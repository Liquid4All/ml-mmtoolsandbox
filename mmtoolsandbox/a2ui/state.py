"""
State management for A2UI extension.

This module maintains the state of the A2UI session, including the current
screen content and interaction history. It faithfully implements the v0.8
protocol's data model, including nested ``valueMap`` structures and JSON
Pointer path resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _resolve_contents(contents: list[dict[str, Any]]) -> dict[str, Any]:
    """Recursively convert a v0.8 ``contents`` list into a nested Python dict.

    The A2UI v0.8 protocol represents data as a list of entries, each with a
    ``key`` and exactly one typed value (``valueString``, ``valueNumber``,
    ``valueBoolean``, or ``valueMap`` for nested structures).

    Args:
        contents: A list of entry dicts from a ``dataModelUpdate`` message.

    Returns:
        A nested dictionary representing the resolved data model.
    """
    result: dict[str, Any] = {}
    for entry in contents:
        key = entry.get("key")
        if key is None:
            continue

        if "valueString" in entry:
            result[key] = entry["valueString"]
        elif "valueNumber" in entry:
            result[key] = entry["valueNumber"]
        elif "valueBoolean" in entry:
            result[key] = entry["valueBoolean"]
        elif "valueMap" in entry:
            result[key] = _resolve_contents(entry["valueMap"])
        # If no recognized value key, skip this entry.
    return result


def _parse_pointer_segments(path: str) -> list[str]:
    """Parse a JSON Pointer path into non-empty segments.

    Shared by both ``_set_at_path`` (write) and ``_resolve_path`` in the
    renderer (read) to ensure consistent path handling.

    Args:
        path: A JSON Pointer path (e.g., ``/user/name`` or ``name``).

    Returns:
        A list of path segments with leading slashes stripped and empty
        strings filtered out.
    """
    return [s for s in path.strip("/").split("/") if s]


def _set_at_path(data: dict[str, Any], path: str, value: Any) -> None:
    """Set a value at a JSON Pointer path in the data model.

    Creates intermediate dicts as needed. A path of ``/`` or empty string
    is handled by the caller (replaces entire model).

    Args:
        data: The root data dict to mutate.
        path: A JSON Pointer path (e.g., ``/user/name``).
        value: The value to set at that path.
    """
    segments = _parse_pointer_segments(path)
    if not segments:
        return
    current = data
    for segment in segments[:-1]:
        current = current.setdefault(segment, {})
    current[segments[-1]] = value


def _get_at_path(data: dict[str, Any], path: str) -> Any:
    """Read a value at a JSON Pointer path in the data model.

    Args:
        data: The root data dict to read from.
        path: A JSON Pointer path (e.g., ``/user/name`` or ``name``).

    Returns:
        The value at the path, or None if the path doesn't exist.
    """
    segments = _parse_pointer_segments(path)
    if not segments:
        return data
    current: Any = data
    for segment in segments:
        if not isinstance(current, dict) or segment not in current:
            return None
        current = current[segment]
    return current


@dataclass
class SurfaceState:
    """Tracks the state of a single UI surface.

    Attributes:
        surface_id: The unique identifier for the surface.
        root_id: The ID of the root component.
        components: A dictionary mapping component IDs to their definitions.
        data_model: The hierarchical data model for this surface.
        last_image_id: The image ID from the most recent render, or ``None``
                       if the surface has been mutated since last render.
    """

    surface_id: str
    root_id: str | None = None
    components: dict[str, dict[str, Any]] = field(default_factory=dict)
    data_model: dict[str, Any] = field(default_factory=dict)
    last_image_id: int | None = None


@dataclass
class UIState:
    """Tracks state for UI interactions.

    This class implements the Singleton pattern (consistent with AppWorld's
    ``AppWorldState``) to ensure a single shared state across the extension.

    Attributes:
        interaction_history: List of interactions performed by the user.
        surfaces: A dictionary mapping surface IDs to SurfaceState objects.
        active_surface_id: The ID of the currently active surface.
    """

    interaction_history: list[dict[str, Any]] = field(default_factory=list)
    surfaces: dict[str, SurfaceState] = field(default_factory=dict)
    active_surface_id: str | None = None

    # Singleton instance
    _instance: UIState | None = None

    @classmethod
    def get_instance(cls) -> UIState:
        """Get the singleton instance of UIState.

        Returns:
            The singleton UIState instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset state for a new scenario.

        This method clears the current instance and creates a new one.
        """
        cls._instance = cls()

    def record_interaction(self, interaction: dict[str, Any]) -> None:
        """Record a user interaction.

        Args:
            interaction: The interaction details in JSON format.
        """
        self.interaction_history.append(interaction)

    def process_message(self, message: dict[str, Any] | list[dict[str, Any]]) -> None:
        """Process an A2UI message and update the internal state.

        Handles the four v0.8 server-to-client message types:
        ``beginRendering``, ``surfaceUpdate``, ``dataModelUpdate``,
        and ``deleteSurface``.

        Args:
            message: A single A2UI message dict or a list of messages.
        """
        if isinstance(message, list):
            for msg in message:
                self.process_message(msg)
            return

        if "beginRendering" in message:
            payload = message["beginRendering"]
            surface_id = payload["surfaceId"]
            if surface_id not in self.surfaces:
                self.surfaces[surface_id] = SurfaceState(
                    surface_id=surface_id, root_id=payload["root"]
                )
            else:
                # Surface may already have components from a prior surfaceUpdate;
                # just set/update the root without discarding existing state.
                self.surfaces[surface_id].root_id = payload["root"]
            self.active_surface_id = surface_id

        elif "surfaceUpdate" in message:
            payload = message["surfaceUpdate"]
            surface_id = payload["surfaceId"]
            if surface_id not in self.surfaces:
                self.surfaces[surface_id] = SurfaceState(surface_id=surface_id)

            surface = self.surfaces[surface_id]
            self.active_surface_id = surface_id
            for component in payload.get("components", []):
                surface.components[component["id"]] = component
            surface.last_image_id = None  # Invalidate cached render

        elif "dataModelUpdate" in message:
            payload = message["dataModelUpdate"]
            surface_id = payload["surfaceId"]
            if surface_id not in self.surfaces:
                return

            surface = self.surfaces[surface_id]
            resolved = _resolve_contents(payload.get("contents", []))
            path = payload.get("path", "/")

            if path == "/" or path == "":
                surface.data_model.update(resolved)
            else:
                _set_at_path(surface.data_model, path, resolved)
            surface.last_image_id = None  # Invalidate cached render

        elif "deleteSurface" in message:
            payload = message["deleteSurface"]
            surface_id = payload["surfaceId"]
            if surface_id in self.surfaces:
                del self.surfaces[surface_id]
            if self.active_surface_id == surface_id:
                self.active_surface_id = None


def get_ui_state() -> UIState:
    """Get the singleton UIState instance.

    Returns:
        The shared UIState instance.
    """
    return UIState.get_instance()


def get_interactive_elements(surface: SurfaceState) -> str:
    """Extract a human-readable summary of interactive elements from a surface.

    Walks the component tree and identifies buttons (with action names),
    text fields, checkboxes, sliders, date pickers, and choice pickers.
    Returns a plain-text summary suitable for inclusion in messages to the
    user simulator so it knows what actions are available.

    Args:
        surface: The surface state to inspect.

    Returns:
        A plain-text summary of interactive elements, or empty string if none.
    """
    elements: list[str] = []

    for comp_id, comp_def in surface.components.items():
        wrapper = comp_def.get("component", {})
        if not wrapper:
            continue

        comp_type = next(iter(wrapper))
        props = wrapper[comp_type]

        if comp_type == "Button":
            action = props.get("action", {})
            action_name = action.get("name", "unknown")
            # Try to get button label from child text component
            child_id = props.get("child", "")
            label = _get_button_label(surface, child_id)
            elements.append(f'Button "{label}" (id: {comp_id}, action: {action_name})')

        elif comp_type == "TextField":
            label = _extract_label(props.get("label", {}))
            path = _extract_path(props.get("text", {}))
            elements.append(f'TextField "{label}" (id: {comp_id}, path: {path})')

        elif comp_type == "CheckBox":
            label = _extract_label(props.get("label", {}))
            path = _extract_path(props.get("value", {}))
            elements.append(f'CheckBox "{label}" (id: {comp_id}, path: {path})')

        elif comp_type == "Slider":
            min_val = props.get("minValue", 0)
            max_val = props.get("maxValue", 100)
            path = _extract_path(props.get("value", {}))
            elements.append(
                f"Slider [{min_val}-{max_val}] (id: {comp_id}, path: {path})"
            )

        elif comp_type == "DateTimeInput":
            label = _extract_label(props.get("label", {}))
            path = _extract_path(props.get("value", {}))
            elements.append(f'DateTimeInput "{label}" (id: {comp_id}, path: {path})')

        elif comp_type == "MultipleChoice":
            options = props.get("options", [])
            option_labels = [_extract_label(opt.get("label", {})) for opt in options]
            path = _extract_path(props.get("selections", {}))
            elements.append(
                f"MultipleChoice [{', '.join(option_labels)}] (id: {comp_id}, path: {path})"
            )

    if not elements:
        return ""

    lines = [
        f"Available actions on surface '{surface.surface_id}':",
    ]
    for el in elements:
        lines.append(f"  - {el}")

    lines.append("")
    lines.append(
        "To perform an action, call ui_user_interact with:\n"
        "  action_name: the action value shown above\n"
        f"  surface_id: '{surface.surface_id}'\n"
        "  component_id: the id value shown above\n"
        "  field_values: optional dict to fill form fields (keyed by component id)"
    )

    return "\n".join(lines)


def summarize_surface(surface: SurfaceState) -> str:
    """Produce a one-line structural summary of a UI surface.

    Counts component types, measures max nesting depth, and lists
    interactive elements with their labels and action bindings.
    Designed for the UI judge evidence — gives structural facts without
    dumping the full JSON tree.

    Args:
        surface: The surface state to summarize.

    Returns:
        A summary string like:
        ``"8 components (2 Card, 3 Text, 2 Button, 1 Image), depth 4,
        buttons: 'Add to Cart' (add_to_cart), 'View Details' (view_details)"``
    """
    type_counts: dict[str, int] = {}
    buttons: list[str] = []

    for comp_id, comp_def in surface.components.items():
        wrapper = comp_def.get("component", {})
        if not wrapper:
            continue
        comp_type = next(iter(wrapper))
        type_counts[comp_type] = type_counts.get(comp_type, 0) + 1

        if comp_type == "Button":
            props = wrapper[comp_type]
            action = props.get("action", {})
            action_name = action.get("name", "?")
            child_id = props.get("child", "")
            label = _get_button_label(surface, child_id)
            buttons.append(f'"{label}" ({action_name})')

    total = sum(type_counts.values())
    type_str = ", ".join(
        f"{count} {name}" for name, count in sorted(type_counts.items())
    )

    # Measure max depth from root
    def _depth(comp_id: str, visited: set[str]) -> int:
        if comp_id in visited or comp_id not in surface.components:
            return 0
        visited.add(comp_id)
        comp_def = surface.components[comp_id]
        wrapper = comp_def.get("component", {})
        if not wrapper:
            return 1
        comp_type = next(iter(wrapper))
        props = wrapper[comp_type]
        child_depths: list[int] = []
        # singular child (Card, Button, Modal)
        for key in ("child", "entryPointChild", "contentChild"):
            cid = props.get(key, "")
            if cid:
                child_depths.append(_depth(cid, visited))
        # children list (Row, Column, List)
        for cid in props.get("children", []):
            if isinstance(cid, str):
                child_depths.append(_depth(cid, visited))
        return 1 + (max(child_depths) if child_depths else 0)

    max_depth = _depth(surface.root_id or "", set()) if surface.root_id else 0

    parts = [f"{total} components ({type_str})", f"depth {max_depth}"]
    if buttons:
        parts.append("buttons: " + ", ".join(buttons))
    else:
        parts.append("no buttons")

    return "; ".join(parts)


def _get_button_label(surface: SurfaceState, child_id: str) -> str:
    """Try to extract the text label from a button's child Text component."""
    if child_id and child_id in surface.components:
        child = surface.components[child_id]
        child_wrapper = child.get("component", {})
        if "Text" in child_wrapper:
            return _extract_label(child_wrapper["Text"].get("text", {}))
    return child_id or "?"


def _extract_label(value_def: Any) -> str:
    """Extract a display string from a BoundValue definition."""
    if isinstance(value_def, dict):
        if "literalString" in value_def:
            return str(value_def["literalString"])
        if "path" in value_def:
            return f"[{value_def['path']}]"
    return str(value_def) if value_def else "?"


def _extract_path(value_def: Any) -> str:
    """Extract the data path from a BoundValue, or 'literal' if static."""
    if isinstance(value_def, dict):
        if "path" in value_def:
            return str(value_def["path"])
    return "literal"


def _resolve_bound_value(value_def: Any, data_model: dict[str, Any]) -> Any:
    """Resolve a BoundValue to its actual value from the data model.

    Args:
        value_def: A BoundValue dict (with 'path' or 'literalString').
        data_model: The surface's data model to resolve paths against.

    Returns:
        The resolved value, or None if unresolvable.
    """
    if isinstance(value_def, dict):
        if "path" in value_def:
            return _get_at_path(data_model, value_def["path"])
        if "literalString" in value_def:
            return value_def["literalString"]
        if "literalNumber" in value_def:
            return value_def["literalNumber"]
        if "literalBoolean" in value_def:
            return value_def["literalBoolean"]
    return None


def apply_field_values(surface: SurfaceState, field_values: dict[str, str]) -> None:
    """Apply user-provided field values to the surface's data model.

    For each field_id → value mapping, finds the component's data binding
    path and writes the value into the data model.

    Args:
        surface: The surface to update.
        field_values: Mapping of component ID → value to set.
    """
    for field_id, value in field_values.items():
        comp = surface.components.get(field_id, {})
        wrapper = comp.get("component", {})
        comp_type = next(iter(wrapper), None)
        if comp_type is None:
            continue
        props = wrapper[comp_type]

        # Extract data binding path based on component type.
        # Only field-like components that have a settable data binding are
        # handled; other component types (layout containers, buttons, etc.)
        # are silently skipped because they don't represent editable fields.
        if comp_type == "TextField":
            path = _extract_path(props.get("text", {}))
        elif comp_type in ("DateTimeInput", "Slider", "CheckBox"):
            path = _extract_path(props.get("value", {}))
        elif comp_type == "MultipleChoice":
            path = _extract_path(props.get("selections", {}))
        else:
            continue

        if path and path != "literal":
            _set_at_path(surface.data_model, path, value)
    surface.last_image_id = None  # Invalidate cached render


def resolve_action_context(surface: SurfaceState, component_id: str) -> dict[str, Any]:
    """Resolve a button's action.context data bindings from the data model.

    Reads the button component's action definition, resolves each context
    entry's BoundValue against the surface's current data model, and returns
    the resolved key-value pairs.

    Args:
        surface: The surface containing the button and data model.
        component_id: The ID of the button component.

    Returns:
        A dictionary of resolved context key-value pairs.
    """
    comp = surface.components.get(component_id, {})
    wrapper = comp.get("component", {})
    comp_type = next(iter(wrapper), None)
    # Only Button components carry an ``action.context`` list of data bindings.
    # Other component types have no action context to resolve.
    if comp_type != "Button":
        return {}

    props = wrapper[comp_type]
    action = props.get("action", {})
    # Each binding is a dict with "key" (str) and "value" (a BoundValue that
    # may be a literal or a path reference into the data model).
    context_bindings = action.get("context", [])

    resolved: dict[str, Any] = {}
    for binding in context_bindings:
        key = binding.get("key")
        if key is None:
            continue
        value_def = binding.get("value", {})
        resolved[key] = _resolve_bound_value(value_def, surface.data_model)

    return resolved
