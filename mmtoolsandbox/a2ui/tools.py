"""
Tools for UI tooling extension based on the A2UI framework.

Reference: https://github.com/google/A2UI

This module provides tools for the agent to discover UI capabilities,
send UI updates, and simulate user interactions.
"""

import base64
import json
from io import BytesIO
from typing import Any

import jsonschema
from PIL import Image

from mmtoolsandbox.a2ui.docs import (
    _SCHEMA_DIR,
    _load_schema,
    get_quick_start,
    get_ui_categories,
    get_ui_item_details,
    get_ui_items_in_category,
    search_ui_docs,
)
from mmtoolsandbox.a2ui.renderer import render_surface
from mmtoolsandbox.a2ui.state import (
    apply_field_values,
    get_interactive_elements,
    get_ui_state,
    resolve_action_context,
)
from mmtoolsandbox.common.execution_context import (
    DatabaseNamespace,
    RoleType,
    get_current_context,
)
from mmtoolsandbox.common.image_id import ImageId
from mmtoolsandbox.common.message_conversion import (
    UI_STATE_SERVER_CLOSE_TAG,
    UI_STATE_SERVER_OPEN_TAG,
    Message,
    add_messages_to_execution_context,
)
from mmtoolsandbox.common.utils import register_as_tool
from mmtoolsandbox.datasets.names import DatasetName
from mmtoolsandbox.tools.vision.common import load_image, store_image

# Define toolboxes that will have access to UI tools
_A2UI_TOOLBOXES = {
    DatasetName.FULL,
}


def _validate_schema(data: dict[str, Any], schema_name: str) -> None:
    """Validate data against a JSON schema.

    Args:
        data: The data to validate.
        schema_name: The name of the schema file to validate against.

    Raises:
        jsonschema.ValidationError: If the data does not match the schema.
    """
    schema = _load_schema(schema_name)
    resolver = jsonschema.RefResolver(
        base_uri=f"file://{_SCHEMA_DIR}/", referrer=schema
    )
    jsonschema.validate(instance=data, schema=schema, resolver=resolver)


def _image_to_data_uri(image: Image.Image) -> str:
    """Convert a PIL Image to a base64 data URI.

    Args:
        image: The PIL Image to convert.

    Returns:
        str: A base64 data URI string (e.g., "data:image/png;base64,...").
    """
    buffer = BytesIO()
    # Default to PNG for lossless UI rendering
    image.save(buffer, format="PNG")
    encoded_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded_string}"


def _replace_placeholders(data: Any, replacements: dict[str, str]) -> Any:
    """Recursively replace string values in a JSON object with replacements.

    Args:
        data: The JSON object (dict, list, or primitive) to traverse.
        replacements: A dictionary mapping placeholder strings to replacement strings.

    Returns:
        Any: The data structure with replacements applied.
    """
    if isinstance(data, str):
        return replacements.get(data, data)
    elif isinstance(data, dict):
        return {k: _replace_placeholders(v, replacements) for k, v in data.items()}
    elif isinstance(data, list):
        return [_replace_placeholders(i, replacements) for i in data]
    return data


def _has_image_placeholder(data: Any) -> bool:
    """Check if data contains unresolved IMAGE_URL_* placeholder strings."""
    if isinstance(data, str):
        return data.startswith("IMAGE_URL_")
    elif isinstance(data, dict):
        return any(_has_image_placeholder(v) for v in data.values())
    elif isinstance(data, list):
        return any(_has_image_placeholder(item) for item in data)
    return False


# ============================================================================
# Discovery Tools
# ============================================================================


@register_as_tool(
    toolboxes=_A2UI_TOOLBOXES,
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def ui_explore_capabilities() -> list[str]:
    """List the categories of UI documentation available.

    Returns categories like 'Components', 'Concepts', 'Examples' that you
    can explore further with `ui_list_items`.

    Returns:
        list[str]: Available documentation categories.
    """
    return get_ui_categories()


@register_as_tool(
    toolboxes=_A2UI_TOOLBOXES,
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def ui_list_items(category: str) -> list[str]:
    """List items in a UI documentation category.

    Args:
        category: The category to browse (e.g., 'Components', 'Examples').

    Returns:
        list[str]: Items with brief descriptions.
    """
    return get_ui_items_in_category(category)


@register_as_tool(
    toolboxes=_A2UI_TOOLBOXES,
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def ui_get_item_details(category: str, item_name: str) -> str:
    """Get the schema, documentation, or example JSON for a specific UI item.

    Args:
        category: The category (e.g., 'Components', 'Examples').
        item_name: The item name (e.g., 'Button', 'booking_form').

    Returns:
        str: Schema with a usage example, concept documentation, or full example JSON.
    """
    return get_ui_item_details(category, item_name)


@register_as_tool(
    toolboxes=_A2UI_TOOLBOXES,
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def ui_search_docs(query: str) -> str:
    """Search across all UI documentation and examples.

    Args:
        query: Search text (e.g., "image", "list", "button").

    Returns:
        str: Matching items ranked by relevance.
    """
    return search_ui_docs(query)


@register_as_tool(
    toolboxes=_A2UI_TOOLBOXES,
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def ui_get_quick_start() -> str:
    """Get a complete guide for rendering UI screens.

    Returns everything you need in one call: the UI message format,
    key patterns, a full working example, and a list of all components.
    Call this FIRST before using `render_ui_screen`.

    Returns:
        str: Quick-start guide with format overview, example, and component list.
    """
    return get_quick_start()


# ============================================================================
# Execution Tools
# ============================================================================


@register_as_tool(
    toolboxes=_A2UI_TOOLBOXES,
    database_namespaces={DatabaseNamespace.IMAGE},
    visible_to=(RoleType.AGENT,),
)
def render_ui_screen(
    ui_json: Any,
    image_placeholders: Any = None,
) -> Any:
    """Render a UI screen and return the result.

    IMPORTANT: After rendering, call `show_ui_to_user()` to display the UI
    to the user. Interactive element metadata is automatically provided to
    the user — you do NOT need to include it manually.

    Tip: Call `ui_get_item_details('Examples', '<name>')` to see full working
    templates you can adapt (e.g., 'booking_form', 'contact_card', 'contact_list').
    Call `ui_get_quick_start` for format overview and full component list.

    Minimal example (renders a heading):
        render_ui_screen(ui_json=[
            {"beginRendering": {"surfaceId": "s1", "root": "root"}},
            {"surfaceUpdate": {"surfaceId": "s1", "components": [
                {"id": "root", "component": {"Text": {"text": {"literalString": "Hello!"}, "usageHint": "h1"}}}
            ]}}
        ])

    Args:
        ui_json: The UI description. Can be:
                 - A list of message dicts (recommended): [{...}, {...}, {...}]
                 - A single message dict: {...}
                 - A JSON string: '[{...}, {...}]'

        image_placeholders: Optional mapping of placeholder strings to image IDs
                            (integers). Example: {"IMAGE_URL_1": 5}
                            In your UI, use the placeholder as the image url:
                            {"Image": {"url": {"literalString": "IMAGE_URL_1"}}}

    Returns:
        dict: Contains "image" (ImageResult) and optionally "ui_interactive_elements".
        str: Error message on failure.
    """
    # Normalize input: accept str, dict, or list
    if isinstance(ui_json, str):
        try:
            data = json.loads(ui_json)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON string. {e}"
    elif isinstance(ui_json, (dict, list)):
        data = ui_json
    else:
        return f"Error: ui_json must be a JSON string, dict, or list. Got {type(ui_json).__name__}."

    # Resolve image placeholders if provided
    if image_placeholders:
        try:
            replacements = {}
            for placeholder, image_id in image_placeholders.items():
                # Validate that image_id is an integer, not a wrapped dict
                if isinstance(image_id, dict):
                    return (
                        f"Error: image_placeholders values must be integers, "
                        f"not dicts. Got {image_id} for '{placeholder}'. "
                        f'Use: {{"IMAGE_URL_1": 5}} not {{"IMAGE_URL_1": {{"image_id": 5}}}}'
                    )
                img = load_image(ImageId(int(image_id)))
                replacements[placeholder] = _image_to_data_uri(img)

            data = _replace_placeholders(data, replacements)
        except Exception as e:
            return f"Error: Failed to resolve image placeholders. {e}"

    try:
        # Validate against server_to_client.json
        if isinstance(data, list):
            for item in data:
                _validate_schema(item, "server_to_client.json")
        else:
            _validate_schema(data, "server_to_client.json")
    except jsonschema.ValidationError as e:
        return f"Error: UI description does not match UI schema. {str(e)}"
    except Exception as e:
        return f"Error: Schema validation failed. {str(e)}"

    # Update internal states
    state = get_ui_state()
    # We use process_message directly to handle both dict and list
    state.process_message(data)

    target_id = state.active_surface_id
    if not target_id or target_id not in state.surfaces:
        return "UI updated, but failed to render image (no active surface found)."

    surface = state.surfaces[target_id]

    # Render to PIL image
    try:
        image = render_surface(surface)
        assert image is not None

        image_result = store_image(image)
        surface.last_image_id = image_result.image_id  # Cache for show_ui_to_user

        # Build interactive elements metadata.
        metadata = get_interactive_elements(surface)

        result: dict[str, Any] = {"image": image_result}
        if metadata:
            result["ui_interactive_elements"] = metadata

        # Warn about unresolved image placeholders. This runs AFTER
        # placeholder resolution, so it catches partially-resolved cases
        # (e.g., agent provided IMAGE_URL_1 but JSON also has IMAGE_URL_2).
        if _has_image_placeholder(data):
            result["warning"] = (
                "Unresolved image placeholder(s) detected (e.g., 'IMAGE_URL_1'). "
                "Pass image_placeholders={'IMAGE_URL_1': <image_id>} to resolve "
                "them to actual images."
            )

        return result

    except Exception as e:
        return f"UI image rendering failed with error: {e}"


def _wrap_ui_state_server(content: str) -> str:
    """Wrap content in ``<ui_state_server>`` XML tags."""
    return f"{UI_STATE_SERVER_OPEN_TAG}\n{content}\n{UI_STATE_SERVER_CLOSE_TAG}"


def _notify_agent_of_interaction(
    action_name: str,
    resolved_context: dict[str, Any],
) -> None:
    """Send an environment-mediated notification to the agent about a user interaction.

    Instead of letting the agent see the raw ``ui_user_interact`` tool call
    (which contains back-end metadata like component IDs and surface IDs),
    the UI State Server sends a clean semantic summary wrapped in
    ``<ui_state_server>`` XML tags so the agent only sees *what* the user
    did, not *how* the UI dispatched it.

    Args:
        action_name: The action the user performed.
        resolved_context: Key-value pairs resolved from the data model.
    """
    inner_lines = ["User Interaction:"]
    inner_lines.append(f"  Action: {action_name}")
    if resolved_context:
        inner_lines.append("  Resolved data:")
        for k, v in resolved_context.items():
            display_val = v if v else "(empty)"
            # Filter out base64 image data — the agent doesn't need raw
            # image bytes, only semantic data like names and counts.
            if isinstance(display_val, str) and display_val.startswith("data:image"):
                display_val = "(image)"
            inner_lines.append(f"    {k}: {display_val}")

    agent_notification = Message(
        sender=RoleType.EXECUTION_ENVIRONMENT,
        recipient=RoleType.AGENT,
        content=_wrap_ui_state_server("\n".join(inner_lines)),
        visible_to=[RoleType.AGENT],
    )
    add_messages_to_execution_context(get_current_context(), [agent_notification])


@register_as_tool(
    toolboxes=_A2UI_TOOLBOXES,
    database_namespaces={DatabaseNamespace.IMAGE, DatabaseNamespace.SANDBOX},
    visible_to=(RoleType.USER,),
)
def ui_user_interact(
    action_name: str,
    surface_id: str,
    component_id: str,
    field_values: Any = None,
) -> str:
    """Click a button or interact with a UI element shown by the agent.

    When the agent shows you a UI screen, it will list interactive elements like:
      Button "Approve Card" (id: approve-btn, action: approve_invite)
      TextField "Party Size" (id: party-size-field, path: partySize)
    Use the values from that listing to fill in this tool's parameters.

    Args:
        action_name: The action to perform, from the 'action' field
                     (e.g. 'approve_invite', 'submit_booking').
        surface_id: The surface ID shown in the agent's message
                    (e.g. 'invite-card', 'booking-form').
        component_id: The component ID shown in the agent's message
                      (e.g. 'approve-btn', 'submit-btn').
        field_values: Optional dict of field values to fill before clicking,
                      keyed by component id. Example:
                      {'party-size-field': '4', 'datetime-field': '2026-04-04T19:30'}

    Returns:
        A text summary of the interaction with resolved context.
    """
    from datetime import datetime, timezone

    state = get_ui_state()
    surface = state.surfaces.get(surface_id)
    if surface is None:
        return f"Error: Surface '{surface_id}' not found. Available: {list(state.surfaces.keys())}"

    # Apply field values to the data model
    if field_values:
        apply_field_values(surface, field_values)

    # Resolve the button's action.context from the (updated) data model
    resolved_context = resolve_action_context(surface, component_id)

    # Build the userAction
    data = {
        "userAction": {
            "name": action_name,
            "surfaceId": surface_id,
            "sourceComponentId": component_id,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "context": resolved_context,
        }
    }

    try:
        _validate_schema(data, "client_to_server.json")
    except jsonschema.ValidationError as e:
        return f"Error: Interaction does not match UI schema. {str(e)}"
    except Exception as e:
        return f"Error: Schema validation failed. {str(e)}"

    # Record the interaction
    state.record_interaction(data)

    # Re-render the surface with updated state
    result_lines = [
        f"User Interaction on surface '{surface_id}':",
        f"  Action: {action_name} (component: {component_id})",
    ]
    if resolved_context:
        result_lines.append("  Resolved context:")
        for k, v in resolved_context.items():
            display_val = v if v else "(empty)"
            if isinstance(display_val, str) and display_val.startswith("data:image"):
                display_val = "(image)"
            result_lines.append(f"    {k}: {display_val}")

    try:
        image = render_surface(surface, highlight_component_id=component_id)
        if image is not None:
            highlighted_image = store_image(image)
            result_lines.append(f"  Updated UI: {highlighted_image}")
    except Exception:
        pass  # Re-rendering is best-effort

    # Notify the agent about the user's interaction via the environment.
    # The agent receives a <ui_state_server> tagged summary (resolved context
    # only, no screenshot — agent already knows the UI).
    _notify_agent_of_interaction(action_name, resolved_context)

    return "\n".join(result_lines)


@register_as_tool(
    toolboxes=_A2UI_TOOLBOXES,
    database_namespaces={DatabaseNamespace.IMAGE},
    visible_to=(RoleType.AGENT,),
)
def show_ui_to_user() -> str:
    """Show the most recently rendered UI screen to the user.

    Sends the latest rendered UI image to the user. The system automatically
    provides the user with interactive element details (buttons, fields) via
    a ``<ui_state_server>`` tagged message.

    Call this after ``render_ui_screen`` to display the result. Write your
    own natural language message to the user in the next turn — do NOT
    include surface IDs or component IDs.

    Returns:
        Confirmation that the UI was shown.
    """
    state = get_ui_state()
    surface_id = state.active_surface_id
    if not surface_id or surface_id not in state.surfaces:
        return "Error: No UI has been rendered yet. Call render_ui_screen first."

    surface = state.surfaces[surface_id]

    # Use cached image from render_ui_screen if available; re-render only
    # if the surface was mutated since the last render.
    if surface.last_image_id is not None:
        image_id = ImageId(surface.last_image_id)
    else:
        try:
            image = render_surface(surface)
            if image is None:
                return "Error: Failed to render surface (no root component)."
            image_result = store_image(image)
            surface.last_image_id = image_result.image_id
            image_id = image_result.image_id
        except Exception as e:
            return f"Error: Failed to render surface. {e}"

    context = get_current_context()

    # 1. AGENT→USER (user-only): image delivery — agent already saw the render
    agent_to_user_msg = Message(
        sender=RoleType.AGENT,
        recipient=RoleType.USER,
        content="",
        image_ids=[image_id],
        visible_to=[RoleType.USER],
    )

    # 2. AGENT→USER (user-only): metadata wrapped in <ui_state_server> XML tag
    metadata = get_interactive_elements(surface)
    messages_to_inject = [agent_to_user_msg]
    if metadata:
        metadata_msg = Message(
            sender=RoleType.AGENT,
            recipient=RoleType.USER,
            content=_wrap_ui_state_server(f"Interactive Elements:\n{metadata}"),
            visible_to=[RoleType.USER],
        )
        messages_to_inject.append(metadata_msg)

    add_messages_to_execution_context(context, messages_to_inject)

    # 3. Tool result returns to agent (EXEC_ENV→AGENT, agent-only)
    return f"UI shown to user. Surface: {surface_id}"


def render_ui_and_wait_for_user_interaction(ui_json: str) -> str:
    """Render a UI for the user and wait for the user interaction."""
    # TODO: Think about how to enable true UI interaction to the front-end.
    raise NotImplementedError
