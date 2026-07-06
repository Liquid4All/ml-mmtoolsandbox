"""
Renderer for A2UI extension.

This module provides functionality to render A2UI surfaces into PIL images.
It is a lightweight, standalone renderer that does not depend on a browser.

Supported protocol components:
    Column, Row, List (explicit + template), Card, Divider,
    Text, Image, Button, TextField, DateTimeInput, CheckBox,
    MultipleChoice, Slider, Icon
"""

from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from mmtoolsandbox.a2ui.state import SurfaceState, _parse_pointer_segments

# --- Constants ---
CANVAS_WIDTH = 600
CANVAS_HEIGHT = 2000  # Initial height, will be cropped
BACKGROUND_COLOR = (245, 245, 245)
TEXT_COLOR = (50, 50, 50)
LABEL_COLOR = (138, 138, 142)
PRIMARY_COLOR = (0, 122, 255)
SECONDARY_COLOR = (142, 142, 147)
BORDER_COLOR = (220, 220, 220)
DIVIDER_COLOR = (235, 235, 235)
SHADOW_OUTER = (215, 215, 215)
SHADOW_INNER = (230, 230, 230)

# Module-level state for highlight rendering.  Only safe in single-threaded use;
# set/cleared inside render_surface() with a try/finally guard.
_highlight_component_id: str | None = None
# When True, images render as compact thumbnails (used inside Row layouts).
_compact_images: bool = False
# When True, vertical Column spacing is tighter (used inside Row children).
_inside_row: bool = False
CARD_BG_COLOR = (255, 255, 255)
CARD_PADDING = 16
PADDING = 16
ROW_GAP = 12
ROW_INNER_PADDING = 8
BODY_COLOR = (80, 80, 80)
FONT_SIZE = 14
SMALL_FONT_SIZE = 12

# Font sizes per usageHint level — provides visual hierarchy for headings.
_HEADING_FONT_SIZES: dict[str, int] = {
    "h1": 26,
    "h2": 22,
    "h3": 18,
    "h4": 16,
    "h5": 14,
    "caption": 12,
}
# Max height for images rendered inside a Row (thumbnails).
_ROW_IMAGE_MAX_HEIGHT = 100


# --- Data Resolution ---


def _resolve_path(data: Any, path: str) -> Any:
    """Resolve a JSON Pointer path against the data model.

    Supports nested dicts and list indexing (e.g., ``/contacts/0/name``).

    Args:
        data: The root data object (typically a dict).
        path: A JSON Pointer path (e.g., ``/user/name`` or ``title``).

    Returns:
        The resolved value, or a placeholder string if resolution fails.
    """
    segments = _parse_pointer_segments(path)
    if not segments:
        return data

    current = data
    for segment in segments:
        if isinstance(current, dict):
            current = current.get(segment)
            if current is None:
                return f"[{path}]"
        elif isinstance(current, list):
            try:
                current = current[int(segment)]
            except (ValueError, IndexError):
                return f"[{path}]"
        else:
            return f"[{path}]"
    return current if current is not None else f"[{path}]"


def _get_value(value_def: Any, data_model: dict[str, Any]) -> str:
    """Resolve a v0.8 BoundValue to a display string.

    Handles ``literalString``, ``literalNumber``, ``literalBoolean``,
    ``literalArray``, and ``path`` references.

    Args:
        value_def: The value definition from a component property.
        data_model: The current data model for the surface.

    Returns:
        The resolved string value.
    """
    if isinstance(value_def, dict):
        if "literalString" in value_def:
            return str(value_def["literalString"])
        if "literalNumber" in value_def:
            return str(value_def["literalNumber"])
        if "literalBoolean" in value_def:
            return str(value_def["literalBoolean"])
        if "literalArray" in value_def:
            return ", ".join(str(v) for v in value_def["literalArray"])
        if "path" in value_def:
            return str(_resolve_path(data_model, value_def["path"]))
    return str(value_def) if value_def else ""


def _get_bool_value(value_def: Any, data_model: dict[str, Any]) -> bool:
    """Resolve a v0.8 BoundValue to a boolean.

    Args:
        value_def: The value definition from a component property.
        data_model: The current data model for the surface.

    Returns:
        The resolved boolean value.
    """
    if isinstance(value_def, dict):
        if "literalBoolean" in value_def:
            return bool(value_def["literalBoolean"])
        if "path" in value_def:
            return bool(_resolve_path(data_model, value_def["path"]))
    return bool(value_def)


def _get_number_value(value_def: Any, data_model: dict[str, Any]) -> float:
    """Resolve a v0.8 BoundValue to a number.

    Args:
        value_def: The value definition from a component property.
        data_model: The current data model for the surface.

    Returns:
        The resolved numeric value.
    """
    if isinstance(value_def, dict):
        if "literalNumber" in value_def:
            return float(value_def["literalNumber"])
        if "path" in value_def:
            resolved = _resolve_path(data_model, value_def["path"])
            try:
                return float(resolved)
            except (TypeError, ValueError):
                return 0.0
    try:
        return float(value_def)
    except (TypeError, ValueError):
        return 0.0


# --- Drawing Helpers ---

_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _get_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a font, falling back to the built-in default. Results are cached."""
    if font_size not in _FONT_CACHE:
        try:
            _FONT_CACHE[font_size] = ImageFont.truetype("Arial.ttf", font_size)
        except OSError:
            _FONT_CACHE[font_size] = ImageFont.load_default()
    return _FONT_CACHE[font_size]


def _draw_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    width: int,
    font_size: int = FONT_SIZE,
    color: tuple[int, int, int] = TEXT_COLOR,
    bold: bool = False,
) -> int:
    """Draw word-wrapped text and return the total height used.

    Args:
        draw: The PIL ImageDraw object.
        text: The text to draw.
        x: The x coordinate.
        y: The y coordinate.
        width: The available width for wrapping.
        font_size: The font size.
        color: The text color.
        bold: Simulate bold by double-drawing with 1px offset.

    Returns:
        The height consumed by the rendered text.
    """
    font = _get_font(font_size)
    line_height = int(font_size * 1.4)

    # Pixel-accurate word wrapping using font.getlength().
    # Split on newlines first so explicit line breaks are respected.
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            # Preserve blank lines as empty entries (renders as vertical gap).
            lines.append("")
            continue
        current_line = ""
        for word in paragraph.split(" "):
            if not word:
                continue
            test_line = (current_line + " " + word).strip() if current_line else word
            if font.getlength(test_line) <= width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                # Handle single word wider than available width.
                if font.getlength(word) > width:
                    # Break word character by character.
                    partial = ""
                    for ch in word:
                        if font.getlength(partial + ch) > width and partial:
                            lines.append(partial)
                            partial = ch
                        else:
                            partial += ch
                    current_line = partial
                else:
                    current_line = word
        if current_line:
            lines.append(current_line)

    current_y = y
    for line in lines:
        if bold:
            draw.text((x + 1, current_y), line, fill=color, font=font)
        draw.text((x, current_y), line, fill=color, font=font)
        current_y += line_height

    return current_y - y if lines else line_height


# --- Resolver for children (explicitList or template) ---


def _resolve_children(
    children_def: dict[str, Any],
    surface: SurfaceState,
) -> list[tuple[str, dict[str, Any] | None]]:
    """Resolve a v0.8 children definition to a list of (component_id, overlay_data) pairs.

    For ``explicitList``, returns the list of IDs with no data overlay.
    For ``template``, iterates over the data model at ``dataBinding`` and
    returns the template component ID repeated with scoped data overlays.

    Args:
        children_def: The children definition from a layout component.
        surface: The surface state (for data model access).

    Returns:
        A list of (component_id, optional overlay dict) tuples.
    """
    if "explicitList" in children_def:
        return [(cid, None) for cid in children_def["explicitList"]]

    if "template" in children_def:
        template = children_def["template"]
        template_id = template.get("componentId", "")
        data_path = template.get("dataBinding", "")
        items = _resolve_path(surface.data_model, data_path)

        result: list[tuple[str, dict[str, Any] | None]] = []
        if isinstance(items, dict):
            for _key, item_data in items.items():
                overlay = item_data if isinstance(item_data, dict) else {}
                result.append((template_id, overlay))
        elif isinstance(items, list):
            for item_data in items:
                overlay = item_data if isinstance(item_data, dict) else {}
                result.append((template_id, overlay))
        return result

    return []


# --- Component Rendering ---


def _render_component(
    component_id: str,
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    surface: SurfaceState,
    data_overlay: dict[str, Any] | None = None,
) -> int:
    """Render a single component recursively, returning the height used.

    Args:
        component_id: The ID of the component to render.
        image: The PIL Image canvas.
        draw: The PIL ImageDraw object.
        x: The x coordinate.
        y: The y coordinate.
        width: The available width.
        surface: The surface state.
        data_overlay: Optional scoped data dict overlaid on the surface data model
                      for template-based rendering.

    Returns:
        The height consumed by this component.
    """
    if component_id not in surface.components:
        return _draw_text(
            draw, f"Missing: {component_id}", x, y, width, color=(255, 0, 0)
        )

    comp_def = surface.components[component_id]
    wrapper = comp_def.get("component", {})
    if not wrapper:
        return 0

    comp_type = next(iter(wrapper))
    props = wrapper[comp_type]

    # Build effective data model (overlay for template items)
    effective_data = dict(surface.data_model)
    if data_overlay:
        effective_data.update(data_overlay)

    # Temporarily swap data model for this render call
    saved_data = surface.data_model
    surface.data_model = effective_data
    try:
        height = _render_component_inner(
            comp_type,
            props,
            component_id,
            image,
            draw,
            x,
            y,
            width,
            surface,
        )
    finally:
        surface.data_model = saved_data

    # Draw highlight border if this is the highlighted component
    if (
        _highlight_component_id
        and component_id == _highlight_component_id
        and height > 0
    ):
        border_color = (220, 50, 50)  # Red
        border_width = 3
        for i in range(border_width):
            draw.rectangle(
                [x - i - 1, y - i - 1, x + width + i, y + height + i],
                outline=border_color,
            )
        # Draw a small label
        label = "CLICKED"
        _draw_text(draw, label, x + 2, max(y - 18, 2), width, color=border_color)

    return height


def _render_component_inner(
    comp_type: str,
    props: dict[str, Any],
    component_id: str,
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    surface: SurfaceState,
) -> int:
    """Dispatch rendering for a specific component type.

    Returns the height consumed.
    """
    if comp_type == "Column":
        return _render_layout(
            props, image, draw, x, y, width, surface, horizontal=False
        )

    elif comp_type == "Row":
        global _compact_images, _inside_row
        saved_compact = _compact_images
        saved_inside_row = _inside_row
        _compact_images = True
        _inside_row = True
        try:
            return _render_layout(
                props, image, draw, x, y, width, surface, horizontal=True
            )
        finally:
            _compact_images = saved_compact
            _inside_row = saved_inside_row

    elif comp_type == "List":
        direction = props.get("direction", "vertical")
        horizontal = direction == "horizontal"
        return _render_layout(
            props, image, draw, x, y, width, surface, horizontal=horizontal
        )

    elif comp_type == "Card":
        return _render_card(props, image, draw, x, y, width, surface)

    elif comp_type == "Divider":
        return _render_divider(props, draw, x, y, width)

    elif comp_type == "Text":
        return _render_text(props, draw, x, y, width, surface)

    elif comp_type == "Button":
        return _render_button(props, image, draw, x, y, width, surface)

    elif comp_type in ("TextField", "DateTimeInput"):
        return _render_input_field(comp_type, props, draw, x, y, width, surface)

    elif comp_type == "CheckBox":
        return _render_checkbox(props, draw, x, y, width, surface)

    elif comp_type == "MultipleChoice":
        return _render_multiple_choice(props, draw, x, y, width, surface)

    elif comp_type == "Slider":
        return _render_slider(props, draw, x, y, width, surface)

    elif comp_type == "Icon":
        return _render_icon(props, draw, x, y, width, surface)

    elif comp_type == "Image":
        return _render_image(props, image, draw, x, y, width, surface)

    else:
        # Unknown protocol component — render placeholder
        return _draw_text(draw, f"[{comp_type}]", x, y, width, color=LABEL_COLOR)


# --- Layout Components ---


def _render_layout(
    props: dict[str, Any],
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    surface: SurfaceState,
    *,
    horizontal: bool,
) -> int:
    """Render children either vertically or horizontally."""
    children = _resolve_children(props.get("children", {}), surface)
    if not children:
        return 0

    if horizontal:
        # Compute widths proportional to component weights (default weight=1).
        weights: list[float] = []
        for child_id, _overlay in children:
            w = 1.0
            if child_id in surface.components:
                w = float(surface.components[child_id].get("weight", 1))
            weights.append(w)
        total_weight = sum(weights) or 1.0

        gap = ROW_GAP
        total_gap = gap * max(len(children) - 1, 0)
        usable_width = width - total_gap

        max_height = 0
        current_x = x
        for idx, (child_id, overlay) in enumerate(children):
            child_width = max(int(usable_width * weights[idx] / total_weight), 1)
            h = _render_component(
                child_id,
                image,
                draw,
                current_x,
                y,
                child_width,
                surface,
                overlay,
            )
            max_height = max(max_height, h)
            current_x += child_width + gap
        return max_height
    else:
        vgap = ROW_INNER_PADDING if _inside_row else PADDING
        height_used = 0
        current_y = y
        for child_id, overlay in children:
            h = _render_component(
                child_id, image, draw, x, current_y, width, surface, overlay
            )
            current_y += h + vgap
            height_used += h + vgap
        return max(height_used - vgap, 0)


def _render_card(
    props: dict[str, Any],
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    surface: SurfaceState,
) -> int:
    child_id = props.get("child")
    inner_w = width - 2 * CARD_PADDING

    # Measure child height on a scratch canvas to avoid drawing twice.
    child_h = 0
    if child_id:
        scratch = Image.new("RGB", (inner_w, CANVAS_HEIGHT), CARD_BG_COLOR)
        scratch_draw = ImageDraw.Draw(scratch)
        child_h = _render_component(
            child_id,
            scratch,
            scratch_draw,
            0,
            0,
            inner_w,
            surface,
        )

    # Draw card background, then render child once on the real canvas.
    total_h = child_h + 2 * CARD_PADDING

    # Shadow layers (simple offset approach, no alpha compositing).
    draw.rounded_rectangle(
        [x + 2, y + 2, x + width + 2, y + total_h + 2],
        radius=10,
        fill=SHADOW_OUTER,
    )
    draw.rounded_rectangle(
        [x + 1, y + 1, x + width + 1, y + total_h + 1],
        radius=10,
        fill=SHADOW_INNER,
    )

    # Card background.
    draw.rounded_rectangle(
        [x, y, x + width, y + total_h],
        radius=10,
        fill=CARD_BG_COLOR,
    )
    if child_id:
        _render_component(
            child_id,
            image,
            draw,
            x + CARD_PADDING,
            y + CARD_PADDING,
            inner_w,
            surface,
        )

    return total_h


def _render_divider(
    props: dict[str, Any],
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
) -> int:
    axis = props.get("axis", "horizontal")
    if axis == "horizontal":
        line_y = y + 12
        draw.line([(x, line_y), (x + width, line_y)], fill=DIVIDER_COLOR, width=1)
        return 24
    return 0


# --- Content Components ---


def _render_text(
    props: dict[str, Any],
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    surface: SurfaceState,
) -> int:
    text_val = _get_value(props.get("text", ""), surface.data_model)
    usage_hint = props.get("usageHint", "body")
    font_size = _HEADING_FONT_SIZES.get(usage_hint, FONT_SIZE)
    bold = usage_hint in ("h1", "h2")
    # Color hierarchy: headings (dark) > body (medium) > caption (light)
    if usage_hint == "caption":
        color = LABEL_COLOR
    elif usage_hint in ("h1", "h2", "h3"):
        color = TEXT_COLOR
    else:
        color = BODY_COLOR
    return _draw_text(
        draw, text_val, x, y, width, font_size=font_size, bold=bold, color=color
    )


def _render_icon(
    props: dict[str, Any],
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    surface: SurfaceState,
) -> int:
    icon_name = _get_value(props.get("name", ""), surface.data_model)
    # Render as a labeled box
    box_size = 24
    draw.rounded_rectangle(
        [x, y, x + box_size, y + box_size],
        radius=4,
        outline=BORDER_COLOR,
        fill=(230, 230, 230),
    )
    _draw_text(
        draw,
        icon_name,
        x + box_size + 4,
        y + 2,
        width - box_size - 4,
        font_size=SMALL_FONT_SIZE,
        color=LABEL_COLOR,
    )
    return box_size


def _render_image(
    props: dict[str, Any],
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    surface: SurfaceState,
) -> int:
    max_height = _ROW_IMAGE_MAX_HEIGHT if _compact_images else 200
    img_height = min(100, max_height) if _compact_images else min(120, max_height)
    url_val = _get_value(props.get("url", ""), surface.data_model)

    if url_val.startswith("data:image"):
        try:
            _header, encoded = url_val.split(",", 1)
            data = base64.b64decode(encoded)
            img: Image.Image = Image.open(io.BytesIO(data))

            aspect_ratio = img.height / img.width
            display_height = min(int(width * aspect_ratio), max_height)
            img = img.resize((width, display_height))
            image.paste(img, (x, y))
            return display_height
        except Exception as e:
            draw.rectangle(
                [x, y, x + width, y + img_height],
                outline=BORDER_COLOR,
                fill=(240, 240, 240),
            )
            _draw_text(
                draw, f"Image Error: {e}", x + PADDING, y + PADDING, width - 2 * PADDING
            )
            return img_height

    # Non-data-URI placeholder — clean, icon-based design.
    draw.rounded_rectangle(
        [x, y, x + width, y + img_height],
        radius=6,
        outline=BORDER_COLOR,
        fill=(245, 245, 245),
    )
    # Draw a simple landscape icon (mountain + sun) centered in placeholder.
    cx, cy = x + width // 2, y + img_height // 2
    icon_scale = min(width, img_height) // 6
    if icon_scale > 4:
        # Sun circle
        sr = max(icon_scale // 2, 3)
        draw.ellipse(
            [
                cx + icon_scale - sr,
                cy - icon_scale - sr,
                cx + icon_scale + sr,
                cy - icon_scale + sr,
            ],
            fill=(200, 200, 200),
        )
        # Mountain triangle
        draw.polygon(
            [
                (cx - icon_scale * 2, cy + icon_scale),
                (cx, cy - icon_scale),
                (cx + icon_scale * 2, cy + icon_scale),
            ],
            fill=(210, 210, 210),
        )
    # Truncated URL label below icon.
    if url_val:
        label = url_val
        # Strip protocol and long query params.
        for prefix in ("https://", "http://"):
            if label.startswith(prefix):
                label = label[len(prefix) :]
                break
        if len(label) > 40:
            label = label[:37] + "..."
        font = _get_font(SMALL_FONT_SIZE)
        label_w = font.getlength(label)
        label_x = x + max(int((width - label_w) / 2), PADDING)
        label_y = cy + icon_scale + 6 if icon_scale > 4 else cy
        draw.text((label_x, label_y), label, fill=LABEL_COLOR, font=font)
    return img_height


# --- Interactive Components ---


def _render_button(
    props: dict[str, Any],
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    surface: SurfaceState,
) -> int:
    btn_height = 40
    variant = props.get("variant", "")
    is_primary = props.get("primary", False) or variant == "primary"
    is_destructive = variant == "destructive"

    if is_destructive:
        bg_color = (220, 50, 50)
        text_color = (255, 255, 255)
    elif is_primary:
        bg_color = PRIMARY_COLOR
        text_color = (255, 255, 255)
    else:
        # Secondary / default — light gray with dark text
        bg_color = (230, 230, 230)
        text_color = TEXT_COLOR

    draw.rounded_rectangle(
        [x, y, x + width, y + btn_height],
        radius=6,
        fill=bg_color,
    )

    child_id = props.get("child")
    if child_id and child_id in surface.components:
        child_def = surface.components[child_id]
        child_wrapper = child_def.get("component", {})
        if "Text" in child_wrapper:
            text_val = _get_value(
                child_wrapper["Text"].get("text", ""), surface.data_model
            )
            font = _get_font(FONT_SIZE)
            max_text_w = width - 2 * PADDING
            # Truncate with ellipsis if text is too wide.
            if font.getlength(text_val) > max_text_w:
                while (
                    len(text_val) > 1 and font.getlength(text_val + "...") > max_text_w
                ):
                    text_val = text_val[:-1]
                text_val = text_val + "..."
            text_w = font.getlength(text_val)
            bbox = font.getbbox(text_val)
            text_h = bbox[3] - bbox[1]
            text_x = x + int((width - text_w) / 2)
            text_y = y + int((btn_height - text_h) / 2)
            draw.text((text_x, text_y), text_val, fill=text_color, font=font)

    return btn_height


def _render_input_field(
    comp_type: str,
    props: dict[str, Any],
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    surface: SurfaceState,
) -> int:
    # Label
    label_val = _get_value(props.get("label", ""), surface.data_model)
    h_label = _draw_text(
        draw, label_val, x, y, width, font_size=SMALL_FONT_SIZE, color=LABEL_COLOR
    )

    # Input box
    box_y = y + h_label + 4
    box_height = 36
    draw.rounded_rectangle(
        [x, box_y, x + width, box_y + box_height],
        radius=6,
        outline=BORDER_COLOR,
        fill=(248, 248, 248),
        width=1,
    )

    # Value
    value_key = "text" if comp_type == "TextField" else "value"
    value_val = _get_value(props.get(value_key, ""), surface.data_model)
    if value_val:
        _draw_text(draw, value_val, x + 8, box_y + 8, width - 16)

    return h_label + 4 + box_height


def _render_checkbox(
    props: dict[str, Any],
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    surface: SurfaceState,
) -> int:
    label_val = _get_value(props.get("label", ""), surface.data_model)
    checked = _get_bool_value(props.get("value", False), surface.data_model)

    box_size = 20
    draw.rounded_rectangle(
        [x, y + 1, x + box_size, y + 1 + box_size],
        radius=3,
        outline=PRIMARY_COLOR if checked else BORDER_COLOR,
        width=2,
    )
    if checked:
        # Draw checkmark
        draw.line(
            [(x + 4, y + 11), (x + 8, y + 16), (x + 16, y + 5)],
            fill=PRIMARY_COLOR,
            width=2,
        )

    _draw_text(
        draw,
        label_val,
        x + box_size + 8,
        y + 2,
        width - box_size - 8,
    )
    return max(box_size + 2, FONT_SIZE + 4)


def _render_multiple_choice(
    props: dict[str, Any],
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    surface: SurfaceState,
) -> int:
    options = props.get("options", [])
    selections_raw = props.get("selections", {})
    selected: list[str] = []
    if isinstance(selections_raw, dict):
        if "literalArray" in selections_raw:
            selected = selections_raw["literalArray"]
        elif "path" in selections_raw:
            resolved = _resolve_path(surface.data_model, selections_raw["path"])
            if isinstance(resolved, list):
                selected = [str(s) for s in resolved]

    height_used = 0
    current_y = y
    for option in options:
        opt_label = _get_value(option.get("label", ""), surface.data_model)
        opt_value = option.get("value", "")
        is_selected = opt_value in selected

        # Draw chip / checkbox style
        chip_h = 32
        bg = PRIMARY_COLOR if is_selected else (245, 245, 245)
        fg = (255, 255, 255) if is_selected else TEXT_COLOR
        outline = PRIMARY_COLOR if is_selected else BORDER_COLOR
        draw.rounded_rectangle(
            [x, current_y, x + width, current_y + chip_h],
            radius=6,
            fill=bg,
            outline=outline,
        )
        _draw_text(
            draw,
            opt_label,
            x + 10,
            current_y + 7,
            width - 20,
            font_size=SMALL_FONT_SIZE,
            color=fg,
        )
        current_y += chip_h + 6
        height_used += chip_h + 6

    return max(height_used - 6, 0)


def _render_slider(
    props: dict[str, Any],
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    surface: SurfaceState,
) -> int:
    min_val = props.get("minValue", 0)
    max_val = props.get("maxValue", 100)
    current_val = _get_number_value(props.get("value", 0), surface.data_model)

    # Label
    label_text = f"{current_val:.0f}"
    h_label = _draw_text(
        draw, label_text, x, y, width, font_size=SMALL_FONT_SIZE, color=LABEL_COLOR
    )

    # Track
    track_y = y + h_label + 6
    track_h = 6
    draw.rounded_rectangle(
        [x, track_y, x + width, track_y + track_h],
        radius=3,
        fill=BORDER_COLOR,
    )

    # Filled portion
    range_span = max_val - min_val
    if range_span > 0:
        ratio = max(0.0, min(1.0, (current_val - min_val) / range_span))
        fill_width = int(width * ratio)
        if fill_width > 0:
            draw.rounded_rectangle(
                [x, track_y, x + fill_width, track_y + track_h],
                radius=3,
                fill=PRIMARY_COLOR,
            )

        # Thumb with white border for depth
        thumb_x = x + fill_width
        thumb_r = 8
        draw.ellipse(
            [thumb_x - thumb_r, track_y - 4, thumb_x + thumb_r, track_y + track_h + 4],
            fill=(255, 255, 255),
            outline=PRIMARY_COLOR,
            width=2,
        )

    return h_label + 6 + track_h + 12


# --- Public API ---


def render_surface(
    surface: SurfaceState,
    highlight_component_id: str | None = None,
) -> Image.Image | None:
    """Render the surface state to a PIL Image.

    Args:
        surface: The surface state to render.
        highlight_component_id: Optional component ID to highlight with a red
                                border (e.g., the button a user clicked).

    Returns:
        A PIL Image representing the UI, or ``None`` if the surface has no root.
    """
    global _highlight_component_id
    if not surface.root_id:
        return None

    # Set module-level highlight so _render_component can read it
    _highlight_component_id = highlight_component_id

    image = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image)

    try:
        total_height = _render_component(
            surface.root_id,
            image,
            draw,
            PADDING,
            PADDING,
            CANVAS_WIDTH - 2 * PADDING,
            surface,
        )
    except Exception as e:
        _draw_text(draw, f"Rendering Error: {e}", PADDING, PADDING, CANVAS_WIDTH)
        total_height = 50
    finally:
        _highlight_component_id = None

    final_height = min(max(total_height + 2 * PADDING, 100), CANVAS_HEIGHT)
    return image.crop((0, 0, CANVAS_WIDTH, final_height))
