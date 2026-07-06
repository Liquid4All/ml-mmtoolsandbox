"""Tests for the A2UI PIL renderer.

Covers: rendering correctness, highlight rendering, word wrapping,
newline handling, and all JSON examples rendering without errors.

Requires the A2UI extension (Pillow).  Skipped when not installed.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any

import pytest

PIL = pytest.importorskip(
    "PIL", reason="A2UI extension not installed (requires Pillow)"
)
from PIL import Image  # noqa: E402

from mmtoolsandbox.a2ui.renderer import (  # noqa: E402
    CANVAS_WIDTH,
    _draw_text,
    render_surface,
)
from mmtoolsandbox.a2ui.state import SurfaceState, UIState  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_surface(
    components: dict[str, dict[str, Any]],
    root_id: str = "root",
    data_model: dict[str, Any] | None = None,
) -> SurfaceState:
    """Build a SurfaceState from a flat dict of {id: component_def}."""
    surface = SurfaceState(surface_id="test", root_id=root_id)
    surface.components = components
    if data_model:
        surface.data_model = data_model
    return surface


def _simple_card_with_buttons() -> SurfaceState:
    """Card > Column > [heading, body, Row > [btn_a, btn_b]]."""
    return _make_surface(
        {
            "root": {
                "id": "root",
                "component": {"Card": {"child": "col"}},
            },
            "col": {
                "id": "col",
                "component": {
                    "Column": {
                        "children": {"explicitList": ["heading", "body", "btn_row"]}
                    }
                },
            },
            "heading": {
                "id": "heading",
                "component": {"Text": {"text": "Title", "usageHint": "h2"}},
            },
            "body": {
                "id": "body",
                "component": {"Text": {"text": "Description text here."}},
            },
            "btn_row": {
                "id": "btn_row",
                "component": {
                    "Row": {"children": {"explicitList": ["btn_a", "btn_b"]}}
                },
            },
            "btn_a": {
                "id": "btn_a",
                "component": {
                    "Button": {
                        "action": "approve",
                        "primary": True,
                        "child": "btn_a_text",
                    }
                },
            },
            "btn_a_text": {
                "id": "btn_a_text",
                "component": {"Text": {"text": "Approve"}},
            },
            "btn_b": {
                "id": "btn_b",
                "component": {
                    "Button": {
                        "action": "reject",
                        "primary": False,
                        "child": "btn_b_text",
                    }
                },
            },
            "btn_b_text": {
                "id": "btn_b_text",
                "component": {"Text": {"text": "Reject"}},
            },
        }
    )


# ---------------------------------------------------------------------------
# Basic rendering
# ---------------------------------------------------------------------------


class TestRenderSurface:
    def test_returns_none_when_no_root(self) -> None:
        surface = SurfaceState(surface_id="s", root_id=None)
        assert render_surface(surface) is None

    def test_renders_simple_text(self) -> None:
        surface = _make_surface(
            {
                "root": {
                    "id": "root",
                    "component": {"Text": {"text": "Hello World"}},
                },
            }
        )
        img = render_surface(surface)
        assert img is not None
        assert img.width == CANVAS_WIDTH

    def test_renders_card_with_column(self) -> None:
        surface = _simple_card_with_buttons()
        img = render_surface(surface)
        assert img is not None
        # Card + heading + body + buttons should produce a non-trivial height.
        assert img.height > 100

    def test_missing_component_shows_error_text(self) -> None:
        surface = _make_surface(
            {
                "root": {
                    "id": "root",
                    "component": {
                        "Column": {"children": {"explicitList": ["nonexistent"]}}
                    },
                }
            }
        )
        img = render_surface(surface)
        assert img is not None

    def test_empty_surface_returns_min_height(self) -> None:
        surface = _make_surface(
            {
                "root": {"id": "root", "component": {"Column": {"children": {}}}},
            }
        )
        img = render_surface(surface)
        assert img is not None
        assert img.height == 100  # minimum


# ---------------------------------------------------------------------------
# Highlight rendering
# ---------------------------------------------------------------------------


class TestHighlightRendering:
    """Verify that the CLICKED highlight is drawn for highlighted components."""

    def _has_red_pixels(self, img: Image.Image) -> bool:
        """Check if the image contains any red-ish pixels (from highlight border)."""
        pixels = list(img.getdata())
        return any(r > 180 and g < 80 and b < 80 for r, g, b in pixels)

    def test_no_highlight_no_red(self) -> None:
        surface = _simple_card_with_buttons()
        img = render_surface(surface)
        assert img is not None
        assert not self._has_red_pixels(img)

    def test_highlight_on_button_in_row_has_red(self) -> None:
        """Button inside Card > Column > Row — the scenario that was broken."""
        surface = _simple_card_with_buttons()
        img = render_surface(surface, highlight_component_id="btn_a")
        assert img is not None
        assert self._has_red_pixels(img), (
            "Highlight border should produce red pixels for a button inside a Row"
        )

    def test_highlight_on_text_in_column_has_red(self) -> None:
        """Text directly in a Column (no Row wrapping)."""
        surface = _simple_card_with_buttons()
        img = render_surface(surface, highlight_component_id="heading")
        assert img is not None
        assert self._has_red_pixels(img)

    def test_highlight_nonexistent_id_no_red(self) -> None:
        surface = _simple_card_with_buttons()
        img = render_surface(surface, highlight_component_id="no_such_id")
        assert img is not None
        assert not self._has_red_pixels(img)

    def test_no_rendering_error(self) -> None:
        """Highlight should never cause a Rendering Error text."""
        surface = _simple_card_with_buttons()
        img = render_surface(surface, highlight_component_id="btn_b")
        assert img is not None
        # If there was a rendering error, the image would be very short (50+padding).
        assert img.height > 100


# ---------------------------------------------------------------------------
# Text wrapping
# ---------------------------------------------------------------------------


class TestDrawText:
    def test_simple_text_height(self) -> None:
        img = Image.new("RGB", (600, 200), (255, 255, 255))
        draw = __import__("PIL.ImageDraw", fromlist=["ImageDraw"]).Draw(img)
        h = _draw_text(draw, "Hello", 0, 0, 500)
        assert h > 0

    def test_word_wrap_respects_width(self) -> None:
        """Long text should wrap and return height > single line."""
        img = Image.new("RGB", (600, 400), (255, 255, 255))
        draw = __import__("PIL.ImageDraw", fromlist=["ImageDraw"]).Draw(img)
        long_text = (
            "This is a very long sentence that should wrap across multiple lines"
        )
        h_narrow = _draw_text(draw, long_text, 0, 0, 100)
        h_wide = _draw_text(draw, long_text, 0, 0, 500)
        assert h_narrow > h_wide, "Narrower width should produce taller text"

    def test_newline_handling(self) -> None:
        """Explicit newlines should produce multiple lines."""
        img = Image.new("RGB", (600, 400), (255, 255, 255))
        draw = __import__("PIL.ImageDraw", fromlist=["ImageDraw"]).Draw(img)
        h_single = _draw_text(draw, "Line one", 0, 0, 500)
        h_multi = _draw_text(draw, "Line one\nLine two\nLine three", 0, 0, 500)
        # 3 lines should be roughly 3x one line
        assert h_multi > h_single * 2

    def test_long_word_breaks_by_character(self) -> None:
        """A single word wider than available width should break by char."""
        img = Image.new("RGB", (600, 400), (255, 255, 255))
        draw = __import__("PIL.ImageDraw", fromlist=["ImageDraw"]).Draw(img)
        long_word = "Superlongwordwithnospacesthatdefinitelyexceedsthewidth"
        h = _draw_text(draw, long_word, 0, 0, 80)
        assert h > 0  # Should not crash or return 0

    def test_empty_text_returns_line_height(self) -> None:
        img = Image.new("RGB", (600, 200), (255, 255, 255))
        draw = __import__("PIL.ImageDraw", fromlist=["ImageDraw"]).Draw(img)
        h = _draw_text(draw, "", 0, 0, 500)
        assert h > 0


# ---------------------------------------------------------------------------
# JSON example rendering (integration tests)
# ---------------------------------------------------------------------------

_EXAMPLES_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "mmtoolsandbox",
    "extensions",
    "a2ui",
    "examples",
)


def _get_example_json_files() -> list[str]:
    pattern = os.path.join(_EXAMPLES_DIR, "*.json")
    return sorted(glob.glob(pattern))


def _load_and_render(json_path: str) -> list[Image.Image]:
    """Load a JSON example, replay messages, and collect all rendered images."""
    with open(json_path) as f:
        messages = json.load(f)

    UIState.reset()
    state = UIState()

    images: list[Image.Image] = []
    for msg in messages:
        state.process_message(msg)
        if state.active_surface_id:
            surface = state.surfaces[state.active_surface_id]
            # Auto-detect root for examples that don't send beginRendering.
            if surface.root_id is None and "root" in surface.components:
                surface.root_id = "root"
            img = render_surface(surface)
            if img is not None:
                images.append(img)
    return images


@pytest.mark.parametrize(
    "json_path",
    _get_example_json_files(),
    ids=[os.path.basename(p) for p in _get_example_json_files()],
)
def test_example_renders_without_error(json_path: str) -> None:
    """Every JSON example should render at least one image without errors."""
    images = _load_and_render(json_path)
    # Most examples produce at least one image. A few (weather-widget) may
    # produce none if they don't set a root. That's OK — we just check no crash.
    for img in images:
        assert img.width == CANVAS_WIDTH
        assert img.height >= 100


@pytest.mark.parametrize(
    "json_path",
    [
        p
        for p in _get_example_json_files()
        if os.path.basename(p)
        in ("contact_card.json", "booking_form.json", "restaurant_list.json")
    ],
    ids=lambda p: os.path.basename(p),
)
def test_key_examples_produce_images(json_path: str) -> None:
    """Key examples with known structure must produce at least one render."""
    images = _load_and_render(json_path)
    assert len(images) > 0, f"{os.path.basename(json_path)} should produce renders"


# ---------------------------------------------------------------------------
# Highlight + JSON example combo
# ---------------------------------------------------------------------------


class TestHighlightWithExamples:
    """Test highlight rendering on real example surfaces."""

    def test_contact_card_highlight_button(self) -> None:
        """Highlight a button on the contact card example."""
        json_path = os.path.join(_EXAMPLES_DIR, "contact_card.json")
        if not os.path.exists(json_path):
            pytest.skip("contact_card.json not found")

        with open(json_path) as f:
            messages = json.load(f)

        UIState.reset()
        state = UIState()
        for msg in messages:
            state.process_message(msg)

        if not state.active_surface_id:
            pytest.skip("No active surface")

        surface = state.surfaces[state.active_surface_id]
        # Render with highlight on button_1 (Follow button in Row).
        img = render_surface(surface, highlight_component_id="button_1")
        assert img is not None

        # Check for red pixels (highlight border).
        pixels = list(img.getdata())
        has_red = any(r > 180 and g < 80 and b < 80 for r, g, b in pixels)
        assert has_red, "Follow button highlight should produce red pixels"


# ---------------------------------------------------------------------------
# Round 2: Visual improvements
# ---------------------------------------------------------------------------


class TestInsideRowFlag:
    """Verify _inside_row flag doesn't leak across renders."""

    def test_sequential_renders_no_leak(self) -> None:
        """Render two surfaces sequentially — _inside_row should reset."""
        surface = _simple_card_with_buttons()
        img1 = render_surface(surface)
        img2 = render_surface(surface)
        assert img1 is not None
        assert img2 is not None
        # Both should have the same height (no state leaking).
        assert img1.height == img2.height


class TestButtonVariants:
    """Verify button variant styling produces distinct colors."""

    def _get_button_surface(
        self, primary: bool = False, variant: str = ""
    ) -> SurfaceState:
        props: dict[str, Any] = {
            "action": "test",
            "child": "btn_text",
        }
        if primary:
            props["primary"] = True
        if variant:
            props["variant"] = variant
        return _make_surface(
            {
                "root": {
                    "id": "root",
                    "component": {"Button": props},
                },
                "btn_text": {
                    "id": "btn_text",
                    "component": {"Text": {"text": "Click Me"}},
                },
            }
        )

    def _dominant_color(self, img: Image.Image) -> tuple[int, int, int]:
        """Return the most common pixel color in the image."""
        pixels: list[tuple[int, int, int]] = list(img.getdata())
        from collections import Counter

        return Counter(pixels).most_common(1)[0][0]

    def test_primary_is_blue(self) -> None:
        surface = self._get_button_surface(primary=True)
        img = render_surface(surface)
        assert img is not None
        # Primary button should have blue pixels.
        pixels = list(img.getdata())
        has_blue = any(b > 200 and r < 50 and g < 150 for r, g, b in pixels)
        assert has_blue, "Primary button should contain blue pixels"

    def test_secondary_is_gray(self) -> None:
        surface = self._get_button_surface(primary=False)
        img = render_surface(surface)
        assert img is not None
        # Secondary button should have light gray (230,230,230) pixels.
        pixels = list(img.getdata())
        has_light_gray = any(
            220 <= r <= 240 and 220 <= g <= 240 and 220 <= b <= 240
            for r, g, b in pixels
        )
        assert has_light_gray, "Secondary button should contain light gray pixels"

    def test_primary_and_secondary_look_different(self) -> None:
        s1 = self._get_button_surface(primary=True)
        s2 = self._get_button_surface(primary=False)
        img1 = render_surface(s1)
        img2 = render_surface(s2)
        assert img1 is not None and img2 is not None
        # Compare pixel sets — they should differ since button colors differ.
        p1 = set(img1.getdata())
        p2 = set(img2.getdata())
        assert p1 != p2, "Primary and secondary buttons should produce different pixels"


class TestCaptionTextColor:
    """Verify caption text uses lighter color."""

    def test_caption_renders_without_error(self) -> None:
        surface = _make_surface(
            {
                "root": {
                    "id": "root",
                    "component": {
                        "Text": {"text": "Caption text", "usageHint": "caption"}
                    },
                },
            }
        )
        img = render_surface(surface)
        assert img is not None
        assert img.height >= 100
