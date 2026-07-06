"""
Tests for the A2UI extension.

Covers:
- State management (WP-2: nested data model, path resolution)
- Renderer (WP-3: path resolution, WP-4: new components)
- Discovery tools (WP-5: examples in returns, WP-6: quick start)
- Tool integration (WP-1: database_namespaces, existing tests preserved)

Requires the A2UI extension (Pillow).  Skipped when not installed.
"""

import json
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

pytest = __import__("pytest")
PIL = pytest.importorskip(
    "PIL", reason="A2UI extension not installed (requires Pillow)"
)
from PIL import Image  # noqa: E402

from mmtoolsandbox.a2ui.docs import (  # noqa: E402
    _DEFINITIONS_CACHE,
    COMPONENT_EXAMPLES,
    get_quick_start,
    get_ui_item_details,
    get_ui_items_in_category,
)
from mmtoolsandbox.a2ui.html_renderer import (  # noqa: E402
    render_surface_to_html,
)
from mmtoolsandbox.a2ui.renderer import (  # noqa: E402
    _get_value,
    _resolve_path,
    render_surface,
)
from mmtoolsandbox.a2ui.state import (  # noqa: E402
    SurfaceState,
    UIState,
    _parse_pointer_segments,
    _resolve_contents,
    _set_at_path,
    get_interactive_elements,
    get_ui_state,
)
from mmtoolsandbox.a2ui.tools import (  # noqa: E402
    render_ui_screen,
    show_ui_to_user,
    ui_explore_capabilities,
    ui_get_item_details,
    ui_get_quick_start,
    ui_list_items,
    ui_search_docs,
    ui_user_interact,
)
from mmtoolsandbox.common.execution_context import RoleType  # noqa: E402
from mmtoolsandbox.common.image_id import ImageId, ImageResult  # noqa: E402

# ============================================================================
# WP-2: State — nested data model processing
# ============================================================================


class TestResolveContents(unittest.TestCase):
    """Test the recursive data model builder for v0.8 protocol contents."""

    def test_flat_string_values(self) -> None:
        contents = [
            {"key": "name", "valueString": "Alice"},
            {"key": "email", "valueString": "alice@example.com"},
        ]
        result = _resolve_contents(contents)
        self.assertEqual(result, {"name": "Alice", "email": "alice@example.com"})

    def test_mixed_value_types(self) -> None:
        contents: list[dict[str, Any]] = [
            {"key": "name", "valueString": "Bob"},
            {"key": "age", "valueNumber": 30},
            {"key": "active", "valueBoolean": True},
        ]
        result = _resolve_contents(contents)
        self.assertEqual(result, {"name": "Bob", "age": 30, "active": True})

    def test_nested_value_map(self) -> None:
        """The core v0.8 pattern: valueMap for hierarchical data."""
        contents = [
            {
                "key": "contacts",
                "valueMap": [
                    {
                        "key": "contact1",
                        "valueMap": [
                            {"key": "name", "valueString": "Alice"},
                            {"key": "phone", "valueString": "+1-555-0001"},
                        ],
                    },
                    {
                        "key": "contact2",
                        "valueMap": [
                            {"key": "name", "valueString": "Bob"},
                            {"key": "phone", "valueString": "+1-555-0002"},
                        ],
                    },
                ],
            }
        ]
        result = _resolve_contents(contents)
        self.assertEqual(
            result,
            {
                "contacts": {
                    "contact1": {"name": "Alice", "phone": "+1-555-0001"},
                    "contact2": {"name": "Bob", "phone": "+1-555-0002"},
                }
            },
        )

    def test_empty_contents(self) -> None:
        self.assertEqual(_resolve_contents([]), {})

    def test_entry_without_value_key_is_skipped(self) -> None:
        contents = [{"key": "orphan"}]
        self.assertEqual(_resolve_contents(contents), {})

    def test_entry_without_key_is_skipped(self) -> None:
        contents = [{"valueString": "no key"}]
        self.assertEqual(_resolve_contents(contents), {})


class TestParsePointerSegments(unittest.TestCase):
    """Test the shared JSON Pointer segment parser."""

    def test_absolute_path(self) -> None:
        self.assertEqual(_parse_pointer_segments("/user/name"), ["user", "name"])

    def test_relative_path(self) -> None:
        self.assertEqual(_parse_pointer_segments("name"), ["name"])

    def test_root_path_returns_empty(self) -> None:
        self.assertEqual(_parse_pointer_segments("/"), [])

    def test_empty_string_returns_empty(self) -> None:
        self.assertEqual(_parse_pointer_segments(""), [])

    def test_trailing_slash_stripped(self) -> None:
        self.assertEqual(_parse_pointer_segments("/a/b/"), ["a", "b"])


class TestSetAtPath(unittest.TestCase):
    """Test JSON Pointer path setting in the data model."""

    def test_simple_path(self) -> None:
        data: dict[str, Any] = {}
        _set_at_path(data, "/name", "Alice")
        self.assertEqual(data, {"name": "Alice"})

    def test_nested_path_creates_intermediates(self) -> None:
        data: dict[str, Any] = {}
        _set_at_path(data, "/user/profile/name", "Bob")
        self.assertEqual(data, {"user": {"profile": {"name": "Bob"}}})

    def test_empty_path_is_noop(self) -> None:
        data: dict[str, Any] = {"existing": True}
        _set_at_path(data, "/", {"replaced": True})
        # Empty segments → noop; caller handles root replacement
        self.assertIn("existing", data)


class TestUIStateProcessMessage(unittest.TestCase):
    """Test that UIState.process_message handles v0.8 protocol correctly."""

    def setUp(self) -> None:
        UIState.reset()

    def test_begin_rendering(self) -> None:
        state = get_ui_state()
        state.process_message(
            {"beginRendering": {"surfaceId": "s1", "root": "root-comp"}}
        )
        self.assertIn("s1", state.surfaces)
        self.assertEqual(state.surfaces["s1"].root_id, "root-comp")
        self.assertEqual(state.active_surface_id, "s1")

    def test_surface_update(self) -> None:
        state = get_ui_state()
        state.process_message({"beginRendering": {"surfaceId": "s1", "root": "col1"}})
        state.process_message(
            {
                "surfaceUpdate": {
                    "surfaceId": "s1",
                    "components": [
                        {
                            "id": "col1",
                            "component": {
                                "Column": {"children": {"explicitList": ["t1"]}}
                            },
                        },
                        {
                            "id": "t1",
                            "component": {"Text": {"text": {"literalString": "Hello"}}},
                        },
                    ],
                }
            }
        )
        self.assertIn("col1", state.surfaces["s1"].components)
        self.assertIn("t1", state.surfaces["s1"].components)

    def test_data_model_update_with_nested_value_map(self) -> None:
        """Verify WP-2: nested valueMap is resolved correctly."""
        state = get_ui_state()
        state.process_message({"beginRendering": {"surfaceId": "s1", "root": "root"}})
        state.process_message(
            {
                "dataModelUpdate": {
                    "surfaceId": "s1",
                    "path": "/",
                    "contents": [
                        {"key": "title", "valueString": "Contacts"},
                        {
                            "key": "contacts",
                            "valueMap": [
                                {
                                    "key": "c1",
                                    "valueMap": [
                                        {"key": "name", "valueString": "Alice"},
                                    ],
                                }
                            ],
                        },
                    ],
                }
            }
        )
        dm = state.surfaces["s1"].data_model
        self.assertEqual(dm["title"], "Contacts")
        self.assertEqual(dm["contacts"]["c1"]["name"], "Alice")

    def test_delete_surface(self) -> None:
        state = get_ui_state()
        state.process_message({"beginRendering": {"surfaceId": "s1", "root": "root"}})
        self.assertIn("s1", state.surfaces)
        state.process_message({"deleteSurface": {"surfaceId": "s1"}})
        self.assertNotIn("s1", state.surfaces)
        self.assertIsNone(state.active_surface_id)

    def test_process_message_list(self) -> None:
        """Process a list of messages (batch)."""
        state = get_ui_state()
        state.process_message(
            [
                {"beginRendering": {"surfaceId": "s1", "root": "r"}},
                {
                    "surfaceUpdate": {
                        "surfaceId": "s1",
                        "components": [
                            {
                                "id": "r",
                                "component": {
                                    "Text": {"text": {"literalString": "Hi"}}
                                },
                            }
                        ],
                    }
                },
            ]
        )
        self.assertIn("s1", state.surfaces)
        self.assertIn("r", state.surfaces["s1"].components)


# ============================================================================
# WP-3: Renderer — path resolution
# ============================================================================


class TestResolvePath(unittest.TestCase):
    """Test JSON Pointer path resolution for the renderer."""

    def test_simple_key(self) -> None:
        self.assertEqual(_resolve_path({"name": "Alice"}, "name"), "Alice")

    def test_nested_path(self) -> None:
        data = {"user": {"profile": {"name": "Bob"}}}
        self.assertEqual(_resolve_path(data, "/user/profile/name"), "Bob")

    def test_leading_slash_stripped(self) -> None:
        data = {"name": "Alice"}
        self.assertEqual(_resolve_path(data, "/name"), "Alice")

    def test_list_index(self) -> None:
        data = {"items": ["a", "b", "c"]}
        self.assertEqual(_resolve_path(data, "/items/1"), "b")

    def test_missing_key_returns_placeholder(self) -> None:
        result = _resolve_path({}, "/missing/key")
        self.assertIn("[", result)

    def test_empty_path_returns_data(self) -> None:
        data = {"x": 1}
        self.assertEqual(_resolve_path(data, ""), data)


class TestGetValue(unittest.TestCase):
    """Test BoundValue resolution (literalString, path, etc.)."""

    def test_literal_string(self) -> None:
        self.assertEqual(_get_value({"literalString": "hello"}, {}), "hello")

    def test_literal_number(self) -> None:
        self.assertEqual(_get_value({"literalNumber": 42}, {}), "42")

    def test_literal_boolean(self) -> None:
        self.assertEqual(_get_value({"literalBoolean": True}, {}), "True")

    def test_path_resolution(self) -> None:
        dm = {"user": {"name": "Alice"}}
        self.assertEqual(_get_value({"path": "/user/name"}, dm), "Alice")

    def test_path_missing_returns_placeholder(self) -> None:
        result = _get_value({"path": "/missing"}, {})
        self.assertIn("[", result)

    def test_plain_string_passthrough(self) -> None:
        self.assertEqual(_get_value("direct", {}), "direct")


# ============================================================================
# WP-4: Renderer — component coverage
# ============================================================================


class TestRendererComponents(unittest.TestCase):
    """Test that the renderer handles additional protocol components."""

    def _make_surface(
        self, components: list[dict[str, Any]], data: dict[str, Any] | None = None
    ) -> SurfaceState:
        surface = SurfaceState(surface_id="test", root_id="root")
        for comp in components:
            surface.components[comp["id"]] = comp
        if data:
            surface.data_model = data
        return surface

    def test_render_card(self) -> None:
        surface = self._make_surface(
            [
                {"id": "root", "component": {"Card": {"child": "inner"}}},
                {
                    "id": "inner",
                    "component": {"Text": {"text": {"literalString": "Card content"}}},
                },
            ]
        )
        img = render_surface(surface)
        self.assertIsNotNone(img)
        self.assertGreater(img.height, 0)  # type: ignore[union-attr]

    def test_render_divider(self) -> None:
        surface = self._make_surface(
            [
                {
                    "id": "root",
                    "component": {"Column": {"children": {"explicitList": ["d1"]}}},
                },
                {"id": "d1", "component": {"Divider": {"axis": "horizontal"}}},
            ]
        )
        img = render_surface(surface)
        self.assertIsNotNone(img)

    def test_render_checkbox(self) -> None:
        surface = self._make_surface(
            [
                {
                    "id": "root",
                    "component": {"Column": {"children": {"explicitList": ["cb"]}}},
                },
                {
                    "id": "cb",
                    "component": {
                        "CheckBox": {
                            "label": {"literalString": "Accept terms"},
                            "value": {"literalBoolean": True},
                        }
                    },
                },
            ]
        )
        img = render_surface(surface)
        self.assertIsNotNone(img)

    def test_render_slider(self) -> None:
        surface = self._make_surface(
            [
                {
                    "id": "root",
                    "component": {"Column": {"children": {"explicitList": ["sl"]}}},
                },
                {
                    "id": "sl",
                    "component": {
                        "Slider": {
                            "value": {"literalNumber": 50},
                            "minValue": 0,
                            "maxValue": 100,
                        }
                    },
                },
            ]
        )
        img = render_surface(surface)
        self.assertIsNotNone(img)

    def test_render_icon(self) -> None:
        surface = self._make_surface(
            [
                {
                    "id": "root",
                    "component": {"Column": {"children": {"explicitList": ["ic"]}}},
                },
                {
                    "id": "ic",
                    "component": {"Icon": {"name": {"literalString": "star"}}},
                },
            ]
        )
        img = render_surface(surface)
        self.assertIsNotNone(img)

    def test_render_button_white_text(self) -> None:
        """WP-4: Button text should be white (not black-on-blue)."""
        surface = self._make_surface(
            [
                {
                    "id": "root",
                    "component": {"Column": {"children": {"explicitList": ["btn"]}}},
                },
                {
                    "id": "btn",
                    "component": {
                        "Button": {
                            "child": "btn-text",
                            "primary": True,
                            "action": {"name": "click"},
                        }
                    },
                },
                {
                    "id": "btn-text",
                    "component": {"Text": {"text": {"literalString": "Click"}}},
                },
            ]
        )
        img = render_surface(surface)
        self.assertIsNotNone(img)

    def test_render_list_with_template(self) -> None:
        """WP-4: Template-based list driven by data model."""
        surface = self._make_surface(
            components=[
                {
                    "id": "root",
                    "component": {
                        "List": {
                            "direction": "vertical",
                            "children": {
                                "template": {
                                    "componentId": "item-tmpl",
                                    "dataBinding": "/items",
                                }
                            },
                        }
                    },
                },
                {"id": "item-tmpl", "component": {"Text": {"text": {"path": "name"}}}},
            ],
            data={
                "items": {
                    "i1": {"name": "First Item"},
                    "i2": {"name": "Second Item"},
                },
            },
        )
        img = render_surface(surface)
        self.assertIsNotNone(img)
        self.assertGreater(img.height, 50)  # type: ignore[union-attr]

    def test_render_multiple_choice(self) -> None:
        surface = self._make_surface(
            [
                {
                    "id": "root",
                    "component": {"Column": {"children": {"explicitList": ["mc"]}}},
                },
                {
                    "id": "mc",
                    "component": {
                        "MultipleChoice": {
                            "selections": {"literalArray": ["a"]},
                            "options": [
                                {"label": {"literalString": "Option A"}, "value": "a"},
                                {"label": {"literalString": "Option B"}, "value": "b"},
                            ],
                        }
                    },
                },
            ]
        )
        img = render_surface(surface)
        self.assertIsNotNone(img)

    def test_render_unknown_component_placeholder(self) -> None:
        """Unknown components should render a placeholder, not crash."""
        surface = self._make_surface(
            [
                {"id": "root", "component": {"FutureWidget": {"foo": "bar"}}},
            ]
        )
        img = render_surface(surface)
        self.assertIsNotNone(img)

    def test_render_no_root_returns_none(self) -> None:
        surface = SurfaceState(surface_id="test")
        self.assertIsNone(render_surface(surface))


# ============================================================================
# Stream 2: Interactive elements extraction
# ============================================================================


class TestGetInteractiveElements(unittest.TestCase):
    """Test extraction of interactive elements from a surface."""

    def _make_surface(
        self, components: list[dict[str, Any]], surface_id: str = "test"
    ) -> SurfaceState:
        surface = SurfaceState(surface_id=surface_id, root_id="root")
        for comp in components:
            surface.components[comp["id"]] = comp
        return surface

    def test_extracts_button(self) -> None:
        surface = self._make_surface(
            [
                {
                    "id": "root",
                    "component": {"Column": {"children": {"explicitList": ["btn"]}}},
                },
                {
                    "id": "btn",
                    "component": {
                        "Button": {
                            "child": "btn-text",
                            "action": {"name": "submit"},
                        }
                    },
                },
                {
                    "id": "btn-text",
                    "component": {"Text": {"text": {"literalString": "Submit"}}},
                },
            ]
        )
        result = get_interactive_elements(surface)
        self.assertIn("Button", result)
        self.assertIn("submit", result)
        self.assertIn("Submit", result)

    def test_extracts_text_field(self) -> None:
        surface = self._make_surface(
            [
                {
                    "id": "root",
                    "component": {"Column": {"children": {"explicitList": ["tf"]}}},
                },
                {
                    "id": "tf",
                    "component": {
                        "TextField": {
                            "label": {"literalString": "Name"},
                            "text": {"path": "/userName"},
                        }
                    },
                },
            ]
        )
        result = get_interactive_elements(surface)
        self.assertIn("TextField", result)
        self.assertIn("Name", result)
        self.assertIn("/userName", result)

    def test_extracts_checkbox(self) -> None:
        surface = self._make_surface(
            [
                {
                    "id": "root",
                    "component": {"Column": {"children": {"explicitList": ["cb"]}}},
                },
                {
                    "id": "cb",
                    "component": {
                        "CheckBox": {
                            "label": {"literalString": "Agree"},
                            "value": {"path": "/agree"},
                        }
                    },
                },
            ]
        )
        result = get_interactive_elements(surface)
        self.assertIn("CheckBox", result)
        self.assertIn("Agree", result)

    def test_empty_for_no_interactive_elements(self) -> None:
        surface = self._make_surface(
            [
                {
                    "id": "root",
                    "component": {"Text": {"text": {"literalString": "Hello"}}},
                },
            ]
        )
        result = get_interactive_elements(surface)
        self.assertEqual(result, "")

    def test_includes_user_action_guidance(self) -> None:
        surface = self._make_surface(
            [
                {
                    "id": "root",
                    "component": {"Column": {"children": {"explicitList": ["btn"]}}},
                },
                {
                    "id": "btn",
                    "component": {
                        "Button": {
                            "child": "btn-text",
                            "action": {"name": "click"},
                        }
                    },
                },
                {
                    "id": "btn-text",
                    "component": {"Text": {"text": {"literalString": "Go"}}},
                },
            ]
        )
        result = get_interactive_elements(surface)
        self.assertIn("ui_user_interact", result)
        self.assertIn("action_name", result)

    def test_includes_surface_id(self) -> None:
        surface = self._make_surface(
            [
                {
                    "id": "root",
                    "component": {"Column": {"children": {"explicitList": ["btn"]}}},
                },
                {
                    "id": "btn",
                    "component": {
                        "Button": {
                            "child": "btn-text",
                            "action": {"name": "go"},
                        }
                    },
                },
                {
                    "id": "btn-text",
                    "component": {"Text": {"text": {"literalString": "Go"}}},
                },
            ],
            surface_id="my-surface",
        )
        result = get_interactive_elements(surface)
        self.assertIn("my-surface", result)


# ============================================================================
# Stream 1: HTML renderer
# ============================================================================


class TestHtmlRenderer(unittest.TestCase):
    """Test HTML rendering of A2UI surfaces."""

    def _make_surface(
        self,
        components: list[dict[str, Any]],
        data: dict[str, Any] | None = None,
        surface_id: str = "test-surface",
    ) -> SurfaceState:
        surface = SurfaceState(surface_id=surface_id, root_id="root")
        for comp in components:
            surface.components[comp["id"]] = comp
        if data:
            surface.data_model = data
        return surface

    def test_returns_html_string(self) -> None:
        surface = self._make_surface(
            [
                {
                    "id": "root",
                    "component": {"Text": {"text": {"literalString": "Hello"}}},
                },
            ]
        )
        result = render_surface_to_html(surface)
        self.assertIsInstance(result, str)
        self.assertIn("<!DOCTYPE html>", result)

    def test_contains_a2ui_surface_element(self) -> None:
        surface = self._make_surface(
            [
                {
                    "id": "root",
                    "component": {"Text": {"text": {"literalString": "Hello"}}},
                },
            ]
        )
        result = render_surface_to_html(surface)
        self.assertIn("a2ui-surface", result)

    def test_contains_inlined_messages(self) -> None:
        """The A2UI JSON messages should be inlined in the HTML as data."""
        surface = self._make_surface(
            [
                {
                    "id": "root",
                    "component": {"Column": {"children": {"explicitList": ["t1"]}}},
                },
                {
                    "id": "t1",
                    "component": {"Text": {"text": {"literalString": "Test"}}},
                },
            ],
            data={"title": "My Title"},
        )
        result = render_surface_to_html(surface)
        # The component definitions should be present in the HTML
        self.assertIn("test-surface", result)
        self.assertIn("root", result)

    def test_contains_import_map(self) -> None:
        """HTML should include ES module import maps for Lit dependencies."""
        surface = self._make_surface(
            [
                {
                    "id": "root",
                    "component": {"Text": {"text": {"literalString": "Hi"}}},
                },
            ]
        )
        result = render_surface_to_html(surface)
        self.assertIn("importmap", result)
        self.assertIn("lit", result)

    def test_contains_surface_id(self) -> None:
        surface = self._make_surface(
            [{"id": "root", "component": {"Text": {"text": {"literalString": "Hi"}}}}],
            surface_id="my-card",
        )
        result = render_surface_to_html(surface)
        self.assertIn("my-card", result)

    def test_custom_a2ui_base_path(self) -> None:
        """Should support configuring where the pre-built Lit JS is served from."""
        surface = self._make_surface(
            [
                {
                    "id": "root",
                    "component": {"Text": {"text": {"literalString": "Hi"}}},
                },
            ]
        )
        result = render_surface_to_html(surface, a2ui_base_url="/custom/path")
        self.assertIn("/custom/path", result)

    def test_none_surface_returns_empty(self) -> None:
        surface = SurfaceState(surface_id="empty")
        result = render_surface_to_html(surface)
        self.assertEqual(result, "")


# ============================================================================
# WP-5: Docs — examples in component detail returns
# ============================================================================


class TestComponentExamples(unittest.TestCase):
    """Test that component detail returns include usage examples."""

    def setUp(self) -> None:
        _DEFINITIONS_CACHE.clear()

    def test_button_details_include_example(self) -> None:
        details = get_ui_item_details("Components", "Button")
        self.assertIn("--- Minimal Usage Example ---", details)
        self.assertIn("btn1", details)
        self.assertIn("action", details)

    def test_text_details_include_example(self) -> None:
        details = get_ui_item_details("Components", "Text")
        self.assertIn("--- Minimal Usage Example ---", details)
        self.assertIn("literalString", details)

    def test_column_details_include_example(self) -> None:
        details = get_ui_item_details("Components", "Column")
        self.assertIn("--- Minimal Usage Example ---", details)
        self.assertIn("explicitList", details)

    def test_all_catalog_components_have_examples(self) -> None:
        """Every component in the catalog should have a COMPONENT_EXAMPLES entry."""
        items = get_ui_items_in_category("Components")
        component_names = [item.split(":")[0].strip() for item in items]
        for name in component_names:
            self.assertIn(
                name,
                COMPONENT_EXAMPLES,
                f"Component '{name}' is missing from COMPONENT_EXAMPLES",
            )


# ============================================================================
# WP-6: Quick Start tool
# ============================================================================


class TestQuickStart(unittest.TestCase):
    """Test the ui_get_quick_start tool."""

    def setUp(self) -> None:
        _DEFINITIONS_CACHE.clear()

    def test_quick_start_contains_protocol_overview(self) -> None:
        result = get_quick_start()
        self.assertIn("beginRendering", result)
        self.assertIn("surfaceUpdate", result)
        self.assertIn("dataModelUpdate", result)

    def test_quick_start_contains_example(self) -> None:
        result = get_quick_start()
        self.assertIn("booking-form", result)

    def test_quick_start_contains_component_list(self) -> None:
        result = get_quick_start()
        self.assertIn("Button", result)
        self.assertIn("Column", result)
        self.assertIn("Text", result)

    def test_quick_start_contains_key_patterns(self) -> None:
        result = get_quick_start()
        self.assertIn("literalString", result)
        self.assertIn("explicitList", result)
        self.assertIn("valueMap", result)


# ============================================================================
# Existing tool tests (preserved + augmented)
# ============================================================================


class TestToolIntegration(unittest.TestCase):
    """Integration tests for the registered tools."""

    def setUp(self) -> None:
        UIState.reset()
        _DEFINITIONS_CACHE.clear()

    @patch("mmtoolsandbox.a2ui.tools.store_image")
    @patch("mmtoolsandbox.a2ui.tools.render_surface")
    def test_render_ui_screen_success(
        self, mock_render_surface: MagicMock, mock_store_image: MagicMock
    ) -> None:
        mock_image = MagicMock()
        mock_render_surface.return_value = mock_image
        mock_store_image.return_value = ImageResult(ImageId(123))

        message = {
            "beginRendering": {
                "surfaceId": "surface-1",
                "root": "root-component-id",
                "catalogId": "example.com:catalog-1",
            },
        }
        result = render_ui_screen(json.dumps(message))
        self.assertIsInstance(result, dict)
        self.assertEqual(result["image"].image_id, ImageId(123))

        state = get_ui_state()
        self.assertIn("surface-1", state.surfaces)
        self.assertEqual(state.surfaces["surface-1"].root_id, "root-component-id")

    def test_render_ui_screen_invalid_json(self) -> None:
        result = render_ui_screen("{invalid json")
        self.assertIn("Error: Invalid JSON string", result)

    def test_render_ui_screen_invalid_schema(self) -> None:
        message = {"beginRendering": {"surfaceId": "surface-1"}}
        result = render_ui_screen(json.dumps(message))
        self.assertIn("Error: UI description does not match UI schema", result)

    @patch("mmtoolsandbox.a2ui.tools.add_messages_to_execution_context")
    @patch("mmtoolsandbox.a2ui.tools.get_current_context")
    def test_ui_user_interact_valid(
        self, mock_get_ctx: MagicMock, mock_add_msgs: MagicMock
    ) -> None:
        # Set up a surface with a button so the tool can find it
        state = get_ui_state()
        state.process_message(
            [
                {"beginRendering": {"surfaceId": "surface-1", "root": "root"}},
                {
                    "surfaceUpdate": {
                        "surfaceId": "surface-1",
                        "components": [
                            {
                                "id": "root",
                                "component": {
                                    "Column": {
                                        "children": {"explicitList": ["button-1"]}
                                    }
                                },
                            },
                            {
                                "id": "button-1",
                                "component": {
                                    "Button": {
                                        "child": "btn-text",
                                        "action": {"name": "click", "context": []},
                                    }
                                },
                            },
                            {
                                "id": "btn-text",
                                "component": {
                                    "Text": {"text": {"literalString": "Go"}}
                                },
                            },
                        ],
                    }
                },
            ]
        )
        result = ui_user_interact(
            action_name="click",
            surface_id="surface-1",
            component_id="button-1",
        )
        self.assertIn("User Interaction on surface", result)
        self.assertIn("Action: click", result)

        self.assertEqual(len(state.interaction_history), 1)
        # Verify agent notification was injected
        mock_add_msgs.assert_called_once()

    def test_ui_user_interact_surface_not_found(self) -> None:
        result = ui_user_interact(
            action_name="click",
            surface_id="nonexistent",
            component_id="button-1",
        )
        self.assertIn("Error: Surface 'nonexistent' not found", result)

    @patch("mmtoolsandbox.a2ui.tools.add_messages_to_execution_context")
    @patch("mmtoolsandbox.a2ui.tools.get_current_context")
    def test_ui_user_interact_with_field_values(
        self, mock_get_ctx: MagicMock, mock_add_msgs: MagicMock
    ) -> None:
        state = get_ui_state()
        state.process_message(
            [
                {"beginRendering": {"surfaceId": "form-1", "root": "root"}},
                {
                    "surfaceUpdate": {
                        "surfaceId": "form-1",
                        "components": [
                            {
                                "id": "root",
                                "component": {
                                    "Column": {
                                        "children": {
                                            "explicitList": ["name-field", "submit-btn"]
                                        }
                                    }
                                },
                            },
                            {
                                "id": "name-field",
                                "component": {
                                    "TextField": {
                                        "label": {"literalString": "Name"},
                                        "text": {"path": "userName"},
                                    }
                                },
                            },
                            {
                                "id": "submit-btn",
                                "component": {
                                    "Button": {
                                        "child": "btn-text",
                                        "action": {
                                            "name": "submit",
                                            "context": [
                                                {
                                                    "key": "userName",
                                                    "value": {"path": "userName"},
                                                }
                                            ],
                                        },
                                    }
                                },
                            },
                            {
                                "id": "btn-text",
                                "component": {
                                    "Text": {"text": {"literalString": "Submit"}}
                                },
                            },
                        ],
                    }
                },
                {
                    "dataModelUpdate": {
                        "surfaceId": "form-1",
                        "path": "/",
                        "contents": [
                            {"key": "userName", "valueString": ""},
                        ],
                    }
                },
            ]
        )
        result = ui_user_interact(
            action_name="submit",
            surface_id="form-1",
            component_id="submit-btn",
            field_values={"name-field": "Alice"},
        )
        self.assertIn("Action: submit", result)
        self.assertIn("userName: Alice", result)
        # Verify data model was updated
        self.assertEqual(state.surfaces["form-1"].data_model["userName"], "Alice")

    @patch("mmtoolsandbox.a2ui.tools.add_messages_to_execution_context")
    @patch("mmtoolsandbox.a2ui.tools.get_current_context")
    def test_ui_user_interact_injects_agent_notification(
        self, mock_get_ctx: MagicMock, mock_add_msgs: MagicMock
    ) -> None:
        """Verify that ui_user_interact sends an EXEC_ENV→AGENT notification."""
        from mmtoolsandbox.common.execution_context import RoleType

        state = get_ui_state()
        state.process_message(
            [
                {"beginRendering": {"surfaceId": "form-1", "root": "root"}},
                {
                    "surfaceUpdate": {
                        "surfaceId": "form-1",
                        "components": [
                            {
                                "id": "root",
                                "component": {
                                    "Column": {
                                        "children": {
                                            "explicitList": ["name-field", "submit-btn"]
                                        }
                                    }
                                },
                            },
                            {
                                "id": "name-field",
                                "component": {
                                    "TextField": {
                                        "label": {"literalString": "Name"},
                                        "text": {"path": "userName"},
                                    }
                                },
                            },
                            {
                                "id": "submit-btn",
                                "component": {
                                    "Button": {
                                        "child": "btn-text",
                                        "action": {
                                            "name": "submit",
                                            "context": [
                                                {
                                                    "key": "userName",
                                                    "value": {"path": "userName"},
                                                }
                                            ],
                                        },
                                    }
                                },
                            },
                            {
                                "id": "btn-text",
                                "component": {
                                    "Text": {"text": {"literalString": "Submit"}}
                                },
                            },
                        ],
                    }
                },
                {
                    "dataModelUpdate": {
                        "surfaceId": "form-1",
                        "path": "/",
                        "contents": [{"key": "userName", "valueString": ""}],
                    }
                },
            ]
        )
        ui_user_interact(
            action_name="submit",
            surface_id="form-1",
            component_id="submit-btn",
            field_values={"name-field": "Bob"},
        )

        # Verify notification was injected
        mock_add_msgs.assert_called_once()
        injected_messages = mock_add_msgs.call_args[0][1]
        self.assertEqual(len(injected_messages), 1)

        msg = injected_messages[0]
        self.assertEqual(msg.sender, RoleType.EXECUTION_ENVIRONMENT)
        self.assertEqual(msg.recipient, RoleType.AGENT)
        self.assertEqual(msg.visible_to, [RoleType.AGENT])
        self.assertIsNone(msg.image_ids)
        self.assertIn("<ui_state_server>", msg.content)
        self.assertIn("</ui_state_server>", msg.content)
        self.assertIn("submit", msg.content)
        self.assertIn("userName: Bob", msg.content)

    def test_ui_explore_capabilities(self) -> None:
        categories = ui_explore_capabilities()
        self.assertIn("Components", categories)
        self.assertIn("Concepts", categories)
        self.assertIn("Examples", categories)

    def test_ui_list_items(self) -> None:
        components = ui_list_items("Components")
        self.assertTrue(len(components) > 0)
        self.assertTrue(any("Button" in c for c in components))

    def test_ui_get_item_details(self) -> None:
        details = ui_get_item_details("Components", "Button")
        self.assertIn("action", details)
        self.assertIn("properties", details)

    def test_ui_search_docs(self) -> None:
        results = ui_search_docs("list")
        self.assertIn("List", results)

    def test_examples_loading(self) -> None:
        items = ui_list_items("Examples")
        for expected in ["booking_form", "contact_list", "contact_card"]:
            self.assertTrue(
                any(expected in item for item in items),
                f"Example '{expected}' not found",
            )
        details = ui_get_item_details("Examples", "booking_form")
        self.assertIn("booking-form", details)

    @patch("mmtoolsandbox.a2ui.tools.store_image")
    @patch("mmtoolsandbox.a2ui.tools.load_image")
    @patch("mmtoolsandbox.a2ui.tools.render_surface")
    def test_render_with_placeholders(
        self,
        mock_render_surface: MagicMock,
        mock_load_image: MagicMock,
        mock_store_image: MagicMock,
    ) -> None:
        dummy_image = Image.new("RGB", (10, 10), color="red")
        mock_load_image.return_value = dummy_image

        rendered_image = Image.new("RGB", (100, 100), color="blue")
        mock_render_surface.return_value = rendered_image

        mock_store_image.return_value = ImageResult(ImageId(456))

        ui_json = json.dumps(
            [
                {
                    "beginRendering": {
                        "surfaceId": "s1",
                        "root": "img1",
                        "catalogId": "example:catalog",
                    }
                },
                {
                    "surfaceUpdate": {
                        "surfaceId": "s1",
                        "components": [
                            {
                                "id": "img1",
                                "component": {
                                    "Image": {"url": {"literalString": "IMAGE_URL_1"}}
                                },
                            }
                        ],
                    }
                },
            ]
        )

        result = render_ui_screen(
            ui_json=ui_json, image_placeholders={"IMAGE_URL_1": 123}
        )
        mock_load_image.assert_called_with(123)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["image"].image_id, ImageId(456))

    def test_ui_get_quick_start_tool(self) -> None:
        """Test the registered quick start tool."""
        result = ui_get_quick_start()
        self.assertIn("Quick Start", result)
        self.assertIn("booking-form", result)
        self.assertIn("beginRendering", result)

    @patch("mmtoolsandbox.a2ui.tools.add_messages_to_execution_context")
    @patch("mmtoolsandbox.a2ui.tools.get_current_context")
    def test_notify_agent_uses_xml_tag_no_image(
        self, mock_get_ctx: MagicMock, mock_add_msgs: MagicMock
    ) -> None:
        """Agent notification must use <ui_state_server> tag and NOT include images."""
        from mmtoolsandbox.a2ui.tools import _notify_agent_of_interaction

        _notify_agent_of_interaction(
            action_name="submit_rsvp",
            resolved_context={"guestName": "Alice", "guestCount": "2"},
        )

        mock_add_msgs.assert_called_once()
        injected = mock_add_msgs.call_args[0][1]
        self.assertEqual(len(injected), 1)

        msg = injected[0]
        self.assertEqual(msg.sender, RoleType.EXECUTION_ENVIRONMENT)
        self.assertEqual(msg.recipient, RoleType.AGENT)
        self.assertEqual(msg.visible_to, [RoleType.AGENT])
        # Must NOT include images — agent already knows the UI
        self.assertIsNone(msg.image_ids)
        # Must use <ui_state_server> XML tag
        self.assertIn("<ui_state_server>", msg.content)
        self.assertIn("</ui_state_server>", msg.content)
        self.assertIn("submit_rsvp", msg.content)
        self.assertIn("guestName: Alice", msg.content)

    @patch("mmtoolsandbox.a2ui.tools.store_image")
    @patch("mmtoolsandbox.a2ui.tools.render_surface")
    @patch("mmtoolsandbox.a2ui.tools.add_messages_to_execution_context")
    @patch("mmtoolsandbox.a2ui.tools.get_current_context")
    def test_show_ui_metadata_uses_xml_tag_not_system_role(
        self,
        mock_get_ctx: MagicMock,
        mock_add_msgs: MagicMock,
        mock_render: MagicMock,
        mock_store: MagicMock,
    ) -> None:
        """Metadata message must be AGENT→USER with <ui_state_server> tag, not SYSTEM→USER."""
        # Setup: render a surface with a button
        state = get_ui_state()
        state.process_message(
            [
                {"beginRendering": {"surfaceId": "card-1", "root": "root"}},
                {
                    "surfaceUpdate": {
                        "surfaceId": "card-1",
                        "components": [
                            {
                                "id": "root",
                                "component": {
                                    "Column": {"children": {"explicitList": ["btn"]}}
                                },
                            },
                            {
                                "id": "btn",
                                "component": {
                                    "Button": {
                                        "child": "btn-text",
                                        "action": {"name": "approve", "context": []},
                                    }
                                },
                            },
                            {
                                "id": "btn-text",
                                "component": {
                                    "Text": {"text": {"literalString": "Approve"}}
                                },
                            },
                        ],
                    }
                },
            ]
        )

        mock_render.return_value = Image.new("RGB", (100, 100))
        mock_image_result = MagicMock()
        mock_image_result.image_id = ImageId(99)
        mock_store.return_value = mock_image_result

        show_ui_to_user()

        mock_add_msgs.assert_called_once()
        injected = mock_add_msgs.call_args[0][1]
        self.assertEqual(len(injected), 2)

        # Message 1: image only (user-only — agent already saw the render)
        img_msg = injected[0]
        self.assertEqual(img_msg.sender, RoleType.AGENT)
        self.assertEqual(img_msg.recipient, RoleType.USER)
        self.assertEqual(img_msg.visible_to, [RoleType.USER])
        self.assertEqual(img_msg.content, "")
        self.assertIsNotNone(img_msg.image_ids)

        # Message 2: metadata (user-only, XML tagged)
        meta_msg = injected[1]
        self.assertEqual(meta_msg.sender, RoleType.AGENT)  # NOT SYSTEM
        self.assertEqual(meta_msg.recipient, RoleType.USER)
        self.assertEqual(meta_msg.visible_to, [RoleType.USER])
        self.assertIn("<ui_state_server>", meta_msg.content)
        self.assertIn("</ui_state_server>", meta_msg.content)
        self.assertIn("approve", meta_msg.content)


if __name__ == "__main__":
    unittest.main()
