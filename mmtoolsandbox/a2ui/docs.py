"""
Documentation and examples for the A2UI extension.

This module provides helper functions to access A2UI protocol documentation,
schemas, and example payloads. It serves as the backend for the UI discovery
tools. The protocol schemas in ``schema/`` are upstream A2UI v0.8 artifacts
and are never modified — this module only provides tooling to present them.
"""

import json
import os
from pathlib import Path
from typing import Any

# Base directory for A2UI samples (relative to this file)
A2UI_SAMPLES_DIR = Path(__file__).parent / "examples"

# Path to schema directory (upstream v0.8 schemas — read-only)
_SCHEMA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "schema"))

# ---------------------------------------------------------------------------
# Minimal usage examples for protocol components.
#
# These are NOT part of the protocol — they are tooling aids that help LLMs
# understand how to compose valid A2UI JSON.  Each example shows the minimal
# component definition(s) needed for that component type.
# ---------------------------------------------------------------------------
COMPONENT_EXAMPLES: dict[str, str] = {
    "Text": (
        '{"id": "t1", "component": {"Text": {"text": {"literalString": "Hello World"}, "usageHint": "h2"}}}'
    ),
    "Image": (
        '{"id": "img1", "component": {"Image": {"url": {"literalString": "IMAGE_URL_1"}}}}'
    ),
    "Icon": (
        '{"id": "icon1", "component": {"Icon": {"name": {"literalString": "star"}}}}'
    ),
    "Video": (
        '{"id": "vid1", "component": {"Video": {"url": {"literalString": "https://example.com/video.mp4"}}}}'
    ),
    "AudioPlayer": (
        '{"id": "audio1", "component": {"AudioPlayer": {"url": {"literalString": "https://example.com/song.mp3"}}}}'
    ),
    "Column": (
        '{"id": "col1", "component": {"Column": {"children": {"explicitList": ["child1", "child2"]}}}}'
    ),
    "Row": (
        '{"id": "row1", "component": {"Row": {"children": {"explicitList": ["left", "right"]}}}}'
    ),
    "List": (
        "# Explicit list:\n"
        '{"id": "list1", "component": {"List": {"direction": "vertical", "children": {"explicitList": ["item1", "item2"]}}}}\n'
        "# Template-based (data-driven) list:\n"
        '{"id": "list2", "component": {"List": {"direction": "vertical", "children": {"template": {"componentId": "item-template", "dataBinding": "/items"}}}}}'
    ),
    "Card": ('{"id": "card1", "component": {"Card": {"child": "card-content"}}}'),
    "Tabs": (
        '{"id": "tabs1", "component": {"Tabs": {"tabItems": [{"title": {"literalString": "Tab 1"}, "child": "tab1-content"}, {"title": {"literalString": "Tab 2"}, "child": "tab2-content"}]}}}'
    ),
    "Divider": ('{"id": "div1", "component": {"Divider": {"axis": "horizontal"}}}'),
    "Modal": (
        '{"id": "modal1", "component": {"Modal": {"entryPointChild": "open-btn", "contentChild": "modal-body"}}}'
    ),
    "Button": (
        "# A Button requires a child component (typically Text) and an action:\n"
        '{"id": "btn1", "component": {"Button": {"child": "btn1-text", "primary": true, "action": {"name": "submit"}}}}\n'
        '{"id": "btn1-text", "component": {"Text": {"text": {"literalString": "Submit"}}}}'
    ),
    "TextField": (
        '{"id": "tf1", "component": {"TextField": {"label": {"literalString": "Your Name"}, "text": {"path": "userName"}}}}'
    ),
    "CheckBox": (
        '{"id": "cb1", "component": {"CheckBox": {"label": {"literalString": "I agree"}, "value": {"literalBoolean": false}}}}'
    ),
    "DateTimeInput": (
        '{"id": "dt1", "component": {"DateTimeInput": {"value": {"path": "selectedDate"}, "enableDate": true, "enableTime": true}}}'
    ),
    "MultipleChoice": (
        '{"id": "mc1", "component": {"MultipleChoice": {"selections": {"path": "selected"}, "options": [{"label": {"literalString": "Option A"}, "value": "a"}, {"label": {"literalString": "Option B"}, "value": "b"}]}}}'
    ),
    "Slider": (
        '{"id": "sl1", "component": {"Slider": {"value": {"literalNumber": 50}, "minValue": 0, "maxValue": 100}}}'
    ),
}

# ---------------------------------------------------------------------------
# Concept documentation (hardcoded pedagogical content for the discovery tools)
# ---------------------------------------------------------------------------
A2UI_DOCS = {
    "Overview": """
# UI System Overview

The UI system allows you to generate rich, interactive user interfaces.
Instead of just generating text, you can send a stream of JSON messages that describe UI components
(like Buttons, Text, Images) and data.

Key Concepts:
1.  **Declarative**: You describe *what* the UI should look like, not *how* to draw it.
2.  **Streamed**: The UI is built incrementally from a stream of messages.
3.  **Platform-Agnostic**: The same JSON can be rendered on Web, Mobile (Flutter), etc.
4.  **Separation of Concerns**:
    *   **Structure**: The component tree (e.g., a Column containing a Text and a Button).
    *   **Data**: The values that populate the UI (e.g., the text string, the button label).
    *   **Catalog**: The set of available components (defined by the client/platform).

To start, explore the "Protocol" concept or look at "Components".
""",
    "Protocol": """
# UI Message Format

The protocol consists of a stream of JSON objects (JSONL). Each object is a message.
There are four main server-to-client message types:

1.  **`beginRendering`**: Signals that the initial set of components and data is ready.
    *   `surfaceId`: The ID of the surface.
    *   `root`: The ID of the root component to display.

2.  **`surfaceUpdate`**: Adds or updates components on a surface.
    *   `surfaceId`: The ID of the surface to update.
    *   `components`: A list of component definitions.

3.  **`dataModelUpdate`**: Updates the data for a surface.
    *   `surfaceId`: The ID of the surface.
    *   `contents`: A list of key-value pairs to update in the data model.

4.  **`deleteSurface`**: Removes a surface.
    *   `surfaceId`: The ID of the surface to remove.

Typical Sequence (3 messages):
1.  Send `beginRendering` to create the surface and specify the root component.
2.  Send `surfaceUpdate` with all component definitions.
3.  Send `dataModelUpdate` with data that components bind to.
""",
    "Components": """
# UI Components Overview

Components are the building blocks of the UI. They are defined in `surfaceUpdate` messages.
Each component has an `id` and a `component` object describing its type and properties.

Common Components:
*   **Layout**: `Column`, `Row`, `List`, `Card`, `Divider`, `Tabs`, `Modal`
*   **Content**: `Text`, `Image`, `Icon`, `Video`, `AudioPlayer`
*   **Input**: `Button`, `TextField`, `CheckBox`, `MultipleChoice`, `Slider`, `DateTimeInput`

Example Component Definition:
```json
{"id": "my-button", "component": {"Button": {"child": "my-button-text", "action": {"name": "submit"}}}}
{"id": "my-button-text", "component": {"Text": {"text": {"literalString": "Click Me"}}}}
```

Use `ui_list_items("Components")` to see all available components, or
`ui_get_item_details("Components", "<name>")` for schema + usage examples.
""",
    "Actions": """
# UI Actions & Events

Interactive components (like `Button`) have an `action` property. When the user interacts with them,
the client sends a `userAction` event back to the agent.

Defining an Action:
```json
"action": {
  "name": "submit_form",
  "context": [
    { "key": "userInput", "value": { "path": "/form/text" } }
  ]
}
```

Handling an Event:
The agent receives a JSON object like:
```json
{
  "userAction": {
    "name": "submit_form",
    "surfaceId": "...",
    "context": { "userInput": "..." }
  }
}
```

Use `ui_list_items("Client Actions")` to see event structures.
""",
    "Data Binding": """
# UI Data Binding

Properties of components can be bound to the data model. This allows the UI to update automatically
when the data changes, without resending the component definition.

A `BoundValue` can be:
1.  **Literal**: A static value.
    `"text": { "literalString": "Hello" }`
2.  **Path**: A reference to a value in the data model.
    `"text": { "path": "/user/name" }`

Data Model Updates (using `dataModelUpdate` message):
```json
{
  "dataModelUpdate": {
    "surfaceId": "s1",
    "path": "/",
    "contents": [
      { "key": "name", "valueString": "Alice" },
      { "key": "contacts", "valueMap": [
        { "key": "c1", "valueMap": [
          { "key": "name", "valueString": "Bob" },
          { "key": "email", "valueString": "bob@example.com" }
        ]}
      ]}
    ]
  }
}
```
Nested data uses `valueMap` which contains further key-value entries.
""",
}

# Cache for loaded definitions
_DEFINITIONS_CACHE: dict[str, Any] = {}


def _load_schema(schema_name: str) -> Any:
    """Load a JSON schema from the schema directory."""
    schema_path = os.path.join(_SCHEMA_DIR, schema_name)
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_all_definitions() -> dict[str, Any]:
    """Load and index all schema definitions and docs."""
    if _DEFINITIONS_CACHE:
        return _DEFINITIONS_CACHE

    definitions: dict[str, Any] = {}

    # 1. Concepts (from A2UI_DOCS)
    for name, content in A2UI_DOCS.items():
        definitions[name] = {
            "category": "Concepts",
            "description": content.strip().split("\n")[0].replace("# ", ""),
            "content": content,
        }

    # 2. Server Actions (server_to_client.json)
    try:
        s2c = _load_schema("server_to_client.json")
        for name, schema in s2c.get("properties", {}).items():
            definitions[name] = {
                "category": "Server Actions",
                "description": schema.get("description", "Server action"),
                "schema": schema,
            }
    except Exception as e:
        print(f"Warning: Failed to load server_to_client.json: {e}")

    # 3. Components (standard_catalog.json)
    try:
        catalog = _load_schema("standard_catalog.json")
        for name, schema in catalog.get("components", {}).items():
            definitions[name] = {
                "category": "Components",
                "description": schema.get("description", "UI Component"),
                "schema": schema,
            }
    except Exception as e:
        print(f"Warning: Failed to load standard_catalog.json: {e}")

    # 4. Client Actions (client_to_server.json)
    try:
        c2s = _load_schema("client_to_server.json")
        for name, schema in c2s.get("properties", {}).items():
            definitions[name] = {
                "category": "Client Actions",
                "description": schema.get("description", "Client action/event"),
                "schema": schema,
            }
    except Exception as e:
        print(f"Warning: Failed to load client_to_server.json: {e}")

    # 5. Examples (dynamically loaded from A2UI_SAMPLES_DIR)
    try:
        if A2UI_SAMPLES_DIR.exists():
            for file_path in A2UI_SAMPLES_DIR.glob("*.json"):
                name = file_path.stem
                definitions[name] = {
                    "category": "Examples",
                    "description": f"Example: {name.replace('_', ' ').title()}",
                    "path": str(file_path),
                }
    except Exception as e:
        print(f"Warning: Failed to load examples: {e}")

    _DEFINITIONS_CACHE.update(definitions)
    return definitions


def get_ui_categories() -> list[str]:
    """Get the list of available UI capability categories."""
    return [
        "Concepts",
        "Components",
        "Server Actions",
        "Client Actions",
        "Examples",
    ]


def get_ui_items_in_category(category: str) -> list[str]:
    """Get a list of items in a specific category with descriptions."""
    definitions = _load_all_definitions()
    items = []
    for name, info in definitions.items():
        if info["category"].lower() == category.lower():
            desc = info.get("description", "")
            if len(desc) > 80:
                desc = desc[:77] + "..."
            items.append(f"{name}: {desc}")
    return sorted(items)


def get_ui_item_details(category: str, item_name: str) -> str:
    """Get detailed documentation for a specific item."""
    definitions = _load_all_definitions()

    # Try exact match first
    info = definitions.get(item_name)

    # If not found, try case-insensitive match
    if not info:
        for name, data in definitions.items():
            if name.lower() == item_name.lower():
                info = data
                item_name = name
                break

    if not info:
        return f"Error: Item '{item_name}' not found."

    # Verify category matches (loose check)
    if info["category"].lower() != category.lower():
        return (
            f"Note: '{item_name}' belongs to category '{info['category']}', not '{category}'.\n\n"
            + _format_item_details(item_name, info)
        )

    return _format_item_details(item_name, info)


def _format_item_details(name: str, info: dict[str, Any]) -> str:
    """Format the details of an item based on its type.

    For components, appends a minimal usage example from
    ``COMPONENT_EXAMPLES`` alongside the raw protocol schema.
    """
    category = info["category"]

    if category == "Concepts":
        return str(info["content"])

    if category == "Examples":
        path = info["path"]
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return json.dumps(data, indent=2)
        except Exception as e:
            return f"Error loading example file: {e}"

    if category in ["Components", "Server Actions", "Client Actions"]:
        schema = info.get("schema", {})
        result = json.dumps(schema, indent=2)

        # Append minimal usage example for components
        example = COMPONENT_EXAMPLES.get(name)
        if example:
            result += f"\n\n--- Minimal Usage Example ---\n{example}"

        return result

    return f"Unknown category: {category}"


def search_ui_docs(query: str) -> str:
    """Search across all UI documentation."""
    definitions = _load_all_definitions()
    query_lower = query.lower()
    results: list[tuple[int, str, str, str]] = []

    for name, info in definitions.items():
        score = 0
        if query_lower in name.lower():
            score += 10
        if query_lower in info.get("description", "").lower():
            score += 5

        content_str = ""
        if "content" in info:
            content_str = info["content"]
        elif "schema" in info:
            content_str = json.dumps(info["schema"])

        if query_lower in content_str.lower():
            score += 1

        if score > 0:
            results.append((score, name, info["category"], info.get("description", "")))

    results.sort(key=lambda x: x[0], reverse=True)

    if not results:
        return f"No results found for '{query}'."

    output = [f"Search results for '{query}':"]
    for _, name, category, desc in results[:10]:
        output.append(f"- {name} ({category}): {desc}")

    return "\n".join(output)


def get_quick_start() -> str:
    """Return a compact protocol overview with a complete working example.

    This is the backend for the ``ui_get_quick_start`` tool — gives an agent
    everything it needs to start generating A2UI JSON in a single call.
    """
    example_content = get_ui_item_details("Examples", "booking_form")
    component_list = get_ui_items_in_category("Components")
    example_list = get_ui_items_in_category("Examples")

    return (
        "# UI Rendering Quick Start\n\n"
        "A UI screen is built from a sequence of 3 JSON messages:\n\n"
        "1. `beginRendering` — Create a surface with a surfaceId and root component ID\n"
        "2. `surfaceUpdate` — Define all components (flat list, each referenced by ID)\n"
        "3. `dataModelUpdate` — Populate data that components bind to via `path`\n\n"
        "## Key Patterns\n"
        '- Components reference children by ID string (e.g., Button\'s `"child": "btn-text"`)\n'
        '- Text values use `{"literalString": "..."}` for static or `{"path": "keyName"}` for data-bound\n'
        '- Layout children use `{"explicitList": ["id1", "id2"]}` for a fixed set\n'
        '- Data model entries use `{"key": "name", "valueString": "value"}` format\n'
        '- Nested data uses `"valueMap"` containing further key-value entries\n\n'
        "## Complete Working Example (Booking Form)\n\n"
        f"```json\n{example_content}\n```\n\n"
        "## Available Templates\n\n"
        "Use `ui_get_item_details('Examples', '<name>')` to load a full template:\n\n"
        + "\n".join(f"- {item}" for item in example_list)
        + "\n\n## Available Components\n\n"
        + "\n".join(f"- {item}" for item in component_list)
    )
