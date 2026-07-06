"""
HTML renderer for A2UI extension.

Generates self-contained HTML pages that use the A2UI Lit renderer (Web
Components) to display interactive UI surfaces in a browser. The HTML
includes:

- ES module import maps for Lit and its dependencies (via CDN)
- Import paths for the pre-built A2UI Lit renderer (served locally)
- The A2UI JSON messages inlined as JavaScript data
- An ``<a2ui-surface>`` Web Component that renders the UI

The A2UI Lit renderer must be pre-built from the upstream A2UI repository
(``npm install && npm run build`` in ``A2UI/renderers/lit/``). This module
does NOT modify the A2UI repo — it only references the compiled output.
"""

from __future__ import annotations

import html
import json
from typing import Any

from mmtoolsandbox.a2ui.state import SurfaceState

# Default CDN URLs for Lit dependencies (same versions as client/index.html)
_DEFAULT_CDN_IMPORTS = {
    "lit": "https://esm.sh/lit@3.1.2",
    "lit/": "https://esm.sh/lit@3.1.2/",
    "@lit/context": "https://esm.sh/@lit/context@1.1.0",
    "lit/decorators.js": "https://esm.sh/lit@3.1.2/decorators.js",
    "lit/directives/style-map.js": "https://esm.sh/lit@3.1.2/directives/style-map.js",
    "lit/directives/class-map.js": "https://esm.sh/lit@3.1.2/directives/class-map.js",
    "@lit-labs/signals": "https://esm.sh/@lit-labs/signals@0.1.3",
    "signal-utils/": "https://esm.sh/signal-utils@0.21.1/",
    "markdown-it": "https://esm.sh/markdown-it@14.1.0",
}

# Default local path for pre-built A2UI renderers
_DEFAULT_A2UI_BASE_URL = "/lib/renderers"


def _build_import_map(a2ui_base_url: str) -> dict[str, str]:
    """Build the ES module import map combining CDN deps and local A2UI paths.

    Args:
        a2ui_base_url: Base URL where the pre-built A2UI renderers are served.

    Returns:
        A dict suitable for a ``<script type="importmap">`` block.
    """
    base = a2ui_base_url.rstrip("/")
    imports = dict(_DEFAULT_CDN_IMPORTS)
    imports["@a2ui/lit"] = f"{base}/lit/dist/src/index.js"
    imports["@a2ui/lit/"] = f"{base}/lit/dist/src/"
    imports["@a2ui/web_core"] = f"{base}/web_core/dist/src/v0_8/index.js"
    imports["@a2ui/web_core/"] = f"{base}/web_core/dist/src/v0_8/"
    return imports


def _surface_to_a2ui_messages(surface: SurfaceState) -> list[dict[str, Any]]:
    """Convert a SurfaceState back to the A2UI v0.8 message sequence.

    Reconstructs the 3-message pattern (beginRendering, surfaceUpdate,
    dataModelUpdate) that the Lit renderer expects.

    Args:
        surface: The surface state to convert.

    Returns:
        A list of A2UI message dicts.
    """
    messages: list[dict[str, Any]] = []

    # 1. beginRendering
    messages.append(
        {
            "beginRendering": {
                "surfaceId": surface.surface_id,
                "root": surface.root_id,
            }
        }
    )

    # 2. surfaceUpdate
    components = list(surface.components.values())
    if components:
        messages.append(
            {
                "surfaceUpdate": {
                    "surfaceId": surface.surface_id,
                    "components": components,
                }
            }
        )

    # 3. dataModelUpdate (if data model is non-empty)
    if surface.data_model:
        # Convert Python dict back to v0.8 contents format
        contents = _dict_to_contents(surface.data_model)
        messages.append(
            {
                "dataModelUpdate": {
                    "surfaceId": surface.surface_id,
                    "path": "/",
                    "contents": contents,
                }
            }
        )

    return messages


def _dict_to_contents(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a Python dict back to v0.8 contents list format.

    Args:
        data: The Python dict to convert.

    Returns:
        A list of v0.8 content entries.
    """
    contents: list[dict[str, Any]] = []
    for key, value in data.items():
        if isinstance(value, str):
            contents.append({"key": key, "valueString": value})
        elif isinstance(value, bool):
            contents.append({"key": key, "valueBoolean": value})
        elif isinstance(value, (int, float)):
            contents.append({"key": key, "valueNumber": value})
        elif isinstance(value, dict):
            contents.append({"key": key, "valueMap": _dict_to_contents(value)})
        else:
            # Fallback: serialize as string
            contents.append({"key": key, "valueString": str(value)})
    return contents


def render_surface_to_html(
    surface: SurfaceState,
    a2ui_base_url: str = _DEFAULT_A2UI_BASE_URL,
) -> str:
    """Render a surface state to a self-contained HTML page.

    The HTML uses the pre-built A2UI Lit renderer to display the UI
    as interactive Web Components. It can be embedded in an iframe or
    served as a standalone page.

    Args:
        surface: The surface state to render.
        a2ui_base_url: Base URL where the pre-built A2UI Lit and web_core
                        JS files are served from.

    Returns:
        A complete HTML string, or empty string if the surface has no root.
    """
    if not surface.root_id:
        return ""

    # Build the A2UI messages from surface state
    a2ui_messages = _surface_to_a2ui_messages(surface)
    messages_json = json.dumps(a2ui_messages, ensure_ascii=False)

    # Build import map
    import_map = _build_import_map(a2ui_base_url)
    import_map_json = json.dumps({"imports": import_map}, indent=2)

    surface_id_escaped = html.escape(surface.surface_id)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UI Surface: {surface_id_escaped}</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f5f5f5;
        }}
        #app {{
            max-width: 600px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        a2ui-surface {{
            display: block;
            width: 100%;
        }}
    </style>
    <script async src="https://ga.jspm.io/npm:es-module-shims@1.10.0/dist/es-module-shims.js"></script>
    <script type="importmap">
    {import_map_json}
    </script>
</head>
<body>
    <div id="app"></div>
    <script type="module">
        import {{ html, render }} from 'lit';
        import * as A2UI from '@a2ui/lit';

        const messages = {messages_json};
        const surfaceId = {json.dumps(surface.surface_id)};

        const processor = A2UI.v0_8.Data.createSignalA2uiMessageProcessor();
        for (const msg of messages) {{
            processor.processMessages([msg]);
        }}

        const surface = processor.getSurfaces().get(surfaceId);

        render(html\`
            <a2ui-surface
                .surfaceId=${{surfaceId}}
                .surface=${{surface ? {{ ...surface }} : undefined}}
                .processor=${{processor}}
                .enableCustomElements=${{true}}
                @a2ui-action=${{(e) => console.log('User Action:', e.detail)}}
            ></a2ui-surface>
        \`, document.getElementById('app'));
    </script>
</body>
</html>"""
