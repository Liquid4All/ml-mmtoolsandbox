# Copyright © 2026 Apple Inc.

"""
Consolidated tool definitions for the MEDIUM toolbox.

This package contains coarser-grained tools that dispatch to the original
fine-grained tools in ``tools.appworld`` and ``tools.tool_sandbox``.  Tools
that are not part of any consolidation group are copied from the APPWORLD
toolbox as-is ("passthrough" tools).

The MEDIUM toolbox is loaded by ``get_all_tools(DatasetName.MEDIUM)`` in
``common/tool_registry.py``, which triggers the imports below and then
calls ``register_passthrough_tools()`` to fill in the ungrouped tools.
"""

from __future__ import annotations

from mmtoolsandbox.common.tool_registry import get_all_tools, register_tool
from mmtoolsandbox.datasets.names import DatasetName

_passthrough_registered: bool = False

# Tools that should never be consolidated (framework-critical or role-specific).
PROTECTED_TOOLS: frozenset[str] = frozenset(
    {
        "end_conversation",
        "send_message_with_image",
        "api_docs_search_api_docs",
        "execute_code",
        "render_ui_screen",
        "show_ui_to_user",
        "ui_user_interact",
    }
)


def _get_consolidated_tool_names() -> set[str]:
    """Return the set of tool names already registered in the MEDIUM toolbox.

    Called after all consolidated tool modules have been imported so their
    ``@register_as_tool`` decorators have fired.
    """
    from mmtoolsandbox.common.tool_registry import _registry

    return set(_registry.get(DatasetName.MEDIUM, {}).keys())


def register_passthrough_tools() -> int:
    """Copy APPWORLD tools that are NOT consolidated into the MEDIUM toolbox.

    This is idempotent — repeated calls are no-ops.

    Returns:
        Number of passthrough tools registered (0 on subsequent calls).
    """
    global _passthrough_registered
    if _passthrough_registered:
        return 0

    # Get the full FULL toolbox (triggers its own lazy loading).
    appworld_tools = get_all_tools(DatasetName.FULL)

    # Figure out which tools are already registered as consolidated tools.
    consolidated_names = _get_consolidated_tool_names()

    # Also collect the set of *original* tool names that have been absorbed
    # into a consolidated tool so we don't register them as passthrough.
    absorbed_names = get_absorbed_tool_names()

    count = 0
    for tool_name, tool_func in appworld_tools.items():
        if tool_name in consolidated_names:
            continue
        if tool_name in absorbed_names:
            continue
        try:
            register_tool(DatasetName.MEDIUM, tool_func)
            count += 1
        except KeyError:
            pass  # Already registered

    _passthrough_registered = True
    return count


def get_absorbed_tool_names() -> set[str]:
    """Return original tool names that have been merged into consolidated tools.

    Each consolidated tool module populates this set via
    ``_register_absorbed()``.
    """
    return set(_ABSORBED_TOOL_NAMES)


# Mutable set populated by consolidated tool modules at import time.
_ABSORBED_TOOL_NAMES: set[str] = set()

# Per-tool mapping: consolidated tool name → set of original names it absorbed.
_ABSORBED_BY_TOOL: dict[str, set[str]] = {}


def mark_tools_absorbed(*tool_names: str) -> None:
    """Mark original tool names as absorbed into a consolidated tool.

    Called by each consolidated tool module to declare which original tools
    it replaces. Use ``mark_tools_absorbed_by`` instead when you need the
    per-tool reverse mapping for search scoring.
    """
    _ABSORBED_TOOL_NAMES.update(tool_names)


def mark_tools_absorbed_by(consolidated_name: str, *original_names: str) -> None:
    """Mark original tool names as absorbed by a specific consolidated tool.

    This records both the flat absorbed set (for passthrough filtering) and
    the per-tool mapping (for search scoring in api_docs).
    """
    _ABSORBED_TOOL_NAMES.update(original_names)
    if consolidated_name not in _ABSORBED_BY_TOOL:
        _ABSORBED_BY_TOOL[consolidated_name] = set()
    _ABSORBED_BY_TOOL[consolidated_name].update(original_names)


def get_absorbed_names_for_tool(tool_name: str) -> set[str]:
    """Return original tool names absorbed by a specific consolidated tool."""
    return _ABSORBED_BY_TOOL.get(tool_name, set())
