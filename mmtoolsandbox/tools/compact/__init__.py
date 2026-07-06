# Copyright © 2026 Apple Inc.

"""
Consolidated tool definitions for the COMPACT toolbox.

Builds on the MEDIUM toolbox by applying additional consolidation:
- Strategy 4: Search/list entity-type parameterization
- Strategy 5: Functional domain clustering
- Strategy 7: Collapse existing consolidated tools by entity subtype

The COMPACT toolbox is loaded by ``get_all_tools(DatasetName.COMPACT)`` in
``common/tool_registry.py``, which:
1. Loads MEDIUM tools and copies them into the COMPACT registry
2. Imports COMPACT-specific modules (triggers @register_as_tool decorators)
3. Calls ``register_passthrough_tools()`` to fill in remaining ungrouped tools
"""

from __future__ import annotations

from mmtoolsandbox.common.tool_registry import get_all_tools
from mmtoolsandbox.datasets.names import DatasetName
from mmtoolsandbox.tools.consolidated import (
    mark_tools_absorbed_by,
)

_passthrough_registered: bool = False

# Tools absorbed by COMPACT-level consolidation (on top of MEDIUM absorbed).
_COMPACT_ABSORBED_TOOL_NAMES: set[str] = set()


def mark_compact_tools_absorbed_by(
    consolidated_name: str, *original_names: str
) -> None:
    """Mark original tool names as absorbed by a COMPACT consolidated tool.

    Records in both the COMPACT-local set and the shared per-tool mapping
    used by api_docs search scoring.
    """
    _COMPACT_ABSORBED_TOOL_NAMES.update(original_names)
    mark_tools_absorbed_by(consolidated_name, *original_names)


def copy_medium_tools_to_compact() -> int:
    """Copy all MEDIUM tools into the COMPACT registry.

    This is called before COMPACT-specific modules are imported so that
    MEDIUM's consolidated tools are available, and COMPACT modules can
    override/replace some of them.

    Returns:
        Number of tools copied.
    """
    from mmtoolsandbox.common.tool_registry import _registry

    medium_tools = get_all_tools(DatasetName.MEDIUM)
    compact_tools = _registry[DatasetName.COMPACT]

    count = 0
    for tool_name, tool_func in medium_tools.items():
        if tool_name not in compact_tools:
            compact_tools[tool_name] = tool_func
            count += 1
    return count


def register_passthrough_tools() -> int:
    """Remove COMPACT-absorbed tools from the COMPACT registry.

    After copy_medium_tools_to_compact() copies everything from MEDIUM, and
    COMPACT modules register their replacements, this function removes the
    now-superseded tools.

    Returns:
        Number of tools removed.
    """
    global _passthrough_registered
    if _passthrough_registered:
        return 0

    from mmtoolsandbox.common.tool_registry import _registry

    compact_tools = _registry[DatasetName.COMPACT]
    count = 0
    for name in _COMPACT_ABSORBED_TOOL_NAMES:
        if name in compact_tools:
            del compact_tools[name]
            count += 1

    _passthrough_registered = True
    return count
