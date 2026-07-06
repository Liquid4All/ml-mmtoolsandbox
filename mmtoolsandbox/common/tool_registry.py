# Copyright © 2026 Apple Inc.

from collections import defaultdict
from typing import Any, Callable

from mmtoolsandbox.datasets.names import DatasetName
from mmtoolsandbox.toolbox.names import ToolboxName

Tool = Callable[..., Any]

_registry: dict[ToolboxName, dict[str, Tool]] = defaultdict(dict)


def register_tool(toolbox_name: ToolboxName, tool: Tool) -> None:
    """Register the given tool in the toolbox."""
    # Check if a tool with the same name has already been registered. The registry does
    # not care, but downstream code uses dictionaries from tool name to function, which
    # would silently override elements if the tool names were not unique.
    tool_name = tool.__name__
    existing_tool = _registry[toolbox_name].get(tool_name)
    if existing_tool is not None:
        raise KeyError(
            f"A tool with name '{tool_name}' has already been registered. Use "
            f"unique names for tools. Existing tool:\n{existing_tool}\nTool with "
            f"duplicate name:\n{tool}"
        )

    _registry[toolbox_name][tool_name] = tool


def deregister_tool(tool_name: str) -> None:
    """Manually deregister a tool.

    Usually you do not have to call this function, but it can be useful in unit tests.
    """
    for name_to_tool in _registry.values():
        name_to_tool.pop(tool_name, None)


def get_all_tools(toolbox_name: ToolboxName) -> dict[str, Tool]:
    """Get the tools for the given toolbox."""
    # Lazy: circular import.  ``mmtoolsandbox.tools`` submodules use
    # ``@register_as_tool`` (from ``common.utils``) which calls
    # ``register_tool`` from THIS module at decoration time.  Importing
    # ``mmtoolsandbox.tools`` at the top level would create the cycle:
    #   tool_registry → tools → common.utils → tool_registry
    # Lazy: same circular pattern as above — a2ui/tools.py uses
    # ``@register_as_tool`` which calls back into this module.
    import mmtoolsandbox.a2ui.tools  # noqa: F401
    import mmtoolsandbox.tools  # noqa: F401

    # Always register AppWorld tools into the unified toolbox.
    from mmtoolsandbox.tools.appworld import (
        register_appworld_tools_to_core_registry,
    )

    register_appworld_tools_to_core_registry()

    # For the MEDIUM toolbox, import consolidated tool modules (triggers
    # @register_as_tool decorators), then copy over any APPWORLD tools
    # that aren't part of a consolidation group.
    if toolbox_name == DatasetName.MEDIUM:
        import mmtoolsandbox.tools.consolidated.alarms  # noqa: F401
        import mmtoolsandbox.tools.consolidated.amazon  # noqa: F401
        import mmtoolsandbox.tools.consolidated.calendar_event  # noqa: F401
        import mmtoolsandbox.tools.consolidated.contacts  # noqa: F401
        import mmtoolsandbox.tools.consolidated.cross_app_account  # noqa: F401
        import mmtoolsandbox.tools.consolidated.cross_app_notification  # noqa: F401
        import mmtoolsandbox.tools.consolidated.cross_app_payment  # noqa: F401
        import mmtoolsandbox.tools.consolidated.file_system  # noqa: F401
        import mmtoolsandbox.tools.consolidated.gmail  # noqa: F401
        import mmtoolsandbox.tools.consolidated.reminder  # noqa: F401
        import mmtoolsandbox.tools.consolidated.settings  # noqa: F401
        import mmtoolsandbox.tools.consolidated.simple_note  # noqa: F401
        import mmtoolsandbox.tools.consolidated.splitwise  # noqa: F401
        import mmtoolsandbox.tools.consolidated.spotify  # noqa: F401
        import mmtoolsandbox.tools.consolidated.todoist  # noqa: F401
        import mmtoolsandbox.tools.consolidated.venmo  # noqa: F401
        from mmtoolsandbox.tools.consolidated import (
            register_passthrough_tools,
        )

        register_passthrough_tools()

    # For the COMPACT toolbox, import COMPACT-specific modules FIRST
    # (so overrides of MEDIUM tools register first), then copy remaining
    # MEDIUM tools (skipping names already registered by COMPACT).
    if toolbox_name == DatasetName.COMPACT:
        import mmtoolsandbox.tools.compact.alarms  # noqa: F401
        import mmtoolsandbox.tools.compact.amazon  # noqa: F401
        import mmtoolsandbox.tools.compact.calendar  # noqa: F401
        import mmtoolsandbox.tools.compact.contacts  # noqa: F401
        import mmtoolsandbox.tools.compact.cross_app  # noqa: F401
        import mmtoolsandbox.tools.compact.file_system  # noqa: F401
        import mmtoolsandbox.tools.compact.gmail  # noqa: F401
        import mmtoolsandbox.tools.compact.messages  # noqa: F401
        import mmtoolsandbox.tools.compact.settings  # noqa: F401
        import mmtoolsandbox.tools.compact.simple_note  # noqa: F401
        import mmtoolsandbox.tools.compact.splitwise  # noqa: F401
        import mmtoolsandbox.tools.compact.spotify  # noqa: F401
        import mmtoolsandbox.tools.compact.supervisor  # noqa: F401
        import mmtoolsandbox.tools.compact.todoist  # noqa: F401
        import mmtoolsandbox.tools.compact.venmo  # noqa: F401
        from mmtoolsandbox.tools.compact import (
            copy_medium_tools_to_compact,
        )
        from mmtoolsandbox.tools.compact import (
            register_passthrough_tools as compact_register_passthrough,
        )

        # Copy remaining MEDIUM tools (skips names already in COMPACT)
        copy_medium_tools_to_compact()

        # Remove absorbed passthrough tools
        compact_register_passthrough()

    # For the MINI toolbox, import all MINI modules. No passthrough —
    # every tool is explicitly defined. All tools fit in LLM context.
    if toolbox_name == DatasetName.MINI:
        import mmtoolsandbox.tools.mini.alarms  # noqa: F401
        import mmtoolsandbox.tools.mini.amazon  # noqa: F401
        import mmtoolsandbox.tools.mini.calendar  # noqa: F401
        import mmtoolsandbox.tools.mini.contacts  # noqa: F401
        import mmtoolsandbox.tools.mini.cross_app  # noqa: F401
        import mmtoolsandbox.tools.mini.device_settings  # noqa: F401
        import mmtoolsandbox.tools.mini.file_system  # noqa: F401
        import mmtoolsandbox.tools.mini.gmail  # noqa: F401
        import mmtoolsandbox.tools.mini.messages  # noqa: F401
        import mmtoolsandbox.tools.mini.reminders  # noqa: F401
        import mmtoolsandbox.tools.mini.simple_note  # noqa: F401
        import mmtoolsandbox.tools.mini.splitwise  # noqa: F401
        import mmtoolsandbox.tools.mini.spotify  # noqa: F401
        import mmtoolsandbox.tools.mini.todoist  # noqa: F401
        import mmtoolsandbox.tools.mini.venmo  # noqa: F401

    return _registry.get(toolbox_name, {})
