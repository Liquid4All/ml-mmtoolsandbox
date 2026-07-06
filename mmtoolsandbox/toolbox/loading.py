# Copyright © 2026 Apple Inc.

"""Loading functionality for toolboxes."""

from typing import Any

from mmtoolsandbox.common.tool_registry import get_all_tools
from mmtoolsandbox.toolbox.names import ToolboxName
from mmtoolsandbox.toolbox.toolbox import Toolbox, ToolboxConfig


def load_toolbox(toolbox_name: ToolboxName, config: dict[str, Any]) -> Toolbox:
    """Load the toolbox with the given name.

    Args:
        toolbox_name:  The name of the toolbox to load.
        config:  The configuration of the toolbox.

    Returns:
        The toolbox.
    """
    name_to_tool = get_all_tools(toolbox_name)
    tools = list(name_to_tool.values())
    toolbox_config = ToolboxConfig.model_validate(config)
    return Toolbox(
        name=toolbox_name,
        config=toolbox_config,
        tools=tools,
    )
