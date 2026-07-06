# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""Consolidated native ToolSandbox tools for the MEDIUM toolbox.

Merges get/set pairs for device settings into single tools.
"""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.toolbox.names import ToolboxName
from mmtoolsandbox.tools.consolidated import mark_tools_absorbed_by


def _get_setting(name: str) -> Any:
    import mmtoolsandbox.tools.tool_sandbox.setting as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# Setting pairs: Low battery mode
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces={DatabaseNamespace.SETTING},
    visible_to=(RoleType.AGENT,),
)
def manage_low_battery_mode(
    action: Literal["get", "set"],
    on: bool | NotGiven = NOT_GIVEN,
) -> bool | None:
    """Get or set low battery mode status.

    Actions:
        get: Request current low battery mode status. Returns a boolean.
        set: Enable or disable low battery mode. Requires on. Enabling
            low battery mode automatically disables cellular, wifi, and
            location services.

    Args:
        action: The operation to perform.
        on: Whether to enable (true) or disable (false) low battery mode
            (required for set).

    Returns:
        Boolean status when action is get, None when action is set.

    Raises:
        ValueError: If low battery mode is already in the requested state.
    """
    if action == "get":
        return _get_setting("get_low_battery_mode_status")()
    elif action == "set":
        _get_setting("set_low_battery_mode_status")(on=on)
        return None
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Setting pairs: Location service
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces={DatabaseNamespace.SETTING},
    visible_to=(RoleType.AGENT,),
)
def manage_location_service(
    action: Literal["get", "set"],
    on: bool | NotGiven = NOT_GIVEN,
) -> bool | None:
    """Get or set location service status.

    Actions:
        get: Request current location service status. Returns a boolean.
        set: Enable or disable location service. Requires on.

    Args:
        action: The operation to perform.
        on: Whether to enable (true) or disable (false) location service
            (required for set).

    Returns:
        Boolean status when action is get, None when action is set.

    Raises:
        ValueError: If location service is already in the requested state.
        PermissionError: If low battery mode is on and trying to enable
            location service.
    """
    if action == "get":
        return _get_setting("get_location_service_status")()
    elif action == "set":
        _get_setting("set_location_service_status")(on=on)
        return None
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Setting pairs: Cellular service
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces={DatabaseNamespace.SETTING},
    visible_to=(RoleType.AGENT,),
)
def manage_cellular_service(
    action: Literal["get", "set"],
    on: bool | NotGiven = NOT_GIVEN,
) -> bool | None:
    """Get or set cellular service status.

    Actions:
        get: Request current cellular service status. Returns a boolean.
        set: Enable or disable cellular service. Requires on.

    Args:
        action: The operation to perform.
        on: Whether to enable (true) or disable (false) cellular service
            (required for set).

    Returns:
        Boolean status when action is get, None when action is set.

    Raises:
        ValueError: If cellular service is already in the requested state.
        PermissionError: If low battery mode is on and trying to enable
            cellular service.
    """
    if action == "get":
        return _get_setting("get_cellular_service_status")()
    elif action == "set":
        _get_setting("set_cellular_service_status")(on=on)
        return None
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Setting pairs: Wifi
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces={DatabaseNamespace.SETTING},
    visible_to=(RoleType.AGENT,),
)
def manage_wifi(
    action: Literal["get", "set"],
    on: bool | NotGiven = NOT_GIVEN,
) -> bool | None:
    """Get or set wifi status.

    Actions:
        get: Request current wifi status. Returns a boolean.
        set: Enable or disable wifi. Requires on.

    Args:
        action: The operation to perform.
        on: Whether to enable (true) or disable (false) wifi
            (required for set).

    Returns:
        Boolean status when action is get, None when action is set.

    Raises:
        ValueError: If wifi is already in the requested state.
        PermissionError: If low battery mode is on and trying to enable
            wifi.
    """
    if action == "get":
        return _get_setting("get_wifi_status")()
    elif action == "set":
        _get_setting("set_wifi_status")(on=on)
        return None
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Mark absorbed tools
# ---------------------------------------------------------------------------

mark_tools_absorbed_by(
    "manage_low_battery_mode",
    "get_low_battery_mode_status",
    "set_low_battery_mode_status",
)
mark_tools_absorbed_by(
    "manage_location_service",
    "get_location_service_status",
    "set_location_service_status",
)
mark_tools_absorbed_by(
    "manage_cellular_service",
    "get_cellular_service_status",
    "set_cellular_service_status",
)
mark_tools_absorbed_by(
    "manage_wifi",
    "get_wifi_status",
    "set_wifi_status",
)
