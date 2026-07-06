# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""COMPACT device-setting tools — collapse MEDIUM get/set setting tools into one."""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.datasets.names import DatasetName
from mmtoolsandbox.tools.compact import mark_compact_tools_absorbed_by


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.consolidated.settings as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# Collapse MEDIUM setting tools: manage_device_setting (replaces 4 MEDIUM tools)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces={DatabaseNamespace.SETTING},
    visible_to=(RoleType.AGENT,),
)
def manage_device_setting(
    setting: Literal["low_battery_mode", "location_service", "cellular", "wifi"],
    action: Literal["get", "set"],
    on: bool | NotGiven = NOT_GIVEN,
) -> bool | None:
    """Get or set device settings: low battery mode, location service, cellular, or wifi.

    Settings:
        low_battery_mode: Enabling automatically disables cellular, wifi,
            and location services.
        location_service: Cannot be enabled while low battery mode is on.
        cellular: Cannot be enabled while low battery mode is on.
        wifi: Cannot be enabled while low battery mode is on.

    Actions:
        get: Return current setting status as a boolean.
        set: Enable or disable the setting. Requires on.

    Args:
        setting: Which device setting to manage.
        action: The operation to perform.
        on: Whether to enable (true) or disable (false) the setting
            (required for set).

    Returns:
        Boolean status when action is get, None when action is set.

    Raises:
        ValueError: If the setting is already in the requested state.
        PermissionError: If low battery mode is on and trying to enable
            a dependent setting.
    """
    func_map = {
        "low_battery_mode": "manage_low_battery_mode",
        "location_service": "manage_location_service",
        "cellular": "manage_cellular_service",
        "wifi": "manage_wifi",
    }
    func_name = func_map[setting]
    kwargs: dict[str, Any] = {"action": action}
    if on is not NOT_GIVEN:
        kwargs["on"] = on
    return _get(func_name)(**kwargs)


mark_compact_tools_absorbed_by(
    "manage_device_setting",
    "manage_low_battery_mode",
    "manage_location_service",
    "manage_cellular_service",
    "manage_wifi",
)
