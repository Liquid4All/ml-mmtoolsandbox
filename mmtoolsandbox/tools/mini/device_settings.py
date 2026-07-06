# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""MINI device_settings tool — device settings and location (native, polars-backed)."""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.datasets.names import DatasetName


@register_as_tool(
    toolboxes={DatasetName.MINI},
    database_namespaces={DatabaseNamespace.SETTING},
    visible_to=(RoleType.AGENT,),
)
def device_settings(
    action: Literal["get", "set", "get_location"],
    setting: str | NotGiven = NOT_GIVEN,
    on: bool | NotGiven = NOT_GIVEN,
) -> bool | dict[str, Any] | None:
    """Get or set device settings (wifi, cellular, battery mode, location service) or get GPS location.

    Actions:
        get - Get a setting value. Requires setting.
        set - Set a setting value. Requires setting and on (true/false).
        get_location - Get current GPS location.

    Settings (for get/set):
        low_battery_mode - Low battery mode. Setting on=true disables
            cellular, wifi, and location_service.
        location_service - Location service. Cannot enable in low battery mode.
        cellular - Cellular service. Cannot enable in low battery mode.
        wifi - WiFi. Cannot enable in low battery mode.

    Args:
        action: "get", "set", or "get_location".
        setting: Setting name: "low_battery_mode", "location_service",
            "cellular", or "wifi" (for get/set).
        on: Enable (true) or disable (false) the setting (for set).

    Returns:
        Boolean value (for get), None (for set),
        or dict with latitude/longitude (for get_location).

    Raises:
        ValueError: If setting is already in requested state.
        PermissionError: If low battery mode prevents enabling a service,
            or location service is off (for get_location).
    """
    import mmtoolsandbox.tools.tool_sandbox.setting as m

    setting_map = {
        "low_battery_mode": (
            m.get_low_battery_mode_status,
            m.set_low_battery_mode_status,
        ),
        "location_service": (
            m.get_location_service_status,
            m.set_location_service_status,
        ),
        "cellular": (m.get_cellular_service_status, m.set_cellular_service_status),
        "wifi": (m.get_wifi_status, m.set_wifi_status),
    }

    if action == "get":
        if isinstance(setting, str):
            getter, _ = setting_map[setting]
            return getter()
        raise ValueError("setting is required for 'get' action")
    elif action == "set":
        if isinstance(setting, str):
            _, setter = setting_map[setting]
            return setter(on=on)
        raise ValueError("setting is required for 'set' action")
    elif action == "get_location":
        return m.get_current_location()
    else:
        raise ValueError(f"Unknown action: {action}")
