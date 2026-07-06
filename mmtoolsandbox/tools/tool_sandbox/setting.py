# Copyright © 2026 Apple Inc.

"""A collection of tools which simulates common functions used for setting."""

from typing import Literal, cast

import polars as pl

from mmtoolsandbox.common.execution_context import (
    DatabaseNamespace,
    RoleType,
    get_current_context,
)
from mmtoolsandbox.common.utils import register_as_tool
from mmtoolsandbox.common.validators import validate_type
from mmtoolsandbox.toolbox.names import ToolboxName


def set_boolean_settings(setting_name: str, on: bool) -> None:
    """Set a boolean setting column to on or off.

    Internal utility — not exposed as a tool.

    Args:
        setting_name: Name of the boolean settings column.
        on: Whether to enable (True) or disable (False) the setting.

    Raises:
        KeyError: If the column is not a boolean type.
        ValueError: If the setting is already in the requested state.
    """
    validate_type(setting_name, "setting_name", str)
    validate_type(on, "on", bool)

    current_context = get_current_context()
    setting_database = current_context.get_database(DatabaseNamespace.SETTING)
    if setting_database.schema.get(setting_name, pl.Null) != pl.Boolean:
        raise KeyError(f"{setting_name} is not a boolean column")
    if bool(setting_database[setting_name][0]) == on:
        raise ValueError(f"{setting_name} already {'enabled' if on else 'disabled'}")
    current_context.update_database(
        namespace=DatabaseNamespace.SETTING,
        dataframe=setting_database.with_columns(~pl.col(setting_name)),
    )


def get_boolean_settings(setting_name: str) -> bool:
    """Get the current value of a boolean setting.

    Internal utility — not exposed as a tool.

    Args:
        setting_name: Name of the boolean settings column.

    Returns:
        Current boolean value of the setting.

    Raises:
        KeyError: If the column is not a boolean type.
    """
    validate_type(setting_name, "setting_name", str)

    current_context = get_current_context()
    setting_database = current_context.get_database(DatabaseNamespace.SETTING)
    if setting_database.schema.get(setting_name, pl.Null) != pl.Boolean:
        raise KeyError(f"{setting_name} is not a boolean column")
    return cast(bool, setting_database[setting_name][0])


@register_as_tool(
    toolboxes={
        ToolboxName.FULL,
    },
    database_namespaces={DatabaseNamespace.SETTING},
    visible_to=(RoleType.AGENT,),
)
def set_low_battery_mode_status(on: bool) -> None:
    """Enable / Disable low battery mode

    Args:
        on: If we want to turn on low battery mode

    Raises:
        ValueError: If low battery mode is already turned on / off
    """
    validate_type(on, "on", bool)

    set_boolean_settings(setting_name="low_battery_mode", on=on)
    # Automatically turn off other dependent services if low battery mode is on
    if on:
        for dependent_setting in ["cellular", "wifi", "location_service"]:
            try:
                set_boolean_settings(dependent_setting, on=False)
            except ValueError:
                pass


@register_as_tool(
    toolboxes={
        ToolboxName.FULL,
    },
    database_namespaces={DatabaseNamespace.SETTING},
    visible_to=(RoleType.AGENT,),
)
def get_low_battery_mode_status() -> bool:
    """Request low battery mode status

    Returns:
        Boolean indicating if the low battery mode is on
    """
    return get_boolean_settings("low_battery_mode")


@register_as_tool(
    toolboxes={
        ToolboxName.FULL,
    },
    database_namespaces={DatabaseNamespace.SETTING},
    visible_to=(RoleType.AGENT,),
)
def set_location_service_status(on: bool) -> None:
    """Enable / Disable location service

    Args:
        on: If we want to turn on location service

    Raises:
        ValueError:         If location service is already turned on / off
        PermissionError:    If low battery mode is on, in which case we are not allowed to turn on location service
    """
    if on and get_low_battery_mode_status():
        raise PermissionError(
            "Location service cannot be turned on in low battery mode"
        )
    set_boolean_settings(setting_name="location_service", on=on)


@register_as_tool(
    toolboxes={
        ToolboxName.FULL,
    },
    database_namespaces={DatabaseNamespace.SETTING},
    visible_to=(RoleType.AGENT,),
)
def get_location_service_status() -> bool:
    """Request location service status

    Returns:
        Boolean indicating if the location service is on
    """
    return get_boolean_settings("location_service")


@register_as_tool(
    toolboxes={
        ToolboxName.FULL,
    },
    database_namespaces={DatabaseNamespace.SETTING},
    visible_to=(RoleType.AGENT,),
)
def set_cellular_service_status(on: bool) -> None:
    """Enable / Disable cellular service

    Args:
        on: If we want to turn on cellular service

    Raises:
        ValueError:         If cellular service is already turned on / off
        PermissionError:    If low battery mode is on, in which case we are not allowed to turn on cellular service
    """
    validate_type(on, "on", bool)

    if on and get_low_battery_mode_status():
        raise PermissionError(
            "Cellular service cannot be turned on in low battery mode"
        )
    set_boolean_settings(setting_name="cellular", on=on)


@register_as_tool(
    toolboxes={
        ToolboxName.FULL,
    },
    database_namespaces={DatabaseNamespace.SETTING},
    visible_to=(RoleType.AGENT,),
)
def get_cellular_service_status() -> bool:
    """Request cellular service status

    Returns:
        Boolean indicating if the cellular service is on
    """
    return get_boolean_settings("cellular")


@register_as_tool(
    toolboxes={
        ToolboxName.FULL,
    },
    database_namespaces={DatabaseNamespace.SETTING},
    visible_to=(RoleType.AGENT,),
)
def set_wifi_status(on: bool) -> None:
    """Enable / Disable wifi

    Args:
        on: If we want to turn on wifi

    Raises:
        ValueError:         If wifi is already turned on / off
        PermissionError:    If low battery mode is on, in which case we are not allowed to turn on wifi
    """
    validate_type(on, "on", bool)

    if on and get_low_battery_mode_status():
        raise PermissionError("Wifi cannot be turned on in low battery mode")
    set_boolean_settings(setting_name="wifi", on=on)


@register_as_tool(
    toolboxes={
        ToolboxName.FULL,
    },
    database_namespaces={DatabaseNamespace.SETTING},
    visible_to=(RoleType.AGENT,),
)
def get_wifi_status() -> bool:
    """Request wifi status

    Returns:
        Boolean indicating if the wifi is on
    """
    return get_boolean_settings("wifi")


@register_as_tool(
    toolboxes={
        ToolboxName.FULL,
    },
    database_namespaces={DatabaseNamespace.SETTING},
    visible_to=(RoleType.AGENT,),
)
def get_current_location() -> dict[Literal["latitude", "longitude"], float]:
    """Request current location latitude and longitude

    Returns:
        A dictionary of latitude and longitude

    Raises:
        PermissionError:    If location_service is not turned on, we are not allowed to access current location
    """
    if not get_location_service_status():
        raise PermissionError("Location service is not enabled.")
    current_context = get_current_context()
    setting_database = current_context.get_database(DatabaseNamespace.SETTING)
    return {
        "latitude": setting_database["latitude"][0],
        "longitude": setting_database["longitude"][0],
    }
