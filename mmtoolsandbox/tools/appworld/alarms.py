"""
Alarms API tools for MMToolSandbox.

This module wraps AppWorld's Phone alarm-related APIs as MMToolSandbox-compatible
tools with transparent authentication (no login/logout tools).

Split from phone.py — all bridge.call_api() calls still use "phone" as the app name.
"""

from typing import Any

from mmtoolsandbox.appworld.bridge import get_appworld_bridge
from mmtoolsandbox.appworld.state import (
    get_appworld_state,
    requires_network,
)
from mmtoolsandbox.tools.appworld import register_appworld_tool


@register_appworld_tool("alarms")
@requires_network
def alarms_show_alarms(
    page_index: int | None = 0, page_limit: int | None = 5
) -> list[dict[str, Any]]:
    """
    Get a list of alarms.

    Args:
        page_index: The index of the page to return. Must be >= 0. (optional)
        page_limit: The maximum number of results to return per page. Must be between 1 and 20. (optional)

    Returns:
        Success: [{"alarm_id": int, "time": str, "repeat_days": list[str], "label": str, "enabled": bool, "snooze_minutes": float, "vibration": bool, "created_at": str (datetime), ...}]
        Failure: {"message": str}
    """
    bridge = get_appworld_bridge()
    appworld_state = get_appworld_state()

    params: dict[str, Any] = {
        "access_token": appworld_state.get_access_token("phone"),
    }
    if page_index is not None:
        params["page_index"] = page_index
    if page_limit is not None:
        params["page_limit"] = page_limit

    response = bridge.call_api("phone", "show_alarms", method="get", **params)

    if isinstance(response, dict) and "error" in response:
        raise ValueError(f"API error: {response['error']}")

    # Extract list from response
    if isinstance(response, list):
        return list(response)
    if isinstance(response, dict):
        # Try common list keys
        for key in response:
            if isinstance(response[key], list):
                return list(response[key])
    return []


@register_appworld_tool("alarms")
@requires_network
def alarms_create_alarm(
    time: str,
    repeat_days: list[str] | None = None,
    label: str | None = None,
    enabled: bool = True,
    snooze_minutes: int = 15,
    vibration: bool | None = True,
) -> dict[str, Any]:
    """
    Create a new alarm.

    Args:
        time: The time of the alarm in HH:MM format.
        repeat_days: Days on which the alarm repeats. (optional)
        label: The label for the alarm. (optional)
        enabled: Whether the alarm is enabled or not. Defaults to true.
        snooze_minutes: The duration of snooze in minutes. Use 0 for no snooze. Defaults to 15. Must be >= 0.
        vibration: Whether the alarm should vibrate or not. (optional)

    Returns:
        Success: {"message": str, "alarm_id": int}
        Failure: {"message": str}
    """
    bridge = get_appworld_bridge()
    appworld_state = get_appworld_state()

    params: dict[str, Any] = {
        "access_token": appworld_state.get_access_token("phone"),
        "time": time,
        "enabled": enabled,
        "snooze_minutes": snooze_minutes,
    }
    if repeat_days is not None:
        params["repeat_days"] = repeat_days
    if label is not None:
        params["label"] = label
    if vibration is not None:
        params["vibration"] = vibration

    response = bridge.call_api("phone", "create_alarm", method="post", **params)

    if isinstance(response, dict) and "error" in response:
        raise ValueError(f"API error: {response['error']}")

    return dict(response) if isinstance(response, dict) else {}


@register_appworld_tool("alarms")
@requires_network
def alarms_show_alarm(alarm_id: int) -> dict[str, Any]:
    """
    Show alarm details.

    Args:
        alarm_id: ID of the alarm to show.

    Returns:
        Success: {"alarm_id": int, "time": str, "repeat_days": list[str], "label": str, "enabled": bool, "snooze_minutes": float, "vibration": bool, "created_at": str (datetime), ...}
        Failure: {"message": str}
    """
    bridge = get_appworld_bridge()
    appworld_state = get_appworld_state()

    params: dict[str, Any] = {
        "access_token": appworld_state.get_access_token("phone"),
        "alarm_id": alarm_id,
    }

    response = bridge.call_api("phone", "show_alarm", method="get", **params)

    if isinstance(response, dict) and "error" in response:
        raise ValueError(f"API error: {response['error']}")

    return dict(response) if isinstance(response, dict) else {}


@register_appworld_tool("alarms")
@requires_network
def alarms_update_alarm(
    alarm_id: int,
    time: str | None = None,
    repeat_days: list[str] | None = None,
    label: str | None = None,
    enabled: bool | None = None,
    snooze_minutes: int | None = None,
    vibration: bool | None = None,
) -> dict[str, Any]:
    """
    Update an alarm's settings.

    Args:
        alarm_id: ID of the alarm to be updated.
        time: The updated time of the alarm in HH:MM format. (optional)
        repeat_days: The updated days on which the alarm should repeat. (optional)
        label: The updated label for the alarm. (optional)
        enabled: Whether the alarm is enabled or not. (optional)
        snooze_minutes: The updated duration of snooze in minutes. Use 0 for no snooze. Must be >= 0. (optional)
        vibration: Whether the alarm should vibrate or not. (optional)

    Returns:
        Success: {"message": str}
        Failure: {"message": str}
    """
    bridge = get_appworld_bridge()
    appworld_state = get_appworld_state()

    params: dict[str, Any] = {
        "access_token": appworld_state.get_access_token("phone"),
        "alarm_id": alarm_id,
    }
    if time is not None:
        params["time"] = time
    if repeat_days is not None:
        params["repeat_days"] = repeat_days
    if label is not None:
        params["label"] = label
    if enabled is not None:
        params["enabled"] = enabled
    if snooze_minutes is not None:
        params["snooze_minutes"] = snooze_minutes
    if vibration is not None:
        params["vibration"] = vibration

    response = bridge.call_api("phone", "update_alarm", method="patch", **params)

    if isinstance(response, dict) and "error" in response:
        raise ValueError(f"API error: {response['error']}")

    return dict(response) if isinstance(response, dict) else {}


@register_appworld_tool("alarms")
@requires_network
def alarms_delete_alarm(alarm_id: int) -> dict[str, Any]:
    """
    Delete an alarm.

    Args:
        alarm_id: ID of the alarm to delete.

    Returns:
        Success: {"message": str}
        Failure: {"message": str}
    """
    bridge = get_appworld_bridge()
    appworld_state = get_appworld_state()

    params: dict[str, Any] = {
        "access_token": appworld_state.get_access_token("phone"),
        "alarm_id": alarm_id,
    }

    response = bridge.call_api("phone", "delete_alarm", method="delete", **params)

    if isinstance(response, dict) and "error" in response:
        raise ValueError(f"API error: {response['error']}")

    return dict(response) if isinstance(response, dict) else {}


@register_appworld_tool("alarms")
@requires_network
def alarms_get_current_date_and_time() -> dict[str, Any]:
    """
    Show current date and time.

    Returns:
        Success: {"date": str, "time": str}
        Failure: {"message": str}
    """
    bridge = get_appworld_bridge()

    params: dict[str, Any] = {}

    response = bridge.call_api(
        "phone", "get_current_date_and_time", method="get", **params
    )

    if isinstance(response, dict) and "error" in response:
        raise ValueError(f"API error: {response['error']}")

    return dict(response) if isinstance(response, dict) else {}
