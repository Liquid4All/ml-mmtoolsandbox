# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""MINI alarms tool — unified alarm management."""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.datasets.names import DatasetName


@register_as_tool(
    toolboxes={DatasetName.MINI},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def alarms(
    action: Literal["create", "show", "update", "delete", "list", "current_time"],
    # create/update params
    alarm_id: int | NotGiven = NOT_GIVEN,
    time: str | NotGiven = NOT_GIVEN,
    repeat_days: list[str] | None | NotGiven = NOT_GIVEN,
    label: str | None | NotGiven = NOT_GIVEN,
    enabled: bool | NotGiven = NOT_GIVEN,
    snooze_minutes: int | NotGiven = NOT_GIVEN,
    vibration: bool | None | NotGiven = NOT_GIVEN,
    # list params
    page_index: int | None | NotGiven = NOT_GIVEN,
    page_limit: int | None | NotGiven = NOT_GIVEN,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Create, update, delete, or list alarms, and check the current time.

    Actions:
        create - Create alarm. Requires time (HH:MM, e.g. "07:30").
            Optional: repeat_days, label, enabled, snooze_minutes, vibration.
        show - Show alarm details. Requires alarm_id.
        update - Update alarm. Requires alarm_id. All other fields optional.
        delete - Delete alarm. Requires alarm_id. Irreversible.
        list - List all alarms. Optional: page_index, page_limit.
        current_time - Show current date and time.

    Args:
        action: The alarm action.
        alarm_id: Alarm ID (for show/update/delete). Returned by create.
        time: Alarm time in HH:MM 24-hour format (e.g., "07:30", "22:00").
        repeat_days: Days the alarm repeats (e.g., ["monday", "wednesday"]).
            Omit or null for a one-time alarm.
        label: Alarm label (e.g., "Morning workout").
        enabled: Whether alarm is enabled (default true).
        snooze_minutes: Snooze duration in minutes. 0 disables snooze.
        vibration: Whether alarm vibrates.
        page_index: Zero-based page index (for list).
        page_limit: Results per page (for list).

    Returns:
        For create/show/update: dict with alarm_id, time, label, enabled,
            repeat_days, snooze_minutes, vibration keys. Pass alarm_id to
            show/update/delete.
        For delete: confirmation dict.
        For list: list of alarm dicts.
        For current_time: dict with date and time.

    Raises:
        ValueError: If alarm_id not found (for show/update/delete).
    """
    import mmtoolsandbox.tools.appworld.alarms as m

    if action == "create":
        kwargs: dict[str, Any] = {"time": time}
        if repeat_days is not NOT_GIVEN:
            kwargs["repeat_days"] = repeat_days
        if label is not NOT_GIVEN:
            kwargs["label"] = label
        if enabled is not NOT_GIVEN:
            kwargs["enabled"] = enabled
        if snooze_minutes is not NOT_GIVEN:
            kwargs["snooze_minutes"] = snooze_minutes
        if vibration is not NOT_GIVEN:
            kwargs["vibration"] = vibration
        return m.alarms_create_alarm(**kwargs)
    elif action == "show":
        return m.alarms_show_alarm(alarm_id=alarm_id)
    elif action == "update":
        kwargs = {"alarm_id": alarm_id}
        if time is not NOT_GIVEN:
            kwargs["time"] = time
        if repeat_days is not NOT_GIVEN:
            kwargs["repeat_days"] = repeat_days
        if label is not NOT_GIVEN:
            kwargs["label"] = label
        if enabled is not NOT_GIVEN:
            kwargs["enabled"] = enabled
        if snooze_minutes is not NOT_GIVEN:
            kwargs["snooze_minutes"] = snooze_minutes
        if vibration is not NOT_GIVEN:
            kwargs["vibration"] = vibration
        return m.alarms_update_alarm(**kwargs)
    elif action == "delete":
        return m.alarms_delete_alarm(alarm_id=alarm_id)
    elif action == "list":
        kwargs = {}
        if page_index is not NOT_GIVEN:
            kwargs["page_index"] = page_index
        if page_limit is not NOT_GIVEN:
            kwargs["page_limit"] = page_limit
        return m.alarms_show_alarms(**kwargs)
    elif action == "current_time":
        return m.alarms_get_current_date_and_time()
    else:
        raise ValueError(f"Unknown action: {action}")
