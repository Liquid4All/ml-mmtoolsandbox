# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""COMPACT alarms tools — CRUD + list for alarm management."""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.datasets.names import DatasetName
from mmtoolsandbox.tools.compact import mark_compact_tools_absorbed_by


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.appworld.alarms as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# CRUD+List: Alarm management (absorbs alarms_show_alarms)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def alarms_manage_alarm(
    action: Literal["create", "show", "update", "delete", "list"],
    alarm_id: int | NotGiven = NOT_GIVEN,
    time: str | NotGiven = NOT_GIVEN,
    repeat_days: list[str] | None = None,
    label: str | None = None,
    enabled: bool | None = None,
    snooze_minutes: int | None = None,
    vibration: bool | None = None,
    # list-action params
    page_index: int | None = 0,
    page_limit: int | None = 5,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Manage alarms: create, view, update, delete, or list.

    Actions:
        create: Create a new alarm. Requires time (HH:MM format).
            Optionally include repeat_days, label, enabled (default true),
            snooze_minutes (default 15), and vibration (default true).
        show: Show alarm details. Requires alarm_id.
        update: Update alarm settings. Requires alarm_id and at least one
            field to update.
        delete: Delete an alarm. Requires alarm_id.
        list: List all alarms. Supports pagination.

    Args:
        action: The operation to perform.
        alarm_id: The alarm ID (for show, update, delete).
        time: The time of the alarm in HH:MM format (for create, update).
        repeat_days: Days on which the alarm repeats (for create, update).
        label: The label for the alarm (for create, update).
        enabled: Whether the alarm is enabled (for create, update).
        snooze_minutes: Snooze duration in minutes; 0 for no snooze
            (for create, update).
        vibration: Whether the alarm should vibrate (for create, update).
        page_index: Zero-based page index (for list).
        page_limit: Maximum results per page (for list).

    Returns:
        Alarm details, action confirmation, or list of alarms.

    Raises:
        ConnectionError: If network is unavailable.
    """
    if action == "create":
        kwargs: dict[str, Any] = {"time": time}
        if repeat_days is not None:
            kwargs["repeat_days"] = repeat_days
        if label is not None:
            kwargs["label"] = label
        if enabled is not None:
            kwargs["enabled"] = enabled
        if snooze_minutes is not None:
            kwargs["snooze_minutes"] = snooze_minutes
        if vibration is not None:
            kwargs["vibration"] = vibration
        return _get("alarms_create_alarm")(**kwargs)
    elif action == "show":
        return _get("alarms_show_alarm")(alarm_id=alarm_id)
    elif action == "update":
        kwargs = {"alarm_id": alarm_id}
        if time is not NOT_GIVEN:
            kwargs["time"] = time
        if repeat_days is not None:
            kwargs["repeat_days"] = repeat_days
        if label is not None:
            kwargs["label"] = label
        if enabled is not None:
            kwargs["enabled"] = enabled
        if snooze_minutes is not None:
            kwargs["snooze_minutes"] = snooze_minutes
        if vibration is not None:
            kwargs["vibration"] = vibration
        return _get("alarms_update_alarm")(**kwargs)
    elif action == "delete":
        return _get("alarms_delete_alarm")(alarm_id=alarm_id)
    elif action == "list":
        kwargs = {}
        if page_index is not None:
            kwargs["page_index"] = page_index
        if page_limit is not None:
            kwargs["page_limit"] = page_limit
        return _get("alarms_show_alarms")(**kwargs)
    else:
        raise ValueError(f"Unknown action: {action}")


mark_compact_tools_absorbed_by(
    "alarms_manage_alarm",
    "alarms_show_alarms",
)
