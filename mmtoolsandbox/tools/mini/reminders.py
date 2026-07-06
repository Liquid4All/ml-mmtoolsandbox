# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""MINI reminders tool — unified reminder management (native, polars-backed)."""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.datasets.names import DatasetName


@register_as_tool(
    toolboxes={DatasetName.MINI},
    database_namespaces={DatabaseNamespace.REMINDER},
    visible_to=(RoleType.AGENT,),
)
def reminders(
    action: Literal["add", "modify", "delete", "search"],
    # add/modify params
    reminder_id: str | NotGiven = NOT_GIVEN,
    content: str | NotGiven = NOT_GIVEN,
    reminder_datetime: str | NotGiven = NOT_GIVEN,
    latitude: float | None | NotGiven = NOT_GIVEN,
    longitude: float | None | NotGiven = NOT_GIVEN,
    # search params
    creation_datetime_lowerbound: str | NotGiven = NOT_GIVEN,
    creation_datetime_upperbound: str | NotGiven = NOT_GIVEN,
    reminder_datetime_lowerbound: str | NotGiven = NOT_GIVEN,
    reminder_datetime_upperbound: str | NotGiven = NOT_GIVEN,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Add, modify, delete, or search reminders by time and location.

    Actions:
        add - Add reminder. Requires content, reminder_datetime.
            Optional: latitude, longitude.
        modify - Modify reminder. Requires reminder_id. At least one of
            content, reminder_datetime, latitude, longitude.
        delete - Delete reminder. Requires reminder_id. Irreversible.
        search - Search reminders. Optional: reminder_id (exact match),
            content (fuzzy match), creation_datetime_lowerbound,
            creation_datetime_upperbound, reminder_datetime_lowerbound,
            reminder_datetime_upperbound, latitude (exact), longitude (exact).

    Args:
        action: The reminder action.
        reminder_id: Reminder UUID string (for modify/delete/search).
            Returned by add.
        content: Reminder text (e.g., "Pick up groceries").
        reminder_datetime: When to remind, as an ISO 8601 datetime string
            (e.g., '2026-04-17T10:00:00+00:00').
        latitude: Location latitude (e.g., 37.3346).
        longitude: Location longitude (e.g., -122.0091).
        creation_datetime_lowerbound: Min creation time, ISO 8601 datetime string.
        creation_datetime_upperbound: Max creation time, ISO 8601 datetime string.
        reminder_datetime_lowerbound: Min reminder time, ISO 8601 datetime string.
        reminder_datetime_upperbound: Max reminder time, ISO 8601 datetime string.

    Returns:
        For add: reminder_id string. Pass to modify/delete/search.
        For modify/delete: None.
        For search: list of dicts with reminder_id, content,
            creation_datetime, reminder_datetime, latitude, longitude.
    """
    import mmtoolsandbox.tools.tool_sandbox.reminder as m

    if action == "add":
        kwargs: dict[str, Any] = {
            "content": content,
            "reminder_datetime": reminder_datetime,
        }
        if latitude is not NOT_GIVEN:
            kwargs["latitude"] = latitude
        if longitude is not NOT_GIVEN:
            kwargs["longitude"] = longitude
        return m.add_reminder(**kwargs)
    elif action == "modify":
        kwargs = {"reminder_id": reminder_id}
        if content is not NOT_GIVEN:
            kwargs["content"] = content
        if reminder_datetime is not NOT_GIVEN:
            kwargs["reminder_datetime"] = reminder_datetime
        if latitude is not NOT_GIVEN:
            kwargs["latitude"] = latitude
        if longitude is not NOT_GIVEN:
            kwargs["longitude"] = longitude
        return m.modify_reminder(**kwargs)
    elif action == "delete":
        return m.remove_reminder(reminder_id=reminder_id)
    elif action == "search":
        kwargs = {}
        if reminder_id is not NOT_GIVEN:
            kwargs["reminder_id"] = reminder_id
        if content is not NOT_GIVEN:
            kwargs["content"] = content
        if creation_datetime_lowerbound is not NOT_GIVEN:
            kwargs["creation_datetime_lowerbound"] = creation_datetime_lowerbound
        if creation_datetime_upperbound is not NOT_GIVEN:
            kwargs["creation_datetime_upperbound"] = creation_datetime_upperbound
        if reminder_datetime_lowerbound is not NOT_GIVEN:
            kwargs["reminder_datetime_lowerbound"] = reminder_datetime_lowerbound
        if reminder_datetime_upperbound is not NOT_GIVEN:
            kwargs["reminder_datetime_upperbound"] = reminder_datetime_upperbound
        if latitude is not NOT_GIVEN:
            kwargs["latitude"] = latitude
        if longitude is not NOT_GIVEN:
            kwargs["longitude"] = longitude
        return m.search_reminder(**kwargs)
    else:
        raise ValueError(f"Unknown action: {action}")
