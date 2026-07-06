# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""Consolidated Calendar Event tools for the MEDIUM toolbox.

CRUD consolidation for calendar events: create, modify, delete merged into
a single ``manage_calendar_event`` tool.
"""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution_context import (
    CalendarRecurrenceFrequencyType,
    RoleType,
)
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.toolbox.names import ToolboxName
from mmtoolsandbox.tools.consolidated import mark_tools_absorbed_by


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.tool_sandbox.calendar as m

    return getattr(m, name)


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces={
        DatabaseNamespace.CALENDAR_EVENTS,
        DatabaseNamespace.CALENDAR_EVENTS_RECURRENCE_STAGING,
    },
    visible_to=(RoleType.AGENT,),
)
def manage_calendar_event(
    action: Literal["create", "modify", "delete"],
    calendar_event_id: str | NotGiven = NOT_GIVEN,
    calendar_id: str | NotGiven = NOT_GIVEN,
    title: str | NotGiven = NOT_GIVEN,
    start_datetime: str | NotGiven = NOT_GIVEN,
    end_datetime: str | NotGiven = NOT_GIVEN,
    description: str | None = None,
    is_all_day: bool | NotGiven = NOT_GIVEN,
    recurrence_frequency: CalendarRecurrenceFrequencyType | None = None,
    recurrence_interval: int | None = None,
    recurrence_until_datetime: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    place_id: str | None = None,
    formatted_address: str | None = None,
    attendee_names: list[str] | None = None,
    attendee_emails: list[str] | None = None,
) -> str | None:
    """Manage calendar events: create, modify, or delete.

    Actions:
        create: Create a new calendar event. Requires calendar_id, title,
            start_datetime, and end_datetime. All other fields are optional.
        modify: Update an existing calendar event. Requires calendar_event_id
            and at least one field to update. For recurring events, modifying
            the parent affects all future instances.
        delete: Remove a calendar event. Requires calendar_event_id. For a
            single recurring instance, only that instance is removed. To
            delete all instances, use the recurrence_parent_id.

    Args:
        action: The operation to perform.
        calendar_event_id: UUID of the calendar event (for modify, delete).
        calendar_id: UUID of the calendar this event belongs to (for create,
            modify).
        title: Title or summary for the event (for create, modify).
        start_datetime: ISO 8601 datetime string with UTC timezone offset
            for event start time, inclusive. E.g. 2024-12-09T15:00-08:00
            (for create, modify).
        end_datetime: ISO 8601 datetime string with UTC timezone offset
            for event end time, exclusive. E.g. 2024-12-09T16:00-08:00
            (for create, modify).
        description: Detailed description of the event (for create, modify).
        is_all_day: Whether this is an all-day event (for create, modify).
        recurrence_frequency: Unit of recurrence frequency (for create,
            modify).
        recurrence_interval: Combined with frequency, forms a recurrence
            rule. E.g. interval=2 + frequency=WEEKLY means biweekly
            (for create, modify).
        recurrence_until_datetime: ISO 8601 representing end of recurrence
            (for create, modify).
        latitude: Latitude of the event location (for create, modify).
        longitude: Longitude of the event location (for create, modify).
        place_id: Place ID of the event location (for create, modify).
        formatted_address: Formatted address of the event location
            (for create, modify).
        attendee_names: List of attendee names. Forms a 1-to-1 mapping with
            attendee_emails (for create, modify).
        attendee_emails: List of attendee emails. Use null for attendees
            without an email (for create, modify).

    Returns:
        The calendar event UUID string when action is create, None otherwise.

    Raises:
        ValueError: If required parameters are missing, no update fields
            are provided for modify, or attendee lists don't match.
        NoDataError: If calendar_event_id not found (for modify, delete).
    """
    if action == "create":
        kwargs: dict[str, Any] = {
            "calendar_id": calendar_id,
            "title": title,
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
        }
        if description is not None:
            kwargs["description"] = description
        if is_all_day is not NOT_GIVEN:
            kwargs["is_all_day"] = is_all_day
        if recurrence_frequency is not None:
            kwargs["recurrence_frequency"] = recurrence_frequency
        if recurrence_interval is not None:
            kwargs["recurrence_interval"] = recurrence_interval
        if recurrence_until_datetime is not None:
            kwargs["recurrence_until_datetime"] = recurrence_until_datetime
        if latitude is not None:
            kwargs["latitude"] = latitude
        if longitude is not None:
            kwargs["longitude"] = longitude
        if place_id is not None:
            kwargs["place_id"] = place_id
        if formatted_address is not None:
            kwargs["formatted_address"] = formatted_address
        if attendee_names is not None:
            kwargs["attendee_names"] = attendee_names
        if attendee_emails is not None:
            kwargs["attendee_emails"] = attendee_emails
        return _get("create_calendar_event")(**kwargs)
    elif action == "modify":
        kwargs = {"calendar_event_id": calendar_event_id}
        if calendar_id is not NOT_GIVEN:
            kwargs["calendar_id"] = calendar_id
        if title is not NOT_GIVEN:
            kwargs["title"] = title
        if start_datetime is not NOT_GIVEN:
            kwargs["start_datetime"] = start_datetime
        if end_datetime is not NOT_GIVEN:
            kwargs["end_datetime"] = end_datetime
        if description is not None:
            kwargs["description"] = description
        if is_all_day is not NOT_GIVEN:
            kwargs["is_all_day"] = is_all_day
        if recurrence_frequency is not None:
            kwargs["recurrence_frequency"] = recurrence_frequency
        if recurrence_interval is not None:
            kwargs["recurrence_interval"] = recurrence_interval
        if recurrence_until_datetime is not None:
            kwargs["recurrence_until_datetime"] = recurrence_until_datetime
        if latitude is not None:
            kwargs["latitude"] = latitude
        if longitude is not None:
            kwargs["longitude"] = longitude
        if place_id is not None:
            kwargs["place_id"] = place_id
        if formatted_address is not None:
            kwargs["formatted_address"] = formatted_address
        if attendee_names is not None:
            kwargs["attendee_names"] = attendee_names
        if attendee_emails is not None:
            kwargs["attendee_emails"] = attendee_emails
        _get("modify_calendar_event")(**kwargs)
        return None
    elif action == "delete":
        _get("remove_calendar_event")(calendar_event_id=calendar_event_id)
        return None
    else:
        raise ValueError(f"Unknown action: {action}")


mark_tools_absorbed_by(
    "manage_calendar_event",
    "create_calendar_event",
    "modify_calendar_event",
    "remove_calendar_event",
)
