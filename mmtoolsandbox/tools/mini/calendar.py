# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""MINI Calendar tool -- 1 workflow-based tool covering calendars and events.

calendar: Unified tool for creating, modifying, deleting, and searching
          both calendars and calendar events.
"""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution_context import (
    CalendarRecurrenceFrequencyType,
    RoleType,
)
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.datasets.names import DatasetName


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.tool_sandbox.calendar as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# calendar -- "I want to manage my calendar"
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.MINI},
    database_namespaces={
        DatabaseNamespace.CALENDARS,
        DatabaseNamespace.CALENDAR_EVENTS,
        DatabaseNamespace.CALENDAR_EVENTS_RECURRENCE_STAGING,
    },
    visible_to=(RoleType.AGENT,),
)
def calendar(
    entity: Literal["calendar", "event"],
    action: Literal["create", "modify", "delete", "search"],
    # -- calendar params --
    calendar_id: str | NotGiven = NOT_GIVEN,
    title: str | NotGiven = NOT_GIVEN,
    # -- event params --
    calendar_event_id: str | NotGiven = NOT_GIVEN,
    start_datetime: str | NotGiven = NOT_GIVEN,
    end_datetime: str | NotGiven = NOT_GIVEN,
    description: str | None | NotGiven = NOT_GIVEN,
    is_all_day: bool | NotGiven = NOT_GIVEN,
    recurrence_frequency: CalendarRecurrenceFrequencyType | NotGiven = NOT_GIVEN,
    recurrence_interval: int | NotGiven = NOT_GIVEN,
    recurrence_until_datetime: str | NotGiven = NOT_GIVEN,
    latitude: float | NotGiven = NOT_GIVEN,
    longitude: float | NotGiven = NOT_GIVEN,
    place_id: str | NotGiven = NOT_GIVEN,
    formatted_address: str | NotGiven = NOT_GIVEN,
    attendee_names: list[str] | NotGiven = NOT_GIVEN,
    attendee_emails: list[str] | NotGiven = NOT_GIVEN,
    # -- event search params --
    datetime_range_lowerbound: str | NotGiven = NOT_GIVEN,
    datetime_range_upperbound: str | NotGiven = NOT_GIVEN,
) -> dict[str, Any] | list[dict[str, Any]] | None:
    """Create, modify, delete, or search calendars and calendar events.

    Dispatch is based on entity x action:

    calendar:
        create: Create a new calendar. Requires title. Returns the
            calendar UUID.
        modify: Modify a calendar's title. Requires calendar_id, title.
        delete: Remove a calendar and all its events. Requires calendar_id.
        search: Search calendars. Optional: calendar_id (exact match),
            title (semantic match). At least one search criterion required.

    event:
        create: Create a calendar event. Requires calendar_id, title,
            start_datetime, end_datetime. Optional: description, is_all_day,
            recurrence_frequency, recurrence_interval,
            recurrence_until_datetime, latitude, longitude, place_id,
            formatted_address, attendee_names, attendee_emails. Returns the
            event UUID.
        modify: Modify a calendar event. Requires calendar_event_id and at
            least one field to update. For recurring events, modifying the
            parent affects all future instances.
        delete: Remove a calendar event. Requires calendar_event_id. For a
            single recurring instance, only that instance is removed. Use
            the recurrence_parent_id to delete all instances.
        search: Search calendar events. At least one criterion required.
            Optional: calendar_id, calendar_event_id, title, description,
            datetime_range_lowerbound, datetime_range_upperbound, is_all_day,
            recurrence_frequency, recurrence_interval,
            recurrence_until_datetime, latitude, longitude, place_id,
            formatted_address, attendee_names, attendee_emails. Note:
            datetime_range_lowerbound and datetime_range_upperbound must both
            be provided or both omitted. Recurring event instances only
            appear when datetime range is provided.

    Args:
        entity: Whether to operate on a calendar or an event.
        action: The operation to perform.
        calendar_id: UUID of the calendar (for calendar modify/delete/search,
            event create/modify/search).
        title: Title for the calendar or event.
        calendar_event_id: UUID of the calendar event (for event
            modify/delete/search).
        start_datetime: ISO 8601 datetime with UTC offset for event start,
            inclusive. E.g. 2024-12-09T15:00-08:00 (for event create/modify).
        end_datetime: ISO 8601 datetime with UTC offset for event end,
            exclusive. E.g. 2024-12-09T16:00-08:00 (for event create/modify).
        description: Detailed event description (for event create/modify/
            search).
        is_all_day: Whether this is an all-day event (for event
            create/modify/search).
        recurrence_frequency: Recurrence frequency unit (for event
            create/modify/search).
        recurrence_interval: Recurrence interval combined with frequency
            forms the rule. E.g. 2 + WEEKLY = biweekly (for event
            create/modify/search).
        recurrence_until_datetime: ISO 8601 end of recurrence (for event
            create/modify/search).
        latitude: Event location latitude (for event create/modify/search).
        longitude: Event location longitude (for event create/modify/search).
        place_id: Event location place ID (for event create/modify/search).
        formatted_address: Event location address (for event
            create/modify/search).
        attendee_names: List of attendee names, 1-to-1 with attendee_emails
            (for event create/modify/search).
        attendee_emails: List of attendee emails. Use null for attendees
            without email (for event create/modify/search).
        datetime_range_lowerbound: ISO 8601 lower bound for event search.
            Any overlap with event time range is a match (for event search).
        datetime_range_upperbound: ISO 8601 upper bound for event search.
            Any overlap with event time range is a match (for event search).

    Returns:
        Calendar/event UUID (for create), list of matches (for search),
        or None (for modify/delete).

    Raises:
        ValueError: If required parameters are missing or invalid
            entity/action combination.
        NoDataError: If calendar or event not found (for modify/delete).
    """
    if entity == "calendar":
        if action == "create":
            return _get("create_calendar")(title=title)
        elif action == "modify":
            return _get("modify_calendar")(calendar_id=calendar_id, title=title)
        elif action == "delete":
            return _get("remove_calendar")(calendar_id=calendar_id)
        elif action == "search":
            kwargs: dict[str, Any] = {}
            if calendar_id is not NOT_GIVEN:
                kwargs["calendar_id"] = calendar_id
            if title is not NOT_GIVEN:
                kwargs["title"] = title
            return _get("search_calendars")(**kwargs)
        else:
            raise ValueError(f"Unknown action '{action}' for entity 'calendar'")

    elif entity == "event":
        if action == "create":
            kwargs = {
                "calendar_id": calendar_id,
                "title": title,
                "start_datetime": start_datetime,
                "end_datetime": end_datetime,
            }
            if description is not NOT_GIVEN and description is not None:
                kwargs["description"] = description
            if is_all_day is not NOT_GIVEN:
                kwargs["is_all_day"] = is_all_day
            if recurrence_frequency is not NOT_GIVEN:
                kwargs["recurrence_frequency"] = recurrence_frequency
            if recurrence_interval is not NOT_GIVEN:
                kwargs["recurrence_interval"] = recurrence_interval
            if recurrence_until_datetime is not NOT_GIVEN:
                kwargs["recurrence_until_datetime"] = recurrence_until_datetime
            if latitude is not NOT_GIVEN:
                kwargs["latitude"] = latitude
            if longitude is not NOT_GIVEN:
                kwargs["longitude"] = longitude
            if place_id is not NOT_GIVEN:
                kwargs["place_id"] = place_id
            if formatted_address is not NOT_GIVEN:
                kwargs["formatted_address"] = formatted_address
            if attendee_names is not NOT_GIVEN:
                kwargs["attendee_names"] = attendee_names
            if attendee_emails is not NOT_GIVEN:
                kwargs["attendee_emails"] = attendee_emails
            return _get("create_calendar_event")(**kwargs)
        elif action == "modify":
            kwargs = {"calendar_event_id": calendar_event_id}
            if calendar_id is not NOT_GIVEN:
                kwargs["calendar_id"] = calendar_id
            if title is not NOT_GIVEN:
                kwargs["title"] = title
            if description is not NOT_GIVEN:
                kwargs["description"] = description
            if start_datetime is not NOT_GIVEN:
                kwargs["start_datetime"] = start_datetime
            if end_datetime is not NOT_GIVEN:
                kwargs["end_datetime"] = end_datetime
            if is_all_day is not NOT_GIVEN:
                kwargs["is_all_day"] = is_all_day
            if recurrence_frequency is not NOT_GIVEN:
                kwargs["recurrence_frequency"] = recurrence_frequency
            if recurrence_interval is not NOT_GIVEN:
                kwargs["recurrence_interval"] = recurrence_interval
            if recurrence_until_datetime is not NOT_GIVEN:
                kwargs["recurrence_until_datetime"] = recurrence_until_datetime
            if latitude is not NOT_GIVEN:
                kwargs["latitude"] = latitude
            if longitude is not NOT_GIVEN:
                kwargs["longitude"] = longitude
            if place_id is not NOT_GIVEN:
                kwargs["place_id"] = place_id
            if formatted_address is not NOT_GIVEN:
                kwargs["formatted_address"] = formatted_address
            if attendee_names is not NOT_GIVEN:
                kwargs["attendee_names"] = attendee_names
            if attendee_emails is not NOT_GIVEN:
                kwargs["attendee_emails"] = attendee_emails
            _get("modify_calendar_event")(**kwargs)
            return None
        elif action == "delete":
            _get("remove_calendar_event")(calendar_event_id=calendar_event_id)
            return None
        elif action == "search":
            kwargs = {}
            if calendar_id is not NOT_GIVEN:
                kwargs["calendar_id"] = calendar_id
            if calendar_event_id is not NOT_GIVEN:
                kwargs["calendar_event_id"] = calendar_event_id
            if title is not NOT_GIVEN:
                kwargs["title"] = title
            if description is not NOT_GIVEN:
                kwargs["description"] = description
            if datetime_range_lowerbound is not NOT_GIVEN:
                kwargs["datetime_range_lowerbound"] = datetime_range_lowerbound
            if datetime_range_upperbound is not NOT_GIVEN:
                kwargs["datetime_range_upperbound"] = datetime_range_upperbound
            if is_all_day is not NOT_GIVEN:
                kwargs["is_all_day"] = is_all_day
            if recurrence_frequency is not NOT_GIVEN:
                kwargs["recurrence_frequency"] = recurrence_frequency
            if recurrence_interval is not NOT_GIVEN:
                kwargs["recurrence_interval"] = recurrence_interval
            if recurrence_until_datetime is not NOT_GIVEN:
                kwargs["recurrence_until_datetime"] = recurrence_until_datetime
            if latitude is not NOT_GIVEN:
                kwargs["latitude"] = latitude
            if longitude is not NOT_GIVEN:
                kwargs["longitude"] = longitude
            if place_id is not NOT_GIVEN:
                kwargs["place_id"] = place_id
            if formatted_address is not NOT_GIVEN:
                kwargs["formatted_address"] = formatted_address
            if attendee_names is not NOT_GIVEN:
                kwargs["attendee_names"] = attendee_names
            if attendee_emails is not NOT_GIVEN:
                kwargs["attendee_emails"] = attendee_emails
            return _get("search_calendar_events")(**kwargs)
        else:
            raise ValueError(f"Unknown action '{action}' for entity 'event'")

    else:
        raise ValueError(f"Unknown entity: {entity}")
