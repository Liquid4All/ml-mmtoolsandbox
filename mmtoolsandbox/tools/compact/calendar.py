# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""COMPACT Calendar tools — merges calendar CRUD (3->1) and calendar search (2->1)."""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution_context import (
    CalendarRecurrenceFrequencyType,
    RoleType,
)
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.datasets.names import DatasetName
from mmtoolsandbox.tools.compact import mark_compact_tools_absorbed_by


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.tool_sandbox.calendar as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# Strategy 5: Calendar management (3 -> 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces={DatabaseNamespace.CALENDARS},
    visible_to=(RoleType.AGENT,),
)
def manage_calendar(
    action: Literal["create", "modify", "delete"],
    title: str | NotGiven = NOT_GIVEN,
    calendar_id: str | NotGiven = NOT_GIVEN,
) -> str | None:
    """Create, modify, or delete a calendar.

    Actions:
        create: Create a new calendar. Requires title.
        modify: Modify an existing calendar's title. Requires calendar_id
            and title.
        delete: Remove a calendar and all its events. Requires calendar_id.

    Args:
        action: The calendar operation to perform.
        title: Title for the calendar (required for create and modify).
            E.g. Work.
        calendar_id: UUID of the calendar (required for modify and delete).

    Returns:
        UUID of the created calendar (for create), None otherwise.

    Raises:
        ValueError: If no matching calendar found for the given
            calendar_id (for modify, delete).
    """
    if action == "create":
        return _get("create_calendar")(title=title)
    elif action == "modify":
        _get("modify_calendar")(calendar_id=calendar_id, title=title)
        return None
    elif action == "delete":
        _get("remove_calendar")(calendar_id=calendar_id)
        return None
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Strategy 4: Calendar search (2 -> 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces={
        DatabaseNamespace.CALENDARS,
        DatabaseNamespace.CALENDAR_EVENTS,
        DatabaseNamespace.CALENDAR_EVENTS_RECURRENCE_STAGING,
    },
    visible_to=(RoleType.AGENT,),
)
def calendar_search(
    target: Literal["calendars", "events"],
    # -- shared --
    calendar_id: str | NotGiven = NOT_GIVEN,
    title: str | NotGiven = NOT_GIVEN,
    # -- events only --
    calendar_event_id: str | NotGiven = NOT_GIVEN,
    description: str | NotGiven = NOT_GIVEN,
    datetime_range_lowerbound: str | NotGiven = NOT_GIVEN,
    datetime_range_upperbound: str | NotGiven = NOT_GIVEN,
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
) -> list[dict[str, Any]]:
    """Search calendars or calendar events based on criteria.

    Targets:
        calendars: Search calendars by calendar_id (exact) and/or title
            (semantic). At least one criterion required.
        events: Search calendar events by multiple criteria. Each field uses
            exact, semantic, or range matching. When datetime_range_lowerbound
            and datetime_range_upperbound are provided, recurring event
            instances are expanded.

    Args:
        target: Whether to search "calendars" or "events".
        calendar_id: Calendar UUID (exact match for both targets).
        title: Title query (semantic match for both targets).
        calendar_event_id: Event UUID (exact match, events only).
        description: Description query (semantic match, events only).
        datetime_range_lowerbound: ISO 8601 datetime lower bound for event
            overlap search (events only). Must be paired with upperbound.
        datetime_range_upperbound: ISO 8601 datetime upper bound for event
            overlap search (events only). Must be paired with lowerbound.
        is_all_day: All-day event filter (exact match, events only).
        recurrence_frequency: Recurrence frequency enum (exact, events only).
        recurrence_interval: Recurrence interval (exact, events only).
        recurrence_until_datetime: ISO 8601 recurrence end (exact, events
            only).
        latitude: Location latitude, matched with 0.01 degree tolerance
            (events only).
        longitude: Location longitude, matched with 0.01 degree tolerance
            (events only).
        place_id: Location place_id (exact match, events only).
        formatted_address: Address query (semantic match, events only).
        attendee_names: List of attendee names (semantic match, events only).
            All names must each find at least one match.
        attendee_emails: List of attendee emails (exact match, events only).
            All emails must each find a match.

    Returns:
        List of matching calendars or calendar events.

    Raises:
        ValueError: If all search criteria are not given, or if datetime
            bounds are not both provided/omitted.
    """
    if target == "calendars":
        kwargs: dict[str, Any] = {}
        if calendar_id is not NOT_GIVEN:
            kwargs["calendar_id"] = calendar_id
        if title is not NOT_GIVEN:
            kwargs["title"] = title
        return _get("search_calendars")(**kwargs)
    elif target == "events":
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
        raise ValueError(f"Unknown target: {target}")


mark_compact_tools_absorbed_by(
    "manage_calendar",
    "create_calendar",
    "modify_calendar",
    "remove_calendar",
)
mark_compact_tools_absorbed_by(
    "calendar_search",
    "search_calendars",
    "search_calendar_events",
)
