# Copyright © 2026 Apple Inc.

"""Unit tests for mmtoolsandbox.tools.printciple.calendar"""

import uuid
from typing import Any, Iterator

import polars as pl
import pytest

from mmtoolsandbox.common.execution_context import (
    CalendarRecurrenceFrequencyType,
    DatabaseNamespace,
    ExecutionContext,
    get_current_context,
    new_context,
)
from mmtoolsandbox.tools.tool_sandbox.calendar import (
    create_calendar,
    modify_calendar,
    modify_calendar_event,
    remove_calendar,
    remove_calendar_event,
    search_calendar_events,
)


@pytest.fixture
def home_calendar_id() -> str:
    return str(uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="Home"))


@pytest.fixture
def work_calendar_id() -> str:
    return str(uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="Work"))


@pytest.fixture
def clean_up_event_id() -> str:
    return str(uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="Clean up codebase"))


@pytest.fixture
def unitttest_event_id() -> str:
    return str(uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="Add unittest"))


@pytest.fixture
def calendars(home_calendar_id: str, work_calendar_id: str) -> list[dict[str, Any]]:
    """Test calendars

    Returns:

    """
    return [
        {"calendar_id": home_calendar_id, "title": "Home"},
        {"calendar_id": work_calendar_id, "title": "Work"},
    ]


@pytest.fixture
def calendar_events(
    home_calendar_id: str,
    work_calendar_id: str,
    clean_up_event_id: str,
    unitttest_event_id: str,
) -> list[dict[str, Any]]:
    """Test calendar events

    Returns:

    """
    return [
        {
            "calendar_event_id": clean_up_event_id,
            "calendar_id": work_calendar_id,
            "title": "Clean up codebase",
            "start_datetime": "2024-12-08T00:00-08:00",
            "end_datetime": "2024-12-09T00:00-08:00",
            "is_all_day": True,
        },
        {
            "calendar_event_id": unitttest_event_id,
            "calendar_id": work_calendar_id,
            "title": "Add unittest",
            "description": "Add some unittests to this domain.",
            "start_datetime": "2024-12-09T15:00-08:00",
            "end_datetime": "2024-12-09T23:00-08:00",
            "is_all_day": False,
            "recurrence_frequency": CalendarRecurrenceFrequencyType.WEEKLY,
            "recurrence_interval": 1,
            "recurrence_until_datetime": "2025-12-09T23:00-08:00",
            "latitude": 37.334606,
            "longitude": -122.009102,
            "place_id": "ChIJ_Yjh6Za1j4AR8IgGUZGDDTs",
            "formatted_address": "One Apple Park Way, Cupertino, CA 95014, USA",
            "attendee_names": ["John A", "John B", "John C"],
            "attendee_emails": ["john_a@me.com", "john_b@me.com", "john_c@me.com"],
        },
    ]


@pytest.fixture(scope="function", autouse=True)
def execution_context(
    calendars: list[dict[str, Any]], calendar_events: list[dict[str, Any]]
) -> Iterator[None]:
    """Autouse fixture which will setup and teardown execution context before and after each test function

    Returns:

    """
    test_context = ExecutionContext()
    test_context.add_to_database(namespace=DatabaseNamespace.CALENDARS, rows=calendars)
    test_context.add_to_database(
        namespace=DatabaseNamespace.CALENDAR_EVENTS, rows=calendar_events
    )
    with new_context(test_context):
        yield


def test_create_calendar() -> None:
    test_title = "Workout"
    create_calendar(title=test_title)
    current_context = get_current_context()
    calendar_database = current_context.get_database(
        namespace=DatabaseNamespace.CALENDARS
    )
    assert calendar_database["title"][-1] == test_title


def test_modify_calendar(work_calendar_id: str) -> None:
    test_title = "Job"
    modify_calendar(calendar_id=work_calendar_id, title=test_title)
    current_context = get_current_context()
    calendar_database = current_context.get_database(
        namespace=DatabaseNamespace.CALENDARS
    )
    assert (
        calendar_database.filter(pl.col("calendar_id") == work_calendar_id)["title"][0]
        == test_title
    )


def test_remove_calendar(work_calendar_id: str, home_calendar_id: str) -> None:
    remove_calendar(calendar_id=work_calendar_id)
    current_context = get_current_context()
    calendar_database = current_context.get_database(
        namespace=DatabaseNamespace.CALENDARS
    )
    # There should only be one calendar left, which is the home calendar.
    assert calendar_database.select(pl.len())["len"][0] == 1
    assert (
        calendar_database.filter(pl.col("calendar_id") == home_calendar_id).select(
            pl.len()
        )["len"][0]
        == 1
    )
    # Calendar events should be empty
    assert current_context.get_database(
        namespace=DatabaseNamespace.CALENDAR_EVENTS
    ).is_empty()


def test_remove_calendar_event(clean_up_event_id: str) -> None:
    remove_calendar_event(calendar_event_id=clean_up_event_id)
    assert (
        get_current_context()
        .get_database(namespace=DatabaseNamespace.CALENDAR_EVENTS)
        .select(pl.len())["len"][0]
        == 1
    )


def test_remove_calendar_event_recurrence_parent(unitttest_event_id: str) -> None:
    # Search to populate some staging
    _ = search_calendar_events(
        datetime_range_lowerbound="2024-12-08T00:00+00:00",
        datetime_range_upperbound="2025-01-07T23:30+00:00",
    )
    assert (
        get_current_context()
        .get_database(namespace=DatabaseNamespace.CALENDAR_EVENTS_RECURRENCE_STAGING)
        .select(pl.len())["len"][0]
        == 5
    )
    # Remove parent recurrence
    remove_calendar_event(calendar_event_id=unitttest_event_id)
    # Should have 1 entry in CalendarEvents, and empty staging
    assert (
        get_current_context()
        .get_database(namespace=DatabaseNamespace.CALENDAR_EVENTS)
        .select(pl.len())["len"][0]
        == 1
    )
    assert (
        get_current_context()
        .get_database(namespace=DatabaseNamespace.CALENDAR_EVENTS_RECURRENCE_STAGING)
        .is_empty()
    )


def test_remove_calendar_event_recurrence_instance(unitttest_event_id: str) -> None:
    # Search to populate some staging
    search_result = search_calendar_events(
        datetime_range_lowerbound="2024-12-08T00:00+00:00",
        datetime_range_upperbound="2025-01-07T23:30+00:00",
    )
    assert (
        get_current_context()
        .get_database(namespace=DatabaseNamespace.CALENDAR_EVENTS_RECURRENCE_STAGING)
        .select(pl.len())["len"][0]
        == 5
    )
    # Remove last instance
    remove_calendar_event(calendar_event_id=search_result[-1]["calendar_event_id"])
    # Should have 4 entries in staging
    assert (
        get_current_context()
        .get_database(namespace=DatabaseNamespace.CALENDAR_EVENTS_RECURRENCE_STAGING)
        .select(pl.len())["len"][0]
        == 4
    )
    # Start time should be excluded in parent
    assert get_current_context().get_database(
        namespace=DatabaseNamespace.CALENDAR_EVENTS
    ).filter(pl.col("calendar_event_id") == unitttest_event_id).to_dicts()[0][
        "recurrence_exclude_datetime"
    ] == ["2025-01-06T15:00:00-08:00"]


def test_modify_calendar_event(clean_up_event_id: str, work_calendar_id: str) -> None:
    with pytest.raises(ValueError):
        modify_calendar_event(calendar_event_id=clean_up_event_id)
    modify_calendar_event(
        calendar_event_id=clean_up_event_id,
        attendee_names=["John A"],
        attendee_emails=["john_a@me.com"],
    )
    assert (
        get_current_context()
        .get_database(namespace=DatabaseNamespace.CALENDAR_EVENTS)
        .select(pl.len())["len"][0]
        == 2
    )
    assert get_current_context().get_database(
        namespace=DatabaseNamespace.CALENDAR_EVENTS
    ).filter(pl.col("calendar_event_id") == clean_up_event_id).to_dicts()[0] == {
        "attendee_emails": ["john_a@me.com"],
        "attendee_names": ["John A"],
        "calendar_event_id": clean_up_event_id,
        "calendar_id": work_calendar_id,
        "description": None,
        "end_datetime": "2024-12-09T00:00:00-08:00",
        "formatted_address": None,
        "is_all_day": True,
        "latitude": None,
        "longitude": None,
        "place_id": None,
        "recurrence_exclude_datetime": None,
        "recurrence_frequency": None,
        "recurrence_interval": None,
        "recurrence_parent_id": None,
        "recurrence_until_datetime": None,
        "start_datetime": "2024-12-08T00:00:00-08:00",
        "title": "Clean up codebase",
    }


def test_modify_calendar_event_recurrence_parent(
    unitttest_event_id: str, work_calendar_id: str
) -> None:
    # Search to populate some staging
    _ = search_calendar_events(
        datetime_range_lowerbound="2024-12-08T00:00+00:00",
        datetime_range_upperbound="2025-01-07T23:30+00:00",
    )
    assert (
        get_current_context()
        .get_database(namespace=DatabaseNamespace.CALENDAR_EVENTS_RECURRENCE_STAGING)
        .select(pl.len())["len"][0]
        == 5
    )
    # Edit parent recurrence
    modify_calendar_event(
        calendar_event_id=unitttest_event_id,
        recurrence_interval=2,
        recurrence_frequency=CalendarRecurrenceFrequencyType.DAILY,
        start_datetime="2024-12-09T22:00-08:00",
    )
    # Should have 2 entries in CalendarEvents, and empty staging
    assert (
        get_current_context()
        .get_database(namespace=DatabaseNamespace.CALENDAR_EVENTS)
        .select(pl.len())["len"][0]
        == 2
    )
    assert (
        get_current_context()
        .get_database(namespace=DatabaseNamespace.CALENDAR_EVENTS_RECURRENCE_STAGING)
        .is_empty()
    )
    assert get_current_context().get_database(
        namespace=DatabaseNamespace.CALENDAR_EVENTS
    ).filter(pl.col("calendar_event_id") == unitttest_event_id).to_dicts()[0] == {
        "calendar_id": work_calendar_id,
        "calendar_event_id": unitttest_event_id,
        "title": "Add unittest",
        "description": "Add some unittests to this domain.",
        "start_datetime": "2024-12-09T22:00-08:00",
        "end_datetime": "2024-12-09T23:00-08:00",
        "is_all_day": False,
        "recurrence_frequency": "DAILY",
        "recurrence_interval": 2,
        "recurrence_until_datetime": "2025-12-09T23:00-08:00",
        "recurrence_parent_id": None,
        "recurrence_exclude_datetime": None,
        "latitude": 37.334606,
        "longitude": -122.009102,
        "place_id": "ChIJ_Yjh6Za1j4AR8IgGUZGDDTs",
        "formatted_address": "One Apple Park Way, Cupertino, CA 95014, USA",
        "attendee_names": ["John A", "John B", "John C"],
        "attendee_emails": ["john_a@me.com", "john_b@me.com", "john_c@me.com"],
    }


def test_modify_calendar_event_recurrence_instance(
    unitttest_event_id: str, work_calendar_id: str
) -> None:
    # Search to populate some staging
    search_result = search_calendar_events(
        datetime_range_lowerbound="2024-12-08T00:00+00:00",
        datetime_range_upperbound="2025-01-07T23:30+00:00",
    )
    assert (
        get_current_context()
        .get_database(namespace=DatabaseNamespace.CALENDAR_EVENTS_RECURRENCE_STAGING)
        .select(pl.len())["len"][0]
        == 5
    )
    # Modify last instance
    calendar_event_id = search_result[-1]["calendar_event_id"]
    modify_calendar_event(
        calendar_event_id=calendar_event_id, start_datetime="2025-01-06T22:00:00-08:00"
    )
    # Should have 4 entries in staging
    assert (
        get_current_context()
        .get_database(namespace=DatabaseNamespace.CALENDAR_EVENTS_RECURRENCE_STAGING)
        .select(pl.len())["len"][0]
        == 4
    )
    # Should have one new entry in events
    assert (
        get_current_context()
        .get_database(namespace=DatabaseNamespace.CALENDAR_EVENTS)
        .select(pl.len())["len"][0]
        == 3
    )
    # Should contain updated start time
    assert get_current_context().get_database(
        namespace=DatabaseNamespace.CALENDAR_EVENTS
    ).filter(pl.col("calendar_event_id") == calendar_event_id).to_dicts()[0] == {
        "calendar_id": work_calendar_id,
        "calendar_event_id": calendar_event_id,
        "title": "Add unittest",
        "description": "Add some unittests to this domain.",
        "start_datetime": "2025-01-06T22:00:00-08:00",
        "end_datetime": "2025-01-06T23:00:00-08:00",
        "is_all_day": False,
        "recurrence_frequency": None,
        "recurrence_interval": None,
        "recurrence_until_datetime": None,
        "recurrence_parent_id": unitttest_event_id,
        "recurrence_exclude_datetime": None,
        "latitude": 37.334606,
        "longitude": -122.009102,
        "place_id": "ChIJ_Yjh6Za1j4AR8IgGUZGDDTs",
        "formatted_address": "One Apple Park Way, Cupertino, CA 95014, USA",
        "attendee_names": ["John A", "John B", "John C"],
        "attendee_emails": ["john_a@me.com", "john_b@me.com", "john_c@me.com"],
    }
    # Start time should be excluded in parent
    assert get_current_context().get_database(
        namespace=DatabaseNamespace.CALENDAR_EVENTS
    ).filter(pl.col("calendar_event_id") == unitttest_event_id).to_dicts()[0][
        "recurrence_exclude_datetime"
    ] == ["2025-01-06T15:00:00-08:00"]
