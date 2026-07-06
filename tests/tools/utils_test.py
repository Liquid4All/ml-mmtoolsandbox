# Copyright © 2026 Apple Inc.

"""Unit tests for mmtoolsandbox.tools.utils"""

import datetime
import uuid
from datetime import date
from typing import Any

import pytest

from mmtoolsandbox.common.execution_context import CalendarRecurrenceFrequencyType
from mmtoolsandbox.common.i18n import Locale
from mmtoolsandbox.tools.utils import (
    add_iso_8601_to_pytree,
    create_recurring_calendar_event_instances_within_time_range,
    find_next_holiday_occurrence,
    get_google_maps_language_code,
    get_holiday_years_for_search,
    iso3166_alpha2_to_cctld,
    iso_8601_offset_to_iso_8601_utc,
    load_iso2_to_uncode_cctld,
    remove_key_from_pytree,
    utc_offset_minutes_to_iso_8601_offset,
)


def test_iso3166_alpha2_to_cctld() -> None:
    assert iso3166_alpha2_to_cctld("US") == "us"
    assert iso3166_alpha2_to_cctld("GB") == "uk"


def test_get_google_maps_language_code() -> None:
    # No modification when country code not provided
    assert get_google_maps_language_code(language_code="en", country_code=None) == "en"
    # No modification when country code has no corresponding local
    assert get_google_maps_language_code(language_code="en", country_code="cn") == "en"
    # Locale code match
    assert (
        get_google_maps_language_code(language_code="ZH", country_code="cn") == "zh-CN"
    )
    # Latin America
    assert (
        get_google_maps_language_code(language_code="ES", country_code="ar") == "es-419"
    )


def test_utc_offset_seconds_to_iso_8601_offset() -> None:
    assert utc_offset_minutes_to_iso_8601_offset(-480) == "-08:00"
    assert utc_offset_minutes_to_iso_8601_offset(-450) == "-07:30"
    assert utc_offset_minutes_to_iso_8601_offset(0) == "+00:00"


def test_add_iso_8601_to_pytree() -> None:
    pytree = [
        {
            "periods": [
                {
                    "open": {
                        "date": "2024-12-09",
                        "time": "1500",
                    },
                    "close": {
                        "date": "2024-12-09",
                        "time": "2300",
                    },
                }
            ]
        }
    ]
    add_iso_8601_to_pytree(pytree, "-08:00")
    assert pytree == [
        {
            "periods": [
                {
                    "open": {
                        "date": "2024-12-09",
                        "time": "1500",
                        "iso_8601_date_time": "2024-12-09T15:00-08:00",
                    },
                    "close": {
                        "date": "2024-12-09",
                        "time": "2300",
                        "iso_8601_date_time": "2024-12-09T23:00-08:00",
                    },
                }
            ]
        }
    ]
    datetime.datetime.fromisoformat(
        pytree[0]["periods"][0]["open"]["iso_8601_date_time"]
    )
    datetime.datetime.fromisoformat(
        pytree[0]["periods"][0]["close"]["iso_8601_date_time"]
    )


def test_cache() -> None:
    mapping_a = load_iso2_to_uncode_cctld()
    mapping_b = load_iso2_to_uncode_cctld()
    assert id(mapping_a) == id(mapping_b)


def test_iso_8601_offset_to_iso_8601_utc() -> None:
    assert (
        iso_8601_offset_to_iso_8601_utc("2024-12-09T15:00-08:00")
        == "2024-12-09T23:00:00+00:00"
    )


@pytest.fixture
def work_calendar_id() -> str:
    return str(uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="Work"))


@pytest.fixture
def unitttest_event_id() -> str:
    return str(uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="Add unittest"))


@pytest.fixture
def recurring_event(
    work_calendar_id: str,
    unitttest_event_id: str,
) -> dict[str, Any]:
    """Test calendar events

    Returns:

    """
    return {
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
        "recurrence_exclude_datetime": ["2024-12-16T15:00:00-08:00"],
        "latitude": 37.334606,
        "longitude": -122.009102,
        "place_id": "ChIJ_Yjh6Za1j4AR8IgGUZGDDTs",
        "formatted_address": "One Apple Park Way, Cupertino, CA 95014, USA",
        "attendee_names": ["John A", "John B", "John C"],
        "attendee_emails": ["john_a@me.com", "john_b@me.com", "john_c@me.com"],
    }


def test_create_recurring_calendar_event_instances_within_time_range(
    work_calendar_id: str, unitttest_event_id: str, recurring_event: dict[str, Any]
) -> None:
    instances = create_recurring_calendar_event_instances_within_time_range(
        recurring_event_parent=recurring_event,
        time_range_lowerbound="2024-12-09T00:00+00:00",
        time_range_upperbound="2024-12-30T23:00+00:00",
    )
    for instance in instances:
        del instance["calendar_event_id"]
    assert instances == [
        {
            "calendar_id": work_calendar_id,
            "title": "Add unittest",
            "description": "Add some unittests to this domain.",
            "start_datetime": "2024-12-09T15:00:00-08:00",
            "end_datetime": "2024-12-09T23:00:00-08:00",
            "is_all_day": False,
            "recurrence_frequency": None,
            "recurrence_interval": None,
            "recurrence_until_datetime": None,
            "recurrence_exclude_datetime": None,
            "latitude": 37.334606,
            "longitude": -122.009102,
            "place_id": "ChIJ_Yjh6Za1j4AR8IgGUZGDDTs",
            "formatted_address": "One Apple Park Way, Cupertino, CA 95014, USA",
            "attendee_names": ["John A", "John B", "John C"],
            "attendee_emails": ["john_a@me.com", "john_b@me.com", "john_c@me.com"],
            "recurrence_parent_id": unitttest_event_id,
        },
        {
            "calendar_id": work_calendar_id,
            "title": "Add unittest",
            "description": "Add some unittests to this domain.",
            "start_datetime": "2024-12-23T15:00:00-08:00",
            "end_datetime": "2024-12-23T23:00:00-08:00",
            "is_all_day": False,
            "recurrence_frequency": None,
            "recurrence_interval": None,
            "recurrence_until_datetime": None,
            "recurrence_exclude_datetime": None,
            "latitude": 37.334606,
            "longitude": -122.009102,
            "place_id": "ChIJ_Yjh6Za1j4AR8IgGUZGDDTs",
            "formatted_address": "One Apple Park Way, Cupertino, CA 95014, USA",
            "attendee_names": ["John A", "John B", "John C"],
            "attendee_emails": ["john_a@me.com", "john_b@me.com", "john_c@me.com"],
            "recurrence_parent_id": unitttest_event_id,
        },
    ]
    # Make sure timezone based comparison is working correctly. The upperbound should just cross the boundary
    # of the next recurrence instance, resulting in a new event.
    assert (
        len(
            create_recurring_calendar_event_instances_within_time_range(
                recurring_event_parent=recurring_event,
                time_range_lowerbound="2024-12-09T00:00+00:00",
                time_range_upperbound="2024-12-30T23:30+00:00",
            )
        )
        == 3
    )


def test_remove_key_from_pytree() -> None:
    pytree = {
        "distance": {"text": "56.0 mi", "value": 90113},
        "duration": {"text": "51 mins", "value": 3052},
        "end_location": {"lat": 38.0247511, "lng": -122.1098884},
        "dummy": {"polyline": {"points": "sdfsdfsd"}},
        "html_instructions": 'Continue onto <b>I-680 N</b><div style="font-size:0.9em">Toll road</div>',
        "polyline": {"points": "sdfsdfsd"},
    }
    remove_key_from_pytree(pytree=pytree, key="polyline")
    assert pytree == {
        "distance": {"text": "56.0 mi", "value": 90113},
        "duration": {"text": "51 mins", "value": 3052},
        "end_location": {"lat": 38.0247511, "lng": -122.1098884},
        "dummy": {},
        "html_instructions": 'Continue onto <b>I-680 N</b><div style="font-size:0.9em">Toll road</div>',
    }


def test_christmas_day() -> None:
    assert find_next_holiday_occurrence(
        "Christmas Day", locale=Locale.en_US, reference_date=date(2025, 6, 6)
    ) == date(2025, 12, 25)

    assert find_next_holiday_occurrence(
        "Christmas Day", locale=Locale.en_US, reference_date=date(2025, 12, 31)
    ) == date(2026, 12, 25)

    assert find_next_holiday_occurrence(
        "Christmas Day", locale=Locale.en_US, reference_date=date(2026, 4, 8)
    ) == date(2026, 12, 25)


def test_get_holiday_years_for_search() -> None:
    assert set(
        get_holiday_years_for_search(
            "Christmas Day",
            locale=Locale.en_US,
            reference_date=date(2025, 1, 1),
        )
    ) == {None, 2025}

    assert set(
        get_holiday_years_for_search(
            "Christmas Day",
            locale=Locale.en_US,
            reference_date=date(2025, 12, 25),
        )
    ) == {None, 2025}

    assert get_holiday_years_for_search(
        "Christmas Day",
        locale=Locale.en_US,
        reference_date=date(2025, 12, 30),
    ) == [2026]
