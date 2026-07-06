# Copyright © 2026 Apple Inc.

import copy
import csv
import datetime
import functools
from datetime import date
from pathlib import Path
from typing import Any
from uuid import uuid4

import holidays
import polars as pl
from dateutil import relativedelta

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution_context import (
    CalendarRecurrenceFrequencyType,
    get_current_context,
)
from mmtoolsandbox.common.i18n import Locale
from mmtoolsandbox.common.validators import (
    validate_iso3166_alpha2_country_code,
    validate_iso_8601_date_time_str_with_timezone,
)


def load_country_code_mapping(
    mapping_file_path: str | Path,
) -> dict[str, tuple[int | None, str | None]]:
    """Loads a mapping from ISO 3166-1 alpha-2 to UNcode and ccTLD. Note that the two are not guaranteed to exist.

    Expected file structure to be a csv file with ISO2, UNcode and ccTLD

    Args:
        mapping_file_path:  path to the csv file storing mapping

    Returns:
        A dict from ISO2 code to a tuple of UN and ccTLD code
    """
    iso2_to_uncode_cctld: dict[str, tuple[int | None, str | None]] = {}
    with open(mapping_file_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            uncode = int(row["UNcode"]) if row["UNcode"] else None
            cctld = row["ccTLD"] if row["ccTLD"] else None
            iso2_to_uncode_cctld[row["ISO2"]] = (uncode, cctld)
    return iso2_to_uncode_cctld


@functools.cache
def load_iso2_to_uncode_cctld() -> dict[str, tuple[int | None, str | None]]:
    """Implement singleton factory for mapping using functools.cache

    Returns:

    """
    return load_country_code_mapping(Path(__file__).parent / "country_code_mapping.csv")


def iso3166_alpha2_to_cctld(country_code: str) -> str:
    """Convert ISO 3166 Alpha 2 country code to ccTLD.

    The only difference that "matters" between the two is gb -> uk. Include others when necessary:
    https://en.wikipedia.org/wiki/Country_code_top-level_domain#Relation_to_ISO_3166-1

    Args:
        country_code:   ISO 3166 Alpha 2 country code to convert.

    Returns:
        ccTLD
    """
    validate_iso3166_alpha2_country_code(country_code)
    cctld: str | None = load_iso2_to_uncode_cctld()[country_code.upper()][1]
    if cctld is None:
        raise KeyError(
            f"Country code {country_code} does not have a corresponding ccTLD"
        )
    return cctld


# UN M49 List of Countries under 419. See https://en.wikipedia.org/wiki/UN_M49
Latin_America_Caribbean_Codes: set[int] = {
    32,
    68,
    74,
    76,
    152,
    170,
    218,
    238,
    239,
    254,
    328,
    600,
    604,
    740,
    858,
    862,
    84,
    188,
    222,
    320,
    340,
    484,
    558,
    591,
    28,
    44,
    52,
    92,
    136,
    192,
    212,
    214,
    308,
    312,
    332,
    388,
    474,
    500,
    531,
    533,
    534,
    535,
    630,
    652,
    659,
    660,
    662,
    663,
    670,
    780,
    796,
    850,
}


def get_google_maps_language_code(language_code: str, country_code: str | None) -> str:
    """Google Maps uses something in between language code and language-COUNTRY to represent language.
    This function attempts to infer corresponding Google Maps language code based on ISO 639 Set 1 language code
    and ISO 3166-1 alpha-2 country code.

    Args:
        language_code:  ISO 639 Set 1 language code
        country_code:   ISO 3166-1 alpha-2 country code, if provided


    Returns:
        Google Maps language code. See https://developers.google.com/maps/faq#languagesupport
    """
    language_code = language_code.lower()
    if country_code is None:
        return language_code
    country_code = country_code.upper()
    # Special handling for latin america. Groups countries with https://en.wikipedia.org/wiki/UN_M49
    if load_iso2_to_uncode_cctld()[country_code][0] in Latin_America_Caribbean_Codes:
        country_code = "419"
    locale_code = f"{language_code}-{country_code}"
    if locale_code in [
        "zh-CN",
        "zh-HK",
        "zh-TW",
        "en-AU",
        "en-GB",
        "fr-CA",
        "pt-BR",
        "pt-PT",
        "es-419",
    ]:
        return locale_code
    return language_code


def utc_offset_minutes_to_iso_8601_offset(offset: int) -> str:
    """Convert a utc offset (in minutes) to ISO 8601

    Args:
        offset:     UTC offset, ranges from (-720, 720), e.g. -480

    Returns:
        ISO 8601 UTC offset sting. E.g. -08:00
    """
    return (
        "+" if offset >= 0 else "-"
    ) + f"{abs(offset) // 60:02}:{abs(offset) % 60:02}"


def add_iso_8601_to_pytree(pytree: Any, iso_utc_offset: str) -> None:
    """Adds ISO 8601 date time string to leaf nodes of a Pytree (borrowing the concept from Jax)

    Recursively unpacks list, tuple and dict to look for the following dict pattern:
        {
            "date": "2024-12-09",
            "time": "1500"
        }
    Once found, adds an ISO 8601 date time string to the dict using iso_utc_offset, like so:
        {
            "date": "2024-12-09",
            "time": "1500",
            "iso_8601_date_time": "2024-12-09T15:00-08:00"
        }

    Args:
        pytree:         Pytree object to add to
        iso_utc_offset: UTC offset to base date time on

    Returns:
        None. Modification done in place.
    """
    if isinstance(pytree, (tuple, list)):
        for subtree in pytree:
            add_iso_8601_to_pytree(subtree, iso_utc_offset)
    elif isinstance(pytree, dict):
        if "date" in pytree and "time" in pytree:
            pytree["iso_8601_date_time"] = (
                f"{pytree['date']}T{pytree['time'][:2]}:{pytree['time'][2:]}{iso_utc_offset}"
            )
        for subtree in pytree.values():
            add_iso_8601_to_pytree(subtree, iso_utc_offset)


def iso_8601_offset_to_iso_8601_utc(iso_date_time_str: str) -> str:
    """Convert an ISO 8601 datetime string with UTC offset to UTC timezone

    Args:
        iso_date_time_str:   Datetime string to be converted.

    Returns:
        Converted datetime string in utc timezone.
    """
    validate_iso_8601_date_time_str_with_timezone(iso_date_time_str)
    return (
        datetime.datetime.fromisoformat(iso_date_time_str)
        .astimezone(datetime.timezone.utc)
        .isoformat()
    )


FREQUENCY_TO_RELATIVE_DELTA_KEY: dict[CalendarRecurrenceFrequencyType, str] = {
    CalendarRecurrenceFrequencyType.YEARLY: "years",
    CalendarRecurrenceFrequencyType.MONTHLY: "months",
    CalendarRecurrenceFrequencyType.WEEKLY: "weeks",
    CalendarRecurrenceFrequencyType.DAILY: "days",
}


def prepare_calendar_event_fields(
    calendar_id: str,
    start_datetime: str,
    end_datetime: str,
    is_all_day: bool = False,
    recurrence_frequency: CalendarRecurrenceFrequencyType | None = None,
    recurrence_interval: int | None = None,
    recurrence_until_datetime: str | None = None,
    attendee_names: list[str] | None = None,
    attendee_emails: list[str] | None = None,
    **kwargs: str | float | None,
) -> dict[str, Any]:
    """Perform necessary modification and validations for create_calendar_event.

    Useful for both create_calendar_event and modify_calendar_event.

    Args:
        calendar_id:                UUID of calendar this event belongs to.
        start_datetime:             ISO 8601 datetime string with UTC timezone offset representing event start time,
                                    inclusive. E.g. 2024-12-09T15:00-08:00.
        end_datetime:               ISO 8601 datetime string with UTC timezone offset representing event end time,
                                    exclusive. E.g. 2024-12-09T16:00-08:00.
        is_all_day:                 Boolean value indicating if this is an all day event. If true, hours and minute
                                    info in start_datetime and end_datetime will be discarded.
        recurrence_frequency:       Unit of recurrence frequency.
        recurrence_interval:        Combined with unit, forms a recurrence rule.
                                    E.g. interval of 2 and frequency of WEEKLY means biweekly.
        recurrence_until_datetime:  ISO 8601 representing end of recurrence. E.g. 2024-12-31T15:00-08:00.
        attendee_names:             A list of attendee names. Forms a 1 to 1 mapping with attendee_email.
        attendee_emails:            A list of attendee emails. Forms a 1 to 1 mapping with attendee_names.
                                    Use null element for attendee without an email.

    Returns:
        An updated dictionary with necessary transformations. kwargs are passed through directly.
    """
    current_context = get_current_context()
    # Validate calendar_id
    if (
        current_context.get_database(namespace=DatabaseNamespace.CALENDARS)
        .filter(pl.col("calendar_id") == calendar_id)
        .is_empty()
    ):
        raise ValueError(f"{calendar_id=} doesn't exist in Calendar Database.")
    # Validate datetime strings
    for datetime_str in (start_datetime, end_datetime) + (
        (recurrence_until_datetime,) if recurrence_until_datetime is not None else ()
    ):
        validate_iso_8601_date_time_str_with_timezone(datetime_str)
    start_datetime_object = datetime.datetime.fromisoformat(start_datetime)
    end_datetime_object = datetime.datetime.fromisoformat(end_datetime)

    if end_datetime_object <= start_datetime_object:
        raise ValueError(
            f"end_datetime should be later than start_datetime, "
            f"instead found {start_datetime=}, {end_datetime=}."
        )
    # Validate frequency and interval
    if (recurrence_frequency is None) ^ (recurrence_interval is None):
        raise ValueError(
            "recurrence_frequency and recurrence_interval must be both provided or both omitted."
        )
    # Validate attendee info
    if (attendee_names is None) ^ (attendee_emails is None):
        raise ValueError(
            "attendee_names and attendee_email must be both provided or both omitted."
        )
    if (
        attendee_names is not None
        and attendee_emails is not None
        and len(attendee_names) != len(attendee_emails)
    ):
        raise ValueError(
            "attendee_names and attendee_emails must have the same length."
        )
    # Process start_datetime and end_datetime based on is_all_day
    if is_all_day:
        # start_datetime should become the first second of the day (since it's inclusive).
        start_datetime = datetime.datetime(
            year=start_datetime_object.year,
            month=start_datetime_object.month,
            day=start_datetime_object.day,
            tzinfo=start_datetime_object.tzinfo,
        ).isoformat()
        # end_datetime should become the first second of the next day, except when it is already the first second of
        # a day (since it's exclusive).
        end_datetime_object = (
            end_datetime_object
            + datetime.timedelta(days=1)
            - datetime.timedelta(microseconds=1)
        )
        end_datetime = datetime.datetime(
            year=end_datetime_object.year,
            month=end_datetime_object.month,
            day=end_datetime_object.day,
            tzinfo=end_datetime_object.tzinfo,
        ).isoformat()
    return {
        "calendar_id": calendar_id,
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
        "is_all_day": is_all_day,
        "recurrence_frequency": recurrence_frequency,
        "recurrence_interval": recurrence_interval,
        "recurrence_until_datetime": recurrence_until_datetime,
        "attendee_names": attendee_names,
        "attendee_emails": attendee_emails,
        **kwargs,
    }


def create_recurring_calendar_event_instances_within_time_range(
    recurring_event_parent: dict[str, Any],
    time_range_lowerbound: str,
    time_range_upperbound: str,
) -> list[dict[str, Any]]:
    """Given a recurring event, create its recurring instances within a time range.

    Any overlap between event start end and range lower upper bound is considered a hit, and an instance
    will be created.

    This is particularly useful for staging recurring instances on search_calendar_events tool call,
    so that they can be modified / deleted later.

    Note that the parent itself is also staged.

    Args:
        recurring_event_parent: A dict with the same schema as CALENDAR_EVENTS, containing the parent recurring event.
        time_range_lowerbound:  ISO 8601. Lowerbound of the time range.
        time_range_upperbound:  ISO 8601. Upperbound of the time range.

    Returns:
        A List of recurring event instances within time range, if they exist. Otherwise, an empty list.
    """
    validate_iso_8601_date_time_str_with_timezone(time_range_lowerbound)
    validate_iso_8601_date_time_str_with_timezone(time_range_upperbound)
    if (
        recurring_event_parent["recurrence_frequency"] is None
        or recurring_event_parent["recurrence_interval"] is None
    ):
        raise ValueError(f"{recurring_event_parent=} is not a recurring event.")
    recurring_event_instances: list[dict[str, Any]] = []
    # Initial start and end time
    instance_start_datetime_obj = datetime.datetime.fromisoformat(
        recurring_event_parent["start_datetime"]
    )
    instance_end_datetime_obj = datetime.datetime.fromisoformat(
        recurring_event_parent["end_datetime"]
    )
    # Time range
    time_range_lowerbound_obj = datetime.datetime.fromisoformat(time_range_lowerbound)
    time_range_upperbound_obj = datetime.datetime.fromisoformat(time_range_upperbound)
    recurrence_until_datetime_obj = (
        datetime.datetime.fromisoformat(
            recurring_event_parent["recurrence_until_datetime"]
        )
        if recurring_event_parent["recurrence_until_datetime"] is not None
        else time_range_upperbound_obj
    )
    # Figure out the recurrence offset
    recurrence_frequency = recurring_event_parent["recurrence_frequency"]
    delta = relativedelta.relativedelta(
        **{
            FREQUENCY_TO_RELATIVE_DELTA_KEY[
                recurrence_frequency
            ]: recurring_event_parent["recurrence_interval"]
        }
    )
    # Figure out excluded start time
    excluded_start_time: set[datetime.datetime] = set()
    if recurring_event_parent["recurrence_exclude_datetime"] is not None:
        excluded_start_time = {
            datetime.datetime.fromisoformat(x)
            for x in recurring_event_parent["recurrence_exclude_datetime"]
        }
    # Iterate through all recurrences to see if there's any range overlap. Skip any exclusion.
    while instance_start_datetime_obj < min(
        recurrence_until_datetime_obj, time_range_upperbound_obj
    ):
        if (
            instance_start_datetime_obj not in excluded_start_time
            and instance_end_datetime_obj > time_range_lowerbound_obj
        ):
            instance = copy.deepcopy(recurring_event_parent)
            # Remove all recurrence rules
            for key in [
                "recurrence_frequency",
                "recurrence_interval",
                "recurrence_until_datetime",
                "recurrence_exclude_datetime",
            ]:
                instance[key] = None
            # Add reference to parent
            instance["recurrence_parent_id"] = recurring_event_parent[
                "calendar_event_id"
            ]
            # Assign new uuid
            instance["calendar_event_id"] = str(uuid4())
            # Update start and end time
            instance["start_datetime"] = instance_start_datetime_obj.isoformat()
            instance["end_datetime"] = instance_end_datetime_obj.isoformat()
            recurring_event_instances.append(instance)
        instance_start_datetime_obj += delta
        instance_end_datetime_obj += delta

    return recurring_event_instances


def remove_key_from_pytree(pytree: Any, key: str) -> None:
    """In-place remove key from all nodes of a Pytree (borrowing the concept from Jax) when applicable

    Recursively unpacks list, tuple and dict. For an example input and key "polyline"
        {
            "distance": {
                "text": "56.0 mi",
                "value": 90113
            },
            "duration": {
                "text": "51 mins",
                "value": 3052
            },
            "end_location": {
                "lat": 38.0247511,
                "lng": -122.1098884
            },
            "dummy": {
                "polyline": {
                    "points": "sdfsdfsd"
                }
            },
            "html_instructions": "Continue onto <b>I-680 N</b><div style=\"font-size:0.9em\">Toll road</div>",
            "polyline": {
                "points": "sdfsdfsd"
            }
        }
    Should be pruned to the following
        {
            "distance": {
                "text": "56.0 mi",
                "value": 90113
            },
            "duration": {
                "text": "51 mins",
                "value": 3052
            },
            "end_location": {
                "lat": 38.0247511,
                "lng": -122.1098884
            },
            "dummy": {
            },
            "html_instructions": "Continue onto <b>I-680 N</b><div style=\"font-size:0.9em\">Toll road</div>"
        }

    Args:
        pytree: Pytree to be processed.
        key:    Key to remove.
    """
    if isinstance(pytree, (tuple, list)):
        for subtree in pytree:
            remove_key_from_pytree(subtree, key)
    elif isinstance(pytree, dict):
        if key in pytree:
            del pytree[key]
        for subtree in pytree.values():
            remove_key_from_pytree(subtree, key)


def find_next_holiday_occurrence(
    holiday: str,
    locale: Locale,
    reference_date: date,
    *,
    max_years_to_check: int = 5,
) -> date:
    """Convert holiday name to a dictionary containing its next occurrence date components.

    This function finds the next occurrence of a specified holiday after the given reference
    date. It retrieves holidays for the locale's country and language for the year of the
    provided date, then looks up the date for the given holiday name. If the holiday has
    already occurred in the current year, it searches in subsequent years.

    Args:
        holiday: Name of the holiday in the locale's language.
        locale: Locale object containing country and language information.
        reference_date: Reference date to find the next occurrence of the holiday after this date.
        max_years_to_check: Maximum number of years to check (default: 5).

    Returns:
        Dictionary with `year`, `month`, and `day` keys representing the next occurrence
        of the holiday.

    Raises:
        ValueError: If the holiday cannot be found within the specified number of years.
    """

    start_year = reference_date.year

    for current_year in range(start_year, start_year + max_years_to_check):
        country_holidays = holidays.country_holidays(
            country=locale.value.country,
            years=current_year,
            language=locale.value.holidays_language_code,
        )

        # Mapping from holiday name to date
        holiday_to_dt = {v: k for k, v in country_holidays.items()}

        holiday_date = holiday_to_dt.get(holiday)

        # If holiday exists and is after or on the reference date, return it
        if holiday_date and (
            current_year > reference_date.year or reference_date <= holiday_date
        ):
            # Return holiday if it occurs after the reference date
            return date(
                year=holiday_date.year, month=holiday_date.month, day=holiday_date.day
            )

        # Update reference date to check the next year
        reference_date = date(year=current_year + 1, month=1, day=1)

    raise ValueError(
        f'Could not find holiday "{holiday}" within {max_years_to_check} years'
    )


def get_holiday_years_for_search(
    holiday_name: str,
    locale: Locale,
    reference_date: date,
) -> list[int | None]:
    """Get years for holiday search based on whether holiday is in current year.

    Args:
        holiday_name: Name of the holiday to search for
        locale: Locale for the holiday lookup
        reference_date: Reference date to determine current year context

    Returns:
        List containing [None, current_year] if holiday is in current year,
        otherwise [holiday_year]
    """
    holiday_date = find_next_holiday_occurrence(
        holiday_name,
        locale=locale,
        reference_date=reference_date,
    )

    if holiday_date.year == reference_date.year:
        return [None, holiday_date.year]

    return [holiday_date.year]


def date_to_dict(date: date) -> dict[str, int]:
    return {
        "year": date.year,
        "month": date.month,
        "day": date.day,
    }
