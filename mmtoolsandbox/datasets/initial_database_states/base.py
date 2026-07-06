# Copyright © 2026 Apple Inc.

from functools import partial
from typing import Any

from dateutil.relativedelta import MO, TH, TU, WE

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution_context import CalendarRecurrenceFrequencyType
from mmtoolsandbox.common.i18n import Locale, Localizer
from mmtoolsandbox.common.relative_time import SerializableRelativeDelta
from mmtoolsandbox.datasets.initial_database_states.registry import (
    register_initial_database_state,
)

# NOTE: The current design requires initial database entries to have a 1:1 mapping across locales.
# This is not necessarily true, for example: if the calendar database is populated with local holidays,
# different locales can have different number of calendar events which breaks the 1:1 mapping.
# Modifying the design to enable variable number of entries per locale is left for future work.


@register_initial_database_state(
    collections=(
        "base",
        "base_cellular_off",
        "base_wifi_off",
        "base_location_service_off",
        "base_low_battery_mode_on",
    ),
    namespace=DatabaseNamespace.CALENDAR_EVENTS,
)
def calendar_events_initial_database_state(
    localize: Localizer,
) -> list[dict[str, Any]]:
    """Initial databases entries for calendar events.

    Note that relative time are dilled as a hack to pass polars schema type check.

    Args:
        localize:

    Returns:

    """
    localize_with_scope = partial(
        localize, scope="initial_database_states/calendar_events"
    )

    return [
        {
            "sandbox_message_index": 0,
            "calendar_event_id": "0800e106-3fee-4288-aff8-c8bfc078d616",
            "calendar_id": "af204f1a-07a3-5c29-b452-9e9deedc36f3",
            "title": localize_with_scope("Gym"),
            "description": None,
            "start_datetime": localize_with_scope("2025-01-01T20:30:00-07:00"),
            "end_datetime": localize_with_scope("2025-01-01T21:00:00-07:00"),
            "is_all_day": False,
            "recurrence_frequency": CalendarRecurrenceFrequencyType.DAILY,
            "recurrence_interval": 2,
            "recurrence_until_datetime": None,
            "recurrence_parent_id": None,
            "recurrence_exclude_datetime": None,
            "latitude": None,
            "longitude": None,
            "place_id": None,
            "formatted_address": None,
            "attendee_emails": None,
            "attendee_names": None,
        },
        {
            "sandbox_message_index": 0,
            "calendar_event_id": "b2e52288-dcb8-4383-a2fe-2d65fb6d6503",
            "calendar_id": "af204f1a-07a3-5c29-b452-9e9deedc36f3",
            "title": localize_with_scope("House Cleaning"),
            "description": None,
            "start_datetime": localize_with_scope("2025-01-25T21:30:00-07:00"),
            "end_datetime": localize_with_scope("2025-01-25T22:00:00-07:00"),
            "is_all_day": False,
            "recurrence_frequency": CalendarRecurrenceFrequencyType.DAILY,
            "recurrence_interval": 2,
            "recurrence_until_datetime": None,
            "recurrence_parent_id": None,
            "recurrence_exclude_datetime": None,
            "latitude": None,
            "longitude": None,
            "place_id": None,
            "formatted_address": None,
            "attendee_emails": None,
            "attendee_names": None,
        },
        {
            "sandbox_message_index": 0,
            "calendar_event_id": "c042e7f5-d9ed-455e-add6-257a0bedc252",
            "calendar_id": "af204f1a-07a3-5c29-b452-9e9deedc36f3",
            "title": localize_with_scope("Reservation at Restaurant Gish for 2"),
            "description": localize_with_scope(
                "https://maps.google.com/maps?saddr%3DCurrent%2520Location%26daddr%3D1998%2520"
                "Homestead%2520Rd%250ASanta%2520Clara,%2520CA%252095050-6963"
            ),
            "start_datetime": SerializableRelativeDelta(
                days=1, hour=20, minute=0, second=0
            ).to_json(),  # Tomorrow
            "end_datetime": SerializableRelativeDelta(
                days=1, hour=22, minute=0, second=0
            ).to_json(),
            "is_all_day": False,
            "recurrence_frequency": None,
            "recurrence_interval": None,
            "recurrence_until_datetime": None,
            "recurrence_parent_id": None,
            "recurrence_exclude_datetime": None,
            "latitude": localize_with_scope(37.343352),
            "longitude": localize_with_scope(-121.959029),
            "place_id": localize_with_scope("ChIJbZQsqpDLj4ARHS2lJFcGpVw"),
            "formatted_address": localize_with_scope(
                "1998 Homestead Rd #113, Santa Clara, CA 95050, USA"
            ),
            "attendee_emails": None,
            "attendee_names": None,
        },
        {
            "sandbox_message_index": 0,
            "calendar_event_id": "a9528b91-4083-4633-8db8-c83dc4c0576a",
            "calendar_id": "af204f1a-07a3-5c29-b452-9e9deedc36f3",
            "title": localize_with_scope("DMV"),
            "description": localize_with_scope(
                "License Renewal. Remember to bring documents."
            ),
            "start_datetime": SerializableRelativeDelta(
                days=3, weekday=WE(1), hour=9, minute=0, second=0
            ).to_json(),  # Wed of next week
            "end_datetime": SerializableRelativeDelta(
                days=3, weekday=WE(1), hour=11, minute=0, second=0
            ).to_json(),
            "is_all_day": False,
            "recurrence_frequency": None,
            "recurrence_interval": None,
            "recurrence_until_datetime": None,
            "recurrence_parent_id": None,
            "recurrence_exclude_datetime": None,
            "latitude": localize_with_scope(37.3502718),
            "longitude": localize_with_scope(-121.9939627),
            "place_id": localize_with_scope("ChIJc_Q56_W1j4ARxZxJ-izgArI"),
            "formatted_address": localize_with_scope(
                "3665 Flora Vista Ave, Santa Clara, CA 95051, USA"
            ),
            "attendee_emails": None,
            "attendee_names": None,
        },
        {
            "sandbox_message_index": 0,
            "calendar_event_id": "a433e311-5c21-4cbf-b541-e5b349209e3b",
            "calendar_id": "87dcf025-a8a0-513e-ad16-d23505a9215c",
            "title": localize_with_scope("1:1 Jake | John"),
            "description": localize_with_scope("One off 1 on 1 discussion."),
            "start_datetime": SerializableRelativeDelta(
                days=-1, hour=10, minute=0, second=0
            ).to_json(),  # Yesterday
            "end_datetime": SerializableRelativeDelta(
                days=-1, hour=11, minute=0, second=0
            ).to_json(),
            "is_all_day": False,
            "recurrence_frequency": None,
            "recurrence_interval": None,
            "recurrence_until_datetime": None,
            "recurrence_parent_id": None,
            "recurrence_exclude_datetime": None,
            "latitude": None,
            "longitude": None,
            "place_id": None,
            "formatted_address": None,
            "attendee_emails": [
                localize_with_scope("jake_a@me.com"),
                localize_with_scope("john_p@me.com"),
            ],
            "attendee_names": [
                localize_with_scope("Jake Adams"),
                localize_with_scope("John Petrucci"),
            ],
        },
        {
            "sandbox_message_index": 0,
            "calendar_event_id": "0d62885f-f1b6-4b40-ba77-bc31d3a30655",
            "calendar_id": "87dcf025-a8a0-513e-ad16-d23505a9215c",
            "title": localize_with_scope("Project Planning"),
            "description": localize_with_scope(
                "Initial Meeting for future project planning. "
                "Please bring your project proposals for discussions."
            ),
            "start_datetime": SerializableRelativeDelta(
                days=-1, hour=13, minute=0, second=0
            ).to_json(),  # Yesterday
            "end_datetime": SerializableRelativeDelta(
                days=-1, hour=17, minute=0, second=0
            ).to_json(),
            "is_all_day": False,
            "recurrence_frequency": None,
            "recurrence_interval": None,
            "recurrence_until_datetime": None,
            "recurrence_parent_id": None,
            "recurrence_exclude_datetime": None,
            "latitude": None,
            "longitude": None,
            "place_id": None,
            "formatted_address": None,
            "attendee_emails": [
                localize_with_scope("jake_a@me.com"),
                localize_with_scope("john_p@me.com"),
                localize_with_scope("jill_m@me.com"),
                localize_with_scope("pesto@meow.com"),
                localize_with_scope("joooosh@me.com"),
                localize_with_scope("twinkle@me.com"),
            ],
            "attendee_names": [
                localize_with_scope("Jake Adams"),
                localize_with_scope("John Petrucci"),
                localize_with_scope("Jill Mulligan"),
                localize_with_scope("Pesto"),
                localize_with_scope("Josh F"),
                localize_with_scope("Jeff G"),
            ],
        },
        {
            "sandbox_message_index": 0,
            "calendar_event_id": "ca0cd818-85ef-4bed-9f55-99621348480b",
            "calendar_id": "87dcf025-a8a0-513e-ad16-d23505a9215c",
            "title": localize_with_scope("Team Event"),
            "description": localize_with_scope("Team Event at great America."),
            "start_datetime": SerializableRelativeDelta(
                hour=9, minute=0, second=0
            ).to_json(),  # Today
            "end_datetime": SerializableRelativeDelta(
                hour=17, minute=0, second=0
            ).to_json(),
            "is_all_day": False,
            "recurrence_frequency": None,
            "recurrence_interval": None,
            "recurrence_until_datetime": None,
            "recurrence_parent_id": None,
            "recurrence_exclude_datetime": None,
            "latitude": localize_with_scope(37.4005684),
            "longitude": localize_with_scope(-121.9773816),
            "place_id": localize_with_scope("ChIJR81ua8_Jj4ARHSaAk12y10s"),
            "formatted_address": localize_with_scope(
                "4911 Great America Pkwy, Santa Clara, CA 95054, United States"
            ),
            "attendee_emails": [
                localize_with_scope("jake_a@me.com"),
                localize_with_scope("john_p@me.com"),
                localize_with_scope("jill_m@me.com"),
                localize_with_scope("pesto@meow.com"),
                localize_with_scope("joooosh@me.com"),
                localize_with_scope("twinkle@me.com"),
            ],
            "attendee_names": [
                localize_with_scope("Jake Adams"),
                localize_with_scope("John Petrucci"),
                localize_with_scope("Jill Mulligan"),
                localize_with_scope("Pesto"),
                localize_with_scope("Josh F"),
                localize_with_scope("Jeff G"),
            ],
        },
        {
            "sandbox_message_index": 0,
            "calendar_event_id": "894f472d-8c84-4100-827d-867ba92b96cb",
            "calendar_id": "87dcf025-a8a0-513e-ad16-d23505a9215c",
            "title": localize_with_scope("1:1 Jake | Jill"),
            "description": None,
            "start_datetime": SerializableRelativeDelta(
                days=1, hour=11, minute=30, second=0
            ).to_json(),  # Tomorrow
            "end_datetime": SerializableRelativeDelta(
                days=1, hour=12, minute=0, second=0
            ).to_json(),
            "is_all_day": False,
            "recurrence_frequency": None,
            "recurrence_interval": None,
            "recurrence_until_datetime": None,
            "recurrence_parent_id": None,
            "recurrence_exclude_datetime": None,
            "latitude": None,
            "longitude": None,
            "place_id": None,
            "formatted_address": None,
            "attendee_emails": [
                localize_with_scope("jake_a@me.com"),
                localize_with_scope("jill_m@me.com"),
            ],
            "attendee_names": [
                localize_with_scope("Jake Adams"),
                localize_with_scope("Jill Mulligan"),
            ],
        },
        {
            "sandbox_message_index": 0,
            "calendar_event_id": "7d88e97d-f3cd-4595-94e8-003ba236855a",
            "calendar_id": "87dcf025-a8a0-513e-ad16-d23505a9215c",
            "title": localize_with_scope("1:1 Jake | Pesto"),
            "description": None,
            "start_datetime": SerializableRelativeDelta(
                days=1, hour=12, minute=0, second=0
            ).to_json(),  # Tomorrow
            "end_datetime": SerializableRelativeDelta(
                days=1, hour=12, minute=30, second=0
            ).to_json(),
            "is_all_day": False,
            "recurrence_frequency": None,
            "recurrence_interval": None,
            "recurrence_until_datetime": None,
            "recurrence_parent_id": None,
            "recurrence_exclude_datetime": None,
            "latitude": None,
            "longitude": None,
            "place_id": None,
            "formatted_address": None,
            "attendee_emails": [
                localize_with_scope("jake_a@me.com"),
                localize_with_scope("pesto@meow.com"),
            ],
            "attendee_names": [
                localize_with_scope("Jake Adams"),
                localize_with_scope("Pesto"),
            ],
        },
        {
            "sandbox_message_index": 0,
            "calendar_event_id": "ec63a2f8-83f0-480b-aae8-9cbe0cc10194",
            "calendar_id": "87dcf025-a8a0-513e-ad16-d23505a9215c",
            "title": localize_with_scope("OoO"),
            "description": None,
            "start_datetime": SerializableRelativeDelta(
                days=1, weekday=MO(1), hour=0, minute=0, second=0
            ).to_json(),  # Monday of next week
            "end_datetime": SerializableRelativeDelta(
                days=2, weekday=TU(1), hour=0, minute=0, second=0
            ).to_json(),  # Tuesday of next week
            "is_all_day": True,
            "recurrence_frequency": None,
            "recurrence_interval": None,
            "recurrence_until_datetime": None,
            "recurrence_parent_id": None,
            "recurrence_exclude_datetime": None,
            "latitude": None,
            "longitude": None,
            "place_id": None,
            "formatted_address": None,
            "attendee_emails": None,
            "attendee_names": None,
        },
        {
            "sandbox_message_index": 0,
            "calendar_event_id": "946f59b0-3116-40b6-8100-5fc7d691e366",
            "calendar_id": "87dcf025-a8a0-513e-ad16-d23505a9215c",
            "title": localize_with_scope("Project Kickoff"),
            "description": None,
            "start_datetime": SerializableRelativeDelta(
                days=2, weekday=TU(1), hour=9, minute=0, second=0
            ).to_json(),  # Tuesday of next week
            "end_datetime": SerializableRelativeDelta(
                days=2, weekday=TU(1), hour=11, minute=0, second=0
            ).to_json(),  # Tuesday of next week
            "is_all_day": False,
            "recurrence_frequency": None,
            "recurrence_interval": None,
            "recurrence_until_datetime": None,
            "recurrence_parent_id": None,
            "recurrence_exclude_datetime": None,
            "latitude": None,
            "longitude": None,
            "place_id": None,
            "formatted_address": None,
            "attendee_emails": None,
            "attendee_names": None,
        },
        {
            "sandbox_message_index": 0,
            "calendar_event_id": "9d3ef9bc-841c-4ebe-8794-bbbbaf35055e",
            "calendar_id": "87dcf025-a8a0-513e-ad16-d23505a9215c",
            "title": localize_with_scope("Brainstorming"),
            "description": None,
            "start_datetime": SerializableRelativeDelta(
                days=2, weekday=TU(1), hour=12, minute=0, second=0
            ).to_json(),  # Tuesday of next week
            "end_datetime": SerializableRelativeDelta(
                days=2, weekday=TU(1), hour=16, minute=0, second=0
            ).to_json(),  # Tuesday of next week
            "is_all_day": False,
            "recurrence_frequency": None,
            "recurrence_interval": None,
            "recurrence_until_datetime": None,
            "recurrence_parent_id": None,
            "recurrence_exclude_datetime": None,
            "latitude": None,
            "longitude": None,
            "place_id": None,
            "formatted_address": None,
            "attendee_emails": [
                localize_with_scope("jake_a@me.com"),
                localize_with_scope("john_p@me.com"),
                localize_with_scope("jill_m@me.com"),
                localize_with_scope("pesto@meow.com"),
                localize_with_scope("joooosh@me.com"),
                localize_with_scope("twinkle@me.com"),
            ],
            "attendee_names": [
                localize_with_scope("Jake Adams"),
                localize_with_scope("John Petrucci"),
                localize_with_scope("Jill Mulligan"),
                localize_with_scope("Pesto"),
                localize_with_scope("Josh F"),
                localize_with_scope("Jeff G"),
            ],
        },
        {
            "sandbox_message_index": 0,
            "calendar_event_id": "966f91a6-9078-4c14-aaee-1706c3320009",
            "calendar_id": "87dcf025-a8a0-513e-ad16-d23505a9215c",
            "title": localize_with_scope("1:1 Jake | Josh"),
            "description": None,
            "start_datetime": SerializableRelativeDelta(
                days=3, weekday=WE(1), hour=9, minute=0, second=0
            ).to_json(),  # Wed of next week
            "end_datetime": SerializableRelativeDelta(
                days=3, weekday=WE(1), hour=9, minute=30, second=0
            ).to_json(),
            "is_all_day": False,
            "recurrence_frequency": None,
            "recurrence_interval": None,
            "recurrence_until_datetime": None,
            "recurrence_parent_id": None,
            "recurrence_exclude_datetime": None,
            "latitude": None,
            "longitude": None,
            "place_id": None,
            "formatted_address": None,
            "attendee_emails": [
                localize_with_scope("jake_a@me.com"),
                localize_with_scope("joooosh@me.com"),
            ],
            "attendee_names": [
                localize_with_scope("Jake Adams"),
                localize_with_scope("Josh F"),
            ],
        },
        {
            "sandbox_message_index": 0,
            "calendar_event_id": "0cd0f1c8-7193-4f4a-942c-157730e412f6",
            "calendar_id": "87dcf025-a8a0-513e-ad16-d23505a9215c",
            "title": localize_with_scope("1:1 Jake | Jeff"),
            "description": None,
            "start_datetime": SerializableRelativeDelta(
                days=3, weekday=WE(1), hour=12, minute=30, second=0
            ).to_json(),  # Wed of next week
            "end_datetime": SerializableRelativeDelta(
                days=3, weekday=WE(1), hour=13, minute=0, second=0
            ).to_json(),
            "is_all_day": False,
            "recurrence_frequency": None,
            "recurrence_interval": None,
            "recurrence_until_datetime": None,
            "recurrence_parent_id": None,
            "recurrence_exclude_datetime": None,
            "latitude": None,
            "longitude": None,
            "place_id": None,
            "formatted_address": None,
            "attendee_emails": [
                localize_with_scope("jake_a@me.com"),
                localize_with_scope("twinkle@me.com"),
            ],
            "attendee_names": [
                localize_with_scope("Jake Adams"),
                localize_with_scope("Jeff G"),
            ],
        },
        {
            "sandbox_message_index": 0,
            "calendar_event_id": "769b87bd-0342-4a07-b9d4-101ee3c8c257",
            "calendar_id": "87dcf025-a8a0-513e-ad16-d23505a9215c",
            "title": localize_with_scope("casual chat"),
            "description": None,
            "start_datetime": SerializableRelativeDelta(
                days=4, weekday=TH(1), hour=14, minute=0, second=0
            ).to_json(),  # Thur of next week
            "end_datetime": SerializableRelativeDelta(
                days=4, weekday=TH(1), hour=15, minute=0, second=0
            ).to_json(),
            "is_all_day": False,
            "recurrence_frequency": None,
            "recurrence_interval": None,
            "recurrence_until_datetime": None,
            "recurrence_parent_id": None,
            "recurrence_exclude_datetime": None,
            "latitude": None,
            "longitude": None,
            "place_id": None,
            "formatted_address": None,
            "attendee_emails": None,
            "attendee_names": None,
        },
    ]


@register_initial_database_state(
    collections=(
        "base",
        "base_cellular_off",
        "base_wifi_off",
        "base_location_service_off",
        "base_low_battery_mode_on",
    ),
    namespace=DatabaseNamespace.CALENDARS,
)
def calendars_initial_database_state(
    localize: Localizer,
) -> list[dict[str, Any]]:
    localize_with_scope = partial(localize, scope="initial_database_states/calendars")

    return [
        {
            "sandbox_message_index": 0,
            "calendar_id": "af204f1a-07a3-5c29-b452-9e9deedc36f3",
            "title": localize_with_scope("Personal"),
        },
        {
            "sandbox_message_index": 0,
            "calendar_id": "87dcf025-a8a0-513e-ad16-d23505a9215c",
            "title": localize_with_scope("Work"),
        },
    ]


@register_initial_database_state(
    collections="base",
    namespace=DatabaseNamespace.SETTING,
)
def setting_initial_database_state(localize: Localizer) -> list[dict[str, Any]]:
    localize_with_scope = partial(localize, scope="initial_database_states/setting")

    return [
        {
            "sandbox_message_index": 0,
            "device_id": "04640c30-ec89-5a3c-9fb4-4af3101c8e04",
            "cellular": True,
            "wifi": True,
            "location_service": True,
            "low_battery_mode": False,
            "locale": localize_with_scope(Locale["en_US"].name),
            "latitude": localize_with_scope(37.2439),
            "longitude": localize_with_scope(-122.0308),
            "place_id": localize_with_scope("ChIJ4YSL3_tKjoARpGsLdUkxppM"),
            "formatted_address": localize_with_scope(
                "15400 Montalvo Rd, Saratoga, CA 95070, USA",
            ),
            "utc_offset_seconds": localize_with_scope(-25200),
        }
    ]


@register_initial_database_state(
    collections="base_cellular_off",
    namespace=DatabaseNamespace.SETTING,
)
def setting_initial_database_state_cellular_off(
    localize: Localizer,
) -> list[dict[str, Any]]:
    state = setting_initial_database_state(localize)
    state[0]["cellular"] = False
    return state


@register_initial_database_state(
    collections="base_wifi_off",
    namespace=DatabaseNamespace.SETTING,
)
def setting_initial_database_state_wifi_off(
    localize: Localizer,
) -> list[dict[str, Any]]:
    state = setting_initial_database_state(localize)
    state[0]["wifi"] = False
    return state


@register_initial_database_state(
    collections="base_location_service_off",
    namespace=DatabaseNamespace.SETTING,
)
def setting_initial_database_state_location_service_off(
    localize: Localizer,
) -> list[dict[str, Any]]:
    state = setting_initial_database_state(localize)
    state[0]["location_service"] = False
    return state


@register_initial_database_state(
    collections="base_low_battery_mode_on",
    namespace=DatabaseNamespace.SETTING,
)
def setting_initial_database_state_low_battery_mode_on(
    localize: Localizer,
) -> list[dict[str, Any]]:
    state = setting_initial_database_state(localize)
    state[0]["low_battery_mode"] = True
    state[0]["cellular"] = False
    state[0]["wifi"] = False
    state[0]["location_service"] = False
    return state
