# Copyright © 2026 Apple Inc.

"""
A collection of tools which simulates common functions used for calendar domain.
These are considered Principle tools which can be used to build up alternative tool box designs.

Key terminologies that might be confusing:
- Calendar: A Calendar containing multiple calendar events. E.g. Work / Home / Holiday
- Calendar Event: An event belonging to one and only one calendar.
"""

import datetime
import functools
from typing import (
    Any,
    Literal,
    cast,
)
from uuid import uuid4

import polars as pl
from polars.exceptions import DuplicateError, NoDataError
from typeguard import typechecked

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution_context import (
    CalendarRecurrenceFrequencyType,
    RoleType,
    get_current_context,
)
from mmtoolsandbox.common.utils import (
    NOT_GIVEN,
    NotGiven,
    embedding_filter_dataframe,
    embedding_list_contains_filter_dataframe,
    exact_match_filter_dataframe,
    exact_match_list_fully_contains_filter_dataframe,
    filter_dataframe,
    range_filter_dataframe,
    register_as_tool,
)
from mmtoolsandbox.common.validators import (
    validate_iso_8601_date_time_str_with_timezone,
)
from mmtoolsandbox.toolbox.names import ToolboxName
from mmtoolsandbox.tools.utils import (
    create_recurring_calendar_event_instances_within_time_range,
    prepare_calendar_event_fields,
)


# Principle creation tool for Calendar.
@register_as_tool(
    toolboxes={
        ToolboxName.FULL,
    },
    database_namespaces={DatabaseNamespace.CALENDARS},
    visible_to=(RoleType.AGENT,),
)
@typechecked
def create_calendar(title: str) -> str:
    """Create a new calendar.

    Calendar events can be added to this calendar using the uuid this function returns.

    Args:
        title:  Title summarizing the new calendar. E.g. Work.

    Returns:
        UUID of created calendar.
    """
    current_context = get_current_context()
    calendar_id = str(uuid4())
    current_context.add_to_database(
        namespace=DatabaseNamespace.CALENDARS,
        rows=[{"calendar_id": calendar_id, "title": title}],
    )
    return calendar_id


# Principle search tool for Calendar.
@register_as_tool(
    toolboxes={
        ToolboxName.FULL,
    },
    database_namespaces={DatabaseNamespace.CALENDARS},
    visible_to=(RoleType.AGENT,),
)
@typechecked
def search_calendars(
    calendar_id: str | NotGiven = NOT_GIVEN,
    title: str | NotGiven = NOT_GIVEN,
) -> list[
    dict[
        Literal["calendar_id", "title"],
        str,
    ]
]:
    """Search for calendars based on criteria.

    Search for calendars based on provided criteria. At least one search criteria should be provided.

    Each field has a search criteria of either
    1. Exact value matching
    2. Semantic similarity string matching with a predefined threshold
    Search results contains all calendar entries that matched all criteria

    Args:
        calendar_id:    String format unique identifier for the calendar. Will be exact matched.
        title:          Search query for title field. Will be matched based on semantic similarity

    Returns:
        List of matching calendars

    Raises:
        ValueError:     If all search criteria are not given
    """
    current_context = get_current_context()
    calendars_dataframe = filter_dataframe(
        dataframe=current_context.get_database(namespace=DatabaseNamespace.CALENDARS),
        filter_criteria=[
            ("calendar_id", calendar_id, exact_match_filter_dataframe),
            (
                "title",
                title,
                functools.partial(embedding_filter_dataframe, threshold=0.7),
            ),
        ],
    )
    return cast(
        list[
            dict[
                Literal["calendar_id", "title"],
                str,
            ]
        ],
        calendars_dataframe.to_dicts(),
    )


# Principle modification tool for Calendar.
@register_as_tool(
    toolboxes={
        ToolboxName.FULL,
    },
    database_namespaces={DatabaseNamespace.CALENDARS},
    visible_to=(RoleType.AGENT,),
)
@typechecked
def modify_calendar(calendar_id: str, title: str) -> None:
    """Modify an existing calendar.

    The only editable field for calendar is its title.

    Args:
        calendar_id:    UUID of calendar to be edited.
        title:          New title for the calendar. E.g. Home.

    Raises:
        polars.exceptions.NoDataError:  If no calendar matching found
    """
    current_context = get_current_context()
    # Check if entry exists
    target_entry = current_context.get_database(
        namespace=DatabaseNamespace.CALENDARS
    ).filter(pl.col("calendar_id") == calendar_id)
    if target_entry.is_empty():
        raise NoDataError(f"No calendar matching {calendar_id=} found")
    current_context.remove_from_database(
        namespace=DatabaseNamespace.CALENDARS,
        predicate=pl.col("calendar_id") == calendar_id,
    )
    current_context.add_to_database(
        namespace=DatabaseNamespace.CALENDARS,
        rows=[{"calendar_id": calendar_id, "title": title}],
    )


# Principle removal tool for Calendar.
@register_as_tool(
    toolboxes={
        ToolboxName.FULL,
    },
    database_namespaces={DatabaseNamespace.CALENDARS},
    visible_to=(RoleType.AGENT,),
)
@typechecked
def remove_calendar(calendar_id: str) -> None:
    """Remove a calendar based on its uuid.

    Note that removing a calendar also removes all associated calendar events.

    Args:
        calendar_id:    UUID of calendar to be removed.
    """
    current_context = get_current_context()
    # Delete calendar db
    current_context.remove_from_database(
        namespace=DatabaseNamespace.CALENDARS,
        predicate=pl.col("calendar_id") == calendar_id,
    )
    # Delete all calendar events belonging to calendar_id
    for calendar_event_id in (
        current_context.get_database(namespace=DatabaseNamespace.CALENDAR_EVENTS)
        .filter(pl.col("calendar_id") == calendar_id)["calendar_event_id"]
        .to_list()
    ):
        current_context.remove_from_database(
            namespace=DatabaseNamespace.CALENDAR_EVENTS,
            predicate=pl.col("calendar_event_id") == calendar_event_id,
        )


# Principle creation tool for Calendar Events.
@register_as_tool(
    toolboxes={
        ToolboxName.FULL,
    },
    database_namespaces={DatabaseNamespace.CALENDAR_EVENTS},
    visible_to=(RoleType.AGENT,),
)
@typechecked
def create_calendar_event(
    calendar_id: str,
    title: str,
    start_datetime: str,
    end_datetime: str,
    description: str | None = None,
    is_all_day: bool = False,
    recurrence_frequency: CalendarRecurrenceFrequencyType | None = None,
    recurrence_interval: int | None = None,
    recurrence_until_datetime: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    place_id: str | None = None,
    formatted_address: str | None = None,
    attendee_names: list[str] | None = None,
    attendee_emails: (
        list[str] | None
    ) = None,  # TODO: This should be list[str | None] | None. Fix json schema dump and fix this.
) -> str:
    """Create a calendar event based on provided arguments.

    Args:
        calendar_id:                UUID of calendar this event belongs to.
        title:                      Title or summary for the event.
        start_datetime:             ISO 8601 datetime string with UTC timezone offset representing event start time,
                                    inclusive. E.g. 2024-12-09T15:00-08:00.
        end_datetime:               ISO 8601 datetime string with UTC timezone offset representing event end time,
                                    exclusive. E.g. 2024-12-09T16:00-08:00.
        description:                Detailed description of the event.
        is_all_day:                 Boolean value indicating if this is an all day event. If true, hours and minute
                                    info in start_datetime and end_datetime will be discarded.
        recurrence_frequency:       Unit of recurrence frequency.
        recurrence_interval:        Combined with unit, forms a recurrence rule.
                                    E.g. interval of 2 and frequency of WEEKLY means biweekly.
        recurrence_until_datetime:  ISO 8601 representing end of recurrence. E.g. 2024-12-31T15:00-08:00.
        latitude:                   Latitude of the event location.
        longitude:                  Longitude of the event location.
        place_id:                   place_id of the event location.
        formatted_address:          formatted_address of the event location.
        attendee_names:             A list of attendee names. Forms a 1 to 1 mapping with attendee_email.
        attendee_emails:            A list of attendee emails. Forms a 1 to 1 mapping with attendee_names.
                                    Use null element for attendee without an email.

    Returns:
        UUID of created calendar event.

    Raises:
        ValueError: if attendee_names and attendee_email are not both present and matching in length
    """
    current_context = get_current_context()
    calendar_event_id = str(uuid4())
    current_context.add_to_database(
        namespace=DatabaseNamespace.CALENDAR_EVENTS,
        rows=[
            prepare_calendar_event_fields(
                calendar_event_id=calendar_event_id,
                calendar_id=calendar_id,
                title=title,
                description=description,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                is_all_day=is_all_day,
                recurrence_frequency=recurrence_frequency,
                recurrence_interval=recurrence_interval,
                recurrence_until_datetime=recurrence_until_datetime,
                latitude=latitude,
                longitude=longitude,
                place_id=place_id,
                formatted_address=formatted_address,
                attendee_names=attendee_names,
                attendee_emails=attendee_emails,
            )
        ],
    )
    return calendar_event_id


# Principle search tool for Calendar Events.
# TODO: All day events should be considered floating, and searched based on current local timezone
#   However at this point it's not clear whether "local timezone" should be considered as a device setting,
#   or determined by datetime_range_lowerbound/upperbound. Figure this out in the future.
@register_as_tool(
    toolboxes={
        ToolboxName.FULL,
    },
    database_namespaces={
        DatabaseNamespace.CALENDAR_EVENTS,
        DatabaseNamespace.CALENDAR_EVENTS_RECURRENCE_STAGING,
    },
    visible_to=(RoleType.AGENT,),
)
@typechecked
def search_calendar_events(
    calendar_id: str | NotGiven = NOT_GIVEN,
    calendar_event_id: str | NotGiven = NOT_GIVEN,
    title: str | NotGiven = NOT_GIVEN,
    description: str | NotGiven = NOT_GIVEN,
    datetime_range_lowerbound: str | NotGiven = NOT_GIVEN,
    datetime_range_upperbound: str | NotGiven = NOT_GIVEN,
    is_all_day: bool | NotGiven = NOT_GIVEN,
    recurrence_frequency: CalendarRecurrenceFrequencyType | NotGiven = NOT_GIVEN,
    recurrence_interval: int | NotGiven = NOT_GIVEN,
    recurrence_until_datetime: int | NotGiven = NOT_GIVEN,
    latitude: float | NotGiven = NOT_GIVEN,
    longitude: float | NotGiven = NOT_GIVEN,
    place_id: str | NotGiven = NOT_GIVEN,
    formatted_address: str | NotGiven = NOT_GIVEN,
    attendee_names: list[str] | NotGiven = NOT_GIVEN,
    attendee_emails: list[str] | NotGiven = NOT_GIVEN,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Search for calendar events based on a series of criteria.

    Search for calendar events based on provided criteria. At least one search criteria should be provided.

    Each field has a search criteria of either
    1. Exact value matching
    2. Semantic similarity string matching with a predefined threshold
    3. Range matching datetime with upperbound or lowerbound

    Caveats:
    1. Instances of recurring events will only appear in addition to the parent recurrence when
        datetime_range_lowerbound and datetime_range_upperbound are provided. This is because unbound
        recurrence event exists.
    2. When recurring events are being searched, parent recurrence (indicated by recurrence_parent_id) is also returned,
    in order to provide information around recurrence frequency and interval.

    Search results contains all calendar entries that matched all criteria
    Args:
        calendar_id:                String format unique identifier for calendar to search under. Will be exact matched.
        calendar_event_id:          String format unique identifier for calendar event. Will be exact matched
        title:                      Title summarizing calendar event. Will be semantically matched
        description:                Detailed description of calendar event. Will be semantically matched
        datetime_range_lowerbound:  Lowerbound of datetime range to search for calendar event.
                                    Any overlap between event [start_datetime, end_datetime) and
                                    [Lowerbound, Upperbound) is considered a match.
        datetime_range_upperbound:  Upperbound of datetime range to search for calendar event.
                                    Any overlap between event [start_datetime, end_datetime) and
                                    [Lowerbound, Upperbound) is considered a match.
        is_all_day:                 Boolean indicating all day events. Will be exact matched.
        recurrence_frequency:       Enum indicating recurrence frequency. Will be exact matched.
        recurrence_interval:        Enum indicating recurrence interval. Will be exact matched.
        recurrence_until_datetime:  ISO 8601 representing end of recurrence. Will be exact matched.
        latitude:                   Latitude of location. Will be matched with 0.01 degree tolerance.
        longitude:                  Longitude of location. Will be matched with 0.01 degree tolerance.
        place_id:                   place_id of location. Will be exact matched.
        formatted_address:          Formatted address of location. Will be semantically matched.
        attendee_names:             List of attendee names. Will be semantically matched. All provided names must each
                                    find at least a match.
        attendee_emails:            List of attendee emails. Will be exact matched. All provided emails must each find
                                    a match.

    Returns:
        A list of matching calendar events
    """

    def sort_events_by_increasing_start_datetime(
        events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return sorted(
            events, key=lambda x: datetime.datetime.fromisoformat(x["start_datetime"])
        )

    if (datetime_range_lowerbound is NOT_GIVEN) ^ (
        datetime_range_upperbound is NOT_GIVEN
    ):
        raise ValueError(
            "datetime_range_lowerbound and datetime_range_upperbound must be both provided or both omitted."
        )
    # First filter by all criteria except datetime_range
    current_context = get_current_context()
    filtered_calendar_event_dataframe = current_context.get_database(
        namespace=DatabaseNamespace.CALENDAR_EVENTS
    )
    if not all(
        x is NOT_GIVEN
        for x in (
            calendar_id,
            calendar_event_id,
            title,
            description,
            is_all_day,
            recurrence_frequency,
            recurrence_interval,
            recurrence_until_datetime,
            latitude,
            longitude,
            place_id,
            formatted_address,
            attendee_names,
            attendee_emails,
        )
    ):
        filtered_calendar_event_dataframe = filter_dataframe(
            dataframe=current_context.get_database(
                namespace=DatabaseNamespace.CALENDAR_EVENTS
            ),
            filter_criteria=[
                ("calendar_id", calendar_id, exact_match_filter_dataframe),
                ("calendar_event_id", calendar_event_id, exact_match_filter_dataframe),
                (
                    "title",
                    title,
                    functools.partial(embedding_filter_dataframe, threshold=0.7),
                ),
                (
                    "description",
                    description,
                    functools.partial(embedding_filter_dataframe, threshold=0.7),
                ),
                ("is_all_day", is_all_day, exact_match_filter_dataframe),
                (
                    "recurrence_frequency",
                    recurrence_frequency,
                    exact_match_filter_dataframe,
                ),
                (
                    "recurrence_interval",
                    recurrence_interval,
                    exact_match_filter_dataframe,
                ),
                (
                    "recurrence_until_datetime",
                    recurrence_until_datetime,
                    exact_match_filter_dataframe,
                ),
                (
                    "latitude",
                    latitude,
                    functools.partial(range_filter_dataframe, value_delta=0.01),
                ),
                (
                    "longitude",
                    longitude,
                    functools.partial(range_filter_dataframe, value_delta=0.01),
                ),
                ("place_id", place_id, exact_match_filter_dataframe),
                (
                    "formatted_address",
                    formatted_address,
                    functools.partial(embedding_filter_dataframe, threshold=0.7),
                ),
                (
                    "attendee_names",
                    attendee_names,
                    embedding_list_contains_filter_dataframe,
                ),
                (
                    "attendee_emails",
                    attendee_emails,
                    exact_match_list_fully_contains_filter_dataframe,
                ),
            ],
        )
    if (
        datetime_range_lowerbound is NOT_GIVEN
        and datetime_range_upperbound is NOT_GIVEN
    ):
        # Sort by increasing start time
        return sort_events_by_increasing_start_datetime(
            filtered_calendar_event_dataframe.to_dicts()
        )
    assert isinstance(datetime_range_lowerbound, str)  # Hint to mypy
    assert isinstance(datetime_range_upperbound, str)  # Hint to mypy
    validate_iso_8601_date_time_str_with_timezone(datetime_range_lowerbound)
    validate_iso_8601_date_time_str_with_timezone(datetime_range_upperbound)

    range_matched_instances: list[dict[str, Any]] = []

    # Find recurrence events and unroll them into staging db within datetime range
    for recurrence_event in filtered_calendar_event_dataframe.filter(
        pl.col("recurrence_frequency").is_not_null()
        & pl.col("recurrence_interval").is_not_null()
    ).to_dicts():
        event_staging_db = current_context.get_database(
            namespace=DatabaseNamespace.CALENDAR_EVENTS_RECURRENCE_STAGING
        )
        recurrence_instances: list[dict[str, Any]] = []
        # Make sure that instance doesn't exist in staging area already
        for instance in create_recurring_calendar_event_instances_within_time_range(
            recurring_event_parent=recurrence_event,
            time_range_lowerbound=datetime_range_lowerbound,
            time_range_upperbound=datetime_range_upperbound,
        ):
            existing_staged_instance_db = event_staging_db.filter(
                (pl.col("recurrence_parent_id") == instance["recurrence_parent_id"])
                & (pl.col("start_datetime") == instance["start_datetime"])
                & (pl.col("end_datetime") == instance["end_datetime"])
            )
            if existing_staged_instance_db.is_empty():
                # Prepare instance to be committed
                recurrence_instances.append(instance)
                # Prepare instance to be returned
                range_matched_instances.append(instance)
            else:
                # Prepare existing cached instance to be returned
                range_matched_instances.extend(existing_staged_instance_db.to_dicts())
        # Also add recurrence parent to show recurrence frequency and interval
        if recurrence_instances:
            range_matched_instances.append(recurrence_event)

        # Commit to staging area. Note that staged instances only need to be excluded when they are modified or deleted.
        current_context.add_to_database(
            namespace=DatabaseNamespace.CALENDAR_EVENTS_RECURRENCE_STAGING,
            rows=recurrence_instances,
        )
    # Recurrence instances that matches all criteria have now been collected in range_matched_instances,
    # Process filtered_calendar_event_dataframe based on range to get the full search result
    # Two conditions need to be met for an overlap
    #   1. end_datetime > datetime_range_lowerbound
    #   2. start_datetime < datetime_range_upperbound
    # Exclude recurrence
    range_matched_instances.extend(
        filtered_calendar_event_dataframe.filter(
            pl.col("end_datetime").map_elements(
                lambda x: datetime.datetime.fromisoformat(x)
                > datetime.datetime.fromisoformat(datetime_range_lowerbound),
                return_dtype=pl.Boolean,
            )
            & pl.col("start_datetime").map_elements(
                lambda x: datetime.datetime.fromisoformat(x)
                < datetime.datetime.fromisoformat(datetime_range_upperbound),
                return_dtype=pl.Boolean,
            )
            & (
                pl.col("recurrence_frequency").is_null()
                | pl.col("recurrence_interval").is_null()
            )
        ).to_dicts()
    )
    # Sort by increasing start time
    return sort_events_by_increasing_start_datetime(range_matched_instances)


# Principle removal tool for Calendar Event.
@register_as_tool(
    toolboxes={
        ToolboxName.FULL,
    },
    database_namespaces={
        DatabaseNamespace.CALENDAR_EVENTS,
        DatabaseNamespace.CALENDAR_EVENTS_RECURRENCE_STAGING,
    },
    visible_to=(RoleType.AGENT,),
)
@typechecked
def remove_calendar_event(calendar_event_id: str) -> None:
    """Remove a calendar event based on its uuid.

    Args:
        calendar_event_id:      UUID of calendar event to be removed. Note that in the case of a single instance of
                                recurring event, using its calendar_event_id only removes this event itself. To delete
                                all instances of a recurring event, use the calendar_event_id of the parent of
                                a recurring instance (e.g. the recurrence_parent_id of the instance).
    """
    current_context = get_current_context()
    calendar_event_db = current_context.get_database(
        namespace=DatabaseNamespace.CALENDAR_EVENTS
    )
    staging_db = current_context.get_database(
        namespace=DatabaseNamespace.CALENDAR_EVENTS_RECURRENCE_STAGING
    )
    calendar_event_db_entry = calendar_event_db.filter(
        pl.col("calendar_event_id") == calendar_event_id
    )
    staging_db_entry = staging_db.filter(
        pl.col("calendar_event_id") == calendar_event_id
    )
    if calendar_event_db_entry.is_empty() and staging_db_entry.is_empty():
        raise ValueError(f"{calendar_event_id=} not found.")
    if (
        staging_db_entry.select(pl.len())["len"][0] > 1
        or calendar_event_db_entry.select(pl.len())["len"][0] > 1
    ):
        raise DuplicateError(f"Duplicated events found for {calendar_event_id=}")
    # If removing single recurrence instance
    if not staging_db_entry.is_empty():
        recurrence_instance_dict = staging_db_entry.to_dicts()[0]
        # Remove from staging database
        current_context.remove_from_database(
            namespace=DatabaseNamespace.CALENDAR_EVENTS_RECURRENCE_STAGING,
            predicate=pl.col("calendar_event_id") == calendar_event_id,
        )
        # Exclude from parent
        parent_recurrence = calendar_event_db.filter(
            pl.col("calendar_event_id")
            == recurrence_instance_dict["recurrence_parent_id"]
        )
        if parent_recurrence.is_empty():
            raise NoDataError(
                f"Parent recurrence with "
                f"calendar_event_id={recurrence_instance_dict['recurrence_parent_id']} not found"
            )
        parent_recurrence_dict = parent_recurrence.to_dicts()[0]
        if parent_recurrence_dict["recurrence_exclude_datetime"] is None:
            parent_recurrence_dict["recurrence_exclude_datetime"] = []
        assert isinstance(parent_recurrence_dict["recurrence_exclude_datetime"], list)
        parent_recurrence_dict["recurrence_exclude_datetime"].append(
            recurrence_instance_dict["start_datetime"]
        )
        current_context.remove_from_database(
            namespace=DatabaseNamespace.CALENDAR_EVENTS,
            predicate=pl.col("calendar_event_id")
            == parent_recurrence_dict["calendar_event_id"],
        )
        current_context.add_to_database(
            namespace=DatabaseNamespace.CALENDAR_EVENTS,
            rows=[parent_recurrence_dict],
        )
    elif not calendar_event_db_entry.is_empty():
        instance_dict = calendar_event_db_entry.to_dicts()[0]
        current_context.remove_from_database(
            namespace=DatabaseNamespace.CALENDAR_EVENTS,
            predicate=pl.col("calendar_event_id") == instance_dict["calendar_event_id"],
        )
        # If a recurrence parent is removed, remove child from everywhere
        if instance_dict["recurrence_frequency"] is not None:
            current_context.remove_from_database(
                namespace=DatabaseNamespace.CALENDAR_EVENTS,
                predicate=pl.col("recurrence_parent_id")
                == instance_dict["calendar_event_id"],
                allow_no_match=True,
            )
            current_context.remove_from_database(
                namespace=DatabaseNamespace.CALENDAR_EVENTS_RECURRENCE_STAGING,
                predicate=pl.col("recurrence_parent_id")
                == instance_dict["calendar_event_id"],
                allow_no_match=True,
            )


# Principle modification tool for Calendar Event.
@register_as_tool(
    toolboxes={
        ToolboxName.FULL,
    },
    database_namespaces={
        DatabaseNamespace.CALENDAR_EVENTS,
        DatabaseNamespace.CALENDAR_EVENTS_RECURRENCE_STAGING,
    },
    visible_to=(RoleType.AGENT,),
)
@typechecked
def modify_calendar_event(
    calendar_event_id: str,
    calendar_id: str | NotGiven = NOT_GIVEN,
    title: str | NotGiven = NOT_GIVEN,
    description: str | NotGiven = NOT_GIVEN,
    start_datetime: str | NotGiven = NOT_GIVEN,
    end_datetime: str | NotGiven = NOT_GIVEN,
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
) -> None:
    """Modify a calendar event based on its uuid.

    Besides calendar_event_id, at least one modification field must be provided.


    Args:
        calendar_event_id:          UUID of calendar event to be modified. Note that in the case of a single instance of
                                    recurring event, using its calendar_event_id only modifies this event itself.
                                    To modify all instances of a recurring event, use the calendar_event_id of
                                    the parent of a recurring instance (e.g. the recurrence_parent_id of the instance).
                                    Modifying a parent recurrence event means previously occurred
                                    child instances are removed, and replaced with new instances with new uuids.
        calendar_id:                New string format unique identifier for calendar this event belongs to.
        title:                      New title summarizing calendar event.
        description:                New detailed description of calendar event.
        start_datetime:             New ISO 8601 datetime string with UTC timezone offset representing event start time,
                                    inclusive. E.g. 2024-12-09T15:00-08:00.
        end_datetime:               New ISO 8601 datetime string with UTC timezone offset representing event end time,
                                    exclusive. E.g. 2024-12-09T16:00-08:00.
        is_all_day:                 Boolean indicating all day events.
        recurrence_frequency:       Enum indicating new recurrence frequency.
        recurrence_interval:        Enum indicating new recurrence interval.
        recurrence_until_datetime:  ISO 8601 representing end of recurrence. Will be exact matched.
        latitude:                   New latitude of location.
        longitude:                  New longitude of location.
        place_id:                   New place_id of location.
        formatted_address:          New formatted address of location.
        attendee_names:             New list of attendee names.
        attendee_emails:            New list of attendee emails.

    """
    if all(
        x is NOT_GIVEN
        for x in [
            calendar_id,
            title,
            description,
            start_datetime,
            end_datetime,
            is_all_day,
            recurrence_frequency,
            recurrence_interval,
            recurrence_until_datetime,
            latitude,
            longitude,
            place_id,
            formatted_address,
            attendee_names,
            attendee_emails,
        ]
    ):
        raise ValueError(
            "No update information given. At least one new field should be provided in order to modify calendar event."
        )
    current_context = get_current_context()
    calendar_event_database = current_context.get_database(
        DatabaseNamespace.CALENDAR_EVENTS
    )
    calendar_event_staging_database = current_context.get_database(
        DatabaseNamespace.CALENDAR_EVENTS_RECURRENCE_STAGING
    )
    # Check if entry exists
    target_entry = pl.concat(
        [calendar_event_database, calendar_event_staging_database]
    ).filter(pl.col("calendar_event_id") == calendar_event_id)
    if target_entry.is_empty():
        raise NoDataError(f"No db entry matching {calendar_event_id=} found")
    # Check if entry is unique
    target_entry_dicts = target_entry.to_dicts()
    if len(target_entry_dicts) > 1:
        raise DuplicateError(f"More than 1 entry with {calendar_event_id=} found")
    target_entry_dict = target_entry_dicts[0]
    # Create updated entry
    for name, value in [
        ("calendar_id", calendar_id),
        ("title", title),
        ("description", description),
        ("start_datetime", start_datetime),
        ("end_datetime", end_datetime),
        ("is_all_day", is_all_day),
        ("recurrence_frequency", recurrence_frequency),
        ("recurrence_interval", recurrence_interval),
        ("recurrence_until_datetime", recurrence_until_datetime),
        ("latitude", latitude),
        ("longitude", longitude),
        ("place_id", place_id),
        ("formatted_address", formatted_address),
        ("attendee_names", attendee_names),
        ("attendee_emails", attendee_emails),
    ]:
        if value is not NOT_GIVEN:
            target_entry_dict[name] = value
    # Validate the updated fields before removing the old event.
    # prepare_calendar_event_fields can raise on invalid input (e.g.
    # end_datetime <= start_datetime), and we must not delete the
    # original event if validation fails.
    prepared_row = prepare_calendar_event_fields(**target_entry_dict)
    # Remove old event (handles recurrence cleanup).
    remove_calendar_event(calendar_event_id=calendar_event_id)
    # Add the validated entry to CALENDAR_EVENTS
    current_context.add_to_database(
        namespace=DatabaseNamespace.CALENDAR_EVENTS,
        rows=[prepared_row],
    )
