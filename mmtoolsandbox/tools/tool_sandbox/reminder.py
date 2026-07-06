# Copyright © 2026 Apple Inc.

"""A collection of tools which simulates common functions used for reminder."""

import datetime
import functools
from typing import (
    Literal,
    Optional,
    Union,
    cast,
)
from uuid import uuid4

import polars as pl
from polars.exceptions import DuplicateError, NoDataError

from mmtoolsandbox.common.execution_context import (
    DatabaseNamespace,
    RoleType,
    get_current_context,
)
from mmtoolsandbox.common.utils import (
    NOT_GIVEN,
    NotGiven,
    exact_match_filter_dataframe,
    filter_dataframe,
    fuzzy_match_filter_dataframe,
    gt_eq_filter_dataframe,
    lt_eq_filter_dataframe,
    register_as_tool,
)
from mmtoolsandbox.common.validators import (
    typechecked,
    validate_iso_8601_date_time_str,
    validate_latitude,
    validate_longitude,
)
from mmtoolsandbox.toolbox.names import ToolboxName


def _validate_reminder_datetime(value: str | NotGiven, name: str) -> None:
    """Validate a reminder datetime parameter if provided."""
    if isinstance(value, NotGiven):
        return
    if not isinstance(value, str):
        raise TypeError(
            f"{name} must be a string in ISO 8601 format, got {type(value).__name__}"
        )
    validate_iso_8601_date_time_str(value)


@register_as_tool(
    toolboxes={
        ToolboxName.FULL,
    },
    database_namespaces={DatabaseNamespace.REMINDER},
    visible_to=(RoleType.AGENT,),
)
@typechecked
def add_reminder(
    content: str,
    reminder_datetime: str,
    latitude: float | None = None,
    longitude: float | None = None,
) -> str:
    """Add a reminder

    Args:
        content:                Content of the reminder
        reminder_datetime:      When the user wants to be reminded. ISO 8601 datetime string (e.g. '2026-04-17T10:00:00+00:00')
        latitude:               Optional. Latitude of the location associated with this reminder
        longitude:              Optional. Longitude of the location associated with this reminder


    Returns:
        String format unique identifier for the reminder, this can be passed to other functions
        which require a unique identifier for this reminder

    """
    _validate_reminder_datetime(reminder_datetime, "reminder_datetime")
    validate_latitude(latitude, "latitude", Optional[float])
    validate_longitude(longitude, "longitude", Optional[float])

    current_context = get_current_context()
    # Create reminder uuid
    reminder_id = str(uuid4())
    current_context.add_to_database(
        namespace=DatabaseNamespace.REMINDER,
        rows=[
            {
                "reminder_id": reminder_id,
                "content": content,
                "creation_datetime": datetime.datetime.now(
                    tz=datetime.timezone.utc
                ).isoformat(),
                "reminder_datetime": reminder_datetime,
                "latitude": latitude,
                "longitude": longitude,
            }
        ],
    )
    return reminder_id


@register_as_tool(
    toolboxes={
        ToolboxName.FULL,
    },
    database_namespaces={DatabaseNamespace.REMINDER},
    visible_to=(RoleType.AGENT,),
)
@typechecked
def modify_reminder(
    reminder_id: str,
    content: str | NotGiven = NOT_GIVEN,
    reminder_datetime: str | NotGiven = NOT_GIVEN,
    latitude: float | None | NotGiven = NOT_GIVEN,
    longitude: float | None | NotGiven = NOT_GIVEN,
) -> None:
    """Modify a reminder with new information provided.

    Creation datetime is updated automatically.

    Args:
        reminder_id:            String format unique identifier of the reminder to be modified
        content:                New content
        reminder_datetime:      When the user wants to be reminded. ISO 8601 datetime string (e.g. '2026-04-17T10:00:00+00:00')
        latitude:               Optional. Latitude of the location associated with this reminder
        longitude:              Optional. Longitude of the location associated with this reminder

    Raises:
        ValueError:     When all arguments were None
        NoDataError:    When the reminder_id cannot be found in database
        DuplicateError: When multiple entries with the same id were found

    """
    _validate_reminder_datetime(reminder_datetime, "reminder_datetime")
    validate_latitude(latitude, "latitude", Union[Optional[float], NotGiven])
    validate_longitude(longitude, "longitude", Union[Optional[float], NotGiven])

    if all(x is NOT_GIVEN for x in [content, reminder_datetime, latitude, longitude]):
        raise ValueError(
            "No update information given. At least one new field should be provided among "
            "[content, reminder_datetime, latitude, longitude] in order to modify reminder"
        )
    current_context = get_current_context()
    reminder_database = current_context.get_database(DatabaseNamespace.REMINDER)
    # Check if entry exists
    target_entry = reminder_database.filter(pl.col("reminder_id") == reminder_id)
    if target_entry.is_empty():
        raise NoDataError(f"No db entry matching {reminder_id=} found")
    # Check if entry is unique
    target_entry_dicts = target_entry.to_dicts()
    if len(target_entry_dicts) > 1:
        raise DuplicateError(f"More than 1 entry with {reminder_id=} found")
    target_entry_dict = target_entry_dicts[0]
    # Create updated entry
    for field_name, field_value in [
        ("content", content),
        (
            "creation_datetime",
            datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        ),
        ("reminder_datetime", reminder_datetime),
        ("latitude", latitude),
        ("longitude", longitude),
    ]:
        if field_value is not NOT_GIVEN:
            target_entry_dict[field_name] = field_value
    # Update database
    current_context.remove_from_database(
        namespace=DatabaseNamespace.REMINDER,
        predicate=pl.col("reminder_id") == reminder_id,
    )
    current_context.add_to_database(
        namespace=DatabaseNamespace.REMINDER,
        rows=[target_entry_dict],
    )


@register_as_tool(
    toolboxes={
        ToolboxName.FULL,
    },
    database_namespaces={DatabaseNamespace.REMINDER},
    visible_to=(RoleType.AGENT,),
)
@typechecked
def search_reminder(
    reminder_id: str | NotGiven = NOT_GIVEN,
    content: str | NotGiven = NOT_GIVEN,
    creation_datetime_lowerbound: str | NotGiven = NOT_GIVEN,
    creation_datetime_upperbound: str | NotGiven = NOT_GIVEN,
    reminder_datetime_lowerbound: str | NotGiven = NOT_GIVEN,
    reminder_datetime_upperbound: str | NotGiven = NOT_GIVEN,
    latitude: float | NotGiven = NOT_GIVEN,
    longitude: float | NotGiven = NOT_GIVEN,
) -> list[
    dict[
        Literal[
            "reminder_id",
            "content",
            "creation_datetime",
            "reminder_datetime",
            "latitude",
            "longitude",
        ],
        str | float,
    ]
]:
    """Search for a reminder based on provided arguments

    Each field has a search criteria of either
    1. Exact value matching
    2. Fuzzy string matching with a predefined threshold
    3. Range matching datetime with upperbound or lowerbound
    Search results contains all reminder entries that matched all criteria

    Args:
        reminder_id:                    String format unique identifier for the reminder, will be exact matched
        content:                        Content of the reminder, will be fuzzy matched
        creation_datetime_lowerbound:   Lowerbound of the creation datetime (ISO 8601 format, e.g. '2026-04-17T10:00:00+00:00').
        creation_datetime_upperbound:   Upperbound of the creation datetime (ISO 8601 format, e.g. '2026-04-17T10:00:00+00:00').
        reminder_datetime_lowerbound:   Lowerbound of the reminder datetime (ISO 8601 format, e.g. '2026-04-17T10:00:00+00:00').
        reminder_datetime_upperbound:   Upperbound of the reminder datetime (ISO 8601 format, e.g. '2026-04-17T10:00:00+00:00').
        latitude:                       Latitude of the location associated with the reminder. Will be exact matched.
        longitude:                      Longitude of the location associated with the reminder. Will be exact matched.

    Returns:
        A List of matching reminders. An empty List if no matching reminders were found

    Raises:
        ValueError: When all arguments were not provided

    """
    _validate_reminder_datetime(
        creation_datetime_lowerbound, "creation_datetime_lowerbound"
    )
    _validate_reminder_datetime(
        creation_datetime_upperbound, "creation_datetime_upperbound"
    )
    _validate_reminder_datetime(
        reminder_datetime_lowerbound, "reminder_datetime_lowerbound"
    )
    _validate_reminder_datetime(
        reminder_datetime_upperbound, "reminder_datetime_upperbound"
    )
    validate_latitude(latitude, "latitude", Union[float, NotGiven])
    validate_longitude(longitude, "longitude", Union[float, NotGiven])

    current_context = get_current_context()
    reminder_dataframe = filter_dataframe(
        dataframe=current_context.get_database(namespace=DatabaseNamespace.REMINDER),
        filter_criteria=[
            ("reminder_id", reminder_id, exact_match_filter_dataframe),
            (
                "content",
                content,
                functools.partial(fuzzy_match_filter_dataframe, threshold=50),
            ),
            (
                "creation_datetime",
                creation_datetime_lowerbound,
                gt_eq_filter_dataframe,
            ),
            (
                "creation_datetime",
                creation_datetime_upperbound,
                lt_eq_filter_dataframe,
            ),
            (
                "reminder_datetime",
                reminder_datetime_lowerbound,
                gt_eq_filter_dataframe,
            ),
            (
                "reminder_datetime",
                reminder_datetime_upperbound,
                lt_eq_filter_dataframe,
            ),
            ("latitude", latitude, exact_match_filter_dataframe),
            ("longitude", longitude, exact_match_filter_dataframe),
        ],
    )
    return cast(
        list[
            dict[
                Literal[
                    "reminder_id",
                    "content",
                    "creation_datetime",
                    "reminder_datetime",
                    "latitude",
                    "longitude",
                ],
                str | float,
            ]
        ],
        reminder_dataframe.to_dicts(),
    )


@register_as_tool(
    toolboxes={
        ToolboxName.FULL,
    },
    database_namespaces={DatabaseNamespace.REMINDER},
    visible_to=(RoleType.AGENT,),
)
@typechecked
def remove_reminder(
    reminder_id: str,
) -> None:
    """Remove a reminder given its unique identifier

    Args:
        reminder_id:    String format unique identifier of the reminder to be removed

    Raises:
        NoDataError:    If the provided reminder_id was not found

    """
    current_context = get_current_context()
    current_context.remove_from_database(
        namespace=DatabaseNamespace.REMINDER,
        predicate=(pl.col("reminder_id") == reminder_id),
    )
