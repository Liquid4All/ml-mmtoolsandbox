# Copyright © 2026 Apple Inc.

from __future__ import annotations

import copy
import datetime
import functools
import json
from typing import Any

import polars as pl
from dateutil.relativedelta import relativedelta, weekday

from mmtoolsandbox.common.databases import DatabaseNamespace as DatabaseNamespace
from mmtoolsandbox.common.execution_context import ExecutionContext

# Mapping containing timestamp keys in each database namespace schema
NAMESPACE_TO_TIMESTAMP_KEYS: dict[DatabaseNamespace, list[str]] = {
    DatabaseNamespace.MESSAGING: ["creation_timestamp"],
    DatabaseNamespace.REMINDER: ["creation_datetime", "reminder_datetime"],
}


def realign_timestamp(
    execution_context: ExecutionContext,
    execution_context_creation_time: datetime.datetime,
    new_execution_context_creation_time: datetime.datetime,
) -> ExecutionContext:
    """Some model eval takes a very long time ( > 1day ) in the case of MOE models, which will render
    timestamp created at scenario creation time invalid.
    Here we piggyback on reference time to implement a hack where we shift timestamp to
    current time from creation time.

    This hacks works under the following assumptions:
    1. All scenario timestamps in the dataset are defined by a diff against creation time, with no exceptions
        1.1 Because of this, we can realign simply by adding the diff between new_creation_time and creation time

    Args:
        execution_context:                      Starting context.
        execution_context_creation_time:        Original creation time when this execution context and the
                                                timestamp within was created.
        new_execution_context_creation_time:    New creation time to align towards.

    Returns:
        Execution context with timestamp realigned against new_execution_context_creation_time
    """
    execution_context = copy.deepcopy(execution_context)
    delta_seconds = (
        new_execution_context_creation_time.timestamp()
        - execution_context_creation_time.timestamp()
    )
    for namespace in NAMESPACE_TO_TIMESTAMP_KEYS:
        for key in NAMESPACE_TO_TIMESTAMP_KEYS[namespace]:
            if namespace not in execution_context._dbs:
                continue
            col_dtype = execution_context.dbs_schemas[namespace].get(key)
            if col_dtype == pl.Float64:
                execution_context._dbs[namespace] = execution_context._dbs[
                    namespace
                ].with_columns(
                    pl.col(key)
                    .map_elements(
                        lambda x: x + delta_seconds if x is not None else None,
                        return_dtype=pl.Float64,
                    )
                    .alias(key)
                )
            elif col_dtype == pl.String:
                execution_context._dbs[namespace] = execution_context._dbs[
                    namespace
                ].with_columns(
                    pl.col(key)
                    .map_elements(
                        lambda x: _shift_iso_datetime(x, delta_seconds)
                        if x is not None
                        else None,
                        return_dtype=pl.String,
                    )
                    .alias(key)
                )
    return execution_context


def _shift_iso_datetime(iso_str: str, delta_seconds: float) -> str:
    """Shift an ISO 8601 datetime string by a number of seconds."""
    dt = datetime.datetime.fromisoformat(iso_str)
    dt += datetime.timedelta(seconds=delta_seconds)
    return dt.isoformat()


class SerializableRelativeDelta(relativedelta):
    """relativedelta that can be serialized into a dict

    Using relativedelta is not a complete solution. You cannot express
    things like "next working day". To support this one would need to use
    rrule.
    """

    dict_keys: set[str] = {
        "years",
        "months",
        "days",
        "leapdays",
        "hours",
        "minutes",
        "seconds",
        "microseconds",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "microsecond",
        "weekday_weekday",
        "weekday_n",
    }

    def to_dict(self) -> dict[str, Any]:
        """Convert relativedelta to dict. Most notably

        1. Extracts N from weekday(N)
        2. Ignores weeks (since it was folded into days during init)


        Returns:
            A dict containing necessary info to reconstruct relativedelta.
        """
        return {
            "years": self.years,
            "months": self.months,
            "days": self.days,
            "leapdays": self.leapdays,
            "hours": self.hours,
            "minutes": self.minutes,
            "seconds": self.seconds,
            "microseconds": self.microseconds,
            "year": self.year,
            "month": self.month,
            "day": self.day,
            "hour": self.hour,
            "minute": self.minute,
            "second": self.second,
            "microsecond": self.microsecond,
            "weekday_weekday": self.weekday.weekday
            if self.weekday is not None
            else None,
            "weekday_n": self.weekday.n if self.weekday is not None else None,
        }

    @classmethod
    def from_dict(
        cls, relative_delta_dict: dict[str, Any]
    ) -> SerializableRelativeDelta:
        """Deserialize a relativedelta from a dict.

        Must follow the same structure to_dict produces

        Args:
            relative_delta_dict:    A dict of the same format as to_dict return value.

        Returns:
            A deserialized SerializableRelativeDelta.
        """
        relative_delta_dict = copy.deepcopy(relative_delta_dict)
        if not set(relative_delta_dict.keys()) == cls.dict_keys:
            raise ValueError(
                f"Expecting dict with keys {cls.dict_keys}, found {set(relative_delta_dict.keys())}"
            )
        if relative_delta_dict["weekday_weekday"] is not None:
            relative_delta_dict["weekday"] = weekday(
                weekday=relative_delta_dict["weekday_weekday"],
                n=relative_delta_dict["weekday_n"],
            )
        del relative_delta_dict["weekday_weekday"]
        del relative_delta_dict["weekday_n"]
        return cls(**relative_delta_dict)

    def to_json(self) -> str:
        """Convert SerializableRelativeDelta to a json string.

        Returns:
            Json string serialization of SerializableRelativeDelta.
        """
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> SerializableRelativeDelta:
        """Deserialize a relativedelta from a json string.

        Must follow the same structure to_json produces

        Args:
            json_str:   A string of the same format as to_json return value.


        Returns:
            A deserialized SerializableRelativeDelta.
        """
        return cls.from_dict(json.loads(json_str))


# Mapping containing relativedelta keys in each database namespace schema
NAMESPACE_TO_RELATIVEDELTA_KEYS: dict[DatabaseNamespace, list[str]] = {
    DatabaseNamespace.CALENDAR_EVENTS: [
        "start_datetime",
        "end_datetime",
        "recurrence_until_datetime",
        "recurrence_exclude_datetime",
    ],
}


def convert_relativedelta_to_absolute(
    time: str | None, reference_time: datetime.datetime
) -> str | None:
    """Convert a possible relativedelta to absolute ISO 8601 datetime string with timezone.

    Args:
        time:           Time string. Could be a ISO 8601 already or SerializableRelativeDelta json_str.
        reference_time: Reference time the relativedelta bases off of

    Returns:
        ISO 8601 datetime string with timezone.
    """
    # Cover null entries
    if time is None:
        return time
    # Try resolving as datetime string. In which case time should already be an absolute
    # time. No additional change needed.
    try:
        datetime.datetime.fromisoformat(time)
        return time
    except (ValueError, TypeError):
        pass
    # Try resolving as SerializableRelativeDelta json_str.
    try:
        delta = SerializableRelativeDelta.from_json(time)
        return (reference_time + delta).isoformat()
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(
            f"Unrecognized time string detected {time=}. "
            f"Should either be an ISO 8601, or a json serialized SerializableRelativeDelta."
        ) from e


def convert_relativedelta_list_to_absolute(
    time_list: list[str | pl.Series | None], reference_time: datetime.datetime
) -> list[str | None] | None:
    """List version of convert_relativedelta_to_absolute

    Args:
        time_list:      A list of time to be converted.
        reference_time: Reference time the relativedelta bases off of

    Returns:
        A list of ISO 8601 datetime string with timezone.
    """
    if time_list is None:
        return None
    if isinstance(time_list, (list, pl.Series)):
        return [
            convert_relativedelta_to_absolute(time=time, reference_time=reference_time)  # type: ignore[arg-type]
            for time in time_list
        ]
    raise ValueError("Column must contain a list of strings.")


def resolve_execution_context_relative_time(
    execution_context: ExecutionContext, reference_time: datetime.datetime
) -> ExecutionContext:
    """Converts relativedelta entries in scenario starting state to absolute time.

    Some model eval takes a very long time ( > 1day ) in the case of MOE models, which will render
    time created at scenario creation time invalid.
    We solve this by defining relative time in relativedelta, and convert them
    into absolute time at Scenario.play

    This solution works under the following assumptions:
    1. All relative time are expressed as relativedelta, dilled into a string.
        1.1 This is to get around polars typing restriction. pl.Object deepcopy is not allowed.

    Args:
        execution_context:  Execution Context containing relative delta
        reference_time:     Reference time to base relative delta off of

    Returns:
        Modified execution context with absolute time only.
    """
    execution_context = copy.deepcopy(execution_context)
    for database_namespace in NAMESPACE_TO_RELATIVEDELTA_KEYS:
        # Check columns that are defined as pl.Objects
        # They can potentially be relativedelta
        for column_name in NAMESPACE_TO_RELATIVEDELTA_KEYS[database_namespace]:
            # Cover string columns
            if (
                execution_context.dbs_schemas[database_namespace][column_name]
                == pl.String
            ):
                map_function = functools.partial(
                    convert_relativedelta_to_absolute, reference_time=reference_time
                )
            elif execution_context.dbs_schemas[database_namespace][
                column_name
            ] == pl.List(pl.String):
                map_function = functools.partial(
                    convert_relativedelta_list_to_absolute,  # type: ignore[arg-type]
                    reference_time=reference_time,
                )  # type: ignore[assignment]
            else:
                raise ValueError(
                    "Relative time column must be either pl.String or pl.List(pl.String) type. "
                    f"Found {execution_context.dbs_schemas[database_namespace][column_name]}"
                )
            # Override column
            execution_context._dbs[database_namespace] = execution_context._dbs[
                database_namespace
            ].with_columns(
                pl.col(column_name)
                .map_elements(
                    function=map_function,
                    return_dtype=execution_context.dbs_schemas[database_namespace][
                        column_name
                    ],
                )
                .alias(column_name)
            )
    return execution_context
