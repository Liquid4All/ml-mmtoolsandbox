# Copyright © 2026 Apple Inc.

"""Unit tests for mmtoolsandbox.common.relative_time"""

import datetime

import dateutil.tz
import polars as pl
import pytest
from dateutil.relativedelta import WE

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution_context import (
    CalendarRecurrenceFrequencyType,
    ExecutionContext,
)
from mmtoolsandbox.common.relative_time import (
    NAMESPACE_TO_RELATIVEDELTA_KEYS,
    NAMESPACE_TO_TIMESTAMP_KEYS,
    SerializableRelativeDelta,
    convert_relativedelta_list_to_absolute,
    convert_relativedelta_to_absolute,
    realign_timestamp,
    resolve_execution_context_relative_time,
)
from mmtoolsandbox.toolbox.loading import load_toolbox
from mmtoolsandbox.toolbox.names import ToolboxName


@pytest.fixture
def random_tuesday() -> datetime.datetime:
    return datetime.datetime(
        year=2025,
        month=4,
        day=15,
        tzinfo=dateutil.tz.tzoffset("Dummy Name", -25200),
    )


@pytest.fixture
def timestamp_execution_context() -> ExecutionContext:
    """ExecutionContext with timestamp populated in relevant database.

    Returns:
        An execution context object
    """
    toolbox = load_toolbox(ToolboxName.FULL, config={})
    test_context = ExecutionContext(toolbox=toolbox, delay_initialization=False)
    # Add entries, remember there's a headguard for each snapshot
    test_context._dbs[DatabaseNamespace.MESSAGING] = pl.DataFrame(
        [
            {
                "sandbox_message_index": 0,
                "message_id": None,
                "sender_person_id": None,
                "sender_phone_number": None,
                "recipient_person_id": None,
                "recipient_phone_number": None,
                "content": None,
                "creation_timestamp": None,
            },
            {
                "sandbox_message_index": 0,
                "message_id": "d674b27a-98a4-4339-a271-1359ce3ff199",
                "sender_person_id": "33a5432d-c92b-4bdd-a1f2-62f78d5d899a",
                "sender_phone_number": "1234567890",
                "recipient_person_id": "314ee6f5-db75-415f-bf7a-db9e741e125e",
                "recipient_phone_number": "0987654321",
                "content": "Test",
                "creation_timestamp": datetime.datetime(
                    year=2025, month=1, day=1
                ).timestamp(),
            },
        ],
        schema=ExecutionContext.dbs_schemas[DatabaseNamespace.MESSAGING],
    )

    # Add timestamp related database entry to test timestamp shifting logic
    test_context._dbs[DatabaseNamespace.REMINDER] = pl.DataFrame(
        [
            {
                "sandbox_message_index": 0,
                "reminder_id": "1",
                "content": "Test",
                "creation_datetime": datetime.datetime(
                    year=2025, month=1, day=1, tzinfo=datetime.timezone.utc
                ).isoformat(),
                "reminder_datetime": datetime.datetime(
                    year=2025, month=1, day=2, tzinfo=datetime.timezone.utc
                ).isoformat(),
            },
            {
                "sandbox_message_index": 0,
                "reminder_id": "2",
                "content": "Test",
                "creation_datetime": datetime.datetime(
                    year=2025, month=1, day=2, tzinfo=datetime.timezone.utc
                ).isoformat(),
            },
        ],
        schema=ExecutionContext.dbs_schemas[DatabaseNamespace.REMINDER],
    )

    return test_context


@pytest.fixture
def relative_time_execution_context(
    random_tuesday: datetime.datetime,
) -> ExecutionContext:
    """ExecutionContext with relative time populated in relevant database.

    Returns:
        An execution context object
    """
    toolbox = load_toolbox(ToolboxName.FULL, config={})
    test_context = ExecutionContext(toolbox=toolbox, delay_initialization=False)
    # Add entries, remember there's a headguard for each snapshot
    test_context._dbs[DatabaseNamespace.CALENDAR_EVENTS] = pl.DataFrame(
        [
            {
                "sandbox_message_index": 0,
                "calendar_event_id": None,
                "calendar_id": None,
                "title": None,
                "description": None,
                "start_datetime": None,
                "end_datetime": None,
                "is_all_day": None,
                "recurrence_frequency": None,
                "recurrence_interval": None,
                "recurrence_until_datetime": None,
                "recurrence_parent_id": None,
                "recurrence_exclude_datetime": None,
                "latitude": None,
                "longitude": None,
                "place_id": None,
                "formatted_address": None,
                "attendee_names": None,
                "attendee_emails": None,
            },
            {
                "sandbox_message_index": 0,
                "calendar_event_id": None,
                "calendar_id": None,
                "title": None,
                "description": None,
                "start_datetime": random_tuesday.isoformat(),
                "end_datetime": SerializableRelativeDelta(hours=1).to_json(),
                "is_all_day": None,
                "recurrence_frequency": CalendarRecurrenceFrequencyType.DAILY,
                "recurrence_interval": 1,
                "recurrence_until_datetime": SerializableRelativeDelta(
                    days=6
                ).to_json(),
                "recurrence_parent_id": None,
                "recurrence_exclude_datetime": [
                    SerializableRelativeDelta(days=1).to_json(),
                    (random_tuesday + SerializableRelativeDelta(days=2)).isoformat(),
                ],
                "latitude": None,
                "longitude": None,
                "place_id": None,
                "formatted_address": None,
                "attendee_names": None,
                "attendee_emails": None,
            },
        ],
        schema=ExecutionContext.dbs_schemas[DatabaseNamespace.CALENDAR_EVENTS],
    )

    return test_context


def test_realign_timestamp(timestamp_execution_context: ExecutionContext) -> None:
    execution_context = realign_timestamp(
        execution_context=timestamp_execution_context,
        execution_context_creation_time=datetime.datetime(year=2025, month=1, day=1),
        new_execution_context_creation_time=datetime.datetime(
            year=2025, month=1, day=1, hour=1
        ),
    )
    for namespace in NAMESPACE_TO_TIMESTAMP_KEYS:
        for timestamp_key in NAMESPACE_TO_TIMESTAMP_KEYS[namespace]:
            col_dtype = ExecutionContext.dbs_schemas[namespace].get(timestamp_key)
            for original, shifted in zip(
                timestamp_execution_context.get_database(
                    namespace=namespace, get_all_history_snapshots=True
                )[timestamp_key].to_list(),
                execution_context.get_database(
                    namespace=namespace, get_all_history_snapshots=True
                )[timestamp_key].to_list(),
            ):
                if shifted is None:
                    assert original is None
                elif col_dtype == pl.Float64:
                    assert (
                        original
                        == shifted - datetime.timedelta(hours=1).total_seconds()
                    )
                elif col_dtype == pl.String:
                    shifted_dt = datetime.datetime.fromisoformat(shifted)
                    expected_dt = shifted_dt - datetime.timedelta(hours=1)
                    assert original == expected_dt.isoformat()


def test_convert_relativedelta_to_absolute(random_tuesday: datetime.datetime) -> None:
    # Check relativedelta
    assert (
        convert_relativedelta_to_absolute(
            time=SerializableRelativeDelta(
                days=3, weekday=WE(1), hour=0, minute=0, second=0
            ).to_json(),
            reference_time=random_tuesday,
        )
        == datetime.datetime(
            year=2025,
            month=4,
            day=23,
            tzinfo=dateutil.tz.tzoffset("Dummy Name", -25200),
        ).isoformat()
    )
    # Check regular timestamp. No modification should be made
    assert (
        convert_relativedelta_to_absolute(
            time=random_tuesday.isoformat(),
            reference_time=random_tuesday,
        )
        == random_tuesday.isoformat()
    )
    # Anything else should raise exceptions
    with pytest.raises(ValueError):
        convert_relativedelta_to_absolute(
            time="2025/04/15", reference_time=random_tuesday
        )


def test_convert_relativedelta_list_to_absolute(
    relative_time_execution_context: ExecutionContext,
    random_tuesday: datetime.datetime,
) -> None:
    assert convert_relativedelta_list_to_absolute(
        time_list=[
            SerializableRelativeDelta(
                days=3, weekday=WE(1), hour=0, minute=0, second=0
            ).to_json(),
            random_tuesday.isoformat(),
        ],
        reference_time=random_tuesday,
    ) == [
        datetime.datetime(
            year=2025,
            month=4,
            day=23,
            tzinfo=dateutil.tz.tzoffset("Dummy Name", -25200),
        ).isoformat(),
        random_tuesday.isoformat(),
    ]
    assert convert_relativedelta_list_to_absolute(
        time_list=relative_time_execution_context._dbs[
            DatabaseNamespace.CALENDAR_EVENTS
        ]["recurrence_exclude_datetime"][-1],
        reference_time=random_tuesday,
    ) == [
        (random_tuesday + SerializableRelativeDelta(days=1)).isoformat(),
        (random_tuesday + SerializableRelativeDelta(days=2)).isoformat(),
    ]


def test_resolve_execution_context_relative_time(
    relative_time_execution_context: ExecutionContext, random_tuesday: datetime.datetime
) -> None:
    execution_context = resolve_execution_context_relative_time(
        execution_context=relative_time_execution_context,
        reference_time=random_tuesday,
    )
    for namespace in NAMESPACE_TO_RELATIVEDELTA_KEYS:
        for relative_delta_key in NAMESPACE_TO_RELATIVEDELTA_KEYS[namespace]:
            for maybe_relative_delta_list, absolute_time_list in zip(
                relative_time_execution_context.get_database(
                    namespace=namespace, get_all_history_snapshots=True
                )[relative_delta_key].to_list(),
                execution_context.get_database(
                    namespace=namespace, get_all_history_snapshots=True
                )[relative_delta_key].to_list(),
            ):
                if not isinstance(maybe_relative_delta_list, list):
                    maybe_relative_delta_list = [maybe_relative_delta_list]
                    absolute_time_list = [absolute_time_list]
                for maybe_relative_delta, absolute_time in zip(
                    maybe_relative_delta_list, absolute_time_list
                ):
                    try:
                        # If it is already an absolute time, no change
                        datetime.datetime.fromisoformat(maybe_relative_delta)
                        assert maybe_relative_delta == absolute_time
                    except (ValueError, TypeError):
                        assert (
                            SerializableRelativeDelta.from_json(maybe_relative_delta)
                            + random_tuesday
                        ).isoformat() == absolute_time
