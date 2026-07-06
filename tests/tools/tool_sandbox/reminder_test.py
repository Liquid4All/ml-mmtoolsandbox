# Copyright © 2026 Apple Inc.

import datetime
import uuid
from typing import Dict, Iterator, Union

import polars as pl
import pytest
from polars.exceptions import DuplicateError, NoDataError

from mmtoolsandbox.common.execution_context import (
    DatabaseNamespace,
    ExecutionContext,
    get_current_context,
    new_context,
)
from mmtoolsandbox.tools.tool_sandbox.reminder import (
    add_reminder,
    modify_reminder,
    remove_reminder,
    search_reminder,
)


@pytest.fixture
def new_reminder() -> Dict[str, Union[str, float, None]]:
    """Provides a dictionary containing information about a new reminder entry

    Returns:

    """
    return {
        "content": "Return chocolate milk",
        "reminder_datetime": (
            datetime.datetime.now(tz=datetime.timezone.utc)
            + datetime.timedelta(days=4, hours=4, minutes=5, seconds=6)
        ).isoformat(),
        "latitude": 37.3237926356735,
        "longitude": -122.03961770355414,
    }


@pytest.fixture(scope="function", autouse=True)
def execution_context() -> Iterator[None]:
    """Autouse fixture which will setup and teardown execution context before and after each test function

    Returns:

    """
    # Set test context
    test_context = ExecutionContext()
    test_context.add_to_database(
        namespace=DatabaseNamespace.REMINDER,
        rows=[
            {
                "reminder_id": str(
                    uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="reminder_0")
                ),
                "content": "Write tests",
                "creation_datetime": (
                    datetime.datetime.now(tz=datetime.timezone.utc)
                    - datetime.timedelta(days=10, hours=1, minutes=2, seconds=3)
                ).isoformat(),
                "reminder_datetime": (
                    datetime.datetime.now(tz=datetime.timezone.utc)
                    - datetime.timedelta(days=8, hours=4, minutes=5, seconds=6)
                ).isoformat(),
                "latitude": None,
                "longitude": None,
            },
            {
                "reminder_id": str(
                    uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="reminder_1")
                ),
                "content": "Buy chocolate milk",
                "creation_datetime": (
                    datetime.datetime.now(tz=datetime.timezone.utc)
                    - datetime.timedelta(hours=1, minutes=2, seconds=3)
                ).isoformat(),
                "reminder_datetime": (
                    datetime.datetime.now(tz=datetime.timezone.utc)
                    + datetime.timedelta(days=2, hours=4, minutes=5, seconds=6)
                ).isoformat(),
                "latitude": 37.3237926356735,
                "longitude": -122.03961770355414,
            },
        ],
    )
    with new_context(test_context):
        yield


def test_add_reminder(new_reminder: Dict[str, Union[str, float, None]]) -> None:
    add_reminder(**new_reminder)
    current_context = get_current_context()
    reminder_database = current_context.get_database(
        namespace=DatabaseNamespace.REMINDER
    )
    assert reminder_database["content"][-1] == new_reminder["content"]
    assert (
        reminder_database["reminder_datetime"][-1] == new_reminder["reminder_datetime"]
    )
    assert reminder_database["latitude"][-1] == new_reminder["latitude"]
    assert reminder_database["longitude"][-1] == new_reminder["longitude"]


def test_remove_reminder() -> None:
    current_context = get_current_context()
    reminder_database = current_context.get_database(
        namespace=DatabaseNamespace.REMINDER
    )
    assert not reminder_database.filter(
        pl.col("reminder_id")
        == str(uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="reminder_1"))
    ).is_empty()
    # Successful
    remove_reminder(
        reminder_id=str(uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="reminder_1"))
    )
    reminder_database = current_context.get_database(
        namespace=DatabaseNamespace.REMINDER
    )
    assert reminder_database.filter(
        pl.col("reminder_id")
        == str(uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="reminder_1"))
    ).is_empty()
    # Error
    with pytest.raises(NoDataError):
        remove_reminder("This should raise error")


def test_modify_reminder() -> None:
    current_context = get_current_context()
    reminder_id = str(uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="reminder_1"))
    creation_datetime = current_context.get_database(
        namespace=DatabaseNamespace.REMINDER
    ).filter(pl.col("reminder_id") == reminder_id)["creation_datetime"][-1]
    modify_reminder(
        reminder_id=reminder_id,
        content="Test",
    )
    # Content should change
    assert (
        current_context.get_database(namespace=DatabaseNamespace.REMINDER).filter(
            pl.col("reminder_id") == reminder_id
        )["content"][-1]
        == "Test"
    )
    # Creation datetime should change (new one is later)
    assert (
        current_context.get_database(namespace=DatabaseNamespace.REMINDER).filter(
            pl.col("reminder_id") == reminder_id
        )["creation_datetime"][-1]
        > creation_datetime
    )
    new_datetime = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
    modify_reminder(
        reminder_id=reminder_id,
        reminder_datetime=new_datetime,
        latitude=None,
        longitude=None,
    )
    assert (
        current_context.get_database(namespace=DatabaseNamespace.REMINDER).filter(
            pl.col("reminder_id") == reminder_id
        )["reminder_datetime"][-1]
        == new_datetime
    )
    assert (
        current_context.get_database(namespace=DatabaseNamespace.REMINDER).filter(
            pl.col("reminder_id") == reminder_id
        )["latitude"][-1]
        is None
    )
    assert (
        current_context.get_database(namespace=DatabaseNamespace.REMINDER).filter(
            pl.col("reminder_id") == reminder_id
        )["longitude"][-1]
        is None
    )
    # Error
    with pytest.raises(ValueError):
        modify_reminder(reminder_id=reminder_id)
    with pytest.raises(NoDataError):
        modify_reminder(
            reminder_id="This should raise error", content="This should raise error"
        )
    current_context.add_to_database(
        namespace=DatabaseNamespace.REMINDER,
        rows=current_context.get_database(
            namespace=DatabaseNamespace.REMINDER
        ).to_dicts()[-1:],
    )
    with pytest.raises(DuplicateError):
        modify_reminder(
            reminder_id=reminder_id,
            content="This should raise error",
        )


def test_search_reminder() -> None:
    current_context = get_current_context()
    reminder_id = str(uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="reminder_1"))
    target_reminder = (
        current_context.get_database(namespace=DatabaseNamespace.REMINDER)
        .filter(pl.col("reminder_id") == reminder_id)
        .to_dicts()[0]
    )
    assert [target_reminder] == search_reminder(reminder_id=reminder_id)
    assert [target_reminder] == search_reminder(content=target_reminder["content"])

    # Datetime range matching — ISO 8601 strings compare lexicographically
    creation_dt = target_reminder["creation_datetime"]
    # Shift by subtracting/adding 10 seconds
    lower = (
        datetime.datetime.fromisoformat(creation_dt) - datetime.timedelta(seconds=10)
    ).isoformat()
    upper = (
        datetime.datetime.fromisoformat(creation_dt) + datetime.timedelta(seconds=10)
    ).isoformat()
    assert [target_reminder] == search_reminder(
        creation_datetime_lowerbound=lower,
        creation_datetime_upperbound=upper,
    )

    reminder_dt = target_reminder["reminder_datetime"]
    lower = (
        datetime.datetime.fromisoformat(reminder_dt) - datetime.timedelta(seconds=10)
    ).isoformat()
    upper = (
        datetime.datetime.fromisoformat(reminder_dt) + datetime.timedelta(seconds=10)
    ).isoformat()
    assert [target_reminder] == search_reminder(
        reminder_datetime_lowerbound=lower,
        reminder_datetime_upperbound=upper,
    )

    assert [target_reminder] == search_reminder(
        latitude=target_reminder["latitude"],
    )
    assert [target_reminder] == search_reminder(
        longitude=target_reminder["longitude"],
    )
    # No arguments
    with pytest.raises(ValueError):
        search_reminder()
    # Exact matching failure
    assert (
        len(
            search_reminder(
                reminder_id=str(
                    uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="wrong id")
                )
            )
        )
        == 0
    )
    # Fuzzy matching failure
    assert len(search_reminder(content="Hello")) == 0
    # Range matching failure — lowerbound after the actual datetime
    upper_bound = (
        datetime.datetime.fromisoformat(reminder_dt) + datetime.timedelta(seconds=10)
    ).isoformat()
    assert (
        len(
            search_reminder(
                reminder_datetime_lowerbound=upper_bound,
            )
        )
        == 0
    )
