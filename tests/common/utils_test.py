# Copyright © 2026 Apple Inc.

"""Unit tests for mmtoolsandbox.common.utils"""

import datetime
import itertools
import json
from functools import partial
from typing import Any, Iterator

import polars as pl
import pytest

from mmtoolsandbox.common.execution_context import (
    DatabaseNamespace,
    ExecutionContext,
    RoleType,
    get_current_context,
    new_context,
)
from mmtoolsandbox.common.tool_registry import deregister_tool
from mmtoolsandbox.common.utils import (
    exact_match_filter_dataframe,
    exact_match_list_fully_contains_filter_dataframe,
    exact_match_struct_field_filter_dataframe,
    fuzzy_match_filter_dataframe,
    fuzzy_match_struct_field_filter_dataframe,
    get_next_iso_weekday_datetime_given_datetime,
    get_outer_tool_names,
    is_close,
    merge_dicts,
    range_filter_dataframe,
    register_as_tool,
)
from mmtoolsandbox.toolbox.names import ToolboxName


@pytest.fixture
def dataframe() -> pl.DataFrame:
    """
    Creates dataframe for matching filter testing

    Returns:
        Testing dataframe
    """
    names = [
        "John Wick",
        "Wick John",
        "Mike Portnoy",
        "Jerry with three hearts",
        "Jerry❤️❤️❤️",
        "Jerry💗",
    ]
    return pl.DataFrame(
        {
            "name": names,
            "age": [10, 20, 30, 30, 30, 30],
            "dob": [
                datetime.datetime(year=2000, month=10, day=2),
                datetime.datetime(year=2001, month=11, day=3),
                datetime.datetime(year=2002, month=12, day=4),
                datetime.datetime(year=1999, month=10, day=2),
                datetime.datetime(year=1999, month=10, day=2),
                datetime.datetime(year=1999, month=10, day=2),
            ],
            "jobs": [
                ["assassin", "actor", "cool dude"],
                None,
                ["Musician", "cool dude"],
                None,
                None,
                None,
            ],
            "person": [
                {"id": f"person_{i}", "name": name} for i, name in enumerate(names)
            ],
        }
    )


def test_exact_match_filter_dataframe(dataframe: pl.DataFrame) -> None:
    # Match success
    assert exact_match_filter_dataframe(
        dataframe=dataframe, column_name="name", value="John Wick"
    ).equals(dataframe[:1])
    # Match failure
    assert exact_match_filter_dataframe(
        dataframe=dataframe, column_name="name", value="john wick"
    ).is_empty()
    # Struct match success
    assert exact_match_struct_field_filter_dataframe(
        dataframe=dataframe,
        column_name="person",
        struct_field_name="name",
        value="John Wick",
    ).equals(dataframe[:1])
    # Struct match failure
    assert exact_match_struct_field_filter_dataframe(
        dataframe=dataframe,
        column_name="person",
        struct_field_name="name",
        value="john wick",
    ).is_empty()


def test_fuzzy_match_filter_dataframe(dataframe: pl.DataFrame) -> None:
    for filter_fun in [
        partial(fuzzy_match_filter_dataframe, column_name="name"),
        partial(
            fuzzy_match_struct_field_filter_dataframe,
            column_name="person",
            struct_field_name="name",
        ),
    ]:
        # Match success
        assert filter_fun(dataframe=dataframe, value="John Wick", threshold=100).equals(
            dataframe[:1]
        )
        # Casing invariant
        assert filter_fun(dataframe=dataframe, value="john wick", threshold=100).equals(
            dataframe[:1]
        )
        # Lower threshold
        assert filter_fun(dataframe=dataframe, value="John Wick", threshold=50).equals(
            dataframe[:2]
        )
        # Match failure
        assert filter_fun(
            dataframe=dataframe, value="John Petrucci", threshold=100
        ).is_empty()


def test_exact_match_list_fully_contains_filter_dataframe() -> None:
    df = pl.DataFrame([{"test": [0, 1, 2]}])
    assert not exact_match_list_fully_contains_filter_dataframe(
        dataframe=df, column_name="test", value=[2, 0]
    ).is_empty()
    assert exact_match_list_fully_contains_filter_dataframe(
        dataframe=df, column_name="test", value=[4]
    ).is_empty()


def test_fuzzy_match_filter_dataframe_for_large_dataframe() -> None:
    # First names starting with `Ja` so when doing a fuzzy search for 'Ja' we expect that we get back each row of the
    # dataframe.
    first_name_options = [
        "Jack",
        "Jackson",
        "Jacob",
        "Jake",
        "James",
        "Jan",
        "Jane",
        "Janet",
        "Jason",
        "Jasper",
        "Jax",
        "Jay",
        "Jayden",
    ]
    last_name_options = ["Appleseed", "John", "Portnoy", "Powers", "Wick"]
    first_last_name_tuples = list(
        itertools.product(first_name_options, last_name_options)
    )
    first_names, last_names = zip(*first_last_name_tuples)
    # The fuzzy matching uses `rapidfuzz.process.extract` under the hood, which has a hard-coded value of 5. This test
    # is to ensure that we set the limit to the length of the dataframe.
    assert len(first_names) > 5

    dataframe = pl.DataFrame(
        {
            "first_name": first_names,
            "last_name": last_names,
            "age": list(range(len(first_names))),
            "person": [
                {"id": f"person_{i}", "first_name": first_name}
                for i, first_name in enumerate(first_names)
            ],
        }
    )
    for filter_fun in [
        partial(fuzzy_match_filter_dataframe, column_name="first_name"),
        partial(
            fuzzy_match_struct_field_filter_dataframe,
            column_name="person",
            struct_field_name="first_name",
        ),
    ]:
        # All names start with 'Ja' so we should get back the original dataframe.
        filtered_df = filter_fun(dataframe=dataframe, value="Ja", threshold=90)
        assert filtered_df.equals(dataframe)


def test_range_filter_dataframe(dataframe: pl.DataFrame) -> None:
    # Earlier
    assert range_filter_dataframe(
        dataframe=dataframe,
        column_name="dob",
        value=datetime.datetime(year=2002, month=12, day=5),
        value_delta=datetime.timedelta(days=-2),
    ).equals(dataframe[2:3])
    # Later
    assert range_filter_dataframe(
        dataframe=dataframe,
        column_name="dob",
        value=datetime.datetime(year=2000, month=9, day=30),
        value_delta=datetime.timedelta(days=5),
    ).equals(dataframe[:1])


def test_is_close() -> None:
    assert is_close(value=1.0, reference=0.8, atol=0.3)
    assert is_close(value=1.0, reference=1.0)
    assert not is_close(value=1.0, reference=0.8)
    assert is_close(value=1, reference=1)


def test_nested_merge() -> None:
    base_dict = {
        "a": {
            "x": 1,
            "y": 2,
        },
        "b": 2,
    }
    overwrite_dict = {
        "a": {
            "y": 20,
            "z": 30,
        },
        "c": 3,
    }

    expected_result = {
        "a": {
            "x": 1,
            "y": 20,
            "z": 30,
        },
        "b": 2,
        "c": 3,
    }
    merged_dict = merge_dicts(base_dict, overwrite_dict)
    assert merged_dict == expected_result


def test_non_dict_merge() -> None:
    base_dict = {
        "a": {
            "x": 1,
        },
        "b": 2,
    }
    overwrite_dict = {"a": 10, "c": 3}

    expected_result = {
        "a": 10,
        "b": 2,
        "c": 3,
    }
    merged_dict = merge_dicts(base_dict, overwrite_dict)
    assert merged_dict == expected_result


def test_empty_overwrite() -> None:
    base_dict = {
        "a": 1,
        "b": {
            "c": 2,
        },
    }
    overwrite_dict: dict[str, Any] = {}

    merged_dict = merge_dicts(base_dict, overwrite_dict)
    assert merged_dict == base_dict


def test_empty_base() -> None:
    base_dict: dict[str, Any] = {}
    overwrite_dict = {
        "a": 1,
        "b": {
            "c": 2,
        },
    }

    merged_dict = merge_dicts(base_dict, overwrite_dict)
    assert merged_dict == overwrite_dict


@pytest.fixture
def deregister_mock_tool_tool_after_test_execution() -> Any:
    yield
    deregister_tool("mock_tool")


@pytest.mark.usefixtures("deregister_mock_tool_tool_after_test_execution")
def test_register_as_tool() -> None:
    toolboxes = {ToolboxName.FULL}
    database_namespaces = None
    visible_to = (RoleType.USER, RoleType.AGENT)

    @register_as_tool(
        toolboxes=toolboxes,
        database_namespaces=database_namespaces,
        visible_to=visible_to,
    )
    def mock_tool() -> None: ...

    assert {} == getattr(mock_tool, "database_namespaces")
    assert visible_to == getattr(mock_tool, "visible_to")


# These must be here due to assumptions made in nearest_outer_tool_frame_name:
#   If there is any outer frame functions / tool, it exists in the current frame f_globals.


@register_as_tool(
    toolboxes={
        ToolboxName.FULL,
    },
    visible_to=(RoleType.AGENT,),
)
def c() -> None: ...


@register_as_tool(
    toolboxes={
        ToolboxName.FULL,
    },
    visible_to=(RoleType.AGENT,),
)
def d() -> None: ...


@register_as_tool(
    toolboxes={
        ToolboxName.FULL,
    },
    visible_to=(RoleType.AGENT,),
)
def e() -> list[str]:
    return get_outer_tool_names()


@register_as_tool(
    toolboxes={
        ToolboxName.FULL,
    },
    visible_to=(RoleType.AGENT,),
)
def f() -> None: ...


@register_as_tool(
    toolboxes={
        ToolboxName.FULL,
    },
    visible_to=(RoleType.AGENT,),
)
def b() -> tuple[list[str], list[str]]:
    d()
    return e(), get_outer_tool_names()


@register_as_tool(
    toolboxes={
        ToolboxName.FULL,
    },
    visible_to=(RoleType.AGENT,),
)
def a() -> tuple[list[str], list[str], list[str]]:
    result = b()
    c()
    return result[0], result[1], get_outer_tool_names()


def test_nearest_outer_tool_frame_name() -> None:
    assert a() == (["a", "b"], ["a"], [])


@pytest.fixture
def tool_trace_execution_context() -> Iterator[None]:
    """Fixture which will setup and teardown execution context before and after each test function

    Returns:

    """
    test_context = ExecutionContext()
    # Add dummy message (tool tracing assumes non-empty SANDBOX DB)
    test_context.add_to_database(
        namespace=DatabaseNamespace.SANDBOX,
        rows=[
            {
                "sender": RoleType.AGENT,
                "recipient": RoleType.EXECUTION_ENVIRONMENT,
                "content": None,
            },
        ],
    )
    # Enable tracing
    test_context.trace_tool = True
    with new_context(test_context):
        yield


@pytest.fixture
def nested_tool_trace_execution_context() -> Iterator[None]:
    """Fixture which will setup and teardown execution context before and after each test function

    Returns:

    """
    test_context = ExecutionContext()
    # Add dummy message (tool tracing assumes non-empty SANDBOX DB)
    test_context.add_to_database(
        namespace=DatabaseNamespace.SANDBOX,
        rows=[
            {
                "sender": RoleType.AGENT,
                "recipient": RoleType.EXECUTION_ENVIRONMENT,
                "content": None,
            },
        ],
    )
    # Enable tracing
    test_context.trace_tool = True
    # Enable nested tracing
    test_context.trace_nested_tool = True
    with new_context(test_context):
        yield


@pytest.mark.usefixtures("tool_trace_execution_context")
def test_tool_trace() -> None:
    a()
    f()
    # Only top level tools should be traced
    traces: list[dict[str, Any]] = [
        json.loads(x)
        for x in get_current_context().get_database(DatabaseNamespace.SANDBOX)[
            "tool_trace"
        ][0]
    ]
    assert traces == [
        {"tool_name": "a", "arguments": {}, "result": [["a", "b"], ["a"], []]},
        {"tool_name": "f", "arguments": {}, "result": None},
    ]


@pytest.mark.usefixtures("nested_tool_trace_execution_context")
def test_nested_tool_trace() -> None:
    a()
    f()
    # Nested tool traces should be traced. Expected trace structure would be
    #    ------------
    #  |             |
    #  a ------      f
    #  |       |
    #  b --    c
    #  |   |
    #  d   e
    #
    traces: list[dict[str, Any]] = [
        json.loads(x)
        for x in get_current_context().get_database(DatabaseNamespace.SANDBOX)[
            "tool_trace"
        ][0]
    ]
    assert traces == [
        {
            "tool_name": "a",
            "arguments": {},
            "result": [["a", "b"], ["a"], []],
            "call_stack_tool_names": [],
            "children_tool_trace": [
                {
                    "tool_name": "b",
                    "arguments": {},
                    "result": [["a", "b"], ["a"]],
                    "call_stack_tool_names": ["a"],
                    "children_tool_trace": [
                        {
                            "tool_name": "d",
                            "arguments": {},
                            "result": None,
                            "call_stack_tool_names": ["a", "b"],
                            "children_tool_trace": [],
                        },
                        {
                            "tool_name": "e",
                            "arguments": {},
                            "result": ["a", "b"],
                            "call_stack_tool_names": ["a", "b"],
                            "children_tool_trace": [],
                        },
                    ],
                },
                {
                    "tool_name": "c",
                    "arguments": {},
                    "result": None,
                    "call_stack_tool_names": ["a"],
                    "children_tool_trace": [],
                },
            ],
        },
        {
            "tool_name": "f",
            "arguments": {},
            "result": None,
            "call_stack_tool_names": [],
            "children_tool_trace": [],
        },
    ]


def test_get_next_iso_weekday_datetime_given_datetime() -> None:
    """Test get_next_iso_weekday_datetime_given_datetime function with various scenarios."""

    # Test case 1: Monday (1) - basic case
    current_dt = datetime.datetime(2023, 8, 15, 10, 30, 45)  # Tuesday, August 15, 2023
    result = get_next_iso_weekday_datetime_given_datetime(current_dt, 1)  # Next Monday
    expected = datetime.datetime(2023, 8, 21, 10, 30, 45)  # Monday, August 21, 2023
    assert result == expected
    assert result.isoweekday() == 1  # Verify it's a Monday

    # Test case 2: Tuesday (2)
    current_dt = datetime.datetime(2023, 8, 15, 14, 20, 30)  # Tuesday, August 15, 2023
    result = get_next_iso_weekday_datetime_given_datetime(current_dt, 2)  # Next Tuesday
    expected = datetime.datetime(2023, 8, 22, 14, 20, 30)  # Tuesday, August 22, 2023
    assert result == expected
    assert result.isoweekday() == 2  # Verify it's a Tuesday

    # Test case 3: Wednesday (3)
    current_dt = datetime.datetime(2023, 8, 15, 9, 15, 0)  # Tuesday, August 15, 2023
    result = get_next_iso_weekday_datetime_given_datetime(
        current_dt, 3
    )  # Next Wednesday
    expected = datetime.datetime(2023, 8, 23, 9, 15, 0)  # Wednesday, August 23, 2023
    assert result == expected
    assert result.isoweekday() == 3  # Verify it's a Wednesday

    # Test case 4: Thursday (4)
    current_dt = datetime.datetime(2023, 8, 15, 16, 45, 12)  # Tuesday, August 15, 2023
    result = get_next_iso_weekday_datetime_given_datetime(
        current_dt, 4
    )  # Next Thursday
    expected = datetime.datetime(2023, 8, 24, 16, 45, 12)  # Thursday, August 24, 2023
    assert result == expected
    assert result.isoweekday() == 4  # Verify it's a Thursday

    # Test case 5: Friday (5)
    current_dt = datetime.datetime(2023, 8, 15, 8, 0, 0)  # Tuesday, August 15, 2023
    result = get_next_iso_weekday_datetime_given_datetime(current_dt, 5)  # Next Friday
    expected = datetime.datetime(2023, 8, 25, 8, 0, 0)  # Friday, August 25, 2023
    assert result == expected
    assert result.isoweekday() == 5  # Verify it's a Friday

    # Test case 6: Saturday (6)
    current_dt = datetime.datetime(2023, 8, 15, 12, 30, 45)  # Tuesday, August 15, 2023
    result = get_next_iso_weekday_datetime_given_datetime(
        current_dt, 6
    )  # Next Saturday
    expected = datetime.datetime(2023, 8, 26, 12, 30, 45)  # Saturday, August 26, 2023
    assert result == expected
    assert result.isoweekday() == 6  # Verify it's a Saturday

    # Test case 7: Sunday (7)
    current_dt = datetime.datetime(2023, 8, 15, 18, 45, 30)  # Tuesday, August 15, 2023
    result = get_next_iso_weekday_datetime_given_datetime(current_dt, 7)  # Next Sunday
    expected = datetime.datetime(2023, 8, 27, 18, 45, 30)  # Sunday, August 27, 2023
    assert result == expected
    assert result.isoweekday() == 7  # Verify it's a Sunday
