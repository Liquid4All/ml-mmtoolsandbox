# Copyright © 2026 Apple Inc.

import json
from typing import Any, Dict

import polars as pl
import pytest

from mmtoolsandbox.common.evaluation import (
    ColumnSimilarityMeasureType,
    _fill_null,
    addition_similarity,
    column_exact_match_similarity,
    column_rouge_l_similarity,
    column_tool_trace_exact_match_similarity,
    snapshot_similarity,
    tool_trace_dependant_similarity,
)
from mmtoolsandbox.common.execution_context import (
    DatabaseNamespace,
    ExecutionContext,
    RoleType,
)


@pytest.fixture
def reference_snapshot() -> pl.DataFrame:
    """Testing snapshot

    Returns:
        A Dataframe object representing a snapshot
    """
    return pl.DataFrame(
        [
            {"sandbox_message_index": 0, "content": "Hello there", "num": 0},
        ]
    )


@pytest.fixture
def snapshot() -> pl.DataFrame:
    """Testing snapshot

    Returns:
        A Dataframe object representing a snapshot
    """
    return pl.DataFrame(
        [
            {"sandbox_message_index": 1, "content": "Hello there", "num": 0},
            {"sandbox_message_index": 1, "content": "Hey there", "num": 1},
        ]
    )


@pytest.fixture
def column_similarities() -> Dict[str, ColumnSimilarityMeasureType]:
    """Testing column similarities

    Returns:

    """
    return {
        "content": column_rouge_l_similarity,
        "num": column_exact_match_similarity,
    }


def test_snapshot_similarity(
    reference_snapshot: pl.DataFrame,
    snapshot: pl.DataFrame,
    column_similarities: Dict[str, ColumnSimilarityMeasureType],
) -> None:
    # Different row count
    assert (
        snapshot_similarity(
            snapshot=snapshot,
            target_dataframe=pl.DataFrame(
                [
                    {"content": "Hello there", "num": 0},
                ]
            ),
            column_similarities=column_similarities,
            reference_snapshot=reference_snapshot,
        )
        == 0
    )
    # Different columns
    assert (
        snapshot_similarity(
            snapshot=snapshot,
            target_dataframe=pl.DataFrame(
                [
                    {
                        "sandbox_message_index": 0,
                        "content": "Hello there",
                        "wrong_column": "?",
                    },
                ]
            ),
            column_similarities=column_similarities,
            reference_snapshot=reference_snapshot,
        )
        == 0
    )
    # Subset columns
    assert (
        snapshot_similarity(
            snapshot=snapshot,
            target_dataframe=pl.DataFrame(
                [
                    {"num": 0},
                    {"num": 1},
                ]
            ),
            column_similarities=column_similarities,
            reference_snapshot=reference_snapshot,
        )
        == 1
    )
    # Similarity shouldn't be affected by row orders
    assert (
        0
        < snapshot_similarity(
            snapshot=snapshot,
            target_dataframe=pl.DataFrame(
                [
                    {"content": "Hi there", "num": 0},
                    {"content": "Hi there", "num": 1},
                ]
            ),
            column_similarities=column_similarities,
            reference_snapshot=reference_snapshot,
        )
        == snapshot_similarity(
            snapshot=snapshot,
            target_dataframe=pl.DataFrame(
                [
                    {"content": "Hi there", "num": 1},
                    {"content": "Hi there", "num": 0},
                ]
            ),
            column_similarities=column_similarities,
            reference_snapshot=reference_snapshot,
        )
        < 1
    )


# TODO: Add tests for None
def test_addition_similarity(
    reference_snapshot: pl.DataFrame,
    snapshot: pl.DataFrame,
    column_similarities: Dict[str, ColumnSimilarityMeasureType],
) -> None:
    assert (
        addition_similarity(
            snapshot=snapshot,
            target_dataframe=pl.DataFrame(
                [
                    {"content": "Hi there", "num": 1},
                    {"content": "Hi there", "num": 0},
                ]
            ),
            column_similarities=column_similarities,
            reference_snapshot=pl.DataFrame(
                [
                    {"sandbox_message_index": 0, "content": "Hello there", "num": 2},
                ]
            ),
        )
        == 0
    )
    assert (
        addition_similarity(
            snapshot=snapshot,
            target_dataframe=pl.DataFrame(
                [
                    {"content": "Hi there", "num": 1},
                    {"content": "Hi there", "num": 0},
                ]
            ),
            column_similarities=column_similarities,
            reference_snapshot=reference_snapshot,
        )
        == 0
    )
    assert (
        addition_similarity(
            snapshot=snapshot,
            target_dataframe=pl.DataFrame(
                [
                    {"content": "Hi there", "num": 0},
                ]
            ),
            column_similarities=column_similarities,
            reference_snapshot=reference_snapshot,
        )
        == 0
    )
    assert (
        0
        < addition_similarity(
            snapshot=snapshot,
            target_dataframe=pl.DataFrame(
                [
                    {"content": "Hi there", "num": 1},
                ]
            ),
            column_similarities=column_similarities,
            reference_snapshot=reference_snapshot,
        )
        < 1
    )


@pytest.fixture
def tool_trace_snapshot() -> pl.DataFrame:
    """Testing snapshot containing a tool_trace.

    Returns:
        Testing snapshot containing a tool_trace.
    """
    return pl.DataFrame(
        [
            {
                "tool_trace": [
                    json.dumps(
                        {
                            "tool_name": "search_holiday",
                            "arguments": {
                                "holiday_name": "Christmas Day",
                                "year": 2024,
                            },
                        },
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {
                            "tool_name": "search_lat_lon",
                            "arguments": {
                                "latitude": 37.334606,
                                "longitude": -122.009102,
                            },
                        },
                        ensure_ascii=False,
                    ),
                ]
            }
        ],
        schema={
            "tool_trace": pl.List(pl.String),
        },
    )


def test_column_tool_trace_exact_match_similarity(
    tool_trace_snapshot: pl.DataFrame,
) -> None:
    # Matching against 1 golden trace. Only present arguments are checked
    assert (
        column_tool_trace_exact_match_similarity(
            dataframe=tool_trace_snapshot,
            column_name="tool_trace",
            value=json.dumps(
                {
                    "tool_name": "search_stock",
                    "arguments": {
                        "query": "Apple",
                    },
                }
            ),
        )["similarity"][0]
        == 0
    )
    assert (
        column_tool_trace_exact_match_similarity(
            dataframe=tool_trace_snapshot,
            column_name="tool_trace",
            value=json.dumps(
                {
                    "tool_name": "search_holiday",
                    "arguments": {
                        "holiday_name": "Christmas Day",
                        "year": None,
                    },
                }
            ),
        )["similarity"][0]
        == 0
    )
    assert (
        column_tool_trace_exact_match_similarity(
            dataframe=tool_trace_snapshot,
            column_name="tool_trace",
            value=json.dumps(
                {
                    "tool_name": "search_holiday",
                    "arguments": {
                        "holiday_name": "Christmas Day",
                        "year": 2024,
                    },
                }
            ),
        )["similarity"][0]
        == 1
    )
    assert (
        column_tool_trace_exact_match_similarity(
            dataframe=tool_trace_snapshot,
            column_name="tool_trace",
            value=json.dumps(
                {
                    "tool_name": "search_holiday",
                    "arguments": {
                        "year": 2024,
                    },
                }
            ),
        )["similarity"][0]
        == 1
    )
    # Test multiple golden traces. Should match if one of them matches
    assert (
        column_tool_trace_exact_match_similarity(
            dataframe=tool_trace_snapshot,
            column_name="tool_trace",
            value=json.dumps(
                [
                    {
                        "tool_name": "search_holiday",
                        "arguments": {
                            "holiday_name": "Christmas Day",
                            "year": None,
                        },
                    },
                    {
                        "tool_name": "search_holiday",
                        "arguments": {
                            "holiday_name": "Christmas Day",
                            "year": 2024,
                        },
                    },
                ]
            ),
        )["similarity"][0]
        == 1
    )
    # Exact match float
    assert (
        column_tool_trace_exact_match_similarity(
            dataframe=tool_trace_snapshot,
            column_name="tool_trace",
            value=json.dumps(
                {
                    "tool_name": "search_lat_lon",
                    "arguments": {
                        "latitude": 37.34,
                        "longitude": -122.01,
                    },
                },
            ),
        )["similarity"][0]
        == 0
    )
    # atol in range
    assert (
        column_tool_trace_exact_match_similarity(
            dataframe=tool_trace_snapshot,
            column_name="tool_trace",
            value=json.dumps(
                {
                    "tool_name": "search_lat_lon",
                    "arguments": {
                        "latitude": 37.34,
                        "longitude": -122.01,
                    },
                },
            ),
            atol_dict={"latitude": 0.2, "longitude": 0.2},
        )["similarity"][0]
        == 1
    )
    # atol out of range
    assert (
        column_tool_trace_exact_match_similarity(
            dataframe=tool_trace_snapshot,
            column_name="tool_trace",
            value=json.dumps(
                {
                    "tool_name": "search_lat_lon",
                    "arguments": {
                        "latitude": 37.34,
                        "longitude": -122.01,
                    },
                },
            ),
            atol_dict={"latitude": 0.002, "longitude": 0.002},
        )["similarity"][0]
        == 0
    )


@pytest.fixture
def tool_trace_dependant_similarity_kwargs() -> dict[str, Any]:
    """Testing kwargs for tool_trace_dependant_similarity.

    Returns:
        Testing kwargs for tool_trace_dependant_similarity.
    """
    return {
        "snapshot": pl.DataFrame(
            [
                {
                    "tool_trace": [
                        json.dumps(
                            {
                                "tool_name": "search_holiday",
                                "arguments": {
                                    "holiday_name": "Christmas Day",
                                    "year": 2024,
                                },
                            },
                            ensure_ascii=False,
                        ),
                    ]
                }
            ],
            schema=ExecutionContext.dbs_schemas[DatabaseNamespace.SANDBOX],
        ),
        "reference_snapshot": pl.DataFrame(
            [
                {
                    "tool_trace": [
                        json.dumps(
                            {"results": 2024},
                            ensure_ascii=False,
                        ),
                    ]
                }
            ],
            schema=ExecutionContext.dbs_schemas[DatabaseNamespace.SANDBOX],
        ),
        "column_similarities": {"tool_trace": column_tool_trace_exact_match_similarity},
        "fill_to": "tool_trace",
    }


def test_tool_trace_dependant_similarity(
    tool_trace_dependant_similarity_kwargs: dict[str, Any],
) -> None:
    # Test extracting 1 argument
    assert (
        tool_trace_dependant_similarity(
            **tool_trace_dependant_similarity_kwargs,
            target_dataframe=pl.DataFrame(
                {
                    "tool_trace": json.dumps(
                        {
                            "tool_name": "search_holiday",
                            "arguments": {
                                "holiday_name": "Christmas Day",
                            },
                        },
                        ensure_ascii=False,
                    ),
                }
            ),
            extractor=lambda x: [{"year": x["results"]}],
        )
        == 1
    )
    # Test extraction error
    assert (
        tool_trace_dependant_similarity(
            **tool_trace_dependant_similarity_kwargs,
            target_dataframe=pl.DataFrame(
                {
                    "tool_trace": json.dumps(
                        {
                            "tool_name": "search_holiday",
                            "arguments": {
                                "holiday_name": "Christmas Day",
                            },
                        },
                        ensure_ascii=False,
                    ),
                }
            ),
            extractor=lambda x: [{"day": x["results"]}],
        )
        == 0
    )
    # Test incorrect extracted argument
    assert (
        tool_trace_dependant_similarity(
            **tool_trace_dependant_similarity_kwargs,
            target_dataframe=pl.DataFrame(
                {
                    "tool_trace": json.dumps(
                        {
                            "tool_name": "search_holiday",
                            "arguments": {
                                "holiday_name": "Christmas Day",
                            },
                        },
                        ensure_ascii=False,
                    ),
                }
            ),
            extractor=lambda x: [{"year": x["results"] + 1}],
        )
        == 0
    )
    # Test prefers provided argument 1 argument
    assert (
        tool_trace_dependant_similarity(
            **tool_trace_dependant_similarity_kwargs,
            target_dataframe=pl.DataFrame(
                {
                    "tool_trace": json.dumps(
                        {
                            "tool_name": "search_holiday",
                            "arguments": {
                                "holiday_name": "Christmas Day",
                                "year": 2024,
                            },
                        },
                        ensure_ascii=False,
                    ),
                }
            ),
            extractor=lambda x: [{"year": x["results"] + 1}],
        )
        == 1
    )
    # Test multiple traces
    assert (
        tool_trace_dependant_similarity(
            **tool_trace_dependant_similarity_kwargs,
            target_dataframe=pl.DataFrame(
                {
                    "tool_trace": json.dumps(
                        [
                            {
                                "tool_name": "search_holiday",
                                "arguments": {
                                    "holiday_name": "Christmas Day",
                                    "year": None,
                                },
                            },
                            {
                                "tool_name": "search_holiday",
                                "arguments": {
                                    "holiday_name": "Christmas Day",
                                },
                            },
                        ],
                        ensure_ascii=False,
                    ),
                }
            ),
            extractor=lambda x: [{"year": x["results"]}],
        )
        == 1
    )
    # Test multiple traces, preferring provided value
    assert (
        tool_trace_dependant_similarity(
            **tool_trace_dependant_similarity_kwargs,
            target_dataframe=pl.DataFrame(
                {
                    "tool_trace": json.dumps(
                        [
                            {
                                "tool_name": "search_holiday",
                                "arguments": {
                                    "holiday_name": "Christmas Day",
                                    "year": 2024,
                                },
                            },
                            {
                                "tool_name": "search_holiday",
                                "arguments": {
                                    "holiday_name": "Christmas Day",
                                },
                            },
                        ],
                        ensure_ascii=False,
                    ),
                }
            ),
            extractor=lambda x: [{"year": x["results"] + 1}],
        )
        == 1
    )


def test_fill_null_with_all_dtypes() -> None:
    """Test that _fill_null behaves like fill_null(strategy='zero') for all non-list dtypes.

    This test covers all column dtypes actually used in evaluation and execution_context:
    - Int32, Float64, String, Boolean, Enum
    - List(Int32), List(Float64), List(String), List(Enum)
    """
    # Create a DataFrame with all dtypes used in execution_context, with null values
    df_with_nulls = pl.DataFrame(
        {
            "int_col": pl.Series([1, None, 3], dtype=pl.Int32),
            "float_col": pl.Series([1.5, None, 3.5], dtype=pl.Float64),
            "string_col": pl.Series(["a", None, "c"], dtype=pl.String),
            "bool_col": pl.Series([True, None, False], dtype=pl.Boolean),
            "enum_col": pl.Series(
                [RoleType.USER, None, RoleType.AGENT],
                dtype=pl.Enum([x for x in RoleType]),
            ),
            "list_int_col": pl.Series([[1, 2], None, [5]], dtype=pl.List(pl.Int32)),
            "list_float_col": pl.Series(
                [[1.1, 2.2], None, [5.5]], dtype=pl.List(pl.Float64)
            ),
            "list_string_col": pl.Series(
                [["a", "b"], None, ["e"]], dtype=pl.List(pl.String)
            ),
            "list_enum_col": pl.Series(
                [[RoleType.USER, RoleType.AGENT], None, [RoleType.SYSTEM]],
                dtype=pl.List(pl.Enum([x for x in RoleType])),
            ),
        }
    )

    # Apply _fill_null
    result = _fill_null(df_with_nulls)

    # Expected behavior:
    # 1. For non-list columns: fill_null(strategy="zero") behavior
    # 2. For list columns: fill with empty lists

    # Test non-list columns match fill_null(strategy="zero")
    for col in ["int_col", "float_col", "string_col", "bool_col", "enum_col"]:
        expected = df_with_nulls[col].fill_null(strategy="zero")
        assert result[col].to_list() == expected.to_list(), (
            f"Column {col} does not match fill_null(strategy='zero') behavior. "
            f"Got {result[col].to_list()}, expected {expected.to_list()}"
        )

    # Test list columns are filled with empty lists
    assert result["list_int_col"].to_list() == [[1, 2], [], [5]]
    assert result["list_float_col"].to_list() == [[1.1, 2.2], [], [5.5]]
    assert result["list_string_col"].to_list() == [["a", "b"], [], ["e"]]
    assert result["list_enum_col"].to_list() == [
        [RoleType.USER, RoleType.AGENT],
        [],
        [RoleType.SYSTEM],
    ]
