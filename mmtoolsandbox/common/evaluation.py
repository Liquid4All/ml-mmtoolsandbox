# Copyright © 2026 Apple Inc.

"""Evaluation primitives for MMToolSandbox scenario outcomes.

Provides column-level similarity functions (exact match, ROUGE-L, datetime,
tool trace, etc.) and snapshot-level similarity measures (addition, removal,
update, guardrail) used to compare database states before and after agent
execution.  Also defines the ``EvaluationResult`` and ``EvaluationCriteria``
dataclasses consumed by ``entity_diff_evaluator`` and
``scenario.play_and_evaluate()``, and the top-level ``evaluate()`` orchestrator
that combines entity-diff scoring with an LLM judge.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import (
    Any,
    Literal,
    Sequence,
    cast,
)

import numpy as np
import polars as pl
from attrs import define, field
from polars.exceptions import SchemaError
from rouge_score import rouge_scorer  # type: ignore[import-untyped]
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]
from typing_extensions import Protocol

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution_context import ExecutionContext, RoleType
from mmtoolsandbox.common.i18n import Locale, get_rouge_scorer_kwargs
from mmtoolsandbox.common.tool_trace_extractors import ToolTraceExtractorType
from mmtoolsandbox.common.utils import NOT_GIVEN, all_logging_disabled, is_close

LOGGER = getLogger(__name__)


class ColumnSimilarityMeasureType(Protocol):
    """Callable type def for column similarity measure functions

    Each similarity measure takes a dataframe, column to calculate similarity on and value,
    to return a Dataframe representing similarity between each row in the dataframe and the value
    Similarities are [0, 1] real values.
    """

    def __call__(
        self,
        dataframe: pl.DataFrame,
        column_name: str,
        value: Any,
        atol_dict: dict[str, float] | None = None,
    ) -> pl.DataFrame: ...


def column_ignore_similarity(
    dataframe: pl.DataFrame,
    column_name: str,
    value: Any,
    atol_dict: dict[str, float] | None = None,
) -> pl.DataFrame:
    """A similarity that always returns maximum similarity scores.

    Args:
        dataframe:      Dataframe to calculate similarity on
        column_name:    Column name to compare
        value:          Value the column should compare against
        atol_dict:      Absolute tolerance for each argument

    Returns:
        A Dataframe containing maximum similarity scores 1.
    """
    # Use explicit list to produce one 1.0 per row.  In polars < 1.0.0, pl.lit()
    # produces a single-row scalar, which breaks flexible_row_matching that
    # expects one score per actual row.
    n = dataframe.height
    return pl.DataFrame({"similarity": [1.0] * n}, schema={"similarity": pl.Float32})


def column_datetime_similarity(
    dataframe: pl.DataFrame,
    column_name: str,
    value: Any,
    atol_dict: dict[str, float] | None = None,
) -> pl.DataFrame:
    """A 0/1 similarity checking that the actual timestamp is strictly after the expected value.

    Used for bookkeeping timestamps (``updated_at``, ``created_at``, etc.)
    on **update** operations.  The expected value should match the entity's
    initial timestamp (set during conversion).  If the agent actually
    modified the entity, AppWorld sets the timestamp to ``DateTime.now()``
    which will be strictly later.  If the entity was not touched, the
    timestamp stays at its initial value and correctly fails.

    Both sides are parsed to epoch seconds so that format differences
    (Z suffix, +offset, space vs T separator, microseconds) don't matter.
    Timezone-naive values are assumed UTC.

    Args:
        dataframe:      Dataframe to calculate similarity on
        column_name:    Column name to compare
        value:          Expected datetime (initial timestamp / lower bound)
        atol_dict:      Absolute tolerance for each argument (unused)

    Returns:
        A Dataframe containing similarity score. 1 if actual > expected, 0 otherwise.
    """
    if value is None:
        return dataframe.select(pl.lit(1, dtype=pl.Float32).alias("similarity"))
    ref_epoch = _parse_datetime_to_epoch(str(value))
    if ref_epoch is None:
        return dataframe.select(pl.lit(0, dtype=pl.Float32).alias("similarity"))
    ref_epoch_s = int(ref_epoch)
    return dataframe.select(
        pl.col(column_name)
        .cast(pl.Utf8)
        .map_elements(
            lambda s: 1.0
            if int(_parse_datetime_to_epoch(s, 0.0) or 0) > ref_epoch_s
            else 0.0,
            return_dtype=pl.Float64,
        )
        .cast(pl.Float32)
        .alias("similarity")
    )


# Keep column_after_similarity as an alias for backward compatibility.
column_after_similarity = column_datetime_similarity


def _parse_datetime_naive(s: str) -> str | None:
    """Parse an ISO 8601-ish datetime and return its timezone-naive representation.

    Strips timezone info so that ``10:00:00Z`` and ``10:00:00-08:00`` both
    become ``2026-04-20T10:00:00``.  This is appropriate when the scenario
    provides no timezone context and the agent is forced to guess an offset.

    Returns:
        Naive ISO 8601 string (no offset, no microseconds), or None on failure.
    """
    from datetime import datetime

    if not isinstance(s, str) or not s:
        return None
    try:
        normalized = s.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        if len(normalized) > 10 and normalized[10] == " ":
            normalized = normalized[:10] + "T" + normalized[11:]
        dt = datetime.fromisoformat(normalized)
        return dt.replace(tzinfo=None, microsecond=0).isoformat()
    except (ValueError, TypeError):
        return None


def column_datetime_naive_equal_similarity(
    dataframe: pl.DataFrame,
    column_name: str,
    value: Any,
    atol_dict: dict[str, float] | None = None,
) -> pl.DataFrame:
    """A 0/1 similarity that compares datetimes after stripping timezone info.

    Scenarios that provide no timezone context to the agent should use this
    instead of ``column_exact_match_similarity`` for user-facing datetime
    fields (``start_datetime``, ``end_datetime``, ``reminder_datetime``,
    ``due_date``).  The agent is forced to guess a timezone offset, so only
    the date-and-time-of-day portion is meaningful.

    Args:
        dataframe:      Dataframe to calculate similarity on.
        column_name:    Column name to compare.
        value:          Expected datetime string (ISO 8601).
        atol_dict:      Absolute tolerance (unused).

    Returns:
        A Dataframe containing similarity score. 1 if naive datetimes match,
        0 otherwise.
    """
    if value is None:
        return dataframe.select(
            pl.col(column_name).is_null().cast(pl.Float32).alias("similarity")
        )
    expected_naive = _parse_datetime_naive(str(value))
    if expected_naive is None:
        return dataframe.select(pl.lit(0, dtype=pl.Float32).alias("similarity"))
    return dataframe.select(
        pl.col(column_name)
        .cast(pl.Utf8)
        .map_elements(
            lambda s: 1.0 if _parse_datetime_naive(s) == expected_naive else 0.0,
            return_dtype=pl.Float64,
        )
        .cast(pl.Float32)
        .alias("similarity")
    )


def _parse_datetime_to_epoch(s: str, default: float | None = None) -> float | None:
    """Parse an ISO 8601-ish datetime string to epoch seconds.

    Handles all common variants:
    - ``2026-03-25T22:03:10``        (canonical ISO 8601)
    - ``2026-03-25T22:03:10Z``       (UTC indicator)
    - ``2026-03-25T22:03:10+00:00``  (explicit offset)
    - ``2026-03-25 22:03:10.926011`` (SQLite / Python str(datetime))

    All inputs are treated as UTC (timezone-naive values are assumed UTC).

    Returns:
        Epoch seconds as float, or *default* if parsing fails.
    """
    from datetime import datetime, timezone

    if not isinstance(s, str) or not s:
        return default
    try:
        normalized = s.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        # Python 3.10 fromisoformat needs T separator
        if len(normalized) > 10 and normalized[10] == " ":
            normalized = normalized[:10] + "T" + normalized[11:]
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return default


def column_exact_match_similarity(
    dataframe: pl.DataFrame,
    column_name: str,
    value: Any,
    atol_dict: dict[str, float] | None = None,
) -> pl.DataFrame:
    """A 0/1 similarity based on exact match.

    Args:
        dataframe:      Dataframe to calculate similarity on
        column_name:    Column name to compare
        value:          Value the column should compare against
        atol_dict:      Absolute tolerance for each argument

    Returns:
        A Dataframe containing similarity score. 0 for no match, 1 for match
    """
    if value is not None:
        # BUG FIX: polars .eq() silently returns 0 for every row when the column
        # dtype is pl.List(...).  In polars < 1.0.0, broadcasting a Python list
        # against a List-type column via .eq() is not supported — the operation
        # produces all-zero results instead of raising an error, making
        # fields like `recipient_ids` (list[Int64]) always score 0.0 even when
        # the agent correctly sent to the right recipient(s).
        #
        # We detect List-type columns via the presence of the `.inner` attribute
        # (pl.List(Int64) has `.inner = Int64`; scalar types like Int64 do not).
        # For these columns we sort both sides before comparing so that list order
        # does not affect the score (e.g., [100, 200] == [200, 100]).
        col_dtype = dataframe.schema[column_name]
        if hasattr(col_dtype, "inner"):  # True only for pl.List(...) dtype
            sorted_expected = str(sorted(value) if isinstance(value, list) else [value])
            # polars < 1.0: map_elements on a List column passes each row's
            # value as a polars Series, not a Python list.  str(Series)
            # produces a multi-line repr that never matches str(list).
            # Convert via .to_list() first so both sides use Python list repr.
            # The PolarsInefficientMapWarning can be ignored — evaluation
            # DataFrames are always tiny (1–5 rows), so the overhead is zero.
            return dataframe.select(
                pl.col(column_name)
                .list.sort()
                .map_elements(lambda s: str(s.to_list()), return_dtype=pl.Utf8)
                .eq(sorted_expected)
                .cast(pl.Float32)
                .alias("similarity")
            )
        # AppWorld stores list columns as JSON strings in SQLite (e.g.
        # '["coworker"]', '[100, 200]').  Polars loads these as pl.Utf8,
        # so the pl.List branch above is never triggered.  When the
        # expected value is a Python list and the column is a string,
        # parse the JSON and compare as sorted lists.
        if isinstance(value, list) and col_dtype in (pl.Utf8, pl.Utf8):
            import json as _json

            expected_sorted_list = sorted(value, key=str)

            def _compare_json_list(s: object) -> float:
                try:
                    actual = _json.loads(s) if isinstance(s, str) else s
                    if not isinstance(actual, list):
                        actual = [actual]
                    return (
                        1.0 if sorted(actual, key=str) == expected_sorted_list else 0.0
                    )
                except (TypeError, ValueError, _json.JSONDecodeError):
                    return 0.0

            return dataframe.select(
                pl.col(column_name)
                .map_elements(_compare_json_list, return_dtype=pl.Float64)
                .cast(pl.Float32)
                .alias("similarity")
            )
        # Datetime normalization: if the expected value parses as an ISO 8601
        # datetime, compare both sides as epoch seconds so that format
        # differences ("Z" vs "+00:00" vs "-07:00") don't cause false
        # negatives.  E.g. "2026-04-20T17:00:00Z" should equal
        # "2026-04-20T10:00:00-07:00".
        if isinstance(value, str):
            expected_epoch = _parse_datetime_to_epoch(value)
            if expected_epoch is not None:
                expected_epoch_int = int(expected_epoch)
                return dataframe.select(
                    pl.col(column_name)
                    .cast(pl.Utf8)
                    .map_elements(
                        lambda s: 1.0
                        if int(_parse_datetime_to_epoch(s, -1.0) or -1)
                        == expected_epoch_int
                        else 0.0,
                        return_dtype=pl.Float64,
                    )
                    .cast(pl.Float32)
                    .alias("similarity")
                )
        try:
            return dataframe.select(
                pl.col(column_name).eq(value).cast(pl.Float32).alias("similarity")
            )
        except Exception:
            # Type mismatch (e.g. comparing a string entity-ref like
            # "messages_text_message_100" against an Int64 column).
            # polars raises ComputeError in this case.  Fall back to
            # string-cast comparison so the evaluator doesn't crash silently.
            return dataframe.select(
                pl.col(column_name)
                .cast(pl.Utf8)
                .eq(str(value))
                .cast(pl.Float32)
                .alias("similarity")
            )
    else:
        return dataframe.select(
            pl.col(column_name).is_null().cast(pl.Float32).alias("similarity")
        )


def column_close_similarity(
    dataframe: pl.DataFrame,
    column_name: str,
    value: Any,
    atol_dict: dict[str, float] | None = None,
) -> pl.DataFrame:
    """A 0/1 similarity based on how close values are.

    Only works on int / float, and requires atol_dict.

    Args:
        dataframe:      Dataframe to calculate similarity on
        column_name:    Column name to compare
        value:          Value the column should compare against
        atol_dict:      Absolute tolerance for each argument

    Returns:
        A Dataframe containing similarity score. 0 for no match, 1 for match
    """
    assert isinstance(value, (int, float)) and atol_dict is not None

    def check_close(x: tuple[str | None]) -> float:
        """UDF checking if x [0] is close to value

        Args:
            x:  single value tuple containing a tool trace

        Returns:
            0 / 1 match score
        """
        if is_close(value=value, reference=x[0], atol=atol_dict.get(column_name, None)):
            return 1
        return 0

    return (
        dataframe.select(pl.col(column_name))
        .map_rows(function=check_close, return_dtype=pl.Float32)
        .select(pl.col("map").alias("similarity"))
    )


def column_one_similarity(
    dataframe: pl.DataFrame,
    column_name: str,
    value: Any,
    atol_dict: dict[str, float] | None = None,
) -> pl.DataFrame:
    """A similarity that always returns 1 as similarity score.

    Used for columns that should be ignored in evaluation — every row
    receives maximum similarity regardless of its value.

    Args:
        dataframe:      Dataframe to calculate similarity on.
        column_name:    Column name (unused — all rows get 1.0).
        value:          Expected value (unused — all rows get 1.0).
        atol_dict:      Absolute tolerance (unused).

    Returns:
        A DataFrame with a ``"similarity"`` column of all 1.0 values.
    """
    return dataframe.with_columns(pl.lit(1.0).alias("similarity")).select("similarity")


def column_contains_similarity(
    dataframe: pl.DataFrame,
    column_name: str,
    value: Any,
    atol_dict: dict[str, float] | None = None,
    case_sensitive: bool = True,
) -> pl.DataFrame:
    """A 0/1 similarity whether the value is contained in column value string

    Args:
        dataframe:      Dataframe to calculate similarity on
        column_name:    Column name to compare, must contain string
        value:          Value the column should compare against, must be string value
        atol_dict:      Absolute tolerance for each argument
        case_sensitive: Whether the string comparison should be case-sensitive. Defaults to True.

    Returns:
        A Dataframe containing similarity score. 0 if the column value do not contain target value, 1 if it does
    """
    # Apply case transformations if needed
    column_expr = pl.col(column_name)
    search_value = value

    if not case_sensitive:
        column_expr = column_expr.str.to_lowercase()
        search_value = str(value).lower()

    return dataframe.select(
        column_expr.str.contains_any([search_value])
        .cast(pl.Float32)
        .alias("similarity")
    )


def column_tool_trace_exact_match_similarity(
    dataframe: pl.DataFrame,
    column_name: str,
    value: Any,
    atol_dict: dict[str, float] | None = None,
) -> pl.DataFrame:
    """A 0/1 similarity whether a tool trace matches a provided trace.

    Function name and argument in provided trace is always matched. Arguments not provided are ignored

    Provided trace can be a list of possible traces, in which case a match is found
    if any of the provided traces matches.

    Args:
        dataframe:      Dataframe to calculate similarity on
        column_name:    Column name to compare, must contain string
        value:          Json dumped 1 or a list of tool traces.
        atol_dict:      Absolute tolerance for each argument

    Returns:
        A Dataframe containing similarity score. 0 if the column value do not contain target value, 1 if it does
    """
    trace: dict[str, Any] | list[dict[str, Any]] = json.loads(value)
    # Normalize into a list of possible golden traces
    golden_traces: list[dict[str, Any]] = (
        [trace] if not isinstance(trace, Sequence) else list(trace)
    )

    def match_trace(x: tuple[str | None]) -> float:
        """UDF calculating trace matching score

        Args:
            x:  single value tuple containing a tool trace

        Returns:
            0 / 1 match score
        """
        if x[0] is None:
            return 0
        # If any trace matches any golden trace, return 1
        for golden_trace in golden_traces:
            for tool_trace_json in x[0]:
                tool_trace = json.loads(tool_trace_json)
                if tool_trace["tool_name"] == golden_trace["tool_name"] and all(
                    is_close(
                        tool_trace["arguments"].get(argument_name, NOT_GIVEN),
                        golden_trace["arguments"][argument_name],
                        atol=atol_dict.get(argument_name, None)
                        if atol_dict is not None
                        else None,
                    )
                    for argument_name in golden_trace["arguments"]
                ):
                    return 1
        return 0

    return (
        dataframe.select(pl.col(column_name))
        .map_rows(function=match_trace, return_dtype=pl.Float32)
        .select(pl.col("map").alias("similarity"))
    )


def column_rouge_l_similarity(
    dataframe: pl.DataFrame,
    column_name: str,
    value: Any,
    atol_dict: dict[str, float] | None = None,
    locale: Locale | None = Locale.en_US,
) -> pl.DataFrame:
    """Similarity defined by ROUGE score. Only applicable for string values

    Args:
        dataframe:      Dataframe to calculate similarity on
        column_name:    Column name to compare, must contain string
        value:          Value the column should compare against, must be string value
        atol_dict:      Absolute tolerance for each argument
        locale:         Locale used to determine the tokenizer for Rouge-L

    Returns:
        A Dataframe containing similarity score.
    """
    with all_logging_disabled():
        scorer_kwargs = get_rouge_scorer_kwargs(locale or Locale.en_US)
        scorer = rouge_scorer.RougeScorer(["rougeL"], **scorer_kwargs)

        def rouge_l_score(x: tuple[str]) -> float:
            """UDF calculating rouge L

            Args:
                x:  single value tuple containing the column in question

            Returns:
                float rouge L f score
            """
            return cast(
                float, scorer.score(target=value, prediction=x[0])["rougeL"].fmeasure
            )

        return (
            dataframe.select(pl.col(column_name))
            .map_rows(function=rouge_l_score, return_dtype=pl.Float32)
            .select(pl.col("map").alias("similarity"))
        )


class SnapshotSimilarityMeasureType(Protocol):
    """Callable type def for snapshot similarity measure functions

    Each similarity measure takes a snapshot, target dataframe, column similarities and reference snapshot
    to return a float representing similarity between snapshot and other specified constraints
    Similarities are [0, 1] real values.
    """

    def __call__(
        self,
        snapshot: pl.DataFrame,
        target_dataframe: pl.DataFrame | None = None,
        column_similarities: dict[str, ColumnSimilarityMeasureType] | None = None,
        reference_snapshot: pl.DataFrame | None = None,
        **kwargs: str | ToolTraceExtractorType,
    ) -> float: ...


def _fill_null(df: pl.DataFrame) -> pl.DataFrame:
    """Fill null values in a dataframe with type-appropriate defaults.

    - Numeric/temporal columns: fill with zero
    - Boolean columns: fill with False
    - String/Categorical columns: fill with ""
    - Enum columns: fill with first category (strategy="zero")
    - List columns: fill with empty lists
    - Struct and other complex types: left as-is

    Args:
        df: DataFrame to fill null values in

    Returns:
        DataFrame with null values filled
    """
    fill_exprs = []
    for col in df.columns:
        dtype = df[col].dtype
        if isinstance(dtype, pl.List):
            fill_exprs.append(pl.col(col).fill_null([]))
        elif dtype in pl.NUMERIC_DTYPES or dtype in (
            pl.Date,
            pl.Datetime,
            pl.Time,
            pl.Duration,
        ):
            fill_exprs.append(pl.col(col).fill_null(strategy="zero"))
        elif dtype == pl.Boolean:
            fill_exprs.append(pl.col(col).fill_null(False))
        elif dtype in (pl.Utf8, pl.Categorical):
            fill_exprs.append(pl.col(col).fill_null(""))
        elif isinstance(dtype, pl.Enum):
            fill_exprs.append(pl.col(col).fill_null(strategy="zero"))
        else:
            # For Struct and other complex types, leave nulls as-is
            pass

    if fill_exprs:
        df = df.with_columns(fill_exprs)

    return df


def snapshot_similarity(
    snapshot: pl.DataFrame,
    target_dataframe: pl.DataFrame | None = None,
    column_similarities: dict[str, ColumnSimilarityMeasureType] | None = None,
    reference_snapshot: pl.DataFrame | None = None,
    **kwargs: str | ToolTraceExtractorType,
) -> float:
    """Measures the similarity between each database snapshot and a target_dataframe.

        1. If the number of rows, or column names doesn't match, similarity is always 0
        2. Within each snapshot, calculate a similarity between each row in the target and each row in the snapshot
            2.1 Row-wise similarity is determined by the geo mean of column similarities
            2.2 Only columns provided in target_dataframe was considered
        3. Find a 1 to 1 mapping between snapshot rows and target rows that maximizes geo mean of row similarities.
            Said geo mean is the snapshot similarity

    Args:
        snapshot:                   The dataframe snapshot to calculate similarity for
        target_dataframe:           The dataframe to calculate similarity with
        column_similarities:        A dictionary of column name to column-wise similarity measure
        reference_snapshot:         Not utilized by this similarity

    Returns:
        A [0, 1] similarity score between target_dataframe and snapshot
    """
    assert target_dataframe is not None and column_similarities is not None
    # Check for row number and columns
    if snapshot.select(pl.len())["len"][0] != target_dataframe.select(pl.len())["len"][
        0
    ] or set(target_dataframe.columns) - set(snapshot.columns):
        return 0.0
    # Create N * N cost matrix (- log similarity). This ensures when assignment cost is minimized
    # through hungarian algorithm, row-wise geo metric mean of similarity is maximized
    cost_matrix: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    for row in target_dataframe.to_dicts():
        # - log similarity and column-wise geo mean
        column_cost_df = pl.concat(
            [
                column_similarities[column_name](
                    dataframe=snapshot,
                    column_name=column_name,
                    value=value,
                ).select(-pl.col("similarity").log().alias(f"similarity_{i}"))
                for i, (column_name, value) in enumerate(row.items())
            ],
            how="horizontal",
        )
        cost_matrix.append(
            column_cost_df.select(
                pl.mean_horizontal(*column_cost_df.columns).alias("mean")
            )["mean"].to_numpy()
        )
    numpy_cost_matrix = np.stack(cost_matrix, axis=0)
    try:
        # Solve for assignment that minimizes cost matrix
        row_ind, col_ind = linear_sum_assignment(numpy_cost_matrix)
    except ValueError:
        # cost matrix is infeasible (always results in inf). Return 0
        return 0
    # Calculate similarity
    return cast(float, np.exp(-numpy_cost_matrix[row_ind, col_ind].mean()).tolist())


def addition_similarity(
    snapshot: pl.DataFrame,
    target_dataframe: pl.DataFrame | None = None,
    column_similarities: dict[str, ColumnSimilarityMeasureType] | None = None,
    reference_snapshot: pl.DataFrame | None = None,
    **kwargs: str | ToolTraceExtractorType,
) -> float:
    """Measures the similarity between each database snapshot and a target_dataframe, if the snapshot is supposed to
    be derived from adding target_dataframe onto reference_snapshot

        1. If snapshot does not fully contain reference_snapshot, similarity is always 0
        2. If it does, the anti join of the two is compared against target_dataframe using snapshot_similarity

    Args:
        snapshot:                   The dataframe snapshot to calculate similarity for
        target_dataframe:           The dataframe to calculate similarity with
        column_similarities:        A dictionary of column name to column-wise similarity measure
        reference_snapshot:         When similarity is 1,
                                    snapshot should be the result of adding target_dataframe into reference_snapshot

    Returns:
        A [0, 1] similarity score between target_dataframe and snapshot
    """
    assert (
        target_dataframe is not None
        and column_similarities is not None
        and reference_snapshot is not None
    )
    # Drop sandbox_message_index
    # Fill null with zero to prevent join failure
    snapshot = _fill_null(snapshot.drop("sandbox_message_index"))
    reference_snapshot = _fill_null(reference_snapshot.drop("sandbox_message_index"))
    target_dataframe = _fill_null(target_dataframe)
    if (
        reference_snapshot.select(pl.len())["len"][0]
        != snapshot.join(reference_snapshot, on=snapshot.columns, how="inner").select(
            pl.len()
        )["len"][0]
    ):
        return 0
    return snapshot_similarity(
        snapshot.join(reference_snapshot, on=snapshot.columns, how="anti"),
        target_dataframe=target_dataframe,
        column_similarities=column_similarities,
        reference_snapshot=reference_snapshot,
    )


def removal_similarity(
    snapshot: pl.DataFrame,
    target_dataframe: pl.DataFrame | None = None,
    column_similarities: dict[str, ColumnSimilarityMeasureType] | None = None,
    reference_snapshot: pl.DataFrame | None = None,
    **kwargs: str | ToolTraceExtractorType,
) -> float:
    """Measures the similarity between each database snapshot and a target_dataframe, if the snapshot is supposed to
    be derived from removing target_dataframe from reference_snapshot

        This can be implemented by swapping snapshot, reference_snapshot and calling addition_similarity

    Args:
        snapshot:                   The dataframe snapshot to calculate similarity for
        target_dataframe:           The dataframe to calculate similarity with
        column_similarities:        A dictionary of column name to column-wise similarity measure
        reference_snapshot:         When similarity is 1,
                                    snapshot should be the result of removing target_dataframe from reference_snapshot

    Returns:
        A [0, 1] similarity score between target_dataframe and snapshot
    """
    assert (
        target_dataframe is not None
        and column_similarities is not None
        and reference_snapshot is not None
    )
    return addition_similarity(
        snapshot=reference_snapshot,
        target_dataframe=target_dataframe,
        column_similarities=column_similarities,
        reference_snapshot=snapshot,
    )


def update_similarity(
    snapshot: pl.DataFrame,
    target_dataframe: pl.DataFrame | None = None,
    column_similarities: dict[str, ColumnSimilarityMeasureType] | None = None,
    reference_snapshot: pl.DataFrame | None = None,
    **kwargs: str | ToolTraceExtractorType,
) -> float:
    """Measures the similarity between each database snapshot and a target_dataframe, if the snapshot is supposed to
    be derived from updating the same of number entries from reference_snapshot into target_dataframe

        1. If snapshot and reference_snapshot doesn't match in row count, similarity is always 0
        2. If the number of different rows between snapshot and reference_snapshot doesn't match target_dataframe
            similarity is always 0
        2. If it does, the anti join of the two is compared against target_dataframe using snapshot_similarity

    Args:
        snapshot:                   The dataframe snapshot to calculate similarity for
        target_dataframe:           The dataframe to calculate similarity with
        column_similarities:        A dictionary of column name to column-wise similarity measure
        reference_snapshot:         When similarity is 1,
                                    snapshot should be the result of adding target_dataframe into reference_snapshot

    Returns:
        A [0, 1] similarity score between target_dataframe and snapshot
    """
    assert (
        target_dataframe is not None
        and column_similarities is not None
        and reference_snapshot is not None
    )
    # Drop sandbox_message_index
    # Fill null with zero to prevent join failure
    snapshot = _fill_null(snapshot.drop("sandbox_message_index"))
    reference_snapshot = _fill_null(reference_snapshot.drop("sandbox_message_index"))
    target_dataframe = _fill_null(target_dataframe)
    if (
        reference_snapshot.select(pl.len())["len"][0]
        != snapshot.select(pl.len())["len"][0]
    ):
        return 0
    return snapshot_similarity(
        snapshot.join(reference_snapshot, on=snapshot.columns, how="anti"),
        target_dataframe=target_dataframe,
        column_similarities=column_similarities,
        reference_snapshot=reference_snapshot,
    )


def tool_trace_dependant_similarity(
    snapshot: pl.DataFrame,
    target_dataframe: pl.DataFrame | None = None,
    column_similarities: dict[str, ColumnSimilarityMeasureType] | None = None,
    reference_snapshot: pl.DataFrame | None = None,
    **kwargs: str | ToolTraceExtractorType,
) -> float:
    """A special similarity only intended to be used for SANDBOX database. Allows one to extract values
    from the tool_trace in reference_snapshot, and fill into target_dataframe. Extractors are allowed to return
    multiple "normalized" version of extracted value, a similarity will be calculated for each, and return the max.

    Args:
        snapshot:               The dataframe snapshot to calculate similarity for
        target_dataframe:       The dataframe to calculate similarity with. Either content or tool_trace column is
                                incomplete, requires value extracted from reference_snapshot to fill in
        column_similarities:    A dictionary of column name to column-wise similarity measure
        reference_snapshot:     Contains tool_trace we wish to extract value from
        **kwargs:               Should contain keyword argument "fill_to", indicating which column to fill extracted
                                value into. Can only choose from Literal["tool_trace", "content"].
                                    - In the case of "tool_trace", extracted values are supplied as kwargs into tool
                                        trace arguments
                                    - In the case of "content", extracted values are supplied to str.format
                                Should contain keyword argument "extractor" of type ToolTraceExtractorType.
                                An extractor function taking 1 tool trace as input, and returns a list of dictionary,
                                containing multiple normalized form of extracted values.
                                Each normalized extracted value will be provided to target_dataframe,
                                calculate similarity, and the max of all normalized forms is taken as final similarity.

    Returns:
        A [0, 1] similarity score between target_dataframe and snapshot.
    """
    assert (
        target_dataframe is not None
        and column_similarities is not None
        and reference_snapshot is not None
    )
    # Check schema to make sure we are working with SANDBOX database. Allows sandbox_message_index to be dropped
    for current_snapshot in (snapshot, reference_snapshot):
        schema = {**current_snapshot.schema}
        # Add sandbox_message_index if dropped
        schema.update(
            {
                "sandbox_message_index": ExecutionContext.dbs_schemas[
                    DatabaseNamespace.SANDBOX
                ]["sandbox_message_index"]
            }
        )
        if (
            schema != ExecutionContext.dbs_schemas[DatabaseNamespace.SANDBOX]
            or current_snapshot.select(pl.len())["len"][0] != 1
        ):
            raise SchemaError(
                "tool_trace_dependant_similarity can only be used with SANDBOX database with only 1 row"
            )
    # Check kwargs
    if "fill_to" not in kwargs:
        raise KeyError(
            "fill_to kwarg of type Literal['tool_trace', 'content'] "
            "must be provided to tool_trace_dependant_similarity. "
        )
    fill_to = cast(Literal["tool_trace", "content"], kwargs["fill_to"])
    if fill_to not in ("tool_trace", "content"):
        raise ValueError("fill_to must be of type Literal['tool_trace', 'content']")
    if "extractor" not in kwargs:
        raise KeyError(
            "extractor kwarg of type ToolTraceExtractorType "
            "must be provided to tool_trace_dependant_similarity. "
        )
    extractor = cast(ToolTraceExtractorType, kwargs["extractor"])
    # Start extraction
    if reference_snapshot["tool_trace"][0] is None:
        return 0
    tool_traces = cast(
        list[dict[Literal["tool_name", "arguments", "result"], Any]],
        [json.loads(x) for x in reference_snapshot["tool_trace"][0]],
    )
    # Extract values from all tool traces in snapshot
    extracted_values: list[dict[str, Any]] = []
    for tool_trace in tool_traces:
        try:
            extracted_values.extend(extractor(tool_trace))
        except (KeyError, IndexError, TypeError, ValueError):
            pass
    # Find the best possible matches
    similarity: float = 0.0
    try:
        for extracted_value in extracted_values:
            # When filling in extracted_value, we consider the following options:
            #   1. Any extracted_value can fill in any trace
            #   2. If an extracted_value overrides an existing kwarg in the trace,
            #       consider the existing kwarg as well.
            if fill_to == "tool_trace":
                trace: dict[str, Any] | list[dict[str, Any]] = json.loads(
                    target_dataframe["tool_trace"][0]
                )
                # Normalize into a list of possible candidate traces
                candidate_traces: list[dict[str, Any]] = (
                    [trace] if not isinstance(trace, Sequence) else list(trace)
                )
                filled_traces: list[dict[str, Any]] = []
                # Filling extracted value to all candidate traces
                for candidate_trace in candidate_traces:
                    # Prefer arguments in extracted_value
                    current_trace = deepcopy(candidate_trace)
                    current_trace["arguments"].update(extracted_value)
                    filled_traces.append(current_trace)
                    # Prefer arguments in candidate_trace
                    current_trace = deepcopy(candidate_trace)
                    extracted_arguments = deepcopy(extracted_value)
                    extracted_arguments.update(current_trace["arguments"])
                    current_trace["arguments"] = extracted_arguments
                    filled_traces.append(current_trace)
                similarity = max(
                    similarity,
                    snapshot_similarity(
                        snapshot=snapshot,
                        target_dataframe=target_dataframe.with_columns(
                            pl.lit(json.dumps(filled_traces, ensure_ascii=False)).alias(
                                "tool_trace"
                            )
                        ),
                        column_similarities=column_similarities,
                        reference_snapshot=reference_snapshot,
                    ),
                )
            elif fill_to == "content":
                candidate_content = cast(str, target_dataframe["content"][0])
                candidate_content = candidate_content.format(**extracted_value)
                similarity = max(
                    similarity,
                    snapshot_similarity(
                        snapshot=snapshot,
                        target_dataframe=target_dataframe.with_columns(
                            pl.lit(candidate_content).alias("content")
                        ),
                        column_similarities=column_similarities,
                        reference_snapshot=reference_snapshot,
                    ),
                )
    except (IndexError, KeyError):
        return 0.0
    return similarity


def guardrail_similarity(
    snapshot: pl.DataFrame,
    target_dataframe: pl.DataFrame | None = None,
    column_similarities: dict[str, ColumnSimilarityMeasureType] | None = None,
    reference_snapshot: pl.DataFrame | None = None,
    **kwargs: str | ToolTraceExtractorType,
) -> float:
    """Similarity which ensures snapshot is identical to reference. Returns 0 otherwise

    Args:
        snapshot:                   The dataframe snapshot to calculate similarity for
        target_dataframe:           Not utilized by this similarity
        column_similarities:        Not utilized by this similarity
        reference_snapshot:         When similarity is 1,
                                    snapshot identical to reference_snapshot

    Returns:
        A [0, 1] similarity score between target_dataframe and snapshot
    """
    assert reference_snapshot is not None
    return float(snapshot.equals(reference_snapshot))


# Default similarity measures for each column
_default_dbs_column_similarities: dict[str, dict[str, ColumnSimilarityMeasureType]] = {
    DatabaseNamespace.SANDBOX: {
        "sandbox_message_index": column_exact_match_similarity,
        "sender": column_exact_match_similarity,
        "recipient": column_exact_match_similarity,
        "content": column_rouge_l_similarity,
        "openai_tool_call_id": column_one_similarity,
        "openai_function_name": column_one_similarity,
        "conversation_active": column_exact_match_similarity,
        "tool_call_exception": column_one_similarity,
        "tool_trace": column_tool_trace_exact_match_similarity,
        "visible_to": column_exact_match_similarity,
        "finish_reason": column_exact_match_similarity,
        "logprobs": column_exact_match_similarity,
        "generation": column_exact_match_similarity,
        "token_ids": column_exact_match_similarity,
        "claude_text_response": column_exact_match_similarity,
        "claude_extended_thinking": column_exact_match_similarity,
        "claude_extended_thinking_signature": column_exact_match_similarity,
        "tool_call_text_response": column_exact_match_similarity,
        "image_ids": column_exact_match_similarity,
        "reasoning_trace": column_one_similarity,
        "openai_reasoning_content": column_one_similarity,
        "openai_reasoning_items": column_one_similarity,
    },
    DatabaseNamespace.IMAGE: {
        "sandbox_message_index": column_exact_match_similarity,
        "image_id": column_exact_match_similarity,
        "image_content": column_exact_match_similarity,  # Base64 encoded image data
    },
    DatabaseNamespace.SETTING: {
        "sandbox_message_index": column_exact_match_similarity,
        "device_id": column_exact_match_similarity,
        "cellular": column_exact_match_similarity,
        "wifi": column_exact_match_similarity,
        "location_service": column_exact_match_similarity,
        "low_battery_mode": column_exact_match_similarity,
        "latitude": column_exact_match_similarity,
        "longitude": column_exact_match_similarity,
        "place_id": column_exact_match_similarity,
        "formatted_address": column_rouge_l_similarity,
        "utc_offset_seconds": column_exact_match_similarity,
    },
    DatabaseNamespace.CONTACT: {
        "sandbox_message_index": column_exact_match_similarity,
        "person_id": column_exact_match_similarity,
        "name": column_exact_match_similarity,
        "phone_number": column_exact_match_similarity,
        "relationship": column_rouge_l_similarity,
        "is_self": column_exact_match_similarity,
    },
    DatabaseNamespace.MESSAGING: {
        "sandbox_message_index": column_exact_match_similarity,
        "message_id": column_exact_match_similarity,
        "sender_person_id": column_exact_match_similarity,
        "sender_phone_number": column_exact_match_similarity,
        "recipient_person_id": column_exact_match_similarity,
        "recipient_phone_number": column_exact_match_similarity,
        "content": column_rouge_l_similarity,
        "creation_timestamp": column_exact_match_similarity,
        "image_ids": column_exact_match_similarity,
    },
    DatabaseNamespace.REMINDER: {
        "sandbox_message_index": column_exact_match_similarity,
        "reminder_id": column_exact_match_similarity,
        "content": column_rouge_l_similarity,
        "creation_datetime": column_exact_match_similarity,
        "reminder_datetime": column_datetime_naive_equal_similarity,
        "latitude": column_exact_match_similarity,
        "longitude": column_exact_match_similarity,
    },
    DatabaseNamespace.NOTES: {
        "note_id": column_exact_match_similarity,
        "content": column_rouge_l_similarity,
        "modification_timestamp": column_exact_match_similarity,
        "image_ids": column_exact_match_similarity,
    },
    DatabaseNamespace.CALENDARS: {
        "sandbox_message_index": column_exact_match_similarity,
        "calendar_id": column_exact_match_similarity,
        "title": column_rouge_l_similarity,
    },
    DatabaseNamespace.CALENDAR_EVENTS: {
        "sandbox_message_index": column_exact_match_similarity,
        "calendar_event_id": column_exact_match_similarity,
        "calendar_id": column_exact_match_similarity,
        "title": column_rouge_l_similarity,
        "description": column_rouge_l_similarity,
        "start_datetime": column_datetime_naive_equal_similarity,
        "end_datetime": column_datetime_naive_equal_similarity,
        "is_all_day": column_exact_match_similarity,
        "recurrence_frequency": column_exact_match_similarity,
        "recurrence_interval": column_exact_match_similarity,
        "recurrence_until_datetime": column_exact_match_similarity,
        "recurrence_parent_id": column_exact_match_similarity,
        "recurrence_exclude_datetime": column_exact_match_similarity,
        "latitude": column_exact_match_similarity,
        "longitude": column_exact_match_similarity,
        "place_id": column_exact_match_similarity,
        "formatted_address": column_exact_match_similarity,
        "attendee_names": column_exact_match_similarity,
        "attendee_emails": column_exact_match_similarity,
    },
}
_default_dbs_column_similarities[
    DatabaseNamespace.CALENDAR_EVENTS_RECURRENCE_STAGING
] = _default_dbs_column_similarities[DatabaseNamespace.CALENDAR_EVENTS]


def get_effective_turn_count(sandbox_database: pl.DataFrame) -> int:
    """Calculate effective turn count.

    Excludes infrastructure messages:
        1. System messages
        2. User ↔ ExecutionEnvironment
        3. Messages with ``visible_to=[RoleType.USER]`` (internal-only)

    Args:
        sandbox_database:  Sandbox database

    Returns:
        Effective count.
    """
    # 1. System messages
    # 2. User ↔ ExecutionEnvironment
    # 3. Internal messages visible only to user role
    system_message_filter = pl.col("sender") != RoleType.SYSTEM
    user_exec_env_filter = ~(
        (
            (pl.col("sender") == RoleType.USER)
            & (pl.col("recipient") == RoleType.EXECUTION_ENVIRONMENT)
        )
        | (
            (pl.col("sender") == RoleType.EXECUTION_ENVIRONMENT)
            & (pl.col("recipient") == RoleType.USER)
        )
    )
    internal_only_filter = pl.col("visible_to") != [RoleType.USER]
    filtered_df = sandbox_database.filter(
        system_message_filter & user_exec_env_filter & internal_only_filter
    )
    if filtered_df.is_empty():
        return 0
    return cast(
        int,
        filtered_df.with_columns(pl.len())["len"][0],
    )


@define
class EvaluationResult:
    """Contains evaluation results: entity diff similarity, judge results, and turn count.

    Attributes:
        similarity: Combined F1 score computed from entity diff precision and
            recall.  Set automatically in ``__attrs_post_init__``.
        turn_count: Total turn count excluding system messages.
        judge_result: Result dictionary from the LLM judge model, or None.
        task_completion_criteria: The criteria string used for the judge model.
        entity_diff_result: Entity diff evaluation result with precision, recall,
            and guardrail pass/fail, or None.
        ui_judge_result: Result from the UI quality judge, or None.
        user_judge_result: Result from the user simulator quality judge, or None.
    """

    # Combined similarity score (entity diff F1, set in __attrs_post_init__)
    similarity: float = field(init=False)
    # Total turn count, excluding system messages
    turn_count: int = 0
    # Result from the judge model
    judge_result: dict[str, Any] | None = None
    # The criteria used for the judge model
    task_completion_criteria: str | None = None
    # Entity diff evaluation result (precision/recall/guardrails)
    entity_diff_result: dict[str, Any] | None = None
    # Result from the UI quality judge (separate from judge_result)
    ui_judge_result: dict[str, Any] | None = None
    # Result from the user simulator quality judge
    user_judge_result: dict[str, Any] | None = None

    def __attrs_post_init__(self) -> None:
        # Similarity is computed from entity diff F1
        if self.entity_diff_result is not None:
            p = self.entity_diff_result.get("overall_precision", 0.0)
            r = self.entity_diff_result.get("overall_recall", 0.0)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            if not self.entity_diff_result.get("guardrail_pass", True):
                f1 = 0.0
            self.similarity = f1
        else:
            self.similarity = 0.0


@define
class EvaluationCriteria:
    """The evaluation criteria for a scenario.

    Attributes:
        task_completion_criteria: Natural-language criteria string for the LLM
            judge, or None to skip judge evaluation.
        entity_diff_evaluator: Optional ``EntityDiffEvaluator`` for unified
            before/after state comparison.  Entity diff F1 is used as the
            primary similarity score.
        enable_ui: When True, the judge appends UI quality rubric dimensions
            (4 extra criteria) to evaluate agent-generated UI screens.
    """

    task_completion_criteria: str | None = None
    # Optional entity diff evaluator for unified before/after state comparison.
    # Entity diff F1 is used as the primary similarity score.
    entity_diff_evaluator: Any | None = None
    # When True, the judge appends UI quality rubric dimensions (4 extra criteria)
    # to evaluate the quality of agent-generated UI screens.
    enable_ui: bool = False


def _save_judge_evidence(path: Path, evidence: list[dict[str, Any]]) -> None:
    """Save judge evidence parts to a JSON file.

    Images are stored as placeholders to keep file sizes manageable (the full
    base64 images live in execution_context.json). The placeholder embeds the
    image's id — read from the preceding ``[image_id=N]`` label part — so a
    reader can restore images by identity rather than by position.
    """
    serializable = []
    last_image_id: int | None = None
    for part in evidence:
        if part.get("type") == "text":
            match = re.fullmatch(r"\[image_id=(\d+)\]", part.get("text", "").strip())
            last_image_id = int(match.group(1)) if match else None
            serializable.append(part)
        elif part.get("type") == "image_url":
            # Store an id-tagged placeholder instead of the full base64 blob.
            if last_image_id is not None:
                url = f"(image_id={last_image_id} saved in execution_context.json)"
            else:
                url = "(saved in execution_context.json)"
            serializable.append({"type": "image_url", "image_url": {"url": url}})
            last_image_id = None
        else:
            serializable.append(part)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def evaluate(
    evaluation_criteria: EvaluationCriteria,
    execution_context: ExecutionContext,
    max_turn_count: int,
    judge_name: str | None = None,
    output_directory: Path | None = None,
    scenario_metadata: dict[str, Any] | None = None,
) -> EvaluationResult:
    """Evaluate a completed ExecutionContext.

    Runs entity diff evaluation and LLM judge, then combines results.

    Args:
        evaluation_criteria:    The criteria defining how to evaluate the ExecutionContext.
        execution_context:      ExecutionContext to be evaluated.
        max_turn_count:         Maximum number of turns allowed.
        judge_name:             Judge class name (from _JUDGE_REGISTRY). If None, uses default.
        output_directory:       Optional directory to save judge evidence for later replay.
        scenario_metadata:      Optional scenario metadata for the user judge
                                (challenge_type, require_disambiguation, etc.).

    Returns:
        An EvaluationResult object containing similarity scores.
    """
    # Calculate turn count
    turn_count = get_effective_turn_count(
        execution_context.get_database(
            DatabaseNamespace.SANDBOX, get_all_history_snapshots=True
        )
    )

    judge_result = None
    ui_judge_result = None
    user_judge_result = None

    entity_diff_result = None
    if evaluation_criteria.entity_diff_evaluator is not None:
        try:
            diff_result = evaluation_criteria.entity_diff_evaluator.evaluate(
                execution_context=execution_context,
            )
            entity_diff_result = diff_result.to_dict()
            LOGGER.info(
                "Entity diff eval: precision=%.3f recall=%.3f guardrail=%s",
                diff_result.overall_precision,
                diff_result.overall_recall,
                diff_result.guardrail_pass,
            )
            if output_directory is not None:
                evaluation_criteria.entity_diff_evaluator.save_evidence(
                    output_directory / "entity_diff_evidence.json"
                )
        except Exception as e:
            LOGGER.warning("Entity diff evaluation failed: %s", e)
    else:
        LOGGER.debug("No entity_diff_evaluator set — skipping entity diff evaluation")

    if evaluation_criteria.task_completion_criteria:
        # Avoid circular import.
        from mmtoolsandbox.roles.base_judge import get_judge_instance

        if judge_name is None:
            raise ValueError(
                "judge_name is required when task_completion_criteria is set. "
                "Pass --judge on the CLI or provide judge_name explicitly."
            )
        judge = get_judge_instance(judge_name)

        # Pass AppWorld initial state to the judge so the "Database Changes"
        # section includes AppWorld diffs (not just MMToolSandbox namespaces).
        appworld_initial = None
        if evaluation_criteria.entity_diff_evaluator is not None:
            appworld_initial = (
                evaluation_criteria.entity_diff_evaluator._initial_appworld
            )

        # Standard 5-criterion judge (always runs when criteria is set)
        judge_result = judge.evaluate(
            execution_context=execution_context,
            criteria=evaluation_criteria.task_completion_criteria,
            appworld_initial=appworld_initial,
        )
        # Separate UI quality judge (only when enable_ui is set)
        if evaluation_criteria.enable_ui:
            ui_judge_result = judge.evaluate_ui(
                execution_context=execution_context,
                criteria=evaluation_criteria.task_completion_criteria,
            )

        # User simulator quality judge (always runs when criteria is set)
        user_judge_result = judge.evaluate_user(
            execution_context=execution_context,
            max_messages=max_turn_count,
            turn_count=turn_count,
            scenario_metadata=scenario_metadata,
            enable_ui=evaluation_criteria.enable_ui,
        )

        # Save judge evidence for offline replay / backfill
        if output_directory is not None and hasattr(judge, "last_evidence"):
            evidence_dir = output_directory / "judge_evidence"
            evidence_dir.mkdir(exist_ok=True, parents=True)
            for key, evidence in judge.last_evidence.items():
                _save_judge_evidence(evidence_dir / f"{key}.json", evidence)

    return EvaluationResult(
        turn_count=turn_count,
        judge_result=judge_result,
        task_completion_criteria=evaluation_criteria.task_completion_criteria,
        entity_diff_result=entity_diff_result,
        ui_judge_result=ui_judge_result,
        user_judge_result=user_judge_result,
    )
