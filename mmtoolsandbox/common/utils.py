# Copyright © 2026 Apple Inc.

from __future__ import annotations

import bisect
import datetime
import functools
import inspect
import json
import logging
import uuid
import warnings
from contextlib import contextmanager
from copy import deepcopy
from inspect import getsource, isfunction, signature
from typing import (
    Any,
    Callable,
    Iterator,
    Literal,
    TypeVar,
)

import numpy as np
import numpy.typing as npt
import polars as pl
from dateutil._common import weekday
from dateutil.relativedelta import relativedelta
from decorator import decorate
from rapidfuzz import fuzz, process, utils
from strenum import StrEnum
from typing_extensions import Protocol

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution_context import (
    RoleType,
    get_current_context,
    new_context_with_attribute,
)
from mmtoolsandbox.common.i18n import Locale
from mmtoolsandbox.common.json_utils import MMToolSandboxJSONEncoder
from mmtoolsandbox.common.tool_registry import register_tool
from mmtoolsandbox.toolbox.names import ToolboxName

T = TypeVar("T")

LOGGER = logging.getLogger(__name__)


# Taken from openai._types
# Sentinel class used until PEP 0661 is accepted
class NotGiven:
    """
    A sentinel singleton class used to distinguish omitted keyword arguments
    from those passed in with the value None (which may have different behavior).

    For example:

    ```py
    def search(
        a: Union[int, NotGiven, None] = NotGiven(),
        b: Union[int, NotGiven, None] = NotGiven(),
    ): ...


    search(a=1, b=2)  # Search in database with constraint a == 1, b==2
    search(a=None, b=3)  # Search in database with constraint a == None, b==3
    search(b=4)  # Search in database with constraint b==4
    ```
    """

    def __bool__(self) -> Literal[False]:
        return False

    def __repr__(self) -> str:
        return "NOT_GIVEN"


NOT_GIVEN = NotGiven()


class DataframeFilterMethodType(Protocol):
    """Callable type def for Dataframe filtering functions."""

    def __call__(
        self,
        dataframe: pl.DataFrame,
        column_name: str,
        value: Any,
        **kwargs: int | Any,
    ) -> pl.DataFrame: ...


def exact_match_filter_dataframe(
    dataframe: pl.DataFrame, column_name: str, value: Any, **kwargs: int | Any
) -> pl.DataFrame:
    """Filter dataframe by exact matching value on 1 column

    Args:
    ----
        dataframe:      Dataframe to filter
        column_name:    Name of column
        value:          Value to match against

    Returns:
    -------
        Filtered dataframe

    """
    return dataframe.filter(pl.col(column_name) == value)


def exact_match_list_fully_contains_filter_dataframe(
    dataframe: pl.DataFrame, column_name: str, value: Any, **kwargs: int | Any
) -> pl.DataFrame:
    """Filter dataframe by exact matching a list value on 1 column of lists.

    Only when a list fully contains value, do we consider it a match

    Args:
    ----
        dataframe:      Dataframe to filter
        column_name:    Name of column
        value:          Value to match against

    Returns:
    -------
        Filtered dataframe

    """
    return dataframe.filter(
        pl.col(column_name).list.set_intersection(value).list.len() == len(value)
    )


def embedding_list_contains_filter_dataframe(
    dataframe: pl.DataFrame, column_name: str, value: Any, **kwargs: int | Any
) -> pl.DataFrame:
    """Filter dataframe by thresholding embedding similarity for string value list on 1 list column.

    All items in value list calculate similarity scores for all items in list column.
    All items in value list must pass threshold for a match to happen.

    Args:
    ----
        dataframe:      Dataframe to filter
        column_name:    Name of column
        value:          Value to match against. Should be a list
        **kwargs:       Additional kwargs for matching, for example
                        threshold. only entries above the threshold are selected

    Returns:
    -------
        Filtered dataframe
    """
    # We delay this import since it pulls in `sentence_transformers`, which is slow.
    from mmtoolsandbox.common.cached_embedding_service import (
        batched_gte_multilingual_base_embedding,
    )

    threshold = kwargs.get("threshold", 0.7)
    assert 0 <= threshold <= 1, (
        "threshold should be within [0, 1], for embedding_list_contains_filter_dataframe"
    )
    assert isinstance(value, list), (
        "value must be a list for embedding_list_contains_filter_dataframe"
    )
    # Calculate embeddings for each column items
    column: list[list[str]] = [
        x if x is not None else [] for x in dataframe.get_column(column_name).to_list()
    ]
    # Gather embedding for unique strings
    string_to_embedding: dict[str, npt.NDArray[np.float64]] = (
        batched_gte_multilingual_base_embedding(
            queries=list({x for row in column for x in row})
        )
    )
    # If no entries were found, return empty dataframe.
    if not string_to_embedding:
        return pl.DataFrame(schema=dataframe.schema)
    # Construct padded embedding matrix of size len(column) * max(len(list)) * embedding_dim
    # Use 0 padding
    padding = np.zeros_like(list(string_to_embedding.values())[0])
    max_list_len = max(len(x) for x in column)
    column_embeddings: npt.NDArray[np.float64] = np.array(
        [
            [string_to_embedding[x] for x in row]
            + [padding] * (max_list_len - len(row))
            for row in column
        ]
    )
    query_string_to_embedding: dict[str, npt.NDArray[np.float64]] = (
        batched_gte_multilingual_base_embedding(queries=value)
    )
    # len(value) * embedding_dim array representing embedding for each item in value
    query_embedding: npt.NDArray[np.float64] = np.array(
        [query_string_to_embedding[item] for item in value]
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"invalid value encountered in divide")
        # Find cosine similarity. Resulting in len(column) * max(len(list)) * len(value)
        cosine_similarity = np.nan_to_num(
            column_embeddings
            @ query_embedding.T
            / (
                np.linalg.norm(column_embeddings, axis=-1, keepdims=True)
                @ np.linalg.norm(query_embedding.T, axis=0, keepdims=True)
            ),
            nan=0,
        )
    # Pick best value item match among list items, len(column) * len(value)
    cosine_similarity = np.max(cosine_similarity, axis=1)
    # Nullify any row with at least one similarity below threshold. Calculate average similarity for the rest
    cosine_similarity = np.mean(
        np.min(
            (cosine_similarity > threshold).astype(np.float64),
            axis=-1,
            keepdims=True,
        )
        * cosine_similarity,
        axis=-1,
    )
    # Create similarity & index pairs, sort by ascending order (cus bisect only works with ascending)
    sim_and_indices: list[tuple[float, int]] = sorted(
        list(zip(cosine_similarity.tolist(), range(cosine_similarity.shape[0]))),
    )
    # Find entries above threshold, reverse order, return
    indices = [
        x[1]
        for x in sim_and_indices[
            bisect.bisect(sim_and_indices, (threshold, len(sim_and_indices))) :
        ]
    ]
    return dataframe[list(reversed(indices))]


def exact_match_struct_field_filter_dataframe(
    dataframe: pl.DataFrame,
    column_name: str,
    struct_field_name: str,
    value: Any,
    **kwargs: int | Any,
) -> pl.DataFrame:
    """Filter dataframe by exact matching string value on 1 struct field of 1 column.

    The column has to contain a struct.

    Args:
    ----
        dataframe:         Dataframe to filter
        column_name:       Name of column
        struct_field_name: Name of the struct field to match.
        value:             Value to match against

    Returns:
    -------
        Filtered dataframe
    """
    return dataframe.filter(
        dataframe[column_name].struct.field(struct_field_name) == value
    )


def subsequence_filter_dataframe(
    dataframe: pl.DataFrame, column_name: str, value: Any, **kwargs: int | Any
) -> pl.DataFrame:
    """Filter dataframe for rows that contains value as a subsequence in column_name

    Args:
    ----
        dataframe:      Dataframe to filter
        column_name:    Name of column
        value:          Value to match against

    Returns:
    -------
        Filtered dataframe

    """
    return dataframe.filter(pl.col(column_name).str.contains(str(value)))


def range_filter_dataframe(
    dataframe: pl.DataFrame, column_name: str, value: Any, **kwargs: int | Any
) -> pl.DataFrame:
    """Filter dataframe for rows whose column_name value fits in a range. The value must implement __lt__ __gt__ __eq__
        __add__. Boundary included.

    Args:
    ----
        dataframe:      Dataframe to filter
        column_name:    Name of column
        value:          Lower or upperbound of the range
        **kwargs:       Must contain value_delta, value + value_delta provides the other end of the bound

    Returns:
    -------
        Filtered dataframe

    """
    try:
        value_delta = kwargs.get("value_delta")
    except KeyError:
        raise KeyError("**kwargs must contain 'value_delta'")
    lower_bound, upperbound = (
        min(value, value + value_delta),
        max(value, value + value_delta),
    )
    return dataframe.filter(
        (pl.col(column_name) >= lower_bound) & (pl.col(column_name) <= upperbound)
    )


def lt_eq_filter_dataframe(
    dataframe: pl.DataFrame, column_name: str, value: Any, **kwargs: int | Any
) -> pl.DataFrame:
    """Filter dataframe for rows whose column_name are less than or equal to value. The value must implement __lt__.

    Args:
    ----
        dataframe:      Dataframe to filter
        column_name:    Name of column
        value:          Upperbound of column value

    Returns:
    -------
        Filtered dataframe

    """
    return dataframe.filter(pl.col(column_name) <= value)


def gt_eq_filter_dataframe(
    dataframe: pl.DataFrame, column_name: str, value: Any, **kwargs: int | Any
) -> pl.DataFrame:
    """Filter dataframe for rows whose column_name are greater than or equal to value. The value must implement __gt__.

    Args:
    ----
        dataframe:      Dataframe to filter
        column_name:    Name of column
        value:          Lowerbound of column value

    Returns:
    -------
        Filtered dataframe

    """
    return dataframe.filter(pl.col(column_name) >= value)


def fuzzy_match_filter_dataframe(
    dataframe: pl.DataFrame, column_name: str, value: Any, **kwargs: int | Any
) -> pl.DataFrame:
    """Filter dataframe by fuzzy matching string value on 1 column. Backed by fuzz.WRatio

    Args:
    ----
        dataframe:      Dataframe to filter
        column_name:    Name of column
        value:          Value to match against
        **kwargs:       Additional kwargs for matching, for example
                        threshold for fuzz.WRatio, only entries above the threshold are selected

    Returns:
    -------
        Filtered dataframe
    """
    threshold = kwargs.get("threshold", 90)
    # Find the indices of top scored rows, process.extract returns a tuple of (string, score, index)
    matches = process.extract(
        query=value,
        choices=dataframe.get_column(column_name),
        processor=utils.default_process,
        scorer=fuzz.WRatio,
        score_cutoff=threshold,
        # The limit is hard-coded to 5 in the `rapidfuzz` library, but we want all matches above the given threshold. So
        # the upper bound of the number of matches is the length of the dataframe.
        limit=len(dataframe),
    )
    return dataframe[[x[-1] for x in matches]]


def fuzzy_match_struct_field_filter_dataframe(
    dataframe: pl.DataFrame,
    column_name: str,
    struct_field_name: str,
    value: Any,
    **kwargs: int | Any,
) -> pl.DataFrame:
    """Filter dataframe by fuzzy matching string value on 1 struct field of 1 column. Backed by fuzz.WRatio

    The column has to contain a struct.

    Args:
    ----
        dataframe:         Dataframe to filter
        column_name:       Name of column
        struct_field_name: Name of the struct field to match.
        value:             Value to match against
        **kwargs:          Additional kwargs for matching, for example
                           threshold for fuzz.WRatio, only entries above the threshold are selected

    Returns:
    -------
        Filtered dataframe
    """
    threshold = kwargs.get("threshold", 90)
    # Find the indices of top scored rows, process.extract returns a tuple of (string, score, index)
    matches = process.extract(
        query=value,
        choices=[
            entry[struct_field_name] for entry in dataframe.get_column(column_name)
        ],
        processor=utils.default_process,
        scorer=fuzz.WRatio,
        score_cutoff=threshold,
        # The limit is hard-coded to 5 in the `rapidfuzz` library, but we want all matches above the given threshold. So
        # the upper bound of the number of matches is the length of the dataframe.
        limit=len(dataframe),
    )
    return dataframe[[x[-1] for x in matches]]


def embedding_filter_dataframe(
    dataframe: pl.DataFrame, column_name: str, value: Any, **kwargs: int | Any
) -> pl.DataFrame:
    """Filter dataframe by thresholding embedding similarity for string value on 1 column.

    Returned dataframe is sorted by decreasing embedding similarity.

    Args:
    ----
        dataframe:      Dataframe to filter
        column_name:    Name of column
        value:          Value to match against
        **kwargs:       Additional kwargs for matching, for example
                        threshold. only entries above the threshold are selected

    Returns:
    -------
        Filtered dataframe
    """
    # We delay this import since it pulls in `sentence_transformers`, which is slow.
    from mmtoolsandbox.common.cached_embedding_service import (
        batched_gte_multilingual_base_embedding,
    )

    threshold = kwargs.get("threshold", 0.7)
    assert 0 <= threshold <= 1, (
        "threshold should be within [0, 1], for embedding_filter_dataframe"
    )
    column = dataframe.get_column(column_name).to_list()
    if not column:
        return pl.DataFrame(schema=dataframe.schema)
    # Calculate embeddings
    string_to_embedding: dict[str, npt.NDArray[np.float64]] = (
        batched_gte_multilingual_base_embedding(queries=column)
    )
    column_embeddings: npt.NDArray[np.float64] = np.array(
        [
            string_to_embedding[column_value]
            for column_value in dataframe.get_column(column_name)
        ]
    )
    query_embedding: npt.NDArray[np.float64] = batched_gte_multilingual_base_embedding(
        queries=[value]
    )[value]
    # Find cosine similarity
    cosine_similarity = (
        column_embeddings
        @ query_embedding
        / (
            np.linalg.norm(column_embeddings, axis=-1)
            * np.linalg.norm(query_embedding, axis=-1)
        )
    )
    # Create similarity & index pairs, sort by ascending order (cus bisect only works with ascending)
    sim_and_indices: list[tuple[float, int]] = sorted(
        list(zip(cosine_similarity.tolist(), range(cosine_similarity.shape[0]))),
    )
    # Find entries above threshold, reverse order, return
    indices = [
        x[1]
        for x in sim_and_indices[
            bisect.bisect(sim_and_indices, (threshold, len(sim_and_indices))) :
        ]
    ]
    return dataframe[list(reversed(indices))]


def filter_dataframe(
    dataframe: pl.DataFrame,
    filter_criteria: list[tuple[str, Any, DataframeFilterMethodType]],
) -> pl.DataFrame:
    """Filter dataframe given a filter method, column name and target value

    Args:
        dataframe:          Dataframe to filter
        filter_criteria:    A list of filter constraints on each column,
                                each tuple contains [column_name, value, filter_method]

    Returns:
        Filtered dataframe
    """
    if all(x[1] is NOT_GIVEN for x in filter_criteria):
        raise ValueError(
            f"No search criteria are given. At least one search criteria should be provided among "
            f"{tuple(x[0] for x in filter_criteria)}"
        )
    for column_name, filter_value, filter_method in filter_criteria:
        if filter_value is not NOT_GIVEN:
            dataframe = filter_method(
                dataframe=dataframe,
                column_name=column_name,
                value=filter_value,
            )
    return dataframe


def polars_multiply_columns_expression(column_names: list[str]) -> pl.Expr:
    """Returns a polars expression that multiplies all columns in column_names

    Args:
        column_names:   Columns to multiply

    Returns:
        A polars expression that multiplies all columns in column_names
    """
    expression = pl.col(column_names[0])
    for column_name in column_names[1:]:
        expression = expression.mul(pl.col(column_name))
    return expression


def get_outer_tool_names() -> list[str]:
    """Return the name of all outer frames that's registered by register_as_tool in order.

    This function makes a few key assumptions:
    0. We are working with a python implementation that has support for sys._get_frame()
        0.1 Should always be the case of CPython.
    1. Tools are identifiable through is_tool attribute.
        1.1 As long as register_as_tool doesn't change this should always apply.
    2. If there is any outer frame functions / tool, it exists in the current frame f_globals.
        2.1 As long as no tools are defined as nested functions / class methods, this should always apply.

    In a call stack of depth N, calling this func on all tool in the stack means O(N^2) complexity.

    Returns:
        The name of the nearest outer frame tool if any, otherwise None.
    """
    outer_tool_names: list[str] = []
    # Starting the most recent outer frame of function caller which means 2 popping the first two in the stack,
    # one for nearest_outer_tool_frame_name, one for its invoker, iterate through all outer frames:
    for frame_info in inspect.stack()[2:]:
        frame = frame_info.frame
        # Try to get the Callable object of the frame
        func: Callable[..., Any] | None = frame.f_globals.get(frame.f_code.co_name)
        # Try to see if it is indeed a Callable, if so, if it is  a tool.
        if func is not None and hasattr(func, "is_tool") and getattr(func, "is_tool"):
            outer_tool_names.append(func.__name__)
    return list(reversed(outer_tool_names))


def add_tool_trace(
    f: Callable[..., Any],
    result: Any,
    *args: NotGiven | Any,
    **kwargs: NotGiven | Any,
) -> None:
    """Add the trace of a tool: tool_name, arguments, execution results into current message in execution_context

    When trace_nested_tool is turned off. Tool traces form a list, denoting the order of tool call execution. Each node
    only contains tool_name, arguments, result. (Mostly for backward compatibility with ToolSandbox scenarios).

    When trace_nested_tool is turned on. Tool traces form an n-nary tree. Each root is the outer level tools issued
    by the agent. Inside each node there are three additional arguments:
        1. call_stack_tool_names, which contains a list of names from call stack root to immediate parent.
        2. children_tool_trace, which contains a list of nested tool traces.
        3. module_name, used to disambiguate tools since nested tool could come from a different tool box with different
            module.
    Post-order traversal of the n-nary tree reveals the runtime nested tool execution order.

    Naturally, tool_trace tree construction also happens in post-order. When a new trace is about to be added:
        1. We look through existing tool traces, to see if the new trace is the parent of any of them
            1.1 If so, remove those existing tool traces and add them as children of the new trace. Retain order.
                1.1.1 Insert new trace to the same location of the removed ones.
            1.2 If not, append new trace.

    TODO: This method currently has an issue where, if a nested tool throws an exception,
        tool trace constructions are partial and therefore invalid. Need to save exception traces
        as well to fix this.

    Args:
        f:          Tool callable
        result:     Return value of tool
        *args:      Positional arguments of tool
        **kwargs:   Keyword arguments of tool

    """
    current_context = get_current_context()
    if current_context.trace_tool:
        message_entry: pl.DataFrame = current_context.get_database(
            namespace=DatabaseNamespace.SANDBOX
        )
        # Note that a trace won't exist if tool execution raised exception
        existing_tool_trace_series: pl.Series | None = message_entry["tool_trace"][0]
        if existing_tool_trace_series is None:
            existing_tool_trace_json_list: list[str] = []
        else:
            existing_tool_trace_json_list = existing_tool_trace_series.to_list()
        # Convert positional arguments into keyword arguments. Signature.parameters is an ordered dict
        argument_names = list(signature(f).parameters.keys())
        all_arguments = {argument_names[i]: args[i] for i in range(len(args))}
        all_arguments = {**all_arguments, **kwargs}
        # Skip sentinels
        all_arguments = {k: v for k, v in all_arguments.items() if v is not NOT_GIVEN}
        # Add tool trace to current `SANDBOX` snapshot. Unfortunately polars doesn't
        # support complex objects so we need to JSON dump the trace into string. We use
        # our own JSON encoder to handle user-defined classes.
        if not current_context.trace_nested_tool:
            existing_tool_trace_json_list.append(
                json.dumps(
                    {
                        "tool_name": f.__name__,
                        "arguments": all_arguments,
                        "result": result,
                    },
                    ensure_ascii=False,
                    cls=MMToolSandboxJSONEncoder,
                )
            )
        else:
            existing_tool_trace: list[dict[str, Any]] = [
                json.loads(x) for x in existing_tool_trace_json_list
            ]
            current_trace = {
                "tool_name": f.__name__,
                "arguments": all_arguments,
                "result": result,
                "call_stack_tool_names": get_outer_tool_names(),
                "children_tool_trace": [],
            }
            # Identify existing tool traces that could be a child. Since tool_trace are a possible partial
            # post-order traversal sequence, possible children should appear at the very end of the list.
            if existing_tool_trace:
                # Find the starting index of children of current_trace
                children_start_ind = len(existing_tool_trace)
                # A matching call_stack_tool_names one short indicates immediate parent
                while (
                    children_start_ind > 0
                    and len(
                        existing_tool_trace[children_start_ind - 1][
                            "call_stack_tool_names"
                        ]
                    )
                    == len(current_trace["call_stack_tool_names"]) + 1
                    and existing_tool_trace[children_start_ind - 1][
                        "call_stack_tool_names"
                    ][:-1]
                    == current_trace["call_stack_tool_names"]
                ):
                    children_start_ind -= 1
                current_trace["children_tool_trace"] = existing_tool_trace[
                    children_start_ind:
                ]
                existing_tool_trace = existing_tool_trace[:children_start_ind]
            # Add in current_trace, dump into json again
            existing_tool_trace.append(current_trace)
            existing_tool_trace_json_list = [
                json.dumps(x, ensure_ascii=False, cls=MMToolSandboxJSONEncoder)
                for x in existing_tool_trace
            ]
        try:
            new_message_entry = message_entry.with_columns(
                pl.lit(pl.Series("tool_trace", [existing_tool_trace_json_list]))
            )
        except BaseException as e:
            # Polars raises a PanicException if the tool trace contains lone surrogates,
            # resulting in an error like this:
            # pyo3_runtime.PanicException: called `Result::unwrap()` on an `Err` value:
            # PyErr { type: <class 'UnicodeEncodeError'>, value:
            # UnicodeEncodeError('utf-8', '생일 \ud9c0혜 동하', 3, 4, 'surrogates not allowed'), traceback: None }
            #
            # Since PanicException is derived from BaseException, we re-raise it as a RuntimeError.
            # This ensures it will be caught by `except Exception` clauses.
            raise RuntimeError() from e

        current_context.update_database(
            namespace=DatabaseNamespace.SANDBOX, dataframe=new_message_entry
        )


def register_as_tool(
    toolboxes: set[ToolboxName],
    database_namespaces: set[DatabaseNamespace] | None = None,
    visible_to: tuple[RoleType, ...] | None = None,
) -> Callable[..., Any]:
    """Decorator factory, making decorator with arguments possible

    Args:
        toolboxes:   The toolboxes to which this tool should be added.
        database_namespaces: The namespaces of the databases that the tool accesses.
        visible_to:  Which roles are this tool visible to. Defaults to Agent.

    Returns:
        decorator function
    """

    def internal_decorator(func: Callable[..., T]) -> Callable[..., T]:
        """Necessary to make decorator factory work. Returns a decorator which marks a function as tool,
        available for Roles to use

        Make use of decorator.decorate to keep function signature intact

        Args:
            func:   function to decorate

        Returns:
            decorated function
        """

        def _f(f: Callable[..., T], *args: Any, **kwargs: Any) -> T:
            current_context = get_current_context()
            # Disable tracing for any tool call happened inside another tool's scope if trace_nested_tool is False.
            with new_context_with_attribute(
                trace_tool=current_context.trace_nested_tool
            ):
                result = f(*args, **kwargs)
            # Add tool trace to messages
            add_tool_trace(f, result, *args, **kwargs)
            return result

        dec = decorate(func, _f)
        setattr(
            dec,
            "database_namespaces",
            {} if database_namespaces is None else database_namespaces,
        )
        setattr(dec, "is_tool", True)
        setattr(
            dec,
            "visible_to",
            visible_to if visible_to is not None else (RoleType.AGENT,),
        )
        for toolbox in toolboxes:
            register_tool(toolbox, dec)

        return dec

    return internal_decorator


@contextmanager
def all_logging_disabled(highest_level: int = logging.CRITICAL) -> Iterator[None]:
    """A context manager that will prevent any logging messages
    triggered during the body from being processed.

    This is useful when a known package constantly emits noisy warnings.
    """
    previous_level = logging.root.manager.disable
    logging.disable(highest_level)
    try:
        yield
    finally:
        logging.disable(previous_level)


def attrs_serialize(inst: Any, field: Any, value: Any) -> Any:
    """Serialize troublesome values for attrs. Note almost all below are irreversible operations"""
    if isinstance(value, functools.partial):
        return value.func.__name__
    if isfunction(value):
        if value.__name__ == "<lambda>":
            return getsource(value).strip()
        else:
            return value.__name__
    if callable(value):
        # Applies to arbitrary classes with a __call__ method, not handled by the above cases.
        return str(value)
    if isinstance(value, pl.DataFrame):
        return value.to_dicts()
    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, Locale):
        return value.name
    return value


def deterministic_uuid(payload: str) -> str:
    """Simple helper used to generate a deterministic uuid string

    Args:
        payload:    Payload string. Any uuid derived from the same payload is guaranteed to be the same

    Returns:
        String uuid
    """
    return str(uuid.uuid5(namespace=uuid.NAMESPACE_URL, name=payload))


def is_close(value: T, reference: T, atol: float | None = None) -> bool:
    """Checks if value is close the reference.

    Allows for near identity comparison for different data types.

    In the case of float values, allow for a diff of at max atol.

    Args:
        value:      Value to compare.
        reference:  Value to compare against.
        atol:       Absolute tolerance, only needed for numerical values

    Returns:
        Boolean indicating if the values are close
    """
    if (
        atol is not None
        and isinstance(value, (float, int))
        and isinstance(reference, (float, int))
    ):
        return abs(value - reference) <= atol  # type: ignore[operator]
    return value == reference


def get_tomorrow_datetime() -> datetime.datetime:
    """Get a datetime object for exactly 1 day from now in the future.

    Returns:
        datetime object for exactly 1 day from now in the future.
    """
    return datetime.datetime.now() + datetime.timedelta(days=1)


def get_next_iso_weekday_datetime_given_datetime(
    current_datetime: datetime.datetime, next_iso_weekday: int
) -> datetime.datetime:
    """Get a datetime object for next weekday relative to current_datetime.

    Defined as the weekday of next week.

    Args:
        current_datetime:   Datetime to calculate from.
        next_iso_weekday:   1 based integer. 1 for Monday.

    Returns:
        datetime object for next weekday relative to current_datetime.
    """
    # weekday uses 0 for monday. the following relative delta means the next_iso_weekday of next week.
    return current_datetime + relativedelta(
        days=next_iso_weekday, weekday=weekday(next_iso_weekday - 1)(1)
    )


def get_next_iso_weekday_datetime(next_iso_weekday: int) -> datetime.datetime:
    """Get a datetime object for next weekday.

    Defined as the weekday of next week.

    Args:
        next_iso_weekday:   1 based integer. 1 for Monday.

    Returns:
        datetime object for next weekday.
    """
    return get_next_iso_weekday_datetime_given_datetime(
        current_datetime=datetime.datetime.now(), next_iso_weekday=next_iso_weekday
    )


def merge_dicts(
    base_dict: dict[str, Any], overwrite_dict: dict[str, Any]
) -> dict[str, Any]:
    """Creates a new dictionary based on merging `overwrite_dict` into `base_dict`."""
    base_dict = deepcopy(base_dict)
    for key, value in overwrite_dict.items():
        # If the value is another dict in both dictionaries, recurse.
        if (
            isinstance(value, dict)
            and key in base_dict
            and isinstance(base_dict[key], dict)
        ):
            base_dict[key] = merge_dicts(base_dict[key], value)
        else:
            # Copy if
            #   - The base dict does not have the key
            #   - The value associated to the key is not another dict.
            base_dict[key] = deepcopy(value)
    return base_dict
