# Copyright © 2026 Apple Inc.

"""
Entity Diff Evaluator — unified before/after state comparison.

Compares initial vs final database state for both MMToolSandbox polars
databases and AppWorld SQLite tables.  Supports create/update/delete checks
and guardrails, with per-entity precision/recall scoring via Hungarian
matching.
"""

from __future__ import annotations

import dataclasses
import json
from collections import defaultdict
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]
from sqlalchemy import text as sa_text

from mmtoolsandbox.appworld.bridge import get_appworld_bridge
from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.evaluation import (
    ColumnSimilarityMeasureType,
    _fill_null,
    column_datetime_similarity,
    column_exact_match_similarity,
    column_ignore_similarity,
    guardrail_similarity,
)
from mmtoolsandbox.common.function_registry import FunctionRegistry

LOGGER = getLogger(__name__)

# Namespaces excluded from guardrail checks (infrastructure, not user data).
_GUARDRAIL_EXCLUDED_NAMESPACES = {
    DatabaseNamespace.SANDBOX,
    DatabaseNamespace.IMAGE,
    DatabaseNamespace.CALENDAR_EVENTS_RECURRENCE_STAGING,
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class EntityDiffSpec:
    """A single expected state change (JSON-serializable).

    Attributes:
        operation: "create", "update", or "delete".
        source: "agentsandbox" or "appworld".
        namespace: MMToolSandbox DatabaseNamespace string (e.g. "REMINDER").
        table: AppWorld table key (e.g. "simple_note.notes").
        entity_data: For create — full expected row as dict.
        entity_id_column: For update/delete — column name of the ID.
        entity_id_value: For update/delete — value of the ID.
        expected_fields: For update — dict of changed field→value.
        column_similarity_measure: column name → similarity function name.
    """

    operation: Literal["create", "update", "delete"]
    source: Literal["agentsandbox", "appworld"]
    namespace: str | None = None
    table: str | None = None
    entity_data: dict[str, Any] | None = None
    entity_id_column: str | None = None
    entity_id_value: Any | None = None
    expected_fields: dict[str, Any] | None = None
    column_similarity_measure: dict[str, str] = dataclasses.field(default_factory=dict)

    @property
    def table_key(self) -> str:
        """Unified key: namespace string for MMToolSandbox, table string for AppWorld."""
        if self.source == "agentsandbox":
            return self.namespace or ""
        return self.table or ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        d: dict[str, Any] = {
            "operation": self.operation,
            "source": self.source,
        }
        if self.namespace is not None:
            d["namespace"] = self.namespace
        if self.table is not None:
            d["table"] = self.table
        if self.entity_data is not None:
            d["entity_data"] = self.entity_data
        if self.entity_id_column is not None:
            d["entity_id_column"] = self.entity_id_column
        if self.entity_id_value is not None:
            d["entity_id_value"] = self.entity_id_value
        if self.expected_fields is not None:
            d["expected_fields"] = self.expected_fields
        if self.column_similarity_measure:
            d["column_similarity_measure"] = self.column_similarity_measure
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EntityDiffSpec":
        """Deserialize from dict."""
        return cls(
            operation=d["operation"],
            source=d["source"],
            namespace=d.get("namespace"),
            table=d.get("table"),
            entity_data=d.get("entity_data"),
            entity_id_column=d.get("entity_id_column"),
            entity_id_value=d.get("entity_id_value"),
            expected_fields=d.get("expected_fields"),
            column_similarity_measure=d.get("column_similarity_measure", {}),
        )


@dataclass
class EntityDiffEvalConfig:
    """Evaluation configuration — just specs.

    Guardrails are derived at runtime:
      MMToolSandbox: get_active_database_namespaces() - spec namespaces - {SANDBOX, IMAGE}
      AppWorld: tables in captured initial state - spec tables
    """

    specs: list[EntityDiffSpec]

    def to_dict(self) -> dict[str, Any]:
        return {"specs": [s.to_dict() for s in self.specs]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EntityDiffEvalConfig":
        return cls(specs=[EntityDiffSpec.from_dict(s) for s in d["specs"]])


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class RowMatchResult:
    """Result of flexible row matching."""

    precision: float  # sum(matched_scores) / num_actual
    recall: float  # sum(matched_scores) / num_expected
    per_expected_scores: list[float]  # score per expected row (0 if unmatched)
    # Per-column scores for each matched expected row.
    # List of dicts: [{col_name: score, ...}, ...] indexed by expected row.
    # Unmatched rows get all-zero dicts.
    per_expected_column_scores: list[dict[str, float]]
    num_actual: int
    num_expected: int
    num_matched: int  # min(M, N) matched pairs
    # Actual values from the matched actual row, indexed by expected row.
    # Unmatched expected rows get an empty dict.
    per_expected_actual_values: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GroupEvalResult:
    """Evaluation result for one (table_key, operation) group."""

    table_key: str
    operation: str
    precision: float
    recall: float
    per_entity_scores: list[float]
    # Per-column breakdown for each expected entity.
    per_entity_column_scores: list[dict[str, float]]
    num_actual: int
    num_expected: int
    # Actual values from the matched actual row for each expected entity.
    # Empty dict for unmatched entities.
    per_entity_actual_values: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EntityDiffEvalResult:
    """Overall evaluation result."""

    overall_precision: float
    overall_recall: float
    group_results: list[GroupEvalResult]
    guardrail_pass: bool
    guardrail_failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_precision": self.overall_precision,
            "overall_recall": self.overall_recall,
            "group_results": [dataclasses.asdict(g) for g in self.group_results],
            "guardrail_pass": self.guardrail_pass,
            "guardrail_failures": self.guardrail_failures,
        }


# ---------------------------------------------------------------------------
# I/O helpers (MMToolSandbox polars databases)
# ---------------------------------------------------------------------------


def read_agentsandbox_initial(
    context: Any,  # ExecutionContext
    namespace: DatabaseNamespace,
) -> pl.DataFrame:
    """Read the initial (pre-agent) snapshot for an MMToolSandbox namespace."""
    df: pl.DataFrame = context.get_database(
        namespace,
        sandbox_message_index=context.first_user_sandbox_message_index,
        drop_sandbox_message_index=False,
        drop_headguard=True,
    )
    return df


def read_agentsandbox_final(
    context: Any,  # ExecutionContext
    namespace: DatabaseNamespace,
) -> pl.DataFrame:
    """Read the final (post-agent) snapshot for an MMToolSandbox namespace."""
    df: pl.DataFrame = context.get_database(
        namespace,
        drop_sandbox_message_index=False,
        drop_headguard=True,
    )
    return df


# ---------------------------------------------------------------------------
# Core matching
# ---------------------------------------------------------------------------


def _resolve_column_similarities(
    column_sim_names: dict[str, str],
) -> dict[str, ColumnSimilarityMeasureType]:
    """Convert column similarity function names to actual callables."""
    result: dict[str, ColumnSimilarityMeasureType] = {}
    for col, name in column_sim_names.items():
        result[col] = FunctionRegistry.get(name)
    return result


_BOOKKEEPING_TIMESTAMP_COLUMNS = {
    "updated_at",
    "created_at",
    "modified_at",
    "deleted_at",
    "last_logged_in",
    "last_updated_at",
    "last_modified_at",
    "added_at",
    "liked_at",
    "joined_at",
    # Amazon: computed from product.delivery_days at order time
    "expected_delivery_at",
    "delivered_at",
    # Amazon: set by backend when return is initiated
    "initiated_at",
    # Spotify: set by backend when song is downloaded
    "downloaded_at",
    # Venmo: set by backend when payment is approved/denied
    "approved_at",
    "denied_at",
    # MMToolSandbox REMINDER: set by datetime.now() at tool call time
    "creation_datetime",
}

# Columns whose values are auto-generated by the backend and not directly
# controllable by the agent.  Always forced to column_ignore_similarity.
_AUTO_GENERATED_COLUMNS = _BOOKKEEPING_TIMESTAMP_COLUMNS | {
    "relationships",
    "compressed_data",
    "file_compressed_data",
    # Splitwise: auto-generated or backend-populated
    "invitation_code",  # random code from get_unique_id(5)
    "debtor_ids",  # populated via ExpenseShare change_if hook
    "participant_ids",  # populated via model pre-save hook
    # Todoist: auto-generated invite code
    "invite_code",  # random code from get_unique_id
    # Spotify: backend-managed player state
    "queue_song_ids",
    # Gmail: label defaults to None, not set during send_email flow
    "label",
    # Boolean/flag fields with backend defaults
    "deleted",
}

# Columns whose values are assigned by MMToolSandbox at runtime and are
# path-dependent — the agent cannot control them and they will differ
# across evaluation runs.
#
# All MMToolSandbox tools generate entity IDs via ``uuid4()``, so the
# actual value depends on execution order and is never reproducible.
# These columns are always forced to ``column_ignore_similarity``
# regardless of what any scenario spec says, so that scenario authors
# do not need to remember to exclude them manually.
_AGENTSANDBOX_RUNTIME_ID_COLUMNS = {
    "reminder_id",
    "calendar_event_id",
    "calendar_id",
    "person_id",
    "note_id",
    "message_id",
}


def _default_column_similarity_for_value(col: str, val: Any) -> str:
    """Pick a default similarity function name based on column name and value."""
    if val is None or val == "" or val == []:
        return "column_ignore_similarity"
    # Runtime-assigned IDs are path-dependent — ignore by default so that newly
    # generated scenario specs do not need to set this explicitly.
    if col in _AGENTSANDBOX_RUNTIME_ID_COLUMNS:
        return "column_ignore_similarity"
    if col in _AUTO_GENERATED_COLUMNS:
        return "column_ignore_similarity"
    if col in ("content", "title", "description", "name", "body", "text", "subject"):
        return "column_rouge_l_similarity"
    if col in ("tags", "data"):
        return "column_ignore_similarity"
    return "column_exact_match_similarity"


def flexible_row_matching(
    actual_rows: pl.DataFrame,
    expected_rows: pl.DataFrame,
    column_similarities: dict[str, ColumnSimilarityMeasureType],
    per_expected_column_similarities: (
        list[dict[str, ColumnSimilarityMeasureType]] | None
    ) = None,
) -> RowMatchResult:
    """Match actual rows against expected rows using Hungarian algorithm.

    Handles M != N (unlike snapshot_similarity which returns 0).

    1. Build M×N cost matrix using column similarities
       (same logic as snapshot_similarity in evaluation.py)
    2. Run scipy.linear_sum_assignment on the rectangular matrix
       (automatically handles M!=N by matching min(M,N) pairs)
    3. Compute per-pair similarity scores
    4. Return precision, recall, and per-row details

    Args:
        actual_rows: M rows (actual new/deleted/changed rows)
        expected_rows: N rows (expected from spec)
        column_similarities: column → similarity function (shared fallback)
        per_expected_column_similarities: optional list of per-expected-row
            column similarity dicts.  When provided, expected row *i* uses
            ``per_expected_column_similarities[i]`` instead of the shared
            ``column_similarities`` for any column present in the per-row
            dict.  This is critical when different expected entities in the
            same group require different similarity functions for the same
            column (e.g. one entity ignores ``email`` while another
            requires exact match).

    Returns:
        RowMatchResult with precision, recall, and per-expected scores.
    """
    num_actual = actual_rows.shape[0]
    num_expected = expected_rows.shape[0]

    if num_expected == 0:
        return RowMatchResult(
            precision=1.0 if num_actual == 0 else 0.0,
            recall=1.0,
            per_expected_scores=[],
            per_expected_column_scores=[],
            num_actual=num_actual,
            num_expected=0,
            num_matched=0,
            per_expected_actual_values=[],
        )

    if num_actual == 0:
        return RowMatchResult(
            precision=0.0,
            recall=0.0,
            per_expected_scores=[0.0] * num_expected,
            per_expected_column_scores=[
                {c: 0.0 for c in column_similarities} for _ in range(num_expected)
            ],
            num_actual=0,
            num_expected=num_expected,
            num_matched=0,
            per_expected_actual_values=[{} for _ in range(num_expected)],
        )

    # Only use columns present in both actual and expected
    common_cols = [
        c
        for c in expected_rows.columns
        if c in actual_rows.columns and c in column_similarities
    ]
    if not common_cols:
        return RowMatchResult(
            precision=0.0,
            recall=0.0,
            per_expected_scores=[0.0] * num_expected,
            per_expected_column_scores=[
                {c: 0.0 for c in column_similarities} for _ in range(num_expected)
            ],
            num_actual=num_actual,
            num_expected=num_expected,
            num_matched=0,
            per_expected_actual_values=[{} for _ in range(num_expected)],
        )

    # Fill nulls for stable matching
    actual_filled = _fill_null(actual_rows)
    expected_filled = _fill_null(expected_rows)

    # Build N×M cost matrix (expected × actual) using arithmetic mean of
    # non-ignored column similarities.
    # Also store per-column similarity scores for each (expected, actual) pair.
    cost_matrix_rows: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    # per_col_sims[expected_idx][actual_idx] = {col_name: similarity}
    per_col_sims: list[list[dict[str, float]]] = []
    # Track non-ignored columns per expected row for per-pair scoring
    non_ignored_per_exp: list[list[str]] = []

    for exp_idx, row in enumerate(expected_filled.select(common_cols).to_dicts()):
        # Resolve which similarity function to use for each column.
        # Per-expected-row overrides take precedence over the shared dict.
        exp_col_sims: dict[str, ColumnSimilarityMeasureType] = {}
        non_ignored_cols: list[str] = []
        for col_name in common_cols:
            if (
                per_expected_column_similarities is not None
                and col_name in per_expected_column_similarities[exp_idx]
            ):
                exp_col_sims[col_name] = per_expected_column_similarities[exp_idx][
                    col_name
                ]
            else:
                exp_col_sims[col_name] = column_similarities[col_name]
            # Track non-ignored columns for arithmetic mean
            if exp_col_sims[col_name] is not column_ignore_similarity:
                non_ignored_cols.append(col_name)

        col_sim_dfs = []
        for i, (col_name, value) in enumerate(row.items()):
            sim_fn = exp_col_sims[col_name]
            sim_df = sim_fn(
                dataframe=actual_filled,
                column_name=col_name,
                value=value,
            )
            raw_score = sim_df["similarity"][0] if num_actual > 0 else 0.0
            sim_score = float(raw_score) if raw_score is not None else 0.0
            LOGGER.info(
                "  [expected=%d] %s: expected=%r actual=%r sim=%.4f (fn=%s)",
                exp_idx,
                col_name,
                value,
                actual_filled[col_name][0] if num_actual > 0 else None,
                sim_score,
                sim_fn.__name__
                if hasattr(sim_fn, "__name__")
                else type(sim_fn).__name__,
            )
            col_sim_dfs.append(
                sim_df.select(pl.col("similarity").alias(f"sim_{col_name}"))
            )

        combined = pl.concat(col_sim_dfs, how="horizontal")

        # Store per-column similarities for this expected row vs all actual rows
        row_col_sims: list[dict[str, float]] = []
        for actual_idx in range(num_actual):
            col_scores: dict[str, float] = {}
            for col_name in common_cols:
                raw = combined[f"sim_{col_name}"][actual_idx]
                col_scores[col_name] = float(raw) if raw is not None else 0.0
            row_col_sims.append(col_scores)
        per_col_sims.append(row_col_sims)

        # Compute cost row as negative arithmetic mean of non-ignored column
        # similarities.  Hungarian assignment minimizes total cost, so the
        # optimal matching maximizes the average non-ignored similarity.
        # When ALL columns are ignored (e.g. join tables with only FK refs
        # to created entities), cost is 0 (perfect match by existence).
        if non_ignored_cols:
            cost_cols = [f"sim_{c}" for c in non_ignored_cols]
            cost_matrix_rows.append(
                combined.select((-pl.mean_horizontal(*cost_cols)).alias("neg_mean"))[
                    "neg_mean"
                ].to_numpy()
            )
        else:
            # All columns ignored — entity verified by existence (anti-join)
            cost_matrix_rows.append(np.zeros(num_actual))
        # Remember which columns are non-ignored for per-pair scoring
        non_ignored_per_exp.append(non_ignored_cols)

    numpy_cost_matrix = np.stack(cost_matrix_rows, axis=0)  # shape (N, M)

    try:
        row_ind, col_ind = linear_sum_assignment(numpy_cost_matrix)
    except ValueError:
        # Cost matrix infeasible (all inf)
        return RowMatchResult(
            precision=0.0,
            recall=0.0,
            per_expected_scores=[0.0] * num_expected,
            per_expected_column_scores=[
                {c: 0.0 for c in common_cols} for _ in range(num_expected)
            ],
            num_actual=num_actual,
            num_expected=num_expected,
            num_matched=0,
            per_expected_actual_values=[{} for _ in range(num_expected)],
        )

    # Pre-extract actual row values (over common_cols) for display.
    actual_row_dicts: list[dict[str, Any]] = actual_rows.select(common_cols).to_dicts()

    # Per-pair similarity scores and per-column breakdowns
    # Score = arithmetic mean of non-ignored column similarities.
    matched_scores: dict[int, float] = {}
    matched_col_scores: dict[int, dict[str, float]] = {}
    matched_actual_values: dict[int, dict[str, Any]] = {}
    for r, c in zip(row_ind, col_ind):
        col_scores = per_col_sims[int(r)][int(c)]
        ni_cols = non_ignored_per_exp[int(r)]
        if ni_cols:
            score = sum(col_scores.get(col, 0.0) for col in ni_cols) / len(ni_cols)
        else:
            score = 1.0  # all columns ignored — verified by existence
        matched_scores[int(r)] = score
        matched_col_scores[int(r)] = col_scores
        matched_actual_values[int(r)] = actual_row_dicts[int(c)]

    # Build per-expected scores (0 if not matched)
    zero_col_scores = {c: 0.0 for c in common_cols}
    per_expected_scores = [matched_scores.get(i, 0.0) for i in range(num_expected)]
    per_expected_column_scores = [
        matched_col_scores.get(i, zero_col_scores) for i in range(num_expected)
    ]
    per_expected_actual_values = [
        matched_actual_values.get(i, {}) for i in range(num_expected)
    ]

    total_matched = sum(per_expected_scores)
    precision = total_matched / num_actual if num_actual > 0 else 0.0
    recall = total_matched / num_expected if num_expected > 0 else 0.0

    return RowMatchResult(
        precision=min(precision, 1.0),
        recall=min(recall, 1.0),
        per_expected_scores=per_expected_scores,
        per_expected_column_scores=per_expected_column_scores,
        num_actual=num_actual,
        num_expected=num_expected,
        num_matched=len(row_ind),
        per_expected_actual_values=per_expected_actual_values,
    )


# ---------------------------------------------------------------------------
# Anti-join helper for large tables
# ---------------------------------------------------------------------------

# Common primary key column names for AppWorld tables and MMToolSandbox namespaces.
_ID_COLUMN_CANDIDATES = (
    "id",
    "note_id",
    "reminder_id",
    "calendar_event_id",
    "calendar_id",
    "person_id",
    "song_id",
    "playlist_id",
    "email_id",
    "thread_id",
    "order_id",
    "product_id",
    "transaction_id",
    "expense_id",
    "task_id",
    "file_id",
    "directory_id",
    "contact_id",
    "alarm_id",
)


def _anti_join(left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    """Compute rows in *left* not in *right* efficiently.

    For large AppWorld tables (10K+ rows) a full-column anti-join is expensive
    and fragile (float precision, datetime formats).  When both sides share
    a recognised ID column we filter on that column only via ``is_in``, which
    is O(N) and immune to type-mismatch issues on non-key columns.

    Falls back to full-column anti-join when no common ID column exists
    (the MMToolSandbox polars tables are small enough for this to be fine).

    Args:
        left: The "larger" side (e.g. final state).
        right: The "reference" side (e.g. initial state).

    Returns:
        Rows in *left* that are not in *right*.
    """
    if left.shape[0] == 0 or right.shape[0] == 0:
        return left

    common_cols = [c for c in left.columns if c in right.columns]
    if not common_cols:
        return left

    # Prefer an ID-column based diff (fast + robust for large tables)
    for id_col in _ID_COLUMN_CANDIDATES:
        if id_col in left.columns and id_col in right.columns:
            # Cast to Utf8 so int-vs-string ID mismatches don't cause failures
            left_ids = left[id_col].cast(pl.Utf8)
            right_ids = right[id_col].cast(pl.Utf8)
            return left.filter(~left_ids.is_in(right_ids))

    # No ID column found — full-column anti-join (MMToolSandbox tables are small)
    return left.join(right, on=common_cols, how="anti")


# ---------------------------------------------------------------------------
# AppWorld SQLite I/O
# ---------------------------------------------------------------------------


def read_appworld_table(table_key: str) -> pl.DataFrame:
    """Read an AppWorld SQLite table into a polars DataFrame.

    Args:
        table_key: e.g. ``"simple_note.notes"`` (app.table)

    Returns:
        polars DataFrame with a dummy ``sandbox_message_index=0`` column
        for compatibility with existing similarity functions.
    """
    bridge = get_appworld_bridge()
    if not bridge.is_initialized:
        raise RuntimeError("AppWorld bridge not initialized")

    app_name, table_name = table_key.split(".", 1)

    try:
        engine = _get_engine_from_bridge(bridge, app_name)
        if engine is None:
            LOGGER.warning(
                "No database engine for app %s (table %s)", app_name, table_key
            )
            return pl.DataFrame()

        query = f"SELECT * FROM {table_name}"  # noqa: S608
        with engine.connect() as conn:
            result = conn.execute(sa_text(query))
            rows = result.fetchall()
            if not rows:
                return pl.DataFrame()
            col_names = list(result.keys())
            dicts = [dict(zip(col_names, row)) for row in rows]
            df = pl.DataFrame(dicts, infer_schema_length=None)
            if "deleted" in df.columns:
                df = df.filter(pl.col("deleted") == False)  # noqa: E712
    except Exception as e:
        LOGGER.warning("Failed to read AppWorld table %s: %s", table_key, e)
        return pl.DataFrame()

    df = df.with_columns(pl.lit(0).alias("sandbox_message_index"))
    return df


def _get_engine_from_bridge(bridge: Any, app_name: str) -> Any:
    """Get the SQLAlchemy engine for an app via the bridge's initialized models."""
    try:
        models = bridge.appworld.models
        if not hasattr(models, app_name):
            return None
        app_models = getattr(models, app_name)
        sql_model = getattr(app_models, "SQLModel", None)
        if (
            sql_model is not None
            and hasattr(sql_model, "db")
            and sql_model.db.engine is not None
        ):
            return sql_model.db.engine
    except Exception as e:
        LOGGER.debug("Failed to get engine from bridge for %s: %s", app_name, e)
    return None


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class EntityDiffEvaluator:
    """Evaluates entity diffs by comparing initial vs final state.

    Usage:
        evaluator = EntityDiffEvaluator(config)
        evaluator.capture_initial_state(execution_context, bridge)
        # ... agent acts ...
        result = evaluator.evaluate(execution_context, bridge)
    """

    def __init__(self, config: EntityDiffEvalConfig) -> None:
        self._config = config
        self._initial_appworld: dict[str, pl.DataFrame] = {}
        self.last_evidence: dict[str, Any] = {}

    @property
    def config(self) -> EntityDiffEvalConfig:
        return self._config

    def capture_initial_state(
        self,
        execution_context: Any | None = None,
        bridge: Any | None = None,
    ) -> None:
        """Capture initial state before agent acts.

        MMToolSandbox snapshots are available via ExecutionContext history,
        so no explicit capture is needed for those.  AppWorld tables are
        external (SQLite) and must be read before the agent modifies them.
        """
        appworld_tables: set[str] = set()
        for spec in self._config.specs:
            if spec.source == "appworld" and spec.table:
                appworld_tables.add(spec.table)

        for table_key in appworld_tables:
            try:
                self._initial_appworld[table_key] = read_appworld_table(table_key)
            except Exception as e:
                LOGGER.warning(
                    "Failed to capture initial state for %s: %s", table_key, e
                )
                self._initial_appworld[table_key] = pl.DataFrame()

    def evaluate(
        self,
        execution_context: Any | None = None,
        bridge: Any | None = None,
    ) -> EntityDiffEvalResult:
        """Evaluate entity diffs after agent finishes.

        Args:
            execution_context: MMToolSandbox ExecutionContext.
            bridge: AppWorld bridge instance.

        Returns:
            EntityDiffEvalResult with precision/recall and guardrail info.

        Side effect:
            Populates ``self.last_evidence`` with per-group table snapshots
            for offline re-evaluation.
        """
        # Step 1: Group specs by (table_key, operation)
        groups: dict[tuple[str, str], list[EntityDiffSpec]] = defaultdict(list)
        for spec in self._config.specs:
            groups[(spec.table_key, spec.operation)].append(spec)

        # Step 2: Evaluate each group, collecting table snapshots
        group_results: list[GroupEvalResult] = []
        table_snapshots: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for (table_key, operation), specs in groups.items():
            result, initial, final = self._evaluate_group(
                table_key, operation, specs, execution_context, bridge
            )
            group_results.append(result)
            if table_key not in table_snapshots:
                table_snapshots[table_key] = {
                    "initial": initial.to_dicts(),
                    "final": final.to_dicts(),
                }

        # Step 3: Check guardrails
        guardrail_pass, guardrail_failures = self._check_guardrails(
            execution_context, bridge
        )

        # Step 4: Combine scores
        if group_results:
            overall_precision = sum(g.precision for g in group_results) / len(
                group_results
            )
            overall_recall = sum(g.recall for g in group_results) / len(group_results)
        else:
            overall_precision = 1.0
            overall_recall = 1.0

        eval_result = EntityDiffEvalResult(
            overall_precision=overall_precision,
            overall_recall=overall_recall,
            group_results=group_results,
            guardrail_pass=guardrail_pass,
            guardrail_failures=guardrail_failures,
        )

        self.last_evidence = {
            "specs": self._config.to_dict(),
            "tables": table_snapshots,
            "result": eval_result.to_dict(),
        }

        return eval_result

    def save_evidence(self, path: Path) -> None:
        """Save entity diff evidence to a JSON file for offline re-evaluation."""
        if not self.last_evidence:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.last_evidence, f, indent=2, ensure_ascii=False, default=str)

    @classmethod
    def evaluate_from_evidence(cls, path: Path) -> EntityDiffEvalResult:
        """Re-run entity diff evaluation from saved evidence.

        Loads specs and table snapshots from a previously saved evidence
        file and re-evaluates, allowing updated similarity logic to take
        effect without re-running the agent.
        """
        with open(path) as f:
            evidence = json.load(f)

        config = EntityDiffEvalConfig.from_dict(evidence["specs"])
        evaluator = cls(config)

        tables: dict[str, dict[str, list[dict[str, Any]]]] = evidence["tables"]

        groups: dict[tuple[str, str], list[EntityDiffSpec]] = defaultdict(list)
        for spec in config.specs:
            groups[(spec.table_key, spec.operation)].append(spec)

        group_results: list[GroupEvalResult] = []
        new_table_snapshots: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for (table_key, operation), specs in groups.items():
            saved = tables.get(table_key)
            if saved is None:
                LOGGER.warning("No saved table data for %s, skipping", table_key)
                continue
            initial = pl.DataFrame(saved["initial"])
            final = pl.DataFrame(saved["final"])

            if operation == "create":
                result = evaluator._evaluate_create(table_key, specs, initial, final)
            elif operation == "delete":
                result = evaluator._evaluate_delete(table_key, specs, initial, final)
            elif operation == "update":
                result = evaluator._evaluate_update(
                    table_key, specs, initial, final, None, None
                )
            else:
                result = GroupEvalResult(
                    table_key=table_key,
                    operation=operation,
                    precision=0.0,
                    recall=0.0,
                    per_entity_scores=[0.0] * len(specs),
                    per_entity_column_scores=[{} for _ in specs],
                    num_actual=0,
                    num_expected=len(specs),
                    per_entity_actual_values=[{} for _ in specs],
                )
            group_results.append(result)
            if table_key not in new_table_snapshots:
                new_table_snapshots[table_key] = saved

        if group_results:
            overall_precision = sum(g.precision for g in group_results) / len(
                group_results
            )
            overall_recall = sum(g.recall for g in group_results) / len(group_results)
        else:
            overall_precision = 1.0
            overall_recall = 1.0

        # Guardrails cannot be re-evaluated from saved evidence (would need
        # all non-spec tables).  Carry forward the original result.
        old_result = evidence.get("result", {})
        guardrail_pass = old_result.get("guardrail_pass", True)
        guardrail_failures = old_result.get("guardrail_failures", [])

        eval_result = EntityDiffEvalResult(
            overall_precision=overall_precision,
            overall_recall=overall_recall,
            group_results=group_results,
            guardrail_pass=guardrail_pass,
            guardrail_failures=guardrail_failures,
        )

        evaluator.last_evidence = {
            "specs": config.to_dict(),
            "tables": new_table_snapshots,
            "result": eval_result.to_dict(),
        }

        return eval_result

    # ---- Internal helpers ----

    def _get_initial_and_final(
        self,
        spec: EntityDiffSpec,
        execution_context: Any | None,
        bridge: Any | None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Get initial and final DataFrames for a spec's table."""
        if spec.source == "agentsandbox" and execution_context is not None:
            assert spec.namespace is not None
            ns = DatabaseNamespace(spec.namespace)
            initial = read_agentsandbox_initial(execution_context, ns)
            final = read_agentsandbox_final(execution_context, ns)
            return initial, final
        if spec.source == "appworld" and spec.table:
            initial = self._initial_appworld.get(spec.table, pl.DataFrame())
            try:
                final = read_appworld_table(spec.table)
            except Exception:
                final = pl.DataFrame()
            return initial, final
        return pl.DataFrame(), pl.DataFrame()

    def _resolve_col_sims(
        self, specs: list[EntityDiffSpec]
    ) -> dict[str, ColumnSimilarityMeasureType]:
        """Merge and resolve column similarity measures from specs.

        Applies a forced override for columns in `_AGENTSANDBOX_RUNTIME_ID_COLUMNS`
        Applies forced overrides for columns that the agent cannot control:
        - ``_AGENTSANDBOX_RUNTIME_ID_COLUMNS`` (e.g. ``reminder_id``): values
          assigned by MMToolSandbox at runtime, path-dependent.
        - ``_AUTO_GENERATED_COLUMNS``: bookkeeping timestamps (``created_at``,
          ``updated_at``, etc.) and auto-populated fields (``relationships``)
          that are set by the backend, not the agent.

        Fixing this centrally means existing scenario JSON files do not need
        to be regenerated.
        """
        merged: dict[str, str] = {}
        for spec in specs:
            merged.update(spec.column_similarity_measure)
        # Force-ignore runtime-assigned ID columns regardless of spec.
        for col in _AGENTSANDBOX_RUNTIME_ID_COLUMNS:
            if col in merged:
                merged[col] = "column_ignore_similarity"
        # Force-ignore auto-generated columns (timestamps, relationships).
        for col in _AUTO_GENERATED_COLUMNS:
            if col in merged:
                merged[col] = "column_ignore_similarity"
        return _resolve_column_similarities(merged)

    def _evaluate_group(
        self,
        table_key: str,
        operation: str,
        specs: list[EntityDiffSpec],
        execution_context: Any | None,
        bridge: Any | None,
    ) -> tuple[GroupEvalResult, pl.DataFrame, pl.DataFrame]:
        """Evaluate a single (table_key, operation) group.

        Returns:
            A tuple of (GroupEvalResult, processed_initial, processed_final).
        """
        LOGGER.info(
            "Evaluating group: table=%s operation=%s specs=%d",
            table_key,
            operation,
            len(specs),
        )
        # Get initial and final from first spec (all share same table_key)
        initial, final = self._get_initial_and_final(
            specs[0], execution_context, bridge
        )
        LOGGER.info(
            "  initial: %d rows, final: %d rows",
            initial.shape[0],
            final.shape[0],
        )

        # Drop sandbox_message_index for comparison
        drop_cols = ["sandbox_message_index"]
        for col in drop_cols:
            if col in initial.columns:
                initial = initial.drop(col)
            if col in final.columns:
                final = final.drop(col)

        # Fill nulls for stable joins
        initial = _fill_null(initial)
        final = _fill_null(final)

        if operation == "create":
            result = self._evaluate_create(table_key, specs, initial, final)
        elif operation == "delete":
            result = self._evaluate_delete(table_key, specs, initial, final)
        elif operation == "update":
            result = self._evaluate_update(
                table_key, specs, initial, final, execution_context, bridge
            )
        else:
            result = GroupEvalResult(
                table_key=table_key,
                operation=operation,
                precision=0.0,
                recall=0.0,
                per_entity_scores=[0.0] * len(specs),
                per_entity_column_scores=[{} for _ in specs],
                num_actual=0,
                num_expected=len(specs),
                per_entity_actual_values=[{} for _ in specs],
            )
        return result, initial, final

    def _evaluate_create(
        self,
        table_key: str,
        specs: list[EntityDiffSpec],
        initial: pl.DataFrame,
        final: pl.DataFrame,
    ) -> GroupEvalResult:
        """Evaluate create operations: anti-join final vs initial → new rows."""
        actual_new = _anti_join(final, initial)

        # Build expected rows from specs
        valid_specs = [s for s in specs if s.entity_data is not None]
        expected_dicts = [s.entity_data for s in valid_specs]
        if not expected_dicts:
            return GroupEvalResult(
                table_key=table_key,
                operation="create",
                precision=1.0 if actual_new.shape[0] == 0 else 0.0,
                recall=1.0,
                per_entity_scores=[],
                per_entity_column_scores=[],
                num_actual=actual_new.shape[0],
                num_expected=0,
                per_entity_actual_values=[],
            )

        expected_df = pl.DataFrame(expected_dicts)

        # Build per-spec column similarity measures so that each expected
        # entity uses its OWN similarity functions.  Previously we merged
        # all specs into one dict (last-write-wins), which silently broke
        # scoring when different entities needed different functions for
        # the same column (e.g. one entity ignores ``email`` while
        # another requires exact match).
        per_spec_col_sims: list[dict[str, ColumnSimilarityMeasureType]] = []
        for spec in valid_specs:
            spec_sims = dict(spec.column_similarity_measure)
            # Force-ignore runtime-assigned ID columns regardless of spec.
            for col in _AGENTSANDBOX_RUNTIME_ID_COLUMNS:
                if col in spec_sims:
                    spec_sims[col] = "column_ignore_similarity"
            # Force-ignore auto-generated columns.
            for col in _AUTO_GENERATED_COLUMNS:
                if col in spec_sims:
                    spec_sims[col] = "column_ignore_similarity"
            resolved = _resolve_column_similarities(spec_sims)
            # Add defaults for any column in expected not covered by spec.
            for col in expected_df.columns:
                if col not in resolved:
                    val = spec.entity_data.get(col) if spec.entity_data else None
                    name = _default_column_similarity_for_value(col, val)
                    resolved[col] = _resolve_column_similarities({col: name})[col]
            per_spec_col_sims.append(resolved)

        # Build a shared/fallback col_sims from the union of all per-spec
        # dicts (needed by flexible_row_matching for common_cols filtering
        # and early-return paths).
        col_sims: dict[str, ColumnSimilarityMeasureType] = {}
        for psc in per_spec_col_sims:
            col_sims.update(psc)

        result = flexible_row_matching(
            actual_new,
            expected_df,
            col_sims,
            per_expected_column_similarities=per_spec_col_sims,
        )
        return GroupEvalResult(
            table_key=table_key,
            operation="create",
            precision=result.precision,
            recall=result.recall,
            per_entity_scores=result.per_expected_scores,
            per_entity_column_scores=result.per_expected_column_scores,
            num_actual=result.num_actual,
            num_expected=result.num_expected,
            per_entity_actual_values=result.per_expected_actual_values,
        )

    def _evaluate_delete(
        self,
        table_key: str,
        specs: list[EntityDiffSpec],
        initial: pl.DataFrame,
        final: pl.DataFrame,
    ) -> GroupEvalResult:
        """Evaluate delete operations: anti-join initial vs final → deleted rows."""
        actual_deleted = _anti_join(initial, final)

        # Build expected deleted rows from specs (just ID columns)
        expected_dicts = []
        for s in specs:
            if s.entity_id_column is not None and s.entity_id_value is not None:
                expected_dicts.append({s.entity_id_column: s.entity_id_value})

        if not expected_dicts:
            return GroupEvalResult(
                table_key=table_key,
                operation="delete",
                precision=1.0 if actual_deleted.shape[0] == 0 else 0.0,
                recall=1.0,
                per_entity_scores=[],
                per_entity_column_scores=[],
                num_actual=actual_deleted.shape[0],
                num_expected=0,
                per_entity_actual_values=[],
            )

        expected_df = pl.DataFrame(expected_dicts)

        # For deletion, we only check ID column matching
        col_sims: dict[str, ColumnSimilarityMeasureType] = {}
        merged = self._resolve_col_sims(specs)
        col_sims.update(merged)
        for col in expected_df.columns:
            if col not in col_sims:
                col_sims[col] = column_exact_match_similarity

        result = flexible_row_matching(actual_deleted, expected_df, col_sims)
        return GroupEvalResult(
            table_key=table_key,
            operation="delete",
            precision=result.precision,
            recall=result.recall,
            per_entity_scores=result.per_expected_scores,
            per_entity_column_scores=result.per_expected_column_scores,
            num_actual=result.num_actual,
            num_expected=result.num_expected,
            per_entity_actual_values=result.per_expected_actual_values,
        )

    def _evaluate_update(
        self,
        table_key: str,
        specs: list[EntityDiffSpec],
        initial: pl.DataFrame,
        final: pl.DataFrame,
        execution_context: Any | None,
        bridge: Any | None,
    ) -> GroupEvalResult:
        """Evaluate update operations: check each entity by ID independently."""
        per_entity_scores: list[float] = []
        per_entity_column_scores: list[dict[str, float]] = []
        per_entity_actual_values: list[dict[str, Any]] = []

        for spec in specs:
            if (
                spec.entity_id_column is None
                or spec.entity_id_value is None
                or spec.expected_fields is None
            ):
                per_entity_scores.append(0.0)
                per_entity_column_scores.append({})
                per_entity_actual_values.append({})
                continue

            id_col = spec.entity_id_column
            id_val = spec.entity_id_value

            # Get the actual row from final state
            if id_col not in final.columns:
                per_entity_scores.append(0.0)
                per_entity_column_scores.append({})
                per_entity_actual_values.append({})
                continue

            actual_row = final.filter(pl.col(id_col) == id_val)
            if actual_row.shape[0] == 0:
                LOGGER.info(
                    "  Update %s=%r: no matching row in final state (cols=%s)",
                    id_col,
                    id_val,
                    final.columns,
                )
                per_entity_scores.append(0.0)
                per_entity_column_scores.append({})
                per_entity_actual_values.append({})
                continue

            # Build expected row
            expected_data = {**spec.expected_fields, id_col: id_val}
            expected_df = pl.DataFrame([expected_data])

            LOGGER.info(
                "  Update %s=%r: found %d row(s), expected_fields=%s",
                id_col,
                id_val,
                actual_row.shape[0],
                list(expected_data.keys()),
            )
            # Log actual values for expected columns
            for col in expected_data:
                if col in actual_row.columns:
                    LOGGER.info(
                        "    %s: actual=%r expected=%r",
                        col,
                        actual_row[col][0],
                        expected_data[col],
                    )

            # Resolve column similarities for this spec
            col_sims = _resolve_column_similarities(spec.column_similarity_measure)
            for col in expected_df.columns:
                if col not in col_sims:
                    val = expected_data.get(col)
                    name = _default_column_similarity_for_value(col, val)
                    col_sims[col] = _resolve_column_similarities({col: name})[col]

            # For update operations, bookkeeping timestamp columns use
            # column_datetime_similarity (strictly-greater-than) instead of
            # column_ignore_similarity to verify the entity was actually
            # modified.  The spec's timestamp should match the entity's
            # initial value; if the agent modified the entity, AppWorld sets
            # the timestamp to DateTime.now() (strictly later -> pass).
            #
            # Exception: gmail threads and drafts do NOT auto-update their
            # updated_at on save — the field stays at its creation-time value.
            # For these tables, timestamps remain ignored.
            _NO_AUTO_UPDATE_TABLES = {
                "gmail.global_email_threads",
                "gmail.user_email_threads",
                "gmail.drafts",
            }
            if table_key not in _NO_AUTO_UPDATE_TABLES:
                for col in expected_df.columns:
                    if col in _BOOKKEEPING_TIMESTAMP_COLUMNS:
                        # Only override if the spec did NOT explicitly set
                        # a similarity for this column (respect explicit
                        # column_ignore_similarity from the spec).
                        if col not in spec.column_similarity_measure:
                            col_sims[col] = column_datetime_similarity

            result = flexible_row_matching(actual_row, expected_df, col_sims)
            per_entity_scores.append(result.recall)
            per_entity_column_scores.append(
                result.per_expected_column_scores[0]
                if result.per_expected_column_scores
                else {}
            )
            per_entity_actual_values.append(
                result.per_expected_actual_values[0]
                if result.per_expected_actual_values
                else {}
            )

        num_expected = len(specs)
        avg_score = sum(per_entity_scores) / num_expected if num_expected > 0 else 0.0

        return GroupEvalResult(
            table_key=table_key,
            operation="update",
            precision=avg_score,
            recall=avg_score,
            per_entity_scores=per_entity_scores,
            per_entity_column_scores=per_entity_column_scores,
            num_actual=len(specs),  # For updates, actual = expected
            num_expected=num_expected,
            per_entity_actual_values=per_entity_actual_values,
        )

    def _check_guardrails(
        self,
        execution_context: Any | None,
        bridge: Any | None,
    ) -> tuple[bool, list[str]]:
        """Check that non-spec databases haven't changed."""
        failures: list[str] = []

        # --- MMToolSandbox guardrails ---
        spec_namespaces: set[str] = set()
        for spec in self._config.specs:
            if spec.source == "agentsandbox" and spec.namespace:
                spec_namespaces.add(spec.namespace)

        if execution_context is not None:
            active_namespaces = execution_context.get_active_database_namespaces()
            guard_namespaces = (
                active_namespaces
                - {DatabaseNamespace(n) for n in spec_namespaces if n}
                - _GUARDRAIL_EXCLUDED_NAMESPACES
            )

            for ns in guard_namespaces:
                try:
                    initial = read_agentsandbox_initial(execution_context, ns)
                    final = read_agentsandbox_final(execution_context, ns)
                    sim = guardrail_similarity(
                        snapshot=final.drop("sandbox_message_index")
                        if "sandbox_message_index" in final.columns
                        else final,
                        reference_snapshot=initial.drop("sandbox_message_index")
                        if "sandbox_message_index" in initial.columns
                        else initial,
                    )
                    if sim < 1.0:
                        failures.append(
                            f"MMToolSandbox guardrail failed for {ns}: "
                            f"database changed unexpectedly"
                        )
                except Exception as e:
                    LOGGER.debug("Guardrail check skipped for %s: %s", ns, e)

        # --- AppWorld guardrails ---
        spec_tables: set[str] = set()
        for spec in self._config.specs:
            if spec.source == "appworld" and spec.table:
                spec_tables.add(spec.table)

        guard_tables = set(self._initial_appworld.keys()) - spec_tables
        for table_key in guard_tables:
            try:
                initial = self._initial_appworld[table_key]
                final = read_appworld_table(table_key)

                for col in ["sandbox_message_index"]:
                    if col in initial.columns:
                        initial = initial.drop(col)
                    if col in final.columns:
                        final = final.drop(col)

                if initial.shape[0] != final.shape[0]:
                    failures.append(
                        f"AppWorld guardrail failed for {table_key}: "
                        f"row count changed ({initial.shape[0]} → {final.shape[0]})"
                    )
                    continue

                id_checked = False
                for id_col in _ID_COLUMN_CANDIDATES:
                    if id_col in initial.columns and id_col in final.columns:
                        initial_ids = set(initial[id_col].cast(pl.Utf8).to_list())
                        final_ids = set(final[id_col].cast(pl.Utf8).to_list())
                        if initial_ids != final_ids:
                            failures.append(
                                f"AppWorld guardrail failed for {table_key}: "
                                f"IDs changed on column {id_col}"
                            )
                        id_checked = True
                        break

                if not id_checked:
                    sim = guardrail_similarity(
                        snapshot=final,
                        reference_snapshot=initial,
                    )
                    if sim < 1.0:
                        failures.append(
                            f"AppWorld guardrail failed for {table_key}: "
                            f"table changed unexpectedly"
                        )
            except Exception as e:
                LOGGER.debug("Guardrail check skipped for %s: %s", table_key, e)

        return len(failures) == 0, failures
