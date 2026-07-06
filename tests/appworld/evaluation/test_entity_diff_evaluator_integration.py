"""Integration tests for EntityDiffEvaluator against real AppWorld databases.

These tests load the AppWorld ``_base_`` task, discover ALL tables across
ALL apps dynamically, then verify that read/diff/eval works correctly on
every one — including large tables (gmail.emails 20K+, file_system.files
23K+, gmail.user_email_threads 47K+).

Requires AppWorld to be installed and data downloaded.  Skipped
automatically when AppWorld is not available.
"""

from __future__ import annotations

from typing import Any, Dict, List

import polars as pl
import pytest

# Skip the entire module if AppWorld is not installed.
try:
    from mmtoolsandbox.appworld import APPWORLD_AVAILABLE

    if not APPWORLD_AVAILABLE:
        raise ImportError("AppWorld not available")
    from mmtoolsandbox.appworld.bridge import (
        AppWorldBridge,
        get_appworld_bridge,
    )
except ImportError:
    pytest.skip(
        "AppWorld not installed — skipping integration tests",
        allow_module_level=True,
    )

from mmtoolsandbox.common.entity_diff_evaluator import (
    EntityDiffEvalConfig,
    EntityDiffEvaluator,
    EntityDiffSpec,
    _anti_join,
    _get_engine_from_bridge,
    read_appworld_table,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _discover_all_tables(bridge: AppWorldBridge) -> List[str]:
    """Discover every (app, table) pair from the initialized base task.

    Excludes FTS shadow tables (*_fts, *_fts_config, etc.) and the
    ``supervisor`` app (always empty in base).
    """
    from sqlalchemy import inspect, text

    _FTS_SUFFIXES = (
        "_fts",
        "_fts_config",
        "_fts_content",
        "_fts_data",
        "_fts_docsize",
        "_fts_idx",
    )
    _SKIP_APPS = {"supervisor"}  # always empty in base task

    tables: List[str] = []
    models = bridge.appworld.models
    for app_name in sorted(dir(models)):
        if app_name.startswith("_") or app_name in _SKIP_APPS:
            continue
        engine = _get_engine_from_bridge(bridge, app_name)
        if engine is None:
            continue
        inspector = inspect(engine)
        for tbl in sorted(inspector.get_table_names()):
            if any(tbl.endswith(suffix) for suffix in _FTS_SUFFIXES):
                continue
            # Skip empty tables (nothing to diff)
            with engine.connect() as conn:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
            if count and count > 0:
                tables.append(f"{app_name}.{tbl}")
    return tables


# Module-scoped bridge and table list (initialized once).
_bridge_instance: AppWorldBridge | None = None
_all_tables: List[str] = []


def _get_bridge() -> AppWorldBridge:
    global _bridge_instance
    if _bridge_instance is None:
        try:
            _bridge_instance = get_appworld_bridge()
            _bridge_instance.initialize("_base_")
        except Exception as e:
            pytest.skip(
                f"AppWorld bridge initialization failed (data not set up?): {e}",
                allow_module_level=True,
            )
    return _bridge_instance


def _get_all_tables() -> List[str]:
    global _all_tables
    if not _all_tables:
        _all_tables = _discover_all_tables(_get_bridge())
    return _all_tables


@pytest.fixture(scope="module")
def bridge() -> AppWorldBridge:  # type: ignore[misc]
    """Initialize AppWorld with the _base_ task once for the whole module."""
    try:
        b = _get_bridge()
    except Exception as e:
        pytest.skip(f"AppWorld bridge initialization failed: {e}")
    yield b
    b.close()
    AppWorldBridge.reset()
    global _bridge_instance
    _bridge_instance = None


def _all_table_ids() -> List[str]:
    """Return table IDs for parametrize (called at collection time)."""
    try:
        return _get_all_tables()
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_table(table_key: str) -> pl.DataFrame:
    """Read an AppWorld table, stripping the dummy sandbox_message_index."""
    df = read_appworld_table(table_key)
    if "sandbox_message_index" in df.columns:
        df = df.drop("sandbox_message_index")
    return df


def _insert_row(
    bridge: AppWorldBridge,
    table_key: str,
    row: Dict[str, Any],
) -> None:
    """INSERT a single row into an AppWorld table via the bridge."""
    app_name, table_name = table_key.split(".", 1)
    cols = list(row.keys())
    placeholders = ", ".join(["?" for _ in cols])
    col_str = ", ".join(cols)
    sql = f"INSERT INTO {table_name} ({col_str}) VALUES ({placeholders})"
    params = [row[c] for c in cols]
    bridge.apply_database_modifications([(app_name, sql, params)])


def _update_row(
    bridge: AppWorldBridge,
    table_key: str,
    id_col: str,
    id_val: Any,
    updates: Dict[str, Any],
) -> None:
    """UPDATE a single row in an AppWorld table via the bridge."""
    app_name, table_name = table_key.split(".", 1)
    set_clause = ", ".join([f"{k} = ?" for k in updates])
    sql = f"UPDATE {table_name} SET {set_clause} WHERE {id_col} = ?"
    params = list(updates.values()) + [id_val]
    bridge.apply_database_modifications([(app_name, sql, params)])


def _delete_row(
    bridge: AppWorldBridge,
    table_key: str,
    id_col: str,
    id_val: Any,
) -> None:
    """DELETE a single row from an AppWorld table via the bridge."""
    app_name, table_name = table_key.split(".", 1)
    sql = f"DELETE FROM {table_name} WHERE {id_col} = ?"
    bridge.apply_database_modifications([(app_name, sql, [id_val])])


# ---------------------------------------------------------------------------
# Tests: read every table into polars
# ---------------------------------------------------------------------------


class TestReadAllTables:
    """Verify we can read every non-empty AppWorld table into polars."""

    @pytest.mark.parametrize("table_key", _all_table_ids())
    def test_read_table(self, bridge: AppWorldBridge, table_key: str) -> None:
        """Table should be readable as a non-empty polars DataFrame."""
        df = _read_table(table_key)
        assert isinstance(df, pl.DataFrame), f"{table_key}: not a DataFrame"
        assert df.shape[0] > 0, f"{table_key}: empty"
        assert df.shape[1] > 0, f"{table_key}: no columns"


# ---------------------------------------------------------------------------
# Tests: anti-join self-consistency on every table
# ---------------------------------------------------------------------------


class TestAntiJoinAllTables:
    """Anti-join of a table with itself must be empty for every table."""

    @pytest.mark.parametrize("table_key", _all_table_ids())
    def test_self_diff_is_empty(self, bridge: AppWorldBridge, table_key: str) -> None:
        df = _read_table(table_key)
        diff = _anti_join(df, df)
        assert diff.shape[0] == 0, (
            f"{table_key}: self anti-join produced {diff.shape[0]} rows "
            f"(table has {df.shape[0]} rows, {df.shape[1]} cols)"
        )

    @pytest.mark.parametrize("table_key", _all_table_ids())
    def test_drop_one_row_detected(
        self, bridge: AppWorldBridge, table_key: str
    ) -> None:
        """Removing one row from the right side should produce exactly 1 diff."""
        df = _read_table(table_key)
        if "id" not in df.columns:
            pytest.skip(f"{table_key}: no 'id' column")

        first_id = df["id"][0]
        without_first = df.filter(pl.col("id") != first_id)
        diff = _anti_join(df, without_first)
        assert diff.shape[0] == 1, (
            f"{table_key}: expected 1 diff row, got {diff.shape[0]}"
        )


# ---------------------------------------------------------------------------
# Tests: full evaluator against real AppWorld
# ---------------------------------------------------------------------------


class TestEvaluatorRealData:
    """End-to-end create/update/delete evaluation on real databases."""

    def test_create_eval(self, bridge: AppWorldBridge) -> None:
        """Insert a note, evaluate create spec → recall=1.0."""
        table_key = "simple_note.notes"
        before = _read_table(table_key)
        new_id = int(before["id"].max()) + 99001  # type: ignore[arg-type]

        spec = EntityDiffSpec(
            operation="create",
            source="appworld",
            table=table_key,
            entity_data={"id": new_id, "title": "__eval_create_test__"},
            column_similarity_measure={
                "id": "column_exact_match_similarity",
                "title": "column_exact_match_similarity",
            },
        )
        evaluator = EntityDiffEvaluator(EntityDiffEvalConfig(specs=[spec]))
        evaluator.capture_initial_state(bridge=bridge)

        _insert_row(
            bridge,
            table_key,
            {
                "id": new_id,
                "user_id": 1,
                "title": "__eval_create_test__",
                "content": "test",
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
            },
        )

        result = evaluator.evaluate(bridge=bridge)
        gr = result.group_results[0]
        assert gr.operation == "create"
        assert gr.recall == pytest.approx(1.0)
        assert gr.num_actual >= 1

        _delete_row(bridge, table_key, "id", new_id)

    def test_update_eval(self, bridge: AppWorldBridge) -> None:
        """Update a note's title, evaluate update spec → recall=1.0."""
        table_key = "simple_note.notes"
        before = _read_table(table_key)
        target_id = before["id"][0]
        original_title = before.filter(pl.col("id") == target_id)["title"][0]

        spec = EntityDiffSpec(
            operation="update",
            source="appworld",
            table=table_key,
            entity_id_column="id",
            entity_id_value=target_id,
            expected_fields={"title": "__eval_update_test__"},
            column_similarity_measure={
                "id": "column_exact_match_similarity",
                "title": "column_exact_match_similarity",
            },
        )
        evaluator = EntityDiffEvaluator(EntityDiffEvalConfig(specs=[spec]))
        evaluator.capture_initial_state(bridge=bridge)

        _update_row(
            bridge,
            table_key,
            "id",
            target_id,
            {
                "title": "__eval_update_test__",
            },
        )

        result = evaluator.evaluate(bridge=bridge)
        assert result.group_results[0].recall == pytest.approx(1.0)

        _update_row(bridge, table_key, "id", target_id, {"title": original_title})

    def test_delete_eval(self, bridge: AppWorldBridge) -> None:
        """Insert then delete a note, evaluate delete spec → recall=1.0."""
        table_key = "simple_note.notes"
        before = _read_table(table_key)
        temp_id = int(before["id"].max()) + 99002  # type: ignore[arg-type]

        _insert_row(
            bridge,
            table_key,
            {
                "id": temp_id,
                "user_id": 1,
                "title": "__eval_delete_target__",
                "content": "to be deleted",
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
            },
        )

        spec = EntityDiffSpec(
            operation="delete",
            source="appworld",
            table=table_key,
            entity_id_column="id",
            entity_id_value=temp_id,
            column_similarity_measure={"id": "column_exact_match_similarity"},
        )
        evaluator = EntityDiffEvaluator(EntityDiffEvalConfig(specs=[spec]))
        evaluator.capture_initial_state(bridge=bridge)

        _delete_row(bridge, table_key, "id", temp_id)

        result = evaluator.evaluate(bridge=bridge)
        assert result.group_results[0].recall == pytest.approx(1.0)

    def test_create_not_performed(self, bridge: AppWorldBridge) -> None:
        """Expect a create but don't do it → recall=0."""
        spec = EntityDiffSpec(
            operation="create",
            source="appworld",
            table="simple_note.notes",
            entity_data={"id": 999999, "title": "__never_created__"},
            column_similarity_measure={
                "id": "column_exact_match_similarity",
                "title": "column_exact_match_similarity",
            },
        )
        evaluator = EntityDiffEvaluator(EntityDiffEvalConfig(specs=[spec]))
        evaluator.capture_initial_state(bridge=bridge)

        result = evaluator.evaluate(bridge=bridge)
        assert result.group_results[0].recall == pytest.approx(0.0)

    def test_cross_app_multi_op(self, bridge: AppWorldBridge) -> None:
        """Create in simple_note + update in todoist → both evaluated."""
        note_table = "simple_note.notes"
        before_notes = _read_table(note_table)
        new_note_id = int(before_notes["id"].max()) + 99003  # type: ignore[arg-type]

        todoist_table = "todoist.tasks"
        before_tasks = _read_table(todoist_table)
        task_id = before_tasks["id"][0]
        original_title = before_tasks.filter(pl.col("id") == task_id)["title"][0]

        specs = [
            EntityDiffSpec(
                operation="create",
                source="appworld",
                table=note_table,
                entity_data={"id": new_note_id, "title": "__cross_app_note__"},
                column_similarity_measure={
                    "id": "column_exact_match_similarity",
                    "title": "column_exact_match_similarity",
                },
            ),
            EntityDiffSpec(
                operation="update",
                source="appworld",
                table=todoist_table,
                entity_id_column="id",
                entity_id_value=task_id,
                expected_fields={"title": "__cross_app_task__"},
                column_similarity_measure={
                    "id": "column_exact_match_similarity",
                    "title": "column_exact_match_similarity",
                },
            ),
        ]

        evaluator = EntityDiffEvaluator(EntityDiffEvalConfig(specs=specs))
        evaluator.capture_initial_state(bridge=bridge)

        _insert_row(
            bridge,
            note_table,
            {
                "id": new_note_id,
                "user_id": 1,
                "title": "__cross_app_note__",
                "content": "test",
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
            },
        )
        _update_row(
            bridge,
            todoist_table,
            "id",
            task_id,
            {
                "title": "__cross_app_task__",
            },
        )

        result = evaluator.evaluate(bridge=bridge)
        assert len(result.group_results) == 2
        assert result.overall_recall == pytest.approx(1.0)

        _delete_row(bridge, note_table, "id", new_note_id)
        _update_row(
            bridge,
            todoist_table,
            "id",
            task_id,
            {
                "title": original_title,
            },
        )
