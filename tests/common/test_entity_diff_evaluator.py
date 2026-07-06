# Copyright © 2026 Apple Inc.

"""Tests for EntityDiffEvaluator and supporting functions.

All tests use mock polars DataFrames — no AppWorld dependency required.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import polars as pl
import pytest

from mmtoolsandbox.common.entity_diff_evaluator import (
    EntityDiffEvalConfig,
    EntityDiffEvalResult,
    EntityDiffEvaluator,
    EntityDiffSpec,
    GroupEvalResult,
    flexible_row_matching,
)
from mmtoolsandbox.common.evaluation import (
    ColumnSimilarityMeasureType,
    column_datetime_similarity,
    column_exact_match_similarity,
    column_ignore_similarity,
    column_rouge_l_similarity,
)

# ---------------------------------------------------------------------------
# column_datetime_similarity tests
# ---------------------------------------------------------------------------


class TestColumnAfterSimilarity:
    """Tests for column_datetime_similarity (strictly-greater-than check)."""

    def test_same_time_returns_zero(self) -> None:
        """Same datetime → 0.0 (must be strictly later, not equal)."""
        df = pl.DataFrame({"updated_at": ["2026-03-25T22:03:10"]})
        result = column_datetime_similarity(df, "updated_at", "2026-03-25T22:03:10")
        assert result["similarity"][0] == pytest.approx(0.0)

    def test_later_time_returns_one(self) -> None:
        """Later datetime → 1.0."""
        df = pl.DataFrame({"updated_at": ["2026-04-01T00:00:00"]})
        result = column_datetime_similarity(df, "updated_at", "2026-03-25T00:00:00")
        assert result["similarity"][0] == pytest.approx(1.0)

    def test_earlier_time_returns_zero(self) -> None:
        """Earlier datetime → 0.0."""
        df = pl.DataFrame({"updated_at": ["2026-03-20T00:00:00"]})
        result = column_datetime_similarity(df, "updated_at", "2026-03-25T00:00:00")
        assert result["similarity"][0] == pytest.approx(0.0)

    def test_no_z_vs_z_same_returns_zero(self) -> None:
        """AppWorld stores no Z, spec has Z — same second → 0.0 (not strictly later)."""
        df = pl.DataFrame({"updated_at": ["2026-03-25T22:03:10"]})
        result = column_datetime_similarity(df, "updated_at", "2026-03-25T22:03:10Z")
        assert result["similarity"][0] == pytest.approx(0.0)

    def test_no_z_vs_z_earlier_returns_zero(self) -> None:
        """Earlier time, mixed Z → 0.0."""
        df = pl.DataFrame({"updated_at": ["2026-03-25T20:00:00"]})
        result = column_datetime_similarity(df, "updated_at", "2026-03-25T22:03:10Z")
        assert result["similarity"][0] == pytest.approx(0.0)

    def test_offset_vs_z_same_returns_zero(self) -> None:
        """'+00:00' offset vs 'Z' — same instant → 0.0 (not strictly later)."""
        df = pl.DataFrame({"updated_at": ["2026-03-25T22:03:10+00:00"]})
        result = column_datetime_similarity(df, "updated_at", "2026-03-25T22:03:10Z")
        assert result["similarity"][0] == pytest.approx(0.0)

    def test_sqlite_space_later(self) -> None:
        """SQLite format, later timestamp → 1.0."""
        df = pl.DataFrame({"updated_at": ["2026-03-25 22:03:11.926011"]})
        result = column_datetime_similarity(df, "updated_at", "2026-03-25T22:03:10Z")
        assert result["similarity"][0] == pytest.approx(1.0)

    def test_sqlite_space_different_day_earlier(self) -> None:
        """SQLite space-separated, earlier day → 0.0."""
        df = pl.DataFrame({"updated_at": ["2026-03-24 10:00:00"]})
        result = column_datetime_similarity(df, "updated_at", "2026-03-25T22:03:10Z")
        assert result["similarity"][0] == pytest.approx(0.0)

    def test_none_value_returns_one(self) -> None:
        """None reference → always passes."""
        df = pl.DataFrame({"updated_at": ["anything"]})
        result = column_datetime_similarity(df, "updated_at", None)
        assert result["similarity"][0] == pytest.approx(1.0)

    def test_used_in_flexible_matching(self) -> None:
        """updated_at with column_datetime_similarity: later time → 1.0."""
        actual = pl.DataFrame(
            {
                "id": [1],
                "updated_at": ["2026-03-26 10:00:00.123"],
            }
        )
        expected = pl.DataFrame(
            {
                "id": [1],
                "updated_at": ["2026-03-25T22:03:10Z"],
            }
        )
        col_sims: Dict[str, ColumnSimilarityMeasureType] = {
            "id": column_exact_match_similarity,
            "updated_at": column_datetime_similarity,
        }
        result = flexible_row_matching(actual, expected, col_sims)
        assert result.recall == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# flexible_row_matching tests
# ---------------------------------------------------------------------------


class TestFlexibleRowMatching:
    """Tests for the core flexible_row_matching function."""

    def test_perfect_match_equal_rows(self) -> None:
        """M=N, rows match perfectly → precision=1, recall=1."""
        actual = pl.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})
        expected = pl.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})
        col_sims = {
            "name": column_exact_match_similarity,
            "age": column_exact_match_similarity,
        }

        result = flexible_row_matching(actual, expected, col_sims)

        assert result.num_actual == 2
        assert result.num_expected == 2
        assert result.num_matched == 2
        assert result.recall == pytest.approx(1.0)
        assert result.precision == pytest.approx(1.0)
        assert len(result.per_expected_scores) == 2
        assert all(s == pytest.approx(1.0) for s in result.per_expected_scores)

    def test_extra_actual_rows(self) -> None:
        """M>N (extra actual rows) → recall=1, precision<1."""
        actual = pl.DataFrame(
            {"name": ["Alice", "Bob", "Charlie"], "age": [30, 25, 40]}
        )
        expected = pl.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})
        col_sims = {
            "name": column_exact_match_similarity,
            "age": column_exact_match_similarity,
        }

        result = flexible_row_matching(actual, expected, col_sims)

        assert result.num_actual == 3
        assert result.num_expected == 2
        assert result.recall == pytest.approx(1.0)
        # precision = 2/3 (2 matched out of 3 actual)
        assert result.precision == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_missing_actual_rows(self) -> None:
        """M<N (missing actual rows) → recall<1, precision=1."""
        actual = pl.DataFrame({"name": ["Alice"], "age": [30]})
        expected = pl.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})
        col_sims = {
            "name": column_exact_match_similarity,
            "age": column_exact_match_similarity,
        }

        result = flexible_row_matching(actual, expected, col_sims)

        assert result.num_actual == 1
        assert result.num_expected == 2
        assert result.precision == pytest.approx(1.0)
        # recall = 1/2 (1 matched out of 2 expected)
        assert result.recall == pytest.approx(0.5)

    def test_partial_match(self) -> None:
        """Rows partially match (some columns differ with exact match → score 0)."""
        actual = pl.DataFrame({"name": ["Alice"], "age": [31]})
        expected = pl.DataFrame({"name": ["Alice"], "age": [30]})
        col_sims = {
            "name": column_exact_match_similarity,
            "age": column_exact_match_similarity,
        }

        result = flexible_row_matching(actual, expected, col_sims)

        # name matches (1.0) but age doesn't (0.0).
        # Arithmetic mean over non-ignored columns: (1.0 + 0.0) / 2 = 0.5
        assert result.per_expected_scores[0] == pytest.approx(0.5)
        assert result.per_expected_column_scores[0]["name"] == pytest.approx(1.0)
        assert result.per_expected_column_scores[0]["age"] == pytest.approx(0.0)

    def test_empty_expected(self) -> None:
        """No expected rows → recall=1, precision depends on actual."""
        actual = pl.DataFrame({"name": ["Alice"]})
        expected = pl.DataFrame({"name": pl.Series([], dtype=pl.Utf8)})
        col_sims = {"name": column_exact_match_similarity}

        result = flexible_row_matching(actual, expected, col_sims)

        assert result.num_expected == 0
        assert result.recall == 1.0
        assert result.precision == 0.0

    def test_empty_actual(self) -> None:
        """No actual rows → precision=0, recall=0."""
        actual = pl.DataFrame({"name": pl.Series([], dtype=pl.Utf8)})
        expected = pl.DataFrame({"name": ["Alice"]})
        col_sims = {"name": column_exact_match_similarity}

        result = flexible_row_matching(actual, expected, col_sims)

        assert result.num_actual == 0
        assert result.precision == 0.0
        assert result.recall == 0.0

    def test_both_empty(self) -> None:
        """Both empty → precision=1, recall=1."""
        actual = pl.DataFrame({"name": pl.Series([], dtype=pl.Utf8)})
        expected = pl.DataFrame({"name": pl.Series([], dtype=pl.Utf8)})
        col_sims = {"name": column_exact_match_similarity}

        result = flexible_row_matching(actual, expected, col_sims)

        assert result.precision == 1.0
        assert result.recall == 1.0

    def test_rouge_similarity(self) -> None:
        """Test with ROUGE-L similarity for text columns."""
        actual = pl.DataFrame({"title": ["Buy groceries from the store"]})
        expected = pl.DataFrame({"title": ["Buy groceries"]})
        col_sims: Dict[str, ColumnSimilarityMeasureType] = {
            "title": column_rouge_l_similarity,
        }

        result = flexible_row_matching(actual, expected, col_sims)

        assert result.per_expected_scores[0] > 0.5  # Partial text match
        assert result.per_expected_scores[0] < 1.0  # Not perfect

    def test_hungarian_optimal_assignment(self) -> None:
        """Verify optimal matching (not greedy)."""
        actual = pl.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        expected = pl.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        col_sims = {
            "id": column_exact_match_similarity,
            "name": column_exact_match_similarity,
        }

        result = flexible_row_matching(actual, expected, col_sims)

        # Perfect 1-to-1 match: row 0↔0, row 1↔1
        assert result.recall == pytest.approx(1.0)
        assert result.precision == pytest.approx(1.0)

    def test_per_expected_column_similarities(self) -> None:
        """Per-entity similarity overrides: entity 0 ignores email, entity 1
        requires exact match.  Without per-expected overrides the merge bug
        would apply the last spec's measure to ALL entities."""
        actual = pl.DataFrame(
            {
                "name": ["Alice", "CFPB"],
                "email": ["alice@hallucinated.com", "cfpb@gov.com"],
            }
        )
        expected = pl.DataFrame(
            {
                "name": ["Alice", "CFPB"],
                "email": ["", "cfpb@gov.com"],
            }
        )
        # Shared fallback uses exact match for everything.
        shared_col_sims: Dict[str, ColumnSimilarityMeasureType] = {
            "name": column_exact_match_similarity,
            "email": column_exact_match_similarity,
        }
        # Per-expected: entity 0 ignores email, entity 1 uses exact match.
        per_expected: list[Dict[str, ColumnSimilarityMeasureType]] = [
            {
                "name": column_exact_match_similarity,
                "email": column_ignore_similarity,
            },
            {
                "name": column_exact_match_similarity,
                "email": column_exact_match_similarity,
            },
        ]

        result = flexible_row_matching(
            actual,
            expected,
            shared_col_sims,
            per_expected_column_similarities=per_expected,
        )

        # Both entities should score perfectly:
        # - entity 0: name=exact match (1.0), email=ignored (1.0) → 1.0
        # - entity 1: name=exact match (1.0), email=exact match (1.0) → 1.0
        assert result.recall == pytest.approx(1.0)
        assert result.precision == pytest.approx(1.0)
        assert result.per_expected_scores[0] == pytest.approx(1.0)
        assert result.per_expected_scores[1] == pytest.approx(1.0)
        # Entity 0's email should be 1.0 (ignored), not 0.0.
        assert result.per_expected_column_scores[0]["email"] == pytest.approx(1.0)
        assert result.per_expected_column_scores[1]["email"] == pytest.approx(1.0)

    def test_per_expected_without_override_fails(self) -> None:
        """Without per-expected overrides, the merged dict causes entity 0
        to use exact match for email (last-write-wins) → score near zero."""
        actual = pl.DataFrame(
            {
                "name": ["Alice", "CFPB"],
                "email": ["alice@hallucinated.com", "cfpb@gov.com"],
            }
        )
        expected = pl.DataFrame(
            {
                "name": ["Alice", "CFPB"],
                "email": ["", "cfpb@gov.com"],
            }
        )
        # Simulates the old merged dict: exact match wins for email.
        merged_col_sims: Dict[str, ColumnSimilarityMeasureType] = {
            "name": column_exact_match_similarity,
            "email": column_exact_match_similarity,
        }

        result = flexible_row_matching(actual, expected, merged_col_sims)

        # Entity 0 email mismatch ("" vs "alice@hallucinated.com") → near-zero.
        assert result.per_expected_column_scores[0]["email"] == pytest.approx(0.0)
        # Entity 1 email matches → 1.0.
        assert result.per_expected_column_scores[1]["email"] == pytest.approx(1.0)
        # Overall score should be degraded because of entity 0.
        # Arithmetic mean: (1.0 + 0.0) / 2 = 0.5
        assert result.per_expected_scores[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# EntityDiffSpec tests
# ---------------------------------------------------------------------------


class TestEntityDiffSpec:
    """Tests for EntityDiffSpec serialization."""

    def test_round_trip(self) -> None:
        """Serialize and deserialize an EntityDiffSpec."""
        spec = EntityDiffSpec(
            operation="create",
            source="agentsandbox",
            namespace="REMINDER",
            entity_data={"reminder_id": "r1", "title": "Test"},
            column_similarity_measure={
                "reminder_id": "column_exact_match_similarity",
                "title": "column_rouge_l_similarity",
            },
        )
        d = spec.to_dict()
        restored = EntityDiffSpec.from_dict(d)

        assert restored.operation == "create"
        assert restored.source == "agentsandbox"
        assert restored.namespace == "REMINDER"
        assert restored.entity_data == {"reminder_id": "r1", "title": "Test"}
        assert restored.table_key == "REMINDER"

    def test_appworld_table_key(self) -> None:
        spec = EntityDiffSpec(
            operation="delete",
            source="appworld",
            table="simple_note.notes",
            entity_id_column="id",
            entity_id_value=42,
        )
        assert spec.table_key == "simple_note.notes"


# ---------------------------------------------------------------------------
# EntityDiffEvalConfig tests
# ---------------------------------------------------------------------------


class TestEntityDiffEvalConfig:
    def test_round_trip(self) -> None:
        config = EntityDiffEvalConfig(
            specs=[
                EntityDiffSpec(
                    operation="create",
                    source="agentsandbox",
                    namespace="NOTES",
                    entity_data={"note_id": "n1", "title": "Hello"},
                    column_similarity_measure={
                        "title": "column_rouge_l_similarity",
                    },
                ),
            ]
        )
        d = config.to_dict()
        restored = EntityDiffEvalConfig.from_dict(d)
        assert len(restored.specs) == 1
        assert restored.specs[0].namespace == "NOTES"


# ---------------------------------------------------------------------------
# EntityDiffEvaluator tests (with mock execution context)
# ---------------------------------------------------------------------------


def _mock_execution_context(
    initial_dbs: Dict[str, pl.DataFrame],
    final_dbs: Dict[str, pl.DataFrame],
    active_namespaces: set[str] | None = None,
) -> MagicMock:
    """Create a mock ExecutionContext returning prescribed snapshots."""
    from mmtoolsandbox.common.databases import DatabaseNamespace

    ctx = MagicMock()
    ctx.first_user_sandbox_message_index = 1

    if active_namespaces is None:
        active_namespaces = set(initial_dbs.keys()) | set(final_dbs.keys())

    ctx.get_active_database_namespaces.return_value = {
        DatabaseNamespace(ns) for ns in active_namespaces
    }

    def get_database(
        namespace: Any,
        sandbox_message_index: Any = None,
        drop_sandbox_message_index: bool = True,
        drop_headguard: bool = True,
    ) -> pl.DataFrame:
        ns_str = str(namespace)
        if sandbox_message_index is not None:
            # Initial snapshot
            return initial_dbs.get(ns_str, pl.DataFrame())
        # Final snapshot
        return final_dbs.get(ns_str, pl.DataFrame())

    ctx.get_database.side_effect = get_database
    return ctx


class TestEntityDiffEvaluatorCreate:
    """Test create operation evaluation."""

    def test_single_create_success(self) -> None:
        """Agent creates exactly the expected row."""
        initial = pl.DataFrame(
            {"note_id": ["n0"], "title": ["Old note"], "sandbox_message_index": [0]}
        )
        final = pl.DataFrame(
            {
                "note_id": ["n0", "n1"],
                "title": ["Old note", "Buy milk"],
                "sandbox_message_index": [0, 2],
            }
        )

        ctx = _mock_execution_context(
            {"NOTES": initial},
            {"NOTES": final},
            active_namespaces={"NOTES"},
        )

        config = EntityDiffEvalConfig(
            specs=[
                EntityDiffSpec(
                    operation="create",
                    source="agentsandbox",
                    namespace="NOTES",
                    entity_data={"note_id": "n1", "title": "Buy milk"},
                    column_similarity_measure={
                        "note_id": "column_exact_match_similarity",
                        "title": "column_rouge_l_similarity",
                    },
                )
            ]
        )

        evaluator = EntityDiffEvaluator(config)
        result = evaluator.evaluate(execution_context=ctx)

        assert len(result.group_results) == 1
        gr = result.group_results[0]
        assert gr.operation == "create"
        assert gr.recall == pytest.approx(1.0)
        assert gr.num_expected == 1
        assert gr.num_actual == 1

    def test_multiple_creates(self) -> None:
        """Agent creates two rows, both expected."""
        initial = pl.DataFrame(
            {
                "note_id": pl.Series([], dtype=pl.Utf8),
                "title": pl.Series([], dtype=pl.Utf8),
                "sandbox_message_index": pl.Series([], dtype=pl.Int64),
            }
        )
        final = pl.DataFrame(
            {
                "note_id": ["n1", "n2"],
                "title": ["First", "Second"],
                "sandbox_message_index": [2, 3],
            }
        )

        ctx = _mock_execution_context(
            {"NOTES": initial},
            {"NOTES": final},
            active_namespaces={"NOTES"},
        )

        config = EntityDiffEvalConfig(
            specs=[
                EntityDiffSpec(
                    operation="create",
                    source="agentsandbox",
                    namespace="NOTES",
                    entity_data={"note_id": "n1", "title": "First"},
                    column_similarity_measure={
                        "note_id": "column_exact_match_similarity",
                        "title": "column_exact_match_similarity",
                    },
                ),
                EntityDiffSpec(
                    operation="create",
                    source="agentsandbox",
                    namespace="NOTES",
                    entity_data={"note_id": "n2", "title": "Second"},
                    column_similarity_measure={
                        "note_id": "column_exact_match_similarity",
                        "title": "column_exact_match_similarity",
                    },
                ),
            ]
        )

        evaluator = EntityDiffEvaluator(config)
        result = evaluator.evaluate(execution_context=ctx)

        gr = result.group_results[0]
        assert gr.recall == pytest.approx(1.0)
        assert gr.precision == pytest.approx(1.0)
        assert gr.num_expected == 2


class TestEntityDiffEvaluatorDelete:
    """Test delete operation evaluation."""

    def test_single_delete_success(self) -> None:
        """Agent deletes the expected row."""
        initial = pl.DataFrame(
            {
                "reminder_id": ["r1", "r2"],
                "title": ["Keep", "Delete me"],
                "sandbox_message_index": [0, 0],
            }
        )
        final = pl.DataFrame(
            {
                "reminder_id": ["r1"],
                "title": ["Keep"],
                "sandbox_message_index": [0],
            }
        )

        ctx = _mock_execution_context(
            {"REMINDER": initial},
            {"REMINDER": final},
            active_namespaces={"REMINDER"},
        )

        config = EntityDiffEvalConfig(
            specs=[
                EntityDiffSpec(
                    operation="delete",
                    source="agentsandbox",
                    namespace="REMINDER",
                    entity_id_column="reminder_id",
                    entity_id_value="r2",
                    column_similarity_measure={
                        "reminder_id": "column_exact_match_similarity",
                    },
                )
            ]
        )

        evaluator = EntityDiffEvaluator(config)
        result = evaluator.evaluate(execution_context=ctx)

        gr = result.group_results[0]
        assert gr.operation == "delete"
        assert gr.recall == pytest.approx(1.0)

    def test_delete_not_performed(self) -> None:
        """Agent fails to delete — row still present."""
        initial = pl.DataFrame(
            {
                "reminder_id": ["r1", "r2"],
                "title": ["Keep", "Should delete"],
                "sandbox_message_index": [0, 0],
            }
        )
        # Final same as initial — nothing deleted
        final = initial.clone()

        ctx = _mock_execution_context(
            {"REMINDER": initial},
            {"REMINDER": final},
            active_namespaces={"REMINDER"},
        )

        config = EntityDiffEvalConfig(
            specs=[
                EntityDiffSpec(
                    operation="delete",
                    source="agentsandbox",
                    namespace="REMINDER",
                    entity_id_column="reminder_id",
                    entity_id_value="r2",
                    column_similarity_measure={
                        "reminder_id": "column_exact_match_similarity",
                    },
                )
            ]
        )

        evaluator = EntityDiffEvaluator(config)
        result = evaluator.evaluate(execution_context=ctx)

        gr = result.group_results[0]
        assert gr.recall == pytest.approx(0.0)


class TestEntityDiffEvaluatorUpdate:
    """Test update operation evaluation."""

    def test_single_update_success(self) -> None:
        """Agent updates one field successfully."""
        initial = pl.DataFrame(
            {
                "note_id": ["n1"],
                "title": ["Old title"],
                "sandbox_message_index": [0],
            }
        )
        final = pl.DataFrame(
            {
                "note_id": ["n1"],
                "title": ["New title"],
                "sandbox_message_index": [2],
            }
        )

        ctx = _mock_execution_context(
            {"NOTES": initial},
            {"NOTES": final},
            active_namespaces={"NOTES"},
        )

        config = EntityDiffEvalConfig(
            specs=[
                EntityDiffSpec(
                    operation="update",
                    source="agentsandbox",
                    namespace="NOTES",
                    entity_id_column="note_id",
                    entity_id_value="n1",
                    expected_fields={"title": "New title"},
                    column_similarity_measure={
                        "note_id": "column_exact_match_similarity",
                        "title": "column_exact_match_similarity",
                    },
                )
            ]
        )

        evaluator = EntityDiffEvaluator(config)
        result = evaluator.evaluate(execution_context=ctx)

        gr = result.group_results[0]
        assert gr.operation == "update"
        assert gr.recall == pytest.approx(1.0)

    def test_update_wrong_value(self) -> None:
        """Agent updates field to wrong value."""
        initial = pl.DataFrame(
            {
                "note_id": ["n1"],
                "title": ["Old"],
                "sandbox_message_index": [0],
            }
        )
        final = pl.DataFrame(
            {
                "note_id": ["n1"],
                "title": ["Wrong value"],
                "sandbox_message_index": [2],
            }
        )

        ctx = _mock_execution_context(
            {"NOTES": initial},
            {"NOTES": final},
            active_namespaces={"NOTES"},
        )

        config = EntityDiffEvalConfig(
            specs=[
                EntityDiffSpec(
                    operation="update",
                    source="agentsandbox",
                    namespace="NOTES",
                    entity_id_column="note_id",
                    entity_id_value="n1",
                    expected_fields={"title": "Expected value"},
                    column_similarity_measure={
                        "note_id": "column_exact_match_similarity",
                        "title": "column_exact_match_similarity",
                    },
                )
            ]
        )

        evaluator = EntityDiffEvaluator(config)
        result = evaluator.evaluate(execution_context=ctx)

        gr = result.group_results[0]
        # title mismatch → score < 1.0
        assert gr.recall < 1.0


class TestEntityDiffEvaluatorGuardrails:
    """Test guardrail checking."""

    def test_guardrail_pass(self) -> None:
        """Non-spec database unchanged → guardrail passes."""
        notes_initial = pl.DataFrame(
            {
                "note_id": pl.Series([], dtype=pl.Utf8),
                "title": pl.Series([], dtype=pl.Utf8),
                "sandbox_message_index": pl.Series([], dtype=pl.Int64),
            }
        )
        notes_final = pl.DataFrame(
            {"note_id": ["n1"], "title": ["New"], "sandbox_message_index": [2]}
        )
        contact_db = pl.DataFrame(
            {"person_id": ["c1"], "name": ["Alice"], "sandbox_message_index": [0]}
        )

        ctx = _mock_execution_context(
            {"NOTES": notes_initial, "CONTACT": contact_db},
            {"NOTES": notes_final, "CONTACT": contact_db},
            active_namespaces={"NOTES", "CONTACT"},
        )

        config = EntityDiffEvalConfig(
            specs=[
                EntityDiffSpec(
                    operation="create",
                    source="agentsandbox",
                    namespace="NOTES",
                    entity_data={"note_id": "n1", "title": "New"},
                    column_similarity_measure={
                        "note_id": "column_exact_match_similarity",
                        "title": "column_exact_match_similarity",
                    },
                )
            ]
        )

        evaluator = EntityDiffEvaluator(config)
        result = evaluator.evaluate(execution_context=ctx)

        assert result.guardrail_pass is True
        assert result.guardrail_failures == []

    def test_guardrail_fail(self) -> None:
        """Non-spec database changed → guardrail fails."""
        notes_initial = pl.DataFrame(
            {
                "note_id": pl.Series([], dtype=pl.Utf8),
                "title": pl.Series([], dtype=pl.Utf8),
                "sandbox_message_index": pl.Series([], dtype=pl.Int64),
            }
        )
        notes_final = pl.DataFrame(
            {"note_id": ["n1"], "title": ["New"], "sandbox_message_index": [2]}
        )
        contact_initial = pl.DataFrame(
            {"person_id": ["c1"], "name": ["Alice"], "sandbox_message_index": [0]}
        )
        contact_final = pl.DataFrame(
            {"person_id": ["c1"], "name": ["Bob"], "sandbox_message_index": [0]}
        )

        ctx = _mock_execution_context(
            {"NOTES": notes_initial, "CONTACT": contact_initial},
            {"NOTES": notes_final, "CONTACT": contact_final},
            active_namespaces={"NOTES", "CONTACT"},
        )

        config = EntityDiffEvalConfig(
            specs=[
                EntityDiffSpec(
                    operation="create",
                    source="agentsandbox",
                    namespace="NOTES",
                    entity_data={"note_id": "n1", "title": "New"},
                    column_similarity_measure={
                        "note_id": "column_exact_match_similarity",
                        "title": "column_exact_match_similarity",
                    },
                )
            ]
        )

        evaluator = EntityDiffEvaluator(config)
        result = evaluator.evaluate(execution_context=ctx)

        assert result.guardrail_pass is False
        assert len(result.guardrail_failures) == 1
        assert "CONTACT" in result.guardrail_failures[0]


# ---------------------------------------------------------------------------
# EntityDiffEvaluator end-to-end behavior
# ---------------------------------------------------------------------------


class TestEntityDiffEvaluator:
    """End-to-end behavior of EntityDiffEvaluator and EntityDiffEvalResult."""

    def test_no_specs(self) -> None:
        """Empty config → perfect scores."""
        config = EntityDiffEvalConfig(specs=[])
        evaluator = EntityDiffEvaluator(config)
        result = evaluator.evaluate()

        assert result.overall_precision == 1.0
        assert result.overall_recall == 1.0
        assert result.guardrail_pass is True

    def test_result_to_dict(self) -> None:
        """EntityDiffEvalResult serializes correctly."""
        result = EntityDiffEvalResult(
            overall_precision=0.8,
            overall_recall=0.9,
            group_results=[
                GroupEvalResult(
                    table_key="NOTES",
                    operation="create",
                    precision=0.8,
                    recall=0.9,
                    per_entity_scores=[0.9],
                    per_entity_column_scores=[{"note_id": 1.0, "title": 0.8}],
                    num_actual=1,
                    num_expected=1,
                )
            ],
            guardrail_pass=True,
        )
        d = result.to_dict()
        assert d["overall_precision"] == 0.8
        assert d["overall_recall"] == 0.9
        assert d["guardrail_pass"] is True
        assert len(d["group_results"]) == 1
