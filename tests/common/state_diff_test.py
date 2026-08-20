# Copyright © 2026 Apple Inc.

"""Unit tests for the unified state-diff renderer.

Pure-logic tests (no polars / execution_context import), so they run even in
environments where the heavier modules fail to import.
"""

from __future__ import annotations

from mmtoolsandbox.common.state_diff import (
    TableDiff,
    _resolve_key,
    render_state_diff,
)


def test_added_rows_shown_with_content() -> None:
    diff = TableDiff(
        label="gmail.emails",
        initial=[{"id": 1, "subject": "hi"}],
        final=[{"id": 1, "subject": "hi"}, {"id": 2, "subject": "new mail"}],
    )
    out = render_state_diff([diff])
    assert "[gmail.emails]" in out
    assert "+1" in out
    assert "ADD" in out
    assert "new mail" in out
    assert "-" not in out.split("\n")[0]  # no deletes counted


def test_deleted_rows_visible() -> None:
    diff = TableDiff(
        label="todoist.projects",
        initial=[{"id": 400, "name": "Archive Hunt"}, {"id": 401, "name": "Keep"}],
        final=[{"id": 401, "name": "Keep"}],
    )
    out = render_state_diff([diff])
    assert "-1" in out
    assert "DEL" in out
    assert "Archive Hunt" in out


def test_updated_rows_show_field_level_change() -> None:
    diff = TableDiff(
        label="notes",
        initial=[{"id": 7, "title": "Lunch", "loc": ""}],
        final=[{"id": 7, "title": "Team Lunch", "loc": "Cafe"}],
    )
    out = render_state_diff([diff])
    assert "~1" in out
    assert "UPD" in out
    assert "title:" in out
    assert "Lunch -> Team Lunch" in out
    assert "loc:" in out


def test_noise_columns_suppressed_and_not_flagged_as_update() -> None:
    # record_hash churn must not register as an update, nor be displayed.
    diff = TableDiff(
        label="t",
        initial=[{"id": 1, "v": "a", "record_hash": "OLD"}],
        final=[{"id": 1, "v": "a", "record_hash": "NEW"}],
    )
    out = render_state_diff([diff])
    assert out == "No state changes detected."


def test_noise_column_hidden_on_added_row() -> None:
    diff = TableDiff(
        label="t",
        initial=[],
        final=[{"id": 1, "v": "a", "record_hash": "X", "sandbox_message_index": 3}],
    )
    out = render_state_diff([diff])
    assert "record_hash" not in out
    assert "sandbox_message_index" not in out
    assert "v=a" in out


def test_non_id_key_detected() -> None:
    diff = TableDiff(
        label="CALENDAR_EVENTS",
        initial=[{"calendar_event_id": "abc", "title": "x"}],
        final=[
            {"calendar_event_id": "abc", "title": "x"},
            {"calendar_event_id": "def", "title": "y"},
        ],
    )
    assert _resolve_key(diff.initial, diff.final) == "calendar_event_id"
    out = render_state_diff([diff])
    assert "+1" in out
    assert "def" in out


def test_prefers_id_over_other_id_columns() -> None:
    rows = [{"id": 1, "user_id": 9}, {"id": 2, "user_id": 9}]
    # user_id is non-unique; id is the right key regardless.
    assert _resolve_key(rows, rows) == "id"


def test_keyless_fallback_add_delete_only() -> None:
    # No id / *_id column and non-unique rows → whole-row diff, no updates.
    diff = TableDiff(
        label="t",
        initial=[{"name": "a"}, {"name": "b"}],
        final=[{"name": "a"}, {"name": "c"}],
    )
    assert _resolve_key(diff.initial, diff.final) is None
    out = render_state_diff([diff])
    assert "no stable key" in out
    assert "name=c" in out  # added
    assert "name=b" in out  # deleted
    assert "UPD" not in out


def test_no_changes_returns_sentinel() -> None:
    diff = TableDiff(
        label="t",
        initial=[{"id": 1, "v": "a"}],
        final=[{"id": 1, "v": "a"}],
    )
    assert render_state_diff([diff]) == "No state changes detected."


def test_multiple_tables_only_changed_ones_rendered() -> None:
    changed = TableDiff("changed", [{"id": 1, "v": 1}], [{"id": 1, "v": 2}])
    unchanged = TableDiff("unchanged", [{"id": 5, "v": 9}], [{"id": 5, "v": 9}])
    out = render_state_diff([changed, unchanged])
    assert "[changed]" in out
    assert "[unchanged]" not in out


def test_no_backend_vocabulary_leaks() -> None:
    diff = TableDiff("todoist.projects", [], [{"id": 1, "name": "P"}])
    out = render_state_diff([diff])
    assert "appworld" not in out.lower()
    assert "namespace" not in out.lower()


def test_value_truncation() -> None:
    long = "x" * 500
    diff = TableDiff("t", [], [{"id": 1, "body": long}])
    out = render_state_diff([diff])
    assert "…" in out
    assert "x" * 500 not in out


def test_overflow_summary() -> None:
    final = [{"id": i, "v": i} for i in range(60)]
    diff = TableDiff("t", [], final)
    out = render_state_diff([diff])
    assert "and 10 more" in out  # 60 added, cap 50
