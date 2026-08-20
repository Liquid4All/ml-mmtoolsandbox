# Copyright © 2026 Apple Inc.

"""Unified initial→final state-diff renderer for judge evidence.

Produces a single, backend-agnostic "State Changes" view from pairs of
initial/final row sets, whatever their origin (MMToolSandbox native namespaces
or external AppWorld tables). The caller materializes each table's rows as a
list of plain dicts and supplies a neutral label; this module classifies
added / updated / deleted rows and renders them with content.

Design notes:
- **Key auto-detection, no hardcoding.** The row identity is discovered from the
  data: prefer a column named ``id``, else the unique non-null ``*_id`` column
  present in both snapshots. If no stable key exists, fall back to a whole-row
  set diff (add/delete only — updates cannot be correlated without an identity).
- **Content, not counts.** Adds/deletes show the row; updates show changed
  fields as ``before -> after``.
- **No leakage.** This module never emits backend vocabulary; the caller chooses
  labels. Storage-bookkeeping columns are suppressed for readability.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

# Storage-bookkeeping columns that carry no signal for the judge and would only
# add noise (or spurious "updates") if shown.
_NOISE_COLUMNS: frozenset[str] = frozenset({"record_hash", "sandbox_message_index"})

# Truncate long field values (e.g. email bodies) so the evidence stays compact.
_MAX_VALUE_LEN = 300

# Safety cap on rows rendered per (table, operation); overflow is summarized.
_MAX_ROWS_PER_OP = 50


@dataclass
class TableDiff:
    """A single table's before/after snapshot to be diffed.

    Attributes:
        label: Neutral, judge-facing name (e.g. ``"todoist.projects"``,
            ``"CALENDAR_EVENTS"``). Chosen by the caller; must not leak backend
            vocabulary.
        initial: Rows before the agent acted, as plain dicts.
        final: Rows after the agent finished, as plain dicts.
    """

    label: str
    initial: list[dict[str, Any]]
    final: list[dict[str, Any]]


def _key_valid_for(col: str, rows: list[dict[str, Any]]) -> bool:
    """Return True if *col* is a usable identity for *rows*.

    Requires the column to be present and non-null in every row and unique
    across the rows. An empty row list is vacuously valid.
    """
    if not rows:
        return True
    values = []
    for row in rows:
        if col not in row or row[col] is None:
            return False
        values.append(row[col])
    return len(set(values)) == len(values)


def _resolve_key(
    initial: list[dict[str, Any]], final: list[dict[str, Any]]
) -> str | None:
    """Auto-detect the identity column, or None for a keyless (full-row) diff.

    Preference: a column named ``id`` first, then any ``*_id`` column, choosing
    the first candidate that is present, non-null, and unique in both snapshots.
    """
    columns: list[str] = []
    seen: set[str] = set()
    for row in initial[:1] + final[:1]:
        for col in row:
            if col not in seen:
                seen.add(col)
                columns.append(col)

    candidates: list[str] = []
    if "id" in seen:
        candidates.append("id")
    candidates.extend(c for c in columns if c != "id" and c.endswith("_id"))

    for col in candidates:
        if _key_valid_for(col, initial) and _key_valid_for(col, final):
            return col
    return None


def _clean_row(row: dict[str, Any]) -> dict[str, Any]:
    """Drop noise columns from a row (values untouched, for comparison)."""
    return {k: v for k, v in row.items() if k not in _NOISE_COLUMNS}


def _fmt_value(value: Any) -> str:
    """Render a single field value, truncated for compactness."""
    text = str(value)
    if len(text) > _MAX_VALUE_LEN:
        return text[:_MAX_VALUE_LEN] + "…"
    return text


def _fmt_row(row: dict[str, Any]) -> str:
    """Render a cleaned row as ``k=v, k=v`` for add/delete lines."""
    return ", ".join(f"{k}={_fmt_value(v)}" for k, v in _clean_row(row).items())


def _changed_fields(
    old: dict[str, Any], new: dict[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """Return {column: (old, new)} for non-noise columns that differ."""
    changed: dict[str, tuple[Any, Any]] = {}
    for col in _clean_row(new):
        if old.get(col) != new.get(col):
            changed[col] = (old.get(col), new.get(col))
    # Columns present only in old (dropped) also count as changes.
    for col in _clean_row(old):
        if col not in new:
            changed[col] = (old.get(col), None)
    return changed


def _row_canonical(row: dict[str, Any]) -> str:
    """Stable string form of a cleaned row for keyless multiset comparison."""
    cleaned = _clean_row(row)
    return json.dumps(
        {k: cleaned[k] for k in sorted(cleaned)},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


def _diff_one(
    diff: TableDiff,
) -> tuple[
    list[dict[str, Any]],
    list[tuple[Any, dict[str, Any], dict[str, Any]]],
    list[dict[str, Any]],
    str | None,
]:
    """Classify one table into (added, updated, deleted, key).

    ``updated`` is empty when no key could be resolved (keyless full-row diff).
    """
    key = _resolve_key(diff.initial, diff.final)

    if key is not None:
        ini_by = {row[key]: row for row in diff.initial}
        fin_by = {row[key]: row for row in diff.final}
        added = [fin_by[k] for k in fin_by if k not in ini_by]
        deleted = [ini_by[k] for k in ini_by if k not in fin_by]
        updated = [
            (k, ini_by[k], fin_by[k])
            for k in fin_by
            if k in ini_by and _clean_row(ini_by[k]) != _clean_row(fin_by[k])
        ]
        return added, updated, deleted, key

    # Keyless: whole-row multiset difference. Updates cannot be correlated.
    ini_canon = Counter(_row_canonical(r) for r in diff.initial)
    fin_canon = Counter(_row_canonical(r) for r in diff.final)
    by_canon_final = {_row_canonical(r): r for r in diff.final}
    by_canon_initial = {_row_canonical(r): r for r in diff.initial}
    added = [by_canon_final[c] for c in (fin_canon - ini_canon).elements()]
    deleted = [by_canon_initial[c] for c in (ini_canon - fin_canon).elements()]
    return added, [], deleted, None


def _render_op(marker: str, label: str, lines: list[str]) -> list[str]:
    """Render one operation block with an overflow summary if needed."""
    out: list[str] = []
    shown = lines[:_MAX_ROWS_PER_OP]
    for line in shown:
        out.append(f"    {marker} {line}")
    overflow = len(lines) - len(shown)
    if overflow > 0:
        out.append(f"    {marker} … and {overflow} more")
    return out


def render_state_diff(tables: list[TableDiff]) -> str:
    """Render a unified, judge-facing state-diff for all supplied tables.

    Args:
        tables: Per-table initial/final snapshots with neutral labels.

    Returns:
        A multi-line "State Changes" string. Tables with no changes are
        omitted; if nothing changed anywhere, returns a single
        "No state changes detected." line.
    """
    blocks: list[str] = []
    for diff in tables:
        added, updated, deleted, key = _diff_one(diff)
        if not (added or updated or deleted):
            continue

        counts = []
        if added:
            counts.append(f"+{len(added)}")
        if updated:
            counts.append(f"~{len(updated)}")
        if deleted:
            counts.append(f"-{len(deleted)}")
        header = f"  [{diff.label}] {' '.join(counts)}"
        if key is None:
            header += " (no stable key — updates shown as delete+add)"
        block = [header]

        if added:
            block += _render_op("ADD", diff.label, [_fmt_row(r) for r in added])
        if updated:
            upd_lines = []
            for _k, old, new in updated:
                changes = _changed_fields(old, new)
                change_str = "; ".join(
                    f"{col}: {_fmt_value(o)} -> {_fmt_value(n)}"
                    for col, (o, n) in changes.items()
                )
                upd_lines.append(change_str)
            block += _render_op("UPD", diff.label, upd_lines)
        if deleted:
            block += _render_op("DEL", diff.label, [_fmt_row(r) for r in deleted])

        blocks.append("\n".join(block))

    if not blocks:
        return "No state changes detected."
    return "\n".join(blocks)
