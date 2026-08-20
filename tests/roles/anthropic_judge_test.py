# Copyright © 2026 Apple Inc.

"""Tests for the Anthropic judge's evidence assembly.

:class:`AnthropicAPIJudge` overrides ``evaluate()`` but inherits the evidence
formatters from :class:`OpenAIAPIJudge`. These tests pin the contract at that
seam: the override must hand ``_format_evidence`` a ``list[TableDiff]`` built via
``_appworld_table_diffs``. They exercise ``evaluate()`` rather than the
formatters directly so the override itself stays covered.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import anthropic
import polars as pl
import pytest

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution_context import ExecutionContext, RoleType
from mmtoolsandbox.roles.anthropic_judge import AnthropicAPIJudge

_PATCH_ANTHROPIC = "mmtoolsandbox.roles.anthropic_judge.anthropic.Anthropic"
_PATCH_READ_TABLE = "mmtoolsandbox.common.entity_diff_evaluator.read_appworld_table"

_RUBRIC_RESPONSE = {
    "result": True,
    "reasoning": "The Groceries project was deleted.",
    "criteria_evaluation": [
        {"criterion": name, "analysis": "ok", "pass": True, "evidence": "ok"}
        for name in (
            "task_completion",
            "instruction_following",
            "tool_use_validity",
            "no_side_effects",
            "information_accuracy",
        )
    ],
}


@pytest.fixture
def appworld_context() -> ExecutionContext:
    """Minimal trajectory with no native DB changes, so only AppWorld shows."""
    context = MagicMock(spec=ExecutionContext)

    sandbox_df = pl.DataFrame(
        {
            "sender": [RoleType.USER, RoleType.AGENT],
            "recipient": [RoleType.AGENT, RoleType.USER],
            "content": ["Delete the Groceries project", "Deleted it."],
            "tool_trace": [None, None],
            "tool_call_exception": [None, None],
            "image_ids": [None, None],
            "sandbox_message_index": [0, 1],
        }
    )

    def get_database(namespace: DatabaseNamespace, **kwargs: Any) -> pl.DataFrame:
        if namespace == DatabaseNamespace.SANDBOX:
            return sandbox_df
        return pl.DataFrame()

    context.get_database.side_effect = get_database
    context.first_user_sandbox_message_index = 0
    context.max_sandbox_message_index = 1
    context.get_active_database_namespaces.return_value = {DatabaseNamespace.SANDBOX}
    return context


def _rubric_message() -> MagicMock:
    """A response whose single block is a real ``TextBlock``.

    ``_parse_judge_response`` filters blocks with
    ``isinstance(block, anthropic.types.TextBlock)``, which a bare MagicMock fails.
    """
    response = MagicMock()
    response.content = [
        anthropic.types.TextBlock(text=json.dumps(_RUBRIC_RESPONSE), type="text")
    ]
    return response


def _evidence_text(judge: AnthropicAPIJudge) -> str:
    return "\n".join(
        part.get("text", "")
        for part in judge.last_evidence["main"]
        if part.get("type") == "text"
    )


def test_evaluate_renders_appworld_state_diff(
    appworld_context: ExecutionContext,
) -> None:
    """``evaluate()`` must convert ``appworld_initial`` into rendered TableDiffs."""
    initial = pl.DataFrame({"id": [400, 401], "name": ["Groceries", "Work"]})
    final = pl.DataFrame({"id": [401], "name": ["Work"]})

    with patch(_PATCH_ANTHROPIC):
        judge = AnthropicAPIJudge("claude-sonnet-4-5-20250929")
        with (
            patch.object(judge, "_messages_create", return_value=_rubric_message()),
            patch(_PATCH_READ_TABLE, return_value=final),
        ):
            result = judge.evaluate(
                execution_context=appworld_context,
                criteria="Delete the Groceries project.",
                appworld_initial={"todoist.projects": initial},
            )

    assert result["result"] is True

    evidence_text = _evidence_text(judge)
    # Deleted rows must be rendered with their content, not just counted.
    assert "todoist.projects" in evidence_text
    assert "DEL id=400" in evidence_text
    assert "Groceries" in evidence_text
    # Backend vocabulary must not leak into judge-facing evidence.
    assert "appworld" not in evidence_text.lower()


def test_evaluate_without_appworld_initial_is_unaffected(
    appworld_context: ExecutionContext,
) -> None:
    """``None`` ``appworld_initial`` renders no state changes and raises nothing."""
    with patch(_PATCH_ANTHROPIC):
        judge = AnthropicAPIJudge("claude-sonnet-4-5-20250929")
        with patch.object(judge, "_messages_create", return_value=_rubric_message()):
            result = judge.evaluate(
                execution_context=appworld_context,
                criteria="Delete the Groceries project.",
                appworld_initial=None,
            )

    assert result["result"] is True
    assert "No state changes detected." in _evidence_text(judge)
