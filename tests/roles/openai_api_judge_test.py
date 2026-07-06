# Copyright © 2026 Apple Inc.

"""Unit tests for OpenAIAPIJudge."""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution_context import ExecutionContext, RoleType
from mmtoolsandbox.roles.base_judge import (
    _UI_RUBRIC_CRITERIA,
    _UI_SYSTEM_PROMPT,
    CriterionResult,
    JudgeResult,
)
from mmtoolsandbox.roles.openai_judge import (
    OpenAIAPIJudge,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def execution_context() -> ExecutionContext:
    """Simple 2-message context with one image (USER→AGENT, AGENT→USER)."""
    context = MagicMock(spec=ExecutionContext)

    sandbox_data = {
        "sender": [RoleType.USER, RoleType.AGENT],
        "recipient": [RoleType.AGENT, RoleType.USER],
        "content": ["Look at this image", "I see it"],
        "tool_trace": [None, None],
        "tool_call_exception": [None, None],
        "image_ids": [[1], None],
        "sandbox_message_index": [0, 1],
    }
    sandbox_df = pl.DataFrame(sandbox_data)

    image_data = {"image_id": [1], "image_content": ["base64_encoded_image_content"]}
    image_df = pl.DataFrame(image_data)

    def get_database(namespace: DatabaseNamespace, **kwargs: Any) -> pl.DataFrame:
        if namespace == DatabaseNamespace.SANDBOX:
            return sandbox_df
        elif namespace == DatabaseNamespace.IMAGE:
            return image_df
        return pl.DataFrame()

    context.get_database.side_effect = get_database
    context.first_user_sandbox_message_index = 0
    context.max_sandbox_message_index = 1
    context.get_active_database_namespaces.return_value = {
        DatabaseNamespace.SANDBOX,
        DatabaseNamespace.IMAGE,
    }
    return context


TOOL_TRACE_CREATE_NOTE = json.dumps(
    {
        "tool_name": "create_note",
        "arguments": {"content": "eggs, milk"},
        "result": "n1",
    },
    ensure_ascii=False,
)

TOOL_TRACE_SEARCH_CONTACTS = json.dumps(
    {
        "tool_name": "search_contacts",
        "arguments": {"query": "Fredrik"},
        "result": [{"name": "Fredrik Thordendal", "phone": "+12453344098"}],
    },
    ensure_ascii=False,
)


@pytest.fixture
def multi_turn_context() -> ExecutionContext:
    """A realistic 6-row trajectory with tool calls and database changes."""
    context = MagicMock(spec=ExecutionContext)

    sandbox_data = {
        "sender": [
            RoleType.SYSTEM,
            RoleType.USER,
            RoleType.AGENT,
            RoleType.EXECUTION_ENVIRONMENT,
            RoleType.AGENT,
            RoleType.USER,
        ],
        "recipient": [
            RoleType.AGENT,
            RoleType.AGENT,
            RoleType.EXECUTION_ENVIRONMENT,
            RoleType.AGENT,
            RoleType.USER,
            RoleType.EXECUTION_ENVIRONMENT,
        ],
        "content": [
            "You are an assistant",
            "Create a note with eggs and milk",
            'create_note(content="eggs, milk")',
            '{"note_id": "n1"}',
            "Done! Created your note.",
            "end_conversation",
        ],
        "tool_trace": [
            None,
            None,
            None,
            [TOOL_TRACE_CREATE_NOTE],
            None,
            None,
        ],
        "tool_call_exception": [None, None, None, None, None, None],
        "image_ids": [None, [1], None, None, None, None],
        "sandbox_message_index": [0, 1, 2, 3, 4, 5],
    }
    sandbox_df = pl.DataFrame(sandbox_data)

    image_data = {"image_id": [1], "image_content": ["base64_user_image"]}
    image_df = pl.DataFrame(image_data)

    notes_initial = pl.DataFrame(
        {
            "note_id": pl.Series([], dtype=pl.Utf8),
            "content": pl.Series([], dtype=pl.Utf8),
        }
    )
    notes_final = pl.DataFrame({"note_id": ["n1"], "content": ["eggs, milk"]})

    def get_database(namespace: DatabaseNamespace, **kwargs: Any) -> pl.DataFrame:
        if namespace == DatabaseNamespace.SANDBOX:
            return sandbox_df
        elif namespace == DatabaseNamespace.IMAGE:
            return image_df
        elif namespace == DatabaseNamespace.NOTES:
            idx = kwargs.get("sandbox_message_index")
            if idx is not None and idx <= 1:
                return notes_initial
            return notes_final
        return pl.DataFrame()

    context.get_database.side_effect = get_database
    context.first_user_sandbox_message_index = 1
    context.max_sandbox_message_index = 5
    context.get_active_database_namespaces.return_value = {
        DatabaseNamespace.SANDBOX,
        DatabaseNamespace.IMAGE,
        DatabaseNamespace.NOTES,
    }
    return context


RUBRIC_RESPONSE = {
    "criteria_evaluation": [
        {
            "criterion": "task_completion",
            "analysis": "create_note was called and returned successfully.",
            "pass": True,
            "evidence": "Note created.",
        },
        {
            "criterion": "instruction_following",
            "analysis": "Content includes eggs and milk as requested.",
            "pass": True,
            "evidence": "Contains eggs, milk.",
        },
        {
            "criterion": "tool_use_validity",
            "analysis": "create_note is appropriate for this task.",
            "pass": True,
            "evidence": "create_note called correctly.",
        },
        {
            "criterion": "no_side_effects",
            "analysis": "Only create_note was called.",
            "pass": True,
            "evidence": "No extra actions.",
        },
        {
            "criterion": "information_accuracy",
            "analysis": "Note content matches the user request.",
            "pass": True,
            "evidence": "Content matches request.",
        },
    ],
    "result": True,
    "reasoning": "All criteria passed.",
}


# ---------------------------------------------------------------------------
# Tests: _format_user_assistant_conversation
# ---------------------------------------------------------------------------


@patch("mmtoolsandbox.roles.openai_judge.OpenAI")
def test_user_assistant_conversation_basic(
    mock_openai_cls: MagicMock, multi_turn_context: ExecutionContext
) -> None:
    """Includes USER and AGENT messages, skips SYSTEM and EXEC_ENV."""
    judge = OpenAIAPIJudge()
    sandbox_db = multi_turn_context.get_database(DatabaseNamespace.SANDBOX)
    conv = judge._format_user_assistant_conversation(sandbox_db.to_dicts())

    assert "[USER]:" in conv
    assert "[AGENT]:" in conv
    assert "Create a note with eggs and milk" in conv
    assert "Done! Created your note." in conv
    # SYSTEM and EXEC_ENV content should not appear
    assert "You are an assistant" not in conv
    assert '{"note_id": "n1"}' not in conv


@patch("mmtoolsandbox.roles.openai_judge.OpenAI")
def test_user_assistant_conversation_with_images(
    mock_openai_cls: MagicMock, multi_turn_context: ExecutionContext
) -> None:
    """USER message with image_ids gets [image_id=N] marker."""
    judge = OpenAIAPIJudge()
    sandbox_db = multi_turn_context.get_database(DatabaseNamespace.SANDBOX)
    conv = judge._format_user_assistant_conversation(sandbox_db.to_dicts())

    assert "[image_id=1]" in conv


@patch("mmtoolsandbox.roles.openai_judge.OpenAI")
def test_user_assistant_conversation_empty(mock_openai_cls: MagicMock) -> None:
    """No USER/AGENT messages returns fallback."""
    sandbox_data = {
        "sender": [RoleType.SYSTEM],
        "recipient": [RoleType.AGENT],
        "content": ["setup"],
        "tool_trace": [None],
        "tool_call_exception": [None],
        "image_ids": [None],
        "sandbox_message_index": [0],
    }
    sandbox_df = pl.DataFrame(sandbox_data)

    judge = OpenAIAPIJudge()
    conv = judge._format_user_assistant_conversation(sandbox_df.to_dicts())
    assert "No conversation" in conv


# ---------------------------------------------------------------------------
# Tests: _format_agent_environment_interactions
# ---------------------------------------------------------------------------


@patch("mmtoolsandbox.roles.openai_judge.OpenAI")
def test_agent_env_interactions_basic(
    mock_openai_cls: MagicMock, multi_turn_context: ExecutionContext
) -> None:
    """Shows AGENT→ENV tool calls and ENV→AGENT results."""
    judge = OpenAIAPIJudge()
    sandbox_db = multi_turn_context.get_database(DatabaseNamespace.SANDBOX)
    actions = judge._format_agent_environment_interactions(sandbox_db.to_dicts())

    assert "[AGENT → ENV]:" in actions
    assert "[ENV → AGENT]:" in actions
    assert "create_note" in actions
    assert "eggs, milk" in actions


@patch("mmtoolsandbox.roles.openai_judge.OpenAI")
def test_agent_env_interactions_with_error(mock_openai_cls: MagicMock) -> None:
    """EXEC_ENV row with tool_call_exception shows ERROR."""
    sandbox_data = {
        "sender": [RoleType.AGENT, RoleType.EXECUTION_ENVIRONMENT],
        "recipient": [RoleType.EXECUTION_ENVIRONMENT, RoleType.AGENT],
        "content": ["bad_call()", ""],
        "tool_trace": [None, None],
        "tool_call_exception": [None, "ValueError: invalid argument"],
        "image_ids": [None, None],
        "sandbox_message_index": [0, 1],
    }
    sandbox_df = pl.DataFrame(sandbox_data)

    judge = OpenAIAPIJudge()
    actions = judge._format_agent_environment_interactions(sandbox_df.to_dicts())

    assert "ERROR" in actions
    assert "invalid argument" in actions


@patch("mmtoolsandbox.roles.openai_judge.OpenAI")
def test_agent_env_interactions_empty(mock_openai_cls: MagicMock) -> None:
    """No AGENT↔EXEC_ENV messages returns fallback."""
    sandbox_data = {
        "sender": [RoleType.USER, RoleType.AGENT],
        "recipient": [RoleType.AGENT, RoleType.USER],
        "content": ["hello", "hi"],
        "tool_trace": [None, None],
        "tool_call_exception": [None, None],
        "image_ids": [None, None],
        "sandbox_message_index": [0, 1],
    }
    sandbox_df = pl.DataFrame(sandbox_data)

    judge = OpenAIAPIJudge()
    actions = judge._format_agent_environment_interactions(sandbox_df.to_dicts())
    assert "No agent actions" in actions


# ---------------------------------------------------------------------------
# Tests: _format_database_diff
# ---------------------------------------------------------------------------


@patch("mmtoolsandbox.roles.openai_judge.OpenAI")
def test_format_database_diff_additions(
    mock_openai_cls: MagicMock, multi_turn_context: ExecutionContext
) -> None:
    """Detects added rows in NOTES."""
    judge = OpenAIAPIJudge()
    diff = judge._format_database_diff(multi_turn_context)
    assert "NOTES" in diff
    assert "eggs, milk" in diff
    assert "+1" in diff or "1 row" in diff


@patch("mmtoolsandbox.roles.openai_judge.OpenAI")
def test_format_database_diff_no_changes(mock_openai_cls: MagicMock) -> None:
    """Reports no changes when initial == final."""
    context = MagicMock(spec=ExecutionContext)
    notes_df = pl.DataFrame({"note_id": ["n1"], "content": ["existing"]})

    def get_database(namespace: DatabaseNamespace, **kwargs: Any) -> pl.DataFrame:
        if namespace == DatabaseNamespace.NOTES:
            return notes_df
        return pl.DataFrame()

    context.get_database.side_effect = get_database
    context.first_user_sandbox_message_index = 0
    context.max_sandbox_message_index = 5
    context.get_active_database_namespaces.return_value = {
        DatabaseNamespace.SANDBOX,
        DatabaseNamespace.IMAGE,
        DatabaseNamespace.NOTES,
    }

    judge = OpenAIAPIJudge()
    diff = judge._format_database_diff(context)
    assert "No database changes" in diff


@patch("mmtoolsandbox.roles.openai_judge.OpenAI")
def test_format_database_diff_skips_sandbox_and_image(
    mock_openai_cls: MagicMock,
) -> None:
    """SANDBOX and IMAGE namespaces are excluded from diff."""
    context = MagicMock(spec=ExecutionContext)
    context.get_active_database_namespaces.return_value = {
        DatabaseNamespace.SANDBOX,
        DatabaseNamespace.IMAGE,
    }
    context.first_user_sandbox_message_index = 0
    context.max_sandbox_message_index = 5

    judge = OpenAIAPIJudge()
    diff = judge._format_database_diff(context)
    assert "No database changes" in diff


# ---------------------------------------------------------------------------
# Tests: evaluate (full flow)
# ---------------------------------------------------------------------------


@patch("mmtoolsandbox.roles.openai_judge.OpenAI")
def test_evaluate_rubric_schema(
    mock_openai_cls: MagicMock, multi_turn_context: ExecutionContext
) -> None:
    """Rubric schema returned with criteria_evaluation."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(RUBRIC_RESPONSE)
    mock_client.chat.completions.create.return_value = mock_response

    judge = OpenAIAPIJudge()
    result = judge.evaluate(multi_turn_context, "Create a note with eggs and milk.")

    assert result["result"] is True
    assert "criteria_evaluation" in result
    assert len(result["criteria_evaluation"]) == 5
    assert all(
        "criterion" in c and "pass" in c and "evidence" in c
        for c in result["criteria_evaluation"]
    )


@patch("mmtoolsandbox.roles.openai_judge.OpenAI")
def test_evaluate_backward_compat(
    mock_openai_cls: MagicMock, multi_turn_context: ExecutionContext
) -> None:
    """result and reasoning keys always present."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(RUBRIC_RESPONSE)
    mock_client.chat.completions.create.return_value = mock_response

    judge = OpenAIAPIJudge()
    result = judge.evaluate(multi_turn_context, "Create a note.")

    assert "result" in result
    assert "reasoning" in result
    assert isinstance(result["result"], bool)
    assert isinstance(result["reasoning"], str)


# ---------------------------------------------------------------------------
# Tests: _format_evidence integration
# ---------------------------------------------------------------------------


@patch("mmtoolsandbox.roles.openai_judge.OpenAI")
def test_format_evidence_multimodal(
    mock_openai_cls: MagicMock, execution_context: ExecutionContext
) -> None:
    """_format_evidence includes images as base64 data URLs."""
    judge = OpenAIAPIJudge()
    evidence_parts = judge._format_evidence(execution_context, "Identify the image.")

    assert isinstance(evidence_parts, list)

    image_parts = [p for p in evidence_parts if p.get("type") == "image_url"]
    assert len(image_parts) == 1
    assert (
        image_parts[0]["image_url"]["url"]
        == "data:image/jpeg;base64,base64_encoded_image_content"
    )


@patch("mmtoolsandbox.roles.openai_judge.OpenAI")
def test_evaluate_multimodal(
    mock_openai_cls: MagicMock, execution_context: ExecutionContext
) -> None:
    """evaluate sends rubric prompt, conversation, and images."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(RUBRIC_RESPONSE)
    mock_client.chat.completions.create.return_value = mock_response

    judge = OpenAIAPIJudge()
    result = judge.evaluate(execution_context, "Identify the image.")

    assert result["result"] is True

    call_args = mock_client.chat.completions.create.call_args
    kwargs = call_args.kwargs
    messages = kwargs["messages"]

    system_content = messages[0]["content"]
    assert "TRUST MODEL" in system_content
    assert "RUBRIC" in system_content
    assert "task_completion" in system_content

    user_content = messages[1]["content"]
    assert isinstance(user_content, list)
    assert any(
        "Task Completion Criteria" in p["text"]
        for p in user_content
        if p["type"] == "text"
    )
    assert any(
        p["type"] == "image_url"
        and p["image_url"]["url"]
        == "data:image/jpeg;base64,base64_encoded_image_content"
        for p in user_content
    )


@patch("mmtoolsandbox.roles.openai_judge.OpenAI")
def test_evaluate_evidence_contains_agent_actions(
    mock_openai_cls: MagicMock, multi_turn_context: ExecutionContext
) -> None:
    """Evidence contains Agent Actions and Conversation sections."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(RUBRIC_RESPONSE)
    mock_client.chat.completions.create.return_value = mock_response

    judge = OpenAIAPIJudge()
    judge.evaluate(multi_turn_context, "Create a note.")

    call_args = mock_client.chat.completions.create.call_args
    user_content = call_args.kwargs["messages"][1]["content"]
    text_parts = [p["text"] for p in user_content if p["type"] == "text"]
    full_text = "\n".join(text_parts)

    assert "Conversation:" in full_text
    assert "Agent Actions:" in full_text
    assert "create_note" in full_text
    assert "Database Changes:" in full_text
    assert "NOTES" in full_text


# ---------------------------------------------------------------------------
# Tests: JudgeResult / CriterionResult dataclasses
# ---------------------------------------------------------------------------


def test_judge_result_roundtrip() -> None:
    """JudgeResult.from_dict → to_dict preserves all fields."""
    raw = {
        "criteria_evaluation": [
            {
                "criterion": "task_completion",
                "analysis": "ok",
                "pass": True,
                "evidence": "done",
            },
        ],
        "result": True,
        "reasoning": "All good.",
    }
    parsed = JudgeResult.from_dict(raw)
    assert parsed.result is True
    assert len(parsed.criteria_evaluation) == 1
    assert parsed.criteria_evaluation[0].passed is True

    serialized = parsed.to_dict()
    assert serialized["result"] is True
    assert serialized["criteria_evaluation"][0]["pass"] is True
    assert "passed" not in serialized["criteria_evaluation"][0]


def test_judge_result_from_dict_tolerates_missing_fields() -> None:
    """from_dict handles missing criteria_evaluation and reasoning gracefully."""
    parsed = JudgeResult.from_dict({"result": False})
    assert parsed.result is False
    assert parsed.criteria_evaluation == []
    assert parsed.reasoning == ""


def test_criterion_result_from_dict_uses_pass_key() -> None:
    """CriterionResult reads 'pass' from JSON but stores as 'passed'."""
    cr = CriterionResult.from_dict(
        {"criterion": "x", "pass": True, "analysis": "a", "evidence": "e"}
    )
    assert cr.passed is True
    d = cr.to_dict()
    assert d["pass"] is True
    assert "passed" not in d


# ---------------------------------------------------------------------------
# Tests: Edge cases — image ID consistency
# ---------------------------------------------------------------------------


@patch("mmtoolsandbox.roles.openai_judge.OpenAI")
def test_image_ids_consistent_across_evidence(
    mock_openai_cls: MagicMock, multi_turn_context: ExecutionContext
) -> None:
    """Every [image_id=N] in Conversation has a matching base64 image in the evidence."""
    judge = OpenAIAPIJudge()
    parts = judge._format_evidence(multi_turn_context, "Test criteria.")

    text_parts = [p["text"] for p in parts if p.get("type") == "text"]
    full_text = "\n".join(text_parts)

    # Extract image IDs referenced in conversation markers
    import re

    referenced_ids = {int(m) for m in re.findall(r"\[image_id=(\d+)\]", full_text)}

    # Extract image IDs provided as base64
    provided_ids: set[int] = set()
    for p in parts:
        if p.get("type") == "image_url":
            # Image exists — multi_turn_context has image_id=1 with "base64_user_image"
            provided_ids.add(1)  # We know from the fixture

    # Every referenced ID should have a corresponding image
    assert referenced_ids, "Expected at least one [image_id=N] marker"
    assert referenced_ids <= provided_ids, (
        f"Image IDs {referenced_ids - provided_ids} referenced in conversation "
        f"but not provided as base64 images"
    )


# ---------------------------------------------------------------------------
# Tests: Edge cases — _format_agent_environment_interactions
# ---------------------------------------------------------------------------


@patch("mmtoolsandbox.roles.openai_judge.OpenAI")
def test_agent_env_interactions_multiple_traces(mock_openai_cls: MagicMock) -> None:
    """Multiple tool traces in a single EXEC_ENV row all appear."""
    sandbox_data = {
        "sender": [RoleType.EXECUTION_ENVIRONMENT],
        "recipient": [RoleType.AGENT],
        "content": ["results"],
        "tool_trace": [[TOOL_TRACE_CREATE_NOTE, TOOL_TRACE_SEARCH_CONTACTS]],
        "tool_call_exception": [None],
        "image_ids": [None],
        "sandbox_message_index": [0],
    }
    sandbox_df = pl.DataFrame(sandbox_data)

    judge = OpenAIAPIJudge()
    actions = judge._format_agent_environment_interactions(sandbox_df.to_dicts())

    assert "create_note" in actions
    assert "search_contacts" in actions
    assert actions.count("[ENV → AGENT]:") == 2


@patch("mmtoolsandbox.roles.openai_judge.OpenAI")
def test_agent_env_interactions_malformed_json(mock_openai_cls: MagicMock) -> None:
    """Malformed JSON in tool_trace falls back to raw content."""
    sandbox_data = {
        "sender": [RoleType.EXECUTION_ENVIRONMENT],
        "recipient": [RoleType.AGENT],
        "content": ["raw fallback content"],
        "tool_trace": [["not valid json {{{"]],
        "tool_call_exception": [None],
        "image_ids": [None],
        "sandbox_message_index": [0],
    }
    sandbox_df = pl.DataFrame(sandbox_data)

    judge = OpenAIAPIJudge()
    actions = judge._format_agent_environment_interactions(sandbox_df.to_dicts())

    assert "raw fallback content" in actions


# ---------------------------------------------------------------------------
# Tests: Edge cases — _format_images
# ---------------------------------------------------------------------------


@patch("mmtoolsandbox.roles.openai_judge.OpenAI")
def test_format_images_dedup_and_cap(mock_openai_cls: MagicMock) -> None:
    """Images are deduplicated and capped at _MAX_IMAGES (10)."""
    context = MagicMock(spec=ExecutionContext)

    # 12 unique image IDs across two messages, with one duplicate
    sandbox_rows = [
        {
            "sender": RoleType.USER,
            "recipient": RoleType.AGENT,
            "image_ids": [1, 2, 3, 4, 5],
        },
        {
            "sender": RoleType.AGENT,
            "recipient": RoleType.USER,
            "image_ids": [3, 6, 7, 8, 9, 10, 11, 12],  # 3 is a duplicate
        },
    ]

    image_data = {
        "image_id": list(range(1, 13)),
        "image_content": [f"b64_{i}" for i in range(1, 13)],
    }
    image_df = pl.DataFrame(image_data)

    def get_database(namespace: DatabaseNamespace, **kwargs: Any) -> pl.DataFrame:
        if namespace == DatabaseNamespace.IMAGE:
            return image_df
        return pl.DataFrame()

    context.get_database.side_effect = get_database

    judge = OpenAIAPIJudge()
    images = judge._format_images(sandbox_rows, context)

    assert len(images) == 10  # Capped at _MAX_IMAGES
    # No duplicates — image_id=3 should appear only once
    urls = [img["image_url"]["url"] for img in images]
    assert len(set(urls)) == 10


@patch("mmtoolsandbox.roles.openai_judge.OpenAI")
def test_format_images_missing_image_id(mock_openai_cls: MagicMock) -> None:
    """Image IDs not found in IMAGE DB are silently skipped."""
    context = MagicMock(spec=ExecutionContext)

    sandbox_rows = [
        {
            "sender": RoleType.USER,
            "recipient": RoleType.AGENT,
            "image_ids": [1, 999],  # 999 doesn't exist
        },
    ]

    image_data = {"image_id": [1], "image_content": ["b64_1"]}
    image_df = pl.DataFrame(image_data)

    def get_database(namespace: DatabaseNamespace, **kwargs: Any) -> pl.DataFrame:
        if namespace == DatabaseNamespace.IMAGE:
            return image_df
        return pl.DataFrame()

    context.get_database.side_effect = get_database

    judge = OpenAIAPIJudge()
    images = judge._format_images(sandbox_rows, context)

    assert len(images) == 1
    assert "b64_1" in images[0]["image_url"]["url"]


# ---------------------------------------------------------------------------
# Tests: Edge cases — _format_database_diff
# ---------------------------------------------------------------------------


@patch("mmtoolsandbox.roles.openai_judge.OpenAI")
def test_format_database_diff_multiple_namespaces(mock_openai_cls: MagicMock) -> None:
    """Additions in multiple namespaces are all reported."""
    context = MagicMock(spec=ExecutionContext)

    notes_initial = pl.DataFrame(
        {
            "note_id": pl.Series([], dtype=pl.Utf8),
            "content": pl.Series([], dtype=pl.Utf8),
        }
    )
    notes_final = pl.DataFrame({"note_id": ["n1"], "content": ["my note"]})

    contacts_initial = pl.DataFrame(
        {
            "person_id": pl.Series([], dtype=pl.Utf8),
            "name": pl.Series([], dtype=pl.Utf8),
        }
    )
    contacts_final = pl.DataFrame({"person_id": ["p1"], "name": ["Alice"]})

    def get_database(namespace: DatabaseNamespace, **kwargs: Any) -> pl.DataFrame:
        idx = kwargs.get("sandbox_message_index")
        if namespace == DatabaseNamespace.NOTES:
            return notes_initial if idx is not None and idx <= 1 else notes_final
        elif namespace == DatabaseNamespace.CONTACT:
            return contacts_initial if idx is not None and idx <= 1 else contacts_final
        return pl.DataFrame()

    context.get_database.side_effect = get_database
    context.first_user_sandbox_message_index = 1
    context.max_sandbox_message_index = 5
    context.get_active_database_namespaces.return_value = {
        DatabaseNamespace.SANDBOX,
        DatabaseNamespace.IMAGE,
        DatabaseNamespace.NOTES,
        DatabaseNamespace.CONTACT,
    }

    judge = OpenAIAPIJudge()
    diff = judge._format_database_diff(context)

    assert "NOTES" in diff
    assert "CONTACT" in diff
    assert "my note" in diff
    assert "Alice" in diff


# ---------------------------------------------------------------------------
# Tests: Edge cases — _format_images with tool-referenced images
# ---------------------------------------------------------------------------


@patch("mmtoolsandbox.roles.openai_judge.OpenAI")
def test_format_images_from_tool_trace(mock_openai_cls: MagicMock) -> None:
    """Images referenced in tool_trace (view_image, crop_image) are included."""
    context = MagicMock(spec=ExecutionContext)

    view_image_trace = json.dumps(
        {
            "tool_name": "view_image",
            "arguments": {"image_id": 0},
            "result": "ImageResult(image_id=0)",
        },
        ensure_ascii=False,
    )

    # No image_ids on any USER/AGENT messages — image is only in tool_trace
    sandbox_rows = [
        {
            "sender": RoleType.USER,
            "recipient": RoleType.AGENT,
            "image_ids": None,
        },
        {
            "sender": RoleType.EXECUTION_ENVIRONMENT,
            "recipient": RoleType.AGENT,
            "image_ids": None,
            "tool_trace": [view_image_trace],
        },
        {
            "sender": RoleType.AGENT,
            "recipient": RoleType.USER,
            "image_ids": None,
        },
    ]

    image_data = {"image_id": [0], "image_content": ["b64_pizza_menu"]}
    image_df = pl.DataFrame(image_data)

    def get_database(namespace: DatabaseNamespace, **kwargs: Any) -> pl.DataFrame:
        if namespace == DatabaseNamespace.IMAGE:
            return image_df
        return pl.DataFrame()

    context.get_database.side_effect = get_database

    judge = OpenAIAPIJudge()
    images = judge._format_images(sandbox_rows, context)  # type: ignore[arg-type]

    assert len(images) == 1
    assert "b64_pizza_menu" in images[0]["image_url"]["url"]


# ---------------------------------------------------------------------------
# Tests: _UI_SYSTEM_PROMPT (standalone UI judge prompt)
# ---------------------------------------------------------------------------


def test_ui_system_prompt_has_4_criteria() -> None:
    """UI judge prompt mentions 4 criteria and includes all UI dimensions."""
    assert "4 criteria" in _UI_SYSTEM_PROMPT
    assert "ALL 4 criteria pass" in _UI_SYSTEM_PROMPT
    for criterion in _UI_RUBRIC_CRITERIA:
        assert criterion in _UI_SYSTEM_PROMPT


def test_ui_system_prompt_no_company_references() -> None:
    """UI judge prompt must NOT mention company names or specific guidelines."""
    for name in ("Apple", "Google", "Material Design", "Fluent", "HIG", "Microsoft"):
        assert name not in _UI_SYSTEM_PROMPT, f"Prompt should not reference '{name}'"


def test_ui_system_prompt_is_standalone() -> None:
    """UI prompt is self-contained — not a fragment appended to _SYSTEM_PROMPT."""
    assert _UI_SYSTEM_PROMPT.startswith("You are an impartial judge")
    assert "OUTPUT FORMAT" in _UI_SYSTEM_PROMPT
    # Must NOT contain the standard 5 criteria
    assert "task_completion" not in _UI_SYSTEM_PROMPT.split("ui_task_focus")[0]
    assert "information_accuracy" not in _UI_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Tests: evaluate_ui (separate UI judge call)
# ---------------------------------------------------------------------------

UI_JUDGE_RESPONSE = {
    "criteria_evaluation": [
        {
            "criterion": "ui_task_focus",
            "analysis": "Clear primary action.",
            "pass": True,
            "evidence": "Title and Book button prominent.",
        },
        {
            "criterion": "ui_structure_and_patterns",
            "analysis": "List pattern with per-item cards.",
            "pass": True,
            "evidence": "Card > Row > Column structure.",
        },
        {
            "criterion": "ui_action_clarity",
            "analysis": "Verb labels on all buttons.",
            "pass": False,
            "evidence": "Generic 'Submit' label on one button.",
        },
        {
            "criterion": "ui_feedback_and_flow",
            "analysis": "Confirmation screen shown after click.",
            "pass": True,
            "evidence": "Booking confirmed message visible.",
        },
    ],
    "result": False,
    "reasoning": "UI action clarity failed due to generic button label.",
}


@patch("mmtoolsandbox.roles.openai_judge.OpenAI")
def test_evaluate_ui_uses_ui_system_prompt(
    mock_openai_cls: MagicMock, multi_turn_context: ExecutionContext
) -> None:
    """evaluate_ui() sends the standalone UI prompt, not the standard one."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(UI_JUDGE_RESPONSE)
    mock_client.chat.completions.create.return_value = mock_response

    judge = OpenAIAPIJudge()
    result = judge.evaluate_ui(multi_turn_context, "Create a booking UI.")

    # Verify the system prompt is the UI-specific one
    call_args = mock_client.chat.completions.create.call_args
    system_content = call_args.kwargs["messages"][0]["content"]
    assert "ui_task_focus" in system_content
    assert "ui_structure_and_patterns" in system_content
    assert "4 criteria" in system_content
    # Must NOT contain standard criteria
    assert "information_accuracy" not in system_content

    # Verify the result has 4 criteria
    assert len(result["criteria_evaluation"]) == 4
    assert result["result"] is False


@patch("mmtoolsandbox.roles.openai_judge.OpenAI")
def test_evaluate_and_evaluate_ui_are_independent(
    mock_openai_cls: MagicMock, multi_turn_context: ExecutionContext
) -> None:
    """evaluate() and evaluate_ui() use different prompts and return independent results."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    # First call returns standard 5-criteria result (all pass)
    # Second call returns UI 4-criteria result (one fails)
    mock_response_standard = MagicMock()
    mock_response_standard.choices[0].message.content = json.dumps(RUBRIC_RESPONSE)
    mock_response_ui = MagicMock()
    mock_response_ui.choices[0].message.content = json.dumps(UI_JUDGE_RESPONSE)
    mock_client.chat.completions.create.side_effect = [
        mock_response_standard,
        mock_response_ui,
    ]

    judge = OpenAIAPIJudge()
    standard_result = judge.evaluate(multi_turn_context, "Create a booking UI.")
    ui_result = judge.evaluate_ui(multi_turn_context, "Create a booking UI.")

    # Standard judge passes (all 5 criteria pass)
    assert standard_result["result"] is True
    assert len(standard_result["criteria_evaluation"]) == 5

    # UI judge fails (ui_action_clarity fails)
    assert ui_result["result"] is False
    assert len(ui_result["criteria_evaluation"]) == 4

    # Verify different system prompts were used
    calls = mock_client.chat.completions.create.call_args_list
    standard_prompt = calls[0].kwargs["messages"][0]["content"]
    ui_prompt = calls[1].kwargs["messages"][0]["content"]
    assert "5 criteria" in standard_prompt
    assert "4 criteria" in ui_prompt
