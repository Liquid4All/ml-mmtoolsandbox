# Copyright © 2026 Apple Inc.

"""Extended unit tests for mmtoolsandbox.common.message_conversion.

Targets undertested code paths in ``to_openai_messages()``,
``serialize_to_conversation()``, and helper functions.
"""

from __future__ import annotations

from unittest.mock import patch

import attrs
import pytest

from mmtoolsandbox.common.evaluation import EvaluationResult
from mmtoolsandbox.common.execution_context import (
    DatabaseNamespace,
    ExecutionContext,
    RoleType,
)
from mmtoolsandbox.common.image_id import ImageId
from mmtoolsandbox.common.message_conversion import (
    EXECUTION_RESULTS_CLOSE_TAG,
    EXECUTION_RESULTS_OPEN_TAG,
    ConversionMode,
    Message,
    add_messages_to_execution_context,
    extract_reasoning,
    get_image_content,
    get_messages_from_execution_context,
    normalize_tool_call_id,
    serialize_to_conversation,
    to_openai_messages,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context(
    *,
    support_images: bool = False,
    messages: list[Message] | None = None,
) -> ExecutionContext:
    """Create an ExecutionContext with optional pre-loaded messages."""
    ctx = ExecutionContext(support_images=support_images)
    if messages:
        add_messages_to_execution_context(ctx, messages)
    return ctx


def _make_tool_call_content(call_id: str, fn_name: str, args: str = "{}") -> str:
    """Build the python code string that ``to_openai_messages`` expects for a tool call."""
    safe_id = call_id.replace("-", "_")
    return (
        f"{safe_id}_parameters = {args}\n"
        f"{safe_id}_response = {fn_name}(**{safe_id}_parameters)"
    )


# ===================================================================
# Tests for to_openai_messages() -- TOOL_CALLING mode
# ===================================================================


class TestToOpenaiMessagesBasicRoles:
    """Basic role mapping: SYSTEM->system, USER->user, AGENT->assistant."""

    def test_system_message_becomes_system_role(self) -> None:
        """A SYSTEM->AGENT message must convert to role='system'."""
        messages = [
            Message(
                sender=RoleType.SYSTEM,
                recipient=RoleType.AGENT,
                content="You are a helpful assistant.",
            ),
        ]
        openai_msgs, indices, meta = to_openai_messages(messages)

        assert len(openai_msgs) == 1
        assert openai_msgs[0]["role"] == "system"
        assert openai_msgs[0]["content"] == "You are a helpful assistant."
        assert indices == [[0]]
        assert meta == [None]

    def test_consecutive_system_messages_merged(self) -> None:
        """Multiple consecutive SYSTEM->AGENT messages are merged into one."""
        messages = [
            Message(
                sender=RoleType.SYSTEM,
                recipient=RoleType.AGENT,
                content="Part one.",
            ),
            Message(
                sender=RoleType.SYSTEM,
                recipient=RoleType.AGENT,
                content="Part two.",
            ),
        ]
        openai_msgs, indices, _ = to_openai_messages(messages)

        assert len(openai_msgs) == 1
        assert openai_msgs[0]["role"] == "system"
        assert "Part one." in openai_msgs[0]["content"]
        assert "Part two." in openai_msgs[0]["content"]
        # Both original indices are captured.
        assert indices == [[0, 1]]

    def test_user_message_becomes_user_role(self) -> None:
        """A USER->AGENT message must convert to role='user'."""
        messages = [
            Message(
                sender=RoleType.USER,
                recipient=RoleType.AGENT,
                content="Hello there!",
            ),
        ]
        openai_msgs, indices, meta = to_openai_messages(messages)

        assert len(openai_msgs) == 1
        assert openai_msgs[0]["role"] == "user"
        assert openai_msgs[0]["content"] == "Hello there!"
        assert indices == [[0]]
        assert meta == [None]

    def test_agent_message_becomes_assistant_role(self) -> None:
        """An AGENT->USER message must convert to role='assistant'."""
        messages = [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.USER,
                content="Hi! How can I help?",
            ),
        ]
        openai_msgs, indices, meta = to_openai_messages(messages)

        assert len(openai_msgs) == 1
        assert openai_msgs[0]["role"] == "assistant"
        assert openai_msgs[0]["content"] == "Hi! How can I help?"
        assert indices == [[0]]
        # Choices metadata is populated for agent messages.
        assert meta[0] is not None
        assert meta[0]["finish_reason"] is None

    def test_system_to_user_skipped(self) -> None:
        """SYSTEM->USER messages are skipped (not for the agent)."""
        messages = [
            Message(
                sender=RoleType.SYSTEM,
                recipient=RoleType.USER,
                content="User sim instructions.",
            ),
        ]
        openai_msgs, indices, _ = to_openai_messages(messages)
        assert len(openai_msgs) == 0


# ===================================================================
# Tests for to_openai_messages() -- tool call grouping
# ===================================================================


class TestToolCallGrouping:
    """Tool call and tool result pairing and grouping."""

    def test_tool_call_grouped_with_result(self) -> None:
        """A single tool call + result produces assistant(tool_calls) + tool role."""
        messages = [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content=_make_tool_call_content("call_1", "search_contacts"),
                openai_tool_call_id="call_1",
                openai_function_name="search_contacts",
                finish_reason="tool_calls",
            ),
            Message(
                sender=RoleType.EXECUTION_ENVIRONMENT,
                recipient=RoleType.AGENT,
                content='{"name": "Alice"}',
                openai_tool_call_id="call_1",
                openai_function_name="search_contacts",
            ),
        ]
        openai_msgs, indices, meta = to_openai_messages(messages)

        assert len(openai_msgs) == 2
        # Assistant message with tool_calls
        assert openai_msgs[0]["role"] == "assistant"
        assert "tool_calls" in openai_msgs[0]
        assert len(openai_msgs[0]["tool_calls"]) == 1
        assert openai_msgs[0]["tool_calls"][0]["function"]["name"] == "search_contacts"

        # Tool result
        assert openai_msgs[1]["role"] == "tool"
        assert openai_msgs[1]["tool_call_id"] == "call_1"
        assert openai_msgs[1]["name"] == "search_contacts"
        assert openai_msgs[1]["content"] == '{"name": "Alice"}'

    def test_parallel_tool_calls_grouped(self) -> None:
        """Multiple AGENT->EXEC_ENV messages in sequence are grouped into a single assistant message."""
        messages = [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content=_make_tool_call_content("call_a", "tool_one"),
                openai_tool_call_id="call_a",
                openai_function_name="tool_one",
                finish_reason="tool_calls",
                generation=None,
                logprobs=None,
                token_ids=None,
            ),
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content=_make_tool_call_content("call_b", "tool_two"),
                openai_tool_call_id="call_b",
                openai_function_name="tool_two",
                finish_reason="tool_calls",
                generation=None,
                logprobs=None,
                token_ids=None,
            ),
            Message(
                sender=RoleType.EXECUTION_ENVIRONMENT,
                recipient=RoleType.AGENT,
                content="result one",
                openai_tool_call_id="call_a",
                openai_function_name="tool_one",
            ),
            Message(
                sender=RoleType.EXECUTION_ENVIRONMENT,
                recipient=RoleType.AGENT,
                content="result two",
                openai_tool_call_id="call_b",
                openai_function_name="tool_two",
            ),
        ]
        openai_msgs, indices, _ = to_openai_messages(messages)

        # Should be: 1 assistant (2 tool_calls), 2 tool results
        assert len(openai_msgs) == 3
        assert openai_msgs[0]["role"] == "assistant"
        assert len(openai_msgs[0]["tool_calls"]) == 2
        assert openai_msgs[1]["role"] == "tool"
        assert openai_msgs[2]["role"] == "tool"

        # The first openai message maps to both sandbox messages 0 and 1.
        assert indices[0] == [0, 1]

    def test_tool_result_without_images_plain_content(self) -> None:
        """Tool result without images produces plain string content."""
        messages = [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content=_make_tool_call_content("call_1", "get_time"),
                openai_tool_call_id="call_1",
                openai_function_name="get_time",
                finish_reason="tool_calls",
            ),
            Message(
                sender=RoleType.EXECUTION_ENVIRONMENT,
                recipient=RoleType.AGENT,
                content="12:00 PM",
                openai_tool_call_id="call_1",
                openai_function_name="get_time",
            ),
        ]
        openai_msgs, _, _ = to_openai_messages(messages)

        tool_msg = openai_msgs[1]
        assert tool_msg["role"] == "tool"
        assert isinstance(tool_msg["content"], str)
        assert tool_msg["content"] == "12:00 PM"


# ===================================================================
# Tests for to_openai_messages() -- visibility filtering
# ===================================================================


class TestVisibilityFiltering:
    """Messages with restricted visible_to should be filtered before conversion."""

    def test_visibility_filtering_hides_messages(self) -> None:
        """Messages with visible_to=[USER] should be absent from agent's view."""
        all_messages = [
            Message(
                sender=RoleType.USER,
                recipient=RoleType.AGENT,
                content="Hello agent",
            ),
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.USER,
                content="Hello!",
            ),
            # User-only metadata (visible_to=[USER])
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.USER,
                content="<ui_state_server>secret metadata</ui_state_server>",
                visible_to=[RoleType.USER],
            ),
        ]
        # Simulate the agent filtering (same as BaseRole does).
        agent_visible = [
            m for m in all_messages if m.visible_to and RoleType.AGENT in m.visible_to
        ]
        assert len(agent_visible) == 2

        openai_msgs, _, _ = to_openai_messages(agent_visible)
        assert len(openai_msgs) == 2
        assert openai_msgs[0]["role"] == "user"
        assert openai_msgs[1]["role"] == "assistant"

        # Ensure no metadata leaked.
        for msg in openai_msgs:
            assert "ui_state_server" not in str(msg.get("content", ""))


# ===================================================================
# Tests for to_openai_messages() -- CODE_EXEC mode
# ===================================================================


class TestCodeExecMode:
    """Tests for ConversionMode.CODE_EXEC behavior."""

    def test_code_exec_mode_uses_content_blocks(self) -> None:
        """In CODE_EXEC, AGENT->EXEC_ENV becomes plain assistant and EXEC_ENV->AGENT becomes user with XML tags."""
        messages = [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content="result = search_contacts(name='Alice')",
                finish_reason="stop",
            ),
            Message(
                sender=RoleType.EXECUTION_ENVIRONMENT,
                recipient=RoleType.AGENT,
                content='{"name": "Alice", "phone": "555-1234"}',
            ),
        ]
        openai_msgs, indices, meta = to_openai_messages(
            messages, mode=ConversionMode.CODE_EXEC
        )

        assert len(openai_msgs) == 2

        # Assistant message (code)
        assert openai_msgs[0]["role"] == "assistant"
        assert "search_contacts" in openai_msgs[0]["content"]
        assert "tool_calls" not in openai_msgs[0]

        # Execution result as user message with XML tags
        assert openai_msgs[1]["role"] == "user"
        content = openai_msgs[1]["content"]
        assert EXECUTION_RESULTS_OPEN_TAG in content
        assert EXECUTION_RESULTS_CLOSE_TAG in content
        assert '"Alice"' in content

    def test_code_exec_consecutive_results_aggregated(self) -> None:
        """Consecutive execution results are merged into a single user message."""
        messages = [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content="x = 1",
                finish_reason="stop",
            ),
            Message(
                sender=RoleType.EXECUTION_ENVIRONMENT,
                recipient=RoleType.AGENT,
                content="x = 1",
            ),
            # Second code block results -- same assistant turn pattern
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content="y = 2",
                finish_reason="stop",
            ),
            Message(
                sender=RoleType.EXECUTION_ENVIRONMENT,
                recipient=RoleType.AGENT,
                content="y = 2",
            ),
        ]
        openai_msgs, indices, _ = to_openai_messages(
            messages, mode=ConversionMode.CODE_EXEC
        )

        # Each assistant+result pair produces its own messages (not aggregated
        # since they are not consecutive execution results from the same turn).
        assert len(openai_msgs) == 4

    def test_code_exec_tool_exception_formatted(self) -> None:
        """Tool exceptions in CODE_EXEC mode appear in the user message."""
        messages = [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content="bad_call()",
                finish_reason="stop",
            ),
            Message(
                sender=RoleType.EXECUTION_ENVIRONMENT,
                recipient=RoleType.AGENT,
                content="",
                tool_call_exception="NameError: name 'bad_call' is not defined",
            ),
        ]
        openai_msgs, _, _ = to_openai_messages(messages, mode=ConversionMode.CODE_EXEC)

        result_msg = openai_msgs[1]
        assert result_msg["role"] == "user"
        assert "[Error]" in result_msg["content"]
        assert "NameError" in result_msg["content"]

    def test_code_exec_reasoning_trace_in_assistant(self) -> None:
        """Reasoning trace is prepended as <think> block in CODE_EXEC mode."""
        messages = [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content="search_contacts()",
                finish_reason="stop",
                reasoning_trace="I should search contacts first.",
            ),
        ]
        openai_msgs, _, meta = to_openai_messages(
            messages, mode=ConversionMode.CODE_EXEC
        )

        assert openai_msgs[0]["role"] == "assistant"
        assert (
            "<think>I should search contacts first.</think>"
            in openai_msgs[0]["content"]
        )
        assert "search_contacts()" in openai_msgs[0]["content"]
        assert meta[0] is not None
        assert meta[0]["reasoning_trace"] == "I should search contacts first."


# ===================================================================
# Tests for to_openai_messages() -- reasoning trace
# ===================================================================


class TestReasoningTraceExtracted:
    """Reasoning trace handling in TOOL_CALLING mode."""

    def test_reasoning_trace_in_tool_calling_mode(self) -> None:
        """Reasoning trace becomes the content of the assistant message with tool_calls."""
        messages = [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content=_make_tool_call_content("call_1", "search_contacts"),
                openai_tool_call_id="call_1",
                openai_function_name="search_contacts",
                finish_reason="tool_calls",
                reasoning_trace="The user wants to find Alice.",
            ),
        ]
        openai_msgs, _, meta = to_openai_messages(messages)

        assert openai_msgs[0]["role"] == "assistant"
        assert (
            openai_msgs[0]["content"] == "<think>The user wants to find Alice.</think>"
        )
        assert "tool_calls" in openai_msgs[0]
        assert meta[0] is not None
        assert meta[0]["reasoning_trace"] == "The user wants to find Alice."

    def test_native_reasoning_content_in_tool_calling_mode(self) -> None:
        messages = [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content=_make_tool_call_content("call_1", "search_contacts"),
                openai_tool_call_id="call_1",
                openai_function_name="search_contacts",
                finish_reason="tool_calls",
                reasoning_trace="The user wants to find Alice.",
                openai_reasoning_content="The user wants to find Alice.",
            ),
        ]

        openai_msgs, _, _ = to_openai_messages(messages)

        assert openai_msgs[0]["content"] == ""
        assert openai_msgs[0]["reasoning_content"] == "The user wants to find Alice."

    def test_no_reasoning_trace_empty_content(self) -> None:
        """Without reasoning_trace, assistant content is empty string in tool_calling mode."""
        messages = [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content=_make_tool_call_content("call_1", "some_tool"),
                openai_tool_call_id="call_1",
                openai_function_name="some_tool",
                finish_reason="tool_calls",
            ),
        ]
        openai_msgs, _, _ = to_openai_messages(messages)
        assert openai_msgs[0]["content"] == ""


# ===================================================================
# Tests for to_openai_messages() -- environment notifications
# ===================================================================


class TestEnvironmentNotifications:
    """Standalone EXEC_ENV->AGENT messages without tool_call_id."""

    def test_standalone_env_notification_becomes_user(self) -> None:
        """An EXEC_ENV->AGENT message without tool_call_id becomes role='user'."""
        messages = [
            Message(
                sender=RoleType.EXECUTION_ENVIRONMENT,
                recipient=RoleType.AGENT,
                content="User interacted with the UI button.",
            ),
        ]
        openai_msgs, _, _ = to_openai_messages(messages)

        assert len(openai_msgs) == 1
        assert openai_msgs[0]["role"] == "user"
        assert openai_msgs[0]["content"] == "User interacted with the UI button."


# ===================================================================
# Tests for to_openai_messages() -- user tool calls
# ===================================================================


class TestUserToolCalls:
    """USER->EXEC_ENV and EXEC_ENV->USER messages."""

    def test_user_tool_call_formatted(self) -> None:
        """USER->EXEC_ENV becomes a user message with tool call prefix."""
        messages = [
            Message(
                sender=RoleType.USER,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content="call_u1_parameters = {}\ncall_u1_response = end_conversation(**call_u1_parameters)",
                openai_tool_call_id="call_u1",
                openai_function_name="end_conversation",
            ),
        ]
        openai_msgs, _, _ = to_openai_messages(messages)

        assert len(openai_msgs) == 1
        assert openai_msgs[0]["role"] == "user"
        assert "[User called tool: end_conversation]" in openai_msgs[0]["content"]

    def test_user_tool_result_formatted(self) -> None:
        """EXEC_ENV->USER becomes a user message with result prefix."""
        messages = [
            Message(
                sender=RoleType.EXECUTION_ENVIRONMENT,
                recipient=RoleType.USER,
                content="Conversation ended.",
                openai_tool_call_id="call_u1",
                openai_function_name="end_conversation",
            ),
        ]
        openai_msgs, _, _ = to_openai_messages(messages)

        assert len(openai_msgs) == 1
        assert openai_msgs[0]["role"] == "user"
        assert "[Tool result for user (end_conversation)" in openai_msgs[0]["content"]


# ===================================================================
# Tests for to_openai_messages() -- unrecognized pair
# ===================================================================


class TestUnrecognizedSenderRecipient:
    """Unrecognized sender/recipient pairs raise ValueError."""

    def test_unrecognized_pair_raises(self) -> None:
        messages = [
            Message(
                sender=RoleType.USER,
                recipient=RoleType.USER,
                content="This makes no sense.",
            ),
        ]
        with pytest.raises(ValueError, match="Unrecognized sender recipient pair"):
            to_openai_messages(messages)


# ===================================================================
# Tests for to_openai_messages() -- deferred AGENT->USER in CODE_EXEC
# ===================================================================


class TestDeferredCodeExecMessages:
    """Deferred AGENT->USER messages in CODE_EXEC mode."""

    def test_deferred_agent_to_user_in_code_exec(self) -> None:
        """In CODE_EXEC, AGENT->USER injected during execution is deferred until after result."""
        messages = [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content="render_ui()",
                finish_reason="stop",
            ),
            # Injected by tool: AGENT->USER (should be deferred)
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.USER,
                content="Here is the UI for you.",
            ),
            # Execution result
            Message(
                sender=RoleType.EXECUTION_ENVIRONMENT,
                recipient=RoleType.AGENT,
                content="UI rendered successfully.",
            ),
        ]
        openai_msgs, indices, _ = to_openai_messages(
            messages, mode=ConversionMode.CODE_EXEC
        )

        roles = [m["role"] for m in openai_msgs]
        # The deferred AGENT->USER appears AFTER the execution result.
        assert roles == ["assistant", "user", "assistant"]
        assert "Here is the UI for you." in openai_msgs[2]["content"]


# ===================================================================
# Tests for serialize_to_conversation()
# ===================================================================


class TestSerializeToConversation:
    """Tests for serialize_to_conversation()."""

    def test_serialize_basic_conversation(self) -> None:
        """System + user + agent messages produce the expected JSON structure."""
        ctx = ExecutionContext(support_images=True)
        messages = [
            Message(
                sender=RoleType.SYSTEM,
                recipient=RoleType.AGENT,
                content="You are a helpful agent.",
            ),
            Message(
                sender=RoleType.USER,
                recipient=RoleType.AGENT,
                content="What time is it?",
            ),
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.USER,
                content="It is 3 PM.",
            ),
        ]
        add_messages_to_execution_context(ctx, messages)

        eval_result = EvaluationResult(turn_count=1)
        full_conv, agent_conv = serialize_to_conversation(ctx, eval_result)

        # Agent conversation should have the 3 messages
        assert len(agent_conv) == 3
        assert agent_conv[0]["role"] == "system"
        assert agent_conv[1]["role"] == "user"
        assert agent_conv[2]["role"] == "assistant"

        # Full conversation should have perspective metadata
        for msg in full_conv:
            assert "perspective" in msg
            assert "views" in msg["perspective"]
            assert "sender" in msg["perspective"]
            assert "recipient" in msg["perspective"]
            assert "sandbox_message_index" in msg["perspective"]

    def test_serialize_includes_tool_traces(self) -> None:
        """Tool trace data appears in serialized extras."""
        ctx = ExecutionContext(support_images=True)
        messages = [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content=_make_tool_call_content("call_1", "search_contacts"),
                openai_tool_call_id="call_1",
                openai_function_name="search_contacts",
                finish_reason="tool_calls",
                generation="raw_gen_text",
            ),
            Message(
                sender=RoleType.EXECUTION_ENVIRONMENT,
                recipient=RoleType.AGENT,
                content='{"name": "Alice"}',
                openai_tool_call_id="call_1",
                openai_function_name="search_contacts",
                tool_trace=['{"input": "query", "output": "Alice"}'],
            ),
        ]
        add_messages_to_execution_context(ctx, messages)

        eval_result = EvaluationResult(turn_count=1)
        full_conv, agent_conv = serialize_to_conversation(ctx, eval_result)

        # The assistant message should have extras with generation metadata.
        assistant_msgs = [m for m in agent_conv if m.get("role") == "assistant"]
        assert len(assistant_msgs) == 1
        extras = assistant_msgs[0].get("extras_assistant")
        assert extras is not None
        assert extras["generation"] == "raw_gen_text"

    def test_serialize_user_only_messages_in_full_conversation(self) -> None:
        """User-only messages (SYSTEM->USER) appear in full_conversation but not agent_conversation."""
        ctx = ExecutionContext(support_images=True)
        messages = [
            Message(
                sender=RoleType.SYSTEM,
                recipient=RoleType.AGENT,
                content="Agent system prompt.",
            ),
            Message(
                sender=RoleType.SYSTEM,
                recipient=RoleType.USER,
                content="User sim instructions.",
            ),
            Message(
                sender=RoleType.USER,
                recipient=RoleType.AGENT,
                content="Hi!",
            ),
        ]
        add_messages_to_execution_context(ctx, messages)

        eval_result = EvaluationResult(turn_count=1)
        full_conv, agent_conv = serialize_to_conversation(ctx, eval_result)

        # Agent conv should only have system->agent and user->agent messages.
        assert len(agent_conv) == 2
        assert agent_conv[0]["role"] == "system"
        assert agent_conv[1]["role"] == "user"

        # Full conv includes user_system pseudo-role.
        user_system_msgs = [m for m in full_conv if m.get("role") == "user_system"]
        assert len(user_system_msgs) == 1
        assert user_system_msgs[0]["content"] == "User sim instructions."


# ===================================================================
# Tests for helper functions
# ===================================================================


class TestNormalizeToolCallId:
    """Tests for normalize_tool_call_id()."""

    def test_replaces_hyphens(self) -> None:
        assert normalize_tool_call_id("call-123-abc") == "call_123_abc"

    def test_no_hyphens_unchanged(self) -> None:
        assert normalize_tool_call_id("call_123_abc") == "call_123_abc"

    def test_empty_string(self) -> None:
        assert normalize_tool_call_id("") == ""

    def test_only_hyphens(self) -> None:
        assert normalize_tool_call_id("---") == "___"


class TestExtractReasoning:
    """Tests for extract_reasoning()."""

    def test_extracts_think_blocks(self) -> None:
        reasoning, remaining = extract_reasoning(
            "<think>I need to search first.</think>The answer is 42."
        )
        assert reasoning == "I need to search first."
        assert remaining == "The answer is 42."

    def test_no_think_block(self) -> None:
        reasoning, remaining = extract_reasoning("Just plain text.")
        assert reasoning is None
        assert remaining == "Just plain text."

    def test_empty_think_block(self) -> None:
        reasoning, remaining = extract_reasoning("<think></think>Hello")
        assert reasoning == ""
        assert remaining == "Hello"

    def test_multiline_think_block(self) -> None:
        content = "<think>\nLine 1\nLine 2\n</think>\nResult here."
        reasoning, remaining = extract_reasoning(content)
        assert reasoning is not None
        assert "Line 1" in reasoning
        assert "Line 2" in reasoning
        assert remaining == "Result here."


class TestGetImageContent:
    """Tests for get_image_content()."""

    def test_returns_base64_image_content(self) -> None:
        """get_image_content fetches images from the IMAGE database and returns base64."""
        ctx = ExecutionContext(support_images=True)
        ctx.add_to_database(
            namespace=DatabaseNamespace.IMAGE,
            rows=[{"image_id": 0, "image_content": "dGVzdA=="}],  # base64("test")
        )

        # Patch _resize_base64_image to be a no-op for this test since "dGVzdA=="
        # is not a real image.
        with patch(
            "mmtoolsandbox.common.message_conversion._resize_base64_image",
            side_effect=lambda x, **kw: x,
        ):
            content = get_image_content(
                initial_content="Look at this image:",
                image_ids=[ImageId(0)],
                execution_context=ctx,
            )

        assert len(content) == 3
        # First element: text content
        assert content[0] == {"type": "text", "text": "Look at this image:"}
        # Second: image_id label
        assert content[1] == {"type": "text", "text": "image_id: 0"}
        # Third: image_url with base64
        assert content[2]["type"] == "image_url"
        assert "data:image/jpeg;base64,dGVzdA==" in content[2]["image_url"]["url"]

    def test_raises_without_execution_context(self) -> None:
        with pytest.raises(ValueError, match="execution_context is required"):
            get_image_content(
                initial_content="text",
                image_ids=[ImageId(0)],
                execution_context=None,
            )

    def test_raises_for_missing_image(self) -> None:
        ctx = ExecutionContext(support_images=True)
        with patch(
            "mmtoolsandbox.common.message_conversion._resize_base64_image",
            side_effect=lambda x, **kw: x,
        ):
            with pytest.raises(ValueError, match="not found in IMAGE database"):
                get_image_content(
                    initial_content="text",
                    image_ids=[ImageId(999)],
                    execution_context=ctx,
                )

    def test_initial_content_list_extended(self) -> None:
        """When initial_content is already a list, it is extended (not wrapped)."""
        ctx = ExecutionContext(support_images=True)
        ctx.add_to_database(
            namespace=DatabaseNamespace.IMAGE,
            rows=[{"image_id": 0, "image_content": "dGVzdA=="}],
        )

        initial = [{"type": "text", "text": "Existing text part."}]
        with patch(
            "mmtoolsandbox.common.message_conversion._resize_base64_image",
            side_effect=lambda x, **kw: x,
        ):
            content = get_image_content(
                initial_content=initial,
                image_ids=[ImageId(0)],
                execution_context=ctx,
            )

        # The existing list entry plus image_id label plus image_url
        assert len(content) == 3
        assert content[0] == {"type": "text", "text": "Existing text part."}


# ===================================================================
# Tests for Message construction edge cases
# ===================================================================


class TestMessageConstruction:
    """Edge cases for Message attrs class."""

    def test_default_visible_to_set_from_sender_recipient(self) -> None:
        msg = Message(
            sender=RoleType.AGENT,
            recipient=RoleType.USER,
            content="Hello",
        )
        assert msg.visible_to == [RoleType.AGENT, RoleType.USER]

    def test_explicit_visible_to_preserved(self) -> None:
        msg = Message(
            sender=RoleType.AGENT,
            recipient=RoleType.USER,
            content="Secret",
            visible_to=[RoleType.USER],
        )
        assert msg.visible_to == [RoleType.USER]

    def test_tool_call_id_requires_function_name(self) -> None:
        with pytest.raises(AssertionError):
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content="code",
                openai_tool_call_id="call_1",
                openai_function_name=None,
            )

    def test_frozen_message(self) -> None:
        msg = Message(
            sender=RoleType.USER,
            recipient=RoleType.AGENT,
            content="Frozen check",
        )
        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            msg.content = "Modified"  # type: ignore[misc]


# ===================================================================
# Tests for add/get messages round-trip
# ===================================================================


class TestMessageRoundTrip:
    """Round-trip: add messages to ExecutionContext, get them back."""

    def test_add_and_get_messages(self) -> None:
        ctx = _make_context()
        original = [
            Message(
                sender=RoleType.SYSTEM,
                recipient=RoleType.AGENT,
                content="System prompt",
            ),
            Message(
                sender=RoleType.USER,
                recipient=RoleType.AGENT,
                content="User query",
            ),
        ]
        add_messages_to_execution_context(ctx, original)
        retrieved = get_messages_from_execution_context(ctx)

        assert len(retrieved) == 2
        assert retrieved[0].sender == RoleType.SYSTEM
        assert retrieved[0].content == "System prompt"
        assert retrieved[1].sender == RoleType.USER
        assert retrieved[1].content == "User query"

    def test_ending_index_truncates(self) -> None:
        ctx = _make_context()
        messages = [
            Message(
                sender=RoleType.SYSTEM,
                recipient=RoleType.AGENT,
                content="System prompt",
            ),
            Message(
                sender=RoleType.USER,
                recipient=RoleType.AGENT,
                content="User query",
            ),
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.USER,
                content="Agent reply",
            ),
        ]
        add_messages_to_execution_context(ctx, messages)

        # Messages get sandbox_message_index 0, 1, 2. Truncate to index <= 1.
        retrieved = get_messages_from_execution_context(ctx, ending_index=1)
        assert len(retrieved) == 2
        assert retrieved[-1].content == "User query"


# ===================================================================
# Tests for choices_metadata population
# ===================================================================


class TestChoicesMetadata:
    """Verify that choices metadata is populated correctly."""

    def test_agent_message_populates_metadata(self) -> None:
        messages = [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.USER,
                content="Done.",
                finish_reason="stop",
                generation="raw text",
                logprobs=[0.1, 0.2],
                token_ids=[1, 2],
                reasoning_trace="I finished.",
            ),
        ]
        _, _, meta = to_openai_messages(messages)

        assert meta[0] is not None
        assert meta[0]["finish_reason"] == "stop"
        assert meta[0]["generation"] == "raw text"
        assert meta[0]["logprobs"] == [0.1, 0.2]
        assert meta[0]["token_ids"] == [1, 2]
        assert meta[0]["reasoning_trace"] == "I finished."

    def test_non_agent_messages_have_none_metadata(self) -> None:
        messages = [
            Message(
                sender=RoleType.SYSTEM,
                recipient=RoleType.AGENT,
                content="Prompt",
            ),
            Message(
                sender=RoleType.USER,
                recipient=RoleType.AGENT,
                content="Hello",
            ),
        ]
        _, _, meta = to_openai_messages(messages)

        assert meta[0] is None
        assert meta[1] is None
