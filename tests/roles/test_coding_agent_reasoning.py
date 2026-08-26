# Copyright © 2026 Apple Inc.

"""Tests for reasoning traces in code-execution mode."""

from types import SimpleNamespace
from unittest.mock import patch

from openai.types.chat import ChatCompletionMessage

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.message_conversion import (
    Message,
    extract_reasoning,
    to_openai_messages_for_code_exec,
)
from mmtoolsandbox.roles.code_execution_agent import (
    CodeExecutionAgent,
    extract_code_blocks,
)


def test_extract_reasoning_then_code_blocks() -> None:
    """Reasoning extraction should not interfere with code block extraction."""
    content = (
        "<think>I need to search for calendar tools.</think>\n\n"
        "```python\napi_docs_search_api_docs(query='calendar')\n```"
    )
    reasoning, cleaned = extract_reasoning(content)
    assert reasoning == "I need to search for calendar tools."
    blocks = extract_code_blocks(cleaned)
    assert len(blocks) == 1
    assert "api_docs_search_api_docs" in blocks[0]


def test_extract_reasoning_no_think_with_code() -> None:
    """Without think tags, code blocks should be extracted normally."""
    content = "Let me search:\n\n```python\nprint('hello')\n```"
    reasoning, cleaned = extract_reasoning(content)
    assert reasoning is None
    assert cleaned == content
    blocks = extract_code_blocks(cleaned)
    assert len(blocks) == 1


def test_code_exec_roundtrip_with_reasoning() -> None:
    """Reasoning trace should be reconstructed with <think> tags in code-exec messages."""
    messages = [
        Message(
            sender=RoleType.AGENT,
            recipient=RoleType.EXECUTION_ENVIRONMENT,
            content="```python\nprint('hello')\n```",
            reasoning_trace="I should print hello.",
        ),
    ]
    openai_msgs = to_openai_messages_for_code_exec(messages)
    assert len(openai_msgs) == 1
    assert openai_msgs[0]["role"] == "assistant"
    assert openai_msgs[0]["content"].startswith("<think>I should print hello.</think>")
    assert "```python" in openai_msgs[0]["content"]


def test_code_exec_native_reasoning_roundtrip() -> None:
    content = "```python\nprint('hello')\n```"
    response = SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                message=ChatCompletionMessage.model_validate({
                    "role": "assistant",
                    "content": content,
                    "reasoning": "I should print hello.",
                }),
                finish_reason="stop",
            )
        ],
    )
    messages = [
        Message(
            sender=RoleType.USER,
            recipient=RoleType.AGENT,
            content="Print hello.",
        )
    ]
    agent = CodeExecutionAgent.__new__(CodeExecutionAgent)

    with (
        patch.object(agent, "messages_validation"),
        patch.object(agent, "filter_messages", return_value=messages),
        patch.object(agent, "_convert_messages", return_value=[]),
        patch.object(agent, "model_inference", return_value=response),
        patch("mmtoolsandbox.roles.code_execution_agent.get_current_context"),
    ):
        response_messages, _ = agent.respond(messages, {})

    assert len(response_messages) == 1
    message = response_messages[0]
    assert message.reasoning_trace == "I should print hello."
    assert message.openai_reasoning_content == message.reasoning_trace

    openai_msgs = to_openai_messages_for_code_exec(response_messages)
    assert openai_msgs[0]["content"] == content
    assert openai_msgs[0]["reasoning_content"] == "I should print hello."


def test_code_exec_roundtrip_without_reasoning() -> None:
    """Without reasoning, code-exec messages should be unchanged."""
    messages = [
        Message(
            sender=RoleType.AGENT,
            recipient=RoleType.EXECUTION_ENVIRONMENT,
            content="```python\nprint('hello')\n```",
        ),
    ]
    openai_msgs = to_openai_messages_for_code_exec(messages)
    assert openai_msgs[0]["content"] == "```python\nprint('hello')\n```"


def test_code_exec_agent_to_user_roundtrip_with_reasoning() -> None:
    """AGENT→USER messages should also reconstruct <think> tags."""
    messages = [
        Message(
            sender=RoleType.AGENT,
            recipient=RoleType.USER,
            content="Done! I sent the message.",
            reasoning_trace="Task is complete, confirming to user.",
        ),
    ]
    openai_msgs = to_openai_messages_for_code_exec(messages)
    assert (
        "<think>Task is complete, confirming to user.</think>"
        in openai_msgs[0]["content"]
    )
    assert "Done! I sent the message." in openai_msgs[0]["content"]


def test_code_exec_agent_to_user_no_reasoning() -> None:
    """AGENT→USER without reasoning should be plain content."""
    messages = [
        Message(
            sender=RoleType.AGENT,
            recipient=RoleType.USER,
            content="Here is the result.",
        ),
    ]
    openai_msgs = to_openai_messages_for_code_exec(messages)
    assert openai_msgs[0]["content"] == "Here is the result."
