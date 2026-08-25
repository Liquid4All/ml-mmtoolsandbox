# Copyright © 2026 Apple Inc.

"""Unit tests for reasoning extraction."""

from openai.types.chat import ChatCompletionMessage

from mmtoolsandbox.common.message_conversion import extract_reasoning
from mmtoolsandbox.roles.openai_agent import _extract_openai_reasoning


def test_extract_reasoning_with_tags() -> None:
    reasoning, remaining = extract_reasoning("<think>I need to search first.</think>")
    assert reasoning == "I need to search first."
    assert remaining == ""


def test_extract_reasoning_with_surrounding_text() -> None:
    reasoning, remaining = extract_reasoning(
        "<think>Thinking...</think>\nHere is the answer."
    )
    assert reasoning == "Thinking..."
    assert remaining == "Here is the answer."


def test_extract_reasoning_no_tags() -> None:
    reasoning, remaining = extract_reasoning("Just a plain answer.")
    assert reasoning is None
    assert remaining == "Just a plain answer."


def test_extract_reasoning_empty_string() -> None:
    reasoning, remaining = extract_reasoning("")
    assert reasoning is None
    assert remaining == ""


def test_extract_reasoning_multiline() -> None:
    content = (
        "<think>\nStep 1: Look up the contact.\n"
        "Step 2: Send the message.\n</think>\n"
        "I'll help you with that."
    )
    reasoning, remaining = extract_reasoning(content)
    assert reasoning is not None
    assert "Step 1" in reasoning
    assert "Step 2" in reasoning
    assert remaining == "I'll help you with that."


def test_extract_reasoning_empty_tags() -> None:
    reasoning, remaining = extract_reasoning("<think></think>Some text")
    assert reasoning == ""
    assert remaining == "Some text"


def test_extract_openai_native_reasoning() -> None:
    message = ChatCompletionMessage.model_validate(
        {
            "role": "assistant",
            "content": "I will search now.",
            "reasoning_content": "I need the search tool.",
        }
    )

    reasoning, content, native_reasoning = _extract_openai_reasoning(message)

    assert reasoning == "I need the search tool."
    assert content == "I will search now."
    assert native_reasoning == reasoning


def test_extract_openai_reasoning_alias() -> None:
    message = ChatCompletionMessage.model_validate(
        {
            "role": "assistant",
            "content": None,
            "reasoning": "I need the search tool.",
        }
    )

    reasoning, content, native_reasoning = _extract_openai_reasoning(message)

    assert reasoning == "I need the search tool."
    assert content == ""
    assert native_reasoning == reasoning


def test_extract_openai_inline_reasoning() -> None:
    message = ChatCompletionMessage(
        role="assistant",
        content="<think>I need the search tool.</think>I will search now.",
    )

    reasoning, content, native_reasoning = _extract_openai_reasoning(message)

    assert reasoning == "I need the search tool."
    assert content == "I will search now."
    assert native_reasoning is None
