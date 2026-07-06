# Copyright © 2026 Apple Inc.

"""Unit tests for ReACT-style reasoning extraction."""

from mmtoolsandbox.common.message_conversion import extract_reasoning


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
