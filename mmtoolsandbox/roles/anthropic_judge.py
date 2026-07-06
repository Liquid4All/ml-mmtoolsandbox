# Copyright © 2026 Apple Inc.

"""Judge role for Claude models via the native Anthropic API.

The judge reuses the evidence-formatting helpers from ``OpenAIAPIJudge``
and swaps the inference calls for the Anthropic SDK.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, cast

import anthropic
from tenacity import retry, stop_after_attempt, wait_random_exponential

from mmtoolsandbox.common.execution_context import ExecutionContext
from mmtoolsandbox.common.utils import all_logging_disabled
from mmtoolsandbox.roles.base_judge import (
    _SYSTEM_PROMPT,
    _UI_SYSTEM_PROMPT,
    JudgeResult,
    get_user_system_prompt,
)
from mmtoolsandbox.roles.openai_judge import OpenAIAPIJudge

LOGGER = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)
_INNER_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """Extract JSON from Claude responses that may contain reasoning text.

    Tries three strategies in order:
    1. The entire response is a code-fenced JSON block.
    2. A code-fenced JSON block is embedded in surrounding reasoning text.
    3. A raw JSON object (``{ ... }``) appears somewhere in the text.

    Falls back to the original text if none of the above match.
    """
    # Strategy 1: entire response is a code fence
    m = _CODE_FENCE_RE.match(text)
    if m:
        return m.group(1)

    # Strategy 2: code fence embedded in reasoning text
    m = _INNER_FENCE_RE.search(text)
    if m:
        return m.group(1)

    # Strategy 3: find the outermost { ... } JSON object
    start = text.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    return text


def _openai_parts_to_anthropic(
    parts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert OpenAI-style content parts to Anthropic content blocks.

    Transforms ``{"type": "image_url", "image_url": {"url": "data:..."}}``
    into ``{"type": "image", "source": {"type": "base64", ...}}``.
    Text blocks pass through unchanged.
    """
    anthropic_blocks: list[dict[str, Any]] = []
    for part in parts:
        if part.get("type") == "image_url":
            data_url: str = part["image_url"]["url"]
            match = re.match(r"data:([^;]+);base64,(.+)", data_url, re.DOTALL)
            if match:
                anthropic_blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": match.group(1),
                            "data": match.group(2),
                        },
                    }
                )
        else:
            anthropic_blocks.append(part)
    return anthropic_blocks


def _parse_judge_response(
    response: anthropic.types.Message, fallback_label: str
) -> dict[str, Any]:
    """Extract text, parse JSON, and wrap as a JudgeResult dict."""
    text_parts = [
        block.text
        for block in response.content
        if isinstance(block, anthropic.types.TextBlock)
    ]
    content = "\n".join(text_parts)
    if not content:
        return {"result": False, "reasoning": f"No response from {fallback_label}."}

    try:
        raw: dict[str, Any] = json.loads(_strip_code_fences(content))
        return JudgeResult.from_dict(raw).to_dict()
    except (json.JSONDecodeError, TypeError, KeyError):
        return {
            "result": False,
            "reasoning": f"Failed to parse {fallback_label} response: {content}",
        }


class AnthropicAPIJudge(OpenAIAPIJudge):
    """Judge using Claude models via the native Anthropic API.

    Inherits evidence-formatting logic from :class:`OpenAIAPIJudge` and
    overrides the inference calls to use the Anthropic SDK.

    Authentication:
        Reads the ``ANTHROPIC_API_KEY`` environment variable via the default
        behavior of :class:`anthropic.Anthropic`.
    """

    def __init__(self, model_name: str) -> None:
        # Skip OpenAIAPIJudge.__init__ — no OpenAI client needed.
        self.model_name = model_name
        self.anthropic_client: anthropic.Anthropic = anthropic.Anthropic(max_retries=10)
        self.last_evidence: dict[str, list[dict[str, Any]]] = {}

    def _messages_create(
        self,
        system_prompt: str,
        evidence_parts: list[dict[str, Any]],
    ) -> anthropic.types.Message:
        """Single-shot call to Claude with a system prompt and user content."""
        anthropic_content = _openai_parts_to_anthropic(evidence_parts)
        with all_logging_disabled(highest_level=logging.WARNING):
            return self.anthropic_client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                system=system_prompt,
                messages=cast(
                    list[anthropic.types.MessageParam],
                    [{"role": "user", "content": anthropic_content}],
                ),
                temperature=0,
            )

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
    )
    def evaluate_from_evidence(
        self,
        evidence_parts: list[dict[str, Any]],
        system_prompt: str,
    ) -> dict[str, Any]:
        """Call the Anthropic judge LLM with pre-built evidence parts."""
        response = self._messages_create(system_prompt, evidence_parts)
        return _parse_judge_response(response, "judge model")

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
    )
    def evaluate(
        self,
        execution_context: ExecutionContext,
        criteria: str,
        appworld_initial: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate the execution context against the task-completion criteria."""
        evidence_parts = self._format_evidence(
            execution_context, criteria, appworld_initial
        )
        self.last_evidence["main"] = evidence_parts
        response = self._messages_create(_SYSTEM_PROMPT, evidence_parts)
        return _parse_judge_response(response, "judge model")

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
    )
    def evaluate_ui(
        self,
        execution_context: ExecutionContext,
        criteria: str,
    ) -> dict[str, Any]:
        """Evaluate UI quality using the 4-criterion UI rubric."""
        evidence_parts = self._format_ui_evidence(execution_context, criteria)
        self.last_evidence["ui"] = evidence_parts
        response = self._messages_create(_UI_SYSTEM_PROMPT, evidence_parts)
        return _parse_judge_response(response, "UI judge model")

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
    )
    def evaluate_user(
        self,
        execution_context: ExecutionContext,
        max_messages: int = 40,
        turn_count: int | None = None,
        scenario_metadata: dict[str, Any] | None = None,
        enable_ui: bool = False,
    ) -> dict[str, Any]:
        """Evaluate user-simulator quality using the 4-criterion user rubric."""
        evidence_parts = self._format_user_evidence(
            execution_context,
            max_messages=max_messages,
            turn_count=turn_count,
            scenario_metadata=scenario_metadata,
        )
        self.last_evidence["user"] = evidence_parts
        system_prompt = get_user_system_prompt(enable_ui=enable_ui)
        response = self._messages_create(system_prompt, evidence_parts)
        return _parse_judge_response(response, "user judge model")


class Claude_4_5_Sonnet_Judge(AnthropicAPIJudge):
    def __init__(self) -> None:
        super().__init__("claude-sonnet-4-5-20250929")
