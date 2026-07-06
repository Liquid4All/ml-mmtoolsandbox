# Copyright © 2026 Apple Inc.

"""Base agent for pure code execution mode.

In this mode the agent generates Python code in markdown fenced blocks (text
completion — no function calling).  The code is extracted, sent to the
ExecutionEnvironment for direct execution, and the result comes back as a
tool-execution message wrapped in ``<execution_results>`` XML tags.

The agent discovers tools dynamically via ``api_docs_search_api_docs()``
which is available in the execution environment.  No tool documentation is
injected into the system prompt by this class — that is handled externally
(e.g. by ``transform_scenario_capabilities``).

``CodeExecutionAgent`` is the base class.  ``OpenAIAPICodingAgent`` extends
it with the legacy ``_wrap_code_for_execution`` behaviour and tool-doc
injection.
"""

from __future__ import annotations

import logging
import re
from typing import (
    Any,
    Callable,
    Literal,
    cast,
)

from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat import (
    ChatCompletionToolParam,
)

from mmtoolsandbox.common.execution_context import (
    RoleType,
    get_current_context,
)
from mmtoolsandbox.common.introspection_databases import IntrospectionDatabaseNamespace
from mmtoolsandbox.common.message_conversion import (
    ConversionMode,
    Message,
    extract_reasoning,
    to_openai_messages,
)
from mmtoolsandbox.common.tool_conversion import (
    generate_tool_call_id,
)
from mmtoolsandbox.roles.openai_agent import OpenAIAPIAgent

LOGGER = logging.getLogger(__name__)

# Marker used as openai_function_name so the ExecutionEnvironment knows to
# execute the code block directly (stdout capture + last-expression eval)
# rather than looking for a ``{tool_id}_response`` variable.
CODE_BLOCK_FUNCTION_NAME = "code_block"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_code_blocks(text: str) -> list[str]:
    """Extract Python code blocks from markdown-formatted text.

    Args:
        text:   The text to extract code blocks from.

    Returns:
        A list of code strings extracted from ```python fenced blocks.
    """
    return re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)


# ---------------------------------------------------------------------------
# CodeExecutionAgent
# ---------------------------------------------------------------------------


class CodeExecutionAgent(OpenAIAPIAgent):
    """Agent that generates code as text and sends it for direct execution.

    The agent writes Python code in markdown fenced blocks.  Code is
    extracted and sent to the ExecutionEnvironment for direct execution
    (stdout capture + last-expression eval, like a REPL).

    Tool discovery happens at runtime via ``api_docs_search_api_docs()``
    inside the execution environment — this class does NOT inject tool
    documentation into the system prompt.
    """

    def respond(
        self,
        messages: list[Message],
        available_tools: dict[str, Callable[..., Any]],
        external_tool_schemas: list[dict[str, Any]] | None = None,
    ) -> tuple[list[Message], dict[IntrospectionDatabaseNamespace, Any]]:
        """Generate a text response, extract code blocks, send for execution.

        Args:
            messages:               Full dialog history.
            available_tools:        Tool names -> callables visible to the agent.
            external_tool_schemas:  Optional external tool schemas.

        Returns:
            Response messages and introspection database entries.
        """
        response_messages: list[Message] = []
        self.messages_validation(messages=messages)
        messages = self.filter_messages(messages=messages)

        if messages[-1].sender == RoleType.SYSTEM:
            return [], {}

        # Convert messages to OpenAI format (no tool-calling constructs)
        execution_context = get_current_context()
        openai_messages = self._convert_messages(messages, execution_context)

        # Debug: log raw messages sent to the API
        LOGGER.debug("=== CodeExecutionAgent API call ===")
        for i, msg in enumerate(openai_messages):
            content = msg.get("content", "")
            if isinstance(content, str):
                LOGGER.debug(
                    "  [%d] role=%s, content=%s",
                    i,
                    msg.get("role"),
                    content,
                )
            elif isinstance(content, list):
                text_parts = [
                    p.get("text", "") for p in content if p.get("type") == "text"
                ]
                LOGGER.debug(
                    "  [%d] role=%s, multimodal (%d parts), text=%s",
                    i,
                    msg.get("role"),
                    len(content),
                    "; ".join(text_parts),
                )
        LOGGER.debug("=== End messages ===")

        # Call model WITHOUT tools parameter (pure text completion)
        response = self.model_inference(
            openai_messages=cast(
                list[
                    dict[
                        Literal[
                            "role", "content", "tool_call_id", "name", "tool_calls"
                        ],
                        Any,
                    ]
                ],
                openai_messages,
            ),
            openai_tools=cast(
                list[ChatCompletionToolParam] | NotGiven,
                NOT_GIVEN,
            ),
        )
        if response.usage is not None:
            self._record_token_usage(
                response.usage.prompt_tokens, response.usage.completion_tokens
            )

        response_content = response.choices[0].message.content
        openai_response_finish_reason: str = response.choices[0].finish_reason

        if response_content is None:
            response_content = ""

        # Extract <think> reasoning before code extraction so tags don't
        # interfere with code block parsing or end up in executed code.
        reasoning_trace, cleaned_content = extract_reasoning(response_content)

        # Extract code blocks from the cleaned (tag-free) content
        code_blocks = extract_code_blocks(cleaned_content)

        if not code_blocks:
            # No code blocks — natural language response to user
            response_messages = [
                Message(
                    sender=self.role_type,
                    recipient=RoleType.USER,
                    content=cleaned_content,
                    reasoning_trace=reasoning_trace,
                    finish_reason=openai_response_finish_reason,
                )
            ]
        else:
            # Use the first code block (prompt instructs model to use a single block)
            code_block = code_blocks[0]
            tool_call_id = generate_tool_call_id()

            # Hook: subclasses can transform the code (e.g. add wrapping)
            final_code = self._prepare_code_for_execution(tool_call_id, code_block)

            # Truncate the response to include only the text up to and including
            # the first code block.  Models sometimes hallucinate fake
            # <execution_results> and subsequent code blocks in a single turn;
            # keeping only the first block prevents those from polluting history.
            first_fence_start = cleaned_content.find("```python")
            first_fence_end = cleaned_content.find(
                "```", first_fence_start + len("```python")
            )
            if first_fence_start >= 0 and first_fence_end >= 0:
                truncated_content = cleaned_content[
                    : first_fence_end + len("```")
                ].rstrip()
            else:
                truncated_content = cleaned_content

            response_messages.append(
                Message(
                    sender=self.role_type,
                    recipient=RoleType.EXECUTION_ENVIRONMENT,
                    content=self._execution_message_content(
                        truncated_content, final_code
                    ),
                    openai_tool_call_id=tool_call_id,
                    openai_function_name=self._get_execution_function_name(),
                    reasoning_trace=reasoning_trace,
                    finish_reason=openai_response_finish_reason,
                )
            )

        return response_messages, {}

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def _prepare_code_for_execution(self, tool_call_id: str, code: str) -> str:
        """Transform extracted code before sending to ExecutionEnvironment.

        The base implementation returns the code unchanged (direct execution).
        ``OpenAIAPICodingAgent`` overrides this to wrap the code so that the
        result is stored in a ``{tool_id}_response`` variable.
        """
        return code

    def _execution_message_content(
        self, truncated_content: str, final_code: str
    ) -> str:
        """Return the content for the AGENT->EXEC_ENV message.

        The base implementation uses the full (truncated) model response, which
        includes the markdown fences.  ``_execute_direct_code_block`` knows how
        to extract code from fenced blocks, so this is correct for the base
        ``code_block`` execution path.

        ``OpenAIAPICodingAgent`` overrides this to return ``final_code`` because
        its execution path (``execute_code``) calls ``code.compile_command``
        directly on ``message.content`` and must receive plain Python — not
        markdown-fenced text.
        """
        return truncated_content

    def _get_execution_function_name(self) -> str:
        """Return the ``openai_function_name`` attached to outgoing messages.

        The ExecutionEnvironment uses this to decide how to execute the code
        and extract results.
        """
        return CODE_BLOCK_FUNCTION_NAME

    def _convert_messages(
        self,
        messages: list[Message],
        execution_context: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Convert internal messages to OpenAI API format.

        The base implementation uses ``<execution_results>`` XML tags in user
        messages.  ``OpenAIAPICodingAgent`` overrides this to use synthetic
        ``tool_calls`` / ``role: "tool"`` messages.
        """
        msgs, _, _ = to_openai_messages(
            messages, execution_context, mode=ConversionMode.CODE_EXEC
        )
        return msgs  # type: ignore[return-value]
