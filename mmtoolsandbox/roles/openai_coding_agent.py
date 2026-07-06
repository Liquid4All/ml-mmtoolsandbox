# Copyright © 2026 Apple Inc.

"""Agent role that uses code generation with synthetic tool_call/tool_result history.

Extends ``CodeExecutionAgent`` with:

1. **Code wrapping** — the extracted code block is rewritten so that the last
   expression is captured in a ``{tool_id}_response`` variable, which the
   ``ExecutionEnvironment`` reads after ``exec()``.
2. **Synthetic tool_call history** — past code executions are represented as
   ``tool_calls`` (assistant) + ``role: "tool"`` (result) in the OpenAI
   message format, giving the model unambiguous signal about what is a tool
   result vs. a real user message.
"""

from __future__ import annotations

import ast
import json
from typing import (
    Any,
)

from mmtoolsandbox.common.execution_context import (
    RoleType,
)
from mmtoolsandbox.common.message_conversion import (
    Message,
    get_image_content,
    normalize_tool_call_id,
)
from mmtoolsandbox.roles.code_execution_agent import (
    CodeExecutionAgent,
)
from mmtoolsandbox.roles.openai_agent import (
    OpenAIResponsesCodeExecAgent,
)
from mmtoolsandbox.roles.registry import register_role_class

# ---------------------------------------------------------------------------
# Code wrapping helper
# ---------------------------------------------------------------------------


def _wrap_code_for_execution(tool_id: str, code: str) -> str:
    """Wrap free-form Python code so the execution environment can extract the result.

    The execution environment looks for a variable named '{tool_id}_response' in locals
    after execution. This function modifies the code to assign the result of the last
    expression to that variable.

    Args:
        tool_id:    The normalized tool call ID (valid Python identifier).
        code:       The raw Python code from the model.

    Returns:
        Wrapped Python code with result assignment.
    """
    tool_id = normalize_tool_call_id(tool_id)
    result_var = f"{tool_id}_response"

    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Let the execution environment handle the syntax error
        return code

    if not tree.body:
        return code

    last_stmt = tree.body[-1]
    if isinstance(last_stmt, ast.Expr):
        # Last statement is a bare expression (e.g., a function call).
        # Get the source lines and replace the last expression with an assignment.
        lines = code.split("\n")
        last_line_start = last_stmt.lineno - 1  # 0-indexed
        last_line_end = last_stmt.end_lineno or last_stmt.lineno  # 1-indexed

        prefix_lines = lines[:last_line_start]
        last_expr_lines = lines[last_line_start:last_line_end]
        suffix_lines = lines[last_line_end:]

        last_expr_code = "\n".join(last_expr_lines)
        wrapped_last = f"{result_var} = {last_expr_code.strip()}"

        result_lines = prefix_lines + [wrapped_last] + suffix_lines
        return "\n".join(result_lines)
    elif isinstance(last_stmt, ast.Assign) and len(last_stmt.targets) == 1:
        # Last statement is a simple assignment (e.g., result = func(...)).
        # Copy the assigned variable to the response variable.
        target = last_stmt.targets[0]
        if isinstance(target, ast.Name):
            return f"{code}\n{result_var} = {target.id}"
        else:
            return f"{code}\n{result_var} = 'Code executed successfully.'"
    else:
        # Last statement is not an expression or simple assignment (e.g., for loop).
        return f"{code}\n{result_var} = 'Code executed successfully.'"


# ---------------------------------------------------------------------------
# Message conversion with synthetic tool_calls
# ---------------------------------------------------------------------------


def to_openai_messages_with_tool_calls(
    messages: list[Message],
    execution_context: Any | None = None,
) -> list[dict[str, Any]]:
    """Convert message history using synthetic tool_call / tool result format.

    AGENT->EXEC_ENV messages become assistant messages with a ``tool_calls``
    entry.  EXEC_ENV->AGENT messages become ``role: "tool"`` messages with a
    matching ``tool_call_id``.  This gives the model unambiguous signal about
    what is a tool result vs. a real user message.

    Args:
        messages:           List of Message objects.
        execution_context:  Optional execution context for image lookup.

    Returns:
        List of OpenAI API format message dicts.
    """
    openai_messages: list[dict[str, Any]] = []

    for message in messages:
        if message.sender == RoleType.SYSTEM and message.recipient == RoleType.AGENT:
            openai_messages.append({"role": "system", "content": message.content})

        elif message.sender == RoleType.SYSTEM and message.recipient == RoleType.USER:
            openai_messages.append({"role": "system", "content": message.content})

        elif message.sender == RoleType.USER and message.recipient == RoleType.AGENT:
            if message.image_ids and execution_context is not None:
                content = get_image_content(
                    initial_content=message.content,
                    image_ids=message.image_ids,
                    execution_context=execution_context,
                )
                openai_messages.append({"role": "user", "content": content})
            else:
                openai_messages.append({"role": "user", "content": message.content})

        elif message.sender == RoleType.AGENT and message.recipient == RoleType.USER:
            openai_messages.append({"role": "assistant", "content": message.content})

        elif (
            message.sender == RoleType.AGENT
            and message.recipient == RoleType.EXECUTION_ENVIRONMENT
        ):
            # Represent as assistant message with a synthetic tool_call
            tool_call_id = message.openai_tool_call_id or "call_unknown"
            func_name = message.openai_function_name or "execute_code"
            tool_call_entry = {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps({"code": message.content}),
                },
            }
            # Aggregate with previous assistant tool_calls if consecutive
            if (
                openai_messages
                and openai_messages[-1].get("role") == "assistant"
                and "tool_calls" in openai_messages[-1]
            ):
                openai_messages[-1]["tool_calls"].append(tool_call_entry)
                continue
            openai_messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call_entry],
                }
            )

        elif (
            message.sender == RoleType.EXECUTION_ENVIRONMENT
            and message.recipient == RoleType.AGENT
        ):
            # Represent as a tool result message
            result_content = message.content
            if message.tool_call_exception:
                result_content += f"\n[Error]: {message.tool_call_exception}"
            openai_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": message.openai_tool_call_id or "call_unknown",
                    "content": result_content,
                }
            )

        else:
            raise ValueError(
                f"Unrecognized sender recipient pair {(message.sender, message.recipient)}"
            )

    return openai_messages


# ---------------------------------------------------------------------------
# OpenAIAPICodingAgent
# ---------------------------------------------------------------------------


class OpenAIAPICodingAgent(CodeExecutionAgent):
    """Agent that uses code generation with synthetic tool_call history.

    Extends ``CodeExecutionAgent`` with:
    - Code wrapping so the ``ExecutionEnvironment`` can extract results via
      the ``{tool_id}_response`` variable convention.
    - Synthetic tool_call / tool_result messages in conversation history for
      unambiguous disambiguation of execution results vs. user messages.
    """

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _prepare_code_for_execution(self, tool_call_id: str, code: str) -> str:
        return _wrap_code_for_execution(tool_call_id, code)

    def _execution_message_content(
        self, _truncated_content: str, final_code: str
    ) -> str:
        return final_code

    def _get_execution_function_name(self) -> str:
        return "execute_code"

    def _convert_messages(
        self,
        messages: list[Message],
        execution_context: Any | None = None,
    ) -> list[dict[str, Any]]:
        return to_openai_messages_with_tool_calls(messages, execution_context)


@register_role_class
class GPT_5_4_2026_03_05_CodeExec_Agent(CodeExecutionAgent):
    model_name = "gpt-5.4-2026-03-05"
    reasoning_effort = "high"


# ---------------------------------------------------------------------------
# Responses API code-execution agents (native reasoning)
# ---------------------------------------------------------------------------


@register_role_class
class GPT_5_4_2026_03_05_Reasoning_CodeExec_Agent(  # type: ignore[misc]
    OpenAIResponsesCodeExecAgent, CodeExecutionAgent
):
    model_name = "gpt-5.4-2026-03-05"
    reasoning_effort = "medium"
