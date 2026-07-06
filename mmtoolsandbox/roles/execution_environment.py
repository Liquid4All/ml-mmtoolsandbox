# Copyright © 2026 Apple Inc.

"""Python Execution Environment"""

from __future__ import annotations

import ast
import code
import io
import json
import logging
import sys
import traceback
from typing import (
    Any,
    Callable,
    Sequence,
    cast,
)

import polars as pl
from attrs import evolve

from mmtoolsandbox.common.execution_context import (
    DatabaseNamespace,
    RoleType,
    get_current_context,
)
from mmtoolsandbox.common.image_id import ImageId, contains_images, extract_image_ids
from mmtoolsandbox.common.introspection_databases import IntrospectionDatabaseNamespace
from mmtoolsandbox.common.json_utils import MMToolSandboxJSONEncoder
from mmtoolsandbox.common.message_conversion import Message, normalize_tool_call_id
from mmtoolsandbox.common.utils import NOT_GIVEN
from mmtoolsandbox.roles.base_role import BaseRole
from mmtoolsandbox.roles.code_execution_agent import CODE_BLOCK_FUNCTION_NAME
from mmtoolsandbox.tools.tool_sandbox.user_tools import end_conversation

# Capture the real SystemExit before any safety guard can monkeypatch it.
# The safety guard replaces builtins.SystemExit with Exception to prevent
# agent code from killing the process, but that causes "except SystemExit"
# to catch ALL exceptions.
_REAL_SYSTEM_EXIT = SystemExit

LOGGER = logging.getLogger(__name__)


def _execute_direct_code_block(
    interactive_console: code.InteractiveConsole,
    message: Message,
    role_type: RoleType,
) -> Message | None:
    """Execute a raw code block directly with stdout capture and last-expression eval.

    This is the execution path for ``CodeExecutionAgent`` (pure code execution
    mode).  Unlike the standard path which looks for a ``{tool_id}_response``
    variable after ``exec()``, this path:

    1. Redirects stdout to capture ``print()`` output.
    2. Parses the AST to separate statements from a trailing expression.
    3. ``exec()`` all statements, ``eval()`` the last expression.
    4. Returns captured stdout + last expression value as the result.

    Safety guardrails (syntax check + runtime guard) are applied identically
    to the ``execute_code`` tool.

    Args:
        interactive_console: The interactive console whose locals contain
                             pre-loaded tools.
        message:             The AGENT -> EXEC_ENV message containing raw code.
        role_type:           The role type of the responder (EXECUTION_ENVIRONMENT).

    Returns:
        A response message, or ``None`` if the sender is SYSTEM.
    """
    # The message content may be the full model response (text + code blocks).
    # Extract the first code block for execution.
    from mmtoolsandbox.roles.code_execution_agent import extract_code_blocks

    code_blocks = extract_code_blocks(message.content)
    code_str = code_blocks[0] if code_blocks else message.content
    locals_dict = cast(dict[str, Any], interactive_console.locals)

    # Safety guard: only apply to non-SYSTEM messages.
    safety_guard = get_current_context().safety_guard
    is_guarded = safety_guard is not None and message.sender != RoleType.SYSTEM

    # Layer 1: Static syntax check
    if (
        is_guarded
        and safety_guard is not None
        and safety_guard.config.enable_syntax_check
    ):
        is_safe, safety_msg = safety_guard.is_syntax_safe(code_str)
        if not is_safe:
            blocked_content = f"Execution blocked: {safety_msg}"
            return Message(
                sender=role_type,
                recipient=message.sender,
                content=blocked_content,
                openai_tool_call_id=message.openai_tool_call_id,
                openai_function_name=message.openai_function_name,
                tool_call_exception=blocked_content,
            )
    # Redirect stdout to capture print() output
    old_stdout = sys.stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Layer 2: Enable runtime monkey-patching
    if (
        is_guarded
        and safety_guard is not None
        and safety_guard.config.enable_runtime_guard
    ):
        safety_guard.enable()

    exec_traceback_str = None
    full_traceback_str = None
    last_value = None

    # NOTE: The safety guard replaces builtins.SystemExit with Exception
    # to prevent agent code from killing the process.  We must match only
    # the REAL SystemExit here, otherwise "except SystemExit" would catch
    # ALL exceptions and re-raise them, preventing error recovery.
    try:
        tree = ast.parse(code_str)

        if tree.body:
            last_stmt = tree.body[-1]
            if isinstance(last_stmt, ast.Expr):
                # Execute all but the last statement
                if len(tree.body) > 1:
                    exec_tree = ast.Module(body=tree.body[:-1], type_ignores=[])
                    exec(compile(exec_tree, "<string>", "exec"), locals_dict)
                # Evaluate the last expression and capture its value
                last_value = eval(
                    compile(ast.Expression(body=last_stmt.value), "<string>", "eval"),
                    locals_dict,
                )
            else:
                # No trailing expression, just exec everything
                exec(code_str, locals_dict)
        else:
            exec(code_str, locals_dict)

    except _REAL_SYSTEM_EXIT:
        raise
    except Exception:
        exec_traceback_str = traceback.format_exc(limit=0).rstrip()
        full_traceback_str = traceback.format_exc()
    finally:
        sys.stdout = old_stdout
        # Layer 2: Restore original functions
        if (
            is_guarded
            and safety_guard is not None
            and safety_guard.config.enable_runtime_guard
        ):
            safety_guard.disable()

    if full_traceback_str is not None:
        LOGGER.debug(
            "Exception in direct code block execution:\n%s", full_traceback_str
        )

    # If this message is from system, do not respond
    if message.sender == RoleType.SYSTEM:
        return None

    output = captured_output.getvalue()

    # Check if the last value contains images
    has_images = contains_images(last_value)

    if has_images:
        image_ids: list[ImageId] | None = extract_image_ids(last_value)
        # For image results, JSON-encode the non-image parts
        parts = []
        if output:
            parts.append(output.rstrip())
        if last_value is not None and not has_images:
            parts.append(repr(last_value))
        content = "\n".join(parts) if parts else ""
    else:
        image_ids = None
        # Build text result: stdout + last expression value
        parts = []
        if output:
            parts.append(output.rstrip())
        if last_value is not None:
            parts.append(repr(last_value))
        content = (
            "\n".join(parts) if parts else "Code executed successfully (no output)"
        )

    if exec_traceback_str is not None:
        if content:
            content += f"\n{exec_traceback_str}"
        else:
            content = exec_traceback_str

    if not image_ids:
        image_ids = None

    return Message(
        sender=role_type,
        recipient=message.sender,
        content=content,
        openai_tool_call_id=message.openai_tool_call_id,
        openai_function_name=message.openai_function_name,
        tool_call_exception=exec_traceback_str,
        image_ids=image_ids,
    )


def respond_to_single_message(
    interactive_console: code.InteractiveConsole,
    message: Message,
    role_type: RoleType,
) -> Message | None:
    """Respond to a single message.

    Args:
        interactive_console: The interactive console to use for executing the function
                             call.
        message:             The message to which to respond.
        role_type:           The role type of the responder (e.g. sender of the
                             response).

    Returns:
        A message if there was a response. More specifically, if the sender is the
        `system` role then the returned value is `None`.
    """
    # Direct code block execution (from CodeExecutionAgent): execute with
    # stdout capture + last-expression eval instead of variable extraction.
    if message.openai_function_name == CODE_BLOCK_FUNCTION_NAME:
        return _execute_direct_code_block(interactive_console, message, role_type)

    # First compile the code string to a command, which checks if the command is
    # valid and complete (regarding completeness see `if command is None` below for
    # details). Note that we cannot use the default `symbol=single` since that assumes
    # that we want to only generate a single statement. This would result in a syntax
    # error for e.g. `a = 1\nb = 2` saying
    #    "multiple statements found while compiling a single statement"
    try:
        command = code.compile_command(message.content, symbol="exec")
    except (OverflowError, SyntaxError, ValueError):
        # We do not want to leak details like the code path so we only include the
        # actual exception in the traceback.
        traceback_str = traceback.format_exc(limit=0)
        return Message(
            sender=role_type,
            recipient=message.sender,
            content=traceback_str,
            openai_tool_call_id=message.openai_tool_call_id,
            openai_function_name=message.openai_function_name,
            tool_call_exception=traceback_str,
        )
    if command is None:
        # `None` is returned when the given code string is incomplete. An example
        # would be `code.compile_command("if True:")`. The LLM should not generate
        # such code so we consider this an error/failure.
        response = f"Error: The given code was incomplete and could not be executed: '{message.content}'"
        return Message(
            sender=role_type,
            recipient=message.sender,
            content=response,
            openai_tool_call_id=message.openai_tool_call_id,
            openai_function_name=message.openai_function_name,
            tool_call_exception=response,
        )

    # We assume NOT_GIVEN to be an invalid tool response value to detect whether tool execution succeeded, since
    # None is a possible tool response value.
    tool_result = NOT_GIVEN
    exec_traceback_str = None
    full_traceback_str = None

    # Safety guard: only apply to non-SYSTEM messages (SYSTEM messages are trusted setup code).
    safety_guard = get_current_context().safety_guard
    is_guarded = safety_guard is not None and message.sender != RoleType.SYSTEM

    # Layer 1: Static syntax check
    if (
        is_guarded
        and safety_guard is not None
        and safety_guard.config.enable_syntax_check
    ):
        is_safe, safety_msg = safety_guard.is_syntax_safe(message.content)
        if not is_safe:
            blocked_content = f"Execution blocked: {safety_msg}"
            return Message(
                sender=role_type,
                recipient=message.sender,
                content=blocked_content,
                openai_tool_call_id=message.openai_tool_call_id,
                openai_function_name=message.openai_function_name,
                tool_call_exception=blocked_content,
            )

    # Execute the code. At this point we know that the code is valid Python, but it
    # can still throw exceptions.
    # The below is effectively the same as interactive_console.runcode with custom exception
    # handling logic.
    try:
        # For non-AGENT roles (like USER), we execute in a temporary copy of locals that includes their specific tools.
        # This allows the User Simulator to call tools like end_conversation without exposing
        # them to the Agent's persistent environment.
        if message.sender != RoleType.AGENT and message.sender != RoleType.SYSTEM:
            exec_locals = dict(interactive_console.locals)
            sender_tools = get_current_context().get_available_tools_for_role(
                message.sender
            )
            for name, tool in sender_tools.items():
                exec_locals[name] = tool
        else:
            exec_locals = cast(dict[str, Any], interactive_console.locals)

        # We execute the command directly.
        # For standard tools, they are trusted and should run without the safety guard.
        # For `execute_code`, it internally manages the safety guard (enable/disable)
        # to ensure the untrusted user code is executed safely.
        exec(command, exec_locals)

        # The tool call result will be stored in a variable called '{tool_id}_response'
        # so we can just extract it from the locals of the interactive console.
        # Note: The variable name uses the normalized ID (hyphens replaced with underscores)
        # to ensure valid Python syntax, even though message.openai_tool_call_id may contain hyphens.
        if message.openai_tool_call_id is not None:
            tool_id = normalize_tool_call_id(message.openai_tool_call_id)
            tool_result_var_name = f"{tool_id}_response"
            tool_result = exec_locals.get(tool_result_var_name, NOT_GIVEN)
    except _REAL_SYSTEM_EXIT:  # Raise as interactive_console.runcode does
        raise
    except Exception:
        # We do not want to leak details like the code path so we only include the
        # actual exception in the traceback.
        exec_traceback_str = traceback.format_exc(limit=0).rstrip()
        full_traceback_str = traceback.format_exc()

    if full_traceback_str is not None:
        LOGGER.debug(
            "Exception in `%s`:\n%s", message.openai_function_name, full_traceback_str
        )

    # Reconstruct what would've been flushed to the console from stdout and stderr in a threadsafe fashion.
    # Dumps result into json.
    content_lines = (
        [json.dumps(tool_result, cls=MMToolSandboxJSONEncoder, ensure_ascii=False)]
        if tool_result is not NOT_GIVEN
        else []
    )
    if exec_traceback_str is not None:
        content_lines.extend(exec_traceback_str.split("\n"))

    # If this message is from system, do not respond
    if message.sender == RoleType.SYSTEM:
        return None

    # Get image ids from result, if the tool returned an ImageResult
    image_ids: list[ImageId] | None = extract_image_ids(tool_result)

    if not image_ids:
        image_ids = None

    # TODO: Right now because we don't immediately respond after processing a message
    # The state update of all messages are attached to the first responded message.
    # Fix this by responding immediately instead. Remember to move tool trace processing here as well
    # Propagate visibility from request to response for user tool calls.
    # When the user sets visible_to to include AGENT, the response should too.
    propagated_visible_to: list[RoleType] | None = None
    if message.sender == RoleType.USER and message.visible_to is not None:
        default_vis = {message.sender, message.recipient}
        actual_vis = set(message.visible_to)
        if actual_vis != default_vis:
            propagated_visible_to = message.visible_to

    return Message(
        sender=role_type,
        recipient=message.sender,
        content="\n".join(content_lines),
        openai_tool_call_id=message.openai_tool_call_id,
        openai_function_name=message.openai_function_name,
        tool_call_exception=exec_traceback_str,
        image_ids=image_ids,
        visible_to=propagated_visible_to,
    )


def respond_to_messages(
    interactive_console: code.InteractiveConsole,
    messages: Sequence[Message],
    role_type: RoleType,
) -> list[Message]:
    """Respond to the given messages.

    Args:
        interactive_console: The interactive console to use for executing the function
                             call.
        messages:            The messages to which to respond.
        role_type:           The role type of the responder (e.g. sender of the
                             response).

    Returns:
        The response messages. Note that it can be an empty list if the messages came
        from the `system` role.
    """
    response_messages = [
        respond_to_single_message(interactive_console, message, role_type)
        for message in messages
    ]
    return [message for message in response_messages if message is not None]


def get_messages_to_process(
    messages: list[Message], recipient: RoleType
) -> list[Message]:
    """Filter out the message to which the execution environment should respond.

    Args:
        messages:   All messages of the current conversation.
        recipient:  The role type of the recipient.

    Returns:
        The messages to which the `recipient` should respond.
    """
    new_messages_index = len(messages) - 1
    while new_messages_index >= 0:
        if messages[new_messages_index].recipient != recipient:
            break
        new_messages_index -= 1
    messages_to_process = messages[new_messages_index + 1 :]
    return messages_to_process


class ExecutionEnvironment(BaseRole):
    """An Execution Environment able to execute python code in an REPL console in a stateful manner
    Note that this happens in the same process and thread as your main process, just under a different scope
    """

    role_type: RoleType = RoleType.EXECUTION_ENVIRONMENT

    def respond(
        self,
        messages: list[Message],
        available_tools: dict[str, Callable[..., Any]],
        external_tool_schemas: list[dict[str, Any]] | None = None,
    ) -> tuple[list[Message], dict[IntrospectionDatabaseNamespace, Any]]:
        """Reads a List of messages and attempt to respond with a Message

        Specifically, reads python source code from other Roles, executes and return with REPL env stdout / stderr
        System could provide necessary imports and init commands at the start, in which case we won't respond,
        just silently execute the code snippet
        User could provide tool call to terminate the conversation
        Agent could provide tool call to help complete the ongoing task

        Message comes from current context, the last k messages should be directed to this role type
        Response are written to current context as well. k new messages, addressed to appropriate recipient

        Args:
            messages:               List of Message objects containing full dialog history until this point
            available_tools:        Dictionary of role facing tool names to tool callable object,
                                    which also contains the execution facing name in tool.__name__
            external_tool_schemas:  Optional list of external tool schemas that are not registered in the repo

        Returns:
            A list of responses messages, and a dictionary of entries into introspection database

        Raises:
            KeyError:   When the last message is not directed to this role
        """
        self.messages_validation(messages)

        # Some LLMs (e.g. GPT-3.5 turbo) can return multiple function calls in a single
        # request, which is called parallel function calling. Thus, the execution
        # environment may have to process multiple messages.
        messages_to_process = get_messages_to_process(
            messages, recipient=self.role_type
        )

        # When external tool schemas are provided, check if related tool calls are predicted
        # Terminate conversation if so
        if external_tool_schemas is not None:
            # Build set of allowed tool names including external tools
            external_tool_names = {
                tool["function"]["name"] for tool in external_tool_schemas
            }
            # Check if any message uses a tool externally available
            for message in messages_to_process:
                if message.openai_function_name in external_tool_names:
                    LOGGER.debug(
                        "Ending conversation since function %s is an external tool.",
                        message.openai_function_name,
                    )
                    end_conversation()
                    return [], {}

        response_messages = respond_to_messages(
            interactive_console=get_current_context().interactive_console,
            messages=messages_to_process,
            role_type=self.role_type,
        )
        # TODO: Cleanly resolve issues where we need to apply System -> ExecutionEnvironment messages
        #   while resuming execution.
        if not response_messages:
            return response_messages, {}
        # At this point the tool calls have been performed and the global execution
        # context reflects the changes introduced by the tool calls.
        current_context = get_current_context()

        # Since tool_trace is technically the outcome of calling a tool, assign tool_trace collected in the last
        # Agent -> ExecutionEnvironment message to ExecutionEnvironment -> Agent messages
        # TODO: This isn't always guaranteed to be correct if one of the tool calls failed. Fix this by logging a
        #   tool_trace for exceptions as well
        # Note that tool traces are being stored in permuted order, while responses messages are in original order.
        # Need to reorder tool trace accordingly.
        tool_trace_series: pl.Series = current_context.get_database(
            DatabaseNamespace.SANDBOX
        )["tool_trace"]
        if not tool_trace_series.is_empty() and tool_trace_series[0] is not None:
            tool_trace_list = tool_trace_series[0].to_list()
            # Erase tool trace collected in Agent -> ExecutionEnvironment message
            current_context.update_database(
                DatabaseNamespace.SANDBOX,
                current_context.get_database(DatabaseNamespace.SANDBOX).with_columns(
                    pl.lit(None).alias("tool_trace")
                ),
            )
            # Assign tool traces to response messages.
            #
            # Traces are appended in execution order by the register_as_tool
            # decorator. For parallel tool calls (without permutation) the
            # execution order matches the message order, so index-based
            # assignment is correct.
            #
            # A tool call that raised an exception won't have a trace, so we
            # skip exception messages. The number of traces may also be less
            # than the number of successful responses if a tool execution
            # produced no trace for any other reason.
            tool_trace_index = 0
            for i in range(len(response_messages)):
                if i == len(response_messages) - 1:
                    # Last message gets all remaining traces. This handles:
                    # 1. Coding agent: single message, multiple tool calls
                    # 2. Partial traces from earlier successful calls
                    remaining_traces = tool_trace_list[tool_trace_index:]
                    if remaining_traces:
                        response_messages[i] = evolve(
                            response_messages[i], tool_trace=remaining_traces
                        )
                    tool_trace_index = len(tool_trace_list)
                elif response_messages[i].tool_call_exception is not None:
                    continue
                elif tool_trace_index < len(tool_trace_list):
                    response_messages[i] = evolve(
                        response_messages[i],
                        tool_trace=[tool_trace_list[tool_trace_index]],
                    )
                    tool_trace_index += 1

        return response_messages, {}
