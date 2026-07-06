# Copyright © 2026 Apple Inc.

"""Agent role for any model that conforms to OpenAI tool use API"""

from __future__ import annotations

import json
import logging
from typing import (
    Any,
    Callable,
    Literal,
    cast,
)

from openai import OpenAI
from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
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
    openai_tool_call_to_python_code,
    to_openai_messages,
    tool_call_with_dict_to_python_code,
)
from mmtoolsandbox.common.tool_conversion import convert_to_openai_tools
from mmtoolsandbox.common.utils import all_logging_disabled
from mmtoolsandbox.roles.base_role import BaseRole
from mmtoolsandbox.roles.registry import register_role_class


# TODO: Refactor all this mess
class OpenAIAPIAgent(BaseRole):
    """Agent role for any model that conforms to OpenAI tool use API"""

    role_type: RoleType = RoleType.AGENT
    model_name: str
    raise_error_on_unknown_tools: bool = True
    tool_images_as_user_message: bool = True
    reasoning_effort: str | None = None

    def __init__(self) -> None:
        super().__init__()
        self.openai_client: OpenAI = OpenAI()

    def respond(
        self,
        messages: list[Message],
        available_tools: dict[str, Callable[..., Any]],
        external_tool_schemas: list[dict[str, Any]] | None = None,
    ) -> tuple[list[Message], dict[IntrospectionDatabaseNamespace, Any]]:
        """Reads a List of messages and attempt to respond with list of Messages.

        Specifically, interprets system, user, execution environment messages and sends out NL response to user, or
        code snippet to execution environment.

        Message comes from current context, the last k messages should be directed to this role type
        Response are written to current context as well. n new messages, addressed to appropriate recipient
        k != n when dealing with parallel function call and responses. Parallel function call are expanded into
        individual messages, parallel function call responses are combined as 1 OpenAI API request

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
        response_messages: list[Message] = []
        self.messages_validation(messages=messages)
        # Keeps only relevant messages
        messages = self.filter_messages(messages=messages)
        # Does not respond to System
        if messages[-1].sender == RoleType.SYSTEM:
            return [], {}
        # Get OpenAI tools if most recent message is from user
        available_tool_names = set(available_tools.keys())

        # Add external tool names to available tool names if external tools are provided
        if external_tool_schemas:
            external_tool_names = {
                tool["function"]["name"] for tool in external_tool_schemas
            }
            available_tool_names.update(external_tool_names)

        openai_tools: list[dict[str, Any]] | NotGiven = NOT_GIVEN
        if (
            messages[-1].sender == RoleType.USER
            or messages[-1].sender == RoleType.EXECUTION_ENVIRONMENT
        ):
            # Convert registered tools to OpenAI format
            converted_tools = convert_to_openai_tools(
                name_to_tool=available_tools,
            )

            # Add external tool schemas if provided
            if external_tool_schemas:
                converted_tools.extend(external_tool_schemas)

            openai_tools = converted_tools

        # Convert to OpenAI messages.
        execution_context = get_current_context()
        openai_messages, _, _ = to_openai_messages(
            messages,
            execution_context,
            tool_images_as_user_message=self.tool_images_as_user_message,
            evict_stale_ui_images=True,
        )
        # Call model
        # We need a cast here since `convert_to_openai_tool` returns a plain dict, but
        # `ChatCompletionToolParam` is a `TypedDict`.
        response = self.model_inference(
            openai_messages=openai_messages,
            openai_tools=cast(
                list[ChatCompletionToolParam] | NotGiven,
                openai_tools,
            ),
        )
        if response.usage is not None:
            self._record_token_usage(
                response.usage.prompt_tokens, response.usage.completion_tokens
            )
        # Parse response
        openai_response_message = response.choices[0].message
        openai_response_finish_reason: str = response.choices[0].finish_reason
        openai_response_log_probs: list[float] | None = None
        openai_response_raw_generation: str | None = None
        openai_response_token_ids: list[int] | None = None
        # Fill in RL utilities
        if (
            response.choices[0].logprobs is not None
            and response.choices[0].logprobs.content is not None
        ):
            openai_response_log_probs = [
                x.logprob for x in response.choices[0].logprobs.content
            ]
            openai_response_raw_generation = openai_response_message.content
            openai_response_token_ids = [
                int(x.token.removeprefix("token_id:"))
                for x in response.choices[0].logprobs.content
            ]

        # Message contains no tool call, aka addressed to user
        # This is intentionally casting tool_calls to boolean,
        # covering both tool_calls = None and tool_calls = []
        if not openai_response_message.tool_calls:
            assert openai_response_message.content is not None
            content = openai_response_message.content
            # Extract <think>...</think> reasoning if present
            reasoning_trace, content = extract_reasoning(content)
            response_messages = [
                Message(
                    sender=self.role_type,
                    recipient=RoleType.USER,
                    content=content,
                    reasoning_trace=reasoning_trace,
                    finish_reason=openai_response_finish_reason,
                    logprobs=openai_response_log_probs,
                    generation=openai_response_raw_generation,
                    token_ids=openai_response_token_ids,
                )
            ]
        else:
            assert openai_tools is not NOT_GIVEN
            # Capture reasoning trace and text response from model content.
            # Text inside <think> tags → reasoning_trace.
            # Remaining text → text_response (preserved for
            # round-tripping to the model on subsequent turns).
            reasoning_trace = None
            text_response = None
            if openai_response_message.content:
                reasoning_trace, remaining = extract_reasoning(
                    openai_response_message.content
                )
                if remaining:
                    text_response = remaining
            for tool_call in openai_response_message.tool_calls:
                # The response contains the agent facing tool name so we need to get
                # the execution facing tool name when creating the Python code.

                # Skip non-function tool calls (e.g., custom tool calls)
                if not isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
                    continue

                # Default to agent facing name if this is not available (e.g. Agent hallucination)
                # Note that in this case, an error will be raised in ExecutionEnvironment, and shown
                # to the agent.
                execution_facing_tool_name = (
                    available_tools[tool_call.function.name].__name__
                    if tool_call.function.name in available_tools
                    else tool_call.function.name
                )
                response_messages.append(
                    Message(
                        sender=self.role_type,
                        recipient=RoleType.EXECUTION_ENVIRONMENT,
                        content=openai_tool_call_to_python_code(
                            tool_call,
                            available_tool_names,
                            execution_facing_tool_name=execution_facing_tool_name,
                            raise_error_on_unknown_tools=self.raise_error_on_unknown_tools,
                        ),
                        openai_tool_call_id=tool_call.id,
                        openai_function_name=tool_call.function.name,
                        reasoning_trace=reasoning_trace,
                        tool_call_text_response=text_response,
                        finish_reason=openai_response_finish_reason,
                        logprobs=openai_response_log_probs,
                        generation=openai_response_raw_generation,
                        token_ids=openai_response_token_ids,
                    )
                )
                # Only attach reasoning and text to the first tool call in a
                # parallel batch, matching the pattern used by the Claude agent.
                reasoning_trace = None
                text_response = None
        return response_messages, {}

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(10),
        # Do not set retry explicitly. Retry on all exceptions.
    )
    def model_inference(
        self,
        openai_messages: list[
            dict[
                Literal["role", "content", "tool_call_id", "name", "tool_calls"],
                Any,
            ]
        ],
        openai_tools: list[ChatCompletionToolParam] | NotGiven,
    ) -> ChatCompletion:
        """Run OpenAI model inference

        Args:
            openai_messages:    List of OpenAI API format messages
            openai_tools:       List of OpenAI API format tools definition

        Returns:
            OpenAI API chat completion object
        """
        kwargs: dict[str, Any] = {}
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        with all_logging_disabled(highest_level=logging.WARNING):
            return self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=cast(list[ChatCompletionMessageParam], openai_messages),
                tools=openai_tools,  # type: ignore[arg-type]  # SDK expects NotGiven→Omit
                **kwargs,
            )


@register_role_class
class GPT_5_4_2026_03_05_Agent(OpenAIAPIAgent):
    model_name = "gpt-5.4-2026-03-05"
    reasoning_effort = "high"


# ---------------------------------------------------------------------------
# Responses API agents (native reasoning support)
# ---------------------------------------------------------------------------
#
# These agents use OpenAI's Responses API (`client.responses.create()`)
# instead of Chat Completions.  This enables:
#   - Explicit reasoning effort control (none → xhigh)
#   - Reasoning summaries for trajectory visualization
#   - Reasoning item passback for multi-step continuity
#
# Reasoning models (GPT-5.4, GPT-5.5) reason internally by default even
# through Chat Completions, but reasoning tokens are invisible and lost
# between turns.  The Responses API exposes reasoning items that can be
# passed back, preserving the model's chain of thought across tool calls.


class OpenAIResponsesAPIAgent(BaseRole):
    """Agent using OpenAI's Responses API with native reasoning support."""

    role_type: RoleType = RoleType.AGENT
    model_name: str
    reasoning_effort: str = "medium"
    raise_error_on_unknown_tools: bool = True
    tool_images_as_user_message: bool = True

    # Fields the Responses API returns on output items but rejects on input.
    _OUTPUT_ONLY_FIELDS = {"status"}

    @staticmethod
    def _sanitize_output_items(items: list[Any]) -> list[dict[str, Any]]:
        """Convert output items to dicts and strip output-only fields."""
        sanitized: list[dict[str, Any]] = []
        for item in items:
            if hasattr(item, "model_dump"):
                d = item.model_dump(exclude_none=True)
            elif isinstance(item, dict):
                d = {k: v for k, v in item.items() if v is not None}
            else:
                d = dict(item)
            for key in OpenAIResponsesAPIAgent._OUTPUT_ONLY_FIELDS:
                d.pop(key, None)
            sanitized.append(d)
        return sanitized

    def __init__(self) -> None:
        super().__init__()
        self.openai_client: OpenAI = OpenAI()
        # Accumulated output items from all responses in the current
        # tool-calling chain.  Includes reasoning + function_call items
        # from every round so they can all be passed back for continuity.
        self._chain_output_items: list[Any] = []
        self._chain_call_ids: set[str] = set()

    def respond(
        self,
        messages: list[Message],
        available_tools: dict[str, Callable[..., Any]],
        external_tool_schemas: list[dict[str, Any]] | None = None,
    ) -> tuple[list[Message], dict[IntrospectionDatabaseNamespace, Any]]:
        self.messages_validation(messages=messages)
        messages = self.filter_messages(messages=messages)
        if messages[-1].sender == RoleType.SYSTEM:
            return [], {}

        available_tool_names = set(available_tools.keys())
        if external_tool_schemas:
            available_tool_names.update(
                t["function"]["name"] for t in external_tool_schemas
            )

        tools = self._build_responses_tools(
            available_tools, external_tool_schemas, messages
        )
        input_items = self._build_responses_input(messages)

        response = self.model_inference(input_items, tools)

        if response.usage is not None:
            self._record_token_usage(
                response.usage.input_tokens, response.usage.output_tokens
            )

        # Accumulate output items for reasoning continuity.
        # If this response contains function calls, the chain continues —
        # append to the accumulated items so the next round sees all prior
        # reasoning.  If it's a final message (no function calls), reset
        # the chain since the next call will be a new user turn.
        has_function_calls = any(
            getattr(item, "type", None) == "function_call" for item in response.output
        )
        if has_function_calls:
            self._chain_output_items.extend(
                self._sanitize_output_items(response.output)
            )
            self._chain_call_ids.update(
                item.call_id
                for item in response.output
                if getattr(item, "type", None) == "function_call"
            )
        else:
            self._chain_output_items = []
            self._chain_call_ids = set()

        return self._parse_responses_output(
            response, available_tools, available_tool_names
        )

    # -- Input construction --------------------------------------------------

    def _build_responses_input(self, messages: list[Message]) -> list[Any]:
        """Build Responses API input, injecting stored reasoning items when
        the current messages contain tool results for a previous response."""
        execution_context = get_current_context()

        if self._chain_output_items and self._chain_call_ids:
            current_result_ids = {
                msg.openai_tool_call_id
                for msg in messages
                if msg.sender == RoleType.EXECUTION_ENVIRONMENT
                and msg.recipient == RoleType.AGENT
                and msg.openai_tool_call_id
            }
            if self._chain_call_ids & current_result_ids:
                return self._build_input_with_reasoning(messages, execution_context)

        openai_msgs, _, _ = to_openai_messages(
            messages,
            execution_context,
            tool_images_as_user_message=self.tool_images_as_user_message,
            evict_stale_ui_images=True,
        )

        # Build a map of tool_call_id → reasoning items from persisted
        # Messages so that historical turns also get their reasoning
        # items injected.
        reasoning_map = self._build_reasoning_map(messages)
        return self._chat_to_responses_format(
            cast(list[dict[str, Any]], openai_msgs), reasoning_map
        )

    def _build_input_with_reasoning(
        self,
        messages: list[Message],
        execution_context: Any,
    ) -> list[Any]:
        """Build input that preserves reasoning continuity.

        Splits the message list at the boundary of the first stored tool
        call: earlier history is converted to chat format; everything from
        that point onward is represented by the accumulated output items
        (which include reasoning from every round in the chain) plus
        ``function_call_output`` items for all tool results.
        """
        split_idx = len(messages)
        for i, msg in enumerate(messages):
            if (
                msg.sender == RoleType.AGENT
                and msg.recipient == RoleType.EXECUTION_ENVIRONMENT
                and msg.openai_tool_call_id in self._chain_call_ids
            ):
                split_idx = i
                break

        items: list[Any] = []
        historical = messages[:split_idx]
        if historical:
            openai_msgs, _, _ = to_openai_messages(
                historical,
                execution_context,
                tool_images_as_user_message=self.tool_images_as_user_message,
                evict_stale_ui_images=True,
            )
            items = self._chat_to_responses_format(
                cast(list[dict[str, Any]], openai_msgs)
            )

        items.extend(self._chain_output_items)

        for msg in messages[split_idx:]:
            if (
                msg.sender == RoleType.EXECUTION_ENVIRONMENT
                and msg.recipient == RoleType.AGENT
                and msg.openai_tool_call_id in self._chain_call_ids
            ):
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.openai_tool_call_id,
                        "output": msg.content,
                    }
                )

        return items

    @staticmethod
    def _build_reasoning_map(
        messages: list[Message],
    ) -> dict[str, list[dict[str, Any]]]:
        """Build a mapping from tool_call_id to deserialized reasoning items.

        Looks at all AGENT→EXEC Messages that carry persisted
        ``openai_reasoning_items`` and returns a dict keyed by
        ``openai_tool_call_id``.
        """
        reasoning_map: dict[str, list[dict[str, Any]]] = {}
        for msg in messages:
            if (
                msg.sender == RoleType.AGENT
                and msg.recipient == RoleType.EXECUTION_ENVIRONMENT
                and msg.openai_reasoning_items
                and msg.openai_tool_call_id
            ):
                reasoning_map[msg.openai_tool_call_id] = json.loads(
                    msg.openai_reasoning_items
                )
        return reasoning_map

    @staticmethod
    def _chat_to_responses_format(
        openai_msgs: list[dict[str, Any]],
        reasoning_map: dict[str, list[dict[str, Any]]] | None = None,
    ) -> list[Any]:
        """Convert Chat Completions messages to Responses API input format.

        Key differences from Chat Completions:
        - Assistant tool_calls become top-level ``function_call`` items
        - Tool results become ``function_call_output`` items
        - Multimodal content uses ``input_text`` / ``input_image`` types
        """
        items: list[Any] = []
        for msg in openai_msgs:
            role = msg.get("role")
            content = msg.get("content", "")

            if role in ("system", "user"):
                if isinstance(content, list):
                    converted = []
                    for part in content:
                        if part.get("type") == "text":
                            converted.append(
                                {"type": "input_text", "text": part["text"]}
                            )
                        elif part.get("type") == "image_url":
                            converted.append(
                                {
                                    "type": "input_image",
                                    "image_url": part["image_url"]["url"],
                                }
                            )
                    items.append({"role": role, "content": converted})
                else:
                    items.append({"role": role, "content": content})

            elif role == "assistant":
                if "tool_calls" in msg and msg["tool_calls"]:
                    if content:
                        items.append({"role": "assistant", "content": content})
                    for tc in msg["tool_calls"]:
                        # Inject persisted reasoning items before the
                        # function_call they preceded.
                        if reasoning_map and tc["id"] in reasoning_map:
                            items.extend(reasoning_map[tc["id"]])
                        fn = tc["function"]
                        items.append(
                            {
                                "type": "function_call",
                                "call_id": tc["id"],
                                "name": fn["name"],
                                "arguments": (
                                    fn["arguments"]
                                    if isinstance(fn["arguments"], str)
                                    else json.dumps(fn["arguments"])
                                ),
                            }
                        )
                else:
                    items.append({"role": "assistant", "content": content})

            elif role == "tool":
                tool_content = msg.get("content", "")
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg["tool_call_id"],
                        "output": (
                            tool_content
                            if isinstance(tool_content, str)
                            else json.dumps(tool_content)
                        ),
                    }
                )

        return items

    # -- Tool construction ---------------------------------------------------

    def _build_responses_tools(
        self,
        available_tools: dict[str, Callable[..., Any]],
        external_tool_schemas: list[dict[str, Any]] | None,
        messages: list[Message],
    ) -> list[dict[str, Any]] | NotGiven:
        """Build tools in Responses API format (flat, not nested under 'function')."""
        if messages[-1].sender not in (
            RoleType.USER,
            RoleType.EXECUTION_ENVIRONMENT,
        ):
            return NOT_GIVEN

        chat_tools = convert_to_openai_tools(
            name_to_tool=available_tools,
        )
        if external_tool_schemas:
            chat_tools.extend(external_tool_schemas)

        responses_tools: list[dict[str, Any]] = []
        for tool in chat_tools:
            if tool.get("type") == "function":
                fn = tool["function"]
                tool_def: dict[str, Any] = {
                    "type": "function",
                    "name": fn["name"],
                    "parameters": fn.get("parameters", {}),
                }
                if fn.get("description"):
                    tool_def["description"] = fn["description"]
                responses_tools.append(tool_def)

        return responses_tools

    # -- Response parsing ----------------------------------------------------

    def _parse_responses_output(
        self,
        response: Any,
        available_tools: dict[str, Callable[..., Any]],
        available_tool_names: set[str],
    ) -> tuple[list[Message], dict[IntrospectionDatabaseNamespace, Any]]:
        """Parse Responses API output into sandbox Messages.

        Extracts reasoning summaries and attaches them to the next tool-call
        or user-facing message as ``reasoning_trace`` for visualization.
        Serializes reasoning output items (including ``encrypted_content``)
        onto the first tool-call Message so they persist across turns.
        """
        response_messages: list[Message] = []
        reasoning_summary: str | None = None
        finish_reason = getattr(response, "status", "completed")

        # Collect reasoning items from this response for persistence.
        # These are serialized onto the first tool-call Message so they
        # can be reconstructed on future turns.
        pending_reasoning_items: list[dict[str, Any]] = []

        for item in response.output:
            item_type = getattr(item, "type", None)

            if item_type == "reasoning":
                summary = getattr(item, "summary", None)
                if summary:
                    texts = [s.text for s in summary if hasattr(s, "text")]
                    if texts:
                        reasoning_summary = "\n".join(texts)
                # Serialize for persistence (includes encrypted_content).
                if hasattr(item, "model_dump"):
                    d = item.model_dump(exclude_none=True)
                    for key in self._OUTPUT_ONLY_FIELDS:
                        d.pop(key, None)
                    pending_reasoning_items.append(d)
                else:
                    pending_reasoning_items.append({"type": "reasoning"})

            elif item_type == "function_call":
                execution_facing_name = (
                    available_tools[item.name].__name__
                    if item.name in available_tools
                    else item.name
                )

                if (
                    item.name not in available_tool_names
                    and self.raise_error_on_unknown_tools
                ):
                    raise KeyError(
                        f"Agent tool call {item.name!r} is not a known allowed "
                        f"tool. Options are {available_tool_names!r}."
                    )

                args = (
                    json.loads(item.arguments)
                    if isinstance(item.arguments, str)
                    else item.arguments
                )
                content = tool_call_with_dict_to_python_code(
                    tool_id=item.call_id,
                    function_name=execution_facing_name,
                    arguments_dict=args,
                )

                # Attach accumulated reasoning items to the first tool call.
                serialized_reasoning: str | None = None
                if pending_reasoning_items:
                    serialized_reasoning = json.dumps(pending_reasoning_items)
                    pending_reasoning_items = []

                response_messages.append(
                    Message(
                        sender=self.role_type,
                        recipient=RoleType.EXECUTION_ENVIRONMENT,
                        content=content,
                        openai_tool_call_id=item.call_id,
                        openai_function_name=item.name,
                        reasoning_trace=reasoning_summary,
                        openai_reasoning_items=serialized_reasoning,
                        finish_reason=finish_reason,
                    )
                )
                reasoning_summary = None

            elif item_type == "message":
                text_parts = []
                for block in getattr(item, "content", []):
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
                text = "".join(text_parts)

                response_messages.append(
                    Message(
                        sender=self.role_type,
                        recipient=RoleType.USER,
                        content=text,
                        reasoning_trace=reasoning_summary,
                        finish_reason=finish_reason,
                    )
                )
                reasoning_summary = None

        return response_messages, {}

    # -- Inference -----------------------------------------------------------

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(10),
    )
    def model_inference(
        self,
        input_items: list[Any],
        tools: list[dict[str, Any]] | NotGiven,
    ) -> Any:
        with all_logging_disabled(highest_level=logging.WARNING):
            return self.openai_client.responses.create(  # type: ignore[call-overload]
                model=self.model_name,
                input=input_items,
                tools=tools,
                reasoning={"effort": self.reasoning_effort, "summary": "auto"},
                include=["reasoning.encrypted_content"],
            )


@register_role_class
class GPT_5_4_2026_03_05_Reasoning_Agent(OpenAIResponsesAPIAgent):
    model_name = "gpt-5.4-2026-03-05"
    reasoning_effort = "medium"


# ---------------------------------------------------------------------------
# Responses API code-execution agents
# ---------------------------------------------------------------------------


class OpenAIResponsesCodeExecAgent(OpenAIResponsesAPIAgent):
    """Code-exec agent using the Responses API with native reasoning.

    Combines ``OpenAIResponsesAPIAgent`` (reasoning effort, summaries,
    encrypted reasoning passback) with ``CodeExecutionAgent``-style
    respond logic (extract code blocks, send to ExecutionEnvironment).

    Registered as a ``CodeExecutionAgent`` subclass for CLI validation.
    """

    def __init__(self) -> None:
        OpenAIResponsesAPIAgent.__init__(self)
        self._chain_exec_ids: set[str] = set()

    def respond(
        self,
        messages: list[Message],
        available_tools: dict[str, Callable[..., Any]],
        external_tool_schemas: list[dict[str, Any]] | None = None,
    ) -> tuple[list[Message], dict[IntrospectionDatabaseNamespace, Any]]:
        from mmtoolsandbox.common.tool_conversion import generate_tool_call_id
        from mmtoolsandbox.roles.code_execution_agent import (
            CODE_BLOCK_FUNCTION_NAME,
            extract_code_blocks,
        )

        self.messages_validation(messages=messages)
        messages = self.filter_messages(messages=messages)
        if messages[-1].sender == RoleType.SYSTEM:
            return [], {}

        input_items = self._build_code_exec_responses_input(messages)

        response = self.model_inference(input_items, NOT_GIVEN)

        if response.usage is not None:
            self._record_token_usage(
                response.usage.input_tokens, response.usage.output_tokens
            )

        # Extract text and reasoning from response output.
        response_content = ""
        reasoning_summary: str | None = None
        pending_reasoning_items: list[dict[str, Any]] = []

        for item in response.output:
            item_type = getattr(item, "type", None)
            if item_type == "reasoning":
                summary = getattr(item, "summary", None)
                if summary:
                    texts = [s.text for s in summary if hasattr(s, "text")]
                    if texts:
                        reasoning_summary = "\n".join(texts)
                if hasattr(item, "model_dump"):
                    d = item.model_dump(exclude_none=True)
                    for key in self._OUTPUT_ONLY_FIELDS:
                        d.pop(key, None)
                    pending_reasoning_items.append(d)
            elif item_type == "message":
                for block in getattr(item, "content", []):
                    if hasattr(block, "text"):
                        response_content += block.text

        finish_reason = getattr(response, "status", "completed")

        reasoning_trace, cleaned_content = extract_reasoning(response_content)
        code_blocks = extract_code_blocks(cleaned_content)

        effective_reasoning = reasoning_summary or reasoning_trace
        serialized_reasoning = (
            json.dumps(pending_reasoning_items) if pending_reasoning_items else None
        )

        response_messages: list[Message] = []
        if not code_blocks:
            response_messages = [
                Message(
                    sender=self.role_type,
                    recipient=RoleType.USER,
                    content=cleaned_content,
                    reasoning_trace=effective_reasoning,
                    openai_reasoning_items=serialized_reasoning,
                    finish_reason=finish_reason,
                )
            ]
            self._chain_output_items = []
            self._chain_call_ids = set()
            self._chain_exec_ids = set()
        else:
            tool_call_id = generate_tool_call_id()

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
                    content=truncated_content,
                    openai_tool_call_id=tool_call_id,
                    openai_function_name=CODE_BLOCK_FUNCTION_NAME,
                    reasoning_trace=effective_reasoning,
                    openai_reasoning_items=serialized_reasoning,
                    finish_reason=finish_reason,
                )
            )
            self._chain_output_items.extend(
                self._sanitize_output_items(response.output)
            )
            self._chain_exec_ids.add(tool_call_id)

        return response_messages, {}

    # -- Input construction (code-exec) --------------------------------------

    def _build_code_exec_responses_input(self, messages: list[Message]) -> list[Any]:
        """Build Responses API input for code-exec mode."""
        execution_context = get_current_context()

        if self._chain_output_items and self._chain_exec_ids:
            has_chain_results = any(
                msg.sender == RoleType.EXECUTION_ENVIRONMENT
                and msg.recipient == RoleType.AGENT
                and msg.openai_tool_call_id in self._chain_exec_ids
                for msg in messages
            )
            if has_chain_results:
                return self._build_code_exec_with_chain(messages, execution_context)

        return self._build_code_exec_full_history(messages, execution_context)

    def _build_code_exec_full_history(
        self,
        messages: list[Message],
        execution_context: Any,
    ) -> list[Any]:
        """Convert full code-exec history, injecting persisted reasoning."""
        openai_msgs, _, _ = to_openai_messages(
            messages,
            execution_context,
            tool_images_as_user_message=self.tool_images_as_user_message,
            mode=ConversionMode.CODE_EXEC,
        )

        # Build content → reasoning items map from persisted Messages.
        content_reasoning: dict[str, list[dict[str, Any]]] = {}
        for msg in messages:
            if msg.sender == RoleType.AGENT and msg.openai_reasoning_items:
                content_reasoning[msg.content] = json.loads(msg.openai_reasoning_items)

        items: list[Any] = []
        for oai_msg in openai_msgs:
            role = oai_msg.get("role")
            content = oai_msg.get("content", "")

            if role in ("system", "user"):
                if isinstance(content, list):
                    converted = []
                    for part in content:
                        if part.get("type") == "text":
                            converted.append(
                                {"type": "input_text", "text": part["text"]}
                            )
                        elif part.get("type") == "image_url":
                            converted.append(
                                {
                                    "type": "input_image",
                                    "image_url": part["image_url"]["url"],
                                }
                            )
                    items.append({"role": role, "content": converted})
                else:
                    items.append({"role": role, "content": content})

            elif role == "assistant":
                # Inject persisted reasoning items before the assistant
                # message.  Match by stripping <think> tags (added by
                # to_openai_messages) and comparing to the stored content.
                if isinstance(content, str) and content_reasoning:
                    _, bare = extract_reasoning(content)
                    if bare in content_reasoning:
                        items.extend(content_reasoning.pop(bare))
                items.append({"role": "assistant", "content": content})

        return items

    def _build_code_exec_with_chain(
        self,
        messages: list[Message],
        execution_context: Any,
    ) -> list[Any]:
        """Build input preserving the current chain's reasoning."""
        from mmtoolsandbox.common.message_conversion import (
            EXECUTION_RESULTS_CLOSE_TAG,
            EXECUTION_RESULTS_OPEN_TAG,
        )

        # Find where the chain starts.
        split_idx = len(messages)
        for i, msg in enumerate(messages):
            if (
                msg.sender == RoleType.AGENT
                and msg.recipient == RoleType.EXECUTION_ENVIRONMENT
                and msg.openai_tool_call_id in self._chain_exec_ids
            ):
                split_idx = i
                break

        # Convert historical part.
        historical = messages[:split_idx]
        items: list[Any] = []
        if historical:
            items = self._build_code_exec_full_history(historical, execution_context)

        # Inject accumulated output items (reasoning + messages from all
        # rounds in the chain).
        items.extend(self._chain_output_items)

        # Append execution results for chain messages as user messages.
        for msg in messages[split_idx:]:
            if (
                msg.sender == RoleType.EXECUTION_ENVIRONMENT
                and msg.recipient == RoleType.AGENT
                and msg.openai_tool_call_id in self._chain_exec_ids
            ):
                items.append(
                    {
                        "role": "user",
                        "content": (
                            f"{EXECUTION_RESULTS_OPEN_TAG}\n"
                            f"{msg.content}\n"
                            f"{EXECUTION_RESULTS_CLOSE_TAG}"
                        ),
                    }
                )

        return items
