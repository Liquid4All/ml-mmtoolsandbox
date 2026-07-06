# Copyright © 2026 Apple Inc.

"""Simulated user role for any model that conforms to OpenAI tool use API"""

from __future__ import annotations

import json
import re
from logging import getLogger
from typing import (
    Any,
    Callable,
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
from tenacity import retry, stop_after_attempt, wait_random_exponential

from mmtoolsandbox.common.execution_context import (
    RoleType,
    get_current_context,
)
from mmtoolsandbox.common.image_id import ImageId
from mmtoolsandbox.common.introspection_databases import IntrospectionDatabaseNamespace
from mmtoolsandbox.common.message_conversion import (
    Message,
    get_image_content,
    openai_tool_call_to_python_code,
    python_code_to_openai_tool_call,
)
from mmtoolsandbox.common.tool_conversion import convert_to_openai_tool
from mmtoolsandbox.common.utils import all_logging_disabled
from mmtoolsandbox.roles.base_role import BaseRole
from mmtoolsandbox.roles.registry import register_role_class
from mmtoolsandbox.tools.tool_sandbox.user_tools import (
    SEND_MESSAGE_WITH_IMAGE_TOOL_NAME,
)

LOGGER = getLogger(__name__)

_SENT_IMAGE_IDS_PATTERN = re.compile(r"\n*\[Sent image IDs: [\d, ]+\]")


class OpenAIAPIUser(BaseRole):
    """Simulated user role for any model that conforms to OpenAI tool use API"""

    role_type: RoleType = RoleType.USER
    model_name: str

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

        Specifically, interprets system & agent messages, sends out valid followup responses back to agent

        Comparing to agents and execution environments. Users and user simulators have a unique challenge. Agents and
        execution environments passively accept messages from other roles, execute them and respond. However, a user
        has the autonomy to decide when to stop the conversation. It must be able to, otherwise the conversation is
        never going to stop.

        The current idea is to instruct the user simulator to issue structured responses indicating end of conversation,
        1 such approach could be, we offer a tool to user simulator. The simulator could issue
        tool call to in order to terminate the conversation. This will be interpreted, and sent to execution env to
        execute.

        Args:
            messages:               List of Message objects containing full dialog history until this point
            available_tools:        Dictionary of role facing tool names to tool callable object,
                                    which also contains the execution facing name in tool.__name__

        Returns:
            A list of responses messages, and a dictionary of entries into introspection database

        Raises:
            KeyError:   When the last message is not directed to this role
        """
        response_messages: list[Message] = []
        self.messages_validation(messages=messages)
        # Keeps only relevant messages
        messages = self.filter_messages(messages=messages)
        # The user simulator is also allowed to respond to messages from `SYSTEM` as
        # they contain the instructions/task for the scenario.

        # Get OpenAI tools if most recent turn is from Agent (again, to terminate the conversation if needed)
        available_tool_names = set(available_tools.keys())
        openai_tools = (
            [convert_to_openai_tool(tool=tool) for tool in available_tools.values()]
            if messages[-1].sender == RoleType.AGENT
            else NOT_GIVEN
        )

        # Convert to OpenAI messages
        openai_messages = self.to_openai_messages(messages=messages)
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

        # Message contains no tool call, aka addressed to agent
        if openai_response_message.tool_calls is None:
            # Not sure why the content field `ChatCompletionMessage` has a type of
            # `str | None`.
            assert openai_response_message.content is not None
            response_messages = [
                Message(
                    sender=self.role_type,
                    recipient=RoleType.AGENT,
                    content=_SENT_IMAGE_IDS_PATTERN.sub(
                        "", openai_response_message.content
                    ),
                )
            ]
        else:
            assert openai_tools is not NOT_GIVEN
            for tool_call in openai_response_message.tool_calls:
                if not isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
                    continue
                # Intercept send_message_with_image: convert it into a
                # USER -> AGENT message with image_ids instead of routing
                # to the execution environment.
                if tool_call.function.name == SEND_MESSAGE_WITH_IMAGE_TOOL_NAME:
                    args = json.loads(tool_call.function.arguments)
                    raw_ids = str(args.get("image_ids", "")).strip().strip("[]")
                    parsed_image_ids = [
                        ImageId(int(id_str.strip()))
                        for id_str in raw_ids.split(",")
                        if id_str.strip()
                    ]
                    response_messages.append(
                        Message(
                            sender=self.role_type,
                            recipient=RoleType.AGENT,
                            content=_SENT_IMAGE_IDS_PATTERN.sub(
                                "", args.get("message", "")
                            ),
                            image_ids=parsed_image_ids,
                            openai_tool_call_id=tool_call.id,
                            openai_function_name=tool_call.function.name,
                        )
                    )
                else:
                    # User tool calls are private to the user and execution
                    # environment. Tools that need to notify the agent (e.g.,
                    # ui_user_interact) inject an environment-mediated
                    # EXEC_ENV→AGENT message instead of leaking the raw call.
                    response_messages.append(
                        Message(
                            sender=self.role_type,
                            recipient=RoleType.EXECUTION_ENVIRONMENT,
                            content=openai_tool_call_to_python_code(
                                tool_call,
                                available_tool_names,
                                execution_facing_tool_name=None,
                            ),
                            openai_tool_call_id=tool_call.id,
                            openai_function_name=tool_call.function.name,
                            visible_to=[
                                RoleType.USER,
                                RoleType.EXECUTION_ENVIRONMENT,
                            ],
                        )
                    )
        return response_messages, {}

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(10),
        # Do not set retry explicitly. Retry on all exceptions.
    )
    def model_inference(
        self,
        openai_messages: list[dict[str, Any]],
        openai_tools: list[ChatCompletionToolParam] | NotGiven,
    ) -> ChatCompletion:
        """Run OpenAI model inference

        Args:
            openai_messages:    List of OpenAI API format messages
            openai_tools:       List of OpenAI API format tools definition

        Returns:
            OpenAI API chat completion object
        """
        with all_logging_disabled():
            return self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=cast(list[ChatCompletionMessageParam], openai_messages),
                tools=openai_tools,  # type: ignore[arg-type]  # SDK expects NotGiven→Omit
            )

    @staticmethod
    def to_openai_messages(
        messages: list[Message],
    ) -> list[dict[str, Any]]:
        """Converts a list of MMToolSandbox messages to OpenAI API messages, from the perspective of a simulated user

        Args:
            messages:   A list of MMToolSandbox messages

        Returns:
            A list of OpenAI API messages
        """
        # Find the index of the LAST AGENT→USER message with images.
        # Only that message gets full image data; older ones get text-only
        # to avoid blowing up the context window with accumulated base64.
        # NOTE: only AGENT→USER images (UI renders) are evicted. USER→AGENT
        # images (user's own photos) are always preserved — they are task
        # context the user needs throughout the conversation.
        # TODO: Revisit this design for multi-image interaction scenarios.
        last_image_msg_idx = -1
        for idx, msg in enumerate(messages):
            if (
                msg.sender == RoleType.AGENT
                and msg.recipient == RoleType.USER
                and msg.image_ids
            ):
                last_image_msg_idx = idx

        openai_messages: list[dict[str, Any]] = []
        for idx, message in enumerate(messages):
            if message.sender == RoleType.SYSTEM and message.recipient == RoleType.USER:
                # Merge consecutive system messages into one to avoid OOD for models
                if openai_messages and openai_messages[-1]["role"] == "system":
                    openai_messages[-1]["content"] += "\n\n" + message.content
                else:
                    openai_messages.append(
                        {"role": "system", "content": message.content}
                    )
            elif (
                message.sender == RoleType.AGENT and message.recipient == RoleType.USER
            ):
                # The roles are in reverse
                # We are the user simulator, simulated response from OpenAI assistant role is the simulated user message
                # which means agent dialog is OpenAI user role
                if message.image_ids and idx == last_image_msg_idx:
                    # Only include full image data for the most recent image
                    content = get_image_content(
                        initial_content=message.content,
                        image_ids=message.image_ids,
                        execution_context=get_current_context(),
                    )
                    openai_messages.append({"role": "user", "content": content})
                elif message.image_ids:
                    # Older image messages: text only, no base64
                    openai_messages.append(
                        {
                            "role": "user",
                            "content": message.content
                            + "\n[Image omitted from context]",
                        }
                    )
                else:
                    openai_messages.append({"role": "user", "content": message.content})
            elif (
                message.sender == RoleType.USER and message.recipient == RoleType.AGENT
            ):
                text = message.content or ""
                if message.image_ids:
                    id_str = ", ".join(str(int(img_id)) for img_id in message.image_ids)
                    text += f"\n\n[Sent image IDs: {id_str}]"
                openai_messages.append({"role": "assistant", "content": text})
            elif (
                message.sender == RoleType.USER
                and message.recipient == RoleType.EXECUTION_ENVIRONMENT
            ):
                # User's own tool calls → "assistant" with tool_calls
                # (from the user model's perspective, its outputs are "assistant")
                tool_call = python_code_to_openai_tool_call(
                    message.content,
                    agent_facing_tool_name=message.openai_function_name,
                )
                tool_call_id = message.openai_tool_call_id or tool_call.id
                # Aggregate parallel tool calls into a single assistant message
                if not openai_messages or "tool_calls" not in openai_messages[-1]:
                    openai_messages.append(
                        {"role": "assistant", "content": "", "tool_calls": []}
                    )
                openai_messages[-1]["tool_calls"].append(
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                )
            elif (
                message.sender == RoleType.EXECUTION_ENVIRONMENT
                and message.recipient == RoleType.USER
            ):
                # Tool results for the user model → "tool" role
                openai_messages.append(
                    {
                        "tool_call_id": message.openai_tool_call_id,
                        "role": "tool",
                        "name": message.openai_function_name,
                        "content": message.content,
                    }
                )
            else:
                raise ValueError(
                    f"Unrecognized sender recipient pair {(message.sender, message.recipient)}"
                )
        return openai_messages


@register_role_class
class GPT_5_4_2026_03_05_User(OpenAIAPIUser):
    model_name = "gpt-5.4-2026-03-05"
