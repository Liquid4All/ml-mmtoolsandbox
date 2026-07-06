# Copyright © 2026 Apple Inc.

"""Interactive CLI user/agent role for human interfacing."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from logging import getLogger
from textwrap import indent
from typing import Any, Callable

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.introspection_databases import IntrospectionDatabaseNamespace
from mmtoolsandbox.common.message_conversion import Message
from mmtoolsandbox.roles.base_role import BaseRole
from mmtoolsandbox.roles.registry import register_role_class

LOGGER = getLogger(__name__)


@dataclass
class InteractiveMessage:
    content: str | None = None
    tool_call: str | None = None


class CONSOLE_COLORS:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


ROLE_TO_COLOR = {
    RoleType.USER: CONSOLE_COLORS.HEADER,
    RoleType.AGENT: CONSOLE_COLORS.OKBLUE,
    RoleType.EXECUTION_ENVIRONMENT: CONSOLE_COLORS.OKGREEN,
}


class CliRole(BaseRole):
    """Interactive CLI user/agent role for a real human."""

    role_type: RoleType
    model_name: str
    print_index: int = 0

    def __init__(self) -> None:
        super().__init__()
        print(f"Interactive role '{self.role_type}'.")

    def respond(
        self,
        messages: list[Message],
        available_tools: dict[str, Callable[..., Any]],
        external_tool_schemas: list[dict[str, Any]] | None = None,
    ) -> tuple[list[Message], dict[IntrospectionDatabaseNamespace, Any]]:
        """Reads a List of messages and attempt to respond with list of Messages.

        Args:
            messages:               List of Message objects containing full dialog history until this point
            available_tools:        Dictionary of role facing tool names to tool callable object,
                                    which also contains the execution facing name in tool.__name__

        Returns:
            A list of responses messages, and a dictionary of entries into introspection database

        Raises:
            KeyError:   When the last message is not directed to this role
        """
        # Catch start of new scenarios ....
        if len(messages) < self.print_index:
            self.print_index = 0

        self.messages_validation(messages=messages)

        # Echo latest messages
        for msg in messages[self.print_index :]:
            sender_color = ROLE_TO_COLOR.get(msg.sender, CONSOLE_COLORS.ENDC)
            recipient_color = ROLE_TO_COLOR.get(msg.recipient, CONSOLE_COLORS.ENDC)
            print(
                f" [{sender_color}{msg.sender}{CONSOLE_COLORS.ENDC}] "
                f"-> [{recipient_color}{msg.recipient}{CONSOLE_COLORS.ENDC}]:"
            )

            # Normalize the message contents.
            try:
                content = json.loads(msg.content)
                if isinstance(content, list):
                    content_lines = content  # Fallback for lists
                else:
                    content_lines = [content]  # Fallback for single objects
            except json.JSONDecodeError:
                content_lines = [msg.content]  # Fallback for non-json, eg response

            for line in content_lines:
                print(indent(f"{sender_color}{line}{CONSOLE_COLORS.ENDC}", "    "))
            print()

        # Don't keep printing from the beginning
        self.print_index = len(messages)

        # Keeps only relevant messages
        messages = self.filter_messages(messages=messages)
        # Agent does not respond to System
        if self.role_type == RoleType.AGENT and messages[-1].sender == RoleType.SYSTEM:
            return [], {}
        available_tool_names = set(available_tools.keys())

        # Ask user for input.
        response = self.user_input(available_tools)

        # Parse response
        if response.tool_call is None:
            # Message contains no tool call, aka addressed to agent
            assert response.content is not None
            recipient_role = (
                RoleType.AGENT if self.role_type == RoleType.USER else RoleType.USER
            )
            response_messages: list[Message] = [
                Message(
                    sender=self.role_type,
                    recipient=recipient_role,
                    content=response.content,
                )
            ]
        else:
            assert response.tool_call is not None
            response_messages = [
                Message(
                    sender=self.role_type,
                    recipient=RoleType.EXECUTION_ENVIRONMENT,
                    content=self.user_tool_call_to_python_code(
                        response.tool_call, available_tool_names
                    ),
                )
            ]
        return response_messages, {}

    def user_input(
        self,
        available_tools: dict[str, Callable[..., Any]],
    ) -> InteractiveMessage:
        """Get the user input from the CLI and return a message.

        Args:
            available_tools:        Dictionary of role facing tool names to tool callable object,
                                    which also contains the execution facing name in tool.__name__

        Returns:
          A message containing the plain text or tool call information.
        """
        available_tool_names = available_tools.keys()
        text = input(f" [{self.role_type}] > ")
        if text == "end" and "end_conversation" in available_tool_names:
            return InteractiveMessage(tool_call="end_conversation()")
        if text != "tool":
            return InteractiveMessage(content=text)
        # Tool call, show available options.
        tool_names_sigs = [
            (f.__name__, inspect.signature(f)) for f in available_tools.values()
        ]
        tools_fmtd = "\n".join([f" - {name}{sig}" for name, sig in tool_names_sigs])
        print(f"Tool options: \n {tools_fmtd}.")
        tool_name = input(f" [{self.role_type}] Tool function call > ")
        return InteractiveMessage(tool_call=tool_name)

    def user_tool_call_to_python_code(
        self, tool_call: str, available_tool_names: set[str]
    ) -> str:
        """Convert a tool call into an execution environment command."""
        return f"print(json.dumps({tool_call}))"


@register_role_class
class CliUser(CliRole):
    role_type: RoleType = RoleType.USER
    model_name = "cli_user"


@register_role_class
class CliAgent(CliRole):
    role_type: RoleType = RoleType.AGENT
    model_name = "cli_agent"
