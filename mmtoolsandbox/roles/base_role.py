# Copyright © 2026 Apple Inc.

"""Base class for all roles in the MMToolSandbox simulation."""

from __future__ import annotations

import dataclasses
from logging import getLogger
from typing import (
    Any,
    Callable,
    cast,
)

from mmtoolsandbox.common.execution_context import (
    RoleType,
)
from mmtoolsandbox.common.introspection_databases import IntrospectionDatabaseNamespace
from mmtoolsandbox.common.message_conversion import Message

LOGGER = getLogger(__name__)


@dataclasses.dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0


class BaseRole:
    """Base class for all roles in the simulation.

    A role is a participant that reads and writes messages via the
    execution context. Concrete roles include dialog agents, user
    simulators, and the code execution environment.

    Roles are designed to be stateless; persistent state is stored in
    the ``ExecutionContext`` databases.

    Attributes:
        role_type: The ``RoleType`` this role represents, or ``None``
            if not yet assigned.
    """

    role_type: RoleType | None = None

    def __init__(self) -> None:
        self._token_usage = TokenUsage()

    def _record_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self._token_usage.prompt_tokens += prompt_tokens
        self._token_usage.completion_tokens += completion_tokens
        self._token_usage.total_tokens += prompt_tokens + completion_tokens
        self._token_usage.api_calls += 1

    @property
    def token_usage(self) -> TokenUsage:
        return self._token_usage

    @classmethod
    def messages_validation(cls, messages: list[Message]) -> None:
        """Validate that the last message is addressed to this role.

        Args:
            messages: Full conversation history. The last element must
                have ``recipient == cls.role_type``.

        Raises:
            KeyError: If the last message is not directed to this role.
        """
        if messages[-1].recipient != cls.role_type:
            raise KeyError(
                f"The last message should be addressed to {cls.role_type}, found {messages[-1].recipient}"
            )

    @classmethod
    def filter_messages(cls, messages: list[Message]) -> list[Message]:
        """Filter messages to only those visible to this role.

        Args:
            messages: Full conversation history to filter.

        Returns:
            Messages where this role is in the ``visible_to`` list.
        """
        return [
            message
            for message in messages
            if cls.role_type in cast(list[RoleType], message.visible_to)
        ]

    def teardown(self) -> None:
        """Clean up the role and free any held resources."""
        pass

    def respond(
        self,
        messages: list[Message],
        available_tools: dict[str, Callable[..., Any]],
        external_tool_schemas: list[dict[str, Any]] | None = None,
    ) -> tuple[list[Message], dict[IntrospectionDatabaseNamespace, Any]]:
        """Generate response messages for the current conversation turn.

        Subclasses must override this method. An empty response list means
        the role acknowledges the message but does not reply (e.g. system
        setup messages processed silently).

        Args:
            messages: Full dialog history up to this point. The last *k*
                messages are directed to this role.
            available_tools: Mapping of tool display names to callables.
                Each callable also carries the execution-facing name in
                ``tool.__name__``.
            external_tool_schemas: Optional tool schemas for tools not
                registered in the repo (triggers conversation end if
                the agent calls one).

        Returns:
            A tuple of (response_messages, introspection_entries).

        Raises:
            KeyError: If the last message is not directed to this role.
        """
        raise NotImplementedError
