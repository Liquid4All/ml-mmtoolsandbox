# Copyright © 2026 Apple Inc.

from __future__ import annotations

from enum import auto
from typing import Any, Protocol

from strenum import StrEnum


class ToolStateType(StrEnum):
    """This enum defines the types of tools for which state can be registered."""

    ENTITY_DATABASE = auto()
    TOOL_CONFIGS = auto()
    TOOL_RETRIEVAL = auto()
    UTILITIES = auto()


class ToolState(Protocol):
    """Protocol defining the `ToolState` interface.

    The `ToolState` is managed by the `ToolStateRegistry`, which is part of the
    `ExecutionContext`. The final tool states are deserialized as part of the
    `ExecutionContext`. Any intermediate changes to a tool state are currently not
    recoverable from a serialized `ExecutionContext`. Additionally, deserialization of
    tool states is currently not implemented.
    """

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError()

    @classmethod
    def from_dict(cls, serialized_dict: dict[str, Any]) -> ToolState:
        raise NotImplementedError()
