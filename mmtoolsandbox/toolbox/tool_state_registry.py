# Copyright © 2026 Apple Inc.

from __future__ import annotations

from logging import getLogger
from typing import Any

from mmtoolsandbox.toolbox.state import ToolState, ToolStateType

LOGGER = getLogger(__name__)


class ToolStateRegistry:
    def __init__(self) -> None:
        self._registry: dict[ToolStateType, ToolState] = {}

    def check_consistency(self, serialized_tool_state_registry: dict[str, Any]) -> None:
        """Check if this registry is consistent with the given serialized version."""
        current_types = set(state_type.value for state_type in self._registry)
        serialized_types = set(extract_tool_state_types(serialized_tool_state_registry))
        if current_types != serialized_types:
            different_types = current_types.symmetric_difference(serialized_types)
            LOGGER.warning(
                "The deserialized and re-loaded tool state registry types are "
                f"different. This means that the toolbox has changed and is "
                "potentially incompatible with a serialized `ExecutionContext`.\n"
                f"Different types:    {sorted(different_types)}\n"
                f"Re-loaded types:    {sorted(current_types)}\n"
                f"Deserialized types: {sorted(serialized_types)}\n"
            )

    def add(self, tool_state_type: ToolStateType, state: ToolState) -> None:
        existing_state = self._registry.get(tool_state_type)
        if existing_state is not None:
            raise ValueError(
                f"Tool state of type '{tool_state_type}' has already been registered. "
                f"Existing value:\n{existing_state}\nDesired value:\n{state}"
            )

        self._registry[tool_state_type] = state

    def get(self, tool_state_type: ToolStateType) -> ToolState:
        """Get the state for the given tool.

        Raises:
            KeyError if no state exists for the given type.
        """
        return self._registry[tool_state_type]

    def get_tool_state_types(self) -> list[ToolStateType]:
        return list(self._registry.keys())

    def to_dict(self) -> dict[str, Any]:
        # The enum objects are not serializable so we only store the enum string
        # value.
        registry_dict = {
            state_type.value: state.to_dict()
            for state_type, state in self._registry.items()
        }
        return {"_registry": registry_dict}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolStateRegistry):
            return False

        return self._registry == other._registry


def extract_tool_state_types(serialized_dict: dict[str, Any]) -> list[ToolStateType]:
    return [ToolStateType(value) for value in serialized_dict["_registry"].keys()]
