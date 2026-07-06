# Copyright © 2026 Apple Inc.

from __future__ import annotations

import hashlib
import inspect
from collections import defaultdict
from inspect import getmodule
from logging import getLogger
from typing import Any, Callable, Iterable, Protocol

import attrs
from pydantic import BaseModel

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.toolbox.tool_state_registry import ToolStateRegistry

Tool = Callable[..., Any]


class ToolboxConfig(BaseModel): ...


LOGGER = getLogger(__name__)


def organize_tools_by_module(tools: Iterable[Tool]) -> dict[str, list[Tool]]:
    module_name_to_tools: dict[str, list[Tool]] = defaultdict(list)
    for tool in tools:
        module = getmodule(tool)
        assert module is not None, tool
        module_name = module.__name__
        module_name_to_tools[module_name].append(tool)
    return module_name_to_tools


def create_import_statement(
    tools: Iterable[Tool],
    tool_allow_list: list[str] | None = None,
    visible_to_role: Any | None = None,
) -> str:
    # We import `RoleType` here to avoid a cyclic dependency.
    from mmtoolsandbox.common.execution_context import RoleType

    filtered_tools = []
    for tool in tools:
        if tool_allow_list is not None and tool.__name__ not in tool_allow_list:
            continue
        if visible_to_role is not None:
            visible_to = getattr(tool, "visible_to", (RoleType.AGENT,))
            if isinstance(visible_to_role, (list, tuple)):
                if not any(role in visible_to for role in visible_to_role):
                    continue
            elif visible_to_role not in visible_to:
                continue
        filtered_tools.append(tool)

    module_name_to_tools = organize_tools_by_module(filtered_tools)

    # Create import statement for JSON serialization of tool call results.
    import_statement = "import json\n"
    import_statement += (
        "from mmtoolsandbox.common.json_utils import MMToolSandboxJSONEncoder\n"
    )
    # Create import statements for tools.
    for module_name, tools in module_name_to_tools.items():
        tool_names = [tool.__name__ for tool in tools]
        import_statement += f"from {module_name} import {', '.join(tool_names)}\n"
    return import_statement


def compute_tool_checksum(tools: Iterable[Tool]) -> str:
    """Compute a checksum for the given tools.

    Note: When computing the checksum we hash the tool's source code including
    docstring and comments.

    Args:
        tools:  The tools for which to compute a checksum.

    Returns:
        The checksum as a hex string.
    """
    # We cannot just use Python's built-in `hash` function since it uses a random hash
    # seed set once at startup. This means that the hash of the same string is different
    # across different Python interpreter processes, see
    # https://stackoverflow.com/a/27522708 . However, we want to get the same checksum
    # across Python interpreter sessions since we use it to detect if a serialized
    # toolbox is compatible with the current code.
    m = hashlib.sha256()
    for tool in tools:
        tool_code_str = inspect.getsource(tool)
        m.update(tool_code_str.encode("utf-8"))
    return m.hexdigest()


class ToolStateRegistryFactory(Protocol):
    """Protocol for a factory constructing a `ToolStateRegistry`."""

    def __call__(
        self,
        toolbox: Toolbox,
        tool_allow_list: list[str] | None,
        tool_deny_list: list[str] | None,
    ) -> ToolStateRegistry:
        return ToolStateRegistry()


class EmptyToolStateRegistryFactory(ToolStateRegistryFactory):
    """Factory creating an empty `ToolStateRegistry`."""

    def __call__(
        self,
        toolbox: Toolbox,
        tool_allow_list: list[str] | None,
        tool_deny_list: list[str] | None,
    ) -> ToolStateRegistry:
        return ToolStateRegistry()


@attrs.define(frozen=True)
class Toolbox:
    name: str
    config: ToolboxConfig
    tools: list[Tool] = attrs.field()
    # The `ToolStateRegistry` is tied to the tools and thus toolbox. However, we do not
    # want to include the registry itself in the toolbox since its contents are
    # mutable and allowed to change throughout a scenario. The `Toolbox` on the other
    # hand is supposed to be static/immutable and the same across all scenarios.
    tool_state_registry_factory: ToolStateRegistryFactory = (
        EmptyToolStateRegistryFactory()
    )

    @tools.validator
    def check(self, attribute: attrs.Attribute[list[Tool]], tools: list[Tool]) -> None:
        # TODO:
        # Right now there is a cyclic dependency between the `ExecutionContext` and
        # `mmtoolsandbox.tools`. Note that `ExecutionContext` instantiates the `Toolbox`
        # upon construction

        #  from mmtoolsandbox.tools.user_tools import end_conversation
        #  if end_conversation not in tools:
        #    raise ValueError(...)

        # The `end_conversation` tool is deeply tied to the orchestration of the
        # MMToolSandbox and must thus always be available. Note that we allow creating a
        # `Toolbox` with no tools at all, see comment above the `create_empty_toolbox()`
        # function for more details.
        if len(tools) > 0 and not any(
            tool.__name__ == "end_conversation"
            and tool.__module__ == "mmtoolsandbox.tools.tool_sandbox.user_tools"
            for tool in tools
        ):
            raise ValueError(
                f"The toolbox must contain the `end_conversation` tool. Tools:\n{tools}"
            )

    def create_tool_state_registry(
        self, tool_allow_list: list[str] | None, tool_deny_list: list[str] | None
    ) -> ToolStateRegistry:
        return self.tool_state_registry_factory(
            self, tool_allow_list=tool_allow_list, tool_deny_list=tool_deny_list
        )

    def get_tool_names(self) -> set[str]:
        return {tool.__name__ for tool in self.tools}

    def create_import_statement(
        self,
        tool_allow_list: list[str] | None = None,
        visible_to_role: Any | None = None,
    ) -> str:
        return create_import_statement(
            self.tools,
            tool_allow_list=tool_allow_list,
            visible_to_role=visible_to_role,
        )

    def extract_database_namespaces(self) -> set[DatabaseNamespace]:
        """Extract which database namespaces the tools need access to."""
        namespaces: set[DatabaseNamespace] = set()
        for tool in self.tools:
            current_namespaces: set[DatabaseNamespace] = getattr(
                tool, "database_namespaces", set()
            )
            namespaces.update(current_namespaces)
        return namespaces

    def serialize_to_dict(self) -> dict[str, Any]:
        """Serialize the toolbox to a dictionary.

        Note that only the toolbox name and config are serialized. During
        deserialization we load the toolbox using its name and configuration. The
        motivation is that correctly serializing the tool implementations is hard and
        there is no immediate use case where this would be needed.
        """
        tools_checksum = compute_tool_checksum(self.tools)
        tool_state_registry_factory_name = type(
            self.tool_state_registry_factory
        ).__qualname__
        return {
            "name": self.name,
            "config": self.config.model_dump(),
            "tools_checksum": tools_checksum,
            "tool_state_registry_factory_name": tool_state_registry_factory_name,
        }

    @classmethod
    def deserialize_from_dict(cls, serialized_dict: dict[str, Any]) -> Toolbox:
        # We import `load_toolbox` here to avoid a cyclic dependency.
        from mmtoolsandbox.toolbox.loading import load_toolbox

        toolbox_name = serialized_dict["name"]
        # We use a default value for `config` for backwards compatibility.
        config = serialized_dict.get("config", {})
        toolbox = load_toolbox(toolbox_name, config=config)

        current_tools_checksum = compute_tool_checksum(toolbox.tools)
        serialized_tools_checksum = serialized_dict["tools_checksum"]
        if serialized_tools_checksum != current_tools_checksum:
            LOGGER.warning(
                "The tool checksum of the serialized toolbox does not match the "
                f"checksum of the re-loaded toolbox: {serialized_tools_checksum} != "
                f"{current_tools_checksum}. This means that the toolbox has changed "
                "and is potentially incompatible with a serialized `ExecutionContext`."
            )

        serialized_registry_factory_name = serialized_dict.get(
            "tool_state_registry_factory_name"
        )
        current_registry_factory_name = type(
            toolbox.tool_state_registry_factory
        ).__qualname__
        if (
            serialized_registry_factory_name is not None
            and current_registry_factory_name != serialized_registry_factory_name
        ):
            LOGGER.warning(
                "The tool state registry factory name of the serialized toolbox does "
                f"not match the re-loaded name: {serialized_registry_factory_name} != "
                f"{current_registry_factory_name}. This means that the factory has "
                "changed and is potentially incompatible with a serialized "
                "`ExecutionContext`."
            )

        return toolbox


def create_empty_toolbox() -> Toolbox:
    """Create an empty toolbox instance.

    This can be used as the default when creating an `ExecutionContext`. The main reason is
    that the code relies on being able to default-construct an `ExecutionContext` and only
    later provide information like the toolbox name."""
    return Toolbox(
        name="empty",
        config=ToolboxConfig(),
        tools=[],
    )
