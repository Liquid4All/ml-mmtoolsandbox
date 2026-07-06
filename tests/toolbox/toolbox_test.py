# Copyright © 2026 Apple Inc.

"""Tests for mmtoolsandbox.toolbox.toolbox"""

from typing import Any

import pytest

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.tool_registry import deregister_tool
from mmtoolsandbox.common.utils import register_as_tool
from mmtoolsandbox.toolbox.loading import load_toolbox
from mmtoolsandbox.toolbox.names import ToolboxName
from mmtoolsandbox.toolbox.toolbox import Toolbox, ToolboxConfig, compute_tool_checksum
from mmtoolsandbox.tools.tool_sandbox.user_tools import end_conversation


@pytest.fixture
def deregister_do_nothing_tool_after_test_execution() -> Any:
    yield
    deregister_tool("do_nothing")


@pytest.mark.usefixtures("deregister_do_nothing_tool_after_test_execution")
def test_compute_tools_checksum() -> None:
    @register_as_tool(toolboxes={ToolboxName.FULL})
    def do_nothing() -> None:
        """This is my docstring."""
        print("Do nothing.")

    checksum = compute_tool_checksum([do_nothing])

    # The checksum should be the same when the tool is the same.
    assert checksum == compute_tool_checksum([do_nothing])

    # The remainder re-defines the `do_nothing` function with changed contents. Note
    # we need to re-define the function since we want it to have the exact same name.

    # Change the docstring and test that the checksum has changed.
    deregister_tool("do_nothing")

    @register_as_tool(  # type: ignore[no-redef]
        toolboxes={ToolboxName.FULL}
    )
    def do_nothing() -> None:
        """This is a different docstring."""
        print("Do nothing.")

    assert checksum != compute_tool_checksum([do_nothing])

    # Change the docstring and test that the checksum has changed.
    deregister_tool("do_nothing")

    @register_as_tool(  # type: ignore[no-redef]
        toolboxes={ToolboxName.FULL}
    )
    def do_nothing() -> None:
        """This is my docstring."""
        print("Now we say something else.")

    assert checksum != compute_tool_checksum([do_nothing])

    # The `register_as_tool` decorator is considered part of the function code and thus
    # changes to it also affect the checksum. Change the `toolboxes` field and ensure
    # that the checksum is different.
    deregister_tool("do_nothing")

    @register_as_tool(  # type: ignore[no-redef]
        toolboxes={ToolboxName.FULL}
    )
    def do_nothing() -> None:
        """This is my docstring."""
        print("Do nothing.")

    assert checksum != compute_tool_checksum([do_nothing])

    # The checksum is different if the number of tools has changed even if we just have
    # the same tool multiple times.
    assert checksum != compute_tool_checksum([do_nothing, do_nothing])


@pytest.mark.parametrize("toolbox_name", ToolboxName)
def test_all_toolboxes_contain_end_conversation_tool(toolbox_name: ToolboxName) -> None:
    config = ToolboxConfig()
    toolbox = load_toolbox(toolbox_name, config=config.model_dump())
    assert end_conversation in toolbox.tools


@pytest.mark.usefixtures("deregister_do_nothing_tool_after_test_execution")
def test_check_that_toolbox_must_contain_end_conversation_tool() -> None:
    @register_as_tool(toolboxes={ToolboxName.FULL})
    def do_nothing() -> None:
        """This is my docstring."""
        print("Do nothing.")

    with pytest.raises(
        ValueError, match=r".*must contain the `end_conversation` tool.*"
    ):
        Toolbox(
            name="my_toolbox",
            config=ToolboxConfig(),
            tools=[do_nothing],
        )

    # Creating a custom toolbox should be supported if it is empty or if we manually
    # specify the `end_conversation` tool.
    Toolbox(
        name="my_toolbox",
        config=ToolboxConfig(),
        tools=[],
    )
    Toolbox(
        name="my_toolbox",
        config=ToolboxConfig(),
        tools=[end_conversation],
    )


@pytest.fixture
def deregister_no_databases_needed_tool_after_test_execution() -> Any:
    yield
    deregister_tool("no_databases_needed")


@pytest.mark.usefixtures("deregister_no_databases_needed_tool_after_test_execution")
def test_extract_database_namespaces_sandbox_db_only() -> None:
    @register_as_tool(
        toolboxes={ToolboxName.FULL},
        database_namespaces=None,
    )
    def no_databases_needed() -> None: ...

    toolbox = Toolbox(
        name="mock",
        config=ToolboxConfig(),
        tools=[no_databases_needed, end_conversation],
    )
    # While `no_databases_needed` does not need access to any databases the
    # `end_conversation` tool accesses the sandbox message database.
    assert {DatabaseNamespace.SANDBOX} == toolbox.extract_database_namespaces()


@pytest.fixture
def deregister_multiple_databases_needed_tool_after_test_execution() -> Any:
    yield
    deregister_tool("multiple_databases_needed")


@pytest.mark.usefixtures(
    "deregister_multiple_databases_needed_tool_after_test_execution"
)
def test_extract_database_namespaces_tool_with_multiple_dbs() -> None:
    DATABASES = {DatabaseNamespace.CONTACT, DatabaseNamespace.SETTING}

    @register_as_tool(
        toolboxes={ToolboxName.FULL},
        database_namespaces=DATABASES,
    )
    def multiple_databases_needed() -> None: ...

    toolbox = Toolbox(
        name="mock",
        config=ToolboxConfig(),
        tools=[multiple_databases_needed, end_conversation],
    )
    # `multiple_databases_needed` accesses multiple databases and the `end_conversation`
    # tool accesses the sandbox message database.
    expected_dbs = {DatabaseNamespace.SANDBOX}.union(DATABASES)
    assert expected_dbs == toolbox.extract_database_namespaces()
