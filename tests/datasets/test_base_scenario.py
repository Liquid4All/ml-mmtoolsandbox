# Copyright © 2026 Apple Inc.

"""Tests for base scenario creation after few-shot example removal."""

import inspect

import polars as pl
import pytest

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.datasets.base_scenario import create_base_scenario
from mmtoolsandbox.toolbox.loading import load_toolbox
from mmtoolsandbox.toolbox.names import ToolboxName
from mmtoolsandbox.toolbox.toolbox import Toolbox


@pytest.fixture(scope="module")
def toolbox() -> Toolbox:
    return load_toolbox(ToolboxName.FULL, {})


class TestBaseScenarioCreation:
    """Verify base scenario contains no few-shot examples."""

    def test_sandbox_has_only_system_messages(self, toolbox: Toolbox) -> None:
        """SANDBOX should contain only SYSTEM→EXEC_ENV and SYSTEM→AGENT messages."""
        scenario = create_base_scenario(toolbox)
        sandbox = scenario.starting_context.get_database(
            DatabaseNamespace.SANDBOX, get_all_history_snapshots=True
        )
        sandbox = scenario.starting_context.drop_headguard(sandbox)

        for row in sandbox.to_dicts():
            assert row["sender"] == RoleType.SYSTEM, (
                f"Expected only SYSTEM sender, got {row['sender']}"
            )

    def test_sandbox_message_count(self, toolbox: Toolbox) -> None:
        """SANDBOX should have exactly 2 messages: import + agent prompt."""
        scenario = create_base_scenario(toolbox)
        sandbox = scenario.starting_context.get_database(
            DatabaseNamespace.SANDBOX, get_all_history_snapshots=True
        )
        sandbox = scenario.starting_context.drop_headguard(sandbox)
        assert sandbox.height == 2

    def test_no_visible_to_user_only_messages(self, toolbox: Toolbox) -> None:
        """No messages should have visible_to=[USER] (the old few-shot marker)."""
        scenario = create_base_scenario(toolbox)
        sandbox = scenario.starting_context.get_database(
            DatabaseNamespace.SANDBOX, get_all_history_snapshots=True
        )
        sandbox = scenario.starting_context.drop_headguard(sandbox)

        user_only = sandbox.filter(pl.col("visible_to") == [RoleType.USER])
        assert user_only.is_empty(), (
            f"Found {user_only.height} messages with visible_to=[USER]"
        )

    def test_no_few_shot_parameter(self) -> None:
        """create_base_scenario should not accept add_user_simulator_few_shot_examples."""
        sig = inspect.signature(create_base_scenario)
        assert "add_user_simulator_few_shot_examples" not in sig.parameters

    def test_no_bart_content_in_sandbox(self, toolbox: Toolbox) -> None:
        """No 'Bart' references should appear in SANDBOX messages."""
        scenario = create_base_scenario(toolbox)
        sandbox = scenario.starting_context.get_database(
            DatabaseNamespace.SANDBOX, get_all_history_snapshots=True
        )
        sandbox = scenario.starting_context.drop_headguard(sandbox)

        for row in sandbox.to_dicts():
            content = row.get("content") or ""
            assert "Bart" not in content, (
                f"Found 'Bart' in SANDBOX message: {content[:80]}"
            )

    def test_end_conversation_in_allow_list(self, toolbox: Toolbox) -> None:
        """end_conversation should always be in the tool allow list."""
        scenario = create_base_scenario(toolbox)
        assert scenario.starting_context.tool_allow_list is not None
        assert "end_conversation" in scenario.starting_context.tool_allow_list

    def test_without_agent_system_prompt(self, toolbox: Toolbox) -> None:
        """With add_agent_system_prompt=False, only the import message remains."""
        scenario = create_base_scenario(toolbox, add_agent_system_prompt=False)
        sandbox = scenario.starting_context.get_database(
            DatabaseNamespace.SANDBOX, get_all_history_snapshots=True
        )
        sandbox = scenario.starting_context.drop_headguard(sandbox)
        assert sandbox.height == 1
        assert sandbox["recipient"][0] == RoleType.EXECUTION_ENVIRONMENT
