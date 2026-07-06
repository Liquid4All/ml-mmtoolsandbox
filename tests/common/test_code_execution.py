# Copyright © 2026 Apple Inc.

"""Unit tests for code execution tool filtering and import generation."""

import unittest
from typing import Any, cast
from unittest.mock import MagicMock, patch

import polars as pl

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution import transform_scenario_capabilities
from mmtoolsandbox.common.execution_context import (
    ExecutionContext,
    RoleType,
    set_current_context,
)
from mmtoolsandbox.common.image_id import ImageResult
from mmtoolsandbox.common.message_conversion import (
    CODE_EXEC_ERROR_PREFIX,
    EXECUTION_RESULTS_CLOSE_TAG,
    EXECUTION_RESULTS_OPEN_TAG,
    Message,
    to_openai_messages_for_code_exec,
)
from mmtoolsandbox.common.scenario import Scenario
from mmtoolsandbox.common.utils import register_as_tool
from mmtoolsandbox.datasets.names import DatasetName
from mmtoolsandbox.roles.code_execution_agent import (
    extract_code_blocks,
)
from mmtoolsandbox.roles.execution_environment import ExecutionEnvironment
from mmtoolsandbox.toolbox.toolbox import Toolbox, ToolboxConfig
from mmtoolsandbox.tools.api_docs import (
    api_docs_search_api_docs,
)
from mmtoolsandbox.tools.code_execution import (
    execute_code,
)
from mmtoolsandbox.tools.tool_sandbox.user_tools import end_conversation

# ===========================================================================
# Dummy Tools Definition
# ===========================================================================


@register_as_tool(
    toolboxes={DatasetName.FULL},
    visible_to=(RoleType.AGENT,),
)
def dummy_agent_tool() -> str:
    """Agent only tool."""
    return "agent"


@register_as_tool(
    toolboxes={DatasetName.FULL},
    visible_to=(RoleType.USER,),
)
def dummy_user_tool() -> str:
    """User only tool."""
    return "user"


@register_as_tool(
    toolboxes={DatasetName.FULL},
    visible_to=(RoleType.AGENT, RoleType.USER),
)
def dummy_shared_tool() -> str:
    """Shared tool."""
    return "shared"


@register_as_tool(
    toolboxes={DatasetName.FULL},
    visible_to=(RoleType.SYSTEM,),
)
def dummy_system_tool() -> str:
    """System only tool."""
    return "system"


@register_as_tool(
    toolboxes={DatasetName.FULL},
    database_namespaces=set(),
)
def dummy_default_tool() -> str:
    """Tool with default visibility (no visible_to attribute)."""
    return "default"


@register_as_tool(
    toolboxes={DatasetName.FULL},
    visible_to=(RoleType.AGENT,),
)
def dummy_hybrid_tool(arg: str) -> str:
    """A dummy tool for hybrid execution testing.

    Args:
        arg: The argument to echo back.
    """
    return f"hybrid_{arg}"


# ===========================================================================
# Test Cases
# ===========================================================================


class TestCodeExecutionFiltering(unittest.TestCase):
    """Tests for Toolbox.create_import_statement filtering logic.

    This test suite verifies:
    1. Role-based filtering (AGENT vs USER).
    2. Allow-list filtering.
    3. Interaction between allow-list and role visibility.
    4. Default visibility behavior.
    5. Module grouping in generated import statements.
    """

    def setUp(self) -> None:
        self.tools = [
            end_conversation,
            execute_code,
            dummy_agent_tool,
            dummy_user_tool,
            dummy_shared_tool,
            dummy_system_tool,
            dummy_default_tool,
        ]

        # Ensure dummy_default_tool does not have the visible_to attribute
        if hasattr(dummy_default_tool, "visible_to"):
            delattr(dummy_default_tool, "visible_to")

        self.toolbox = Toolbox(
            name="test_toolbox", config=ToolboxConfig(), tools=self.tools
        )

    def test_filter_by_agent_role(self) -> None:
        """Test that only agent-visible tools are imported for AGENT role."""
        import_statement = self.toolbox.create_import_statement(
            visible_to_role=RoleType.AGENT
        )

        # Agent-visible tools should be present
        self.assertIn("dummy_agent_tool", import_statement)
        self.assertIn("dummy_shared_tool", import_statement)

        # Default visibility is AGENT, so it should be present
        self.assertIn("dummy_default_tool", import_statement)

        # User-only and System-only tools should be excluded
        self.assertNotIn("dummy_user_tool", import_statement)
        self.assertNotIn("dummy_system_tool", import_statement)

        # end_conversation is a user tool, so it should not be visible to AGENT
        self.assertNotIn("end_conversation", import_statement)

    def test_filter_by_user_role(self) -> None:
        """Test that only user-visible tools are imported for USER role."""
        import_statement = self.toolbox.create_import_statement(
            visible_to_role=RoleType.USER
        )

        # User-visible tools should be present
        self.assertIn("dummy_shared_tool", import_statement)
        self.assertIn("dummy_user_tool", import_statement)

        # end_conversation is a user tool, so it should be visible to USER
        self.assertIn("end_conversation", import_statement)

        # Agent-only and System-only tools should be excluded
        self.assertNotIn("dummy_agent_tool", import_statement)
        self.assertNotIn("dummy_system_tool", import_statement)

        # Default visibility is AGENT, so dummy_default_tool should NOT be present for USER
        self.assertNotIn("dummy_default_tool", import_statement)

    def test_filter_by_allow_list(self) -> None:
        """Test that tool_allow_list restricts the imported tools."""
        # Allow only shared tool
        import_statement = self.toolbox.create_import_statement(
            tool_allow_list=["dummy_shared_tool"], visible_to_role=RoleType.AGENT
        )

        self.assertIn("dummy_shared_tool", import_statement)

        # Other tools should be excluded
        self.assertNotIn("dummy_agent_tool", import_statement)
        self.assertNotIn("dummy_user_tool", import_statement)
        self.assertNotIn("end_conversation", import_statement)

    def test_allow_list_respects_visibility(self) -> None:
        """Test that allow list cannot expose tools not visible to the role."""
        # Try to allow user tool for agent role (should fail to import)
        import_statement = self.toolbox.create_import_statement(
            tool_allow_list=["dummy_user_tool", "dummy_agent_tool"],
            visible_to_role=RoleType.AGENT,
        )

        # Agent tool should be present
        self.assertIn("dummy_agent_tool", import_statement)

        # User tool should be excluded despite being in allow list
        self.assertNotIn("dummy_user_tool", import_statement)

    def test_empty_allow_list(self) -> None:
        """Test that empty allow list results in no tools."""
        import_statement = self.toolbox.create_import_statement(
            tool_allow_list=[], visible_to_role=RoleType.AGENT
        )

        # No tools should be present
        self.assertNotIn("dummy_agent_tool", import_statement)
        self.assertNotIn("dummy_shared_tool", import_statement)
        self.assertNotIn("end_conversation", import_statement)

    def test_allow_list_with_non_existent_tools(self) -> None:
        """Test that non-existent tools in allow list are ignored."""
        import_statement = self.toolbox.create_import_statement(
            tool_allow_list=["non_existent_tool", "dummy_agent_tool"],
            visible_to_role=RoleType.AGENT,
        )

        self.assertIn("dummy_agent_tool", import_statement)
        self.assertNotIn("non_existent_tool", import_statement)

    def test_module_grouping(self) -> None:
        """Test that tools from the same module are grouped in the import statement."""
        # dummy_agent_tool and dummy_shared_tool are in the same module (this test file)
        import_statement = self.toolbox.create_import_statement(
            visible_to_role=RoleType.AGENT
        )

        # Verify that we don't have multiple lines for the same module
        lines = import_statement.splitlines()
        module_lines = [
            line
            for line in lines
            if f"from {dummy_agent_tool.__module__} import" in line
        ]

        # There should be exactly one line for this module
        self.assertEqual(len(module_lines), 1)
        line = module_lines[0]
        self.assertIn("dummy_agent_tool", line)
        self.assertIn("dummy_shared_tool", line)

    def test_none_allow_list(self) -> None:
        """Test that None allow list results in all visible tools."""
        import_statement = self.toolbox.create_import_statement(
            tool_allow_list=None, visible_to_role=RoleType.AGENT
        )

        # All agent-visible tools should be present
        self.assertIn("dummy_agent_tool", import_statement)
        self.assertIn("dummy_shared_tool", import_statement)
        self.assertIn("dummy_default_tool", import_statement)

        # User-only tools should still be excluded due to role visibility
        self.assertNotIn("dummy_user_tool", import_statement)
        self.assertNotIn("end_conversation", import_statement)

    def test_scenario_allow_list_behavior(self) -> None:
        """Test behavior mimicking a scenario with specific allow list."""
        # Simulate a scenario allow list
        scenario_allow_list = [
            "dummy_shared_tool",
            "dummy_user_tool",  # Should be filtered out because it's not visible to AGENT
            "non_existent_tool",  # Should be ignored
        ]

        import_statement = self.toolbox.create_import_statement(
            tool_allow_list=scenario_allow_list, visible_to_role=RoleType.AGENT
        )

        # Only allowed AND visible tools should be present
        self.assertIn("dummy_shared_tool", import_statement)

        # Allowed but not visible
        self.assertNotIn("dummy_user_tool", import_statement)

        # Not allowed
        self.assertNotIn("dummy_agent_tool", import_statement)
        self.assertNotIn("dummy_default_tool", import_statement)

    def test_execute_code_inclusion(self) -> None:
        """Test that execute_code is included when added to allow list."""
        scenario_allow_list = ["dummy_shared_tool"]

        discovery_tools = [
            "api_docs_search_api_docs",
        ]
        tool_allow_list = list(
            set(scenario_allow_list + discovery_tools + ["execute_code"])
        )

        import_statement = self.toolbox.create_import_statement(
            tool_allow_list=tool_allow_list, visible_to_role=RoleType.AGENT
        )

        # execute_code should be present
        self.assertIn("execute_code", import_statement)

        # dummy_shared_tool should be present
        self.assertIn("dummy_shared_tool", import_statement)

        # dummy_agent_tool should NOT be present (not in allow list)
        self.assertNotIn("dummy_agent_tool", import_statement)

    def test_role_tool_allow_list(self) -> None:
        """Test that role_tool_allow_list restricts tools for specific roles."""
        # Set up a context with role_tool_allow_list
        ctx = ExecutionContext(toolbox=self.toolbox)
        ctx.tool_allow_list = [
            "dummy_shared_tool",
            "dummy_agent_tool",
            "end_conversation",
        ]
        ctx.role_tool_allow_list = {RoleType.AGENT: ["dummy_shared_tool"]}

        # Agent should only see dummy_shared_tool
        agent_tools = ctx.get_available_tools_for_role(RoleType.AGENT)
        self.assertIn("dummy_shared_tool", agent_tools)
        self.assertNotIn("dummy_agent_tool", agent_tools)
        self.assertNotIn("end_conversation", agent_tools)

        # User should see end_conversation and dummy_shared_tool (since no role_tool_allow_list for USER)
        user_tools = ctx.get_available_tools_for_role(RoleType.USER)
        self.assertIn("end_conversation", user_tools)
        self.assertIn("dummy_shared_tool", user_tools)
        self.assertNotIn("dummy_agent_tool", user_tools)


class TestExecutionTransformation(unittest.TestCase):
    """Tests for transform_scenario_capabilities."""

    def setUp(self) -> None:
        # Create a dummy toolbox with end_conversation and discovery tools
        self.tools = [
            end_conversation,
            execute_code,
        ]
        self.toolbox = Toolbox(
            name="test_toolbox", config=ToolboxConfig(), tools=self.tools
        )

        # Create a dummy execution context
        self.ctx = ExecutionContext(toolbox=self.toolbox)

        # Mock the database
        sandbox_db = pl.DataFrame(
            {
                "sender": [RoleType.SYSTEM, RoleType.SYSTEM],
                "recipient": [RoleType.AGENT, RoleType.EXECUTION_ENVIRONMENT],
                "content": ["original system prompt", "original import statement"],
                "conversation_active": [True, True],
            }
        )
        self.ctx._dbs[DatabaseNamespace.SANDBOX] = sandbox_db

        # Create a dummy scenario
        self.scenario = MagicMock(spec=Scenario)
        self.scenario.starting_context = self.ctx
        self.scenario.categories = []
        self.scenario.max_messages = 10

    def test_end_conversation_exclusion(self) -> None:
        """Test that end_conversation is excluded from the import statement."""
        # Set tool_allow_list to something that doesn't include end_conversation explicitly
        # (though transform_scenario_capabilities adds it to the list, create_import_statement filters it out)
        self.ctx.tool_allow_list = ["some_other_tool"]

        transformed_scenario = transform_scenario_capabilities(
            self.scenario, enable_coding_tool=True
        )

        # Get the updated import statement from the database
        sandbox_db = transformed_scenario.starting_context._dbs[
            DatabaseNamespace.SANDBOX
        ]
        import_statement_row = sandbox_db.filter(
            (pl.col("sender") == RoleType.SYSTEM)
            & (pl.col("recipient") == RoleType.EXECUTION_ENVIRONMENT)
        )
        import_statement = import_statement_row["content"][0]

        # Verify end_conversation is NOT present
        self.assertNotIn(
            "from mmtoolsandbox.tools.tool_sandbox.user_tools import end_conversation",
            import_statement,
        )

    def test_execute_code_inclusion(self) -> None:
        """Test that execute_code is included in the import statement."""
        self.ctx.tool_allow_list = ["some_other_tool"]

        # Let's add execute_code to the toolbox for this test
        from mmtoolsandbox.tools.code_execution import execute_code

        self.toolbox.tools.append(execute_code)

        transformed_scenario = transform_scenario_capabilities(
            self.scenario, enable_coding_tool=True
        )

        sandbox_db = transformed_scenario.starting_context._dbs[
            DatabaseNamespace.SANDBOX
        ]
        import_statement_row = sandbox_db.filter(
            (pl.col("sender") == RoleType.SYSTEM)
            & (pl.col("recipient") == RoleType.EXECUTION_ENVIRONMENT)
        )
        import_statement = import_statement_row["content"][0]

        self.assertIn("execute_code", import_statement)

    def test_role_tool_allow_list_set(self) -> None:
        """Test that role_tool_allow_list is set correctly for the agent (Explicit Mode)."""
        self.ctx.tool_allow_list = ["some_other_tool", "end_conversation"]

        # We need to patch deepcopy here as well because transform_scenario_capabilities uses it
        with patch(
            "mmtoolsandbox.common.execution.copy.deepcopy", side_effect=lambda x: x
        ):
            # Explicitly pass the allow list to simulate Pattern 1 (Pre-enabled)
            transformed_scenario = transform_scenario_capabilities(
                self.scenario,
                enable_tool_search=True,
                enable_coding_tool=True,
                exposed_tool_allow_list=self.ctx.tool_allow_list,
            )

        # Agent should only have discovery + coding tools (not original tools)
        agent_tools = transformed_scenario.starting_context.role_tool_allow_list.get(
            RoleType.AGENT
        )
        assert agent_tools is not None
        self.assertIn("execute_code", agent_tools)
        self.assertIn("api_docs_search_api_docs", agent_tools)
        self.assertNotIn("some_other_tool", agent_tools)
        self.assertNotIn("end_conversation", agent_tools)

        # Global allow list is unchanged — keeps original tools for other roles.
        # Only agent's view is restricted via role_tool_allow_list.
        global_allow_list = transformed_scenario.starting_context.tool_allow_list
        assert global_allow_list is not None
        self.assertIn("some_other_tool", global_allow_list)
        self.assertIn("end_conversation", global_allow_list)

    def test_default_mode_preserves_tools(self) -> None:
        """Test that default transformation preserves existing tools and adds discovery."""
        self.ctx.tool_allow_list = ["some_other_tool", "end_conversation"]

        with patch(
            "mmtoolsandbox.common.execution.copy.deepcopy", side_effect=lambda x: x
        ):
            # Call with both flags -> Hybrid Mode
            transformed_scenario = transform_scenario_capabilities(
                self.scenario, enable_tool_search=True, enable_coding_tool=True
            )

        agent_tools = transformed_scenario.starting_context.role_tool_allow_list.get(
            RoleType.AGENT
        )
        assert agent_tools is not None

        # Should contain execute_code and discovery tools
        self.assertIn("execute_code", agent_tools)
        self.assertIn("api_docs_search_api_docs", agent_tools)

        # Should NOT contain original tools — agent must discover via search
        self.assertNotIn("some_other_tool", agent_tools)
        self.assertNotIn("end_conversation", agent_tools)

    def test_default_mode_with_empty_allow_list(self) -> None:
        """Test that default transformation works with empty allow list (Restricted Start)."""
        self.ctx.tool_allow_list = []

        with patch(
            "mmtoolsandbox.common.execution.copy.deepcopy", side_effect=lambda x: x
        ):
            # Call with both flags
            transformed_scenario = transform_scenario_capabilities(
                self.scenario, enable_tool_search=True, enable_coding_tool=True
            )

        agent_tools = transformed_scenario.starting_context.role_tool_allow_list.get(
            RoleType.AGENT
        )
        assert agent_tools is not None

        # Should contain execute_code and discovery tools
        self.assertIn("execute_code", agent_tools)
        self.assertIn("api_docs_search_api_docs", agent_tools)

        # Should NOT contain other tools (since none were allowed initially)
        self.assertNotIn("some_other_tool", agent_tools)

    def test_standard_mode(self) -> None:
        """Test that standard mode (both flags False) does not modify tools or prompts."""
        self.ctx.tool_allow_list = ["some_other_tool"]

        with patch(
            "mmtoolsandbox.common.execution.copy.deepcopy", side_effect=lambda x: x
        ):
            transformed_scenario = transform_scenario_capabilities(
                self.scenario, enable_tool_search=False, enable_coding_tool=False
            )

        # In standard mode, role_tool_allow_list might not be explicitly set if it wasn't before,
        # but if it is, it shouldn't contain the new tools.
        # Let's check the global allow list instead, or role_tool_allow_list if it exists.
        if transformed_scenario.starting_context.role_tool_allow_list:
            agent_tools = (
                transformed_scenario.starting_context.role_tool_allow_list.get(
                    RoleType.AGENT, []
                )
            )
            self.assertNotIn("execute_code", agent_tools)
            self.assertNotIn("api_docs_search_api_docs", agent_tools)

        global_tools = transformed_scenario.starting_context.tool_allow_list
        if global_tools is not None:
            self.assertNotIn("execute_code", global_tools)
            self.assertNotIn("api_docs_search_api_docs", global_tools)

        # Verify system prompt is unchanged
        sandbox_db = transformed_scenario.starting_context._dbs[
            DatabaseNamespace.SANDBOX
        ]
        system_prompt_row = sandbox_db.filter(
            (pl.col("sender") == RoleType.SYSTEM)
            & (pl.col("recipient") == RoleType.AGENT)
        )
        system_prompt = system_prompt_row["content"][0]
        self.assertEqual(system_prompt, "original system prompt")

        # Verify NO import statement is injected
        import_statement_row = sandbox_db.filter(
            (pl.col("sender") == RoleType.SYSTEM)
            & (pl.col("recipient") == RoleType.EXECUTION_ENVIRONMENT)
        )
        import_statement = import_statement_row["content"][0]
        self.assertEqual(import_statement, "original import statement")

    def test_search_only_mode(self) -> None:
        """Test that search-only mode adds discovery tools but not execute_code."""
        self.ctx.tool_allow_list = ["some_other_tool"]

        with patch(
            "mmtoolsandbox.common.execution.copy.deepcopy", side_effect=lambda x: x
        ):
            transformed_scenario = transform_scenario_capabilities(
                self.scenario, enable_tool_search=True, enable_coding_tool=False
            )

        agent_tools = transformed_scenario.starting_context.role_tool_allow_list.get(
            RoleType.AGENT
        )
        assert agent_tools is not None

        self.assertIn("api_docs_search_api_docs", agent_tools)
        self.assertNotIn("execute_code", agent_tools)
        # Agent should NOT see original tools — must discover via search
        self.assertNotIn("some_other_tool", agent_tools)

        # Verify system prompt
        sandbox_db = transformed_scenario.starting_context._dbs[
            DatabaseNamespace.SANDBOX
        ]
        system_prompt_row = sandbox_db.filter(
            (pl.col("sender") == RoleType.SYSTEM)
            & (pl.col("recipient") == RoleType.AGENT)
        )
        system_prompt = system_prompt_row["content"][0]
        self.assertIn("Discovery Process", system_prompt)
        self.assertNotIn("execute_code", system_prompt)

        # Verify import statement IS injected (replaces original)
        import_statement_row = sandbox_db.filter(
            (pl.col("sender") == RoleType.SYSTEM)
            & (pl.col("recipient") == RoleType.EXECUTION_ENVIRONMENT)
        )
        import_statement = import_statement_row["content"][0]
        self.assertNotEqual(import_statement, "original import statement")


class TestCodeExecutionReturnValues(unittest.TestCase):
    """Tests for execute_code return values, especially regarding images."""

    def setUp(self) -> None:
        self.context = MagicMock(spec=ExecutionContext)
        self.context.interactive_console = MagicMock()
        self.context.interactive_console.locals = {}
        self.context.safety_guard = None
        self.context.trace_nested_tool = False
        self.context.trace_tool = False
        set_current_context(self.context)

    def test_single_image(self) -> None:
        """Test that a single ImageResult is returned as is."""
        code = "ImageResult(1)"
        # Mock ImageResult in locals so eval works
        self.context.interactive_console.locals = {"ImageResult": ImageResult}

        result = execute_code(code)
        self.assertIsInstance(result, ImageResult)
        self.assertEqual(result.image_id, 1)

    def test_multiple_images(self) -> None:
        """Test that multiple ImageResults are returned as a list."""
        code = "[ImageResult(1), ImageResult(2)]"
        self.context.interactive_console.locals = {"ImageResult": ImageResult}

        result = execute_code(code)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], ImageResult)
        self.assertEqual(result[0].image_id, 1)
        self.assertEqual(result[1].image_id, 2)

    def test_mixed_text_and_image(self) -> None:
        """Test that mixed text and image are returned as a list."""
        code = """
print("Hello")
ImageResult(1)
"""
        self.context.interactive_console.locals = {"ImageResult": ImageResult}

        result = execute_code(code)
        # Expecting a list: ["Hello", ImageResult(1)]
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "Hello")
        self.assertIsInstance(result[1], ImageResult)
        self.assertEqual(result[1].image_id, 1)

    def test_mixed_text_and_multiple_images(self) -> None:
        """Test that mixed text and multiple images are returned as a nested list."""
        code = """
print("Hello")
[ImageResult(1), ImageResult(2)]
"""
        self.context.interactive_console.locals = {"ImageResult": ImageResult}

        result = execute_code(code)
        # Expecting a list: ["Hello", [ImageResult(1), ImageResult(2)]]
        # Note: execute_code no longer flattens the list
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "Hello")
        self.assertIsInstance(result[1], list)
        self.assertEqual(len(result[1]), 2)
        self.assertEqual(result[1][0].image_id, 1)
        self.assertEqual(result[1][1].image_id, 2)

    def test_mixed_text_and_image_in_list(self) -> None:
        """Test that mixed text and image in a list are returned as a nested list."""
        # Case where the last expression is explicitly a mixed list
        code = """
print("Start")
["Text", ImageResult(1)]
"""
        self.context.interactive_console.locals = {"ImageResult": ImageResult}

        result = execute_code(code)
        # Expecting: ["Start", ["Text", ImageResult(1)]]
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "Start")
        self.assertIsInstance(result[1], list)
        self.assertEqual(result[1][0], "Text")
        self.assertIsInstance(result[1][1], ImageResult)
        self.assertEqual(result[1][1].image_id, 1)

    def test_tuple_of_images(self) -> None:
        """Test that a tuple of ImageResults is returned as a tuple."""
        code = "(ImageResult(1), ImageResult(2))"
        self.context.interactive_console.locals = {"ImageResult": ImageResult}

        result = execute_code(code)
        # Expecting a tuple: (ImageResult(1), ImageResult(2))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], ImageResult)
        self.assertEqual(result[0].image_id, 1)
        self.assertEqual(result[1].image_id, 2)

    def test_dict_of_images(self) -> None:
        """Test that a dict containing ImageResults is returned as a dict."""
        code = "{'img1': ImageResult(1), 'img2': ImageResult(2)}"
        self.context.interactive_console.locals = {"ImageResult": ImageResult}

        result = execute_code(code)
        # Expecting a dict
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result["img1"], ImageResult)
        self.assertEqual(result["img1"].image_id, 1)
        self.assertIsInstance(result["img2"], ImageResult)
        self.assertEqual(result["img2"].image_id, 2)

    def test_nested_structure_with_images(self) -> None:
        """Test that a complex nested structure with images is returned as is."""
        code = """
{'data': [ImageResult(1), {'nested': ImageResult(2)}], 'meta': 'info'}
"""
        self.context.interactive_console.locals = {"ImageResult": ImageResult}

        result = execute_code(code)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["meta"], "info")
        self.assertIsInstance(result["data"], list)
        self.assertEqual(result["data"][0].image_id, 1)
        self.assertEqual(result["data"][1]["nested"].image_id, 2)

    def test_mixed_text_and_nested_structure(self) -> None:
        """Test that mixed text and nested structure are returned as a list."""
        code = """
print("Log")
{'data': ImageResult(1)}
"""
        self.context.interactive_console.locals = {"ImageResult": ImageResult}

        result = execute_code(code)
        # Expecting: ["Log", {'data': ImageResult(1)}]
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "Log")
        self.assertIsInstance(result[1], dict)
        self.assertEqual(result[1]["data"].image_id, 1)

    def test_set_of_images(self) -> None:
        """Test that a set of ImageResults is returned as a set."""
        code = "{ImageResult(1), ImageResult(2)}"
        self.context.interactive_console.locals = {"ImageResult": ImageResult}

        result = execute_code(code)
        # Expecting a set
        self.assertIsInstance(result, set)
        self.assertEqual(len(result), 2)
        # Sets are unordered, so we check if items are present
        ids = {img.image_id for img in result}
        self.assertEqual(ids, {1, 2})

    def test_circular_reference(self) -> None:
        """Test that circular references are handled gracefully."""
        code = """
l = []
l.append(l)
l.append(ImageResult(1))
l
"""
        self.context.interactive_console.locals = {"ImageResult": ImageResult}

        result = execute_code(code)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIs(result[0], result)  # Circular reference
        self.assertEqual(result[1].image_id, 1)


class TestHybridExecution(unittest.TestCase):
    """Tests for Hybrid Execution Mode."""

    def setUp(self) -> None:
        self.tools = [
            dummy_hybrid_tool,
            execute_code,
            end_conversation,
        ]
        self.toolbox = Toolbox(
            name="test_toolbox", config=ToolboxConfig(), tools=self.tools
        )
        self.ctx = ExecutionContext(toolbox=self.toolbox)

        # Mock the database
        sandbox_db = pl.DataFrame(
            {
                "sender": [RoleType.SYSTEM, RoleType.SYSTEM],
                "recipient": [RoleType.AGENT, RoleType.EXECUTION_ENVIRONMENT],
                "content": ["original system prompt", "original import statement"],
                "conversation_active": [True, True],
                "sandbox_message_index": [0, 1],
                "tool_trace": [None, None],
            }
        )
        self.ctx._dbs[DatabaseNamespace.SANDBOX] = sandbox_db

        # Create a dummy scenario using a simple class instead of MagicMock
        # to avoid pickling issues with deepcopy
        class MockScenario:
            def __init__(self, context: ExecutionContext) -> None:
                self.starting_context = context
                self.categories: list[Any] = []
                self.max_messages = 10

        self.scenario = MockScenario(self.ctx)

        # Set initial allow list
        self.ctx.tool_allow_list = ["dummy_hybrid_tool"]

    def test_hybrid_tool_visibility(self) -> None:
        """Test that both execute_code and standard tools are visible to the agent."""
        with patch(
            "mmtoolsandbox.common.execution.copy.deepcopy", side_effect=lambda x: x
        ):
            transformed_scenario = transform_scenario_capabilities(
                cast(Scenario, self.scenario),
                enable_tool_search=True,
                enable_coding_tool=True,
            )

        # Get the agent's allowed tools
        agent_tools = transformed_scenario.starting_context.role_tool_allow_list.get(
            RoleType.AGENT
        )
        assert agent_tools is not None

        # In Hybrid Mode, the agent should only see discovery + coding tools
        self.assertIn("execute_code", agent_tools)
        self.assertIn("api_docs_search_api_docs", agent_tools)
        # Original tools should NOT be visible — agent discovers via search
        self.assertNotIn("dummy_hybrid_tool", agent_tools)

    def test_system_prompt_update(self) -> None:
        """Test that the system prompt mentions both capabilities."""
        with patch(
            "mmtoolsandbox.common.execution.copy.deepcopy", side_effect=lambda x: x
        ):
            transformed_scenario = transform_scenario_capabilities(
                cast(Scenario, self.scenario),
                enable_tool_search=True,
                enable_coding_tool=True,
            )

        sandbox_db = transformed_scenario.starting_context._dbs[
            DatabaseNamespace.SANDBOX
        ]
        system_prompt_row = sandbox_db.filter(
            (pl.col("sender") == RoleType.SYSTEM)
            & (pl.col("recipient") == RoleType.AGENT)
        )
        system_prompt = system_prompt_row["content"][0]

        # Check for key phrases indicating hybrid mode
        self.assertIn("execute_code", system_prompt)
        self.assertIn("You have access to a set of tools", system_prompt)
        self.assertIn("function calling", system_prompt)
        self.assertIn("Auto-Enable", system_prompt)

    def test_hybrid_execution_runtime(self) -> None:
        """Test that both execute_code and standard tools can be executed."""
        # Setup context
        set_current_context(self.ctx)

        # Initialize ExecutionEnvironment
        env = ExecutionEnvironment()

        # 1. Test Standard Tool Execution
        # The agent calls dummy_hybrid_tool("test")
        # In the real system, OpenAIAPIAgent converts this to a python call string
        msg_standard = Message(
            sender=RoleType.AGENT,
            recipient=RoleType.EXECUTION_ENVIRONMENT,
            content='call_1_response = dummy_hybrid_tool("standard")',
            openai_tool_call_id="call_1",
            openai_function_name="dummy_hybrid_tool",
        )

        # We need to ensure the tool is in the interactive console locals
        # This is normally done by the import statement, but here we manually inject it
        cast(dict[str, Any], self.ctx.interactive_console.locals)[
            "dummy_hybrid_tool"
        ] = dummy_hybrid_tool
        cast(dict[str, Any], self.ctx.interactive_console.locals)["execute_code"] = (
            execute_code
        )

        responses, _ = env.respond([msg_standard], {}, None)
        self.assertEqual(len(responses), 1)
        self.assertIn("hybrid_standard", responses[0].content)

        # 2. Test execute_code Execution
        # The agent calls execute_code('print("hello")')
        msg_exec = Message(
            sender=RoleType.AGENT,
            recipient=RoleType.EXECUTION_ENVIRONMENT,
            content="call_2_response = execute_code('print(\"hello_exec\")')",
            openai_tool_call_id="call_2",
            openai_function_name="execute_code",
        )

        responses, _ = env.respond([msg_exec], {}, None)
        self.assertEqual(len(responses), 1)
        self.assertIn("hello_exec", responses[0].content)


class TestStandardHybridMode(unittest.TestCase):
    """Tests for Standard Hybrid Mode behavior and tool exposure."""

    def setUp(self) -> None:
        self.tools = [
            dummy_agent_tool,
            dummy_user_tool,
            end_conversation,
            execute_code,
            api_docs_search_api_docs,
            end_conversation,
        ]
        self.toolbox = Toolbox(
            name="test_toolbox", config=ToolboxConfig(), tools=self.tools
        )
        self.ctx = ExecutionContext(toolbox=self.toolbox)

        # Mock database
        sandbox_db = pl.DataFrame(
            {
                "sender": [RoleType.SYSTEM, RoleType.SYSTEM],
                "recipient": [RoleType.AGENT, RoleType.EXECUTION_ENVIRONMENT],
                "content": ["original system prompt", "original import statement"],
                "conversation_active": [True, True],
                "sandbox_message_index": [0, 1],
                "tool_trace": [None, None],
            }
        )
        self.ctx._dbs[DatabaseNamespace.SANDBOX] = sandbox_db

        # Create scenario
        self.scenario = MagicMock(spec=Scenario)
        self.scenario.starting_context = self.ctx
        self.scenario.categories = []
        self.scenario.max_messages = 10

        # Set initial allowed tools (standard scenario behavior)
        # Note: end_conversation is typically added by BaseScenario, so we include it here
        self.ctx.tool_allow_list = ["dummy_agent_tool", "end_conversation"]

    def test_standard_hybrid_mode_transformation(self) -> None:
        """Verify standard hybrid mode behavior (explicit exposed_tool_allow_list provided)."""

        # Patch deepcopy to avoid issues with MagicMock
        with patch(
            "mmtoolsandbox.common.execution.copy.deepcopy", side_effect=lambda x: x
        ):
            # Call transform_scenario_capabilities WITH exposed_tool_allow_list
            # This simulates the "Pattern 1" where we want specific tools pre-enabled
            transformed_scenario = transform_scenario_capabilities(
                self.scenario,
                enable_tool_search=True,
                enable_coding_tool=True,
                exposed_tool_allow_list=self.ctx.tool_allow_list,
            )

        ctx = transformed_scenario.starting_context

        # 1. Verify Tool Access & Visibility for AGENT
        # We check what is actually available to the agent (filtered by visibility)
        available_agent_tools = ctx.get_available_tools_for_role(RoleType.AGENT)

        # Original tools should NOT be visible to agent — must discover via search
        self.assertNotIn("dummy_agent_tool", available_agent_tools)

        # execute_code should be present
        self.assertIn("execute_code", available_agent_tools)

        # Discovery tools should be present
        self.assertIn("api_docs_search_api_docs", available_agent_tools)

        # end_conversation should NOT be present for AGENT (it's a user tool)
        self.assertNotIn("end_conversation", available_agent_tools)

        # 2. Verify Tool Access & Visibility for USER
        # The transformation should NOT modify User's access.
        # User should see end_conversation (visible to USER)
        user_tools = ctx.get_available_tools_for_role(RoleType.USER)
        self.assertIn("end_conversation", user_tools)

        # User should NOT see execute_code (visible to AGENT)
        self.assertNotIn("execute_code", user_tools)

        # User should NOT see discovery tools (visible to AGENT)
        self.assertNotIn("api_docs_search_api_docs", user_tools)

        # User should NOT see agent tools
        self.assertNotIn("dummy_agent_tool", user_tools)

        # 3. Verify Execution Environment Setup (Import Statement)
        sandbox_db = ctx._dbs[DatabaseNamespace.SANDBOX]
        import_statement_row = sandbox_db.filter(
            (pl.col("sender") == RoleType.SYSTEM)
            & (pl.col("recipient") == RoleType.EXECUTION_ENVIRONMENT)
        )
        import_statement = import_statement_row["content"][0]

        # Should import only discovery/coding tools — scenario tools are
        # added dynamically by register_tools() on discovery
        self.assertNotIn("dummy_agent_tool", import_statement)
        self.assertIn("execute_code", import_statement)
        self.assertIn("api_docs_search_api_docs", import_statement)

        # end_conversation (User tool) should NOT be imported for AGENT execution environment
        self.assertNotIn("end_conversation", import_statement)

        # 4. Verify System Prompt Modification
        system_prompt_row = sandbox_db.filter(
            (pl.col("sender") == RoleType.SYSTEM)
            & (pl.col("recipient") == RoleType.AGENT)
        )
        system_prompt = system_prompt_row["content"][0]

        # Check for key phrases
        self.assertIn("You have access to a set of tools", system_prompt)
        self.assertIn("execute_code", system_prompt)
        self.assertIn("Discovery Process", system_prompt)
        self.assertIn("Auto-Enable", system_prompt)


class TestAutoImport(unittest.TestCase):
    """Tests for Auto-Import (Implicit Registration) via Search."""

    def setUp(self) -> None:
        self.tools = [
            dummy_agent_tool,
            dummy_user_tool,
            dummy_shared_tool,
            execute_code,
            api_docs_search_api_docs,
            end_conversation,
        ]
        self.toolbox = Toolbox(
            name="test_toolbox", config=ToolboxConfig(), tools=self.tools
        )
        self.ctx = ExecutionContext(toolbox=self.toolbox)

        # Initially only execute_code and search are allowed
        self.ctx.tool_allow_list = ["execute_code", "api_docs_search_api_docs"]
        self.ctx.role_tool_allow_list = {
            RoleType.AGENT: ["execute_code", "api_docs_search_api_docs"]
        }
        # Use pure_code_exec so api_docs_search_api_docs returns structured results
        self.ctx.pure_code_exec = True

        # Initialize interactive console locals with ONLY allowed tools
        self.ctx.interactive_console.locals = {
            "execute_code": execute_code,
            "api_docs_search_api_docs": api_docs_search_api_docs,
        }

        set_current_context(self.ctx)

    def test_search_enables_tool(self) -> None:
        """Test that searching for a tool automatically enables it."""
        # Verify tool is not yet available
        self.assertNotIn(
            "dummy_agent_tool", self.ctx.get_available_tools_for_role(RoleType.AGENT)
        )
        self.assertNotIn("dummy_agent_tool", self.ctx.interactive_console.locals)

        # Search for the tool
        # This should trigger register_tools as a side effect
        results = api_docs_search_api_docs("dummy_agent_tool")

        # Verify search results
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["tool_name"], "dummy_agent_tool")

        # Verify tool is now available in role_tool_allow_list
        self.assertIn(
            "dummy_agent_tool", self.ctx.get_available_tools_for_role(RoleType.AGENT)
        )

        # Verify tool is now available in locals
        self.assertIn("dummy_agent_tool", self.ctx.interactive_console.locals)

        # Verify we can call it
        result = execute_code("dummy_agent_tool()")
        self.assertEqual(result, "'agent'")

    def test_search_does_not_enable_user_tools(self) -> None:
        """Test that searching for a user-only tool does not enable it."""
        # Search for user tool
        results = api_docs_search_api_docs("dummy_user_tool")

        # Should not find it (search respects visibility)
        found = any(r["tool_name"] == "dummy_user_tool" for r in results)
        self.assertFalse(found)

        # Should NOT be enabled
        self.assertNotIn("dummy_user_tool", self.ctx.interactive_console.locals)
        self.assertNotIn(
            "dummy_user_tool", self.ctx.get_available_tools_for_role(RoleType.AGENT)
        )


class TestDiscoveryToolVisibility(unittest.TestCase):
    """Tests that discovery tools have full visibility of the toolbox."""

    def setUp(self) -> None:
        self.tools = [
            dummy_agent_tool,
            dummy_user_tool,
            execute_code,
            api_docs_search_api_docs,
            end_conversation,
        ]
        self.toolbox = Toolbox(
            name="test_toolbox", config=ToolboxConfig(), tools=self.tools
        )
        self.ctx = ExecutionContext(toolbox=self.toolbox)

        # Restricted Execution: Only execute_code and discovery tools are allowed initially
        self.ctx.tool_allow_list = [
            "execute_code",
            "api_docs_search_api_docs",
        ]
        self.ctx.role_tool_allow_list = {RoleType.AGENT: self.ctx.tool_allow_list}
        # Use pure_code_exec so api_docs_search_api_docs returns structured results
        self.ctx.pure_code_exec = True

        # Initialize interactive console locals with ONLY allowed tools
        # This mimics the behavior of transform_scenario_capabilities in restricted mode
        self.ctx.interactive_console.locals = {
            tool.__name__: tool
            for tool in self.tools
            if tool.__name__ in self.ctx.tool_allow_list
        }

        set_current_context(self.ctx)

    def test_discovery_sees_hidden_tools(self) -> None:
        """Test that api_docs can see tools not in the allow list."""
        # Verify dummy_agent_tool is NOT in locals (simulating restricted execution)
        self.assertNotIn("dummy_agent_tool", self.ctx.interactive_console.locals)

        # Use api_docs_search_api_docs("dummy_agent_tool")
        # This should find the tool even if it's not in locals, because api_docs looks at the full toolbox.
        response = api_docs_search_api_docs(query="dummy_agent_tool")
        results = response

        self.assertTrue(len(results) > 0, "Discovery tool failed to find hidden tool")
        self.assertEqual(results[0]["tool_name"], "dummy_agent_tool")

    def test_discovery_hides_user_tools(self) -> None:
        """Test that api_docs does NOT see tools visible only to USER."""
        # dummy_user_tool is in the toolbox but visible_to=(RoleType.USER,)

        # Search for it
        response = api_docs_search_api_docs(query="dummy_user_tool")
        results = response

        # Should NOT find it
        found = any(r["tool_name"] == "dummy_user_tool" for r in results)
        self.assertFalse(found, "Discovery tool found a USER-only tool!")


class TestPureCodeExecTransformation(unittest.TestCase):
    """Tests for transform_scenario_capabilities with enable_pure_code_exec=True."""

    def setUp(self) -> None:
        self.tools = [
            dummy_agent_tool,
            dummy_user_tool,
            dummy_shared_tool,
            dummy_system_tool,
            execute_code,
            api_docs_search_api_docs,
            end_conversation,
        ]
        self.toolbox = Toolbox(
            name="test_toolbox", config=ToolboxConfig(), tools=self.tools
        )
        self.ctx = ExecutionContext(toolbox=self.toolbox)

        sandbox_db = pl.DataFrame(
            {
                "sender": [RoleType.SYSTEM, RoleType.SYSTEM],
                "recipient": [RoleType.AGENT, RoleType.EXECUTION_ENVIRONMENT],
                "content": ["original system prompt", "original import statement"],
                "conversation_active": [True, True],
            }
        )
        self.ctx._dbs[DatabaseNamespace.SANDBOX] = sandbox_db

        self.scenario = MagicMock(spec=Scenario)
        self.scenario.starting_context = self.ctx
        self.scenario.categories = []
        self.scenario.max_messages = 10

    def test_tool_allow_list_set_to_none(self) -> None:
        """Pure code exec should set tool_allow_list to None (all tools available)."""
        self.ctx.tool_allow_list = ["some_other_tool"]

        transformed = transform_scenario_capabilities(
            self.scenario, enable_pure_code_exec=True
        )

        self.assertIsNone(transformed.starting_context.tool_allow_list)

    def test_pure_code_exec_flag_set(self) -> None:
        """Pure code exec should set pure_code_exec=True on the context."""
        transformed = transform_scenario_capabilities(
            self.scenario, enable_pure_code_exec=True
        )

        self.assertTrue(transformed.starting_context.pure_code_exec)

    def test_agent_role_tool_allow_list_cleared(self) -> None:
        """Pure code exec should remove agent from role_tool_allow_list."""
        self.ctx.role_tool_allow_list = {
            RoleType.AGENT: ["some_tool"],
            RoleType.USER: ["user_tool"],
        }

        transformed = transform_scenario_capabilities(
            self.scenario, enable_pure_code_exec=True
        )

        role_list = transformed.starting_context.role_tool_allow_list
        self.assertNotIn(RoleType.AGENT, role_list)
        # User list should be preserved
        self.assertIn(RoleType.USER, role_list)

    def test_system_prompt_updated(self) -> None:
        """Pure code exec should replace the system prompt."""
        transformed = transform_scenario_capabilities(
            self.scenario, enable_pure_code_exec=True
        )

        sandbox_db = transformed.starting_context._dbs[DatabaseNamespace.SANDBOX]
        system_prompt_row = sandbox_db.filter(
            (pl.col("sender") == RoleType.SYSTEM)
            & (pl.col("recipient") == RoleType.AGENT)
        )
        system_prompt = system_prompt_row["content"][0]

        self.assertNotEqual(system_prompt, "original system prompt")
        # Should contain pure code exec instructions
        self.assertIn("```python", system_prompt)

    def test_import_statement_injected(self) -> None:
        """Pure code exec should inject an import statement for all agent-visible tools."""
        self.ctx.tool_allow_list = ["some_other_tool"]

        transformed = transform_scenario_capabilities(
            self.scenario, enable_pure_code_exec=True
        )

        sandbox_db = transformed.starting_context._dbs[DatabaseNamespace.SANDBOX]
        import_row = sandbox_db.filter(
            (pl.col("sender") == RoleType.SYSTEM)
            & (pl.col("recipient") == RoleType.EXECUTION_ENVIRONMENT)
        )
        import_statement = import_row["content"][0]

        self.assertNotEqual(import_statement, "original import statement")
        # Agent-visible tools should be imported
        self.assertIn("dummy_agent_tool", import_statement)
        self.assertIn("dummy_shared_tool", import_statement)

    def test_import_excludes_user_only_tools(self) -> None:
        """Pure code exec import should not include user-only tools."""
        transformed = transform_scenario_capabilities(
            self.scenario, enable_pure_code_exec=True
        )

        sandbox_db = transformed.starting_context._dbs[DatabaseNamespace.SANDBOX]
        import_row = sandbox_db.filter(
            (pl.col("sender") == RoleType.SYSTEM)
            & (pl.col("recipient") == RoleType.EXECUTION_ENVIRONMENT)
        )
        import_statement = import_row["content"][0]

        self.assertNotIn("dummy_user_tool", import_statement)

    def test_mutually_exclusive_with_coding_tool(self) -> None:
        """Pure code exec and enable_coding_tool should not be used together
        (they produce different system prompts / tool setups)."""
        # When both are True, enable_pure_code_exec takes precedence
        # (tool_allow_list becomes None). Verify the pure-exec behavior wins.
        transformed = transform_scenario_capabilities(
            self.scenario,
            enable_pure_code_exec=True,
            enable_coding_tool=True,
        )
        self.assertIsNone(transformed.starting_context.tool_allow_list)

    def test_reasoning_instruction_appended_to_code_exec_prompt(self) -> None:
        """Verify reasoning instruction is appended in code-exec mode."""
        transformed = transform_scenario_capabilities(
            self.scenario, enable_pure_code_exec=True, enable_reasoning=True
        )

        sandbox_db = transformed.starting_context._dbs[DatabaseNamespace.SANDBOX]
        system_prompt_row = sandbox_db.filter(
            (pl.col("sender") == RoleType.SYSTEM)
            & (pl.col("recipient") == RoleType.AGENT)
        )
        system_prompt = system_prompt_row["content"][0]

        # Should have both code-exec prompt AND reasoning instruction
        self.assertIn("```python", system_prompt)
        self.assertIn("<think>", system_prompt)


class TestExtractCodeBlocks(unittest.TestCase):
    """Tests for extract_code_blocks helper."""

    def test_single_block(self) -> None:
        text = "Here is code:\n```python\nprint('hello')\n```\nDone."
        blocks = extract_code_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].strip(), "print('hello')")

    def test_multiple_blocks(self) -> None:
        text = "```python\na = 1\n```\ntext\n```python\nb = 2\n```"
        blocks = extract_code_blocks(text)
        self.assertEqual(len(blocks), 2)
        self.assertIn("a = 1", blocks[0])
        self.assertIn("b = 2", blocks[1])

    def test_no_blocks(self) -> None:
        text = "No code here, just text."
        blocks = extract_code_blocks(text)
        self.assertEqual(len(blocks), 0)

    def test_non_python_blocks_ignored(self) -> None:
        text = "```javascript\nconsole.log('hi')\n```\n```python\nprint('hi')\n```"
        blocks = extract_code_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertIn("print('hi')", blocks[0])

    def test_multiline_code(self) -> None:
        text = "```python\ndef foo():\n    return 42\n\nresult = foo()\n```"
        blocks = extract_code_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertIn("def foo():", blocks[0])
        self.assertIn("result = foo()", blocks[0])


class TestToOpenaiMessagesForCodeExec(unittest.TestCase):
    """Tests for to_openai_messages_for_code_exec message conversion."""

    def test_system_to_agent(self) -> None:
        messages = [
            Message(
                sender=RoleType.SYSTEM,
                recipient=RoleType.AGENT,
                content="You are a helpful assistant.",
            )
        ]
        result = to_openai_messages_for_code_exec(messages)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[0]["content"], "You are a helpful assistant.")

    def test_user_to_agent(self) -> None:
        messages = [
            Message(
                sender=RoleType.USER,
                recipient=RoleType.AGENT,
                content="Hello!",
            )
        ]
        result = to_openai_messages_for_code_exec(messages)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "user")
        self.assertEqual(result[0]["content"], "Hello!")

    def test_agent_to_user(self) -> None:
        messages = [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.USER,
                content="I can help with that.",
            )
        ]
        result = to_openai_messages_for_code_exec(messages)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "assistant")

    def test_agent_to_exec_env(self) -> None:
        """Agent code block messages become assistant messages."""
        messages = [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content="Let me run:\n```python\nprint('hi')\n```",
            )
        ]
        result = to_openai_messages_for_code_exec(messages)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "assistant")
        self.assertIn("```python", result[0]["content"])

    def test_exec_env_to_agent(self) -> None:
        """Execution results are wrapped in XML tags as user messages."""
        messages = [
            Message(
                sender=RoleType.EXECUTION_ENVIRONMENT,
                recipient=RoleType.AGENT,
                content="hi",
            )
        ]
        result = to_openai_messages_for_code_exec(messages)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "user")
        self.assertIn(EXECUTION_RESULTS_OPEN_TAG, result[0]["content"])
        self.assertIn(EXECUTION_RESULTS_CLOSE_TAG, result[0]["content"])
        self.assertIn("hi", result[0]["content"])

    def test_exec_env_error(self) -> None:
        """Execution errors include the error prefix."""
        messages = [
            Message(
                sender=RoleType.EXECUTION_ENVIRONMENT,
                recipient=RoleType.AGENT,
                content="output",
                tool_call_exception="NameError: name 'x' is not defined",
            )
        ]
        result = to_openai_messages_for_code_exec(messages)
        self.assertEqual(len(result), 1)
        self.assertIn(CODE_EXEC_ERROR_PREFIX, result[0]["content"])
        self.assertIn("NameError", result[0]["content"])

    def test_consecutive_exec_results_aggregated(self) -> None:
        """Consecutive execution results are merged into a single user message."""
        messages = [
            Message(
                sender=RoleType.EXECUTION_ENVIRONMENT,
                recipient=RoleType.AGENT,
                content="result1",
            ),
            Message(
                sender=RoleType.EXECUTION_ENVIRONMENT,
                recipient=RoleType.AGENT,
                content="result2",
            ),
        ]
        result = to_openai_messages_for_code_exec(messages)
        # Both results should be in a single user message
        self.assertEqual(len(result), 1)
        self.assertIn("result1", result[0]["content"])
        self.assertIn("result2", result[0]["content"])

    def test_unrecognized_sender_recipient_raises(self) -> None:
        """Unrecognized message pairs should raise ValueError."""
        messages = [
            Message(
                sender=RoleType.EXECUTION_ENVIRONMENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content="bad",
            )
        ]
        with self.assertRaises(ValueError):
            to_openai_messages_for_code_exec(messages)


class TestBasePromptMaskPreservesScenarioMessages(unittest.TestCase):
    """Tests that transform_scenario_capabilities only modifies the first
    SYSTEM→AGENT message (the base prompt), preserving subsequent
    scenario-specific SYSTEM→AGENT messages unchanged.
    """

    def setUp(self) -> None:
        self.tools = [
            dummy_agent_tool,
            dummy_user_tool,
            execute_code,
            api_docs_search_api_docs,
            end_conversation,
        ]
        self.toolbox = Toolbox(
            name="test_toolbox", config=ToolboxConfig(), tools=self.tools
        )
        self.ctx = ExecutionContext(toolbox=self.toolbox)

        # Two SYSTEM→AGENT messages: base prompt + scenario-specific instruction
        sandbox_db = pl.DataFrame(
            {
                "sender": [
                    RoleType.SYSTEM,
                    RoleType.SYSTEM,
                    RoleType.SYSTEM,
                ],
                "recipient": [
                    RoleType.AGENT,
                    RoleType.EXECUTION_ENVIRONMENT,
                    RoleType.AGENT,
                ],
                "content": [
                    "base system prompt",
                    "import statement",
                    "scenario-specific UI instructions",
                ],
                "conversation_active": [True, True, True],
            }
        )
        self.ctx._dbs[DatabaseNamespace.SANDBOX] = sandbox_db

        self.scenario = MagicMock(spec=Scenario)
        self.scenario.starting_context = self.ctx
        self.scenario.categories = []
        self.scenario.max_messages = 10

    def test_code_exec_only_overwrites_first_system_agent(self) -> None:
        """Code-exec prompt should replace the first SYSTEM→AGENT message only."""
        transformed = transform_scenario_capabilities(
            self.scenario, enable_tool_search=True, enable_coding_tool=True
        )
        sandbox_db = transformed.starting_context._dbs[DatabaseNamespace.SANDBOX]
        sys_agent_rows = sandbox_db.filter(
            (pl.col("sender") == RoleType.SYSTEM)
            & (pl.col("recipient") == RoleType.AGENT)
        )

        self.assertEqual(sys_agent_rows.height, 2)
        # First row: overwritten with code-exec prompt
        self.assertIn("execute_code", sys_agent_rows["content"][0])
        self.assertNotEqual("base system prompt", sys_agent_rows["content"][0])
        # Second row: preserved unchanged
        self.assertEqual(
            "scenario-specific UI instructions", sys_agent_rows["content"][1]
        )

    def test_reasoning_only_appends_to_first_system_agent(self) -> None:
        """Reasoning instructions should only be appended to the first SYSTEM→AGENT."""
        transformed = transform_scenario_capabilities(
            self.scenario,
            enable_tool_search=True,
            enable_coding_tool=True,
            enable_reasoning=True,
        )
        sandbox_db = transformed.starting_context._dbs[DatabaseNamespace.SANDBOX]
        sys_agent_rows = sandbox_db.filter(
            (pl.col("sender") == RoleType.SYSTEM)
            & (pl.col("recipient") == RoleType.AGENT)
        )

        # First row: has reasoning instruction appended
        self.assertIn("<think>", sys_agent_rows["content"][0])
        # Second row: unchanged — no reasoning instruction leaked
        self.assertNotIn("<think>", sys_agent_rows["content"][1])
        self.assertEqual(
            "scenario-specific UI instructions", sys_agent_rows["content"][1]
        )

    def test_coding_guidelines_only_appends_to_first_system_agent(self) -> None:
        """Coding environment guidelines should only be appended to the first SYSTEM→AGENT."""
        transformed = transform_scenario_capabilities(
            self.scenario,
            enable_tool_search=True,
            enable_coding_tool=True,
        )
        sandbox_db = transformed.starting_context._dbs[DatabaseNamespace.SANDBOX]
        sys_agent_rows = sandbox_db.filter(
            (pl.col("sender") == RoleType.SYSTEM)
            & (pl.col("recipient") == RoleType.AGENT)
        )

        # First row: has coding guidelines appended
        self.assertIn("blocked", sys_agent_rows["content"][0])
        # Second row: unchanged
        self.assertEqual(
            "scenario-specific UI instructions", sys_agent_rows["content"][1]
        )

    def test_single_system_agent_still_works(self) -> None:
        """When there's only one SYSTEM→AGENT message, it should still be overwritten."""
        sandbox_db = pl.DataFrame(
            {
                "sender": [RoleType.SYSTEM, RoleType.SYSTEM],
                "recipient": [RoleType.AGENT, RoleType.EXECUTION_ENVIRONMENT],
                "content": ["original prompt", "import statement"],
                "conversation_active": [True, True],
            }
        )
        self.ctx._dbs[DatabaseNamespace.SANDBOX] = sandbox_db

        transformed = transform_scenario_capabilities(
            self.scenario, enable_tool_search=True, enable_coding_tool=True
        )
        sandbox_db = transformed.starting_context._dbs[DatabaseNamespace.SANDBOX]
        sys_agent_rows = sandbox_db.filter(
            (pl.col("sender") == RoleType.SYSTEM)
            & (pl.col("recipient") == RoleType.AGENT)
        )

        self.assertEqual(sys_agent_rows.height, 1)
        self.assertIn("execute_code", sys_agent_rows["content"][0])
        self.assertNotEqual("original prompt", sys_agent_rows["content"][0])


if __name__ == "__main__":
    unittest.main()
