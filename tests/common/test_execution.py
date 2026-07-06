# Copyright © 2026 Apple Inc.

"""Unit tests for mmtoolsandbox/common/execution.py orchestration functions.

Tests cover:
- transform_scenario_capabilities() — capability rewrites (tool search, coding,
  pure code exec, reasoning, UI)
- maybe_repeat_scenarios() — scenario repetition with suffix naming
- get_category_summary() — result aggregation by category
- get_category_to_scenario_count() — category counting with dedup
- _is_retryable_api_error() — transient error pattern matching
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution import (
    SCENARIO_REPEAT_SUFFIX,
    _is_retryable_api_error,
    get_category_summary,
    get_category_to_scenario_count,
    maybe_repeat_scenarios,
    transform_scenario_capabilities,
)
from mmtoolsandbox.common.execution_context import (
    ExecutionContext,
    RoleType,
    ScenarioCategories,
)
from mmtoolsandbox.common.prompt_templates import (
    SECTION_CODE_ENVIRONMENT,
    SECTION_EXTENDED_REASONING,
    SECTION_REASONING,
    UI_AGENT_TOOLS,
    UI_USER_TOOLS,
)
from mmtoolsandbox.common.scenario import Scenario
from mmtoolsandbox.datasets.names import DatasetName
from mmtoolsandbox.toolbox.toolbox import Toolbox, ToolboxConfig
from mmtoolsandbox.tools.tool_sandbox.user_tools import end_conversation

# ---------------------------------------------------------------------------
# Shared constants and fixtures
# ---------------------------------------------------------------------------

_DATASET_NAME = DatasetName.FULL

_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def _make_toolbox() -> Toolbox:
    """Create a minimal toolbox containing only ``end_conversation``."""
    return Toolbox(
        name=_DATASET_NAME,
        config=ToolboxConfig(),
        tools=[end_conversation],
    )


def _make_execution_context(toolbox: Toolbox | None = None) -> ExecutionContext:
    """Create a minimal ``ExecutionContext`` suitable for unit tests."""
    if toolbox is None:
        toolbox = _make_toolbox()
    ctx = ExecutionContext(
        toolbox=toolbox,
        delay_initialization=True,
        safety_guard_config=None,
    )
    ctx.add_to_database(
        namespace=DatabaseNamespace.SETTING,
        rows=[
            {
                "device_id": "test-device-001",
                "cellular": True,
                "wifi": True,
                "location_service": True,
                "low_battery_mode": False,
                "locale": "en_US",
                "utc_offset_seconds": -25200,
                "latitude": 37.3346,
                "longitude": -122.0091,
            }
        ],
    )
    ctx.tool_allow_list = ["end_conversation"]
    return ctx


def _make_scenario(
    ctx: ExecutionContext | None = None,
    *,
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    user_message: str = "Hello, please help me.",
    max_messages: int = 20,
) -> Scenario:
    """Build a minimal ``Scenario`` with system prompt and one user message."""
    if ctx is None:
        ctx = _make_execution_context()
    import_stmt = ctx.toolbox.create_import_statement()
    ctx.add_to_database(
        namespace=DatabaseNamespace.SANDBOX,
        rows=[
            {
                "sender": RoleType.SYSTEM,
                "recipient": RoleType.EXECUTION_ENVIRONMENT,
                "content": import_stmt,
            },
            {
                "sender": RoleType.SYSTEM,
                "recipient": RoleType.AGENT,
                "content": system_prompt,
            },
            {
                "sender": RoleType.USER,
                "recipient": RoleType.AGENT,
                "content": user_message,
            },
        ],
    )
    return Scenario(
        starting_context=ctx,
        max_messages=max_messages,
    )


@pytest.fixture
def base_scenario() -> Scenario:
    """A fresh minimal scenario for transform tests."""
    return _make_scenario()


# ---------------------------------------------------------------------------
# Tests: transform_scenario_capabilities
# ---------------------------------------------------------------------------


class TestTransformScenarioCapabilities:
    """Tests for transform_scenario_capabilities()."""

    def test_no_transforms_returns_original(self, base_scenario: Scenario) -> None:
        """When all flags are False the same scenario object is returned."""
        result = transform_scenario_capabilities(base_scenario)
        # Should be the exact same object (no deep copy).
        assert result is base_scenario

    def test_enable_tool_search_adds_api_docs(self, base_scenario: Scenario) -> None:
        """Enabling tool search adds api_docs_search_api_docs to the allow list."""
        result = transform_scenario_capabilities(base_scenario, enable_tool_search=True)
        assert result.starting_context.tool_allow_list is not None
        assert "api_docs_search_api_docs" in result.starting_context.tool_allow_list

    def test_enable_tool_search_sets_agent_role_allow_list(
        self, base_scenario: Scenario
    ) -> None:
        """Tool search mode sets a role-specific allow list for the agent."""
        result = transform_scenario_capabilities(base_scenario, enable_tool_search=True)
        agent_tools = result.starting_context.role_tool_allow_list[RoleType.AGENT]
        assert "api_docs_search_api_docs" in agent_tools

    def test_enable_coding_tool_adds_execute_code(
        self, base_scenario: Scenario
    ) -> None:
        """Enabling the coding tool adds execute_code to the global allow list."""
        result = transform_scenario_capabilities(base_scenario, enable_coding_tool=True)
        assert result.starting_context.tool_allow_list is not None
        assert "execute_code" in result.starting_context.tool_allow_list

    def test_enable_pure_code_exec_sets_tool_allow_list_none(
        self, base_scenario: Scenario
    ) -> None:
        """Pure code exec mode clears the tool allow list (None = all tools)."""
        result = transform_scenario_capabilities(
            base_scenario, enable_pure_code_exec=True
        )
        assert result.starting_context.tool_allow_list is None

    def test_enable_pure_code_exec_sets_pure_code_exec_flag(
        self, base_scenario: Scenario
    ) -> None:
        """Pure code exec mode sets the pure_code_exec flag on the context."""
        result = transform_scenario_capabilities(
            base_scenario, enable_pure_code_exec=True
        )
        assert result.starting_context.pure_code_exec is True

    def test_enable_pure_code_exec_implies_tool_search(
        self, base_scenario: Scenario
    ) -> None:
        """Pure code exec implies tool search; the system prompt should reflect
        this by being rewritten (not the original default prompt)."""
        original_prompt = _get_base_system_prompt(base_scenario)
        result = transform_scenario_capabilities(
            base_scenario, enable_pure_code_exec=True
        )
        new_prompt = _get_base_system_prompt(result)
        assert new_prompt != original_prompt

    def test_enable_pure_code_exec_removes_agent_from_role_allow_list(
        self, base_scenario: Scenario
    ) -> None:
        """Pure code exec removes the AGENT key from role_tool_allow_list
        because the agent discovers tools dynamically."""
        # Pre-seed a role_tool_allow_list with AGENT
        base_scenario.starting_context.role_tool_allow_list = {
            RoleType.AGENT: ["some_tool"],
        }
        result = transform_scenario_capabilities(
            base_scenario, enable_pure_code_exec=True
        )
        assert RoleType.AGENT not in result.starting_context.role_tool_allow_list

    def test_enable_reasoning_standard_appends_reasoning_section(
        self, base_scenario: Scenario
    ) -> None:
        """Standard reasoning appends the SECTION_REASONING block to the
        base system prompt."""
        result = transform_scenario_capabilities(
            base_scenario,
            enable_reasoning=True,
            enable_tool_search=True,  # need at least one flag to trigger transform
        )
        prompt = _get_base_system_prompt(result)
        assert SECTION_REASONING in prompt
        assert SECTION_EXTENDED_REASONING not in prompt

    def test_enable_reasoning_extended_appends_extended_section(
        self, base_scenario: Scenario
    ) -> None:
        """Extended reasoning appends the SECTION_EXTENDED_REASONING block."""
        result = transform_scenario_capabilities(
            base_scenario,
            enable_reasoning=True,
            reasoning_effort="extended",
            enable_tool_search=True,
        )
        prompt = _get_base_system_prompt(result)
        assert SECTION_EXTENDED_REASONING in prompt

    def test_enable_coding_tool_appends_code_environment_section(
        self, base_scenario: Scenario
    ) -> None:
        """Coding tool mode appends the SECTION_CODE_ENVIRONMENT to the
        base system prompt (only when NOT in pure_code_exec mode)."""
        result = transform_scenario_capabilities(base_scenario, enable_coding_tool=True)
        prompt = _get_base_system_prompt(result)
        assert SECTION_CODE_ENVIRONMENT in prompt

    def test_enable_ui_adds_agent_tools(self, base_scenario: Scenario) -> None:
        """Enabling UI adds all UI_AGENT_TOOLS to the global allow list."""
        result = transform_scenario_capabilities(
            base_scenario, enable_ui=True, enable_tool_search=True
        )
        for tool_name in UI_AGENT_TOOLS:
            assert result.starting_context.tool_allow_list is not None
            assert tool_name in result.starting_context.tool_allow_list

    def test_enable_ui_adds_user_tools(self, base_scenario: Scenario) -> None:
        """Enabling UI adds UI_USER_TOOLS to the user's role allow list."""
        result = transform_scenario_capabilities(
            base_scenario, enable_ui=True, enable_tool_search=True
        )
        user_tools = result.starting_context.role_tool_allow_list[RoleType.USER]
        for tool_name in UI_USER_TOOLS:
            assert tool_name in user_tools

    def test_enable_ui_sets_evaluation_enable_ui(self, base_scenario: Scenario) -> None:
        """Enabling UI sets enable_ui on evaluation_criteria if present."""
        result = transform_scenario_capabilities(
            base_scenario, enable_ui=True, enable_tool_search=True
        )
        assert result.evaluation_criteria.enable_ui is True

    def test_transform_deep_copies_scenario(self, base_scenario: Scenario) -> None:
        """Transformation must deep-copy: the original scenario is not mutated."""
        original_allow_list = list(base_scenario.starting_context.tool_allow_list or [])
        _ = transform_scenario_capabilities(base_scenario, enable_tool_search=True)
        assert base_scenario.starting_context.tool_allow_list == original_allow_list

    def test_end_conversation_always_in_global_allow_list(
        self, base_scenario: Scenario
    ) -> None:
        """end_conversation must be in the global allow list after any transform
        so the user simulator can call it."""
        result = transform_scenario_capabilities(base_scenario, enable_tool_search=True)
        assert result.starting_context.tool_allow_list is not None
        assert "end_conversation" in result.starting_context.tool_allow_list

    def test_tool_search_pins_scenario_tools(self, base_scenario: Scenario) -> None:
        """Original scenario tools should be pinned so they are not evicted
        from the LRU cache during tool discovery."""
        original_tools = list(base_scenario.starting_context.tool_allow_list or [])
        result = transform_scenario_capabilities(base_scenario, enable_tool_search=True)
        for tool_name in original_tools:
            assert tool_name in result.starting_context.pinned_tools

    def test_tool_search_with_coding_adds_both(self, base_scenario: Scenario) -> None:
        """Tool search + coding tool adds both api_docs_search_api_docs and
        execute_code to the agent's role allow list."""
        result = transform_scenario_capabilities(
            base_scenario,
            enable_tool_search=True,
            enable_coding_tool=True,
        )
        agent_tools = result.starting_context.role_tool_allow_list[RoleType.AGENT]
        assert "api_docs_search_api_docs" in agent_tools
        assert "execute_code" in agent_tools

    def test_no_duplicate_tools_in_allow_list(self, base_scenario: Scenario) -> None:
        """Tools should not be duplicated in the global allow list."""
        result = transform_scenario_capabilities(base_scenario, enable_tool_search=True)
        allow_list = result.starting_context.tool_allow_list
        assert allow_list is not None
        assert len(allow_list) == len(set(allow_list))


# ---------------------------------------------------------------------------
# Tests: maybe_repeat_scenarios
# ---------------------------------------------------------------------------


class TestMaybeRepeatScenarios:
    """Tests for maybe_repeat_scenarios()."""

    def test_single_repeat_returns_with_suffix(self) -> None:
        """A single repeat (num_scenario_repeats=1) returns one entry per
        scenario with the /run_0 suffix."""
        scenario = _make_scenario()
        result = maybe_repeat_scenarios(1, [("test_scenario", scenario)])
        assert len(result) == 1
        name, _ = result[0]
        assert name == f"test_scenario{SCENARIO_REPEAT_SUFFIX}0"

    def test_multiple_repeats_adds_suffix(self) -> None:
        """Multiple repeats produce N entries with sequential /run_N suffixes."""
        scenario = _make_scenario()
        result = maybe_repeat_scenarios(3, [("test_scenario", scenario)])
        assert len(result) == 3
        names = [name for name, _ in result]
        assert names == [
            f"test_scenario{SCENARIO_REPEAT_SUFFIX}0",
            f"test_scenario{SCENARIO_REPEAT_SUFFIX}1",
            f"test_scenario{SCENARIO_REPEAT_SUFFIX}2",
        ]

    def test_repeated_scenarios_are_independent_copies(self) -> None:
        """Each repeated scenario should be a deep copy so mutations do not
        propagate across repeats."""
        scenario = _make_scenario()
        result = maybe_repeat_scenarios(2, [("test_scenario", scenario)])
        _, s0 = result[0]
        _, s1 = result[1]
        assert s0 is not s1
        assert s0.starting_context is not s1.starting_context

    def test_conflicting_name_raises_value_error(self) -> None:
        """A scenario name ending with the repeat suffix should raise."""
        scenario = _make_scenario()
        bad_name = f"conflict{SCENARIO_REPEAT_SUFFIX}"
        with pytest.raises(ValueError, match="conflicts with the prefix"):
            maybe_repeat_scenarios(1, [(bad_name, scenario)])

    def test_multiple_scenarios_all_repeated(self) -> None:
        """All input scenarios are repeated the specified number of times."""
        s1 = _make_scenario()
        s2 = _make_scenario()
        result = maybe_repeat_scenarios(2, [("alpha", s1), ("beta", s2)])
        assert len(result) == 4
        names = {name for name, _ in result}
        assert names == {
            f"alpha{SCENARIO_REPEAT_SUFFIX}0",
            f"alpha{SCENARIO_REPEAT_SUFFIX}1",
            f"beta{SCENARIO_REPEAT_SUFFIX}0",
            f"beta{SCENARIO_REPEAT_SUFFIX}1",
        }


# ---------------------------------------------------------------------------
# Tests: get_category_summary
# ---------------------------------------------------------------------------


class TestGetCategorySummary:
    """Tests for get_category_summary()."""

    def test_aggregates_turn_count_to_all_categories(self) -> None:
        """Every result's turn_count should appear in ALL_CATEGORIES."""
        results = [
            _make_result(
                "s1", categories=[ScenarioCategories.SINGLE_TOOL_CALL], turn_count=5
            ),
            _make_result(
                "s2", categories=[ScenarioCategories.SINGLE_TOOL_CALL], turn_count=10
            ),
        ]
        summary = get_category_summary(results)
        assert summary["ALL_CATEGORIES"]["turn_count"] == [5, 10]

    def test_aggregates_per_category(self) -> None:
        """Results should be grouped under their specific categories."""
        results = [
            _make_result(
                "s1", categories=[ScenarioCategories.SINGLE_TOOL_CALL], turn_count=3
            ),
            _make_result(
                "s2", categories=[ScenarioCategories.MULTIPLE_TOOL_CALL], turn_count=7
            ),
        ]
        summary = get_category_summary(results)
        assert summary[ScenarioCategories.SINGLE_TOOL_CALL]["turn_count"] == [3]
        assert summary[ScenarioCategories.MULTIPLE_TOOL_CALL]["turn_count"] == [7]

    def test_aggregates_judge_results(self) -> None:
        """Judge results (result + criteria_evaluation) should be aggregated."""
        results = [
            _make_result(
                "s1",
                categories=[ScenarioCategories.SINGLE_TOOL_CALL],
                turn_count=5,
                judge_result={
                    "result": True,
                    "criteria_evaluation": [
                        {"criterion": "task_completion", "pass": True},
                        {"criterion": "no_side_effects", "pass": False},
                    ],
                },
            ),
        ]
        summary = get_category_summary(results)
        cat_summary = summary[ScenarioCategories.SINGLE_TOOL_CALL]
        assert cat_summary["judge_pass"] == [1.0]
        assert cat_summary["judge_task_completion"] == [1.0]
        assert cat_summary["judge_no_side_effects"] == [0.0]

    def test_empty_results_returns_empty_summary(self) -> None:
        """Empty input should produce an empty summary dict."""
        summary = get_category_summary([])
        assert len(summary) == 0


# ---------------------------------------------------------------------------
# Tests: get_category_to_scenario_count
# ---------------------------------------------------------------------------


class TestGetCategoryToScenarioCount:
    """Tests for get_category_to_scenario_count()."""

    def test_counts_all_categories(self) -> None:
        """ALL_CATEGORIES should count every scenario once."""
        s1 = _make_scenario()
        s1.categories = [ScenarioCategories.SINGLE_TOOL_CALL]
        s2 = _make_scenario()
        s2.categories = [ScenarioCategories.MULTIPLE_TOOL_CALL]
        count = get_category_to_scenario_count({"s1": s1, "s2": s2})
        assert count["ALL_CATEGORIES"] == 2

    def test_counts_specific_categories(self) -> None:
        """Each specific category should accumulate its own count."""
        s1 = _make_scenario()
        s1.categories = [ScenarioCategories.SINGLE_TOOL_CALL]
        s2 = _make_scenario()
        s2.categories = [ScenarioCategories.SINGLE_TOOL_CALL]
        count = get_category_to_scenario_count({"s1": s1, "s2": s2})
        assert count[ScenarioCategories.SINGLE_TOOL_CALL] == 2


# ---------------------------------------------------------------------------
# Tests: _is_retryable_api_error
# ---------------------------------------------------------------------------


class TestIsRetryableApiError:
    """Tests for _is_retryable_api_error()."""

    @pytest.mark.parametrize(
        "exception_type",
        [
            "RateLimitError",
            "APIError",
            "APIStatusError",
            "APIConnectionError",
            "VertexAINonRetryableError",
            "VertexAIRateLimitError",
            "Timeout",
            "TimeoutError",
            "ConnectionError",
            "ServiceUnavailable",
            "BadGateway",
        ],
    )
    def test_retryable_exception_types(self, exception_type: str) -> None:
        """Known retryable exception types should return True."""
        result = {"exception_type": exception_type, "traceback": ""}
        assert _is_retryable_api_error(result) is True

    @pytest.mark.parametrize(
        "status_code",
        ["429", "500", "502", "503"],
    )
    def test_retryable_status_codes_in_traceback(self, status_code: str) -> None:
        """Retryable HTTP status codes in the traceback should match."""
        result = {
            "exception_type": "SomeError",
            "traceback": f"HTTP {status_code} response",
        }
        assert _is_retryable_api_error(result) is True

    def test_non_retryable_exception(self) -> None:
        """Agent logic errors (ValueError, KeyError) should NOT be retryable."""
        result = {"exception_type": "ValueError", "traceback": "some traceback"}
        assert _is_retryable_api_error(result) is False

    def test_missing_fields_returns_false(self) -> None:
        """When exception_type and traceback are both absent, return False."""
        assert _is_retryable_api_error({}) is False

    def test_none_fields_returns_false(self) -> None:
        """Explicit None values should not cause errors."""
        result: dict[str, Any] = {"exception_type": None, "traceback": None}
        assert _is_retryable_api_error(result) is False

    def test_resource_exhausted_in_traceback(self) -> None:
        """RESOURCE_EXHAUSTED (gRPC code) in traceback should be retryable."""
        result = {
            "exception_type": "GoogleAPIError",
            "traceback": "google.api_core.exceptions.ResourceExhausted: RESOURCE_EXHAUSTED",
        }
        assert _is_retryable_api_error(result) is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_base_system_prompt(scenario: Scenario) -> str:
    """Extract the base SYSTEM->AGENT prompt from a scenario's SANDBOX DB."""
    sandbox_db = scenario.starting_context._dbs[DatabaseNamespace.SANDBOX]
    sys_agent_rows = sandbox_db.filter(
        (pl.col("sender") == RoleType.SYSTEM) & (pl.col("recipient") == RoleType.AGENT)
    )
    assert len(sys_agent_rows) > 0, "No SYSTEM->AGENT prompt found"
    return str(sys_agent_rows["content"][0])


def _make_result(
    name: str,
    *,
    categories: list[ScenarioCategories],
    turn_count: int = 5,
    judge_result: dict[str, Any] | None = None,
    ui_judge_result: dict[str, Any] | None = None,
    user_judge_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal result dict matching the shape from run_scenario()."""
    return {
        "name": name,
        "categories": categories,
        "traceback": None,
        "exception_type": None,
        "similarity": 0.5,
        "turn_count": turn_count,
        "judge_result": judge_result,
        "ui_judge_result": ui_judge_result,
        "user_judge_result": user_judge_result,
        "task_completion_criteria": None,
        "entity_diff_result": None,
    }
