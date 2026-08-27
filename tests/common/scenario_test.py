# Copyright © 2026 Apple Inc.

import copy
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mmtoolsandbox.common.evaluation import EvaluationResult
from mmtoolsandbox.common.execution import run_scenario
from mmtoolsandbox.common.execution_context import set_current_context
from mmtoolsandbox.common.scenario import Scenario, ScenarioExtension, ScenarioResult
from mmtoolsandbox.datasets.base_scenario import create_base_scenario
from mmtoolsandbox.roles.base_role import BaseRole
from mmtoolsandbox.toolbox.toolbox import Toolbox, ToolboxConfig
from mmtoolsandbox.tools.tool_sandbox.user_tools import end_conversation


@pytest.fixture
def toolbox() -> Toolbox:
    return Toolbox(
        name="my_toolbox",
        config=ToolboxConfig(),
        tools=[end_conversation],
    )


def test_tool_reference_checks(toolbox: Toolbox) -> None:
    base_scenario = create_base_scenario(toolbox)
    extension = ScenarioExtension(
        name="my_scenario",
        base_scenario=base_scenario,
        tool_allow_list=None,
        tool_deny_list=None,
    )
    # Ensure that we can generate scenarios from a valid `ScenarioExtension`.
    assert extension.get_extended_scenario() is not None

    # Generating the scenarios should fail if we specify a non-existent tool in the
    # allow list.
    extension_with_invalid_allow_list = copy.deepcopy(extension)
    extension_with_invalid_allow_list.tool_allow_list = ["this_tool_does_not_exist"]
    with pytest.raises(RuntimeError, match=r"not part of.*toolbox"):
        extension_with_invalid_allow_list.get_extended_scenario()

    # Generating the scenarios should fail if we specify a non-existent tool in the
    # allow list.
    extension_with_invalid_deny_list = copy.deepcopy(extension)
    extension_with_invalid_deny_list.tool_allow_list = ["this_tool_does_not_exist"]
    with pytest.raises(RuntimeError, match=r"not part of.*toolbox"):
        extension_with_invalid_deny_list.get_extended_scenario()


def test_run_scenario_propagates_task_completion_criteria() -> None:
    # Mock Scenario
    mock_scenario = MagicMock(spec=Scenario)
    mock_scenario.categories = ["test_category"]
    mock_scenario.starting_context = MagicMock()
    mock_scenario.max_messages = 10

    # Mock EvaluationResult with task_completion_criteria
    mock_eval_result = MagicMock(spec=EvaluationResult)
    mock_eval_result.milestone_similarity = 1.0
    mock_eval_result.minefield_similarity = 0.0
    mock_eval_result.similarity = 1.0
    mock_eval_result.turn_count = 5
    mock_eval_result.milestone_mapping = {}
    mock_eval_result.minefield_mapping = {}
    mock_eval_result.judge_result = "Success"
    mock_eval_result.task_completion_criteria = "Complete the task successfully."

    # Mock ScenarioResult
    mock_scenario_result = MagicMock(spec=ScenarioResult)
    mock_scenario_result.evaluation_result = mock_eval_result

    # Setup play_and_evaluate return value
    mock_scenario.play_and_evaluate.return_value = (mock_scenario_result, [], [])

    # Mock Roles
    mock_agent_type = MagicMock(spec=BaseRole)
    mock_user_type = MagicMock(spec=BaseRole)

    # Mock output directory
    mock_output_dir = MagicMock(spec=Path)

    # Run scenario
    with patch("mmtoolsandbox.common.execution._load_role") as mock_load_role:
        mock_load_role.return_value = MagicMock()

        result = run_scenario(
            name_and_scenario=("test_scenario", mock_scenario),
            dataset_name="test_dataset",
            agent_type=mock_agent_type,
            user_type=mock_user_type,
            output_directory=mock_output_dir,
            agent_config=None,
            user_config=None,
            fail_on_error=False,
        )

    # Verify task_completion_criteria is in the result
    assert "task_completion_criteria" in result
    assert result["task_completion_criteria"] == "Complete the task successfully."


def _make_scenario_with_starting_context(toolbox: Toolbox) -> Scenario:
    """Build a Scenario via the normal extension pipeline so starting_context is set."""
    base_scenario = create_base_scenario(toolbox)
    extension = ScenarioExtension(
        name="save_path_test",
        base_scenario=base_scenario,
        tool_allow_list=None,
        tool_deny_list=None,
    )
    scenarios = extension.get_extended_scenario()
    return next(iter(scenarios.values()))


def _noop_play_that_sets_context(scenario: Scenario) -> None:
    """Replace Scenario.play with a no-op that still seeds the thread-local context.

    play_and_evaluate's finally block calls get_current_context(); we need it to
    return a valid ExecutionContext even though we skip real turn execution.
    """
    ctx = copy.deepcopy(scenario.starting_context)
    ctx.execute_delayed_initialization()
    set_current_context(ctx)


def test_play_and_evaluate_saves_trajectory_when_judge_fails(
    toolbox: Toolbox, tmp_path: Path
) -> None:
    """Judge failure should NOT lose the agent trajectory.

    play() succeeds, evaluate() raises -> conversation.json must still be
    written and the returned ScenarioResult must carry a stub EvaluationResult
    (judge_result=None). play_and_evaluate itself must NOT re-raise.
    """
    scenario = _make_scenario_with_starting_context(toolbox)
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    with (
        patch.object(
            Scenario,
            "play",
            autospec=True,
            side_effect=lambda self, **kw: _noop_play_that_sets_context(self),
        ),
        patch(
            "mmtoolsandbox.common.scenario.evaluate",
            side_effect=RuntimeError("simulated judge failure"),
        ) as evaluate_mock,
    ):
        result, _, _ = scenario.play_and_evaluate(
            roles={},
            output_directory=output_dir,
            scenario_name="save_path_test",
            dataset_name="test_dataset",
            judge_name=None,
        )

    assert evaluate_mock.call_count == 1

    # Trajectory is saved even though the judge raised.
    trajectory_dir = output_dir / "trajectories" / "save_path_test"
    assert (trajectory_dir / "conversation.json").exists(), (
        "conversation.json must be written when only the judge fails"
    )
    assert (trajectory_dir / "execution_context.json").exists()

    # Stub evaluation result carries no judge verdict — backfill_judges fills this in.
    assert isinstance(result.evaluation_result, EvaluationResult)
    assert result.evaluation_result.judge_result is None
    assert result.evaluation_result.entity_diff_result is None


def test_play_and_evaluate_does_not_save_when_play_fails(
    toolbox: Toolbox, tmp_path: Path
) -> None:
    """Agent failure must keep the current behavior: no trajectory saved, exception re-raised.

    Saving a partial agent trajectory would incorrectly mark the scenario
    ``complete`` under ``--continue``. We want ``--retry-failed`` to rerun it.
    """
    scenario = _make_scenario_with_starting_context(toolbox)
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    with patch.object(
        Scenario,
        "play",
        autospec=True,
        side_effect=RuntimeError("simulated agent failure"),
    ):
        with pytest.raises(RuntimeError, match="simulated agent failure"):
            scenario.play_and_evaluate(
                roles={},
                output_directory=output_dir,
                scenario_name="save_path_test",
                dataset_name="test_dataset",
                judge_name=None,
            )

    trajectory_dir = output_dir / "trajectories" / "save_path_test"
    assert not (trajectory_dir / "conversation.json").exists(), (
        "conversation.json must NOT be written when the agent rollout fails"
    )
