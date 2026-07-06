# Copyright © 2026 Apple Inc.

import pytest

from mmtoolsandbox.common.execution_context import ExecutionContext
from mmtoolsandbox.common.scenario import Scenario
from mmtoolsandbox.datasets.dataset import (
    NAME_SUFFIX_FOR_MODIFIED_DATASET,
    VERSION_SUFFIX_FOR_MODIFIED_DATASET,
    Dataset,
    filter_dataset_by_name,
    filter_dataset_by_tags,
)
from mmtoolsandbox.toolbox.toolbox import Toolbox, ToolboxConfig

MOCK_DATASET = Dataset(
    name="test",
    scenario_name_to_def={
        "scenario_0": Scenario(starting_context=ExecutionContext(), tags={"01"}),
        "scenario_1": Scenario(starting_context=ExecutionContext(), tags={"01", "12"}),
        "scenario_2": Scenario(starting_context=ExecutionContext(), tags={"2", "12"}),
    },
    toolbox=Toolbox(
        name="test_toolbox",
        config=ToolboxConfig(),
        tools=[],
    ),
    version="0.1.0",
)


def test_filter_dataset_original_scenario_subset() -> None:
    # Filtering by all scenarios of the dataset should return the original dataset.
    assert MOCK_DATASET == filter_dataset_by_name(
        MOCK_DATASET,
        scenario_names=list(MOCK_DATASET.scenario_name_to_def.keys()),
    )


def test_filter_dataset_empty_scenario_subset() -> None:
    # Not filtering by any scenarios should return the original dataset.
    assert MOCK_DATASET == filter_dataset_by_name(MOCK_DATASET, scenario_names=[])


def test_filter_dataset_using_subset() -> None:
    # Pick the last `N`` scenario names of the original dataset for filtering.
    N = 2
    scenario_name_subset = list(MOCK_DATASET.scenario_name_to_def.keys())[-N:]
    # Make sure that the mock dataset has enough scenarios so that we actually have a
    # subset of size `N`.
    assert len(scenario_name_subset) == N

    filtered_dataset = filter_dataset_by_name(
        MOCK_DATASET,
        scenario_names=scenario_name_subset,
    )
    assert MOCK_DATASET.name + NAME_SUFFIX_FOR_MODIFIED_DATASET == filtered_dataset.name
    expected_scenario_name_to_def = {
        name: scenario
        for name, scenario in MOCK_DATASET.scenario_name_to_def.items()
        if name in scenario_name_subset
    }
    assert expected_scenario_name_to_def == filtered_dataset.scenario_name_to_def
    assert (
        MOCK_DATASET.version + VERSION_SUFFIX_FOR_MODIFIED_DATASET
        == filtered_dataset.version
    )


def test_filter_dataset_non_existent_scenario() -> None:
    all_scenario_names = list(MOCK_DATASET.scenario_name_to_def.keys())
    non_existent_scenario = "this scenario does not exist"
    assert non_existent_scenario not in all_scenario_names

    with pytest.raises(KeyError, match=r"does not contain the following scenario\(s\)"):
        filter_dataset_by_name(
            MOCK_DATASET,
            scenario_names=all_scenario_names[:-1] + [non_existent_scenario],
        )


def test_filter_tags_empty() -> None:
    result_dset = filter_dataset_by_tags(MOCK_DATASET, None)
    assert result_dset.scenario_name_to_def.keys() == {
        "scenario_0",
        "scenario_1",
        "scenario_2",
    }


@pytest.mark.parametrize(
    "expr,expected_scens",
    [
        ("01", {0, 1}),
        ("12", {1, 2}),
        ("2", {2}),
        ("(2)", {2}),
        ("01 | 12", {0, 1, 2}),
        ("01|12", {0, 1, 2}),
        ("01 & 12", {1}),
        ("01 & ~12", {0}),
        ("01 & ~(12)", {0}),
        ("01 | 01", {0, 1}),
        ("12 | 01 & 2", {1, 2}),
        ("(12 | 01) & 2", {2}),
    ],
)
def test_filter_tags_legal(expr: str, expected_scens: set[int]) -> None:
    expected_scenario_names = {f"scenario_{i}" for i in expected_scens}
    result_dset = filter_dataset_by_tags(MOCK_DATASET, expr)
    assert result_dset.scenario_name_to_def.keys() == expected_scenario_names


def test_filter_tags_illegal() -> None:
    with pytest.raises(ValueError, match="No scenario matches"):
        filter_dataset_by_tags(MOCK_DATASET, "01 & 2")  # empty set

    with pytest.raises(ValueError, match="Malformed"):
        filter_dataset_by_tags(MOCK_DATASET, "(01 & 2")

    with pytest.raises(ValueError, match="unknown_tags"):
        filter_dataset_by_tags(MOCK_DATASET, "01 & unknown_tag")
