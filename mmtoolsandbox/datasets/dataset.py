# Copyright © 2026 Apple Inc.

from __future__ import annotations

from typing import Sequence

from attrs import define

from mmtoolsandbox.common.scenario import Scenario, filter_scenarios_by_tag_expression
from mmtoolsandbox.toolbox.toolbox import Toolbox

# A common use case is to run a subset of scenarios from a dataset. In this case we
# want to update the original dataset name and version so that we can distinguish it
# from the official dataset.
NAME_SUFFIX_FOR_MODIFIED_DATASET = "_local_mods"
VERSION_SUFFIX_FOR_MODIFIED_DATASET = ".dev"


@define(frozen=True)
class Dataset:
    # The name of the dataset.
    # Note that this is a string instead of a `DatasetName` enum value since we
    # potentially add the `NAME_SUFFIX_FOR_MODIFIED_DATASET` to it.
    name: str

    # Mapping from scenario name to its definition.
    scenario_name_to_def: dict[str, Scenario]

    # The toolbox defining the available tools.
    toolbox: Toolbox

    # Semantic version of the dataset in the format of 'major.minor.patch'. For more
    # details see https://packaging.python.org/en/latest/discussions/versioning
    # Note: We rely on developer's diligence to update the version when changing the
    # dataset. There are no automatic checks to enforce updating the versions.
    version: str

    def __len__(self) -> int:
        return len(self.scenario_name_to_def)

    def __repr__(self) -> str:
        return f"< Dataset '{self.name}' (# Scen. {len(self)}) >"


def filter_dataset_by_name(
    dataset: Dataset,
    scenario_names: Sequence[str],
) -> Dataset:
    if len(scenario_names) == 0:
        return dataset

    # Check if the given scenario name subset contains all scenarios of the original
    # dataset.
    original_scenario_names = set(dataset.scenario_name_to_def.keys())
    scenario_names_subset = set(scenario_names)
    if scenario_names_subset == original_scenario_names:
        return dataset

    # Raise an exception if some of the given scenarios are not part of the dataset.
    missing_scenarios = scenario_names_subset - original_scenario_names
    if len(missing_scenarios) > 0:
        msg = (
            f"{dataset} does not contain the following scenario(s)"
            f"{sorted(missing_scenarios)}"
        )
        if 0 < len(original_scenario_names) <= 10:
            msg += f"\nTry instead: {original_scenario_names}"

        raise KeyError(msg)

    filtered_name_to_scenario = {
        name: scenario
        for name, scenario in dataset.scenario_name_to_def.items()
        if name in scenario_names_subset
    }
    return Dataset(
        name=dataset.name + NAME_SUFFIX_FOR_MODIFIED_DATASET,
        scenario_name_to_def=filtered_name_to_scenario,
        toolbox=dataset.toolbox,
        version=dataset.version + VERSION_SUFFIX_FOR_MODIFIED_DATASET,
    )


def filter_dataset_by_tags(dataset: Dataset, tag_expression: str | None) -> Dataset:
    if tag_expression is None or tag_expression == "":
        return dataset

    scenarios = dataset.scenario_name_to_def
    filtered_scenarios = filter_scenarios_by_tag_expression(scenarios, tag_expression)
    return filter_dataset_by_name(
        dataset, scenario_names=list(filtered_scenarios.keys())
    )
