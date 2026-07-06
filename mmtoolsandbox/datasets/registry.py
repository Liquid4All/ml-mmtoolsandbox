# Copyright © 2026 Apple Inc.

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mmtoolsandbox.common.i18n import DefaultLocalizer, Locale, Localizer
from mmtoolsandbox.common.scenario import Scenario
from mmtoolsandbox.datasets.dataset import Dataset
from mmtoolsandbox.datasets.names import DatasetName
from mmtoolsandbox.toolbox.loading import load_toolbox
from mmtoolsandbox.toolbox.toolbox import Toolbox


def populate_dataset_registry() -> None:
    # Importing datasets at module level results in circular imports,
    # therefore datasets are not registered until this function is called explicitly.
    # The unified scenarios module registers the FULL dataset.
    from mmtoolsandbox.datasets import scenarios as scenarios  # noqa: F811


class ScenarioFactory(Protocol):
    def __call__(
        self,
        base_scenarios: dict[str, Scenario],
        toolbox: Toolbox,
        config: dict[str, Any],
        locale: Locale,
        localize: Localizer,
    ) -> dict[str, Scenario]:
        """Construct scenarios for a dataset.

        Args:
            base_scenarios: Base scenarios to extends.
            toolbox:        Toolbox to use for scenarios.
            config:         Config options to customize scenario creation.
            locale:         Locale for scenarios.
            localize:       Localizer instance to localize scenarios.

        Returns:
            Dictionary of named scenarios.
        """
        ...


class BaseScenarioFactory(Protocol):
    def __call__(
        self,
        toolbox: Toolbox,
        locale: Locale,
        localize: Localizer,
    ) -> dict[str, Scenario]:
        """Construct base scenarios.

        Args:
            toolbox:  Toolbox to use for base scenarios.
            locale:   Locale to use for base scenarios.
            localize: Localizer instance to localize base scenarios.

        Returns:
            Dictionary of named base scenarios.
        """
        ...


@dataclass(frozen=True, kw_only=True)
class DatasetRegistryEntry:
    scenario_factory: ScenarioFactory
    base_scenario_factory: BaseScenarioFactory
    localized_data_path: Path | str | None = None


_registry: dict[DatasetName, dict[str, DatasetRegistryEntry]] = {}


def register_datasets(
    dataset_names: DatasetName | list[DatasetName],
    *,
    version: str,
    entry: DatasetRegistryEntry,
) -> None:
    """Function used to add dataset registry entries.

    This function saves the provided entry to the dataset registry which enables retrieving them by name.

    Args:
        dataset_names: Dataset name or list of dataset names.
        version:       Version string for the datasets.
        entry:         Entry to register.
    """

    if isinstance(dataset_names, DatasetName):
        dataset_names = [dataset_names]

    for dataset_name in dataset_names:
        if dataset_name not in _registry:
            _registry[dataset_name] = {}

        _registry[dataset_name][version] = entry


def _get_dataset_registry_entry(
    dataset_name: DatasetName, version: str | None
) -> tuple[DatasetRegistryEntry, str]:
    """Find an entry from the dataset registry.

    Args:
        dataset_name: Name of the dataset.
        version:      Version of the dataset. If None, the latest version is used.

    Returns:
        Tuple where the first element is the entry and the second element is the version.
    """
    if dataset_name not in _registry:
        populate_dataset_registry()

    try:
        entries = _registry[dataset_name]
    except KeyError:
        raise RuntimeError(f"Dataset '{dataset_name}' does not exist.")

    if version is None:
        version = max(entries.keys())

    try:
        entry = entries[version]
    except KeyError:
        raise RuntimeError(
            f"Dataset '{dataset_name}' with version '{version}' does not exist."
        )

    return entry, version


def load_dataset(
    dataset_name: DatasetName,
    config: dict[str, Any],
    locale: Locale = Locale.en_US,
    version: str | None = None,
) -> Dataset:
    """Get the dataset with the given name.

    Args:
        dataset_name: Name of the dataset to get.
        config:       Configuration of the dataset.
        locale:       Locale for which to load the dataset.
        version:      Version of the dataset. Defaults to None which means the latest version.

    Returns:
        The dataset.
    """

    result = load_scenarios(
        dataset_name,
        config=frozenset(config.items()),
        locale=locale,
        version=version,
    )

    return Dataset(
        name=dataset_name,
        scenario_name_to_def=result.scenarios,
        toolbox=result.toolbox,
        version=result.version,
    )


@dataclass(frozen=True, kw_only=True)
class LoadScenariosResult:
    base_scenarios: dict[str, Scenario]
    scenarios: dict[str, Scenario]
    version: str
    toolbox: Toolbox


@functools.cache
def load_scenarios(
    dataset_name: DatasetName,
    config: frozenset[tuple[str, Any]],
    locale: Locale = Locale.en_US,
    version: str | None = None,
) -> LoadScenariosResult:
    """Load scenarios for the given dataset.

    This function looks up an entry from the registry and calls the factory function
    associated with the entry to load the scenarios.

    Args:
        dataset_name:  Name of the dataset.
        config:        Configuration of the dataset. Needs to be mapped
                        back to a dict from frozenset due to cache support.
        locale:        Locale for the scenario. Defaults to en_US.
        version:       Version of the dataset. Defaults to None which means the latest version.

    Returns:
        LoadScenariosResult instance which contains base scenarios, extended scenarios,
        the loaded dataset version and toolbox.
    """
    toolbox = load_toolbox(dataset_name, config={})

    entry, version = _get_dataset_registry_entry(dataset_name, version=version)

    if path := entry.localized_data_path:
        localize = Localizer.from_path(path, locale)
    else:
        localize = DefaultLocalizer

    base_scenarios = entry.base_scenario_factory(
        toolbox=toolbox,
        locale=locale,
        localize=localize,
    )

    scenarios = entry.scenario_factory(
        base_scenarios=base_scenarios,
        toolbox=toolbox,
        config=dict(config),
        locale=locale,
        localize=localize,
    )

    return LoadScenariosResult(
        base_scenarios=base_scenarios,
        scenarios=scenarios,
        version=version,
        toolbox=toolbox,
    )
