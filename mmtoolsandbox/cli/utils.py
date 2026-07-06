# Copyright © 2026 Apple Inc.

from __future__ import annotations

from typing import Any

from mmtoolsandbox.common.i18n import Locale
from mmtoolsandbox.datasets.dataset import (
    Dataset,
    filter_dataset_by_name,
    filter_dataset_by_tags,
)
from mmtoolsandbox.datasets.names import DatasetName
from mmtoolsandbox.datasets.registry import load_dataset

# The scenarios to play back when the `--test_mode` flag is set.
TEST_SCENARIO_NAMES = [
    "send_message_with_contact_content_cellular_off_multiple_user_turn",
    "send_message_with_contact_content_cellular_off_multiple_user_turn_10_distraction_tools",
    "send_message_with_contact_content_cellular_off_3_distraction_tools_arg_description_scrambled",
    # "remove_contact_by_phone_multiple_user_turn",
    # "find_temperature_f_with_location_and_time_diff_multiple_user_turn",
]


def resolve_dataset(
    dataset_name: DatasetName,
    dataset_config: dict[str, Any],
    locale: Locale,
    desired_scenario_names: list[str] | None,
    desired_scenario_tag_expression: str | None,
) -> Dataset:
    """Resolve the scenarios to run.

    Args:
        dataset_name:                    Name of the dataset to use.
        dataset_config:                  Configuration of the dataset.
        locale:                          Locale of the dataset.
        desired_scenario_names:          Name of scenarios to run. If empty all scenarios will be returned.
        desired_scenario_tag_expression: Tag expression to filter scenarios.

    Returns:
        The dataset with optionally filtered scenarios.
    """
    dataset = load_dataset(dataset_name, dataset_config, locale)
    dataset = filter_dataset_by_name(
        dataset, scenario_names=desired_scenario_names or []
    )
    dataset = filter_dataset_by_tags(
        dataset, tag_expression=desired_scenario_tag_expression
    )
    return dataset
