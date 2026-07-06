# Copyright © 2026 Apple Inc.

import json
from collections import Counter
from logging import getLogger
from pathlib import Path
from typing import Any

import mmtoolsandbox
from mmtoolsandbox.common.utils_git import get_git_sha, has_local_changes

LOGGER = getLogger(__name__)


def write_result_summary(
    result_summaries: list[dict[str, Any]],
    category_summary: dict[str, dict[str, list[float]]],
    output_directory: Path,
    num_scenario_repeats: int,
) -> None:
    """Write the `result_summary.json` file.

    Args:
        result_summaries:     The result summaries for each scenario.
        category_summary:     Mapping from scenario category name to results.
        output_directory:     The directory under which to store the results file.
        num_scenario_repeats:  The number of times each scenario should be repeated.
    """
    git_sha = get_git_sha()
    if git_sha is not None and has_local_changes():
        git_sha += " + local changes"

    total = len(result_summaries)
    passed = sum(
        1
        for r in result_summaries
        if r.get("judge_result") is not None and r["judge_result"].get("result") is True
    )

    with open(output_directory / "result_summary.json", "w") as f:
        json.dump(
            {
                "pass_rate": round(passed / total, 4) if total > 0 else 0.0,
                "per_scenario_results": result_summaries,
                "category_aggregated_results": {
                    category: {k: sum(v) / len(v) for k, v in aggregation.items()}
                    for category, aggregation in category_summary.items()
                },
                "num_scenario_repeats": num_scenario_repeats,
                "git_sha": git_sha,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )


def write_aggregated_metrics_json(
    dataset_name: str,
    *,
    dataset_version: str,
    category_summary: dict[str, dict[str, list[float]]],
    output_directory: Path,
) -> None:
    prefix = (
        f"MMToolSandbox/{mmtoolsandbox.__version__}/"
        f"{dataset_name}/{dataset_version}/categories"
    )
    metrics = {}
    for category_str, metric_name_to_values in category_summary.items():
        for metric_name, values in metric_name_to_values.items():
            avg_metric = sum(values) / len(values)
            metrics[f"{prefix}/{category_str}/{metric_name}/avg"] = avg_metric

    wandb_metrics_path = output_directory / "wandb_metrics.json"
    with open(wandb_metrics_path, "wt") as f:
        json.dump(metrics, f, indent=2)


def write_exceptions_summary(
    result_summaries: list[dict[str, Any]], output_directory: Path
) -> None:
    with open(output_directory / "exceptions.json", "w") as f:
        json.dump(
            Counter(
                summary["exception_type"]
                for summary in result_summaries
                if summary["exception_type"]
            ),
            f,
            indent=4,
            ensure_ascii=False,
        )
