#!/usr/bin/env python3
# Copyright © 2026 Apple Inc.

"""Standalone script to start the visualization server for existing results."""

import argparse
from pathlib import Path

from mmtoolsandbox.viz import start_visualizer_server


def _discover_experiments(parent_directory: Path, marker: str) -> dict[str, Path]:
    """Recursively find experiment directories under ``parent_directory``.

    A directory is considered an experiment if it contains a file named
    ``marker``. The experiment name is the path relative to
    ``parent_directory`` (or the directory's basename if the marker lives
    directly in ``parent_directory``).
    """
    experiments: dict[str, Path] = {}
    for marker_path in sorted(parent_directory.rglob(marker)):
        if not marker_path.is_file():
            continue
        exp_dir = marker_path.parent
        if exp_dir == parent_directory:
            name = parent_directory.name
        else:
            name = str(exp_dir.relative_to(parent_directory))
        experiments[name] = exp_dir
    return experiments


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Start visualization server for MMToolSandbox results"
    )
    parser.add_argument(
        "parent_directory",
        type=Path,
        help=(
            "Parent directory to search recursively for experiment runs. "
            "Each subdirectory containing the result-summary marker file "
            "becomes a selectable experiment."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=None,
        help="Path to directory containing scenario JSON files (e.g. scenario_0001.json)",
    )
    parser.add_argument(
        "--user-exports-dir",
        type=Path,
        default=None,
        help="Path to directory containing user export JSON files (user_001.json, ...)",
    )
    parser.add_argument(
        "--result-summary",
        type=str,
        default="result_summary.json",
        help=(
            "Result summary filename — also used as the marker for "
            "experiment discovery (default: result_summary.json)"
        ),
    )
    parser.add_argument(
        "--auto-refresh",
        action="store_true",
        help="Enable auto-refresh (default: disabled for viewing existing results)",
    )

    args = parser.parse_args()

    parent = args.parent_directory.expanduser()
    if not parent.exists():
        print(f"Error: Directory {parent} does not exist")
        return 1
    if not parent.is_dir():
        print(f"Error: {parent} is not a directory")
        return 1

    experiments = _discover_experiments(parent, args.result_summary)
    if not experiments:
        print(
            f"Error: No experiments found under {parent} "
            f"(looking for '{args.result_summary}')"
        )
        return 1

    if len(experiments) == 1:
        only = next(iter(experiments.values()))
        print(f"Starting visualization server for: {only}")
    else:
        print(f"Starting visualization server with {len(experiments)} experiments:")
        for name, path in experiments.items():
            print(f"  - {name}: {path}")
    print(f"Auto-refresh: {'enabled' if args.auto_refresh else 'disabled'}")

    start_visualizer_server(
        experiments,
        port=args.port,
        auto_refresh=args.auto_refresh,
        scenario_dir=args.scenario_dir,
        user_exports_dir=args.user_exports_dir,
        result_summary_name=args.result_summary,
    )

    return 0


if __name__ == "__main__":
    exit(main())
