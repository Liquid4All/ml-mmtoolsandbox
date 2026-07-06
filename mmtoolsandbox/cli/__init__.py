# Copyright © 2026 Apple Inc.

"""Run all scenarios in the mmtoolsandbox."""

from __future__ import annotations

import argparse
import datetime
import json
import random
from logging import DEBUG, ERROR, WARNING, getLogger
from pathlib import Path
from typing import (
    Any,
    Type,
    cast,
)

from mmtoolsandbox.cli.utils import (
    TEST_SCENARIO_NAMES,
    resolve_dataset,
)
from mmtoolsandbox.common.execution import run_dataset
from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.i18n import Locale
from mmtoolsandbox.datasets.names import DatasetName
from mmtoolsandbox.roles.base_judge import _JUDGE_REGISTRY
from mmtoolsandbox.roles.base_role import BaseRole
from mmtoolsandbox.roles.code_execution_agent import CodeExecutionAgent
from mmtoolsandbox.roles.openai_agent import (
    GPT_5_4_2026_03_05_Agent,
)
from mmtoolsandbox.roles.openai_judge import GPT_5_4_2026_03_05_Judge
from mmtoolsandbox.roles.openai_user import GPT_5_4_2026_03_05_User
from mmtoolsandbox.roles.registry import (
    get_role_class_by_name,
    get_role_classes_by_role_type,
)
from mmtoolsandbox.roles.role_config import ConfigMixin, RoleConfig

DEFAULT_AGENT_TYPE = GPT_5_4_2026_03_05_Agent
DEFAULT_USER_TYPE = GPT_5_4_2026_03_05_User
DEFAULT_JUDGE_TYPE = GPT_5_4_2026_03_05_Judge

LOGGER = getLogger(__name__)


def load_json_str(json_str: str) -> dict[str, Any]:
    """Helper to parse a JSON string.

    Can be used as `type` in the argparser to perform JSON validation when parsing the
    arguments.
    """
    return cast(dict[str, Any], json.loads(json_str))


def main() -> None:
    random.seed(42)
    parser = argparse.ArgumentParser(description=__doc__)

    agent_role_types = get_role_classes_by_role_type(RoleType.AGENT)
    user_role_types = get_role_classes_by_role_type(RoleType.USER)
    all_role_types = list(agent_role_types.values()) + list(user_role_types.values())

    parser.add_argument(
        "--agent",
        help="Agent type.",
        default=DEFAULT_AGENT_TYPE.__name__,
        choices=agent_role_types.keys(),
    )
    parser.add_argument(
        "--user",
        help="User type.",
        default=DEFAULT_USER_TYPE.__name__,
        choices=user_role_types.keys(),
    )
    parser.add_argument(
        "--judge",
        help="Judge model for evaluation.",
        default=DEFAULT_JUDGE_TYPE.__name__,
        choices=_JUDGE_REGISTRY.keys(),
    )
    parser.add_argument(
        "-d",
        "--dataset",
        help="Dataset name.",
        type=str,
        default=str(DatasetName.FULL),
        choices=[str(n) for n in DatasetName],
    )
    parser.add_argument(
        "--dataset-config",
        type=load_json_str,
        default="{}",
        help="JSON config for the dataset.",
    )
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        help="Path to a directory of scenario JSON files (for APPWORLD dataset).",
    )
    parser.add_argument(
        "--image-base-path",
        type=Path,
        help="Base directory used to resolve relative `image_paths` entries in "
        "scenario JSONs. Absolute paths are passed through unchanged.",
    )
    parser.add_argument(
        "-l",
        "--locale",
        help="Locale for scenario evaluation.",
        choices=[locale.name for locale in Locale],
        default="en_US",
    )
    scenario_selection_group = parser.add_mutually_exclusive_group()
    scenario_selection_group.add_argument(
        "-t",
        "--test_mode",
        action="store_true",
        help="Only run a few scenarios rather than the full suite.",
    )
    scenario_selection_group.add_argument(
        "-s",
        "--scenarios",
        nargs="*",
        help=(
            "Optional. Can be used to process the given subset of scenarios of the "
            "chosen dataset."
        ),
        required=False,
    )
    scenario_selection_group.add_argument(
        "--tag-expression",
        help="Optional. Filter scenarios using boolean algebra with tag. Ex: (tag|tag2)&~tag3",
        required=False,
    )
    parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        default=16,
        help="Max number of processes for running scenarios in parallel.",
    )
    parser.add_argument(
        "-r",
        "--repeats",
        type=int,
        default=1,
        help="Number of times to repeat each scenario.",
    )
    output_dir_group = parser.add_mutually_exclusive_group()
    output_dir_group.add_argument(
        "-o",
        "--output-base-dir",
        type=Path,
        default=Path("data"),
        help=(
            "The output base directory under which to create a directory to store "
            "the results"
        ),
    )
    output_dir_group.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="The output directory under which to store the results.",
    )
    parser.add_argument(
        "--fail-on-error",
        help="Whether to stop and fail if an error occurs in the scenario.",
        action="store_true",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Rerun only scenarios that failed due to API errors (429, timeouts, etc.) "
        "in a previous run. Requires --output-dir pointing to an existing experiment.",
    )
    parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="Skip scenarios that already have trajectories in the output directory. "
        "Useful for resuming a crashed or interrupted run.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Turn on debug logging.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Start a web server to visualize conversations as they are generated.",
    )
    parser.add_argument(
        "--enable-tool-search",
        action="store_true",
        help="Enable dynamic tool discovery via api_docs.",
    )
    parser.add_argument(
        "--enable-coding-tool",
        action="store_true",
        help="Enable the execute_code tool for Python execution. Best work together with enable-tool-search.",
    )
    parser.add_argument(
        "--code-execution-mode",
        action="store_true",
        help="Enable pure code execution mode: agent generates code in markdown blocks "
        "(no function calling). All tools are pre-loaded; agent discovers them via "
        "api_docs_search_api_docs. Mutually exclusive with --enable-coding-tool.",
    )
    parser.add_argument(
        "--enable-reasoning",
        action="store_true",
        help="Enable ReACT-style reasoning traces. The agent outputs "
        "<think>...</think> before each tool call or response.",
    )
    parser.add_argument(
        "--enable-ui",
        action="store_true",
        help="Enable interactive UI presentation via UI tools. Agent presents "
        "structured results as rendered UI screens instead of plain text.",
    )
    parser.add_argument(
        "--image-input",
        action="store_true",
        help="Enable multimodal image input. Adds image handling instructions "
        "to the agent system prompt for scenarios involving user-provided images.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="standard",
        choices=["standard", "extended"],
        help="Depth of reasoning traces: 'standard' for brief rationale, "
        "'extended' for reflection, replanning, and self-correction. "
        "Requires --enable-reasoning.",
    )
    # Let role types add their custom cli arguments to the parser
    for role_type in all_role_types:
        if issubclass(role_type, ConfigMixin):
            role_type.extend_cli_parser(parser)

    args = parser.parse_args()

    if args.code_execution_mode and args.enable_coding_tool:
        parser.error(
            "--code-execution-mode and --enable-coding-tool are mutually exclusive."
        )

    if args.enable_coding_tool and not args.enable_tool_search:
        parser.error(
            "--enable-coding-tool requires --enable-tool-search. "
            "The coding tool needs tool discovery to be useful."
        )

    if args.reasoning_effort != "standard" and not args.enable_reasoning:
        parser.error("--reasoning-effort requires --enable-reasoning to be set.")

    if args.verbose:
        getLogger("mmtoolsandbox").setLevel(DEBUG)

    # Silence noisy loggers
    getLogger("httpx").setLevel(ERROR)
    getLogger("httpcore").setLevel(ERROR)
    getLogger("openai").setLevel(ERROR)
    getLogger("mmtoolsandbox.common.image_utils").setLevel(WARNING)

    agent_type = get_role_class_by_name(args.agent)
    user_type = get_role_class_by_name(args.user)

    # Validate agent type matches execution mode
    if args.enable_reasoning and getattr(agent_type, "is_native_thinking_model", False):
        parser.error(
            f"{agent_type.__name__} is a natively thinking model — reasoning "
            "traces are captured automatically from the model's chain-of-thought "
            "output and do not require --enable-reasoning.  Using it would inject "
            "a redundant prompt instruction and risk duplicating the reasoning trace."
        )
    if args.code_execution_mode:
        if not issubclass(agent_type, CodeExecutionAgent):
            parser.error(
                f"--code-execution-mode requires a CodeExecutionAgent subclass, "
                f"but got {agent_type.__name__}. Use a *_CodeExec_Agent variant."
            )
    else:
        if issubclass(agent_type, CodeExecutionAgent):
            parser.error(
                f"{agent_type.__name__} is a CodeExecutionAgent and requires "
                f"--code-execution-mode to be set."
            )

    # Let role types validate the args first to allow checking also that there are no
    # agent-specific args given when the agent is not active.
    for role_type in all_role_types:
        if issubclass(role_type, ConfigMixin):
            role_type.validate_cli_args(args, agent_type, user_type)

    # Then create a config
    agent_config = _maybe_make_config(agent_type, args)
    user_config = _maybe_make_config(user_type, args)

    # Resolve the output directory.
    if args.output_dir is None:
        agent_desc = getattr(agent_type, "model_name", agent_type.__name__)
        user_desc = getattr(user_type, "model_name", user_type.__name__)
        time_str = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        output_directory = (
            args.output_base_dir
            / f"agent_{agent_desc}_user_{user_desc}_{time_str}_{args.locale}"
        )
    else:
        output_directory = args.output_dir

    # The parser for `--test_mode` and `--scenarios` are in a mutually exclusive group
    # so we can safely ignore the value of `args.scenarios` when `args.test_mode` is
    # true.
    scenario_names = TEST_SCENARIO_NAMES if args.test_mode else args.scenarios

    if args.scenario_dir:
        args.dataset_config["scenario_dir"] = str(args.scenario_dir)

    if args.image_base_path:
        args.dataset_config["image_base_path"] = str(args.image_base_path)

    # Inject CLI flags into dataset_config so scenario factories can read them.
    # This bridges CLI flags (used by transform_scenario_capabilities) with
    # dataset-level factories (which only receive config, not CLI args).
    if args.enable_ui:
        args.dataset_config["enable_ui"] = True
    if args.image_input:
        args.dataset_config["image_input"] = True

    # Add selected scenario names to dataset_config so factories can skip
    # loading heavy per-scenario assets (e.g. images) for unselected files.
    if scenario_names:
        args.dataset_config["scenario_names"] = tuple(sorted(scenario_names))

    dataset_name = DatasetName(args.dataset)  # < explicitly convert to `StrEnum`
    dataset = resolve_dataset(
        dataset_name=dataset_name,
        locale=Locale[args.locale],
        dataset_config=args.dataset_config,
        desired_scenario_names=scenario_names,
        desired_scenario_tag_expression=args.tag_expression,
    )

    # When running a single scenario it makes most sense to directly show errors in the
    # console instead of only in the `result_summary.json`.
    fail_on_error = args.fail_on_error
    if len(dataset.scenario_name_to_def) == 1 and not args.fail_on_error:
        fail_on_error = True
        LOGGER.info(
            "Enabling `--fail-on-error` since a single scenario is being processed."
        )

    # Start visualization server if requested (before running scenarios)
    if args.visualize:
        import threading

        from mmtoolsandbox.viz import start_visualizer_server

        # Start server in a background thread
        server_thread = threading.Thread(
            target=start_visualizer_server,
            args=(output_directory,),
            daemon=True,
        )
        server_thread.start()

    if args.enable_reasoning:
        LOGGER.info(
            "\033[92mReasoning enabled (thinking effort: %s)\033[0m",
            args.reasoning_effort,
        )

    if args.retry_failed:
        if args.output_dir is None:
            parser.error(
                "--retry-failed requires --output-dir pointing to a previous run"
            )
        summary_path = output_directory / "result_summary.json"
        if not summary_path.exists():
            parser.error(
                f"--retry-failed: no result_summary.json found in {output_directory}"
            )

    run_dataset(
        agent_type=agent_type,
        user_type=user_type,
        dataset=dataset,
        processes=args.parallel,
        output_directory=output_directory,
        fail_on_error=fail_on_error,
        agent_config=agent_config,
        user_config=user_config,
        num_scenario_repeats=args.repeats,
        enable_tool_search=args.enable_tool_search,
        enable_coding_tool=args.enable_coding_tool,
        enable_pure_code_exec=args.code_execution_mode,
        enable_reasoning=args.enable_reasoning,
        reasoning_effort=args.reasoning_effort,
        enable_ui=args.enable_ui,
        image_input=args.image_input,
        retry_failed=args.retry_failed,
        continue_run=args.continue_run,
        judge_name=args.judge,
    )
    # Log a message to get a timestamp for when processing has ended.
    LOGGER.info("Done.")

    # Keep the visualization server running after scenarios complete
    if args.visualize:
        # Disable auto-refresh now that scenarios are complete
        from mmtoolsandbox.viz.visualizer import (
            ConversationHTTPHandler,
        )

        ConversationHTTPHandler.auto_refresh = False
        LOGGER.info(
            "Scenarios complete. Auto-refresh disabled. Visualization server is still running. Press Ctrl+C to stop."
        )
        try:
            # Keep main thread alive
            import time

            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            LOGGER.info("Stopping visualization server.")


def _maybe_make_config(
    role_type: Type[BaseRole], args: argparse.Namespace
) -> RoleConfig | None:
    if issubclass(role_type, ConfigMixin):
        return role_type.make_config(args)
    else:
        return None


if __name__ == "__main__":
    main()
