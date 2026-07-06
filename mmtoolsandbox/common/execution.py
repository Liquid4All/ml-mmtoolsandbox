# Copyright © 2026 Apple Inc.

"""Orchestration layer for running MMToolSandbox scenarios and datasets.

Provides ``run_dataset()`` as the main entry point invoked by the CLI, which
parallelizes scenario execution across multiple processes.  ``run_scenario()``
plays a single scenario end-to-end (role instantiation, capability transforms,
evaluation, and result collection).  ``transform_scenario_capabilities()``
rewrites a scenario's system prompt and tool configuration to enable tool
search, code execution, reasoning, and UI modes.  Result aggregation helpers
produce per-category summaries and metrics files.
"""

from __future__ import annotations

import contextlib
import copy
import json
import multiprocessing
import random
import traceback
from collections import Counter, defaultdict
from functools import partial
from logging import getLogger
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Type,
)

import polars as pl
from tenacity import RetryError
from tqdm import tqdm

from mmtoolsandbox.common.execution_context import (
    DatabaseNamespace,
    RoleType,
    ScenarioCategories,
)
from mmtoolsandbox.common.results import (
    write_aggregated_metrics_json,
    write_exceptions_summary,
    write_result_summary,
)
from mmtoolsandbox.common.scenario import Scenario
from mmtoolsandbox.datasets.dataset import Dataset
from mmtoolsandbox.roles.base_role import BaseRole
from mmtoolsandbox.roles.cli_role import CliRole
from mmtoolsandbox.roles.execution_environment import ExecutionEnvironment
from mmtoolsandbox.roles.role_config import ConfigMixin, RoleConfig

LOGGER = getLogger(__name__)

# Note that we use `/` to automatically organize the repeated runs in subfolders, e.g.:
# mmtoolsandbox -s wifi_off --output-dir /tmp/wifi_off -r 2
# tree /tmp/wifi_off/trajectories
# /tmp/wifi_off/trajectories
# └── wifi_off
#     ├── run_0
#     │   ├── conversation.json
#     │   ├── execution_context.json
#     │   └── pretty_print.txt
#     └── run_1
#         ├── conversation.json
#         ├── execution_context.json
#         └── pretty_print.txt
SCENARIO_REPEAT_SUFFIX = "/run_"


# Exception patterns that indicate transient API/infrastructure errors worth
# retrying.  Checked against both ``exception_type`` and ``traceback`` fields
# in the result summary.  Agent-logic errors (ValueError, KeyError, …) are
# intentionally absent — those would fail identically on retry.
_RETRYABLE_EXCEPTION_PATTERNS = {
    "RateLimitError",
    "APIError",
    "APIStatusError",
    "APIConnectionError",
    "AuthenticationError",
    "VertexAINonRetryableError",
    "VertexAIRateLimitError",
    "Timeout",
    "TimeoutError",
    "ConnectionError",
    "RESOURCE_EXHAUSTED",
    "ServiceUnavailable",
    "BadGateway",
    "401",
    "429",
    "500",
    "502",
    "503",
}


def _is_retryable_api_error(result: dict[str, Any]) -> bool:
    """Return True if *result* failed due to a transient API/infra error."""
    exc_type = result.get("exception_type") or ""
    tb = result.get("traceback") or ""
    return any(p in exc_type or p in tb for p in _RETRYABLE_EXCEPTION_PATTERNS)


# ---------------------------------------------------------------------------
# Prompt construction is centralized in prompt_templates.py.  We import
# the composer and the auth/reasoning/code constants that
# transform_scenario_capabilities() uses for find-and-replace transforms.
# ---------------------------------------------------------------------------
from mmtoolsandbox.common.prompt_templates import (  # noqa: E402
    RULE_LOGIN_REQUIRED,
    RULE_LOGIN_SKIP,
    SECTION_AUTH_AUTO_LOGIN,
    SECTION_AUTH_MANUAL,
    SECTION_CODE_ENVIRONMENT,
    SECTION_EXTENDED_REASONING,
    SECTION_REASONING,
    UI_AGENT_TOOLS,
    UI_USER_TOOLS,
    compose_agent_prompt,
)


def transform_scenario_capabilities(
    scenario: Scenario,
    enable_tool_search: bool = False,
    enable_coding_tool: bool = False,
    enable_pure_code_exec: bool = False,
    enable_reasoning: bool = False,
    reasoning_effort: str = "standard",
    exposed_tool_allow_list: list[str] | None = None,
    enable_ui: bool = False,
    image_input: bool = False,
) -> Scenario:
    """Transform a scenario to enable specific capabilities.

    Args:
        scenario: The scenario to transform
        enable_tool_search: If True, expose discovery tools.
        enable_coding_tool: If True, expose execute_code and pre-load tools into REPL.
        enable_pure_code_exec: If True, pre-load tools into REPL for pure code
            execution mode (no execute_code wrapper). Implies enable_tool_search.
        enable_reasoning: If True, append ReACT-style reasoning instructions to the
            system prompt so the agent emits ``<think>`` blocks before each action.
        reasoning_effort: Depth of reasoning — ``"standard"`` for brief rationale,
            ``"extended"`` for reflection, replanning, and self-correction.
        exposed_tool_allow_list: Optional list of tools to expose to the agent.
                                 If None, defaults to scenario.tool_allow_list.

    Returns:
        A transformed copy of the scenario
    """
    if (
        not enable_tool_search
        and not enable_coding_tool
        and not enable_pure_code_exec
        and not enable_reasoning
        and not enable_ui
    ):
        return scenario

    # enable_pure_code_exec implies tool search (agent discovers tools via api search)
    if enable_pure_code_exec:
        enable_tool_search = True

    scenario = copy.deepcopy(scenario)
    ctx = scenario.starting_context

    if enable_pure_code_exec:
        # Pure code exec: ALL tools are available and pre-loaded into the REPL.
        # The agent doesn't see tool schemas — it discovers them dynamically
        # via api_docs_search_api_docs().  We keep tool_allow_list=None so
        # nothing is filtered out.
        ctx.tool_allow_list = None
        ctx.pure_code_exec = True
        if ctx.role_tool_allow_list is not None:
            ctx.role_tool_allow_list.pop(RoleType.AGENT, None)
    else:
        added_tools = []
        if enable_tool_search:
            added_tools.append("api_docs_search_api_docs")
        if enable_coding_tool:
            added_tools.append("execute_code")
        if enable_ui:
            added_tools.extend(UI_AGENT_TOOLS)

        # Pin scenario-defined tools + UI tools so they're never evicted
        # by the LRU cache when the agent discovers many AppWorld tools.
        if ctx.tool_allow_list is not None:
            ctx.pinned_tools.update(ctx.tool_allow_list)
        ctx.pinned_tools.update(added_tools)

        if enable_tool_search:
            # Tool search (with or without coding tool): the agent only
            # sees discovery/coding tools.  Scenario tools are registered
            # dynamically by register_tools() when the agent discovers
            # them via api_docs_search_api_docs.
            #
            # Ensure tool_allow_list is a concrete list (not None) so that
            # register_tools() can dynamically append discovered tools.
            # We seed it with the discovery/coding tools; original tools
            # for other roles (USER) are kept if already present.
            if ctx.tool_allow_list is None:
                ctx.tool_allow_list = list(added_tools)
            else:
                for t in added_tools:
                    if t not in ctx.tool_allow_list:
                        ctx.tool_allow_list.append(t)
            if ctx.role_tool_allow_list is None:
                ctx.role_tool_allow_list = {}
            ctx.role_tool_allow_list[RoleType.AGENT] = list(added_tools)
        else:
            # No tool search: use the scenario's original allow list
            if exposed_tool_allow_list is not None:
                base_allow_list = exposed_tool_allow_list
            else:
                base_allow_list = ctx.tool_allow_list if ctx.tool_allow_list else []
            ctx.tool_allow_list = list(set(base_allow_list + added_tools))
            if ctx.role_tool_allow_list is None:
                ctx.role_tool_allow_list = {}
            ctx.role_tool_allow_list[RoleType.AGENT] = ctx.tool_allow_list

    # Inject UI tools, user interaction tools, and judge rubric.
    #   Agent: UI_AGENT_TOOLS added to tool_allow_list
    #   User:  ui_user_interact added to role_tool_allow_list[USER] + global list
    #   Judge: evaluation_criteria.enable_ui = True (activates UI quality rubric)
    if enable_ui:
        if ctx.role_tool_allow_list is None:
            ctx.role_tool_allow_list = {}
        user_tools = list(ctx.role_tool_allow_list.get(RoleType.USER, []))
        for tool_name in UI_USER_TOOLS:
            if tool_name not in user_tools:
                user_tools.append(tool_name)
            if ctx.tool_allow_list is not None and tool_name not in ctx.tool_allow_list:
                ctx.tool_allow_list.append(tool_name)
        ctx.role_tool_allow_list[RoleType.USER] = user_tools
        if hasattr(scenario, "evaluation_criteria"):
            scenario.evaluation_criteria.enable_ui = True

    # Ensure end_conversation is always in the global tool_allow_list
    # so the user simulator can call it regardless of execution mode.
    # This is safe for the agent role because end_conversation is registered
    # with visible_to=(RoleType.USER,), so get_available_tools_for_role(AGENT)
    # filters it out. Additionally, in tool-search mode the agent's
    # role_tool_allow_list only contains discovery/coding tools, providing
    # a second layer of exclusion.
    if (
        ctx.tool_allow_list is not None
        and "end_conversation" not in ctx.tool_allow_list
    ):
        ctx.tool_allow_list.append("end_conversation")

    sandbox_db = ctx._dbs[DatabaseNamespace.SANDBOX]

    # ---------------------------------------------------------------------------
    # Identify the base system prompt.
    #
    # The FIRST SYSTEM→AGENT row is the base scenario's default system prompt.
    # The execution framework may overwrite or append to it (e.g., code-execution
    # instructions, reasoning instructions, coding guidelines).
    #
    # Subsequent SYSTEM→AGENT rows (added by ScenarioExtensions) are scenario-
    # specific instructions and MUST be preserved unchanged.
    # ---------------------------------------------------------------------------
    sys_agent_mask = (sandbox_db["sender"] == RoleType.SYSTEM) & (
        sandbox_db["recipient"] == RoleType.AGENT
    )
    first_idx = sys_agent_mask.arg_true()
    base_prompt_mask = (
        pl.Series([i == first_idx[0] for i in range(sandbox_db.height)])
        if len(first_idx) > 0
        else pl.Series([False] * sandbox_db.height)
    )

    # Read scenario context flags from runtime_metadata.
    _runtime_meta = getattr(scenario, "runtime_metadata", None)
    auto_login = _runtime_meta.get("auto_login", False) if _runtime_meta else False

    # 1. Overwrite the base system prompt based on execution mode.
    if enable_pure_code_exec:
        system_prompt: str | None = compose_agent_prompt(
            pure_code_exec=True,
            auto_login=auto_login,
            support_images=image_input,
            enable_ui=enable_ui,
            enable_reasoning=reasoning_effort if enable_reasoning else None,
        )
    elif enable_tool_search and enable_coding_tool:
        system_prompt = compose_agent_prompt(
            enable_tool_search=True,
            enable_coding_tool=True,
            auto_login=auto_login,
            support_images=image_input,
            enable_ui=enable_ui,
        )
    elif enable_tool_search:
        system_prompt = compose_agent_prompt(
            enable_tool_search=True,
            auto_login=auto_login,
            support_images=image_input,
            enable_ui=enable_ui,
        )
    elif enable_ui:
        system_prompt = compose_agent_prompt(
            auto_login=auto_login,
            support_images=image_input,
            enable_ui=True,
        )
    else:
        system_prompt = None

    if system_prompt:
        sandbox_db = sandbox_db.with_columns(
            pl.when(base_prompt_mask)
            .then(pl.lit(system_prompt))
            .otherwise(pl.col("content"))
            .alias("content")
        )

    # 1b. If auto_login is enabled and the prompt was NOT already composed
    #     with auto_login (i.e., system_prompt is None), do find-and-replace
    #     on the existing base prompt.
    if auto_login and system_prompt is None:
        sandbox_db = sandbox_db.with_columns(
            pl.when(base_prompt_mask)
            .then(
                pl.col("content")
                .str.replace(SECTION_AUTH_MANUAL, SECTION_AUTH_AUTO_LOGIN, literal=True)
                .str.replace(RULE_LOGIN_REQUIRED, RULE_LOGIN_SKIP, literal=True)
            )
            .otherwise(pl.col("content"))
            .alias("content")
        )

    # 2. Pre-load tools into REPL
    #    pure_code_exec: None = all tools (agent discovers dynamically)
    #    tool_search:    only discovery/coding tools (rest added by register_tools)
    #    default:        scenario's allow list
    if enable_pure_code_exec:
        repl_allow_list = None
    elif enable_tool_search:
        repl_allow_list = ctx.role_tool_allow_list[RoleType.AGENT]
    else:
        repl_allow_list = ctx.tool_allow_list

    import_statement = ctx.toolbox.create_import_statement(
        tool_allow_list=repl_allow_list,
        visible_to_role=[RoleType.AGENT],
    )
    sandbox_db = sandbox_db.with_columns(
        pl.when(
            (pl.col("sender") == RoleType.SYSTEM)
            & (pl.col("recipient") == RoleType.EXECUTION_ENVIRONMENT)
        )
        .then(pl.lit(import_statement))
        .otherwise(pl.col("content"))
        .alias("content")
    )

    # 3. Append optional augmentations to the base system prompt.
    # For pure code-exec, reasoning is handled inside the template.
    if enable_reasoning and not enable_pure_code_exec:
        instruction = (
            SECTION_EXTENDED_REASONING
            if reasoning_effort == "extended"
            else SECTION_REASONING
        )
        sandbox_db = sandbox_db.with_columns(
            pl.when(base_prompt_mask)
            .then(pl.col("content") + pl.lit(instruction))
            .otherwise(pl.col("content"))
            .alias("content")
        )

    if enable_coding_tool and not enable_pure_code_exec:
        sandbox_db = sandbox_db.with_columns(
            pl.when(base_prompt_mask)
            .then(pl.col("content") + pl.lit(SECTION_CODE_ENVIRONMENT))
            .otherwise(pl.col("content"))
            .alias("content")
        )

    ctx._dbs[DatabaseNamespace.SANDBOX] = sandbox_db

    return scenario


def is_human_role(role_type: Type[BaseRole]) -> bool:
    """Return whether the given role type represents a human (CLI) player.

    Args:
        role_type: A ``BaseRole`` subclass to check.

    Returns:
        True if ``role_type`` is a subclass of ``CliRole``.
    """
    return issubclass(role_type, CliRole)


def run_scenario(
    name_and_scenario: tuple[str, Scenario],
    *,
    dataset_name: str,
    agent_type: type[BaseRole],
    user_type: type[BaseRole],
    output_directory: Path,
    agent_config: RoleConfig | None,
    user_config: RoleConfig | None,
    fail_on_error: bool,
    enable_tool_search: bool = False,
    enable_coding_tool: bool = False,
    enable_pure_code_exec: bool = False,
    enable_reasoning: bool = False,
    reasoning_effort: str = "standard",
    enable_ui: bool = False,
    image_input: bool = False,
    judge_name: str | None = None,
) -> dict[str, Any]:
    """Play and evaluate a scenario.

    This is a necessary utility function to make multiprocessing work.

    Args:
        name_and_scenario: Scenario name and Scenario object.
        dataset_name: The name of the dataset to which the scenario belongs.
        agent_type: Agent type.
        user_type: User type.
        output_directory: Directory to write output into.
        agent_config: Agent-specific config.
        user_config: User-specific config.
        fail_on_error: Whether to stop and fail if an error occurs.
        enable_tool_search: If True, expose discovery tools.
        enable_coding_tool: If True, expose execute_code and pre-load tools.
        enable_pure_code_exec: If True, pre-load tools for pure code exec mode.
        enable_reasoning: If True, enable ReACT-style reasoning traces.
        reasoning_effort: Depth of reasoning — "standard" or "extended".
        enable_ui: If True, enable interactive UI presentation.
        image_input: If True, enable multimodal image input.
        judge_name: Judge class name from ``_JUDGE_REGISTRY``. If None, uses
            the default judge.

    Returns:
        A dictionary containing evaluation results (similarity, turn_count,
        judge_result, ui_judge_result, user_judge_result, entity_diff_result).
    """
    name, scenario = name_and_scenario
    roles = {}

    # Force-import sentence_transformers early to avoid circular import
    # errors when it's lazily imported inside embedding_filter_dataframe.
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        pass

    # Optional per-scenario runtime context (e.g. AppWorld tool registration,
    # bridge lifecycle, time freeze).  Wraps the entire run including
    # transform_scenario_capabilities so that tools registered in the worker
    # are available for import generation.
    if scenario.runtime_context_factory is not None:
        runtime_cm = scenario.runtime_context_factory(scenario, output_directory)
    else:
        runtime_cm = contextlib.nullcontext()

    try:
        with runtime_cm:
            # Transform scenario capabilities if enabled
            scenario = transform_scenario_capabilities(
                scenario,
                enable_tool_search=enable_tool_search,
                enable_coding_tool=enable_coding_tool,
                enable_pure_code_exec=enable_pure_code_exec,
                enable_reasoning=enable_reasoning,
                reasoning_effort=reasoning_effort,
                enable_ui=enable_ui,
                image_input=image_input,
            )

            num_tools = len(
                scenario.starting_context.get_available_tools_for_role(RoleType.AGENT)
            )
            LOGGER.info("Scenario '%s': %d tools available for agent", name, num_tools)

            agent = _load_role(agent_type, agent_config)
            user = _load_role(user_type, user_config)

            # In dynamic tool discovery modes, the agent may call tools that
            # aren't yet in the allow list (they get registered on the fly).
            # Disable the hard error so unknown tool calls are handled
            # gracefully instead of crashing the scenario.
            if enable_tool_search or enable_coding_tool or enable_pure_code_exec:
                if hasattr(agent, "raise_error_on_unknown_tools"):
                    agent.raise_error_on_unknown_tools = False

            roles = {
                RoleType.USER: user,
                RoleType.EXECUTION_ENVIRONMENT: ExecutionEnvironment(),
                RoleType.AGENT: agent,
            }
            output_directory.mkdir(parents=True, exist_ok=True)

            # Configure the safety guard to allow writes to the output directory.
            # This must be done before play_and_evaluate, which deep-copies the
            # starting_context — so the write path config is inherited by the copy.
            scenario.starting_context.set_safety_guard_output_directory(
                output_directory
            )

            result, _, _ = scenario.play_and_evaluate(
                roles=roles,
                output_directory=output_directory,
                scenario_name=name,
                dataset_name=dataset_name,
                judge_name=judge_name,
            )
        return {
            "name": name,
            "categories": scenario.categories,
            "traceback": None,
            "exception_type": None,
            "similarity": result.evaluation_result.similarity,
            "turn_count": result.evaluation_result.turn_count,
            "judge_result": result.evaluation_result.judge_result,
            "task_completion_criteria": result.evaluation_result.task_completion_criteria,
            "entity_diff_result": result.evaluation_result.entity_diff_result,
            "ui_judge_result": result.evaluation_result.ui_judge_result,
            "user_judge_result": result.evaluation_result.user_judge_result,
            "token_usage": result.token_usage,
        }
    except Exception as e:
        if fail_on_error:
            raise

        LOGGER.error("Scenario '%s' failed: %s\n%s", name, e, traceback.format_exc())

        if isinstance(e, RetryError):
            original_exception = e.last_attempt.exception()
            exception_type = f"RetryError[{type(original_exception).__name__}]"
        else:
            exception_type = type(e).__name__

        return {
            "name": name,
            "categories": scenario.categories,
            "traceback": traceback.format_exc(),
            "exception_type": exception_type,
            "similarity": 0,
            "turn_count": scenario.max_messages,
            "judge_result": None,
            "task_completion_criteria": None,
            "entity_diff_result": None,
            "ui_judge_result": None,
            "user_judge_result": None,
            "token_usage": {},
        }
    finally:
        for role in roles.values():
            role.teardown()


# ---------------------------------------------------------------------------
# Helpers for --continue mode
# ---------------------------------------------------------------------------


def _find_completed_trajectories(output_directory: Path) -> set[str]:
    """Scan output directory for scenarios that already have a conversation.json.

    Handles both direct layout (``trajectories/<name>/conversation.json``)
    and the run-indexed layout (``trajectories/<name>/run_N/conversation.json``)
    produced by the main CLI. For run-indexed layouts, the returned name
    includes the ``/run_N`` suffix so it matches the keys stored in
    ``result_summary.json`` (as produced by ``maybe_repeat_scenarios``).

    Args:
        output_directory: Root output directory containing ``trajectories/``.

    Returns:
        Set of scenario names with completed trajectories.
    """
    trajectories_dir = output_directory / "trajectories"
    if not trajectories_dir.exists():
        return set()

    completed: set[str] = set()
    for scenario_dir in trajectories_dir.iterdir():
        if not scenario_dir.is_dir():
            continue
        if (scenario_dir / "conversation.json").exists():
            completed.add(scenario_dir.name)
            continue
        for run_dir in scenario_dir.iterdir():
            if (
                run_dir.is_dir()
                and run_dir.name.startswith("run_")
                and (run_dir / "conversation.json").exists()
            ):
                completed.add(f"{scenario_dir.name}/{run_dir.name}")
    return completed


def _load_prior_results_by_name(
    output_directory: Path,
) -> dict[str, dict[str, Any]]:
    """Load prior result_summary.json and index by scenario name.

    Args:
        output_directory: Root output directory.

    Returns:
        Mapping of scenario name to its result dict, or empty dict if
        no prior summary exists.
    """
    summary_path = output_directory / "result_summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, encoding="utf-8") as f:
        results = json.load(f).get("per_scenario_results", [])
    return {r["name"]: r for r in results}


def _make_stub_result(
    name: str, output_directory: Path | None = None
) -> dict[str, Any]:
    """Create a result entry for a completed trajectory.

    When ``output_directory`` is provided, reads actual per-scenario data
    from disk (conversation, entity diff, judge evidence, token usage).
    Falls back to zeroed defaults for any file that is missing or
    unreadable.

    Args:
        name: Scenario name (may include ``/run_N`` suffix).
        output_directory: Root output directory containing ``trajectories/``.

    Returns:
        A result dict with fields matching ``run_scenario()`` output.
    """
    result: dict[str, Any] = {
        "name": name,
        "categories": [],
        "similarity": 0,
        "turn_count": 0,
        "task_completion_criteria": None,
        "judge_result": None,
        "user_judge_result": None,
        "entity_diff_result": None,
        "exception_type": None,
        "traceback": None,
    }

    if output_directory is None:
        return result

    if "/run_" in name:
        base_name, run_suffix = name.split("/run_", 1)
        run_dir = output_directory / "trajectories" / base_name / f"run_{run_suffix}"
    else:
        base_name = name
        run_dir = output_directory / "trajectories" / base_name / "run_0"
    if not run_dir.exists():
        run_dir = output_directory / "trajectories" / base_name
    if not run_dir.exists():
        return result

    def _load_json(path: Path) -> Any:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    conv_path = run_dir / "conversation.json"
    if conv_path.exists():
        conv = _load_json(conv_path)
        msgs = conv if isinstance(conv, list) else conv.get("messages", [])
        result["turn_count"] = len(msgs)

    ed_path = run_dir / "entity_diff_evidence.json"
    if ed_path.exists():
        ed = _load_json(ed_path)
        ed_result = ed.get("result", {})
        result["entity_diff_result"] = ed_result
        p = ed_result.get("overall_precision")
        r = ed_result.get("overall_recall")
        guardrail_pass = ed_result.get("guardrail_pass", True)
        if p is not None and r is not None and (p + r) > 0 and guardrail_pass:
            result["similarity"] = 2 * p * r / (p + r)

    # Prefer parsed judge results if available; fall back to raw evidence
    jr_path = run_dir / "judge_evidence" / "judge_results.json"
    if jr_path.exists():
        jr = _load_json(jr_path)
        result["judge_result"] = jr.get("judge_result")
        result["user_judge_result"] = jr.get("user_judge_result")
        result["ui_judge_result"] = jr.get("ui_judge_result")
    else:
        for key, filename in [
            ("judge_result", "main.json"),
            ("user_judge_result", "user.json"),
            ("ui_judge_result", "ui.json"),
        ]:
            path = run_dir / "judge_evidence" / filename
            if path.exists():
                result[key] = _load_json(path)

    tu_path = run_dir / "token_usage.json"
    if tu_path.exists():
        result["token_usage"] = _load_json(tu_path)

    return result


def run_dataset(
    *,
    agent_type: type[BaseRole],
    user_type: type[BaseRole],
    dataset: Dataset,
    processes: int,
    output_directory: Path,
    fail_on_error: bool,
    agent_config: RoleConfig | None,
    user_config: RoleConfig | None,
    num_scenario_repeats: int = 1,
    enable_tool_search: bool = False,
    enable_coding_tool: bool = False,
    enable_pure_code_exec: bool = False,
    enable_reasoning: bool = False,
    reasoning_effort: str = "standard",
    enable_ui: bool = False,
    image_input: bool = False,
    retry_failed: bool = False,
    continue_run: bool = False,
    judge_name: str | None = None,
) -> None:
    """Entry point for MMToolSandbox scenario execution.

    Args:
        agent_type: The agent type to use.
        user_type: The user type to use.
        dataset: The dataset defining the scenarios to run.
        processes: Number of processes to run in parallel.
        output_directory: The directory for model outputs.
        fail_on_error: Whether to stop and fail if an error occurs.
        agent_config: Configuration of the agent.
        user_config: Configuration of the user.
        num_scenario_repeats: The number of times each scenario should be
            repeated.
        enable_tool_search: If True, expose discovery tools.
        enable_coding_tool: If True, expose execute_code and pre-load tools.
        enable_pure_code_exec: If True, pre-load tools for pure code exec mode.
        enable_reasoning: If True, enable ReACT-style reasoning traces.
        reasoning_effort: Depth of reasoning — "standard" or "extended".
        enable_ui: If True, enable interactive UI presentation.
        image_input: If True, enable multimodal image input.
        retry_failed: If True, only rerun scenarios that failed due to
            transient API errors in a previous run.
        judge_name: Judge class name from ``_JUDGE_REGISTRY``. If None, uses
            the default judge.
    """
    # Show all rows and all columns when converting polars dataframes to strings.
    # Sadly, there is no way to specify an unlimited format length for strings. Note
    # that for tracebacks or long explanations from Claude 3 Opus a value of `1000` was
    # insufficient.
    pl.Config.set_tbl_rows(-1).set_tbl_cols(-1).set_fmt_str_lengths(10000)
    pl.Config.set_tbl_formatting("ASCII_FULL")

    LOGGER.info(
        f"Using dataset '{dataset.name}', version: '{dataset.version}', number of "
        f"scenarios: {len(dataset.scenario_name_to_def)}"
    )
    LOGGER.info("Storing outputs under '%s'.", output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)
    latest_link = output_directory.parent / "latest"
    latest_link.unlink(missing_ok=True)
    latest_link.symlink_to(output_directory, target_is_directory=True)

    # Print a category-wise count before playing scenarios
    category_counter = get_category_to_scenario_count(dataset.scenario_name_to_def)
    LOGGER.info(
        "Number of test cases per category:\n"
        + json.dumps(
            {str(k): v for k, v in category_counter.most_common(len(category_counter))},
            indent=4,
            ensure_ascii=False,
        ),
    )
    # Print a necessary tool-wise count before playing scenarios
    necessary_tool_counter = get_necessary_tool_name_to_scenario_count(
        dataset.scenario_name_to_def
    )
    LOGGER.info(
        "Number of test cases per necessary tool name:\n"
        + json.dumps(
            {
                str(k): v
                for k, v in necessary_tool_counter.most_common(
                    len(necessary_tool_counter)
                )
            },
            indent=4,
            ensure_ascii=False,
        ),
    )
    # Shuffle scenarios for load balancing
    name_and_scenario_list = maybe_repeat_scenarios(
        num_scenario_repeats, dataset.scenario_name_to_def.items()
    )

    # --retry-failed: keep previous successful/non-retryable results and only
    # rerun scenarios that failed due to transient API errors.
    previous_kept_results: list[dict[str, Any]] = []
    if retry_failed:
        prev_summary_path = output_directory / "result_summary.json"
        with open(prev_summary_path, encoding="utf-8") as f:
            prev_results: list[dict[str, Any]] = json.load(f)["per_scenario_results"]
        retryable_names = {
            r["name"] for r in prev_results if _is_retryable_api_error(r)
        }
        previous_kept_results = [
            r for r in prev_results if r["name"] not in retryable_names
        ]
        name_and_scenario_list = [
            (name, s) for name, s in name_and_scenario_list if name in retryable_names
        ]
        LOGGER.info(
            "Retry mode: rerunning %d API-error scenarios (keeping %d others)",
            len(name_and_scenario_list),
            len(previous_kept_results),
        )

    # --continue: skip scenarios whose trajectories already exist on disk.
    # Both sides use the "<name>/run_N" format produced by maybe_repeat_scenarios
    # and returned by _find_completed_trajectories, so we compare directly.
    if continue_run:
        completed_names = _find_completed_trajectories(output_directory)
        if completed_names:
            prior_results_by_name = _load_prior_results_by_name(output_directory)
            for name in completed_names:
                previous_kept_results.append(
                    prior_results_by_name.get(
                        name, _make_stub_result(name, output_directory)
                    )
                )

            name_and_scenario_list = [
                (name, s)
                for name, s in name_and_scenario_list
                if name not in completed_names
            ]
            LOGGER.info(
                "Continue mode: skipping %d completed, running %d remaining",
                len(completed_names),
                len(name_and_scenario_list),
            )

    random.shuffle(name_and_scenario_list)
    num_scenarios = len(name_and_scenario_list)
    if processes > 1 and num_scenarios > 1:
        for role_type in {agent_type, user_type}:
            if is_human_role(role_type):
                raise RuntimeError(
                    f"The '{role_type.__name__}' can only be used when processing a "
                    "single scenario."
                )

        # As described in e.g. https://stackoverflow.com/a/66113051 the default option
        # for starting a process is to fork the parent process, which by design can
        # cause dead locks. Switching to the `spawn` instead of `fork` method
        # for starting a new process eliminated the deadlock.
        mpctx = multiprocessing.get_context("spawn")
        # Use maxtasksperchild=1 when scenarios have runtime context factories
        # (e.g. AppWorld) to guarantee process isolation and prevent freezegun
        # datetime state from leaking between scenarios.
        needs_process_isolation = any(
            s.runtime_context_factory is not None
            for s in dataset.scenario_name_to_def.values()
        )
        pool_kwargs: dict[str, Any] = {}
        if needs_process_isolation:
            pool_kwargs["maxtasksperchild"] = 1
        with mpctx.Pool(min(processes, num_scenarios), **pool_kwargs) as pool:
            result_summaries = []
            run_fn = partial(
                run_scenario,
                dataset_name=dataset.name,
                agent_type=agent_type,
                user_type=user_type,
                output_directory=output_directory,
                agent_config=agent_config,
                user_config=user_config,
                fail_on_error=fail_on_error,
                enable_tool_search=enable_tool_search,
                enable_coding_tool=enable_coding_tool,
                enable_pure_code_exec=enable_pure_code_exec,
                enable_reasoning=enable_reasoning,
                reasoning_effort=reasoning_effort,
                enable_ui=enable_ui,
                image_input=image_input,
                judge_name=judge_name,
            )
            for result in pool.imap_unordered(run_fn, name_and_scenario_list):
                result_summaries.append(result)
                LOGGER.info(
                    "Completed %d/%d scenarios",
                    len(result_summaries),
                    num_scenarios,
                )
    else:
        result_summaries = []
        tqdm_iterator = tqdm(name_and_scenario_list, desc="Scenarios")
        for name_and_scenario in tqdm_iterator:
            result_summaries.append(
                run_scenario(
                    name_and_scenario,
                    dataset_name=dataset.name,
                    agent_type=agent_type,
                    user_type=user_type,
                    output_directory=output_directory,
                    agent_config=agent_config,
                    user_config=user_config,
                    fail_on_error=fail_on_error,
                    enable_tool_search=enable_tool_search,
                    enable_coding_tool=enable_coding_tool,
                    enable_pure_code_exec=enable_pure_code_exec,
                    enable_reasoning=enable_reasoning,
                    reasoning_effort=reasoning_effort,
                    enable_ui=enable_ui,
                    image_input=image_input,
                    judge_name=judge_name,
                )
            )

    # Merge previous kept results (from --retry-failed or --continue) with new results.
    result_summaries = previous_kept_results + result_summaries

    # Aggregate results by category
    category_summary = get_category_summary(result_summaries)
    write_result_summary(
        result_summaries=result_summaries,
        category_summary=category_summary,
        output_directory=output_directory,
        num_scenario_repeats=num_scenario_repeats,
    )

    # Write a `metrics.json` file with aggregated metrics, suitable for
    # uploading to dashboards like wandb.
    write_aggregated_metrics_json(
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        category_summary=category_summary,
        output_directory=output_directory,
    )

    # Write `exceptions.json` file to show exceptions when collecting results
    write_exceptions_summary(
        result_summaries=result_summaries,
        output_directory=output_directory,
    )


def maybe_repeat_scenarios(
    num_scenario_repeats: int, name_and_scenario_list: Iterable[tuple[str, Scenario]]
) -> list[tuple[str, Scenario]]:
    """Create scenarios with repetitions (if enabled).

    Args:
        num_scenario_repeats:  The number of times to repeat each scenario. If the value
                               is `<= 1` then this is a no-op.
        name_and_scenario_list: The list of name and scenario tuples to repeat.

    Returns:
        `name_and_scenario_list` if  `num_scenario_repeats <= 1`, otherwise a new list
        where each scenario is repeated `num_scenario_repeats` times.
    """
    repeated_scenarios = []
    for name, scenario in name_and_scenario_list:
        if name.endswith(SCENARIO_REPEAT_SUFFIX):
            raise ValueError(
                "The scenario name conflicts with the prefix for scenario repetitions. "
                f"Change the scenario name.\n"
                f"Scenario name: '{name}'\n"
                f"Scenario repetition prefix: '{SCENARIO_REPEAT_SUFFIX}'"
            )

        for i in range(num_scenario_repeats):
            repeated_name = f"{name}{SCENARIO_REPEAT_SUFFIX}{i}"
            # We create a copy of the scenario in case the execution mutates it.
            repeated_scenarios.append((repeated_name, copy.deepcopy(scenario)))
    return repeated_scenarios


def get_category_summary(
    result_summary: list[dict[str, Any]],
) -> dict[str, dict[str, list[float]]]:
    """Aggregate per test case result summary into category wise summary.

    Args:
        result_summary:     A list of results for each test case.

    Returns:
        Category wise summary.
    """

    def _aggregate_judge(
        target: dict[str, list[float]],
        judge_result: dict[str, Any],
        prefix: str = "judge",
    ) -> None:
        """Append judge metrics (overall + per-criterion) to target dict."""
        target[f"{prefix}_pass"].append(float(judge_result.get("result", False)))
        for criterion_eval in judge_result.get("criteria_evaluation", []):
            key = f"{prefix}_{criterion_eval['criterion']}"
            target[key].append(float(criterion_eval.get("pass", False)))

    def _aggregate_token_usage(
        target: dict[str, list[float]],
        token_usage: dict[str, Any],
    ) -> None:
        """Append per-role and combined token metrics to target dict."""
        agent_usage = token_usage.get("agent", {})
        user_usage = token_usage.get("user", {})
        for prefix, usage in [("agent", agent_usage), ("user", user_usage)]:
            for metric in ("prompt_tokens", "completion_tokens", "total_tokens"):
                target[f"{prefix}_{metric}"].append(float(usage.get(metric, 0)))
        total_prompt = sum(u.get("prompt_tokens", 0) for u in token_usage.values())
        total_completion = sum(
            u.get("completion_tokens", 0) for u in token_usage.values()
        )
        target["total_prompt_tokens"].append(float(total_prompt))
        target["total_completion_tokens"].append(float(total_completion))
        target["total_tokens"].append(float(total_prompt + total_completion))

    # Aggregate results by category
    category_summary: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for current_summary in result_summary:
        judge_result = current_summary.get("judge_result")
        ui_judge_result = current_summary.get("ui_judge_result")
        user_judge_result = current_summary.get("user_judge_result")
        token_usage = current_summary.get("token_usage", {})
        for category in current_summary["categories"]:
            category_summary[category]["turn_count"].append(
                current_summary["turn_count"]
            )
            if judge_result is not None:
                _aggregate_judge(category_summary[category], judge_result)
            if ui_judge_result is not None:
                _aggregate_judge(
                    category_summary[category], ui_judge_result, prefix="ui_judge"
                )
            if user_judge_result is not None:
                _aggregate_judge(
                    category_summary[category],
                    user_judge_result,
                    prefix="user_judge",
                )
            if token_usage:
                _aggregate_token_usage(category_summary[category], token_usage)
        category_summary["ALL_CATEGORIES"]["turn_count"].append(
            current_summary["turn_count"]
        )
        if judge_result is not None:
            _aggregate_judge(category_summary["ALL_CATEGORIES"], judge_result)
        if ui_judge_result is not None:
            _aggregate_judge(
                category_summary["ALL_CATEGORIES"], ui_judge_result, prefix="ui_judge"
            )
        if user_judge_result is not None:
            _aggregate_judge(
                category_summary["ALL_CATEGORIES"],
                user_judge_result,
                prefix="user_judge",
            )
        if token_usage:
            _aggregate_token_usage(category_summary["ALL_CATEGORIES"], token_usage)
    return category_summary


def get_category_to_scenario_count(
    name_to_scenario: dict[str, Scenario],
) -> Counter[ScenarioCategories | str]:
    """Count number of scenarios based on ScenarioCategories.

    Args:
        name_to_scenario:   A dict with scenario name as keys, scenario objects as values.

    Returns:
        A counter object containing counts for each category.
    """
    category_counter: Counter[ScenarioCategories | str] = Counter()
    for scenario in name_to_scenario.values():
        for category in scenario.categories:
            category_counter[category] += 1
        category_counter["ALL_CATEGORIES"] += 1
    return category_counter


def get_necessary_tool_name_to_scenario_count(
    name_to_scenario: dict[str, Scenario],
) -> Counter[ScenarioCategories | str]:
    """Count number of scenarios based on necessary tool names.

    Args:
        name_to_scenario:   A dict with scenario name as keys, scenario objects as values.

    Returns:
        A counter object containing counts for each necessary tool names.
    """
    tool_name_counter: Counter[ScenarioCategories | str] = Counter()
    # Necessary tool names can be deducted from allowed tools in NO_DISTRACTION_TOOLS category
    # Then the total count equals the count from this category * number of augmentations.
    augmentation_categories: set[ScenarioCategories | str] = set()
    for scenario in name_to_scenario.values():
        if ScenarioCategories.NO_DISTRACTION_TOOLS in scenario.categories:
            assert scenario.starting_context.tool_allow_list is not None
            for necessary_tool in scenario.starting_context.tool_allow_list:
                tool_name_counter[necessary_tool] += 1
        augmentation_categories |= {
            ScenarioCategories.NO_DISTRACTION_TOOLS,
            ScenarioCategories.THREE_DISTRACTION_TOOLS,
            ScenarioCategories.TEN_DISTRACTION_TOOLS,
            ScenarioCategories.ALL_TOOLS_AVAILABLE,
        } & set(scenario.categories)
    for necessary_tool in tool_name_counter:
        tool_name_counter[necessary_tool] *= len(augmentation_categories)
    return tool_name_counter


def _load_role(role_type: type[BaseRole], config: RoleConfig | None) -> BaseRole:
    if issubclass(role_type, ConfigMixin):
        assert config is not None
        return role_type.from_config(config)
    else:
        return role_type()
