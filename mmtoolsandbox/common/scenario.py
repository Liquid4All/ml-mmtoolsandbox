# Copyright © 2026 Apple Inc.

"""Scenario definitions and execution for MMToolSandbox evaluations.

Defines ``Scenario`` (a single evaluation case with starting state and criteria),
``ScenarioExtension`` (a builder that extends a base scenario with messages, tools,
and categories), and ``ScenarioResult`` (the output of a completed run).  The
``Scenario.play()`` method drives the multi-turn actor loop, while
``play_and_evaluate()`` combines play with evaluation and result serialization.
``filter_scenarios_by_tag_expression()`` supports boolean tag queries for scenario
selection.
"""

from __future__ import annotations

import copy
import datetime
import json
import logging
import re
import traceback
from contextlib import AbstractContextManager
from pathlib import Path
from typing import (
    Any,
    Callable,
    cast,
)

import numpy as np
import polars as pl
from attrs import Factory, define
from dateutil.relativedelta import relativedelta
from dateutil.tz import tzoffset
from tqdm import tqdm

from mmtoolsandbox.common.databases import (
    DatabaseNamespace,
)
from mmtoolsandbox.common.evaluation import (
    EvaluationCriteria,
    EvaluationResult,
    evaluate,
)
from mmtoolsandbox.common.execution_context import (
    ExecutionContext,
    RoleType,
    ScenarioCategories,
    get_current_context,
    set_current_context,
)
from mmtoolsandbox.common.image_id import ImageId
from mmtoolsandbox.common.image_utils import load_image_as_base64
from mmtoolsandbox.common.introspection_databases import (
    get_introspection_json_file_name,
    get_introspection_pretty_print_file_name,
)
from mmtoolsandbox.common.message_conversion import (
    add_messages_to_execution_context,
    get_messages_from_execution_context,
    serialize_to_conversation,
    serialize_user_conversation,
)
from mmtoolsandbox.common.relative_time import (
    realign_timestamp,
    resolve_execution_context_relative_time,
)
from mmtoolsandbox.common.tool_conversion import convert_to_openai_tool
from mmtoolsandbox.roles.base_role import BaseRole

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Image delivery enforcement
# ---------------------------------------------------------------------------

_ROUGE_L_THRESHOLD = 0.7

# Lazy-initialized singleton — avoids re-loading the stemmer on every call.
_rouge_scorer_instance = None


def _get_rouge_scorer() -> Any:
    global _rouge_scorer_instance
    if _rouge_scorer_instance is None:
        from rouge_score import rouge_scorer  # type: ignore[import-untyped]

        _rouge_scorer_instance = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _rouge_scorer_instance


def _enforce_image_delivery(
    response_messages: list[Any],
    delivery_map: list[dict[str, Any]],
    delivered: set[int],
) -> None:
    """Auto-attach images to user→agent messages when the LLM forgot.

    For each text-only USER→AGENT message, computes ROUGE-L against each
    round's scripted Query.  If the score exceeds the threshold and those
    images haven't been delivered yet, they are attached to the message.

    Args:
        response_messages: Messages just produced by the user sim.
        delivery_map:      List of ``{"query": str, "image_ids": [int]}``
                           from ``runtime_metadata["image_delivery_map"]``.
        delivered:         Mutable set tracking which image IDs have been sent.
    """
    scorer = _get_rouge_scorer()

    for msg in response_messages:
        if msg.sender != RoleType.USER or msg.recipient != RoleType.AGENT:
            continue

        if msg.image_ids:
            delivered.update(int(img_id) for img_id in msg.image_ids)
            continue

        # User sim sent text-only — check if it matches a round query
        if not msg.content:
            continue

        for entry in delivery_map:
            image_ids = entry["image_ids"]
            if all(img_id in delivered for img_id in image_ids):
                continue  # already delivered

            score = scorer.score(target=entry["query"], prediction=msg.content)[
                "rougeL"
            ].fmeasure
            if score >= _ROUGE_L_THRESHOLD:
                object.__setattr__(
                    msg,
                    "image_ids",
                    [ImageId(img_id) for img_id in image_ids],
                )
                delivered.update(image_ids)
                break  # one match per message


@define
class ScenarioResult:
    """Output of Scenario Play, saving both the execution context after the rollout is collected,
    and evaluation result

    Attributes:
        ending_context: The ``ExecutionContext`` after the scenario rollout.
        evaluation_result: The ``EvaluationResult`` from post-rollout evaluation.
        token_usage: Per-role token usage accumulated during the scenario.
    """

    ending_context: ExecutionContext
    evaluation_result: EvaluationResult
    token_usage: dict[str, Any] = Factory(dict)


@define
class Scenario:
    """Test case scenario defining a single evaluation case.

    Each scenario contains an execution context defining starting state, and an
    evaluation object defining evaluation criteria.

    Attributes:
        starting_context: Initial ``ExecutionContext`` containing the world
            state (databases, toolbox, system prompt).
        evaluation_criteria: Criteria used to evaluate the scenario outcome.
        max_messages: Maximum number of total messages in the rollout.
        categories: List of ``ScenarioCategories`` tags for filtering and
            reporting.
        tags: Free-form string tags for search and filtering.
        reference_time: ISO 8601 reference time for timestamp adjustment.
        image_paths: Optional file paths for images loaded into the IMAGE
            database at scenario start.
        runtime_metadata: Optional metadata dict for external setup (e.g.
            AppWorld bridge init, entity staging, time freeze).
        runtime_context_factory: Optional callable that creates a context
            manager wrapping ``play_and_evaluate()`` for per-scenario
            runtime setup and teardown.
    """

    # Initial context, contains initial world state
    starting_context: ExecutionContext
    # Evaluation criteria
    evaluation_criteria: EvaluationCriteria = Factory(EvaluationCriteria)
    # Max number of total messages in roll out
    max_messages: int = 40
    # Category tags
    categories: list[ScenarioCategories] = Factory(list)
    # Custom tags for search
    tags: set[str] = Factory(set)
    # Reference time
    reference_time: str | None = None
    # Optional image paths for scenarios that include images
    image_paths: list[str] | None = None
    # Optional runtime metadata for scenarios that need external setup
    # (e.g. AppWorld bridge init, entity staging, time freeze).
    runtime_metadata: dict[str, Any] | None = None
    # Optional factory that creates a context manager wrapping play_and_evaluate.
    # Used for per-scenario runtime setup/teardown (e.g. AppWorld bridge lifecycle).
    # Must be a top-level function (not a lambda/closure) to support pickling
    # for multiprocessing.  Signature: (scenario, output_directory) -> ContextManager
    runtime_context_factory: (
        Callable[["Scenario", Path], AbstractContextManager[None]] | None
    ) = None

    def prepare(self, execution_context: ExecutionContext) -> None:
        """Prepare the scenario for running.

        This function is intended to perform work that you do *not* want to happen when
        just loading a dataset (e.g. any type of resource-intensive or slow operation).
        """
        execution_context.execute_delayed_initialization()

    def next_image_id(self) -> ImageId:
        """Get the next available image ID for this scenario.

        This method looks at the current number of images in the starting context
        and returns the next sequential ID that would be assigned.

        Returns:
            ImageId: The next available image ID
        """
        from mmtoolsandbox.common.execution_context import DatabaseNamespace

        # Check if IMAGE database exists
        if DatabaseNamespace.IMAGE not in self.starting_context._dbs:
            return ImageId(0)

        # Count existing images in the starting context
        image_db = self.starting_context.get_database(DatabaseNamespace.IMAGE)
        return ImageId(len(image_db))

    def timestamp_adjustment_hack(self) -> None:
        """Adjust initial database timestamps based on current time at `play` invocation.

        This prevents issues where evaluation took very long, and test scenarios constructed at the start
        of test invocation are no longer valid.
        """
        assert self.reference_time is not None
        reference_datetime = datetime.datetime.fromisoformat(self.reference_time)
        current_datetime = datetime.datetime.now()
        self.starting_context = realign_timestamp(
            execution_context=self.starting_context,
            execution_context_creation_time=reference_datetime,
            new_execution_context_creation_time=current_datetime,
        )
        # Override reference time for consistency.
        self.reference_time = current_datetime.isoformat()

    def convert_relativedelta_to_absolute(self) -> None:
        """Converts relativedelta entries in starting state to absolute time."""
        # Get current time with context timezone. Remove microseconds.
        current_datetime = datetime.datetime.now(
            tz=tzoffset(
                "Dummy Name",
                self.starting_context.get_database(
                    namespace=DatabaseNamespace.SETTING
                ).to_dicts()[0]["utc_offset_seconds"],
            )
        ) + relativedelta(microsecond=0)
        self.starting_context = resolve_execution_context_relative_time(
            execution_context=self.starting_context, reference_time=current_datetime
        )
        # Override reference time for consistency.
        self.reference_time = current_datetime.isoformat()

    def adjust_time(self) -> None:
        """Adjust time concepts in local database to make scenarios valid.

        This is used to prevent cases where, scenario intends to test models ability
        to interact with an event 1 hr in the future, but because
        1. Time concepts are created at init time
        2. Scenario.play took more than 1 hr after init before kickoff
        Scenario was no longer valid by the time it reached Scenario.play

        Different datasets have different methods for solving this issue.

        Some datasets store a reference time alongside absolute event times;
        relative relationships are recalculated at ``Scenario.play`` and
        readjusted to the new reference time.

        Other datasets store event times as ``relativedelta`` and resolve them
        to absolute times at ``Scenario.play``.
        """
        if self.reference_time is not None:
            self.timestamp_adjustment_hack()
        # Only apply this for datasets that ship relativedelta starting state.
        elif self.starting_context.toolbox.name in []:
            self.convert_relativedelta_to_absolute()

    def _inject_reference_time_into_prompt(self) -> None:
        """Prepend the scenario's reference time to the agent's base system prompt.

        Agents need the current time to resolve relative references in user
        messages ("tomorrow at 10AM", "next Monday", etc.).  The oracle agent
        bypasses this because it knows ground-truth timestamps, but regular
        agents cannot resolve them without an explicit clock.
        """
        if self.reference_time is None:
            return
        try:
            ref_dt = datetime.datetime.fromisoformat(
                self.reference_time.replace("Z", "+00:00")
            )
        except (ValueError, TypeError):
            return
        time_str = ref_dt.strftime("%A, %B %d, %Y at %I:%M %p")
        time_line = f"Current date and time: {time_str}\n\n"

        sandbox_db = self.starting_context._dbs[DatabaseNamespace.SANDBOX]
        sys_agent_mask = (sandbox_db["sender"] == RoleType.SYSTEM) & (
            sandbox_db["recipient"] == RoleType.AGENT
        )
        first_idx = sys_agent_mask.arg_true()
        if len(first_idx) == 0:
            return
        base_prompt_mask = pl.Series(
            [i == first_idx[0] for i in range(sandbox_db.height)]
        )
        self.starting_context._dbs[DatabaseNamespace.SANDBOX] = sandbox_db.with_columns(
            pl.when(base_prompt_mask)
            .then(pl.lit(time_line) + pl.col("content"))
            .otherwise(pl.col("content"))
            .alias("content")
        )

    def play(
        self,
        roles: dict[RoleType, BaseRole],
        scenario_name: str,
        *,
        dataset_name: str,
    ) -> ExecutionContext:
        """Play out the scenario and return execution context

        Args:
            roles:          A mapping indicating which Role we should use for each role
                            type.
            scenario_name:  The scenario name.
            dataset_name:   The name of the dataset to which the scenario belongs.

        Returns:
            Execution context after playing out the scenario

        """
        # Adjust initial database timestamps based on current time at `play` invocation.
        self.adjust_time()

        # Freeze time to the scenario's reference_time so that all code
        # (including tools) sees a consistent clock.  For AppWorld scenarios
        # this nests harmlessly inside the outer freeze from
        # appworld_runtime_context; for agentsandbox scenarios it ensures
        # datetime.now() returns the reference_time rather than wall-clock.
        time_freezer = None
        if self.reference_time is not None:
            from freezegun import (  # type: ignore[import-not-found,unused-ignore]
                freeze_time,
            )

            try:
                ref_dt = datetime.datetime.fromisoformat(
                    self.reference_time.replace("Z", "+00:00")
                )
                time_freezer = freeze_time(ref_dt)
                time_freezer.start()
            except (ValueError, TypeError):
                time_freezer = None

        try:
            return self._play_inner(roles, scenario_name, dataset_name=dataset_name)
        finally:
            if time_freezer is not None:
                time_freezer.stop()

    def _play_inner(
        self,
        roles: dict[RoleType, BaseRole],
        scenario_name: str,
        *,
        dataset_name: str,
    ) -> ExecutionContext:
        """Inner play loop, separated so ``play`` can wrap it in a time freeze."""
        self._inject_reference_time_into_prompt()

        set_current_context(copy.deepcopy(self.starting_context))
        execution_context = get_current_context()
        self.prepare(execution_context)

        # Right now the `ToolboxName` type is just an alias of the `DatasetName`. Note
        # that we use `startswith` instead of an equality check since when running a
        # subset of a dataset's scenarios we append `_local_mods` to the dataset name.
        assert dataset_name.startswith(execution_context.toolbox.name), (
            "The toolbox in the `ExecutionContext` has a name of "
            f"'{execution_context.toolbox.name}' which does not match the dataset name "
            f"of '{dataset_name}'."
        )

        # Prepare InteractiveConsole by consuming system message addressed to it
        sandbox_db = execution_context.get_database(
            DatabaseNamespace.SANDBOX,
            drop_sandbox_message_index=False,
            get_all_history_snapshots=True,
        )
        max_sandbox_message_index = execution_context.max_sandbox_message_index
        for message_index in range(max_sandbox_message_index + 1):
            if (
                sandbox_db["recipient"][message_index] == RoleType.EXECUTION_ENVIRONMENT
                and sandbox_db["sender"][message_index] == RoleType.SYSTEM
            ):
                actor = roles[sandbox_db["recipient"][message_index]]
                response_messages, introspection_entry_dict = actor.respond(
                    messages=get_messages_from_execution_context(
                        execution_context=get_current_context(),
                        ending_index=message_index,
                    ),
                    available_tools=execution_context.get_available_tools_for_role(
                        role_type=cast(RoleType, actor.role_type)
                    ),
                    external_tool_schemas=execution_context.external_tool_schemas,
                )
                # Add response message to SANDBOX database
                add_messages_to_execution_context(
                    execution_context=get_current_context(), messages=response_messages
                )
                # Add introspection entries
                for (
                    introspection_namespace,
                    entries,
                ) in introspection_entry_dict.items():
                    get_current_context().add_to_introspection_database(
                        namespace=introspection_namespace, rows=entries
                    )

        # Since this should only be processing system message, there should be no new
        # messages after this.
        assert (
            get_current_context().max_sandbox_message_index == max_sandbox_message_index
        )

        # Ensure that the latest message is not from the system role. The user/agents
        # skip system messages and do not produce new messages. Thus, we would be
        # processing the same system message in an infinite loop.
        sender = sandbox_db["sender"][-1]
        recipient = sandbox_db["recipient"][-1]
        if sender == RoleType.SYSTEM and recipient == RoleType.AGENT:
            # Note that we perform this check here instead of a `post-init` validator
            # because the scenario might be created progressively (e.g. starting with
            # a partially initialized `Scenario` object).
            raise RuntimeError(
                f"The definition of scenario '{scenario_name}' is invalid. The last "
                f"message of the scenario should not be from the {RoleType.SYSTEM} to "
                f"the {RoleType.AGENT} since the agent does not respond to messages "
                f"from the {RoleType.SYSTEM} so the scenario playback would be stuck "
                "in an infinite loop."
            )

        # Start processing non-system messages
        _rm = getattr(self, "runtime_metadata", None)
        image_delivery_map = (_rm.get("image_delivery_map") if _rm else None) or []
        delivered_images: set[int] = set()
        with tqdm(total=self.max_messages, desc=scenario_name) as pbar:
            while (
                sandbox_db["conversation_active"][-1]
                and sandbox_db["sandbox_message_index"][-1]
                < self.max_messages + max_sandbox_message_index
            ):
                actor = roles[sandbox_db["recipient"][-1]]
                response_messages, introspection_entry_dict = actor.respond(
                    messages=get_messages_from_execution_context(
                        execution_context=get_current_context(),
                    ),
                    available_tools=execution_context.get_available_tools_for_role(
                        role_type=cast(RoleType, actor.role_type)
                    ),
                    external_tool_schemas=execution_context.external_tool_schemas,
                )
                # Guarantee image delivery: if the user sim sent text
                # that matches a round query but forgot the images,
                # attach them automatically.
                if image_delivery_map and actor.role_type == RoleType.USER:
                    _enforce_image_delivery(
                        response_messages, image_delivery_map, delivered_images
                    )
                # Add response message to SANDBOX database
                add_messages_to_execution_context(
                    execution_context=get_current_context(), messages=response_messages
                )
                # Add introspection entries
                for (
                    introspection_namespace,
                    entries,
                ) in introspection_entry_dict.items():
                    get_current_context().add_to_introspection_database(
                        namespace=introspection_namespace, rows=entries
                    )
                sandbox_db = get_current_context().get_database(
                    DatabaseNamespace.SANDBOX, drop_sandbox_message_index=False
                )
                pbar.update()
            # Update max turns on successful end.
            pbar.total = pbar.n
            pbar.update(0)

        return get_current_context()

    def play_and_evaluate(
        self,
        roles: dict[RoleType, BaseRole],
        output_directory: Path | None,
        scenario_name: str,
        *,
        dataset_name: str,
        judge_name: str | None = None,
    ) -> tuple[ScenarioResult, list[dict[str, Any]], list[dict[str, Any]]]:
        """Play out the scenario and evaluate according to evaluation criteria.

        Args:
            roles: A mapping indicating which Role to use for each role type.
            output_directory: Directory to write results to. When None, skips
                writing results.
            scenario_name: Unique name for scenario. Used to serialize message
                history.
            dataset_name: The name of the dataset to which the scenario belongs.
            judge_name: Judge class name from ``_JUDGE_REGISTRY`` in
                ``base_judge.py``. If None, uses the default judge.

        Returns:
            A tuple of (ScenarioResult, serialized conversation, available tools).
        """
        # If an exception occurs during playback we want to save the conversation and
        # execution context histories before re-raising the exception to skip
        # evaluation.
        token_usage_data: dict[str, Any] = {}
        try:
            self.play(
                roles=roles,
                scenario_name=scenario_name,
                dataset_name=dataset_name,
            )
        except Exception:
            raise
        finally:
            execution_context = get_current_context()

            # Extract token usage from roles
            for role_type_key, role in roles.items():
                usage = role.token_usage
                token_usage_data[role_type_key.value] = {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "api_calls": usage.api_calls,
                }

            # Write pretty print messages
            # Skip user simulator few shot messages
            pretty_print_str = (
                "Note that User Simulator few shot messages have been omitted\n"
                + str(
                    execution_context.get_database(
                        DatabaseNamespace.SANDBOX,
                        get_all_history_snapshots=True,
                        drop_sandbox_message_index=False,
                    )
                    .filter(
                        (pl.col("visible_to") != [RoleType.USER])
                        | (pl.col("visible_to").is_null())
                    )
                    .drop(
                        [
                            "openai_tool_call_id",
                            "conversation_active",
                        ]
                    )
                )
            )

        # Compute the scenario-level output directory for saving judge evidence.
        scenario_output_directory: Path | None = None
        if output_directory is not None:
            scenario_output_directory = (
                output_directory / "trajectories" / scenario_name
            )

        # Write the conversation to a JSON file in a generic format.
        # Use a placeholder evaluation_result for serialization — the real one
        # is written after the judge runs.
        placeholder_eval = EvaluationResult()
        full_conversation, agent_conversation = serialize_to_conversation(
            execution_context=execution_context,
            evaluation_result=placeholder_eval,
        )

        # Build the list of OpenAI tool schemas available to the agent.
        available_openai_tools: list[dict[str, Any]] = [
            convert_to_openai_tool(
                tool=tool,
                name=name,
            )
            for name, tool in get_current_context()
            .get_available_tools_for_role(role_type=RoleType.AGENT)
            .items()
        ]

        # Save trajectories BEFORE running the judge so they are preserved even
        # if the judge fails.
        if output_directory is not None:
            scenario_output_directory = (
                output_directory / "trajectories" / scenario_name
            )
            scenario_output_directory.mkdir(exist_ok=True, parents=True)
            with open(
                scenario_output_directory / "pretty_print.txt", "w", encoding="utf-8"
            ) as f:
                f.write(pretty_print_str)

            # Write execution_context
            with open(
                scenario_output_directory / "execution_context.json",
                "w",
                encoding="utf-8",
            ) as f:
                # We'll have to ditch dill InteractiveConsole here because
                # dill creates a bytes instead of raw string
                f.write(
                    json.dumps(
                        execution_context.to_dict(serialize_console=False),
                        ensure_ascii=False,
                        indent=4,
                    )
                )
            # Write pretty print files for non-empty introspection databases.
            introspection_db_manager = (
                execution_context.get_introspection_database_manager()
            )
            for db_name in introspection_db_manager.get_populated_databases():
                db = introspection_db_manager.get_database(db_name)
                pretty_print_fname = get_introspection_pretty_print_file_name(db_name)
                with open(
                    scenario_output_directory / pretty_print_fname,
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(str(db))

                json_fname = get_introspection_json_file_name(db_name)
                with open(
                    scenario_output_directory / json_fname,
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        introspection_db_manager.get_database_as_json_object(db_name),
                        f,
                        ensure_ascii=False,
                        indent=4,
                    )
            with open(
                scenario_output_directory / "conversation.json",
                "w",
                encoding="utf-8",
            ) as f:
                available_user_tools: list[dict[str, Any]] = [
                    convert_to_openai_tool(
                        tool=tool,
                        name=name,
                    )
                    for name, tool in get_current_context()
                    .get_available_tools_for_role(role_type=RoleType.USER)
                    .items()
                ]
                json.dump(
                    {
                        "messages": full_conversation,
                        "tools": available_openai_tools,
                        "user_tools": available_user_tools,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            # Write user-facing conversation for debugging user simulator behavior
            user_conversation = serialize_user_conversation(
                execution_context=execution_context,
            )
            with open(
                scenario_output_directory / "user_conversation.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {"messages": user_conversation},
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            with open(
                scenario_output_directory / "token_usage.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(token_usage_data, f, indent=2, ensure_ascii=False)

        # Extract scenario_metadata for the user judge.  AppWorld scenarios
        # store design parameters (challenge_type, image_arrival, etc.) under
        # runtime_metadata["scenario_metadata"].  These are passed through to
        # the user judge so it can interpret intentional script behaviors
        # (e.g., error_correction challenges) rather than penalizing them.
        scenario_metadata = None
        if self.runtime_metadata and "scenario_metadata" in self.runtime_metadata:
            scenario_metadata = self.runtime_metadata["scenario_metadata"]

        # Run the judge. If it fails, use a sentinel result (-1 similarity)
        # so trajectories are preserved and the judge can be backfilled later.
        try:
            evaluation_result = evaluate(
                evaluation_criteria=self.evaluation_criteria,
                execution_context=execution_context,
                max_turn_count=self.max_messages,
                judge_name=judge_name,
                output_directory=scenario_output_directory,
                scenario_metadata=scenario_metadata,
            )
        except Exception:
            LOGGER.error(
                "Judge failed for scenario '%s', saving trajectory with "
                "sentinel evaluation (similarity=-1):\n%s",
                scenario_name,
                traceback.format_exc(),
            )
            evaluation_result = EvaluationResult(
                turn_count=self.max_messages,
                judge_result=None,
                entity_diff_result=None,
            )

        return (
            ScenarioResult(
                ending_context=execution_context,
                evaluation_result=evaluation_result,
                token_usage=token_usage_data,
            ),
            full_conversation,
            available_openai_tools,
        )


def compute_missing_tools(
    tool_names: list[str] | None, toolbox_tool_names: set[str]
) -> set[str]:
    """Compute which elements in `tool_names` are not in `toolbox_tool_names`.

    Args:
        tool_names:  The list of tool names expected to exist in `toolbox_tool_names`.
                     If `None` there is nothing to check.
        toolbox_tool_names:  The name of all tools in the toolbox.

    Returns:
        The elements in `tool_names` that are not in `toolbox_tool_names`.

    """
    if tool_names is None:
        return set()
    missing_tools = set(tool_names).difference(toolbox_tool_names)
    return missing_tools


# Mapping from scenario namespace strings to DatabaseNamespace enums.
_MMTOOLSANDBOX_NAMESPACE_MAP: dict[str, DatabaseNamespace] = {
    "contact": DatabaseNamespace.CONTACT,
    "messaging": DatabaseNamespace.MESSAGING,
    "reminder": DatabaseNamespace.REMINDER,
    "notes": DatabaseNamespace.NOTES,
    "calendars": DatabaseNamespace.CALENDARS,
    "calendar_events": DatabaseNamespace.CALENDAR_EVENTS,
    "setting": DatabaseNamespace.SETTING,
}


def _stage_mmtoolsandbox_entities(
    context: ExecutionContext,
    entities: dict[str, list[dict[str, Any]]],
) -> None:
    """Insert pre-seeded MMToolSandbox entities into the execution context.

    Must be called **before** conversation messages are added to SANDBOX
    so that the evaluator's initial-state snapshot includes these entities.
    """
    for ns_str, rows in entities.items():
        if not rows:
            continue
        db_namespace = _MMTOOLSANDBOX_NAMESPACE_MAP.get(ns_str)
        if db_namespace is None:
            continue
        prepared = [
            {k: v for k, v in row.items() if k != "sandbox_message_index"}
            for row in rows
        ]
        context.add_to_database(namespace=db_namespace, rows=prepared)


@define
class ScenarioExtension:
    """Extends a few fields over base scenario to form a valid test scenario.

    Attributes:
        name: Name for the resulting extended scenario.
        base_scenario: Base ``Scenario`` to extend.
        messages: Messages to append to the base scenario's SANDBOX database.
        tool_allow_list: Additional tool names to add to the allow list.
        tool_deny_list: Additional tool names to add to the deny list.
        categories: Additional ``ScenarioCategories`` to tag the scenario with.
        image_paths: Optional file paths for images to load at scenario start.
        task_completion_criteria: Optional criteria string for the LLM judge.
        user_tool_allow_list: Optional tool allow list for the user role.
            When set, the user simulator can call these tools in addition to
            ``end_conversation``.
        max_messages: Optional override for max messages (defaults to base
            scenario value).
        reference_time: Optional ISO 8601 reference time for timestamp
            adjustment.
        runtime_metadata: Optional metadata dict for external setup (e.g.
            AppWorld bridge init, entity staging).
        enable_ui: When True, activates the UI quality rubric in the judge.
    """

    # Name for the resulting extended scenario
    name: str
    # Base scenario to extend on
    base_scenario: Scenario
    # Messages to extend to the starting context of base scenario
    messages: list[dict[str, str | list[RoleType] | list[ImageId]]] = Factory(list)
    # Tool allow list to extend to starting context of base scenario
    tool_allow_list: list[str] | None = None
    # Tool deny list to extend to starting context of base scenario
    tool_deny_list: list[str] | None = None
    # Categories to extend to scenario
    categories: list[ScenarioCategories] = Factory(list)
    # Optional image paths for scenarios that include images
    image_paths: list[str] | None = None
    # Optional task completion criteria using llm judge model.
    task_completion_criteria: str | None = None
    # Optional tool allow list for the user role. When set, the user simulator
    # can call these tools in addition to `end_conversation` (always included).
    # When None (default), only `end_conversation` is available to the user.
    user_tool_allow_list: list[str] | None = None
    # Optional override for max messages (defaults to base scenario value)
    max_messages: int | None = None
    # Optional reference time (e.g. device_state_current_time for AppWorld scenarios)
    reference_time: str | None = None
    # Optional runtime metadata for scenarios that need external setup
    # (e.g. AppWorld bridge init, entity staging).
    # Keys: appworld_entities, agentsandbox_entities, appworld_base_task,
    #        device_state_id, description, difficulty, and any custom keys.
    runtime_metadata: dict[str, Any] | None = None
    # When True, the judge appends UI quality rubric dimensions to evaluate
    # the quality of agent-generated UI screens.
    enable_ui: bool = False

    def get_extended_scenario(self) -> dict[str, Scenario]:
        """Get an extended scenario based on specified extensions

        Returns:
            A dictionary containing extended scenario and name
        """
        scenario: Scenario = copy.deepcopy(self.base_scenario)

        # Load all the initial images in the scenario and register them in the IMAGE database
        # This should be done before adding any message that might reference one of those images
        # to keep the databases chronologically consistent.
        if self.image_paths:
            for image_path in self.image_paths:
                image_base64 = load_image_as_base64(image_path)
                scenario.starting_context.add_image(image_base64)

        # Stage pre-seeded entities (contacts, reminders, etc.)
        # BEFORE adding messages so they appear in the initial-state snapshot
        # that the evaluator reads at first_user_sandbox_message_index.
        if self.runtime_metadata:
            agentsandbox_entities = self.runtime_metadata.get(
                "agentsandbox_entities", {}
            )
            if agentsandbox_entities:
                _stage_mmtoolsandbox_entities(
                    scenario.starting_context, agentsandbox_entities
                )

        scenario.starting_context.add_to_database(
            namespace=DatabaseNamespace.SANDBOX, rows=self.messages
        )
        if self.tool_allow_list is not None:
            if scenario.starting_context.tool_allow_list is None:
                scenario.starting_context.tool_allow_list = []
            scenario.starting_context.tool_allow_list.extend(self.tool_allow_list)

        # User tool allow list: always restrict the user role.
        # Default: only end_conversation. When user_tool_allow_list is set,
        # those tools are added on top of end_conversation.
        user_tools = list(self.user_tool_allow_list or [])
        if "end_conversation" not in user_tools:
            user_tools.append("end_conversation")
        # Add user tools to global allow list so they pass the global filter
        if scenario.starting_context.tool_allow_list is not None:
            for tool_name in user_tools:
                if tool_name not in scenario.starting_context.tool_allow_list:
                    scenario.starting_context.tool_allow_list.append(tool_name)
        # Set role-specific allow list for USER
        scenario.starting_context.role_tool_allow_list[RoleType.USER] = user_tools

        # Consistency check: ensure that all tools from the allow list exist in the
        # toolbox.
        toolbox = scenario.starting_context.get_toolbox()
        toolbox_tool_names = toolbox.get_tool_names()
        missing_tool_allow_list_names = compute_missing_tools(
            scenario.starting_context.tool_allow_list, toolbox_tool_names
        )
        if len(missing_tool_allow_list_names) > 0:
            raise RuntimeError(
                f"The tool allow list of scenario '{self.name}' contains tools that "
                f"are not part of the '{toolbox.name}' toolbox. Add the following "
                "tools to the toolbox by extending their `register_as_tool` decorator:"
                f"\n{sorted(missing_tool_allow_list_names)}. Also, double check if the module is imported"
                f" in mmtoolsandbox.tools.__init__ or not."
            )

        if self.tool_deny_list is not None:
            if scenario.starting_context.tool_deny_list is None:
                scenario.starting_context.tool_deny_list = []
            scenario.starting_context.tool_deny_list.extend(self.tool_deny_list)

        # Consistency check: ensure that all tools from the deny list exist in the
        # toolbox.
        missing_tool_deny_list_names = compute_missing_tools(
            scenario.starting_context.tool_deny_list, toolbox_tool_names
        )
        if len(missing_tool_deny_list_names) > 0:
            raise RuntimeError(
                f"The tool deny list of scenario '{self.name}' contains tools that "
                f"are not part of the '{toolbox.name}' toolbox. Add the following "
                "tools to the toolbox by extending their `register_as_too` decorator:"
                f"\n{sorted(missing_tool_deny_list_names)}. Also, double check if the module is imported"
                f" in mmtoolsandbox.tools.__init__ or not."
            )
        scenario.evaluation_criteria = EvaluationCriteria(
            task_completion_criteria=self.task_completion_criteria,
            enable_ui=self.enable_ui,
        )

        scenario.categories.extend(self.categories)

        if self.image_paths is not None:
            scenario.image_paths = self.image_paths

        if self.max_messages is not None:
            scenario.max_messages = self.max_messages

        if self.reference_time is not None:
            scenario.reference_time = self.reference_time

        if self.runtime_metadata is not None:
            scenario.runtime_metadata = self.runtime_metadata

        return {self.name: scenario}


def filter_scenarios_by_tag_expression(
    scenarios: dict[str, Scenario], tag_expression: str
) -> dict[str, Scenario]:
    """Filter scenarios using a boolean tag expression.

    Supports ``|`` (OR), ``&`` (AND), ``~`` (NOT), and parentheses for
    grouping.  Each token in the expression must match a tag present in
    the dataset.

    Example::

        filter_scenarios_by_tag_expression(scenarios, "(vision|calendar)&~ui")

    Args:
        scenarios: Mapping of scenario names to ``Scenario`` objects.
        tag_expression: Boolean expression over scenario tags.

    Returns:
        Subset of ``scenarios`` matching the expression.

    Raises:
        ValueError: If the expression contains no tags, references unknown
            tags, or a tag name contains reserved characters.
    """
    special_chars = "|()&~ "
    all_scen_names, all_scenarios = zip(*scenarios.items())
    all_tags_in_dataset = set()
    for scen in all_scenarios:
        all_tags_in_dataset |= scen.tags

    # Create a boolean columns per tag. Each column has length of the dataset.
    tag_scen_table = {
        tag: np.array([(tag in scen.tags) for scen in all_scenarios])
        for tag in all_tags_in_dataset
    }

    # Validate current tag names don't contain the special chars used in expression
    for tag in all_tags_in_dataset:
        for ch in special_chars:
            if ch in tag:
                raise ValueError(f"Illegal character `{ch}` found in {tag=}")
    tags_in_expr = re.split(f"[{special_chars}]+", tag_expression)
    tags_in_expr = [tag for tag in tags_in_expr if tag != ""]  # remove resulting empty
    if not tags_in_expr:
        raise ValueError(f"Could not find a single tag in {tag_expression}")

    # Ensure all specified tags exist in the dataset
    if unknown_tags := set(tags_in_expr) - all_tags_in_dataset:
        raise ValueError(f"{unknown_tags=} not found in {all_tags_in_dataset=}")

    # Replace tags which may not be valid python with temp variables x0, x1, ...
    # Enclose the expression to one extra parenthesis to capture the last variable
    tag_to_temp_tag = {tag: f"temp_tag_{i}" for i, tag in enumerate(tags_in_expr)}

    # Assume that none tags are actually called temp_tag_0 etc
    assert not any(tag in tag_to_temp_tag.values() for tag in tags_in_expr)
    tag_expression_enclosed = f"({tag_expression})"
    for tag_name, temp_var in tag_to_temp_tag.items():
        tag_expression_enclosed = re.sub(
            pattern=f"{tag_name}([{special_chars}])",  # example my_tag([()|&~ ])
            repl=temp_var + r"\1",  # keep the special char
            string=tag_expression_enclosed,
        )

    # Evaluate the expression with temporary variables as numpy boolean vectors
    temp_var_table = {
        temp: tag_scen_table[tag] for tag, temp in tag_to_temp_tag.items()
    }
    try:
        filt = eval(tag_expression_enclosed, temp_var_table)
    except SyntaxError as e:
        raise ValueError(f"Malformed {tag_expression=}") from e
    # Make sure the result is as expected
    assert isinstance(filt, np.ndarray) and len(filt) == len(all_scenarios)

    # Get scenario names back from numpy world
    filtered_scenario_names = np.array(all_scen_names)[filt].tolist()

    if len(filtered_scenario_names) == 0:
        raise ValueError(f"No scenario matches {tag_expression=}")

    return {name: scenarios[name] for name in filtered_scenario_names}
