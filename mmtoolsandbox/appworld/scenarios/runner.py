"""
Scenario Runner Adapter for AppWorld scenarios.

This module provides adapters and hooks that integrate AppWorld task execution
with MMToolSandbox's scenario runner. It handles:
- Initializing AppWorld when an AppWorld-based scenario starts
- Applying database modifications (both Appworld SQL and MMToolSandbox entities)
- Running setup code
- Cleanup after scenario completion
"""

from __future__ import annotations

import atexit
import base64
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Generator,
)

import attrs

from mmtoolsandbox.appworld.bridge import (
    AppWorldBridge,
    AppWorldNotInstalledError,
    get_appworld_bridge,
)
from mmtoolsandbox.appworld.state import (
    get_appworld_state,
)
from mmtoolsandbox.common.databases import DatabaseNamespace

if TYPE_CHECKING:
    from mmtoolsandbox.common.execution_context import ExecutionContext

# Mapping from namespace string to DatabaseNamespace enum
_NAMESPACE_MAPPING: dict[str, DatabaseNamespace] = {
    "contact": DatabaseNamespace.CONTACT,
    "messaging": DatabaseNamespace.MESSAGING,
    "reminder": DatabaseNamespace.REMINDER,
    "notes": DatabaseNamespace.NOTES,
    "calendars": DatabaseNamespace.CALENDARS,
    "calendar_events": DatabaseNamespace.CALENDAR_EVENTS,
    "setting": DatabaseNamespace.SETTING,
}


@dataclass
class AppWorldScenarioRunner:
    """
    Manages the lifecycle of AppWorld for scenario execution.

    This runner is responsible for:
    - Initializing AppWorld with the correct task state
    - Applying database modifications
    - Running setup code
    - Cleaning up after scenario completion
    - Syncing state between AppWorld and MMToolSandbox

    Usage:
        runner = AppWorldScenarioRunner()

        # For AppWorld task scenarios
        with runner.run_scenario(task_id="123", database_modifications=[...]):
            # Scenario executes here
            pass

        # For cross-system scenarios (no specific task)
        with runner.run_scenario(base_task_id="123", database_modifications=[...]):
            # Scenario executes here
            pass
    """

    _active: bool = field(default=False, init=False)
    _current_task_id: str | None = field(default=None, init=False)

    @staticmethod
    def apply_starting_state(
        execution_context: "ExecutionContext",
        starting_state: dict[str, list[dict[str, Any]]],
    ) -> None:
        """
        Apply MMToolSandbox starting state to the execution context.

        This enables cross-environment tasks to modify MMToolSandbox databases
        (contacts, calendar events, reminders, etc.) alongside Appworld databases.

        Args:
            execution_context: The MMToolSandbox execution context
            starting_state: Dict mapping namespace to list of entity dicts
                Example: {
                    "contact": [{"person_id": "...", "name": "John", ...}],
                    "calendar_events": [{"calendar_event_id": "...", ...}],
                }

        Raises:
            ValueError: If an unknown namespace is encountered.
            KeyError: If entity fields don't match the database schema.
        """
        from mmtoolsandbox.common.execution_context import ExecutionContext

        for namespace_str, entities in starting_state.items():
            if not entities:
                continue

            if namespace_str not in _NAMESPACE_MAPPING:
                raise ValueError(
                    f"Unknown MMToolSandbox namespace: {namespace_str!r}. "
                    f"Valid options: {list(_NAMESPACE_MAPPING.keys())}"
                )

            db_namespace = _NAMESPACE_MAPPING[namespace_str]

            # Remove sandbox_message_index if present (added automatically)
            prepared = [
                {k: v for k, v in entity.items() if k != "sandbox_message_index"}
                for entity in entities
            ]

            # Validate column names before inserting so errors are clear
            schema_columns = set(ExecutionContext.dbs_schemas[db_namespace].keys())
            schema_columns.discard("sandbox_message_index")
            entity_columns = {k for row in prepared for k in row.keys()}
            unknown_columns = entity_columns - schema_columns
            if unknown_columns:
                raise KeyError(
                    f"Entity staging failed for namespace {namespace_str!r}: "
                    f"unknown columns {unknown_columns}. "
                    f"Expected columns (excluding sandbox_message_index): "
                    f"{sorted(schema_columns)}. "
                    f"Provided columns: {sorted(entity_columns)}. "
                    f"Hint: check that the scenario conversion script maps field "
                    f"names to the MMToolSandbox schema (e.g. 'modification_datetime' "
                    f"should be 'modification_timestamp' for notes)."
                )

            execution_context.add_to_database(
                namespace=db_namespace,
                rows=prepared,
            )

    @contextmanager
    def run_scenario(
        self,
        task_id: str | None = None,
        base_task_id: str | None = None,
        database_modifications: list[tuple[str, str, list[Any]]] | None = None,
        setup_code: str | None = None,
        starting_state: dict[str, list[dict[str, Any]]] | None = None,
        execution_context: "ExecutionContext" | None = None,
    ) -> Generator[AppWorldBridge, None, None]:
        """
        Context manager for running an AppWorld-based scenario.

        Args:
            task_id: AppWorld task ID for direct task scenarios
            base_task_id: Base task ID for cross-system scenarios
            database_modifications: SQL modifications to apply to Appworld databases
            setup_code: Python code to run in AppWorld before scenario starts
            starting_state: MMToolSandbox entity modifications to apply
                Dict mapping namespace to list of entity dicts (e.g.,
                {"contact": [...], "calendar_events": [...]})
            execution_context: Required if starting_state is provided

        Yields:
            The initialized AppWorldBridge instance

        Example:
            runner = AppWorldScenarioRunner()

            with runner.run_scenario(
                task_id="task_123",
                database_modifications=[
                    ("spotify", "UPDATE ...", []),
                ],
                setup_code="apis.spotify.login(...)",
                starting_state={
                    "contact": [{"person_id": "p1", "name": "John", ...}],
                },
                execution_context=scenario.starting_context,
            ) as bridge:
                # Run scenario using bridge
                pass
        """
        effective_task_id = task_id or base_task_id

        # Apply MMToolSandbox starting state if provided
        if starting_state and execution_context:
            self.apply_starting_state(execution_context, starting_state)
        elif starting_state and not execution_context:
            print(
                "Warning: starting_state provided but execution_context is None. "
                "MMToolSandbox entities will not be applied."
            )

        if effective_task_id is None:
            # No AppWorld task - just yield None and let scenario run
            yield None  # type: ignore[misc]
            return

        try:
            # Initialize AppWorld
            bridge = get_appworld_bridge()

            # Check if task exists, fallback if necessary
            try:
                import os

                from appworld.common.path_store import path_store

                from mmtoolsandbox.appworld.bridge import load_task_ids

                # Ensure path_store is configured
                try:
                    load_task_ids("train")
                except Exception:
                    pass

                task_dir = os.path.join(path_store.data, "tasks", effective_task_id)
                if not os.path.exists(task_dir):
                    print(
                        f"Warning: Task '{effective_task_id}' not found at {task_dir}."
                    )
                    # Try to find a valid task from train/dev sets
                    for dataset in ["train", "dev", "test_normal", "test_challenge"]:
                        try:
                            task_ids = load_task_ids(dataset)
                            if task_ids:
                                effective_task_id = task_ids[0]
                                print(
                                    f"Using fallback task: {effective_task_id} (from {dataset})"
                                )
                                break
                        except Exception:
                            continue
            except ImportError:
                pass  # AppWorld might not be fully installed
            except Exception as e:
                print(f"Warning: Failed to check task existence: {e}")

            bridge.initialize(effective_task_id)
            self._active = True
            self._current_task_id = effective_task_id

            # Reset AppWorld state tracking
            state = get_appworld_state()
            state.reset()

            # Apply database modifications if provided
            if database_modifications:
                bridge.apply_database_modifications(database_modifications)

            # Run setup code if provided
            if setup_code:
                bridge.run_setup_code(setup_code)

            yield bridge

        finally:
            # Cleanup
            self._cleanup()

    def _cleanup(self) -> None:
        """Clean up AppWorld resources."""
        if self._active:
            try:
                bridge = get_appworld_bridge()
                bridge.close()
            except Exception:
                pass  # Ignore cleanup errors

            # Reset state
            get_appworld_state().reset()
            self._active = False
            self._current_task_id = None

    @property
    def is_active(self) -> bool:
        """Check if an AppWorld scenario is currently running."""
        return self._active

    @property
    def current_task_id(self) -> str | None:
        """Get the current task ID if a scenario is running."""
        return self._current_task_id


# Global singleton instance
_scenario_runner: AppWorldScenarioRunner | None = None


def get_scenario_runner() -> AppWorldScenarioRunner:
    """Get the singleton AppWorldScenarioRunner instance."""
    global _scenario_runner
    if _scenario_runner is None:
        _scenario_runner = AppWorldScenarioRunner()
    return _scenario_runner


def reset_scenario_runner() -> None:
    """Reset the scenario runner. Useful for testing."""
    global _scenario_runner
    if _scenario_runner is not None:
        _scenario_runner._cleanup()
        _scenario_runner = None


# Register cleanup on exit
atexit.register(lambda: reset_scenario_runner())


class AppWorldScenarioHooks:
    """
    Hooks for integrating AppWorld scenarios with MMToolSandbox's scenario lifecycle.

    These hooks can be used to:
    - Initialize AppWorld before scenario starts
    - Sync state during execution
    - Evaluate using AppWorld's evaluation
    - Clean up after scenario ends
    """

    @staticmethod
    def before_scenario_play(
        scenario_name: str,
        scenario_metadata: dict[str, Any] | None = None,
        execution_context: "ExecutionContext" | None = None,
    ) -> AppWorldBridge | None:
        """
        Hook to call before a scenario starts playing.

        Args:
            scenario_name: Name of the scenario
            scenario_metadata: Optional metadata containing AppWorld config
            execution_context: Optional execution context for applying starting_state

        Returns:
            AppWorldBridge if this is an AppWorld scenario, None otherwise
        """
        if scenario_metadata is None:
            return None

        # Check if this is an AppWorld scenario
        task_id = scenario_metadata.get("appworld_task_id")
        base_task_id = scenario_metadata.get("base_task_id")
        database_modifications = scenario_metadata.get("database_modifications")
        setup_code = scenario_metadata.get("setup_code")
        starting_state = scenario_metadata.get("starting_state")

        # Apply MMToolSandbox starting state if provided
        if starting_state and execution_context:
            AppWorldScenarioRunner.apply_starting_state(
                execution_context, starting_state
            )
        elif starting_state and not execution_context:
            print(
                f"Warning: starting_state provided for {scenario_name} but "
                "execution_context is None. MMToolSandbox entities will not be applied."
            )

        if task_id is None and base_task_id is None:
            return None

        try:
            # Initialize AppWorld
            runner = get_scenario_runner()
            effective_task_id = task_id or base_task_id

            if effective_task_id is None:
                print(f"Warning: No task ID provided for {scenario_name}")
                return None

            bridge = get_appworld_bridge()
            bridge.initialize(effective_task_id)

            # Reset state
            get_appworld_state().reset()

            # Apply modifications
            if database_modifications:
                bridge.apply_database_modifications(database_modifications)

            # Run setup
            if setup_code:
                bridge.run_setup_code(setup_code)

            return bridge

        except AppWorldNotInstalledError:
            print(
                f"Warning: AppWorld not installed, skipping AppWorld initialization for {scenario_name}"
            )
            return None
        except Exception as e:
            print(f"Warning: Failed to initialize AppWorld for {scenario_name}: {e}")
            return None

    @staticmethod
    def after_scenario_play(
        scenario_name: str,
        scenario_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Hook to call after a scenario finishes playing.

        Args:
            scenario_name: Name of the scenario
            scenario_metadata: Optional metadata containing AppWorld config

        Returns:
            AppWorld evaluation result if applicable
        """
        if scenario_metadata is None:
            return None

        task_id = scenario_metadata.get("appworld_task_id")
        if task_id is None:
            return None

        try:
            bridge = get_appworld_bridge()
            if bridge.is_initialized:
                # Get AppWorld evaluation
                eval_result = bridge.evaluate()

                # Cleanup
                bridge.close()
                get_appworld_state().reset()

                return {
                    "appworld_task_id": task_id,
                    "appworld_evaluation": eval_result,
                }
        except Exception as e:
            print(f"Warning: Failed to evaluate AppWorld scenario {scenario_name}: {e}")

        return None

    @staticmethod
    def sync_state() -> None:
        """
        Sync state between AppWorld and MMToolSandbox.

        This should be called periodically during scenario execution
        to ensure state dependencies are properly tracked.
        """
        state = get_appworld_state()
        state.sync_with_agentsandbox_state()


def run_appworld_scenario(
    scenario_dict: dict[str, Any],
    roles: dict[str, Any] | None = None,
    execution_context: "ExecutionContext" | None = None,
) -> dict[str, Any]:
    """
    Run an AppWorld-based scenario.

    This is a convenience function that handles the full lifecycle of
    running an AppWorld scenario within MMToolSandbox.

    Args:
        scenario_dict: Scenario definition from task_adapters
        roles: Optional role mappings for the scenario
        execution_context: Optional execution context for applying starting_state

    Returns:
        Dictionary with scenario results including:
        - execution_context: Final execution context
        - appworld_evaluation: AppWorld evaluation result
        - conversation: Conversation history
    """

    metadata = scenario_dict.get("metadata", {})
    task_id = metadata.get("appworld_task_id")
    database_modifications = scenario_dict.get("database_modifications")
    setup_code = scenario_dict.get("setup_code")
    starting_state = scenario_dict.get("starting_state")

    runner = get_scenario_runner()

    with runner.run_scenario(
        task_id=task_id,
        database_modifications=database_modifications,
        setup_code=setup_code,
        starting_state=starting_state,
        execution_context=execution_context,
    ) as bridge:
        # The actual scenario execution would happen here
        # This is a placeholder - the real implementation would
        # integrate with MMToolSandbox's Scenario.play()

        result = {
            "scenario_name": scenario_dict.get("name"),
            "task_id": task_id,
            "bridge_initialized": bridge is not None,
        }

        # Get evaluation if bridge is active
        if bridge is not None and bridge.is_initialized:
            try:
                eval_result = bridge.evaluate()
                result["appworld_evaluation"] = eval_result
            except Exception as e:
                result["evaluation_error"] = str(e)

        return result


def create_appworld_scenario_extension(
    scenario_dict: dict[str, Any],
    base_scenario: Any,  # Scenario type from mmtoolsandbox
) -> Any:
    """
    Create a ScenarioExtension from an AppWorld scenario definition.

    This bridges the gap between AppWorld's task format and MMToolSandbox's
    ScenarioExtension format.

    Args:
        scenario_dict: Scenario definition from task_adapters
        base_scenario: Base Scenario object from MMToolSandbox

    Returns:
        ScenarioExtension configured for the AppWorld task
    """
    from mmtoolsandbox.common.scenario import ScenarioExtension

    return ScenarioExtension(
        name=scenario_dict.get("name", "unnamed_appworld_scenario"),
        base_scenario=base_scenario,
        messages=scenario_dict.get("messages", []),
        tool_allow_list=scenario_dict.get("tool_allow_list"),
        categories=scenario_dict.get("categories", []),
    )


def _initialize_scenario_base(
    toolbox: Any | None = None,
    include_appworld_tools: bool = True,
    support_images: bool = False,
) -> Any:
    """
    Helper to initialize a base scenario with AppWorld tools and base state.
    """
    from mmtoolsandbox.common.i18n import DefaultLocalizer
    from mmtoolsandbox.datasets.base_scenario import create_base_scenario
    from mmtoolsandbox.datasets.initial_database_states.base import (
        calendar_events_initial_database_state,
        calendars_initial_database_state,
        setting_initial_database_state,
    )

    # Get or create toolbox
    if toolbox is None:
        from mmtoolsandbox.toolbox.toolbox import create_empty_toolbox

        toolbox = create_empty_toolbox()

        if include_appworld_tools:
            try:
                from mmtoolsandbox.tools.appworld import get_appworld_tools

                appworld_tools = get_appworld_tools()
                toolbox = attrs.evolve(
                    toolbox, tools=list(toolbox.tools) + list(appworld_tools.values())
                )
            except Exception as e:
                print(f"Warning: Failed to add Appworld tools: {e}")

    # Create base scenario
    scenario = create_base_scenario(
        toolbox=toolbox,
        add_agent_system_prompt=True,
        support_images=support_images,
    )

    # Load MMToolSandbox base state by calling initial state functions directly
    # (bypasses the decorator registry which can fail due to circular imports)
    _base_state_loaders = [
        (DatabaseNamespace.SETTING, setting_initial_database_state),
        (DatabaseNamespace.CALENDARS, calendars_initial_database_state),
        (DatabaseNamespace.CALENDAR_EVENTS, calendar_events_initial_database_state),
    ]
    for ns, loader_fn in _base_state_loaders:
        try:
            rows = loader_fn(DefaultLocalizer)
            scenario.starting_context.add_to_database(namespace=ns, rows=rows)
        except Exception as e:
            print(f"Warning: Failed to load base state for {ns}: {e}")

    return scenario


def create_appworld_scenario(
    scenario_dict: dict[str, Any],
    toolbox: Any | None = None,
    include_appworld_tools: bool = True,
) -> Any:
    """
    Create a complete Scenario from a converted visual-tool-calling scenario dict.

    This handles:
    1. Creating the base MMToolSandbox scenario with image support
    2. Loading MMToolSandbox base state (contact, calendars, etc.)
    3. Applying MMToolSandbox entity modifications (agentsandbox_entities)
    4. Loading images into the IMAGE database
    5. Adding messages to SANDBOX database
    6. Setting up the combined toolbox (MMToolSandbox + Appworld)

    The returned scenario can then be run with:
        runner = get_scenario_runner()
        with runner.run_scenario(
            base_task_id="_base_",
            database_modifications=sql_mods,
            execution_context=scenario.starting_context,
        ) as bridge:
            scenario.play(roles)

    Args:
        scenario_dict: Scenario dict from load_appworld_task()
        toolbox: Optional toolbox (creates combined toolbox if None)
        include_appworld_tools: If True, include Appworld tools in toolbox

    Returns:
        Scenario object with base state loaded and entity modifications applied
    """
    # Create base scenario with image support and base state loaded
    scenario = _initialize_scenario_base(
        toolbox=toolbox,
        include_appworld_tools=include_appworld_tools,
        support_images=True,
    )

    # Apply MMToolSandbox entity modifications
    agentsandbox_entities = scenario_dict.get("agentsandbox_entities")
    if agentsandbox_entities:
        AppWorldScenarioRunner.apply_starting_state(
            scenario.starting_context,
            agentsandbox_entities,
        )

    # Load images into IMAGE database
    image_paths = scenario_dict.get("image_paths", [])
    if image_paths:
        image_rows = []
        for idx, img_path in enumerate(image_paths):
            try:
                with open(img_path, "rb") as img_f:
                    image_data = base64.b64encode(img_f.read()).decode("utf-8")
                image_rows.append(
                    {
                        "image_id": idx,
                        "image_content": image_data,
                    }
                )
            except Exception as e:
                print(f"Warning: Failed to load image {img_path}: {e}")

        if image_rows:
            scenario.starting_context.add_to_database(
                namespace=DatabaseNamespace.IMAGE,
                rows=image_rows,
            )

    # Add task messages to conversation
    messages = scenario_dict.get("messages", [])
    if messages:
        scenario.starting_context.add_to_database(
            namespace=DatabaseNamespace.SANDBOX,
            rows=messages,
        )

    # No tool_allow_list restriction - agent is free to use any tool
    # (end_conversation is already set by create_base_scenario)

    return scenario


def run_appworld_task(
    scenario_dict: dict[str, Any],
    roles: dict[str, Any] | None = None,
    toolbox: Any | None = None,
) -> dict[str, Any]:
    """
    Complete end-to-end execution of a visual-tool-calling task.

    This function handles the entire lifecycle:
    1. Creates MMToolSandbox scenario with image support and correct base
    2. Applies MMToolSandbox entity modifications
    3. Loads images into IMAGE database
    4. Initializes Appworld with _base_ task
    5. Converts appworld_entities to SQL and applies modifications
    6. Returns results including evaluation info

    Args:
        scenario_dict: Scenario dict from load_appworld_task()
        roles: Optional role mappings for scenario execution
        toolbox: Optional toolbox

    Returns:
        Dictionary with execution results including:
        - scenario_name: Name/ID of the executed scenario
        - appworld_evaluation: Appworld evaluation result (if applicable)
        - execution_context: Final execution context

    Example:
        from mmtoolsandbox.appworld.scenarios.task_adapters import (
            load_appworld_task,
        )
        from mmtoolsandbox.appworld.scenarios.runner import (
            run_appworld_task,
        )

        scenario_dict = load_appworld_task("path/to/scenario_0010.json")
        result = run_appworld_task(scenario_dict)
    """
    from mmtoolsandbox.appworld.scenarios.task_adapters import (
        convert_appworld_entities_to_sql,
    )

    # Create scenario with base state, images, and entities loaded
    scenario = create_appworld_scenario(scenario_dict, toolbox)

    # Convert appworld_entities to SQL modifications
    appworld_entities = scenario_dict.get("appworld_entities", {})
    database_modifications = None
    if appworld_entities:
        database_modifications = convert_appworld_entities_to_sql(appworld_entities)

    # Use _base_ task for AppWorld initialization
    base_task_id = scenario_dict.get("appworld_base_task", "_base_")

    runner = get_scenario_runner()

    result: dict[str, Any] = {
        "scenario_name": scenario_dict.get("task_id"),
        "agentsandbox_base": scenario_dict.get("agentsandbox_base", "base"),
        "appworld_base_task": base_task_id,
    }

    with runner.run_scenario(
        base_task_id=base_task_id,
        database_modifications=database_modifications,
        execution_context=scenario.starting_context,
    ) as bridge:
        result["bridge_initialized"] = bridge is not None

        # Update base_task_id in result if fallback occurred
        if runner.current_task_id and runner.current_task_id != base_task_id:
            result["appworld_base_task"] = runner.current_task_id
            print(
                f"Note: Updated base_task_id to {runner.current_task_id} after fallback"
            )

        # Capture initial state for entity diff evaluation if specs are present
        entity_diff_specs = scenario_dict.get("entity_diff_specs")
        if entity_diff_specs:
            from mmtoolsandbox.common.entity_diff_evaluator import (
                EntityDiffEvalConfig,
                EntityDiffEvaluator,
            )

            eval_config = EntityDiffEvalConfig.from_dict({"specs": entity_diff_specs})
            evaluator = EntityDiffEvaluator(eval_config)
            evaluator.capture_initial_state(
                execution_context=scenario.starting_context,
                bridge=bridge,
            )
            result["entity_diff_evaluator"] = evaluator

        # Note: Actual scenario.play(roles) would happen here
        # The caller can implement their own execution loop

        # Get Appworld evaluation if available
        if bridge is not None and bridge.is_initialized:
            try:
                eval_result = bridge.evaluate()
                result["appworld_evaluation"] = eval_result
            except Exception as e:
                result["evaluation_error"] = str(e)

        result["execution_context"] = scenario.starting_context

    return result


def create_cross_environment_scenario(
    scenario_dict: dict[str, Any],
    toolbox: Any | None = None,
    include_appworld_tools: bool = True,
) -> Any:
    """
    Create a complete Scenario from a cross-environment scenario dict.

    This is the main entry point for running cross-environment tasks loaded
    from JSON files. It handles:
    1. Creating the base MMToolSandbox scenario with correct base variant
    2. Loading MMToolSandbox base state (contact, calendars, etc.)
    3. Applying MMToolSandbox entity modifications (starting_state)
    4. Setting up the combined toolbox (MMToolSandbox + Appworld)

    The returned scenario can then be run with:
        runner = get_scenario_runner()
        with runner.run_scenario(
            base_task_id=scenario_dict["base_task_id"],
            database_modifications=scenario_dict["database_modifications"],
            execution_context=scenario.starting_context,
        ) as bridge:
            scenario.play(roles)

    Args:
        scenario_dict: Scenario dict from task_adapters (e.g., from load_cross_environment_task)
        toolbox: Optional toolbox (creates combined toolbox if None)
        include_appworld_tools: If True, include Appworld tools in toolbox

    Returns:
        Scenario object with base state loaded and entity modifications applied
    """
    # Create base scenario with base state loaded
    scenario = _initialize_scenario_base(
        toolbox=toolbox,
        include_appworld_tools=include_appworld_tools,
        support_images=False,
    )

    # Apply MMToolSandbox entity modifications (starting_state)
    starting_state = scenario_dict.get("starting_state")
    if starting_state:
        AppWorldScenarioRunner.apply_starting_state(
            scenario.starting_context,
            starting_state,
        )

    # Add task messages to conversation
    messages = scenario_dict.get("messages", [])
    if messages:
        scenario.starting_context.add_to_database(
            namespace=DatabaseNamespace.SANDBOX,
            rows=messages,
        )

    # Set tool allow list
    tool_allow_list = scenario_dict.get("tool_allow_list")
    if tool_allow_list:
        # Always include end_conversation
        if "end_conversation" not in tool_allow_list:
            tool_allow_list = ["end_conversation"] + tool_allow_list
        scenario.starting_context.tool_allow_list = tool_allow_list

    return scenario


def run_cross_environment_task(
    scenario_dict: dict[str, Any],
    roles: dict[str, Any] | None = None,
    toolbox: Any | None = None,
) -> dict[str, Any]:
    """
    Complete end-to-end execution of a cross-environment task.

    This function handles the entire lifecycle:
    1. Creates MMToolSandbox scenario with correct base
    2. Applies MMToolSandbox entity modifications
    3. Initializes Appworld with base task
    4. Applies Appworld database modifications
    5. Returns results including Appworld evaluation

    Args:
        scenario_dict: Scenario dict from load_cross_environment_task()
        roles: Optional role mappings for scenario execution
        toolbox: Optional toolbox

    Returns:
        Dictionary with execution results including:
        - scenario_name: Name of the executed scenario
        - appworld_evaluation: Appworld evaluation result (if applicable)
        - execution_context: Final execution context

    Example:
        from mmtoolsandbox.appworld.scenarios.task_adapters import (
            load_cross_environment_task,
        )
        from mmtoolsandbox.appworld.scenarios.runner import (
            run_cross_environment_task,
        )

        # Load task from JSON
        scenario_dict = load_cross_environment_task("path/to/task.json")

        # Run the task
        result = run_cross_environment_task(scenario_dict)
    """
    # Create scenario with base state loaded
    scenario = create_cross_environment_scenario(scenario_dict, toolbox)

    # Get Appworld configuration
    base_task_id = scenario_dict.get("base_task_id")
    database_modifications = scenario_dict.get("database_modifications")
    setup_code = scenario_dict.get("setup_code")

    runner = get_scenario_runner()

    result = {
        "scenario_name": scenario_dict.get("name"),
        "agentsandbox_base": scenario_dict.get("agentsandbox_base", "base"),
        "appworld_base_task": base_task_id,
    }

    with runner.run_scenario(
        base_task_id=base_task_id,
        database_modifications=database_modifications,
        setup_code=setup_code,
        # starting_state already applied in create_cross_environment_scenario
        execution_context=scenario.starting_context,
    ) as bridge:
        result["bridge_initialized"] = bridge is not None

        # Note: Actual scenario.play(roles) would happen here
        # The caller can implement their own execution loop

        # Get Appworld evaluation if available
        if bridge is not None and bridge.is_initialized:
            try:
                eval_result = bridge.evaluate()
                result["appworld_evaluation"] = eval_result
            except Exception as e:
                result["evaluation_error"] = str(e)

        result["execution_context"] = scenario.starting_context

    return result
