# Copyright © 2026 Apple Inc.

"""Unified scenario factory for MMToolSandbox.

Provides scenario factories for all dataset levels:
- ``FULL``: ~500 fine-grained tools
- ``MEDIUM``: ~300 consolidated tools (CRUD merging, cross-app dedup)
- ``COMPACT``: ~165 further-consolidated tools
- ``MINI``: ~30 workflow-based mega-tools

All levels use ``ScenarioExtension.get_extended_scenario()`` to produce
``Scenario`` instances with a ``runtime_context_factory`` for per-scenario
bridge setup in spawned worker processes.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any, Mapping

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution_context import (
    RoleType,
    ScenarioCategories,
)
from mmtoolsandbox.common.i18n import DefaultLocalizer, Locale, Localizer
from mmtoolsandbox.common.image_id import ImageId
from mmtoolsandbox.common.prompt_templates import (
    SECTION_UI_USER_INTERACTION,
    SECTION_USER_IMAGE_DELIVERY_BLIND,
    SECTION_USER_IMAGE_VIEWING,
    UI_USER_TOOLS,
    USER_INSTRUCTION,
    USER_INSTRUCTION_SCRIPTED,
    compose_agent_prompt,
    get_challenge_type_instructions,
)
from mmtoolsandbox.common.scenario import Scenario, ScenarioExtension
from mmtoolsandbox.datasets.names import DatasetName
from mmtoolsandbox.datasets.registry import (
    DatasetRegistryEntry,
    register_datasets,
)
from mmtoolsandbox.toolbox.toolbox import Toolbox

LOGGER = logging.getLogger(__name__)


def _rebind_toolbox(ctx: Any, toolbox: Toolbox) -> None:
    """Replace the toolbox on an execution context and rebuild derived state."""
    ctx.toolbox = toolbox
    ctx.name_to_tool = {tool.__name__: tool for tool in toolbox.tools}


# ---------------------------------------------------------------------------
# Scenario utilities
# ---------------------------------------------------------------------------


def update_scenarios_empty_toolist(
    scenarios: list[ScenarioExtension],
) -> list[ScenarioExtension]:
    """Create copies of scenarios with empty tool-allow lists.

    Useful for code execution mode where tools are pre-loaded into the
    REPL and ``tool_allow_list`` is not used for filtering.

    Args:
        scenarios: Scenario extensions to copy.

    Returns:
        Deep copies of the input scenarios with ``tool_allow_list`` set to
        ``[]`` and ``"_empty_tool_list"`` appended to each name.
    """
    import copy

    new_scenarios = []
    for s in scenarios:
        s = copy.deepcopy(s)
        s.name = s.name + "_empty_tool_list"
        s.tool_allow_list = []
        new_scenarios.append(s)
    return new_scenarios


# ---------------------------------------------------------------------------
# ScenarioDataSchema — JSON ↔ ScenarioExtension conversion
# ---------------------------------------------------------------------------


def _resolve_constants(text: str, *, scripted: bool = False) -> str:
    """Resolve template constants in text.

    Args:
        text: Text possibly containing ``{{USER_INSTRUCTION}}`` placeholders.
        scripted: When True, resolve to the structured supervisor
            prompt (``USER_INSTRUCTION_SCRIPTED``).  When False (default),
            resolve to the simple prompt (``USER_INSTRUCTION``).
    """
    if isinstance(text, str):
        prompt = USER_INSTRUCTION_SCRIPTED if scripted else USER_INSTRUCTION
        return text.replace("{{USER_INSTRUCTION}}", prompt)
    return text


def _reverse_resolve_constants(text: str) -> str:
    """Reverse-resolve constants back to template placeholders."""
    if isinstance(text, str):
        # Try scripted first (longer string avoids partial matches)
        text = text.replace(USER_INSTRUCTION_SCRIPTED, "{{USER_INSTRUCTION}}")
        text = text.replace(USER_INSTRUCTION, "{{USER_INSTRUCTION}}")
    return text


@dataclasses.dataclass
class ScenarioDataSchema:
    """Data schema for JSON-serialized scenarios.

    Core fields (used by all scenarios):
        name, base_scenario, messages, milestones, image_paths,
        tool_allow_list, task_completion_criteria.

    Extended fields (used by visual tool calling scenarios):
        categories, max_messages, metadata, appworld_entities,
        agentsandbox_entities, appworld_base_task, device_state_id,
        reference_time, description, difficulty.
    """

    name: str
    base_scenario: str
    image_paths: list[str] | None = None
    messages: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    tool_allow_list: list[str] | None = None
    milestones: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    task_completion_criteria: str | None = None
    # Extended fields
    categories: list[str] | None = None
    max_messages: int | None = None
    metadata: dict[str, Any] | None = None
    appworld_entities: dict[str, list[dict[str, Any]]] | None = None
    agentsandbox_entities: dict[str, list[dict[str, Any]]] | None = None
    appworld_base_task: str | None = None
    device_state_id: str | None = None
    reference_time: str | None = None
    description: str | None = None
    difficulty: str | None = None
    entity_diff_specs: list[dict[str, Any]] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScenarioDataSchema":
        """Create a ScenarioDataSchema from a raw JSON dictionary.

        Args:
            data: Dictionary parsed from a scenario JSON file. Must contain
                ``"name"`` and ``"base_scenario"`` keys; all other fields
                are optional.

        Returns:
            A populated ScenarioDataSchema instance.
        """
        return cls(
            name=data["name"],
            base_scenario=data["base_scenario"],
            image_paths=data.get("image_paths"),
            messages=data.get("messages", []),
            tool_allow_list=data.get("tool_allow_list"),
            milestones=data.get("milestones", []),
            task_completion_criteria=data.get("task_completion_criteria"),
            categories=data.get("categories"),
            max_messages=data.get("max_messages"),
            metadata=data.get("metadata"),
            appworld_entities=data.get("appworld_entities"),
            agentsandbox_entities=data.get("agentsandbox_entities"),
            appworld_base_task=data.get("appworld_base_task"),
            device_state_id=data.get("device_state_id"),
            reference_time=data.get("reference_time"),
            description=data.get("description"),
            difficulty=data.get("difficulty"),
            entity_diff_specs=data.get("entity_diff_specs"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Optional fields that are ``None`` are omitted from the output.

        Returns:
            Dictionary suitable for ``json.dump()``.
        """
        data: dict[str, Any] = {
            "name": self.name,
            "base_scenario": self.base_scenario,
            "messages": self.messages,
            "milestones": self.milestones,
        }
        if self.image_paths:
            data["image_paths"] = self.image_paths
        if self.tool_allow_list:
            data["tool_allow_list"] = self.tool_allow_list
        if self.task_completion_criteria:
            data["task_completion_criteria"] = self.task_completion_criteria
        if self.categories:
            data["categories"] = self.categories
        if self.max_messages is not None:
            data["max_messages"] = self.max_messages
        if self.metadata:
            data["metadata"] = self.metadata
        if self.appworld_entities:
            data["appworld_entities"] = self.appworld_entities
        if self.agentsandbox_entities:
            data["agentsandbox_entities"] = self.agentsandbox_entities
        if self.appworld_base_task:
            data["appworld_base_task"] = self.appworld_base_task
        if self.device_state_id:
            data["device_state_id"] = self.device_state_id
        if self.reference_time:
            data["reference_time"] = self.reference_time
        if self.description:
            data["description"] = self.description
        if self.difficulty:
            data["difficulty"] = self.difficulty
        if self.entity_diff_specs:
            data["entity_diff_specs"] = self.entity_diff_specs
        return data

    def to_scenario_extension(
        self,
        base_scenarios: Mapping[str, Scenario],
        *,
        scripted_user_prompt: bool = False,
    ) -> ScenarioExtension:
        """Convert to a runtime ScenarioExtension.

        Resolves string role types to ``RoleType`` enums, converts integer
        image IDs to ``ImageId``, maps category strings to
        ``ScenarioCategories``, and packs extended fields into
        ``runtime_metadata``.

        Args:
            base_scenarios: Mapping of base scenario names to ``Scenario``
                instances. The ``base_scenario`` field must be a key in
                this mapping.
            scripted_user_prompt: When True, resolve ``{{USER_INSTRUCTION}}``
                to the structured supervisor prompt.  When False,
                resolve to the simple prompt.

        Returns:
            A ScenarioExtension ready for ``get_extended_scenario()``.

        Raises:
            ValueError: If ``base_scenario`` is not found in
                ``base_scenarios``.
        """
        if self.base_scenario not in base_scenarios:
            raise ValueError(f"Base scenario '{self.base_scenario}' not found.")

        # Resolve messages
        challenge_type = None
        if self.metadata and "scenario_metadata" in self.metadata:
            challenge_type = self.metadata["scenario_metadata"].get("challenge_type")

        messages = []
        for msg in self.messages:
            resolved_msg = msg.copy()
            resolved_msg["sender"] = RoleType(msg["sender"])
            resolved_msg["recipient"] = RoleType(msg["recipient"])
            resolved_msg["content"] = _resolve_constants(
                msg["content"], scripted=scripted_user_prompt
            )
            # Inject blind image delivery instructions when the scenario
            # has images the user must deliver via send_message_with_image.
            if (
                self.image_paths
                and resolved_msg["sender"] == RoleType.SYSTEM
                and resolved_msg["recipient"] == RoleType.USER
                and isinstance(resolved_msg["content"], str)
                and "## YOUR TASK" in resolved_msg["content"]
            ):
                resolved_msg["content"] = resolved_msg["content"].replace(
                    "## YOUR TASK",
                    SECTION_USER_IMAGE_DELIVERY_BLIND + "\n\n## YOUR TASK",
                )
            # Inject challenge-type-specific guardrail instructions into
            # the user simulator system prompt (SYSTEM→USER message).
            if (
                challenge_type
                and resolved_msg["sender"] == RoleType.SYSTEM
                and resolved_msg["recipient"] == RoleType.USER
                and isinstance(resolved_msg["content"], str)
            ):
                section = get_challenge_type_instructions(challenge_type)
                if section:
                    resolved_msg["content"] = resolved_msg["content"].replace(
                        "## YOUR TASK", section + "## YOUR TASK"
                    )
            if "image_ids" in msg:
                resolved_msg["image_ids"] = [ImageId(i) for i in msg["image_ids"]]
            messages.append(resolved_msg)

        # Resolve categories
        categories = []
        if self.categories:
            for cat in self.categories:
                try:
                    categories.append(ScenarioCategories(cat))
                except ValueError:
                    pass  # Skip unknown categories

        # Build runtime metadata from extended fields
        runtime_metadata: dict[str, Any] | None = None
        extended_keys = {
            "appworld_entities": self.appworld_entities,
            "agentsandbox_entities": self.agentsandbox_entities,
            "appworld_base_task": self.appworld_base_task,
            "device_state_id": self.device_state_id,
            "description": self.description,
            "difficulty": self.difficulty,
            "entity_diff_specs": self.entity_diff_specs,
        }
        # Merge explicit metadata dict first, then overlay extended keys
        if self.metadata:
            runtime_metadata = dict(self.metadata)
        for key, value in extended_keys.items():
            if value is not None:
                if runtime_metadata is None:
                    runtime_metadata = {}
                runtime_metadata[key] = value

        return ScenarioExtension(
            name=self.name,
            base_scenario=base_scenarios[self.base_scenario],
            image_paths=self.image_paths,
            messages=messages,
            tool_allow_list=self.tool_allow_list,
            task_completion_criteria=self.task_completion_criteria,
            categories=categories,
            max_messages=self.max_messages,
            reference_time=self.reference_time,
            runtime_metadata=runtime_metadata,
        )

    @classmethod
    def from_scenario_extension(
        cls, extension: ScenarioExtension, base_scenarios: Mapping[str, Scenario]
    ) -> "ScenarioDataSchema":
        """Create a ScenarioDataSchema from a runtime ScenarioExtension.

        Reverses ``to_scenario_extension``: converts ``RoleType`` enums back
        to strings, ``ImageId`` to ints, and unpacks ``runtime_metadata``
        into the extended fields.

        Args:
            extension: The ScenarioExtension to serialize.
            base_scenarios: Mapping of names to base ``Scenario`` objects,
                used to look up the base scenario's name.

        Returns:
            A ScenarioDataSchema suitable for JSON serialization via
            ``to_dict()``.

        Raises:
            ValueError: If the extension's base scenario is not found
                in ``base_scenarios``.
        """
        # Find base scenario name
        base_scenario_name = None
        for name, scenario in base_scenarios.items():
            if extension.base_scenario == scenario:
                base_scenario_name = name
                break

        if base_scenario_name is None:
            raise ValueError("Base scenario not found in provided base_scenarios map.")

        # Dump messages
        messages = []
        for msg in extension.messages:
            dumped_msg = msg.copy()
            dumped_msg["sender"] = str(msg["sender"])
            dumped_msg["recipient"] = str(msg["recipient"])
            dumped_msg["content"] = _reverse_resolve_constants(str(msg["content"]))
            if "image_ids" in msg:
                dumped_msg["image_ids"] = [int(i) for i in msg["image_ids"]]  # type: ignore[assignment]
            messages.append(dumped_msg)

        # Extract extended fields from runtime_metadata
        rm = extension.runtime_metadata or {}

        return cls(
            name=extension.name,
            base_scenario=base_scenario_name,
            image_paths=extension.image_paths,
            messages=messages,
            tool_allow_list=extension.tool_allow_list,
            milestones=[],
            task_completion_criteria=extension.task_completion_criteria,
            categories=[str(c) for c in extension.categories]
            if extension.categories
            else None,
            max_messages=extension.max_messages,
            reference_time=extension.reference_time,
            metadata={
                k: v
                for k, v in rm.items()
                if k
                not in {
                    "appworld_entities",
                    "agentsandbox_entities",
                    "appworld_base_task",
                    "device_state_id",
                    "description",
                    "difficulty",
                    "entity_diff_specs",
                }
            }
            or None,
            appworld_entities=rm.get("appworld_entities"),
            agentsandbox_entities=rm.get("agentsandbox_entities"),
            appworld_base_task=rm.get("appworld_base_task"),
            device_state_id=rm.get("device_state_id"),
            description=rm.get("description"),
            difficulty=rm.get("difficulty"),
            entity_diff_specs=rm.get("entity_diff_specs"),
        )


# ---------------------------------------------------------------------------
# Scenario factory
# ---------------------------------------------------------------------------


def _create_execution_context(
    config: dict[str, Any],
) -> Any:
    """Create a lightweight execution context for the FULL dataset.

    The full toolbox is loaded via ``get_all_tools(DatasetName.FULL)``
    which auto-registers all app tools into the core registry.
    """
    import mmtoolsandbox.tools.code_execution  # noqa: F401
    from mmtoolsandbox.common.execution_context import ExecutionContext
    from mmtoolsandbox.toolbox.loading import load_toolbox

    toolbox = load_toolbox(DatasetName.FULL, {})

    return ExecutionContext(
        toolbox=toolbox,
        tool_allow_list=None,
        delay_initialization=True,
        support_images=True,
    )


def _create_base_scenarios(
    toolbox: Toolbox,
    locale: Locale = Locale.en_US,
    localize: Localizer = DefaultLocalizer,
) -> dict[str, Scenario]:
    """Create the base scenario for the FULL dataset.

    Lightweight context (settings, calendars, calendar events) — entities
    are staged per-scenario from JSON metadata.
    """
    from mmtoolsandbox.common.evaluation import EvaluationCriteria
    from mmtoolsandbox.common.execution_context import ScenarioCategories as SC
    from mmtoolsandbox.common.message_conversion import (
        Message,
        add_messages_to_execution_context,
    )
    from mmtoolsandbox.datasets.initial_database_states.base import (
        calendar_events_initial_database_state,
        calendars_initial_database_state,
        setting_initial_database_state,
    )

    starting_context = _create_execution_context({})
    starting_context.trace_tool = True

    _DEFAULT_AGENT_PROMPT = compose_agent_prompt(
        enable_tool_search=True, enable_coding_tool=True
    )
    add_messages_to_execution_context(
        starting_context,
        [
            Message(
                sender=RoleType.SYSTEM,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content="PLACEHOLDER_IMPORTS",
                conversation_active=False,
            ),
            Message(
                sender=RoleType.SYSTEM,
                recipient=RoleType.AGENT,
                content=_DEFAULT_AGENT_PROMPT,
                conversation_active=True,
            ),
        ],
    )

    for ns, loader_fn in [
        (DatabaseNamespace.SETTING, setting_initial_database_state),
        (DatabaseNamespace.CALENDARS, calendars_initial_database_state),
        (DatabaseNamespace.CALENDAR_EVENTS, calendar_events_initial_database_state),
    ]:
        try:
            rows = loader_fn(DefaultLocalizer)
            starting_context.add_to_database(namespace=ns, rows=rows)
        except Exception as e:
            LOGGER.warning("Failed to load base state for %s: %s", ns, e)

    base_scenario = Scenario(
        starting_context=starting_context,
        evaluation_criteria=EvaluationCriteria(),
        max_messages=60,
        categories=[SC.VISION_SYNTHETIC],
    )

    return {"base": base_scenario}


def create_scenarios(
    base_scenarios: dict[str, Scenario],
    toolbox: Toolbox,
    config: dict[str, Any],
    locale: Locale,
    localize: Localizer,
) -> dict[str, Scenario]:
    """Scenario factory for the FULL dataset.

    Loads scenarios from individual JSON files in ``config["scenario_dir"]``,
    converts each via ``ScenarioDataSchema``, and attaches the
    runtime context factory for per-scenario bridge lifecycle.

    Args:
        base_scenarios: Named base scenarios providing starting context.
        toolbox: Toolbox with registered tools for this dataset.
        config: Dataset configuration. Recognized keys:
            ``scenario_dir``: Path to directory of scenario JSON files.
            ``auto_login``: If True, all apps are pre-authenticated.
            ``enable_ui``: If True, inject A2UI tools and UI prompts.
            ``limit``: Optional max number of scenarios to load.
        locale: Locale for localized scenario content.
        localize: Localizer callable for string localization.

    Returns:
        Mapping of scenario names to fully instantiated ``Scenario`` objects,
        each with ``runtime_context_factory`` set to
        ``appworld_runtime_context``.

    Raises:
        ValueError: If ``scenario_dir`` is set but does not exist.
    """
    from mmtoolsandbox.appworld.runtime import appworld_runtime_context

    scenario_dir_str = config.get("scenario_dir")
    if not scenario_dir_str:
        return {}

    scenario_dir = Path(scenario_dir_str)
    if not scenario_dir.is_dir():
        raise ValueError(f"scenario_dir does not exist: {scenario_dir}")

    auto_login = config.get("auto_login", False)
    enable_ui = config.get("enable_ui", False)
    limit = config.get("limit")
    image_base_path_str = config.get("image_base_path")
    image_base_path = Path(image_base_path_str) if image_base_path_str else None

    json_files = sorted(scenario_dir.glob("*.json"))

    # Optional name pre-filter (passed in via the CLI's `-s` flag) so we can
    # skip JSON parsing AND per-scenario image loads/resizes for files we
    # won't run. Names are matched against the JSON filename stem, which is
    # also the scenario name written into each file.
    name_filter = config.get("scenario_names")
    if name_filter:
        wanted = set(name_filter)
        json_files = [p for p in json_files if p.stem in wanted]

    if limit is not None:
        json_files = json_files[:limit]

    LOGGER.info("Loading %d scenario files from %s", len(json_files), scenario_dir)

    scenarios: dict[str, Scenario] = {}
    for json_path in json_files:
        try:
            with open(json_path) as f:
                data = json.load(f)

            schema = ScenarioDataSchema.from_dict(data)

            # Resolve relative image_paths against image_base_path so the
            # downstream image loader sees absolute paths only.
            if schema.image_paths and image_base_path is not None:
                schema.image_paths = [
                    p if Path(p).is_absolute() else str(image_base_path / p)
                    for p in schema.image_paths
                ]

            extension = schema.to_scenario_extension(
                base_scenarios, scripted_user_prompt=True
            )

            if extension.runtime_metadata is not None:
                if auto_login:
                    extension.runtime_metadata["auto_login"] = True

            # Ensure send_message_with_image is available to user simulator
            if extension.user_tool_allow_list is None:
                extension.user_tool_allow_list = []
            if "send_message_with_image" not in extension.user_tool_allow_list:
                extension.user_tool_allow_list.append("send_message_with_image")

            # UI injection
            if enable_ui:
                for tool_name in UI_USER_TOOLS:
                    if tool_name not in extension.user_tool_allow_list:
                        extension.user_tool_allow_list.append(tool_name)

                for msg in extension.messages:
                    if (
                        msg.get("sender") == RoleType.SYSTEM
                        and msg.get("recipient") == RoleType.USER
                    ):
                        content = str(msg.get("content", ""))
                        insertion = SECTION_UI_USER_INTERACTION
                        if extension.image_paths:
                            insertion += SECTION_USER_IMAGE_VIEWING
                        if "## YOUR TASK" in content:
                            content = content.replace(
                                "## YOUR TASK",
                                insertion + "\n\n## YOUR TASK",
                            )
                        else:
                            content += insertion
                        msg["content"] = content
                        break

                extension.enable_ui = True

            for name, scenario in extension.get_extended_scenario().items():
                if scenario.reference_time:
                    import datetime as _dt

                    from mmtoolsandbox.common.relative_time import (
                        resolve_execution_context_relative_time,
                    )

                    try:
                        ref_dt = _dt.datetime.fromisoformat(
                            scenario.reference_time.replace("Z", "+00:00")
                        )
                        if ref_dt.tzinfo is None:
                            ref_dt = ref_dt.replace(tzinfo=_dt.timezone.utc)
                    except (ValueError, TypeError):
                        ref_dt = _dt.datetime.now(tz=_dt.timezone.utc)
                    scenario.starting_context = resolve_execution_context_relative_time(
                        scenario.starting_context, ref_dt
                    )

                scenario.runtime_context_factory = appworld_runtime_context
                scenarios[name] = scenario

        except Exception as e:
            LOGGER.warning("Failed to load %s: %s", json_path.name, e)
            continue

    LOGGER.info("Loaded %d scenarios", len(scenarios))
    return scenarios


# ---------------------------------------------------------------------------
# Dataset registration
# ---------------------------------------------------------------------------

register_datasets(
    dataset_names=DatasetName.FULL,
    version="1.0.0",
    entry=DatasetRegistryEntry(
        scenario_factory=create_scenarios,
        base_scenario_factory=_create_base_scenarios,
    ),
)

# ---------------------------------------------------------------------------
# MEDIUM dataset — consolidated toolbox variant of FULL
# ---------------------------------------------------------------------------


def _build_medium_remap_table() -> dict[str, str]:
    """Build a mapping from original tool names to consolidated MEDIUM names.

    Lazily imports absorbed tool names from each consolidated module.
    """
    import mmtoolsandbox.tools.consolidated.alarms  # noqa: F401
    import mmtoolsandbox.tools.consolidated.amazon  # noqa: F401
    import mmtoolsandbox.tools.consolidated.calendar_event  # noqa: F401
    import mmtoolsandbox.tools.consolidated.contacts  # noqa: F401
    import mmtoolsandbox.tools.consolidated.cross_app_account  # noqa: F401
    import mmtoolsandbox.tools.consolidated.cross_app_notification  # noqa: F401
    import mmtoolsandbox.tools.consolidated.cross_app_payment  # noqa: F401
    import mmtoolsandbox.tools.consolidated.file_system  # noqa: F401
    import mmtoolsandbox.tools.consolidated.gmail  # noqa: F401
    import mmtoolsandbox.tools.consolidated.reminder  # noqa: F401
    import mmtoolsandbox.tools.consolidated.settings  # noqa: F401
    import mmtoolsandbox.tools.consolidated.simple_note  # noqa: F401
    import mmtoolsandbox.tools.consolidated.splitwise  # noqa: F401
    import mmtoolsandbox.tools.consolidated.spotify  # noqa: F401
    import mmtoolsandbox.tools.consolidated.todoist  # noqa: F401
    import mmtoolsandbox.tools.consolidated.venmo  # noqa: F401
    from mmtoolsandbox.tools.consolidated import get_absorbed_tool_names

    absorbed = get_absorbed_tool_names()
    # For scenarios that specify a tool_allow_list, we simply drop absorbed
    # tools from the list — the agent will use the consolidated version
    # instead, and consolidated tools are always available (not gated by
    # tool_allow_list in MEDIUM scenarios since scenarios
    # typically use tool_allow_list=None).
    return {name: "" for name in absorbed}


def _remap_scenario_tool_allow_list(
    tool_allow_list: list[str] | None,
    absorbed: set[str],
) -> list[str] | None:
    """Remove absorbed tool names from a tool_allow_list."""
    if tool_allow_list is None:
        return None
    return [name for name in tool_allow_list if name not in absorbed]


def create_medium_scenarios(
    base_scenarios: dict[str, Scenario],
    toolbox: Toolbox,
    config: dict[str, Any],
    locale: Locale,
    localize: Localizer,
) -> dict[str, Scenario]:
    """Scenario factory for the MEDIUM dataset.

    Delegates to ``create_scenarios`` then remaps tool_allow_list
    entries so absorbed tools are removed (the consolidated replacements
    are always available in the MEDIUM toolbox).
    """
    scenarios = create_scenarios(base_scenarios, toolbox, config, locale, localize)

    remap = _build_medium_remap_table()
    absorbed = set(remap.keys())

    for scenario in scenarios.values():
        ctx = scenario.starting_context
        _rebind_toolbox(ctx, toolbox)
        ctx.tool_allow_list = _remap_scenario_tool_allow_list(
            ctx.tool_allow_list, absorbed
        )
        for role, allow_list in (ctx.role_tool_allow_list or {}).items():
            ctx.role_tool_allow_list[role] = [
                name for name in allow_list if name not in absorbed
            ]

    return scenarios


register_datasets(
    dataset_names=DatasetName.MEDIUM,
    version="1.0.0",
    entry=DatasetRegistryEntry(
        scenario_factory=create_medium_scenarios,
        base_scenario_factory=_create_base_scenarios,
    ),
)


# ---------------------------------------------------------------------------
# COMPACT dataset — further consolidated toolbox variant of MEDIUM
# ---------------------------------------------------------------------------


def _build_compact_absorbed() -> set[str]:
    """Return all tool names absorbed at the MEDIUM + COMPACT levels."""
    from mmtoolsandbox.common.tool_registry import get_all_tools
    from mmtoolsandbox.tools.compact import _COMPACT_ABSORBED_TOOL_NAMES
    from mmtoolsandbox.tools.consolidated import get_absorbed_tool_names

    # Trigger COMPACT imports so _COMPACT_ABSORBED_TOOL_NAMES is populated.
    get_all_tools(DatasetName.COMPACT)
    return get_absorbed_tool_names() | _COMPACT_ABSORBED_TOOL_NAMES


def create_compact_scenarios(
    base_scenarios: dict[str, Scenario],
    toolbox: Toolbox,
    config: dict[str, Any],
    locale: Locale,
    localize: Localizer,
) -> dict[str, Scenario]:
    """Scenario factory for the COMPACT dataset."""
    scenarios = create_scenarios(base_scenarios, toolbox, config, locale, localize)

    absorbed = _build_compact_absorbed()

    for scenario in scenarios.values():
        ctx = scenario.starting_context
        _rebind_toolbox(ctx, toolbox)
        ctx.tool_allow_list = _remap_scenario_tool_allow_list(
            ctx.tool_allow_list, absorbed
        )
        for role, allow_list in (ctx.role_tool_allow_list or {}).items():
            ctx.role_tool_allow_list[role] = [
                name for name in allow_list if name not in absorbed
            ]

    return scenarios


register_datasets(
    dataset_names=DatasetName.COMPACT,
    version="1.0.0",
    entry=DatasetRegistryEntry(
        scenario_factory=create_compact_scenarios,
        base_scenario_factory=_create_base_scenarios,
    ),
)


# ---------------------------------------------------------------------------
# MINI dataset — workflow-based toolbox variant
# ---------------------------------------------------------------------------


def _build_mini_absorbed() -> set[str]:
    """Return all tool names NOT present in the MINI toolbox."""
    from mmtoolsandbox.common.tool_registry import get_all_tools

    full_tools = get_all_tools(DatasetName.FULL)
    mini_tools = get_all_tools(DatasetName.MINI)
    return set(full_tools.keys()) - set(mini_tools.keys())


def create_mini_scenarios(
    base_scenarios: dict[str, Scenario],
    toolbox: Toolbox,
    config: dict[str, Any],
    locale: Locale,
    localize: Localizer,
) -> dict[str, Scenario]:
    """Scenario factory for the MINI dataset."""
    scenarios = create_scenarios(base_scenarios, toolbox, config, locale, localize)

    absorbed = _build_mini_absorbed()

    for scenario in scenarios.values():
        ctx = scenario.starting_context
        _rebind_toolbox(ctx, toolbox)
        ctx.tool_allow_list = _remap_scenario_tool_allow_list(
            ctx.tool_allow_list, absorbed
        )
        for role, allow_list in (ctx.role_tool_allow_list or {}).items():
            ctx.role_tool_allow_list[role] = [
                name for name in allow_list if name not in absorbed
            ]

    return scenarios


register_datasets(
    dataset_names=DatasetName.MINI,
    version="1.0.0",
    entry=DatasetRegistryEntry(
        scenario_factory=create_mini_scenarios,
        base_scenario_factory=_create_base_scenarios,
    ),
)
