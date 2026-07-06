"""
AppWorld Scenarios Module for MMToolSandbox.

This module provides utilities to convert AppWorld tasks to MMToolSandbox
scenarios, enabling AppWorld tasks to run in MMToolSandbox's conversational
framework.

Key components:
- task_adapters: Convert AppWorld tasks to MMToolSandbox scenario format
- runner: Manage AppWorld lifecycle during scenario execution
"""

from mmtoolsandbox.appworld.scenarios.runner import (
    AppWorldScenarioHooks,
    AppWorldScenarioRunner,
    create_appworld_scenario_extension,
    get_scenario_runner,
    reset_scenario_runner,
    run_appworld_scenario,
)
from mmtoolsandbox.appworld.scenarios.task_adapters import (
    CROSS_SYSTEM_TEMPLATES,
    AppWorldScenarioConfig,
    appworld_task_to_scenario,
    create_cross_system_scenario,
    create_from_template,
    load_appworld_scenarios,
)

__all__ = [
    # Task adapters
    "AppWorldScenarioConfig",
    "appworld_task_to_scenario",
    "load_appworld_scenarios",
    "create_cross_system_scenario",
    "create_from_template",
    "CROSS_SYSTEM_TEMPLATES",
    # Runner
    "AppWorldScenarioRunner",
    "AppWorldScenarioHooks",
    "get_scenario_runner",
    "reset_scenario_runner",
    "run_appworld_scenario",
    "create_appworld_scenario_extension",
]
