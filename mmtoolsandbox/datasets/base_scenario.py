# Copyright © 2026 Apple Inc.

from mmtoolsandbox.common.execution_context import (
    DatabaseNamespace,
    ExecutionContext,
    RoleType,
)
from mmtoolsandbox.common.i18n import DefaultLocalizer, Locale, Localizer
from mmtoolsandbox.common.scenario import Scenario
from mmtoolsandbox.toolbox.toolbox import Toolbox


def create_base_scenario(
    toolbox: Toolbox,
    locale: Locale = Locale.en_US,
    localize: Localizer = DefaultLocalizer,
    add_agent_system_prompt: bool = True,
    support_images: bool = False,
) -> Scenario:
    """Create a base scenario with system messages and tool tracing enabled.

    The returned scenario is incomplete — it has no evaluation criteria or
    user/agent messages.  Callers extend it via ``ScenarioExtension``.

    Args:
        toolbox: The toolbox to use.
        locale: Locale of the base scenario.
        localize: Localizer instance used to localize scenarios.
        add_agent_system_prompt: Whether to add a system prompt for the agent.
        support_images: Whether to support images in messages.

    Returns:
        The base scenario.
    """
    scenario = Scenario(
        starting_context=ExecutionContext(
            toolbox,
            delay_initialization=True,
            support_images=support_images,
        )
    )
    scenario.starting_context.trace_tool = True

    rows = [
        {
            "sender": RoleType.SYSTEM,
            "recipient": RoleType.EXECUTION_ENVIRONMENT,
            "content": toolbox.create_import_statement(),
        },
    ]
    if add_agent_system_prompt:
        agent_system_prompt = (
            "Don't make assumptions about what values to plug into functions. "
            + "Ask for clarification if a user request is ambiguous."
        )
        if locale != Locale.en_US:
            agent_system_prompt += f" The user's locale is {locale.name}."
        rows += [
            {
                "sender": RoleType.SYSTEM,
                "recipient": RoleType.AGENT,
                "content": agent_system_prompt,
            },
        ]

    scenario.starting_context.add_to_database(
        namespace=DatabaseNamespace.SANDBOX, rows=rows
    )
    scenario.starting_context.tool_allow_list = ["end_conversation"]
    return scenario
