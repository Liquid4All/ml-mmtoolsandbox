# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""Consolidated Reminder tools for the MEDIUM toolbox.

CRUD consolidation for reminders: add, modify, delete merged into
a single ``manage_reminder`` tool.
"""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.toolbox.names import ToolboxName
from mmtoolsandbox.tools.consolidated import mark_tools_absorbed_by


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.tool_sandbox.reminder as m

    return getattr(m, name)


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces={DatabaseNamespace.REMINDER},
    visible_to=(RoleType.AGENT,),
)
def manage_reminder(
    action: Literal["add", "modify", "delete"],
    reminder_id: str | NotGiven = NOT_GIVEN,
    content: str | NotGiven = NOT_GIVEN,
    reminder_datetime: str | NotGiven = NOT_GIVEN,
    latitude: float | None | NotGiven = NOT_GIVEN,
    longitude: float | None | NotGiven = NOT_GIVEN,
) -> str | None:
    """Manage reminders: add, modify, or delete.

    Actions:
        add: Create a new reminder. Requires content and reminder_datetime.
            Optionally include latitude and longitude for location-based
            reminders.
        modify: Update an existing reminder. Requires reminder_id and at
            least one of content, reminder_datetime, latitude, or longitude.
        delete: Remove a reminder. Requires reminder_id.

    Args:
        action: The operation to perform.
        reminder_id: Unique identifier of the reminder (for modify, delete).
        content: Content of the reminder (for add, modify).
        reminder_datetime: When the user wants to be reminded. ISO 8601
            datetime string, e.g. '2026-04-17T10:00:00+00:00' (for add, modify).
        latitude: Latitude of the location associated with this reminder
            (for add, modify). Pass None to clear.
        longitude: Longitude of the location associated with this reminder
            (for add, modify). Pass None to clear.

    Returns:
        The reminder ID string when action is add, None otherwise.

    Raises:
        ValueError: If required parameters are missing or no update fields
            are provided for modify.
        NoDataError: If reminder_id not found (for modify, delete).
    """
    if action == "add":
        kwargs: dict[str, Any] = {
            "content": content,
            "reminder_datetime": reminder_datetime,
        }
        if latitude is not NOT_GIVEN:
            kwargs["latitude"] = latitude
        if longitude is not NOT_GIVEN:
            kwargs["longitude"] = longitude
        return _get("add_reminder")(**kwargs)
    elif action == "modify":
        kwargs = {"reminder_id": reminder_id}
        if content is not NOT_GIVEN:
            kwargs["content"] = content
        if reminder_datetime is not NOT_GIVEN:
            kwargs["reminder_datetime"] = reminder_datetime
        if latitude is not NOT_GIVEN:
            kwargs["latitude"] = latitude
        if longitude is not NOT_GIVEN:
            kwargs["longitude"] = longitude
        _get("modify_reminder")(**kwargs)
        return None
    elif action == "delete":
        _get("remove_reminder")(reminder_id=reminder_id)
        return None
    else:
        raise ValueError(f"Unknown action: {action}")


mark_tools_absorbed_by(
    "manage_reminder",
    "add_reminder",
    "modify_reminder",
    "remove_reminder",
)
