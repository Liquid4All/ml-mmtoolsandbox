# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""Cross-app notification tools for the MEDIUM toolbox.

Consolidates identical notification tools across Todoist, Venmo, and
Splitwise into a single parameterized tool.

Note: Spotify and Amazon do NOT have notification tools.
"""

from __future__ import annotations

import importlib
from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.toolbox.names import ToolboxName
from mmtoolsandbox.tools.consolidated import mark_tools_absorbed_by

NotificationApp = Literal["todoist", "venmo", "splitwise"]

_NOTIFICATION_APPS = ("todoist", "venmo", "splitwise")


def _get_func(app: str, suffix: str) -> Any:
    mod = importlib.import_module(f"mmtoolsandbox.tools.appworld.{app}")
    return getattr(mod, f"{app}_{suffix}")


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def manage_notifications(
    app: NotificationApp,
    action: Literal["list", "delete_all", "mark_all", "count", "delete", "mark"],
    notification_id: int | NotGiven = NOT_GIVEN,
    read: bool | NotGiven = NOT_GIVEN,
    page_index: int | None = 0,
    page_limit: int | None = 5,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Manage notifications on Todoist, Venmo, or Splitwise.

    Actions:
        list: Show your notifications. Optionally filter by read status.
        delete_all: Delete all your notifications.
        mark_all: Mark all notifications as read or unread. Requires read.
        count: Get the count of your notifications. Optionally filter by read.
        delete: Delete a single notification. Requires notification_id.
        mark: Mark a single notification as read or unread. Requires
            notification_id and read.

    Args:
        app: The app to manage notifications on.
        action: The operation to perform.
        notification_id: ID of the notification (for delete, mark).
        read: Read status filter (for list, count) or target state (for
            mark_all, mark). True = read, False = unread.
        page_index: Zero-based page index for pagination (for list).
        page_limit: Maximum results per page (for list).

    Returns:
        List of notifications, count, or action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into the app.
    """
    if action == "list":
        kwargs: dict[str, Any] = {}
        if read is not NOT_GIVEN:
            kwargs["read"] = read
        kwargs["page_index"] = page_index
        kwargs["page_limit"] = page_limit
        return _get_func(app, "show_notifications")(**kwargs)
    elif action == "delete_all":
        return _get_func(app, "delete_notifications")()
    elif action == "mark_all":
        return _get_func(app, "mark_notifications")(read=read)
    elif action == "count":
        kwargs = {}
        if read is not NOT_GIVEN:
            kwargs["read"] = read
        return _get_func(app, "show_notifications_count")(**kwargs)
    elif action == "delete":
        return _get_func(app, "delete_notification")(
            notification_id=notification_id,
        )
    elif action == "mark":
        return _get_func(app, "mark_notification")(
            notification_id=notification_id,
            read=read,
        )
    else:
        raise ValueError(f"Unknown action: {action}")


# Mark absorbed tools
_ABSORBED: list[str] = []
for _app in _NOTIFICATION_APPS:
    _ABSORBED.extend(
        [
            f"{_app}_show_notifications",
            f"{_app}_delete_notifications",
            f"{_app}_mark_notifications",
            f"{_app}_show_notifications_count",
            f"{_app}_delete_notification",
            f"{_app}_mark_notification",
        ]
    )
mark_tools_absorbed_by("manage_notifications", *_ABSORBED)
