# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""Consolidated Gmail tools for the MEDIUM toolbox.

Thread management (show/delete/mark/label/snooze) and draft lifecycle
(create/show/update/delete/send) each collapsed into a single tool.
"""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.toolbox.names import ToolboxName
from mmtoolsandbox.tools.consolidated import mark_tools_absorbed_by


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.appworld.gmail as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# Thread management (14 original tools -> 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def gmail_manage_thread(
    action: Literal[
        "show",
        "delete",
        "mark_read",
        "mark_unread",
        "archive",
        "unarchive",
        "star",
        "unstar",
        "mark_spam",
        "unmark_spam",
        "label",
        "unlabel",
        "snooze",
        "unsnooze",
    ],
    email_thread_id: int,
    label: str | NotGiven = NOT_GIVEN,
    snooze_until: str | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Manage a Gmail email thread: view details or change its state.

    Actions:
        show: Show detailed information about the thread including emails
            and drafts within it.
        delete: Permanently delete the thread.
        mark_read: Mark the thread as read.
        mark_unread: Mark the thread as unread.
        archive: Archive the thread. Also removes spam and snooze status.
        unarchive: Unarchive the thread.
        star: Star the thread.
        unstar: Unstar the thread.
        mark_spam: Mark the thread as spam. Also removes archived and
            snooze status.
        unmark_spam: Mark the thread as not spam.
        label: Assign a label to the thread. Requires the label parameter.
        unlabel: Remove the label from the thread.
        snooze: Snooze the thread until a future date/time. It will
            reappear unread in your inbox/outbox at that time. Requires
            the snooze_until parameter in YYYY-MM-DD|HH:MM:SS format.
        unsnooze: Unsnooze the thread.

    Args:
        action: The operation to perform on the thread.
        email_thread_id: The ID of the email thread.
        label: The label to assign (required for the "label" action).
        snooze_until: Date/time to snooze until in YYYY-MM-DD|HH:MM:SS
            format (required for the "snooze" action).

    Returns:
        Thread details or action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Gmail.
    """
    if action == "show":
        return _get("gmail_show_thread")(email_thread_id=email_thread_id)
    elif action == "delete":
        return _get("gmail_delete_thread")(email_thread_id=email_thread_id)
    elif action == "mark_read":
        return _get("gmail_mark_thread_read")(email_thread_id=email_thread_id)
    elif action == "mark_unread":
        return _get("gmail_mark_thread_unread")(email_thread_id=email_thread_id)
    elif action == "archive":
        return _get("gmail_mark_thread_archived")(email_thread_id=email_thread_id)
    elif action == "unarchive":
        return _get("gmail_mark_thread_unarchived")(email_thread_id=email_thread_id)
    elif action == "star":
        return _get("gmail_mark_thread_starred")(email_thread_id=email_thread_id)
    elif action == "unstar":
        return _get("gmail_mark_thread_unstarred")(email_thread_id=email_thread_id)
    elif action == "mark_spam":
        return _get("gmail_mark_thread_spam")(email_thread_id=email_thread_id)
    elif action == "unmark_spam":
        return _get("gmail_mark_thread_not_spam")(email_thread_id=email_thread_id)
    elif action == "label":
        return _get("gmail_label_thread")(email_thread_id=email_thread_id, label=label)
    elif action == "unlabel":
        return _get("gmail_unlabel_thread")(email_thread_id=email_thread_id)
    elif action == "snooze":
        return _get("gmail_snooze_thread")(
            email_thread_id=email_thread_id, snooze_until=snooze_until
        )
    elif action == "unsnooze":
        return _get("gmail_unsnooze_thread")(email_thread_id=email_thread_id)
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Draft management (5 original tools -> 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def gmail_manage_draft(
    action: Literal["create", "show", "update", "delete", "send"],
    draft_id: int | NotGiven = NOT_GIVEN,
    recipient_email_addresses: list[str] | NotGiven = NOT_GIVEN,
    body: str | NotGiven = NOT_GIVEN,
    subject: str | None = None,
    belongs_to_email_thread_id: int | None = None,
    response_to_email_id: int | None = None,
    attachment_file_paths: list[str] | None = None,
    scheduled_send_at: str | None = None,
    file_system_access_token: str | None = None,
) -> dict[str, Any]:
    """Manage Gmail drafts: create, view, update, delete, or send.

    Actions:
        create: Create a new draft. Requires recipient_email_addresses
            and body. Optionally set subject (must be None for replies),
            belongs_to_email_thread_id (required for replies/forwards),
            response_to_email_id, attachment_file_paths,
            scheduled_send_at (YYYY-MM-DD|HH:MM:SS), and
            file_system_access_token (for attachments).
        show: Show detailed draft information. Requires draft_id.
        update: Update a draft. Requires draft_id and at least one of
            recipient_email_addresses, subject, body,
            belongs_to_email_thread_id, response_to_email_id, or
            scheduled_send_at. Pass scheduled_send_at=null to remove
            scheduled delivery.
        delete: Delete a draft. Requires draft_id.
        send: Send a draft immediately. Requires draft_id. Optionally
            provide file_system_access_token if the draft has
            attachments.

    Args:
        action: The operation to perform.
        draft_id: The draft ID (for show, update, delete, send).
        recipient_email_addresses: List of recipient email addresses
            (for create, update).
        body: The body of the draft (for create, update).
        subject: The subject of the draft (for create, update). Must
            be None for replies.
        belongs_to_email_thread_id: The email thread ID the draft
            belongs to (for create, update).
        response_to_email_id: The email ID the draft responds to
            (for create, update).
        attachment_file_paths: List of absolute file paths from the
            file_system app to attach (for create).
        scheduled_send_at: Future send time in YYYY-MM-DD|HH:MM:SS
            format (for create, update).
        file_system_access_token: Access token from file_system login,
            needed for attachments (for create, send).

    Returns:
        Draft details or action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Gmail.
    """
    if action == "create":
        kwargs: dict[str, Any] = {
            "recipient_email_addresses": recipient_email_addresses,
            "body": body,
        }
        if subject is not None:
            kwargs["subject"] = subject
        if belongs_to_email_thread_id is not None:
            kwargs["belongs_to_email_thread_id"] = belongs_to_email_thread_id
        if response_to_email_id is not None:
            kwargs["response_to_email_id"] = response_to_email_id
        if attachment_file_paths is not None:
            kwargs["attachment_file_paths"] = attachment_file_paths
        if scheduled_send_at is not None:
            kwargs["scheduled_send_at"] = scheduled_send_at
        if file_system_access_token is not None:
            kwargs["file_system_access_token"] = file_system_access_token
        return _get("gmail_create_draft")(**kwargs)
    elif action == "show":
        return _get("gmail_show_draft")(draft_id=draft_id)
    elif action == "update":
        kwargs = {"draft_id": draft_id}
        if recipient_email_addresses is not NOT_GIVEN:
            kwargs["email_addresses"] = recipient_email_addresses
        if subject is not None:
            kwargs["subject"] = subject
        if body is not NOT_GIVEN:
            kwargs["body"] = body
        if belongs_to_email_thread_id is not None:
            kwargs["belongs_to_email_thread_id"] = belongs_to_email_thread_id
        if response_to_email_id is not None:
            kwargs["response_to_email_id"] = response_to_email_id
        if scheduled_send_at is not None:
            kwargs["scheduled_send_at"] = scheduled_send_at
        return _get("gmail_update_draft")(**kwargs)
    elif action == "delete":
        return _get("gmail_delete_draft")(draft_id=draft_id)
    elif action == "send":
        kwargs = {"draft_id": draft_id}
        if file_system_access_token is not None:
            kwargs["file_system_access_token"] = file_system_access_token
        return _get("gmail_send_email_from_draft")(**kwargs)
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Mark absorbed tools
# ---------------------------------------------------------------------------

mark_tools_absorbed_by(
    "gmail_manage_thread",
    "gmail_show_thread",
    "gmail_delete_thread",
    "gmail_mark_thread_read",
    "gmail_mark_thread_unread",
    "gmail_mark_thread_archived",
    "gmail_mark_thread_unarchived",
    "gmail_mark_thread_starred",
    "gmail_mark_thread_unstarred",
    "gmail_mark_thread_spam",
    "gmail_mark_thread_not_spam",
    "gmail_label_thread",
    "gmail_unlabel_thread",
    "gmail_snooze_thread",
    "gmail_unsnooze_thread",
)
mark_tools_absorbed_by(
    "gmail_manage_draft",
    "gmail_create_draft",
    "gmail_show_draft",
    "gmail_update_draft",
    "gmail_delete_draft",
    "gmail_send_email_from_draft",
)
