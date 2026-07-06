# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""COMPACT Gmail tools — merges thread listing (6→1), compose (4→1), attachments (3→1)."""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.datasets.names import DatasetName
from mmtoolsandbox.tools.compact import mark_compact_tools_absorbed_by


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.appworld.gmail as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# Strategy 4: Thread listing (6 → 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def gmail_show_threads(
    category: Literal["inbox", "outbox", "snoozed", "starred", "archived", "spam"],
    query: str | None = "",
    label: str | None = None,
    starred: bool | None = None,
    archived: bool | None = None,
    spam: bool | None = None,
    snoozed: bool | None = None,
    read: bool | None = None,
    attachment: bool | None = None,
    from_email: str | None = None,
    to_email: str | None = None,
    min_created_at: str | None = None,
    max_created_at: str | None = None,
    page_index: int | None = 0,
    page_limit: int | None = 5,
    sort_by: str | None = None,
) -> list[dict[str, Any]]:
    """Show or search Gmail email threads in inbox, outbox, snoozed, starred, archived, or spam.

    Args:
        category: Which thread category to show.
        query: Search query string.
        label: Filter by label.
        starred: Filter by starred status.
        archived: Filter by archived status.
        spam: Filter by spam status.
        snoozed: Filter by snoozed status.
        read: Filter by read status.
        attachment: Filter by whether thread has attachments.
        from_email: Filter by sender email.
        to_email: Filter by recipient email.
        min_created_at: Minimum creation date (YYYY-MM-DD).
        max_created_at: Maximum creation date (YYYY-MM-DD).
        page_index: Zero-based page index for pagination.
        page_limit: Maximum results per page.
        sort_by: Sort field. Prefix with '-' for descending.

    Returns:
        List of matching email threads.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Gmail.
    """
    func_map = {
        "inbox": "gmail_show_inbox_threads",
        "outbox": "gmail_show_outbox_threads",
        "snoozed": "gmail_show_snoozed_threads",
        "starred": "gmail_show_starred_threads",
        "archived": "gmail_show_archived_threads",
        "spam": "gmail_show_spam_threads",
    }
    kwargs: dict[str, Any] = {
        "query": query,
        "read": read,
        "attachment": attachment,
        "from_email": from_email,
        "to_email": to_email,
        "min_created_at": min_created_at,
        "max_created_at": max_created_at,
        "page_index": page_index,
        "page_limit": page_limit,
        "sort_by": sort_by,
    }
    if label is not None:
        kwargs["label"] = label
    if starred is not None:
        kwargs["starred"] = starred
    if archived is not None:
        kwargs["archived"] = archived
    if spam is not None:
        kwargs["spam"] = spam
    if snoozed is not None:
        kwargs["snoozed"] = snoozed
    return _get(func_map[category])(**kwargs)


# ---------------------------------------------------------------------------
# Strategy 5: Email compose (4 → 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def gmail_compose(
    action: Literal["send", "reply", "forward_email", "forward_thread"],
    email_addresses: list[str] | NotGiven = NOT_GIVEN,
    subject: str | NotGiven = NOT_GIVEN,
    body: str | NotGiven = NOT_GIVEN,
    email_thread_id: int | NotGiven = NOT_GIVEN,
    email_id: int | NotGiven = NOT_GIVEN,
    attachment_file_paths: list[str] | None = None,
    file_system_access_token: str | None = None,
    save_as_draft: bool | None = False,
) -> dict[str, Any]:
    """Send, reply to, or forward emails in Gmail.

    Actions:
        send: Send a new email. Requires email_addresses, subject, and body.
        reply: Reply to an email in a thread. Requires email_thread_id,
            email_id, and body. Optionally override email_addresses.
        forward_email: Forward a specific email from a thread. Requires
            email_thread_id, email_id, and email_addresses.
        forward_thread: Forward an entire thread. Requires email_thread_id
            and email_addresses.

    Args:
        action: The compose operation to perform.
        email_addresses: List of recipient email addresses.
        subject: Email subject (for send).
        body: Email body (for send, reply).
        email_thread_id: Thread ID (for reply, forward_email, forward_thread).
        email_id: Email ID within thread (for reply, forward_email).
        attachment_file_paths: File paths to attach (for send, reply).
        file_system_access_token: Access token for file attachments.
        save_as_draft: If true, create draft instead of sending (for
            forward_email, forward_thread).

    Returns:
        Sent email or draft details.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Gmail.
    """
    if action == "send":
        kwargs: dict[str, Any] = {
            "email_addresses": email_addresses,
            "subject": subject,
            "body": body,
        }
        if attachment_file_paths is not None:
            kwargs["attachment_file_paths"] = attachment_file_paths
        if file_system_access_token is not None:
            kwargs["file_system_access_token"] = file_system_access_token
        return _get("gmail_send_email")(**kwargs)
    elif action == "reply":
        kwargs = {
            "email_thread_id": email_thread_id,
            "email_id": email_id,
            "body": body,
        }
        if email_addresses is not NOT_GIVEN:
            kwargs["email_addresses"] = email_addresses
        if attachment_file_paths is not None:
            kwargs["attachment_file_paths"] = attachment_file_paths
        if file_system_access_token is not None:
            kwargs["file_system_access_token"] = file_system_access_token
        return _get("gmail_reply_to_email")(**kwargs)
    elif action == "forward_email":
        kwargs = {
            "email_thread_id": email_thread_id,
            "email_id": email_id,
            "email_addresses": email_addresses,
        }
        if save_as_draft is not None:
            kwargs["draft_not_send"] = save_as_draft
        return _get("gmail_forward_email_from_thread")(**kwargs)
    elif action == "forward_thread":
        kwargs = {
            "email_thread_id": email_thread_id,
            "email_addresses": email_addresses,
        }
        if save_as_draft is not None:
            kwargs["draft_not_send"] = save_as_draft
        return _get("gmail_forward_email_thread")(**kwargs)
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Strategy 5: Attachment management (3 → 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def gmail_manage_attachment(
    action: Literal["download", "upload_to_draft", "remove_from_draft"],
    attachment_id: int | NotGiven = NOT_GIVEN,
    draft_id: int | NotGiven = NOT_GIVEN,
    file_system_access_token: str | NotGiven = NOT_GIVEN,
    attachment_file_paths: list[str] | NotGiven = NOT_GIVEN,
    download_to_file_path: str | None = None,
    overwrite: bool | None = False,
) -> dict[str, Any]:
    """Download, upload, or remove Gmail email attachments.

    Actions:
        download: Download an attachment. Requires attachment_id and
            file_system_access_token.
        upload_to_draft: Upload attachments to a draft. Requires draft_id,
            attachment_file_paths, and file_system_access_token.
        remove_from_draft: Remove an attachment from a draft. Requires
            draft_id and attachment_id.

    Args:
        action: The attachment operation to perform.
        attachment_id: ID of the attachment (for download, remove_from_draft).
        draft_id: ID of the draft (for upload_to_draft, remove_from_draft).
        file_system_access_token: Access token (for download, upload_to_draft).
        attachment_file_paths: Paths to files to upload (for upload_to_draft).
        download_to_file_path: Destination path (for download).
        overwrite: Whether to overwrite existing files.

    Returns:
        Attachment details or action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Gmail.
    """
    if action == "download":
        kwargs: dict[str, Any] = {
            "attachment_id": attachment_id,
            "file_system_access_token": file_system_access_token,
        }
        if download_to_file_path is not None:
            kwargs["download_to_file_path"] = download_to_file_path
        if overwrite is not None:
            kwargs["overwrite"] = overwrite
        return _get("gmail_download_attachment")(**kwargs)
    elif action == "upload_to_draft":
        kwargs = {
            "draft_id": draft_id,
            "attachment_file_paths": attachment_file_paths,
            "file_system_access_token": file_system_access_token,
        }
        if overwrite is not None:
            kwargs["overwrite"] = overwrite
        return _get("gmail_upload_attachments_to_draft")(**kwargs)
    elif action == "remove_from_draft":
        return _get("gmail_remove_attachment_from_draft")(
            draft_id=draft_id, attachment_id=attachment_id
        )
    else:
        raise ValueError(f"Unknown action: {action}")


mark_compact_tools_absorbed_by(
    "gmail_show_threads",
    "gmail_show_inbox_threads",
    "gmail_show_outbox_threads",
    "gmail_show_snoozed_threads",
    "gmail_show_starred_threads",
    "gmail_show_archived_threads",
    "gmail_show_spam_threads",
)
mark_compact_tools_absorbed_by(
    "gmail_compose",
    "gmail_send_email",
    "gmail_reply_to_email",
    "gmail_forward_email_from_thread",
    "gmail_forward_email_thread",
)
mark_compact_tools_absorbed_by(
    "gmail_manage_attachment",
    "gmail_download_attachment",
    "gmail_upload_attachments_to_draft",
    "gmail_remove_attachment_from_draft",
)


# ---------------------------------------------------------------------------
# CRUD+List: Gmail draft management (absorbs gmail_show_drafts)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def gmail_manage_draft(
    action: Literal["create", "show", "update", "delete", "send", "list"],
    draft_id: int | NotGiven = NOT_GIVEN,
    recipient_email_addresses: list[str] | NotGiven = NOT_GIVEN,
    body: str | NotGiven = NOT_GIVEN,
    subject: str | None = None,
    belongs_to_email_thread_id: int | None = None,
    response_to_email_id: int | None = None,
    attachment_file_paths: list[str] | None = None,
    scheduled_send_at: str | None = None,
    file_system_access_token: str | None = None,
    # list-action params
    query: str | None = "",
    recipient_email: str | None = None,
    attachment: bool | None = None,
    scheduled: bool | None = None,
    min_created_at: str | None = None,
    max_created_at: str | None = None,
    page_index: int | None = 0,
    page_limit: int | None = 5,
    sort_by: str | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Manage Gmail drafts: create, view, update, delete, send, or list.

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
        list: List or search your drafts. Supports query,
            recipient_email, attachment, scheduled,
            belongs_to_email_thread_id, response_to_email_id,
            date range, pagination, and sorting.

    Args:
        action: The operation to perform.
        draft_id: The draft ID (for show, update, delete, send).
        recipient_email_addresses: List of recipient email addresses
            (for create, update).
        body: The body of the draft (for create).
        subject: The subject of the draft (for create, update). Must
            be None for replies.
        belongs_to_email_thread_id: The email thread ID the draft
            belongs to (for create, update, list filter).
        response_to_email_id: The email ID the draft responds to
            (for create, update, list filter).
        attachment_file_paths: List of absolute file paths from the
            file_system app to attach (for create).
        scheduled_send_at: Future send time in YYYY-MM-DD|HH:MM:SS
            format (for create, update).
        file_system_access_token: Access token from file_system login,
            needed for attachments (for create, send).
        query: Search query string (for list).
        recipient_email: Filter by recipient email (for list).
        attachment: Filter by attachment status (for list).
        scheduled: Filter by scheduled status (for list).
        min_created_at: Minimum created_at date YYYY-MM-DD (for list).
        max_created_at: Maximum created_at date YYYY-MM-DD (for list).
        page_index: Zero-based page index (for list).
        page_limit: Maximum results per page (for list).
        sort_by: Sort attribute prefixed with +/- for direction. Valid
            attributes: created_at, updated_at (for list).

    Returns:
        Draft details, action confirmation, or list of drafts.

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
    elif action == "list":
        kwargs = {}
        if query is not None:
            kwargs["query"] = query
        if recipient_email is not None:
            kwargs["recipient_email"] = recipient_email
        if attachment is not None:
            kwargs["attachment"] = attachment
        if scheduled is not None:
            kwargs["scheduled"] = scheduled
        if belongs_to_email_thread_id is not None:
            kwargs["belongs_to_email_thread_id"] = belongs_to_email_thread_id
        if response_to_email_id is not None:
            kwargs["response_to_email_id"] = response_to_email_id
        if min_created_at is not None:
            kwargs["min_created_at"] = min_created_at
        if max_created_at is not None:
            kwargs["max_created_at"] = max_created_at
        if page_index is not None:
            kwargs["page_index"] = page_index
        if page_limit is not None:
            kwargs["page_limit"] = page_limit
        if sort_by is not None:
            kwargs["sort_by"] = sort_by
        return _get("gmail_show_drafts")(**kwargs)
    else:
        raise ValueError(f"Unknown action: {action}")


mark_compact_tools_absorbed_by(
    "gmail_manage_draft",
    "gmail_show_drafts",
)
