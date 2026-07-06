# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""MINI Gmail tools -- 2 workflow-based tools covering all Gmail operations.

gmail_read: browsing threads, viewing emails/drafts, labels, categories, status,
            and thread state management (mark read/unread, archive, star, spam,
            label, snooze, delete).
gmail_write: composing/sending emails, managing drafts, and handling attachments.
"""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.datasets.names import DatasetName


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.appworld.gmail as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# gmail_read -- "I want to read, browse, or organize my email"
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.MINI},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def gmail_read(
    domain: Literal[
        "threads",
        "email",
        "drafts",
        "labels",
        "categories",
        "thread_action",
    ],
    # -- threads params --
    category: Literal["inbox", "outbox", "snoozed", "starred", "archived", "spam"]
    | NotGiven = NOT_GIVEN,
    query: str | None = None,
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
    page_index: int | None = None,
    page_limit: int | None = None,
    sort_by: str | None = None,
    # -- email params --
    email_thread_id: int | NotGiven = NOT_GIVEN,
    email_id: int | NotGiven = NOT_GIVEN,
    # -- drafts params --
    draft_id: int | NotGiven = NOT_GIVEN,
    recipient_email: str | None = None,
    scheduled: bool | None = None,
    belongs_to_email_thread_id: int | None = None,
    response_to_email_id: int | None = None,
    # -- thread_action params --
    thread_action: Literal[
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
        "delete",
    ]
    | NotGiven = NOT_GIVEN,
    snooze_until: str | NotGiven = NOT_GIVEN,
    # -- email sub-action --
    email_action: Literal["show", "delete"] | NotGiven = NOT_GIVEN,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Read, browse, and organize Gmail threads, emails, and drafts.

    Domains:
        threads: Show threads by category (inbox/outbox/snoozed/starred/
            archived/spam) with full search filters. Requires category.
            Optional filters: query, label, starred, archived, spam, snoozed,
            read, attachment, from_email, to_email, min_created_at,
            max_created_at, page_index, page_limit, sort_by.
        email: View or delete a specific email. Requires email_thread_id and
            email_id. Set email_action to "show" (default) to view details or
            "delete" to delete the email from the thread.
        drafts: List or view drafts. To list drafts pass optional filters
            (query, recipient_email, attachment, scheduled,
            belongs_to_email_thread_id, response_to_email_id, min_created_at,
            max_created_at, page_index, page_limit, sort_by). To view a
            single draft pass draft_id.
        labels: Search email thread labels. Optional: query, page_index,
            page_limit.
        categories: Show the number of email threads in each category.
        thread_action: Manage thread state. Requires email_thread_id and
            thread_action (mark_read, mark_unread, archive, unarchive, star,
            unstar, mark_spam, unmark_spam, label, unlabel, snooze, unsnooze,
            delete). The "label" action also requires the label parameter.
            The "snooze" action also requires snooze_until in
            YYYY-MM-DD|HH:MM:SS format.

    Args:
        domain: The area of Gmail to interact with.
        category: Thread category to browse (for threads domain).
        query: Search query string (for threads, drafts, labels).
        label: Label filter (for threads) or label to assign (for
            thread_action "label").
        starred: Filter by starred status (for threads).
        archived: Filter by archived status (for threads).
        spam: Filter by spam status (for threads).
        snoozed: Filter by snoozed status (for threads).
        read: Filter by read status (for threads, categories).
        attachment: Filter by attachment presence (for threads, drafts).
        from_email: Filter by sender email (for threads).
        to_email: Filter by recipient email (for threads).
        min_created_at: Min date filter YYYY-MM-DD (for threads, drafts).
        max_created_at: Max date filter YYYY-MM-DD (for threads, drafts).
        page_index: Page index for pagination (for threads, drafts, labels).
        page_limit: Max results per page (for threads, drafts, labels).
        sort_by: Sort attribute prefixed with +/- (for threads, drafts).
        email_thread_id: Email thread ID (for email, thread_action).
        email_id: Email ID (for email domain).
        draft_id: Draft ID (for drafts domain, single draft view).
        recipient_email: Filter drafts by recipient (for drafts).
        scheduled: Filter drafts by scheduled status (for drafts).
        belongs_to_email_thread_id: Filter drafts by thread (for drafts).
        response_to_email_id: Filter drafts by reply-to email (for drafts).
        thread_action: Action to perform on a thread (for thread_action).
        snooze_until: Snooze until datetime YYYY-MM-DD|HH:MM:SS (for
            thread_action "snooze").
        email_action: "show" or "delete" for the email domain. Defaults
            to "show".

    Returns:
        For threads: list of thread dicts with email_thread_id, subject,
            from_email, created_at. Pass email_thread_id to email/thread_action.
        For email: dict with email_id, subject, body, from_email, to_email.
        For drafts: single draft dict or list of draft dicts.
        For labels: list of label strings.
        For categories: dict with category name -> thread count.
        For thread_action: confirmation dict.

    Raises:
        ValueError: If required parameters are missing for the chosen domain.
    """
    if domain == "threads":
        # Dispatch to gmail_show_{category}_threads
        if category is NOT_GIVEN:
            raise ValueError("category is required for threads domain")
        fn_name = f"gmail_show_{category}_threads"
        kwargs: dict[str, Any] = {}
        if query is not None:
            kwargs["query"] = query
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
        if read is not None:
            kwargs["read"] = read
        if attachment is not None:
            kwargs["attachment"] = attachment
        if from_email is not None:
            kwargs["from_email"] = from_email
        if to_email is not None:
            kwargs["to_email"] = to_email
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
        return _get(fn_name)(**kwargs)

    elif domain == "email":
        if email_thread_id is NOT_GIVEN or email_id is NOT_GIVEN:
            raise ValueError(
                "email_thread_id and email_id are required for email domain"
            )
        action = email_action if email_action is not NOT_GIVEN else "show"
        if action == "show":
            return _get("gmail_show_email")(email_id=email_id)
        elif action == "delete":
            return _get("gmail_delete_email_in_thread")(
                email_thread_id=email_thread_id, email_id=email_id
            )
        else:
            raise ValueError(f"Unknown email_action: {action}")

    elif domain == "drafts":
        # Single draft view if draft_id is given with no list filters
        if draft_id is not NOT_GIVEN:
            return _get("gmail_show_draft")(draft_id=draft_id)
        # List drafts
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

    elif domain == "labels":
        kwargs = {}
        if query is not None:
            kwargs["query"] = query
        if page_index is not None:
            kwargs["page_index"] = page_index
        if page_limit is not None:
            kwargs["page_limit"] = page_limit
        return _get("gmail_search_labels")(**kwargs)

    elif domain == "categories":
        kwargs = {}
        if read is not None:
            kwargs["read"] = read
        return _get("gmail_show_category_sizes")(**kwargs)

    elif domain == "thread_action":
        if email_thread_id is NOT_GIVEN:
            raise ValueError("email_thread_id is required for thread_action domain")
        if thread_action is NOT_GIVEN:
            raise ValueError("thread_action is required for thread_action domain")
        dispatch: dict[str, str] = {
            "mark_read": "gmail_mark_thread_read",
            "mark_unread": "gmail_mark_thread_unread",
            "archive": "gmail_mark_thread_archived",
            "unarchive": "gmail_mark_thread_unarchived",
            "star": "gmail_mark_thread_starred",
            "unstar": "gmail_mark_thread_unstarred",
            "mark_spam": "gmail_mark_thread_spam",
            "unmark_spam": "gmail_mark_thread_not_spam",
            "label": "gmail_label_thread",
            "unlabel": "gmail_unlabel_thread",
            "snooze": "gmail_snooze_thread",
            "unsnooze": "gmail_unsnooze_thread",
            "delete": "gmail_delete_thread",
        }
        fn_name = dispatch.get(thread_action)  # type: ignore[arg-type, assignment]
        if fn_name is None:
            raise ValueError(f"Unknown thread_action: {thread_action}")

        if thread_action == "label":
            if label is None:
                raise ValueError(
                    "label parameter is required for thread_action 'label'"
                )
            return _get(fn_name)(email_thread_id=email_thread_id, label=label)
        elif thread_action == "snooze":
            if snooze_until is NOT_GIVEN:
                raise ValueError("snooze_until is required for thread_action 'snooze'")
            return _get(fn_name)(
                email_thread_id=email_thread_id, snooze_until=snooze_until
            )
        else:
            return _get(fn_name)(email_thread_id=email_thread_id)

    else:
        raise ValueError(f"Unknown domain: {domain}")


# ---------------------------------------------------------------------------
# gmail_write -- "I want to compose, send, or manage email drafts"
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.MINI},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def gmail_write(
    domain: Literal["compose", "draft", "attachment", "status"],
    action: Literal[
        "send",
        "reply",
        "forward_email",
        "forward_thread",
        "create",
        "update",
        "delete",
        "download",
        "upload",
        "remove",
        "set",
    ],
    # -- compose params --
    email_addresses: list[str] | NotGiven = NOT_GIVEN,
    subject: str | NotGiven = NOT_GIVEN,
    body: str | NotGiven = NOT_GIVEN,
    email_thread_id: int | NotGiven = NOT_GIVEN,
    email_id: int | NotGiven = NOT_GIVEN,
    save_as_draft: bool | None = None,
    # -- draft params --
    draft_id: int | NotGiven = NOT_GIVEN,
    recipient_email_addresses: list[str] | NotGiven = NOT_GIVEN,
    belongs_to_email_thread_id: int | None = None,
    response_to_email_id: int | None = None,
    scheduled_send_at: str | None = None,
    # -- attachment params --
    attachment_id: int | NotGiven = NOT_GIVEN,
    attachment_file_paths: list[str] | None = None,
    download_to_file_path: str | None = None,
    overwrite: bool | None = None,
    # -- status params --
    status_text: str | NotGiven = NOT_GIVEN,
    status_expiry: str | NotGiven = NOT_GIVEN,
    # -- shared --
    file_system_access_token: str | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Compose, send, and manage Gmail drafts, attachments, and status.

    Domains:
        compose: Send, reply, or forward emails. Requires action.
            action="send": Send a new email. Requires email_addresses,
                subject, body. Optional: attachment_file_paths,
                file_system_access_token.
            action="reply": Reply to an email. Requires email_thread_id,
                email_id, body. Optional: email_addresses (defaults to
                sender), attachment_file_paths, file_system_access_token.
            action="forward_email": Forward a single email. Requires
                email_thread_id, email_id, email_addresses. Optional:
                save_as_draft.
            action="forward_thread": Forward entire thread. Requires
                email_thread_id, email_addresses. Optional: save_as_draft.
        draft: Manage email drafts. Requires action.
            action="create": Create a draft. Requires
                recipient_email_addresses, body. Optional: subject,
                belongs_to_email_thread_id, response_to_email_id,
                attachment_file_paths, scheduled_send_at,
                file_system_access_token.
            action="update": Update a draft. Requires draft_id. Optional:
                email_addresses, subject, body,
                belongs_to_email_thread_id, response_to_email_id,
                scheduled_send_at. Pass scheduled_send_at=null to
                remove scheduled delivery.
            action="delete": Delete a draft. Requires draft_id.
            action="send": Send a draft immediately. Requires draft_id.
                Optional: file_system_access_token.
        attachment: Manage email attachments. Requires action.
            action="download": Download an attachment. Requires
                attachment_id, file_system_access_token. Optional:
                download_to_file_path, overwrite.
            action="upload": Upload attachments to a draft. Requires
                draft_id, attachment_file_paths, file_system_access_token.
                Optional: overwrite.
            action="remove": Remove attachment from draft. Requires
                draft_id, attachment_id.
        status: Set your availability status.
            action="set": Set status. Requires status_text.
                Optional: status_expiry.

    Args:
        domain: The area of Gmail writing to interact with.
        action: The specific operation (send/reply/forward_email/
            forward_thread for compose; create/update/delete/send for draft;
            download/upload/remove for attachment; set for status).
        email_addresses: Recipient email addresses (for compose send/reply/
            forward, draft update).
        subject: Email subject (for compose send, draft create/update).
        body: Email body (for compose send/reply, draft create/update).
        email_thread_id: Thread ID (for compose reply/forward).
        email_id: Email ID (for compose reply/forward_email).
        save_as_draft: Save as draft instead of sending (for compose
            forward).
        draft_id: Draft ID (for draft update/delete/send, attachment
            upload/remove).
        recipient_email_addresses: Recipient emails (for draft create).
        belongs_to_email_thread_id: Thread ID for draft reply/forward
            (for draft create/update).
        response_to_email_id: Email ID being replied to (for draft
            create/update).
        scheduled_send_at: Future send time YYYY-MM-DD|HH:MM:SS (for draft
            create/update).
        attachment_id: Attachment ID (for attachment download/remove).
        attachment_file_paths: File paths to attach (for compose send/reply,
            draft create, attachment upload).
        download_to_file_path: Path to save downloaded attachment (for
            attachment download).
        overwrite: Whether to overwrite existing files (for attachment
            download/upload).
        file_system_access_token: File system access token for attachment
            operations.
        status_text: Availability status text (for status domain).
        status_expiry: Status expiry datetime (for status domain).

    Returns:
        For compose send/reply: dict with email details. Sends immediately —
            externally visible and irreversible.
        For compose forward: dict. If save_as_draft=true, creates a draft
            instead of sending.
        For draft create: dict with draft_id. Pass to draft update/delete/send.
        For draft send: sends the draft. Externally visible and irreversible.
        For attachment: download/upload/remove confirmation dict.
        For status: confirmation dict.

    Raises:
        ValueError: If required parameters are missing.
    """
    if domain == "compose":
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
            raise ValueError(f"Unknown compose action: {action}")

    elif domain == "draft":
        if action == "create":
            kwargs = {
                "recipient_email_addresses": recipient_email_addresses,
                "body": body,
            }
            if subject is not NOT_GIVEN:
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
        elif action == "update":
            kwargs = {"draft_id": draft_id}
            if email_addresses is not NOT_GIVEN:
                kwargs["email_addresses"] = email_addresses
            if subject is not NOT_GIVEN:
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
            raise ValueError(f"Unknown draft action: {action}")

    elif domain == "attachment":
        if action == "download":
            kwargs = {
                "attachment_id": attachment_id,
                "file_system_access_token": file_system_access_token,
            }
            if download_to_file_path is not None:
                kwargs["download_to_file_path"] = download_to_file_path
            if overwrite is not None:
                kwargs["overwrite"] = overwrite
            return _get("gmail_download_attachment")(**kwargs)
        elif action == "upload":
            kwargs = {
                "draft_id": draft_id,
                "attachment_file_paths": attachment_file_paths,
                "file_system_access_token": file_system_access_token,
            }
            if overwrite is not None:
                kwargs["overwrite"] = overwrite
            return _get("gmail_upload_attachments_to_draft")(**kwargs)
        elif action == "remove":
            return _get("gmail_remove_attachment_from_draft")(
                draft_id=draft_id, attachment_id=attachment_id
            )
        else:
            raise ValueError(f"Unknown attachment action: {action}")

    elif domain == "status":
        if status_text is NOT_GIVEN:
            raise ValueError("status_text is required for status domain")
        status_kwargs: dict[str, Any] = {"status": status_text}
        if status_expiry is not NOT_GIVEN:
            status_kwargs["status_expiry"] = status_expiry
        return _get("gmail_set_status")(**status_kwargs)

    else:
        raise ValueError(f"Unknown domain: {domain}")
