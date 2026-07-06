# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""COMPACT Messages tool -- consolidates 10 text/voice message tools into 1."""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.datasets.names import DatasetName
from mmtoolsandbox.tools.compact import mark_compact_tools_absorbed_by


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.appworld.messages as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# Strategy 6: Text/Voice message consolidation (10 -> 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def messages_manage(
    message_type: Literal["text", "voice"],
    action: Literal["send", "delete", "show", "search", "show_window"],
    phone_number: str | NotGiven = NOT_GIVEN,
    message: str | NotGiven = NOT_GIVEN,
    message_id: int | NotGiven = NOT_GIVEN,
    query: str | None = "",
    only_latest_per_contact: bool | None = False,
    min_datetime: str | None = None,
    max_datetime: str | None = None,
    pagination_order: Literal["ascending", "descending"] | None = "descending",
    page_index: int | None = 0,
    page_limit: int | None = 5,
    sort_by: str | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Send, delete, show, search, or browse text and voice messages.

    Actions:
        send: Send a message. Requires phone_number and message.
        delete: Delete a message. Requires message_id.
        show: Show a single message by ID. Requires message_id.
        search: Search or list messages. Supports query, phone_number,
            only_latest_per_contact, page_index, page_limit, and sort_by.
        show_window: Show messages with a contact around a date range.
            Requires phone_number. Supports min_datetime, max_datetime,
            pagination_order, page_index, and page_limit.

    Args:
        message_type: Whether to operate on "text" or "voice" messages.
        action: The operation to perform.
        phone_number: Phone number of the contact (for send, search,
            show_window).
        message: Message content (for send).
        message_id: The ID of the message (for show, delete). Maps to
            text_message_id or voice_message_id depending on message_type.
        query: Search query string (for search).
        only_latest_per_contact: If true, show only the latest message per
            contact (for search).
        min_datetime: Minimum datetime in YYYY-MM-DD|HH:MM:SS format
            (for show_window).
        max_datetime: Maximum datetime in YYYY-MM-DD|HH:MM:SS format
            (for show_window).
        pagination_order: "ascending" or "descending" page order
            (for show_window).
        page_index: Zero-based page index (for search, show_window).
        page_limit: Maximum results per page (for search, show_window).
        sort_by: Sort field prefixed with +/- for direction. Valid
            attributes: created_at (for search).

    Returns:
        Message details, list of messages, or action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
    """
    prefix = message_type  # "text" or "voice"
    id_param = f"{prefix}_message_id"

    if action == "send":
        func_name = f"messages_send_{prefix}_message"
        return _get(func_name)(phone_number=phone_number, message=message)

    elif action == "delete":
        func_name = f"messages_delete_{prefix}_message"
        return _get(func_name)(**{id_param: message_id})

    elif action == "show":
        func_name = f"messages_show_{prefix}_message"
        return _get(func_name)(**{id_param: message_id})

    elif action == "search":
        func_name = f"messages_search_{prefix}_messages"
        kwargs: dict[str, Any] = {}
        if query is not None:
            kwargs["query"] = query
        if phone_number is not NOT_GIVEN and phone_number is not None:
            kwargs["phone_number"] = phone_number
        if only_latest_per_contact is not None:
            kwargs["only_latest_per_contact"] = only_latest_per_contact
        if page_index is not None:
            kwargs["page_index"] = page_index
        if page_limit is not None:
            kwargs["page_limit"] = page_limit
        if sort_by is not None:
            kwargs["sort_by"] = sort_by
        return _get(func_name)(**kwargs)

    elif action == "show_window":
        func_name = f"messages_show_{prefix}_message_window"
        kwargs = {"phone_number": phone_number}
        if min_datetime is not None:
            kwargs["min_datetime"] = min_datetime
        if max_datetime is not None:
            kwargs["max_datetime"] = max_datetime
        if pagination_order is not None:
            kwargs["pagination_order"] = pagination_order
        if page_index is not None:
            kwargs["page_index"] = page_index
        if page_limit is not None:
            kwargs["page_limit"] = page_limit
        return _get(func_name)(**kwargs)

    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Absorption declarations
# ---------------------------------------------------------------------------

mark_compact_tools_absorbed_by(
    "messages_manage",
    "messages_send_text_message",
    "messages_delete_text_message",
    "messages_show_text_message",
    "messages_search_text_messages",
    "messages_show_text_message_window",
    "messages_send_voice_message",
    "messages_delete_voice_message",
    "messages_show_voice_message",
    "messages_search_voice_messages",
    "messages_show_voice_message_window",
)
