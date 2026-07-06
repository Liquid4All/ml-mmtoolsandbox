# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""MINI messages tool — unified text and voice message management."""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.datasets.names import DatasetName


@register_as_tool(
    toolboxes={DatasetName.MINI},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def messages(
    message_type: Literal["text", "voice"],
    action: Literal["send", "delete", "show", "search", "show_window"],
    # send params
    phone_number: str | NotGiven = NOT_GIVEN,
    message: str | NotGiven = NOT_GIVEN,
    # show/delete params
    message_id: int | NotGiven = NOT_GIVEN,
    # search params
    query: str | None | NotGiven = NOT_GIVEN,
    only_latest_per_contact: bool | None | NotGiven = NOT_GIVEN,
    page_index: int | None | NotGiven = NOT_GIVEN,
    page_limit: int | None | NotGiven = NOT_GIVEN,
    sort_by: str | None | NotGiven = NOT_GIVEN,
    # show_window params
    min_datetime: str | None | NotGiven = NOT_GIVEN,
    max_datetime: str | None | NotGiven = NOT_GIVEN,
    pagination_order: Literal["ascending", "descending"] | None | NotGiven = NOT_GIVEN,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Send, search, view, or delete text and voice messages.

    Actions:
        send - Send message. Requires phone_number, message.
            Externally visible — sends to another person.
        delete - Delete message. Requires message_id. Irreversible.
        show - Show message details. Requires message_id.
        search - Search messages. Optional: query, phone_number,
            only_latest_per_contact, page_index, page_limit, sort_by.
        show_window - Show message window with a contact. Requires
            phone_number. Optional: min_datetime, max_datetime
            (YYYY-MM-DD|HH:MM:SS format, e.g., "2026-04-30|14:00:00"),
            pagination_order, page_index, page_limit.

    Args:
        message_type: "text" or "voice".
        action: The message action.
        phone_number: Contact phone number (e.g., "+14155551234").
        message: Message content text.
        message_id: Message ID (maps to text_message_id or
            voice_message_id depending on message_type).
        query: Search query — matches message content.
        only_latest_per_contact: If true, return only latest message per
            contact.
        page_index: Zero-based page index.
        page_limit: Results per page.
        sort_by: Sort attribute with +/- prefix (e.g., "-created_at").
        min_datetime: Min datetime (YYYY-MM-DD|HH:MM:SS).
        max_datetime: Max datetime (YYYY-MM-DD|HH:MM:SS).
        pagination_order: "ascending" or "descending" page order.

    Returns:
        For send/show: dict with message_id, phone_number, message,
            created_at. Pass message_id to show/delete.
        For delete: confirmation dict.
        For search: list of message dicts.
        For show_window: list of messages in chronological order.
    """
    import mmtoolsandbox.tools.appworld.messages as m

    if action == "send":
        return getattr(m, f"messages_send_{message_type}_message")(
            phone_number=phone_number, message=message
        )
    elif action == "delete":
        id_key = f"{message_type}_message_id"
        return getattr(m, f"messages_delete_{message_type}_message")(
            **{id_key: message_id}
        )
    elif action == "show":
        id_key = f"{message_type}_message_id"
        return getattr(m, f"messages_show_{message_type}_message")(
            **{id_key: message_id}
        )
    elif action == "search":
        kwargs: dict[str, Any] = {}
        if query is not NOT_GIVEN:
            kwargs["query"] = query
        if phone_number is not NOT_GIVEN:
            kwargs["phone_number"] = phone_number
        if only_latest_per_contact is not NOT_GIVEN:
            kwargs["only_latest_per_contact"] = only_latest_per_contact
        if page_index is not NOT_GIVEN:
            kwargs["page_index"] = page_index
        if page_limit is not NOT_GIVEN:
            kwargs["page_limit"] = page_limit
        if sort_by is not NOT_GIVEN:
            kwargs["sort_by"] = sort_by
        return getattr(m, f"messages_search_{message_type}_messages")(**kwargs)
    elif action == "show_window":
        kwargs = {"phone_number": phone_number}
        if min_datetime is not NOT_GIVEN:
            kwargs["min_datetime"] = min_datetime
        if max_datetime is not NOT_GIVEN:
            kwargs["max_datetime"] = max_datetime
        if pagination_order is not NOT_GIVEN:
            kwargs["pagination_order"] = pagination_order
        if page_index is not NOT_GIVEN:
            kwargs["page_index"] = page_index
        if page_limit is not NOT_GIVEN:
            kwargs["page_limit"] = page_limit
        return getattr(m, f"messages_show_{message_type}_message_window")(**kwargs)
    else:
        raise ValueError(f"Unknown action: {action}")
