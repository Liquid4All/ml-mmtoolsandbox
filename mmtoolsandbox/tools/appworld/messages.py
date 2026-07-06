"""
Messages API tools for MMToolSandbox.

This module wraps AppWorld's Phone text/voice message APIs as MMToolSandbox-compatible
tools with transparent authentication (no login/logout tools).

Split from phone.py — all bridge.call_api() calls still use "phone" as the app name.
"""

from typing import Any

from mmtoolsandbox.appworld.bridge import get_appworld_bridge
from mmtoolsandbox.appworld.state import (
    get_appworld_state,
    requires_network,
)
from mmtoolsandbox.tools.appworld import register_appworld_tool

# ---------------------------------------------------------------------------
# Text messages
# ---------------------------------------------------------------------------


@register_appworld_tool("messages")
@requires_network
def messages_show_text_message_window(
    phone_number: str,
    min_datetime: str | None = "1500-01-01|00:00:00",
    max_datetime: str | None = "3000-01-01|00:00:00",
    pagination_order: str | None = "descending",
    page_index: int | None = 0,
    page_limit: int | None = 5,
) -> list[dict[str, Any]]:
    """
    Show text messages with a contact around a given date and time.

    Args:
        phone_number: The phone number of the contact to show messages with.
        min_datetime: The minimum datetime to show messages on or after in YYYY-MM-DD|HH:MM:SS format. (optional)
        max_datetime: The maximum datetime to show messages on or before in YYYY-MM-DD|HH:MM:SS format. (optional)
        pagination_order: If set to ascending, as page_index increases, the results will have newer messages. If set to descending, as page_index increases, the results will have older messages. The messages within each page will always be oldest to newest. Must be one of: 'ascending', 'descending'. (optional)
        page_index: The index of the page to return. Must be >= 0. (optional)
        page_limit: The maximum number of results to return per page. Must be between 1 and 20. (optional)

    Returns:
        Success: [{"text_message_id": int, "sender": dict, "receiver": dict, "message": str, "sent_at": str (datetime)}]
        Failure: {"message": str}
    """
    bridge = get_appworld_bridge()
    appworld_state = get_appworld_state()

    params: dict[str, Any] = {
        "access_token": appworld_state.get_access_token("phone"),
        "phone_number": phone_number,
    }
    if min_datetime is not None:
        params["min_datetime"] = min_datetime
    if max_datetime is not None:
        params["max_datetime"] = max_datetime
    if pagination_order is not None:
        params["pagination_order"] = pagination_order
    if page_index is not None:
        params["page_index"] = page_index
    if page_limit is not None:
        params["page_limit"] = page_limit

    response = bridge.call_api(
        "phone", "show_text_message_window", method="get", **params
    )

    if isinstance(response, dict) and "error" in response:
        raise ValueError(f"API error: {response['error']}")

    # Extract list from response
    if isinstance(response, list):
        return list(response)
    if isinstance(response, dict):
        # Try common list keys
        for key in response:
            if isinstance(response[key], list):
                return list(response[key])
    return []


@register_appworld_tool("messages")
@requires_network
def messages_search_text_messages(
    query: str | None = "",
    phone_number: str | None = None,
    only_latest_per_contact: bool | None = False,
    page_index: int | None = 0,
    page_limit: int | None = 5,
    sort_by: str | None = None,
) -> list[dict[str, Any]]:
    """
    Show or search your text messages.

    Args:
        query: The search query string. (optional)
        phone_number: The phone number of the contact to show messages with. (optional)
        only_latest_per_contact: If set to true, only the latest message from each contact will be shown. (optional)
        page_index: The index of the page to return. Must be >= 0. (optional)
        page_limit: The maximum number of results to return per page. Must be between 1 and 20. (optional)
        sort_by: The attribute to sort the messages by prefixed with +/- to reflect ascending/descending. Valid attributes: created_at. If both query and sort_by are given and non-empty, results will be first ranked by query relevance, then paginated, and will then be sorted by the given attribute within each page. If both query and sort_by are not given, null, or empty, sort_by will default to -created_at. (optional)

    Returns:
        Success: [{"text_message_id": int, "sender": dict, "receiver": dict, "message": str, "sent_at": str (datetime)}]
        Failure: {"message": str}
    """
    bridge = get_appworld_bridge()
    appworld_state = get_appworld_state()

    params: dict[str, Any] = {
        "access_token": appworld_state.get_access_token("phone"),
    }
    if query is not None:
        params["query"] = query
    if phone_number is not None:
        params["phone_number"] = phone_number
    if only_latest_per_contact is not None:
        params["only_latest_per_contact"] = only_latest_per_contact
    if page_index is not None:
        params["page_index"] = page_index
    if page_limit is not None:
        params["page_limit"] = page_limit
    if sort_by is not None:
        params["sort_by"] = sort_by

    response = bridge.call_api("phone", "search_text_messages", method="get", **params)

    if isinstance(response, dict) and "error" in response:
        raise ValueError(f"API error: {response['error']}")

    # Extract list from response
    if isinstance(response, list):
        return list(response)
    if isinstance(response, dict):
        # Try common list keys
        for key in response:
            if isinstance(response[key], list):
                return list(response[key])
    return []


@register_appworld_tool("messages")
@requires_network
def messages_show_text_message(text_message_id: int) -> dict[str, Any]:
    """
    Show text message details.

    Args:
        text_message_id: ID of the text message to show.

    Returns:
        Success: {"text_message_id": int, "sender": dict, "receiver": dict, "message": str, "sent_at": str (datetime)}
        Failure: {"message": str}
    """
    bridge = get_appworld_bridge()
    appworld_state = get_appworld_state()

    params: dict[str, Any] = {
        "access_token": appworld_state.get_access_token("phone"),
        "text_message_id": text_message_id,
    }

    response = bridge.call_api("phone", "show_text_message", method="get", **params)

    if isinstance(response, dict) and "error" in response:
        raise ValueError(f"API error: {response['error']}")

    return dict(response) if isinstance(response, dict) else {}


@register_appworld_tool("messages")
@requires_network
def messages_delete_text_message(text_message_id: int) -> dict[str, Any]:
    """
    Delete a text message.

    Args:
        text_message_id: ID of the text message to be deleted.

    Returns:
        Success: {"message": str}
        Failure: {"message": str}
    """
    bridge = get_appworld_bridge()
    appworld_state = get_appworld_state()

    params: dict[str, Any] = {
        "access_token": appworld_state.get_access_token("phone"),
        "text_message_id": text_message_id,
    }

    response = bridge.call_api(
        "phone", "delete_text_message", method="delete", **params
    )

    if isinstance(response, dict) and "error" in response:
        raise ValueError(f"API error: {response['error']}")

    return dict(response) if isinstance(response, dict) else {}


@register_appworld_tool("messages")
@requires_network
def messages_send_text_message(phone_number: str, message: str) -> dict[str, Any]:
    """
    Send a text message on the given phone number.

    Args:
        phone_number: The phone number of the contact to send the message to.
        message: The content of the text message. Minimum length: 1.

    Returns:
        Success: {"message": str, "text_message_id": int}
        Failure: {"message": str}
    """
    bridge = get_appworld_bridge()
    appworld_state = get_appworld_state()

    params: dict[str, Any] = {
        "access_token": appworld_state.get_access_token("phone"),
        "phone_number": phone_number,
        "message": message,
    }

    response = bridge.call_api("phone", "send_text_message", method="post", **params)

    if isinstance(response, dict) and "error" in response:
        raise ValueError(f"API error: {response['error']}")

    return dict(response) if isinstance(response, dict) else {}


# ---------------------------------------------------------------------------
# Voice messages
# ---------------------------------------------------------------------------


@register_appworld_tool("messages")
@requires_network
def messages_show_voice_message_window(
    phone_number: str,
    min_datetime: str | None = "1500-01-01|00:00:00",
    max_datetime: str | None = "3000-01-01|00:00:00",
    pagination_order: str | None = "descending",
    page_index: int | None = 0,
    page_limit: int | None = 5,
) -> list[dict[str, Any]]:
    """
    Show voice messages with a contact around a given date and time.

    Args:
        phone_number: The phone number of the contact to show messages with.
        min_datetime: The minimum datetime to show messages on or after in YYYY-MM-DD|HH:MM:SS format. (optional)
        max_datetime: The maximum datetime to show messages on or before in YYYY-MM-DD|HH:MM:SS format. (optional)
        pagination_order: If set to ascending, as page_index increases, the results will have newer messages. If set to descending, as page_index increases, the results will have older messages. The messages within each page will always be oldest to newest. Must be one of: 'ascending', 'descending'. (optional)
        page_index: The index of the page to return. Must be >= 0. (optional)
        page_limit: The maximum number of results to return per page. Must be between 1 and 20. (optional)

    Returns:
        Success: [{"voice_message_id": int, "sender": dict, "receiver": dict, "message": str, "sent_at": str (datetime)}]
        Failure: {"message": str}
    """
    bridge = get_appworld_bridge()
    appworld_state = get_appworld_state()

    params: dict[str, Any] = {
        "access_token": appworld_state.get_access_token("phone"),
        "phone_number": phone_number,
    }
    if min_datetime is not None:
        params["min_datetime"] = min_datetime
    if max_datetime is not None:
        params["max_datetime"] = max_datetime
    if pagination_order is not None:
        params["pagination_order"] = pagination_order
    if page_index is not None:
        params["page_index"] = page_index
    if page_limit is not None:
        params["page_limit"] = page_limit

    response = bridge.call_api(
        "phone", "show_voice_message_window", method="get", **params
    )

    if isinstance(response, dict) and "error" in response:
        raise ValueError(f"API error: {response['error']}")

    # Extract list from response
    if isinstance(response, list):
        return list(response)
    if isinstance(response, dict):
        # Try common list keys
        for key in response:
            if isinstance(response[key], list):
                return list(response[key])
    return []


@register_appworld_tool("messages")
@requires_network
def messages_search_voice_messages(
    query: str | None = "",
    phone_number: str | None = None,
    only_latest_per_contact: bool | None = False,
    page_index: int | None = 0,
    page_limit: int | None = 5,
    sort_by: str | None = None,
) -> list[dict[str, Any]]:
    """
    Show or search voice messages between the user and a contact.

    Args:
        query: The search query string. (optional)
        phone_number: The phone number of the contact to show voice_messages with. (optional)
        only_latest_per_contact: If set to true, only the latest message from each contact will be shown. (optional)
        page_index: The index of the page to return. Must be >= 0. (optional)
        page_limit: The maximum number of results to return per page. Must be between 1 and 20. (optional)
        sort_by: The attribute to sort the voice messages by prefixed with +/- to reflect ascending/descending. Valid attributes: created_at. If both query and sort_by are given and non-empty, results will be first ranked by query relevance, then paginated, and will then be sorted by the given attribute within each page. If both query and sort_by are not given, null, or empty, sort_by will default to -created_at. (optional)

    Returns:
        Success: [{"voice_message_id": int, "sender": dict, "receiver": dict, "message": str, "sent_at": str (datetime)}]
        Failure: {"message": str}
    """
    bridge = get_appworld_bridge()
    appworld_state = get_appworld_state()

    params: dict[str, Any] = {
        "access_token": appworld_state.get_access_token("phone"),
    }
    if query is not None:
        params["query"] = query
    if phone_number is not None:
        params["phone_number"] = phone_number
    if only_latest_per_contact is not None:
        params["only_latest_per_contact"] = only_latest_per_contact
    if page_index is not None:
        params["page_index"] = page_index
    if page_limit is not None:
        params["page_limit"] = page_limit
    if sort_by is not None:
        params["sort_by"] = sort_by

    response = bridge.call_api("phone", "search_voice_messages", method="get", **params)

    if isinstance(response, dict) and "error" in response:
        raise ValueError(f"API error: {response['error']}")

    # Extract list from response
    if isinstance(response, list):
        return list(response)
    if isinstance(response, dict):
        # Try common list keys
        for key in response:
            if isinstance(response[key], list):
                return list(response[key])
    return []


@register_appworld_tool("messages")
@requires_network
def messages_show_voice_message(voice_message_id: int) -> dict[str, Any]:
    """
    Show voice message details.

    Args:
        voice_message_id: ID of the voice message to show.

    Returns:
        Success: {"voice_message_id": int, "sender": dict, "receiver": dict, "message": str, "sent_at": str (datetime)}
        Failure: {"message": str}
    """
    bridge = get_appworld_bridge()
    appworld_state = get_appworld_state()

    params: dict[str, Any] = {
        "access_token": appworld_state.get_access_token("phone"),
        "voice_message_id": voice_message_id,
    }

    response = bridge.call_api("phone", "show_voice_message", method="get", **params)

    if isinstance(response, dict) and "error" in response:
        raise ValueError(f"API error: {response['error']}")

    return dict(response) if isinstance(response, dict) else {}


@register_appworld_tool("messages")
@requires_network
def messages_delete_voice_message(voice_message_id: int) -> dict[str, Any]:
    """
    Delete a voice message.

    Args:
        voice_message_id: The ID of the voice message to delete.

    Returns:
        Success: {"message": str}
        Failure: {"message": str}
    """
    bridge = get_appworld_bridge()
    appworld_state = get_appworld_state()

    params: dict[str, Any] = {
        "access_token": appworld_state.get_access_token("phone"),
        "voice_message_id": voice_message_id,
    }

    response = bridge.call_api(
        "phone", "delete_voice_message", method="delete", **params
    )

    if isinstance(response, dict) and "error" in response:
        raise ValueError(f"API error: {response['error']}")

    return dict(response) if isinstance(response, dict) else {}


@register_appworld_tool("messages")
@requires_network
def messages_send_voice_message(phone_number: str, message: str) -> dict[str, Any]:
    """
    Send a voice message on the given phone number.

    Args:
        phone_number: The phone number of the contact to send the voice message to.
        message: The message text of the voice_message. Minimum length: 1.

    Returns:
        Success: {"message": str, "voice_message_id": int}
        Failure: {"message": str}
    """
    bridge = get_appworld_bridge()
    appworld_state = get_appworld_state()

    params: dict[str, Any] = {
        "access_token": appworld_state.get_access_token("phone"),
        "phone_number": phone_number,
        "message": message,
    }

    response = bridge.call_api("phone", "send_voice_message", method="post", **params)

    if isinstance(response, dict) and "error" in response:
        raise ValueError(f"API error: {response['error']}")

    return dict(response) if isinstance(response, dict) else {}
