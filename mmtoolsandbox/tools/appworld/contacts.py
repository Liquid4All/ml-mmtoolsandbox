"""
Contacts API tools for MMToolSandbox.

This module wraps AppWorld's Phone contact-related APIs as MMToolSandbox-compatible
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


@register_appworld_tool("contacts")
@requires_network
def contacts_show_contact_relationships() -> list[dict[str, Any]]:
    """
    Get a list of all relationships available in your contact book.

    Returns:
        Success: [str]
        Failure: {"message": str}
    """
    bridge = get_appworld_bridge()
    appworld_state = get_appworld_state()

    params: dict[str, Any] = {
        "access_token": appworld_state.get_access_token("phone"),
    }

    response = bridge.call_api(
        "phone", "show_contact_relationships", method="get", **params
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


@register_appworld_tool("contacts")
@requires_network
def contacts_search_contacts(
    query: str | None = "",
    relationship: str | None = None,
    page_index: int | None = 0,
    page_limit: int | None = 5,
) -> list[dict[str, Any]]:
    """
    Search your contact book for relatives' information.

    Args:
        query: Search query for the contacts list. (optional)
        relationship: Relationship with the person in the contacts list to filter by. (optional)
        page_index: The index of the page to return. Must be >= 0. (optional)
        page_limit: The maximum number of results to return per page. Must be between 1 and 20. (optional)

    Returns:
        Success: [{"contact_id": int, "first_name": str, "last_name": str, "email": str, "phone_number": str, "relationships": list[str], "birthday": str, "home_address": str, ...}]
        Failure: {"message": str}
    """
    bridge = get_appworld_bridge()
    appworld_state = get_appworld_state()

    params: dict[str, Any] = {
        "access_token": appworld_state.get_access_token("phone"),
    }
    if query is not None:
        params["query"] = query
    if relationship is not None:
        params["relationship"] = relationship
    if page_index is not None:
        params["page_index"] = page_index
    if page_limit is not None:
        params["page_limit"] = page_limit

    response = bridge.call_api("phone", "search_contacts", method="get", **params)

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


@register_appworld_tool("contacts")
@requires_network
def contacts_add_contact(
    first_name: str,
    last_name: str,
    email: str,
    phone_number: str | None = None,
    relationships: list[str] | None = None,
    birthday: str | None = None,
    home_address: str | None = None,
    work_address: str | None = None,
) -> dict[str, Any]:
    """
    Add a new contact.

    Each contact must have a unique email address.

    Args:
        first_name: First name of the contact.
        last_name: Last name of the contact.
        email: Email of the contact. Must be a valid email address.
        phone_number: Phone number of the contact. (optional)
        relationships: Relationship with the contact. (optional)
        birthday: Birthday of the contact in YYYY-MM-DD format. (optional)
        home_address: Home address of the contact. (optional)
        work_address: Work address of the contact. (optional)

    Returns:
        Success: {"message": str, "contact_id": int}
        Failure: {"message": str}
    """
    bridge = get_appworld_bridge()
    appworld_state = get_appworld_state()

    params: dict[str, Any] = {
        "access_token": appworld_state.get_access_token("phone"),
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
    }
    if phone_number is not None:
        params["phone_number"] = phone_number
    if relationships is not None:
        params["relationships"] = relationships
    if birthday is not None:
        params["birthday"] = birthday
    if home_address is not None:
        params["home_address"] = home_address
    if work_address is not None:
        params["work_address"] = work_address

    response = bridge.call_api("phone", "add_contact", method="post", **params)

    if isinstance(response, dict) and "error" in response:
        raise ValueError(f"API error: {response['error']}")

    return dict(response) if isinstance(response, dict) else {}


@register_appworld_tool("contacts")
@requires_network
def contacts_update_contact(
    contact_id: int,
    first_name: str | None = None,
    last_name: str | None = None,
    phone_number: str | None = None,
    email: str | None = None,
    relationships: list[str] | None = None,
    birthday: str | None = None,
    home_address: str | None = None,
    work_address: str | None = None,
) -> dict[str, Any]:
    """
    Update contact information.

    Args:
        contact_id: ID of the contact to update.
        first_name: Updated first name of the contact. (optional)
        last_name: Updated last name of the contact. (optional)
        phone_number: Updated phone number of the contact. (optional)
        email: Updated email of the contact. Must be a valid email address. (optional)
        relationships: Updated relationship with the contact. (optional)
        birthday: Updated birthday of the contact in YYYY-MM-DD format. (optional)
        home_address: Updated home address of the contact. (optional)
        work_address: Updated work address of the contact. (optional)

    Returns:
        Success: {"message": str}
        Failure: {"message": str}
    """
    bridge = get_appworld_bridge()
    appworld_state = get_appworld_state()

    params: dict[str, Any] = {
        "access_token": appworld_state.get_access_token("phone"),
        "contact_id": contact_id,
    }
    if first_name is not None:
        params["first_name"] = first_name
    if last_name is not None:
        params["last_name"] = last_name
    if phone_number is not None:
        params["phone_number"] = phone_number
    if email is not None:
        params["email"] = email
    if relationships is not None:
        params["relationships"] = relationships
    if birthday is not None:
        params["birthday"] = birthday
    if home_address is not None:
        params["home_address"] = home_address
    if work_address is not None:
        params["work_address"] = work_address

    response = bridge.call_api("phone", "update_contact", method="patch", **params)

    if isinstance(response, dict) and "error" in response:
        raise ValueError(f"API error: {response['error']}")

    return dict(response) if isinstance(response, dict) else {}


@register_appworld_tool("contacts")
@requires_network
def contacts_delete_contact(contact_id: int) -> dict[str, Any]:
    """
    Delete contact information.

    Args:
        contact_id: ID of the contact to be deleted.

    Returns:
        Success: {"message": str}
        Failure: {"message": str}
    """
    bridge = get_appworld_bridge()
    appworld_state = get_appworld_state()

    params: dict[str, Any] = {
        "access_token": appworld_state.get_access_token("phone"),
        "contact_id": contact_id,
    }

    response = bridge.call_api("phone", "delete_contact", method="delete", **params)

    if isinstance(response, dict) and "error" in response:
        raise ValueError(f"API error: {response['error']}")

    return dict(response) if isinstance(response, dict) else {}


@register_appworld_tool("contacts")
@requires_network
def contacts_show_profile(phone_number: str | None = None) -> dict[str, Any]:
    """
    Show public profile information of a user.

    Args:
        phone_number: Phone number of the person you want to see the profile information of. (optional)

    Returns:
        Success: {"first_name": str, "last_name": str, "phone_number": str, "registered_at": str (datetime)}
        Failure: {"message": str}
    """
    bridge = get_appworld_bridge()
    appworld_state = get_appworld_state()

    params: dict[str, Any] = {}
    if phone_number is not None:
        params["phone_number"] = phone_number

    response = bridge.call_api("phone", "show_profile", method="get", **params)

    if isinstance(response, dict) and "error" in response:
        raise ValueError(f"API error: {response['error']}")

    return dict(response) if isinstance(response, dict) else {}
