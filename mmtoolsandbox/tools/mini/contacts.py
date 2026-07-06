# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""MINI contacts tool — unified contact book management."""

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
def contacts(
    action: Literal[
        "add", "update", "delete", "search", "show_relationships", "show_profile"
    ],
    # add/update params
    contact_id: int | NotGiven = NOT_GIVEN,
    first_name: str | NotGiven = NOT_GIVEN,
    last_name: str | NotGiven = NOT_GIVEN,
    email: str | NotGiven = NOT_GIVEN,
    phone_number: str | None | NotGiven = NOT_GIVEN,
    relationships: list[str] | None | NotGiven = NOT_GIVEN,
    birthday: str | None | NotGiven = NOT_GIVEN,
    home_address: str | None | NotGiven = NOT_GIVEN,
    work_address: str | None | NotGiven = NOT_GIVEN,
    # search params
    query: str | None | NotGiven = NOT_GIVEN,
    relationship: str | None | NotGiven = NOT_GIVEN,
    page_index: int | None | NotGiven = NOT_GIVEN,
    page_limit: int | None | NotGiven = NOT_GIVEN,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Add, update, delete, or search contacts and view phone profile.

    Actions:
        add - Add contact. Requires first_name, last_name, email.
            Optional: phone_number, relationships, birthday, home_address,
            work_address.
        update - Update contact. Requires contact_id. All other fields optional.
        delete - Delete contact. Requires contact_id. Irreversible.
        search - Search contacts. Optional: query, relationship, page_index,
            page_limit.
        show_relationships - List available relationship types.
        show_profile - Show phone profile. Optional: phone_number.

    Args:
        action: The contact action.
        contact_id: Contact ID (for update/delete). Returned by add.
        first_name: First name.
        last_name: Last name.
        email: Email address (e.g., "alice@example.com").
        phone_number: Phone number (e.g., "+14155551234").
        relationships: Relationship labels (e.g., ["friend", "colleague"]).
            Use show_relationships to see valid values.
        birthday: Birthday in YYYY-MM-DD format (e.g., "1990-05-15").
        home_address: Home address string.
        work_address: Work address string.
        query: Search query — matches name, email, phone.
        relationship: Filter by relationship type.
        page_index: Zero-based page index.
        page_limit: Results per page.

    Returns:
        For add/show_profile: dict with contact_id, first_name, last_name,
            email, phone_number, relationships, birthday.
        For update/delete: confirmation dict.
        For search: list of contact dicts.
        For show_relationships: list of relationship type strings.

    Raises:
        ValueError: If contact_id not found (for update/delete).
    """
    import mmtoolsandbox.tools.appworld.contacts as m

    if action == "add":
        kwargs: dict[str, Any] = {
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
        }
        if phone_number is not NOT_GIVEN:
            kwargs["phone_number"] = phone_number
        if relationships is not NOT_GIVEN:
            kwargs["relationships"] = relationships
        if birthday is not NOT_GIVEN:
            kwargs["birthday"] = birthday
        if home_address is not NOT_GIVEN:
            kwargs["home_address"] = home_address
        if work_address is not NOT_GIVEN:
            kwargs["work_address"] = work_address
        return m.contacts_add_contact(**kwargs)
    elif action == "update":
        kwargs = {"contact_id": contact_id}
        if first_name is not NOT_GIVEN:
            kwargs["first_name"] = first_name
        if last_name is not NOT_GIVEN:
            kwargs["last_name"] = last_name
        if phone_number is not NOT_GIVEN:
            kwargs["phone_number"] = phone_number
        if email is not NOT_GIVEN:
            kwargs["email"] = email
        if relationships is not NOT_GIVEN:
            kwargs["relationships"] = relationships
        if birthday is not NOT_GIVEN:
            kwargs["birthday"] = birthday
        if home_address is not NOT_GIVEN:
            kwargs["home_address"] = home_address
        if work_address is not NOT_GIVEN:
            kwargs["work_address"] = work_address
        return m.contacts_update_contact(**kwargs)
    elif action == "delete":
        return m.contacts_delete_contact(contact_id=contact_id)
    elif action == "search":
        kwargs = {}
        if query is not NOT_GIVEN:
            kwargs["query"] = query
        if relationship is not NOT_GIVEN:
            kwargs["relationship"] = relationship
        if page_index is not NOT_GIVEN:
            kwargs["page_index"] = page_index
        if page_limit is not NOT_GIVEN:
            kwargs["page_limit"] = page_limit
        return m.contacts_search_contacts(**kwargs)
    elif action == "show_relationships":
        return m.contacts_show_contact_relationships()
    elif action == "show_profile":
        kwargs = {}
        if phone_number is not NOT_GIVEN:
            kwargs["phone_number"] = phone_number
        return m.contacts_show_profile(**kwargs)
    else:
        raise ValueError(f"Unknown action: {action}")
