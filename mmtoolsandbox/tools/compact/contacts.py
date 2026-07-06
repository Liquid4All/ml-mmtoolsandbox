# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""COMPACT contacts tools — CRUD + search for contact management."""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.datasets.names import DatasetName
from mmtoolsandbox.tools.compact import mark_compact_tools_absorbed_by


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.appworld.contacts as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# CRUD+Search: Contact management (absorbs contacts_search_contacts)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def contacts_manage_contact(
    action: Literal["add", "update", "delete", "search"],
    contact_id: int | NotGiven = NOT_GIVEN,
    first_name: str | NotGiven = NOT_GIVEN,
    last_name: str | NotGiven = NOT_GIVEN,
    email: str | NotGiven = NOT_GIVEN,
    phone_number: str | None = None,
    relationships: list[str] | None = None,
    birthday: str | None = None,
    home_address: str | None = None,
    work_address: str | None = None,
    # search-action params
    query: str | None = "",
    relationship: str | None = None,
    page_index: int | None = 0,
    page_limit: int | None = 5,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Manage contacts: add, update, delete, or search.

    Each contact must have a unique email address.

    Actions:
        add: Add a new contact. Requires first_name, last_name, and email.
            Optionally include phone_number, relationships, birthday
            (YYYY-MM-DD), home_address, and work_address.
        update: Update contact information. Requires contact_id and at least
            one field to update.
        delete: Delete a contact. Requires contact_id.
        search: Search your contact book. Supports query, relationship
            filter, and pagination.

    Args:
        action: The operation to perform.
        contact_id: The contact ID (for update, delete).
        first_name: First name of the contact (for add, update).
        last_name: Last name of the contact (for add, update).
        email: Email of the contact (for add, update).
        phone_number: Phone number of the contact (for add, update).
        relationships: Relationship with the contact (for add, update).
        birthday: Birthday in YYYY-MM-DD format (for add, update).
        home_address: Home address of the contact (for add, update).
        work_address: Work address of the contact (for add, update).
        query: Search query for contacts (for search).
        relationship: Relationship to filter by (for search).
        page_index: Zero-based page index (for search).
        page_limit: Maximum results per page (for search).

    Returns:
        Contact details, action confirmation, or list of contacts.

    Raises:
        ConnectionError: If network is unavailable.
    """
    if action == "add":
        kwargs: dict[str, Any] = {
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
        }
        if phone_number is not None:
            kwargs["phone_number"] = phone_number
        if relationships is not None:
            kwargs["relationships"] = relationships
        if birthday is not None:
            kwargs["birthday"] = birthday
        if home_address is not None:
            kwargs["home_address"] = home_address
        if work_address is not None:
            kwargs["work_address"] = work_address
        return _get("contacts_add_contact")(**kwargs)
    elif action == "update":
        kwargs = {"contact_id": contact_id}
        if first_name is not NOT_GIVEN:
            kwargs["first_name"] = first_name
        if last_name is not NOT_GIVEN:
            kwargs["last_name"] = last_name
        if email is not NOT_GIVEN:
            kwargs["email"] = email
        if phone_number is not None:
            kwargs["phone_number"] = phone_number
        if relationships is not None:
            kwargs["relationships"] = relationships
        if birthday is not None:
            kwargs["birthday"] = birthday
        if home_address is not None:
            kwargs["home_address"] = home_address
        if work_address is not None:
            kwargs["work_address"] = work_address
        return _get("contacts_update_contact")(**kwargs)
    elif action == "delete":
        return _get("contacts_delete_contact")(contact_id=contact_id)
    elif action == "search":
        kwargs = {}
        if query is not None:
            kwargs["query"] = query
        if relationship is not None:
            kwargs["relationship"] = relationship
        if page_index is not None:
            kwargs["page_index"] = page_index
        if page_limit is not None:
            kwargs["page_limit"] = page_limit
        return _get("contacts_search_contacts")(**kwargs)
    else:
        raise ValueError(f"Unknown action: {action}")


mark_compact_tools_absorbed_by(
    "contacts_manage_contact",
    "contacts_search_contacts",
)
