# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""COMPACT Splitwise tools — merges balance (4->1), expenses (2->1), payments (2->1), receipts (6->1)."""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.datasets.names import DatasetName
from mmtoolsandbox.tools.compact import mark_compact_tools_absorbed_by


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.appworld.splitwise as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# Strategy 4: Balance viewing (4 -> 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def splitwise_show_balance(
    scope: Literal["person", "all_people", "group", "all_groups"],
    email: str | NotGiven = NOT_GIVEN,
    group_id: int | None | NotGiven = NOT_GIVEN,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Show Splitwise balance for a person, all people, a group, or all groups.

    Scope values:
        person: Show amounts you and a given person owe each other, broken
            down by group. Requires email.
        all_people: Show aggregate amounts you owe to each person and they
            owe to you. No extra parameters needed.
        group: Show detailed breakdown of amounts members owe each other in
            a group. Optionally pass group_id (None for non-grouped) and
            email to filter by member.
        all_groups: Show aggregate amounts you owe to others or others owe
            you for each group. No extra parameters needed.

    Args:
        scope: Which balance view to show.
        email: Email of the person (for person scope) or member to filter
            (for group scope).
        group_id: Group ID (for group scope). None shows non-grouped
            balance.

    Returns:
        Balance information (dict or list depending on scope).

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Splitwise.
    """
    if scope == "person":
        return _get("splitwise_show_person_balance")(email=email)
    elif scope == "all_people":
        return _get("splitwise_show_people_balance")()
    elif scope == "group":
        kwargs: dict[str, Any] = {}
        if group_id is not NOT_GIVEN:
            kwargs["group_id"] = group_id
        if email is not NOT_GIVEN:
            kwargs["email"] = email
        return _get("splitwise_show_group_balance")(**kwargs)
    elif scope == "all_groups":
        return _get("splitwise_show_groups_balance")()
    else:
        raise ValueError(f"Unknown scope: {scope}")


# ---------------------------------------------------------------------------
# Strategy 4: Expense listing (2 -> 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def splitwise_show_expenses(
    scope: Literal["group", "no_group"],
    group_id: int | NotGiven = NOT_GIVEN,
    query: str | None = "",
    participant_email: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    min_created_at: str | None = None,
    max_created_at: str | None = None,
    deleted: bool | None = False,
    page_index: int | None = 0,
    page_limit: int | None = 5,
    sort_by: str | None = None,
) -> list[dict[str, Any]]:
    """Show or search Splitwise expenses in a group or outside any group.

    Scope values:
        group: Show expenses from a given group. Requires group_id.
        no_group: Show expenses not part of any group.

    Args:
        scope: Whether to show group or non-group expenses.
        group_id: The ID of the group (required for "group" scope).
        query: Search query string.
        participant_email: Email of payer or debtor to filter by.
        min_amount: Minimum expense amount filter. Omit for no lower bound.
        max_amount: Maximum expense amount filter. Omit for no upper bound.
        min_created_at: Minimum creation date (YYYY-MM-DD). Omit for no lower bound.
        max_created_at: Maximum creation date (YYYY-MM-DD). Omit for no upper bound.
        deleted: Filter by deleted status.
        page_index: Zero-based page index for pagination.
        page_limit: Maximum results per page.
        sort_by: Sort field prefixed with +/- for ascending/descending.
            Valid: created_at, amount.

    Returns:
        List of matching expenses.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Splitwise.
    """
    kwargs: dict[str, Any] = {}
    if query is not None:
        kwargs["query"] = query
    if participant_email is not None:
        kwargs["participant_email"] = participant_email
    if min_amount is not None:
        kwargs["min_amount"] = min_amount
    if max_amount is not None:
        kwargs["max_amount"] = max_amount
    if min_created_at is not None:
        kwargs["min_created_at"] = min_created_at
    if max_created_at is not None:
        kwargs["max_created_at"] = max_created_at
    if deleted is not None:
        kwargs["deleted"] = deleted
    if page_index is not None:
        kwargs["page_index"] = page_index
    if page_limit is not None:
        kwargs["page_limit"] = page_limit
    if sort_by is not None:
        kwargs["sort_by"] = sort_by

    if scope == "group":
        kwargs["group_id"] = group_id
        return _get("splitwise_show_group_expenses")(**kwargs)
    elif scope == "no_group":
        return _get("splitwise_show_no_group_expenses")(**kwargs)
    else:
        raise ValueError(f"Unknown scope: {scope}")


# ---------------------------------------------------------------------------
# Strategy 4: Payment listing (2 -> 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def splitwise_show_payments(
    scope: Literal["group", "no_group"],
    group_id: int | NotGiven = NOT_GIVEN,
    query: str | None = "",
    participant_email: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    min_created_at: str | None = None,
    max_created_at: str | None = None,
    deleted: bool | None = False,
    page_index: int | None = 0,
    page_limit: int | None = 5,
    sort_by: str | None = None,
) -> list[dict[str, Any]]:
    """Show or search Splitwise payments in a group or outside any group.

    Scope values:
        group: Show payments from a given group. Requires group_id.
        no_group: Show payments not part of any group.

    Args:
        scope: Whether to show group or non-group payments.
        group_id: The ID of the group (required for "group" scope).
        query: Search query string.
        participant_email: Email of payer or receiver to filter by.
        min_amount: Minimum payment amount filter. Omit for no lower bound.
        max_amount: Maximum payment amount filter. Omit for no upper bound.
        min_created_at: Minimum creation date (YYYY-MM-DD). Omit for no lower bound.
        max_created_at: Maximum creation date (YYYY-MM-DD). Omit for no upper bound.
        deleted: Filter by deleted status.
        page_index: Zero-based page index for pagination.
        page_limit: Maximum results per page.
        sort_by: Sort field prefixed with +/- for ascending/descending.
            Valid: created_at, amount.

    Returns:
        List of matching payments.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Splitwise.
    """
    kwargs: dict[str, Any] = {}
    if query is not None:
        kwargs["query"] = query
    if participant_email is not None:
        kwargs["participant_email"] = participant_email
    if min_amount is not None:
        kwargs["min_amount"] = min_amount
    if max_amount is not None:
        kwargs["max_amount"] = max_amount
    if min_created_at is not None:
        kwargs["min_created_at"] = min_created_at
    if max_created_at is not None:
        kwargs["max_created_at"] = max_created_at
    if deleted is not None:
        kwargs["deleted"] = deleted
    if page_index is not None:
        kwargs["page_index"] = page_index
    if page_limit is not None:
        kwargs["page_limit"] = page_limit
    if sort_by is not None:
        kwargs["sort_by"] = sort_by

    if scope == "group":
        kwargs["group_id"] = group_id
        return _get("splitwise_show_group_payments")(**kwargs)
    elif scope == "no_group":
        return _get("splitwise_show_no_group_payments")(**kwargs)
    else:
        raise ValueError(f"Unknown scope: {scope}")


# ---------------------------------------------------------------------------
# Strategy 5: Receipt management (6 -> 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def splitwise_manage_receipt(
    entity_type: Literal["expense", "payment"],
    action: Literal["attach", "download", "delete"],
    entity_id: int | NotGiven = NOT_GIVEN,
    receipt_file_path: str | NotGiven = NOT_GIVEN,
    file_system_access_token: str | NotGiven = NOT_GIVEN,
    download_to_file_path: str | None = None,
    overwrite: bool | None = False,
) -> dict[str, Any]:
    """Attach, download, or delete receipt files on Splitwise expenses or payments.

    Actions:
        attach: Attach a receipt file. Requires entity_id,
            receipt_file_path, and file_system_access_token.
        download: Download a receipt file. Requires entity_id and
            file_system_access_token.
        delete: Delete a receipt file. Requires entity_id only.

    Args:
        entity_type: Whether the receipt belongs to an "expense" or
            "payment".
        action: The receipt operation to perform.
        entity_id: The ID of the expense or payment.
        receipt_file_path: File path to attach as receipt (for attach).
        file_system_access_token: Access token from file_system login
            (for attach, download).
        download_to_file_path: Destination path for download. Defaults to
            ~/downloads directory.
        overwrite: Whether to overwrite existing files.

    Returns:
        Operation result details.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Splitwise.
    """
    # Build the function name: splitwise_{action}_{entity_type}_receipt_file
    id_key = "expense_id" if entity_type == "expense" else "payment_id"

    if action == "attach":
        func_name = f"splitwise_attach_{entity_type}_receipt_file"
        kwargs: dict[str, Any] = {
            id_key: entity_id,
            "receipt_file_path": receipt_file_path,
            "file_system_access_token": file_system_access_token,
        }
        if overwrite is not None:
            kwargs["overwrite"] = overwrite
        return _get(func_name)(**kwargs)
    elif action == "download":
        func_name = f"splitwise_download_{entity_type}_receipt_file"
        kwargs = {
            id_key: entity_id,
            "file_system_access_token": file_system_access_token,
        }
        if download_to_file_path is not None:
            kwargs["download_to_file_path"] = download_to_file_path
        if overwrite is not None:
            kwargs["overwrite"] = overwrite
        return _get(func_name)(**kwargs)
    elif action == "delete":
        func_name = f"splitwise_delete_{entity_type}_receipt_file"
        return _get(func_name)(**{id_key: entity_id})
    else:
        raise ValueError(f"Unknown action: {action}")


mark_compact_tools_absorbed_by(
    "splitwise_show_balance",
    "splitwise_show_person_balance",
    "splitwise_show_people_balance",
    "splitwise_show_group_balance",
    "splitwise_show_groups_balance",
)
mark_compact_tools_absorbed_by(
    "splitwise_show_expenses",
    "splitwise_show_group_expenses",
    "splitwise_show_no_group_expenses",
)
mark_compact_tools_absorbed_by(
    "splitwise_show_payments",
    "splitwise_show_group_payments",
    "splitwise_show_no_group_payments",
)
mark_compact_tools_absorbed_by(
    "splitwise_manage_receipt",
    "splitwise_attach_expense_receipt_file",
    "splitwise_delete_expense_receipt_file",
    "splitwise_download_expense_receipt_file",
    "splitwise_attach_payment_receipt_file",
    "splitwise_delete_payment_receipt_file",
    "splitwise_download_payment_receipt_file",
)


# ---------------------------------------------------------------------------
# Lazy import helper — dispatch to MEDIUM consolidated splitwise tools
# ---------------------------------------------------------------------------


def _get_consolidated(name: str) -> Any:
    import mmtoolsandbox.tools.consolidated.splitwise as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# CRUD+List: Splitwise group management (absorbs splitwise_show_groups)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def splitwise_manage_group(
    action: Literal["create", "show", "update", "delete", "undelete", "list"],
    group_id: int | NotGiven = NOT_GIVEN,
    name: str | NotGiven = NOT_GIVEN,
    member_emails: list[str] | NotGiven = NOT_GIVEN,
    description: str | None | NotGiven = NOT_GIVEN,
    # list-action params
    include_deleted: bool | None = False,
    page_index: int | None = 0,
    page_limit: int | None = 5,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Manage Splitwise groups: create, view, update, delete, restore, or list.

    Actions:
        create: Create a new group. Requires name and member_emails.
            Optionally include a description.
        show: View details of a group. Requires group_id.
        update: Update group name or description. Requires group_id and at
            least one of name or description.
        delete: Delete a group. Any member can undelete it later. Requires
            group_id.
        undelete: Restore a previously deleted group. Requires group_id.
        list: List groups you are a member of. Supports filtering by
            deleted status and pagination.

    Args:
        action: The operation to perform.
        group_id: The group ID (for show, update, delete, undelete).
        name: Group name (for create, update).
        member_emails: Emails of users to add to the group besides yourself
            (for create).
        description: Group description (for create, update).
        include_deleted: Include deleted groups in results (for list).
        page_index: Zero-based page index (for list).
        page_limit: Maximum results per page (for list).

    Returns:
        Group details, action confirmation, or list of groups.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Splitwise.
    """
    if action == "create":
        kwargs: dict[str, Any] = {
            "name": name,
            "member_emails": member_emails,
        }
        if description is not NOT_GIVEN:
            kwargs["description"] = description
        return _get("splitwise_create_group")(**kwargs)
    elif action == "show":
        return _get("splitwise_show_group")(group_id=group_id)
    elif action == "update":
        kwargs = {"group_id": group_id}
        if name is not NOT_GIVEN:
            kwargs["name"] = name
        if description is not NOT_GIVEN:
            kwargs["description"] = description
        return _get("splitwise_update_group")(**kwargs)
    elif action == "delete":
        return _get("splitwise_delete_group")(group_id=group_id)
    elif action == "undelete":
        return _get("splitwise_undelete_group")(group_id=group_id)
    elif action == "list":
        kwargs = {}
        if include_deleted is not None:
            kwargs["delete"] = include_deleted
        if page_index is not None:
            kwargs["page_index"] = page_index
        if page_limit is not None:
            kwargs["page_limit"] = page_limit
        return _get("splitwise_show_groups")(**kwargs)
    else:
        raise ValueError(f"Unknown action: {action}")


mark_compact_tools_absorbed_by(
    "splitwise_manage_group",
    "splitwise_show_groups",
)


# ---------------------------------------------------------------------------
# Strategy 5: Splitwise invitation management (3 -> 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def splitwise_manage_invitation(
    action: Literal["accept", "exit", "regenerate"],
    invitation_code: str | NotGiven = NOT_GIVEN,
    group_id: int | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Accept a group invitation, exit a group, or regenerate a group invitation code.

    Actions:
        accept: Accept a group invitation shared with you by a member.
            Requires invitation_code.
        exit: Exit from a group you are a part of. Requires group_id.
        regenerate: Regenerate the invitation code for a group, invalidating
            the old code. Requires group_id.

    Args:
        action: The invitation operation to perform.
        invitation_code: The group invitation code (required for accept).
        group_id: The ID of the group (required for exit and regenerate).

    Returns:
        Operation result details.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Splitwise.
    """
    if action == "accept":
        return _get("splitwise_accept_group_invitation")(
            invitation_code=invitation_code,
        )
    elif action == "exit":
        return _get("splitwise_exit_group")(group_id=group_id)
    elif action == "regenerate":
        return _get("splitwise_regenerate_invitation_code")(
            group_id=group_id,
        )
    else:
        raise ValueError(f"Unknown action: {action}")


mark_compact_tools_absorbed_by(
    "splitwise_manage_invitation",
    "splitwise_accept_group_invitation",
    "splitwise_exit_group",
    "splitwise_regenerate_invitation_code",
)


# ---------------------------------------------------------------------------
# Strategy 7: Splitwise comment management — collapse MEDIUM by entity subtype
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def splitwise_manage_comment(
    entity_type: Literal["expense", "payment"],
    action: Literal["post", "show", "update", "delete"],
    entity_id: int | NotGiven = NOT_GIVEN,
    comment_id: int | NotGiven = NOT_GIVEN,
    comment: str | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Post, view, update, or delete comments on Splitwise expenses or payments.

    Actions:
        post: Post a new comment. Requires entity_id and comment.
        show: View a single comment. Requires comment_id.
        update: Update a comment you posted. Requires comment_id and
            comment.
        delete: Delete a comment you posted. Requires comment_id.
            Irreversible.

    Args:
        entity_type: Whether the comment belongs to an "expense" or
            "payment".
        action: The operation to perform.
        entity_id: The expense or payment ID (for post).
        comment_id: The comment ID (for show, update, delete).
        comment: The comment text (for post, update).

    Returns:
        Comment details or action confirmation.
    """
    func_map = {
        "expense": "splitwise_manage_expense_comment",
        "payment": "splitwise_manage_payment_comment",
    }
    func_name = func_map[entity_type]
    id_param = f"{entity_type}_id"

    kwargs: dict[str, Any] = {"action": action}
    if action == "post":
        kwargs[id_param] = entity_id
        kwargs["comment"] = comment
    elif action == "show":
        kwargs["comment_id"] = comment_id
    elif action == "update":
        kwargs["comment_id"] = comment_id
        kwargs["comment"] = comment
    elif action == "delete":
        kwargs["comment_id"] = comment_id
    else:
        raise ValueError(f"Unknown action: {action}")

    return _get_consolidated(func_name)(**kwargs)


mark_compact_tools_absorbed_by(
    "splitwise_manage_comment",
    "splitwise_manage_expense_comment",
    "splitwise_manage_payment_comment",
)
