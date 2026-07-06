# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""Consolidated Splitwise tools for the MEDIUM toolbox.

CRUD consolidation for groups, expenses, payments, and comments, plus
a symmetric pair merge for group member add/remove.
"""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.toolbox.names import ToolboxName
from mmtoolsandbox.tools.consolidated import mark_tools_absorbed_by


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.appworld.splitwise as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# CRUD: Group management
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def splitwise_manage_group(
    action: Literal["create", "show", "update", "delete", "undelete"],
    group_id: int | NotGiven = NOT_GIVEN,
    name: str | NotGiven = NOT_GIVEN,
    member_emails: list[str] | NotGiven = NOT_GIVEN,
    description: str | None | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Manage Splitwise groups: create, view, update, delete, or restore.

    Actions:
        create: Create a new group. Requires name and member_emails.
            Optionally include a description.
        show: View details of a group. Requires group_id.
        update: Update group name or description. Requires group_id and at
            least one of name or description.
        delete: Delete a group. Any member can undelete it later. Requires
            group_id.
        undelete: Restore a previously deleted group. Requires group_id.

    Args:
        action: The operation to perform.
        group_id: The group ID (for show, update, delete, undelete).
        name: Group name (for create, update).
        member_emails: Emails of users to add to the group besides yourself
            (for create).
        description: Group description (for create, update).

    Returns:
        Group details or action confirmation.

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
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# CRUD: Expense management
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def splitwise_manage_expense(
    action: Literal["record", "show", "update", "delete", "undelete"],
    expense_id: int | NotGiven = NOT_GIVEN,
    description: str | NotGiven = NOT_GIVEN,
    paid_amount: float | NotGiven = NOT_GIVEN,
    payer_email: str | NotGiven = NOT_GIVEN,
    debtor_emails: list[str] | NotGiven = NOT_GIVEN,
    debt_amounts: list[float] | None | NotGiven = NOT_GIVEN,
    group_id: int | None | NotGiven = NOT_GIVEN,
    receipt_file_path: str | None | NotGiven = NOT_GIVEN,
    file_system_access_token: str | None | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Manage Splitwise expenses: record, view, update, delete, or restore.

    Actions:
        record: Record a new shared expense. Requires description,
            paid_amount, payer_email, and debtor_emails. If debt_amounts is
            not provided, each debtor owes an equal share. Optionally attach
            to a group_id and include a receipt file.
        show: View details of an expense. Requires expense_id.
        update: Update expense fields. Requires expense_id and at least one
            of description, paid_amount, payer_email, debtor_emails, or
            debt_amounts.
        delete: Delete an expense. Anyone involved can undelete it later.
            Requires expense_id.
        undelete: Restore a previously deleted expense. Requires expense_id.

    Args:
        action: The operation to perform.
        expense_id: The expense ID (for show, update, delete, undelete).
        description: A short note or description of the expense (for record,
            update).
        paid_amount: The total amount paid (for record, update).
        payer_email: Email of the user who paid (for record, update).
        debtor_emails: Emails of users who owe a share (for record, update).
        debt_amounts: Amounts owed by each debtor. Must match the length of
            debtor_emails if provided. If omitted, equal split is assumed
            (for record, update).
        group_id: The group this expense belongs to. None for ungrouped
            (for record).
        receipt_file_path: Absolute file path from the file_system app to
            attach as a receipt (for record).
        file_system_access_token: Access token from file_system app login.
            Only needed when attaching a receipt (for record).

    Returns:
        Expense details or action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Splitwise.
    """
    if action == "record":
        kwargs: dict[str, Any] = {
            "description": description,
            "paid_amount": paid_amount,
            "payer_email": payer_email,
            "debtor_emails": debtor_emails,
        }
        if debt_amounts is not NOT_GIVEN:
            kwargs["debt_amounts"] = debt_amounts
        if group_id is not NOT_GIVEN:
            kwargs["group_id"] = group_id
        if receipt_file_path is not NOT_GIVEN:
            kwargs["receipt_file_path"] = receipt_file_path
        if file_system_access_token is not NOT_GIVEN:
            kwargs["file_system_access_token"] = file_system_access_token
        return _get("splitwise_record_expense")(**kwargs)
    elif action == "show":
        return _get("splitwise_show_expense")(expense_id=expense_id)
    elif action == "update":
        kwargs = {"expense_id": expense_id}
        if description is not NOT_GIVEN:
            kwargs["description"] = description
        if paid_amount is not NOT_GIVEN:
            kwargs["paid_amount"] = paid_amount
        if payer_email is not NOT_GIVEN:
            kwargs["payer_email"] = payer_email
        if debtor_emails is not NOT_GIVEN:
            kwargs["debtor_emails"] = debtor_emails
        if debt_amounts is not NOT_GIVEN:
            kwargs["debt_amounts"] = debt_amounts
        return _get("splitwise_update_expense")(**kwargs)
    elif action == "delete":
        return _get("splitwise_delete_expense")(expense_id=expense_id)
    elif action == "undelete":
        return _get("splitwise_undelete_expense")(expense_id=expense_id)
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# CRUD: Payment management
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def splitwise_manage_payment(
    action: Literal["record", "show", "update", "delete", "undelete"],
    payment_id: int | NotGiven = NOT_GIVEN,
    payer_email: str | NotGiven = NOT_GIVEN,
    receiver_email: str | NotGiven = NOT_GIVEN,
    amount: float | NotGiven = NOT_GIVEN,
    group_id: int | None | NotGiven = NOT_GIVEN,
    description: str | None | NotGiven = NOT_GIVEN,
    receipt_file_path: str | None | NotGiven = NOT_GIVEN,
    file_system_access_token: str | None | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Manage Splitwise payments: record, view, update, delete, or restore.

    This only records payments on Splitwise and does not move real money.

    Actions:
        record: Record a new payment. Requires payer_email, receiver_email,
            and amount. Optionally attach to a group_id, include a
            description, and attach a receipt file.
        show: View details of a payment. Requires payment_id.
        update: Update payment amount or description. Requires payment_id
            and at least one of amount or description.
        delete: Delete a payment. Anyone involved can undelete it later.
            Requires payment_id.
        undelete: Restore a previously deleted payment. Requires payment_id.

    Args:
        action: The operation to perform.
        payment_id: The payment ID (for show, update, delete, undelete).
        payer_email: Email of the user who made the payment (for record).
        receiver_email: Email of the user who received the payment
            (for record).
        amount: The payment amount (for record, update).
        group_id: The group to record the payment in. None for ungrouped
            (for record).
        description: A short note or description (for record, update).
        receipt_file_path: Receipt file path to attach as evidence of
            payment (for record).
        file_system_access_token: Access token from file_system app login.
            Only needed when attaching a receipt (for record).

    Returns:
        Payment details or action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Splitwise.
    """
    if action == "record":
        kwargs: dict[str, Any] = {
            "payer_email": payer_email,
            "receiver_email": receiver_email,
            "amount": amount,
        }
        if group_id is not NOT_GIVEN:
            kwargs["group_id"] = group_id
        if description is not NOT_GIVEN:
            kwargs["description"] = description
        if receipt_file_path is not NOT_GIVEN:
            kwargs["receipt_file_path"] = receipt_file_path
        if file_system_access_token is not NOT_GIVEN:
            kwargs["file_system_access_token"] = file_system_access_token
        return _get("splitwise_record_payment")(**kwargs)
    elif action == "show":
        return _get("splitwise_show_payment")(payment_id=payment_id)
    elif action == "update":
        kwargs = {"payment_id": payment_id}
        if amount is not NOT_GIVEN:
            kwargs["amount"] = amount
        if description is not NOT_GIVEN:
            kwargs["description"] = description
        return _get("splitwise_update_payment")(**kwargs)
    elif action == "delete":
        return _get("splitwise_delete_payment")(payment_id=payment_id)
    elif action == "undelete":
        return _get("splitwise_undelete_payment")(payment_id=payment_id)
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# CRUD: Expense comment management
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def splitwise_manage_expense_comment(
    action: Literal["post", "show", "update", "delete"],
    expense_id: int | NotGiven = NOT_GIVEN,
    comment_id: int | NotGiven = NOT_GIVEN,
    comment: str | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Manage comments on Splitwise expenses: post, view, update, or delete.

    Actions:
        post: Post a new comment on an expense. Requires expense_id and
            comment.
        show: View a single expense comment. Requires comment_id.
        update: Update a comment you posted. Requires comment_id and
            comment.
        delete: Delete a comment you posted. Requires comment_id.

    Args:
        action: The operation to perform.
        expense_id: The expense ID (for post).
        comment_id: The comment ID (for show, update, delete).
        comment: The comment text (for post, update).

    Returns:
        Comment details or action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Splitwise.
    """
    if action == "post":
        return _get("splitwise_post_expense_comment")(
            expense_id=expense_id, comment=comment
        )
    elif action == "show":
        return _get("splitwise_show_expense_comment")(comment_id=comment_id)
    elif action == "update":
        return _get("splitwise_update_expense_comment")(
            comment_id=comment_id, comment=comment
        )
    elif action == "delete":
        return _get("splitwise_delete_expense_comment")(comment_id=comment_id)
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# CRUD: Payment comment management
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def splitwise_manage_payment_comment(
    action: Literal["post", "show", "update", "delete"],
    payment_id: int | NotGiven = NOT_GIVEN,
    comment_id: int | NotGiven = NOT_GIVEN,
    comment: str | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Manage comments on Splitwise payments: post, view, update, or delete.

    Actions:
        post: Post a new comment on a payment. Requires payment_id and
            comment.
        show: View a single payment comment. Requires comment_id.
        update: Update a comment you posted. Requires comment_id and
            comment.
        delete: Delete a comment you posted. Requires comment_id.

    Args:
        action: The operation to perform.
        payment_id: The payment ID (for post).
        comment_id: The comment ID (for show, update, delete).
        comment: The comment text (for post, update).

    Returns:
        Comment details or action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Splitwise.
    """
    if action == "post":
        return _get("splitwise_post_payment_comment")(
            payment_id=payment_id, comment=comment
        )
    elif action == "show":
        return _get("splitwise_show_payment_comment")(comment_id=comment_id)
    elif action == "update":
        return _get("splitwise_update_payment_comment")(
            comment_id=comment_id, comment=comment
        )
    elif action == "delete":
        return _get("splitwise_delete_payment_comment")(comment_id=comment_id)
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Symmetric pair: Group member add/remove
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def splitwise_manage_group_member(
    group_id: int,
    user_email: str,
    action: Literal["add", "remove"],
) -> dict[str, Any]:
    """Add or remove a member from a Splitwise group.

    Args:
        group_id: The ID of the group.
        user_email: Email of the user to add or remove.
        action: "add" to add the user to the group, "remove" to remove
            them.

    Returns:
        Action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Splitwise.
    """
    if action == "add":
        return _get("splitwise_add_member_to_group")(
            group_id=group_id, user_email=user_email
        )
    elif action == "remove":
        return _get("splitwise_remove_member_from_group")(
            group_id=group_id, user_email=user_email
        )
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Mark absorbed tools
# ---------------------------------------------------------------------------

mark_tools_absorbed_by(
    "splitwise_manage_group",
    "splitwise_create_group",
    "splitwise_show_group",
    "splitwise_update_group",
    "splitwise_delete_group",
    "splitwise_undelete_group",
)
mark_tools_absorbed_by(
    "splitwise_manage_expense",
    "splitwise_record_expense",
    "splitwise_show_expense",
    "splitwise_update_expense",
    "splitwise_delete_expense",
    "splitwise_undelete_expense",
)
mark_tools_absorbed_by(
    "splitwise_manage_payment",
    "splitwise_record_payment",
    "splitwise_show_payment",
    "splitwise_update_payment",
    "splitwise_delete_payment",
    "splitwise_undelete_payment",
)
mark_tools_absorbed_by(
    "splitwise_manage_expense_comment",
    "splitwise_post_expense_comment",
    "splitwise_show_expense_comment",
    "splitwise_update_expense_comment",
    "splitwise_delete_expense_comment",
)
mark_tools_absorbed_by(
    "splitwise_manage_payment_comment",
    "splitwise_post_payment_comment",
    "splitwise_show_payment_comment",
    "splitwise_update_payment_comment",
    "splitwise_delete_payment_comment",
)
mark_tools_absorbed_by(
    "splitwise_manage_group_member",
    "splitwise_add_member_to_group",
    "splitwise_remove_member_from_group",
)
