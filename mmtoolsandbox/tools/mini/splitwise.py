# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""Consolidated Splitwise tools for the MINI toolbox.

Two workflow-based tools covering all Splitwise functionality:
- splitwise_manage: expenses, payments, groups, balances, receipts, settle up
- splitwise_social: group membership, invitations, comments, activity
"""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.datasets.names import DatasetName


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.appworld.splitwise as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# splitwise_manage
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.MINI},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def splitwise_manage(
    domain: Literal["group", "expense", "payment", "balance", "receipt", "settle"],
    action: Literal[
        "create",
        "show",
        "update",
        "delete",
        "undelete",
        "list",
        "record",
        "person",
        "all_people",
        "group",
        "all_groups",
        "attach",
        "download",
        "settle",
        "settle_up",
    ],
    # group params
    group_id: int | None | NotGiven = NOT_GIVEN,
    name: str | NotGiven = NOT_GIVEN,
    member_emails: list[str] | NotGiven = NOT_GIVEN,
    description: str | None | NotGiven = NOT_GIVEN,
    delete: bool | None | NotGiven = NOT_GIVEN,
    # expense params
    expense_id: int | NotGiven = NOT_GIVEN,
    paid_amount: float | NotGiven = NOT_GIVEN,
    payer_email: str | NotGiven = NOT_GIVEN,
    debtor_emails: list[str] | None | NotGiven = NOT_GIVEN,
    debt_amounts: list[float] | None | NotGiven = NOT_GIVEN,
    # payment params
    payment_id: int | NotGiven = NOT_GIVEN,
    receiver_email: str | NotGiven = NOT_GIVEN,
    amount: float | NotGiven = NOT_GIVEN,
    # list/search params
    scope: Literal["group", "no_group"] | NotGiven = NOT_GIVEN,
    query: str | None | NotGiven = NOT_GIVEN,
    participant_email: str | None | NotGiven = NOT_GIVEN,
    min_amount: float | None | NotGiven = NOT_GIVEN,
    max_amount: float | None | NotGiven = NOT_GIVEN,
    min_created_at: str | None | NotGiven = NOT_GIVEN,
    max_created_at: str | None | NotGiven = NOT_GIVEN,
    deleted: bool | None | NotGiven = NOT_GIVEN,
    page_index: int | None | NotGiven = NOT_GIVEN,
    page_limit: int | None | NotGiven = NOT_GIVEN,
    sort_by: str | None | NotGiven = NOT_GIVEN,
    # balance params
    email: str | NotGiven = NOT_GIVEN,
    # receipt params
    entity_type: Literal["expense", "payment"] | NotGiven = NOT_GIVEN,
    receipt_file_path: str | None | NotGiven = NOT_GIVEN,
    file_system_access_token: str | None | NotGiven = NOT_GIVEN,
    download_to_file_path: str | None | NotGiven = NOT_GIVEN,
    overwrite: bool | None | NotGiven = NOT_GIVEN,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Track shared expenses, payments, and balances on Splitwise.

    Domains and actions:
        group:
            create - Create group. Requires name, member_emails.
                Optional: description.
            show - Show group details. Requires group_id.
            update - Update group. Requires group_id. Optional: name, description.
            delete - Delete group. Requires group_id.
            undelete - Restore deleted group. Requires group_id.
            list - List groups. Optional: delete (filter), page_index, page_limit.
        expense:
            record - Record expense. Requires description, paid_amount, payer_email,
                debtor_emails. Optional: debt_amounts, group_id, receipt_file_path,
                file_system_access_token.
            show - Show expense. Requires expense_id.
            update - Update expense. Requires expense_id.
            delete - Delete expense. Requires expense_id.
            undelete - Restore expense. Requires expense_id.
            list - List expenses. Requires scope ("group" or "no_group").
                If scope="group", requires group_id.
        payment:
            record - Record payment. Requires payer_email, receiver_email, amount.
                Optional: group_id, description, receipt_file_path,
                file_system_access_token.
            show - Show payment. Requires payment_id.
            update - Update payment. Requires payment_id.
            delete - Delete payment. Requires payment_id.
            undelete - Restore payment. Requires payment_id.
            list - List payments. Requires scope ("group" or "no_group").
                If scope="group", requires group_id.
        balance:
            person - Balance with one person. Requires email.
            all_people - Balance with all people.
            group - Group balance. Optional: group_id, email.
            all_groups - Balance across all groups.
        receipt:
            attach - Attach receipt. Requires entity_type (expense/payment),
                expense_id or payment_id, receipt_file_path,
                file_system_access_token.
            download - Download receipt. Requires entity_type, expense_id or
                payment_id, file_system_access_token.
            delete - Delete receipt. Requires entity_type, expense_id or payment_id.
        settle:
            settle_up - Settle balance. Requires email. Optional: group_id,
                description.

    Args:
        domain: The financial domain.
        action: The specific action within the domain.
        group_id: Group ID.
        name: Group name (for create/update).
        member_emails: Member emails (for group create).
        description: Description (for group/expense/payment).
        delete: Filter by deleted status (for group list).
        expense_id: Expense ID.
        paid_amount: Total amount paid (for expense record).
        payer_email: Email of payer (for expense/payment).
        debtor_emails: Emails of debtors (for expense).
        debt_amounts: Amounts owed by each debtor (for expense).
        payment_id: Payment ID.
        receiver_email: Email of payment receiver.
        amount: Payment amount.
        scope: "group" or "no_group" (for expense/payment list).
        query: Search query (for list actions).
        participant_email: Filter by participant (for list).
        min_amount: Min amount filter.
        max_amount: Max amount filter.
        min_created_at: Min date YYYY-MM-DD.
        max_created_at: Max date YYYY-MM-DD.
        deleted: Filter by deleted status (for expense/payment list).
        page_index: Page index.
        page_limit: Results per page.
        sort_by: Sort attribute with +/- prefix.
        email: Person email (for balance/settle).
        entity_type: "expense" or "payment" (for receipt).
        receipt_file_path: Path to receipt file (for attach).
        file_system_access_token: File system token (for receipt).
        download_to_file_path: Download path (for receipt download).
        overwrite: Overwrite existing file (for receipt).

    Returns:
        For group/expense/payment create/record: dict with entity ID
            (group_id, expense_id, payment_id). Pass to show/update/delete.
        For delete: confirmation dict. Irreversible (use undelete to restore).
        For balance: dict or list of balance breakdowns.
        For settle: creates a payment to settle debts. Externally visible.
        For receipt attach/download/delete: confirmation dict.

    Raises:
        PermissionError: If not logged into Splitwise.
    """
    if domain == "group":
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
            if delete is not NOT_GIVEN:
                kwargs["delete"] = delete
            if page_index is not NOT_GIVEN:
                kwargs["page_index"] = page_index
            if page_limit is not NOT_GIVEN:
                kwargs["page_limit"] = page_limit
            return _get("splitwise_show_groups")(**kwargs)
        else:
            raise ValueError(f"Unknown group action: {action}")

    elif domain == "expense":
        if action == "record":
            kwargs = {
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
        elif action == "list":
            kwargs = {}
            if query is not NOT_GIVEN:
                kwargs["query"] = query
            if participant_email is not NOT_GIVEN:
                kwargs["participant_email"] = participant_email
            if min_amount is not NOT_GIVEN:
                kwargs["min_amount"] = min_amount
            if max_amount is not NOT_GIVEN:
                kwargs["max_amount"] = max_amount
            if min_created_at is not NOT_GIVEN:
                kwargs["min_created_at"] = min_created_at
            if max_created_at is not NOT_GIVEN:
                kwargs["max_created_at"] = max_created_at
            if deleted is not NOT_GIVEN:
                kwargs["deleted"] = deleted
            if page_index is not NOT_GIVEN:
                kwargs["page_index"] = page_index
            if page_limit is not NOT_GIVEN:
                kwargs["page_limit"] = page_limit
            if sort_by is not NOT_GIVEN:
                kwargs["sort_by"] = sort_by
            if scope == "group":
                kwargs["group_id"] = group_id
                return _get("splitwise_show_group_expenses")(**kwargs)
            else:
                return _get("splitwise_show_no_group_expenses")(**kwargs)
        else:
            raise ValueError(f"Unknown expense action: {action}")

    elif domain == "payment":
        if action == "record":
            kwargs = {
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
        elif action == "list":
            kwargs = {}
            if query is not NOT_GIVEN:
                kwargs["query"] = query
            if participant_email is not NOT_GIVEN:
                kwargs["participant_email"] = participant_email
            if min_amount is not NOT_GIVEN:
                kwargs["min_amount"] = min_amount
            if max_amount is not NOT_GIVEN:
                kwargs["max_amount"] = max_amount
            if min_created_at is not NOT_GIVEN:
                kwargs["min_created_at"] = min_created_at
            if max_created_at is not NOT_GIVEN:
                kwargs["max_created_at"] = max_created_at
            if deleted is not NOT_GIVEN:
                kwargs["deleted"] = deleted
            if page_index is not NOT_GIVEN:
                kwargs["page_index"] = page_index
            if page_limit is not NOT_GIVEN:
                kwargs["page_limit"] = page_limit
            if sort_by is not NOT_GIVEN:
                kwargs["sort_by"] = sort_by
            if scope == "group":
                kwargs["group_id"] = group_id
                return _get("splitwise_show_group_payments")(**kwargs)
            else:
                return _get("splitwise_show_no_group_payments")(**kwargs)
        else:
            raise ValueError(f"Unknown payment action: {action}")

    elif domain == "balance":
        if action == "person":
            return _get("splitwise_show_person_balance")(email=email)
        elif action == "all_people":
            return _get("splitwise_show_people_balance")()
        elif action == "group":
            kwargs = {}
            if group_id is not NOT_GIVEN:
                kwargs["group_id"] = group_id
            if email is not NOT_GIVEN:
                kwargs["email"] = email
            return _get("splitwise_show_group_balance")(**kwargs)
        elif action == "all_groups":
            return _get("splitwise_show_groups_balance")()
        else:
            raise ValueError(f"Unknown balance action: {action}")

    elif domain == "receipt":
        if action == "attach":
            if entity_type == "expense":
                kwargs = {
                    "expense_id": expense_id,
                    "receipt_file_path": receipt_file_path,
                    "file_system_access_token": file_system_access_token,
                }
                if overwrite is not NOT_GIVEN:
                    kwargs["overwrite"] = overwrite
                return _get("splitwise_attach_expense_receipt_file")(**kwargs)
            elif entity_type == "payment":
                kwargs = {
                    "payment_id": payment_id,
                    "receipt_file_path": receipt_file_path,
                    "file_system_access_token": file_system_access_token,
                }
                if overwrite is not NOT_GIVEN:
                    kwargs["overwrite"] = overwrite
                return _get("splitwise_attach_payment_receipt_file")(**kwargs)
            else:
                raise ValueError(f"Unknown entity_type: {entity_type}")
        elif action == "download":
            if entity_type == "expense":
                kwargs = {
                    "expense_id": expense_id,
                    "file_system_access_token": file_system_access_token,
                }
                if download_to_file_path is not NOT_GIVEN:
                    kwargs["download_to_file_path"] = download_to_file_path
                if overwrite is not NOT_GIVEN:
                    kwargs["overwrite"] = overwrite
                return _get("splitwise_download_expense_receipt_file")(**kwargs)
            elif entity_type == "payment":
                kwargs = {
                    "payment_id": payment_id,
                    "file_system_access_token": file_system_access_token,
                }
                if download_to_file_path is not NOT_GIVEN:
                    kwargs["download_to_file_path"] = download_to_file_path
                if overwrite is not NOT_GIVEN:
                    kwargs["overwrite"] = overwrite
                return _get("splitwise_download_payment_receipt_file")(**kwargs)
            else:
                raise ValueError(f"Unknown entity_type: {entity_type}")
        elif action == "delete":
            if entity_type == "expense":
                return _get("splitwise_delete_expense_receipt_file")(
                    expense_id=expense_id
                )
            elif entity_type == "payment":
                return _get("splitwise_delete_payment_receipt_file")(
                    payment_id=payment_id
                )
            else:
                raise ValueError(f"Unknown entity_type: {entity_type}")
        else:
            raise ValueError(f"Unknown receipt action: {action}")

    elif domain == "settle":
        if action == "settle_up":
            kwargs = {"email": email}
            if group_id is not NOT_GIVEN:
                kwargs["group_id"] = group_id
            if description is not NOT_GIVEN:
                kwargs["description"] = description
            return _get("splitwise_settle_up")(**kwargs)
        else:
            raise ValueError(f"Unknown settle action: {action}")

    else:
        raise ValueError(f"Unknown domain: {domain}")


# ---------------------------------------------------------------------------
# splitwise_social
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.MINI},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def splitwise_social(
    domain: Literal["member", "invitation", "comment", "activity"],
    action: Literal[
        "add",
        "remove",
        "accept",
        "exit",
        "regenerate",
        "post",
        "show",
        "update",
        "delete",
        "list",
    ],
    # member params
    group_id: int | NotGiven = NOT_GIVEN,
    user_email: str | NotGiven = NOT_GIVEN,
    # invitation params
    invitation_code: str | NotGiven = NOT_GIVEN,
    # comment params
    entity_type: Literal["expense", "payment"] | NotGiven = NOT_GIVEN,
    expense_id: int | NotGiven = NOT_GIVEN,
    payment_id: int | NotGiven = NOT_GIVEN,
    comment_id: int | NotGiven = NOT_GIVEN,
    comment: str | NotGiven = NOT_GIVEN,
    page_index: int | None | NotGiven = NOT_GIVEN,
    page_limit: int | None | NotGiven = NOT_GIVEN,
    # activity params
    show_expenses: bool | None | NotGiven = NOT_GIVEN,
    show_payments: bool | None | NotGiven = NOT_GIVEN,
    sort_by: str | None | NotGiven = NOT_GIVEN,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Manage Splitwise groups, invitations, and comments.

    Domains and actions:
        member:
            add - Add member to group. Requires group_id, user_email.
            remove - Remove member from group. Requires group_id, user_email.
        invitation:
            accept - Accept group invitation. Requires invitation_code.
            exit - Exit a group. Requires group_id.
            regenerate - Regenerate invitation code. Requires group_id.
        comment:
            post - Post comment. Requires entity_type (expense/payment),
                expense_id or payment_id, comment.
            show - Show comment. Requires entity_type, comment_id.
            update - Update comment. Requires entity_type, comment_id, comment.
            delete - Delete comment. Requires entity_type, comment_id.
            list - List comments. Requires entity_type, expense_id or payment_id.
                Optional: page_index, page_limit.
        activity:
            show - Show expense/payment history. Optional: show_expenses,
                show_payments, page_index, page_limit, sort_by.

    Args:
        domain: The social domain.
        action: The specific action within the domain.
        group_id: Group ID (for member/invitation actions).
        user_email: User email (for member add/remove).
        invitation_code: Invitation code (for accept).
        entity_type: "expense" or "payment" (for comment actions).
        expense_id: Expense ID (for comment on expense).
        payment_id: Payment ID (for comment on payment).
        comment_id: Comment ID (for show/update/delete).
        comment: Comment text (for post/update).
        page_index: Page index for pagination.
        page_limit: Results per page.
        show_expenses: Include expenses in activity (for activity show).
        show_payments: Include payments in activity (for activity show).
        sort_by: Sort attribute with +/- prefix (for activity show).

    Returns:
        For member add/remove: confirmation dict.
        For invitation accept/exit: confirmation dict.
        For invitation regenerate: dict with new invitation_code.
        For comment post: dict with comment_id.
        For comment delete: confirmation. Irreversible.
        For activity show: list of expense/payment activity dicts.
    """
    if domain == "member":
        if action == "add":
            return _get("splitwise_add_member_to_group")(
                group_id=group_id, user_email=user_email
            )
        elif action == "remove":
            return _get("splitwise_remove_member_from_group")(
                group_id=group_id, user_email=user_email
            )
        else:
            raise ValueError(f"Unknown member action: {action}")

    elif domain == "invitation":
        if action == "accept":
            return _get("splitwise_accept_group_invitation")(
                invitation_code=invitation_code
            )
        elif action == "exit":
            return _get("splitwise_exit_group")(group_id=group_id)
        elif action == "regenerate":
            return _get("splitwise_regenerate_invitation_code")(group_id=group_id)
        else:
            raise ValueError(f"Unknown invitation action: {action}")

    elif domain == "comment":
        if entity_type == "expense":
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
            elif action == "list":
                kwargs: dict[str, Any] = {"expense_id": expense_id}
                if page_index is not NOT_GIVEN:
                    kwargs["page_index"] = page_index
                if page_limit is not NOT_GIVEN:
                    kwargs["page_limit"] = page_limit
                return _get("splitwise_show_expense_comments")(**kwargs)
            else:
                raise ValueError(f"Unknown comment action: {action}")
        elif entity_type == "payment":
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
            elif action == "list":
                kwargs = {"payment_id": payment_id}
                if page_index is not NOT_GIVEN:
                    kwargs["page_index"] = page_index
                if page_limit is not NOT_GIVEN:
                    kwargs["page_limit"] = page_limit
                return _get("splitwise_show_payment_comments")(**kwargs)
            else:
                raise ValueError(f"Unknown comment action: {action}")
        else:
            raise ValueError(f"Unknown entity_type: {entity_type}")

    elif domain == "activity":
        if action == "show":
            kwargs = {}
            if show_expenses is not NOT_GIVEN:
                kwargs["show_expenses"] = show_expenses
            if show_payments is not NOT_GIVEN:
                kwargs["show_payments"] = show_payments
            if page_index is not NOT_GIVEN:
                kwargs["page_index"] = page_index
            if page_limit is not NOT_GIVEN:
                kwargs["page_limit"] = page_limit
            if sort_by is not NOT_GIVEN:
                kwargs["sort_by"] = sort_by
            return _get("splitwise_show_activity")(**kwargs)
        else:
            raise ValueError(f"Unknown activity action: {action}")

    else:
        raise ValueError(f"Unknown domain: {domain}")
