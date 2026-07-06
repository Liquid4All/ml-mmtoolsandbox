# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""Consolidated Venmo tools for the MEDIUM toolbox.

CRUD consolidation for transactions, transaction comments, and payment
requests, plus symmetric pair merges for friendship, likes, balance
management, and payment request responses.
"""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.toolbox.names import ToolboxName
from mmtoolsandbox.tools.consolidated import mark_tools_absorbed_by


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.appworld.venmo as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# CRUD: Transaction management
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def venmo_manage_transaction(
    action: Literal["create", "show", "update"],
    transaction_id: int | NotGiven = NOT_GIVEN,
    receiver_email: str | NotGiven = NOT_GIVEN,
    amount: float | NotGiven = NOT_GIVEN,
    description: str | NotGiven = NOT_GIVEN,
    payment_card_id: int | None | NotGiven = NOT_GIVEN,
    private: bool | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Manage Venmo transactions: create (send money), view, or update.

    Actions:
        create: Send money to a user. Requires receiver_email and amount.
            description defaults to empty string, private defaults to false.
            If payment_card_id is not provided, Venmo balance will be used.
        show: View details of a transaction. Requires transaction_id.
        update: Update description or privacy of a transaction. Requires
            transaction_id and at least one of description or private.

    Args:
        action: The operation to perform.
        transaction_id: The transaction ID (for show, update).
        receiver_email: Email address of the receiver (for create).
        amount: Amount of the transaction (for create).
        description: Description of or note about the transaction
            (for create, update).
        payment_card_id: ID of the payment card to use. If not provided,
            Venmo balance will be used (for create).
        private: Whether the transaction is private (for create, update).

    Returns:
        Transaction details or action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Venmo.
    """
    if action == "create":
        kwargs: dict[str, Any] = {
            "receiver_email": receiver_email,
            "amount": amount,
        }
        if description is not NOT_GIVEN:
            kwargs["description"] = description
        if payment_card_id is not NOT_GIVEN:
            kwargs["payment_card_id"] = payment_card_id
        if private is not NOT_GIVEN:
            kwargs["private"] = private
        return _get("venmo_create_transaction")(**kwargs)
    elif action == "show":
        return _get("venmo_show_transaction")(transaction_id=transaction_id)
    elif action == "update":
        kwargs = {"transaction_id": transaction_id}
        if description is not NOT_GIVEN:
            kwargs["description"] = description
        if private is not NOT_GIVEN:
            kwargs["private"] = private
        return _get("venmo_update_transaction")(**kwargs)
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# CRUD: Transaction comment management
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def venmo_manage_transaction_comment(
    action: Literal["create", "show", "update", "delete"],
    transaction_id: int | NotGiven = NOT_GIVEN,
    comment_id: int | NotGiven = NOT_GIVEN,
    comment: str | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Manage Venmo transaction comments: create, view, update, or delete.

    Actions:
        create: Add a comment to a transaction. Requires transaction_id
            and comment.
        show: View details of a comment. Requires comment_id.
        update: Update a comment's text. Requires comment_id and comment.
        delete: Delete a comment. Requires comment_id.

    Args:
        action: The operation to perform.
        transaction_id: The transaction ID (for create).
        comment_id: The comment ID (for show, update, delete).
        comment: The comment text (for create, update).

    Returns:
        Comment details or action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Venmo.
    """
    if action == "create":
        return _get("venmo_create_transaction_comment")(
            transaction_id=transaction_id, comment=comment
        )
    elif action == "show":
        return _get("venmo_show_transaction_comment")(comment_id=comment_id)
    elif action == "update":
        return _get("venmo_update_transaction_comment")(
            comment_id=comment_id, comment=comment
        )
    elif action == "delete":
        return _get("venmo_delete_transaction_comment")(comment_id=comment_id)
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# CRUD: Payment request management
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def venmo_manage_payment_request(
    action: Literal["create", "update", "delete"],
    payment_request_id: int | NotGiven = NOT_GIVEN,
    user_email: str | NotGiven = NOT_GIVEN,
    amount: float | None | NotGiven = NOT_GIVEN,
    description: str | None | NotGiven = NOT_GIVEN,
    private: bool | None | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Manage Venmo payment requests: create, update, or delete.

    Actions:
        create: Send a payment request. Requires user_email and amount.
            description defaults to empty string, private defaults to false.
        update: Update a payment request. Requires payment_request_id and
            at least one of amount, description, or private.
        delete: Delete a payment request. Requires payment_request_id.

    Args:
        action: The operation to perform.
        payment_request_id: The payment request ID (for update, delete).
        user_email: Email address of the user to request payment from
            (for create).
        amount: Amount of the payment request (for create, update).
        description: Description of or note about the payment request
            (for create, update).
        private: Privacy of the transaction on approval (for create, update).

    Returns:
        Payment request details or action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Venmo.
    """
    if action == "create":
        kwargs: dict[str, Any] = {
            "user_email": user_email,
            "amount": amount,
        }
        if description is not NOT_GIVEN:
            kwargs["description"] = description
        if private is not NOT_GIVEN:
            kwargs["private"] = private
        return _get("venmo_create_payment_request")(**kwargs)
    elif action == "update":
        kwargs = {"payment_request_id": payment_request_id}
        if amount is not NOT_GIVEN:
            kwargs["amount"] = amount
        if description is not NOT_GIVEN:
            kwargs["description"] = description
        if private is not NOT_GIVEN:
            kwargs["private"] = private
        return _get("venmo_update_payment_request")(**kwargs)
    elif action == "delete":
        return _get("venmo_delete_payment_request")(
            payment_request_id=payment_request_id
        )
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Symmetric pair: Friendship management
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def venmo_manage_friendship(
    user_email: str,
    action: Literal["add", "remove"],
) -> dict[str, Any]:
    """Add or remove a friend on Venmo.

    Args:
        user_email: Email address of the user.
        action: "add" to add as friend, "remove" to remove from friends.

    Returns:
        Action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Venmo.
    """
    if action == "add":
        return _get("venmo_add_friend")(user_email=user_email)
    return _get("venmo_remove_friend")(user_email=user_email)


# ---------------------------------------------------------------------------
# Symmetric pair: Transaction like/unlike
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def venmo_toggle_transaction_like(
    transaction_id: int,
    like: bool,
) -> dict[str, Any]:
    """Like or unlike a Venmo transaction.

    Args:
        transaction_id: The ID of the transaction.
        like: True to like, False to unlike.

    Returns:
        Action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Venmo.
    """
    if like:
        return _get("venmo_like_transaction")(transaction_id=transaction_id)
    return _get("venmo_unlike_transaction")(transaction_id=transaction_id)


# ---------------------------------------------------------------------------
# Symmetric pair: Comment like/unlike
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def venmo_toggle_comment_like(
    comment_id: int,
    like: bool,
) -> dict[str, Any]:
    """Like or unlike a Venmo transaction comment.

    Args:
        comment_id: The ID of the transaction comment.
        like: True to like, False to unlike.

    Returns:
        Action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Venmo.
    """
    if like:
        return _get("venmo_like_transaction_comment")(comment_id=comment_id)
    return _get("venmo_unlike_transaction_comment")(comment_id=comment_id)


# ---------------------------------------------------------------------------
# Symmetric pair: Balance add/withdraw
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def venmo_manage_balance(
    action: Literal["add", "withdraw"],
    amount: float,
    payment_card_id: int,
) -> dict[str, Any]:
    """Add money to or withdraw money from your Venmo balance.

    Actions:
        add: Add money to Venmo balance from a payment card.
        withdraw: Withdraw money from Venmo balance to a payment card.

    Args:
        action: "add" to add funds, "withdraw" to withdraw funds.
        amount: The amount to add or withdraw.
        payment_card_id: ID of the payment card to use.

    Returns:
        Action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Venmo.
    """
    if action == "add":
        return _get("venmo_add_to_venmo_balance")(
            amount=amount, payment_card_id=payment_card_id
        )
    return _get("venmo_withdraw_from_venmo_balance")(
        amount=amount, payment_card_id=payment_card_id
    )


# ---------------------------------------------------------------------------
# Symmetric pair: Payment request approve/deny
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def venmo_respond_to_payment_request(
    action: Literal["approve", "deny"],
    payment_request_id: int,
    payment_card_id: int | None | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Approve or deny a received payment request.

    Actions:
        approve: Approve the payment request. If payment_card_id is not
            provided, Venmo balance will be used.
        deny: Deny the payment request.

    Args:
        action: "approve" to approve, "deny" to deny.
        payment_request_id: ID of the payment request.
        payment_card_id: ID of the payment card to use for approval.
            If not provided, Venmo balance will be used (for approve only).

    Returns:
        Action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Venmo.
    """
    if action == "approve":
        kwargs: dict[str, Any] = {"payment_request_id": payment_request_id}
        if payment_card_id is not NOT_GIVEN:
            kwargs["payment_card_id"] = payment_card_id
        return _get("venmo_approve_payment_request")(**kwargs)
    return _get("venmo_deny_payment_request")(payment_request_id=payment_request_id)


# ---------------------------------------------------------------------------
# Mark absorbed tools
# ---------------------------------------------------------------------------

mark_tools_absorbed_by(
    "venmo_manage_transaction",
    "venmo_create_transaction",
    "venmo_show_transaction",
    "venmo_update_transaction",
)
mark_tools_absorbed_by(
    "venmo_manage_transaction_comment",
    "venmo_create_transaction_comment",
    "venmo_show_transaction_comment",
    "venmo_update_transaction_comment",
    "venmo_delete_transaction_comment",
)
mark_tools_absorbed_by(
    "venmo_manage_payment_request",
    "venmo_create_payment_request",
    "venmo_update_payment_request",
    "venmo_delete_payment_request",
)
mark_tools_absorbed_by(
    "venmo_manage_friendship",
    "venmo_add_friend",
    "venmo_remove_friend",
)
mark_tools_absorbed_by(
    "venmo_toggle_transaction_like",
    "venmo_like_transaction",
    "venmo_unlike_transaction",
)
mark_tools_absorbed_by(
    "venmo_toggle_comment_like",
    "venmo_like_transaction_comment",
    "venmo_unlike_transaction_comment",
)
mark_tools_absorbed_by(
    "venmo_manage_balance",
    "venmo_add_to_venmo_balance",
    "venmo_withdraw_from_venmo_balance",
)
mark_tools_absorbed_by(
    "venmo_respond_to_payment_request",
    "venmo_approve_payment_request",
    "venmo_deny_payment_request",
)
