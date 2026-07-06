# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""Consolidated Venmo tools for the MINI toolbox.

Two workflow-based tools covering all Venmo functionality:
- venmo_transact: financial operations (transactions, payment requests, balance)
- venmo_social: social interactions (friends, likes, comments, feed)
"""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.datasets.names import DatasetName


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.appworld.venmo as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# venmo_transact
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.MINI},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def venmo_transact(
    domain: Literal["transaction", "payment_request", "balance", "finances"],
    action: Literal[
        "create",
        "show",
        "update",
        "list",
        "download_receipt",
        "delete",
        "approve",
        "deny",
        "show_received",
        "show_sent",
        "remind",
        "add",
        "withdraw",
        "show_balance",
        "show_transfers",
        "download_transfer_receipt",
    ],
    transaction_id: int | NotGiven = NOT_GIVEN,
    receiver_email: str | NotGiven = NOT_GIVEN,
    amount: float | NotGiven = NOT_GIVEN,
    note: str | NotGiven = NOT_GIVEN,
    audience: bool | NotGiven = NOT_GIVEN,
    payment_card_id: int | None | NotGiven = NOT_GIVEN,
    # payment_request params
    payment_request_id: int | NotGiven = NOT_GIVEN,
    # list/search params
    query: str | None | NotGiven = NOT_GIVEN,
    user_email: str | None | NotGiven = NOT_GIVEN,
    min_created_at: str | None | NotGiven = NOT_GIVEN,
    max_created_at: str | None | NotGiven = NOT_GIVEN,
    min_like_count: int | None | NotGiven = NOT_GIVEN,
    max_like_count: int | None | NotGiven = NOT_GIVEN,
    min_amount: float | None | NotGiven = NOT_GIVEN,
    max_amount: float | None | NotGiven = NOT_GIVEN,
    private: bool | None | NotGiven = NOT_GIVEN,
    direction: str | None | NotGiven = NOT_GIVEN,
    page_index: int | None | NotGiven = NOT_GIVEN,
    page_limit: int | None | NotGiven = NOT_GIVEN,
    sort_by: str | None | NotGiven = NOT_GIVEN,
    status: Literal["pending", "approved", "denied", "cancelled"]
    | None
    | NotGiven = NOT_GIVEN,
    # receipt params
    file_system_access_token: str | NotGiven = NOT_GIVEN,
    download_to_file_path: str | None | NotGiven = NOT_GIVEN,
    overwrite: bool | None | NotGiven = NOT_GIVEN,
    # balance params
    bank_transfer_id: int | NotGiven = NOT_GIVEN,
    transfer_type: Literal["standard", "instant"] | None | NotGiven = NOT_GIVEN,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Send, receive, and manage money on Venmo.

    Domains and actions:
        transaction:
            create - Send money. Requires receiver_email, amount.
                Optional: note (description), audience (private), payment_card_id.
            show - View transaction details. Requires transaction_id.
            update - Update note/audience. Requires transaction_id.
            list - Search/list transactions. Optional: query, user_email,
                min/max_created_at, min/max_like_count, min/max_amount,
                private, direction, page_index, page_limit, sort_by.
            download_receipt - Download transaction receipt. Requires
                transaction_id, file_system_access_token.
        payment_request:
            create - Request money. Requires receiver_email (user_email param),
                amount. Optional: note, audience (private).
            update - Update request. Requires payment_request_id.
            delete - Delete request. Requires payment_request_id.
            approve - Approve request. Requires payment_request_id.
                Optional: payment_card_id.
            deny - Deny request. Requires payment_request_id.
            show_received - List received requests. Optional: query, status,
                page_index, page_limit.
            show_sent - List sent requests. Optional: query, status,
                page_index, page_limit.
            remind - Send reminder. Requires payment_request_id.
        balance:
            add - Add to balance. Requires amount, payment_card_id.
            withdraw - Withdraw from balance. Requires amount, payment_card_id.
        finances:
            show_balance - Show Venmo balance.
            show_transfers - Show bank transfer history.
            download_transfer_receipt - Download transfer receipt.
                Requires bank_transfer_id, file_system_access_token.

    Args:
        domain: The financial domain.
        action: The specific action within the domain.
        transaction_id: Transaction ID (for show/update/download_receipt).
        receiver_email: Receiver's email (for transaction create).
        amount: Amount (for create, add, withdraw).
        note: Description/note (for create, update).
        audience: Privacy flag - True means private (for create, update).
        payment_card_id: Payment card ID (optional for create, approve, add, withdraw).
        payment_request_id: Payment request ID (for update/delete/approve/deny/remind).
        query: Search query string (for list actions).
        user_email: Filter by user email (for transaction list, payment_request create).
        min_created_at: Min date filter YYYY-MM-DD (for transaction list).
        max_created_at: Max date filter YYYY-MM-DD (for transaction list).
        min_like_count: Min like count filter (for transaction list).
        max_like_count: Max like count filter (for transaction list).
        min_amount: Min amount filter (for transaction list).
        max_amount: Max amount filter (for transaction list).
        private: Privacy filter (for transaction list).
        direction: Direction filter - sent/received (for transaction list).
        page_index: Page index for pagination.
        page_limit: Results per page.
        sort_by: Sort attribute with +/- prefix (for transaction list).
        status: Status filter (for payment request list).
        file_system_access_token: File system access token (for receipts).
        download_to_file_path: Download path (for receipts).
        overwrite: Overwrite existing file (for receipts).
        bank_transfer_id: Bank transfer ID (for download_transfer_receipt).
        transfer_type: Transfer type filter (for show_transfers).

    Returns:
        For transaction create: dict with transaction_id. Sends money —
            externally visible and irreversible.
        For transaction show/update: transaction detail dict.
        For transaction list: list of transaction dicts.
        For payment_request create: sends money request. Externally visible.
        For payment_request approve: transfers money. Irreversible.
        For balance add/withdraw: confirmation dict. Moves real money.
        For finances: balance dict or transfer list.

    Raises:
        PermissionError: If not logged into Venmo.
    """
    if domain == "transaction":
        if action == "create":
            kwargs: dict[str, Any] = {
                "receiver_email": receiver_email,
                "amount": amount,
            }
            if note is not NOT_GIVEN:
                kwargs["description"] = note
            if payment_card_id is not NOT_GIVEN:
                kwargs["payment_card_id"] = payment_card_id
            if audience is not NOT_GIVEN:
                kwargs["private"] = audience
            return _get("venmo_create_transaction")(**kwargs)
        elif action == "show":
            return _get("venmo_show_transaction")(transaction_id=transaction_id)
        elif action == "update":
            kwargs = {"transaction_id": transaction_id}
            if note is not NOT_GIVEN:
                kwargs["description"] = note
            if audience is not NOT_GIVEN:
                kwargs["private"] = audience
            return _get("venmo_update_transaction")(**kwargs)
        elif action == "list":
            kwargs = {}
            if query is not NOT_GIVEN:
                kwargs["query"] = query
            if user_email is not NOT_GIVEN:
                kwargs["user_email"] = user_email
            if min_created_at is not NOT_GIVEN:
                kwargs["min_created_at"] = min_created_at
            if max_created_at is not NOT_GIVEN:
                kwargs["max_created_at"] = max_created_at
            if min_like_count is not NOT_GIVEN:
                kwargs["min_like_count"] = min_like_count
            if max_like_count is not NOT_GIVEN:
                kwargs["max_like_count"] = max_like_count
            if min_amount is not NOT_GIVEN:
                kwargs["min_amount"] = min_amount
            if max_amount is not NOT_GIVEN:
                kwargs["max_amount"] = max_amount
            if private is not NOT_GIVEN:
                kwargs["private"] = private
            if direction is not NOT_GIVEN:
                kwargs["direction"] = direction
            if page_index is not NOT_GIVEN:
                kwargs["page_index"] = page_index
            if page_limit is not NOT_GIVEN:
                kwargs["page_limit"] = page_limit
            if sort_by is not NOT_GIVEN:
                kwargs["sort_by"] = sort_by
            return _get("venmo_show_transactions")(**kwargs)
        elif action == "download_receipt":
            kwargs = {
                "transaction_id": transaction_id,
                "file_system_access_token": file_system_access_token,
            }
            if download_to_file_path is not NOT_GIVEN:
                kwargs["download_to_file_path"] = download_to_file_path
            if overwrite is not NOT_GIVEN:
                kwargs["overwrite"] = overwrite
            return _get("venmo_download_transaction_receipt")(**kwargs)
        else:
            raise ValueError(f"Unknown transaction action: {action}")

    elif domain == "payment_request":
        if action == "create":
            kwargs = {
                "user_email": user_email,
                "amount": amount,
            }
            if note is not NOT_GIVEN:
                kwargs["description"] = note
            if audience is not NOT_GIVEN:
                kwargs["private"] = audience
            return _get("venmo_create_payment_request")(**kwargs)
        elif action == "update":
            kwargs = {"payment_request_id": payment_request_id}
            if amount is not NOT_GIVEN:
                kwargs["amount"] = amount
            if note is not NOT_GIVEN:
                kwargs["description"] = note
            if audience is not NOT_GIVEN:
                kwargs["private"] = audience
            return _get("venmo_update_payment_request")(**kwargs)
        elif action == "delete":
            return _get("venmo_delete_payment_request")(
                payment_request_id=payment_request_id
            )
        elif action == "approve":
            kwargs = {"payment_request_id": payment_request_id}
            if payment_card_id is not NOT_GIVEN:
                kwargs["payment_card_id"] = payment_card_id
            return _get("venmo_approve_payment_request")(**kwargs)
        elif action == "deny":
            return _get("venmo_deny_payment_request")(
                payment_request_id=payment_request_id
            )
        elif action == "show_received":
            kwargs = {}
            if query is not NOT_GIVEN:
                kwargs["query"] = query
            if status is not NOT_GIVEN:
                kwargs["status"] = status
            if page_index is not NOT_GIVEN:
                kwargs["page_index"] = page_index
            if page_limit is not NOT_GIVEN:
                kwargs["page_limit"] = page_limit
            return _get("venmo_show_received_payment_requests")(**kwargs)
        elif action == "show_sent":
            kwargs = {}
            if query is not NOT_GIVEN:
                kwargs["query"] = query
            if status is not NOT_GIVEN:
                kwargs["status"] = status
            if page_index is not NOT_GIVEN:
                kwargs["page_index"] = page_index
            if page_limit is not NOT_GIVEN:
                kwargs["page_limit"] = page_limit
            return _get("venmo_show_sent_payment_requests")(**kwargs)
        elif action == "remind":
            return _get("venmo_remind_payment_request")(
                payment_request_id=payment_request_id
            )
        else:
            raise ValueError(f"Unknown payment_request action: {action}")

    elif domain == "balance":
        if action == "add":
            return _get("venmo_add_to_venmo_balance")(
                amount=amount, payment_card_id=payment_card_id
            )
        elif action == "withdraw":
            return _get("venmo_withdraw_from_venmo_balance")(
                amount=amount, payment_card_id=payment_card_id
            )
        else:
            raise ValueError(f"Unknown balance action: {action}")

    elif domain == "finances":
        if action == "show_balance":
            return _get("venmo_show_venmo_balance")()
        elif action == "show_transfers":
            kwargs = {}
            if transfer_type is not NOT_GIVEN:
                kwargs["transfer_type"] = transfer_type
            if page_index is not NOT_GIVEN:
                kwargs["page_index"] = page_index
            if page_limit is not NOT_GIVEN:
                kwargs["page_limit"] = page_limit
            return _get("venmo_show_bank_transfer_history")(**kwargs)
        elif action == "download_transfer_receipt":
            kwargs = {
                "bank_transfer_id": bank_transfer_id,
                "file_system_access_token": file_system_access_token,
            }
            if download_to_file_path is not NOT_GIVEN:
                kwargs["download_to_file_path"] = download_to_file_path
            if overwrite is not NOT_GIVEN:
                kwargs["overwrite"] = overwrite
            return _get("venmo_download_bank_transfer_receipt")(**kwargs)
        else:
            raise ValueError(f"Unknown finances action: {action}")

    else:
        raise ValueError(f"Unknown domain: {domain}")


# ---------------------------------------------------------------------------
# venmo_social
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.MINI},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def venmo_social(
    domain: Literal["friend", "like", "comment", "feed"],
    action: Literal[
        "add",
        "remove",
        "search",
        "toggle",
        "create",
        "show",
        "update",
        "delete",
        "list",
    ],
    user_email: str | NotGiven = NOT_GIVEN,
    query: str | None | NotGiven = NOT_GIVEN,
    page_index: int | None | NotGiven = NOT_GIVEN,
    page_limit: int | None | NotGiven = NOT_GIVEN,
    # like params
    entity_type: Literal["transaction", "comment"] | NotGiven = NOT_GIVEN,
    entity_id: int | NotGiven = NOT_GIVEN,
    like: bool | NotGiven = NOT_GIVEN,
    # comment params
    transaction_id: int | NotGiven = NOT_GIVEN,
    comment_id: int | NotGiven = NOT_GIVEN,
    comment: str | NotGiven = NOT_GIVEN,
    sort_by: str | None | NotGiven = NOT_GIVEN,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Browse and interact with the Venmo social feed.

    Domains and actions:
        friend:
            add - Add a friend. Requires user_email.
            remove - Remove a friend. Requires user_email.
            search - Search friends. Optional: query, page_index, page_limit.
        like:
            toggle - Like or unlike. Requires entity_type (transaction/comment),
                entity_id, and like (true=like, false=unlike).
        comment:
            create - Post a comment. Requires transaction_id, comment.
            show - View comment details. Requires comment_id.
            update - Update a comment. Requires comment_id, comment.
            delete - Delete a comment. Requires comment_id.
            list - List comments on a transaction. Requires transaction_id.
                Optional: page_index, page_limit, sort_by.
        feed:
            show - Show social feed. Optional: page_index, page_limit.

    Args:
        domain: The social domain.
        action: The specific action within the domain.
        user_email: Email address (for friend add/remove).
        query: Search query (for friend search).
        page_index: Page index for pagination.
        page_limit: Results per page.
        entity_type: Entity type for like - "transaction" or "comment".
        entity_id: Entity ID for like (transaction_id or comment_id).
        like: True to like, False to unlike.
        transaction_id: Transaction ID (for comment create/list).
        comment_id: Comment ID (for comment show/update/delete).
        comment: Comment text (for comment create/update).
        sort_by: Sort attribute with +/- prefix (for comment list).

    Returns:
        For friend add/remove: confirmation dict.
        For friend search: list of user dicts.
        For like toggle: confirmation dict.
        For comment create: dict with comment_id.
        For comment delete: confirmation. Irreversible.
        For feed show: list of feed item dicts.
    """
    if domain == "friend":
        if action == "add":
            return _get("venmo_add_friend")(user_email=user_email)
        elif action == "remove":
            return _get("venmo_remove_friend")(user_email=user_email)
        elif action == "search":
            kwargs: dict[str, Any] = {}
            if query is not NOT_GIVEN:
                kwargs["query"] = query
            if page_index is not NOT_GIVEN:
                kwargs["page_index"] = page_index
            if page_limit is not NOT_GIVEN:
                kwargs["page_limit"] = page_limit
            return _get("venmo_search_friends")(**kwargs)
        else:
            raise ValueError(f"Unknown friend action: {action}")

    elif domain == "like":
        if action == "toggle":
            if entity_type == "transaction":
                if like:
                    return _get("venmo_like_transaction")(transaction_id=entity_id)
                else:
                    return _get("venmo_unlike_transaction")(transaction_id=entity_id)
            elif entity_type == "comment":
                if like:
                    return _get("venmo_like_transaction_comment")(comment_id=entity_id)
                else:
                    return _get("venmo_unlike_transaction_comment")(
                        comment_id=entity_id
                    )
            else:
                raise ValueError(f"Unknown entity_type: {entity_type}")
        else:
            raise ValueError(f"Unknown like action: {action}")

    elif domain == "comment":
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
        elif action == "list":
            kwargs = {"transaction_id": transaction_id}
            if page_index is not NOT_GIVEN:
                kwargs["page_index"] = page_index
            if page_limit is not NOT_GIVEN:
                kwargs["page_limit"] = page_limit
            return _get("venmo_show_transaction_comments")(**kwargs)
        else:
            raise ValueError(f"Unknown comment action: {action}")

    elif domain == "feed":
        if action == "show":
            kwargs = {}
            if page_index is not NOT_GIVEN:
                kwargs["page_index"] = page_index
            if page_limit is not NOT_GIVEN:
                kwargs["page_limit"] = page_limit
            return _get("venmo_show_social_feed")(**kwargs)
        else:
            raise ValueError(f"Unknown feed action: {action}")

    else:
        raise ValueError(f"Unknown domain: {domain}")
