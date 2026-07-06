# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""COMPACT Venmo tools — merges payment request listing (2->1)."""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.datasets.names import DatasetName
from mmtoolsandbox.tools.compact import mark_compact_tools_absorbed_by


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.appworld.venmo as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# Strategy 4: Payment request listing (2 -> 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def venmo_show_payment_requests(
    direction: Literal["received", "sent"],
    query: str | None = "",
    status: Literal["pending", "approved", "denied", "cancelled"] | None = None,
    page_index: int | None = 0,
    page_limit: int | None = 5,
) -> list[dict[str, Any]]:
    """Show or search Venmo payment requests, filtered by direction (received or sent).

    Args:
        direction: Whether to show received or sent payment requests.
        query: Search query string.
        status: Filter payment requests by status.
        page_index: Zero-based page index for pagination.
        page_limit: Maximum results per page.

    Returns:
        List of matching payment requests.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Venmo.
    """
    kwargs: dict[str, Any] = {}
    if query is not None:
        kwargs["query"] = query
    if status is not None:
        kwargs["status"] = status
    if page_index is not None:
        kwargs["page_index"] = page_index
    if page_limit is not None:
        kwargs["page_limit"] = page_limit

    func_map = {
        "received": "venmo_show_received_payment_requests",
        "sent": "venmo_show_sent_payment_requests",
    }
    return _get(func_map[direction])(**kwargs)


mark_compact_tools_absorbed_by(
    "venmo_show_payment_requests",
    "venmo_show_received_payment_requests",
    "venmo_show_sent_payment_requests",
)


# ---------------------------------------------------------------------------
# Lazy import helper — dispatch to MEDIUM consolidated venmo tools
# ---------------------------------------------------------------------------


def _get_consolidated(name: str) -> Any:
    import mmtoolsandbox.tools.consolidated.venmo as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# Strategy 5: Venmo finances (3 -> 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def venmo_manage_finances(
    action: Literal[
        "show_balance", "show_transfer_history", "download_transfer_receipt"
    ],
    # -- show_transfer_history params --
    transfer_type: Literal["standard", "instant"] | None | NotGiven = NOT_GIVEN,
    page_index: int | None | NotGiven = NOT_GIVEN,
    page_limit: int | None | NotGiven = NOT_GIVEN,
    # -- download_transfer_receipt params --
    bank_transfer_id: int | NotGiven = NOT_GIVEN,
    file_system_access_token: str | NotGiven = NOT_GIVEN,
    download_to_file_path: str | None = None,
    overwrite: bool | None = False,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Show Venmo balance, bank transfer history, or download a bank transfer receipt.

    Actions:
        show_balance: Show your Venmo balance. No extra parameters needed.
        show_transfer_history: Show history of money transfers between Venmo
            and payment cards. Optional filters: transfer_type, page_index,
            page_limit.
        download_transfer_receipt: Download receipt of a bank transfer.
            Requires bank_transfer_id and file_system_access_token.

    Args:
        action: The finance operation to perform.
        transfer_type: Filter bank transfers by type (for
            show_transfer_history).
        page_index: Zero-based page index for pagination (for
            show_transfer_history).
        page_limit: Maximum results per page (for show_transfer_history).
        bank_transfer_id: ID of the bank transfer (for
            download_transfer_receipt).
        file_system_access_token: Access token from file_system login (for
            download_transfer_receipt).
        download_to_file_path: Destination path for download. Defaults to
            ~/downloads directory (for download_transfer_receipt).
        overwrite: Whether to overwrite existing files (for
            download_transfer_receipt).

    Returns:
        Balance info (dict), transfer list, or download result.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Venmo.
    """
    if action == "show_balance":
        return _get("venmo_show_venmo_balance")()
    elif action == "show_transfer_history":
        kwargs: dict[str, Any] = {}
        if transfer_type is not NOT_GIVEN and transfer_type is not None:
            kwargs["transfer_type"] = transfer_type
        if page_index is not NOT_GIVEN and page_index is not None:
            kwargs["page_index"] = page_index
        if page_limit is not NOT_GIVEN and page_limit is not None:
            kwargs["page_limit"] = page_limit
        return _get("venmo_show_bank_transfer_history")(**kwargs)
    elif action == "download_transfer_receipt":
        kwargs = {
            "bank_transfer_id": bank_transfer_id,
            "file_system_access_token": file_system_access_token,
        }
        if download_to_file_path is not None:
            kwargs["download_to_file_path"] = download_to_file_path
        if overwrite is not None:
            kwargs["overwrite"] = overwrite
        return _get("venmo_download_bank_transfer_receipt")(**kwargs)
    else:
        raise ValueError(f"Unknown action: {action}")


mark_compact_tools_absorbed_by(
    "venmo_manage_finances",
    "venmo_show_venmo_balance",
    "venmo_show_bank_transfer_history",
    "venmo_download_bank_transfer_receipt",
)


# ---------------------------------------------------------------------------
# Strategy 7: Venmo toggle like — collapse MEDIUM by entity subtype
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def venmo_toggle_like(
    entity_type: Literal["transaction", "comment"],
    entity_id: int,
    like: bool,
) -> dict[str, Any]:
    """Like or unlike a Venmo transaction or transaction comment.

    Args:
        entity_type: The type of entity to like or unlike.
        entity_id: The ID of the entity.
        like: True to like, False to unlike.

    Returns:
        Action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Venmo.
    """
    func_map: dict[str, tuple[str, str]] = {
        "transaction": ("venmo_toggle_transaction_like", "transaction_id"),
        "comment": ("venmo_toggle_comment_like", "comment_id"),
    }
    func_name, param_name = func_map[entity_type]
    return _get_consolidated(func_name)(**{param_name: entity_id, "like": like})


mark_compact_tools_absorbed_by(
    "venmo_toggle_like",
    "venmo_toggle_transaction_like",
    "venmo_toggle_comment_like",
)
