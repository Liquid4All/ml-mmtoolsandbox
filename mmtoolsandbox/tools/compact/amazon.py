# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""COMPACT Amazon tools — orders, returns, collection views, subscriptions."""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.datasets.names import DatasetName
from mmtoolsandbox.tools.compact import mark_compact_tools_absorbed_by


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.appworld.amazon as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# Strategy 5: Order management (5 → 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def amazon_manage_order(
    action: Literal["list", "show", "place", "download_receipt", "show_purchases"],
    order_id: int | NotGiven = NOT_GIVEN,
    query: str | None = "",
    page_index: int | None = 0,
    page_limit: int | None = 5,
    sort_by: str | None = None,
    payment_card_id: int | NotGiven = NOT_GIVEN,
    address_id: int | NotGiven = NOT_GIVEN,
    file_system_access_token: str | NotGiven = NOT_GIVEN,
    download_to_file_path: str | None = None,
    overwrite: bool | None = False,
) -> dict[str, Any] | list[dict[str, Any]]:
    """List, show, place, download receipt of, or show purchases for Amazon orders.

    Actions:
        list: Search or list past orders. Optionally filter with query,
            page_index, page_limit, and sort_by.
        show: Show details of a specific order. Requires order_id.
        place: Place an order for all items in your cart. Requires
            payment_card_id and address_id.
        download_receipt: Download the receipt of a past order. Requires
            order_id and file_system_access_token. Optionally set
            download_to_file_path and overwrite.
        show_purchases: Show products you have purchased in the past.
            Optionally set page_index and page_limit.

    Args:
        action: The order operation to perform.
        order_id: The order ID (for show, download_receipt).
        query: Search query string (for list).
        page_index: Zero-based page index for pagination (for list,
            show_purchases).
        page_limit: Maximum results per page (for list, show_purchases).
        sort_by: Sort field prefixed with +/- for ascending/descending.
            Valid attributes for list: created_at (for list).
        payment_card_id: Payment card ID (for place).
        address_id: Shipping address ID (for place).
        file_system_access_token: Access token from file_system login
            (for download_receipt).
        download_to_file_path: Destination file path in the file system.
            If not passed, saved in ~/downloads (for download_receipt).
        overwrite: Whether to overwrite if file already exists
            (for download_receipt).

    Returns:
        Order details, list of orders, or action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Amazon.
    """
    if action == "list":
        kwargs: dict[str, Any] = {}
        if query is not None:
            kwargs["query"] = query
        if page_index is not None:
            kwargs["page_index"] = page_index
        if page_limit is not None:
            kwargs["page_limit"] = page_limit
        if sort_by is not None:
            kwargs["sort_by"] = sort_by
        return _get("amazon_show_orders")(**kwargs)
    elif action == "show":
        return _get("amazon_show_order")(order_id=order_id)
    elif action == "place":
        return _get("amazon_place_order")(
            payment_card_id=payment_card_id, address_id=address_id
        )
    elif action == "download_receipt":
        kwargs = {
            "order_id": order_id,
            "file_system_access_token": file_system_access_token,
        }
        if download_to_file_path is not None:
            kwargs["download_to_file_path"] = download_to_file_path
        if overwrite is not None:
            kwargs["overwrite"] = overwrite
        return _get("amazon_download_order_receipt")(**kwargs)
    elif action == "show_purchases":
        kwargs = {}
        if page_index is not None:
            kwargs["page_index"] = page_index
        if page_limit is not None:
            kwargs["page_limit"] = page_limit
        return _get("amazon_show_product_purchases")(**kwargs)
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Strategy 5: Return management (4 → 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def amazon_manage_return(
    action: Literal["list", "show", "initiate", "show_deliverers"],
    return_id: int | NotGiven = NOT_GIVEN,
    order_id: int | NotGiven = NOT_GIVEN,
    product_id: int | NotGiven = NOT_GIVEN,
    deliverer_id: int | NotGiven = NOT_GIVEN,
    quantity: int | NotGiven = NOT_GIVEN,
    page_index: int | None = 0,
    page_limit: int | None = 5,
    sort_by: str | None = "-initiated_at",
) -> dict[str, Any] | list[dict[str, Any]]:
    """List, show, initiate product returns, or show return deliverers on Amazon.

    Actions:
        list: List your product returns. Optionally filter by order_id,
            page_index, page_limit, and sort_by.
        show: Show details of a specific return. Requires return_id.
        initiate: Initiate a product return. Requires order_id, product_id,
            deliverer_id, and quantity.
        show_deliverers: List available return deliverers. No additional
            arguments required.

    Args:
        action: The return operation to perform.
        return_id: The return ID (for show).
        order_id: The order ID (for list filter, initiate).
        product_id: The product ID to return (for initiate).
        deliverer_id: The deliverer ID assigned to the return (for initiate).
        quantity: Quantity of the product to return (for initiate).
        page_index: Zero-based page index for pagination (for list).
        page_limit: Maximum results per page (for list).
        sort_by: Sort field prefixed with +/- for ascending/descending.
            Valid attributes: quantity, initiated_at, returned_at (for list).

    Returns:
        Return details, list of returns/deliverers, or action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Amazon.
    """
    if action == "list":
        kwargs: dict[str, Any] = {}
        if order_id is not NOT_GIVEN:
            kwargs["order_id"] = order_id
        if page_index is not None:
            kwargs["page_index"] = page_index
        if page_limit is not None:
            kwargs["page_limit"] = page_limit
        if sort_by is not None:
            kwargs["sort_by"] = sort_by
        return _get("amazon_show_returns")(**kwargs)
    elif action == "show":
        return _get("amazon_show_return")(return_id=return_id)
    elif action == "initiate":
        return _get("amazon_initiate_return")(
            order_id=order_id,
            product_id=product_id,
            deliverer_id=deliverer_id,
            quantity=quantity,
        )
    elif action == "show_deliverers":
        return _get("amazon_show_return_deliverers")()
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Strategy 4: Show collection (2 → 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def amazon_show_collection(
    collection: Literal["cart", "wish_list"],
) -> dict[str, Any] | list[dict[str, Any]]:
    """Show the contents of your Amazon cart or wish list.

    Args:
        collection: Which collection to show — "cart" or "wish_list".

    Returns:
        Collection contents (cart details or wish list items).

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Amazon.
    """
    func_map = {
        "cart": "amazon_show_cart",
        "wish_list": "amazon_show_wish_list",
    }
    return _get(func_map[collection])()


# ---------------------------------------------------------------------------
# Strategy 4: Clear collection (2 → 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def amazon_clear_collection(
    collection: Literal["cart", "wish_list"],
) -> dict[str, Any]:
    """Clear all items from your Amazon cart or wish list.

    Args:
        collection: Which collection to clear — "cart" or "wish_list".

    Returns:
        Action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Amazon.
    """
    func_map = {
        "cart": "amazon_clear_cart",
        "wish_list": "amazon_clear_wish_list",
    }
    return _get(func_map[collection])()


# ---------------------------------------------------------------------------
# Strategy 4: Update collection quantity (2 → 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def amazon_update_collection_quantity(
    collection: Literal["cart", "wish_list"],
    product_id: int,
    quantity: int,
) -> dict[str, Any]:
    """Update the quantity of a product in your Amazon cart or wish list.

    Args:
        collection: Which collection to update — "cart" or "wish_list".
        product_id: The ID of the product to update.
        quantity: The new quantity for the product.

    Returns:
        Updated product details.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Amazon.
    """
    func_map = {
        "cart": "amazon_update_product_quantity_in_cart",
        "wish_list": "amazon_update_product_quantity_in_wish_list",
    }
    return _get(func_map[collection])(product_id=product_id, quantity=quantity)


# ---------------------------------------------------------------------------
# Mark absorbed tools
# ---------------------------------------------------------------------------

mark_compact_tools_absorbed_by(
    "amazon_manage_order",
    "amazon_show_orders",
    "amazon_show_order",
    "amazon_place_order",
    "amazon_download_order_receipt",
    "amazon_show_product_purchases",
)
mark_compact_tools_absorbed_by(
    "amazon_manage_return",
    "amazon_show_returns",
    "amazon_show_return",
    "amazon_initiate_return",
    "amazon_show_return_deliverers",
)
mark_compact_tools_absorbed_by(
    "amazon_show_collection",
    "amazon_show_cart",
    "amazon_show_wish_list",
)
mark_compact_tools_absorbed_by(
    "amazon_clear_collection",
    "amazon_clear_cart",
    "amazon_clear_wish_list",
)
mark_compact_tools_absorbed_by(
    "amazon_update_collection_quantity",
    "amazon_update_product_quantity_in_cart",
    "amazon_update_product_quantity_in_wish_list",
)


# ---------------------------------------------------------------------------
# Lazy import helper — dispatch to MEDIUM consolidated amazon tools
# ---------------------------------------------------------------------------


def _get_consolidated(name: str) -> Any:
    import mmtoolsandbox.tools.consolidated.amazon as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# Strategy 7: Amazon collection product — collapse MEDIUM by entity subtype
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def amazon_manage_collection_product(
    collection: Literal["cart", "wish_list"],
    product_id: int,
    action: Literal["add", "remove"],
    quantity: int | NotGiven = NOT_GIVEN,
    clear_cart_first: bool | NotGiven = NOT_GIVEN,
    clear_wish_list_first: bool | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Add or remove a product from your Amazon cart or wish list.

    Actions:
        add: Add a product. Optionally set quantity (defaults to 1).
            For cart, optionally set clear_cart_first.
            For wish_list, optionally set clear_wish_list_first.
        remove: Remove a product from the collection.

    Args:
        collection: Which collection to manage — "cart" or "wish_list".
        product_id: The ID of the product.
        action: "add" to add the product, "remove" to remove it.
        quantity: Quantity to add (for add). Defaults to 1.
        clear_cart_first: If true, clear the cart before adding
            (for add, cart only).
        clear_wish_list_first: If true, clear the wish list before adding
            (for add, wish_list only).

    Returns:
        Action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Amazon.
    """
    if collection == "cart":
        kwargs: dict[str, Any] = {"product_id": product_id, "action": action}
        if quantity is not NOT_GIVEN:
            kwargs["quantity"] = quantity
        if clear_cart_first is not NOT_GIVEN:
            kwargs["clear_cart_first"] = clear_cart_first
        return _get_consolidated("amazon_manage_cart_product")(**kwargs)
    elif collection == "wish_list":
        kwargs = {"product_id": product_id, "action": action}
        if quantity is not NOT_GIVEN:
            kwargs["quantity"] = quantity
        if clear_wish_list_first is not NOT_GIVEN:
            kwargs["clear_wish_list_first"] = clear_wish_list_first
        return _get_consolidated("amazon_manage_wish_list_product")(**kwargs)
    else:
        raise ValueError(f"Unknown collection: {collection}")


mark_compact_tools_absorbed_by(
    "amazon_manage_collection_product",
    "amazon_manage_cart_product",
    "amazon_manage_wish_list_product",
)
