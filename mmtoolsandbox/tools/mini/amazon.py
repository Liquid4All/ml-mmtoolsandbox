# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"
"""MINI Amazon tools — 3 workflow-based tools covering all Amazon functionality."""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.datasets.names import DatasetName


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.appworld.amazon as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# Tool 1: amazon_browse — "I want to find or learn about products"
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.MINI},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def amazon_browse(
    domain: Literal[
        "search_products",
        "search_sellers",
        "search_types",
        "show_product",
        "show_seller",
        "show_reviews",
        "show_questions",
        "show_answers",
        "show_features",
        "show_ratings",
        "recommendations",
        "browsing_history",
        "clear_browsing_history",
        "update_browsing_tracking",
        "last_purchase",
    ],
    query: str | NotGiven = NOT_GIVEN,
    page_index: int | NotGiven = NOT_GIVEN,
    page_limit: int | NotGiven = NOT_GIVEN,
    sort_by: str | NotGiven = NOT_GIVEN,
    product_id: int | NotGiven = NOT_GIVEN,
    seller_id: int | NotGiven = NOT_GIVEN,
    product_type: str | NotGiven = NOT_GIVEN,
    color: str | NotGiven = NOT_GIVEN,
    relative_size: str | NotGiven = NOT_GIVEN,
    min_price: float | NotGiven = NOT_GIVEN,
    max_price: float | NotGiven = NOT_GIVEN,
    min_product_rating: float | NotGiven = NOT_GIVEN,
    max_product_rating: float | NotGiven = NOT_GIVEN,
    min_seller_rating: float | NotGiven = NOT_GIVEN,
    max_seller_rating: float | NotGiven = NOT_GIVEN,
    question_id: int | NotGiven = NOT_GIVEN,
    user_email: str | NotGiven = NOT_GIVEN,
    min_rating: int | NotGiven = NOT_GIVEN,
    max_rating: int | NotGiven = NOT_GIVEN,
    is_verified: bool | NotGiven = NOT_GIVEN,
    track_browsing_history: bool | NotGiven = NOT_GIVEN,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Browse, search, and discover products on Amazon.

    Domains:
        search_products: Search products with filters. Params: query,
            product_type, color, relative_size, min_price, max_price,
            min_product_rating, max_product_rating, min_seller_rating,
            max_seller_rating, seller_id, page_index, page_limit, sort_by.
        search_sellers: Search for sellers. Params: query, page_index,
            page_limit.
        search_types: Search product types. Params: query, page_index,
            page_limit.
        show_product: Show product details. Requires product_id.
        show_seller: Show seller details. Requires seller_id.
        show_reviews: Show product reviews. Requires product_id. Params:
            query, user_email, min_rating, max_rating, is_verified,
            page_index, page_limit, sort_by.
        show_questions: Show product questions. Requires product_id. Params:
            query, user_email, page_index, page_limit, sort_by.
        show_answers: Show answers to a product question. Requires
            question_id. Params: query, user_email, is_verified, page_index,
            page_limit, sort_by.
        show_features: Show product feature choices (colors, sizes, sellers).
            Params: product_type.
        show_ratings: Show rating distribution. Requires product_id.
        recommendations: Show recommended products. Params: page_index,
            page_limit.
        browsing_history: Show browsing history. Params: page_index,
            page_limit.
        clear_browsing_history: Clear your browsing history.
        update_browsing_tracking: Update tracking preference. Requires
            track_browsing_history.
        last_purchase: Show last purchase of a product. Requires product_id.

    Args:
        domain: The browsing action to perform.
        query: Search query string.
        page_index: Page index for paginated results.
        page_limit: Maximum results per page.
        sort_by: Sort attribute prefixed with +/- for ascending/descending.
        product_id: The product ID.
        seller_id: The seller ID.
        product_type: The product type to filter by.
        color: Product color filter.
        relative_size: Product relative size filter.
        min_price: Minimum price filter.
        max_price: Maximum price filter.
        min_product_rating: Minimum product rating filter.
        max_product_rating: Maximum product rating filter.
        min_seller_rating: Minimum seller rating filter.
        max_seller_rating: Maximum seller rating filter.
        question_id: The question ID (for show_answers).
        user_email: Filter by user email.
        min_rating: Minimum review rating filter.
        max_rating: Maximum review rating filter.
        is_verified: Filter by verified purchaser.
        track_browsing_history: Whether to track browsing history.

    Returns:
        For search_products: list of product dicts with product_id, title,
            price, rating. Pass product_id to show_product/show_reviews.
        For show_product/show_seller: detail dict.
        For show_reviews/show_questions/show_answers: list of dicts.
        For recommendations/browsing_history: list of product dicts.
    """
    if domain == "search_products":
        kwargs: dict[str, Any] = {}
        if query is not NOT_GIVEN:
            kwargs["query"] = query
        if product_type is not NOT_GIVEN:
            kwargs["product_type"] = product_type
        if color is not NOT_GIVEN:
            kwargs["color"] = color
        if relative_size is not NOT_GIVEN:
            kwargs["relative_size"] = relative_size
        if min_price is not NOT_GIVEN:
            kwargs["min_price"] = min_price
        if max_price is not NOT_GIVEN:
            kwargs["max_price"] = max_price
        if min_product_rating is not NOT_GIVEN:
            kwargs["min_product_rating"] = min_product_rating
        if max_product_rating is not NOT_GIVEN:
            kwargs["max_product_rating"] = max_product_rating
        if min_seller_rating is not NOT_GIVEN:
            kwargs["min_seller_rating"] = min_seller_rating
        if max_seller_rating is not NOT_GIVEN:
            kwargs["max_seller_rating"] = max_seller_rating
        if seller_id is not NOT_GIVEN:
            kwargs["seller_id"] = seller_id
        if page_index is not NOT_GIVEN:
            kwargs["page_index"] = page_index
        if page_limit is not NOT_GIVEN:
            kwargs["page_limit"] = page_limit
        if sort_by is not NOT_GIVEN:
            kwargs["sort_by"] = sort_by
        return _get("amazon_search_products")(**kwargs)

    elif domain == "search_sellers":
        kwargs = {}
        if query is not NOT_GIVEN:
            kwargs["query"] = query
        if page_index is not NOT_GIVEN:
            kwargs["page_index"] = page_index
        if page_limit is not NOT_GIVEN:
            kwargs["page_limit"] = page_limit
        return _get("amazon_search_sellers")(**kwargs)

    elif domain == "search_types":
        kwargs = {}
        if query is not NOT_GIVEN:
            kwargs["query"] = query
        if page_index is not NOT_GIVEN:
            kwargs["page_index"] = page_index
        if page_limit is not NOT_GIVEN:
            kwargs["page_limit"] = page_limit
        return _get("amazon_search_product_types")(**kwargs)

    elif domain == "show_product":
        return _get("amazon_show_product")(product_id=product_id)

    elif domain == "show_seller":
        return _get("amazon_show_seller")(seller_id=seller_id)

    elif domain == "show_reviews":
        kwargs = {"product_id": product_id}
        if query is not NOT_GIVEN:
            kwargs["query"] = query
        if user_email is not NOT_GIVEN:
            kwargs["user_email"] = user_email
        if min_rating is not NOT_GIVEN:
            kwargs["min_rating"] = min_rating
        if max_rating is not NOT_GIVEN:
            kwargs["max_rating"] = max_rating
        if is_verified is not NOT_GIVEN:
            kwargs["is_verified"] = is_verified
        if page_index is not NOT_GIVEN:
            kwargs["page_index"] = page_index
        if page_limit is not NOT_GIVEN:
            kwargs["page_limit"] = page_limit
        if sort_by is not NOT_GIVEN:
            kwargs["sort_by"] = sort_by
        return _get("amazon_show_product_reviews")(**kwargs)

    elif domain == "show_questions":
        kwargs = {"product_id": product_id}
        if query is not NOT_GIVEN:
            kwargs["query"] = query
        if user_email is not NOT_GIVEN:
            kwargs["user_email"] = user_email
        if page_index is not NOT_GIVEN:
            kwargs["page_index"] = page_index
        if page_limit is not NOT_GIVEN:
            kwargs["page_limit"] = page_limit
        if sort_by is not NOT_GIVEN:
            kwargs["sort_by"] = sort_by
        return _get("amazon_show_product_questions")(**kwargs)

    elif domain == "show_answers":
        kwargs = {"question_id": question_id}
        if query is not NOT_GIVEN:
            kwargs["query"] = query
        if user_email is not NOT_GIVEN:
            kwargs["user_email"] = user_email
        if is_verified is not NOT_GIVEN:
            kwargs["is_verified"] = is_verified
        if page_index is not NOT_GIVEN:
            kwargs["page_index"] = page_index
        if page_limit is not NOT_GIVEN:
            kwargs["page_limit"] = page_limit
        if sort_by is not NOT_GIVEN:
            kwargs["sort_by"] = sort_by
        return _get("amazon_show_product_question_answers")(**kwargs)

    elif domain == "show_features":
        kwargs = {}
        if product_type is not NOT_GIVEN:
            kwargs["product_type"] = product_type
        return _get("amazon_show_product_feature_choices")(**kwargs)

    elif domain == "show_ratings":
        return _get("amazon_show_product_rating_distribution")(product_id=product_id)

    elif domain == "recommendations":
        kwargs = {}
        if page_index is not NOT_GIVEN:
            kwargs["page_index"] = page_index
        if page_limit is not NOT_GIVEN:
            kwargs["page_limit"] = page_limit
        return _get("amazon_show_recommended_products")(**kwargs)

    elif domain == "browsing_history":
        kwargs = {}
        if page_index is not NOT_GIVEN:
            kwargs["page_index"] = page_index
        if page_limit is not NOT_GIVEN:
            kwargs["page_limit"] = page_limit
        return _get("amazon_show_browsing_history")(**kwargs)

    elif domain == "clear_browsing_history":
        return _get("amazon_clear_browsing_history")()

    elif domain == "update_browsing_tracking":
        return _get("amazon_update_browsing_history_tracking")(
            track_browsing_history=track_browsing_history
        )

    elif domain == "last_purchase":
        return _get("amazon_show_last_product_purchase")(product_id=product_id)

    else:
        raise ValueError(f"Unknown domain: {domain}")


# ---------------------------------------------------------------------------
# Tool 2: amazon_shop — "I want to buy things"
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.MINI},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def amazon_shop(
    domain: Literal["cart", "wish_list", "order", "promo", "gift_wrap", "move"],
    action: Literal[
        "show",
        "clear",
        "add",
        "remove",
        "update_qty",
        "list",
        "place",
        "download_receipt",
        "show_purchases",
        "apply",
        "cart_to_wish_list",
        "wish_list_to_cart",
    ],
    product_id: int | NotGiven = NOT_GIVEN,
    quantity: int | NotGiven = NOT_GIVEN,
    clear_cart_first: bool | NotGiven = NOT_GIVEN,
    clear_wish_list_first: bool | NotGiven = NOT_GIVEN,
    order_id: int | NotGiven = NOT_GIVEN,
    payment_card_id: int | NotGiven = NOT_GIVEN,
    address_id: int | NotGiven = NOT_GIVEN,
    file_system_access_token: str | NotGiven = NOT_GIVEN,
    download_to_file_path: str | NotGiven = NOT_GIVEN,
    overwrite: bool | NotGiven = NOT_GIVEN,
    promo_code: str | NotGiven = NOT_GIVEN,
    query: str | NotGiven = NOT_GIVEN,
    page_index: int | NotGiven = NOT_GIVEN,
    page_limit: int | NotGiven = NOT_GIVEN,
    sort_by: str | NotGiven = NOT_GIVEN,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Shop on Amazon: manage cart, wish list, orders, promos, and gift wrapping.

    Domains and their valid actions:
        cart:
            show: Show your cart contents.
            clear: Clear your entire cart.
            add: Add a product to cart. Requires product_id. Params: quantity,
                clear_cart_first.
            remove: Remove a product from cart. Requires product_id.
            update_qty: Update product quantity in cart. Requires product_id,
                quantity.
        wish_list:
            show: Show your wish list.
            clear: Clear your wish list.
            add: Add a product to wish list. Requires product_id. Params:
                quantity, clear_wish_list_first.
            remove: Remove a product from wish list. Requires product_id.
            update_qty: Update product quantity in wish list. Requires
                product_id, quantity.
        order:
            list: List/search past orders. Params: query, page_index,
                page_limit, sort_by.
            show: Show order details. Requires order_id.
            place: Place an order for cart items. Requires payment_card_id,
                address_id.
            download_receipt: Download order receipt. Requires order_id,
                file_system_access_token. Params: download_to_file_path,
                overwrite.
            show_purchases: Show purchased products. Params: page_index,
                page_limit.
        promo:
            apply: Apply a promo code to cart. Requires promo_code.
            remove: Remove promo code from cart.
        gift_wrap:
            add: Add gift wrapping to a cart product. Requires product_id.
                Params: quantity.
            remove: Remove gift wrapping from a cart product. Requires
                product_id.
        move:
            cart_to_wish_list: Move product from cart to wish list. Requires
                product_id. Params: quantity.
            wish_list_to_cart: Move product from wish list to cart. Requires
                product_id. Params: quantity.

    Args:
        domain: The shopping domain.
        action: The action to perform within the domain.
        product_id: The product ID.
        quantity: Product quantity.
        clear_cart_first: If true, clear cart before adding.
        clear_wish_list_first: If true, clear wish list before adding.
        order_id: The order ID.
        payment_card_id: Payment card ID for placing orders.
        address_id: Shipping address ID for placing orders.
        file_system_access_token: File system access token for downloads.
        download_to_file_path: File path to download receipt to.
        overwrite: Whether to overwrite existing file.
        promo_code: The promo code string.
        query: Search query for orders.
        page_index: Page index for paginated results.
        page_limit: Maximum results per page.
        sort_by: Sort attribute prefixed with +/- for ascending/descending.

    Returns:
        For cart/wish_list show: list of item dicts with product_id, quantity.
        For cart/wish_list add/remove/clear: confirmation dict.
        For order place: dict with order_id. Places order and charges
            payment card — externally visible, irreversible.
        For order show/list: order detail dict or list.
        For order cancel: cancels the order. May be irreversible.
        For promo apply/remove: confirmation dict.
        For gift_wrap: confirmation dict.
    """
    if domain == "cart":
        if action == "show":
            return _get("amazon_show_cart")()
        elif action == "clear":
            return _get("amazon_clear_cart")()
        elif action == "add":
            kwargs: dict[str, Any] = {"product_id": product_id}
            if quantity is not NOT_GIVEN:
                kwargs["quantity"] = quantity
            if clear_cart_first is not NOT_GIVEN:
                kwargs["clear_cart_first"] = clear_cart_first
            return _get("amazon_add_product_to_cart")(**kwargs)
        elif action == "remove":
            return _get("amazon_delete_product_from_cart")(product_id=product_id)
        elif action == "update_qty":
            return _get("amazon_update_product_quantity_in_cart")(
                product_id=product_id, quantity=quantity
            )
        else:
            raise ValueError(f"Unknown action for cart: {action}")

    elif domain == "wish_list":
        if action == "show":
            return _get("amazon_show_wish_list")()
        elif action == "clear":
            return _get("amazon_clear_wish_list")()
        elif action == "add":
            kwargs = {"product_id": product_id}
            if quantity is not NOT_GIVEN:
                kwargs["quantity"] = quantity
            if clear_wish_list_first is not NOT_GIVEN:
                kwargs["clear_wish_list_first"] = clear_wish_list_first
            return _get("amazon_add_product_to_wish_list")(**kwargs)
        elif action == "remove":
            return _get("amazon_delete_product_from_wish_list")(product_id=product_id)
        elif action == "update_qty":
            return _get("amazon_update_product_quantity_in_wish_list")(
                product_id=product_id, quantity=quantity
            )
        else:
            raise ValueError(f"Unknown action for wish_list: {action}")

    elif domain == "order":
        if action == "list":
            kwargs = {}
            if query is not NOT_GIVEN:
                kwargs["query"] = query
            if page_index is not NOT_GIVEN:
                kwargs["page_index"] = page_index
            if page_limit is not NOT_GIVEN:
                kwargs["page_limit"] = page_limit
            if sort_by is not NOT_GIVEN:
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
            if download_to_file_path is not NOT_GIVEN:
                kwargs["download_to_file_path"] = download_to_file_path
            if overwrite is not NOT_GIVEN:
                kwargs["overwrite"] = overwrite
            return _get("amazon_download_order_receipt")(**kwargs)
        elif action == "show_purchases":
            kwargs = {}
            if page_index is not NOT_GIVEN:
                kwargs["page_index"] = page_index
            if page_limit is not NOT_GIVEN:
                kwargs["page_limit"] = page_limit
            return _get("amazon_show_product_purchases")(**kwargs)
        else:
            raise ValueError(f"Unknown action for order: {action}")

    elif domain == "promo":
        if action == "apply":
            return _get("amazon_apply_promo_code_to_cart")(promo_code=promo_code)
        elif action == "remove":
            return _get("amazon_remove_promo_code_from_cart")()
        else:
            raise ValueError(f"Unknown action for promo: {action}")

    elif domain == "gift_wrap":
        if action == "add":
            kwargs = {"product_id": product_id}
            if quantity is not NOT_GIVEN:
                kwargs["quantity"] = quantity
            return _get("amazon_add_gift_wrapping_to_product")(**kwargs)
        elif action == "remove":
            return _get("amazon_remove_gift_wrapping_from_product")(
                product_id=product_id
            )
        else:
            raise ValueError(f"Unknown action for gift_wrap: {action}")

    elif domain == "move":
        if action == "cart_to_wish_list":
            kwargs = {"product_id": product_id}
            if quantity is not NOT_GIVEN:
                kwargs["quantity"] = quantity
            return _get("amazon_move_product_from_cart_to_wish_list")(**kwargs)
        elif action == "wish_list_to_cart":
            kwargs = {"product_id": product_id}
            if quantity is not NOT_GIVEN:
                kwargs["quantity"] = quantity
            return _get("amazon_move_product_from_wish_list_to_cart")(**kwargs)
        else:
            raise ValueError(f"Unknown action for move: {action}")

    else:
        raise ValueError(f"Unknown domain: {domain}")


# ---------------------------------------------------------------------------
# Tool 3: amazon_manage — "I want to manage addresses, reviews, or returns"
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.MINI},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def amazon_manage(
    domain: Literal["address", "review", "question", "answer", "return"],
    action: Literal[
        "add",
        "update",
        "delete",
        "list",
        "show",
        "write",
        "initiate",
        "show_deliverers",
    ],
    address_id: int | NotGiven = NOT_GIVEN,
    name: str | NotGiven = NOT_GIVEN,
    street_address: str | NotGiven = NOT_GIVEN,
    city: str | NotGiven = NOT_GIVEN,
    state: str | NotGiven = NOT_GIVEN,
    country: str | NotGiven = NOT_GIVEN,
    zip_code: int | NotGiven = NOT_GIVEN,
    product_id: int | NotGiven = NOT_GIVEN,
    review_id: int | NotGiven = NOT_GIVEN,
    rating: int | NotGiven = NOT_GIVEN,
    title: str | NotGiven = NOT_GIVEN,
    text: str | NotGiven = NOT_GIVEN,
    question_id: int | NotGiven = NOT_GIVEN,
    question: str | NotGiven = NOT_GIVEN,
    question_answer_id: int | NotGiven = NOT_GIVEN,
    answer: str | NotGiven = NOT_GIVEN,
    return_id: int | NotGiven = NOT_GIVEN,
    order_id: int | NotGiven = NOT_GIVEN,
    deliverer_id: int | NotGiven = NOT_GIVEN,
    quantity: int | NotGiven = NOT_GIVEN,
    page_index: int | NotGiven = NOT_GIVEN,
    page_limit: int | NotGiven = NOT_GIVEN,
    sort_by: str | NotGiven = NOT_GIVEN,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Manage Amazon addresses, reviews, questions, answers, and returns.

    Domains and their valid actions:
        address:
            add: Add a new address. Requires name, street_address, city,
                state, country, zip_code.
            update: Update an address. Requires address_id. Params: name,
                street_address, city, state, country, zip_code.
            delete: Delete an address. Requires address_id.
            list: List all your addresses.
        review:
            write: Write a product review. Requires product_id, rating.
                Params: title, text.
            update: Update a review. Requires review_id. Params: rating,
                title, text.
            delete: Delete a review. Requires review_id.
        question:
            write: Post a product question. Requires product_id, question.
            update: Update a question. Requires question_id. Params: question.
            delete: Delete a question. Requires question_id.
        answer:
            write: Write an answer to a question. Requires question_id, answer.
            update: Update an answer. Requires question_answer_id. Params:
                answer.
            delete: Delete an answer. Requires question_answer_id.
        return:
            list: List your returns. Params: order_id, page_index, page_limit,
                sort_by.
            show: Show return details. Requires return_id.
            initiate: Initiate a return. Requires order_id, product_id,
                deliverer_id, quantity.
            show_deliverers: List available return deliverers.

    Args:
        domain: The management domain.
        action: The action to perform within the domain.
        address_id: The address ID.
        name: Address name, e.g. "Home" or "Work".
        street_address: Street address line.
        city: City name.
        state: State name.
        country: Country name.
        zip_code: 5-digit zip code.
        product_id: The product ID.
        review_id: The review ID.
        rating: Product rating value.
        title: Review title.
        text: Review body text.
        question_id: The question ID.
        question: The question text.
        question_answer_id: The question answer ID.
        answer: The answer text.
        return_id: The return ID.
        order_id: The order ID.
        deliverer_id: The deliverer ID for returns.
        quantity: Quantity for returns.
        page_index: Page index for paginated results.
        page_limit: Maximum results per page.
        sort_by: Sort attribute prefixed with +/- for ascending/descending.

    Returns:
        For address add: dict with address_id.
        For address delete: confirmation. Irreversible.
        For review/question/answer write: dict with entity ID.
        For review/question/answer delete: confirmation. Irreversible.
        For return initiate: dict with return_id. Initiates a product return.
    """
    if domain == "address":
        if action == "add":
            return _get("amazon_add_address")(
                name=name,
                street_address=street_address,
                city=city,
                state=state,
                country=country,
                zip_code=zip_code,
            )
        elif action == "update":
            kwargs: dict[str, Any] = {"address_id": address_id}
            if name is not NOT_GIVEN:
                kwargs["name"] = name
            if street_address is not NOT_GIVEN:
                kwargs["street_address"] = street_address
            if city is not NOT_GIVEN:
                kwargs["city"] = city
            if state is not NOT_GIVEN:
                kwargs["state"] = state
            if country is not NOT_GIVEN:
                kwargs["country"] = country
            if zip_code is not NOT_GIVEN:
                kwargs["zip_code"] = zip_code
            return _get("amazon_update_address")(**kwargs)
        elif action == "delete":
            return _get("amazon_delete_address")(address_id=address_id)
        elif action == "list":
            return _get("amazon_show_addresses")()
        else:
            raise ValueError(f"Unknown action for address: {action}")

    elif domain == "review":
        if action == "write":
            kwargs = {"product_id": product_id, "rating": rating}
            if title is not NOT_GIVEN:
                kwargs["title"] = title
            if text is not NOT_GIVEN:
                kwargs["text"] = text
            return _get("amazon_write_product_review")(**kwargs)
        elif action == "update":
            kwargs = {"review_id": review_id}
            if rating is not NOT_GIVEN:
                kwargs["rating"] = rating
            if title is not NOT_GIVEN:
                kwargs["title"] = title
            if text is not NOT_GIVEN:
                kwargs["text"] = text
            return _get("amazon_update_product_review")(**kwargs)
        elif action == "delete":
            return _get("amazon_delete_product_review")(review_id=review_id)
        else:
            raise ValueError(f"Unknown action for review: {action}")

    elif domain == "question":
        if action == "write":
            return _get("amazon_write_product_question")(
                product_id=product_id, question=question
            )
        elif action == "update":
            kwargs = {"question_id": question_id}
            if question is not NOT_GIVEN:
                kwargs["question"] = question
            return _get("amazon_update_product_question")(**kwargs)
        elif action == "delete":
            return _get("amazon_delete_product_question")(question_id=question_id)
        else:
            raise ValueError(f"Unknown action for question: {action}")

    elif domain == "answer":
        if action == "write":
            return _get("amazon_write_product_question_answer")(
                question_id=question_id, answer=answer
            )
        elif action == "update":
            kwargs = {"question_answer_id": question_answer_id}
            if answer is not NOT_GIVEN:
                kwargs["answer"] = answer
            return _get("amazon_update_product_question_answer")(**kwargs)
        elif action == "delete":
            return _get("amazon_delete_product_question_answer")(
                question_answer_id=question_answer_id
            )
        else:
            raise ValueError(f"Unknown action for answer: {action}")

    elif domain == "return":
        if action == "list":
            kwargs = {}
            if order_id is not NOT_GIVEN:
                kwargs["order_id"] = order_id
            if page_index is not NOT_GIVEN:
                kwargs["page_index"] = page_index
            if page_limit is not NOT_GIVEN:
                kwargs["page_limit"] = page_limit
            if sort_by is not NOT_GIVEN:
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
            raise ValueError(f"Unknown action for return: {action}")

    else:
        raise ValueError(f"Unknown domain: {domain}")
