# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""Consolidated Amazon tools for the MEDIUM toolbox.

CRUD consolidation for addresses, product reviews, product questions, and
question answers, plus symmetric pair merges for cart/wish-list products,
gift wrapping, browsing history, promo codes, and cart/wish-list moves.
"""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.toolbox.names import ToolboxName
from mmtoolsandbox.tools.consolidated import mark_tools_absorbed_by


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.appworld.amazon as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# CRUD: Address management
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def amazon_manage_address(
    action: Literal["add", "update", "delete"],
    address_id: int | NotGiven = NOT_GIVEN,
    name: str | NotGiven = NOT_GIVEN,
    street_address: str | NotGiven = NOT_GIVEN,
    city: str | NotGiven = NOT_GIVEN,
    state: str | NotGiven = NOT_GIVEN,
    country: str | NotGiven = NOT_GIVEN,
    zip_code: int | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Manage Amazon addresses: add, update, or delete.

    Actions:
        add: Add a new address. Requires name, street_address, city, state,
            country, and zip_code.
        update: Update an existing address. Requires address_id and at least
            one of name, street_address, city, state, country, or zip_code.
        delete: Delete an address. Requires address_id.

    Args:
        action: The operation to perform.
        address_id: The address ID (for update, delete).
        name: Name of the address, e.g. "Home" or "Work" (for add, update).
        street_address: Street address line (for add, update).
        city: City name (for add, update).
        state: State name (for add, update).
        country: Country name (for add, update).
        zip_code: 5-digit zip code (for add, update).

    Returns:
        Address details or action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Amazon.
    """
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
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# CRUD: Product review management
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def amazon_manage_product_review(
    action: Literal["write", "update", "delete"],
    product_id: int | NotGiven = NOT_GIVEN,
    review_id: int | NotGiven = NOT_GIVEN,
    rating: int | NotGiven = NOT_GIVEN,
    title: str | NotGiven = NOT_GIVEN,
    text: str | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Manage Amazon product reviews: write, update, or delete.

    Actions:
        write: Write a new product review. Requires product_id and rating.
            Optionally include title and text.
        update: Update an existing review. Requires review_id and at least
            one of rating, title, or text.
        delete: Delete a review. Requires review_id.

    Args:
        action: The operation to perform.
        product_id: The product ID (for write).
        review_id: The review ID (for update, delete).
        rating: Product rating value (for write, update).
        title: Review title (for write, update).
        text: Review body text (for write, update).

    Returns:
        Review details or action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Amazon.
    """
    if action == "write":
        kwargs: dict[str, Any] = {"product_id": product_id, "rating": rating}
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
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# CRUD: Product question management
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def amazon_manage_product_question(
    action: Literal["write", "update", "delete"],
    product_id: int | NotGiven = NOT_GIVEN,
    question_id: int | NotGiven = NOT_GIVEN,
    question: str | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Manage Amazon product questions: write, update, or delete.

    Actions:
        write: Post a new question about a product. Requires product_id
            and question.
        update: Update an existing question. Requires question_id and
            optionally question.
        delete: Delete a question. Requires question_id.

    Args:
        action: The operation to perform.
        product_id: The product ID (for write).
        question_id: The question ID (for update, delete).
        question: The question text (for write, update).

    Returns:
        Question details or action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Amazon.
    """
    if action == "write":
        return _get("amazon_write_product_question")(
            product_id=product_id, question=question
        )
    elif action == "update":
        kwargs: dict[str, Any] = {"question_id": question_id}
        if question is not NOT_GIVEN:
            kwargs["question"] = question
        return _get("amazon_update_product_question")(**kwargs)
    elif action == "delete":
        return _get("amazon_delete_product_question")(question_id=question_id)
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# CRUD: Product question answer management
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def amazon_manage_product_question_answer(
    action: Literal["write", "update", "delete"],
    question_id: int | NotGiven = NOT_GIVEN,
    question_answer_id: int | NotGiven = NOT_GIVEN,
    answer: str | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Manage Amazon product question answers: write, update, or delete.

    Actions:
        write: Write an answer to a product question. Requires question_id
            and answer.
        update: Update an existing answer. Requires question_answer_id and
            optionally answer.
        delete: Delete an answer. Requires question_answer_id.

    Args:
        action: The operation to perform.
        question_id: The question ID (for write).
        question_answer_id: The question answer ID (for update, delete).
        answer: The answer text (for write, update).

    Returns:
        Answer details or action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Amazon.
    """
    if action == "write":
        return _get("amazon_write_product_question_answer")(
            question_id=question_id, answer=answer
        )
    elif action == "update":
        kwargs: dict[str, Any] = {"question_answer_id": question_answer_id}
        if answer is not NOT_GIVEN:
            kwargs["answer"] = answer
        return _get("amazon_update_product_question_answer")(**kwargs)
    elif action == "delete":
        return _get("amazon_delete_product_question_answer")(
            question_answer_id=question_answer_id
        )
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Symmetric pair: Cart product add/remove
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def amazon_manage_cart_product(
    product_id: int,
    action: Literal["add", "remove"],
    quantity: int | NotGiven = NOT_GIVEN,
    clear_cart_first: bool | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Add or remove a product from your Amazon cart.

    Actions:
        add: Add a product to the cart. Optionally set quantity (defaults
            to 1) and clear_cart_first (defaults to false).
        remove: Remove a product from the cart.

    Args:
        product_id: The ID of the product.
        action: "add" to add to cart, "remove" to remove from cart.
        quantity: Quantity to add (for add). Defaults to 1.
        clear_cart_first: If true, clear the cart before adding (for add).

    Returns:
        Action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Amazon.
    """
    if action == "add":
        kwargs: dict[str, Any] = {"product_id": product_id}
        if quantity is not NOT_GIVEN:
            kwargs["quantity"] = quantity
        if clear_cart_first is not NOT_GIVEN:
            kwargs["clear_cart_first"] = clear_cart_first
        return _get("amazon_add_product_to_cart")(**kwargs)
    return _get("amazon_delete_product_from_cart")(product_id=product_id)


# ---------------------------------------------------------------------------
# Symmetric pair: Wish list product add/remove
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def amazon_manage_wish_list_product(
    product_id: int,
    action: Literal["add", "remove"],
    quantity: int | NotGiven = NOT_GIVEN,
    clear_wish_list_first: bool | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Add or remove a product from your Amazon wish list.

    Actions:
        add: Add a product to the wish list. Optionally set quantity
            (defaults to 1) and clear_wish_list_first (defaults to false).
        remove: Remove a product from the wish list.

    Args:
        product_id: The ID of the product.
        action: "add" to add to wish list, "remove" to remove from wish list.
        quantity: Quantity to add (for add). Defaults to 1.
        clear_wish_list_first: If true, clear the wish list before adding
            (for add).

    Returns:
        Action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Amazon.
    """
    if action == "add":
        kwargs: dict[str, Any] = {"product_id": product_id}
        if quantity is not NOT_GIVEN:
            kwargs["quantity"] = quantity
        if clear_wish_list_first is not NOT_GIVEN:
            kwargs["clear_wish_list_first"] = clear_wish_list_first
        return _get("amazon_add_product_to_wish_list")(**kwargs)
    return _get("amazon_delete_product_from_wish_list")(product_id=product_id)


# ---------------------------------------------------------------------------
# Symmetric pair: Gift wrapping add/remove
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def amazon_manage_gift_wrapping(
    product_id: int,
    action: Literal["add", "remove"],
    quantity: int | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Add or remove gift wrapping for a product in your Amazon cart.

    Actions:
        add: Add gift wrapping to a cart product. Optionally set quantity
            (defaults to 1). If already gift wrapped, quantity is updated.
        remove: Remove gift wrapping from a cart product.

    Args:
        product_id: The ID of the product in your cart.
        action: "add" to add gift wrapping, "remove" to remove it.
        quantity: Quantity to gift wrap (for add). Defaults to 1.

    Returns:
        Action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Amazon.
    """
    if action == "add":
        kwargs: dict[str, Any] = {"product_id": product_id}
        if quantity is not NOT_GIVEN:
            kwargs["quantity"] = quantity
        return _get("amazon_add_gift_wrapping_to_product")(**kwargs)
    return _get("amazon_remove_gift_wrapping_from_product")(product_id=product_id)


# ---------------------------------------------------------------------------
# Symmetric pair: Browsing history product add/remove
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def amazon_manage_browsing_history_product(
    product_id: int,
    action: Literal["add", "remove"],
) -> dict[str, Any]:
    """Add or remove a product from your Amazon browsing history.

    Args:
        product_id: The ID of the product.
        action: "add" to add to browsing history, "remove" to remove from it.

    Returns:
        Action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Amazon.
    """
    if action == "add":
        return _get("amazon_add_product_to_browsing_history")(product_id=product_id)
    return _get("amazon_remove_product_from_browsing_history")(product_id=product_id)


# ---------------------------------------------------------------------------
# Symmetric pair: Cart promo code apply/remove
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def amazon_manage_cart_promo_code(
    action: Literal["apply", "remove"],
    promo_code: str | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Apply or remove a promo code from your Amazon cart.

    Actions:
        apply: Apply a promo code. Requires promo_code.
        remove: Remove the current promo code from the cart.

    Args:
        action: "apply" to apply a promo code, "remove" to remove it.
        promo_code: The promo code string (for apply).

    Returns:
        Action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Amazon.
    """
    if action == "apply":
        return _get("amazon_apply_promo_code_to_cart")(promo_code=promo_code)
    return _get("amazon_remove_promo_code_from_cart")()


# ---------------------------------------------------------------------------
# Symmetric pair: Move product between cart and wish list
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def amazon_move_product(
    product_id: int,
    direction: Literal["cart_to_wish_list", "wish_list_to_cart"],
    quantity: int | None = 1,
) -> dict[str, Any]:
    """Move a product between your Amazon cart and wish list.

    Args:
        product_id: The ID of the product to move.
        direction: "cart_to_wish_list" or "wish_list_to_cart".
        quantity: Quantity to move. Defaults to 1.

    Returns:
        Action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Amazon.
    """
    kwargs: dict[str, Any] = {"product_id": product_id}
    if quantity is not None:
        kwargs["quantity"] = quantity
    if direction == "cart_to_wish_list":
        return _get("amazon_move_product_from_cart_to_wish_list")(**kwargs)
    return _get("amazon_move_product_from_wish_list_to_cart")(**kwargs)


# ---------------------------------------------------------------------------
# Mark absorbed tools
# ---------------------------------------------------------------------------

mark_tools_absorbed_by(
    "amazon_manage_address",
    "amazon_add_address",
    "amazon_update_address",
    "amazon_delete_address",
)
mark_tools_absorbed_by(
    "amazon_manage_product_review",
    "amazon_write_product_review",
    "amazon_update_product_review",
    "amazon_delete_product_review",
)
mark_tools_absorbed_by(
    "amazon_manage_product_question",
    "amazon_write_product_question",
    "amazon_update_product_question",
    "amazon_delete_product_question",
)
mark_tools_absorbed_by(
    "amazon_manage_product_question_answer",
    "amazon_write_product_question_answer",
    "amazon_update_product_question_answer",
    "amazon_delete_product_question_answer",
)
mark_tools_absorbed_by(
    "amazon_manage_cart_product",
    "amazon_add_product_to_cart",
    "amazon_delete_product_from_cart",
)
mark_tools_absorbed_by(
    "amazon_manage_wish_list_product",
    "amazon_add_product_to_wish_list",
    "amazon_delete_product_from_wish_list",
)
mark_tools_absorbed_by(
    "amazon_manage_gift_wrapping",
    "amazon_add_gift_wrapping_to_product",
    "amazon_remove_gift_wrapping_from_product",
)
mark_tools_absorbed_by(
    "amazon_manage_browsing_history_product",
    "amazon_add_product_to_browsing_history",
    "amazon_remove_product_from_browsing_history",
)
mark_tools_absorbed_by(
    "amazon_manage_cart_promo_code",
    "amazon_apply_promo_code_to_cart",
    "amazon_remove_promo_code_from_cart",
)
mark_tools_absorbed_by(
    "amazon_move_product",
    "amazon_move_product_from_cart_to_wish_list",
    "amazon_move_product_from_wish_list_to_cart",
)
