# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""Cross-app payment card tools for the MEDIUM toolbox.

Consolidates identical payment card CRUD tools across Spotify, Amazon,
and Venmo into a single parameterized tool.
"""

from __future__ import annotations

import importlib
from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.toolbox.names import ToolboxName
from mmtoolsandbox.tools.consolidated import mark_tools_absorbed_by

PaymentCardApp = Literal["spotify", "amazon", "venmo"]

_PAYMENT_CARD_APPS = ("spotify", "amazon", "venmo")


def _get_func(app: str, suffix: str) -> Any:
    mod = importlib.import_module(f"mmtoolsandbox.tools.appworld.{app}")
    return getattr(mod, f"{app}_{suffix}")


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def manage_payment_card(
    app: PaymentCardApp,
    action: Literal["list", "show", "add", "update", "delete"],
    payment_card_id: int | NotGiven = NOT_GIVEN,
    card_name: str | NotGiven = NOT_GIVEN,
    owner_name: str | NotGiven = NOT_GIVEN,
    card_number: int | NotGiven = NOT_GIVEN,
    expiry_year: int | NotGiven = NOT_GIVEN,
    expiry_month: int | NotGiven = NOT_GIVEN,
    cvv_number: int | NotGiven = NOT_GIVEN,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Manage payment cards on Spotify, Amazon, or Venmo.

    Actions:
        list: Show all your payment cards. No additional parameters needed.
        show: Show details of a specific card. Requires payment_card_id.
        add: Add a new payment card. Requires card_name, owner_name,
            card_number, expiry_year, expiry_month, cvv_number.
        update: Update a payment card's name. Requires payment_card_id and
            card_name.
        delete: Delete a payment card. Requires payment_card_id.

    Args:
        app: The app to manage payment cards on.
        action: The operation to perform.
        payment_card_id: ID of the payment card (for show, update, delete).
        card_name: Display name for the card (for add, update).
        owner_name: Full name of the card owner (for add).
        card_number: 16-digit card number (for add).
        expiry_year: Card expiration year (for add).
        expiry_month: Card expiration month (for add).
        cvv_number: 3-digit CVV number (for add).

    Returns:
        Card details, list of cards, or action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into the app.
    """
    if action == "list":
        return _get_func(app, "show_payment_cards")()
    elif action == "show":
        return _get_func(app, "show_payment_card")(payment_card_id=payment_card_id)
    elif action == "add":
        return _get_func(app, "add_payment_card")(
            card_name=card_name,
            owner_name=owner_name,
            card_number=card_number,
            expiry_year=expiry_year,
            expiry_month=expiry_month,
            cvv_number=cvv_number,
        )
    elif action == "update":
        return _get_func(app, "update_payment_card")(
            payment_card_id=payment_card_id,
            card_name=card_name,
        )
    elif action == "delete":
        return _get_func(app, "delete_payment_card")(payment_card_id=payment_card_id)
    else:
        raise ValueError(f"Unknown action: {action}")


# Mark absorbed tools
_ABSORBED: list[str] = []
for _app in _PAYMENT_CARD_APPS:
    _ABSORBED.extend(
        [
            f"{_app}_show_payment_cards",
            f"{_app}_add_payment_card",
            f"{_app}_show_payment_card",
            f"{_app}_update_payment_card",
            f"{_app}_delete_payment_card",
        ]
    )
mark_tools_absorbed_by("manage_payment_card", *_ABSORBED)
