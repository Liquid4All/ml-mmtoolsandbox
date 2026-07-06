# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""COMPACT cross-app tools — merges subscription management across Spotify and Amazon (8->1)."""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.datasets.names import DatasetName
from mmtoolsandbox.tools.compact import mark_compact_tools_absorbed_by


def _get_func(app: str, suffix: str) -> Any:
    import importlib

    module = importlib.import_module(f"mmtoolsandbox.tools.appworld.{app}")
    return getattr(module, f"{app}_{suffix}")


# ---------------------------------------------------------------------------
# Strategy 7: Cross-app subscription management (8 -> 1)
# ---------------------------------------------------------------------------

# Mapping from unified action names to per-app function suffixes and
# the app-specific subscription type name (premium vs prime).
_ACTION_MAP: dict[str, dict[str, str]] = {
    "show_plans": {
        "spotify": "show_premium_plans",
        "amazon": "show_prime_plans",
    },
    "subscribe": {
        "spotify": "subscribe_premium",
        "amazon": "subscribe_prime",
    },
    "show_history": {
        "spotify": "show_premium_subscriptions",
        "amazon": "show_prime_subscriptions",
    },
    "download_receipt": {
        "spotify": "download_premium_subscription_receipt",
        "amazon": "download_prime_subscription_receipt",
    },
}


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def app_manage_subscription(
    app: Literal["spotify", "amazon"],
    action: Literal["show_plans", "subscribe", "show_history", "download_receipt"],
    payment_card_id: int | NotGiven = NOT_GIVEN,
    duration: str | NotGiven = NOT_GIVEN,
    subscription_id: int | NotGiven = NOT_GIVEN,
    file_system_access_token: str | NotGiven = NOT_GIVEN,
    download_to_file_path: str | None = None,
    overwrite: bool | None = False,
    page_index: int | None = 0,
    page_limit: int | None = 5,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Show plans, subscribe, view history, or download receipts for Spotify premium or Amazon prime subscriptions.

    Actions:
        show_plans: Show available subscription plans. No extra parameters.
        subscribe: Subscribe to premium/prime. Requires payment_card_id
            and duration.
        show_history: Show subscription history. Supports page_index and
            page_limit.
        download_receipt: Download a subscription receipt. Requires
            subscription_id and file_system_access_token.

    Args:
        app: Which app's subscription to manage.
        action: The subscription operation to perform.
        payment_card_id: Payment card ID (for subscribe).
        duration: Subscription duration (for subscribe).
        subscription_id: ID of the subscription (for download_receipt).
            Maps to premium_subscription_id for Spotify,
            prime_subscription_id for Amazon.
        file_system_access_token: Access token from file_system login
            (for download_receipt).
        download_to_file_path: Destination path for receipt download.
            Defaults to ~/downloads directory.
        overwrite: Whether to overwrite existing files.
        page_index: Zero-based page index (for show_history).
        page_limit: Maximum results per page (for show_history).

    Returns:
        Subscription information or operation result.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into the app.
    """
    suffix = _ACTION_MAP[action][app]
    func = _get_func(app, suffix)

    if action == "show_plans":
        return func()

    elif action == "subscribe":
        return func(payment_card_id=payment_card_id, duration=duration)

    elif action == "show_history":
        kwargs: dict[str, Any] = {}
        if page_index is not None:
            kwargs["page_index"] = page_index
        if page_limit is not None:
            kwargs["page_limit"] = page_limit
        return func(**kwargs)

    elif action == "download_receipt":
        # Map unified subscription_id to app-specific parameter name
        id_key = (
            "premium_subscription_id" if app == "spotify" else "prime_subscription_id"
        )
        kwargs = {
            id_key: subscription_id,
            "file_system_access_token": file_system_access_token,
        }
        if download_to_file_path is not None:
            kwargs["download_to_file_path"] = download_to_file_path
        if overwrite is not None:
            kwargs["overwrite"] = overwrite
        return func(**kwargs)

    else:
        raise ValueError(f"Unknown action: {action}")


mark_compact_tools_absorbed_by(
    "app_manage_subscription",
    "spotify_show_premium_plans",
    "spotify_subscribe_premium",
    "spotify_show_premium_subscriptions",
    "spotify_download_premium_subscription_receipt",
    "amazon_show_prime_plans",
    "amazon_subscribe_prime",
    "amazon_show_prime_subscriptions",
    "amazon_download_prime_subscription_receipt",
)
