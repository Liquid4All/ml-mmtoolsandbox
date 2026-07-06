# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""Cross-app consolidated tools for the MINI toolbox.

Three tools that span multiple apps:
- app_account: account management (login, logout, signup, etc.)
- app_service: notifications, payment cards, subscriptions
- supervisor_show: supervisor information
"""

from __future__ import annotations

import importlib
from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.datasets.names import DatasetName


def _get_app_func(app: str, suffix: str) -> Any:
    """Lazily import an app module and return {app}_{suffix}."""
    module = importlib.import_module(f"mmtoolsandbox.tools.appworld.{app}")
    return getattr(module, f"{app}_{suffix}")


# ---------------------------------------------------------------------------
# app_account
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.MINI},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def app_account(
    app: Literal[
        "spotify",
        "amazon",
        "gmail",
        "todoist",
        "venmo",
        "splitwise",
        "simple_note",
        "file_system",
    ],
    action: Literal[
        "login",
        "logout",
        "signup",
        "delete",
        "show_account",
        "update_name",
        "show_profile",
        "search_users",
        "send_verification",
        "verify",
        "send_reset_code",
        "reset_password",
    ],
    # login
    username: str | NotGiven = NOT_GIVEN,
    password: str | NotGiven = NOT_GIVEN,
    # signup / update_name
    first_name: str | NotGiven = NOT_GIVEN,
    last_name: str | NotGiven = NOT_GIVEN,
    # signup
    email: str | NotGiven = NOT_GIVEN,
    # search_users / show_profile
    query: str | None | NotGiven = NOT_GIVEN,
    page_index: int | None | NotGiven = NOT_GIVEN,
    page_limit: int | None | NotGiven = NOT_GIVEN,
    # verification
    verification_code: str | NotGiven = NOT_GIVEN,
    # reset_password
    password_reset_code: str | NotGiven = NOT_GIVEN,
    new_password: str | NotGiven = NOT_GIVEN,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Manage app accounts: login, logout, signup, profile, and password management.

    Actions:
        login - Log in. Requires username (email), password.
        logout - Log out.
        signup - Create account. Requires first_name, last_name, email, password.
        delete - Delete account.
        show_account - Show private account info.
        update_name - Update name. Optional: first_name, last_name.
        show_profile - Show public profile. Optional: email.
        search_users - Search users. Optional: query, page_index, page_limit.
            Supported apps: spotify, amazon, gmail, todoist, venmo, splitwise.
        send_verification - Send verification code. Requires email.
            Not supported by: gmail.
        verify - Verify account. Requires email, verification_code.
            Not supported by: gmail.
        send_reset_code - Send password reset code. Requires email.
        reset_password - Reset password. Requires email, password_reset_code,
            new_password.

    Args:
        app: The application name.
        action: The account action.
        username: Account email for login.
        password: Account password for login/signup.
        first_name: First name for signup/update.
        last_name: Last name for signup/update.
        email: Email address.
        query: Search query (for search_users).
        page_index: Page index (for search_users).
        page_limit: Results per page (for search_users).
        verification_code: Verification code (for verify).
        password_reset_code: Reset code (for reset_password).
        new_password: New password (for reset_password).

    Returns:
        For login/signup: session confirmation dict.
        For delete: deletes the account. Irreversible.
        For show_account/show_profile: dict with name, email, etc.
        For search_users: list of user dicts.
        For reset_password: confirmation dict.
    """
    if action == "login":
        return _get_app_func(app, "login")(username=username, password=password)
    elif action == "logout":
        return _get_app_func(app, "logout")()
    elif action == "signup":
        return _get_app_func(app, "signup")(
            first_name=first_name,
            last_name=last_name,
            email=email,
            password=password,
        )
    elif action == "delete":
        return _get_app_func(app, "delete_account")()
    elif action == "show_account":
        return _get_app_func(app, "show_account")()
    elif action == "update_name":
        kwargs: dict[str, Any] = {}
        if first_name is not NOT_GIVEN:
            kwargs["first_name"] = first_name
        if last_name is not NOT_GIVEN:
            kwargs["last_name"] = last_name
        return _get_app_func(app, "update_account_name")(**kwargs)
    elif action == "show_profile":
        kwargs = {}
        if email is not NOT_GIVEN:
            kwargs["email"] = email
        return _get_app_func(app, "show_profile")(**kwargs)
    elif action == "search_users":
        kwargs = {}
        if query is not NOT_GIVEN:
            kwargs["query"] = query
        if page_index is not NOT_GIVEN:
            kwargs["page_index"] = page_index
        if page_limit is not NOT_GIVEN:
            kwargs["page_limit"] = page_limit
        return _get_app_func(app, "search_users")(**kwargs)
    elif action == "send_verification":
        return _get_app_func(app, "send_verification_code")(email=email)
    elif action == "verify":
        return _get_app_func(app, "verify_account")(
            email=email, verification_code=verification_code
        )
    elif action == "send_reset_code":
        return _get_app_func(app, "send_password_reset_code")(email=email)
    elif action == "reset_password":
        return _get_app_func(app, "reset_password")(
            email=email,
            password_reset_code=password_reset_code,
            new_password=new_password,
        )
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# app_service
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.MINI},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def app_service(
    service: Literal["notification", "payment_card", "subscription"],
    app: Literal[
        "spotify",
        "amazon",
        "venmo",
        "todoist",
        "splitwise",
    ],
    action: Literal[
        "list",
        "delete_all",
        "mark_all",
        "count",
        "delete",
        "mark",
        "show",
        "add",
        "update",
        "show_plans",
        "subscribe",
        "show_history",
        "download_receipt",
    ],
    # notification params
    notification_id: int | NotGiven = NOT_GIVEN,
    read: bool | None | NotGiven = NOT_GIVEN,
    # payment_card params
    payment_card_id: int | NotGiven = NOT_GIVEN,
    card_name: str | NotGiven = NOT_GIVEN,
    owner_name: str | NotGiven = NOT_GIVEN,
    card_number: int | NotGiven = NOT_GIVEN,
    expiry_year: int | NotGiven = NOT_GIVEN,
    expiry_month: int | NotGiven = NOT_GIVEN,
    cvv_number: int | NotGiven = NOT_GIVEN,
    # subscription params
    duration: str | NotGiven = NOT_GIVEN,
    subscription_id: int | NotGiven = NOT_GIVEN,
    file_system_access_token: str | NotGiven = NOT_GIVEN,
    download_to_file_path: str | None | NotGiven = NOT_GIVEN,
    overwrite: bool | None | NotGiven = NOT_GIVEN,
    # pagination
    page_index: int | None | NotGiven = NOT_GIVEN,
    page_limit: int | None | NotGiven = NOT_GIVEN,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Manage notifications, payment cards, and subscriptions across apps.

    Services and actions:
        notification (apps: todoist, venmo, splitwise):
            list - List notifications. Optional: read (filter), page_index, page_limit.
            delete_all - Delete all notifications.
            mark_all - Mark all as read/unread. Requires read.
            count - Count notifications. Optional: read.
            delete - Delete one notification. Requires notification_id.
            mark - Mark one notification. Requires notification_id, read.
        payment_card (apps: spotify, amazon, venmo):
            list - List payment cards.
            show - Show card details. Requires payment_card_id.
            add - Add card. Requires card_name, owner_name, card_number,
                expiry_year, expiry_month, cvv_number.
            update - Update card name. Requires payment_card_id, card_name.
            delete - Delete card. Requires payment_card_id.
        subscription (apps: spotify, amazon):
            show_plans - Show available plans.
            subscribe - Subscribe. Requires payment_card_id, duration.
            show_history - Show subscription history.
                Optional: page_index, page_limit.
            download_receipt - Download receipt. Requires subscription_id,
                file_system_access_token.

    Args:
        service: The service type.
        app: The application name.
        action: The specific action.
        notification_id: Notification ID (for delete/mark).
        read: Read status (for notification filter/mark).
        payment_card_id: Card ID (for show/update/delete/subscribe).
        card_name: Card name (for add/update).
        owner_name: Card owner name (for add).
        card_number: 16-digit card number (for add).
        expiry_year: Card expiry year (for add).
        expiry_month: Card expiry month (for add).
        cvv_number: 3-digit CVV (for add).
        duration: Subscription duration (for subscribe).
        subscription_id: Subscription ID (for download_receipt).
        file_system_access_token: File system token (for download_receipt).
        download_to_file_path: Download path (for download_receipt).
        overwrite: Overwrite existing file (for download_receipt).
        page_index: Page index for pagination.
        page_limit: Results per page.

    Returns:
        For notification list: list of notification dicts.
        For notification delete/delete_all: confirmation. Irreversible.
        For payment_card add: dict with payment_card_id.
        For payment_card delete: confirmation. Irreversible.
        For subscription subscribe: dict. Charges payment card —
            externally visible.
        For subscription download_receipt: file download confirmation.
    """
    if service == "notification":
        if action == "list":
            kwargs: dict[str, Any] = {}
            if read is not NOT_GIVEN:
                kwargs["read"] = read
            if page_index is not NOT_GIVEN:
                kwargs["page_index"] = page_index
            if page_limit is not NOT_GIVEN:
                kwargs["page_limit"] = page_limit
            return _get_app_func(app, "show_notifications")(**kwargs)
        elif action == "delete_all":
            return _get_app_func(app, "delete_notifications")()
        elif action == "mark_all":
            return _get_app_func(app, "mark_notifications")(read=read)
        elif action == "count":
            kwargs = {}
            if read is not NOT_GIVEN:
                kwargs["read"] = read
            return _get_app_func(app, "show_notifications_count")(**kwargs)
        elif action == "delete":
            return _get_app_func(app, "delete_notification")(
                notification_id=notification_id
            )
        elif action == "mark":
            return _get_app_func(app, "mark_notification")(
                notification_id=notification_id, read=read
            )
        else:
            raise ValueError(f"Unknown notification action: {action}")

    elif service == "payment_card":
        if action == "list":
            return _get_app_func(app, "show_payment_cards")()
        elif action == "show":
            return _get_app_func(app, "show_payment_card")(
                payment_card_id=payment_card_id
            )
        elif action == "add":
            return _get_app_func(app, "add_payment_card")(
                card_name=card_name,
                owner_name=owner_name,
                card_number=card_number,
                expiry_year=expiry_year,
                expiry_month=expiry_month,
                cvv_number=cvv_number,
            )
        elif action == "update":
            return _get_app_func(app, "update_payment_card")(
                payment_card_id=payment_card_id, card_name=card_name
            )
        elif action == "delete":
            return _get_app_func(app, "delete_payment_card")(
                payment_card_id=payment_card_id
            )
        else:
            raise ValueError(f"Unknown payment_card action: {action}")

    elif service == "subscription":
        # Map app to subscription naming: spotify=premium, amazon=prime
        sub_type = "premium" if app == "spotify" else "prime"
        if action == "show_plans":
            return _get_app_func(app, f"show_{sub_type}_plans")()
        elif action == "subscribe":
            return _get_app_func(app, f"subscribe_{sub_type}")(
                payment_card_id=payment_card_id, duration=duration
            )
        elif action == "show_history":
            kwargs = {}
            if page_index is not NOT_GIVEN:
                kwargs["page_index"] = page_index
            if page_limit is not NOT_GIVEN:
                kwargs["page_limit"] = page_limit
            return _get_app_func(app, f"show_{sub_type}_subscriptions")(**kwargs)
        elif action == "download_receipt":
            kwargs = {
                f"{sub_type}_subscription_id": subscription_id,
                "file_system_access_token": file_system_access_token,
            }
            if download_to_file_path is not NOT_GIVEN:
                kwargs["download_to_file_path"] = download_to_file_path
            if overwrite is not NOT_GIVEN:
                kwargs["overwrite"] = overwrite
            return _get_app_func(app, f"download_{sub_type}_subscription_receipt")(
                **kwargs
            )
        else:
            raise ValueError(f"Unknown subscription action: {action}")

    else:
        raise ValueError(f"Unknown service: {service}")


# ---------------------------------------------------------------------------
# supervisor_show
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.MINI},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def supervisor_show(
    entity_type: Literal["profile", "addresses", "payment_cards", "passwords"],
) -> dict[str, Any] | list[dict[str, Any]]:
    """View supervisor and admin information across apps.

    Entity types:
        profile - Show supervisor's profile.
        addresses - Show supervisor's addresses.
        payment_cards - Show supervisor's payment cards.
        passwords - Show supervisor's app account passwords.

    Args:
        entity_type: The type of supervisor information to view.

    Returns:
        For profile: dict with name, email, phone.
        For addresses: list of address dicts.
        For payment_cards: list of card dicts with payment_card_id.
        For passwords: dict with app passwords. Contains sensitive data.

    Raises:
        ConnectionError: If network is unavailable.
    """
    import mmtoolsandbox.tools.appworld.supervisor as m

    dispatch = {
        "profile": m.supervisor_show_profile,
        "addresses": m.supervisor_show_addresses,
        "payment_cards": m.supervisor_show_payment_cards,
        "passwords": m.supervisor_show_account_passwords,
    }
    return dispatch[entity_type]()
