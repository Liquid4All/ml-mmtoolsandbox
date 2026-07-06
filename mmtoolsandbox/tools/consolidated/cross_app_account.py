# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""Cross-app account, authentication, and profile tools for the MEDIUM toolbox.

Consolidates identical account/auth/profile tools across 6-8 AppWorld apps
into single parameterized tools with an ``app`` parameter.
"""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import register_as_tool
from mmtoolsandbox.toolbox.names import ToolboxName
from mmtoolsandbox.tools.consolidated import mark_tools_absorbed_by

# All apps that have the full account/auth suite.
_FULL_ACCOUNT_APPS = (
    "spotify",
    "amazon",
    "gmail",
    "todoist",
    "venmo",
    "splitwise",
    "simple_note",
    "file_system",
)

# Apps that support email verification (Gmail does not).
_VERIFICATION_APPS = (
    "spotify",
    "amazon",
    "todoist",
    "venmo",
    "splitwise",
    "simple_note",
    "file_system",
)

# Apps that have search_users.
_SEARCH_USERS_APPS = (
    "spotify",
    "gmail",
    "todoist",
    "venmo",
    "splitwise",
)

FullAccountApp = Literal[
    "spotify",
    "amazon",
    "gmail",
    "todoist",
    "venmo",
    "splitwise",
    "simple_note",
    "file_system",
]

VerificationApp = Literal[
    "spotify",
    "amazon",
    "todoist",
    "venmo",
    "splitwise",
    "simple_note",
    "file_system",
]

SearchUsersApp = Literal[
    "spotify",
    "gmail",
    "todoist",
    "venmo",
    "splitwise",
]


def _get_original_func(app: str, func_suffix: str) -> Any:
    """Import and return the original per-app function."""
    import importlib

    module = importlib.import_module(f"mmtoolsandbox.tools.appworld.{app}")
    return getattr(module, f"{app}_{func_suffix}")


# ---------------------------------------------------------------------------
# Consolidated tools
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def app_show_account(
    app: FullAccountApp,
) -> dict[str, Any]:
    """Show your private account information on spotify, amazon, gmail, todoist, venmo, splitwise, simple_note, or file_system.

    Unlike show_profile, this returns private details (email, settings, etc.)
    and requires you to be logged in.

    Args:
        app: The app to show account information for.

    Returns:
        Account details including private information.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into the app.
    """
    return _get_original_func(app, "show_account")()


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def app_signup(
    app: FullAccountApp,
    first_name: str,
    last_name: str,
    email: str,
    password: str,
) -> dict[str, Any]:
    """Create a new account on spotify, amazon, gmail, todoist, venmo, splitwise, simple_note, or file_system.

    Args:
        app: The app to create an account on.
        first_name: Your first name.
        last_name: Your last name.
        email: Your email address.
        password: Your password.

    Returns:
        Signup confirmation with account details.

    Raises:
        ConnectionError: If network is unavailable.
    """
    return _get_original_func(app, "signup")(
        first_name=first_name,
        last_name=last_name,
        email=email,
        password=password,
    )


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def app_delete_account(
    app: FullAccountApp,
) -> dict[str, Any]:
    """Permanently delete your account on spotify, amazon, gmail, todoist, venmo, splitwise, simple_note, or file_system.

    Args:
        app: The app to delete your account from.

    Returns:
        Deletion confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into the app.
    """
    return _get_original_func(app, "delete_account")()


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def app_update_account_name(
    app: FullAccountApp,
    first_name: str | None = None,
    last_name: str | None = None,
) -> dict[str, Any]:
    """Update your first or last name on spotify, amazon, gmail, todoist, venmo, splitwise, simple_note, or file_system.

    At least one of first_name or last_name must be provided.

    Args:
        app: The app to update your name on.
        first_name: Your updated first name.
        last_name: Your updated last name.

    Returns:
        Updated account details.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into the app.
    """
    return _get_original_func(app, "update_account_name")(
        first_name=first_name,
        last_name=last_name,
    )


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def app_login(
    app: FullAccountApp,
    username: str,
    password: str,
) -> dict[str, Any]:
    """Log in to your account on spotify, amazon, gmail, todoist, venmo, splitwise, simple_note, or file_system.

    Args:
        app: The app to log in to.
        username: Your account email address.
        password: Your account password.

    Returns:
        Login confirmation with session details.

    Raises:
        ConnectionError: If network is unavailable.
    """
    return _get_original_func(app, "login")(
        username=username,
        password=password,
    )


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def app_logout(
    app: FullAccountApp,
) -> dict[str, Any]:
    """Log out from your account on spotify, amazon, gmail, todoist, venmo, splitwise, simple_note, or file_system.

    Args:
        app: The app to log out from.

    Returns:
        Logout confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into the app.
    """
    return _get_original_func(app, "logout")()


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def app_send_verification_code(
    app: VerificationApp,
    email: str,
) -> dict[str, Any]:
    """Send an account verification code on spotify, amazon, todoist, venmo, splitwise, simple_note, or file_system.

    Not available for Gmail (Gmail accounts are pre-verified).

    Args:
        app: The app to send the verification code for.
        email: The email address to send the code to.

    Returns:
        Confirmation that the code was sent.

    Raises:
        ConnectionError: If network is unavailable.
    """
    return _get_original_func(app, "send_verification_code")(email=email)


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def app_verify_account(
    app: VerificationApp,
    email: str,
    verification_code: str,
) -> dict[str, Any]:
    """Verify an account on spotify, amazon, todoist, venmo, splitwise, simple_note, or file_system.

    Not available for Gmail (Gmail accounts are pre-verified).

    Args:
        app: The app to verify the account for.
        email: The email address the code was sent to.
        verification_code: The verification code received via email.

    Returns:
        Verification confirmation.

    Raises:
        ConnectionError: If network is unavailable.
    """
    return _get_original_func(app, "verify_account")(
        email=email,
        verification_code=verification_code,
    )


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def app_send_password_reset_code(
    app: FullAccountApp,
    email: str,
) -> dict[str, Any]:
    """Send a password reset code on spotify, amazon, gmail, todoist, venmo, splitwise, simple_note, or file_system.

    Args:
        app: The app to send the reset code for.
        email: The email address to send the code to.

    Returns:
        Confirmation that the reset code was sent.

    Raises:
        ConnectionError: If network is unavailable.
    """
    return _get_original_func(app, "send_password_reset_code")(email=email)


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def app_reset_password(
    app: FullAccountApp,
    email: str,
    password_reset_code: str,
    new_password: str,
) -> dict[str, Any]:
    """Reset your password on spotify, amazon, gmail, todoist, venmo, splitwise, simple_note, or file_system.

    Args:
        app: The app to reset the password for.
        email: Your email address.
        password_reset_code: The reset code received via email.
        new_password: Your new password.

    Returns:
        Confirmation that the password was reset.

    Raises:
        ConnectionError: If network is unavailable.
    """
    return _get_original_func(app, "reset_password")(
        email=email,
        password_reset_code=password_reset_code,
        new_password=new_password,
    )


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def app_show_profile(
    app: FullAccountApp,
    email: str | None = None,
) -> dict[str, Any]:
    """Show public profile information of a user on spotify, amazon, gmail, todoist, venmo, splitwise, simple_note, or file_system.

    If email is not provided, shows your own profile.

    Args:
        app: The app to show the profile from.
        email: Email of the user whose profile to view. If omitted, shows your own.

    Returns:
        Public profile information (name, email, etc.).

    Raises:
        ConnectionError: If network is unavailable.
    """
    return _get_original_func(app, "show_profile")(email=email)


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def app_search_users(
    app: SearchUsersApp,
    query: str | None = "",
    page_index: int | None = 0,
    page_limit: int | None = 5,
) -> list[dict[str, Any]]:
    """Search for users by name or email on spotify, gmail, todoist, venmo, or splitwise.

    Not available for SimpleNote or FileSystem.

    Args:
        app: The app to search users on.
        query: Search query (name or email substring).
        page_index: Zero-based page index for pagination.
        page_limit: Maximum number of results per page.

    Returns:
        List of matching user profiles.

    Raises:
        ConnectionError: If network is unavailable.
    """
    return _get_original_func(app, "search_users")(
        query=query,
        page_index=page_index,
        page_limit=page_limit,
    )


# ---------------------------------------------------------------------------
# Mark absorbed tools
# ---------------------------------------------------------------------------

# All original per-app tools that are replaced by the consolidated versions.
# Each consolidated tool absorbs its corresponding per-app variants.
mark_tools_absorbed_by(
    "app_show_account", *[f"{_app}_show_account" for _app in _FULL_ACCOUNT_APPS]
)
mark_tools_absorbed_by("app_signup", *[f"{_app}_signup" for _app in _FULL_ACCOUNT_APPS])
mark_tools_absorbed_by(
    "app_delete_account", *[f"{_app}_delete_account" for _app in _FULL_ACCOUNT_APPS]
)
mark_tools_absorbed_by(
    "app_update_account_name",
    *[f"{_app}_update_account_name" for _app in _FULL_ACCOUNT_APPS],
)
mark_tools_absorbed_by("app_login", *[f"{_app}_login" for _app in _FULL_ACCOUNT_APPS])
mark_tools_absorbed_by("app_logout", *[f"{_app}_logout" for _app in _FULL_ACCOUNT_APPS])
mark_tools_absorbed_by(
    "app_send_password_reset_code",
    *[f"{_app}_send_password_reset_code" for _app in _FULL_ACCOUNT_APPS],
)
mark_tools_absorbed_by(
    "app_reset_password", *[f"{_app}_reset_password" for _app in _FULL_ACCOUNT_APPS]
)
mark_tools_absorbed_by(
    "app_show_profile", *[f"{_app}_show_profile" for _app in _FULL_ACCOUNT_APPS]
)
mark_tools_absorbed_by(
    "app_send_verification_code",
    *[f"{_app}_send_verification_code" for _app in _VERIFICATION_APPS],
)
mark_tools_absorbed_by(
    "app_verify_account", *[f"{_app}_verify_account" for _app in _VERIFICATION_APPS]
)
mark_tools_absorbed_by(
    "app_search_users", *[f"{_app}_search_users" for _app in _SEARCH_USERS_APPS]
)
