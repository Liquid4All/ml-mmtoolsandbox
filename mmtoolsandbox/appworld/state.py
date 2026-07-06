"""
State dependency management for AppWorld APIs.

This module adds MMToolSandbox-style state dependency checks to AppWorld APIs.
It enables scenarios where:
- Network must be available for API calls
- User must be authenticated to an app before using its APIs
- Rate limits are enforced
- Cross-system state dependencies (e.g., cellular status affects network)

Key Features:
- Decorators for easy state requirement declaration
- State sync with MMToolSandbox's ExecutionContext
- Support for cross-system scenarios
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
)

if TYPE_CHECKING:
    pass


class AppWorldStateKey(Enum):
    """Keys for AppWorld state tracking."""

    NETWORK_ENABLED = auto()
    AUTHENTICATED_APPS = auto()
    RATE_LIMIT_REMAINING = auto()
    CURRENT_USER_ID = auto()
    ACCESS_TOKENS = auto()


@dataclass
class AppWorldState:
    """
    Tracks state for AppWorld API dependencies.

    This singleton class maintains the current state of the AppWorld
    integration, including network status, authentication, and rate limits.

    Attributes:
        network_enabled: Whether network/HTTP calls are allowed
        authenticated_apps: Set of app names the user is logged into
        rate_limit_remaining: Number of API calls remaining
        current_user_id: ID of the currently logged-in user
        access_tokens: Mapping of app name -> access token
    """

    network_enabled: bool = True
    authenticated_apps: set[str] = field(default_factory=set)
    rate_limit_remaining: int = 2000
    current_user_id: str | None = None
    access_tokens: dict[str, str] = field(default_factory=dict)

    # Singleton instance
    _instance: AppWorldState | None = None

    @classmethod
    def get_instance(cls) -> AppWorldState:
        """Get the singleton instance of AppWorldState."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset state for a new scenario."""
        cls._instance = cls()

    @classmethod
    def set_instance(cls, state: AppWorldState) -> None:
        """Set the singleton instance (useful for testing)."""
        cls._instance = state

    def check_network(self) -> None:
        """
        Raise ConnectionError if network is disabled.

        Raises:
            ConnectionError: If network_enabled is False
        """
        if not self.network_enabled:
            raise ConnectionError(
                "Network is not available. Enable cellular or WiFi first."
            )

    def check_authenticated(self, app: str) -> None:
        """
        Raise PermissionError if not authenticated to the specified app.

        Args:
            app: The app name to check authentication for

        Raises:
            PermissionError: If not authenticated to the app
        """
        if app not in self.authenticated_apps:
            raise PermissionError(
                f"Not authenticated to {app}. Call {app}_login() first."
            )

    def check_rate_limit(self) -> None:
        """
        Raise RuntimeError if rate limit is exceeded.

        Also decrements the rate limit counter on each check.

        Raises:
            RuntimeError: If rate_limit_remaining is 0 or less
        """
        if self.rate_limit_remaining <= 0:
            raise RuntimeError(
                "API rate limit exceeded. Wait before making more calls."
            )
        self.rate_limit_remaining -= 1

    def login(self, app: str, access_token: str, user_id: str | None = None) -> None:
        """
        Record a successful login to an app.

        Args:
            app: The app name that was logged into
            access_token: The access token received from login
            user_id: Optional user ID for the logged-in user
        """
        self.authenticated_apps.add(app)
        self.access_tokens[app] = access_token
        if user_id:
            self.current_user_id = user_id

    def logout(self, app: str) -> None:
        """
        Record a logout from an app.

        Args:
            app: The app name to log out of
        """
        self.authenticated_apps.discard(app)
        self.access_tokens.pop(app, None)

    def logout_all(self) -> None:
        """Log out of all apps."""
        self.authenticated_apps.clear()
        self.access_tokens.clear()
        self.current_user_id = None

    def get_access_token(self, app: str) -> str:
        """
        Get the access token for an app.

        Args:
            app: The app name to get the token for

        Returns:
            The access token string

        Raises:
            PermissionError: If not authenticated to the app
        """
        if app not in self.access_tokens:
            raise PermissionError(
                f"No access token for {app}. Call {app}_login() first."
            )
        return self.access_tokens[app]

    def sync_with_agentsandbox(self) -> None:
        """
        Sync AppWorld state with MMToolSandbox's ExecutionContext.

        This enables cross-system state dependencies, e.g., cellular
        status in MMToolSandbox affecting network availability in AppWorld.
        """
        try:
            from mmtoolsandbox.common.databases import DatabaseNamespace
            from mmtoolsandbox.common.execution_context import (
                get_current_context,
            )

            context = get_current_context()
            setting_db = context.get_database(DatabaseNamespace.SETTING)

            # Network available if cellular OR wifi is on
            cellular_on = setting_db.get("cellular_service", [False])[0]  # type: ignore[attr-defined]
            wifi_on = setting_db.get("wifi_status", [False])[0]  # type: ignore[attr-defined]
            self.network_enabled = cellular_on or wifi_on

        except (ImportError, AttributeError, KeyError):
            # MMToolSandbox context not available, keep current network state
            pass

    def sync_with_agentsandbox_state(self) -> None:
        """Alias for sync_with_agentsandbox for backward compatibility."""
        self.sync_with_agentsandbox()

    def to_dict(self) -> dict[str, Any]:
        """Convert state to a dictionary for serialization."""
        return {
            "network_enabled": self.network_enabled,
            "authenticated_apps": list(self.authenticated_apps),
            "rate_limit_remaining": self.rate_limit_remaining,
            "current_user_id": self.current_user_id,
            "access_tokens": dict(self.access_tokens),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppWorldState:
        """Create state from a dictionary."""
        state = cls(
            network_enabled=data.get("network_enabled", True),
            authenticated_apps=set(data.get("authenticated_apps", [])),
            rate_limit_remaining=data.get("rate_limit_remaining", 2000),
            current_user_id=data.get("current_user_id"),
            access_tokens=dict(data.get("access_tokens", {})),
        )
        return state


def get_appworld_state() -> AppWorldState:
    """
    Get the singleton AppWorldState instance.

    Returns:
        The shared AppWorldState instance
    """
    return AppWorldState.get_instance()


def requires_network(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator: API requires network connectivity.

    Wraps a function to check network availability before execution.
    Also syncs with MMToolSandbox state if available.

    Args:
        func: The function to wrap

    Returns:
        Wrapped function that checks network first

    Example:
        @requires_network
        def spotify_login(email: str, password: str) -> dict:
            ...
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        state = get_appworld_state()
        # Sync with MMToolSandbox state for cross-system dependencies
        state.sync_with_agentsandbox()
        state.check_network()
        return func(*args, **kwargs)

    return wrapper


def requires_auth(app: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator factory: API requires authentication to a specific app.

    Args:
        app: The app name that requires authentication

    Returns:
        Decorator that wraps functions to check authentication

    Example:
        @requires_auth("spotify")
        def spotify_get_playlists() -> list[dict]:
            ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            state = get_appworld_state()
            state.check_authenticated(app)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def requires_rate_limit(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator: API call counts against rate limit.

    Args:
        func: The function to wrap

    Returns:
        Wrapped function that checks/decrements rate limit

    Example:
        @requires_rate_limit
        @requires_network
        def some_api_call() -> dict:
            ...
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        state = get_appworld_state()
        state.check_rate_limit()
        return func(*args, **kwargs)

    return wrapper


def with_state_sync(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator: Sync AppWorld state with MMToolSandbox before execution.

    Use this when you want to ensure state is synced but don't
    necessarily require network.

    Args:
        func: The function to wrap

    Returns:
        Wrapped function that syncs state first
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        state = get_appworld_state()
        state.sync_with_agentsandbox()
        return func(*args, **kwargs)

    return wrapper
