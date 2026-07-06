"""Unit tests for AppWorld tool wrappers.

Tests verify that each tool wrapper:
1. Calls bridge.call_api with the correct app name, endpoint, and HTTP method.
2. Forwards parameters correctly (including access_token injection).
3. Respects @requires_network / @requires_auth decorators.
4. Handles dict vs list response shapes.
5. Raises ValueError on API error responses.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mmtoolsandbox.appworld.state import AppWorldState

# ---------------------------------------------------------------------------
# Patch target
# ---------------------------------------------------------------------------
# Tool modules bind ``get_appworld_bridge`` locally via ``from ... import``.
# Patching the source function does NOT affect the already-bound names.
# Instead we patch ``AppWorldBridge.get_instance`` (a classmethod) which is
# called by every ``get_appworld_bridge()`` invocation at runtime.
#
# State (``AppWorldState``) is controlled via ``set_instance`` in fixtures --
# ``get_appworld_state()`` delegates to ``AppWorldState.get_instance()`` which
# returns whatever we set.  No patch needed.
_BRIDGE_CLS_PATCH = "mmtoolsandbox.appworld.bridge.AppWorldBridge.get_instance"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_appworld_state() -> None:
    """Reset the AppWorldState singleton before each test."""
    AppWorldState.reset()


@pytest.fixture()
def mock_bridge() -> MagicMock:
    """Return a MagicMock standing in for AppWorldBridge."""
    bridge = MagicMock()
    bridge.call_api.return_value = {"status": "ok"}
    return bridge


@pytest.fixture()
def authenticated_state() -> AppWorldState:
    """Return an AppWorldState that is online and authenticated to all apps."""
    state = AppWorldState(
        network_enabled=True,
        authenticated_apps={
            "gmail",
            "amazon",
            "spotify",
            "todoist",
            "venmo",
            "simple_note",
            "file_system",
            "phone",
        },
        access_tokens={
            "gmail": "tok_gmail",
            "amazon": "tok_amazon",
            "spotify": "tok_spotify",
            "todoist": "tok_todoist",
            "venmo": "tok_venmo",
            "simple_note": "tok_simple_note",
            "file_system": "tok_file_system",
            "phone": "tok_phone",
        },
    )
    AppWorldState.set_instance(state)
    return state


@pytest.fixture()
def offline_state() -> AppWorldState:
    """Return an AppWorldState with network disabled."""
    state = AppWorldState(network_enabled=False)
    AppWorldState.set_instance(state)
    return state


@pytest.fixture()
def unauthenticated_state() -> AppWorldState:
    """Return an AppWorldState that is online but not authenticated."""
    state = AppWorldState(
        network_enabled=True,
        authenticated_apps=set(),
        access_tokens={},
    )
    AppWorldState.set_instance(state)
    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_tool(module_path: str, func_name: str) -> Any:
    """Dynamically import a tool function by module path and name."""
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


# ===================================================================
# 1. Parametrized endpoint routing tests (happy path)
# ===================================================================

# Each tuple: (import_path, func_name, kwargs, expected_app, expected_endpoint, expected_method)

_DICT_RETURNING_TOOLS: list[tuple[str, str, dict[str, Any], str, str, str]] = [
    # Gmail
    (
        "mmtoolsandbox.tools.appworld.gmail",
        "gmail_send_email",
        {
            "email_addresses": ["bob@test.com"],
            "subject": "Hi",
            "body": "Hello",
        },
        "gmail",
        "send_email",
        "post",
    ),
    (
        "mmtoolsandbox.tools.appworld.gmail",
        "gmail_show_email",
        {"email_id": 42},
        "gmail",
        "show_email",
        "get",
    ),
    (
        "mmtoolsandbox.tools.appworld.gmail",
        "gmail_show_thread",
        {"email_thread_id": 7},
        "gmail",
        "show_thread",
        "get",
    ),
    # Amazon
    (
        "mmtoolsandbox.tools.appworld.amazon",
        "amazon_show_account",
        {},
        "amazon",
        "show_account",
        "get",
    ),
    # Spotify
    (
        "mmtoolsandbox.tools.appworld.spotify",
        "spotify_show_playlist",
        {"playlist_id": 99},
        "spotify",
        "show_playlist",
        "get",
    ),
    # Todoist
    (
        "mmtoolsandbox.tools.appworld.todoist",
        "todoist_create_task",
        {"project_id": 1, "title": "Buy milk"},
        "todoist",
        "create_task",
        "post",
    ),
    (
        "mmtoolsandbox.tools.appworld.todoist",
        "todoist_show_tasks",
        {"project_id": 0},
        "todoist",
        "show_tasks",
        "get",
    ),
    # Venmo
    (
        "mmtoolsandbox.tools.appworld.venmo",
        "venmo_show_venmo_balance",
        {},
        "venmo",
        "show_venmo_balance",
        "get",
    ),
    (
        "mmtoolsandbox.tools.appworld.venmo",
        "venmo_create_transaction",
        {
            "receiver_email": "alice@test.com",
            "amount": 25.0,
        },
        "venmo",
        "create_transaction",
        "post",
    ),
    # Simple Note
    (
        "mmtoolsandbox.tools.appworld.simple_note",
        "simple_note_create_note",
        {"title": "Groceries", "content": "Eggs, Bread"},
        "simple_note",
        "create_note",
        "post",
    ),
    # File System
    (
        "mmtoolsandbox.tools.appworld.file_system",
        "file_system_show_file",
        {"file_path": "/home/user/readme.txt"},
        "file_system",
        "show_file",
        "get",
    ),
    # Supervisor (network-only, no auth)
    (
        "mmtoolsandbox.tools.appworld.supervisor",
        "supervisor_show_profile",
        {},
        "supervisor",
        "show_profile",
        "get",
    ),
]

_LIST_RETURNING_TOOLS: list[tuple[str, str, dict[str, Any], str, str, str]] = [
    # Gmail -- inbox threads (returns list)
    (
        "mmtoolsandbox.tools.appworld.gmail",
        "gmail_show_inbox_threads",
        {},
        "gmail",
        "show_inbox_threads",
        "get",
    ),
    # Amazon -- search products (returns list)
    (
        "mmtoolsandbox.tools.appworld.amazon",
        "amazon_search_products",
        {"query": "headphones"},
        "amazon",
        "search_products",
        "get",
    ),
    # Spotify -- search songs (returns list)
    (
        "mmtoolsandbox.tools.appworld.spotify",
        "spotify_search_songs",
        {"query": "bohemian"},
        "spotify",
        "search_songs",
        "get",
    ),
    # Simple Note -- search notes (returns list)
    (
        "mmtoolsandbox.tools.appworld.simple_note",
        "simple_note_search_notes",
        {},
        "simple_note",
        "search_notes",
        "get",
    ),
    # File System -- show directory (returns list)
    (
        "mmtoolsandbox.tools.appworld.file_system",
        "file_system_show_directory",
        {},
        "file_system",
        "show_directory",
        "get",
    ),
    # Contacts -- search contacts (returns list, routes to "phone" app)
    (
        "mmtoolsandbox.tools.appworld.contacts",
        "contacts_search_contacts",
        {"query": "Smith"},
        "phone",
        "search_contacts",
        "get",
    ),
]


@pytest.mark.parametrize(
    "module_path, func_name, kwargs, expected_app, expected_endpoint, expected_method",
    _DICT_RETURNING_TOOLS,
    ids=[t[1] for t in _DICT_RETURNING_TOOLS],
)
def test_dict_tool_calls_correct_endpoint(
    mock_bridge: MagicMock,
    authenticated_state: AppWorldState,
    module_path: str,
    func_name: str,
    kwargs: dict[str, Any],
    expected_app: str,
    expected_endpoint: str,
    expected_method: str,
) -> None:
    """Each dict-returning tool routes to the correct bridge.call_api endpoint."""
    tool_fn = _import_tool(module_path, func_name)

    with patch(_BRIDGE_CLS_PATCH, return_value=mock_bridge):
        result = tool_fn(**kwargs)

    mock_bridge.call_api.assert_called_once()
    call_args = mock_bridge.call_api.call_args
    assert call_args[0][0] == expected_app
    assert call_args[0][1] == expected_endpoint
    assert call_args[1]["method"] == expected_method
    assert isinstance(result, dict)


@pytest.mark.parametrize(
    "module_path, func_name, kwargs, expected_app, expected_endpoint, expected_method",
    _LIST_RETURNING_TOOLS,
    ids=[t[1] for t in _LIST_RETURNING_TOOLS],
)
def test_list_tool_calls_correct_endpoint(
    mock_bridge: MagicMock,
    authenticated_state: AppWorldState,
    module_path: str,
    func_name: str,
    kwargs: dict[str, Any],
    expected_app: str,
    expected_endpoint: str,
    expected_method: str,
) -> None:
    """Each list-returning tool routes to the correct bridge.call_api endpoint."""
    tool_fn = _import_tool(module_path, func_name)
    mock_bridge.call_api.return_value = [{"id": 1, "name": "item"}]

    with patch(_BRIDGE_CLS_PATCH, return_value=mock_bridge):
        result = tool_fn(**kwargs)

    mock_bridge.call_api.assert_called_once()
    call_args = mock_bridge.call_api.call_args
    assert call_args[0][0] == expected_app
    assert call_args[0][1] == expected_endpoint
    assert call_args[1]["method"] == expected_method
    assert isinstance(result, list)


# ===================================================================
# 2. Parameter forwarding tests
# ===================================================================


class TestGmailSendEmailParams:
    """Verify gmail_send_email forwards all parameters to bridge.call_api."""

    def test_required_params_forwarded(
        self,
        mock_bridge: MagicMock,
        authenticated_state: AppWorldState,
    ) -> None:
        from mmtoolsandbox.tools.appworld.gmail import gmail_send_email

        with patch(_BRIDGE_CLS_PATCH, return_value=mock_bridge):
            gmail_send_email(
                email_addresses=["a@b.com", "c@d.com"],
                subject="Test",
                body="Body text",
            )

        call_kwargs = mock_bridge.call_api.call_args[1]
        assert call_kwargs["email_addresses"] == ["a@b.com", "c@d.com"]
        assert call_kwargs["subject"] == "Test"
        assert call_kwargs["body"] == "Body text"
        assert call_kwargs["access_token"] == "tok_gmail"

    def test_optional_attachments_forwarded(
        self,
        mock_bridge: MagicMock,
        authenticated_state: AppWorldState,
    ) -> None:
        from mmtoolsandbox.tools.appworld.gmail import gmail_send_email

        with patch(_BRIDGE_CLS_PATCH, return_value=mock_bridge):
            gmail_send_email(
                email_addresses=["a@b.com"],
                subject="With attachment",
                body="See attached",
                attachment_file_paths=["/home/user/doc.pdf"],
                file_system_access_token="fs_tok",
            )

        call_kwargs = mock_bridge.call_api.call_args[1]
        assert call_kwargs["attachment_file_paths"] == ["/home/user/doc.pdf"]
        assert call_kwargs["file_system_access_token"] == "fs_tok"


class TestTodoistCreateTaskParams:
    """Verify todoist_create_task forwards all parameters to bridge.call_api."""

    def test_required_and_default_params(
        self,
        mock_bridge: MagicMock,
        authenticated_state: AppWorldState,
    ) -> None:
        from mmtoolsandbox.tools.appworld.todoist import todoist_create_task

        with patch(_BRIDGE_CLS_PATCH, return_value=mock_bridge):
            todoist_create_task(project_id=5, title="Write tests")

        call_kwargs = mock_bridge.call_api.call_args[1]
        assert call_kwargs["project_id"] == 5
        assert call_kwargs["title"] == "Write tests"
        assert call_kwargs["access_token"] == "tok_todoist"
        # Default order_index should be forwarded
        assert call_kwargs["order_index"] == -1

    def test_optional_params_forwarded(
        self,
        mock_bridge: MagicMock,
        authenticated_state: AppWorldState,
    ) -> None:
        from mmtoolsandbox.tools.appworld.todoist import todoist_create_task

        with patch(_BRIDGE_CLS_PATCH, return_value=mock_bridge):
            todoist_create_task(
                project_id=5,
                title="Review PR",
                due_date="2026-05-01",
                priority="high",
                description="Urgent review",
            )

        call_kwargs = mock_bridge.call_api.call_args[1]
        assert call_kwargs["due_date"] == "2026-05-01"
        assert call_kwargs["priority"] == "high"
        assert call_kwargs["description"] == "Urgent review"


class TestVenmoCreateTransactionParams:
    """Verify venmo_create_transaction forwards parameters correctly."""

    def test_required_params(
        self,
        mock_bridge: MagicMock,
        authenticated_state: AppWorldState,
    ) -> None:
        from mmtoolsandbox.tools.appworld.venmo import venmo_create_transaction

        with patch(_BRIDGE_CLS_PATCH, return_value=mock_bridge):
            venmo_create_transaction(
                receiver_email="alice@test.com",
                amount=50.0,
            )

        call_kwargs = mock_bridge.call_api.call_args[1]
        assert call_kwargs["receiver_email"] == "alice@test.com"
        assert call_kwargs["amount"] == 50.0
        assert call_kwargs["access_token"] == "tok_venmo"
        # Default values
        assert call_kwargs["description"] == ""
        assert call_kwargs["private"] is False

    def test_optional_payment_card(
        self,
        mock_bridge: MagicMock,
        authenticated_state: AppWorldState,
    ) -> None:
        from mmtoolsandbox.tools.appworld.venmo import venmo_create_transaction

        with patch(_BRIDGE_CLS_PATCH, return_value=mock_bridge):
            venmo_create_transaction(
                receiver_email="bob@test.com",
                amount=10.0,
                payment_card_id=42,
                private=True,
            )

        call_kwargs = mock_bridge.call_api.call_args[1]
        assert call_kwargs["payment_card_id"] == 42
        assert call_kwargs["private"] is True


class TestSimpleNoteCreateNoteParams:
    """Verify simple_note_create_note forwards parameters correctly."""

    def test_with_tags(
        self,
        mock_bridge: MagicMock,
        authenticated_state: AppWorldState,
    ) -> None:
        from mmtoolsandbox.tools.appworld.simple_note import simple_note_create_note

        with patch(_BRIDGE_CLS_PATCH, return_value=mock_bridge):
            simple_note_create_note(
                title="Shopping",
                content="Buy groceries",
                tags=["personal", "urgent"],
                pinned=True,
            )

        call_kwargs = mock_bridge.call_api.call_args[1]
        assert call_kwargs["title"] == "Shopping"
        assert call_kwargs["content"] == "Buy groceries"
        assert call_kwargs["tags"] == ["personal", "urgent"]
        assert call_kwargs["pinned"] is True
        assert call_kwargs["access_token"] == "tok_simple_note"


class TestContactsSearchParams:
    """Verify contacts_search_contacts routes to 'phone' app with correct params."""

    def test_routes_to_phone_app(
        self,
        mock_bridge: MagicMock,
        authenticated_state: AppWorldState,
    ) -> None:
        from mmtoolsandbox.tools.appworld.contacts import contacts_search_contacts

        mock_bridge.call_api.return_value = [{"name": "John Smith"}]

        with patch(_BRIDGE_CLS_PATCH, return_value=mock_bridge):
            result = contacts_search_contacts(query="Smith", relationship="friend")

        call_args = mock_bridge.call_api.call_args
        # contacts module routes to "phone" as the app name
        assert call_args[0][0] == "phone"
        assert call_args[0][1] == "search_contacts"
        call_kwargs = call_args[1]
        assert call_kwargs["query"] == "Smith"
        assert call_kwargs["relationship"] == "friend"
        assert call_kwargs["access_token"] == "tok_phone"
        assert result == [{"name": "John Smith"}]


# ===================================================================
# 3. @requires_network decorator tests
# ===================================================================

_NETWORK_ONLY_TOOLS: list[tuple[str, str, dict[str, Any]]] = [
    (
        "mmtoolsandbox.tools.appworld.gmail",
        "gmail_show_profile",
        {},
    ),
    (
        "mmtoolsandbox.tools.appworld.amazon",
        "amazon_search_products",
        {},
    ),
    (
        "mmtoolsandbox.tools.appworld.spotify",
        "spotify_search_songs",
        {},
    ),
    (
        "mmtoolsandbox.tools.appworld.supervisor",
        "supervisor_show_profile",
        {},
    ),
]


@pytest.mark.parametrize(
    "module_path, func_name, kwargs",
    _NETWORK_ONLY_TOOLS,
    ids=[t[1] for t in _NETWORK_ONLY_TOOLS],
)
def test_requires_network_raises_when_offline(
    mock_bridge: MagicMock,
    offline_state: AppWorldState,
    module_path: str,
    func_name: str,
    kwargs: dict[str, Any],
) -> None:
    """@requires_network raises ConnectionError when network is disabled."""
    tool_fn = _import_tool(module_path, func_name)

    with patch(_BRIDGE_CLS_PATCH, return_value=mock_bridge):
        with pytest.raises(ConnectionError, match="Network is not available"):
            tool_fn(**kwargs)

    mock_bridge.call_api.assert_not_called()


# ===================================================================
# 4. @requires_auth decorator tests
# ===================================================================

_AUTH_REQUIRED_TOOLS: list[tuple[str, str, dict[str, Any], str]] = [
    (
        "mmtoolsandbox.tools.appworld.gmail",
        "gmail_send_email",
        {"email_addresses": ["a@b.com"], "subject": "s", "body": "b"},
        "gmail",
    ),
    (
        "mmtoolsandbox.tools.appworld.spotify",
        "spotify_show_playlist",
        {"playlist_id": 1},
        "spotify",
    ),
    (
        "mmtoolsandbox.tools.appworld.todoist",
        "todoist_create_task",
        {"project_id": 1, "title": "t"},
        "todoist",
    ),
    (
        "mmtoolsandbox.tools.appworld.venmo",
        "venmo_show_venmo_balance",
        {},
        "venmo",
    ),
    (
        "mmtoolsandbox.tools.appworld.simple_note",
        "simple_note_create_note",
        {"title": "t", "content": "c"},
        "simple_note",
    ),
    (
        "mmtoolsandbox.tools.appworld.file_system",
        "file_system_show_file",
        {"file_path": "/tmp/f.txt"},
        "file_system",
    ),
]


@pytest.mark.parametrize(
    "module_path, func_name, kwargs, required_app",
    _AUTH_REQUIRED_TOOLS,
    ids=[t[1] for t in _AUTH_REQUIRED_TOOLS],
)
def test_requires_auth_raises_when_not_logged_in(
    mock_bridge: MagicMock,
    unauthenticated_state: AppWorldState,
    module_path: str,
    func_name: str,
    kwargs: dict[str, Any],
    required_app: str,
) -> None:
    """@requires_auth raises PermissionError when not logged in to the app."""
    tool_fn = _import_tool(module_path, func_name)

    with patch(_BRIDGE_CLS_PATCH, return_value=mock_bridge):
        with pytest.raises(
            PermissionError, match=f"Not authenticated to {required_app}"
        ):
            tool_fn(**kwargs)

    mock_bridge.call_api.assert_not_called()


# ===================================================================
# 5. API error response handling
# ===================================================================


class TestApiErrorResponse:
    """Verify that tools raise ValueError on API error responses."""

    def test_dict_tool_raises_on_error(
        self,
        mock_bridge: MagicMock,
        authenticated_state: AppWorldState,
    ) -> None:
        from mmtoolsandbox.tools.appworld.gmail import gmail_show_email

        mock_bridge.call_api.return_value = {"error": "Email not found"}

        with patch(_BRIDGE_CLS_PATCH, return_value=mock_bridge):
            with pytest.raises(ValueError, match="API error: Email not found"):
                gmail_show_email(email_id=999)

    def test_list_tool_raises_on_error(
        self,
        mock_bridge: MagicMock,
        authenticated_state: AppWorldState,
    ) -> None:
        from mmtoolsandbox.tools.appworld.amazon import amazon_search_products

        mock_bridge.call_api.return_value = {"error": "Invalid query"}

        with patch(_BRIDGE_CLS_PATCH, return_value=mock_bridge):
            with pytest.raises(ValueError, match="API error: Invalid query"):
                amazon_search_products(query="???")


# ===================================================================
# 6. Response shape handling
# ===================================================================


class TestResponseShapeHandling:
    """Verify tools correctly handle dict/list/nested response shapes."""

    def test_dict_response_returned_as_dict(
        self,
        mock_bridge: MagicMock,
        authenticated_state: AppWorldState,
    ) -> None:
        from mmtoolsandbox.tools.appworld.venmo import venmo_show_venmo_balance

        mock_bridge.call_api.return_value = {"balance": 150.0}

        with patch(_BRIDGE_CLS_PATCH, return_value=mock_bridge):
            result = venmo_show_venmo_balance()

        assert result == {"balance": 150.0}

    def test_list_response_returned_as_list(
        self,
        mock_bridge: MagicMock,
        authenticated_state: AppWorldState,
    ) -> None:
        from mmtoolsandbox.tools.appworld.spotify import spotify_search_songs

        songs = [{"id": 1, "title": "Song A"}, {"id": 2, "title": "Song B"}]
        mock_bridge.call_api.return_value = songs

        with patch(_BRIDGE_CLS_PATCH, return_value=mock_bridge):
            result = spotify_search_songs(query="test")

        assert result == songs

    def test_nested_list_extracted_from_dict(
        self,
        mock_bridge: MagicMock,
        authenticated_state: AppWorldState,
    ) -> None:
        """When bridge returns a dict with a list value, list-returning tools
        extract the first list found."""
        from mmtoolsandbox.tools.appworld.gmail import gmail_show_inbox_threads

        mock_bridge.call_api.return_value = {
            "threads": [{"id": 1}, {"id": 2}],
        }

        with patch(_BRIDGE_CLS_PATCH, return_value=mock_bridge):
            result = gmail_show_inbox_threads()

        assert result == [{"id": 1}, {"id": 2}]

    def test_empty_dict_when_response_is_non_dict(
        self,
        mock_bridge: MagicMock,
        authenticated_state: AppWorldState,
    ) -> None:
        """Dict-returning tools return {} when bridge gives a non-dict response."""
        from mmtoolsandbox.tools.appworld.supervisor import supervisor_show_profile

        mock_bridge.call_api.return_value = "unexpected string"

        with patch(_BRIDGE_CLS_PATCH, return_value=mock_bridge):
            result = supervisor_show_profile()

        assert result == {}

    def test_empty_list_when_response_is_non_list_non_dict(
        self,
        mock_bridge: MagicMock,
        authenticated_state: AppWorldState,
    ) -> None:
        """List-returning tools return [] when bridge gives a non-list/non-dict."""
        from mmtoolsandbox.tools.appworld.amazon import amazon_search_products

        mock_bridge.call_api.return_value = 42

        with patch(_BRIDGE_CLS_PATCH, return_value=mock_bridge):
            result = amazon_search_products()

        assert result == []

    def test_empty_list_when_dict_has_no_list_values(
        self,
        mock_bridge: MagicMock,
        authenticated_state: AppWorldState,
    ) -> None:
        """List-returning tools return [] when dict has no list values."""
        from mmtoolsandbox.tools.appworld.file_system import file_system_show_directory

        mock_bridge.call_api.return_value = {"count": 0}

        with patch(_BRIDGE_CLS_PATCH, return_value=mock_bridge):
            result = file_system_show_directory()

        assert result == []


# ===================================================================
# 7. Gmail login state management
# ===================================================================


class TestGmailLoginStateSideEffect:
    """Verify gmail_login registers auth state on successful response."""

    def test_login_registers_auth(
        self,
        mock_bridge: MagicMock,
    ) -> None:
        from mmtoolsandbox.tools.appworld.gmail import gmail_login

        state = AppWorldState(network_enabled=True)
        AppWorldState.set_instance(state)

        mock_bridge.call_api.return_value = {
            "access_token": "new_tok",
            "user_id": "u123",
        }

        with patch(_BRIDGE_CLS_PATCH, return_value=mock_bridge):
            result = gmail_login(username="user@test.com", password="pass")

        assert result["access_token"] == "new_tok"
        assert "gmail" in state.authenticated_apps
        assert state.access_tokens["gmail"] == "new_tok"

    def test_login_does_not_register_on_error(
        self,
        mock_bridge: MagicMock,
    ) -> None:
        from mmtoolsandbox.tools.appworld.gmail import gmail_login

        state = AppWorldState(network_enabled=True)
        AppWorldState.set_instance(state)

        mock_bridge.call_api.return_value = {"error": "Invalid credentials"}

        with patch(_BRIDGE_CLS_PATCH, return_value=mock_bridge):
            with pytest.raises(ValueError, match="API error"):
                gmail_login(username="user@test.com", password="bad")

        assert "gmail" not in state.authenticated_apps


# ===================================================================
# 8. Supervisor -- network-only (no auth required)
# ===================================================================


class TestSupervisorNoAuth:
    """Supervisor tools require network but NOT authentication."""

    def test_show_profile_no_access_token_in_params(
        self,
        mock_bridge: MagicMock,
        authenticated_state: AppWorldState,
    ) -> None:
        from mmtoolsandbox.tools.appworld.supervisor import supervisor_show_profile

        with patch(_BRIDGE_CLS_PATCH, return_value=mock_bridge):
            supervisor_show_profile()

        call_kwargs = mock_bridge.call_api.call_args[1]
        # Supervisor tools do not inject access_token
        assert "access_token" not in call_kwargs

    def test_show_profile_works_unauthenticated(
        self,
        mock_bridge: MagicMock,
        unauthenticated_state: AppWorldState,
    ) -> None:
        """Supervisor tools should NOT raise PermissionError."""
        from mmtoolsandbox.tools.appworld.supervisor import supervisor_show_profile

        with patch(_BRIDGE_CLS_PATCH, return_value=mock_bridge):
            result = supervisor_show_profile()

        assert isinstance(result, dict)


# ===================================================================
# 9. Optional parameter omission (None -> not forwarded)
# ===================================================================


class TestOptionalParamOmission:
    """When optional params are left at None, they are NOT sent to bridge."""

    def test_gmail_show_profile_no_email(
        self,
        mock_bridge: MagicMock,
        authenticated_state: AppWorldState,
    ) -> None:
        from mmtoolsandbox.tools.appworld.gmail import gmail_show_profile

        with patch(_BRIDGE_CLS_PATCH, return_value=mock_bridge):
            gmail_show_profile()  # email defaults to None

        call_kwargs = mock_bridge.call_api.call_args[1]
        assert "email" not in call_kwargs

    def test_gmail_show_profile_with_email(
        self,
        mock_bridge: MagicMock,
        authenticated_state: AppWorldState,
    ) -> None:
        from mmtoolsandbox.tools.appworld.gmail import gmail_show_profile

        with patch(_BRIDGE_CLS_PATCH, return_value=mock_bridge):
            gmail_show_profile(email="test@example.com")

        call_kwargs = mock_bridge.call_api.call_args[1]
        assert call_kwargs["email"] == "test@example.com"

    def test_file_system_show_directory_defaults(
        self,
        mock_bridge: MagicMock,
        authenticated_state: AppWorldState,
    ) -> None:
        from mmtoolsandbox.tools.appworld.file_system import file_system_show_directory

        mock_bridge.call_api.return_value = []

        with patch(_BRIDGE_CLS_PATCH, return_value=mock_bridge):
            file_system_show_directory()

        call_kwargs = mock_bridge.call_api.call_args[1]
        # Default values are forwarded when not None
        assert call_kwargs["directory_path"] == "/"
        assert call_kwargs["entry_type"] == "all"
        assert call_kwargs["recursive"] is True
        # substring defaults to None -> not forwarded
        assert "substring" not in call_kwargs
