"""Unit tests for mmtoolsandbox.appworld.bridge.

Tests the AppWorldBridge singleton, its lifecycle methods (initialize, close,
reset), API call routing, database modifications, context manager protocol,
and helper functions — all WITHOUT requiring AppWorld to be installed.

Every external dependency (appworld.Environment, appworld.Requester,
SQLAlchemy engines) is mocked at the boundary.
"""

from __future__ import annotations

import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mmtoolsandbox.appworld.bridge import (
    AppWorldBridge,
    AppWorldNotInstalledError,
    _configure_appworld_root,
    get_appworld_bridge,
    load_task_ids,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def _reset_singleton() -> Any:
    """Reset the AppWorldBridge singleton before and after each test.

    Patches close() on any existing instance to avoid calling into
    real AppWorld teardown.
    """
    # Before: forcibly clear singleton without calling close
    AppWorldBridge._instance = None

    yield

    # After: forcibly clear singleton without calling close
    if AppWorldBridge._instance is not None:
        AppWorldBridge._instance._appworld = None
        AppWorldBridge._instance = None


@pytest.fixture()
def bridge() -> AppWorldBridge:
    """Return a fresh AppWorldBridge instance (not via singleton)."""
    return AppWorldBridge()


@pytest.fixture()
def mock_appworld_module() -> MagicMock:
    """Create a mock that mimics the ``appworld`` top-level module.

    The mock has an ``AppWorld`` class whose instances expose
    ``.requester``, ``.task``, ``.models``, ``.execute()``,
    ``.evaluate()``, and ``.close()``.
    """
    mock_module = MagicMock(spec=["AppWorld"])
    mock_appworld_cls = MagicMock()
    mock_module.AppWorld = mock_appworld_cls

    # AppWorld() returns an instance with the expected attributes
    instance = MagicMock()
    instance.requester = MagicMock()
    instance.task = MagicMock()
    instance.models = MagicMock()
    instance.execute = MagicMock(return_value="exec_result")
    instance.evaluate = MagicMock(return_value={"score": 1.0})
    instance.close = MagicMock()
    mock_appworld_cls.return_value = instance

    return mock_module


def _init_bridge(bridge: AppWorldBridge, mock_appworld_module: MagicMock) -> MagicMock:
    """Helper: initialise a bridge with mocked appworld module.

    Returns the mock AppWorld *instance* that was created.
    """
    bridge._appworld_module = mock_appworld_module
    with (
        patch.object(bridge, "_ensure_appworld_imported"),
        patch("mmtoolsandbox.appworld.bridge._configure_appworld_root"),
        patch.object(bridge, "_force_clear_freezegun_state"),
    ):
        bridge.initialize("test_task_001")
    return mock_appworld_module.AppWorld.return_value  # type: ignore[no-any-return]


# ============================================================================
# Singleton Behaviour
# ============================================================================


class TestGetAppworldBridge:
    """Tests for the ``get_appworld_bridge()`` module-level accessor."""

    def test_returns_appworld_bridge_instance(self) -> None:
        result = get_appworld_bridge()
        assert isinstance(result, AppWorldBridge)

    def test_returns_same_instance_on_repeated_calls(self) -> None:
        first = get_appworld_bridge()
        second = get_appworld_bridge()
        assert first is second

    def test_get_instance_returns_singleton(self) -> None:
        a = AppWorldBridge.get_instance()
        b = AppWorldBridge.get_instance()
        assert a is b

    def test_reset_clears_singleton(self) -> None:
        first = get_appworld_bridge()
        with patch(
            "mmtoolsandbox.appworld.bridge.AppWorldBridge._force_clear_freezegun_state"
        ):
            AppWorldBridge.reset()
        second = get_appworld_bridge()
        assert first is not second


# ============================================================================
# Initialization
# ============================================================================


class TestInitialize:
    """Tests for ``AppWorldBridge.initialize()``."""

    def test_initialize_creates_appworld_instance(
        self, bridge: AppWorldBridge, mock_appworld_module: MagicMock
    ) -> None:
        instance = _init_bridge(bridge, mock_appworld_module)
        mock_appworld_module.AppWorld.assert_called_once_with(
            task_id="test_task_001",
            experiment_name="mmtoolsandbox",
            ground_truth_mode="minimal",
            max_api_calls_per_interaction=1000,
        )
        assert bridge._appworld is instance

    def test_initialize_with_custom_params(
        self, bridge: AppWorldBridge, mock_appworld_module: MagicMock
    ) -> None:
        bridge._appworld_module = mock_appworld_module
        with (
            patch.object(bridge, "_ensure_appworld_imported"),
            patch("mmtoolsandbox.appworld.bridge._configure_appworld_root"),
            patch.object(bridge, "_force_clear_freezegun_state"),
        ):
            bridge.initialize(
                task_id="custom_task",
                experiment_name="my_experiment",
                ground_truth_mode="full",
                max_api_calls_per_interaction=500,
            )
        mock_appworld_module.AppWorld.assert_called_once_with(
            task_id="custom_task",
            experiment_name="my_experiment",
            ground_truth_mode="full",
            max_api_calls_per_interaction=500,
        )

    def test_initialize_returns_self_for_chaining(
        self, bridge: AppWorldBridge, mock_appworld_module: MagicMock
    ) -> None:
        bridge._appworld_module = mock_appworld_module
        with (
            patch.object(bridge, "_ensure_appworld_imported"),
            patch("mmtoolsandbox.appworld.bridge._configure_appworld_root"),
            patch.object(bridge, "_force_clear_freezegun_state"),
        ):
            result = bridge.initialize("task_123")
        assert result is bridge

    def test_initialize_closes_existing_instance_first(
        self, bridge: AppWorldBridge, mock_appworld_module: MagicMock
    ) -> None:
        # Initialize once
        first_instance = _init_bridge(bridge, mock_appworld_module)

        # Reset the mock call count, set up a second instance
        mock_appworld_module.AppWorld.reset_mock()
        second_instance = MagicMock()
        mock_appworld_module.AppWorld.return_value = second_instance

        # Initialize again — should close the first
        bridge._appworld_module = mock_appworld_module
        with (
            patch.object(bridge, "_ensure_appworld_imported"),
            patch("mmtoolsandbox.appworld.bridge._configure_appworld_root"),
            patch.object(bridge, "_force_clear_freezegun_state"),
        ):
            bridge.initialize("task_002")

        first_instance.close.assert_called_once()
        assert bridge._appworld is second_instance

    def test_initialize_calls_force_clear_freezegun_state(
        self, bridge: AppWorldBridge, mock_appworld_module: MagicMock
    ) -> None:
        bridge._appworld_module = mock_appworld_module
        with (
            patch.object(bridge, "_ensure_appworld_imported"),
            patch("mmtoolsandbox.appworld.bridge._configure_appworld_root"),
            patch.object(bridge, "_force_clear_freezegun_state") as mock_clear,
        ):
            bridge.initialize("task_123")
        mock_clear.assert_called_once()

    def test_is_initialized_false_before_init(self, bridge: AppWorldBridge) -> None:
        assert bridge.is_initialized is False

    def test_is_initialized_true_after_init(
        self, bridge: AppWorldBridge, mock_appworld_module: MagicMock
    ) -> None:
        _init_bridge(bridge, mock_appworld_module)
        assert bridge.is_initialized is True


# ============================================================================
# Lazy Import
# ============================================================================


class TestEnsureAppworldImported:
    """Tests for ``_ensure_appworld_imported()``."""

    def test_imports_appworld_on_first_call(self, bridge: AppWorldBridge) -> None:
        fake_module = types.ModuleType("appworld")
        with patch.dict("sys.modules", {"appworld": fake_module}):
            bridge._ensure_appworld_imported()
        assert bridge._appworld_module is fake_module

    def test_raises_custom_error_when_not_installed(
        self, bridge: AppWorldBridge
    ) -> None:
        with patch.dict("sys.modules", {"appworld": None}):
            with pytest.raises(AppWorldNotInstalledError, match="not installed"):
                bridge._ensure_appworld_imported()

    def test_idempotent_on_second_call(self, bridge: AppWorldBridge) -> None:
        sentinel = MagicMock()
        bridge._appworld_module = sentinel
        bridge._ensure_appworld_imported()
        assert bridge._appworld_module is sentinel


# ============================================================================
# Property Access Guards
# ============================================================================


class TestPropertyGuards:
    """Tests that properties raise RuntimeError when not initialized."""

    def test_requester_raises_when_not_initialized(
        self, bridge: AppWorldBridge
    ) -> None:
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = bridge.requester

    def test_task_raises_when_not_initialized(self, bridge: AppWorldBridge) -> None:
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = bridge.task

    def test_appworld_raises_when_not_initialized(self, bridge: AppWorldBridge) -> None:
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = bridge.appworld

    def test_requester_returns_value_when_initialized(
        self, bridge: AppWorldBridge, mock_appworld_module: MagicMock
    ) -> None:
        instance = _init_bridge(bridge, mock_appworld_module)
        result = bridge.requester
        assert result is instance.requester

    def test_task_returns_value_when_initialized(
        self, bridge: AppWorldBridge, mock_appworld_module: MagicMock
    ) -> None:
        instance = _init_bridge(bridge, mock_appworld_module)
        result = bridge.task
        assert result is instance.task

    def test_appworld_returns_value_when_initialized(
        self, bridge: AppWorldBridge, mock_appworld_module: MagicMock
    ) -> None:
        instance = _init_bridge(bridge, mock_appworld_module)
        result = bridge.appworld
        # bridge.appworld returns self._appworld, which IS the instance
        assert result is instance


# ============================================================================
# call_api
# ============================================================================


class TestCallApi:
    """Tests for ``call_api()``."""

    def test_call_api_raises_when_not_initialized(self, bridge: AppWorldBridge) -> None:
        with pytest.raises(RuntimeError, match="not initialized"):
            bridge.call_api("spotify", "login")

    def test_call_api_delegates_to_requester_request(
        self, bridge: AppWorldBridge, mock_appworld_module: MagicMock
    ) -> None:
        instance = _init_bridge(bridge, mock_appworld_module)
        instance.requester.request.return_value = {"status": "ok"}

        result = bridge.call_api(
            "spotify", "login", email="user@test.com", password="pw"
        )

        instance.requester.request.assert_called_once_with(
            _app_name="spotify",
            _api_name="login",
            email="user@test.com",
            password="pw",
        )
        assert result == {"status": "ok"}

    def test_call_api_passes_method_kwarg_is_ignored(
        self, bridge: AppWorldBridge, mock_appworld_module: MagicMock
    ) -> None:
        """The ``method`` parameter is accepted but not forwarded."""
        instance = _init_bridge(bridge, mock_appworld_module)
        instance.requester.request.return_value = {}

        bridge.call_api("amazon", "search", method="get", query="laptop")

        # method is NOT forwarded — only _app_name, _api_name, **params
        instance.requester.request.assert_called_once_with(
            _app_name="amazon", _api_name="search", query="laptop"
        )


# ============================================================================
# execute
# ============================================================================


class TestExecute:
    """Tests for ``execute()``."""

    def test_execute_raises_when_not_initialized(self, bridge: AppWorldBridge) -> None:
        with pytest.raises(RuntimeError, match="not initialized"):
            bridge.execute("print('hello')")

    def test_execute_delegates_to_appworld(
        self, bridge: AppWorldBridge, mock_appworld_module: MagicMock
    ) -> None:
        instance = _init_bridge(bridge, mock_appworld_module)
        instance.execute.return_value = "result_42"

        result = bridge.execute("x = 42")

        instance.execute.assert_called_once_with("x = 42")
        assert result == "result_42"


# ============================================================================
# run_setup_code
# ============================================================================


class TestRunSetupCode:
    """Tests for ``run_setup_code()``."""

    def test_run_setup_code_raises_when_not_initialized(
        self, bridge: AppWorldBridge
    ) -> None:
        with pytest.raises(RuntimeError, match="not initialized"):
            bridge.run_setup_code("apis.spotify.login()")

    def test_run_setup_code_delegates_to_execute(
        self, bridge: AppWorldBridge, mock_appworld_module: MagicMock
    ) -> None:
        instance = _init_bridge(bridge, mock_appworld_module)
        instance.execute.return_value = "setup_done"

        result = bridge.run_setup_code("apis.spotify.login()")

        instance.execute.assert_called_once_with("apis.spotify.login()")
        assert result == "setup_done"


# ============================================================================
# evaluate
# ============================================================================


class TestEvaluate:
    """Tests for ``evaluate()``."""

    def test_evaluate_raises_when_not_initialized(self, bridge: AppWorldBridge) -> None:
        with pytest.raises(RuntimeError, match="not initialized"):
            bridge.evaluate()

    def test_evaluate_delegates_to_appworld(
        self, bridge: AppWorldBridge, mock_appworld_module: MagicMock
    ) -> None:
        instance = _init_bridge(bridge, mock_appworld_module)
        instance.evaluate.return_value = {"tgc_score": 0.85}

        result = bridge.evaluate()

        instance.evaluate.assert_called_once()
        assert result == {"tgc_score": 0.85}


# ============================================================================
# close / context manager
# ============================================================================


class TestClose:
    """Tests for ``close()`` and the context manager protocol."""

    def test_close_calls_appworld_close(
        self, bridge: AppWorldBridge, mock_appworld_module: MagicMock
    ) -> None:
        instance = _init_bridge(bridge, mock_appworld_module)
        bridge.close()
        instance.close.assert_called_once()
        assert bridge._appworld is None

    def test_close_idempotent_when_not_initialized(
        self, bridge: AppWorldBridge
    ) -> None:
        # Should not raise
        bridge.close()
        assert bridge._appworld is None

    def test_close_swallows_appworld_exception(
        self, bridge: AppWorldBridge, mock_appworld_module: MagicMock
    ) -> None:
        instance = _init_bridge(bridge, mock_appworld_module)
        instance.close.side_effect = RuntimeError("cleanup failed")
        # Should not raise
        bridge.close()
        assert bridge._appworld is None

    def test_context_manager_calls_close_on_exit(
        self, bridge: AppWorldBridge, mock_appworld_module: MagicMock
    ) -> None:
        instance = _init_bridge(bridge, mock_appworld_module)
        with bridge:
            assert bridge.is_initialized
        instance.close.assert_called_once()
        assert bridge._appworld is None

    def test_context_manager_calls_close_on_exception(
        self, bridge: AppWorldBridge, mock_appworld_module: MagicMock
    ) -> None:
        instance = _init_bridge(bridge, mock_appworld_module)
        with pytest.raises(ValueError, match="boom"):
            with bridge:
                raise ValueError("boom")
        instance.close.assert_called_once()
        assert bridge._appworld is None


# ============================================================================
# reset
# ============================================================================


class TestReset:
    """Tests for ``AppWorldBridge.reset()``."""

    def test_reset_clears_instance(self) -> None:
        _ = AppWorldBridge.get_instance()
        assert AppWorldBridge._instance is not None
        with patch.object(AppWorldBridge, "_force_clear_freezegun_state"):
            AppWorldBridge.reset()
        assert AppWorldBridge._instance is None

    def test_reset_closes_instance(self, mock_appworld_module: MagicMock) -> None:
        bridge = AppWorldBridge.get_instance()
        instance = _init_bridge(bridge, mock_appworld_module)

        with patch.object(AppWorldBridge, "_force_clear_freezegun_state"):
            AppWorldBridge.reset()

        instance.close.assert_called_once()

    def test_reset_safe_when_no_instance(self) -> None:
        AppWorldBridge._instance = None
        with patch.object(AppWorldBridge, "_force_clear_freezegun_state"):
            # Should not raise
            AppWorldBridge.reset()

    def test_reset_calls_force_clear_freezegun_state(self) -> None:
        _ = AppWorldBridge.get_instance()
        with patch.object(AppWorldBridge, "_force_clear_freezegun_state") as mock_clear:
            AppWorldBridge.reset()
        mock_clear.assert_called_once()

    def test_reset_calls_requester_close_all(self) -> None:
        _ = AppWorldBridge.get_instance()
        mock_requester = MagicMock()
        with (
            patch.object(AppWorldBridge, "_force_clear_freezegun_state"),
            patch.dict(
                "sys.modules",
                {
                    "appworld": MagicMock(),
                    "appworld.requester": MagicMock(Requester=mock_requester),
                    "appworld.environment": MagicMock(AppWorld=MagicMock()),
                },
            ),
        ):
            AppWorldBridge.reset()
        mock_requester.close_all.assert_called_once()

    def test_reset_calls_appworld_close_all(self) -> None:
        _ = AppWorldBridge.get_instance()
        mock_appworld_env = MagicMock()
        with (
            patch.object(AppWorldBridge, "_force_clear_freezegun_state"),
            patch.dict(
                "sys.modules",
                {
                    "appworld": MagicMock(),
                    "appworld.requester": MagicMock(Requester=MagicMock()),
                    "appworld.environment": MagicMock(AppWorld=mock_appworld_env),
                },
            ),
        ):
            AppWorldBridge.reset()
        mock_appworld_env.close_all.assert_called_once()


# ============================================================================
# apply_database_modifications
# ============================================================================


class TestApplyDatabaseModifications:
    """Tests for ``apply_database_modifications()``."""

    def test_raises_when_not_initialized(self, bridge: AppWorldBridge) -> None:
        with pytest.raises(RuntimeError, match="not initialized"):
            bridge.apply_database_modifications([("admin", "SELECT 1", [])])

    def test_executes_sql_on_correct_engine(
        self, bridge: AppWorldBridge, mock_appworld_module: MagicMock
    ) -> None:
        instance = _init_bridge(bridge, mock_appworld_module)

        # Build a mock model hierarchy:
        # models.spotify.SQLModel.db.engine → mock engine
        mock_conn = MagicMock()
        mock_engine = MagicMock()
        # engine.connect() is used as a context manager
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_sql_model = MagicMock()
        mock_sql_model.db.engine = mock_engine

        mock_spotify = MagicMock()
        mock_spotify.SQLModel = mock_sql_model

        instance.models.spotify = mock_spotify

        bridge.apply_database_modifications(
            [
                ("spotify", "UPDATE playlists SET name = ? WHERE id = ?", ["Test", 1]),
            ]
        )

        # Verify connection was used
        mock_engine.connect.assert_called_once()
        mock_conn.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

    def test_converts_question_mark_placeholders_to_named(
        self, bridge: AppWorldBridge, mock_appworld_module: MagicMock
    ) -> None:
        instance = _init_bridge(bridge, mock_appworld_module)

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_sql_model = MagicMock()
        mock_sql_model.db.engine = mock_engine

        mock_admin = MagicMock()
        mock_admin.SQLModel = mock_sql_model
        instance.models.admin = mock_admin

        bridge.apply_database_modifications(
            [
                (
                    "admin",
                    "INSERT INTO users (id, name, email) VALUES (?, ?, ?)",
                    [9999, "User", "u@example.com"],
                ),
            ]
        )

        # Check that execute was called with the named-parameter SQL
        assert mock_conn.execute.call_count == 1
        called_args = mock_conn.execute.call_args
        # The first argument is a text() object — check its string representation
        sql_arg = called_args[0][0]
        param_dict = called_args[0][1]
        assert param_dict == {"p0": 9999, "p1": "User", "p2": "u@example.com"}

    def test_warns_on_missing_app(
        self,
        bridge: AppWorldBridge,
        mock_appworld_module: MagicMock,
    ) -> None:
        instance = _init_bridge(bridge, mock_appworld_module)
        # Make models have no attribute for "nonexistent_app"
        instance.models = MagicMock(spec=[])

        with pytest.raises(RuntimeError, match="not found"):
            bridge.apply_database_modifications(
                [
                    ("nonexistent_app", "SELECT 1", []),
                ]
            )

    def test_warns_on_missing_engine(
        self,
        bridge: AppWorldBridge,
        mock_appworld_module: MagicMock,
    ) -> None:
        instance = _init_bridge(bridge, mock_appworld_module)

        mock_sql_model = MagicMock()
        mock_sql_model.db.engine = None

        mock_app = MagicMock()
        mock_app.SQLModel = mock_sql_model
        instance.models.myapp = mock_app

        with pytest.raises(RuntimeError, match="No database engine"):
            bridge.apply_database_modifications(
                [
                    ("myapp", "SELECT 1", []),
                ]
            )

    def test_warns_on_sql_execution_error(
        self,
        bridge: AppWorldBridge,
        mock_appworld_module: MagicMock,
    ) -> None:
        instance = _init_bridge(bridge, mock_appworld_module)

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = RuntimeError("SQL error")
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_sql_model = MagicMock()
        mock_sql_model.db.engine = mock_engine

        mock_app = MagicMock()
        mock_app.SQLModel = mock_sql_model
        instance.models.badapp = mock_app

        with pytest.raises(RuntimeError, match="Failed to apply modification"):
            bridge.apply_database_modifications(
                [
                    ("badapp", "INVALID SQL ?", [1]),
                ]
            )

    def test_multiple_modifications_all_executed(
        self, bridge: AppWorldBridge, mock_appworld_module: MagicMock
    ) -> None:
        instance = _init_bridge(bridge, mock_appworld_module)

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_sql_model = MagicMock()
        mock_sql_model.db.engine = mock_engine

        # Both apps share the same mock engine for simplicity
        mock_app = MagicMock()
        mock_app.SQLModel = mock_sql_model
        instance.models.app_a = mock_app
        instance.models.app_b = mock_app

        bridge.apply_database_modifications(
            [
                ("app_a", "UPDATE t SET x = ? WHERE id = ?", ["val", 1]),
                ("app_b", "DELETE FROM t WHERE id = ?", [2]),
            ]
        )

        assert mock_conn.execute.call_count == 2
        assert mock_conn.commit.call_count == 2


# ============================================================================
# _extract_insert_info
# ============================================================================


class TestExtractInsertInfo:
    """Tests for the static ``_extract_insert_info()`` helper."""

    def test_simple_insert(self) -> None:
        sql = "INSERT INTO users (id, name, email) VALUES (?, ?, ?)"
        params = [42, "Alice", "alice@test.com"]
        table, id_val = AppWorldBridge._extract_insert_info(sql, params)
        assert table == "users"
        assert id_val == 42

    def test_insert_or_replace(self) -> None:
        sql = "INSERT OR REPLACE INTO notes (id, title) VALUES (?, ?)"
        params = [7, "My Note"]
        table, id_val = AppWorldBridge._extract_insert_info(sql, params)
        assert table == "notes"
        assert id_val == 7

    def test_insert_without_id_column(self) -> None:
        sql = "INSERT INTO logs (timestamp, message) VALUES (?, ?)"
        params = ["2026-01-01", "hello"]
        table, id_val = AppWorldBridge._extract_insert_info(sql, params)
        assert table == "logs"
        assert id_val is None

    def test_unparseable_sql(self) -> None:
        sql = "UPDATE users SET name = ? WHERE id = ?"
        params = ["Bob", 1]
        table, id_val = AppWorldBridge._extract_insert_info(sql, params)
        assert table is None
        assert id_val is None

    def test_id_not_first_column(self) -> None:
        sql = "INSERT INTO songs (title, id, artist) VALUES (?, ?, ?)"
        params = ["Song A", 99, "Artist B"]
        table, id_val = AppWorldBridge._extract_insert_info(sql, params)
        assert table == "songs"
        assert id_val == 99

    def test_id_index_out_of_range(self) -> None:
        # More column names than parameters (malformed input)
        sql = "INSERT INTO t (name, id) VALUES (?, ?)"
        params = ["only_one"]
        table, id_val = AppWorldBridge._extract_insert_info(sql, params)
        assert table == "t"
        assert id_val is None

    def test_case_insensitive(self) -> None:
        sql = "insert into Contacts (id, phone) values (?, ?)"
        params = [5, "555-1234"]
        table, id_val = AppWorldBridge._extract_insert_info(sql, params)
        assert table == "Contacts"
        assert id_val == 5


# ============================================================================
# _update_fts_for_inserted_rows
# ============================================================================


class TestUpdateFtsForInsertedRows:
    """Tests for ``_update_fts_for_inserted_rows()``."""

    def test_noop_when_not_initialized(self, bridge: AppWorldBridge) -> None:
        # Should not raise even though _appworld is None
        bridge._update_fts_for_inserted_rows([("app", "table", 1)])

    def test_calls_set_search_text_on_searchable_record(
        self, bridge: AppWorldBridge, mock_appworld_module: MagicMock
    ) -> None:
        instance = _init_bridge(bridge, mock_appworld_module)

        mock_record = MagicMock()

        # The model class must pass isinstance(attr, type) check in the
        # real code, so we use a real class with mock methods.
        class FakeNoteModel:
            table_name = "notes"

            @staticmethod
            def searchable() -> bool:
                return True

            @staticmethod
            def find_one(id: int) -> MagicMock:  # noqa: A002
                return mock_record

        # Build the models mock so that iterating over dir() finds the model
        mock_app_models = MagicMock()
        mock_app_models.NoteModel = FakeNoteModel
        # Override dir() to return our model
        type(mock_app_models).__dir__ = MagicMock(  # type: ignore[method-assign]
            return_value=["NoteModel", "__class__"]
        )
        instance.models.simple_note = mock_app_models

        bridge._update_fts_for_inserted_rows([("simple_note", "notes", 42)])

        mock_record.set_search_text.assert_called_once()

    def test_skips_non_searchable_model(
        self, bridge: AppWorldBridge, mock_appworld_module: MagicMock
    ) -> None:
        instance = _init_bridge(bridge, mock_appworld_module)

        mock_find_one = MagicMock()

        class FakeLogModel:
            table_name = "logs"

            @staticmethod
            def searchable() -> bool:
                return False

            find_one = mock_find_one

        mock_app_models = MagicMock()
        mock_app_models.LogModel = FakeLogModel
        type(mock_app_models).__dir__ = MagicMock(  # type: ignore[method-assign]
            return_value=["LogModel"]
        )
        instance.models.myapp = mock_app_models

        bridge._update_fts_for_inserted_rows([("myapp", "logs", 1)])

        mock_find_one.assert_not_called()

    def test_handles_exception_gracefully(
        self,
        bridge: AppWorldBridge,
        mock_appworld_module: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        instance = _init_bridge(bridge, mock_appworld_module)

        class FakeBrokenModel:
            table_name = "broken"

            @staticmethod
            def searchable() -> bool:
                return True

            @staticmethod
            def find_one(id: int) -> None:  # noqa: A002
                raise RuntimeError("DB error")

        mock_app_models = MagicMock()
        mock_app_models.BrokenModel = FakeBrokenModel
        type(mock_app_models).__dir__ = MagicMock(  # type: ignore[method-assign]
            return_value=["BrokenModel"]
        )
        instance.models.myapp = mock_app_models

        # Should not raise
        bridge._update_fts_for_inserted_rows([("myapp", "broken", 1)])

        captured = capsys.readouterr()
        assert "Failed to update FTS" in captured.out


# ============================================================================
# AppWorldNotInstalledError
# ============================================================================


class TestAppWorldNotInstalledError:
    """Tests for the custom import error."""

    def test_is_import_error_subclass(self) -> None:
        assert issubclass(AppWorldNotInstalledError, ImportError)

    def test_message_mentions_install(self) -> None:
        err = AppWorldNotInstalledError()
        assert "not installed" in str(err).lower()
        assert "pip install" in str(err)


# ============================================================================
# _configure_appworld_root
# ============================================================================


class TestConfigureAppworldRoot:
    """Tests for the ``_configure_appworld_root()`` module-level helper."""

    def test_sets_appworld_root_when_sibling_exists(self, tmp_path: Any) -> None:
        """When a sibling appworld/data/tasks dir exists, APPWORLD_ROOT is set."""
        import os

        # Build: tmp_path/repo/mmtoolsandbox/appworld/bridge.py (fake __file__)
        # and:   tmp_path/appworld/data/tasks/
        repo_dir = tmp_path / "repo"
        fake_file = repo_dir / "mmtoolsandbox" / "appworld" / "bridge.py"
        fake_file.parent.mkdir(parents=True)
        fake_file.touch()

        sibling_appworld = tmp_path / "appworld"
        (sibling_appworld / "data" / "tasks").mkdir(parents=True)

        # Patch Path(__file__) resolution chain used in _configure_appworld_root
        # The function does: Path(__file__).parent.parent.parent => repo_root
        # Then: repo_root.parent / "appworld" => sibling check
        from pathlib import Path

        original_path = Path.__new__

        with (
            patch.dict(os.environ, {}, clear=False),
            patch("mmtoolsandbox.appworld.bridge.Path") as mock_path,
        ):
            # Make Path(__file__) work
            mock_file = MagicMock()
            mock_file.parent.parent.parent = repo_dir
            mock_path.return_value = mock_file

            # repo_dir.parent / "appworld" => sibling_appworld
            # This is handled by the real Path object, so we just need
            # the mock chain to return the right thing
            # Actually _configure_appworld_root does:
            #   repo_root = Path(__file__).parent.parent.parent
            #   sibling_appworld = repo_root.parent / "appworld"
            # repo_root is the real tmp_path/repo Path, so .parent / "appworld" works

            with patch.dict("sys.modules", {"appworld.common.path_store": None}):
                _configure_appworld_root()

            assert os.environ.get("APPWORLD_ROOT") == str(sibling_appworld)

        # Clean up env
        os.environ.pop("APPWORLD_ROOT", None)

    def test_does_not_crash_without_appworld_installed(self) -> None:
        """_configure_appworld_root should not crash when appworld is absent."""
        import os

        with (
            patch.dict(os.environ, {}, clear=False),
            patch.dict(
                "sys.modules",
                {
                    "appworld": None,
                    "appworld.common": None,
                    "appworld.common.path_store": None,
                },
            ),
        ):
            try:
                _configure_appworld_root()
            except (ImportError, ModuleNotFoundError):
                pass  # Expected when openai compat patch fails too


# ============================================================================
# load_task_ids
# ============================================================================


class TestLoadTaskIds:
    """Tests for the ``load_task_ids()`` module-level function."""

    def test_raises_when_appworld_not_installed(self) -> None:
        with (
            patch("mmtoolsandbox.appworld.bridge._configure_appworld_root"),
            patch.dict("sys.modules", {"appworld": None}),
        ):
            with pytest.raises(AppWorldNotInstalledError):
                load_task_ids("dev")

    def test_returns_task_ids_from_appworld(self) -> None:
        mock_appworld = MagicMock()
        mock_appworld.load_task_ids.return_value = ["task_001", "task_002"]

        with (
            patch("mmtoolsandbox.appworld.bridge._configure_appworld_root"),
            patch.dict("sys.modules", {"appworld": mock_appworld}),
        ):
            result = load_task_ids("dev")
        assert result == ["task_001", "task_002"]

    def test_passes_dataset_name_to_appworld(self) -> None:
        mock_appworld = MagicMock()
        mock_appworld.load_task_ids.return_value = []

        with (
            patch("mmtoolsandbox.appworld.bridge._configure_appworld_root"),
            patch.dict("sys.modules", {"appworld": mock_appworld}),
        ):
            load_task_ids("test_challenge")
        mock_appworld.load_task_ids.assert_called_once_with("test_challenge")

    def test_default_dataset_is_dev(self) -> None:
        mock_appworld = MagicMock()
        mock_appworld.load_task_ids.return_value = []

        with (
            patch("mmtoolsandbox.appworld.bridge._configure_appworld_root"),
            patch.dict("sys.modules", {"appworld": mock_appworld}),
        ):
            load_task_ids()
        mock_appworld.load_task_ids.assert_called_once_with("dev")


# ============================================================================
# _force_clear_freezegun_state (static method)
# ============================================================================


class TestForceClearFreezegunState:
    """Tests for ``_force_clear_freezegun_state()``."""

    def test_does_not_crash_when_appworld_not_installed(self) -> None:
        """The method should be fully resilient to missing imports."""
        with patch.dict(
            "sys.modules",
            {
                "appworld": None,
                "appworld.environment": None,
                "appworld.requester": None,
                "freezegun": None,
                "freezegun.api": None,
            },
        ):
            # Should not raise
            AppWorldBridge._force_clear_freezegun_state()

    def test_stops_and_clears_time_freezers(self) -> None:
        mock_freezer = MagicMock()
        mock_freezer.undo_changes = []
        mock_appworld_env = MagicMock()
        mock_appworld_env.id_to_time_freezer = {"id1": mock_freezer}

        mock_requester = MagicMock()
        mock_requester.time_freezers_or_ids = []

        with patch.dict(
            "sys.modules",
            {
                "appworld": MagicMock(),
                "appworld.environment": MagicMock(AppWorld=mock_appworld_env),
                "appworld.requester": MagicMock(Requester=mock_requester),
                "freezegun": None,
                "freezegun.api": None,
            },
        ):
            AppWorldBridge._force_clear_freezegun_state()

        mock_freezer.stop.assert_called_once()
        assert mock_appworld_env.id_to_time_freezer == {}

    def test_handles_stop_failure_gracefully(self) -> None:
        mock_freezer = MagicMock()
        mock_freezer.stop.side_effect = IndexError("pop from empty list")
        mock_freezer.undo_changes = []
        mock_appworld_env = MagicMock()
        mock_appworld_env.id_to_time_freezer = {"id1": mock_freezer}

        mock_requester = MagicMock()
        mock_requester.time_freezers_or_ids = []

        with patch.dict(
            "sys.modules",
            {
                "appworld": MagicMock(),
                "appworld.environment": MagicMock(AppWorld=mock_appworld_env),
                "appworld.requester": MagicMock(Requester=mock_requester),
                "freezegun": None,
                "freezegun.api": None,
            },
        ):
            # Should not raise despite stop() failing
            AppWorldBridge._force_clear_freezegun_state()

        # Still cleared
        assert mock_appworld_env.id_to_time_freezer == {}
