"""
Bridge between AppWorld and MMToolSandbox.

This module provides the adapter layer that imports AppWorld components
and makes them available to MMToolSandbox's execution environment.

The bridge handles:
- AppWorld initialization and cleanup
- API call routing
- Task loading and state management
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

# Lazy imports to avoid import errors if appworld is not installed
if TYPE_CHECKING:
    pass


class AppWorldNotInstalledError(ImportError):
    """Raised when AppWorld is not installed or not accessible."""

    def __init__(self) -> None:
        super().__init__(
            "AppWorld is not installed. Install it with:\n"
            "  pip install appworld\n"
            "Or add the appworld repository as a sibling directory."
        )


class AppWorldBridge:
    """
    Bridges AppWorld's execution environment with MMToolSandbox's.

    This singleton class manages the lifecycle of AppWorld instances
    and provides a unified interface for making API calls.

    Usage:
        bridge = get_appworld_bridge()
        bridge.initialize("task_id_123")
        result = bridge.call_api("spotify", "login", email="...", password="...")
        bridge.close()
    """

    _instance: AppWorldBridge | None = None
    _appworld: Any | None = None  # Type: _AppWorld
    _appworld_module: Any | None = None

    def __init__(self) -> None:
        """Initialize the bridge. Use get_appworld_bridge() instead."""
        self._appworld = None
        self._appworld_module = None
        self._requester = None

    @classmethod
    def get_instance(cls) -> AppWorldBridge:
        """Get the singleton instance of AppWorldBridge."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance and all AppWorld state. Useful for testing."""
        if cls._instance is not None:
            cls._instance.close()
            cls._instance = None

        # Force-clear all freezegun tracking state to prevent cascading
        # "pop from empty list" errors from stale state, then do a full
        # close_all to clean up remaining resources (DB caches, etc.).
        cls._force_clear_freezegun_state()

        try:
            from appworld.requester import Requester

            try:
                Requester.close_all()
            except Exception:
                pass
        except ImportError:
            pass

        try:
            from appworld.environment import AppWorld

            try:
                AppWorld.close_all()
            except Exception:
                pass
        except ImportError:
            pass

    def _ensure_appworld_imported(self) -> None:
        """Lazily import AppWorld module."""
        if self._appworld_module is None:
            try:
                import appworld

                self._appworld_module = appworld
            except ImportError as e:
                raise AppWorldNotInstalledError() from e

    @staticmethod
    def _force_clear_freezegun_state() -> None:
        """Force-clear all freezegun tracking state to prevent cascading errors.

        When a freeze_time object is tracked in AppWorld.id_to_time_freezer but
        its underlying freezegun internal state (freeze_factories) is already
        empty, calling stop() raises IndexError("pop from empty list").
        Once this happens, close_all() never reaches clear(), so the stale
        entries persist and every subsequent scenario fails.

        A secondary effect is that datetime.datetime stays monkey-patched to
        FakeDatetime, which causes FastAPI/Pydantic to reject datetime type
        annotations when building the app for the next scenario.

        This method safely stops each tracked freezer (ignoring errors),
        force-clears the tracking structures, and restores datetime if
        freezegun left it monkey-patched — including module-level references
        that freezegun patched via undo_changes.
        """
        # Collect all freezer objects so we can extract undo_changes later
        # if stop() fails.
        all_freezers: list[Any] = []

        try:
            from appworld.environment import AppWorld

            for time_freezer in list(AppWorld.id_to_time_freezer.values()):
                all_freezers.append(time_freezer)
                try:
                    time_freezer.stop()
                except Exception:
                    pass
            AppWorld.id_to_time_freezer.clear()
        except ImportError:
            pass

        try:
            from appworld.requester import Requester

            for tf in list(Requester.time_freezers_or_ids):
                try:
                    if not isinstance(tf, str):
                        all_freezers.append(tf)
                        from appworld.apps.lib.apis.local_remote import (
                            unset_local_date_and_time,
                        )

                        unset_local_date_and_time(tf)
                except Exception:
                    pass
            Requester.time_freezers_or_ids = []
            Requester.time_freezer_id_to_remote_apis_url = {}
        except ImportError:
            pass

        # Restore datetime and clean up freezegun module-level state.
        # When stop() fails mid-way, datetime.datetime stays as FakeDatetime
        # which breaks FastAPI/Pydantic type validation.
        try:
            import datetime

            from freezegun.api import (  # type: ignore[import-not-found,unused-ignore]
                freeze_factories,
                ignore_lists,
                real_date,
                real_datetime,
                tick_flags,
                tz_offsets,
            )

            if freeze_factories or datetime.datetime is not real_datetime:
                datetime.datetime = real_datetime  # type: ignore[misc]
                datetime.date = real_date  # type: ignore[misc]
                freeze_factories.clear()
                ignore_lists.clear()
                tick_flags.clear()
                tz_offsets.clear()

            # Apply undo_changes from any freezer whose stop() failed.
            # These restore module-level datetime references (e.g.
            # appworld.apps.datetime) that freezegun patched during start().
            for freezer in all_freezers:
                for change in getattr(freezer, "undo_changes", []):
                    try:
                        module_or_object, attribute, original_value = change
                        setattr(module_or_object, attribute, original_value)
                    except Exception:
                        pass
                freezer.undo_changes = []
        except ImportError:
            pass

    def initialize(
        self,
        task_id: str,
        experiment_name: str = "mmtoolsandbox",
        ground_truth_mode: str = "minimal",
        max_api_calls_per_interaction: int = 1000,  # AppWorld default
    ) -> AppWorldBridge:
        """
        Initialize AppWorld for a specific task.

        Args:
            task_id: The AppWorld task ID to load
            experiment_name: Name for the experiment (for logging/output)
            ground_truth_mode: Ground truth loading mode ("none", "full", etc.)
            max_api_calls_per_interaction: Maximum API calls per interaction (default: 1000, AppWorld default)

        Returns:
            self for method chaining

        Raises:
            AppWorldNotInstalledError: If AppWorld is not installed
        """
        _configure_appworld_root()
        self._ensure_appworld_imported()

        # Close any existing instance
        if self._appworld is not None:
            self.close()

        # Force-clear stale freezegun state before creating a new AppWorld.
        # AppWorld.__init__ calls close_all() which will fail with "pop from
        # empty list" if id_to_time_freezer has entries whose freezegun
        # internal state is already cleaned up.
        self._force_clear_freezegun_state()

        # Initialize new AppWorld instance
        assert self._appworld_module is not None  # Set by _ensure_appworld_imported
        self._appworld = self._appworld_module.AppWorld(
            task_id=task_id,
            experiment_name=experiment_name,
            ground_truth_mode=ground_truth_mode,
            max_api_calls_per_interaction=max_api_calls_per_interaction,
        )

        return self

    @property
    def is_initialized(self) -> bool:
        """Check if AppWorld is initialized."""
        return self._appworld is not None

    @property
    def requester(self) -> Any:  # Type: Requester
        """
        Get the AppWorld requester for making API calls.

        Returns:
            The Requester instance for HTTP-like API calls

        Raises:
            RuntimeError: If AppWorld is not initialized
        """
        if self._appworld is None:
            raise RuntimeError(
                "AppWorld not initialized. Call initialize(task_id) first."
            )
        return self._appworld.requester

    @property
    def task(self) -> Any:  # Type: AppWorldTask
        """
        Get the current AppWorld task.

        Returns:
            The Task instance with instruction, supervisor, etc.

        Raises:
            RuntimeError: If AppWorld is not initialized
        """
        if self._appworld is None:
            raise RuntimeError(
                "AppWorld not initialized. Call initialize(task_id) first."
            )
        return self._appworld.task

    @property
    def appworld(self) -> Any:  # Type: _AppWorld
        """
        Get the raw AppWorld instance.

        Returns:
            The AppWorld instance

        Raises:
            RuntimeError: If AppWorld is not initialized
        """
        if self._appworld is None:
            raise RuntimeError(
                "AppWorld not initialized. Call initialize(task_id) first."
            )
        return self._appworld

    def call_api(
        self,
        app: str,
        endpoint: str,
        method: str = "post",
        **params: Any,
    ) -> Any:
        """
        Call an AppWorld API endpoint.

        Args:
            app: The app name (e.g., "spotify", "amazon")
            endpoint: The API name (e.g., "login", "search_songs")
            method: HTTP method (ignored - the requester.request() method looks this up)
            **params: Parameters to pass to the API

        Returns:
            The parsed JSON response from the API (dict or list)

        Raises:
            RuntimeError: If AppWorld is not initialized
        """
        if self._appworld is None:
            raise RuntimeError(
                "AppWorld not initialized. Call initialize(task_id) first."
            )

        # Use requester.request() which properly looks up the API path and method
        # from the API docs, rather than constructing URLs directly
        requester = self.requester
        return requester.request(_app_name=app, _api_name=endpoint, **params)

    def execute(self, code: str) -> Any:
        """
        Execute Python code in AppWorld's environment.

        This is a passthrough to AppWorld.execute() for running
        arbitrary code with access to the `apis` object.

        Args:
            code: Python code to execute

        Returns:
            The result from AppWorld.execute()

        Raises:
            RuntimeError: If AppWorld is not initialized
        """
        if self._appworld is None:
            raise RuntimeError(
                "AppWorld not initialized. Call initialize(task_id) first."
            )
        return self._appworld.execute(code)

    def apply_database_modifications(
        self,
        modifications: list[tuple[str, str, list[Any]]],
    ) -> None:
        """
        Apply custom database modifications for a scenario.

        This allows MMToolSandbox scenarios to customize the AppWorld
        database state beyond what the base task provides.

        After all SQL statements are executed, any rows inserted into
        searchable tables have their FTS index entries updated so that
        they are discoverable via AppWorld search APIs.

        Args:
            modifications: List of (app_name, sql_statement, parameters) tuples

        Example:
            bridge.apply_database_modifications([
                # Add a custom contact
                ("admin", "INSERT INTO users (id, name, email) VALUES (?, ?, ?)",
                 [9999, "Custom User", "custom@example.com"]),
                # Modify a playlist
                ("spotify", "UPDATE playlists SET name = ? WHERE id = ?",
                 ["Agent Test Playlist", 1]),
                # Add money to Venmo balance
                ("venmo", "UPDATE accounts SET balance = ? WHERE user_id = ?",
                 [500.00, 1]),
            ])

        Raises:
            RuntimeError: If AppWorld is not initialized
        """
        if self._appworld is None:
            raise RuntimeError(
                "AppWorld not initialized. Call initialize(task_id) first."
            )

        # Access the model collection to get database connections
        models = self._appworld.models

        from sqlalchemy import text

        # Track inserted rows so we can update FTS indices afterwards.
        # Each entry is (app_name, table_name, record_id).
        inserted_rows: list[tuple[str, str, Any]] = []

        staging_errors: list[str] = []

        for app_name, sql_statement, parameters in modifications:
            try:
                # Get the app's model collection (Munch of model classes + SQLModel)
                if hasattr(models, app_name):
                    app_models = getattr(models, app_name)
                    # Access engine via SQLModel.db.engine (not app_models.db directly)
                    sql_model = getattr(app_models, "SQLModel", None)
                    if (
                        sql_model is not None
                        and hasattr(sql_model, "db")
                        and sql_model.db.engine is not None
                    ):
                        engine = sql_model.db.engine
                        # Convert ? placeholders to :p0, :p1, ... for SQLAlchemy text()
                        parts = sql_statement.split("?")
                        named_sql = parts[0]
                        param_dict = {}
                        for i in range(len(parts) - 1):
                            named_sql += f":p{i}" + parts[i + 1]
                            param_dict[f"p{i}"] = parameters[i]
                        with engine.connect() as conn:
                            conn.execute(text(named_sql), param_dict)
                            conn.commit()

                        # If this is an INSERT, track for FTS update
                        sql_upper = sql_statement.strip().upper()
                        if sql_upper.startswith("INSERT"):
                            table_name, record_id = self._extract_insert_info(
                                sql_statement, parameters
                            )
                            if table_name and record_id is not None:
                                inserted_rows.append((app_name, table_name, record_id))
                    else:
                        staging_errors.append(
                            f"No database engine found for app '{app_name}'"
                        )
                else:
                    staging_errors.append(f"App '{app_name}' not found in models")
            except Exception as e:
                staging_errors.append(
                    f"Failed to apply modification to {app_name}: {e}\n"
                    f"  SQL: {sql_statement}\n"
                    f"  Params: {parameters}"
                )

        # Update FTS indices for inserted rows in searchable tables
        if inserted_rows:
            self._update_fts_for_inserted_rows(inserted_rows)

        if staging_errors:
            raise RuntimeError(
                f"Entity staging failed with {len(staging_errors)} error(s):\n"
                + "\n".join(f"  - {e}" for e in staging_errors)
            )

    @staticmethod
    def _extract_insert_info(
        sql_statement: str, parameters: list[Any]
    ) -> tuple[str | None, Any]:
        """Extract the table name and id value from an INSERT statement.

        Returns (table_name, id_value) or (None, None) if not parseable.
        """
        import re

        # Match: INSERT [OR REPLACE] INTO <table> (<cols>) VALUES (...)
        m = re.match(
            r"INSERT\s+(?:OR\s+\w+\s+)?INTO\s+(\w+)\s*\(([^)]+)\)",
            sql_statement,
            re.IGNORECASE,
        )
        if not m:
            return None, None

        table_name = m.group(1)
        columns = [c.strip() for c in m.group(2).split(",")]

        if "id" in columns:
            id_index = columns.index("id")
            if id_index < len(parameters):
                return table_name, parameters[id_index]

        return table_name, None

    def _update_fts_for_inserted_rows(
        self, inserted_rows: list[tuple[str, str, Any]]
    ) -> None:
        """Update FTS search text for newly inserted rows.

        For each inserted row, finds the corresponding ORM model class,
        loads the record, and calls set_search_text() to populate the
        FTS virtual table so that search APIs can find the record.
        """
        if self._appworld is None:
            return
        models = self._appworld.models

        for app_name, table_name, record_id in inserted_rows:
            try:
                app_models = getattr(models, app_name, None)
                if app_models is None:
                    continue

                # Find the model class whose table_name matches
                model_cls = None
                for attr_name in dir(app_models):
                    attr = getattr(app_models, attr_name, None)
                    if (
                        attr is not None
                        and isinstance(attr, type)
                        and hasattr(attr, "table_name")
                        and hasattr(attr, "searchable")
                        and getattr(attr, "table_name", None) == table_name
                    ):
                        model_cls = attr
                        break

                if model_cls is None or not model_cls.searchable():
                    continue

                record = model_cls.find_one(id=record_id)
                if record is not None:
                    record.set_search_text()
            except Exception as e:
                print(
                    f"Warning: Failed to update FTS for "
                    f"{app_name}.{table_name} id={record_id}: {e}"
                )

    def run_setup_code(self, setup_code: str) -> Any:
        """
        Run setup code in AppWorld's environment before the scenario starts.

        This is useful for:
        - Creating custom users or data
        - Setting up specific app states
        - Preparing test fixtures

        Args:
            setup_code: Python code to execute for setup

        Returns:
            The result from execution

        Example:
            bridge.run_setup_code('''
                # Create a custom playlist
                apis.spotify.login(email="user@example.com", password="pass123")
                apis.spotify.create_playlist(name="Test Playlist", is_public=True)
            ''')
        """
        if self._appworld is None:
            raise RuntimeError(
                "AppWorld not initialized. Call initialize(task_id) first."
            )
        return self._appworld.execute(setup_code)

    def evaluate(self) -> Any:
        """
        Run AppWorld's evaluation on the current task.

        Returns:
            Evaluation result with TGC score, passed/failed tests, etc.

        Raises:
            RuntimeError: If AppWorld is not initialized
        """
        if self._appworld is None:
            raise RuntimeError(
                "AppWorld not initialized. Call initialize(task_id) first."
            )
        return self._appworld.evaluate()

    def close(self) -> None:
        """Cleanup AppWorld resources."""
        if self._appworld is not None:
            try:
                self._appworld.close()
            except Exception:
                pass  # Ignore cleanup errors
            self._appworld = None

    def __enter__(self) -> AppWorldBridge:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Context manager exit - cleanup resources."""
        self.close()


def get_appworld_bridge() -> AppWorldBridge:
    """
    Get the singleton AppWorldBridge instance.

    Returns:
        The shared AppWorldBridge instance
    """
    return AppWorldBridge.get_instance()


def _configure_appworld_root() -> None:
    """
    Configure APPWORLD_ROOT environment variable for AppWorld.

    Looks for the appworld data directory in common locations:
    1. Sibling 'appworld' directory (relative to MMToolSandbox)
    2. User-specified APPWORLD_ROOT environment variable
    3. Current working directory (fallback)

    Also updates the path_store singleton if it was already created.
    """
    import os

    # Fix OpenAI SDK 1.99+ compatibility issue where ChatCompletionMessageToolCall
    # became a Union type instead of a class. Patch it to use the actual class.
    try:
        import openai.types.chat.chat_completion_message_tool_call as tool_call_module
        from openai.types.chat.chat_completion_message_function_tool_call import (
            ChatCompletionMessageFunctionToolCall,
        )

        # Patch the module so MMToolSandbox imports get the actual class
        tool_call_module.ChatCompletionMessageToolCall = (
            ChatCompletionMessageFunctionToolCall  # type: ignore[assignment,unused-ignore]
        )
    except (ImportError, AttributeError):
        pass  # Older OpenAI version, no patch needed

    # Look for sibling appworld directory
    # ml-mmtoolsandbox and appworld are expected as sibling directories.
    repo_root = Path(__file__).parent.parent.parent  # mmtoolsandbox package root
    sibling_appworld = repo_root.parent / "appworld"

    appworld_root = None
    if sibling_appworld.exists() and (sibling_appworld / "data" / "tasks").exists():
        appworld_root = str(sibling_appworld)
    elif (repo_root / "data" / "tasks").exists():
        appworld_root = str(repo_root)

    if appworld_root:
        os.environ["APPWORLD_ROOT"] = appworld_root

        # Also update path_store singleton if it was already created
        try:
            from appworld.common.path_store import path_store

            path_store.update_root(appworld_root)
        except ImportError:
            pass  # AppWorld not installed


def load_task_ids(dataset: str = "dev") -> list[str]:
    """
    Load task IDs from an AppWorld dataset.

    Args:
        dataset: Dataset name ("train", "dev", "test_normal", "test_challenge")

    Returns:
        List of task ID strings

    Raises:
        AppWorldNotInstalledError: If AppWorld is not installed
    """
    _configure_appworld_root()

    try:
        from appworld import load_task_ids as _load_task_ids

        result: list[str] = _load_task_ids(dataset)
        return result
    except ImportError as e:
        raise AppWorldNotInstalledError() from e
