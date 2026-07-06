"""AppWorld per-scenario runtime lifecycle for MMToolSandbox.

Runs in a spawned worker process.  Handles:
1. AppWorld bridge lifecycle (init, SQL mods, supervisor, time freeze, auto-login)
2. Expired payment card fixes
3. Entity diff evaluator initial-state capture

This is a top-level function (not a closure) so it can be pickled for
multiprocessing.
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any, Iterator

from mmtoolsandbox.common.scenario import Scenario

LOGGER = logging.getLogger(__name__)

_APPS_WITH_LOGIN: frozenset[str] = frozenset(
    {
        "amazon",
        "spotify",
        "gmail",
        "venmo",
        "todoist",
        "splitwise",
        "phone",
        "file_system",
        "simple_note",
    }
)


@contextlib.contextmanager
def appworld_runtime_context(
    scenario: Scenario,
    output_directory: Path,
) -> Iterator[None]:
    """Context manager that sets up and tears down AppWorld for a single scenario.

    Runs in a spawned worker process.  The scenario's toolbox is already
    fully populated (AppWorld + native tools) at load time via
    ``register_appworld_tools_to_core_registry()``, so no tool registration
    or merging is needed here.

    Reads configuration from ``scenario.runtime_metadata``.
    """
    import datetime as _dt_mod

    # Silence noisy loggers in spawned worker processes
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("httpcore").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.ERROR)

    # --- AppWorld bridge lifecycle ---
    from mmtoolsandbox.appworld.bridge import AppWorldBridge
    from mmtoolsandbox.appworld.scenarios.runner import get_scenario_runner
    from mmtoolsandbox.appworld.scenarios.task_adapters import (
        convert_appworld_entities_to_sql,
    )
    from mmtoolsandbox.appworld.state import AppWorldState

    rm = scenario.runtime_metadata or {}

    AppWorldBridge.reset()
    AppWorldState.reset()

    appworld_entities = rm.get("appworld_entities", {})
    db_mods = (
        convert_appworld_entities_to_sql(appworld_entities)
        if appworld_entities
        else None
    )

    # Clear the Amazon cart so pre-existing items don't interfere with
    # scenario evaluation.  No scenario stages cart_entries, so this is
    # always safe.
    _cart_clear: list[tuple[str, str, list[object]]] = [
        ("amazon", "DELETE FROM cart_entries", [])
    ]
    if db_mods is None:
        db_mods = _cart_clear
    else:
        db_mods = _cart_clear + db_mods

    base_task_id = rm.get("appworld_base_task", "_base_")
    runner = get_scenario_runner()

    # Redirect AppWorld's disk output to a per-scenario subdirectory
    scenario_id = rm.get("original_scenario_id", f"unknown_{id(scenario)}")
    appworld_output_path = output_directory / "outputs" / scenario_id

    try:
        from appworld.common.path_store import PathStore

        _original_experiment_outputs = PathStore.experiment_outputs.fget  # type: ignore[attr-defined,unused-ignore]

        @property  # type: ignore[misc,unused-ignore]
        def _redirected_experiment_outputs(self):  # type: ignore[no-untyped-def,unused-ignore]
            return str(appworld_output_path)

        PathStore.experiment_outputs = _redirected_experiment_outputs  # type: ignore[method-assign,unused-ignore]
    except ImportError:
        _original_experiment_outputs = None

    try:
        with runner.run_scenario(
            base_task_id=base_task_id,
            database_modifications=db_mods,
            execution_context=scenario.starting_context,
        ) as bridge:
            # Freeze time to scenario's reference_time
            scenario_time = scenario.reference_time or rm.get("reference_time")
            _time_freezer = None
            if bridge and bridge.is_initialized and scenario_time:
                try:
                    from appworld.apps.lib.apis.local_remote import (
                        set_local_date_and_time,
                    )

                    dt = _dt_mod.datetime.fromisoformat(
                        scenario_time.replace("Z", "+00:00")
                    )
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=_dt_mod.timezone.utc)
                    _time_freezer = set_local_date_and_time(dt)
                    LOGGER.info("AppWorld time set to %s", dt)
                except Exception as e:
                    LOGGER.warning("Failed to set AppWorld time: %s", e)

            try:
                # Set up supervisor from device_state_id
                device_state_id = rm.get("device_state_id")
                if bridge and bridge.is_initialized and device_state_id:
                    escaped_email = json.dumps(device_state_id)
                    escaped_instruction = json.dumps(rm.get("instruction", ""))
                    setup_code = (
                        "from appworld.apps.admin.models import MainUser\n"
                        "from appworld.apps.supervisor.models import Supervisor\n"
                        f"main_user = MainUser.find_one(email={escaped_email})\n"
                        "if main_user:\n"
                        f"    Supervisor.reset_from_main_user(main_user.id, task_instruction={escaped_instruction})\n"
                        "    print(f'Supervisor set up for {main_user.first_name} {main_user.last_name} ({main_user.email})')\n"
                        "else:\n"
                        f"    print(f'Warning: No MainUser found for email {escaped_email}')\n"
                    )
                    try:
                        bridge.run_setup_code(setup_code)
                    except Exception as e:
                        LOGGER.warning("Failed to set up supervisor: %s", e)

                # Auto-login to relevant apps if requested
                if rm.get("auto_login") and bridge and bridge.is_initialized:
                    _auto_login_apps(bridge, rm)

                # Fix expired payment cards
                if bridge and bridge.is_initialized and scenario_time:
                    _fix_expired_payment_cards(bridge, scenario_time)

                # Log AppWorld entity staging verification
                if bridge and bridge.is_initialized and appworld_entities:
                    _log_appworld_entity_staging(bridge, appworld_entities)

                # Capture initial state for entity diff evaluation
                entity_diff_specs = rm.get("entity_diff_specs")
                if entity_diff_specs and bridge and bridge.is_initialized:
                    from mmtoolsandbox.common.entity_diff_evaluator import (
                        EntityDiffEvalConfig,
                        EntityDiffEvaluator,
                    )

                    config = EntityDiffEvalConfig.from_dict(
                        {"specs": entity_diff_specs}
                    )
                    evaluator = EntityDiffEvaluator(config)
                    evaluator.capture_initial_state(
                        execution_context=scenario.starting_context,
                        bridge=bridge,
                    )
                    scenario.evaluation_criteria.entity_diff_evaluator = evaluator

                yield
            finally:
                if _time_freezer is not None:
                    try:
                        from appworld.apps.lib.apis.local_remote import (
                            unset_local_date_and_time,
                        )

                        unset_local_date_and_time(_time_freezer)
                    except Exception:
                        pass
    finally:
        if _original_experiment_outputs is not None:
            PathStore.experiment_outputs = property(_original_experiment_outputs)  # type: ignore[assignment,unused-ignore]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fix_expired_payment_cards(
    bridge: Any,
    scenario_time: str,
) -> None:
    """Update expired Amazon payment cards so they're valid at scenario time.

    The AppWorld base DB has payment cards with expiry dates in 2022-2024,
    but scenarios run in 2026+.  This updates any expired cards to expire
    2 years after the scenario's reference time.
    """
    import datetime as _dt

    try:
        dt = _dt.datetime.fromisoformat(scenario_time.replace("Z", "+00:00"))
        target_year = dt.year + 2

        bridge.apply_database_modifications(
            [
                (
                    "amazon",
                    f"UPDATE payment_cards SET expiry_year = {target_year} "
                    f"WHERE expiry_year < {dt.year}",
                    [],
                ),
                (
                    "admin",
                    f"UPDATE payment_cards SET expiry_year = {target_year} "
                    f"WHERE expiry_year < {dt.year}",
                    [],
                ),
            ]
        )
    except Exception as e:
        LOGGER.warning("Failed to fix expired payment cards: %s", e)


def _auto_login_apps(
    bridge: Any,
    metadata: dict[str, Any],
) -> None:
    """Auto-login to all apps that require authentication."""
    from mmtoolsandbox.appworld.state import AppWorldState

    profile = bridge.call_api("supervisor", "show_profile", method="get")
    email = profile.get("email")
    phone_number = profile.get("phone_number")
    if not email:
        return

    credentials = bridge.call_api("supervisor", "show_account_passwords", method="get")
    password_by_app = {c["account_name"]: c["password"] for c in credentials}

    state = AppWorldState.get_instance()
    for app in _APPS_WITH_LOGIN:
        password = password_by_app.get(app)
        if not password:
            continue
        username = phone_number if app == "phone" else email
        if not username:
            continue
        try:
            response = bridge.call_api(
                app,
                "login",
                method="post",
                username=username,
                password=password,
            )
            if isinstance(response, dict) and "access_token" in response:
                state.login(app, response["access_token"], response.get("user_id"))
                LOGGER.info("Auto-login: %s OK", app)
        except Exception as e:
            LOGGER.warning("Auto-login: %s failed: %s", app, e)


def _log_appworld_entity_staging(
    bridge: Any,
    appworld_entities: dict[str, list[dict[str, Any]]],
) -> None:
    """Log verification of staged AppWorld entities."""
    from sqlalchemy import text as sa_text

    LOGGER.debug("--- AppWorld DB Verification (staged entities) ---")
    for table_key, entities in appworld_entities.items():
        app_name, table_name = table_key.split(".", 1)
        staged_ids = [e.get("id") for e in entities if e.get("id") is not None]
        try:
            if bridge._appworld is None:
                continue
            models = bridge._appworld.models
            app_models = getattr(models, app_name, None)
            if app_models is None:
                LOGGER.debug("  %s: app '%s' not found in models", table_key, app_name)
                continue
            sql_model = getattr(app_models, "SQLModel", None)
            if (
                sql_model is None
                or not hasattr(sql_model, "db")
                or sql_model.db.engine is None
            ):
                LOGGER.debug(
                    "  %s: no database engine for app '%s'", table_key, app_name
                )
                continue
            engine = sql_model.db.engine
            with engine.connect() as conn:
                total = conn.execute(
                    sa_text(f"SELECT COUNT(*) FROM {table_name}")
                ).scalar()
                if staged_ids:
                    placeholders = ",".join(str(i) for i in staged_ids)
                    rows = conn.execute(
                        sa_text(
                            f"SELECT * FROM {table_name} WHERE id IN ({placeholders})"
                        )
                    ).fetchall()
                else:
                    rows = []
                LOGGER.debug(
                    "  %s: %d total rows, %d staged", table_key, total, len(rows)
                )
                for row in rows:
                    LOGGER.debug("    %s", dict(row._mapping))
        except Exception as e:
            LOGGER.debug("  %s: query failed: %s", table_key, e)
    LOGGER.debug("--- End DB Verification ---")
