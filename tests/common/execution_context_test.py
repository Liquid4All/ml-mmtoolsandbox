# Copyright © 2026 Apple Inc.

"""Unit tests for mmtoolsandbox.common.execution_context"""

import code
import copy
import datetime
import queue
import random
import threading
from typing import Any, Callable, Optional, cast
from unittest.mock import MagicMock

import polars as pl
import pytest

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution_context import (
    MAX_ACTIVE_TOOLS,
    ExecutionContext,
    RoleType,
    get_current_context,
    get_locale,
    new_context,
    new_context_with_attribute,
    set_current_context,
)
from mmtoolsandbox.common.i18n import Locale
from mmtoolsandbox.common.introspection_databases import IntrospectionDatabaseNamespace
from mmtoolsandbox.common.relative_time import realign_timestamp
from mmtoolsandbox.toolbox.loading import load_toolbox
from mmtoolsandbox.toolbox.names import ToolboxName
from mmtoolsandbox.toolbox.state import ToolState, ToolStateType
from mmtoolsandbox.toolbox.tool_state_registry import ToolStateRegistry
from mmtoolsandbox.toolbox.toolbox import (
    Toolbox,
    ToolboxConfig,
    ToolStateRegistryFactory,
)


@pytest.fixture(params=[Locale.en_US])
def locale(request: pytest.FixtureRequest) -> Locale:
    return cast(Locale, request.param)


@pytest.fixture
def default_execution_context() -> ExecutionContext:
    """Default init ExecutionContext

    Returns:
        A default init ExecutionContext object
    """
    return ExecutionContext()


@pytest.fixture
def populated_execution_context(locale: Locale) -> ExecutionContext:
    """Execution context with a few entries populated in SETTINGS database and SANDBOX database

    Returns:
        A default init ExecutionContext object
    """
    toolbox = load_toolbox(ToolboxName.FULL, config={})
    test_context = ExecutionContext(toolbox=toolbox, delay_initialization=False)
    test_context.get_tool_state_registry().add(ToolStateType.UTILITIES, MockToolState())
    # Add 3 entries into SANDBOX, remember there's a headguard for each snapshot
    test_context._dbs[DatabaseNamespace.SANDBOX] = pl.DataFrame(
        [
            {
                "sandbox_message_index": 0,
                "sender": None,
                "recipient": None,
                "content": None,
                "conversation_active": None,
                "openai_tool_call_id": None,
                "openai_function_name": None,
                "tool_call_exception": None,
                "tool_trace": None,
                "visible_to": None,
                "finish_reason": None,
                "logprobs": None,
                "generation": None,
                "token_ids": None,
                "claude_text_response": None,
                "claude_extended_thinking": None,
                "claude_extended_thinking_signature": None,
                "tool_call_text_response": None,
                "image_ids": None,
                "reasoning_trace": None,
                "openai_reasoning_items": None,
            },
            {
                "sandbox_message_index": 0,
                "sender": RoleType.SYSTEM,
                "recipient": RoleType.AGENT,
                "content": "Be Good",
                "conversation_active": True,
                "openai_tool_call_id": None,
                "openai_function_name": None,
                "tool_call_exception": None,
                "tool_trace": None,
                "visible_to": None,
                "finish_reason": None,
                "logprobs": None,
                "generation": None,
                "claude_text_response": None,
                "claude_extended_thinking": None,
                "claude_extended_thinking_signature": None,
                "tool_call_text_response": None,
                "image_ids": None,
                "reasoning_trace": None,
                "openai_reasoning_items": None,
            },
            {
                "sandbox_message_index": 1,
                "sender": None,
                "recipient": None,
                "content": None,
                "conversation_active": None,
                "openai_tool_call_id": None,
                "openai_function_name": None,
                "tool_call_exception": None,
                "tool_trace": None,
                "visible_to": None,
                "finish_reason": None,
                "logprobs": None,
                "generation": None,
                "token_ids": None,
                "claude_text_response": None,
                "claude_extended_thinking": None,
                "claude_extended_thinking_signature": None,
                "tool_call_text_response": None,
                "image_ids": None,
                "reasoning_trace": None,
                "openai_reasoning_items": None,
            },
            {
                "sandbox_message_index": 1,
                "sender": RoleType.SYSTEM,
                "recipient": RoleType.USER,
                "content": "Be Good",
                "conversation_active": True,
                "openai_tool_call_id": None,
                "openai_function_name": None,
                "tool_call_exception": None,
                "tool_trace": None,
                "visible_to": None,
                "finish_reason": None,
                "logprobs": None,
                "generation": None,
                "token_ids": None,
                "claude_text_response": None,
                "claude_extended_thinking": None,
                "claude_extended_thinking_signature": None,
                "tool_call_text_response": None,
                "image_ids": None,
                "reasoning_trace": None,
                "openai_reasoning_items": None,
            },
            {
                "sandbox_message_index": 2,
                "sender": None,
                "recipient": None,
                "content": None,
                "conversation_active": None,
                "openai_tool_call_id": None,
                "openai_function_name": None,
                "tool_call_exception": None,
                "tool_trace": None,
                "visible_to": None,
                "finish_reason": None,
                "logprobs": None,
                "generation": None,
                "token_ids": None,
                "claude_text_response": None,
                "claude_extended_thinking": None,
                "claude_extended_thinking_signature": None,
                "tool_call_text_response": None,
                "image_ids": None,
                "reasoning_trace": None,
                "openai_reasoning_items": None,
            },
            {
                "sandbox_message_index": 2,
                "sender": RoleType.SYSTEM,
                "recipient": RoleType.EXECUTION_ENVIRONMENT,
                "content": "import json\n",
                "conversation_active": True,
                "openai_tool_call_id": None,
                "openai_function_name": None,
                "tool_call_exception": None,
                "tool_trace": None,
                "visible_to": None,
                "finish_reason": None,
                "logprobs": None,
                "generation": None,
                "token_ids": None,
                "claude_text_response": None,
                "claude_extended_thinking": None,
                "claude_extended_thinking_signature": None,
                "tool_call_text_response": None,
                "image_ids": None,
                "reasoning_trace": None,
                "openai_reasoning_items": None,
            },
            {
                "sandbox_message_index": 3,
                "sender": None,
                "recipient": None,
                "content": None,
                "conversation_active": None,
                "openai_tool_call_id": None,
                "openai_function_name": None,
                "tool_call_exception": None,
                "tool_trace": None,
                "visible_to": None,
                "finish_reason": None,
                "logprobs": None,
                "generation": None,
                "token_ids": None,
                "claude_text_response": None,
                "claude_extended_thinking": None,
                "claude_extended_thinking_signature": None,
                "tool_call_text_response": None,
                "image_ids": None,
                "reasoning_trace": None,
                "openai_reasoning_items": None,
            },
            {
                "sandbox_message_index": 3,
                "sender": RoleType.USER,
                "recipient": RoleType.AGENT,
                "content": "Hi",
                "conversation_active": True,
                "openai_tool_call_id": None,
                "openai_function_name": None,
                "tool_call_exception": None,
                "tool_trace": None,
                "visible_to": None,
                "finish_reason": None,
                "logprobs": None,
                "generation": None,
                "token_ids": None,
                "claude_text_response": None,
                "claude_extended_thinking": None,
                "claude_extended_thinking_signature": None,
                "tool_call_text_response": None,
                "image_ids": None,
                "reasoning_trace": None,
                "openai_reasoning_items": None,
            },
        ],
        schema=ExecutionContext.dbs_schemas[DatabaseNamespace.SANDBOX],
    )

    test_context._dbs[DatabaseNamespace.SETTING] = pl.DataFrame(
        [
            {
                "sandbox_message_index": 0,
                "device_id": None,
                "cellular": None,
                "wifi": None,
                "location_service": None,
                "low_battery_mode": None,
                "locale": None,
                "latitude": None,
                "longitude": None,
                "utc_offset_seconds": None,
            },
            {
                "sandbox_message_index": 1,
                "device_id": None,
                "cellular": None,
                "wifi": None,
                "location_service": None,
                "low_battery_mode": None,
                "locale": None,
                "latitude": None,
                "longitude": None,
                "utc_offset_seconds": None,
            },
            {
                "sandbox_message_index": 1,
                "device_id": "1",
                "cellular": True,
                "wifi": False,
                "location_service": True,
                "low_battery_mode": False,
                "locale": locale.name,
                "latitude": 0,
                "longitude": 0,
                "utc_offset_seconds": None,
            },
            {
                "sandbox_message_index": 3,
                "device_id": None,
                "cellular": None,
                "wifi": None,
                "location_service": None,
                "low_battery_mode": None,
                "locale": None,
                "latitude": None,
                "longitude": None,
                "utc_offset_seconds": None,
            },
            {
                "sandbox_message_index": 3,
                "device_id": "1",
                "cellular": False,
                "wifi": False,
                "location_service": True,
                "low_battery_mode": False,
                "locale": locale.name,
                "latitude": 0,
                "longitude": 0,
                "utc_offset_seconds": None,
            },
        ],
        schema=ExecutionContext.dbs_schemas[DatabaseNamespace.SETTING],
    )

    # Add introspection data so that we exercise the logic in the serialization
    # roundtrip test.
    test_context.add_to_introspection_database(
        IntrospectionDatabaseNamespace.MISC,
        rows=[
            {"description": "LLM inference runtime"},
            {"content": "13.52 ms"},
        ],
    )

    # Add timestamp related database entry to test timestamp shifting logic
    test_context._dbs[DatabaseNamespace.REMINDER] = pl.DataFrame(
        [
            {
                "sandbox_message_index": 0,
                "reminder_id": "1",
                "content": "Test",
                "creation_datetime": datetime.datetime(
                    year=2025, month=1, day=1, tzinfo=datetime.timezone.utc
                ).isoformat(),
                "reminder_datetime": datetime.datetime(
                    year=2025, month=1, day=2, tzinfo=datetime.timezone.utc
                ).isoformat(),
            },
            {
                "sandbox_message_index": 0,
                "reminder_id": "2",
                "content": "Test",
                "creation_datetime": datetime.datetime(
                    year=2025, month=1, day=2, tzinfo=datetime.timezone.utc
                ).isoformat(),
            },
        ],
        schema=ExecutionContext.dbs_schemas[DatabaseNamespace.REMINDER],
    )

    # Add other attributes
    test_context.tool_allow_list = ["end_conversation"]
    test_context.tool_deny_list = ["get_current_location"]
    test_context.trace_tool = False
    command = code.compile_command("a=1", symbol="exec")
    assert command is not None
    test_context.interactive_console.runcode(command)

    return test_context


@pytest.fixture
def headguard_execution_context(locale: Locale) -> ExecutionContext:
    """Execution context with a only a headguard populated in Reminder database

    Returns:
        A default init ExecutionContext object
    """
    toolbox = load_toolbox(ToolboxName.FULL, config={})
    test_context = ExecutionContext(toolbox=toolbox, delay_initialization=False)
    test_context.get_tool_state_registry().add(ToolStateType.UTILITIES, MockToolState())
    # Add 3 entries into SANDBOX, remember there's a headguard for each snapshot
    test_context._dbs[DatabaseNamespace.SANDBOX] = pl.DataFrame(
        [
            {
                "sandbox_message_index": 0,
                "sender": None,
                "recipient": None,
                "content": None,
                "conversation_active": None,
                "openai_tool_call_id": None,
                "openai_function_name": None,
                "tool_call_exception": None,
                "tool_trace": None,
                "visible_to": None,
                "image_ids": None,
                "reasoning_trace": None,
                "openai_reasoning_items": None,
            },
            {
                "sandbox_message_index": 0,
                "sender": RoleType.SYSTEM,
                "recipient": RoleType.AGENT,
                "content": "Be Good",
                "conversation_active": True,
                "openai_tool_call_id": None,
                "openai_function_name": None,
                "tool_call_exception": None,
                "tool_trace": None,
                "visible_to": None,
                "image_ids": None,
                "reasoning_trace": None,
                "openai_reasoning_items": None,
            },
            {
                "sandbox_message_index": 1,
                "sender": None,
                "recipient": None,
                "content": None,
                "conversation_active": None,
                "openai_tool_call_id": None,
                "openai_function_name": None,
                "tool_call_exception": None,
                "tool_trace": None,
                "visible_to": None,
                "image_ids": None,
                "reasoning_trace": None,
                "openai_reasoning_items": None,
            },
            {
                "sandbox_message_index": 1,
                "sender": RoleType.SYSTEM,
                "recipient": RoleType.USER,
                "content": "Be Good",
                "conversation_active": True,
                "openai_tool_call_id": None,
                "openai_function_name": None,
                "tool_call_exception": None,
                "tool_trace": None,
                "visible_to": None,
                "image_ids": None,
                "reasoning_trace": None,
                "openai_reasoning_items": None,
            },
            {
                "sandbox_message_index": 2,
                "sender": None,
                "recipient": None,
                "content": None,
                "conversation_active": None,
                "openai_tool_call_id": None,
                "openai_function_name": None,
                "tool_call_exception": None,
                "tool_trace": None,
                "visible_to": None,
                "image_ids": None,
                "reasoning_trace": None,
                "openai_reasoning_items": None,
            },
            {
                "sandbox_message_index": 2,
                "sender": RoleType.SYSTEM,
                "recipient": RoleType.EXECUTION_ENVIRONMENT,
                "content": "import json\n",
                "conversation_active": True,
                "openai_tool_call_id": None,
                "openai_function_name": None,
                "tool_call_exception": None,
                "tool_trace": None,
                "visible_to": None,
                "image_ids": None,
                "reasoning_trace": None,
                "openai_reasoning_items": None,
            },
            {
                "sandbox_message_index": 3,
                "sender": None,
                "recipient": None,
                "content": None,
                "conversation_active": None,
                "openai_tool_call_id": None,
                "openai_function_name": None,
                "tool_call_exception": None,
                "tool_trace": None,
                "visible_to": None,
                "image_ids": None,
                "reasoning_trace": None,
                "openai_reasoning_items": None,
            },
            {
                "sandbox_message_index": 3,
                "sender": RoleType.USER,
                "recipient": RoleType.AGENT,
                "content": "Hi",
                "conversation_active": True,
                "openai_tool_call_id": None,
                "openai_function_name": None,
                "tool_call_exception": None,
                "tool_trace": None,
                "visible_to": None,
                "image_ids": None,
                "reasoning_trace": None,
                "openai_reasoning_items": None,
            },
        ],
        schema=ExecutionContext.dbs_schemas[DatabaseNamespace.SANDBOX],
    )

    # Add timestamp related database entry to test timestamp shifting logic
    test_context._dbs[DatabaseNamespace.REMINDER] = pl.DataFrame(
        [
            {
                "sandbox_message_index": 0,
                "reminder_id": None,
                "content": None,
                "latitude": None,
                "longitude": None,
                "creation_datetime": None,
                "reminder_datetime": None,
            },
        ],
        schema=ExecutionContext.dbs_schemas[DatabaseNamespace.REMINDER],
    )

    return test_context


def test_drop_headguard() -> None:
    assert ExecutionContext.drop_headguard(
        pl.DataFrame({"sandbox_message_index": 0, "a": None, "b": None})
    ).is_empty()
    assert ExecutionContext.drop_headguard(
        pl.DataFrame({"a": None, "b": None})
    ).is_empty()


def test_max_sandbox_message_index(default_execution_context: ExecutionContext) -> None:
    # Empty SANDBOX database
    assert default_execution_context.max_sandbox_message_index == -1
    # Add an entry
    default_execution_context._dbs[DatabaseNamespace.SANDBOX] = (
        default_execution_context._dbs[DatabaseNamespace.SANDBOX].vstack(
            pl.DataFrame(
                {
                    "sandbox_message_index": 0,
                    "sender": RoleType.SYSTEM,
                    "recipient": RoleType.AGENT,
                    "content": "Be Good",
                    "conversation_active": True,
                    "openai_tool_call_id": None,
                    "openai_function_name": None,
                    "tool_call_exception": None,
                    "tool_trace": None,
                    "visible_to": None,
                    "finish_reason": None,
                    "logprobs": None,
                    "generation": None,
                    "token_ids": None,
                    "claude_text_response": None,
                    "claude_extended_thinking": None,
                    "claude_extended_thinking_signature": None,
                    "tool_call_text_response": None,
                    "image_ids": None,
                    "reasoning_trace": None,
                    "openai_reasoning_content": None,
                    "openai_reasoning_items": None,
                },
                schema=ExecutionContext.dbs_schemas[DatabaseNamespace.SANDBOX],
            )
        )
    )
    assert default_execution_context.max_sandbox_message_index == 0


def test_get_most_recent_snapshot_sandbox_message_index_empty(
    default_execution_context: ExecutionContext,
) -> None:
    assert (
        default_execution_context.get_most_recent_snapshot_sandbox_message_index(
            namespace=DatabaseNamespace.MESSAGING, query_index=1
        )
        == 0
    )


def test_get_most_recent_snapshot_sandbox_message_index_populated(
    populated_execution_context: ExecutionContext,
) -> None:
    # Out of bounds
    assert (
        populated_execution_context.get_most_recent_snapshot_sandbox_message_index(
            namespace=DatabaseNamespace.SETTING, query_index=0
        )
        == 0
    )
    # Bisect
    assert (
        populated_execution_context.get_most_recent_snapshot_sandbox_message_index(
            namespace=DatabaseNamespace.SETTING, query_index=2
        )
        == 1
    )
    # Max allowed
    assert (
        populated_execution_context.get_most_recent_snapshot_sandbox_message_index(
            namespace=DatabaseNamespace.SETTING, query_index=4
        )
        == 3
    )
    # Error
    with pytest.raises(IndexError):
        populated_execution_context.get_most_recent_snapshot_sandbox_message_index(
            namespace=DatabaseNamespace.SETTING, query_index=5
        )


def test_maybe_create_snapshot(populated_execution_context: ExecutionContext) -> None:
    settings_database = copy.deepcopy(
        populated_execution_context._dbs[DatabaseNamespace.SETTING]
    )
    populated_execution_context._maybe_create_snapshot(
        namespace=DatabaseNamespace.SETTING
    )
    # New snapshot should exist, matching previous latest exactly except index
    assert (
        populated_execution_context._dbs[DatabaseNamespace.SETTING]
        .filter(pl.col("sandbox_message_index") == 3)
        .drop("sandbox_message_index")
        .equals(
            populated_execution_context._dbs[DatabaseNamespace.SETTING]
            .filter(pl.col("sandbox_message_index") == 4)
            .drop("sandbox_message_index")
        )
    )
    # Previous latest should not be modified
    assert (
        populated_execution_context._dbs[DatabaseNamespace.SETTING]
        .filter(pl.col("sandbox_message_index") <= 3)
        .equals(settings_database)
    )
    # New snapshots should not be created after the snapshot for max_sandbox_message_index + 1 exists
    settings_database_with_snapshot = copy.deepcopy(
        populated_execution_context._dbs[DatabaseNamespace.SETTING]
    )
    populated_execution_context._maybe_create_snapshot(
        namespace=DatabaseNamespace.SETTING
    )
    assert populated_execution_context._dbs[DatabaseNamespace.SETTING].equals(
        settings_database_with_snapshot
    )


def test_get_database(populated_execution_context: ExecutionContext) -> None:
    # Default values
    assert populated_execution_context.get_database(
        namespace=DatabaseNamespace.SETTING
    ).equals(
        ExecutionContext.drop_headguard(
            populated_execution_context._dbs[DatabaseNamespace.SETTING]
            .filter(pl.col("sandbox_message_index") == 3)
            .drop("sandbox_message_index")
        )
    )
    # Don't drop index
    assert populated_execution_context.get_database(
        namespace=DatabaseNamespace.SETTING, drop_sandbox_message_index=False
    ).equals(
        ExecutionContext.drop_headguard(
            populated_execution_context._dbs[DatabaseNamespace.SETTING].filter(
                pl.col("sandbox_message_index") == 3
            )
        )
    )
    # Earlier index
    assert populated_execution_context.get_database(
        namespace=DatabaseNamespace.SETTING,
        sandbox_message_index=2,
        drop_sandbox_message_index=False,
    ).equals(
        ExecutionContext.drop_headguard(
            populated_execution_context._dbs[DatabaseNamespace.SETTING].filter(
                pl.col("sandbox_message_index") == 1
            )
        )
    )
    # All history
    assert populated_execution_context.get_database(
        namespace=DatabaseNamespace.SETTING,
        get_all_history_snapshots=True,
        drop_sandbox_message_index=False,
    ).equals(
        ExecutionContext.drop_headguard(
            populated_execution_context._dbs[DatabaseNamespace.SETTING]
        )
    )
    # Don't drop headguard
    assert populated_execution_context.get_database(
        namespace=DatabaseNamespace.SETTING, drop_headguard=False
    ).equals(
        populated_execution_context._dbs[DatabaseNamespace.SETTING]
        .filter(pl.col("sandbox_message_index") == 3)
        .drop("sandbox_message_index")
    )


def test_add_to_database(
    locale: Locale, populated_execution_context: ExecutionContext
) -> None:
    # Incorrect column name
    with pytest.raises(KeyError):
        populated_execution_context.add_to_database(
            namespace=DatabaseNamespace.SANDBOX,
            rows=[
                {"wrong_name": "test"},
            ],
        )
    # Adding messages
    populated_execution_context.add_to_database(
        namespace=DatabaseNamespace.SANDBOX,
        rows=[
            {
                "sender": RoleType.AGENT,
                "recipient": RoleType.USER,
                "content": "Hey",
                "conversation_active": True,
            },
            {
                "sender": RoleType.USER,
                "recipient": RoleType.AGENT,
                "content": "Howdy",
                "conversation_active": True,
            },
        ],
    )
    assert ExecutionContext.drop_headguard(
        populated_execution_context._dbs[DatabaseNamespace.SANDBOX].filter(
            (pl.col("sandbox_message_index") >= 4)
            & (pl.col("sandbox_message_index") <= 5)
        )
    ).equals(
        pl.DataFrame(
            [
                {
                    "sandbox_message_index": 4,
                    "sender": RoleType.AGENT,
                    "recipient": RoleType.USER,
                    "content": "Hey",
                    "conversation_active": True,
                    "openai_tool_call_id": None,
                    "openai_function_name": None,
                    "tool_call_exception": None,
                    "tool_trace": None,
                },
                {
                    "sandbox_message_index": 5,
                    "sender": RoleType.USER,
                    "recipient": RoleType.AGENT,
                    "content": "Howdy",
                    "conversation_active": True,
                    "openai_tool_call_id": None,
                    "openai_function_name": None,
                    "tool_call_exception": None,
                    "tool_trace": None,
                },
            ],
            schema=ExecutionContext.dbs_schemas[DatabaseNamespace.SANDBOX],
        )
    )
    # Adding to non SANDBOX database, snapshot should be created
    # Note that normally SETTINGS should only have 1 row in each snapshot. The following is just for testing purposes
    populated_execution_context.add_to_database(
        namespace=DatabaseNamespace.SETTING,
        rows=[
            {
                "device_id": "1",
                "cellular": True,
                "wifi": False,
                "location_service": True,
                "low_battery_mode": False,
                "locale": locale.name,
                "latitude": 0,
                "longitude": 0,
            }
        ],
    )
    assert ExecutionContext.drop_headguard(
        populated_execution_context._dbs[DatabaseNamespace.SETTING].filter(
            pl.col("sandbox_message_index") == 6
        )
    ).equals(
        pl.DataFrame(
            [
                {
                    "sandbox_message_index": 6,
                    "device_id": "1",
                    "cellular": False,
                    "wifi": False,
                    "location_service": True,
                    "low_battery_mode": False,
                    "locale": locale.name,
                    "latitude": 0,
                    "longitude": 0,
                    "utc_offset_seconds": None,
                },
                {
                    "sandbox_message_index": 6,
                    "device_id": "1",
                    "cellular": True,
                    "wifi": False,
                    "location_service": True,
                    "low_battery_mode": False,
                    "locale": locale.name,
                    "latitude": 0,
                    "longitude": 0,
                    "utc_offset_seconds": None,
                },
            ],
            schema=ExecutionContext.dbs_schemas[DatabaseNamespace.SETTING],
        )
    )
    # Adding to non SANDBOX database, snapshot should not be created
    populated_execution_context.add_to_database(
        namespace=DatabaseNamespace.SETTING,
        rows=[
            {
                "device_id": "1",
                "cellular": True,
                "wifi": True,
                "location_service": True,
                "low_battery_mode": False,
                "locale": locale.name,
                "latitude": 0,
                "longitude": 0,
                "utc_offset_seconds": None,
            }
        ],
    )
    assert ExecutionContext.drop_headguard(
        populated_execution_context._dbs[DatabaseNamespace.SETTING].filter(
            pl.col("sandbox_message_index") == 6
        )
    ).equals(
        pl.DataFrame(
            [
                {
                    "sandbox_message_index": 6,
                    "device_id": "1",
                    "cellular": False,
                    "wifi": False,
                    "location_service": True,
                    "low_battery_mode": False,
                    "locale": locale.name,
                    "latitude": 0,
                    "longitude": 0,
                    "utc_offset_seconds": None,
                },
                {
                    "sandbox_message_index": 6,
                    "device_id": "1",
                    "cellular": True,
                    "wifi": False,
                    "location_service": True,
                    "low_battery_mode": False,
                    "locale": locale.name,
                    "latitude": 0,
                    "longitude": 0,
                    "utc_offset_seconds": None,
                },
                {
                    "sandbox_message_index": 6,
                    "device_id": "1",
                    "cellular": True,
                    "wifi": True,
                    "location_service": True,
                    "low_battery_mode": False,
                    "locale": locale.name,
                    "latitude": 0,
                    "longitude": 0,
                    "utc_offset_seconds": None,
                },
            ],
            schema=ExecutionContext.dbs_schemas[DatabaseNamespace.SETTING],
        )
    )


def test_add_to_empty_database(headguard_execution_context: ExecutionContext) -> None:
    # Make sure adding to an empty database with only 0 index headguard will populate
    # headguard to future indices correctly.
    headguard_execution_context.add_to_database(
        namespace=DatabaseNamespace.REMINDER,
        rows=[
            {
                "reminder_id": "1",
                "content": "2",
                "latitude": None,
                "longitude": None,
                "creation_datetime": None,
                "reminder_datetime": None,
            },
        ],
    )
    assert headguard_execution_context.get_database(
        namespace=DatabaseNamespace.REMINDER,
        get_all_history_snapshots=True,
        drop_sandbox_message_index=False,
        drop_headguard=False,
    ).to_dicts() == [
        {
            "sandbox_message_index": 0,
            "reminder_id": None,
            "content": None,
            "latitude": None,
            "longitude": None,
            "creation_datetime": None,
            "reminder_datetime": None,
        },
        {
            "sandbox_message_index": 4,
            "reminder_id": None,
            "content": None,
            "latitude": None,
            "longitude": None,
            "creation_datetime": None,
            "reminder_datetime": None,
        },
        {
            "sandbox_message_index": 4,
            "reminder_id": "1",
            "content": "2",
            "latitude": None,
            "longitude": None,
            "creation_datetime": None,
            "reminder_datetime": None,
        },
    ]


def test_remove_from_database(populated_execution_context: ExecutionContext) -> None:
    # Remove from sandbox
    with pytest.raises(KeyError):
        populated_execution_context.remove_from_database(
            namespace=DatabaseNamespace.SANDBOX,
            predicate=pl.col("sandbox_message_index") == 0,
        )
    # Remove 1. Headguard should remain, get should show empty if headguard is dropped, snapshot ind should be 4
    populated_execution_context.remove_from_database(
        namespace=DatabaseNamespace.SETTING,
        predicate=(pl.col("cellular") == pl.lit(False)),
    )
    assert (
        populated_execution_context.get_most_recent_snapshot_sandbox_message_index(
            namespace=DatabaseNamespace.SETTING, query_index=4
        )
        == 4
    )
    assert populated_execution_context.get_database(
        namespace=DatabaseNamespace.SETTING
    ).is_empty()
    assert populated_execution_context.get_database(
        namespace=DatabaseNamespace.SETTING,
        drop_headguard=False,
        drop_sandbox_message_index=False,
    ).equals(
        pl.DataFrame(
            {
                "sandbox_message_index": 4,
                "device_id": None,
                "cellular": None,
                "wifi": None,
                "location_service": None,
                "low_battery_mode": None,
                "locale": None,
                "latitude": None,
                "longitude": None,
                "place_id": None,
                "formatted_address": None,
                "utc_offset_seconds": None,
            },
            schema=ExecutionContext.dbs_schemas[DatabaseNamespace.SETTING],
        )
    )


def test_update_database(populated_execution_context: ExecutionContext) -> None:
    # Update sandbox. Should not create snapshot
    populated_execution_context.update_database(
        namespace=DatabaseNamespace.SANDBOX,
        dataframe=populated_execution_context.get_database(
            namespace=DatabaseNamespace.SANDBOX
        ).with_columns(pl.lit("Hey").alias("content")),
    )
    assert populated_execution_context.get_database(
        namespace=DatabaseNamespace.SANDBOX, drop_sandbox_message_index=False
    ).equals(
        pl.DataFrame(
            {
                "sandbox_message_index": 3,
                "sender": RoleType.USER,
                "recipient": RoleType.AGENT,
                "content": "Hey",
                "conversation_active": True,
                "openai_tool_call_id": None,
                "openai_function_name": None,
                "tool_call_exception": None,
                "tool_trace": None,
                "visible_to": None,
                "finish_reason": None,
                "logprobs": None,
                "generation": None,
                "token_ids": None,
                "claude_text_response": None,
                "claude_extended_thinking": None,
                "claude_extended_thinking_signature": None,
                "tool_call_text_response": None,
                "image_ids": None,
                "reasoning_trace": None,
                "openai_reasoning_content": None,
                "openai_reasoning_items": None,
            },
            schema=ExecutionContext.dbs_schemas[DatabaseNamespace.SANDBOX],
        )
    )
    # Update Setting. Should create snapshot. Should contain headguard
    populated_execution_context.update_database(
        namespace=DatabaseNamespace.SETTING,
        dataframe=populated_execution_context.get_database(
            namespace=DatabaseNamespace.SETTING
        ).with_columns(pl.lit(True).alias("cellular")),
    )
    assert populated_execution_context.get_database(
        namespace=DatabaseNamespace.SETTING,
        drop_sandbox_message_index=False,
        drop_headguard=False,
    ).equals(
        pl.DataFrame(
            [
                {
                    "sandbox_message_index": 4,
                    "device_id": None,
                    "cellular": None,
                    "wifi": None,
                    "location_service": None,
                    "low_battery_mode": None,
                    "locale": None,
                    "latitude": None,
                    "longitude": None,
                    "utc_offset_seconds": None,
                },
                {
                    "sandbox_message_index": 4,
                    "device_id": "1",
                    "cellular": True,
                    "wifi": False,
                    "location_service": True,
                    "low_battery_mode": False,
                    "locale": "en_US",
                    "latitude": 0,
                    "longitude": 0,
                    "utc_offset_seconds": None,
                },
            ],
            schema=ExecutionContext.dbs_schemas[DatabaseNamespace.SETTING],
        )
    )


def test_new_context(
    default_execution_context: ExecutionContext,
    populated_execution_context: ExecutionContext,
) -> None:
    set_current_context(default_execution_context)
    with new_context(populated_execution_context):
        assert (
            get_current_context()
            .get_database(DatabaseNamespace.SANDBOX)
            .equals(populated_execution_context.get_database(DatabaseNamespace.SANDBOX))
        )
    assert (
        get_current_context()
        .get_database(DatabaseNamespace.SANDBOX)
        .equals(default_execution_context.get_database(DatabaseNamespace.SANDBOX))
    )
    # Check release during exception
    with pytest.raises(RuntimeError):
        with new_context(populated_execution_context):
            raise RuntimeError()
    assert (
        get_current_context()
        .get_database(DatabaseNamespace.SANDBOX)
        .equals(default_execution_context.get_database(DatabaseNamespace.SANDBOX))
    )


def test_new_context_with_attribute(
    default_execution_context: ExecutionContext,
) -> None:
    set_current_context(default_execution_context)
    with new_context_with_attribute(trace_tool=True):
        assert get_current_context().trace_tool
    assert not get_current_context().trace_tool
    # Check release during exception
    with pytest.raises(RuntimeError):
        with new_context_with_attribute(trace_tool=True):
            raise RuntimeError()
    assert not get_current_context().trace_tool


def test_context_copying(mock_toolbox: Toolbox) -> None:
    context = ExecutionContext(mock_toolbox, delay_initialization=False)

    # Set a local variable in the interactive console. We will use this variable to
    # ensure that we can create deep copies of the execution context. Also import a
    # module and a user-defined function so we can test these as well.
    console = context.interactive_console
    assert "a" not in console.locals
    console.runsource("a=0")
    assert console.locals["a"] == 0
    console.runsource("import math; b = math.degrees(math.pi)")
    assert pytest.approx(180.0) == console.locals["b"]
    console.runsource("from mmtoolsandbox.common.utils import deterministic_uuid")
    assert "deterministic_uuid" in console.locals

    # Create a copy and change the value of `a` to allow testing that we indeed created
    # a deep copy.
    clone = copy.deepcopy(context).interactive_console
    assert clone.locals["a"] == 0
    clone.runsource("a=1")
    assert clone.locals["a"] == 1
    # The variable in the original context should be unchanged.
    assert console.locals["a"] == 0

    # Test that the `math` module was correctly copied. `InteractiveConsole` internally
    # catches exceptions and just prints them so we use the existence of a variable as
    # a proxy for the statement being executed successfully. The lines below ensure that
    # this approach can indeed be used to test that statement execution failed.
    clone.runsource("c = json.dumps({})")
    assert "c" not in clone.locals

    # If for some reason the `math` import was not copied correctly then the next
    # statement would raise an exception that the `InteractiveConsole` would catch
    # internally. The variable `c` would then be undefined.
    clone.runsource("c = math.degrees(math.pi)")
    assert pytest.approx(console.locals["b"]) == clone.locals["c"]

    # Ensure that the user-defined `deterministic_uuid` function exists in the cloned
    # console and can be called.
    assert "deterministic_uuid" in clone.locals
    assert "d" not in clone.locals
    clone.runsource("d = deterministic_uuid(payload='test')")
    assert "d" in clone.locals


def test_serialization_roundtrip(populated_execution_context: ExecutionContext) -> None:
    copied_context = ExecutionContext.from_dict(
        populated_execution_context.to_dict(serialize_console=True)
    )
    for namespace in populated_execution_context.get_active_database_namespaces():
        assert copied_context._dbs[namespace].equals(
            populated_execution_context._dbs[namespace]
        )

    assert (
        populated_execution_context.get_introspection_database_manager().serialize_to_dict()
        == copied_context.get_introspection_database_manager().serialize_to_dict()
    )
    assert copied_context.tool_allow_list == populated_execution_context.tool_allow_list
    assert copied_context.trace_tool == populated_execution_context.trace_tool
    assert copied_context.toolbox == populated_execution_context.toolbox
    # We have manually added a tool state to the `populated_execution_context`, which
    # is not set up as part of the `ToolStateRegistryFactory`. We do not actually
    # serialize the registry, but re-load it. Thus, the element should not exist in
    # the deserialized `ExecutionContext`.
    assert [] == copied_context.get_tool_state_registry().get_tool_state_types()
    assert (
        populated_execution_context.interactive_console.locals["a"]
        == copied_context.interactive_console.locals["a"]
    )
    assert (
        copied_context.active_tool_cache
        == populated_execution_context.active_tool_cache
    )

    # Check that the contents of the `InteractiveConsole` are not restored when
    # serializing it is not being deserialized.
    copied_context = ExecutionContext.from_dict(
        populated_execution_context.to_dict(serialize_console=False)
    )
    assert "a" not in copied_context.interactive_console.locals


def test_active_databases_default_context(
    default_execution_context: ExecutionContext,
) -> None:
    # When default constructing an `ExecutionContext` we add all databases for which
    # schemas are defined. The reason is that during scenario definition the starting
    # execution context is default-constructed, but we still want the ability to add
    # messages to databases. Only later when the toolbox gets set we know the active
    # databases.
    assert (
        set(ExecutionContext.dbs_schemas.keys())
        == default_execution_context.get_active_database_namespaces()
    )


def test_active_databases_with_toolbox() -> None:
    toolbox = load_toolbox(ToolboxName.FULL, config={})
    execution_context = ExecutionContext(toolbox)
    # Consistency check.
    assert (
        toolbox.extract_database_namespaces()
        == execution_context.get_active_database_namespaces()
    )


class MockToolState(ToolState):
    def __init__(self) -> None:
        self.number = 42

    def to_dict(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_dict(cls, serialized_dict: dict[str, Any]) -> ToolState:
        return MockToolState()


@pytest.fixture
def mock_toolbox() -> Toolbox:
    class MockToolStateRegistryFactory(ToolStateRegistryFactory):
        def __call__(
            self,
            toolbox: Toolbox,
            tool_allow_list: Optional[list[str]],
            tool_deny_list: Optional[list[str]],
        ) -> ToolStateRegistry:
            registry = ToolStateRegistry()
            registry.add(ToolStateType.UTILITIES, MockToolState())
            return registry

    toolbox = Toolbox(
        name="test_toolbox",
        config=ToolboxConfig(),
        tools=[],
        tool_state_registry_factory=MockToolStateRegistryFactory(),
    )
    return toolbox


def test_delayed_initialization(mock_toolbox: Toolbox) -> None:
    # When delayed initialization is enabled the tool state registry member should not
    # exist.
    execution_context = ExecutionContext(
        toolbox=mock_toolbox,
        delay_initialization=True,
    )
    with pytest.raises(RuntimeError, match=r".*execute_delayed_initialization.*"):
        execution_context.get_tool_state_registry()

    # Manually trigger delayed initialization.
    execution_context.execute_delayed_initialization()
    tool_state_registry = execution_context.get_tool_state_registry()
    tool_state = tool_state_registry.get(ToolStateType.UTILITIES)
    assert isinstance(tool_state, MockToolState)
    assert 42 == tool_state.number

    # Disable delayed initialization. The tool state registry should be populated right
    # away.
    execution_context = ExecutionContext(
        toolbox=mock_toolbox,
        delay_initialization=False,
    )
    tool_state_registry = execution_context.get_tool_state_registry()
    tool_state = tool_state_registry.get(ToolStateType.UTILITIES)
    assert isinstance(tool_state, MockToolState)
    assert 42 == tool_state.number


def test_repeated_delayed_initialization(mock_toolbox: Toolbox) -> None:
    execution_context = ExecutionContext(
        toolbox=mock_toolbox,
        delay_initialization=True,
    )
    execution_context.execute_delayed_initialization()
    with pytest.raises(RuntimeError, match=r".*has already been performed.*"):
        execution_context.execute_delayed_initialization()

    # The delayed initialization is already performed as part of the constructor.
    execution_context = ExecutionContext(
        toolbox=mock_toolbox,
        delay_initialization=False,
    )
    with pytest.raises(RuntimeError, match=r".*has already been performed.*"):
        execution_context.execute_delayed_initialization()


def test_thread_safety() -> None:
    random.seed(42)
    q: queue.Queue[Any] = queue.Queue()
    execution_contexts: list[ExecutionContext] = [ExecutionContext() for _ in range(32)]

    def thread_func() -> None:
        """Set and get default_execution_context for half of the threads.

        Keep results in result_execution_context_queue.

        Args:

        """
        target = random.choice(execution_contexts)
        set_current_context(target)
        q.put((id(target), id(get_current_context())))

    threads: list[threading.Thread] = []
    for _ in execution_contexts:
        x = threading.Thread(target=thread_func)
        threads.append(x)
        x.start()
    for thread in threads:
        thread.join()
    # Check result
    while not q.empty():
        target_id, result_id = q.get()
        assert target_id == result_id
        q.task_done()
    q.join()


def test_realign_timestamp(populated_execution_context: ExecutionContext) -> None:
    execution_context = realign_timestamp(
        execution_context=populated_execution_context,
        execution_context_creation_time=datetime.datetime(year=2025, month=1, day=1),
        new_execution_context_creation_time=datetime.datetime(
            year=2025, month=1, day=1, hour=1
        ),
    )
    for datetime_key in ["creation_datetime", "reminder_datetime"]:
        for original, shifted in zip(
            populated_execution_context.get_database(
                namespace=DatabaseNamespace.REMINDER, get_all_history_snapshots=True
            )[datetime_key].to_list(),
            execution_context.get_database(
                namespace=DatabaseNamespace.REMINDER, get_all_history_snapshots=True
            )[datetime_key].to_list(),
        ):
            if shifted is None:
                assert original is None
            else:
                shifted_dt = datetime.datetime.fromisoformat(shifted)
                expected_dt = shifted_dt - datetime.timedelta(hours=1)
                assert original == expected_dt.isoformat()


@pytest.mark.parametrize("locale", [locale for locale in Locale])
def test_get_user_locale(
    locale: Locale, populated_execution_context: ExecutionContext
) -> None:
    with new_context(populated_execution_context):
        assert get_locale() == locale


def test_dynamic_role_tool_allow_list_update() -> None:
    """Test that updating role_tool_allow_list dynamically updates available tools."""
    # Create mock tools
    tool1 = MagicMock()
    tool1.__name__ = "tool1"
    tool1.visible_to = (RoleType.AGENT,)
    # Mock database_namespaces for Toolbox validation
    tool1.database_namespaces = set()

    tool2 = MagicMock()
    tool2.__name__ = "tool2"
    tool2.visible_to = (RoleType.AGENT,)
    tool2.database_namespaces = set()

    tool3 = MagicMock()
    tool3.__name__ = "tool3"
    tool3.visible_to = (RoleType.AGENT,)
    tool3.database_namespaces = set()

    # Create toolbox
    # We need end_conversation because Toolbox validation requires it
    end_conversation = MagicMock()
    end_conversation.__name__ = "end_conversation"
    end_conversation.__module__ = "mmtoolsandbox.tools.tool_sandbox.user_tools"
    end_conversation.database_namespaces = set()

    toolbox = Toolbox(
        name="test_toolbox",
        config=ToolboxConfig(),
        tools=[tool1, tool2, tool3, end_conversation],
    )

    # Create context
    context = ExecutionContext(toolbox=toolbox)

    # Initial state: only tool1 allowed
    context.role_tool_allow_list = {RoleType.AGENT: ["tool1"]}

    # Verify initial state
    available_tools = context.get_available_tools_for_role(RoleType.AGENT)
    assert "tool1" in available_tools
    assert "tool2" not in available_tools

    # Dynamically update role_tool_allow_list (simulate enable_tool)
    context.role_tool_allow_list[RoleType.AGENT].append("tool2")

    # Verify updated state
    updated_available_tools = context.get_available_tools_for_role(RoleType.AGENT)
    assert "tool1" in updated_available_tools
    assert "tool2" in updated_available_tools
    assert "tool3" not in updated_available_tools


def test_register_tools_standard_mode_safety() -> None:
    """Test that register_tools does NOT restrict tools in Standard Mode."""
    # Create mock tools
    tool1 = MagicMock()
    tool1.__name__ = "tool1"
    tool1.visible_to = (RoleType.AGENT,)
    tool1.database_namespaces = set()

    tool2 = MagicMock()
    tool2.__name__ = "tool2"
    tool2.visible_to = (RoleType.AGENT,)
    tool2.database_namespaces = set()

    # Create toolbox
    end_conversation = MagicMock()
    end_conversation.__name__ = "end_conversation"
    end_conversation.__module__ = "mmtoolsandbox.tools.tool_sandbox.user_tools"
    end_conversation.database_namespaces = set()

    toolbox = Toolbox(
        name="test_toolbox",
        config=ToolboxConfig(),
        tools=[tool1, tool2, end_conversation],
    )

    # Create context with restricted tool_allow_list (Standard Mode)
    # Only tool2 is allowed initially
    context = ExecutionContext(toolbox=toolbox, tool_allow_list=["tool2"])
    assert RoleType.AGENT not in context.role_tool_allow_list

    # Verify initial state
    available = context.get_available_tools_for_role(RoleType.AGENT)
    assert "tool2" in available
    assert "tool1" not in available

    # Register tool1
    context.register_tools(["tool1"])

    # Verify role_tool_allow_list is STILL not set for AGENT
    # This ensures we didn't accidentally switch to restricted mode
    assert RoleType.AGENT not in context.role_tool_allow_list

    # Verify tool1 is now available (added to tool_allow_list)
    available = context.get_available_tools_for_role(RoleType.AGENT)
    assert "tool1" in available
    assert "tool2" in available


def test_register_tools_lru_eviction() -> None:
    """Test that register_tools respects LRU eviction policy."""
    limit = MAX_ACTIVE_TOOLS
    num_tools = limit + 5

    # Create mock tools
    tools = []
    for i in range(num_tools):
        tool = MagicMock()
        tool.__name__ = f"tool_{i}"
        tool.visible_to = (RoleType.AGENT,)
        tool.database_namespaces = set()
        tools.append(tool)

    # Create toolbox
    end_conversation = MagicMock()
    end_conversation.__name__ = "end_conversation"
    end_conversation.__module__ = "mmtoolsandbox.tools.tool_sandbox.user_tools"
    end_conversation.database_namespaces = set()

    toolbox = Toolbox(
        name="test_toolbox",
        config=ToolboxConfig(),
        tools=cast(list[Callable[..., Any]], tools + [end_conversation]),
    )

    # Create context — tool_allow_list must be a non-None list to simulate hybrid
    # mode (pure_code_exec mode sets tool_allow_list=None and skips LRU entirely).
    context = ExecutionContext(toolbox=toolbox)
    context.tool_allow_list = []
    context.role_tool_allow_list = {RoleType.AGENT: []}

    # Register limit tools (fill the cache)
    for i in range(limit):
        context.register_tools([f"tool_{i}"])

    assert len(context.active_tool_cache) == limit
    assert context.active_tool_cache[0] == "tool_0"  # Oldest
    assert context.active_tool_cache[-1] == f"tool_{limit - 1}"  # Newest
    assert "tool_0" in context.interactive_console.locals
    assert "tool_0" in context.role_tool_allow_list[RoleType.AGENT]

    # Register (limit + 1)th tool (trigger eviction)
    context.register_tools([f"tool_{limit}"])

    assert len(context.active_tool_cache) == limit
    assert context.active_tool_cache[0] == "tool_1"  # tool_0 should be gone
    assert context.active_tool_cache[-1] == f"tool_{limit}"

    # Verify eviction from locals and allow list
    assert "tool_0" not in context.interactive_console.locals
    assert "tool_0" not in context.role_tool_allow_list[RoleType.AGENT]
    assert f"tool_{limit}" in context.interactive_console.locals
    assert f"tool_{limit}" in context.role_tool_allow_list[RoleType.AGENT]

    # Re-register an existing tool (should move to end)
    context.register_tools(["tool_1"])
    assert context.active_tool_cache[-1] == "tool_1"
    assert len(context.active_tool_cache) == limit
    # tool_2 should now be the oldest
    assert context.active_tool_cache[0] == "tool_2"


def test_register_tools_lru_execution_update() -> None:
    """Test that executing a tool updates its position in the LRU cache."""
    limit = MAX_ACTIVE_TOOLS

    # Create mock tools
    tools = []
    for i in range(limit):
        tool = MagicMock()
        tool.__name__ = f"tool_{i}"
        tool.visible_to = (RoleType.AGENT,)
        tool.database_namespaces = set()
        tools.append(tool)

    end_conversation = MagicMock()
    end_conversation.__name__ = "end_conversation"
    end_conversation.__module__ = "mmtoolsandbox.tools.tool_sandbox.user_tools"
    end_conversation.database_namespaces = set()

    toolbox = Toolbox(
        name="test_toolbox",
        config=ToolboxConfig(),
        tools=cast(list[Callable[..., Any]], tools + [end_conversation]),
    )

    context = ExecutionContext(toolbox=toolbox)
    context.tool_allow_list = []  # non-None: simulate hybrid mode (pure_code_exec skips LRU)
    context.role_tool_allow_list = {RoleType.AGENT: []}

    # Register limit tools (fill the cache)
    for i in range(limit):
        context.register_tools([f"tool_{i}"])

    # Cache is full: tool_0 is oldest, tool_{limit-1} is newest
    assert context.active_tool_cache[0] == "tool_0"
    assert context.active_tool_cache[-1] == f"tool_{limit - 1}"

    # Execute tool_0 via the interactive console
    # This should trigger the wrapper and move tool_0 to the end of the cache
    context.interactive_console.runsource("tool_0()")

    # Verify tool_0 is now the most recently used (at the end)
    assert context.active_tool_cache[-1] == "tool_0"
    # Verify tool_1 is now the oldest (at the front)
    assert context.active_tool_cache[0] == "tool_1"

    # Verify the underlying mock was actually called
    tools[0].assert_called_once()


def test_sandbox_schema_includes_reasoning_trace() -> None:
    """Verify reasoning_trace column exists in SANDBOX schema."""
    schema = ExecutionContext.dbs_schemas[DatabaseNamespace.SANDBOX]
    assert "reasoning_trace" in schema
    assert schema["reasoning_trace"] == pl.String
