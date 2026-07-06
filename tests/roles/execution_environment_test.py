# Copyright © 2026 Apple Inc.

"""Unit tests for mmtoolsandbox.roles.execution_environment"""

import queue
import random
import textwrap
import threading
from typing import Any, Iterator

import pytest

from mmtoolsandbox.common.execution_context import (
    DatabaseNamespace,
    ExecutionContext,
    RoleType,
    get_current_context,
    new_context,
)
from mmtoolsandbox.common.message_conversion import (
    Message,
    add_messages_to_execution_context,
    get_messages_from_execution_context,
)
from mmtoolsandbox.common.utils import deterministic_uuid
from mmtoolsandbox.roles.execution_environment import ExecutionEnvironment


@pytest.fixture(scope="function", autouse=True)
def execution_context() -> Iterator[None]:
    """Autouse fixture which will setup and teardown execution context before and after each test function

    Returns:

    """
    # Disable safety guard for these tests as they use modules/functions that might be blocked
    # (e.g. warnings, time.sleep) and we want to test the execution environment logic itself.
    with new_context(ExecutionContext(safety_guard_config=None)):
        yield


@pytest.fixture
def execution_environment() -> ExecutionEnvironment:
    """Execution environment object used for testing

    Returns:
        An execution environment object
    """
    return ExecutionEnvironment()


def test_execution_environment_syntax_error(
    execution_environment: ExecutionEnvironment,
) -> None:
    add_messages_to_execution_context(
        execution_context=get_current_context(),
        messages=[
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content="improt math",
            )
        ],
    )
    response_messages, _ = execution_environment.respond(
        messages=get_messages_from_execution_context(
            execution_context=get_current_context()
        ),
        available_tools={},
    )
    add_messages_to_execution_context(
        execution_context=get_current_context(), messages=response_messages
    )
    content = textwrap.dedent("""\
      File "<input>", line 1
        improt math
               ^^^^
    SyntaxError: invalid syntax
    """)
    assert get_messages_from_execution_context(execution_context=get_current_context())[
        -1
    ] == Message(
        sender=RoleType.EXECUTION_ENVIRONMENT,
        recipient=RoleType.AGENT,
        content=content,
        conversation_active=True,
        tool_call_exception=content,
    )


def test_execution_environment_warning(
    execution_environment: ExecutionEnvironment,
) -> None:
    add_messages_to_execution_context(
        execution_context=get_current_context(),
        messages=[
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content="import warnings; warnings.warn('foo')",
            )
        ],
    )
    response_messages, _ = execution_environment.respond(
        messages=get_messages_from_execution_context(
            execution_context=get_current_context()
        ),
        available_tools={},
    )
    add_messages_to_execution_context(
        execution_context=get_current_context(), messages=response_messages
    )
    # No exception set for warnings.
    assert get_messages_from_execution_context(execution_context=get_current_context())[
        -1
    ] == Message(
        sender=RoleType.EXECUTION_ENVIRONMENT,
        recipient=RoleType.AGENT,
        content="",
        conversation_active=True,
        tool_call_exception=None,
    )


def test_execution_environment_raise_error(
    execution_environment: ExecutionEnvironment,
) -> None:
    add_messages_to_execution_context(
        execution_context=get_current_context(),
        messages=[
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content='raise ValueError("This is a test error")',
            )
        ],
    )
    response_messages, _ = execution_environment.respond(
        messages=get_messages_from_execution_context(
            execution_context=get_current_context()
        ),
        available_tools={},
    )
    add_messages_to_execution_context(
        execution_context=get_current_context(), messages=response_messages
    )
    content = "ValueError: This is a test error"
    assert get_messages_from_execution_context(execution_context=get_current_context())[
        -1
    ] == Message(
        sender=RoleType.EXECUTION_ENVIRONMENT,
        recipient=RoleType.AGENT,
        content=content,
        conversation_active=True,
        tool_call_exception=content,
    )


def test_execution_environment_incomplete_command(
    execution_environment: ExecutionEnvironment,
) -> None:
    message = Message(
        sender=RoleType.AGENT,
        recipient=RoleType.EXECUTION_ENVIRONMENT,
        content="if True:",  # < incomplete code
    )
    add_messages_to_execution_context(
        execution_context=get_current_context(), messages=[message]
    )
    response_messages, _ = execution_environment.respond(
        messages=get_messages_from_execution_context(
            execution_context=get_current_context()
        ),
        available_tools={},
    )
    add_messages_to_execution_context(
        execution_context=get_current_context(), messages=response_messages
    )
    content = (
        "Error: The given code was incomplete and could not be executed: "
        f"'{message.content}'"
    )
    assert get_messages_from_execution_context(execution_context=get_current_context())[
        -1
    ] == Message(
        sender=RoleType.EXECUTION_ENVIRONMENT,
        recipient=RoleType.AGENT,
        content=content,
        conversation_active=True,
        tool_call_exception=content,
    )


def test_execution_environment_successful_execution(
    execution_environment: ExecutionEnvironment,
) -> None:
    add_messages_to_execution_context(
        execution_context=get_current_context(),
        messages=[
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content="a = 1\nb = 2\ncall_c_response = a + b",
                openai_tool_call_id="call_c",
                openai_function_name="dummy",
            )
        ],
    )
    response_messages, _ = execution_environment.respond(
        messages=get_messages_from_execution_context(
            execution_context=get_current_context()
        ),
        available_tools={},
    )
    add_messages_to_execution_context(
        execution_context=get_current_context(), messages=response_messages
    )
    content = "3"
    assert get_messages_from_execution_context(execution_context=get_current_context())[
        -1
    ] == Message(
        sender=RoleType.EXECUTION_ENVIRONMENT,
        recipient=RoleType.AGENT,
        content=textwrap.dedent(content),
        conversation_active=True,
        openai_tool_call_id="call_c",
        openai_function_name="dummy",
    )


def test_valid_parallel_tool_call(
    execution_environment: ExecutionEnvironment,
) -> None:
    # Set up the environment by adding a system message that specifies which tools to
    # import.
    add_messages_to_execution_context(
        execution_context=get_current_context(),
        messages=[
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content=(
                    "import json\n"
                    "from mmtoolsandbox.common.execution_context import get_current_context, DatabaseNamespace\n"
                    "import polars as pl\n"
                    "def search_contacts(name):\n"
                    "    db = get_current_context().get_database(DatabaseNamespace.CONTACT)\n"
                    "    return json.dumps(db.filter(pl.col('name').str.contains(name)).to_dicts())\n"
                ),
                conversation_active=True,
            )
        ],
    )
    response_messages, _ = execution_environment.respond(
        messages=get_messages_from_execution_context(
            execution_context=get_current_context()
        ),
        available_tools={},
    )
    add_messages_to_execution_context(
        execution_context=get_current_context(), messages=response_messages
    )
    get_current_context().add_to_database(
        DatabaseNamespace.CONTACT,
        rows=[
            {
                "person_id": deterministic_uuid(payload="Tomas Haake"),
                "name": "Tomas Haake",
                "phone_number": "+11233344455",
                "relationship": "self",
                "is_self": True,
            },
            {
                "person_id": deterministic_uuid(payload="Fredrik Thordendal"),
                "name": "Fredrik Thordendal",
                "phone_number": "+12453344098",
                "relationship": "friend",
                "is_self": False,
            },
        ],
    )

    # Agents can issue parallel tool calls. The expectation is that these are
    # independent and can thus be executed in any order. This is true for the tool calls
    # defined below.
    add_messages_to_execution_context(
        execution_context=get_current_context(),
        messages=[
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content="call_ucWrTNIM4Prh1KlK77k1Zjeu_parameters = {'name': 'Tomas Haake'}\n"
                "call_ucWrTNIM4Prh1KlK77k1Zjeu_response = search_contacts(**call_ucWrTNIM4Prh1KlK77k1Zjeu_parameters)",
                conversation_active=True,
                openai_tool_call_id="call_ucWrTNIM4Prh1KlK77k1Zjeu",
                openai_function_name="search_contacts",
            ),
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content="call_e60fwDpTW2yecfBg2BQPNy88_parameters = {'name': 'Fredrik Thordendal'}\n"
                "call_e60fwDpTW2yecfBg2BQPNy88_response = search_contacts(**call_e60fwDpTW2yecfBg2BQPNy88_parameters)",
                conversation_active=True,
                openai_tool_call_id="call_e60fwDpTW2yecfBg2BQPNy88",
                openai_function_name="search_contacts",
            ),
        ],
    )
    response_messages, _ = execution_environment.respond(
        messages=get_messages_from_execution_context(
            execution_context=get_current_context()
        ),
        available_tools={},
    )
    add_messages_to_execution_context(
        execution_context=get_current_context(), messages=response_messages
    )
    # Without setting `get_all_history_snapshots` we only get the latest tool call back.
    df = get_current_context().get_database(
        DatabaseNamespace.SANDBOX, get_all_history_snapshots=True
    )
    assert RoleType.EXECUTION_ENVIRONMENT == df[-2]["sender"][0]
    assert RoleType.AGENT == df[-2]["recipient"][0]
    assert "search_contacts" == df[-2]["openai_function_name"][0]
    # If all possible orderings of the tool calls succeed then the response should match
    # the original requests.
    assert "Tomas Haake" in df[-2]["content"][0]

    assert RoleType.EXECUTION_ENVIRONMENT == df[-1]["sender"][0]
    assert RoleType.AGENT == df[-1]["recipient"][0]
    assert "search_contacts" == df[-1]["openai_function_name"][0]
    assert "Fredrik Thordendal" in df[-1]["content"][0]


def test_parallel_tool_call_permutation_disabled(
    execution_environment: ExecutionEnvironment,
) -> None:
    # Set up the environment by adding a system message that specifies which tools to
    # import.
    add_messages_to_execution_context(
        execution_context=get_current_context(),
        messages=[
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content=(
                    "import json\n"
                    "from mmtoolsandbox.tools.tool_sandbox.setting import set_cellular_service_status\n"
                    "def send_message_with_phone_number(phone_number, content):\n"
                    "    return json.dumps({'status': 'sent', 'phone_number': phone_number})\n"
                ),
                conversation_active=True,
            )
        ],
    )
    # Disable parallel tool call permutation.
    response_messages, _ = execution_environment.respond(
        messages=get_messages_from_execution_context(
            execution_context=get_current_context()
        ),
        available_tools={},
    )
    get_current_context().add_to_database(
        namespace=DatabaseNamespace.SETTING,
        rows=[{"cellular": False}],
    )
    get_current_context().add_to_database(
        DatabaseNamespace.CONTACT,
        rows=[
            {
                "person_id": deterministic_uuid(payload="Tomas Haake"),
                "name": "Tomas Haake",
                "phone_number": "+11233344455",
                "relationship": "self",
                "is_self": True,
            },
            {
                "person_id": deterministic_uuid(payload="Fredrik Thordendal"),
                "name": "Fredrik Thordendal",
                "phone_number": "+12453344098",
                "relationship": "friend",
                "is_self": False,
            },
        ],
    )
    add_messages_to_execution_context(
        execution_context=get_current_context(), messages=response_messages
    )
    # In the example below the tool calls depend on each other, but when executed
    # sequentially in the given order they would succeed.
    add_messages_to_execution_context(
        execution_context=get_current_context(),
        messages=[
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content="call_ucWrTNIM4Prh1KlK77k1Zjeu_parameters = {'on': True}\n"
                "call_ucWrTNIM4Prh1KlK77k1Zjeu_response = set_cellular_service_status(**call_ucWrTNIM4Prh1KlK77k1Zjeu_parameters)",
                conversation_active=True,
                openai_tool_call_id="call_ucWrTNIM4Prh1KlK77k1Zjeu",
                openai_function_name="set_cellular_service_status",
            ),
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content="call_e60fwDpTW2yecfBg2BQPNy88_parameters = {'phone_number': '+12453344098', 'content': 'Hi'}\n"
                "call_e60fwDpTW2yecfBg2BQPNy88_response = send_message_with_phone_number(**call_e60fwDpTW2yecfBg2BQPNy88_parameters)",
                conversation_active=True,
                openai_tool_call_id="call_e60fwDpTW2yecfBg2BQPNy88",
                openai_function_name="send_message_with_phone_number",
            ),
        ],
    )
    response_messages, _ = execution_environment.respond(
        messages=get_messages_from_execution_context(
            execution_context=get_current_context()
        ),
        available_tools={},
    )
    add_messages_to_execution_context(
        execution_context=get_current_context(), messages=response_messages
    )
    # Cellular service should be on at this point
    assert get_current_context().get_database(DatabaseNamespace.SETTING)["cellular"][0]

    # Without setting `get_all_history_snapshots` we only get the latest tool call back.
    df = get_current_context().get_database(
        DatabaseNamespace.SANDBOX, get_all_history_snapshots=True
    )

    # Execution should happen in original order
    assert RoleType.EXECUTION_ENVIRONMENT == df[-2]["sender"][0]
    assert RoleType.AGENT == df[-2]["recipient"][0]
    assert "set_cellular_service_status" == df[-2]["openai_function_name"][0]

    assert RoleType.EXECUTION_ENVIRONMENT == df[-1]["sender"][0]
    assert RoleType.AGENT == df[-1]["recipient"][0]
    assert "send_message_with_phone_number" == df[-1]["openai_function_name"][0]


def test_thread_safety(
    execution_environment: ExecutionEnvironment,
) -> None:
    random.seed(42)
    q: queue.Queue[Any] = queue.Queue()

    def thread_func() -> None:
        """Executes dummy tool call and extracts result.

        Keep results in q.

        Args:

        """
        # Use a context with safety guard disabled for this thread
        with new_context(ExecutionContext(safety_guard_config=None)):
            # Value for mock function to return
            value = random.randint(0, 256)
            add_messages_to_execution_context(
                execution_context=get_current_context(),
                messages=[
                    Message(
                        sender=RoleType.AGENT,
                        recipient=RoleType.EXECUTION_ENVIRONMENT,
                        content=f"import time\n"
                        f"import random\n"
                        f"time.sleep(random.random())\n"  # Use sleep to mimic random tool invocation duration.
                        f"call_ucWrTNIM4Prh1KlK77k1Zjeu_parameters = {{}}\n"
                        f"call_ucWrTNIM4Prh1KlK77k1Zjeu_response = {value}",
                        conversation_active=True,
                        openai_tool_call_id="call_ucWrTNIM4Prh1KlK77k1Zjeu",
                        openai_function_name="dummy",
                    ),
                ],
            )
            response_messages, _ = execution_environment.respond(
                messages=get_messages_from_execution_context(
                    execution_context=get_current_context()
                ),
                available_tools={},
            )
            add_messages_to_execution_context(
                execution_context=get_current_context(), messages=response_messages
            )
            # Without setting `get_all_history_snapshots` we only get the latest tool call back.
            df = get_current_context().get_database(
                DatabaseNamespace.SANDBOX, get_all_history_snapshots=True
            )
            q.put((f"{value}", df[-1]["content"][0]))

    threads: list[threading.Thread] = []
    for _ in range(1024):
        x = threading.Thread(target=thread_func)
        threads.append(x)
        x.start()
    for thread in threads:
        thread.join()
    # Check result
    while not q.empty():
        target, result = q.get()
        assert target == result

        q.task_done()
    q.join()


def test_thread_safety_error(
    execution_environment: ExecutionEnvironment,
) -> None:
    random.seed(42)
    q: queue.Queue[Any] = queue.Queue()

    def thread_func() -> None:
        """Raises dummy error.

        Keep results in q.

        Args:

        """
        # Use a context with safety guard disabled for this thread
        with new_context(ExecutionContext(safety_guard_config=None)):
            # Value for mock function to return
            value = random.randint(0, 256)
            add_messages_to_execution_context(
                execution_context=get_current_context(),
                messages=[
                    Message(
                        sender=RoleType.AGENT,
                        recipient=RoleType.EXECUTION_ENVIRONMENT,
                        content=f"import time\n"
                        f"import random\n"
                        f"time.sleep(random.random())\n"  # Use sleep to mimic random tool invocation duration.
                        f"raise ValueError({value})",
                        conversation_active=True,
                        openai_tool_call_id="call_ucWrTNIM4Prh1KlK77k1Zjeu",
                        openai_function_name="dummy",
                    ),
                ],
            )
            response_messages, _ = execution_environment.respond(
                messages=get_messages_from_execution_context(
                    execution_context=get_current_context()
                ),
                available_tools={},
            )
            add_messages_to_execution_context(
                execution_context=get_current_context(), messages=response_messages
            )
            # Without setting `get_all_history_snapshots` we only get the latest tool call back.
            df = get_current_context().get_database(
                DatabaseNamespace.SANDBOX, get_all_history_snapshots=True
            )
            q.put((f"ValueError: {value}", df[-1]["content"][0]))

    threads: list[threading.Thread] = []
    for _ in range(1024):
        x = threading.Thread(target=thread_func)
        threads.append(x)
        x.start()
    for thread in threads:
        thread.join()
    # Check result
    while not q.empty():
        target, result = q.get()
        assert target == result

        q.task_done()
    q.join()


def test_external_tool_schemas(
    execution_environment: ExecutionEnvironment,
) -> None:
    # Define external tool schema for get_status function
    external_tool_schemas = [
        {
            "function": {
                "name": "get_status",
                "description": "Get the current status",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
        }
    ]

    add_messages_to_execution_context(
        execution_context=get_current_context(),
        messages=[
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content="get_status()",
                openai_function_name="get_status",
                openai_tool_call_id="call_0",
            )
        ],
    )
    response_messages, _ = execution_environment.respond(
        messages=get_messages_from_execution_context(
            execution_context=get_current_context()
        ),
        available_tools={},
        external_tool_schemas=external_tool_schemas,
    )
    assert not response_messages
    assert (
        not get_current_context()
        .get_database(DatabaseNamespace.SANDBOX)
        .to_dicts()[0]["conversation_active"]
    )
