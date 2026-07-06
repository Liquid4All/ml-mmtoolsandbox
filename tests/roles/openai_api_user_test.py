# Copyright © 2026 Apple Inc.

"""Tests for OpenAIAPIUser, specifically tool-calling message conversion."""

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.message_conversion import Message, _get_message_views
from mmtoolsandbox.roles.openai_user import OpenAIAPIUser


class TestToOpenAIMessages:
    """Test OpenAIAPIUser.to_openai_messages handles all message types correctly."""

    def test_basic_system_and_agent_messages(self) -> None:
        """System→User becomes 'system', Agent→User becomes 'user'."""
        messages = [
            Message(
                sender=RoleType.SYSTEM,
                recipient=RoleType.USER,
                content="You are a user.",
            ),
            Message(sender=RoleType.AGENT, recipient=RoleType.USER, content="Hello!"),
        ]
        result = OpenAIAPIUser.to_openai_messages(messages)
        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "You are a user."}
        assert result[1] == {"role": "user", "content": "Hello!"}

    def test_user_to_agent_becomes_assistant(self) -> None:
        """User→Agent becomes 'assistant' (the user model's own output)."""
        messages = [
            Message(
                sender=RoleType.USER,
                recipient=RoleType.AGENT,
                content="I want to book a table.",
            ),
        ]
        result = OpenAIAPIUser.to_openai_messages(messages)
        assert len(result) == 1
        assert result[0] == {"role": "assistant", "content": "I want to book a table."}

    def test_user_tool_call_becomes_assistant_with_tool_calls(self) -> None:
        """USER→EXEC_ENV with tool metadata becomes 'assistant' with tool_calls."""
        messages = [
            Message(
                sender=RoleType.USER,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content=(
                    'call_abc123_parameters = {"action_name": "click", "surface_id": "s1", "component_id": "b1"}\n'
                    "call_abc123_response = ui_user_interact(**call_abc123_parameters)"
                ),
                openai_tool_call_id="call_abc123",
                openai_function_name="ui_user_interact",
            ),
        ]
        result = OpenAIAPIUser.to_openai_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "tool_calls" in result[0]
        assert len(result[0]["tool_calls"]) == 1
        tc = result[0]["tool_calls"][0]
        assert tc["id"] == "call_abc123"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "ui_user_interact"

    def test_exec_env_to_user_becomes_tool_result(self) -> None:
        """EXEC_ENV→USER with tool metadata becomes 'tool' role message."""
        messages = [
            Message(
                sender=RoleType.EXECUTION_ENVIRONMENT,
                recipient=RoleType.USER,
                content='User Interaction: {"userAction": {}}',
                openai_tool_call_id="call_abc123",
                openai_function_name="ui_user_interact",
            ),
        ]
        result = OpenAIAPIUser.to_openai_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_abc123"
        assert result[0]["name"] == "ui_user_interact"
        assert "User Interaction" in result[0]["content"]

    def test_full_tool_call_round_trip(self) -> None:
        """Full flow: system, agent shows UI, user calls tool, gets result, responds."""
        messages = [
            # System instructs user
            Message(
                sender=RoleType.SYSTEM,
                recipient=RoleType.USER,
                content="You are a user.",
            ),
            # Agent shows UI to user
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.USER,
                content="Here is a booking UI.",
            ),
            # User calls ui_user_interact
            Message(
                sender=RoleType.USER,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content=(
                    'call_xyz_parameters = {"action_name": "submit", "surface_id": "s1", "component_id": "b1"}\n'
                    "call_xyz_response = ui_user_interact(**call_xyz_parameters)"
                ),
                openai_tool_call_id="call_xyz",
                openai_function_name="ui_user_interact",
            ),
            # Execution environment returns tool result
            Message(
                sender=RoleType.EXECUTION_ENVIRONMENT,
                recipient=RoleType.USER,
                content="User Interaction: submitted",
                openai_tool_call_id="call_xyz",
                openai_function_name="ui_user_interact",
            ),
            # User sends follow-up text to agent
            Message(
                sender=RoleType.USER,
                recipient=RoleType.AGENT,
                content="I clicked submit.",
            ),
        ]
        result = OpenAIAPIUser.to_openai_messages(messages)
        assert len(result) == 5
        # system
        assert result[0]["role"] == "system"
        # agent message → user role
        assert result[1]["role"] == "user"
        # user's tool call → assistant with tool_calls
        assert result[2]["role"] == "assistant"
        assert "tool_calls" in result[2]
        assert result[2]["tool_calls"][0]["id"] == "call_xyz"
        # tool result
        assert result[3]["role"] == "tool"
        assert result[3]["tool_call_id"] == "call_xyz"
        # user follow-up → assistant
        assert result[4]["role"] == "assistant"
        assert result[4]["content"] == "I clicked submit."

    def test_parallel_user_tool_calls_aggregated(self) -> None:
        """Multiple consecutive USER→EXEC_ENV messages aggregate into one assistant message."""
        messages = [
            Message(
                sender=RoleType.USER,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content=(
                    'call_a_parameters = {"action_name": "click", "surface_id": "s1", "component_id": "b1"}\n'
                    "call_a_response = ui_user_interact(**call_a_parameters)"
                ),
                openai_tool_call_id="call_a",
                openai_function_name="ui_user_interact",
            ),
            Message(
                sender=RoleType.USER,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content=(
                    "call_b_parameters = {}\n"
                    "call_b_response = end_conversation(**call_b_parameters)"
                ),
                openai_tool_call_id="call_b",
                openai_function_name="end_conversation",
            ),
        ]
        result = OpenAIAPIUser.to_openai_messages(messages)
        # Both should be in a single assistant message
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert len(result[0]["tool_calls"]) == 2
        assert result[0]["tool_calls"][0]["id"] == "call_a"
        assert result[0]["tool_calls"][1]["id"] == "call_b"

    def test_unrecognized_sender_recipient_raises(self) -> None:
        """Unknown message pair raises ValueError."""
        messages = [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content="x",
            ),
        ]
        try:
            OpenAIAPIUser.to_openai_messages(messages)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unrecognized" in str(e)


class TestRuntimeVisibility:
    """Test that user tool calls set visible_to correctly."""

    def test_ui_user_interact_visible_to_agent(self) -> None:
        """Non-end_conversation tool calls include AGENT in visible_to."""
        msg = Message(
            sender=RoleType.USER,
            recipient=RoleType.EXECUTION_ENVIRONMENT,
            content="call_x_parameters = {}\ncall_x_response = ui_user_interact(**call_x_parameters)",
            openai_tool_call_id="call_x",
            openai_function_name="ui_user_interact",
            visible_to=[RoleType.USER, RoleType.EXECUTION_ENVIRONMENT, RoleType.AGENT],
        )
        assert RoleType.AGENT in msg.visible_to  # type: ignore[operator]

    def test_end_conversation_not_visible_to_agent(self) -> None:
        """end_conversation should NOT include AGENT in visible_to."""
        msg = Message(
            sender=RoleType.USER,
            recipient=RoleType.EXECUTION_ENVIRONMENT,
            content="call_x_parameters = {}\ncall_x_response = end_conversation(**call_x_parameters)",
            openai_tool_call_id="call_x",
            openai_function_name="end_conversation",
            visible_to=[RoleType.USER, RoleType.EXECUTION_ENVIRONMENT],
        )
        assert RoleType.AGENT not in msg.visible_to  # type: ignore[operator]


class TestGetMessageViews:
    """Test _get_message_views classification."""

    def test_agent_to_user(self) -> None:
        views = _get_message_views(None, RoleType.AGENT, RoleType.USER)
        assert "agent" in views
        assert "user" in views

    def test_user_to_agent(self) -> None:
        views = _get_message_views(None, RoleType.USER, RoleType.AGENT)
        assert "agent" in views
        assert "user" in views

    def test_agent_to_exec_env(self) -> None:
        views = _get_message_views(None, RoleType.AGENT, RoleType.EXECUTION_ENVIRONMENT)
        assert views == ["agent"]

    def test_exec_env_to_agent(self) -> None:
        views = _get_message_views(None, RoleType.EXECUTION_ENVIRONMENT, RoleType.AGENT)
        assert views == ["agent"]

    def test_system_to_agent(self) -> None:
        views = _get_message_views(None, RoleType.SYSTEM, RoleType.AGENT)
        assert views == ["agent"]

    def test_system_to_user(self) -> None:
        views = _get_message_views(None, RoleType.SYSTEM, RoleType.USER)
        assert views == ["user"]

    def test_user_to_exec_env_default(self) -> None:
        """Default visible_to for USER→EXEC_ENV is user-only."""
        views = _get_message_views(None, RoleType.USER, RoleType.EXECUTION_ENVIRONMENT)
        assert views == ["user"]

    def test_user_to_exec_env_with_agent(self) -> None:
        """When visible_to includes AGENT, views include both."""
        views = _get_message_views(
            [RoleType.USER, RoleType.EXECUTION_ENVIRONMENT, RoleType.AGENT],
            RoleType.USER,
            RoleType.EXECUTION_ENVIRONMENT,
        )
        assert "agent" in views
        assert "user" in views

    def test_exec_env_to_user_default(self) -> None:
        views = _get_message_views(None, RoleType.EXECUTION_ENVIRONMENT, RoleType.USER)
        assert views == ["user"]

    def test_exec_env_to_user_with_agent(self) -> None:
        views = _get_message_views(
            [RoleType.EXECUTION_ENVIRONMENT, RoleType.USER, RoleType.AGENT],
            RoleType.EXECUTION_ENVIRONMENT,
            RoleType.USER,
        )
        assert "agent" in views
        assert "user" in views


class TestFilterMessages:
    """Test that filter_messages correctly includes/excludes messages per the visibility table.

    Visibility table:
    | Message Type                       | Agent | User |
    |------------------------------------|-------|------|
    | Agent system prompt (SYS→AGT)      |  Yes  |  No  |
    | User system prompt (SYS→USR)       |  No   |  Yes |
    | User text to agent (USR→AGT)       |  Yes  |  Yes |
    | Agent text to user (AGT→USR)       |  Yes  |  Yes |
    | Agent tool call (AGT→EXE)          |  Yes  |  No  |
    | Agent tool result (EXE→AGT)        |  Yes  |  No  |
    | User tool call ui_user_interact    |  Yes  |  Yes |
    | User tool result ui_user_interact  |  Yes  |  Yes |
    | User tool call end_conversation    |  No   |  Yes |
    | User tool result end_conversation  |  No   |  Yes |
    """

    ALL_MESSAGES = [
        Message(
            sender=RoleType.SYSTEM,
            recipient=RoleType.AGENT,
            content="agent prompt",
        ),
        Message(
            sender=RoleType.SYSTEM,
            recipient=RoleType.USER,
            content="user prompt",
        ),
        Message(
            sender=RoleType.USER,
            recipient=RoleType.AGENT,
            content="user text",
        ),
        Message(
            sender=RoleType.AGENT,
            recipient=RoleType.USER,
            content="agent text",
        ),
        Message(
            sender=RoleType.AGENT,
            recipient=RoleType.EXECUTION_ENVIRONMENT,
            content="agent tool call",
            openai_tool_call_id="tc1",
            openai_function_name="web_search",
        ),
        Message(
            sender=RoleType.EXECUTION_ENVIRONMENT,
            recipient=RoleType.AGENT,
            content="agent tool result",
            openai_tool_call_id="tc1",
            openai_function_name="web_search",
        ),
        # ui_user_interact: visible to agent
        Message(
            sender=RoleType.USER,
            recipient=RoleType.EXECUTION_ENVIRONMENT,
            content="user ui interact",
            openai_tool_call_id="tc2",
            openai_function_name="ui_user_interact",
            visible_to=[RoleType.USER, RoleType.EXECUTION_ENVIRONMENT, RoleType.AGENT],
        ),
        Message(
            sender=RoleType.EXECUTION_ENVIRONMENT,
            recipient=RoleType.USER,
            content="ui interact result",
            openai_tool_call_id="tc2",
            openai_function_name="ui_user_interact",
            visible_to=[RoleType.USER, RoleType.EXECUTION_ENVIRONMENT, RoleType.AGENT],
        ),
        # end_conversation: NOT visible to agent
        Message(
            sender=RoleType.USER,
            recipient=RoleType.EXECUTION_ENVIRONMENT,
            content="end conv",
            openai_tool_call_id="tc3",
            openai_function_name="end_conversation",
            visible_to=[RoleType.USER, RoleType.EXECUTION_ENVIRONMENT],
        ),
        Message(
            sender=RoleType.EXECUTION_ENVIRONMENT,
            recipient=RoleType.USER,
            content="end conv result",
            openai_tool_call_id="tc3",
            openai_function_name="end_conversation",
            visible_to=[RoleType.EXECUTION_ENVIRONMENT, RoleType.USER],
        ),
    ]

    def test_agent_sees_correct_messages(self) -> None:
        """Agent should see: system→agent, user↔agent, tool calls/results, ui_user_interact."""
        from mmtoolsandbox.roles.openai_agent import OpenAIAPIAgent

        filtered = OpenAIAPIAgent.filter_messages(self.ALL_MESSAGES)
        contents = {m.content for m in filtered}
        # Agent SHOULD see these
        assert "agent prompt" in contents
        assert "user text" in contents
        assert "agent text" in contents
        assert "agent tool call" in contents
        assert "agent tool result" in contents
        assert "user ui interact" in contents
        assert "ui interact result" in contents
        # Agent should NOT see these
        assert "user prompt" not in contents
        assert "end conv" not in contents
        assert "end conv result" not in contents

    def test_user_sees_correct_messages(self) -> None:
        """User should see: system→user, user↔agent, user tool calls/results."""
        filtered = OpenAIAPIUser.filter_messages(self.ALL_MESSAGES)
        contents = {m.content for m in filtered}
        # User SHOULD see these
        assert "user prompt" in contents
        assert "user text" in contents
        assert "agent text" in contents
        assert "user ui interact" in contents
        assert "ui interact result" in contents
        assert "end conv" in contents
        assert "end conv result" in contents
        # User should NOT see these
        assert "agent prompt" not in contents
        assert "agent tool call" not in contents
        assert "agent tool result" not in contents
