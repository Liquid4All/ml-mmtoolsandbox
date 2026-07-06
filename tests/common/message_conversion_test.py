# Copyright © 2026 Apple Inc.

"""Unit tests for mmtoolsandbox.roles.message_conversion"""

from typing import Tuple

import pytest
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from mmtoolsandbox.common.execution_context import (
    DatabaseNamespace,
    ExecutionContext,
    RoleType,
)
from mmtoolsandbox.common.image_id import ImageId
from mmtoolsandbox.common.message_conversion import (
    Message,
    add_messages_to_execution_context,
    get_messages_from_execution_context,
    openai_tool_call_to_python_code,
    python_code_to_openai_tool_call,
    to_openai_messages,
)


@pytest.fixture
def python_code_and_openai_tool_call() -> Tuple[str, ChatCompletionMessageToolCall]:
    """Contains test datapoint for OpenAI tool call and python code conversion

    Returns:
        A tuple of ChatCompletionMessageToolCall and corresponding python code
    """
    tool_id = "call_1293YXHD"
    name = "test"
    arguments = '{"a": 42}'
    return (
        "call_1293YXHD_parameters = {'a': 42}\n"
        "call_1293YXHD_response = test(**call_1293YXHD_parameters)"
    ), ChatCompletionMessageToolCall(
        id=tool_id,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


def test_openai_tool_call_to_python_code(
    python_code_and_openai_tool_call: Tuple[str, ChatCompletionMessageToolCall],
) -> None:
    python_code, tool_call = python_code_and_openai_tool_call
    assert (
        openai_tool_call_to_python_code(
            tool_call, {"test"}, execution_facing_tool_name=None
        )
        == python_code
    )


def test_python_code_to_openai_tool_call(
    python_code_and_openai_tool_call: Tuple[str, ChatCompletionMessageToolCall],
) -> None:
    python_code, tool_call = python_code_and_openai_tool_call
    assert python_code_to_openai_tool_call(python_code, None) == tool_call


def test_message_construction() -> None:
    message = Message(sender="AGENT", recipient="USER", content="Hi")
    # Test sender recipient converter
    assert isinstance(message.sender, RoleType)
    assert isinstance(message.recipient, RoleType)
    # Test Message post init hook
    assert message.visible_to == [RoleType.AGENT, RoleType.USER]
    # Test visible_to converter
    message = Message(
        sender=RoleType.AGENT,
        recipient=RoleType.USER,
        content="Hi",
        visible_to=[RoleType.USER],
    )
    assert message.visible_to == [RoleType.USER]


@pytest.fixture
def execution_context_with_images() -> ExecutionContext:
    """Create an ExecutionContext with IMAGE database populated.

    Returns:
        ExecutionContext with test images in the IMAGE database.
    """
    context = ExecutionContext(support_images=True)
    # Add test images to IMAGE database
    context.add_to_database(
        namespace=DatabaseNamespace.IMAGE,
        rows=[
            {"image_id": 1, "image_content": "base64encodedimage1"},
            {"image_id": 2, "image_content": "base64encodedimage2"},
        ],
    )
    return context


def test_tool_images_as_user_message(
    execution_context_with_images: ExecutionContext,
) -> None:
    """Test tool_images_as_user_message parameter with multiple images."""
    messages = [
        Message(
            sender=RoleType.EXECUTION_ENVIRONMENT,
            recipient=RoleType.AGENT,
            content="Found 2 images",
            openai_tool_call_id="call_456",
            openai_function_name="search_images",
            image_ids=[ImageId(1), ImageId(2)],
        )
    ]

    # Test with tool_images_as_user_message=True (separate user message)
    openai_messages_as_user_msg, indices_mapping_as_user_msg, _ = to_openai_messages(
        messages, execution_context_with_images, tool_images_as_user_message=True
    )

    expected_messages_as_user_msg = [
        {
            "role": "tool",
            "tool_call_id": "call_456",
            "name": "search_images",
            "content": "Found 2 images",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "image_id: 1"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,base64encodedimage1"},
                },
                {"type": "text", "text": "image_id: 2"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,base64encodedimage2"},
                },
            ],
        },
    ]

    assert openai_messages_as_user_msg == expected_messages_as_user_msg
    assert indices_mapping_as_user_msg == [[0], [0]]

    # Test with tool_images_as_user_message=False (inline in tool message)
    openai_messages_inline, indices_mapping_inline, _ = to_openai_messages(
        messages, execution_context_with_images, tool_images_as_user_message=False
    )

    expected_messages_inline = [
        {
            "role": "tool",
            "tool_call_id": "call_456",
            "name": "search_images",
            "content": [
                {"type": "text", "text": "Found 2 images"},
                {"type": "text", "text": "image_id: 1"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,base64encodedimage1"},
                },
                {"type": "text", "text": "image_id: 2"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,base64encodedimage2"},
                },
            ],
        },
    ]

    assert openai_messages_inline == expected_messages_inline
    assert indices_mapping_inline == [[0]]


@pytest.fixture
def execution_context_with_parallel_tool_calls_and_images() -> ExecutionContext:
    """Create an ExecutionContext simulating parallel tool calls with image results.

    This simulates the scenario where:
    1. Agent makes 2 parallel tool calls
    2. Both tools return results with images

    Returns:
        ExecutionContext with SANDBOX and IMAGE databases populated.
    """
    context = ExecutionContext(support_images=True)

    # Add images to IMAGE database
    context.add_to_database(
        namespace=DatabaseNamespace.IMAGE,
        rows=[
            {"image_id": 0, "image_content": "base64encodedimage1"},
            {"image_id": 1, "image_content": "base64encodedimage2"},
        ],
    )

    # Add messages simulating parallel tool calls
    # First, add the agent's parallel tool calls (2 tool calls in one turn)
    messages_tool_calls = [
        Message(
            sender=RoleType.AGENT,
            recipient=RoleType.EXECUTION_ENVIRONMENT,
            content="call_123_parameters = {'query': 'search1'}\ncall_123_response = search_tool(**call_123_parameters)",
            openai_tool_call_id="call_123",
            openai_function_name="search_tool",
            finish_reason="tool_calls",
            generation=None,
            logprobs=None,
            token_ids=None,
        ),
        Message(
            sender=RoleType.AGENT,
            recipient=RoleType.EXECUTION_ENVIRONMENT,
            content="call_456_parameters = {'query': 'search2'}\ncall_456_response = search_tool(**call_456_parameters)",
            openai_tool_call_id="call_456",
            openai_function_name="search_tool",
            finish_reason="tool_calls",
            generation=None,
            logprobs=None,
            token_ids=None,
        ),
    ]
    add_messages_to_execution_context(context, messages_tool_calls)

    # Add the execution environment's responses with images
    messages_tool_results = [
        Message(
            sender=RoleType.EXECUTION_ENVIRONMENT,
            recipient=RoleType.AGENT,
            content="Found result 1",
            openai_tool_call_id="call_123",
            openai_function_name="search_tool",
            image_ids=[ImageId(0)],
        ),
        Message(
            sender=RoleType.EXECUTION_ENVIRONMENT,
            recipient=RoleType.AGENT,
            content="Found result 2",
            openai_tool_call_id="call_456",
            openai_function_name="search_tool",
            image_ids=[ImageId(1)],
        ),
    ]
    add_messages_to_execution_context(context, messages_tool_results)

    return context


def test_parallel_tool_calls_with_images_ordering(
    execution_context_with_parallel_tool_calls_and_images: ExecutionContext,
) -> None:
    """Test that tool results are grouped before user messages with images.

    When tool_images_as_user_message=True, the OpenAI message ordering should be:
    1. Assistant message with tool_calls
    2. All tool result messages (role="tool")
    3. All user messages with images (role="user")

    This test verifies that tool results are NOT interleaved with user messages.
    """
    messages = get_messages_from_execution_context(
        execution_context_with_parallel_tool_calls_and_images
    )

    openai_messages, indices_mapping, _ = to_openai_messages(
        messages,
        execution_context_with_parallel_tool_calls_and_images,
        tool_images_as_user_message=True,
    )

    # Expected message ordering:
    # 1. assistant message with tool_calls
    # 2. tool message for call_123
    # 3. tool message for call_456
    # 4. user message with both images (from call_123 and call_456)

    # Extract roles
    roles = [msg["role"] for msg in openai_messages]

    print(f"Message roles: {roles}")
    print("OpenAI messages:")
    for i, msg in enumerate(openai_messages):
        print(f"  {i}: role={msg['role']}, keys={list(msg.keys())}")

    # Assert the correct ordering: all tool messages before any user messages
    # Find indices of tool and user messages
    tool_indices = [i for i, role in enumerate(roles) if role == "tool"]
    user_indices = [i for i, role in enumerate(roles) if role == "user"]

    # All tool messages should come before all user messages
    if tool_indices and user_indices:
        max_tool_index = max(tool_indices)
        min_user_index = min(user_indices)

        assert max_tool_index < min_user_index, (
            f"Tool messages and user messages are interleaved. "
            f"Tool message indices: {tool_indices}, User message indices: {user_indices}. "
            f"Expected all tool messages (max index {max_tool_index}) to come before "
            f"all user messages (min index {min_user_index})."
        )

    # Verify the expected structure
    assert len(openai_messages) == 4, f"Expected 4 messages, got {len(openai_messages)}"
    assert roles == ["assistant", "tool", "tool", "user"], (
        f"Expected roles ['assistant', 'tool', 'tool', 'user'], got {roles}"
    )

    # Verify the assistant message has both tool calls
    assert openai_messages[0]["role"] == "assistant"
    assert "tool_calls" in openai_messages[0]
    assert len(openai_messages[0]["tool_calls"]) == 2

    # Verify tool messages
    assert openai_messages[1]["role"] == "tool"
    assert openai_messages[1]["tool_call_id"] == "call_123"
    assert openai_messages[1]["content"] == "Found result 1"

    assert openai_messages[2]["role"] == "tool"
    assert openai_messages[2]["tool_call_id"] == "call_456"
    assert openai_messages[2]["content"] == "Found result 2"

    # Verify single user message with both images
    assert openai_messages[3]["role"] == "user"
    assert isinstance(openai_messages[3]["content"], list)
    # Should have 2 images in the content
    image_items = [
        item for item in openai_messages[3]["content"] if "image_url" in item
    ]
    assert len(image_items) == 2, f"Expected 2 images, got {len(image_items)}"

    # Verify indices_mapping
    # Sandbox messages: [0, 1] are parallel tool calls, [2, 3] are tool results
    # OpenAI messages: [0] assistant, [1] tool, [2] tool, [3] user with images
    assert len(indices_mapping) == 4, f"Expected 4 mappings, got {len(indices_mapping)}"

    # Assistant message should map to both parallel tool call sandbox messages
    assert indices_mapping[0] == [0, 1], (
        f"Assistant message should map to sandbox messages [0, 1], got {indices_mapping[0]}"
    )

    # First tool message should map to first tool result sandbox message
    assert indices_mapping[1] == [2], (
        f"First tool message should map to sandbox message [2], got {indices_mapping[1]}"
    )

    # Second tool message should map to second tool result sandbox message
    assert indices_mapping[2] == [3], (
        f"Second tool message should map to sandbox message [3], got {indices_mapping[2]}"
    )

    # User message with images should map to both tool result sandbox messages (deferred then flushed)
    assert indices_mapping[3] == [2, 3], (
        f"User message with images should map to sandbox messages [2, 3], got {indices_mapping[3]}"
    )


def test_reasoning_trace_preserved_in_to_openai_messages() -> None:
    """Reasoning trace should appear in assistant content when tool_calls are present."""
    messages = [
        Message(sender=RoleType.USER, recipient=RoleType.AGENT, content="Hi"),
        Message(
            sender=RoleType.AGENT,
            recipient=RoleType.EXECUTION_ENVIRONMENT,
            content=(
                "call_1_parameters = {}\n"
                "call_1_response = search_contacts(**call_1_parameters)"
            ),
            openai_tool_call_id="call_1",
            openai_function_name="search_contacts",
            reasoning_trace="I need to find the contact first.",
        ),
    ]
    openai_msgs, _, _ = to_openai_messages(messages)
    assistant_msg = openai_msgs[1]
    assert assistant_msg["role"] == "assistant"
    assert (
        assistant_msg["content"] == "<think>I need to find the contact first.</think>"
    )
    assert len(assistant_msg["tool_calls"]) == 1


def test_no_reasoning_trace_gives_empty_content() -> None:
    """Without reasoning_trace, assistant content should be empty string."""
    messages = [
        Message(sender=RoleType.USER, recipient=RoleType.AGENT, content="Hi"),
        Message(
            sender=RoleType.AGENT,
            recipient=RoleType.EXECUTION_ENVIRONMENT,
            content=(
                "call_1_parameters = {}\ncall_1_response = foo(**call_1_parameters)"
            ),
            openai_tool_call_id="call_1",
            openai_function_name="foo",
        ),
    ]
    openai_msgs, _, _ = to_openai_messages(messages)
    assert openai_msgs[1]["content"] == ""


def test_reasoning_trace_on_message_field() -> None:
    """Verify the reasoning_trace field works on Message."""
    msg = Message(
        sender=RoleType.AGENT,
        recipient=RoleType.USER,
        content="Hello",
        reasoning_trace="I should greet the user.",
    )
    assert msg.reasoning_trace == "I should greet the user."

    msg_none = Message(sender=RoleType.AGENT, recipient=RoleType.USER, content="Hello")
    assert msg_none.reasoning_trace is None


def test_deferred_agent_to_user_indices_mapping_sync() -> None:
    """Deferred AGENT→USER messages must not create duplicate indices_mapping entries.

    When the agent makes a tool call and a tool injects an AGENT→USER message
    (e.g., show_ui_to_user), the injected message is deferred until all tool
    results are in. The mapping entry must be created only once — at flush time,
    not at defer time.
    """
    messages = [
        # System prompt
        Message(
            sender=RoleType.SYSTEM,
            recipient=RoleType.AGENT,
            content="You are an agent.",
        ),
        # Second system prompt (triggers merge)
        Message(
            sender=RoleType.SYSTEM,
            recipient=RoleType.AGENT,
            content="Use UI tools.",
        ),
        # User message
        Message(
            sender=RoleType.USER,
            recipient=RoleType.AGENT,
            content="Create a card.",
        ),
        # Agent tool call
        Message(
            sender=RoleType.AGENT,
            recipient=RoleType.EXECUTION_ENVIRONMENT,
            content=(
                "call_1_parameters = {'id': 's1'}\n"
                "call_1_response = show_ui(**call_1_parameters)"
            ),
            openai_tool_call_id="call_1",
            openai_function_name="show_ui",
            finish_reason="tool_calls",
        ),
        # Injected AGENT→USER (deferred because call_1 result is pending)
        Message(
            sender=RoleType.AGENT,
            recipient=RoleType.USER,
            content="Here is your card!",
        ),
        # Tool result (clears pending, flushes deferred)
        Message(
            sender=RoleType.EXECUTION_ENVIRONMENT,
            recipient=RoleType.AGENT,
            content="UI shown to user.",
            openai_tool_call_id="call_1",
            openai_function_name="show_ui",
        ),
        # User response
        Message(
            sender=RoleType.USER,
            recipient=RoleType.AGENT,
            content="Looks great!",
        ),
    ]

    openai_msgs, indices_mapping, choices_metadata = to_openai_messages(messages)

    # CRITICAL: mapping must be in sync with messages
    assert len(openai_msgs) == len(indices_mapping), (
        f"indices_mapping desync: {len(openai_msgs)} openai messages "
        f"but {len(indices_mapping)} mapping entries"
    )
    assert len(openai_msgs) == len(choices_metadata), (
        f"choices_metadata desync: {len(openai_msgs)} openai messages "
        f"but {len(choices_metadata)} metadata entries"
    )

    # Verify expected message roles:
    # system (merged), user, assistant (tool_calls), tool, assistant (deferred), user
    roles = [msg["role"] for msg in openai_msgs]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "user"]

    # Verify mapping ORDER: each openai message maps to the correct source.
    # Messages list (indices 0-6):
    #   0: SYSTEM→AGENT  "You are an agent."
    #   1: SYSTEM→AGENT  "Use UI tools."        (merged into 0)
    #   2: USER→AGENT    "Create a card."
    #   3: AGENT→EE      tool call (show_ui)
    #   4: AGENT→USER    "Here is your card!"   (deferred)
    #   5: EE→AGENT      "UI shown to user."    (tool result)
    #   6: USER→AGENT    "Looks great!"
    #
    # OpenAI messages (indices 0-5):
    #   0: system  (merged 0+1)  → mapping [0, 1]
    #   1: user    (msg 2)       → mapping [2]
    #   2: assistant+tool_calls  → mapping [3]
    #   3: tool    (msg 5)       → mapping [5]
    #   4: assistant (deferred)  → mapping [4]
    #   5: user    (msg 6)       → mapping [6]
    assert indices_mapping[0] == [0, 1], f"system: {indices_mapping[0]}"
    assert indices_mapping[1] == [2], f"user: {indices_mapping[1]}"
    assert indices_mapping[2] == [3], f"assistant tool_calls: {indices_mapping[2]}"
    assert indices_mapping[3] == [5], (
        f"tool result should map to msg 5 (EE→AGENT), got {indices_mapping[3]}"
    )
    assert indices_mapping[4] == [4], (
        f"deferred assistant should map to msg 4 (AGENT→USER), got {indices_mapping[4]}"
    )
    assert indices_mapping[5] == [6], f"user response: {indices_mapping[5]}"


def test_env_event_not_visible_to_agent() -> None:
    """AGENT→USER with visible_to=[USER] must not appear in agent's view.

    When the environment injects a metadata message (e.g., UI interactive
    elements) as AGENT→USER with visible_to=[USER], the agent must never
    see it. This tests the visibility filtering, not the serialization.
    """
    messages = [
        Message(sender=RoleType.SYSTEM, recipient=RoleType.AGENT, content="Prompt"),
        Message(sender=RoleType.USER, recipient=RoleType.AGENT, content="Hello"),
        Message(sender=RoleType.AGENT, recipient=RoleType.USER, content="Hi!"),
        # Metadata message: AGENT→USER but user-only
        Message(
            sender=RoleType.AGENT,
            recipient=RoleType.USER,
            content="<ui_state_server>\nInteractive Elements:\n  Button\n</ui_state_server>",
            visible_to=[RoleType.USER],
        ),
    ]

    # Filter for agent visibility (same as BaseRole.filter_messages)
    agent_visible = [
        m for m in messages if m.visible_to and RoleType.AGENT in m.visible_to
    ]
    assert len(agent_visible) == 3, (
        f"Expected 3 agent-visible messages, got {len(agent_visible)}"
    )

    openai_msgs, _, _ = to_openai_messages(agent_visible)
    # None of the agent's messages should contain ui_state_server content
    for msg in openai_msgs:
        content_str = str(msg.get("content", ""))
        assert "ui_state_server" not in content_str, (
            f"Agent should not see <ui_state_server> content: {content_str[:100]}"
        )


# --- UI Image Eviction Tests ---


def _make_render_ui_tool_pair(call_id: str, image_id: int) -> Tuple[Message, Message]:
    """Helper: create a render_ui_screen tool call + tool result pair."""
    tool_call = Message(
        sender=RoleType.AGENT,
        recipient=RoleType.EXECUTION_ENVIRONMENT,
        content=f"{call_id}_params = {{}}\n{call_id}_resp = render_ui_screen(**{call_id}_params)",
        openai_tool_call_id=call_id,
        openai_function_name="render_ui_screen",
        finish_reason="tool_calls",
    )
    tool_result = Message(
        sender=RoleType.EXECUTION_ENVIRONMENT,
        recipient=RoleType.AGENT,
        content=f'{{"image": "ImageResult(image_id={image_id})", "ui_interactive_elements": "Button"}}',
        openai_tool_call_id=call_id,
        openai_function_name="render_ui_screen",
        image_ids=[ImageId(image_id)],
    )
    return tool_call, tool_result


@pytest.fixture
def execution_context_with_ui_images() -> ExecutionContext:
    """ExecutionContext with 3 images for UI eviction tests."""
    context = ExecutionContext(support_images=True)
    context.add_to_database(
        namespace=DatabaseNamespace.IMAGE,
        rows=[
            {"image_id": 1, "image_content": "base64_ui_render_1"},
            {"image_id": 2, "image_content": "base64_ui_render_2"},
            {"image_id": 3, "image_content": "base64_ui_render_3"},
            {"image_id": 10, "image_content": "base64_user_photo"},
            {"image_id": 20, "image_content": "base64_view_image_result"},
        ],
    )
    return context


def test_stale_render_ui_screen_images_evicted(
    execution_context_with_ui_images: ExecutionContext,
) -> None:
    """With evict_stale_ui_images=True, only the latest render_ui_screen
    image should be kept. Older renders get a text placeholder."""
    call1, result1 = _make_render_ui_tool_pair("call_r1", 1)
    call2, result2 = _make_render_ui_tool_pair("call_r2", 2)
    messages = [call1, result1, call2, result2]

    openai_msgs, _, _ = to_openai_messages(
        messages, execution_context_with_ui_images, evict_stale_ui_images=True
    )

    # Find tool result messages
    tool_results = [m for m in openai_msgs if m.get("role") == "tool"]
    assert len(tool_results) == 2

    # First render result: evicted (text placeholder, no image)
    first_result = tool_results[0]
    assert "[UI image omitted" in str(first_result["content"])
    assert isinstance(first_result["content"], str), (
        "Evicted result should be plain text, not multimodal"
    )

    # Second render result: kept (has image data)
    second_result = tool_results[1]
    content = second_result["content"]
    # Content should be multimodal (list) with image_url entries
    assert isinstance(content, list), (
        "Latest render result should have multimodal content with images"
    )
    has_image = any(
        item.get("type") == "image_url" for item in content if isinstance(item, dict)
    )
    assert has_image, "Latest render result must include inline image"


def test_non_ui_tool_images_not_evicted(
    execution_context_with_ui_images: ExecutionContext,
) -> None:
    """Images from non-render_ui_screen tools must never be evicted,
    even when evict_stale_ui_images=True."""
    # A view_image tool call + result with image
    view_call = Message(
        sender=RoleType.AGENT,
        recipient=RoleType.EXECUTION_ENVIRONMENT,
        content="call_v1_params = {}\ncall_v1_resp = view_image(**call_v1_params)",
        openai_tool_call_id="call_v1",
        openai_function_name="view_image",
        finish_reason="tool_calls",
    )
    view_result = Message(
        sender=RoleType.EXECUTION_ENVIRONMENT,
        recipient=RoleType.AGENT,
        content="Image displayed",
        openai_tool_call_id="call_v1",
        openai_function_name="view_image",
        image_ids=[ImageId(20)],
    )
    # Also a render_ui_screen call after it
    render_call, render_result = _make_render_ui_tool_pair("call_r1", 1)

    messages = [view_call, view_result, render_call, render_result]

    openai_msgs, _, _ = to_openai_messages(
        messages, execution_context_with_ui_images, evict_stale_ui_images=True
    )

    tool_results = [m for m in openai_msgs if m.get("role") == "tool"]
    assert len(tool_results) == 2

    # view_image result: must still have image (NOT evicted)
    view_result_msg = tool_results[0]
    content = view_result_msg["content"]
    assert isinstance(content, list), (
        "view_image result should have multimodal content (not evicted)"
    )
    has_image = any(
        item.get("type") == "image_url" for item in content if isinstance(item, dict)
    )
    assert has_image, "view_image result must retain its image"


def test_user_provided_images_not_evicted(
    execution_context_with_ui_images: ExecutionContext,
) -> None:
    """User-sent images (USER→AGENT) must never be evicted by UI eviction."""
    user_msg_with_image = Message(
        sender=RoleType.USER,
        recipient=RoleType.AGENT,
        content="Here's a photo of the restaurant",
        image_ids=[ImageId(10)],
    )
    # Two render cycles after user's image
    call1, result1 = _make_render_ui_tool_pair("call_r1", 1)
    call2, result2 = _make_render_ui_tool_pair("call_r2", 2)
    messages = [user_msg_with_image, call1, result1, call2, result2]

    openai_msgs, _, _ = to_openai_messages(
        messages, execution_context_with_ui_images, evict_stale_ui_images=True
    )

    # The first message should be the user's message with image preserved
    user_openai = openai_msgs[0]
    assert user_openai["role"] == "user"
    content = user_openai["content"]
    assert isinstance(content, list), (
        "User message with image should have multimodal content"
    )
    has_image = any(
        item.get("type") == "image_url" for item in content if isinstance(item, dict)
    )
    assert has_image, "User-provided image must never be evicted"


def test_single_render_ui_screen_preserved(
    execution_context_with_ui_images: ExecutionContext,
) -> None:
    """A single render_ui_screen call should keep its image (nothing to evict)."""
    call1, result1 = _make_render_ui_tool_pair("call_r1", 1)
    messages = [call1, result1]

    openai_msgs, _, _ = to_openai_messages(
        messages, execution_context_with_ui_images, evict_stale_ui_images=True
    )

    tool_results = [m for m in openai_msgs if m.get("role") == "tool"]
    assert len(tool_results) == 1
    content = tool_results[0]["content"]
    assert isinstance(content, list), (
        "Single render result should have multimodal content with image"
    )
    has_image = any(
        item.get("type") == "image_url" for item in content if isinstance(item, dict)
    )
    assert has_image, "Single render must keep its image"


def test_eviction_with_tool_images_as_user_message(
    execution_context_with_ui_images: ExecutionContext,
) -> None:
    """With tool_images_as_user_message=True, stale UI render images
    should NOT be added to the deferred image buffer."""
    call1, result1 = _make_render_ui_tool_pair("call_r1", 1)
    call2, result2 = _make_render_ui_tool_pair("call_r2", 2)
    messages = [call1, result1, call2, result2]

    openai_msgs, _, _ = to_openai_messages(
        messages,
        execution_context_with_ui_images,
        tool_images_as_user_message=True,
        evict_stale_ui_images=True,
    )

    # Should have: assistant, tool (evicted), assistant, tool (text-only), user (latest image)
    # The deferred user message should only contain the LATEST render's image
    user_msgs = [m for m in openai_msgs if m.get("role") == "user"]
    assert len(user_msgs) == 1, "Should have exactly one deferred user image message"

    # The user message should only have the latest image (image_id=2)
    content = user_msgs[0]["content"]
    assert isinstance(content, list)
    image_urls = [
        item
        for item in content
        if isinstance(item, dict) and item.get("type") == "image_url"
    ]
    assert len(image_urls) == 1, (
        f"Should have exactly 1 image (latest only), got {len(image_urls)}"
    )


def test_visualization_preserves_all_images(
    execution_context_with_ui_images: ExecutionContext,
) -> None:
    """With evict_stale_ui_images=False (default, used by serialize_to_conversation),
    ALL render_ui_screen images must be preserved."""
    call1, result1 = _make_render_ui_tool_pair("call_r1", 1)
    call2, result2 = _make_render_ui_tool_pair("call_r2", 2)
    messages = [call1, result1, call2, result2]

    # Default: evict_stale_ui_images=False (same as serialize_to_conversation)
    openai_msgs, _, _ = to_openai_messages(messages, execution_context_with_ui_images)

    tool_results = [m for m in openai_msgs if m.get("role") == "tool"]
    assert len(tool_results) == 2

    # BOTH render results should have images (no eviction)
    for idx, result_msg in enumerate(tool_results):
        content = result_msg["content"]
        assert isinstance(content, list), (
            f"Tool result {idx} should have multimodal content (no eviction)"
        )
        has_image = any(
            item.get("type") == "image_url"
            for item in content
            if isinstance(item, dict)
        )
        assert has_image, (
            f"Tool result {idx} must retain image when eviction is disabled"
        )
