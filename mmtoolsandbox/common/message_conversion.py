# Copyright © 2026 Apple Inc.

"""Serialization bridge between the internal MMToolSandbox message format and LLM API formats.

Defines the ``Message`` attrs class that all sandbox roles read and write, and
the ``to_openai_messages()`` converter that translates message histories into
OpenAI-compatible dictionaries for both tool-calling and code-execution modes.
Also provides ``serialize_to_conversation()`` for trajectory persistence,
``get_image_content()`` for multimodal image handling with automatic
downscaling, and helper functions for Python code / OpenAI tool-call
round-trip conversion.
"""

from __future__ import annotations  # < for type hints from lazy imports without strings

import ast
import base64
import functools
import io
import json
import re
from collections import defaultdict
from itertools import chain
from typing import (
    Any,
    Literal,
    cast,
)

import attrs
import polars as pl
from attrs import define, field
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from strenum import StrEnum

from mmtoolsandbox.common.evaluation import EvaluationResult
from mmtoolsandbox.common.execution_context import (
    DatabaseNamespace,
    ExecutionContext,
    RoleType,
)
from mmtoolsandbox.common.image_id import ImageId

# Tags and markers used when formatting execution results for code-exec mode.
EXECUTION_RESULTS_OPEN_TAG = "<execution_results>"
EXECUTION_RESULTS_CLOSE_TAG = "</execution_results>"
CODE_EXEC_ERROR_PREFIX = "[Error]"

# Tags for UI State Server environment-mediated messages (A2UI extension).
UI_STATE_SERVER_OPEN_TAG = "<ui_state_server>"
UI_STATE_SERVER_CLOSE_TAG = "</ui_state_server>"

# Tag pattern for ReACT-style reasoning traces.
_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# Matches any character not allowed in a Python identifier.  Used to
# normalize tool_call IDs from serving backends that emit non-identifier
# characters (e.g. vLLM's ``chatcmpl-tool-XXX`` with hyphens, Kimi's
# ``functions.NAME:INDEX`` with dots and colons).
_NON_IDENTIFIER_CHAR = re.compile(r"[^a-zA-Z0-9_]")


class ConversionMode(StrEnum):
    """Mode for :func:`to_openai_messages` controlling how tool interactions are
    serialised.

    ``TOOL_CALLING`` – standard function-calling API format (``tool_calls`` /
    ``role: "tool"``).

    ``CODE_EXEC`` – code-execution agent format: assistant text with inline
    code and ``<execution_results>`` XML tags in user messages.
    """

    TOOL_CALLING = "tool_calling"
    CODE_EXEC = "code_exec"


# Pseudo-roles for messages that are only visible to the *user* perspective.
# These appear in the "full conversation" returned by `serialize_to_conversation``
# and are consumed by the trajectory visualizer
# (``viz/templates/trajectory.html``) to render user-side system prompts and
# user-initiated tool interactions with distinct styling. NOT valid
# OpenAI API roles — they exist solely for visualization / debugging.
#
#   USER_SYSTEM      – a system instruction sent to the user simulator
#                      (sender=SYSTEM, recipient=USER)
#   USER_TOOL_CALL   – the user calling a tool on their own behalf
#                      (sender=USER, recipient=EXECUTION_ENVIRONMENT)
#   USER_TOOL_RESULT – the execution-environment response to the user
#                      (sender=EXECUTION_ENVIRONMENT, recipient=USER)
PSEUDO_ROLE_USER_SYSTEM = "user_system"
PSEUDO_ROLE_USER_TOOL_CALL = "user_tool_call"
PSEUDO_ROLE_USER_TOOL_RESULT = "user_tool_result"
PSEUDO_ROLE_ENV_EVENT = "env_event"


# Maximum dimension (width or height) for images sent to the LLM API.
# Images larger than this are downscaled to save context tokens while
# preserving enough detail for the model to understand the content.
_MAX_IMAGE_DIMENSION = 1024


@define(frozen=True)
class Message:
    """Messages each role reads and writes.

    Attributes:
        sender: The role that sent this message.
        recipient: The role that receives this message.
        content: Text content of the message.
        conversation_active: Whether the conversation is still active after this
            message.  None inherits the previous value.
        openai_tool_call_id: OpenAI tool call ID for correlating tool
            calls with results.
        openai_function_name: OpenAI function name for this tool call or result.
        tool_call_exception: Exception string if the tool call raised an error.
        tool_trace: List of JSON-serialized tool trace entries for this message.
        visible_to: Roles that can see this message.  Defaults to sender and
            recipient.
        finish_reason: LLM finish reason from ChatCompletion (e.g. "stop",
            "tool_calls").
        logprobs: Log probabilities from the LLM response, for RL training.
        generation: Raw generation text before post-processing.
        token_ids: Token IDs corresponding to the raw generation text.
        claude_text_response: Text response associated with a Claude tool use
            block.
        claude_extended_thinking: Extended thinking text from Claude.
        claude_extended_thinking_signature: Signature for Claude extended
            thinking verification.
        reasoning_trace: Reasoning text extracted from a native field or inline
            ``<think>`` tags.
        openai_reasoning_content: Reasoning returned in OpenAI-compatible APIs'
            native ``reasoning_content`` extension.
        image_ids: List of ``ImageId`` references attached to this message.
    """

    sender: RoleType = field(converter=RoleType)
    recipient: RoleType = field(converter=RoleType)
    content: str
    conversation_active: bool | None = None
    # Optional fields to support OpenAI API
    openai_tool_call_id: str | None = None
    openai_function_name: str | None = None
    # Optional field for storing exceptions that occurred as part of the tool call.
    tool_call_exception: str | None = None
    # Optional field tracing tool execution for this message
    tool_trace: list[str] | None = None
    # Message visibility. By default, should be visible to sender and recipient
    visible_to: list[RoleType] | None = None
    # In the case of parallel function call where 1 OpenAI message will be split into
    # multiple MMToolSandbox messages, the following will be filled in all corresponding
    # MMToolSandbox messages from OpenAI message.

    # Useful for RL training to determine data validity
    finish_reason: str | None = None
    # Useful for RL training
    logprobs: list[float] | None = None
    # Raw generation text before post-processing. Useful for RL training
    generation: str | None = None
    # Token ids corresponding to raw generation text. Useful for RL training
    token_ids: list[int] | None = None

    # Optional field to support Claude API
    # Text response associated with tool use block
    claude_text_response: str | None = None
    # Extended thinking associated with tool uses block
    claude_extended_thinking: str | None = None
    # Extended thinking signature associated with tool uses block
    claude_extended_thinking_signature: str | None = None

    # Text the model produced alongside tool calls (e.g. "Let me look
    # that up for you").  Used by OpenAI / Qwen agents; Claude uses
    # claude_text_response instead.
    tool_call_text_response: str | None = None

    # Optional field for storing reasoning traces
    reasoning_trace: str | None = None

    # Optional field for round-tripping the Chat Completions reasoning_content extension
    openai_reasoning_content: str | None = None

    # Optional field for storing serialized OpenAI Responses API reasoning
    # output items (JSON).  Includes encrypted_content for round-tripping
    # reasoning state across turns.
    openai_reasoning_items: str | None = None

    # Optional field for image identifiers
    image_ids: list[ImageId] | None = None

    def __attrs_post_init__(self) -> None:
        # Bypass frozen. See https://github.com/python-attrs/attrs/issues/120.
        # Assign default visibility
        if self.visible_to is None:
            object.__setattr__(self, "visible_to", [self.sender, self.recipient])

        # The tool call ID and function name must be either both set or unset.
        if self.openai_tool_call_id is None:
            assert self.openai_function_name is None
        else:
            assert self.openai_function_name is not None


def get_messages_from_execution_context(
    execution_context: ExecutionContext, ending_index: int | None = None
) -> list[Message]:
    """Access database to get all historical messages in execution_context.

    Args:
        execution_context:  Execution context to extract messages from.
        ending_index:       Optional index to provide get_messages. Will truncate message history till ending_index
                            if provided. Utility for processing system message, which could contain multiple entries
                            before each was responded to.

    Returns:
        List of Message object
    """
    sandbox_database = execution_context.get_database(
        namespace=DatabaseNamespace.SANDBOX,
        get_all_history_snapshots=True,
        drop_sandbox_message_index=False,
    )
    if ending_index is not None:
        sandbox_database = sandbox_database.filter(
            pl.col("sandbox_message_index") <= ending_index
        )
    return [
        Message(**row)
        for row in sandbox_database.drop("sandbox_message_index").to_dicts()
    ]


def add_messages_to_execution_context(
    execution_context: ExecutionContext, messages: list[Message]
) -> None:
    """Add a list of Messages to execution_context.

    Args:
        execution_context:  Execution context to extract messages from.
        messages:           Messages to be added to the database
    """
    execution_context.add_to_database(
        namespace=DatabaseNamespace.SANDBOX,
        rows=[attrs.asdict(x) for x in messages],
    )


def normalize_tool_call_id(tool_id: str) -> str:
    """Return a version of *tool_id* that is safe as a Python identifier.

    Non-identifier characters are replaced with underscores.  This is
    needed because ``tool_call_to_python_code`` builds assignments of the
    form ``{tool_id}_parameters = ...`` and different serving backends
    emit IDs with characters that break Python syntax (e.g. ``-`` from
    vLLM, ``.`` and ``:`` from Kimi's native format).
    """
    return _NON_IDENTIFIER_CHAR.sub("_", tool_id)


def openai_tool_call_to_python_code(
    tool_call: ChatCompletionMessageFunctionToolCall,
    available_tool_names: set[str],
    execution_facing_tool_name: str | None,
    raise_error_on_unknown_tools: bool = True,
) -> str:
    """Converts OpenAI ChatCompletionMessageFunctionToolCall to python code

    Args:
        tool_call:                      ChatCompletionMessageFunctionToolCall object
        available_tool_names:           Set of tools available
        execution_facing_tool_name:     The execution facing name of the function. In the
                                        case of tool name scrambling the OpenAI API in- and
                                        outputs are filled with scrambled tool names. When
                                        executing the code we need to use the actual tool
                                        name. If `None` the tool name stored in `tool_call`
                                        will be used.
        raise_error_on_unknown_tools:   If true, trajectory errors out upon an unknown tool
                                        If false, generate python code as is and let the console
                                        raise an error.

    Returns:
        A python code making the tool call

    Raises:
        KeyError:   If the tool name is not a known tool
    """
    tool_id = tool_call.id
    agent_facing_tool_name = tool_call.function.name
    arguments_json = tool_call.function.arguments

    # Check if function name is known allowed tool
    if (
        raise_error_on_unknown_tools
        and agent_facing_tool_name not in available_tool_names
    ):
        raise KeyError(
            f"Agent tool call {agent_facing_tool_name=} is not a known allowed tool. "
            f"Options are {available_tool_names=}"
        )
    function_name = (
        agent_facing_tool_name
        if execution_facing_tool_name is None
        else execution_facing_tool_name
    )
    return tool_call_to_python_code(
        tool_id=tool_id,
        function_name=function_name,
        arguments_json=arguments_json,
    )


def tool_call_to_python_code(
    *, tool_id: str, function_name: str, arguments_json: str
) -> str:
    """Create the Python code for calling a tool.

    Args:
        tool_id:        The tool ID for this call.
        function_name:  The name of the tool.
        arguments_json: The tool arguments as a JSON string.

    Returns:
        The Python code to call the tool.
    """
    try:
        args_dict = json.loads(arguments_json)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Given arguments are not valid JSON: {repr(arguments_json)}"
        ) from e
    return tool_call_with_dict_to_python_code(
        tool_id=tool_id,
        function_name=function_name,
        arguments_dict=args_dict,
    )


def tool_call_with_dict_to_python_code(
    *,
    tool_id: str,
    function_name: str,
    arguments_dict: dict[str, Any],
) -> str:
    """Create the Python code for calling a tool.

    Args:
        tool_id:        The tool ID for this call.
        function_name:  The name of the tool.
        arguments_dict: The dictionary of arguments to pass to the tool.

    Returns:
        The Python code to call the tool.
    """
    # Sanitize tool_id to be a valid Python identifier
    tool_id = normalize_tool_call_id(tool_id)

    return (
        f"{tool_id}_parameters = {arguments_dict}\n"
        f"{tool_id}_response = {function_name}(**{tool_id}_parameters)"
    )


def python_code_to_openai_tool_call(
    python_code: str,
    agent_facing_tool_name: str | None,
) -> ChatCompletionMessageFunctionToolCall:
    """Converts python code to OpenAI ChatCompletionMessageToolCall.

    Execution facing tool_name shall be converted back to agent facing tool_name.

    Args:
        python_code:            A python code making the tool call
        agent_facing_tool_name: The agent facing name of the function. In the
                                case of tool name scrambling the OpenAI API in- and
                                outputs are filled with scrambled tool names. When
                                executing the code we need to use the actual tool
                                name. If `None` the tool name stored in python
                                will be used.

    Returns:
        ChatCompletionMessageToolCall object


    Raises:
        KeyError:   If the tool name is not a known tool
    """
    pattern = r"^(?P<tool_id>.+)_parameters = (?P<arguments>[^\n]+)\n(?P=tool_id)_response = (?P<name>[^\(]+)"
    match = re.match(pattern=pattern, string=python_code)
    if match is None:
        # Free-form code (e.g., from a coding agent) that doesn't follow the standard
        # tool call pattern. Create a synthetic tool call with the code as an argument.
        return ChatCompletionMessageFunctionToolCall(
            id="code_execution",
            type="function",
            function=Function(
                name="execute_code",
                arguments=json.dumps({"code": python_code}, ensure_ascii=False),
            ),
        )
    function_name = (
        match.group("name")
        if agent_facing_tool_name is None
        else agent_facing_tool_name
    )
    return ChatCompletionMessageFunctionToolCall(
        id=match.group("tool_id"),
        type="function",
        function=Function(
            name=function_name,
            arguments=json.dumps(
                ast.literal_eval(match.group("arguments")), ensure_ascii=False
            ),
        ),
    )


def _resize_base64_image(
    base64_content: str, max_dim: int = _MAX_IMAGE_DIMENSION
) -> str:
    """Downscale a base64-encoded image if either dimension exceeds *max_dim*.

    Returns the original base64 string unchanged if no resizing is needed.
    Falls back to the original on any decode/resize error.
    """
    try:
        from PIL import Image as PILImage
        from PIL.Image import Resampling

        raw = base64.b64decode(base64_content)
        img: PILImage.Image = PILImage.open(io.BytesIO(raw))
        w, h = img.size
        if w <= max_dim and h <= max_dim:
            return base64_content

        # Compute new size preserving aspect ratio
        scale = max_dim / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Resampling.LANCZOS)

        buf = io.BytesIO()
        fmt = img.format or "JPEG"
        if fmt.upper() == "PNG" and img.mode == "RGBA":
            img.save(buf, format="PNG")
        else:
            # Convert to RGB for JPEG (handles RGBA, P, L modes)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception:
        # If anything goes wrong (bad image data, missing PIL), return as-is
        return base64_content


def get_image_content(
    initial_content: str | list[dict[str, Any]] | None,
    image_ids: list[ImageId],
    execution_context: ExecutionContext | None,
) -> list[dict[str, Any]]:
    """Convert text content and image IDs into OpenAI multimodal message content format.

    Args:
        initial_content:    Optional initial content to prepend to the content list.
        image_ids:          List of image IDs to fetch from the IMAGE database.
        execution_context:  Execution context for looking up images from the IMAGE database.

    Returns:
        List of content parts with text and base64-encoded images in OpenAI format.
    """
    # User message with images - lookup images from IMAGE database
    if execution_context is None:
        raise ValueError("execution_context is required to fetch image content")

    content: list[dict[str, Any]] = []

    if isinstance(initial_content, str):
        content.append({"type": "text", "text": initial_content})
    elif isinstance(initial_content, list):
        content.extend(initial_content)

    # Get the IMAGE database
    image_db = execution_context.get_database(DatabaseNamespace.IMAGE)

    # For each image, add a new text content with the image id and a new image_url
    # content with the base64-encoded image.
    for image_id in image_ids:
        # Find the image in the database
        image_row = image_db.filter(pl.col("image_id") == image_id)
        if image_row.is_empty():
            raise ValueError(
                f"Image with image_id='{image_id}' not found in IMAGE database"
            )
        image_content = image_row["image_content"][0]

        # Downscale large images to save context tokens
        image_content = _resize_base64_image(image_content)

        content.append({"type": "text", "text": f"image_id: {image_id}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_content}"},
            }
        )

    return content


def extract_reasoning(content: str) -> tuple[str | None, str]:
    """Extract ``<think>`` tags from *content*.

    Returns:
        A tuple of ``(reasoning_text, remaining_content)``.  When no
        ``<think>`` tag is found, *reasoning_text* is ``None`` and
        *remaining_content* is the original *content*.
    """
    match = _THINK_PATTERN.search(content)
    if match:
        reasoning = match.group(1).strip()
        remaining = _THINK_PATTERN.sub("", content).strip()
        return reasoning, remaining
    return None, content


def to_openai_messages_for_code_exec(
    messages: list[Message],
    execution_context: ExecutionContext | None = None,
) -> list[dict[str, Any]]:
    """Convert message history to OpenAI format for code execution agents.

    .. deprecated::
        Use ``to_openai_messages(messages, execution_context, mode=ConversionMode.CODE_EXEC)``
        instead.  This wrapper exists for backward compatibility.
    """
    msgs, _, _ = to_openai_messages(
        messages, execution_context, mode=ConversionMode.CODE_EXEC
    )
    return msgs  # type: ignore[return-value]


def to_openai_messages(
    messages: list[Message],
    execution_context: ExecutionContext | None = None,
    tool_images_as_user_message: bool = False,
    evict_stale_ui_images: bool = False,
    mode: ConversionMode = ConversionMode.TOOL_CALLING,
) -> tuple[
    list[
        dict[
            Literal[
                "role",
                "content",
                "tool_call_id",
                "name",
                "tool_calls",
                "reasoning_content",
            ],
            Any,
        ]
    ],
    list[list[int]],
    list[
        dict[
            Literal[
                "finish_reason",
                "generation",
                "logprobs",
                "token_ids",
                "reasoning_trace",
            ],
            Any,
        ]
        | None
    ],
]:
    """Converts a list of MMToolSandbox messages to OpenAI API messages.

    Multiple sandbox messages could be compressed into a single OpenAI API message. Because of this,
    we return a mapping between the indices of these two in addition. This is useful for serialization.

    In addition, we also return useful metadata for agent actions, coming from ChatCompletion.choices
    that doesn't exist in messages. Including finish reason, raw generation text and logprobs. This is
    primarily used for RL training.

    **mode=TOOL_CALLING** (default) — standard function-calling API format:
    AGENT→EXEC_ENV becomes ``assistant`` with ``tool_calls``, EXEC_ENV→AGENT
    becomes ``role: "tool"``.

    **mode=CODE_EXEC** — code-execution agent format: AGENT→EXEC_ENV becomes a
    plain ``assistant`` message with code content, EXEC_ENV→AGENT becomes a
    ``role: "user"`` message with ``<execution_results>`` XML tags.  Consecutive
    execution results are aggregated into one user message.

    Images in tool call results are added as separate user messages when
    tool_images_as_user_message=True (as is required for models that don't support image
    content in tool messages such as GPT-4o). Example order of messages:
    1. role="assistant" (tool call)
    2. role="tool" (tool call result)
    3. role="user" (image results)

    For parallel tool calls in a single message, the OpenAI API expects sequential tool call
    result messages without any interleaved user messages. To ensure the correct ordering of
    the image results we buffer them while processing and flush them as a single user message
    when all results for a given tool call have been appended. This results in the following order:
    1. role="assistant" (2 parallel tool calls)
    2. role="tool" (first tool call result)
    3. role="tool" (second tool call result)
    4. role="user" (both image results).

    Args:
        messages:                     A list of MMToolSandbox messages
        execution_context:            Optional execution context for looking up images by ID
        tool_images_as_user_message:  Whether to add images as a separate user message after tool message.
                                      Required for models that don't support images in tool messages.
                                      Only used in TOOL_CALLING mode.
        evict_stale_ui_images:        When True, only the LAST render_ui_screen tool result keeps
                                      its image; older renders are replaced with a text placeholder.
                                      Also evicts stale UI documentation tool results (e.g.,
                                      ui_get_quick_start, ui_get_item_details) — only the most
                                      recent result per tool name is kept in full.
                                      Use True for LLM API calls (saves tokens). Use False (default)
                                      for serialization/visualization (preserves all content).
                                      Only used in TOOL_CALLING mode.
        mode:                         Conversion mode — see :class:`ConversionMode`.

    Returns:
        A list of OpenAI API messages, a list of openai_messages index -> messages index mapping
        and choices metadata.
    """
    openai_messages: list[
        dict[
            Literal[
                "role",
                "content",
                "tool_call_id",
                "name",
                "tool_calls",
                "reasoning_content",
            ],
            Any,
        ]
    ] = []
    choices_metadata_list: list[
        dict[
            Literal[
                "finish_reason",
                "generation",
                "logprobs",
                "token_ids",
                "reasoning_trace",
            ],
            Any,
        ]
        | None
    ] = []
    # kth entry in this list contains the list of messages indices openai_messages[k] maps to
    indices_mapping: list[list[int]] = []

    is_code_exec = mode == ConversionMode.CODE_EXEC

    # Buffer for image IDs from tool results that need to be deferred.
    # Each entry is (image_ids, sandbox_message_index)
    deferred_tool_images: list[tuple[list[ImageId], int]] = []

    # CODE_EXEC mode state: deferred AGENT→USER messages injected by tools
    # during execute_code.  These must be flushed after the execution result
    # to preserve the assistant → user (result) ordering.
    deferred_code_exec_messages: list[tuple[dict[str, Any], int]] = []
    awaiting_exec_result = False

    # Pre-scan: find the index of the LAST render_ui_screen tool result with
    # images.  When evict_stale_ui_images is True, only that message retains
    # full image data; older renders get a text placeholder.  This saves
    # tokens for the LLM API while keeping visualization unaffected (which
    # calls this function with evict_stale_ui_images=False).
    #
    # Also track the last occurrence of each UI documentation tool so that
    # older (stale) doc lookups can be replaced with a short placeholder.
    _UI_RENDER_TOOL_NAME = "render_ui_screen"
    _UI_DOC_TOOL_NAMES = frozenset(
        {
            "ui_get_quick_start",
            "ui_get_item_details",
            "ui_list_items",
            "ui_search_docs",
            "ui_explore_capabilities",
        }
    )
    last_ui_render_idx = -1
    _last_ui_doc_idx: dict[str, int] = {}
    if evict_stale_ui_images:
        for scan_idx, scan_msg in enumerate(messages):
            if (
                scan_msg.sender == RoleType.EXECUTION_ENVIRONMENT
                and scan_msg.recipient == RoleType.AGENT
            ):
                fn = getattr(scan_msg, "openai_function_name", None)
                if fn == _UI_RENDER_TOOL_NAME and scan_msg.image_ids:
                    last_ui_render_idx = scan_idx
                elif fn in _UI_DOC_TOOL_NAMES:
                    _last_ui_doc_idx[fn] = scan_idx

    def flush_deferred_tool_images() -> None:
        """Flush all deferred tool images as a single user message to openai_messages."""
        if not tool_images_as_user_message or not deferred_tool_images:
            return

        # Unpack buffered tool results
        nested_image_ids, sandbox_msg_indices = zip(*deferred_tool_images)
        image_ids = list(chain(*nested_image_ids))

        # Create a single user message with all images
        image_content = get_image_content(
            initial_content=None,
            image_ids=image_ids,
            execution_context=execution_context,
        )
        openai_messages.append({"role": "user", "content": image_content})
        choices_metadata_list.append(None)
        indices_mapping.append(list(sandbox_msg_indices))

        deferred_tool_images.clear()

    # Buffer for AGENT→USER messages injected by tools during execution
    # (e.g., show_image_to_user adds AGENT→USER). These must be deferred
    # until after all pending tool results to preserve the assistant (with
    # tool_calls) → tool (results) ordering required by the OpenAI API.
    deferred_agent_to_user: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    pending_tool_call_ids: set[str] = set()

    def flush_deferred_agent_to_user() -> None:
        """Flush deferred AGENT→USER messages once all tool results are in."""
        nonlocal deferred_agent_to_user
        for msg_dict, meta, idx in deferred_agent_to_user:
            openai_messages.append(msg_dict)  # type: ignore[arg-type]
            choices_metadata_list.append(meta)  # type: ignore[arg-type]
            indices_mapping.append([idx])
        deferred_agent_to_user = []

    def flush_deferred_code_exec_messages() -> None:
        """Flush deferred AGENT→USER messages in CODE_EXEC mode."""
        nonlocal deferred_code_exec_messages
        for msg_dict, idx in deferred_code_exec_messages:
            openai_messages.append(msg_dict)  # type: ignore[arg-type]
            choices_metadata_list.append(None)
            indices_mapping.append([idx])
        deferred_code_exec_messages = []

    for i, message in enumerate(messages):
        # Flush deferred tool images before processing any non-tool-result message
        if not (
            message.sender == RoleType.EXECUTION_ENVIRONMENT
            and message.recipient == RoleType.AGENT
        ):
            flush_deferred_tool_images()

        # Process messages
        if message.sender == RoleType.SYSTEM and message.recipient == RoleType.AGENT:
            # Merge consecutive system messages into one to avoid OOD for models
            if openai_messages and openai_messages[-1]["role"] == "system":
                openai_messages[-1]["content"] += "\n\n" + message.content
                # Merge into existing mapping entry — no new openai message was created.
                indices_mapping[-1].append(i)
                continue
            else:
                openai_messages.append({"role": "system", "content": message.content})
                choices_metadata_list.append(None)

        elif message.sender == RoleType.SYSTEM and message.recipient == RoleType.USER:
            # User-simulator instructions — not for the agent. Skip.
            # Note: these are normally pre-filtered by the agent_filter in
            # serialize_to_conversation(), but we handle them defensively here.
            # The User agent has its own to_openai_messages() in openai_api_user.py.
            continue

        elif message.sender == RoleType.USER and message.recipient == RoleType.AGENT:
            # Handle images in user message
            if message.image_ids:
                content = get_image_content(
                    initial_content=message.content,
                    image_ids=message.image_ids,
                    execution_context=execution_context,
                )
                openai_messages.append({"role": "user", "content": content})
                choices_metadata_list.append(None)
            else:
                openai_messages.append({"role": "user", "content": message.content})
                choices_metadata_list.append(None)

        elif (
            message.sender == RoleType.EXECUTION_ENVIRONMENT
            and message.recipient == RoleType.AGENT
        ):
            if is_code_exec:
                # CODE_EXEC mode: render as user message with XML tags
                awaiting_exec_result = False
                result_content = message.content
                if message.tool_call_exception:
                    result_content += (
                        f"\n{CODE_EXEC_ERROR_PREFIX}: {message.tool_call_exception}"
                    )
                result_text = f"{EXECUTION_RESULTS_OPEN_TAG}\n{result_content}\n{EXECUTION_RESULTS_CLOSE_TAG}"

                # If the execution result contains images, build multimodal content
                execution_content: str | list[dict[str, Any]]
                if message.image_ids and execution_context is not None:
                    execution_content = get_image_content(
                        initial_content=result_text,
                        image_ids=message.image_ids,
                        execution_context=execution_context,
                    )
                else:
                    execution_content = result_text

                # Aggregate with previous user message if consecutive results
                if openai_messages and openai_messages[-1]["role"] == "user":
                    prev_content = openai_messages[-1]["content"]
                    if (
                        isinstance(prev_content, str)
                        and EXECUTION_RESULTS_OPEN_TAG in prev_content
                    ):
                        if isinstance(execution_content, list):
                            openai_messages[-1]["content"] = [
                                {"type": "text", "text": prev_content},
                            ] + execution_content
                        else:
                            openai_messages[-1]["content"] += f"\n\n{execution_content}"
                        indices_mapping[-1].append(i)
                        flush_deferred_code_exec_messages()
                        continue
                    elif isinstance(prev_content, list) and any(
                        isinstance(p, dict)
                        and p.get("type") == "text"
                        and EXECUTION_RESULTS_OPEN_TAG in p.get("text", "")
                        for p in prev_content
                    ):
                        if isinstance(execution_content, list):
                            openai_messages[-1]["content"] += execution_content
                        else:
                            openai_messages[-1]["content"].append(
                                {"type": "text", "text": f"\n\n{execution_content}"}
                            )
                        indices_mapping[-1].append(i)
                        flush_deferred_code_exec_messages()
                        continue
                openai_messages.append({"role": "user", "content": execution_content})
                choices_metadata_list.append(None)
                # Flush deferred after appending
                flush_deferred_code_exec_messages()

            elif message.openai_tool_call_id is not None:
                # Tool result for agent's own tool call — render as "tool" role
                assert message.openai_function_name is not None, message

                # Track tool result delivery
                pending_tool_call_ids.discard(message.openai_tool_call_id)

                # Handle images in tool results
                if message.image_ids:
                    # Check if this is a stale render_ui_screen result that
                    # should have its image evicted (only when called from the
                    # agent's LLM inference path, not from visualization).
                    is_stale_ui_render = (
                        evict_stale_ui_images
                        and message.openai_function_name == _UI_RENDER_TOOL_NAME
                        and i != last_ui_render_idx
                    )
                    if is_stale_ui_render:
                        # Evict: keep text content, replace image with
                        # placeholder.  The sandbox DB retains the full image
                        # for visualization.
                        openai_messages.append(
                            {
                                "tool_call_id": message.openai_tool_call_id,
                                "role": "tool",
                                "name": message.openai_function_name,
                                "content": message.content
                                + "\n[UI image omitted — superseded by newer render]",
                            }
                        )
                        choices_metadata_list.append(None)
                    elif tool_images_as_user_message:
                        # Tool images as user message mode: Add tool message and add images to buffer
                        # for later flushing to a user message (see docstring for details).
                        openai_messages.append(
                            {
                                "tool_call_id": message.openai_tool_call_id,
                                "role": "tool",
                                "name": message.openai_function_name,
                                "content": message.content,
                            }
                        )
                        choices_metadata_list.append(None)
                        deferred_tool_images.append((message.image_ids, i))
                    else:
                        # Default mode: Add images inline to the tool message content
                        content = get_image_content(
                            initial_content=message.content,
                            image_ids=message.image_ids,
                            execution_context=execution_context,
                        )
                        openai_messages.append(
                            {
                                "tool_call_id": message.openai_tool_call_id,
                                "role": "tool",
                                "name": message.openai_function_name,
                                "content": content,
                            }
                        )
                        choices_metadata_list.append(None)
                else:
                    # No images — add the tool message normally, but replace
                    # stale UI doc results with a short placeholder to keep
                    # the context window manageable.
                    tool_content = message.content
                    if (
                        _last_ui_doc_idx
                        and message.openai_function_name in _last_ui_doc_idx
                        and i != _last_ui_doc_idx[message.openai_function_name]
                    ):
                        tool_content = (
                            "[UI docs omitted — superseded by a newer lookup]"
                        )
                    openai_messages.append(
                        {
                            "tool_call_id": message.openai_tool_call_id,
                            "role": "tool",
                            "name": message.openai_function_name,
                            "content": tool_content,
                        }
                    )
                    choices_metadata_list.append(None)

            else:
                # Standalone environment notification (e.g., UI interaction
                # summary).  These are NOT tool results — they carry no
                # tool_call_id and therefore must not use the "tool" role
                # (which requires a matching assistant tool_calls entry).
                # Render as "user" role instead.
                if message.image_ids:
                    env_content = get_image_content(
                        initial_content=message.content,
                        image_ids=message.image_ids,
                        execution_context=execution_context,
                    )
                    openai_messages.append({"role": "user", "content": env_content})
                else:
                    openai_messages.append({"role": "user", "content": message.content})
                choices_metadata_list.append(None)

        elif (
            message.sender == RoleType.AGENT
            and message.recipient == RoleType.EXECUTION_ENVIRONMENT
        ):
            if is_code_exec:
                # CODE_EXEC mode: plain assistant message with code content
                awaiting_exec_result = True
                assistant_content = message.content
                if (
                    message.reasoning_trace
                    and message.openai_reasoning_content is None
                ):
                    assistant_content = f"<think>{message.reasoning_trace}</think>\n\n{assistant_content}"
                assistant_message = {"role": "assistant", "content": assistant_content}
                if message.openai_reasoning_content is not None:
                    assistant_message["reasoning_content"] = (
                        message.openai_reasoning_content
                    )
                openai_messages.append(assistant_message)
                choices_metadata_list.append(
                    {
                        "finish_reason": message.finish_reason,
                        "generation": message.generation,
                        "logprobs": message.logprobs,
                        "token_ids": message.token_ids,
                        "reasoning_trace": message.reasoning_trace,
                        "claude_extended_thinking": message.claude_extended_thinking,  # type: ignore[dict-item]
                    }
                )
            else:
                # TOOL_CALLING mode: aggregate multiple function calls
                if not openai_messages or "tool_calls" not in openai_messages[-1]:
                    # Create a new message with tool calls.  Reconstruct
                    # the assistant text content from reasoning traces and
                    # any text response that accompanied the tool calls.
                    parts = []
                    if (
                        message.reasoning_trace
                        and message.openai_reasoning_content is None
                    ):
                        parts.append(f"<think>{message.reasoning_trace}</think>")
                    text_resp = (
                        message.tool_call_text_response or message.claude_text_response
                    )
                    if text_resp:
                        parts.append(text_resp)
                    assistant_content = "\n\n".join(parts) if parts else ""
                    assistant_message = {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": [],
                    }
                    if message.openai_reasoning_content is not None:
                        assistant_message["reasoning_content"] = (
                            message.openai_reasoning_content
                        )
                    openai_messages.append(assistant_message)
                    choices_metadata_list.append(
                        {
                            "finish_reason": message.finish_reason,
                            "generation": message.generation,
                            "logprobs": message.logprobs,
                            "token_ids": message.token_ids,
                            "reasoning_trace": message.reasoning_trace,
                            "claude_extended_thinking": message.claude_extended_thinking,  # type: ignore[dict-item]
                        }
                    )
                # Add tool call
                tool_call_dict = python_code_to_openai_tool_call(
                    message.content, message.openai_function_name
                ).model_dump(mode="dict", exclude_unset=True)
                if message.openai_tool_call_id is not None:
                    tool_call_dict["id"] = message.openai_tool_call_id
                openai_messages[-1]["tool_calls"].append(tool_call_dict)
                # Track pending tool call IDs
                pending_tool_call_ids.add(tool_call_dict["id"])
                # Make sure that parallel tool calls have the same choices metadata
                assert (
                    choices_metadata_list[-1] is not None
                    and choices_metadata_list[-1]["finish_reason"]
                    == message.finish_reason
                    and choices_metadata_list[-1]["generation"] == message.generation
                    and choices_metadata_list[-1]["logprobs"] == message.logprobs
                    and choices_metadata_list[-1]["token_ids"] == message.token_ids
                )
        elif message.sender == RoleType.AGENT and message.recipient == RoleType.USER:
            if is_code_exec and awaiting_exec_result:
                # CODE_EXEC mode: defer tool-injected message until after
                # the execution result to avoid breaking message ordering.
                assistant_content = message.content
                if message.reasoning_trace:
                    assistant_content = f"<think>{message.reasoning_trace}</think>\n\n{assistant_content}"
                deferred_code_exec_messages.append(
                    ({"role": "assistant", "content": assistant_content}, i)
                )
                continue
            elif not is_code_exec and pending_tool_call_ids:
                # Defer: this was injected by a tool during execution
                deferred_agent_to_user.append(
                    (
                        {"role": "assistant", "content": message.content},
                        {
                            "finish_reason": message.finish_reason,
                            "generation": message.generation,
                            "logprobs": message.logprobs,
                            "token_ids": message.token_ids,
                            "reasoning_trace": message.reasoning_trace,
                            "claude_extended_thinking": message.claude_extended_thinking,
                        },
                        i,
                    )
                )
                # Skip mapping — flush_deferred_agent_to_user() handles it.
                continue
            else:
                assistant_content = message.content
                if (
                    is_code_exec
                    and message.reasoning_trace
                    and message.openai_reasoning_content is None
                ):
                    assistant_content = f"<think>{message.reasoning_trace}</think>\n\n{assistant_content}"
                assistant_message = {"role": "assistant", "content": assistant_content}
                if message.openai_reasoning_content is not None:
                    assistant_message["reasoning_content"] = (
                        message.openai_reasoning_content
                    )
                openai_messages.append(assistant_message)
                choices_metadata_list.append(
                    {
                        "finish_reason": message.finish_reason,
                        "generation": message.generation,
                        "logprobs": message.logprobs,
                        "token_ids": message.token_ids,
                        "reasoning_trace": message.reasoning_trace,
                        "claude_extended_thinking": message.claude_extended_thinking,  # type: ignore[dict-item]
                    }
                )
        elif (
            message.sender == RoleType.USER
            and message.recipient == RoleType.EXECUTION_ENVIRONMENT
        ):
            openai_messages.append(_user_tool_call_openai_msg(message))  # type: ignore[arg-type]
            choices_metadata_list.append(None)
        elif (
            message.sender == RoleType.EXECUTION_ENVIRONMENT
            and message.recipient == RoleType.USER
        ):
            openai_messages.append(_user_tool_result_openai_msg(message))  # type: ignore[arg-type]
            choices_metadata_list.append(None)
        else:
            raise ValueError(
                f"Unrecognized sender recipient pair {(message.sender, message.recipient)}"
            )
        # Process mapping
        if (
            message.sender == RoleType.AGENT
            and message.recipient == RoleType.EXECUTION_ENVIRONMENT
            and "tool_calls" in openai_messages[-1]
            and len(openai_messages[-1]["tool_calls"]) > 1
        ):
            # Parallel tool calls - aggregate into existing mapping
            indices_mapping[-1].append(i)
        else:
            # Normal case: one sandbox message -> one openai message
            indices_mapping.append([i])

        # Flush deferred AGENT→USER messages AFTER the current message's
        # mapping entry is in place so the ordering stays correct.
        if not pending_tool_call_ids and deferred_agent_to_user:
            flush_deferred_agent_to_user()

    # Flush any remaining deferred tool images at the end
    flush_deferred_tool_images()

    # Flush any remaining deferred AGENT→USER messages
    flush_deferred_agent_to_user()
    flush_deferred_code_exec_messages()

    return openai_messages, indices_mapping, choices_metadata_list


def openai_messages_to_langchain_messages(
    openai_messages: list[dict[str, Any]],
) -> list[BaseMessage]:
    """Convert OpenAI dict messages to langchain strongly typed messages

    Args:
        openai_messages:    OpenAI messages to convert

    Returns:
        langchain messages between human, assistant and tool
    """
    langchain_messages: list[BaseMessage] = []
    for message in openai_messages:
        if message["role"] == "user":
            langchain_messages.append(HumanMessage(content=message.get("content", "")))
        elif message["role"] == "assistant":
            tool_calls = message.get("tool_calls", [])
            langchain_tool_calls: list[ToolCall] = []
            for tool_call in tool_calls:
                # Langchain args is a dict object instead of json str
                try:
                    args_dict = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError as e:
                    raise RuntimeError(
                        f"Given arguments are not valid JSON: {repr(tool_call['arguments'])}"
                    ) from e
                langchain_tool_calls.append(
                    ToolCall(
                        name=tool_call["function"]["name"],
                        args=args_dict,
                        id=tool_call["id"],
                        type="TOOL_TYPE_FUNCTION",  # type: ignore[typeddict-item]
                    )
                )
            langchain_messages.append(
                AIMessage(
                    content=message.get("content", "")
                    if message.get("content", "") is not None
                    else "",
                    tool_calls=langchain_tool_calls,
                )
            )

        elif message["role"] == "tool":
            langchain_messages.append(
                ToolMessage(
                    tool_call_id=message["tool_call_id"],
                    content=message.get("content", ""),
                )
            )
    return langchain_messages


def get_snapshot_indices_to_databases(
    execution_context: ExecutionContext,
) -> dict[int, dict[str, pl.DataFrame]]:
    """Create a mapping of snapshot index -> database update that happened at said index.

    Args:
        execution_context:  The execution context containing databases.

    Returns:
        Mapping from sandbox message index to a dict of namespace to DataFrame.
    """
    snapshot_indices_to_databases: dict[int, dict[str, pl.DataFrame]] = defaultdict(
        dict
    )
    for namespace in execution_context.get_active_database_namespaces() - {
        DatabaseNamespace.SANDBOX
    }:
        # Find indices where a new snapshot was created, add to the mapping
        update_indices = (
            execution_context.get_database(
                namespace=namespace,
                get_all_history_snapshots=True,
                drop_sandbox_message_index=False,
                drop_headguard=False,
            )
            .select("sandbox_message_index")
            .unique()["sandbox_message_index"]
            .to_list()
        )
        for update_index in update_indices:
            snapshot_indices_to_databases[update_index][namespace] = (
                execution_context.get_database(
                    namespace=namespace,
                    sandbox_message_index=update_index,
                )
            )

    return snapshot_indices_to_databases


def _user_tool_call_openai_msg(
    message: Message,
) -> dict[str, Any]:
    """Format a USER→EXEC_ENV message as an OpenAI 'user' role message."""
    return {
        "role": "user",
        "content": f"[User called tool: {message.openai_function_name}]\n{message.content}",
    }


def _user_tool_result_openai_msg(
    message: Message,
) -> dict[str, Any]:
    """Format an EXEC_ENV→USER message as an OpenAI 'user' role message."""
    return {
        "role": "user",
        "content": f"[Tool result for user ({message.openai_function_name}): {message.content}]",
    }


@functools.lru_cache(maxsize=None)
def _get_message_views_cached(
    visible_to: tuple[RoleType, ...] | None, sender: RoleType, recipient: RoleType
) -> tuple[str, ...]:
    """Return which perspectives can see a message, with LRU caching.

    Accepts tuples (not lists) so that arguments are hashable for caching.
    The public wrapper ``_get_message_views`` converts list → tuple before
    calling this.

    Returns:
        Tuple of view name strings, e.g. ``("agent",)``, ``("agent", "user")``.
    """
    effective: tuple[RoleType, ...] = (
        visible_to if visible_to is not None else (sender, recipient)
    )
    views: list[str] = []
    if RoleType.AGENT in effective:
        views.append("agent")
    if RoleType.USER in effective:
        views.append("user")
    return tuple(views)


def _get_message_views(
    visible_to: list[RoleType] | None, sender: RoleType, recipient: RoleType
) -> list[str]:
    """Return which perspective views a message belongs to based on visible_to.

    Args:
        visible_to: The visible_to field from the message, or None for default.
        sender: The message sender.
        recipient: The message recipient.

    Returns:
        List of view names: "agent", "user", or both.
    """
    return list(
        _get_message_views_cached(
            tuple(visible_to) if visible_to is not None else None,
            sender,
            recipient,
        )
    )


def serialize_to_conversation(
    execution_context: ExecutionContext,
    evaluation_result: EvaluationResult,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Serialize an Execution Context and evaluation result.

    This function serializes the following objects:
        1. Serializes Sandbox database into messages, including the stacktrace if the
           session errored out with an exception.
        2. All other database snapshots, attached to the message where the update
           happened.

    Args:
        execution_context:      The final execution context object after playing out the scenario
        evaluation_result:      Evaluation result of the scenario.

    Returns:
        A tuple of (full_conversation, agent_conversation):

        - **full_conversation**: All messages merged from both agent and user
          perspectives, sorted by ``sandbox_message_index``.  Each message dict
          carries a ``"perspective"`` key with ``views``, ``sender``,
          ``recipient`` and ``sandbox_message_index``.  User-only messages use
          pseudo-roles (see ``PSEUDO_ROLE_*`` constants).
        - **agent_conversation**: Agent-view only messages in standard OpenAI
          format (for ``to_llm_data_conversation`` and backward compat), without
          perspective metadata.
    """
    # Ignore user simulator few-shot messages
    # Fetch full SANDBOX once, then partition into agent-subset and user-only
    full_sandbox_db: pl.DataFrame = execution_context.get_database(
        DatabaseNamespace.SANDBOX,
        drop_sandbox_message_index=False,
        get_all_history_snapshots=True,
    )
    non_fewshot_filter = (pl.col("visible_to") != [RoleType.USER]) | (
        pl.col("visible_to").is_null()
    )
    agent_filter = (
        (pl.col("sender") == RoleType.AGENT) | (pl.col("recipient") == RoleType.AGENT)
    ) & non_fewshot_filter

    message_subset_database = full_sandbox_db.filter(agent_filter)
    subset_to_snapshot_indices_mapping: list[int] = cast(
        list[int], message_subset_database["sandbox_message_index"].to_list()
    )
    message_subset = [
        Message(**row)
        for row in message_subset_database.drop("sandbox_message_index").to_dicts()
    ]
    # Convert SANDBOX database to OpenAI API messages
    if execution_context is not None and execution_context.pure_code_exec:
        # Pure code exec mode: use CODE_EXEC conversion which shows
        # assistant text + <execution_results> tags instead of tool_calls.
        openai_messages_result = to_openai_messages(
            messages=message_subset,
            execution_context=execution_context,
            mode=ConversionMode.CODE_EXEC,
        )
        openai_messages: list[dict[str, Any]] = openai_messages_result[0]  # type: ignore[assignment]
        openai_messages_to_subset_indices_mapping = openai_messages_result[1]
        choices_metadata_list: list[dict[str, Any] | None] = openai_messages_result[2]  # type: ignore[assignment]
    else:
        result = to_openai_messages(
            messages=message_subset,
            execution_context=execution_context,
        )
        openai_messages = result[0]  # type: ignore[assignment]
        openai_messages_to_subset_indices_mapping = result[1]
        choices_metadata_list = result[2]  # type: ignore[assignment]
    # Create a mapping of snapshot index -> database update that happened at said index
    snapshot_indices_to_databases: dict[int, dict[str, pl.DataFrame]] = (
        get_snapshot_indices_to_databases(execution_context)
    )

    # Find a one to many mapping between OpenAI messages and its original snapshot index
    openai_messages_to_snapshot_indices_mapping: list[list[int]] = [
        [subset_to_snapshot_indices_mapping[i] for i in subset_indices]
        for subset_indices in openai_messages_to_subset_indices_mapping
    ]
    messages: list[dict[str, Any]] = []
    for openai_message, snapshot_indices, choices_metadata in zip(
        openai_messages,
        openai_messages_to_snapshot_indices_mapping,
        choices_metadata_list,
    ):
        # Add message
        messages.append(openai_message)

        current_extras_key = f"extras_{openai_message['role']}"
        # Populate metadata
        if choices_metadata is not None:
            # Assign as defaultdict for further extension below
            messages[-1][current_extras_key] = defaultdict(None, choices_metadata)
        for snapshot_index in snapshot_indices:
            # Try to add db update
            if snapshot_index in snapshot_indices_to_databases:
                for database_name, database in snapshot_indices_to_databases[
                    snapshot_index
                ].items():
                    if current_extras_key not in messages[-1]:
                        messages[-1][current_extras_key] = defaultdict()
                    if "database_update" not in messages[-1][current_extras_key]:
                        messages[-1][current_extras_key]["database_update"] = {}
                    messages[-1][current_extras_key]["database_update"][
                        database_name
                    ] = database.to_dicts()

    # --- Build agent_conversation (clean copy without perspective metadata) ---
    agent_conversation = [msg.copy() for msg in messages]

    # --- Add perspective metadata to each agent message ---
    for msg_dict, subset_indices in zip(
        messages, openai_messages_to_subset_indices_mapping
    ):
        # Identify the source Message whose sender/recipient/visible_to
        # determines the perspective metadata for this OpenAI message.
        if msg_dict.get("role") == "tool" and len(subset_indices) == 1:
            perspective_source = message_subset[subset_indices[0]]
        elif msg_dict.get("role") == "tool":
            # Multiple messages grouped into one tool result — find the
            # EXEC_ENV message which is the actual tool result.
            perspective_source = message_subset[subset_indices[0]]
            for idx in subset_indices:
                candidate = message_subset[idx]
                if candidate.sender == RoleType.EXECUTION_ENVIRONMENT:
                    perspective_source = candidate
                    break
        else:
            perspective_source = message_subset[subset_indices[0]]

        views = _get_message_views(
            perspective_source.visible_to,
            perspective_source.sender,
            perspective_source.recipient,
        )
        msg_dict["perspective"] = {
            "views": views,
            "sender": str(perspective_source.sender),
            "recipient": str(perspective_source.recipient),
            "sandbox_message_index": subset_to_snapshot_indices_mapping[
                subset_indices[0]
            ],
        }

    # --- Serialize user-only messages (not in agent subset) ---
    # Query messages visible to USER but NOT already in the agent subset
    agent_snapshot_indices = set(subset_to_snapshot_indices_mapping)
    # Pre-filter at polars level: exclude headguard and already-seen agent messages
    user_only_filter = (
        non_fewshot_filter
        & (pl.col("sandbox_message_index") != 0)
        & ~pl.col("sandbox_message_index").is_in(list(agent_snapshot_indices))
    )
    user_candidates_db = full_sandbox_db.filter(user_only_filter)
    user_only_messages: list[dict[str, Any]] = []
    for row in user_candidates_db.to_dicts():
        sandbox_msg_idx = row["sandbox_message_index"]
        msg = Message(**{k: v for k, v in row.items() if k != "sandbox_message_index"})
        views = _get_message_views(msg.visible_to, msg.sender, msg.recipient)
        if "user" not in views:
            continue  # Not visible to user at all

        # Determine pseudo-role for user-only messages
        if msg.sender == RoleType.SYSTEM and msg.recipient == RoleType.USER:
            user_msg: dict[str, Any] = {
                "role": PSEUDO_ROLE_USER_SYSTEM,
                "content": msg.content,
            }
        elif (
            msg.sender == RoleType.USER
            and msg.recipient == RoleType.EXECUTION_ENVIRONMENT
        ):
            # User tool call — reconstruct tool_calls format
            tool_call_dict = python_code_to_openai_tool_call(
                msg.content, agent_facing_tool_name=msg.openai_function_name
            ).model_dump(mode="dict", exclude_unset=True)
            if msg.openai_tool_call_id is not None:
                tool_call_dict["id"] = msg.openai_tool_call_id
            user_msg = {
                "role": PSEUDO_ROLE_USER_TOOL_CALL,
                "content": "",
                "tool_calls": [tool_call_dict],
            }
        elif (
            msg.sender == RoleType.EXECUTION_ENVIRONMENT
            and msg.recipient == RoleType.USER
        ):
            user_msg = {
                "role": PSEUDO_ROLE_USER_TOOL_RESULT,
                "content": msg.content,
                "name": msg.openai_function_name or "",
                "tool_call_id": msg.openai_tool_call_id or "",
            }
        elif msg.sender == RoleType.AGENT and msg.recipient == RoleType.USER:
            # Agent message visible only to user (e.g., UI image or metadata
            # with <ui_state_server> tag injected by show_ui_to_user).
            env_content: str | list[dict[str, Any]]
            if msg.image_ids and execution_context is not None:
                env_content = get_image_content(
                    initial_content=msg.content if msg.content else None,
                    image_ids=msg.image_ids,
                    execution_context=execution_context,
                )
            else:
                env_content = msg.content
            # env_event is a pseudo-role for visualization only — it is NOT
            # sent to any LLM API. During rollout, the user model sees this
            # as role="user" via its own to_openai_messages(). The pseudo-role
            # lets the trajectory viewer label it "environment" to distinguish
            # it from natural user messages.
            user_msg = {
                "role": PSEUDO_ROLE_ENV_EVENT,
                "content": env_content,
            }
        else:
            continue  # Skip other unrecognized pairs

        user_msg["perspective"] = {
            "views": views,
            "sender": str(msg.sender),
            "recipient": str(msg.recipient),
            "sandbox_message_index": sandbox_msg_idx,
        }
        user_only_messages.append(user_msg)

    # --- Merge agent messages and user-only messages by sandbox_message_index ---
    full_conversation = sorted(
        messages + user_only_messages,
        key=lambda m: m.get("perspective", {}).get("sandbox_message_index", 0),
    )

    return full_conversation, agent_conversation


def serialize_user_conversation(
    execution_context: ExecutionContext,
) -> list[dict[str, Any]]:
    """Serialize the user-facing conversation from an Execution Context.

    Returns the conversation as seen by the user simulator: SYSTEM->USER,
    AGENT->USER, USER->AGENT, and USER<->EXECUTION_ENVIRONMENT messages.
    Roles are shown from the user sim's perspective (agent messages appear
    as "user" role, user sim messages as "assistant" role).

    This is intended for debugging the user simulator's behavior (e.g.
    verifying send_message_with_image tool calls).

    Args:
        execution_context: The final execution context after playing the scenario.

    Returns:
        A list of per-turn dictionary elements in OpenAI message format
        (from the user sim's perspective).
    """
    # Filter to USER-related messages
    sandbox_db: pl.DataFrame = execution_context.get_database(
        DatabaseNamespace.SANDBOX,
        drop_sandbox_message_index=False,
        get_all_history_snapshots=True,
    ).filter(
        (pl.col("sender") == RoleType.USER) | (pl.col("recipient") == RoleType.USER)
    )
    user_messages = [
        Message(**row) for row in sandbox_db.drop("sandbox_message_index").to_dicts()
    ]

    # Convert to OpenAI format from user sim perspective (reversed roles)
    openai_messages: list[dict[str, Any]] = []
    for message in user_messages:
        sender_recipient = (message.sender, message.recipient)
        if sender_recipient == (RoleType.SYSTEM, RoleType.USER):
            openai_messages.append(
                {
                    "role": "system",
                    "content": message.content,
                }
            )
        elif sender_recipient == (RoleType.AGENT, RoleType.USER):
            # Agent messages appear as "user" role to the user sim
            openai_messages.append(
                {
                    "role": "user",
                    "content": message.content,
                }
            )
        elif sender_recipient == (RoleType.USER, RoleType.AGENT):
            msg: dict[str, Any] = {
                "role": "assistant",
                "content": message.content,
            }
            if message.image_ids:
                msg["image_ids"] = [int(i) for i in message.image_ids]
            openai_messages.append(msg)
        elif sender_recipient == (RoleType.USER, RoleType.EXECUTION_ENVIRONMENT):
            # User tool call (e.g. end_conversation, send_message_with_image)
            openai_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "is_tool_call": True,
                    "tool_name": message.openai_function_name,
                }
            )
        elif sender_recipient == (RoleType.EXECUTION_ENVIRONMENT, RoleType.USER):
            # Tool result back to user
            openai_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_name": message.openai_function_name,
                }
            )
        else:
            # Skip unexpected pairs
            continue

    return openai_messages
