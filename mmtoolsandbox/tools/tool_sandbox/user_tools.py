# Copyright © 2026 Apple Inc.

"""A collection of tools dedicated for user access, mostly to support user simulation."""

import polars as pl

from mmtoolsandbox.common.execution_context import (
    DatabaseNamespace,
    RoleType,
    get_current_context,
)
from mmtoolsandbox.common.utils import register_as_tool
from mmtoolsandbox.toolbox.names import ToolboxName

SEND_MESSAGE_WITH_IMAGE_TOOL_NAME = "send_message_with_image"


@register_as_tool(
    # The `end_conversation` tool is deeply tied to the orchestration of the
    # MMToolSandbox and is thus needed for all toolboxes.
    toolboxes=set(ToolboxName),
    database_namespaces={DatabaseNamespace.SANDBOX},
    visible_to=(RoleType.USER,),
)
def end_conversation() -> None:
    """Finish the conversation

    Trigger this tool when you think the agent have completed the task for you,
    or the agent is unable to complete the task. Either way this tool will stop the conversation

    Raises:
        ValueError: If conversation already ended
    """
    current_context = get_current_context()
    sandbox_database = current_context.get_database(DatabaseNamespace.SANDBOX)
    if not sandbox_database["conversation_active"][-1]:
        raise ValueError("Conversation already ended")
    current_context.update_database(
        DatabaseNamespace.SANDBOX,
        dataframe=sandbox_database.with_columns(~pl.col("conversation_active")),
    )


@register_as_tool(
    toolboxes=set(ToolboxName),
    database_namespaces={DatabaseNamespace.SANDBOX},
    visible_to=(RoleType.USER,),
)
def send_message_with_image(message: str, image_ids: str) -> str:
    """Send a message to the agent along with one or more images.

    Use this tool when you need to share an image with the agent as part of your message.

    Args:
        message: The text message to send to the agent alongside the image(s).
        image_ids: A comma-separated string of integer image IDs to attach,
                   e.g. "0" for a single image or "1,2" for multiple images.
    """
    # This tool body should never run — the user simulator intercepts calls to
    # send_message_with_image and converts them into Message objects with
    # image_ids before they reach the execution environment.
    raise RuntimeError(
        "send_message_with_image should be intercepted by the user simulator, "
        "not executed by the execution environment."
    )
