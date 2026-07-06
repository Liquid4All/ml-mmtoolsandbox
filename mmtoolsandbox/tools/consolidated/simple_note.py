# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""Consolidated Simple Note tools for the MEDIUM toolbox.

CRUD consolidation for notes: create, show, update, delete merged into
a single ``simple_note_manage_note`` tool.
"""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.toolbox.names import ToolboxName
from mmtoolsandbox.tools.consolidated import mark_tools_absorbed_by


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.appworld.simple_note as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# CRUD: Note management
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def simple_note_manage_note(
    action: Literal["create", "show", "update", "delete"],
    note_id: int | NotGiven = NOT_GIVEN,
    title: str | NotGiven = NOT_GIVEN,
    content: str | NotGiven = NOT_GIVEN,
    tags: list[str] | None = None,
    pinned: bool | None = None,
) -> dict[str, Any]:
    """Manage Simple Note notes: create, view, update, or delete.

    Actions:
        create: Create a new note. Requires title and content. Optionally
            include tags and pinned status (defaults to false).
        show: View detailed information of a note, including its content.
            Requires note_id.
        update: Update a note's title, content, tags, and/or pinned status.
            Requires note_id and at least one of title, content, tags, or
            pinned.
        delete: Delete a note. Requires note_id.

    Args:
        action: The operation to perform.
        note_id: The note ID (for show, update, delete).
        title: Note title (for create, update).
        content: Note content (for create, update).
        tags: Tags for the note (for create, update).
        pinned: Pinned status of the note (for create, update).

    Returns:
        Note details or action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Simple Note.
    """
    if action == "create":
        kwargs: dict[str, Any] = {"title": title, "content": content}
        if tags is not None:
            kwargs["tags"] = tags
        if pinned is not None:
            kwargs["pinned"] = pinned
        return _get("simple_note_create_note")(**kwargs)
    elif action == "show":
        return _get("simple_note_show_note")(note_id=note_id)
    elif action == "update":
        kwargs = {"note_id": note_id}
        if title is not NOT_GIVEN:
            kwargs["title"] = title
        if content is not NOT_GIVEN:
            kwargs["content"] = content
        if tags is not None:
            kwargs["tags"] = tags
        if pinned is not None:
            kwargs["pinned"] = pinned
        return _get("simple_note_update_note")(**kwargs)
    elif action == "delete":
        return _get("simple_note_delete_note")(note_id=note_id)
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Mark absorbed tools
# ---------------------------------------------------------------------------

mark_tools_absorbed_by(
    "simple_note_manage_note",
    "simple_note_create_note",
    "simple_note_show_note",
    "simple_note_update_note",
    "simple_note_delete_note",
)
