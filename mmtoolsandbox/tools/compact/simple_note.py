# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""COMPACT Simple Note tools — CRUD + search for note management."""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.datasets.names import DatasetName
from mmtoolsandbox.tools.compact import mark_compact_tools_absorbed_by


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.appworld.simple_note as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# CRUD+Search: Note management (absorbs simple_note_search_notes)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def simple_note_manage_note(
    action: Literal["create", "show", "update", "delete", "search"],
    note_id: int | NotGiven = NOT_GIVEN,
    title: str | NotGiven = NOT_GIVEN,
    content: str | NotGiven = NOT_GIVEN,
    tags: list[str] | None = None,
    pinned: bool | None = None,
    # search-action params
    query: str | None = "",
    dont_reorder_pinned: bool | None = None,
    page_index: int | None = 0,
    page_limit: int | None = 5,
    sort_by: str | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Manage Simple Note notes: create, view, update, delete, or search.

    Actions:
        create: Create a new note. Requires title and content. Optionally
            include tags and pinned status (defaults to false).
        show: View detailed information of a note, including its content.
            Requires note_id.
        update: Update a note's title, content, tags, and/or pinned status.
            Requires note_id and at least one of title, content, tags, or
            pinned.
        delete: Delete a note. Requires note_id.
        search: Search your notes. Supports query, tags, pinned,
            dont_reorder_pinned, pagination, and sorting. This will not
            show contents of the notes. Pinned notes shown first by
            default unless dont_reorder_pinned is true.

    Args:
        action: The operation to perform.
        note_id: The note ID (for show, update, delete).
        title: Note title (for create, update).
        content: Note content (for create, update).
        tags: Tags for the note (for create, update, search filter).
        pinned: Pinned status (for create, update, search filter).
        query: Search query for notes (for search).
        dont_reorder_pinned: If true, pinned notes will not be shown first
            (for search).
        page_index: Zero-based page index (for search).
        page_limit: Maximum results per page (for search).
        sort_by: Sort attribute prefixed with +/- for direction. Valid
            attributes: created_at, updated_at (for search).

    Returns:
        Note details, action confirmation, or list of notes.

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
    elif action == "search":
        kwargs = {}
        if query is not None:
            kwargs["query"] = query
        if tags is not None:
            kwargs["tags"] = tags
        if pinned is not None:
            kwargs["pinned"] = pinned
        if dont_reorder_pinned is not None:
            kwargs["dont_reorder_pinned"] = dont_reorder_pinned
        if page_index is not None:
            kwargs["page_index"] = page_index
        if page_limit is not None:
            kwargs["page_limit"] = page_limit
        if sort_by is not None:
            kwargs["sort_by"] = sort_by
        return _get("simple_note_search_notes")(**kwargs)
    else:
        raise ValueError(f"Unknown action: {action}")


mark_compact_tools_absorbed_by(
    "simple_note_manage_note",
    "simple_note_search_notes",
)
