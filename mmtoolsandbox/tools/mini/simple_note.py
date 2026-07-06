# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""MINI simple_note tool — unified notes management."""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.datasets.names import DatasetName


@register_as_tool(
    toolboxes={DatasetName.MINI},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def simple_note(
    action: Literal["create", "show", "update", "delete", "search", "add_content"],
    # CRUD params
    note_id: int | NotGiven = NOT_GIVEN,
    title: str | NotGiven = NOT_GIVEN,
    content: str | NotGiven = NOT_GIVEN,
    tags: list[str] | None | NotGiven = NOT_GIVEN,
    pinned: bool | None | NotGiven = NOT_GIVEN,
    # search params
    query: str | None | NotGiven = NOT_GIVEN,
    dont_reorder_pinned: bool | None | NotGiven = NOT_GIVEN,
    page_index: int | None | NotGiven = NOT_GIVEN,
    page_limit: int | None | NotGiven = NOT_GIVEN,
    sort_by: str | None | NotGiven = NOT_GIVEN,
    # add_content params
    append_or_prepend: str | NotGiven = NOT_GIVEN,
    added_content: str | NotGiven = NOT_GIVEN,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Create, update, delete, or search notes with tags and pinning.

    Actions:
        create - Create note. Requires title, content.
            Optional: tags, pinned.
        show - Show note with full content. Requires note_id.
        update - Update note. Requires note_id. Optional: title, content,
            tags, pinned.
        delete - Delete note. Requires note_id. Irreversible.
        search - Search notes (titles only, no content). Optional: query,
            tags, pinned, dont_reorder_pinned, page_index, page_limit,
            sort_by.
        add_content - Append or prepend text to a note. Requires note_id,
            append_or_prepend ("append"/"prepend"), added_content.

    Args:
        action: The note action.
        note_id: Note ID (for show/update/delete/add_content). Returned
            by create.
        title: Note title (e.g., "Meeting Notes").
        content: Note body text.
        tags: Comma-separated tag list (e.g., ["work", "urgent"]).
        pinned: Whether to pin the note to the top of the list.
        query: Search query — matches title and tags.
        dont_reorder_pinned: If true, pinned notes are not shown first.
        page_index: Zero-based page index.
        page_limit: Results per page.
        sort_by: Sort attribute with +/- prefix (e.g., "-created_at").
        append_or_prepend: "append" or "prepend" (for add_content).
        added_content: Text to add (for add_content).

    Returns:
        For create/show/update: dict with note_id, title, content, tags,
            pinned, created_at, updated_at. Pass note_id to other actions.
        For delete: confirmation dict.
        For search: list of note dicts (title only, use show for content).
        For add_content: updated note dict.

    Raises:
        ValueError: If note_id not found (for show/update/delete).
        PermissionError: If not logged into simple_note.
    """
    import mmtoolsandbox.tools.appworld.simple_note as m

    if action == "create":
        kwargs: dict[str, Any] = {"title": title, "content": content}
        if tags is not NOT_GIVEN:
            kwargs["tags"] = tags
        if pinned is not NOT_GIVEN:
            kwargs["pinned"] = pinned
        return m.simple_note_create_note(**kwargs)
    elif action == "show":
        return m.simple_note_show_note(note_id=note_id)
    elif action == "update":
        kwargs = {"note_id": note_id}
        if title is not NOT_GIVEN:
            kwargs["title"] = title
        if content is not NOT_GIVEN:
            kwargs["content"] = content
        if tags is not NOT_GIVEN:
            kwargs["tags"] = tags
        if pinned is not NOT_GIVEN:
            kwargs["pinned"] = pinned
        return m.simple_note_update_note(**kwargs)
    elif action == "delete":
        return m.simple_note_delete_note(note_id=note_id)
    elif action == "search":
        kwargs = {}
        if query is not NOT_GIVEN:
            kwargs["query"] = query
        if tags is not NOT_GIVEN:
            kwargs["tags"] = tags
        if pinned is not NOT_GIVEN:
            kwargs["pinned"] = pinned
        if dont_reorder_pinned is not NOT_GIVEN:
            kwargs["dont_reorder_pinned"] = dont_reorder_pinned
        if page_index is not NOT_GIVEN:
            kwargs["page_index"] = page_index
        if page_limit is not NOT_GIVEN:
            kwargs["page_limit"] = page_limit
        if sort_by is not NOT_GIVEN:
            kwargs["sort_by"] = sort_by
        return m.simple_note_search_notes(**kwargs)
    elif action == "add_content":
        return m.simple_note_add_content_to_note(
            note_id=note_id,
            append_or_prepend=append_or_prepend,
            added_content=added_content,
        )
    else:
        raise ValueError(f"Unknown action: {action}")
