# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""Consolidated File System tools for the MEDIUM toolbox.

CRUD consolidation for files (create, show, update, delete) and directories
(create, show, delete) merged into ``file_system_manage_file`` and
``file_system_manage_directory``.
"""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.toolbox.names import ToolboxName
from mmtoolsandbox.tools.consolidated import mark_tools_absorbed_by


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.appworld.file_system as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# CRUD: File management
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def file_system_manage_file(
    action: Literal["create", "show", "update", "delete"],
    file_path: str | NotGiven = NOT_GIVEN,
    content: str | NotGiven = NOT_GIVEN,
    overwrite: bool | None = None,
) -> dict[str, Any]:
    """Manage files in the file system: create, view, update, or delete.

    Actions:
        create: Create a new file. Requires file_path. Optionally include
            content and overwrite flag (defaults to false).
        show: Show a file's content and other details. Requires file_path.
        update: Update a file's content. Requires file_path and content.
        delete: Delete a file. Requires file_path.

    Args:
        action: The operation to perform.
        file_path: Path of the file. Path can be absolute, starting with
            '/', or relative to the user's home directory, starting with
            '~/'.
        content: The file content (for create, update).
        overwrite: Whether to overwrite an existing file (for create).

    Returns:
        File details or action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into file_system.
    """
    if action == "create":
        kwargs: dict[str, Any] = {"file_path": file_path}
        if content is not NOT_GIVEN:
            kwargs["content"] = content
        if overwrite is not None:
            kwargs["overwrite"] = overwrite
        return _get("file_system_create_file")(**kwargs)
    elif action == "show":
        return _get("file_system_show_file")(file_path=file_path)
    elif action == "update":
        return _get("file_system_update_file")(file_path=file_path, content=content)
    elif action == "delete":
        return _get("file_system_delete_file")(file_path=file_path)
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# CRUD: Directory management
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def file_system_manage_directory(
    action: Literal["create", "show", "delete"],
    directory_path: str | NotGiven = NOT_GIVEN,
    recursive: bool | None = None,
    allow_if_exists: bool | None = None,
    substring: str | None = None,
    entry_type: Literal["all", "files", "directories"] | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Manage directories in the file system: create, list contents, or delete.

    Actions:
        create: Create a directory. Requires directory_path. Optionally
            set recursive (default false) and allow_if_exists (default
            true).
        show: Show files and/or sub-directories in a directory. Optionally
            filter by substring or entry_type, and set recursive listing.
            directory_path defaults to '/'.
        delete: Delete a directory with its sub-directories and files.
            Requires directory_path.

    Args:
        action: The operation to perform.
        directory_path: Path of the directory. Path can be absolute,
            starting with '/', or relative to the user's home directory,
            starting with '~/'.
        recursive: For create: create parent directories recursively.
            For show: list contents recursively (default true).
        allow_if_exists: If true, do not raise an error if the directory
            already exists (for create, default true).
        substring: Filter entries containing this substring, ignoring case
            (for show).
        entry_type: Show 'all', only 'files', or only 'directories'
            (for show, default 'all').

    Returns:
        Directory details, list of entries, or action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into file_system.
    """
    if action == "create":
        kwargs: dict[str, Any] = {"directory_path": directory_path}
        if recursive is not None:
            kwargs["recursive"] = recursive
        if allow_if_exists is not None:
            kwargs["allow_if_exists"] = allow_if_exists
        return _get("file_system_create_directory")(**kwargs)
    elif action == "show":
        kwargs = {}
        if directory_path is not NOT_GIVEN:
            kwargs["directory_path"] = directory_path
        if substring is not None:
            kwargs["substring"] = substring
        if entry_type is not None:
            kwargs["entry_type"] = entry_type
        if recursive is not None:
            kwargs["recursive"] = recursive
        return _get("file_system_show_directory")(**kwargs)
    elif action == "delete":
        return _get("file_system_delete_directory")(directory_path=directory_path)
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Mark absorbed tools
# ---------------------------------------------------------------------------

mark_tools_absorbed_by(
    "file_system_manage_file",
    "file_system_create_file",
    "file_system_show_file",
    "file_system_update_file",
    "file_system_delete_file",
)
mark_tools_absorbed_by(
    "file_system_manage_directory",
    "file_system_create_directory",
    "file_system_show_directory",
    "file_system_delete_directory",
)
