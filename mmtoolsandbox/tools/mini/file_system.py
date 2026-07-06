# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""MINI file_system tool — unified file/directory CRUD and operations."""

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
def file_system(
    domain: Literal["file", "directory", "operation"],
    action: Literal[
        "create",
        "show",
        "update",
        "delete",
        "copy",
        "move",
        "compress",
        "decompress",
        "exists",
    ],
    # file params
    file_path: str | NotGiven = NOT_GIVEN,
    content: str | None | NotGiven = NOT_GIVEN,
    overwrite: bool | None | NotGiven = NOT_GIVEN,
    # directory params
    directory_path: str | None | NotGiven = NOT_GIVEN,
    recursive: bool | None | NotGiven = NOT_GIVEN,
    allow_if_exists: bool | None | NotGiven = NOT_GIVEN,
    substring: str | None | NotGiven = NOT_GIVEN,
    entry_type: Literal["all", "files", "directories"] | None | NotGiven = NOT_GIVEN,
    # operation params
    target_type: Literal["file", "directory"] | NotGiven = NOT_GIVEN,
    source_file_path: str | NotGiven = NOT_GIVEN,
    destination_file_path: str | NotGiven = NOT_GIVEN,
    source_directory_path: str | NotGiven = NOT_GIVEN,
    destination_directory_path: str | NotGiven = NOT_GIVEN,
    retain_dates: bool | None | NotGiven = NOT_GIVEN,
    # compress/decompress
    compressed_file_path: str | None | NotGiven = NOT_GIVEN,
    decompressed_directory_path: str | None | NotGiven = NOT_GIVEN,
    delete_directory: bool | None | NotGiven = NOT_GIVEN,
    delete_compressed_file: bool | None | NotGiven = NOT_GIVEN,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Create, read, update, delete, copy, move, compress, or decompress files and directories.

    Domains and actions:
        file:
            create - Create file. Requires file_path. Optional: content, overwrite.
            show - Show file content. Requires file_path.
            update - Update file content. Requires file_path, content.
            delete - Delete file. Requires file_path.
        directory:
            create - Create directory. Requires directory_path.
                Optional: recursive, allow_if_exists.
            show - List directory contents. Optional: directory_path, substring,
                entry_type, recursive.
            delete - Delete directory. Requires directory_path.
        operation:
            copy - Copy. Requires target_type ("file"/"directory").
                For file: source_file_path, destination_file_path.
                For directory: source_directory_path, destination_directory_path.
                Optional: overwrite, retain_dates.
            move - Move. Same params as copy.
            compress - Compress directory. Requires directory_path.
                Optional: compressed_file_path, delete_directory, overwrite.
            decompress - Decompress file. Requires compressed_file_path.
                Optional: decompressed_directory_path, overwrite, retain_dates,
                delete_compressed_file.
            exists - Check existence. Requires target_type.
                For file: file_path. For directory: directory_path.

    Args:
        domain: The file system domain.
        action: The specific action.
        file_path: Path of the file.
        content: File content.
        overwrite: Whether to overwrite.
        directory_path: Path of the directory.
        recursive: Recursive operation.
        allow_if_exists: Don't error if directory exists.
        substring: Filter by substring (for directory show).
        file_path: Absolute path or ~/relative (e.g., "~/documents/notes.txt").
        content: File content text.
        overwrite: Whether to overwrite existing file.
        directory_path: Absolute path or ~/relative (e.g., "~/documents/").
        recursive: Recursive operation (for directory create/show).
        allow_if_exists: Don't error if directory exists (for create).
        substring: Filter entries containing this substring (for directory show).
        entry_type: "all", "files", or "directories" (for directory show).
        target_type: "file" or "directory" (for copy/move/exists).
        source_file_path: Source path (for copy/move file).
        destination_file_path: Destination path (for copy/move file).
        source_directory_path: Source path (for copy/move directory).
        destination_directory_path: Destination path (for copy/move directory).
        retain_dates: Retain original timestamps during copy/decompress.
        compressed_file_path: Path to .zip file (for compress/decompress).
        decompressed_directory_path: Output directory (for decompress).
        delete_directory: Delete source directory after compression.
        delete_compressed_file: Delete .zip after decompression.

    Returns:
        For create/show/update: dict with file_path, content, size, etc.
        For delete: confirmation dict. Irreversible.
        For directory show: list of entry dicts with name, type, size.
        For copy/move: confirmation dict.
        For exists: dict with "exists" boolean key.

    Raises:
        PermissionError: If not logged into file_system.
    """
    import mmtoolsandbox.tools.appworld.file_system as m

    if domain == "file":
        if action == "create":
            kwargs: dict[str, Any] = {"file_path": file_path}
            if content is not NOT_GIVEN:
                kwargs["content"] = content
            if overwrite is not NOT_GIVEN:
                kwargs["overwrite"] = overwrite
            return m.file_system_create_file(**kwargs)
        elif action == "show":
            return m.file_system_show_file(file_path=file_path)
        elif action == "update":
            return m.file_system_update_file(file_path=file_path, content=content)
        elif action == "delete":
            return m.file_system_delete_file(file_path=file_path)
        else:
            raise ValueError(f"Unknown file action: {action}")

    elif domain == "directory":
        if action == "create":
            kwargs = {"directory_path": directory_path}
            if recursive is not NOT_GIVEN:
                kwargs["recursive"] = recursive
            if allow_if_exists is not NOT_GIVEN:
                kwargs["allow_if_exists"] = allow_if_exists
            return m.file_system_create_directory(**kwargs)
        elif action == "show":
            kwargs = {}
            if directory_path is not NOT_GIVEN:
                kwargs["directory_path"] = directory_path
            if substring is not NOT_GIVEN:
                kwargs["substring"] = substring
            if entry_type is not NOT_GIVEN:
                kwargs["entry_type"] = entry_type
            if recursive is not NOT_GIVEN:
                kwargs["recursive"] = recursive
            return m.file_system_show_directory(**kwargs)
        elif action == "delete":
            return m.file_system_delete_directory(directory_path=directory_path)
        else:
            raise ValueError(f"Unknown directory action: {action}")

    elif domain == "operation":
        if action == "copy":
            if target_type == "file":
                kwargs = {
                    "source_file_path": source_file_path,
                    "destination_file_path": destination_file_path,
                }
                if overwrite is not NOT_GIVEN:
                    kwargs["overwrite"] = overwrite
                if retain_dates is not NOT_GIVEN:
                    kwargs["retain_dates"] = retain_dates
                return m.file_system_copy_file(**kwargs)
            elif target_type == "directory":
                kwargs = {
                    "source_directory_path": source_directory_path,
                    "destination_directory_path": destination_directory_path,
                }
                if overwrite is not NOT_GIVEN:
                    kwargs["overwrite"] = overwrite
                if retain_dates is not NOT_GIVEN:
                    kwargs["retain_dates"] = retain_dates
                return m.file_system_copy_directory(**kwargs)
            else:
                raise ValueError(f"Unknown target_type: {target_type}")
        elif action == "move":
            if target_type == "file":
                kwargs = {
                    "source_file_path": source_file_path,
                    "destination_file_path": destination_file_path,
                }
                if overwrite is not NOT_GIVEN:
                    kwargs["overwrite"] = overwrite
                if retain_dates is not NOT_GIVEN:
                    kwargs["retain_dates"] = retain_dates
                return m.file_system_move_file(**kwargs)
            elif target_type == "directory":
                kwargs = {
                    "source_directory_path": source_directory_path,
                    "destination_directory_path": destination_directory_path,
                }
                if overwrite is not NOT_GIVEN:
                    kwargs["overwrite"] = overwrite
                if retain_dates is not NOT_GIVEN:
                    kwargs["retain_dates"] = retain_dates
                return m.file_system_move_directory(**kwargs)
            else:
                raise ValueError(f"Unknown target_type: {target_type}")
        elif action == "compress":
            kwargs = {"directory_path": directory_path}
            if compressed_file_path is not NOT_GIVEN:
                kwargs["compressed_file_path"] = compressed_file_path
            if delete_directory is not NOT_GIVEN:
                kwargs["delete_directory"] = delete_directory
            if overwrite is not NOT_GIVEN:
                kwargs["overwrite"] = overwrite
            return m.file_system_compress_directory(**kwargs)
        elif action == "decompress":
            kwargs = {"compressed_file_path": compressed_file_path}
            if decompressed_directory_path is not NOT_GIVEN:
                kwargs["decompressed_directory_path"] = decompressed_directory_path
            if overwrite is not NOT_GIVEN:
                kwargs["overwrite"] = overwrite
            if retain_dates is not NOT_GIVEN:
                kwargs["retain_dates"] = retain_dates
            if delete_compressed_file is not NOT_GIVEN:
                kwargs["delete_compressed_file"] = delete_compressed_file
            return m.file_system_decompress_file(**kwargs)
        elif action == "exists":
            if target_type == "file":
                return m.file_system_file_exists(file_path=file_path)
            elif target_type == "directory":
                return m.file_system_directory_exists(directory_path=directory_path)
            else:
                raise ValueError(f"Unknown target_type: {target_type}")
        else:
            raise ValueError(f"Unknown operation action: {action}")

    else:
        raise ValueError(f"Unknown domain: {domain}")
