# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""COMPACT file system tools — merges copy/move/compress/decompress/exists into one tool."""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.datasets.names import DatasetName
from mmtoolsandbox.tools.compact import mark_compact_tools_absorbed_by


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.appworld.file_system as m

    return getattr(m, name)


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def file_system_operation(
    action: Literal["copy", "move", "compress", "decompress", "exists"],
    target_type: Literal["file", "directory"],
    source_path: str | NotGiven = NOT_GIVEN,
    destination_path: str | NotGiven = NOT_GIVEN,
    overwrite: bool | None = False,
    retain_dates: bool | None = None,
    delete_source: bool | None = False,
) -> dict[str, Any]:
    """Copy, move, compress, decompress, or check existence of files and directories in the file system.

    Actions:
        copy: Copy a file or directory. Requires source_path and
            destination_path.
        move: Move a file or directory. Requires source_path and
            destination_path.
        compress: Compress a directory into an archive. Requires source_path
            (directory). Optionally set destination_path for the archive,
            delete_source to remove the original directory.
        decompress: Decompress an archive. Requires source_path (archive).
            Optionally set destination_path for extraction location,
            delete_source to remove the archive.
        exists: Check if a file or directory exists. Requires source_path.

    Args:
        action: The operation to perform.
        target_type: Whether to operate on a "file" or "directory".
        source_path: Source file/directory path (required for all actions).
        destination_path: Destination path (for copy, move, compress,
            decompress).
        overwrite: Whether to overwrite if destination exists.
        retain_dates: Whether to preserve timestamps (for copy, move,
            decompress).
        delete_source: Whether to delete the source after compress/decompress.

    Returns:
        Operation result or existence check result.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into file_system.
    """
    if action == "exists":
        if target_type == "file":
            return _get("file_system_file_exists")(file_path=source_path)
        return _get("file_system_directory_exists")(directory_path=source_path)

    if action == "copy":
        kwargs: dict[str, Any] = {
            "overwrite": overwrite,
        }
        if retain_dates is not None:
            kwargs["retain_dates"] = retain_dates
        if target_type == "file":
            kwargs["source_file_path"] = source_path
            kwargs["destination_file_path"] = destination_path
            return _get("file_system_copy_file")(**kwargs)
        kwargs["source_directory_path"] = source_path
        kwargs["destination_directory_path"] = destination_path
        return _get("file_system_copy_directory")(**kwargs)

    if action == "move":
        kwargs = {
            "overwrite": overwrite,
        }
        if retain_dates is not None:
            kwargs["retain_dates"] = retain_dates
        if target_type == "file":
            kwargs["source_file_path"] = source_path
            kwargs["destination_file_path"] = destination_path
            return _get("file_system_move_file")(**kwargs)
        kwargs["source_directory_path"] = source_path
        kwargs["destination_directory_path"] = destination_path
        return _get("file_system_move_directory")(**kwargs)

    if action == "compress":
        kwargs = {"directory_path": source_path, "overwrite": overwrite}
        if destination_path is not NOT_GIVEN:
            kwargs["compressed_file_path"] = destination_path
        if delete_source:
            kwargs["delete_directory"] = True
        return _get("file_system_compress_directory")(**kwargs)

    if action == "decompress":
        kwargs = {
            "compressed_file_path": source_path,
            "overwrite": overwrite,
        }
        if retain_dates is not None:
            kwargs["retain_dates"] = retain_dates
        if destination_path is not NOT_GIVEN:
            kwargs["decompressed_directory_path"] = destination_path
        if delete_source:
            kwargs["delete_compressed_file"] = True
        return _get("file_system_decompress_file")(**kwargs)

    raise ValueError(f"Unknown action: {action}")


mark_compact_tools_absorbed_by(
    "file_system_operation",
    "file_system_copy_file",
    "file_system_move_file",
    "file_system_copy_directory",
    "file_system_move_directory",
    "file_system_compress_directory",
    "file_system_decompress_file",
    "file_system_file_exists",
    "file_system_directory_exists",
)
