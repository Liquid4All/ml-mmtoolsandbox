# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""COMPACT Todoist tools — merges collaboration (4->1), attachments (3->1)."""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.datasets.names import DatasetName
from mmtoolsandbox.tools.compact import mark_compact_tools_absorbed_by


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.appworld.todoist as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# Strategy 5: Collaboration management (4 -> 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def todoist_manage_collaboration(
    action: Literal["invite", "accept", "decline", "remove"],
    project_id: int | NotGiven = NOT_GIVEN,
    email: str | NotGiven = NOT_GIVEN,
    invite_code: str | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Invite, accept, decline, or remove collaborators on Todoist projects.

    Actions:
        invite: Send a project invite. Requires project_id and email.
        accept: Accept a project invite. Requires invite_code.
        decline: Decline (delete) a project invite. Requires invite_code.
        remove: Remove a collaborator from a project. Requires project_id
            and email.

    Args:
        action: The collaboration operation to perform.
        project_id: The ID of the project (for invite, remove).
        email: Email of the user (for invite, remove).
        invite_code: The invite code (for accept, decline).

    Returns:
        Operation result details.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Todoist.
    """
    if action == "invite":
        return _get("todoist_send_project_invite")(project_id=project_id, email=email)
    elif action == "accept":
        return _get("todoist_accept_project_invite")(invite_code=invite_code)
    elif action == "decline":
        return _get("todoist_delete_project_invite")(invite_code=invite_code)
    elif action == "remove":
        return _get("todoist_remove_collaborator_from_project")(
            project_id=project_id, email=email
        )
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Strategy 5: Attachment management (3 -> 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def todoist_manage_attachment(
    action: Literal["upload", "download", "delete"],
    task_comment_id: int | NotGiven = NOT_GIVEN,
    upload_from_file_path: str | NotGiven = NOT_GIVEN,
    attachment_file_name: str | NotGiven = NOT_GIVEN,
    file_system_access_token: str | NotGiven = NOT_GIVEN,
    download_to_file_path: str | None = None,
    overwrite: bool | None = False,
) -> dict[str, Any]:
    """Upload, download, or delete attachments on Todoist task comments.

    Actions:
        upload: Upload an attachment to a task comment. Requires
            task_comment_id, upload_from_file_path, and
            file_system_access_token.
        download: Download an attachment from a task comment. Requires
            task_comment_id, attachment_file_name, and
            file_system_access_token.
        delete: Delete an attachment from a task comment. Requires
            task_comment_id and attachment_file_name.

    Args:
        action: The attachment operation to perform.
        task_comment_id: The ID of the task comment.
        upload_from_file_path: File path to upload (for upload).
        attachment_file_name: Name of the attached file (for download,
            delete).
        file_system_access_token: Access token from file_system login
            (for upload, download).
        download_to_file_path: Destination path for download. Defaults to
            ~/downloads directory.
        overwrite: Whether to overwrite existing files.

    Returns:
        Operation result details.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Todoist.
    """
    if action == "upload":
        kwargs: dict[str, Any] = {
            "task_comment_id": task_comment_id,
            "upload_from_file_path": upload_from_file_path,
            "file_system_access_token": file_system_access_token,
        }
        if overwrite is not None:
            kwargs["overwrite"] = overwrite
        return _get("todoist_upload_attachment")(**kwargs)
    elif action == "download":
        kwargs = {
            "task_comment_id": task_comment_id,
            "attachment_file_name": attachment_file_name,
            "file_system_access_token": file_system_access_token,
        }
        if download_to_file_path is not None:
            kwargs["download_to_file_path"] = download_to_file_path
        if overwrite is not None:
            kwargs["overwrite"] = overwrite
        return _get("todoist_download_attachment")(**kwargs)
    elif action == "delete":
        return _get("todoist_delete_attachment")(
            task_comment_id=task_comment_id,
            attachment_file_name=attachment_file_name,
        )
    else:
        raise ValueError(f"Unknown action: {action}")


mark_compact_tools_absorbed_by(
    "todoist_manage_collaboration",
    "todoist_send_project_invite",
    "todoist_accept_project_invite",
    "todoist_delete_project_invite",
    "todoist_remove_collaborator_from_project",
)
mark_compact_tools_absorbed_by(
    "todoist_manage_attachment",
    "todoist_upload_attachment",
    "todoist_download_attachment",
    "todoist_delete_attachment",
)


# ---------------------------------------------------------------------------
# CRUD+List: Project management (absorbs todoist_show_projects)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def todoist_manage_project(
    action: Literal["create", "show", "update", "delete", "list"],
    project_id: int | NotGiven = NOT_GIVEN,
    name: str | NotGiven = NOT_GIVEN,
    color: str | None | NotGiven = NOT_GIVEN,
    description: str | NotGiven = NOT_GIVEN,
    is_favorite: bool | None | NotGiven = NOT_GIVEN,
    is_archived: bool | None | NotGiven = NOT_GIVEN,
    # list-action params
    query: str | None = "",
    page_index: int | None = 0,
    page_limit: int | None = 5,
    sort_by: str | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Manage Todoist projects: create, view, update, delete, or list.

    Actions:
        create: Create a new project. Requires name. color defaults to
            "charcoal", description defaults to empty string, is_favorite
            defaults to false.
        show: View details of a project. Requires project_id. Use 0 for
            your Inbox project.
        update: Update project properties. Requires project_id and at least
            one of name, color, description, is_favorite, or is_archived.
        delete: Delete a project. Requires project_id.
        list: List or search your projects. Supports query, color,
            is_favorite, is_archived, pagination, and sorting.

    Args:
        action: The operation to perform.
        project_id: The project ID (for show, update, delete). Use 0 for
            your Inbox project.
        name: Project name (for create, update).
        color: Project color (for create, update, list filter).
        description: Project description (for create, update).
        is_favorite: Whether the project is a favorite (for create, update,
            list filter).
        is_archived: Whether the project is archived (for update, list
            filter).
        query: Search query for projects (for list).
        page_index: Zero-based page index (for list).
        page_limit: Maximum results per page (for list).
        sort_by: Sort attribute prefixed with +/- for direction. Valid
            attributes: created_at (for list).

    Returns:
        Project details, action confirmation, or list of projects.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Todoist.
    """
    if action == "create":
        kwargs: dict[str, Any] = {"name": name}
        if color is not NOT_GIVEN:
            kwargs["color"] = color
        if description is not NOT_GIVEN:
            kwargs["description"] = description
        if is_favorite is not NOT_GIVEN:
            kwargs["is_favorite"] = is_favorite
        return _get("todoist_create_project")(**kwargs)
    elif action == "show":
        return _get("todoist_show_project")(project_id=project_id)
    elif action == "update":
        kwargs = {"project_id": project_id}
        if name is not NOT_GIVEN:
            kwargs["name"] = name
        if color is not NOT_GIVEN:
            kwargs["color"] = color
        if description is not NOT_GIVEN:
            kwargs["description"] = description
        if is_favorite is not NOT_GIVEN:
            kwargs["is_favorite"] = is_favorite
        if is_archived is not NOT_GIVEN:
            kwargs["is_archived"] = is_archived
        return _get("todoist_update_project")(**kwargs)
    elif action == "delete":
        return _get("todoist_delete_project")(project_id=project_id)
    elif action == "list":
        kwargs = {}
        if query is not None:
            kwargs["query"] = query
        if color is not NOT_GIVEN and color is not None:
            kwargs["color"] = color
        if is_favorite is not NOT_GIVEN and is_favorite is not None:
            kwargs["is_favorite"] = is_favorite
        if is_archived is not NOT_GIVEN and is_archived is not None:
            kwargs["is_archived"] = is_archived
        if page_index is not None:
            kwargs["page_index"] = page_index
        if page_limit is not None:
            kwargs["page_limit"] = page_limit
        if sort_by is not None:
            kwargs["sort_by"] = sort_by
        return _get("todoist_show_projects")(**kwargs)
    else:
        raise ValueError(f"Unknown action: {action}")


mark_compact_tools_absorbed_by(
    "todoist_manage_project",
    "todoist_show_projects",
)


# ---------------------------------------------------------------------------
# CRUD+List: Task management (absorbs todoist_show_tasks)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def todoist_manage_task(
    action: Literal["create", "show", "update", "delete", "list"],
    project_id: int | NotGiven = NOT_GIVEN,
    task_id: int | NotGiven = NOT_GIVEN,
    title: str | NotGiven = NOT_GIVEN,
    section_id: int | None | NotGiven = NOT_GIVEN,
    description: str | None | NotGiven = NOT_GIVEN,
    due_date: str | None | NotGiven = NOT_GIVEN,
    duration: float | None | NotGiven = NOT_GIVEN,
    duration_unit: Literal["minute", "day"] | None | NotGiven = NOT_GIVEN,
    order_index: int | None | NotGiven = NOT_GIVEN,
    priority: Literal["low", "medium", "high", "urgent"] | None | NotGiven = NOT_GIVEN,
    is_completed: bool | None | NotGiven = NOT_GIVEN,
    # list-action params
    assignee_email: str | None = None,
    assigner_email: str | None = None,
    due_today: bool | None = False,
    label_id: int | None = None,
    overdue: bool | None = False,
    min_due_date: str | None = None,
    max_due_date: str | None = None,
    sort_by: str | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Manage Todoist tasks: create, view, update, delete, or list.

    Actions:
        create: Create a new task. Requires project_id and title. Use
            project_id 0 for the default/inbox project. description defaults
            to empty string, order_index defaults to -1, priority defaults
            to "medium".
        show: View detailed task information. Requires task_id.
        update: Update task properties. Requires task_id and at least one
            field to change. Pass due_date=null to remove the due date.
            Pass duration=0 to remove duration.
        delete: Delete a task. Requires task_id.
        list: List tasks within a project. Requires project_id. Supports
            filtering by section_id, assignee_email, assigner_email,
            priority, is_completed, due_today, label_id, overdue,
            min_due_date, max_due_date, and sort_by.

    Args:
        action: The operation to perform.
        project_id: The project ID (for create, list). Use 0 for inbox.
        task_id: The task ID (for show, update, delete).
        title: Task title (for create, update).
        section_id: Section ID within the project (for create, list filter).
        description: Task description (for create, update).
        due_date: Due date in YYYY-MM-DD format (for create, update).
        duration: Task duration (for create, update).
        duration_unit: Unit of the task duration (for create, update).
        order_index: Position index in the task list (for create, update).
        priority: Task priority (for create, update, list filter).
        is_completed: Whether the task is completed (for update, list
            filter).
        assignee_email: Email of the assignee to filter by (for list).
        assigner_email: Email of the assigner to filter by (for list).
        due_today: If true, only tasks due today (for list).
        label_id: Label ID to filter tasks by (for list).
        overdue: If true, only overdue tasks (for list).
        min_due_date: Minimum due date in YYYY-MM-DD format (for list).
        max_due_date: Maximum due date in YYYY-MM-DD format (for list).
        sort_by: Sort attribute prefixed with +/- for direction. Valid
            attributes: order_index, due_date, priority (for list).

    Returns:
        Task details or action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Todoist.
    """
    if action == "create":
        kwargs: dict[str, Any] = {"project_id": project_id, "title": title}
        if section_id is not NOT_GIVEN:
            kwargs["section_id"] = section_id
        if description is not NOT_GIVEN:
            kwargs["description"] = description
        if due_date is not NOT_GIVEN:
            kwargs["due_date"] = due_date
        if duration is not NOT_GIVEN:
            kwargs["duration"] = duration
        if duration_unit is not NOT_GIVEN:
            kwargs["duration_unit"] = duration_unit
        if order_index is not NOT_GIVEN:
            kwargs["order_index"] = order_index
        if priority is not NOT_GIVEN:
            kwargs["priority"] = priority
        return _get("todoist_create_task")(**kwargs)
    elif action == "show":
        return _get("todoist_show_task")(task_id=task_id)
    elif action == "update":
        kwargs = {"task_id": task_id}
        if title is not NOT_GIVEN:
            kwargs["title"] = title
        if description is not NOT_GIVEN:
            kwargs["description"] = description
        if due_date is not NOT_GIVEN:
            kwargs["due_date"] = "None" if due_date is None else due_date
        if duration is not NOT_GIVEN:
            kwargs["duration"] = duration
        if duration_unit is not NOT_GIVEN:
            kwargs["duration_unit"] = duration_unit
        if is_completed is not NOT_GIVEN:
            kwargs["is_completed"] = is_completed
        if order_index is not NOT_GIVEN:
            kwargs["order_index"] = order_index
        if priority is not NOT_GIVEN:
            kwargs["priority"] = priority
        return _get("todoist_update_task")(**kwargs)
    elif action == "delete":
        return _get("todoist_delete_task")(task_id=task_id)
    elif action == "list":
        kwargs = {"project_id": project_id}
        if section_id is not NOT_GIVEN and section_id is not None:
            kwargs["section_id"] = section_id
        if assignee_email is not None:
            kwargs["assignee_email"] = assignee_email
        if assigner_email is not None:
            kwargs["assigner_email"] = assigner_email
        if priority is not NOT_GIVEN and priority is not None:
            kwargs["priority"] = priority
        if is_completed is not NOT_GIVEN and is_completed is not None:
            kwargs["is_completed"] = is_completed
        if due_today is not None:
            kwargs["due_today"] = due_today
        if label_id is not None:
            kwargs["label_id"] = label_id
        if overdue is not None:
            kwargs["overdue"] = overdue
        if min_due_date is not None:
            kwargs["min_due_date"] = min_due_date
        if max_due_date is not None:
            kwargs["max_due_date"] = max_due_date
        if sort_by is not None:
            kwargs["sort_by"] = sort_by
        return _get("todoist_show_tasks")(**kwargs)
    else:
        raise ValueError(f"Unknown action: {action}")


mark_compact_tools_absorbed_by(
    "todoist_manage_task",
    "todoist_show_tasks",
)


# ---------------------------------------------------------------------------
# CRUD+List: Section management (absorbs todoist_show_sections)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def todoist_manage_section(
    action: Literal["create", "update", "delete", "list"],
    project_id: int | NotGiven = NOT_GIVEN,
    section_id: int | NotGiven = NOT_GIVEN,
    name: str | None | NotGiven = NOT_GIVEN,
    order_index: int | None | NotGiven = NOT_GIVEN,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Manage Todoist sections within a project: create, update, delete, or list.

    Actions:
        create: Create a new section. Requires project_id and name.
            order_index defaults to -1 (append to end).
        update: Update section properties. Requires section_id and at least
            one of name or order_index.
        delete: Delete a section. Requires section_id.
        list: List all sections within a project. Requires project_id.

    Args:
        action: The operation to perform.
        project_id: The project ID (for create, list). Use 0 for your Inbox
            project.
        section_id: The section ID (for update, delete).
        name: Section name (for create, update).
        order_index: Position index in the section list (for create, update).
            Use negative values to insert from the end (-1 = append).

    Returns:
        Section details, action confirmation, or list of sections.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Todoist.
    """
    if action == "create":
        kwargs: dict[str, Any] = {"project_id": project_id, "name": name}
        if order_index is not NOT_GIVEN:
            kwargs["order_index"] = order_index
        return _get("todoist_create_section")(**kwargs)
    elif action == "update":
        kwargs = {"section_id": section_id}
        if name is not NOT_GIVEN:
            kwargs["name"] = name
        if order_index is not NOT_GIVEN:
            kwargs["order_index"] = order_index
        return _get("todoist_update_section")(**kwargs)
    elif action == "delete":
        return _get("todoist_delete_section")(section_id=section_id)
    elif action == "list":
        return _get("todoist_show_sections")(project_id=project_id)
    else:
        raise ValueError(f"Unknown action: {action}")


mark_compact_tools_absorbed_by(
    "todoist_manage_section",
    "todoist_show_sections",
)


# ---------------------------------------------------------------------------
# CRUD+List: Sub-task management (absorbs todoist_show_sub_tasks)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def todoist_manage_sub_task(
    action: Literal["create", "update", "delete", "list"],
    task_id: int | NotGiven = NOT_GIVEN,
    sub_task_id: int | NotGiven = NOT_GIVEN,
    title: str | NotGiven = NOT_GIVEN,
    description: str | None | NotGiven = NOT_GIVEN,
    due_date: str | None | NotGiven = NOT_GIVEN,
    duration: float | None | NotGiven = NOT_GIVEN,
    duration_unit: Literal["minute", "day"] | None | NotGiven = NOT_GIVEN,
    priority: Literal["low", "medium", "high", "urgent"] | None | NotGiven = NOT_GIVEN,
    order_index: int | None | NotGiven = NOT_GIVEN,
    is_completed: bool | None | NotGiven = NOT_GIVEN,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Manage Todoist sub-tasks: create, update, delete, or list.

    Actions:
        create: Create a new sub-task within a task. Requires task_id and
            title. description defaults to empty string, priority defaults
            to "medium", order_index defaults to -1.
        update: Update sub-task properties. Requires sub_task_id and at
            least one field to change. Pass due_date=null to remove the
            due date. Pass duration=0 to remove duration.
        delete: Delete a sub-task. Requires sub_task_id.
        list: List all sub-tasks within a task. Requires task_id.

    Args:
        action: The operation to perform.
        task_id: The parent task ID (for create, list).
        sub_task_id: The sub-task ID (for update, delete).
        title: Sub-task title (for create, update).
        description: Sub-task description (for create, update).
        due_date: Due date in YYYY-MM-DD format (for create, update).
        duration: Sub-task duration (for create, update).
        duration_unit: Unit of the sub-task duration (for create, update).
        priority: Sub-task priority (for create, update).
        order_index: Position index in the sub-task list (for create,
            update).
        is_completed: Whether the sub-task is completed (for update only).

    Returns:
        Sub-task details, action confirmation, or list of sub-tasks.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Todoist.
    """
    if action == "create":
        kwargs: dict[str, Any] = {"task_id": task_id, "title": title}
        if description is not NOT_GIVEN:
            kwargs["description"] = description
        if due_date is not NOT_GIVEN:
            kwargs["due_date"] = due_date
        if duration is not NOT_GIVEN:
            kwargs["duration"] = duration
        if duration_unit is not NOT_GIVEN:
            kwargs["duration_unit"] = duration_unit
        if priority is not NOT_GIVEN:
            kwargs["priority"] = priority
        if order_index is not NOT_GIVEN:
            kwargs["order_index"] = order_index
        return _get("todoist_create_sub_task")(**kwargs)
    elif action == "update":
        kwargs = {"sub_task_id": sub_task_id}
        if title is not NOT_GIVEN:
            kwargs["title"] = title
        if description is not NOT_GIVEN:
            kwargs["description"] = description
        if due_date is not NOT_GIVEN:
            kwargs["due_date"] = "None" if due_date is None else due_date
        if duration is not NOT_GIVEN:
            kwargs["duration"] = duration
        if duration_unit is not NOT_GIVEN:
            kwargs["duration_unit"] = duration_unit
        if priority is not NOT_GIVEN:
            kwargs["priority"] = priority
        if is_completed is not NOT_GIVEN:
            kwargs["is_completed"] = is_completed
        if order_index is not NOT_GIVEN:
            kwargs["order_index"] = order_index
        return _get("todoist_update_sub_task")(**kwargs)
    elif action == "delete":
        return _get("todoist_delete_sub_task")(sub_task_id=sub_task_id)
    elif action == "list":
        return _get("todoist_show_sub_tasks")(task_id=task_id)
    else:
        raise ValueError(f"Unknown action: {action}")


mark_compact_tools_absorbed_by(
    "todoist_manage_sub_task",
    "todoist_show_sub_tasks",
)


# ---------------------------------------------------------------------------
# CRUD+Search: Label management (absorbs todoist_search_labels)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def todoist_manage_label(
    action: Literal["create", "show", "update", "delete", "search"],
    label_id: int | NotGiven = NOT_GIVEN,
    name: str | NotGiven = NOT_GIVEN,
    color: str | None | NotGiven = NOT_GIVEN,
    # search-action params
    query: str | None = "",
    task_id: int | None = None,
    task_attached: bool | None = True,
    page_index: int | None = 0,
    page_limit: int | None = 5,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Manage Todoist labels: create, view, update, delete, or search.

    Actions:
        create: Create a new label. Requires name. color defaults to
            "charcoal".
        show: View label details. Requires label_id.
        update: Update label properties. Requires label_id and at least
            one of name or color.
        delete: Delete a label. Requires label_id.
        search: Search your labels or labels attached to a task. Supports
            query, task_id, color, task_attached, and pagination.

    Args:
        action: The operation to perform.
        label_id: The label ID (for show, update, delete).
        name: Label name (for create, update).
        color: Label color (for create, update, search filter).
        query: Search query string (for search).
        task_id: Task ID to filter labels by (for search).
        task_attached: If true with task_id, return labels on the task;
            if false, return your labels not on the task (for search).
        page_index: Zero-based page index (for search).
        page_limit: Maximum results per page (for search).

    Returns:
        Label details, action confirmation, or list of labels.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Todoist.
    """
    if action == "create":
        kwargs: dict[str, Any] = {"name": name}
        if color is not NOT_GIVEN:
            kwargs["color"] = color
        return _get("todoist_create_label")(**kwargs)
    elif action == "show":
        return _get("todoist_show_label")(label_id=label_id)
    elif action == "update":
        kwargs = {"label_id": label_id}
        if name is not NOT_GIVEN:
            kwargs["name"] = name
        if color is not NOT_GIVEN:
            kwargs["color"] = color
        return _get("todoist_update_label")(**kwargs)
    elif action == "delete":
        return _get("todoist_delete_label")(label_id=label_id)
    elif action == "search":
        kwargs = {}
        if query is not None:
            kwargs["query"] = query
        if task_id is not None:
            kwargs["task_id"] = task_id
        if color is not NOT_GIVEN and color is not None:
            kwargs["color"] = color
        if task_attached is not None:
            kwargs["task_attached"] = task_attached
        if page_index is not None:
            kwargs["page_index"] = page_index
        if page_limit is not None:
            kwargs["page_limit"] = page_limit
        return _get("todoist_search_labels")(**kwargs)
    else:
        raise ValueError(f"Unknown action: {action}")


mark_compact_tools_absorbed_by(
    "todoist_manage_label",
    "todoist_search_labels",
)


# ---------------------------------------------------------------------------
# CRUD+List: Task comment management (absorbs todoist_show_task_comments)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def todoist_manage_task_comment(
    action: Literal["post", "show", "update", "delete", "list"],
    task_id: int | NotGiven = NOT_GIVEN,
    task_comment_id: int | NotGiven = NOT_GIVEN,
    content: str | None | NotGiven = NOT_GIVEN,
    attachment_file_paths: list[str] | None | NotGiven = NOT_GIVEN,
    file_system_access_token: str | None | NotGiven = NOT_GIVEN,
    # list-action params
    page_index: int | None = 0,
    page_limit: int | None = 5,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Manage comments on Todoist tasks: post, view, update, delete, or list.

    Actions:
        post: Post a new comment on a task. Requires task_id and content.
            Optionally attach files via attachment_file_paths (requires
            file_system_access_token).
        show: View a task comment. Requires task_comment_id.
        update: Update comment content. Requires task_comment_id and
            content.
        delete: Delete a task comment. Requires task_comment_id.
        list: List comments on a task. Requires task_id. Supports
            pagination.

    Args:
        action: The operation to perform.
        task_id: The task ID (for post, list).
        task_comment_id: The comment ID (for show, update, delete).
        content: Comment content (for post, update).
        attachment_file_paths: Paths to files to attach (for post).
        file_system_access_token: File system access token, needed when
            attaching files (for post).
        page_index: Zero-based page index (for list).
        page_limit: Maximum results per page (for list).

    Returns:
        Comment details, action confirmation, or list of comments.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Todoist.
    """
    if action == "post":
        kwargs: dict[str, Any] = {"task_id": task_id, "content": content}
        if attachment_file_paths is not NOT_GIVEN:
            kwargs["attachment_file_paths"] = attachment_file_paths
        if file_system_access_token is not NOT_GIVEN:
            kwargs["file_system_access_token"] = file_system_access_token
        return _get("todoist_post_task_comment")(**kwargs)
    elif action == "show":
        return _get("todoist_show_task_comment")(task_comment_id=task_comment_id)
    elif action == "update":
        kwargs = {"task_comment_id": task_comment_id}
        if content is not NOT_GIVEN:
            kwargs["content"] = content
        return _get("todoist_update_task_comment")(**kwargs)
    elif action == "delete":
        return _get("todoist_delete_task_comment")(task_comment_id=task_comment_id)
    elif action == "list":
        kwargs = {"task_id": task_id}
        if page_index is not None:
            kwargs["page_index"] = page_index
        if page_limit is not None:
            kwargs["page_limit"] = page_limit
        return _get("todoist_show_task_comments")(**kwargs)
    else:
        raise ValueError(f"Unknown action: {action}")


mark_compact_tools_absorbed_by(
    "todoist_manage_task_comment",
    "todoist_show_task_comments",
)
