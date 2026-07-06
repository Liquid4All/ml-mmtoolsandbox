# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""Consolidated Todoist tools for the MEDIUM toolbox.

CRUD consolidation for projects, sections, tasks, sub-tasks, labels,
task comments, and a symmetric pair merge for task label add/remove.
"""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.toolbox.names import ToolboxName
from mmtoolsandbox.tools.consolidated import mark_tools_absorbed_by


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.appworld.todoist as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# CRUD: Project management
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def todoist_manage_project(
    action: Literal["create", "show", "update", "delete"],
    project_id: int | NotGiven = NOT_GIVEN,
    name: str | NotGiven = NOT_GIVEN,
    color: str | None | NotGiven = NOT_GIVEN,
    description: str | NotGiven = NOT_GIVEN,
    is_favorite: bool | None | NotGiven = NOT_GIVEN,
    is_archived: bool | None | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Manage Todoist projects: create, view, update, or delete.

    Actions:
        create: Create a new project. Requires name. color defaults to
            "charcoal", description defaults to empty string, is_favorite
            defaults to false.
        show: View details of a project. Requires project_id. Use 0 for
            your Inbox project.
        update: Update project properties. Requires project_id and at least
            one of name, color, description, is_favorite, or is_archived.
        delete: Delete a project. Requires project_id.

    Args:
        action: The operation to perform.
        project_id: The project ID (for show, update, delete). Use 0 for
            your Inbox project.
        name: Project name (for create, update).
        color: Project color (for create, update).
        description: Project description (for create, update).
        is_favorite: Whether the project is a favorite (for create, update).
        is_archived: Whether the project is archived (for update only).

    Returns:
        Project details or action confirmation.

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
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# CRUD: Section management
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def todoist_manage_section(
    action: Literal["create", "update", "delete"],
    project_id: int | NotGiven = NOT_GIVEN,
    section_id: int | NotGiven = NOT_GIVEN,
    name: str | None | NotGiven = NOT_GIVEN,
    order_index: int | None | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Manage Todoist sections within a project: create, update, or delete.

    Actions:
        create: Create a new section. Requires project_id and name.
            order_index defaults to -1 (append to end).
        update: Update section properties. Requires section_id and at least
            one of name or order_index.
        delete: Delete a section. Requires section_id.

    Args:
        action: The operation to perform.
        project_id: The project ID (for create). Use 0 for your Inbox
            project.
        section_id: The section ID (for update, delete).
        name: Section name (for create, update).
        order_index: Position index in the section list (for create, update).
            Use negative values to insert from the end (-1 = append).

    Returns:
        Section details or action confirmation.

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
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# CRUD: Task management
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def todoist_manage_task(
    action: Literal["create", "show", "update", "delete"],
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
) -> dict[str, Any]:
    """Manage Todoist tasks: create, view, update, or delete.

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

    Args:
        action: The operation to perform.
        project_id: The project ID (for create). Use 0 for inbox.
        task_id: The task ID (for show, update, delete).
        title: Task title (for create, update).
        section_id: Section ID within the project (for create).
        description: Task description (for create, update).
        due_date: Due date in YYYY-MM-DD format (for create, update).
        duration: Task duration (for create, update).
        duration_unit: Unit of the task duration (for create, update).
        order_index: Position index in the task list (for create, update).
        priority: Task priority (for create, update).
        is_completed: Whether the task is completed (for update only).

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
            kwargs["due_date"] = "None" if due_date is None else due_date
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
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# CRUD: Sub-task management
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def todoist_manage_sub_task(
    action: Literal["create", "update", "delete"],
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
) -> dict[str, Any]:
    """Manage Todoist sub-tasks: create, update, or delete.

    Actions:
        create: Create a new sub-task within a task. Requires task_id and
            title. description defaults to empty string, priority defaults
            to "medium", order_index defaults to -1.
        update: Update sub-task properties. Requires sub_task_id and at
            least one field to change. Pass due_date=null to remove the
            due date. Pass duration=0 to remove duration.
        delete: Delete a sub-task. Requires sub_task_id.

    Args:
        action: The operation to perform.
        task_id: The parent task ID (for create).
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
        Sub-task details or action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Todoist.
    """
    if action == "create":
        kwargs: dict[str, Any] = {"task_id": task_id, "title": title}
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
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# CRUD: Label management
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def todoist_manage_label(
    action: Literal["create", "show", "update", "delete"],
    label_id: int | NotGiven = NOT_GIVEN,
    name: str | NotGiven = NOT_GIVEN,
    color: str | None | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Manage Todoist labels: create, view, update, or delete.

    Actions:
        create: Create a new label. Requires name. color defaults to
            "charcoal".
        show: View label details. Requires label_id.
        update: Update label properties. Requires label_id and at least
            one of name or color.
        delete: Delete a label. Requires label_id.

    Args:
        action: The operation to perform.
        label_id: The label ID (for show, update, delete).
        name: Label name (for create, update).
        color: Label color (for create, update).

    Returns:
        Label details or action confirmation.

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
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# CRUD: Task comment management
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def todoist_manage_task_comment(
    action: Literal["post", "show", "update", "delete"],
    task_id: int | NotGiven = NOT_GIVEN,
    task_comment_id: int | NotGiven = NOT_GIVEN,
    content: str | None | NotGiven = NOT_GIVEN,
    attachment_file_paths: list[str] | None | NotGiven = NOT_GIVEN,
    file_system_access_token: str | None | NotGiven = NOT_GIVEN,
) -> dict[str, Any]:
    """Manage comments on Todoist tasks: post, view, update, or delete.

    Actions:
        post: Post a new comment on a task. Requires task_id and content.
            Optionally attach files via attachment_file_paths (requires
            file_system_access_token).
        show: View a task comment. Requires task_comment_id.
        update: Update comment content. Requires task_comment_id and
            content.
        delete: Delete a task comment. Requires task_comment_id.

    Args:
        action: The operation to perform.
        task_id: The task ID (for post).
        task_comment_id: The comment ID (for show, update, delete).
        content: Comment content (for post, update).
        attachment_file_paths: Paths to files to attach (for post).
        file_system_access_token: File system access token, needed when
            attaching files (for post).

    Returns:
        Comment details or action confirmation.

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
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Symmetric pair: Task label add/remove
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={ToolboxName.MEDIUM},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def todoist_manage_task_label(
    task_id: int,
    label_id: int,
    action: Literal["add", "remove"],
) -> dict[str, Any]:
    """Add or remove a label from a Todoist task.

    Args:
        task_id: The ID of the task.
        label_id: The ID of the label.
        action: "add" to attach the label, "remove" to detach it.

    Returns:
        Action confirmation.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Todoist.
    """
    if action == "add":
        return _get("todoist_add_label_to_task")(task_id=task_id, label_id=label_id)
    elif action == "remove":
        return _get("todoist_remove_label_from_task")(
            task_id=task_id, label_id=label_id
        )
    else:
        raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Mark absorbed tools
# ---------------------------------------------------------------------------

mark_tools_absorbed_by(
    "todoist_manage_project",
    "todoist_create_project",
    "todoist_show_project",
    "todoist_update_project",
    "todoist_delete_project",
)
mark_tools_absorbed_by(
    "todoist_manage_section",
    "todoist_create_section",
    "todoist_update_section",
    "todoist_delete_section",
)
mark_tools_absorbed_by(
    "todoist_manage_task",
    "todoist_create_task",
    "todoist_show_task",
    "todoist_update_task",
    "todoist_delete_task",
)
mark_tools_absorbed_by(
    "todoist_manage_sub_task",
    "todoist_create_sub_task",
    "todoist_update_sub_task",
    "todoist_delete_sub_task",
)
mark_tools_absorbed_by(
    "todoist_manage_label",
    "todoist_create_label",
    "todoist_show_label",
    "todoist_update_label",
    "todoist_delete_label",
)
mark_tools_absorbed_by(
    "todoist_manage_task_comment",
    "todoist_post_task_comment",
    "todoist_show_task_comment",
    "todoist_update_task_comment",
    "todoist_delete_task_comment",
)
mark_tools_absorbed_by(
    "todoist_manage_task_label",
    "todoist_add_label_to_task",
    "todoist_remove_label_from_task",
)
