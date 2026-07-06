# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""MINI Todoist tools -- 2 workflow-based tools covering all Todoist operations.

todoist_manage: All CRUD operations on projects, sections, tasks, sub-tasks,
               labels, comments, and task-label associations.
todoist_collaborate: Collaboration features (invitations, attachments).
"""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven, register_as_tool
from mmtoolsandbox.datasets.names import DatasetName


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.appworld.todoist as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# todoist_manage -- "I want to manage my tasks and projects"
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.MINI},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def todoist_manage(
    entity: Literal[
        "project", "section", "task", "sub_task", "label", "comment", "task_label"
    ],
    action: Literal[
        "create",
        "show",
        "update",
        "delete",
        "list",
        "search",
        "add",
        "remove",
        "assign",
    ],
    # -- IDs --
    project_id: int | NotGiven = NOT_GIVEN,
    section_id: int | None | NotGiven = NOT_GIVEN,
    task_id: int | NotGiven = NOT_GIVEN,
    sub_task_id: int | NotGiven = NOT_GIVEN,
    label_id: int | NotGiven = NOT_GIVEN,
    task_comment_id: int | NotGiven = NOT_GIVEN,
    # -- project params --
    name: str | NotGiven = NOT_GIVEN,
    color: str | None | NotGiven = NOT_GIVEN,
    description: str | None | NotGiven = NOT_GIVEN,
    is_favorite: bool | None | NotGiven = NOT_GIVEN,
    is_archived: bool | None | NotGiven = NOT_GIVEN,
    # -- section params --
    order_index: int | None | NotGiven = NOT_GIVEN,
    # -- task params --
    title: str | NotGiven = NOT_GIVEN,
    due_date: str | None | NotGiven = NOT_GIVEN,
    duration: float | None | NotGiven = NOT_GIVEN,
    duration_unit: Literal["minute", "day"] | None | NotGiven = NOT_GIVEN,
    priority: Literal["low", "medium", "high", "urgent"] | None | NotGiven = NOT_GIVEN,
    is_completed: bool | None | NotGiven = NOT_GIVEN,
    assignee_email: str | None | NotGiven = NOT_GIVEN,
    assigner_email: str | None | NotGiven = NOT_GIVEN,
    # -- task list filters --
    due_today: bool | None | NotGiven = NOT_GIVEN,
    overdue: bool | None | NotGiven = NOT_GIVEN,
    min_due_date: str | None | NotGiven = NOT_GIVEN,
    max_due_date: str | None | NotGiven = NOT_GIVEN,
    sort_by: str | None | NotGiven = NOT_GIVEN,
    # -- label search params --
    query: str | None | NotGiven = NOT_GIVEN,
    task_attached: bool | None | NotGiven = NOT_GIVEN,
    page_index: int | None | NotGiven = NOT_GIVEN,
    page_limit: int | None | NotGiven = NOT_GIVEN,
    # -- comment params --
    content: str | None | NotGiven = NOT_GIVEN,
    attachment_file_paths: list[str] | None | NotGiven = NOT_GIVEN,
    file_system_access_token: str | None | NotGiven = NOT_GIVEN,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Manage Todoist tasks, projects, sections, and labels.

    Dispatch is based on entity x action:

    project:
        create: Create a project. Requires name. Optional: color, description,
            is_favorite.
        show: View project details. Requires project_id. Use 0 for Inbox.
        update: Update project. Requires project_id + at least one of name,
            color, description, is_favorite, is_archived.
        delete: Delete project. Requires project_id.
        list: List projects. Optional: query, color, is_favorite, is_archived,
            page_index, page_limit, sort_by.

    section:
        create: Create section. Requires project_id, name. Optional:
            order_index (defaults to -1).
        list: List sections. Requires project_id.
        update: Update section. Requires section_id. Optional: name,
            order_index.
        delete: Delete section. Requires section_id.

    task:
        create: Create task. Requires project_id, title. Optional:
            section_id, description, due_date, duration, duration_unit,
            order_index, priority.
        show: View task. Requires task_id.
        update: Update task. Requires task_id + fields to change.
            Pass due_date=null to remove. Pass duration=0 to remove.
        delete: Delete task. Requires task_id.
        list: List tasks in project. Requires project_id. Optional:
            section_id, assignee_email, assigner_email, priority,
            is_completed, due_today, label_id, overdue, min_due_date,
            max_due_date, sort_by.
        assign: Assign/unassign task. Requires task_id. Optional:
            assignee_email (None to unassign).

    sub_task:
        create: Create sub-task. Requires task_id, title. Optional:
            description, due_date, duration, duration_unit, priority,
            order_index.
        update: Update sub-task. Requires sub_task_id + fields.
        delete: Delete sub-task. Requires sub_task_id.
        list: List sub-tasks. Requires task_id.

    label:
        create: Create label. Requires name. Optional: color.
        show: View label. Requires label_id.
        update: Update label. Requires label_id. Optional: name, color.
        delete: Delete label. Requires label_id.
        search: Search labels. Optional: query, task_id, color,
            task_attached, page_index, page_limit.

    comment:
        create: Post comment. Requires task_id, content. Optional:
            attachment_file_paths, file_system_access_token.
        show: View comment. Requires task_comment_id.
        update: Update comment. Requires task_comment_id, content.
        delete: Delete comment. Requires task_comment_id.
        list: List comments. Requires task_id. Optional: page_index,
            page_limit.

    task_label:
        add: Add label to task. Requires task_id, label_id.
        remove: Remove label from task. Requires task_id, label_id.

    Args:
        entity: The type of object to manage.
        action: The operation to perform.
        project_id: Project ID.
        section_id: Section ID.
        task_id: Task ID.
        sub_task_id: Sub-task ID.
        label_id: Label ID.
        task_comment_id: Task comment ID.
        name: Name (for project, section, label).
        color: Color (for project, label).
        description: Description (for project, task, sub_task).
        is_favorite: Favorite status (for project).
        is_archived: Archived status (for project).
        order_index: Position index (for section, task, sub_task).
        title: Title (for task, sub_task).
        due_date: Due date YYYY-MM-DD (for task, sub_task).
        duration: Duration value (for task, sub_task).
        duration_unit: Duration unit (for task, sub_task).
        priority: Priority level (for task, sub_task).
        is_completed: Completion status (for task, sub_task).
        assignee_email: Assignee email (for task assign/list).
        assigner_email: Assigner email (for task list).
        due_today: Filter tasks due today (for task list).
        overdue: Filter overdue tasks (for task list).
        min_due_date: Min due date filter (for task list).
        max_due_date: Max due date filter (for task list).
        sort_by: Sort attribute with +/- prefix (for task list, project
            list).
        query: Search query (for project list, label search).
        task_attached: Filter labels by task attachment (for label search).
        page_index: Page index for pagination.
        page_limit: Max results per page.
        content: Comment content (for comment create/update).
        attachment_file_paths: File paths to attach (for comment create).
        file_system_access_token: File system token (for comment create).

    Returns:
        For create: dict with entity ID (project_id, task_id, etc.).
            Pass to show/update/delete.
        For show: dict with full entity details.
        For list/search: list of entity dicts.
        For delete: confirmation dict. Irreversible.
        For task_label add/remove: confirmation dict.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Todoist.
        ValueError: If invalid entity/action combination or missing params.
    """
    # --- PROJECT ---
    if entity == "project":
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
            if query is not NOT_GIVEN:
                kwargs["query"] = query
            if color is not NOT_GIVEN:
                kwargs["color"] = color
            if is_favorite is not NOT_GIVEN:
                kwargs["is_favorite"] = is_favorite
            if is_archived is not NOT_GIVEN:
                kwargs["is_archived"] = is_archived
            if page_index is not NOT_GIVEN:
                kwargs["page_index"] = page_index
            if page_limit is not NOT_GIVEN:
                kwargs["page_limit"] = page_limit
            if sort_by is not NOT_GIVEN:
                kwargs["sort_by"] = sort_by
            return _get("todoist_show_projects")(**kwargs)
        else:
            raise ValueError(f"Invalid action '{action}' for entity 'project'")

    # --- SECTION ---
    elif entity == "section":
        if action == "create":
            kwargs = {"project_id": project_id, "name": name}
            if order_index is not NOT_GIVEN:
                kwargs["order_index"] = order_index
            return _get("todoist_create_section")(**kwargs)
        elif action == "list":
            return _get("todoist_show_sections")(project_id=project_id)
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
            raise ValueError(f"Invalid action '{action}' for entity 'section'")

    # --- TASK ---
    elif entity == "task":
        if action == "create":
            kwargs = {"project_id": project_id, "title": title}
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
        elif action == "list":
            kwargs = {"project_id": project_id}
            if section_id is not NOT_GIVEN:
                kwargs["section_id"] = section_id
            if assignee_email is not NOT_GIVEN:
                kwargs["assignee_email"] = assignee_email
            if assigner_email is not NOT_GIVEN:
                kwargs["assigner_email"] = assigner_email
            if priority is not NOT_GIVEN:
                kwargs["priority"] = priority
            if is_completed is not NOT_GIVEN:
                kwargs["is_completed"] = is_completed
            if due_today is not NOT_GIVEN:
                kwargs["due_today"] = due_today
            if label_id is not NOT_GIVEN:
                kwargs["label_id"] = label_id
            if overdue is not NOT_GIVEN:
                kwargs["overdue"] = overdue
            if min_due_date is not NOT_GIVEN:
                kwargs["min_due_date"] = min_due_date
            if max_due_date is not NOT_GIVEN:
                kwargs["max_due_date"] = max_due_date
            if sort_by is not NOT_GIVEN:
                kwargs["sort_by"] = sort_by
            return _get("todoist_show_tasks")(**kwargs)
        elif action == "assign":
            kwargs = {"task_id": task_id}
            if assignee_email is not NOT_GIVEN:
                kwargs["assignee_email"] = assignee_email
            return _get("todoist_assign_or_unassign_task")(**kwargs)
        else:
            raise ValueError(f"Invalid action '{action}' for entity 'task'")

    # --- SUB_TASK ---
    elif entity == "sub_task":
        if action == "create":
            kwargs = {"task_id": task_id, "title": title}
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
        elif action == "list":
            return _get("todoist_show_sub_tasks")(task_id=task_id)
        else:
            raise ValueError(f"Invalid action '{action}' for entity 'sub_task'")

    # --- LABEL ---
    elif entity == "label":
        if action == "create":
            kwargs = {"name": name}
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
            if query is not NOT_GIVEN:
                kwargs["query"] = query
            if task_id is not NOT_GIVEN:
                kwargs["task_id"] = task_id
            if color is not NOT_GIVEN:
                kwargs["color"] = color
            if task_attached is not NOT_GIVEN:
                kwargs["task_attached"] = task_attached
            if page_index is not NOT_GIVEN:
                kwargs["page_index"] = page_index
            if page_limit is not NOT_GIVEN:
                kwargs["page_limit"] = page_limit
            return _get("todoist_search_labels")(**kwargs)
        else:
            raise ValueError(f"Invalid action '{action}' for entity 'label'")

    # --- COMMENT ---
    elif entity == "comment":
        if action == "create":
            kwargs = {"task_id": task_id, "content": content}
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
            if page_index is not NOT_GIVEN:
                kwargs["page_index"] = page_index
            if page_limit is not NOT_GIVEN:
                kwargs["page_limit"] = page_limit
            return _get("todoist_show_task_comments")(**kwargs)
        else:
            raise ValueError(f"Invalid action '{action}' for entity 'comment'")

    # --- TASK_LABEL ---
    elif entity == "task_label":
        if action == "add":
            return _get("todoist_add_label_to_task")(task_id=task_id, label_id=label_id)
        elif action == "remove":
            return _get("todoist_remove_label_from_task")(
                task_id=task_id, label_id=label_id
            )
        else:
            raise ValueError(f"Invalid action '{action}' for entity 'task_label'")

    else:
        raise ValueError(f"Unknown entity: {entity}")


# ---------------------------------------------------------------------------
# todoist_collaborate -- "I want to work with others on tasks"
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.MINI},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def todoist_collaborate(
    domain: Literal["invitation", "attachment"],
    action: Literal[
        "invite", "accept", "decline", "remove", "upload", "download", "delete"
    ],
    # -- invitation params --
    project_id: int | NotGiven = NOT_GIVEN,
    email: str | NotGiven = NOT_GIVEN,
    invite_code: str | NotGiven = NOT_GIVEN,
    # -- attachment params --
    task_comment_id: int | NotGiven = NOT_GIVEN,
    upload_from_file_path: str | NotGiven = NOT_GIVEN,
    attachment_file_name: str | NotGiven = NOT_GIVEN,
    file_system_access_token: str | NotGiven = NOT_GIVEN,
    download_to_file_path: str | None = None,
    overwrite: bool | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Collaborate on Todoist: comments, sharing, and import/export.

    Domains:
        invitation: Manage project collaboration invitations.
            action="invite": Send project invite. Requires project_id, email.
            action="accept": Accept invite. Requires invite_code.
            action="decline": Decline/delete invite. Requires invite_code.
            action="remove": Remove collaborator. Requires project_id, email.
        attachment: Manage task comment attachments.
            action="upload": Upload attachment. Requires task_comment_id,
                upload_from_file_path, file_system_access_token. Optional:
                overwrite.
            action="download": Download attachment. Requires task_comment_id,
                attachment_file_name, file_system_access_token. Optional:
                download_to_file_path, overwrite.
            action="delete": Delete attachment. Requires task_comment_id,
                attachment_file_name.

    Args:
        domain: The collaboration area.
        action: The operation to perform.
        project_id: Project ID (for invitation invite/remove).
        email: Collaborator email (for invitation invite/remove).
        invite_code: Invitation code (for invitation accept/decline).
        task_comment_id: Comment ID (for attachment operations).
        upload_from_file_path: File path to upload (for attachment upload).
        attachment_file_name: Name of attached file (for attachment
            download/delete).
        file_system_access_token: File system token (for attachment
            upload/download).
        download_to_file_path: Path to save downloaded file (for attachment
            download).
        overwrite: Whether to overwrite existing files (for attachment
            upload/download).

    Returns:
        For invitation invite: sends invite. Externally visible.
        For attachment upload/download/delete: confirmation dict.

    Raises:
        ConnectionError: If network is unavailable.
        PermissionError: If not logged into Todoist.
        ValueError: If invalid domain/action or missing parameters.
    """
    if domain == "invitation":
        if action == "invite":
            return _get("todoist_send_project_invite")(
                project_id=project_id, email=email
            )
        elif action == "accept":
            return _get("todoist_accept_project_invite")(invite_code=invite_code)
        elif action == "decline":
            return _get("todoist_delete_project_invite")(invite_code=invite_code)
        elif action == "remove":
            return _get("todoist_remove_collaborator_from_project")(
                project_id=project_id, email=email
            )
        else:
            raise ValueError(f"Unknown invitation action: {action}")

    elif domain == "attachment":
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
            raise ValueError(f"Unknown attachment action: {action}")

    else:
        raise ValueError(f"Unknown domain: {domain}")
