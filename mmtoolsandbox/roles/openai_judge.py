# Copyright © 2026 Apple Inc.

"""Judge role for evaluating task completion using OpenAI API."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from tenacity import retry, stop_after_attempt, wait_random_exponential

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution_context import ExecutionContext, RoleType
from mmtoolsandbox.common.state_diff import TableDiff, render_state_diff
from mmtoolsandbox.common.utils import all_logging_disabled
from mmtoolsandbox.roles.base_judge import (
    _INFRASTRUCTURE_NAMESPACES,
    _MAX_IMAGES,
    _SYSTEM_PROMPT,
    _UI_SYSTEM_PROMPT,
    JudgeResult,
    get_user_system_prompt,
)

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# User judge evidence constants
# ---------------------------------------------------------------------------

# Regex for parsing scripted round image requirements from the SYSTEM→USER
# instruction, e.g. "ROUND 2 [provide: image_1, image_3]".
_USER_PROVIDE_PATTERN = re.compile(r"ROUND\s+(\d+)\s*\[provide:\s*(.*?)\]")

# Regex for extracting per-round Query text from the SYSTEM→USER instruction.
# Matches "ROUND N [provide: ...]<newline>Query: ..." up to "Instructions:".
_USER_QUERY_PATTERN = re.compile(
    r"ROUND\s+\d+(?:\s*\[provide:.*?\])?\s*\n"
    r"Query:\s*(.*?)(?=\nInstructions:)",
    re.DOTALL,
)

# If the actual user message is shorter than this fraction of the scripted
# Query, flag it as potentially truncated.  0.5 = less than half the length.
_USER_QUERY_TRUNCATION_THRESHOLD = 0.5

# Metadata keys forwarded from scenario_metadata to the judge evidence.
_USER_METADATA_KEYS = (
    "challenge_type",
    "require_disambiguation",
    "num_user_rounds",
    "image_arrival",
)

_UI_TOOL_NAMES = frozenset({"render_ui_screen", "show_ui_to_user", "ui_user_interact"})


def _summarize_ui_json(ui_json: Any) -> str:
    """Produce a structural summary from a render_ui_screen ui_json argument.

    Parses the surfaceUpdate messages to count component types, measure
    nesting depth, and list buttons with their labels and action bindings.
    Works on the raw tool trace argument — no live SurfaceState needed.
    """
    if isinstance(ui_json, str):
        try:
            ui_json = json.loads(ui_json)
        except (json.JSONDecodeError, TypeError):
            return "(unparseable ui_json)"

    messages = ui_json if isinstance(ui_json, list) else [ui_json]

    components: dict[str, dict[str, Any]] = {}
    root_id: str | None = None
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if "beginRendering" in msg:
            root_id = msg["beginRendering"].get("root")
        if "surfaceUpdate" in msg:
            for comp in msg["surfaceUpdate"].get("components", []):
                comp_id = comp.get("id", "")
                if comp_id:
                    components[comp_id] = comp

    if not components:
        return "(no components)"

    type_counts: dict[str, int] = {}
    buttons: list[str] = []

    for comp_id, comp_def in components.items():
        wrapper = comp_def.get("component", {})
        if not wrapper:
            continue
        comp_type = next(iter(wrapper))
        type_counts[comp_type] = type_counts.get(comp_type, 0) + 1

        if comp_type == "Button":
            props = wrapper["Button"]
            action = props.get("action", {})
            action_name = action.get("name", "?")
            child_id = props.get("child", "")
            label = child_id
            if child_id and child_id in components:
                child_wrapper = components[child_id].get("component", {})
                if "Text" in child_wrapper:
                    text_prop = child_wrapper["Text"].get("text", {})
                    label = text_prop.get("literalString", child_id)
            buttons.append(f'"{label}" ({action_name})')

    total = sum(type_counts.values())
    type_str = ", ".join(
        f"{count} {name}" for name, count in sorted(type_counts.items())
    )

    def _depth(cid: str, visited: set[str]) -> int:
        if cid in visited or cid not in components:
            return 0
        visited.add(cid)
        wrapper = components[cid].get("component", {})
        if not wrapper:
            return 1
        comp_type = next(iter(wrapper))
        props = wrapper[comp_type]
        child_depths: list[int] = []
        for key in ("child", "entryPointChild", "contentChild"):
            child = props.get(key, "")
            if child:
                child_depths.append(_depth(child, visited))
        for child in props.get("children", []):
            if isinstance(child, str):
                child_depths.append(_depth(child, visited))
        return 1 + (max(child_depths) if child_depths else 0)

    max_depth = _depth(root_id or "", set()) if root_id else 0

    parts = [f"{total} components ({type_str})", f"depth {max_depth}"]
    if buttons:
        parts.append("buttons: " + ", ".join(buttons))
    else:
        parts.append("no buttons")

    return "; ".join(parts)


class OpenAIAPIJudge:
    """Judge that uses OpenAI API to evaluate task completion with rubric scoring."""

    def __init__(self, model_name: str = "gpt-4o") -> None:
        self.client = OpenAI()
        self.model_name = model_name
        # Cached evidence from the most recent evaluate* calls.
        # Keys: "main", "user", "ui".  Values: list of OpenAI content parts.
        self.last_evidence: dict[str, list[dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Evidence formatting helpers
    # ------------------------------------------------------------------

    def _format_user_assistant_conversation(
        self, sandbox_rows: list[dict[str, Any]]
    ) -> str:
        """Format the USER↔AGENT conversation thread.

        Includes all messages where sender or recipient is USER or AGENT (both
        directions), skipping SYSTEM and EXECUTION_ENVIRONMENT rows. Annotates
        messages that carry images with [image_id=N] markers.
        """
        lines: list[str] = []
        for row in sandbox_rows:
            sender = row["sender"]
            recipient = row["recipient"]
            # Only include USER↔AGENT exchanges
            is_user_to_agent = sender == RoleType.USER and recipient == RoleType.AGENT
            is_agent_to_user = sender == RoleType.AGENT and recipient == RoleType.USER
            if not (is_user_to_agent or is_agent_to_user):
                continue
            content = row.get("content") or ""
            role_label = "[USER]" if sender == RoleType.USER else "[AGENT]"
            line = f"{role_label}: {content}"
            # Annotate images
            image_ids = row.get("image_ids")
            if image_ids:
                markers = " ".join(
                    f"[image_id={img_id}]" for img_id in image_ids if img_id is not None
                )
                if markers:
                    line += f" {markers}"
            lines.append(line)
        return "\n".join(lines) if lines else "No conversation."

    def _format_agent_environment_interactions(
        self, sandbox_rows: list[dict[str, Any]]
    ) -> str:
        """Format the chronological AGENT↔EXECUTION_ENVIRONMENT action trace.

        Shows what the agent sent to the environment and what came back, including
        tool call results (from tool_trace) and errors (from tool_call_exception).
        """
        lines: list[str] = []
        for row in sandbox_rows:
            sender = row["sender"]
            recipient = row["recipient"]

            # AGENT → EXEC_ENV: show what the agent sent
            if sender == RoleType.AGENT and recipient == RoleType.EXECUTION_ENVIRONMENT:
                content = row.get("content") or ""
                lines.append(f"[AGENT → ENV]: {content}")

            # EXEC_ENV → AGENT: show results or errors
            elif (
                sender == RoleType.EXECUTION_ENVIRONMENT and recipient == RoleType.AGENT
            ):
                exception = row.get("tool_call_exception")
                if exception:
                    lines.append(f"[ENV → AGENT]: ERROR: {exception}")
                    continue

                tool_trace = row.get("tool_trace")
                if tool_trace:
                    for trace_json in tool_trace:
                        try:
                            trace = json.loads(trace_json)
                            name = trace.get("tool_name", "unknown")
                            args = trace.get("arguments", {})
                            result = trace.get("result")
                            args_str = ", ".join(
                                f"{k}={json.dumps(v, ensure_ascii=False)}"
                                for k, v in args.items()
                            )
                            lines.append(
                                f"[ENV → AGENT]: {name}({args_str}) → "
                                f"{json.dumps(result, ensure_ascii=False)}"
                            )
                        except (json.JSONDecodeError, TypeError):
                            lines.append(f"[ENV → AGENT]: {row.get('content', '')}")
                else:
                    content = row.get("content") or ""
                    lines.append(f"[ENV → AGENT]: {content}")

        return "\n".join(lines) if lines else "No agent actions."

    def _native_table_diffs(
        self, execution_context: ExecutionContext
    ) -> list[TableDiff]:
        """Gather initial→final row sets for MMToolSandbox native namespaces.

        Returns one :class:`TableDiff` per changed namespace (skipping
        infrastructure namespaces and unchanged ones). Rows are plain dicts, so
        the shared renderer can key + classify them uniformly.
        """
        initial_idx = execution_context.first_user_sandbox_message_index
        final_idx = execution_context.max_sandbox_message_index
        tables: list[TableDiff] = []

        for ns in execution_context.get_active_database_namespaces():
            if ns in _INFRASTRUCTURE_NAMESPACES:
                continue
            try:
                initial = execution_context.get_database(
                    ns, sandbox_message_index=initial_idx
                )
                final = execution_context.get_database(
                    ns, sandbox_message_index=final_idx
                )
            except (IndexError, KeyError):
                continue

            if initial.is_empty() and final.is_empty():
                continue
            # Cheap short-circuit for unchanged namespaces (avoids a to_dicts
            # on large tables). The renderer would also emit nothing, but this
            # skips the work.
            if initial.equals(final):
                continue

            tables.append(
                TableDiff(
                    label=str(ns),
                    initial=initial.to_dicts(),
                    final=final.to_dicts(),
                )
            )
        return tables

    def _appworld_table_diffs(
        self, appworld_initial: dict[str, Any] | None
    ) -> list[TableDiff]:
        """Gather initial→final row sets for AppWorld tables.

        Initial rows come from the pre-agent snapshots captured by the entity
        diff evaluator; final rows are read live from the AppWorld SQLite.
        """
        if not appworld_initial:
            return []
        try:
            from mmtoolsandbox.common.entity_diff_evaluator import (
                read_appworld_table,
            )
        except Exception:
            return []

        tables: list[TableDiff] = []
        for table_key, initial_df in appworld_initial.items():
            try:
                final_df = read_appworld_table(table_key)
                tables.append(
                    TableDiff(
                        label=table_key,
                        initial=initial_df.to_dicts() if initial_df is not None else [],
                        final=final_df.to_dicts() if final_df is not None else [],
                    )
                )
            except Exception:
                continue
        return tables

    def _format_database_diff(
        self,
        execution_context: ExecutionContext,
        appworld_tables: list[TableDiff] | None = None,
    ) -> str:
        """Render a unified initial→final state diff for the judge.

        Combines MMToolSandbox native namespaces (gathered from the execution
        context) with any pre-built AppWorld ``TableDiff``s, then delegates to
        the shared :func:`render_state_diff`, which classifies added / updated /
        deleted rows with content — the same view regardless of storage backend.

        Args:
            execution_context: MMToolSandbox execution context.
            appworld_tables: Pre-built AppWorld table diffs. ``None`` or empty
                means no external tables.
        """
        tables = self._native_table_diffs(execution_context)
        if appworld_tables:
            tables += appworld_tables
        return render_state_diff(tables)

    def _format_images(
        self,
        sandbox_rows: list[dict[str, Any]],
        execution_context: ExecutionContext,
    ) -> list[dict[str, Any]]:
        """Collect relevant images, labeled with their id, deduped and capped.

        Selection priority (for the ``_MAX_IMAGES`` cap):
        1. Images from every USER→AGENT message (user-provided). Images delivered
           in later turns via ``send_message_with_image`` are intercepted into
           USER→AGENT messages rather than the tool trace, so every user turn
           must be scanned for them to be visible to the judge.
        2. Tool results that reference images (view_image, crop_image, etc.).
        3. Last AGENT→USER message images (agent output).

        Each returned image is preceded by a ``[image_id=N]`` text part so the
        judge can bind it to the ``[image_id=N]`` references in the conversation
        text. Images are displayed sorted by id (ordering is otherwise
        meaningless once labeled).
        """
        try:
            image_db = execution_context.get_database(DatabaseNamespace.IMAGE)
        except (IndexError, KeyError):
            return []

        if image_db.is_empty():
            return []

        image_lookup: dict[int, str] = {}
        for row in image_db.to_dicts():
            image_lookup[row["image_id"]] = row["image_content"]

        user_ids: list[int] = []
        last_agent_ids: list[int] = []
        tool_image_ids: list[int] = []

        for row in sandbox_rows:
            sender = row.get("sender")
            recipient = row.get("recipient")

            # Source 1: every USER→AGENT message.
            image_ids = row.get("image_ids")
            if image_ids:
                if sender == RoleType.USER and recipient == RoleType.AGENT:
                    user_ids.extend(i for i in image_ids if i is not None)
                if sender == RoleType.AGENT and recipient == RoleType.USER:
                    last_agent_ids = [i for i in image_ids if i is not None]

            # Source 2: image_ids from tool_trace (view_image, crop_image results)
            if sender == RoleType.EXECUTION_ENVIRONMENT:
                tool_trace = row.get("tool_trace")
                if tool_trace:
                    for trace_json in tool_trace:
                        try:
                            trace = json.loads(trace_json)
                            args = trace.get("arguments", {})
                            # Collect image_id from tool arguments
                            if "image_id" in args and isinstance(args["image_id"], int):
                                tool_image_ids.append(args["image_id"])
                        except (json.JSONDecodeError, TypeError):
                            continue

        # Priority order for the cap: user-delivered, then tool-referenced, then
        # the agent's output images. Dedup preserving first-seen so user images
        # are never truncated in favor of tool/agent ones.
        ordered = user_ids + tool_image_ids + last_agent_ids
        seen: set[int] = set()
        selected: list[int] = []
        for img_id in ordered:
            if img_id not in seen and img_id in image_lookup:
                seen.add(img_id)
                selected.append(img_id)
        selected = selected[:_MAX_IMAGES]

        # Display sorted by id, each image preceded by its [image_id=N] label.
        parts: list[dict[str, Any]] = []
        for img_id in sorted(selected):
            parts.append({"type": "text", "text": f"[image_id={img_id}]"})
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_lookup[img_id]}"
                    },
                }
            )
        return parts

    # ------------------------------------------------------------------
    # Evidence assembly
    # ------------------------------------------------------------------

    def _format_evidence(
        self,
        execution_context: ExecutionContext,
        criteria: str,
        appworld_tables: list[TableDiff] | None = None,
    ) -> list[dict[str, Any]]:
        """Build a curated evidence package for the judge model."""
        sandbox_db = execution_context.get_database(
            DatabaseNamespace.SANDBOX,
            get_all_history_snapshots=True,
            drop_sandbox_message_index=False,
        )
        sandbox_rows = sandbox_db.to_dicts()

        conversation = self._format_user_assistant_conversation(sandbox_rows)
        agent_actions = self._format_agent_environment_interactions(sandbox_rows)
        db_diff = self._format_database_diff(execution_context, appworld_tables)

        text = f"Task Completion Criteria:\n{criteria}\n\n"
        text += f"Conversation:\n{conversation}\n\n"
        text += f"Agent Actions:\n{agent_actions}\n\n"
        text += f"Database Changes:\n{db_diff}\n"

        parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
        parts += self._format_images(sandbox_rows, execution_context)
        parts += [{"type": "text", "text": "\nEvaluate if the task was completed."}]
        return parts

    # ------------------------------------------------------------------
    # UI judge evidence
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_ui_requirements(criteria: str) -> str:
        """Extract the 'UI Requirements:' section from task completion criteria."""
        marker = "UI Requirements:"
        idx = criteria.find(marker)
        if idx == -1:
            return criteria
        return criteria[idx:]

    def _format_ui_images(
        self,
        sandbox_rows: list[dict[str, Any]],
        execution_context: ExecutionContext,
    ) -> list[dict[str, Any]]:
        """Collect only UI-rendered images, excluding user-provided images."""
        try:
            image_db = execution_context.get_database(DatabaseNamespace.IMAGE)
        except (IndexError, KeyError):
            return []

        if image_db.is_empty():
            return []

        image_lookup: dict[int, str] = {}
        for row in image_db.to_dicts():
            image_lookup[row["image_id"]] = row["image_content"]

        ui_image_ids: list[int] = []
        for row in sandbox_rows:
            sender = row.get("sender")
            recipient = row.get("recipient")
            image_ids = row.get("image_ids")

            # render_ui_screen / ui_user_interact results returned to agent
            if sender == RoleType.EXECUTION_ENVIRONMENT and image_ids:
                tool_trace = row.get("tool_trace")
                if tool_trace:
                    for trace_json in tool_trace:
                        try:
                            trace = json.loads(trace_json)
                            if trace.get("tool_name") in _UI_TOOL_NAMES:
                                ui_image_ids.extend(
                                    i for i in image_ids if i is not None
                                )
                                break
                        except (json.JSONDecodeError, TypeError):
                            continue

            # show_ui_to_user injected images (AGENT→USER, visible only to user)
            if (
                sender == RoleType.AGENT
                and recipient == RoleType.USER
                and row.get("visible_to") == [RoleType.USER]
                and image_ids
            ):
                ui_image_ids.extend(i for i in image_ids if i is not None)

        # Deduplicate preserving order, then cap
        seen: set[int] = set()
        unique_ids: list[int] = []
        for img_id in ui_image_ids:
            if img_id not in seen and img_id in image_lookup:
                seen.add(img_id)
                unique_ids.append(img_id)
        unique_ids = unique_ids[:_MAX_IMAGES]

        return [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_lookup[img_id]}"},
            }
            for img_id in unique_ids
            if img_id in image_lookup
        ]

    def _format_ui_evidence(
        self,
        execution_context: ExecutionContext,
        criteria: str,
    ) -> list[dict[str, Any]]:
        """Build a focused evidence package for the UI quality judge.

        Unlike ``_format_evidence``, this includes only UI-relevant signals:
        render_ui_screen component trees, UI tool calls, user interactions,
        and rendered screenshots (no user-provided images or functional tools).
        """
        sandbox_db = execution_context.get_database(
            DatabaseNamespace.SANDBOX,
            get_all_history_snapshots=True,
            drop_sandbox_message_index=False,
        )
        sandbox_rows = sandbox_db.to_dicts()

        ui_requirements = self._extract_ui_requirements(criteria)

        # Extract render_ui_screen structural summaries
        render_summaries: list[str] = []
        # Extract all UI-related tool interactions
        ui_trace_lines: list[str] = []
        # Extract user interactions
        interaction_lines: list[str] = []

        render_idx = 0
        for row in sandbox_rows:
            sender = row.get("sender")
            recipient = row.get("recipient")

            # AGENT → ENV: agent calling a UI tool
            if sender == RoleType.AGENT and recipient == RoleType.EXECUTION_ENVIRONMENT:
                content = row.get("content") or ""
                func_name = row.get("openai_function_name") or ""
                if func_name in _UI_TOOL_NAMES or any(
                    t in content for t in _UI_TOOL_NAMES
                ):
                    ui_trace_lines.append(f"[AGENT → ENV]: {content[:500]}")

            # ENV → AGENT: results from UI tools
            if sender == RoleType.EXECUTION_ENVIRONMENT and recipient == RoleType.AGENT:
                tool_trace = row.get("tool_trace")
                if tool_trace:
                    for trace_json in tool_trace:
                        try:
                            trace = json.loads(trace_json)
                            name = trace.get("tool_name", "")
                            if name not in _UI_TOOL_NAMES:
                                continue
                            args = trace.get("arguments", {})
                            result = trace.get("result")

                            if name == "render_ui_screen":
                                render_idx += 1
                                ui_json = args.get("ui_json", "")
                                summary = _summarize_ui_json(ui_json)
                                render_summaries.append(
                                    f"Render #{render_idx}: {summary}"
                                )
                                ui_trace_lines.append(
                                    f"[ENV → AGENT]: render_ui_screen → "
                                    f"{json.dumps(result, ensure_ascii=False)[:300]}"
                                )
                            elif name == "ui_user_interact":
                                args_str = json.dumps(args, ensure_ascii=False)
                                interaction_lines.append(
                                    f"ui_user_interact({args_str}) → "
                                    f"{json.dumps(result, ensure_ascii=False)[:300]}"
                                )
                            else:
                                ui_trace_lines.append(
                                    f"[ENV → AGENT]: {name}() → "
                                    f"{json.dumps(result, ensure_ascii=False)[:300]}"
                                )
                        except (json.JSONDecodeError, TypeError):
                            continue

            # ENV → AGENT: ui_state_server notifications
            if sender == RoleType.EXECUTION_ENVIRONMENT and recipient == RoleType.AGENT:
                content = row.get("content") or ""
                if "<ui_state_server>" in content:
                    ui_trace_lines.append(f"[ENV → AGENT]: {content}")

            # USER → ENV: user interaction tool calls
            if sender == RoleType.USER and recipient == RoleType.EXECUTION_ENVIRONMENT:
                tool_trace = row.get("tool_trace")
                if tool_trace:
                    for trace_json in tool_trace:
                        try:
                            trace = json.loads(trace_json)
                            if trace.get("tool_name") == "ui_user_interact":
                                args = trace.get("arguments", {})
                                result = trace.get("result")
                                interaction_lines.append(
                                    f"ui_user_interact("
                                    f"{json.dumps(args, ensure_ascii=False)}) → "
                                    f"{json.dumps(result, ensure_ascii=False)[:300]}"
                                )
                        except (json.JSONDecodeError, TypeError):
                            continue

        # Assemble text
        text = f"UI Requirements:\n{ui_requirements}\n\n"

        if render_summaries:
            text += "UI Structure Summary:\n"
            text += "\n".join(render_summaries)
            text += "\n\n"
        else:
            text += "UI Structure Summary: No render_ui_screen calls found.\n\n"

        if ui_trace_lines:
            text += "UI Tool Trace:\n"
            text += "\n".join(ui_trace_lines)
            text += "\n\n"

        if interaction_lines:
            text += "User Interactions:\n"
            text += "\n".join(interaction_lines)
            text += "\n"
        else:
            text += "User Interactions: None.\n"

        parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
        parts += self._format_ui_images(sandbox_rows, execution_context)
        parts += [{"type": "text", "text": "\nEvaluate the UI quality."}]
        return parts

    # ------------------------------------------------------------------
    # User judge evidence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_tool_call_text_leak(content: str) -> bool:
        """Return True if content is tool-call syntax leaked as plain text.

        The user simulator LLM sometimes writes function-call syntax
        (e.g., ``end_conversation(**params)``) as message text instead of
        using the tool-calling API.  These should be filtered from the
        conversation evidence shown to the judge.
        """
        return "toolu_vrtx_" in content or "end_conversation(**" in content

    @staticmethod
    def _build_image_delivery_summary(
        user_instruction: str,
        sandbox_rows: list[dict[str, Any]],
    ) -> str | None:
        """Cross-reference scripted [provide: image_N] tags against actual delivery.

        Parses the SYSTEM→USER instruction for required image deliveries per
        round, then checks which image IDs actually appeared on USER→AGENT
        messages in the conversation.

        Args:
            user_instruction: The full SYSTEM→USER instruction text containing
                ``[provide: image_N, ...]`` tags.
            sandbox_rows: List of sandbox DB row dicts from the trajectory.

        Returns:
            A multi-line summary string for the judge, or ``None`` if no
            images are required by the script.
        """
        required_images: dict[int, list[int]] = {}
        for match in _USER_PROVIDE_PATTERN.finditer(user_instruction):
            round_num = int(match.group(1))
            names = [s.strip() for s in match.group(2).split(",")]
            # image_N names are 1-indexed in the script; convert to 0-indexed IDs
            ids: list[int] = []
            for name in names:
                id_match = re.match(r"image_(\d+)", name)
                if id_match:
                    ids.append(int(id_match.group(1)) - 1)
            required_images[round_num] = ids

        if not required_images:
            return None

        # Collect the union of all image IDs that appeared on USER→AGENT messages.
        delivered: set[int] = set()
        for row in sandbox_rows:
            if (
                row.get("sender") == RoleType.USER
                and row.get("recipient") == RoleType.AGENT
            ):
                img = row.get("image_ids")
                if img:
                    delivered.update(i for i in img if i is not None)

        lines = ["Image Delivery Verification:"]
        all_ok = True
        for round_num in sorted(required_images):
            req = required_images[round_num]
            missing = [i for i in req if i not in delivered]
            if missing:
                lines.append(
                    f"  Round {round_num}: required image IDs {req},"
                    f" MISSING IDs {missing}"
                )
                all_ok = False
            else:
                lines.append(
                    f"  Round {round_num}: required image IDs {req}, all delivered"
                )
        if all_ok:
            lines.append("  All required images were delivered.")
        else:
            lines.append("  WARNING: Some required images were NOT delivered.")
        return "\n".join(lines)

    @staticmethod
    def _build_query_delivery_summary(
        user_instruction: str,
        sandbox_rows: list[dict[str, Any]],
    ) -> str | None:
        """Compare each round's scripted Query against actual USER-AGENT messages.

        Uses content-aware best-match: for each scripted Query, finds the
        actual message with the highest word overlap (Jaccard similarity),
        rather than naive sequential matching.  This handles conversations
        where brief acknowledgments or mid-round interactions shift message
        indices.

        When the best-matching message is still below the truncation
        threshold, both texts are shown side-by-side for the judge.

        Args:
            user_instruction: The full SYSTEM→USER instruction text containing
                ``Query:`` fields for each round.
            sandbox_rows: List of sandbox DB row dicts from the trajectory.

        Returns:
            A multi-line summary string for the judge, or ``None`` if no
            scripted queries are found in the instruction.
        """
        scripted_queries = [
            m.group(1).strip() for m in _USER_QUERY_PATTERN.finditer(user_instruction)
        ]
        if not scripted_queries:
            return None

        # Collect USER→AGENT messages, excluding tool-call text leaks.
        actual_messages: list[str] = []
        for row in sandbox_rows:
            if (
                row.get("sender") == RoleType.USER
                and row.get("recipient") == RoleType.AGENT
            ):
                content = row.get("content", "")
                if OpenAIAPIJudge._is_tool_call_text_leak(content):
                    continue
                actual_messages.append(content)

        if not actual_messages:
            return None

        def _word_set(text: str) -> set[str]:
            return set(text.lower().split())

        lines = ["Round Query Delivery Check:"]
        for i, scripted in enumerate(scripted_queries):
            round_num = i + 1
            if not actual_messages:
                lines.append(f"  Round {round_num}: NOT delivered (conversation ended)")
                continue

            scripted_words = _word_set(scripted)
            # Find best-matching actual message by word overlap (Jaccard).
            best_ratio = 0.0
            best_idx = 0
            best_jaccard = 0.0
            for j, msg in enumerate(actual_messages):
                msg_words = _word_set(msg)
                union = scripted_words | msg_words
                if union:
                    jaccard = len(scripted_words & msg_words) / len(union)
                else:
                    jaccard = 0.0
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_idx = j
                    best_ratio = len(msg) / max(len(scripted), 1)

            if best_jaccard >= 0.3:
                # Good content match found.
                if best_ratio < _USER_QUERY_TRUNCATION_THRESHOLD:
                    lines.append(
                        f"  Round {round_num}: WARNING — best matching message"
                        f" is {best_ratio:.0%} the length of scripted Query"
                        f" (Jaccard={best_jaccard:.2f})."
                        f" Key details may be missing."
                    )
                    lines.append(f"    Scripted: {scripted[:200]}")
                    lines.append(f"    Actual:   {actual_messages[best_idx][:200]}")
                else:
                    lines.append(
                        f"  Round {round_num}: delivered"
                        f" ({best_ratio:.0%} length, Jaccard={best_jaccard:.2f})"
                    )
                # Remove matched message so it's not reused for another round.
                actual_messages.pop(best_idx)
            elif i < len(actual_messages):
                # Low overlap — fall back to sequential position.
                ratio = len(actual_messages[i]) / max(len(scripted), 1)
                if ratio < _USER_QUERY_TRUNCATION_THRESHOLD:
                    lines.append(
                        f"  Round {round_num}: WARNING — actual message is"
                        f" {ratio:.0%} the length of scripted Query."
                        f" Key details may be missing."
                    )
                    lines.append(f"    Scripted: {scripted[:200]}")
                    lines.append(f"    Actual:   {actual_messages[i][:200]}")
                else:
                    lines.append(
                        f"  Round {round_num}: delivered ({ratio:.0%} length ratio)"
                    )
                actual_messages.pop(i if i < len(actual_messages) else 0)
            else:
                lines.append(f"  Round {round_num}: NOT delivered (conversation ended)")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # User judge evidence builder
    # ------------------------------------------------------------------

    def _format_user_evidence(
        self,
        execution_context: ExecutionContext,
        max_messages: int = 40,
        turn_count: int | None = None,
        scenario_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Build evidence package for the user simulator judge.

        Key difference from ``_format_evidence``: includes the hidden SYSTEM→USER
        instruction (the agent judge never sees this) and separates user tool
        calls from conversation messages.

        The evidence includes several *computed verification summaries* that
        pre-check common failure modes (missing images, truncated queries).
        These give the judge reliable, structured signals instead of requiring
        it to cross-reference raw text — which LLM judges do unreliably.

        Args:
            execution_context: The execution context containing the trajectory.
            max_messages: Scenario message budget (from ``Scenario.max_messages``).
            turn_count: Actual number of turns in the conversation.
            scenario_metadata: Optional scenario metadata dict with keys like
                ``challenge_type``, ``require_disambiguation``,
                ``num_user_rounds``, ``image_arrival``.

        Returns:
            A list of OpenAI-format content parts (text + images) for the judge.
        """
        sandbox_db = execution_context.get_database(
            DatabaseNamespace.SANDBOX,
            get_all_history_snapshots=True,
            drop_sandbox_message_index=False,
        )
        sandbox_rows = sandbox_db.to_dicts()

        # --- 1. Ground truth: the hidden SYSTEM→USER instruction ---
        user_instruction = ""
        for row in sandbox_rows:
            if (
                row.get("sender") == RoleType.SYSTEM
                and row.get("recipient") == RoleType.USER
            ):
                user_instruction = row.get("content", "")

        # --- 2. USER↔AGENT conversation (natural language only) ---
        # Filter out tool-call text leaks where the user LLM wrote
        # function-call syntax as plain text instead of using the tool API.
        user_agent_lines: list[str] = []
        for row in sandbox_rows:
            sender = row.get("sender")
            recipient = row.get("recipient")
            content = row.get("content", "")
            if sender == RoleType.USER and recipient == RoleType.AGENT:
                if OpenAIAPIJudge._is_tool_call_text_leak(content):
                    continue
                image_ids = row.get("image_ids")
                if image_ids:
                    ids_str = ", ".join(str(i) for i in image_ids if i is not None)
                    user_agent_lines.append(
                        f"[USER → AGENT]: {content} [image_ids={ids_str}]"
                    )
                else:
                    user_agent_lines.append(f"[USER → AGENT]: {content}")
            elif sender == RoleType.AGENT and recipient == RoleType.USER:
                user_agent_lines.append(f"[AGENT → USER]: {content}")
        conversation = (
            "\n".join(user_agent_lines) if user_agent_lines else "No conversation."
        )

        # --- 3. User tool calls (USER→EXEC_ENV) ---
        # Exclude end_conversation: it's a framework mechanism (not a user
        # quality signal) and may appear as a text leak rather than a real
        # tool call.  We also skip the corresponding ENV→USER response.
        skip_next_env_response = False
        user_tool_lines: list[str] = []
        for row in sandbox_rows:
            sender = row.get("sender")
            recipient = row.get("recipient")
            func_name = row.get("openai_function_name")
            content = row.get("content", "")
            if sender == RoleType.USER and recipient == RoleType.EXECUTION_ENVIRONMENT:
                if func_name == "end_conversation":
                    skip_next_env_response = True
                    continue
                user_tool_lines.append(f"[USER tool call]: {func_name or content}")
            elif (
                sender == RoleType.EXECUTION_ENVIRONMENT and recipient == RoleType.USER
            ):
                if skip_next_env_response:
                    skip_next_env_response = False
                    continue
                user_tool_lines.append(f"[ENV → USER]: {content}")
        user_tool_calls = (
            "\n".join(user_tool_lines) if user_tool_lines else "No user tool calls."
        )

        # --- 4. Available tools (excluding end_conversation) ---
        user_tools = [
            name
            for name in execution_context.get_available_tools_for_role(
                RoleType.USER
            ).keys()
            if name != "end_conversation"
        ]
        user_tool_list = (
            ", ".join(sorted(user_tools)) if user_tools else "No tools available."
        )

        # --- 5. Assemble core evidence text ---
        text = f"User Instruction (SYSTEM→USER, ground truth):\n{user_instruction}\n\n"
        text += f"User Available Tools: {user_tool_list}\n\n"
        text += f"Conversation:\n{conversation}\n\n"
        text += f"User Tool Calls:\n{user_tool_calls}\n"

        # --- 6. Scenario context ---
        text += (
            f"\nScenario limits: max_messages={max_messages},"
            f" actual_turn_count={turn_count}\n"
        )
        if scenario_metadata:
            meta_parts = [
                f"{k}={scenario_metadata[k]}"
                for k in _USER_METADATA_KEYS
                if k in scenario_metadata
            ]
            if meta_parts:
                text += f"Scenario metadata: {', '.join(meta_parts)}\n"

        # --- 7. Computed verification summaries ---
        # These give the judge pre-checked, structured signals so it doesn't
        # have to do error-prone cross-referencing on raw text.
        image_summary = self._build_image_delivery_summary(
            user_instruction, sandbox_rows
        )
        if image_summary:
            text += "\n" + image_summary + "\n"

        query_summary = self._build_query_delivery_summary(
            user_instruction, sandbox_rows
        )
        if query_summary:
            text += "\n" + query_summary + "\n"

        # --- 8. Final assembly ---
        parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
        parts += self._format_images(sandbox_rows, execution_context)
        parts += [
            {
                "type": "text",
                "text": "\nEvaluate the user simulator's quality.",
            }
        ]
        return parts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _extra_headers(self) -> dict[str, str]:
        """Per-call headers to inject into the API request.

        Override in subclasses to add auth headers.
        """
        return {}

    @property
    def _temperature(self) -> float | None:
        """Return temperature for the judge API call.

        GPT-5 and o-series models do not support ``temperature=0``.
        Returns ``None`` for those models so the API uses its default.
        """
        model = getattr(self, "model_name", "")
        if "gpt-5" in model or ":o3" in model or ":o4" in model:
            return None
        return 0

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
    )
    def evaluate_from_evidence(
        self,
        evidence_parts: list[dict[str, Any]],
        system_prompt: str,
    ) -> dict[str, Any]:
        """Call the judge LLM with pre-built evidence parts.

        Used by the backfill script to replay saved evidence without
        reconstructing the execution context.

        Args:
            evidence_parts: OpenAI-format content parts (text + image_url).
            system_prompt: The system prompt to use (main, UI, or user).

        Returns:
            Parsed judge result dict.
        """
        with all_logging_disabled(highest_level=logging.WARNING):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=cast(
                    list[ChatCompletionMessageParam],
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": evidence_parts},
                    ],
                ),
                response_format={"type": "json_object"},
                temperature=self._temperature,
                extra_headers=self._extra_headers(),
            )

        content = response.choices[0].message.content or ""
        try:
            raw: dict[str, Any] = json.loads(content)
            return JudgeResult.from_dict(raw).to_dict()
        except (json.JSONDecodeError, TypeError, KeyError):
            return {
                "result": False,
                "reasoning": f"Failed to parse judge response: {content}",
            }

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
    )
    def evaluate(
        self,
        execution_context: ExecutionContext,
        criteria: str,
        appworld_initial: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate the execution context against the criteria.

        Args:
            execution_context: The execution context containing the trajectory.
            criteria: The task completion criteria.
            appworld_initial: Optional dict mapping AppWorld table keys to
                initial-state DataFrames for computing DB diffs.

        Returns:
            A dictionary containing the evaluation result with per-criterion rubric scores.
        """
        evidence_parts = self._format_evidence(
            execution_context,
            criteria,
            appworld_tables=self._appworld_table_diffs(appworld_initial),
        )
        self.last_evidence["main"] = evidence_parts

        with all_logging_disabled(highest_level=logging.WARNING):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=cast(
                    list[ChatCompletionMessageParam],
                    [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": evidence_parts},
                    ],
                ),
                response_format={"type": "json_object"},
                temperature=self._temperature,
                extra_headers=self._extra_headers(),
            )

        content = response.choices[0].message.content
        if content is None:
            return {"result": False, "reasoning": "No response from judge model."}

        try:
            raw: dict[str, Any] = json.loads(content)
            # Parse through dataclass for validation, then serialize back to dict
            return JudgeResult.from_dict(raw).to_dict()
        except (json.JSONDecodeError, TypeError, KeyError):
            return {
                "result": False,
                "reasoning": f"Failed to parse judge response: {content}",
            }

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
    )
    def evaluate_ui(
        self,
        execution_context: ExecutionContext,
        criteria: str,
    ) -> dict[str, Any]:
        """Evaluate UI quality using 4 UI-specific rubric criteria.

        This is a separate evaluation from ``evaluate()`` — it uses a dedicated
        UI quality prompt with 4 criteria (task focus, structure, action clarity,
        feedback) and returns its own independent pass/fail result.

        Args:
            execution_context: The execution context containing the trajectory.
            criteria: The task completion criteria (provides context for the UI
                quality assessment).

        Returns:
            A dictionary with ``result``, ``reasoning``, and 4-entry
            ``criteria_evaluation`` list.
        """
        evidence_parts = self._format_ui_evidence(execution_context, criteria)
        self.last_evidence["ui"] = evidence_parts

        with all_logging_disabled(highest_level=logging.WARNING):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=cast(
                    list[ChatCompletionMessageParam],
                    [
                        {"role": "system", "content": _UI_SYSTEM_PROMPT},
                        {"role": "user", "content": evidence_parts},
                    ],
                ),
                response_format={"type": "json_object"},
                temperature=self._temperature,
                extra_headers=self._extra_headers(),
            )

        content = response.choices[0].message.content
        if content is None:
            return {"result": False, "reasoning": "No response from UI judge model."}

        try:
            raw: dict[str, Any] = json.loads(content)
            return JudgeResult.from_dict(raw).to_dict()
        except (json.JSONDecodeError, TypeError, KeyError):
            return {
                "result": False,
                "reasoning": f"Failed to parse UI judge response: {content}",
            }

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
    )
    def evaluate_user(
        self,
        execution_context: ExecutionContext,
        max_messages: int = 40,
        turn_count: int | None = None,
        scenario_metadata: dict[str, Any] | None = None,
        enable_ui: bool = False,
    ) -> dict[str, Any]:
        """Evaluate user simulator quality using the 4-criterion user rubric.

        Args:
            execution_context: The execution context containing the trajectory.
            max_messages: Scenario message budget.
            turn_count: Actual number of turns in the conversation.
            scenario_metadata: Optional scenario metadata dict.
            enable_ui: When True, append UI interaction rules to the rubric.

        Returns:
            A dictionary containing the evaluation result with per-criterion
            rubric scores.
        """
        evidence_parts = self._format_user_evidence(
            execution_context,
            max_messages=max_messages,
            turn_count=turn_count,
            scenario_metadata=scenario_metadata,
        )
        self.last_evidence["user"] = evidence_parts

        system_prompt = get_user_system_prompt(enable_ui=enable_ui)
        with all_logging_disabled(highest_level=logging.WARNING):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=cast(
                    list[ChatCompletionMessageParam],
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": evidence_parts},
                    ],
                ),
                temperature=self._temperature,
                response_format={"type": "json_object"},
                extra_headers=self._extra_headers(),
            )

        content = response.choices[0].message.content or ""
        try:
            raw: dict[str, Any] = json.loads(content)
            return JudgeResult.from_dict(raw).to_dict()
        except (json.JSONDecodeError, TypeError, KeyError):
            return {
                "result": False,
                "reasoning": f"Failed to parse user judge response: {content}",
            }


class GPT_5_4_2026_03_05_Judge(OpenAIAPIJudge):
    def __init__(self) -> None:
        super().__init__("gpt-5.4-2026-03-05")
