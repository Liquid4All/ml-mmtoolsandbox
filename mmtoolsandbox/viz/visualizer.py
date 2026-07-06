# Copyright © 2026 Apple Inc.

"""Trajectory visualization utility for mmtoolsandbox."""

from __future__ import annotations

import copy
import http.server
import json
import re
import socketserver
import threading
import urllib.parse
import webbrowser
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from mmtoolsandbox.common.message_conversion import (
    EXECUTION_RESULTS_CLOSE_TAG,
    EXECUTION_RESULTS_OPEN_TAG,
    extract_reasoning,
)

# Pattern to match ImageResult(image_id=N) in tool results
_IMAGE_RESULT_PATTERN = re.compile(r"ImageResult\(image_id=(\d+)\)")


class _FileCache:
    """Mtime-based cache for JSON files and directory listings."""

    def __init__(self) -> None:
        self._json_cache: dict[str, tuple[float, Any]] = {}
        self._dir_cache: dict[str, tuple[float, list[str]]] = {}
        self._lock = threading.Lock()

    def load_json(self, path: Path) -> Any:
        key = str(path)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        with self._lock:
            if key in self._json_cache:
                cached_mtime, cached_data = self._json_cache[key]
                if cached_mtime == mtime:
                    return cached_data
        with open(path, "r") as f:
            data = json.load(f)
        with self._lock:
            self._json_cache[key] = (mtime, data)
        return data

    def list_subdirs(self, directory: Path) -> list[str]:
        key = str(directory)
        try:
            mtime = directory.stat().st_mtime
        except OSError:
            return []
        with self._lock:
            if key in self._dir_cache:
                cached_mtime, cached_data = self._dir_cache[key]
                if cached_mtime == mtime:
                    return cached_data
        result = sorted(d.name for d in directory.iterdir() if d.is_dir())
        with self._lock:
            self._dir_cache[key] = (mtime, result)
        return result

    def invalidate(self) -> None:
        with self._lock:
            self._json_cache.clear()
            self._dir_cache.clear()


_file_cache = _FileCache()


@lru_cache(maxsize=1)
def _get_jinja_env() -> Environment:
    template_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
    )
    env.filters["urlquote"] = lambda s: urllib.parse.quote(str(s), safe="")
    return env


def _load_user_profiles(
    user_exports_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Load user export files and build an email → profile lookup.

    Extracts basic info and social circle (spouse, children, friends,
    coworkers, manager) from the admin section of each user export.

    Args:
        user_exports_dir: Directory containing user_NNN.json files.

    Returns:
        Dict mapping email (lowercase) to profile dict.
    """
    profiles: dict[str, dict[str, Any]] = {}
    for user_file in sorted(user_exports_dir.glob("user_*.json")):
        try:
            with open(user_file) as f:
                data = json.load(f)
        except Exception:
            continue

        email = data.get("email", "")
        if not email:
            continue

        admin = data.get("admin", {})
        main_users = admin.get("main_users", [])
        if not main_users:
            continue
        main_user = main_users[0]

        profile: dict[str, Any] = {
            "user_id": data.get("user_id"),
            "email": email,
            "first_name": main_user.get("first_name", ""),
            "last_name": main_user.get("last_name", ""),
            "phone_number": main_user.get("phone_number", ""),
            "birthday": main_user.get("birthday", ""),
            "sex": main_user.get("sex", ""),
        }

        # Addresses
        for addr in admin.get("user_addresses", []):
            details = addr.get("address_details", {})
            if addr.get("name") == "Home":
                profile["home_address"] = details
            elif addr.get("name") == "Work":
                profile["work_address"] = details

        # Social circle
        social: list[dict[str, str]] = []

        # Family
        families = admin.get("families", [])
        if families:
            family = families[0]
            if family.get("husband", {}).get("email") == email:
                wife = family.get("wife", {})
                if wife:
                    social.append(
                        {
                            "name": wife.get("full_name", ""),
                            "email": wife.get("email", ""),
                            "id": str(family.get("wife_id", "")),
                            "relation": "spouse",
                        }
                    )
            else:
                husband = family.get("husband", {})
                if husband:
                    social.append(
                        {
                            "name": husband.get("full_name", ""),
                            "email": husband.get("email", ""),
                            "id": str(family.get("husband_id", "")),
                            "relation": "spouse",
                        }
                    )
            for child in family.get("children", []):
                social.append(
                    {
                        "name": child.get("full_name", ""),
                        "email": child.get("email", ""),
                        "id": str(child.get("id", "")),
                        "relation": "child",
                    }
                )

        # Friends
        for f in admin.get("friendships", []):
            friend = f.get("friend", {})
            social.append(
                {
                    "name": friend.get("full_name", ""),
                    "email": friend.get("email", ""),
                    "id": str(f.get("friend_id", "")),
                    "relation": "friend",
                }
            )

        # Company & colleagues
        employees = admin.get("company_employees", [])
        if employees:
            emp = employees[0]
            company = emp.get("company", {})
            profile["company_name"] = company.get("name", "")
            manager_info = company.get("manager", {})
            for cw in emp.get("coworkers", []):
                if cw.get("is_manager") and manager_info:
                    social.append(
                        {
                            "name": manager_info.get("full_name", ""),
                            "email": manager_info.get("email", ""),
                            "id": str(cw.get("id", "")),
                            "relation": "manager",
                        }
                    )
                else:
                    social.append(
                        {
                            "name": cw.get("full_name", ""),
                            "email": cw.get("email", ""),
                            "id": str(cw.get("id", "")),
                            "relation": "coworker",
                        }
                    )

        profile["social_circle"] = social
        profiles[email.lower()] = profile

    return profiles


def _load_image_db(trajectory_path: Path) -> dict[int, str]:
    """Load the IMAGE database from execution_context.json.

    Args:
        trajectory_path: Path to conversation.json file.

    Returns:
        Dict mapping image_id → base64-encoded image content.
    """
    ec_path = trajectory_path.parent / "execution_context.json"
    if not ec_path.exists():
        return {}
    try:
        ec = _file_cache.load_json(ec_path)
        if ec is None:
            return {}
        image_rows = ec.get("_dbs", {}).get("IMAGE", [])
        image_db: dict[int, str] = {}
        for row in image_rows:
            img_id = row.get("image_id")
            content = row.get("image_content")
            if img_id is not None and content and content != "null":
                image_db[img_id] = content
        return image_db
    except Exception:
        return {}


def _resolve_image_results(
    text: str, image_db: dict[int, str]
) -> tuple[str, list[dict[str, str]]]:
    """Find ImageResult(image_id=N) references in text and resolve to inline images.

    Args:
        text: The text content (e.g., tool result) that may contain ImageResult references.
        image_db: Dict mapping image_id → base64 content.

    Returns:
        Tuple of (text, list_of_image_dicts).  The text is returned unchanged
        (ImageResult references are NOT stripped); the caller decides how to
        render them.  Each image dict has 'image_id' and 'data_url' keys.
    """
    images: list[dict[str, str]] = []
    for match in _IMAGE_RESULT_PATTERN.finditer(text):
        img_id = int(match.group(1))
        if img_id in image_db:
            images.append(
                {
                    "image_id": str(img_id),
                    "data_url": f"data:image/jpeg;base64,{image_db[img_id]}",
                }
            )
    return text, images


def _inject_judge_aggregation(result_summary: dict[str, Any]) -> None:
    """Calculate judge success aggregation on the fly to support existing results.

    Aggregates overall judge pass rate and per-criterion pass rates from the
    rubric schema (criteria_evaluation array).

    Args:
        result_summary: The result summary dictionary to update in-place.
    """
    judge_aggregation: dict[str, list[float]] = defaultdict(list)
    criterion_aggregation: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for result in result_summary.get("per_scenario_results", []):
        judge_result = result.get("judge_result")
        if not judge_result:
            continue

        success = 1.0 if judge_result.get("result") else 0.0
        criteria_evals = judge_result.get("criteria_evaluation", [])

        def _aggregate_to(category: str) -> None:
            judge_aggregation[category].append(success)
            for criterion_eval in criteria_evals:
                key = f"judge_{criterion_eval.get('criterion', 'unknown')}"
                criterion_aggregation[category][key].append(
                    1.0 if criterion_eval.get("pass") else 0.0
                )

        # Add to ALL_CATEGORIES
        _aggregate_to("ALL_CATEGORIES")

        # Add to specific categories
        for category in result.get("categories", []):
            _aggregate_to(category)

    # Update result_summary with calculated judge stats
    if "category_aggregated_results" in result_summary:
        for category, successes in judge_aggregation.items():
            if category in result_summary["category_aggregated_results"]:
                cat = result_summary["category_aggregated_results"][category]
                cat["judge_success"] = (
                    sum(successes) / len(successes) if successes else 0.0
                )
                # Per-criterion pass rates
                for key, values in criterion_aggregation.get(category, {}).items():
                    cat[key] = sum(values) / len(values) if values else 0.0


def _inject_ui_judge_aggregation(result_summary: dict[str, Any]) -> None:
    """Calculate UI judge success aggregation on the fly.

    Parallel to ``_inject_judge_aggregation`` but reads from ``ui_judge_result``
    and writes keys prefixed with ``ui_judge_``.
    """
    ui_judge_aggregation: dict[str, list[float]] = defaultdict(list)
    ui_criterion_aggregation: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for result in result_summary.get("per_scenario_results", []):
        ui_judge_result = result.get("ui_judge_result")
        if not ui_judge_result:
            continue

        success = 1.0 if ui_judge_result.get("result") else 0.0
        criteria_evals = ui_judge_result.get("criteria_evaluation", [])

        def _aggregate_to(category: str) -> None:
            ui_judge_aggregation[category].append(success)
            for criterion_eval in criteria_evals:
                key = f"ui_judge_{criterion_eval.get('criterion', 'unknown')}"
                ui_criterion_aggregation[category][key].append(
                    1.0 if criterion_eval.get("pass") else 0.0
                )

        _aggregate_to("ALL_CATEGORIES")
        for category in result.get("categories", []):
            _aggregate_to(category)

    if "category_aggregated_results" in result_summary:
        for category, successes in ui_judge_aggregation.items():
            if category in result_summary["category_aggregated_results"]:
                cat = result_summary["category_aggregated_results"][category]
                cat["ui_judge_success"] = (
                    sum(successes) / len(successes) if successes else 0.0
                )
                for key, values in ui_criterion_aggregation.get(category, {}).items():
                    cat[key] = sum(values) / len(values) if values else 0.0


def _inject_user_judge_aggregation(result_summary: dict[str, Any]) -> None:
    """Calculate user simulator judge success aggregation on the fly.

    Parallel to ``_inject_judge_aggregation`` but reads from ``user_judge_result``
    and writes keys prefixed with ``user_judge_``.
    """
    user_judge_aggregation: dict[str, list[float]] = defaultdict(list)
    user_criterion_aggregation: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for result in result_summary.get("per_scenario_results", []):
        user_judge_result = result.get("user_judge_result")
        if not user_judge_result:
            continue

        success = 1.0 if user_judge_result.get("result") else 0.0
        criteria_evals = user_judge_result.get("criteria_evaluation", [])

        def _aggregate_to(category: str) -> None:
            user_judge_aggregation[category].append(success)
            for criterion_eval in criteria_evals:
                key = f"user_judge_{criterion_eval.get('criterion', 'unknown')}"
                user_criterion_aggregation[category][key].append(
                    1.0 if criterion_eval.get("pass") else 0.0
                )

        _aggregate_to("ALL_CATEGORIES")
        for category in result.get("categories", []):
            _aggregate_to(category)

    if "category_aggregated_results" in result_summary:
        for category, successes in user_judge_aggregation.items():
            if category in result_summary["category_aggregated_results"]:
                cat = result_summary["category_aggregated_results"][category]
                cat["user_judge_success"] = (
                    sum(successes) / len(successes) if successes else 0.0
                )
                for key, values in user_criterion_aggregation.get(category, {}).items():
                    cat[key] = sum(values) / len(values) if values else 0.0


def _inject_entity_diff_aggregation(result_summary: dict[str, Any]) -> None:
    """Calculate entity diff F1 aggregation per category."""
    diff_aggregation: dict[str, list[float]] = defaultdict(list)

    for result in result_summary.get("per_scenario_results", []):
        entity_diff_result = result.get("entity_diff_result")
        if not entity_diff_result:
            continue

        p = entity_diff_result.get("overall_precision", 0.0)
        r = entity_diff_result.get("overall_recall", 0.0)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        def _aggregate_to(category: str) -> None:
            diff_aggregation[category].append(f1)

        _aggregate_to("ALL_CATEGORIES")
        for category in result.get("categories", []):
            _aggregate_to(category)

    if "category_aggregated_results" in result_summary:
        for category, scores in diff_aggregation.items():
            if category in result_summary["category_aggregated_results"]:
                result_summary["category_aggregated_results"][category][
                    "entity_diff_f1"
                ] = sum(scores) / len(scores) if scores else 0.0


def _detect_code_exec_mode(messages: list[dict[str, Any]]) -> bool:
    """Detect whether a conversation uses code-execution mode.

    In code-execution mode, execution results appear as user messages
    wrapped in ``<execution_results>`` XML tags instead of ``role: "tool"``
    messages.

    Args:
        messages: List of message dicts from conversation JSON.

    Returns:
        True if the conversation uses code-execution mode.
    """
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str) and EXECUTION_RESULTS_OPEN_TAG in content:
            return True
    return False


def _parse_assistant_content_segments(
    content: str,
) -> list[dict[str, str]]:
    """Split assistant message content into text and code segments.

    Handles both markdown-fenced code blocks (`` ```python ... ``` ``) and
    plain Python code (detected by context).

    Args:
        content: The assistant message content string.

    Returns:
        List of ``{"type": "text"|"code", "content": str}`` dicts.
    """
    pattern = r"(```python\s*\n.*?```)"
    parts = re.split(pattern, content, flags=re.DOTALL)
    segments: list[dict[str, str]] = []
    for part in parts:
        if part.startswith("```python"):
            # Strip the fences to get raw code
            code = re.sub(r"^```python\s*\n", "", part)
            code = re.sub(r"\n?```$", "", code)
            segments.append({"type": "code", "content": code})
        elif part.strip():
            segments.append({"type": "text", "content": part.strip()})
    return segments


_EXEC_RESULTS_PATTERN = re.compile(
    rf"{re.escape(EXECUTION_RESULTS_OPEN_TAG)}\n?(.*?)\n?{re.escape(EXECUTION_RESULTS_CLOSE_TAG)}",
    re.DOTALL,
)


def _extract_execution_results(content: str) -> list[str]:
    """Extract content from ``<execution_results>`` XML tags.

    Args:
        content: The user message content string.

    Returns:
        List of extracted result strings.
    """
    return _EXEC_RESULTS_PATTERN.findall(content)


def prepare_message_data(
    message_data: dict[str, Any],
    image_db: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Prepare message data for template rendering.

    Args:
        message_data: Raw message data from conversation JSON
        image_db: Optional dict mapping image_id → base64 content for inline rendering.

    Returns:
        Processed data ready for Jinja2 template
    """
    messages = message_data.get("messages", [])
    tools = message_data.get("tools", [])
    user_tools = message_data.get("user_tools", [])
    is_code_exec_mode = _detect_code_exec_mode(messages)
    has_perspectives = any("perspective" in m for m in messages)
    resolved_images = image_db or {}

    # Process messages to add formatted tool call arguments
    processed_messages = []
    for message in messages:
        processed_msg = message.copy()

        # --- Code-execution mode processing ---
        if is_code_exec_mode:
            content = message.get("content", "")

            # User messages with <execution_results> tags
            if (
                message.get("role") == "user"
                and isinstance(content, str)
                and EXECUTION_RESULTS_OPEN_TAG in content
            ):
                processed_msg["execution_results"] = _extract_execution_results(content)

            # Assistant messages with markdown-fenced code blocks
            elif message.get("role") == "assistant" and isinstance(content, str):
                # Extract reasoning BEFORE parsing code segments so <think>
                # tags don't end up inside text segments.
                extras = message.get("extras_assistant", {})
                thinking = extras.get("claude_extended_thinking") if extras else None
                reasoning = extras.get("reasoning_trace") if extras else None
                if thinking:
                    processed_msg["reasoning_trace"] = thinking
                    _, content = extract_reasoning(content)
                elif reasoning:
                    processed_msg["reasoning_trace"] = reasoning
                    _, content = extract_reasoning(content)
                else:
                    extracted, cleaned = extract_reasoning(content)
                    if extracted:
                        processed_msg["reasoning_trace"] = extracted
                        content = cleaned

                if "```python" in content:
                    processed_msg["content_segments"] = (
                        _parse_assistant_content_segments(content)
                    )

        # --- Standard tool call processing (works for all modes) ---
        if "tool_calls" in message and message["tool_calls"]:
            processed_tool_calls = []
            for tool_call in message["tool_calls"]:
                processed_tc = tool_call.copy()
                if "function" in tool_call:
                    function = tool_call["function"].copy()
                    args_str = function.get("arguments", "")

                    # Pretty-print JSON arguments
                    try:
                        args_dict = json.loads(args_str) if args_str else {}
                        function["formatted_args"] = json.dumps(args_dict, indent=2)

                        # Special handling for execute_code to extract raw code for better visualization
                        if (
                            function.get("name") == "execute_code"
                            and "code" in args_dict
                        ):
                            function["formatted_code"] = args_dict["code"]
                            function["is_code_execution"] = True

                    except (json.JSONDecodeError, TypeError):
                        function["formatted_args"] = args_str

                    processed_tc["function"] = function
                processed_tool_calls.append(processed_tc)

            processed_msg["tool_calls"] = processed_tool_calls

        # --- Reasoning trace extraction (ReACT-style <think> tags or Claude extended thinking) ---
        if message.get("role") == "assistant":
            # First check extras for Claude extended thinking, then reasoning_trace
            extras = message.get("extras_assistant", {})
            thinking = extras.get("claude_extended_thinking") if extras else None
            reasoning = extras.get("reasoning_trace") if extras else None
            if thinking:
                processed_msg["reasoning_trace"] = thinking
                # Also strip any <think> tags from content to avoid duplication
                if isinstance(processed_msg.get("content", ""), str):
                    _, cleaned = extract_reasoning(processed_msg["content"])
                    processed_msg["content"] = cleaned
            elif reasoning:
                processed_msg["reasoning_trace"] = reasoning
                # Also strip any <think> tags from content to avoid duplication
                if isinstance(processed_msg.get("content", ""), str):
                    _, cleaned = extract_reasoning(processed_msg["content"])
                    processed_msg["content"] = cleaned
            # Fall back to extracting inline <think> tags from content
            elif isinstance(message.get("content", ""), str):
                content = message["content"]
                reasoning, cleaned = extract_reasoning(content)
                if reasoning is not None:
                    processed_msg["reasoning_trace"] = reasoning
                    processed_msg["content"] = cleaned

        # --- Perspective metadata ---
        if "perspective" in message:
            processed_msg["perspective"] = message["perspective"]
            views = message["perspective"].get("views", [])
            if "agent" in views and "user" in views:
                processed_msg["perspective_class"] = "view-both"
            elif "agent" in views:
                processed_msg["perspective_class"] = "view-agent-only"
            else:
                processed_msg["perspective_class"] = "view-user-only"

        # --- Resolve ImageResult references to inline images ---
        if resolved_images:
            content = processed_msg.get("content", "")
            # Check string content (tool results, text messages)
            if isinstance(content, str) and "ImageResult" in content:
                _, inline_images = _resolve_image_results(content, resolved_images)
                if inline_images:
                    processed_msg["inline_images"] = inline_images
            # Check list content — but SKIP if list already contains image_url parts
            # (the image is already rendered inline by the template)
            elif isinstance(content, list):
                has_existing_images = any(
                    isinstance(p, dict) and p.get("type") == "image_url"
                    for p in content
                )
                if not has_existing_images:
                    all_images: list[dict[str, str]] = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text", "")
                            if "ImageResult" in text:
                                _, imgs = _resolve_image_results(text, resolved_images)
                                all_images.extend(imgs)
                    if all_images:
                        processed_msg["inline_images"] = all_images

        processed_messages.append(processed_msg)

    # Add the pretty-printed full JSON schema to each tool definition so the
    # template can render a collapsible "full schema" block.
    def _process_tools(raw_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for tool in raw_tools:
            processed = tool.copy()
            processed["full_schema"] = json.dumps(tool, indent=2)
            result.append(processed)
        return result

    return {
        "messages": processed_messages,
        "tools": _process_tools(tools),
        "user_tools": _process_tools(user_tools),
        "has_perspectives": has_perspectives,
    }


def _build_entity_diff_matches(
    specs: list[dict[str, Any]],
    entity_diff_result: dict[str, Any] | None,
) -> tuple[list[dict[str, Any] | None] | None, list[dict[str, Any]] | None]:
    """Match entity_diff_specs to their evaluation scores.

    Returns:
        (matches, extras) where:
        - matches: list parallel to specs, each entry is
          {"score": float, "column_scores": {col: float}} or None if no match.
        - extras: list of {"table_key", "operation", "num_extra"} for groups
          where num_actual > num_expected (agent created extra entities).
    """
    if not entity_diff_result or not specs:
        return None, None

    group_results = entity_diff_result.get("group_results", [])
    if not group_results:
        return None, None

    # Build lookup: (table_key, operation) → group_result
    group_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for group in group_results:
        key = (group["table_key"], group["operation"])
        group_lookup[key] = group

    # Track how many specs we've seen per (table_key, operation)
    # to index into per_entity_scores
    spec_counters: dict[tuple[str, str], int] = {}
    matches: list[dict[str, Any] | None] = []

    for spec in specs:
        table_key = spec.get("table") or spec.get("namespace", "")
        operation = spec.get("operation", "")
        key = (table_key, operation)

        group = group_lookup.get(key)
        if group is None:
            matches.append(None)
            continue

        idx = spec_counters.get(key, 0)
        spec_counters[key] = idx + 1

        per_entity = group.get("per_entity_scores", [])
        per_col = group.get("per_entity_column_scores", [])
        per_actual = group.get("per_entity_actual_values", [])

        if idx < len(per_entity):
            match: dict[str, Any] = {"score": per_entity[idx]}
            if idx < len(per_col):
                match["column_scores"] = per_col[idx]
            if idx < len(per_actual):
                match["actual_values"] = per_actual[idx]
            matches.append(match)
        else:
            matches.append(None)

    # Find extras: groups where num_actual > num_expected
    extras: list[dict[str, Any]] = []
    for group in group_results:
        num_extra = group.get("num_actual", 0) - group.get("num_expected", 0)
        if num_extra > 0:
            extras.append(
                {
                    "table_key": group["table_key"],
                    "operation": group["operation"],
                    "num_extra": num_extra,
                    "num_actual": group["num_actual"],
                    "num_expected": group["num_expected"],
                    "precision": group["precision"],
                }
            )

    return matches, extras if extras else None


def render_error_page(
    output_directory: Path,
    scenario: str,
    run: str,
    result_summary_name: str = "result_summary.json",
    experiments: list[str] | None = None,
    current_experiment: str = "",
) -> str:
    """Render error page for a failed run without conversation file.

    Args:
        output_directory: Base output directory
        scenario: Scenario name
        run: Run name
        result_summary_name: Filename of the result summary JSON.

    Returns:
        Rendered HTML string
    """
    # Load result_summary.json
    result_summary_path = output_directory / result_summary_name
    scenario_result = None
    nav_info: dict[str, str | None] | None = None
    full_scenario_name = f"{scenario}/{run}"

    if result_summary_path.exists():
        result_summary = _file_cache.load_json(result_summary_path)

        # Find the matching scenario result
        for result in result_summary.get("per_scenario_results", []):
            if result.get("name") == full_scenario_name:
                scenario_result = result
                break

        # Calculate navigation info from result_summary
        # Group results by scenario
        scenarios_dict: dict[str, list[str]] = {}
        for result in result_summary.get("per_scenario_results", []):
            name = result.get("name", "")
            if "/" in name:
                base_scenario, run_name = name.rsplit("/", 1)
                if base_scenario not in scenarios_dict:
                    scenarios_dict[base_scenario] = []
                scenarios_dict[base_scenario].append(run_name)

        # Sort everything
        all_scenarios = sorted(scenarios_dict.keys())
        for s in scenarios_dict:
            scenarios_dict[s] = sorted(scenarios_dict[s])

        # Find current positions
        try:
            scenario_idx = all_scenarios.index(scenario)
            run_idx = scenarios_dict[scenario].index(run)
        except (ValueError, KeyError):
            scenario_idx = -1
            run_idx = -1

        # Calculate prev/next
        nav_info = {
            "prev_scenario": None,
            "next_scenario": None,
            "prev_run": None,
            "next_run": None,
        }

        if scenario_idx > 0:
            prev_scenario = all_scenarios[scenario_idx - 1]
            prev_runs = scenarios_dict.get(prev_scenario, [])
            if prev_runs:
                # Check if conversation file exists
                conv_path = (
                    output_directory
                    / "trajectories"
                    / prev_scenario
                    / prev_runs[0]
                    / "conversation.json"
                )
                if conv_path.exists():
                    nav_info["prev_scenario"] = (
                        f"trajectories/{prev_scenario}/{prev_runs[0]}/conversation.json"
                    )
                else:
                    nav_info["prev_scenario"] = (
                        f"?scenario={urllib.parse.quote(prev_scenario, safe='')}"
                        f"&run={urllib.parse.quote(prev_runs[0], safe='')}"
                    )

        if scenario_idx < len(all_scenarios) - 1:
            next_scenario = all_scenarios[scenario_idx + 1]
            next_runs = scenarios_dict.get(next_scenario, [])
            if next_runs:
                conv_path = (
                    output_directory
                    / "trajectories"
                    / next_scenario
                    / next_runs[0]
                    / "conversation.json"
                )
                if conv_path.exists():
                    nav_info["next_scenario"] = (
                        f"trajectories/{next_scenario}/{next_runs[0]}/conversation.json"
                    )
                else:
                    nav_info["next_scenario"] = (
                        f"?scenario={urllib.parse.quote(next_scenario, safe='')}"
                        f"&run={urllib.parse.quote(next_runs[0], safe='')}"
                    )

        if scenario_idx >= 0 and run_idx > 0:
            all_runs = scenarios_dict[scenario]
            prev_run_name = all_runs[run_idx - 1]
            conv_path = (
                output_directory
                / "trajectories"
                / scenario
                / prev_run_name
                / "conversation.json"
            )
            if conv_path.exists():
                nav_info["prev_run"] = (
                    f"trajectories/{scenario}/{prev_run_name}/conversation.json"
                )
            else:
                nav_info["prev_run"] = (
                    f"?scenario={urllib.parse.quote(scenario, safe='')}"
                    f"&run={urllib.parse.quote(prev_run_name, safe='')}"
                )

        if scenario_idx >= 0 and run_idx < len(scenarios_dict[scenario]) - 1:
            all_runs = scenarios_dict[scenario]
            next_run_name = all_runs[run_idx + 1]
            conv_path = (
                output_directory
                / "trajectories"
                / scenario
                / next_run_name
                / "conversation.json"
            )
            if conv_path.exists():
                nav_info["next_run"] = (
                    f"trajectories/{scenario}/{next_run_name}/conversation.json"
                )
            else:
                nav_info["next_run"] = (
                    f"?scenario={urllib.parse.quote(scenario, safe='')}"
                    f"&run={urllib.parse.quote(next_run_name, safe='')}"
                )

    # Render template
    env = _get_jinja_env()
    template = env.get_template("trajectory.html")
    return template.render(
        messages=[],
        tools=[],
        scenario_name=full_scenario_name,
        scenario_result=scenario_result,
        nav_info=nav_info,
        experiments=experiments or [],
        current_experiment=current_experiment,
    )


def render_trajectory_html(
    trajectory_path: Path,
    scenario_dir: Path | None = None,
    user_profiles: dict[str, dict[str, Any]] | None = None,
    result_summary_name: str = "result_summary.json",
    experiments: list[str] | None = None,
    current_experiment: str = "",
) -> str:
    """Render trajectory HTML from a conversation.json file.

    Args:
        trajectory_path: Path to conversation.json file
        scenario_dir: Optional path to directory containing scenario JSON files
        user_profiles: Optional email → profile lookup from user exports
        result_summary_name: Filename of the result summary JSON.

    Returns:
        Rendered HTML string
    """
    # Load message data from conversation file
    message_data = _file_cache.load_json(trajectory_path)

    # Load image database for inline image rendering
    image_db = _load_image_db(trajectory_path)

    # Prepare data for template
    template_data = prepare_message_data(message_data, image_db=image_db)

    # Extract scenario name from path
    # Path structure: .../trajectories/<scenario_name>/<run_name>/conversation.json
    parts = trajectory_path.parts
    scenario_name = None
    scenario_result = None
    nav_info: dict[str, str | None] | None = None

    if "trajectories" in parts:
        traj_idx = parts.index("trajectories")
        if traj_idx + 2 < len(parts):
            base_scenario = parts[traj_idx + 1]
            # Detect flat layout: trajectories/<scenario>/conversation.json
            # vs nested layout: trajectories/<scenario>/<run>/conversation.json
            if parts[traj_idx + 2] == "conversation.json":
                # Flat layout — no run subdirectory
                run_name = None
                is_flat_layout = True
                scenario_name = base_scenario
                output_directory = trajectory_path.parents[2]
            else:
                # Nested layout — has run subdirectory
                run_name = parts[traj_idx + 2]
                is_flat_layout = False
                scenario_name = f"{base_scenario}/{run_name}"
                output_directory = trajectory_path.parents[3]

            result_summary_path = output_directory / result_summary_name

            if result_summary_path.exists():
                result_summary = _file_cache.load_json(result_summary_path)

                # Find the matching scenario result
                for result in result_summary.get("per_scenario_results", []):
                    if result.get("name") == scenario_name:
                        scenario_result = result
                        break

            # Calculate navigation info
            trajectories_dir = output_directory / "trajectories"
            if trajectories_dir.exists():
                # Get all scenarios sorted (cached by dir mtime)
                all_scenarios = _file_cache.list_subdirs(trajectories_dir)

                # Get all runs for current scenario (cached by dir mtime)
                scenario_traj_dir = trajectories_dir / base_scenario
                all_runs = (
                    _file_cache.list_subdirs(scenario_traj_dir)
                    if scenario_traj_dir.exists()
                    else []
                )

                # Find current positions.  Compute the scenario and run indices
                # independently so a missing run never resets the scenario index.
                try:
                    scenario_idx = all_scenarios.index(base_scenario)
                except ValueError:
                    scenario_idx = -1
                if run_name is not None and run_name in all_runs:
                    run_idx = all_runs.index(run_name)
                else:
                    run_idx = -1

                # Calculate prev/next
                nav_info = {
                    "prev_scenario": None,
                    "next_scenario": None,
                    "prev_run": None,
                    "next_run": None,
                }

                _flat_layout = is_flat_layout

                if scenario_idx > 0:
                    prev_scenario = all_scenarios[scenario_idx - 1]
                    if _flat_layout:
                        nav_info["prev_scenario"] = (
                            f"trajectories/{prev_scenario}/conversation.json"
                        )
                    else:
                        prev_runs = _file_cache.list_subdirs(
                            trajectories_dir / prev_scenario
                        )
                        if prev_runs:
                            nav_info["prev_scenario"] = (
                                f"trajectories/{prev_scenario}/{prev_runs[0]}/conversation.json"
                            )

                if scenario_idx < len(all_scenarios) - 1:
                    next_scenario = all_scenarios[scenario_idx + 1]
                    if _flat_layout:
                        nav_info["next_scenario"] = (
                            f"trajectories/{next_scenario}/conversation.json"
                        )
                    else:
                        next_runs = _file_cache.list_subdirs(
                            trajectories_dir / next_scenario
                        )
                        if next_runs:
                            nav_info["next_scenario"] = (
                                f"trajectories/{next_scenario}/{next_runs[0]}/conversation.json"
                            )

                if run_idx > 0:
                    nav_info["prev_run"] = (
                        f"trajectories/{base_scenario}/{all_runs[run_idx - 1]}/conversation.json"
                    )

                if run_idx < len(all_runs) - 1:
                    nav_info["next_run"] = (
                        f"trajectories/{base_scenario}/{all_runs[run_idx + 1]}/conversation.json"
                    )

    # Load raw scenario JSON if scenario_dir is provided
    scenario_json: dict[str, Any] | None = None
    if scenario_dir and scenario_name:
        # scenario_name is like "scenario_0001/run_0" or just "scenario_0001"
        # The JSON file is named after the base scenario (before the /)
        scenario_base = (
            scenario_name.split("/")[0] if "/" in scenario_name else scenario_name
        )
        scenario_json_path = scenario_dir / f"{scenario_base}.json"
        if scenario_json_path.exists():
            try:
                scenario_json = _file_cache.load_json(scenario_json_path)
            except Exception:
                pass

    # Look up user profile from device_state_id
    user_profile: dict[str, Any] | None = None
    if user_profiles and scenario_json:
        device_state_id = scenario_json.get("device_state_id", "")
        if device_state_id:
            user_profile = user_profiles.get(device_state_id.lower())

    env = _get_jinja_env()

    # Build per-entity match scores for the Raw Scenario entity diff specs.
    # Maps (table_or_namespace, operation, entity_index) → score + column_scores
    entity_diff_matches: list[dict[str, Any] | None] | None = None
    entity_diff_extras: list[dict[str, Any]] | None = None
    if scenario_json and scenario_result:
        entity_diff_matches, entity_diff_extras = _build_entity_diff_matches(
            scenario_json.get("entity_diff_specs", []),
            scenario_result.get("entity_diff_result"),
        )

    # Render template
    template = env.get_template("trajectory.html")
    return template.render(
        **template_data,
        scenario_name=scenario_name,
        scenario_result=scenario_result,
        nav_info=nav_info,
        scenario_json=scenario_json,
        user_profile=user_profile,
        entity_diff_matches=entity_diff_matches,
        entity_diff_extras=entity_diff_extras,
        experiments=experiments or [],
        current_experiment=current_experiment,
    )


def find_trajectory_files(output_directory: Path) -> list[Path]:
    """Find all conversation.json files in the output directory.

    Args:
        output_directory: Base output directory

    Returns:
        List of paths to conversation.json files
    """
    return sorted(output_directory.glob("**/conversation.json"))


def generate_index_html(
    output_directory: Path,
    auto_refresh: bool = True,
    result_summary_name: str = "result_summary.json",
    experiments: list[str] | None = None,
    current_experiment: str = "",
) -> str:
    """Generate an index page listing all trajectory results.

    Args:
        output_directory: Base output directory
        auto_refresh: Whether to enable auto-refresh by default
        result_summary_name: Filename of the result summary JSON.

    Returns:
        HTML string for the index page
    """
    trajectory_files = find_trajectory_files(output_directory)

    # Group by scenario
    scenarios: dict[str, list[dict[str, Any]]] = {}
    for traj_file in trajectory_files:
        # Extract scenario name from path: trajectories/<scenario_name>/run_X/conversation.json
        parts = traj_file.parts
        if "trajectories" in parts:
            traj_idx = parts.index("trajectories")
            if traj_idx + 1 < len(parts):
                scenario_name = parts[traj_idx + 1]
                if scenario_name not in scenarios:
                    scenarios[scenario_name] = []

                rel_path = str(traj_file.relative_to(output_directory))
                run_name = traj_file.parent.name
                scenarios[scenario_name].append(
                    {"rel_path": rel_path, "run_name": run_name, "success": None}
                )

    # Load result_summary.json if it exists
    result_summary_path = output_directory / result_summary_name
    result_summary = None
    success_count = 0
    failure_count = 0
    judge_success_count = 0
    judge_total_count = 0
    scenario_aggregation: dict[str, dict[str, Any]] = {}

    if result_summary_path.exists():
        result_summary = copy.deepcopy(_file_cache.load_json(result_summary_path))
        _inject_judge_aggregation(result_summary)
        _inject_ui_judge_aggregation(result_summary)
        _inject_user_judge_aggregation(result_summary)
        _inject_entity_diff_aggregation(result_summary)

        # Calculate success/failure counts and per-scenario aggregation
        for result in result_summary.get("per_scenario_results", []):
            if result.get("exception_type") is None:
                success_count += 1
            else:
                failure_count += 1

            # Calculate judge success
            judge_result = result.get("judge_result")
            if judge_result:
                judge_total_count += 1
                if judge_result.get("result"):
                    judge_success_count += 1

            # Mark success/failure in scenarios dict and add missing runs
            full_scenario_name = result.get("name", "")
            if "/" in full_scenario_name:
                base_scenario, run_name = full_scenario_name.rsplit("/", 1)

                # Ensure scenario exists in dict
                if base_scenario not in scenarios:
                    scenarios[base_scenario] = []

                # Check if this run is already in the scenarios dict
                run_exists = False
                for conv in scenarios[base_scenario]:
                    if conv["run_name"] == run_name:
                        conv["success"] = result.get("exception_type") is None
                        run_exists = True
                        break

                # If run doesn't exist (failed run with no conversation.json), add it
                if not run_exists:
                    scenarios[base_scenario].append(
                        {
                            "rel_path": None,  # No conversation file exists
                            "run_name": run_name,
                            "success": result.get("exception_type") is None,
                        }
                    )

            # Aggregate by scenario (strip /run_X suffix)
            scenario_name = result.get("name", "")
            # Split on last / to get scenario name without run_X
            if "/" in scenario_name:
                base_scenario = scenario_name.rsplit("/", 1)[0]
            else:
                base_scenario = scenario_name

            if base_scenario not in scenario_aggregation:
                scenario_aggregation[base_scenario] = {
                    "turn_counts": [],
                    "successes": 0,
                    "failures": 0,
                    "judge_successes": 0,
                    "judge_total": 0,
                    "user_judge_successes": 0,
                    "user_judge_total": 0,
                    "ui_judge_successes": 0,
                    "ui_judge_total": 0,
                    "entity_diff_scores": [],
                }

            scenario_aggregation[base_scenario]["turn_counts"].append(
                result.get("turn_count", 0)
            )
            if result.get("exception_type") is None:
                scenario_aggregation[base_scenario]["successes"] += 1
            else:
                scenario_aggregation[base_scenario]["failures"] += 1

            if judge_result:
                scenario_aggregation[base_scenario]["judge_total"] += 1
                if judge_result.get("result"):
                    scenario_aggregation[base_scenario]["judge_successes"] += 1

            user_judge_result = result.get("user_judge_result")
            if user_judge_result:
                scenario_aggregation[base_scenario]["user_judge_total"] += 1
                if user_judge_result.get("result"):
                    scenario_aggregation[base_scenario]["user_judge_successes"] += 1

            ui_judge_result = result.get("ui_judge_result")
            if ui_judge_result:
                scenario_aggregation[base_scenario]["ui_judge_total"] += 1
                if ui_judge_result.get("result"):
                    scenario_aggregation[base_scenario]["ui_judge_successes"] += 1

            entity_diff_result = result.get("entity_diff_result")
            if entity_diff_result:
                p = entity_diff_result.get("overall_precision", 0.0)
                r = entity_diff_result.get("overall_recall", 0.0)
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                scenario_aggregation[base_scenario]["entity_diff_scores"].append(f1)

        # Calculate averages
        for scenario_name, data in scenario_aggregation.items():
            data["avg_turn_count"] = (
                sum(data["turn_counts"]) / len(data["turn_counts"])
                if data["turn_counts"]
                else 0
            )
            data["total_runs"] = data["successes"] + data["failures"]
            data["judge_success_rate"] = (
                data["judge_successes"] / data["judge_total"]
                if data["judge_total"] > 0
                else None
            )
            data["user_judge_success_rate"] = (
                data["user_judge_successes"] / data["user_judge_total"]
                if data["user_judge_total"] > 0
                else None
            )
            data["ui_judge_success_rate"] = (
                data["ui_judge_successes"] / data["ui_judge_total"]
                if data["ui_judge_total"] > 0
                else None
            )
            ed_scores = data["entity_diff_scores"]
            data["entity_diff_avg_f1"] = (
                sum(ed_scores) / len(ed_scores) if ed_scores else None
            )

    env = _get_jinja_env()

    # Render template
    template = env.get_template("index.html")
    return template.render(
        output_directory=str(output_directory),
        total_trajectories=len(trajectory_files),
        scenarios=dict(sorted(scenarios.items())),
        result_summary=result_summary,
        success_count=success_count,
        failure_count=failure_count,
        judge_success_count=judge_success_count,
        judge_total_count=judge_total_count,
        scenario_aggregation=dict(sorted(scenario_aggregation.items())),
        auto_refresh=auto_refresh,
        experiments=experiments or [],
        current_experiment=current_experiment,
    )


class ConversationHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler for serving trajectory visualization."""

    output_directories: dict[str, Path] = {}
    default_experiment: str = ""
    auto_refresh: bool = True
    scenario_dir: Path | None = None
    user_profiles: dict[str, dict[str, Any]] = {}
    result_summary_name: str = "result_summary.json"

    @classmethod
    def _resolve_experiment(cls, params: dict[str, list[str]]) -> tuple[str, Path]:
        name = params.get("experiment", [cls.default_experiment])[0]
        if name not in cls.output_directories:
            name = cls.default_experiment
        return name, cls.output_directories[name]

    def do_GET(self) -> None:
        """Handle GET requests."""
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        experiments = list(self.output_directories.keys())
        is_multi = len(experiments) > 1
        current_exp_name, current_dir = self._resolve_experiment(params)

        if parsed.path in ("/", "/index.html"):
            # Serve index page
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            html = generate_index_html(
                current_dir,
                auto_refresh=self.auto_refresh,
                result_summary_name=self.result_summary_name,
                experiments=experiments if is_multi else [],
                current_experiment=current_exp_name if is_multi else "",
            )
            self.wfile.write(html.encode("utf-8"))
        elif parsed.path == "/view":
            if "path" in params:
                # Serve individual conversation from file. parse_qs has already
                # decoded percent-escapes (so %2F → /). Both literal slashes
                # and %2F-encoded slashes in the path value are accepted.
                path_param = params["path"][0]
                conv_path = current_dir / path_param
                if conv_path.exists():
                    self.send_response(200)
                    self.send_header("Content-type", "text/html; charset=utf-8")
                    self.end_headers()
                    try:
                        html = render_trajectory_html(
                            conv_path,
                            scenario_dir=self.scenario_dir,
                            user_profiles=self.user_profiles,
                            result_summary_name=self.result_summary_name,
                            experiments=experiments if is_multi else [],
                            current_experiment=current_exp_name if is_multi else "",
                        )
                        self.wfile.write(html.encode("utf-8"))
                    except Exception as e:
                        error_html = f"<html><body><h1>Error rendering trajectory</h1><pre>{e}</pre></body></html>"
                        self.wfile.write(error_html.encode("utf-8"))
                elif is_multi:
                    # Trajectory doesn't exist in this experiment — redirect
                    # to the experiment's index so the dropdown switch still
                    # lands somewhere sensible.
                    self.send_response(302)
                    self.send_header(
                        "Location",
                        f"/?experiment={urllib.parse.quote(current_exp_name, safe='')}",
                    )
                    self.end_headers()
                else:
                    self.send_error(404, "Conversation file not found")
            elif "scenario" in params and "run" in params:
                # Serve error page for failed run without conversation file
                scenario = params.get("scenario", [""])[0]
                run = params.get("run", [""])[0]

                self.send_response(200)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.end_headers()
                try:
                    html = render_error_page(
                        current_dir,
                        scenario,
                        run,
                        result_summary_name=self.result_summary_name,
                        experiments=experiments if is_multi else [],
                        current_experiment=current_exp_name if is_multi else "",
                    )
                    self.wfile.write(html.encode("utf-8"))
                except Exception as e:
                    error_html = f"<html><body><h1>Error rendering error page</h1><pre>{e}</pre></body></html>"
                    self.wfile.write(error_html.encode("utf-8"))
            else:
                self.send_error(400, "Missing required parameters")
        else:
            self.send_error(404)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress log messages."""
        pass


def start_visualizer_server(
    output_directory: Path | dict[str, Path],
    port: int = 8000,
    auto_refresh: bool = True,
    scenario_dir: Path | None = None,
    user_exports_dir: Path | None = None,
    result_summary_name: str = "result_summary.json",
) -> None:
    """Start a web server to visualize trajectory results from an output directory.

    Args:
        output_directory: Either a single Path, or a dict of {experiment_name: Path}
            to expose multiple experiments behind one server (selectable via the
            ``?experiment=<name>`` query parameter).
        port: Port to run the web server on (default: 8000)
        auto_refresh: Whether to enable auto-refresh by default (default: True)
        scenario_dir: Optional path to directory containing scenario JSON files
        user_exports_dir: Optional path to directory containing user export JSON files
        result_summary_name: Filename of the result summary JSON (default: result_summary.json)
    """
    # Normalize to dict
    if isinstance(output_directory, dict):
        output_directories = output_directory
    else:
        output_directories = {output_directory.name: output_directory}

    if not output_directories:
        raise ValueError("At least one output directory is required")

    default_experiment = next(iter(output_directories))

    ConversationHTTPHandler.output_directories = output_directories
    ConversationHTTPHandler.default_experiment = default_experiment
    ConversationHTTPHandler.auto_refresh = auto_refresh
    ConversationHTTPHandler.scenario_dir = scenario_dir
    ConversationHTTPHandler.result_summary_name = result_summary_name

    # Load user profiles if directory provided
    if user_exports_dir and user_exports_dir.is_dir():
        profiles = _load_user_profiles(user_exports_dir)
        ConversationHTTPHandler.user_profiles = profiles
        print(f"Loaded {len(profiles)} user profiles from {user_exports_dir}")
    else:
        ConversationHTTPHandler.user_profiles = {}

    # Try to find an available port
    max_attempts = 10
    original_port = port
    server_started = False
    httpd = None

    for attempt in range(max_attempts):
        try:
            httpd = socketserver.ThreadingTCPServer(("", port), ConversationHTTPHandler)
            httpd.daemon_threads = True
            httpd.allow_reuse_address = True
            server_started = True
            break
        except OSError as e:
            if e.errno == 48:  # Address already in use (macOS)
                port += 1
            elif e.errno == 98:  # Address already in use (Linux)
                port += 1
            else:
                raise

    if not server_started or httpd is None:
        print(
            f"Error: Could not find an available port after trying ports {original_port}-{port}"
        )
        return

    url = f"http://localhost:{port}"
    print(f"\n{'=' * 60}")
    print("Trajectory Visualizer")
    print(f"{'=' * 60}")
    print(f"\nServer running at: {url}")
    if len(output_directories) == 1:
        print(f"Output directory: {next(iter(output_directories.values()))}")
    else:
        print(f"Experiments ({len(output_directories)}):")
        for name, path in output_directories.items():
            marker = " (default)" if name == default_experiment else ""
            print(f"  - {name}: {path}{marker}")
    print("\nPress Ctrl+C to stop the server")
    print(f"{'=' * 60}\n")

    # Try to open in browser
    try:
        webbrowser.open(url)
    except Exception:
        pass

    # Serve forever - when running in a daemon thread, this will be
    # terminated when the main thread exits
    try:
        httpd.serve_forever()
    finally:
        # Ensure cleanup happens even if forcefully terminated
        try:
            httpd.shutdown()
            httpd.server_close()
        except Exception:
            pass
