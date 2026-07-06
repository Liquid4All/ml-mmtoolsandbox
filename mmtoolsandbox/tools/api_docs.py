# Copyright © 2026 Apple Inc.

"""
Api Docs tools for discovering available functions.

This module provides tools for discovering and getting documentation for
all available functions in the execution context.
"""

import inspect
import json
import logging
from typing import Any, Callable, get_type_hints

import docstring_parser

from mmtoolsandbox.common.execution_context import RoleType, get_current_context
from mmtoolsandbox.common.tool_conversion import format_returns_section
from mmtoolsandbox.common.utils import register_as_tool
from mmtoolsandbox.datasets.names import DatasetName

# ============================================================================
# Helper functions for tool introspection
# ============================================================================

LOGGER = logging.getLogger(__name__)


def _get_all_tools() -> dict[str, Callable[..., Any]]:
    """Return all agent-visible tools from the current execution context.

    Returns:
        Mapping of tool names to callables, filtered to tools visible
        to ``RoleType.AGENT``. Returns an empty dict on failure.
    """
    try:
        ctx = get_current_context()
        # Use name_to_tool for full visibility (locals only has enabled tools)
        all_tools = ctx.name_to_tool

        # Filter tools based on visibility to AGENT
        visible_tools = {}
        for name, tool in all_tools.items():
            if hasattr(tool, "visible_to"):
                if RoleType.AGENT in tool.visible_to:
                    visible_tools[name] = tool
            else:
                # Default: visible to AGENT
                visible_tools[name] = tool

        return visible_tools
    except Exception:
        LOGGER.debug("Failed to get tools from execution context", exc_info=True)
        return {}


def _get_domain_for_tool(name: str, func: Callable[..., Any]) -> str:
    """Infer the domain/app name for a tool from its module path.

    Args:
        name: The tool's registered name (e.g. ``"gmail_send_email"``).
        func: The tool callable (used to read ``__module__``).

    Returns:
        A short domain string such as ``"gmail"``, ``"contact"``, or
        ``"other"`` if no domain can be inferred.
    """
    module = getattr(func, "__module__", "")

    # AppWorld tools (mmtoolsandbox.tools.appworld.gmail, etc.)
    if "tools.appworld" in module:
        return module.split(".")[-1]

    # Consolidated tools (mmtoolsandbox.tools.consolidated.spotify, etc.)
    if "tools.consolidated" in module:
        submodule = module.split(".")[-1]
        if submodule.startswith("cross_app"):
            return "cross_app"
        return submodule

    # Compact tools (mmtoolsandbox.tools.compact.spotify, etc.)
    if "tools.compact" in module:
        submodule = module.split(".")[-1]
        if submodule.startswith("cross_app"):
            return "cross_app"
        return submodule

    # Mini tools (mmtoolsandbox.tools.mini.spotify, etc.)
    if "tools.mini" in module:
        submodule = module.split(".")[-1]
        if submodule.startswith("cross_app"):
            return "cross_app"
        return submodule

    # Native tools
    if "tool_sandbox" in module:
        return module.split(".")[-1]
    elif "vision" in module:
        return "vision"

    # Infer from function name
    if "_" in name:
        return name.split("_")[0]

    return "other"


def _get_all_domains() -> dict[str, dict[str, Callable[..., Any]]]:
    """Group all agent-visible tools by domain.

    Returns:
        Nested mapping of ``{domain: {tool_name: callable}}``.
    """
    tools = _get_all_tools()
    domains: dict[str, dict[str, Callable[..., Any]]] = {}

    for name, func in tools.items():
        domain = _get_domain_for_tool(name, func)
        if domain not in domains:
            domains[domain] = {}
        domains[domain][name] = func

    return domains


def _extract_tool_doc(func: Callable[..., Any], app_name: str) -> dict[str, Any]:
    """Extract structured documentation from a tool function.

    Uses ``docstring_parser`` for consistent parsing with
    ``tool_conversion.py`` (function-calling mode).

    Args:
        func: The tool callable to document.
        app_name: Domain/app name to include in the output.

    Returns:
        Dict with keys ``app_name``, ``tool_name``, ``description``,
        ``parameters``, and ``returns``.
    """
    func_name = getattr(func, "__name__", str(func))

    docstring = docstring_parser.parse(
        inspect.getdoc(func) or "",
        style=docstring_parser.DocstringStyle.GOOGLE,
    )
    description = "" if docstring.description is None else docstring.description.strip()

    returns = format_returns_section(docstring)

    # Build param descriptions from parsed docstring
    param_descriptions = {
        p.arg_name: (p.description or "").strip() for p in docstring.params
    }

    # Build parameters list
    parameters: list[dict[str, Any]] = []
    try:
        hints = get_type_hints(func)
    except Exception:
        LOGGER.debug("Failed to get type hints for %s", func_name, exc_info=True)
        hints = {}

    try:
        sig = inspect.signature(func)
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_info: dict[str, Any] = {
                "name": param_name,
                "type": "any",
                "required": param.default is inspect.Parameter.empty,
                "description": param_descriptions.get(param_name, ""),
                "default": None,
            }

            if param_name in hints:
                hint = hints[param_name]
                type_name = getattr(hint, "__name__", str(hint))
                if "typing." in str(type_name):
                    type_name = str(hint).replace("typing.", "")
                param_info["type"] = type_name

            if param.default is not inspect.Parameter.empty:
                try:
                    json.dumps(param.default)
                    param_info["default"] = param.default
                except (TypeError, ValueError):
                    param_info["default"] = str(param.default)
                param_info["required"] = False

            parameters.append(param_info)
    except Exception:
        LOGGER.debug("Failed to extract parameters for %s", func_name, exc_info=True)

    return {
        "app_name": app_name,
        "tool_name": func_name,
        "description": description,
        "parameters": parameters,
        "returns": returns,
    }


# ============================================================================
# API Docs tools
# ============================================================================

# Toolboxes that should have api_docs tools available
_API_DOCS_TOOLBOXES = {
    DatasetName.FULL,
    DatasetName.MEDIUM,
    DatasetName.COMPACT,
    DatasetName.MINI,
}


@register_as_tool(
    toolboxes=_API_DOCS_TOOLBOXES,
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def api_docs_search_api_docs(
    query: str | None = "",
    page_index: int | None = 0,
    page_limit: int | None = 5,
) -> list[dict[str, Any]] | str:
    """
    Search for relevant API docs.

    Side Effect:
        Tools found by this search are automatically registered in the execution context
        (Implicit Registration), making them available for immediate use in code execution.

    Args:
        query: The search query string. (optional)
        page_index: The index of the page to return. (optional)
        page_limit: The maximum number of results to return per page. (optional)

    Returns:
        dict[str, Any] containing 'message' and 'results'
    """
    all_tools = _get_all_tools()
    query_lower = (query or "").lower().strip()

    if not query_lower:
        # No query - return all tools paginated
        warning_msg = (
            "Warning: No query provided. Returning a random selection of tools. "
            "Please provide a query to search for specific tools."
        )
        print(warning_msg)
        results = []
        for name, func in all_tools.items():
            domain = _get_domain_for_tool(name, func)
            results.append(_extract_tool_doc(func, domain))
        start_idx = (page_index or 0) * (page_limit or 5)
        end_idx = start_idx + (page_limit or 5)

        # Register tools in the current page
        page_results = results[start_idx:end_idx]
        registered_tools: list[str] = []
        try:
            ctx = get_current_context()
            tool_names = [r["tool_name"] for r in page_results]
            ctx.register_tools(tool_names)
            registered_tools = tool_names
        except Exception:
            LOGGER.debug("Failed to register tools in paginated results", exc_info=True)

        if get_current_context().pure_code_exec:
            return page_results
        return (
            f"Successfully registered {len(registered_tools)} tools: {registered_tools}"
        )

    # Build search text for each tool (matching AppWorld's computed_search_text logic)
    # AppWorld uses: app_name + tool_name (spaces instead of underscores) + description
    scored_results: list[tuple[float, dict[str, Any]]] = []
    query_terms = query_lower.split()

    # Import absorbed-name mapping for consolidated tool search scoring
    from mmtoolsandbox.tools.consolidated import get_absorbed_names_for_tool

    for name, func in all_tools.items():
        domain = _get_domain_for_tool(name, func)
        docstring = func.__doc__ or ""
        description = docstring.strip().split("\n")[0] if docstring else ""

        # Build search text similar to AppWorld's computed_search_text
        # app_name + tool_name (with underscores as spaces) + description
        # We also include the raw name to allow exact matches on function names
        search_text = f"{domain} {name} {name.replace('_', ' ')} {description}".lower()

        # Build expanded name text: includes absorbed original tool names so
        # consolidated tools inherit the search ranking of the tools they replace.
        absorbed = get_absorbed_names_for_tool(name)
        absorbed_text = " ".join(absorbed).lower() if absorbed else ""
        name_text = f"{name} {absorbed_text}".lower()

        # Score based on term matches (simple relevance ranking)
        score = 0.0
        for term in query_terms:
            if term in search_text or term in absorbed_text:
                # Higher score for matches in name (or absorbed names) vs description
                if term in name_text:
                    score += 2.0
                elif term in domain.lower():
                    score += 1.5
                else:
                    score += 1.0

        if score > 0:
            scored_results.append((score, _extract_tool_doc(func, domain)))

    # Sort by score descending (higher relevance first)
    scored_results.sort(key=lambda x: -x[0])
    results = [doc for _, doc in scored_results]

    # Apply pagination
    start_idx = (page_index or 0) * (page_limit or 5)
    end_idx = start_idx + (page_limit or 5)
    page_results = results[start_idx:end_idx]

    # Register found tools
    registered_tools = []
    try:
        ctx = get_current_context()
        tool_names = [r["tool_name"] for r in page_results]
        ctx.register_tools(tool_names)
        registered_tools = tool_names
    except Exception:
        LOGGER.debug("Failed to register tools from search results", exc_info=True)

    if get_current_context().pure_code_exec:
        return page_results
    return f"Successfully registered {len(registered_tools)} tools: {registered_tools}"
