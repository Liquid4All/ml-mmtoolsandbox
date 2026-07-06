"""
AppWorld Tool Registry for MMToolSandbox.

This module provides the tool registry and generated API wrappers for all
AppWorld apps (457+ APIs across 13 apps).  All tools are registered into
the core MMToolSandbox tool registry under ``DatasetName.FULL`` so they
are auto-discovered by ``get_all_tools(DatasetName.FULL)``.

The generated wrappers (gmail.py, amazon.py, …) call through the AppWorld
bridge at runtime — the bridge is only initialised in spawned worker
processes, so importing this module in the main process is safe (no
freezegun contamination).
"""

from collections import defaultdict
from typing import (
    Any,
    Callable,
    TypeVar,
)

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.tool_registry import register_tool
from mmtoolsandbox.datasets.names import DatasetName

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Local registry for AppWorld tools (populated at import time by decorators)
# ---------------------------------------------------------------------------

_appworld_registry: dict[str, dict[str, Callable[..., Any]]] = defaultdict(dict)
_tools_loaded: bool = False
_core_registry_populated: bool = False


def register_appworld_tool(
    app: str,
    name: str | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to register a function as an AppWorld tool.

    Args:
        app: The AppWorld app this tool belongs to (e.g., "spotify", "amazon")
        name: Optional tool name (defaults to function name)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        tool_name = name or func.__name__
        _appworld_registry[app][tool_name] = func
        return func

    return decorator


def _load_tools() -> None:
    """Import all generated tool modules to trigger registration.

    This uses a lazy self-import because the submodules (gmail.py, amazon.py,
    etc.) use ``@register_appworld_tool`` decorators that populate
    ``_appworld_registry`` at decoration time.  Importing them eagerly at
    module level would force registration before the registry dict exists.
    The ``_tools_loaded`` flag ensures this only happens once.
    """
    global _tools_loaded
    if _tools_loaded:
        return

    # Lazy: self-import triggers @register_appworld_tool decorators in each
    # submodule.  Cannot be top-level because the decorator (defined above)
    # must exist before the submodules are imported.
    from mmtoolsandbox.tools.appworld import (  # noqa: F401
        alarms,
        amazon,
        contacts,
        file_system,
        gmail,
        messages,
        phone,
        simple_note,
        splitwise,
        spotify,
        supervisor,
        todoist,
        venmo,
    )

    # phone.py is imported but only its account/auth tools are kept.
    # Tools that overlap with contacts, messages, and alarms are removed
    # below so the agent sees only the split-domain versions.
    _phone_tools_to_keep = frozenset(
        {
            "phone_show_account",
            "phone_signup",
            "phone_delete_account",
            "phone_update_account_name",
            "phone_login",
            "phone_logout",
            "phone_send_password_reset_code",
            "phone_reset_password",
        }
    )
    phone_tools = _appworld_registry.get("phone", {})
    for tool_name in list(phone_tools.keys()):
        if tool_name not in _phone_tools_to_keep:
            del phone_tools[tool_name]

    _tools_loaded = True


def get_appworld_tools(app: str | None = None) -> dict[str, Callable[..., Any]]:
    """Get registered AppWorld tools.

    Args:
        app: Optional app name to filter tools.  If None, returns all tools.

    Returns:
        Dictionary mapping tool name to tool function.
    """
    _load_tools()

    if app is not None:
        return dict(_appworld_registry.get(app, {}))

    all_tools: dict[str, Callable[..., Any]] = {}
    for app_name, tools in _appworld_registry.items():
        for tool_name, tool_func in tools.items():
            if tool_name.startswith(f"{app_name}_"):
                all_tools[tool_name] = tool_func
            else:
                all_tools[f"{app_name}_{tool_name}"] = tool_func
    return all_tools


def get_appworld_apps() -> list[str]:
    """Get list of registered AppWorld apps."""
    _load_tools()
    return list(_appworld_registry.keys())


def get_tool_count() -> int:
    """Get the total number of registered AppWorld tools."""
    _load_tools()
    return sum(len(tools) for tools in _appworld_registry.values())


def clear_appworld_registry() -> None:
    """Clear all registered AppWorld tools.  Useful for testing."""
    global _tools_loaded, _core_registry_populated
    _appworld_registry.clear()
    _tools_loaded = False
    _core_registry_populated = False


# ---------------------------------------------------------------------------
# Core registry integration
# ---------------------------------------------------------------------------


def register_appworld_tools_to_core_registry() -> int:
    """Register AppWorld tools into MMToolSandbox's core tool registry.

    Populates ``_registry[DatasetName.FULL]`` with all AppWorld tool
    wrappers (457+ APIs).  Native tools are registered separately via
    their ``@register_as_tool`` decorators.

    This function is **idempotent** — repeated calls are no-ops.

    Returns:
        Number of tools registered (0 on subsequent calls).
    """
    global _core_registry_populated
    if _core_registry_populated:
        return 0

    _load_tools()

    registered_count = 0

    all_appworld = get_appworld_tools()
    for tool_name, tool_func in all_appworld.items():
        if not hasattr(tool_func, "is_tool"):
            setattr(tool_func, "is_tool", True)
        if not hasattr(tool_func, "database_namespaces"):
            setattr(tool_func, "database_namespaces", set())
        if not hasattr(tool_func, "visible_to"):
            setattr(tool_func, "visible_to", (RoleType.AGENT,))

        try:
            register_tool(DatasetName.FULL, tool_func)
            registered_count += 1
        except KeyError:
            pass

    _core_registry_populated = True
    return registered_count


# Backward-compatible wrapper (used by external scripts and tests)
def register_with_agentsandbox(
    toolbox_names: set[Any] | None = None,
    tool_filter: Callable[[str], bool] | None = None,
) -> int:
    """Register AppWorld tools with MMToolSandbox's core tool registry.

    .. deprecated::
        Use ``register_appworld_tools_to_core_registry()`` instead.
        This function is kept for backward compatibility.
    """
    return register_appworld_tools_to_core_registry()


def create_appworld_toolbox(
    include_agentsandbox_tools: bool = False,
    tool_filter: Callable[[str], bool] | None = None,
) -> dict[str, Callable[..., Any]]:
    """Create a toolbox dictionary with AppWorld tools.

    Useful for manually injecting tools into an execution context
    or for running scenarios outside the standard MMToolSandbox pipeline.
    """
    _load_tools()

    tools: dict[str, Callable[..., Any]] = {}

    for t_name, func in get_appworld_tools().items():
        if tool_filter is None or tool_filter(t_name):
            tools[t_name] = func

    if include_agentsandbox_tools:
        from mmtoolsandbox.common.tool_registry import get_all_tools

        agentsandbox_tools = get_all_tools(DatasetName.FULL)
        tools.update(agentsandbox_tools)

    return tools


__all__ = [
    "register_appworld_tool",
    "get_appworld_tools",
    "get_appworld_apps",
    "get_tool_count",
    "clear_appworld_registry",
    "register_appworld_tools_to_core_registry",
    "register_with_agentsandbox",
    "create_appworld_toolbox",
]
