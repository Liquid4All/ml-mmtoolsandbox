"""
AppWorld Extension for MMToolSandbox.

This extension integrates AppWorld's 457 APIs across 9 day-to-day apps
(Amazon, Spotify, Venmo, etc.) with MMToolSandbox's conversational
evaluation framework.

Key Components:
- bridge.py: Adapter between AppWorld and MMToolSandbox
- state.py: State dependency layer for AppWorld APIs
- tools/: Wrapped AppWorld APIs as MMToolSandbox tools
- scenarios/: AppWorld tasks converted to MMToolSandbox scenarios
- evaluation/: Adapted evaluation for AppWorld-based scenarios

Usage:
    # Check if AppWorld is available
    from mmtoolsandbox.appworld import APPWORLD_AVAILABLE

    # Import to register all AppWorld tools (only if available)
    if APPWORLD_AVAILABLE:
        from mmtoolsandbox.appworld import tools

    # Convert an AppWorld task to a scenario
    from mmtoolsandbox.appworld.scenarios import task_adapters
    scenario = task_adapters.appworld_task_to_scenario("task_id_123")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mmtoolsandbox.appworld.bridge import AppWorldBridge
    from mmtoolsandbox.appworld.state import AppWorldState

# Try to add appworld to path if it exists as a sibling directory of the repo root.
# This must happen BEFORE 'import appworld' to prioritize local source over installed package.
# Path: mmtoolsandbox/appworld/__init__.py → repo_root is 3 parents up → sibling is ../appworld
_REPO_ROOT = Path(__file__).parent.parent.parent
APPWORLD_PATHS = [
    _REPO_ROOT.parent / "appworld" / "src",
    _REPO_ROOT.parent / "appworld",
]
for appworld_path in APPWORLD_PATHS:
    if appworld_path.exists() and str(appworld_path) not in sys.path:
        sys.path.insert(0, str(appworld_path))
        break

# Check if AppWorld is installed
try:
    import appworld  # noqa: F401

    APPWORLD_AVAILABLE = True
except ImportError:
    APPWORLD_AVAILABLE = False


def get_appworld_bridge() -> AppWorldBridge:
    """Lazy import to avoid circular dependencies.

    Raises:
        ImportError: If AppWorld is not installed
    """
    if not APPWORLD_AVAILABLE:
        raise ImportError(
            "AppWorld is not installed. Clone it as a sibling directory:\n"
            "  git clone https://github.com/StonyBrookNLP/appworld.git ../appworld\n"
            "  cd ../appworld && pip install -e . --no-deps && appworld install && appworld download data"
        )
    from mmtoolsandbox.appworld.bridge import get_appworld_bridge as _get

    return _get()


def get_appworld_state() -> AppWorldState:
    """Lazy import to avoid circular dependencies."""
    from mmtoolsandbox.appworld.state import get_appworld_state as _get

    return _get()


def is_appworld_scenario(scenario: Any) -> bool:
    """Check if a scenario uses AppWorld tools/tasks.

    Args:
        scenario: A scenario dict or object with metadata

    Returns:
        True if the scenario uses AppWorld
    """
    if not APPWORLD_AVAILABLE:
        return False

    metadata = getattr(scenario, "metadata", None) or scenario.get("metadata", {})
    return bool(metadata.get("appworld_task_id"))


# Re-export commonly used items
__all__ = [
    "APPWORLD_AVAILABLE",
    "get_appworld_bridge",
    "get_appworld_state",
    "is_appworld_scenario",
]
