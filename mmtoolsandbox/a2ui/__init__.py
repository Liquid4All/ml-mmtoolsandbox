"""
A2UI Extension for MMToolSandbox.

This extension adds support for the A2UI protocol, allowing agents to
interact with a simulated UI.
"""

from mmtoolsandbox.a2ui.html_renderer import render_surface_to_html
from mmtoolsandbox.a2ui.state import UIState, get_ui_state
from mmtoolsandbox.a2ui.tools import (
    render_ui_screen,
    show_ui_to_user,
    ui_explore_capabilities,
    ui_get_item_details,
    ui_get_quick_start,
    ui_list_items,
    ui_search_docs,
    ui_user_interact,
)

__all__ = [
    "UIState",
    "get_ui_state",
    "render_surface_to_html",
    "render_ui_screen",
    "show_ui_to_user",
    "ui_user_interact",
    "ui_explore_capabilities",
    "ui_get_quick_start",
    "ui_list_items",
    "ui_get_item_details",
    "ui_search_docs",
]
