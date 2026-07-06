# Copyright © 2026 Apple Inc.

# ruff: noqa: F401, I001
# Make sure to import submodule here so that it's visible to inspect.
# Tool registration happens at import time via @register_as_tool decorators.

# Vision Tools (core for visual scenarios)
from mmtoolsandbox.tools.vision import images as vision_images
from mmtoolsandbox.tools.vision import web_search as vision_web_search

# ToolSandbox Tools
from mmtoolsandbox.tools.tool_sandbox import calendar as calendar
from mmtoolsandbox.tools.tool_sandbox import reminder as reminder
from mmtoolsandbox.tools.tool_sandbox import setting as setting
from mmtoolsandbox.tools.tool_sandbox import user_tools as user_tools

# Code Execution Mode Tools
from mmtoolsandbox.tools import code_execution as code_execution
from mmtoolsandbox.tools import api_docs as api_docs
