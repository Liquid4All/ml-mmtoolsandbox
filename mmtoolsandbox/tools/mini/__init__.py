# Copyright © 2026 Apple Inc.

"""
Tool definitions for the MINI toolbox (~35 workflow-based tools).

Every tool in MINI is explicitly defined — there is no passthrough.
Each tool dispatches directly to original AppWorld/native tools.
All tools fit in the LLM system prompt; tool search is not needed.

The MINI toolbox is loaded by ``get_all_tools(DatasetName.MINI)`` in
``common/tool_registry.py``, which imports all MINI modules.
"""
