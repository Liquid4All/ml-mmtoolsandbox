# Copyright © 2026 Apple Inc.

"""Code execution tool for code execution mode.

This module provides an `execute_code` tool that allows agents to execute
Python code. Allowed functions are pre-injected into the execution context's
interactive console locals. Additional tools are automatically enabled and
injected when discovered via `api_docs_search_api_docs`.

Usage:
    # Agent calls execute_code as a tool
    execute_code(code='''
        # Discover available functions (automatically enables them)
        results = api_docs_search_api_docs("send message")
        print(results)

        # Call functions directly
        result = some_function(param1="value")
    ''')
"""

import ast
import io
import sys
import traceback
from typing import Any, cast

from mmtoolsandbox.common.execution_context import RoleType, get_current_context
from mmtoolsandbox.common.image_id import contains_images
from mmtoolsandbox.common.utils import register_as_tool
from mmtoolsandbox.datasets.names import DatasetName


@register_as_tool(
    toolboxes={
        DatasetName.FULL,
        DatasetName.MEDIUM,
        DatasetName.COMPACT,
        DatasetName.MINI,
    },
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def execute_code(code: str) -> Any:
    """Execute Python code with access to all available functions.

    Use api_docs_search_api_docs(query) to discover functions, then call them directly.

    Args:
        code: Python code to execute. The last expression's value is returned
              automatically (like a REPL). Use print() for intermediate output.

    Returns:
        Any print() output plus the last expression's value, or error message.
        If the result contains images, the raw structure (list, dict, etc.) is returned.
        If there is also text output, a list [text_output, result_structure] is returned.
    """
    context = get_current_context()
    locals_dict = cast(dict[str, Any], context.interactive_console.locals)
    safety_guard = context.safety_guard

    # Layer 1: Static syntax check
    if safety_guard and safety_guard.config.enable_syntax_check:
        is_safe, safety_msg = safety_guard.is_syntax_safe(code)
        if not is_safe:
            return f"Execution blocked: {safety_msg}"

    old_stdout = sys.stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Layer 2: Enable runtime monkey-patching
    if safety_guard and safety_guard.config.enable_runtime_guard:
        safety_guard.enable()

    try:
        # Parse the code to separate statements from final expression
        tree = ast.parse(code)
        last_value = None

        if tree.body:
            # Check if the last statement is an expression
            last_stmt = tree.body[-1]
            if isinstance(last_stmt, ast.Expr):
                # Execute all but the last statement
                if len(tree.body) > 1:
                    exec_tree = ast.Module(body=tree.body[:-1], type_ignores=[])
                    exec(compile(exec_tree, "<string>", "exec"), locals_dict)
                # Evaluate the last expression and capture its value
                last_value = eval(
                    compile(ast.Expression(body=last_stmt.value), "<string>", "eval"),
                    locals_dict,
                )
            else:
                # No trailing expression, just exec everything
                exec(code, locals_dict)
        else:
            exec(code, locals_dict)

        output = captured_output.getvalue()

        # Build result from: printed output + last expression value + result variable
        parts = []
        if output:
            parts.append(output.rstrip())

        # Check if the last value contains images
        has_images = contains_images(last_value)

        if has_images:
            # If we have images, we return the structure containing images.
            # If there is also text output, we return a list containing both.
            if output:
                return [output.rstrip(), last_value]
            return last_value

        # Standard text-only response
        if last_value is not None:
            parts.append(repr(last_value))
        elif "result" in locals_dict:
            parts.append(f"result = {repr(locals_dict['result'])}")

        if parts:
            return "\n".join(parts)

        return "Code executed successfully (no output)"

    except Exception:
        output = captured_output.getvalue()
        error_msg = traceback.format_exc(limit=3)
        if output:
            return f"{output}\nError: {error_msg}"
        return f"Error: {error_msg}"

    finally:
        sys.stdout = old_stdout
        # Layer 2: Restore original functions
        if safety_guard and safety_guard.config.enable_runtime_guard:
            safety_guard.disable()
