# Copyright © 2026 Apple Inc.

"""Registry for named evaluation functions.

Provides a global name-to-callable mapping so that evaluation function
references serialized in JSON scenario specs (e.g. ``entity_diff_specs``)
can be resolved back to Python callables at evaluation time.
"""

from __future__ import annotations

from typing import Any, Callable

from mmtoolsandbox.common import evaluation


class FunctionRegistry:
    """Global registry mapping string names to evaluation callables.

    Used by ``EntityDiffEvaluator`` to resolve similarity function names
    from JSON scenario specs into actual Python functions at evaluation time.

    Attributes:
        _registry: Internal mapping of function names to callables.
    """

    _registry: dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, name: str, func: Callable[..., Any]) -> None:
        """Register a function under the given name.

        Args:
            name: String identifier for the function.
            func: The callable to register.
        """
        cls._registry[name] = func

    @classmethod
    def get(cls, name: str) -> Callable[..., Any]:
        """Retrieve a registered function by name.

        Args:
            name: String identifier of the function to retrieve.

        Returns:
            The callable registered under the given name.

        Raises:
            ValueError: If no function is registered under the given name.
        """
        if name not in cls._registry:
            raise ValueError(f"Function '{name}' not found in registry.")
        return cls._registry[name]

    @classmethod
    def get_name(cls, func: Callable[..., Any]) -> str:
        """Look up the registered name of a callable.

        Args:
            func: The callable whose registered name to find.

        Returns:
            The string name under which the callable is registered.

        Raises:
            ValueError: If the callable is not found in the registry.
        """
        for name, f in cls._registry.items():
            if f == func:
                return name
        raise ValueError(f"Function '{func}' not found in registry.")


# Register common similarity functions
FunctionRegistry.register("addition_similarity", evaluation.addition_similarity)
FunctionRegistry.register("snapshot_similarity", evaluation.snapshot_similarity)
FunctionRegistry.register("column_after_similarity", evaluation.column_after_similarity)
FunctionRegistry.register(
    "column_datetime_similarity", evaluation.column_datetime_similarity
)
FunctionRegistry.register(
    "column_exact_match_similarity", evaluation.column_exact_match_similarity
)
FunctionRegistry.register(
    "column_contains_similarity", evaluation.column_contains_similarity
)
FunctionRegistry.register(
    "column_tool_trace_exact_match_similarity",
    evaluation.column_tool_trace_exact_match_similarity,
)
FunctionRegistry.register(
    "column_rouge_l_similarity", evaluation.column_rouge_l_similarity
)
FunctionRegistry.register("column_close_similarity", evaluation.column_close_similarity)
FunctionRegistry.register(
    "column_datetime_naive_equal_similarity",
    evaluation.column_datetime_naive_equal_similarity,
)
FunctionRegistry.register("column_one_similarity", evaluation.column_one_similarity)
FunctionRegistry.register(
    "column_ignore_similarity", evaluation.column_ignore_similarity
)
FunctionRegistry.register("guardrail_similarity", evaluation.guardrail_similarity)
FunctionRegistry.register("removal_similarity", evaluation.removal_similarity)
FunctionRegistry.register("update_similarity", evaluation.update_similarity)
