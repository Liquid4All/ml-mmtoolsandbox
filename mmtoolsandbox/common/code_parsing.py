# Copyright © 2026 Apple Inc.

"""AST-based code analysis for safety guard.

Provides lightweight import and function call extraction using Python's
stdlib ``ast`` module (no ``libcst`` dependency).
"""

from __future__ import annotations

import ast


def parse_imports(code: str) -> set[str]:
    """Extract top-level module names from all import statements in *code*.

    ``import os.path`` and ``from os.path import join`` both yield ``"os"``.

    Returns an empty set if the code cannot be parsed.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                modules.add(node.module.split(".")[0])
    return modules


class _FunctionPathVisitor(ast.NodeVisitor):
    """Walk an AST and collect dotted function-call paths.

    Handles:
    - ``os.remove(...)``       -> ``"os.remove"``
    - ``subprocess.run(...)``  -> ``"subprocess.run"``
    - ``pathlib.Path(...).unlink()`` -> ``"pathlib.Path.unlink"``
    - ``open(...)``            -> ``"builtins.open"``
    - Aliased imports: ``import os as myos; myos.remove(...)`` -> ``"os.remove"``
    """

    BUILTIN_FUNCTIONS = frozenset(
        {
            "exit",
            "quit",
            "open",
            "breakpoint",
            "compile",
            "exec",
            "eval",
            "__import__",
            "getattr",
            "setattr",
            "delattr",
            "vars",
            "globals",
        }
    )

    def __init__(self, aliases: dict[str, str]) -> None:
        self.aliases = aliases
        self.paths: set[str] = set()
        # Track variable -> constructor class, e.g. ``p = pathlib.Path(...)``
        self.instance_map: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _attr_chain(node: ast.expr) -> list[str] | None:
        """Return the dotted name chain for an Attribute/Name node, or None."""
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
            parts.reverse()
            return parts
        # e.g. chained call like ``pathlib.Path(...).unlink()``
        if isinstance(node, ast.Call):
            inner = _FunctionPathVisitor._attr_chain(node.func)
            if inner is not None:
                parts.reverse()
                return inner + parts
        return None

    def _resolve(self, dotted: str) -> str:
        """Resolve the first segment through import aliases."""
        parts = dotted.split(".")
        if parts[0] in self.aliases:
            parts[0] = self.aliases[parts[0]]
        return ".".join(parts)

    # ------------------------------------------------------------------
    # Visitor callbacks
    # ------------------------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        # Track ``p = pathlib.Path(...)`` so ``p.unlink()`` resolves correctly.
        if isinstance(node.value, ast.Call):
            chain = self._attr_chain(node.value.func)
            if chain is not None:
                resolved = self._resolve(".".join(chain))
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.instance_map[target.id] = resolved
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        chain = self._attr_chain(node.func)
        if chain is not None:
            # Check instance map for the first segment
            if chain[0] in self.instance_map:
                resolved = self.instance_map[chain[0]] + "." + ".".join(chain[1:])
            else:
                resolved = self._resolve(".".join(chain))
            # Prefix bare builtins
            if resolved in self.BUILTIN_FUNCTIONS:
                resolved = f"builtins.{resolved}"
            self.paths.add(resolved)
        self.generic_visit(node)


def _collect_aliases(tree: ast.Module) -> dict[str, str]:
    """Build a mapping from alias names to their true module paths."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname is not None:
                    aliases[alias.asname] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                true_name = f"{module}.{alias.name}" if module else alias.name
                if alias.asname is not None:
                    aliases[alias.asname] = true_name
                else:
                    aliases[alias.name] = true_name
    return aliases


def parse_code_function_paths(code: str) -> set[str]:
    """Return the set of dotted function-call paths found in *code*.

    Example::

        >>> parse_code_function_paths("import os; os.remove('x')")
        {'os.remove'}

    Returns an empty set if the code cannot be parsed.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    aliases = _collect_aliases(tree)
    visitor = _FunctionPathVisitor(aliases)
    visitor.visit(tree)
    return visitor.paths


# ---------------------------------------------------------------------------
# Dangerous dunder attribute detection
# ---------------------------------------------------------------------------

# Dunder attributes that can be used to escape a sandbox.
DANGEROUS_DUNDER_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "__subclasses__",
        "__globals__",
        "__code__",
        "__builtins__",
        "__import__",
        "__loader__",
        "__spec__",
        "__dict__",
        "__class__",
    }
)


def parse_dunder_attr_access(code: str) -> set[str]:
    """Return the set of dangerous dunder attribute names accessed in *code*.

    Detects patterns like ``object.__subclasses__()``, ``func.__globals__``,
    ``__import__("os")``, etc.

    Returns an empty set if the code cannot be parsed.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    found: set[str] = set()
    for node in ast.walk(tree):
        # obj.__subclasses__, func.__globals__, etc.
        if isinstance(node, ast.Attribute) and node.attr in DANGEROUS_DUNDER_ATTRIBUTES:
            found.add(node.attr)
        # Bare __import__("os") or __builtins__
        if isinstance(node, ast.Name) and node.id in DANGEROUS_DUNDER_ATTRIBUTES:
            found.add(node.id)
    return found
