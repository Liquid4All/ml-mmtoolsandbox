# Copyright © 2026 Apple Inc.

"""Safety guard for agent-generated code execution.

Provides three layers of protection:

1. **Static syntax analysis** — checks the AST before execution to reject
   disallowed imports, dangerous function calls, and dunder attribute access.
2. **Runtime guards** — modifies the runtime environment to enforce security
   boundaries. This includes path-restricted ``open()``, ``sys.modules`` proxy,
   ``SystemExit`` swap, and memory limits (via ``resource.setrlimit``).
3. **Execution limits** — wraps the execution flow to enforce time and output
   constraints (timeout and result size checks).

Modelled after the language-level restrictions of ``appworld.common.safety_guard``
but uses only stdlib ``ast`` (no ``libcst``), has no dependency on appworld,
and runs in-process without containerization.
"""

from __future__ import annotations

import builtins
import io
import resource
import signal
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from types import ModuleType
from typing import IO, Any, TypeVar, cast

from mmtoolsandbox.common.code_parsing import (
    parse_code_function_paths,
    parse_dunder_attr_access,
    parse_imports,
)

# ---------------------------------------------------------------------------
# Default configuration constants
# ---------------------------------------------------------------------------

DISALLOWED_MODULE_TO_FUNCTION_NAMES: dict[str, list[str]] = {
    "builtins": [
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
    ],
    "sys": ["exit"],
    "matplotlib.pyplot": ["show"],
    "plt": ["show"],
    "os": [
        "_exit",
        "open",
        "read",
        "write",
        "close",
        "walk",
        "kill",
        "system",
        "putenv",
        "remove",
        "removedirs",
        "rmdir",
        "fchdir",
        "setuid",
        "fork",
        "forkpty",
        "killpg",
        "rename",
        "renames",
        "truncate",
        "replace",
        "unlink",
        "fchmod",
        "fchown",
        "chmod",
        "chown",
        "chroot",
        "lchflags",
        "lchmod",
        "lchown",
        "chdir",
        "environ",
        "getcwd",
        "listdir",
        "scandir",
        "popen",
        "execl",
        "execle",
        "execlp",
        "execlpe",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
    ],
    "io": [
        "open",
        "open_code",
        "FileIO",
    ],
    "shutil": [
        "rmtree",
        "move",
        "chown",
        "copy",
        "copy2",
        "copyfile",
        "copytree",
        "make_archive",
        "get_archive_formats",
    ],
    "subprocess": [
        "Popen",
        "call",
        "check_call",
        "check_output",
        "run",
        "getoutput",
        "getstatusoutput",
    ],
    "pathlib.Path": [
        "open",
        "write_text",
        "read_bytes",
        "write_bytes",
        "unlink",
        "rmdir",
        "rename",
        "replace",
        "chmod",
        "lchmod",
        "chown",
        "lchown",
        "touch",
        "symlink_to",
        "link_to",
        "mkdir",
        "expanduser",
    ],
    "fileinput": [
        "input",
        "filename",
        "nextfile",
        "close",
        "lineno",
    ],
    "glob": [
        "glob",
        "iglob",
    ],
    "json": [
        "dump",
        "load",
    ],
    "tempfile": [
        "mktemp",
        "mkdtemp",
        "mkstemp",
        "NamedTemporaryFile",
        "TemporaryDirectory",
    ],
    "zipfile": [
        "ZipFile",
    ],
    "shelve": [
        "open",
    ],
    "dbm": [
        "open",
        "close",
    ],
    "pickle": [
        "dump",
        "load",
    ],
    "codecs": [
        "open",
    ],
    "bz2": [
        "open",
    ],
    "gzip": [
        "open",
    ],
    "tarfile": [
        "open",
    ],
    "csv": [
        "reader",
        "writer",
        "DictReader",
        "DictWriter",
    ],
    "time": [
        "sleep",
    ],
    "pdb": [
        "set_trace",
    ],
    "urllib.request": [
        "urlretrieve",
        "urlcleanup",
        "urlopen",
        "URLopener",
        "FancyURLopener",
    ],
}

DISALLOWED_FUNCTION_PATHS: set[str] = {
    f"{module_name}.{function_name}"
    for module_name, function_names in DISALLOWED_MODULE_TO_FUNCTION_NAMES.items()
    for function_name in function_names
}

ALLOWED_MODULE_NAMES: set[str] = {
    "mmtoolsandbox",
    "array",
    "builtins",
    "calendar",
    "collections",
    "contextlib",
    "copy",
    "csv",
    "time",
    "datetime",
    "enum",
    "fractions",
    "functools",
    "heapq",
    "itertools",
    "json",
    "math",
    "matplotlib",
    "numbers",
    "numpy",
    "operator",
    "os",
    "pathlib",
    "pendulum",
    "PIL",
    "pprint",
    "random",
    "re",
    "string",
    "textwrap",
    "uuid",
    "yaml",
    "bisect",
    "difflib",
    "typing",
    "unittest",
    "io",
    "decimal",
    "dataclasses",
    "email",
    "hashlib",
    "pydoc",
    "queue",
    "reprlib",
    "statistics",
    "abc",
    "ast",
    "base64",
    "colorsys",
    "contextvars",
    "encodings",
    "errno",
    "fnmatch",
    "genericpath",
    "getopt",
    "graphlib",
    "hmac",
    "html",
    "keyword",
    "opcode",
    "optparse",
    "pydoc_data",
    "quopri",
    "shlex",
    "sre_compile",
    "sre_constants",
    "sre_parse",
    "stringprep",
    "struct",
    "token",
    "traceback",
    "urllib",
    "weakref",
    "xml",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _natural_join(items: Sequence[str], by: str = "and") -> str:
    """Join a list of strings with commas and a final conjunction."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f" {by} ".join([", ".join(items[:-1]), items[-1]])


class _RestrictedModules(dict):  # type: ignore[type-arg]
    """A ``sys.modules`` proxy that blocks lookups for dangerous modules.

    Prevents agent code from fishing out blocked modules via
    ``sys.modules["subprocess"]``.
    """

    _BLOCKED_MODULES: frozenset[str] = frozenset(
        {
            "subprocess",
            "shutil",
            "ctypes",
            "socket",
            "http",
            "ftplib",
            "smtplib",
            "telnetlib",
            "xmlrpc",
            "multiprocessing",
            "signal",
            "pty",
            "resource",
            "mmap",
            "importlib",
        }
    )

    def __init__(
        self, original: dict[str, ModuleType], allowed_modules: set[str]
    ) -> None:
        super().__init__(original)
        self._allowed = allowed_modules

    def __getitem__(self, key: str) -> ModuleType:
        top_level = key.split(".")[0]
        if top_level in self._BLOCKED_MODULES:
            raise PermissionError(
                f"Access to module '{key}' via sys.modules is not allowed."
            )
        return cast(ModuleType, super().__getitem__(key))

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, PermissionError):
            return default


class LimitedStringIO(io.StringIO):
    """A StringIO that raises an error if the content exceeds a size limit."""

    def __init__(self, limit: int | None = None, initial_value: str = "") -> None:
        super().__init__(initial_value)
        self._limit = limit

    def write(self, s: str) -> int:
        if self._limit is not None and self.tell() + len(s) > self._limit:
            raise RuntimeError(
                f"Output limit of {self._limit} characters exceeded. "
                "Please reduce the amount of printed output."
            )
        return super().write(s)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SafetyGuardConfig:
    """Configuration for :class:`SafetyGuard`.

    All fields have sensible defaults so that ``SafetyGuardConfig()`` gives a
    fully-enabled guard.
    """

    enable_syntax_check: bool = True
    enable_runtime_guard: bool = True
    timeout_seconds: int | None = 100  # None = no timeout
    memory_limit_mb: int | None = 512  # None = no limit
    max_output_chars: int | None = 10000  # None = no limit
    allowed_modules: set[str] = field(default_factory=lambda: set(ALLOWED_MODULE_NAMES))
    disallowed_function_paths: set[str] = field(
        default_factory=lambda: set(DISALLOWED_FUNCTION_PATHS)
    )
    allowed_read_paths: list[str] | None = None  # None = allow reads from anywhere
    allowed_write_paths: list[str] | None = None  # None = block all writes


# ---------------------------------------------------------------------------
# SafetyGuard
# ---------------------------------------------------------------------------

ReturnT = TypeVar("ReturnT")


class SafetyGuard:
    """Safety guard for agent-generated code.

    Layer 1 (static): AST-based check of imports, function calls, and dunder
    attributes.

    Layer 2 (runtime): Modifies the runtime environment to enforce security
    boundaries. These are applied via ``enable()``/``disable()``:
    - **Path-restricted ``open()``** — enforces read/write path policies.
    - **``sys.modules`` proxy** — blocks module fishing.
    - **``SystemExit`` swap** — prevents process exit.
    - **Memory limit** — enforces memory usage limits via OS-level resource controls.

    Layer 3 (execution): Wraps the execution flow to enforce limits:
    - **Timeout** — enforces execution time limits.
    - **Output limit** — enforces output size limits (via helper methods).
    """

    def __init__(self, config: SafetyGuardConfig | None = None) -> None:
        self.config = config or SafetyGuardConfig()

        # Save originals for restore in disable().
        self._original_open: Callable[..., Any] = builtins.open
        self._original_io_open: Callable[..., Any] = io.open
        self._original_io_open_code: Callable[..., Any] = io.open_code
        self._original_system_exit: type = builtins.SystemExit
        self._original_sys_modules: dict[str, ModuleType] | None = None
        self._original_rlimit_as: tuple[int, int] | None = None

    # ------------------------------------------------------------------
    # Layer 1: Static syntax check
    # ------------------------------------------------------------------

    def is_syntax_safe(self, code: str) -> tuple[bool, str]:
        """Check whether the given code passes static safety analysis.

        This method parses the code into an AST and checks for:
        - Disallowed imports (e.g., subprocess, os)
        - Disallowed function calls (e.g., open, exec)
        - Disallowed dunder attributes (e.g., __subclasses__)

        Args:
            code: The Python code string to analyze.

        Returns:
            A tuple (is_safe, reason). If safe, returns (True, "").
            If unsafe, returns (False, reason_string).
            Note: Code with syntax errors is considered safe (it will fail at exec time).
        """
        imported_module_names = parse_imports(code)
        prohibited_module_names = sorted(
            imported_module_names - self.config.allowed_modules
        )
        if prohibited_module_names:
            word = "module" if len(prohibited_module_names) == 1 else "modules"
            return (
                False,
                f"Usage of the following {word} is not allowed: "
                + _natural_join(prohibited_module_names)
                + ".",
            )

        used_function_paths = parse_code_function_paths(code)
        prohibited_function_paths = sorted(
            used_function_paths & self.config.disallowed_function_paths
        )
        if prohibited_function_paths:
            word = "function" if len(prohibited_function_paths) == 1 else "functions"
            return (
                False,
                f"Usage of the following {word} is not allowed: "
                + _natural_join(prohibited_function_paths)
                + ".",
            )

        dunder_attrs = parse_dunder_attr_access(code)
        if dunder_attrs:
            word = "attribute" if len(dunder_attrs) == 1 else "attributes"
            return (
                False,
                f"Access to the following {word} is not allowed: "
                + _natural_join(sorted(dunder_attrs))
                + ".",
            )

        return True, ""

    # ------------------------------------------------------------------
    # Layer 2: Runtime guards
    # ------------------------------------------------------------------

    def enable(self) -> None:
        """Activate runtime guards to restrict the execution environment.

        This method performs the following monkey-patching:
        1. Replaces `builtins.open` with a path-restricted version.
        2. Swaps `sys.modules` with a proxy that blocks access to dangerous modules.
        3. Replaces `sys.exit` to prevent the agent from killing the process.
        4. Sets memory limits using `resource.setrlimit` (if configured).

        This must be paired with a subsequent call to :meth:`disable` to restore
        the original environment.
        """
        guard = self  # capture for closure

        def _restricted_open(
            file: str,
            mode: str = "r",
            buffering: int = -1,
            encoding: str | None = None,
            errors: str | None = None,
            newline: str | None = None,
            closefd: bool = True,
            opener: Callable[[str, int], int] | None = None,
        ) -> IO[Any]:
            import os.path as _osp

            is_write = "w" in mode or "a" in mode or "x" in mode or "+" in mode

            if is_write:
                write_paths = guard.config.allowed_write_paths
                if write_paths is None:
                    raise PermissionError("Writing to OS file system is disabled.")
                resolved = _osp.realpath(str(file))
                if not any(resolved.startswith(_osp.realpath(p)) for p in write_paths):
                    raise PermissionError(
                        f"Writing to '{file}' is not allowed. "
                        f"Writes are restricted to: {write_paths}"
                    )
            else:
                read_paths = guard.config.allowed_read_paths
                if read_paths is not None:
                    resolved = _osp.realpath(str(file))
                    if not any(
                        resolved.startswith(_osp.realpath(p)) for p in read_paths
                    ):
                        raise PermissionError(
                            f"Reading from '{file}' is not allowed. "
                            f"Reads are restricted to: {read_paths}"
                        )

            return cast(
                IO[Any],
                guard._original_open(
                    file, mode, buffering, encoding, errors, newline, closefd, opener
                ),
            )

        # 1. Path-restricted open()
        builtins.open = _restricted_open  # type: ignore[assignment]
        io.open = _restricted_open  # type: ignore[assignment]
        io.open_code = _restricted_open  # type: ignore[assignment]

        # 2. SystemExit → Exception (prevent process exit)
        builtins.SystemExit = Exception  # type: ignore[assignment, misc]

        # 3. sys.modules proxy (block module fishing)
        self._original_sys_modules = sys.modules
        sys.modules = _RestrictedModules(sys.modules, self.config.allowed_modules)

        # 4. Memory limit (Unix/macOS only, not Windows)
        if self.config.memory_limit_mb is not None and resource is not None:
            try:
                soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                self._original_rlimit_as = (soft, hard)
                limit_bytes = self.config.memory_limit_mb * 1024 * 1024
                # Ensure we don't exceed the hard limit
                if hard != resource.RLIM_INFINITY and limit_bytes > hard:
                    limit_bytes = hard
                resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, hard))
            except (ValueError, OSError):
                # Ignore errors if we can't set the limit (e.g. permission denied)
                pass

    def disable(self) -> None:
        """Restore all runtime guards."""
        # Restore sys.modules FIRST — other restores may trigger imports.
        if self._original_sys_modules is not None:
            sys.modules = self._original_sys_modules
            self._original_sys_modules = None

        builtins.open = self._original_open
        io.open = self._original_io_open
        io.open_code = self._original_io_open_code
        builtins.SystemExit = self._original_system_exit  # type: ignore[assignment, misc]

        # Restore memory limit
        if self._original_rlimit_as is not None and resource is not None:
            try:
                resource.setrlimit(resource.RLIMIT_AS, self._original_rlimit_as)
            except (ValueError, OSError):
                pass
            self._original_rlimit_as = None

    # ------------------------------------------------------------------
    # Layer 3: Execution & Timeout
    # ------------------------------------------------------------------

    def run(
        self,
        function: Callable[..., ReturnT],
        *args: Any,
        **kwargs: Any,
    ) -> ReturnT:
        """Execute a function with all safety guards enabled.

        This method encapsulates the entire safety lifecycle:
        1. Enables runtime guards (if configured).
        2. Executes the function with a timeout (if configured).
        3. Disables runtime guards (in a finally block).

        Args:
            function: The callable to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            The return value of the executed function.

        Raises:
            TimeoutError: If execution exceeds the configured timeout.
            Exception: Any exception raised by the function.
        """
        if self.config.enable_runtime_guard:
            self.enable()

        try:
            # Use configured timeout
            return self._timeout_call(function, *args, **kwargs)
        finally:
            if self.config.enable_runtime_guard:
                self.disable()

    def _timeout_call(
        self,
        function: Callable[..., ReturnT],
        *args: Any,
        **kwargs: Any,
    ) -> ReturnT:
        """Call *function* with a POSIX ``SIGALRM``-based timeout."""
        timeout_seconds = self.config.timeout_seconds

        if timeout_seconds is None:
            return function(*args, **kwargs)

        timeout_seconds = int(timeout_seconds)

        # SIGALRM is POSIX-only.
        if sys.platform == "win32":
            return function(*args, **kwargs)

        function_name = getattr(function, "__name__", repr(function))
        timeout_message = f"Function {function_name} execution timed out after {timeout_seconds} seconds."

        def _handler(signum: int, frame: Any) -> None:
            raise TimeoutError(timeout_message)

        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout_seconds)
        try:
            result = function(*args, **kwargs)
        except TimeoutError as exc:
            raise Exception(timeout_message) from exc
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        return result

    def check_result_size(self, obj: Any) -> None:
        """Check if the string representation of an object exceeds the limit.

        Args:
            obj: The object to check.

        Raises:
            RuntimeError: If the object is too large.
        """
        limit = self.config.max_output_chars
        if limit is None:
            return

        # We use repr() as a proxy for size. This is not perfect but covers
        # the most common case of large JSON/text dumps.
        # sys.getsizeof is not recursive and thus misleading for containers.
        try:
            size = len(repr(obj))
        except Exception:
            # If repr fails, we can't check size easily. Assume it's fine or
            # let the subsequent serialization fail.
            return

        if size > limit:
            raise RuntimeError(
                f"Result object size ({size} chars) exceeds the limit of {limit} characters. "
                "Please return a smaller result."
            )
