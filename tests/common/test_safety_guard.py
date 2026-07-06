# Copyright © 2026 Apple Inc.

"""Unit tests for mmtoolsandbox.common.safety_guard and code_parsing."""

from __future__ import annotations

import builtins
import os
import sys
import tempfile
from unittest.mock import patch

import pytest

from mmtoolsandbox.common.code_parsing import (
    parse_code_function_paths,
    parse_dunder_attr_access,
    parse_imports,
)
from mmtoolsandbox.common.safety_guard import (
    LimitedStringIO,
    SafetyGuard,
    SafetyGuardConfig,
)

# ===========================================================================
# Tests for code_parsing
# ===========================================================================


class TestParseImports:
    def test_import_statement(self) -> None:
        assert parse_imports("import os") == {"os"}

    def test_import_dotted(self) -> None:
        assert parse_imports("import os.path") == {"os"}

    def test_from_import(self) -> None:
        assert parse_imports("from os.path import join") == {"os"}

    def test_multiple_imports(self) -> None:
        code = "import json\nimport subprocess\nfrom os import remove"
        assert parse_imports(code) == {"json", "subprocess", "os"}

    def test_mmtoolsandbox_import(self) -> None:
        code = "from mmtoolsandbox.tools.code_execution import execute_code"
        assert parse_imports(code) == {"mmtoolsandbox"}

    def test_syntax_error_returns_empty(self) -> None:
        assert parse_imports("import (") == set()

    def test_no_imports(self) -> None:
        assert parse_imports("x = 1 + 2") == set()


class TestParseCodeFunctionPaths:
    def test_simple_call(self) -> None:
        paths = parse_code_function_paths("os.remove('file')")
        assert "os.remove" in paths

    def test_subprocess_run(self) -> None:
        paths = parse_code_function_paths("import subprocess; subprocess.run(['ls'])")
        assert "subprocess.run" in paths

    def test_chained_call(self) -> None:
        paths = parse_code_function_paths(
            "import pathlib; pathlib.Path('/tmp').unlink()"
        )
        assert "pathlib.Path.unlink" in paths

    def test_aliased_import(self) -> None:
        code = "import os as myos\nmyos.remove('file')"
        paths = parse_code_function_paths(code)
        assert "os.remove" in paths

    def test_builtin_open(self) -> None:
        paths = parse_code_function_paths("open('file', 'w')")
        assert "builtins.open" in paths

    def test_safe_code(self) -> None:
        code = 'import json\nx = json.dumps({"a": 1})'
        paths = parse_code_function_paths(code)
        assert "os.remove" not in paths
        assert "subprocess.run" not in paths

    def test_syntax_error_returns_empty(self) -> None:
        assert parse_code_function_paths("def (") == set()

    def test_builtin_compile(self) -> None:
        paths = parse_code_function_paths("compile('code', '<string>', 'exec')")
        assert "builtins.compile" in paths

    def test_builtin_exec(self) -> None:
        paths = parse_code_function_paths("exec('print(1)')")
        assert "builtins.exec" in paths

    def test_builtin_eval(self) -> None:
        paths = parse_code_function_paths("eval('1+1')")
        assert "builtins.eval" in paths


class TestParseDunderAttrAccess:
    def test_subclasses(self) -> None:
        assert "__subclasses__" in parse_dunder_attr_access("object.__subclasses__()")

    def test_globals(self) -> None:
        assert "__globals__" in parse_dunder_attr_access("func.__globals__")

    def test_code_attr(self) -> None:
        assert "__code__" in parse_dunder_attr_access("func.__code__")

    def test_builtins_attr(self) -> None:
        assert "__builtins__" in parse_dunder_attr_access("x = __builtins__")

    def test_import_dunder(self) -> None:
        assert "__import__" in parse_dunder_attr_access('__import__("os")')

    def test_safe_code_no_dunders(self) -> None:
        assert parse_dunder_attr_access('x = {"a": 1}\ny = x.get("a")') == set()

    def test_syntax_error(self) -> None:
        assert parse_dunder_attr_access("def (") == set()

    def test_safe_dunder_not_flagged(self) -> None:
        """__init__ and __name__ are not in the dangerous set."""
        assert (
            parse_dunder_attr_access("cls.__init__(self)\nprint(cls.__name__)") == set()
        )


# ===========================================================================
# Tests for SafetyGuard — Layer 1 (static syntax check)
# ===========================================================================


class TestSafetyGuardSyntaxCheck:
    @pytest.fixture()
    def guard(self) -> SafetyGuard:
        return SafetyGuard()

    def test_safe_code_passes(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe('import json\nx = json.dumps({"a": 1})')
        assert is_safe is True
        assert msg == ""

    def test_mmtoolsandbox_import_passes(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe("from mmtoolsandbox.tools import some_tool")
        assert is_safe is True

    def test_disallowed_import_blocked(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe("import subprocess")
        assert is_safe is False
        assert "subprocess" in msg

    def test_disallowed_import_shutil(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe("import shutil")
        assert is_safe is False
        assert "shutil" in msg

    def test_disallowed_function_call_blocked(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe("os.remove('important_file')")
        assert is_safe is False
        assert "os.remove" in msg

    def test_disallowed_subprocess_run(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe("import subprocess\nsubprocess.run(['ls'])")
        assert is_safe is False
        assert "subprocess" in msg

    def test_tool_call_code_passes(self, guard: SafetyGuard) -> None:
        code = (
            'call_123_parameters = {"key": "value"}\n'
            "call_123_response = some_tool(**call_123_parameters)"
        )
        is_safe, msg = guard.is_syntax_safe(code)
        assert is_safe is True

    def test_syntax_error_code_passes(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe("def (")
        assert is_safe is True

    def test_multiple_allowed_imports(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe(
            "import json\nimport math\nimport re\nimport datetime"
        )
        assert is_safe is True

    def test_compile_blocked(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe("compile('x', '<s>', 'exec')")
        assert is_safe is False
        assert "builtins.compile" in msg

    def test_exec_blocked(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe("exec('print(1)')")
        assert is_safe is False
        assert "builtins.exec" in msg

    def test_eval_blocked(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe("eval('1+1')")
        assert is_safe is False
        assert "builtins.eval" in msg

    def test_dunder_import_blocked(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe('__import__("os")')
        assert is_safe is False
        assert "__import__" in msg

    def test_marshal_import_blocked(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe("import marshal")
        assert is_safe is False
        assert "marshal" in msg

    def test_types_import_blocked(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe("import types")
        assert is_safe is False
        assert "types" in msg

    def test_dis_import_blocked(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe("import dis")
        assert is_safe is False
        assert "dis" in msg

    def test_inspect_import_blocked(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe("import inspect")
        assert is_safe is False
        assert "inspect" in msg

    def test_os_listdir_blocked(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe("os.listdir('/')")
        assert is_safe is False
        assert "os.listdir" in msg

    def test_os_getcwd_blocked(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe("os.getcwd()")
        assert is_safe is False
        assert "os.getcwd" in msg

    def test_subclasses_blocked(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe("object.__subclasses__()")
        assert is_safe is False
        assert "__subclasses__" in msg

    def test_globals_blocked(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe("func.__globals__")
        assert is_safe is False
        assert "__globals__" in msg

    def test_code_attr_blocked(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe("func.__code__")
        assert is_safe is False
        assert "__code__" in msg

    def test_matplotlib_import_allowed(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe("import matplotlib.pyplot as plt")
        assert is_safe is True

    def test_numpy_import_allowed(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe("import numpy as np")
        assert is_safe is True


# ===========================================================================
# Tests for SafetyGuard — Layer 2 (runtime guards)
# ===========================================================================


class TestSafetyGuardRuntime:
    @pytest.fixture()
    def guard(self) -> SafetyGuard:
        return SafetyGuard()

    # --- Path-restricted open() ---

    def test_open_read_allowed(self, guard: SafetyGuard) -> None:
        guard.enable()
        try:
            f = builtins.open(__file__, "r")
            f.close()
        finally:
            guard.disable()

    def test_open_write_blocked_by_default(self, guard: SafetyGuard) -> None:
        guard.enable()
        try:
            with pytest.raises(PermissionError, match="disabled"):
                builtins.open("/tmp/test_safety_guard_write", "w")
        finally:
            guard.disable()

    def test_write_allowed_in_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SafetyGuardConfig(allowed_write_paths=[tmpdir])
            guard = SafetyGuard(config)
            guard.enable()
            try:
                target = os.path.join(tmpdir, "test.txt")
                f = builtins.open(target, "w")
                f.write("hello")
                f.close()
                f = builtins.open(target, "r")
                assert f.read() == "hello"
                f.close()
            finally:
                guard.disable()

    def test_write_blocked_outside_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SafetyGuardConfig(allowed_write_paths=[tmpdir])
            guard = SafetyGuard(config)
            guard.enable()
            try:
                with pytest.raises(PermissionError, match="not allowed"):
                    builtins.open("/tmp/should_not_exist_safety_test", "w")
            finally:
                guard.disable()

    def test_read_path_restriction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            allowed_file = os.path.join(tmpdir, "ok.txt")
            with open(allowed_file, "w") as f:
                f.write("hello")

            config = SafetyGuardConfig(allowed_read_paths=[tmpdir])
            guard = SafetyGuard(config)
            guard.enable()
            try:
                f = builtins.open(allowed_file, "r")
                assert f.read() == "hello"
                f.close()

                with pytest.raises(PermissionError, match="not allowed"):
                    builtins.open("/etc/hostname", "r")
            finally:
                guard.disable()

    def test_disable_restores_open(self, guard: SafetyGuard) -> None:
        original_open = builtins.open
        guard.enable()
        guard.disable()
        assert builtins.open is original_open

    # --- SystemExit swap ---

    def test_system_exit_swapped(self, guard: SafetyGuard) -> None:
        guard.enable()
        try:
            assert builtins.SystemExit is Exception  # type: ignore[comparison-overlap]
        finally:
            guard.disable()

    def test_system_exit_restored(self, guard: SafetyGuard) -> None:
        guard.enable()
        guard.disable()
        assert builtins.SystemExit is not Exception  # type: ignore[comparison-overlap]

    # --- sys.modules proxy ---

    def test_sys_modules_blocks_subprocess(self, guard: SafetyGuard) -> None:
        guard.enable()
        try:
            with pytest.raises(PermissionError, match="subprocess"):
                sys.modules["subprocess"]
        finally:
            guard.disable()

    def test_sys_modules_blocks_shutil(self, guard: SafetyGuard) -> None:
        guard.enable()
        try:
            with pytest.raises(PermissionError, match="shutil"):
                sys.modules["shutil"]
        finally:
            guard.disable()

    def test_sys_modules_blocks_ctypes(self, guard: SafetyGuard) -> None:
        guard.enable()
        try:
            with pytest.raises(PermissionError, match="ctypes"):
                sys.modules["ctypes"]
        finally:
            guard.disable()

    def test_sys_modules_allows_json(self, guard: SafetyGuard) -> None:
        guard.enable()
        try:
            mod = sys.modules.get("json")
            assert mod is not None or mod is None  # just check no PermissionError
        finally:
            guard.disable()

    def test_sys_modules_restored(self, guard: SafetyGuard) -> None:
        original = sys.modules
        guard.enable()
        guard.disable()
        assert sys.modules is original


# ===========================================================================
# Tests for timeout_call
# ===========================================================================


class TestTimeoutCall:
    @pytest.fixture()
    def guard(self) -> SafetyGuard:
        return SafetyGuard()

    def test_no_timeout(self, guard: SafetyGuard) -> None:
        guard.config.timeout_seconds = None
        assert guard._timeout_call(lambda: 42) == 42

    def test_function_completes_in_time(self, guard: SafetyGuard) -> None:
        guard.config.timeout_seconds = 5
        assert guard._timeout_call(lambda: 42) == 42

    @pytest.mark.skipif(sys.platform == "win32", reason="SIGALRM is POSIX only")
    def test_timeout_fires(self, guard: SafetyGuard) -> None:
        def infinite_loop() -> None:
            while True:
                pass

        guard.config.timeout_seconds = 1
        with pytest.raises(Exception, match="timed out"):
            guard._timeout_call(infinite_loop)

    def test_timeout_with_args(self, guard: SafetyGuard) -> None:
        guard.config.timeout_seconds = None
        assert guard._timeout_call(lambda a, b: a + b, 3, 7) == 10


# ===========================================================================
# Tests for SafetyGuardConfig
# ===========================================================================


class TestSafetyGuardConfig:
    def test_default_enabled(self) -> None:
        config = SafetyGuardConfig()
        assert config.enable_syntax_check is True
        assert config.enable_runtime_guard is True
        assert config.timeout_seconds == 100
        assert config.allowed_read_paths is None
        assert config.allowed_write_paths is None

    def test_custom_config(self) -> None:
        config = SafetyGuardConfig(
            enable_syntax_check=False,
            enable_runtime_guard=False,
            timeout_seconds=None,
        )
        assert config.enable_syntax_check is False
        assert config.enable_runtime_guard is False
        assert config.timeout_seconds is None

    def test_guard_with_none_config_uses_defaults(self) -> None:
        guard = SafetyGuard(config=None)
        assert guard.config.enable_syntax_check is True

    def test_guard_disabled_passes_everything(self) -> None:
        config = SafetyGuardConfig(
            enable_syntax_check=True,
            allowed_modules={"subprocess", "os", "shutil"},
            disallowed_function_paths=set(),
        )
        guard = SafetyGuard(config)
        is_safe, msg = guard.is_syntax_safe("import subprocess; subprocess.run(['ls'])")
        assert is_safe is True

    def test_allowed_read_paths_config(self) -> None:
        config = SafetyGuardConfig(allowed_read_paths=["/tmp", "/data"])
        assert config.allowed_read_paths == ["/tmp", "/data"]

    def test_allowed_write_paths_config(self) -> None:
        config = SafetyGuardConfig(allowed_write_paths=["/output"])
        assert config.allowed_write_paths == ["/output"]


# ===========================================================================
# Tests for Output Size Limits
# ===========================================================================


class TestLimitedStringIO:
    def test_write_within_limit(self) -> None:
        s = LimitedStringIO(limit=10)
        s.write("hello")
        assert s.getvalue() == "hello"

    def test_write_exceeds_limit(self) -> None:
        s = LimitedStringIO(limit=5)
        with pytest.raises(RuntimeError, match="Output limit"):
            s.write("hello world")

    def test_incremental_write_exceeds_limit(self) -> None:
        s = LimitedStringIO(limit=5)
        s.write("hi")
        with pytest.raises(RuntimeError, match="Output limit"):
            s.write(" world")

    def test_no_limit(self) -> None:
        s = LimitedStringIO(limit=None)
        s.write("hello world" * 100)
        assert len(s.getvalue()) > 0


class TestCheckObjectSize:
    @pytest.fixture()
    def guard(self) -> SafetyGuard:
        return SafetyGuard()

    def test_small_object_passes(self, guard: SafetyGuard) -> None:
        guard.config.max_output_chars = 100
        guard.check_result_size("hello")

    def test_large_object_fails(self, guard: SafetyGuard) -> None:
        guard.config.max_output_chars = 5
        with pytest.raises(RuntimeError, match="Result object size"):
            guard.check_result_size("hello world")

    def test_no_limit_passes(self, guard: SafetyGuard) -> None:
        guard.config.max_output_chars = None
        guard.check_result_size("hello world" * 100)

    def test_repr_failure_ignored(self, guard: SafetyGuard) -> None:
        class BadRepr:
            def __repr__(self) -> str:
                raise ValueError("Bad repr")

        # Should not raise RuntimeError
        guard.config.max_output_chars = 10
        guard.check_result_size(BadRepr())


# ===========================================================================
# Tests for Memory Limit
# ===========================================================================


class TestMemoryLimit:
    def test_memory_limit_set_on_enable(self) -> None:
        with patch("mmtoolsandbox.common.safety_guard.resource") as mock_resource:
            # Mock getrlimit to return some values
            # Hard limit needs to be larger than requested limit (10MB) for this test
            mock_resource.getrlimit.return_value = (1000000, 20000000)
            mock_resource.RLIMIT_AS = 1
            mock_resource.RLIM_INFINITY = -1

            config = SafetyGuardConfig(memory_limit_mb=10)
            guard = SafetyGuard(config)

            guard.enable()

            # Check if setrlimit was called with correct values
            # 10MB = 10 * 1024 * 1024 = 10485760 bytes
            mock_resource.setrlimit.assert_called_with(1, (10485760, 20000000))

            guard.disable()
            # Check if restored
            mock_resource.setrlimit.assert_called_with(1, (1000000, 20000000))

    def test_memory_limit_respects_hard_limit(self) -> None:
        with patch("mmtoolsandbox.common.safety_guard.resource") as mock_resource:
            # Hard limit is small (1MB)
            mock_resource.getrlimit.return_value = (500000, 1000000)
            mock_resource.RLIMIT_AS = 1
            mock_resource.RLIM_INFINITY = -1

            # Try to set 10MB limit
            config = SafetyGuardConfig(memory_limit_mb=10)
            guard = SafetyGuard(config)

            guard.enable()

            # Should be capped at hard limit (1000000)
            mock_resource.setrlimit.assert_called_with(1, (1000000, 1000000))

            guard.disable()

    def test_memory_limit_ignored_if_resource_missing(self) -> None:
        with patch("mmtoolsandbox.common.safety_guard.resource", None):
            config = SafetyGuardConfig(memory_limit_mb=10)
            guard = SafetyGuard(config)
            # Should not raise exception
            guard.enable()
            guard.disable()


# ===========================================================================
# Tests for Advanced Safety Scenarios
# ===========================================================================


class TestSafetyGuardAdvanced:
    @pytest.fixture()
    def guard(self) -> SafetyGuard:
        return SafetyGuard()

    # --- Static Analysis Bypass ---

    def test_importlib_blocked(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe("import importlib")
        assert is_safe is False
        assert "importlib" in msg

    def test_getattr_blocked(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe("getattr(object, '__subclasses__')")
        assert is_safe is False
        assert "getattr" in msg

    def test_setattr_blocked(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe("setattr(object, 'x', 1)")
        assert is_safe is False
        assert "setattr" in msg

    def test_delattr_blocked(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe("delattr(object, 'x')")
        assert is_safe is False
        assert "delattr" in msg

    def test_vars_blocked(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe("vars(object)")
        assert is_safe is False
        assert "vars" in msg

    def test_globals_blocked_function(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe("globals()")
        assert is_safe is False
        assert "globals" in msg

    def test_attribute_traversal_blocked(self, guard: SafetyGuard) -> None:
        # Accessing __dict__ is blocked by dunder check
        is_safe, msg = guard.is_syntax_safe("import math; math.__dict__['__class__']")
        assert is_safe is False
        assert "__dict__" in msg

    # --- Runtime Monkey-Patch Bypass ---

    def test_restore_open_blocked(self, guard: SafetyGuard) -> None:
        # Accessing __dict__ on builtins is blocked statically
        is_safe, msg = guard.is_syntax_safe(
            "import builtins; builtins.open = builtins.__dict__['open']"
        )
        assert is_safe is False
        assert "__dict__" in msg

    def test_subclasses_traversal_blocked(self, guard: SafetyGuard) -> None:
        # Accessing __subclasses__ is blocked statically
        is_safe, msg = guard.is_syntax_safe("object.__subclasses__()")
        assert is_safe is False
        assert "__subclasses__" in msg

    # --- Reflection and Introspection ---

    def test_inspect_blocked(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe("import inspect")
        assert is_safe is False
        assert "inspect" in msg

    # --- File System Boundary ---

    def test_path_traversal_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SafetyGuardConfig(allowed_write_paths=[tmpdir])
            guard = SafetyGuard(config)
            guard.enable()
            try:
                # Try to write outside via traversal
                # Note: realpath resolves '..', so this checks if resolved path starts with allowed path
                target = os.path.join(tmpdir, "..", "outside.txt")
                with pytest.raises(PermissionError, match="not allowed"):
                    builtins.open(target, "w")
            finally:
                guard.disable()

    def test_absolute_path_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SafetyGuardConfig(allowed_write_paths=[tmpdir])
            guard = SafetyGuard(config)
            guard.enable()
            try:
                with pytest.raises(PermissionError, match="not allowed"):
                    builtins.open("/etc/passwd", "w")
            finally:
                guard.disable()

    # --- sys.modules Guard ---

    def test_sys_modules_os_popen_blocked(self, guard: SafetyGuard) -> None:
        # Even if they get os module, popen should be blocked statically
        is_safe, msg = guard.is_syntax_safe("import os; os.popen('ls')")
        assert is_safe is False
        assert "os.popen" in msg

    def test_sys_modules_os_spawn_blocked(self, guard: SafetyGuard) -> None:
        is_safe, msg = guard.is_syntax_safe("import os; os.spawnl(os.P_WAIT, 'ls')")
        assert is_safe is False
        assert "os.spawnl" in msg
