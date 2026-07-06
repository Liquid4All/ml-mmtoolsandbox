# Copyright © 2026 Apple Inc.

"""Tests for mmtoolsandbox.common.typecheck

Note that type check look up is already registered through entry point.
Integration testing can be achieved without importing from the module
"""

from enum import auto

import pytest
from strenum import StrEnum
from typeguard import TypeCheckError, typechecked


def test_type_check() -> None:
    class EnumClass(StrEnum):
        enum1 = auto()
        enum2 = auto()

    @typechecked
    def test_enum(value: EnumClass) -> None:
        pass

    test_enum(EnumClass.enum1)
    test_enum("enum1")  # type: ignore[arg-type]
    with pytest.raises(TypeCheckError):
        test_enum("enum3")  # type: ignore[arg-type]
