# Copyright © 2026 Apple Inc.

"""Tests for mmtoolsandbox.common.function_registry."""

import pytest

from mmtoolsandbox.common import evaluation
from mmtoolsandbox.common.function_registry import FunctionRegistry


def test_get_registered_function() -> None:
    func = FunctionRegistry.get("addition_similarity")
    assert func == evaluation.addition_similarity


def test_get_unregistered_function_raises() -> None:
    with pytest.raises(ValueError, match="not found in registry"):
        FunctionRegistry.get("non_existent_function")


def test_get_name_of_registered_function() -> None:
    name = FunctionRegistry.get_name(evaluation.addition_similarity)
    assert name == "addition_similarity"


def test_get_name_of_unregistered_function_raises() -> None:
    with pytest.raises(ValueError, match="not found in registry"):
        FunctionRegistry.get_name(lambda x: x)
