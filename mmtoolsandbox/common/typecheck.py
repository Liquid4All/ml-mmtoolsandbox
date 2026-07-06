# Copyright © 2026 Apple Inc.

"""Typecheck utilities for common data types.

This file builds on top of typeguard.typechecked, and is meant to replace
mmtoolsandbox.common.validators in the long run. For now, we only implement checks for
built-in types and a small set of custom types. In the future, most type validations
will be built into typechecked.

Checker look up function custom_checker_lookup is exposed and registered to typeguard as an entry point.
See project.entry-points."typeguard.checker_lookup" in pyproject.toml, as well as
https://typeguard.readthedocs.io/en/stable/extending.html
https://python-poetry.org/docs/pyproject/#entry-points
"""

from __future__ import annotations

from inspect import isclass
from typing import Any, Type

from strenum import StrEnum
from typeguard import TypeCheckerCallable, TypeCheckError, TypeCheckMemo


def check_str_enum(
    value: Any, origin_type: Type[StrEnum], args: tuple[Any, ...], memo: TypeCheckMemo
) -> None:
    """When type checking for StrEnum, allow str as well.

    Args:
        value: The value to be type checked
        origin_type: The origin type
        args: The generic arguments from the annotation (empty tuple when the annotation was not parametrized)
        memo: The memo object (TypeCheckMemo)
    """
    if isinstance(value, StrEnum):
        return
    if isinstance(value, str) and value in list(origin_type):
        return
    raise TypeCheckError(
        f"{repr(value)} is not one of the allowed enums: {list(origin_type)}"
    )


def custom_checker_lookup(
    origin_type: Any, args: tuple[Any, ...], extras: tuple[Any, ...]
) -> TypeCheckerCallable | None:
    if isclass(origin_type) and issubclass(origin_type, StrEnum):
        return check_str_enum

    return None
