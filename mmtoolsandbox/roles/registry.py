# Copyright © 2026 Apple Inc.

from typing import TypeVar

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.roles.base_role import BaseRole

_registry: dict[str, type[BaseRole]] = {}

T = TypeVar("T", bound=BaseRole)


def register_role_class(cls: type[T]) -> type[T]:
    name = cls.__name__
    if name in _registry:
        raise ValueError(f"Role {name} already registered {_registry[name]=} vs {cls=}")
    _registry[cls.__name__] = cls
    return cls


def get_role_classes_by_role_type(role_type: RoleType) -> dict[str, type[BaseRole]]:
    return {name: cls for name, cls in _registry.items() if cls.role_type == role_type}


def get_role_class_by_name(role_name: str) -> type[BaseRole]:
    return _registry[role_name]
