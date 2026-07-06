# Copyright © 2026 Apple Inc.

from __future__ import annotations

from typing import Any, Callable, Sequence

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.i18n import DefaultLocalizer, Localizer

InitialDatabaseStateFn = Callable[[Localizer], list[dict[str, Any]]]

_registry = {}


def register_initial_database_state(
    collections: Sequence[str] | str, namespace: DatabaseNamespace
) -> Callable[[InitialDatabaseStateFn], InitialDatabaseStateFn]:
    if isinstance(collections, str):
        collections = [collections]

    def decorator(func: InitialDatabaseStateFn) -> InitialDatabaseStateFn:
        for collection in collections:
            _registry[(collection, namespace)] = func
        return func

    return decorator


def get_initial_database_states(
    database_collection: str,
    namespace: DatabaseNamespace,
    localize: Localizer = DefaultLocalizer,
) -> list[dict[str, Any]]:
    return _registry[(database_collection, namespace)](localize)
