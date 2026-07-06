# Copyright © 2026 Apple Inc.

from enum import auto

from strenum import StrEnum


class DatasetName(StrEnum):
    """This enum defines the available datasets."""

    FULL = auto()
    # Consolidated toolbox (~300 tools): merges symmetric pairs, CRUD groups,
    # and cross-app duplicates from FULL into fewer, coarser tools.
    MEDIUM = auto()
    # Further consolidated toolbox (~150 tools): merges search/list tools by
    # entity type, clusters functional domains, and collapses entity subtypes.
    COMPACT = auto()
    # Workflow-based toolbox (~35 tools): each app gets 1-3 tools organized by
    # user intent (discover, transact, manage). Fits entirely in LLM context
    # without tool search.
    MINI = auto()
