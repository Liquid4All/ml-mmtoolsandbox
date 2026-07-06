# Copyright © 2026 Apple Inc.

"""Unit tests for mmtoolsandbox.common.cached_embedding_service"""

import copy
import sys
from typing import Iterator

import numpy as np
import numpy.typing as npt
import pytest
from sentence_transformers import SentenceTransformer

from mmtoolsandbox.common.cached_embedding_service import (
    _EMBEDDING_CACHE,
    batched_embedding_cache,
    get_sentence_transformer,
)


def mock_embedding_service(queries: list[str]) -> dict[str, npt.NDArray[np.float64]]:
    """A mocked embedding service returning fake embeddings

    Args:
        queries:    A list of string to embed.

    Returns:
        Mocked embeddings
    """
    return {query: np.array(hash(query), dtype=np.float64) for query in queries}


@batched_embedding_cache
def decorated_mock_embedding_service(
    queries: list[str],
) -> dict[str, npt.NDArray[np.float64]]:
    """A mocked embedding service returning fake embeddings

    Args:
        queries:    A list of string to embed.

    Returns:
        Mocked embeddings
    """
    return mock_embedding_service(queries=queries)


@pytest.fixture(scope="function", autouse=True)
def clear_and_restore_cache() -> Iterator[None]:
    """Autouse fixture which clears and restores cache before each test function

    Returns:

    """
    cache = copy.deepcopy(_EMBEDDING_CACHE)
    _EMBEDDING_CACHE.clear()
    yield
    _EMBEDDING_CACHE.clear()
    _EMBEDDING_CACHE.update(cache)


def test_cache_decoration() -> None:
    # Initial cache is empty
    assert not _EMBEDDING_CACHE
    assert mock_embedding_service(["a"]) == decorated_mock_embedding_service(["a"])
    # This time there should be a hit and a miss
    assert _EMBEDDING_CACHE
    assert mock_embedding_service(["a", "b"]) == decorated_mock_embedding_service(
        ["a", "b"]
    )


@pytest.mark.skipif(
    not sys.platform.startswith("darwin"),
    reason="Test intended for local mac development execution only. "
    "Other location might not have access to Huggingface repo necessary for this test.",
)
def test_singleton_meta() -> None:
    model1: SentenceTransformer = get_sentence_transformer(
        "Alibaba-NLP/gte-multilingual-base",
        trust_remote_code=True,
        revision="7fc06782350c1a83f88b15dd4b38ef853d3b8503",
    )
    model2: SentenceTransformer = get_sentence_transformer(
        "Alibaba-NLP/gte-multilingual-base",
        trust_remote_code=True,
        revision="7fc06782350c1a83f88b15dd4b38ef853d3b8503",
    )
    assert id(model1) == id(model2)
