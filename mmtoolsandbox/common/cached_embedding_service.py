# Copyright © 2026 Apple Inc.

"""This file hosts functions calling into web hosted embedding services.

These functions generally accepts a list of strings, and maintains a cache in process memory using a global variable.
One can think of this process as lazily building a search index.

Performance benchmarking on embedding 1k short queries without caching:

batched_openai_003_small_embedding:                                 5s
batched_all_gte_multilingual_base_embedding (M2 Ultra Local CPU):   3s
"""

import logging
import os
from collections import defaultdict
from functools import wraps
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Disable tokenizer parallelism

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# An in memory embedding cache. First level key is the embedding function name, second level key is the
# string embedded.
_EMBEDDING_CACHE: dict[str, dict[str, npt.NDArray[np.float64]]] = defaultdict(dict)


class BatchedEmbeddingFunction(Protocol):
    """Callable Protocol for batched embedding function"""

    # User defined callable will always have a __name__
    __name__: str

    def __call__(self, queries: list[str]) -> dict[str, npt.NDArray[np.float64]]:
        """A batched embedding function.

        Args:
            queries:    A list of queries to be embedded.

        Returns:
            A dictionary of query to embedding
        """


# Cache of model_name → SentenceTransformer instance (singleton per model).
_sentence_transformer_instances: dict[str, SentenceTransformer] = {}


def get_sentence_transformer(
    model_name_or_path: str, **kwargs: Any
) -> SentenceTransformer:
    """Get or create a singleton SentenceTransformer for the given model name.

    Forces CPU device to avoid MPS (Metal Performance Shaders) crashes on Apple Silicon
    with certain torch/sentence-transformers version combinations.
    """
    if model_name_or_path not in _sentence_transformer_instances:
        _sentence_transformer_instances[model_name_or_path] = SentenceTransformer(
            model_name_or_path, device="cpu", **kwargs
        )
    return _sentence_transformer_instances[model_name_or_path]


def batched_embedding_cache(
    batched_embedding_function: BatchedEmbeddingFunction,
) -> BatchedEmbeddingFunction:
    """Decorator function implementing cache for batched embedding functions.

    Retrieve embeddings found in _EMBEDDING_CACHE, call batched_embedding_function for the rest.

    Args:
        batched_embedding_function:     Embedding function to call in case of cache miss.

    Returns:
        Decorated batched embedding function
    """

    @wraps(batched_embedding_function)
    def cached_batched_embedding_function(
        queries: list[str],
    ) -> dict[str, npt.NDArray[np.float64]]:
        # Fetch embedding function name for embedding cache lookup
        cache = _EMBEDDING_CACHE[batched_embedding_function.__name__]
        cache_hit = {query: cache[query] for query in queries if query in cache}
        remaining_queries = [query for query in queries if query not in cache]
        cache_miss: dict[str, npt.NDArray[np.float64]] = {}
        if remaining_queries:
            cache_miss = batched_embedding_function(queries=remaining_queries)
            # Add to cache
            _EMBEDDING_CACHE[batched_embedding_function.__name__] = {
                **_EMBEDDING_CACHE[batched_embedding_function.__name__],
                **cache_miss,
            }
        return {**cache_hit, **cache_miss}

    return cached_batched_embedding_function


@batched_embedding_cache
def batched_openai_003_small_embedding(
    queries: list[str],
) -> dict[str, npt.NDArray[np.float64]]:
    """Generates embeddings with OpenAI text-embedding-3-small embedding service.

    NOTE: This implementation has not cleared legal approval yet. We are still discussing with legal team.

    Args:
        queries:    A list of queries to be embedded.

    Returns:
        A dictionary of query to embedding
    """
    client = OpenAI()
    response = client.embeddings.create(input=queries, model="text-embedding-3-small")
    return {
        query: np.array(data.embedding, dtype=np.float64)
        for query, data in zip(queries, response.data)
    }


@batched_embedding_cache
def batched_openai_003_large_embedding(
    queries: list[str],
) -> dict[str, npt.NDArray[np.float64]]:
    """Generates embeddings with OpenAI text-embedding-3-large embedding service.

    NOTE: This implementation has not cleared legal approval yet. We are still discussing with legal team.

    Args:
        queries:    A list of queries to be embedded.

    Returns:
        A dictionary of query to embedding
    """
    client = OpenAI()
    response = client.embeddings.create(input=queries, model="text-embedding-3-large")
    return {
        query: np.array(data.embedding, dtype=np.float64)
        for query, data in zip(queries, response.data)
    }


@batched_embedding_cache
def batched_gte_multilingual_base_embedding(
    queries: list[str],
) -> dict[str, npt.NDArray[np.float64]]:
    """Generates embeddings with Alibaba-NLP/gte-multilingual-base embedding model.

    Executes embedding locally

    Args:
        queries:    A list of queries to be embedded.

    Returns:
        A dictionary of query to embedding
    """
    if not queries:
        return {}
    # Disable logging
    previous_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    model: SentenceTransformer = get_sentence_transformer(
        "Alibaba-NLP/gte-multilingual-base",
        trust_remote_code=True,
        revision="7fc06782350c1a83f88b15dd4b38ef853d3b8503",
    )
    embeddings = model.encode(queries, show_progress_bar=False)
    # Reset logging
    logging.disable(previous_level)
    return {
        query: np.array(embedding, dtype=np.float64)
        for query, embedding in zip(queries, embeddings)
    }
