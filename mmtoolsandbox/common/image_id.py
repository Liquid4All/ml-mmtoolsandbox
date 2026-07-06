# Copyright © 2026 Apple Inc.

"""Image identifier types and utilities.

Provides ``ImageId`` (a ``NewType`` over ``int``) and ``ImageResult``
(a wrapper returned by image-producing tools), plus recursive helpers
for detecting and extracting image IDs from nested structures.
"""

from __future__ import annotations

from typing import Any, NewType

ImageId = NewType("ImageId", int)
"""Opaque integer identifier for images stored in the IMAGE database."""


class ImageResult:
    """Wrapper returned by image-producing tools.

    Tool functions that create images return an ``ImageResult`` so the
    execution environment can detect image outputs and attach them to
    response messages.

    Attributes:
        image_id: The identifier of the produced image.
    """

    def __init__(self, image_id: ImageId) -> None:
        self._image_id = image_id

    @property
    def image_id(self) -> ImageId:
        """The identifier of the produced image."""
        return self._image_id

    def __repr__(self) -> str:
        return f"ImageResult(image_id={self._image_id})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self._image_id == other._image_id
        return False

    def __hash__(self) -> int:
        return hash(self._image_id)


def contains_images(obj: Any, visited: set[int] | None = None) -> bool:
    """Recursively check if the object contains any ImageResult.

    Args:
        obj: The object to check for ImageResult instances.
        visited: Set of already-visited object IDs to prevent infinite recursion.

    Returns:
        True if any ImageResult is found in the object tree.
    """
    if visited is None:
        visited = set()
    obj_id = id(obj)
    if obj_id in visited:
        return False
    visited.add(obj_id)
    if isinstance(obj, ImageResult):
        return True
    elif isinstance(obj, (list, tuple, set)):
        return any(contains_images(item, visited) for item in obj)
    elif isinstance(obj, dict):
        return any(contains_images(value, visited) for value in obj.values())
    return False


def extract_image_ids(obj: Any, visited: set[int] | None = None) -> list[ImageId]:
    """Recursively extract image IDs from a potentially nested structure.

    Args:
        obj: The object to extract ImageResult IDs from.
        visited: Set of already-visited object IDs to prevent infinite recursion.

    Returns:
        A list of ImageId values found in the object tree.
    """
    if visited is None:
        visited = set()
    obj_id = id(obj)
    if obj_id in visited:
        return []
    visited.add(obj_id)

    image_ids: list[ImageId] = []
    if isinstance(obj, ImageResult):
        image_ids.append(obj.image_id)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            image_ids.extend(extract_image_ids(item, visited))
    elif isinstance(obj, dict):
        for value in obj.values():
            image_ids.extend(extract_image_ids(value, visited))
    return image_ids
