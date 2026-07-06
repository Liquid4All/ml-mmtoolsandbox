# Copyright © 2026 Apple Inc.

"""Unit tests for mmtoolsandbox.tools.vision.common"""

import base64
from io import BytesIO
from typing import Iterator

import pytest
from PIL import Image

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution_context import (
    ExecutionContext,
    get_current_context,
    new_context,
)
from mmtoolsandbox.common.image_id import ImageId
from mmtoolsandbox.tools.vision.common import (
    NORMALIZED_COORD_MAX,
    NORMALIZED_COORD_MIN,
    load_image,
    map_normalized_box,
    map_normalized_coordinate,
    map_normalized_point,
    store_image,
)


def create_test_image(width: int = 200, height: int = 100, color: str = "red") -> str:
    """Create a test image and return it as base64 string.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        color: Color name

    Returns:
        Base64-encoded image string
    """
    img = Image.new("RGB", (width, height), color=color)
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@pytest.fixture
def context_with_image() -> Iterator[ExecutionContext]:
    """Context with an image in IMAGE database."""
    test_context = ExecutionContext()
    # Add a test image to the IMAGE database
    test_context.add_to_database(
        namespace=DatabaseNamespace.IMAGE,
        rows=[
            {
                "image_id": 0,
                "image_content": create_test_image(),
            }
        ],
    )
    with new_context(test_context):
        yield get_current_context()


# Tests for map_normalized_coordinate
def test_map_normalized_coordinate_valid() -> None:
    """Test mapping valid normalized coordinates to pixel coordinates."""
    # Test min value
    assert map_normalized_coordinate(NORMALIZED_COORD_MIN, 100) == 0
    # Test max value
    assert map_normalized_coordinate(NORMALIZED_COORD_MAX, 100) == 100
    # Test middle value
    assert map_normalized_coordinate(500, 100) == 50


def test_map_normalized_coordinate_invalid() -> None:
    """Test that invalid coordinates raise ValueError."""
    with pytest.raises(ValueError, match="outside the valid range"):
        map_normalized_coordinate(-1, 100)
    with pytest.raises(ValueError, match="outside the valid range"):
        map_normalized_coordinate(1001, 100)


# Tests for map_normalized_point
def test_map_normalized_point_valid() -> None:
    """Test mapping valid normalized points to pixel coordinates."""
    # Test with a 200x100 image
    image_size = (200, 100)

    # Test point
    assert map_normalized_point((500, 500), image_size) == (100, 50)

    # Test box
    assert map_normalized_box((250, 250, 750, 750), image_size) == (50, 25, 150, 75)


# Tests for load_image
def test_load_image(context_with_image: ExecutionContext) -> None:
    """Test loading an image from the IMAGE database."""
    image = load_image(ImageId(0))
    assert isinstance(image, Image.Image)
    assert image.size == (200, 100)


def test_load_image_not_found(context_with_image: ExecutionContext) -> None:
    """Test that loading a non-existent image raises ValueError."""
    with pytest.raises(ValueError, match="not found"):
        load_image(ImageId(999))


# Tests for store_image
def test_store_image(context_with_image: ExecutionContext) -> None:
    """Test storing an image to the execution context."""
    # Create a new image
    new_image = Image.new("RGB", (50, 50), color="green")

    # Store it
    result = store_image(new_image)

    # Verify it was added
    assert result.image_id == ImageId(1)

    # Check the database state
    image_db = context_with_image.get_database(DatabaseNamespace.IMAGE)
    assert len(image_db) == 2  # Original test image + new image
    assert image_db["image_id"][1] == 1
    assert image_db["image_content"][1] is not None

    # Verify it can be loaded back
    loaded_image = load_image(result.image_id)
    assert loaded_image.size == (50, 50)
