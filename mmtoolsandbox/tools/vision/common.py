# Copyright © 2026 Apple Inc.

"""
A collection of core vision tools for image manipulation.
"""

import base64
from io import BytesIO

import polars as pl
from PIL import Image

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution_context import get_current_context
from mmtoolsandbox.common.image_id import ImageId, ImageResult

# Coordinate range for normalized coordinates
NORMALIZED_COORD_MIN = 0
NORMALIZED_COORD_MAX = 1000


def map_normalized_coordinate(coord: int, image_dimension: int) -> int:
    """Map a normalized coordinate in [0, 1000] to a pixel coordinate.

    Args:
        coord: Normalized coordinate value in the range [0, 1000].
        image_dimension: The corresponding image dimension (width or height) in pixels.

    Returns:
        The pixel coordinate in the range [0, image_dimension].

    Raises:
        ValueError: If coord is outside the range [0, 1000].
    """
    if not (NORMALIZED_COORD_MIN <= coord <= NORMALIZED_COORD_MAX):
        raise ValueError(
            f"Coordinate {coord} is outside the valid range "
            f"[{NORMALIZED_COORD_MIN}, {NORMALIZED_COORD_MAX}]"
        )
    return int(coord * image_dimension / NORMALIZED_COORD_MAX)


def map_normalized_point(
    point: tuple[int, int], image_size: tuple[int, int]
) -> tuple[int, int]:
    """Map normalized 2D point coordinates to pixel coordinates.

    Args:
        point: Tuple of (x, y) normalized coordinates in the range [0, 1000].
        image_size: Tuple of (width, height) in pixels.

    Returns:
        Tuple of (pixel_x, pixel_y) coordinates.

    Raises:
        ValueError: If coordinates are outside the range [0, 1000].
    """
    x, y = point
    width, height = image_size
    pixel_x = map_normalized_coordinate(x, width)
    pixel_y = map_normalized_coordinate(y, height)
    return pixel_x, pixel_y


def map_normalized_box(
    box: tuple[int, int, int, int], image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Map normalized box coordinates to pixel coordinates.

    Args:
        box: Tuple of (left, upper, right, lower) normalized coordinates in the range [0, 1000].
        image_size: Tuple of (width, height) in pixels.

    Returns:
        Tuple of (pixel_left, pixel_upper, pixel_right, pixel_lower) coordinates.

    Raises:
        ValueError: If coordinates are outside the range [0, 1000].
    """
    left, upper, right, lower = box
    width, height = image_size
    pixel_left = map_normalized_coordinate(left, width)
    pixel_upper = map_normalized_coordinate(upper, height)
    pixel_right = map_normalized_coordinate(right, width)
    pixel_lower = map_normalized_coordinate(lower, height)
    return pixel_left, pixel_upper, pixel_right, pixel_lower


def validate_image_ids(image_ids: list[ImageId]) -> None:
    """Validate that all image IDs exist in the IMAGE database.

    Args:
        image_ids: List of image IDs to validate.

    Raises:
        ValueError: If any image IDs are not found in the IMAGE database.
    """
    if not image_ids:
        return

    execution_context = get_current_context()
    image_db = execution_context.get_database(DatabaseNamespace.IMAGE)

    # Get all image IDs from the database
    existing_ids = set(image_db["image_id"].to_list())

    # Check which requested IDs are missing
    requested_ids = {int(img_id) for img_id in image_ids}
    missing_ids = requested_ids - existing_ids

    if missing_ids:
        raise ValueError(f"Image IDs not found in database: {sorted(missing_ids)}")


def load_image(image_id: ImageId) -> Image.Image:
    """Load an image from the IMAGE database by its ID.

    Args:
        image_id: The ID of the image to load.

    Returns:
        PIL Image object.

    Raises:
        ValueError: If the image ID is not found.
    """
    execution_context = get_current_context()

    # Load image from IMAGE database
    image_db = execution_context.get_database(DatabaseNamespace.IMAGE)
    image_row = image_db.filter(pl.col("image_id") == image_id)

    if image_row.is_empty():
        raise ValueError(f"Image with ID '{image_id}' not found")

    image_base64 = image_row["image_content"][0]

    # Decode base64 to image
    image_bytes = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_bytes))


def store_image(image: Image.Image) -> ImageResult:
    """Store an image in the execution context and return its new ID.

    Args:
        image: PIL Image object to store.

    Returns:
        ImageResult containing the new image ID.
    """
    execution_context = get_current_context()

    # Encode back to base64
    buffer = BytesIO()

    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    image.save(buffer, format="JPEG")
    cropped_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Add cropped image to execution context
    new_image_id = execution_context.add_image(cropped_base64)
    return ImageResult(new_image_id)
