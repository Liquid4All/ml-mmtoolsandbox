# Copyright © 2026 Apple Inc.

"""Image utilities for loading and processing images in scenarios."""

import base64
import io
import logging
import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Generator

from PIL import Image

LOGGER = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_FORMATS = {"PNG", "JPEG", "JPG", "GIF", "WEBP"}

# Maximum image size for API calls (in pixels)
MAX_IMAGE_SIZE = (2048, 2048)

# Maximum file size (in bytes) - 20MB
MAX_FILE_SIZE = 20 * 1024 * 1024


@contextmanager
def open_image(image_path: str) -> Generator[tuple[Image.Image, int], None, None]:
    """Context manager to open an image file with PIL.

    Args:
        image_path: Path to the image file

    Yields:
        tuple: (PIL.Image, file_size) - The opened PIL Image object and file size in bytes
    """
    with open(image_path, "rb") as f:
        file_size = os.fstat(f.fileno()).st_size
        if file_size > MAX_FILE_SIZE:
            LOGGER.warning(
                "Large image file: %s (%d bytes). Will resize before encoding.",
                image_path,
                file_size,
            )
        with Image.open(io.BytesIO(f.read())) as img:
            yield img, file_size


@lru_cache(maxsize=256)
def load_image_as_base64(image_path: str) -> str:
    """Load an image from disk and convert it to base64 format.

    Args:
        image_path: Path to the image file (relative to workspace or absolute)

    Returns:
        Base64 encoded image string

    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image format is not supported
        OSError: If the image file is corrupted or cannot be opened
    """
    try:
        with open_image(image_path) as (img, _):
            # Check format
            if img.format not in SUPPORTED_FORMATS:
                raise ValueError(
                    f"Unsupported image format: {img.format}. Supported: {SUPPORTED_FORMATS}"
                )

            # Resize if too large
            if img.size[0] > MAX_IMAGE_SIZE[0] or img.size[1] > MAX_IMAGE_SIZE[1]:
                LOGGER.info(
                    "Resizing image from %s to fit within %s", img.size, MAX_IMAGE_SIZE
                )
                img.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)

            # Convert to RGB if necessary (for JPEG compatibility)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # Save to bytes and encode as base64
            buffer = io.BytesIO()
            # Use JPEG for smaller file sizes in API calls
            img.save(buffer, format="JPEG", quality=85, optimize=True)
            buffer.seek(0)

            image_bytes = buffer.read()
            base64_string = base64.b64encode(image_bytes).decode("utf-8")

            LOGGER.debug(
                "Successfully loaded image: %s (%d bytes)", image_path, len(image_bytes)
            )
            return base64_string

    except Exception as e:
        LOGGER.error("Failed to load image %s: %s", image_path, e)
        raise


def get_image_info(image_path: str) -> dict[str, Any]:
    """Get basic information about an image file.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary with image information (format, size, dimensions)

    Raises:
        FileNotFoundError: If the image file doesn't exist
        OSError: If the image file is corrupted or cannot be opened
    """
    with open_image(image_path) as (img, file_size):
        return {
            "format": img.format,
            "mode": img.mode,
            "size": img.size,
            "width": img.size[0],
            "height": img.size[1],
            "file_size": file_size,
        }


def validate_image_path(image_path: str) -> bool:
    """Validate that an image path points to a supported image file.

    Args:
        image_path: Path to validate

    Returns:
        True if the path is valid and points to a supported image
    """
    try:
        get_image_info(image_path)
        return True
    except Exception:
        return False
