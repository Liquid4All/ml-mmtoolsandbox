# Copyright © 2026 Apple Inc.

"""Unit tests for mmtoolsandbox.tools.vision.images"""

import base64
import os
import shutil
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Iterator

import pytest
from PIL import Image

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution_context import (
    ExecutionContext,
    RoleType,
    get_current_context,
    new_context,
    set_current_context,
)
from mmtoolsandbox.common.image_id import ImageId
from mmtoolsandbox.common.message_conversion import Message
from mmtoolsandbox.common.safety_guard import SafetyGuardConfig
from mmtoolsandbox.roles.openai_user import OpenAIAPIUser
from mmtoolsandbox.tools.vision.common import load_image
from mmtoolsandbox.tools.vision.images import (
    crop_image,
    enhance_image,
    resize_image,
    rotate_image,
    save_image,
    show_image_to_user,
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


# Tests for crop_image
def test_crop_image(context_with_image: ExecutionContext) -> None:
    # Crop the center quarter of the 200x100 image
    # Normalized coords: left=250, upper=250, right=750, lower=750
    # Expected pixel coords: left=50, upper=25, right=150, lower=75
    result = crop_image(
        image_id=ImageId(0),
        left=250,
        upper=250,
        right=750,
        lower=750,
    )

    # Verify a new image was created
    assert result.image_id == ImageId(1)

    # Check the database state
    image_db = context_with_image.get_database(DatabaseNamespace.IMAGE)
    assert len(image_db) == 2  # Original image + cropped image
    assert image_db["image_id"][1] == 1
    assert image_db["image_content"][1] is not None

    # Load and verify the cropped image dimensions
    cropped_image = load_image(result.image_id)
    # Width: 150 - 50 = 100, Height: 75 - 25 = 50
    assert cropped_image.size == (100, 50)


# Tests for save_image
def test_save_image(context_with_image: ExecutionContext, tmp_path: Path) -> None:
    result = save_image(image_id=ImageId(0))

    # Verify the result contains a file path
    assert isinstance(result, str)
    assert "Image saved to:" in result
    assert ".jpg" in result or ".jpeg" in result.lower()


# Tests for resize_image
def test_resize_image(context_with_image: ExecutionContext) -> None:
    # Resize the 200x100 image to 100x50
    result = resize_image(image_id=ImageId(0), width=100, height=50)

    # Verify a new image was created
    assert result.image_id == ImageId(1)

    # Check the database state
    image_db = context_with_image.get_database(DatabaseNamespace.IMAGE)
    assert len(image_db) == 2  # Original image + resized image

    # Load and verify the resized image dimensions
    resized_image = load_image(result.image_id)
    assert resized_image.size == (100, 50)


# Tests for rotate_image
def test_rotate_image(context_with_image: ExecutionContext) -> None:
    # Rotate the 200x100 image by 90 degrees
    result = rotate_image(image_id=ImageId(0), angle=90, expand=True)

    # Verify a new image was created
    assert result.image_id == ImageId(1)

    # Check the database state
    image_db = context_with_image.get_database(DatabaseNamespace.IMAGE)
    assert len(image_db) == 2  # Original image + rotated image

    # Load and verify the rotated image dimensions
    # With expand=True, 200x100 rotated 90 degrees becomes 100x200
    rotated_image = load_image(result.image_id)
    assert rotated_image.size == (100, 200)


# Tests for enhance_image
def test_enhance_image(context_with_image: ExecutionContext) -> None:
    # Enhance the image contrast by factor 2.0
    result = enhance_image(image_id=ImageId(0), factor=2.0)

    # Verify a new image was created
    assert result.image_id == ImageId(1)

    # Check the database state
    image_db = context_with_image.get_database(DatabaseNamespace.IMAGE)
    assert len(image_db) == 2  # Original image + enhanced image

    # Load and verify the enhanced image dimensions (should be same as original)
    enhanced_image = load_image(result.image_id)
    assert enhanced_image.size == (200, 100)


# Tests for show_image_to_user
def test_show_image_to_user(context_with_image: ExecutionContext) -> None:
    image_id = ImageId(0)

    # Call the tool with default message
    result = show_image_to_user(image_id)
    assert result == "Image sent to user."

    # Verify message in database
    sandbox_db = context_with_image.get_database(
        DatabaseNamespace.SANDBOX, get_all_history_snapshots=True
    )
    assert len(sandbox_db) == 1
    message_row = sandbox_db.row(0, named=True)
    assert message_row["sender"] == RoleType.AGENT
    assert message_row["recipient"] == RoleType.USER
    assert message_row["content"] == "Here is the image"
    assert message_row["image_ids"] == [image_id]

    # Call the tool with custom message
    result = show_image_to_user(image_id, message="Check this out!")
    assert result == "Image sent to user."

    # Verify second message in database
    sandbox_db = context_with_image.get_database(
        DatabaseNamespace.SANDBOX, get_all_history_snapshots=True
    )
    assert len(sandbox_db) == 2
    message_row = sandbox_db.row(1, named=True)
    assert message_row["content"] == "Check this out!"
    assert message_row["image_ids"] == [image_id]

    # Verify OpenAIAPIUser conversion
    message = Message(
        sender=RoleType.AGENT,
        recipient=RoleType.USER,
        content="Here is the image",
        image_ids=[image_id],
    )

    openai_messages = OpenAIAPIUser.to_openai_messages([message])
    assert len(openai_messages) == 1
    assert openai_messages[0]["role"] == "user"
    content = openai_messages[0]["content"]
    assert isinstance(content, list)
    assert len(content) == 3  # text, text(image_id), image_url
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "Here is the image"
    assert content[2]["type"] == "image_url"
    # We don't check the exact base64 string here as it depends on the image content
    assert "data:image/jpeg;base64," in content[2]["image_url"]["url"]


# Tests for save_image with SafetyGuard
def test_save_image_with_safety_guard() -> None:
    # Create a safe directory for allowed writes
    safe_dir = tempfile.mkdtemp()
    allowed_write_paths = [safe_dir]

    # Configure SafetyGuard via ExecutionContext
    config = SafetyGuardConfig(
        allowed_write_paths=allowed_write_paths,
        enable_runtime_guard=True,
    )

    # Create a new context with the safety guard
    context = ExecutionContext(safety_guard_config=config)

    # Add a test image to the IMAGE database
    context.add_to_database(
        namespace=DatabaseNamespace.IMAGE,
        rows=[
            {
                "image_id": 0,
                "image_content": create_test_image(),
            }
        ],
    )

    # Set as current context
    set_current_context(context)
    guard = context.safety_guard

    try:
        # Enable safety guard
        if guard:
            guard.enable()

        # This should succeed now that save_image uses allowed_write_paths
        result_path_msg = save_image(ImageId(0))

        # Verify the file was created in the safe directory
        # Result format: "Image saved to: /path/to/file"
        path_str = result_path_msg.replace("Image saved to: ", "")
        assert path_str.startswith(safe_dir)
        assert os.path.exists(path_str)

    finally:
        # Clean up
        if guard:
            guard.disable()
        shutil.rmtree(safe_dir)
