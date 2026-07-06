# Copyright © 2026 Apple Inc.

"""Tests for image support functionality."""

import base64
from io import BytesIO
from pathlib import Path

import polars as pl
import pytest
from PIL import Image, ImageOps

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution_context import (
    ExecutionContext,
    RoleType,
    get_current_context,
    set_current_context,
)
from mmtoolsandbox.common.image_id import ImageId, ImageResult
from mmtoolsandbox.common.image_utils import (
    get_image_info,
    load_image_as_base64,
    validate_image_path,
)
from mmtoolsandbox.common.message_conversion import (
    Message,
    add_messages_to_execution_context,
)
from mmtoolsandbox.common.scenario import ScenarioExtension
from mmtoolsandbox.datasets.base_scenario import create_base_scenario
from mmtoolsandbox.toolbox.toolbox import create_empty_toolbox

# Path to test image
TEST_IMAGE_PATH = Path(__file__).parent / "test_data" / "gelato.jpg"


def create_test_image(
    path: Path, width: int = 100, height: int = 100, color: str = "red"
) -> None:
    """Create a simple test image with the specified dimensions and color.

    Args:
        path: Path where the image should be saved
        width: Image width in pixels
        height: Image height in pixels
        color: Color name (e.g., "red", "blue", "green")
    """
    img = Image.new("RGB", (width, height), color=color)
    img.save(path, "PNG")


class TestImageUtils:
    """Test image utility functions."""

    def test_load_image_as_base64(self, tmp_path: Path) -> None:
        """Test loading an image and converting to base64."""
        tmp_file = tmp_path / "test_image.png"
        create_test_image(tmp_file, color="red")

        # Test loading the image
        base64_data = load_image_as_base64(str(tmp_file))
        assert base64_data is not None
        assert isinstance(base64_data, str)
        assert len(base64_data) > 0

    def test_load_nonexistent_image(self) -> None:
        """Test loading a non-existent image raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_image_as_base64("nonexistent_image.png")

    def test_get_image_info(self, tmp_path: Path) -> None:
        """Test getting image information."""
        tmp_file = tmp_path / "test_image.png"
        create_test_image(tmp_file, width=200, height=150, color="blue")

        info = get_image_info(str(tmp_file))
        assert info["format"] == "PNG"
        assert info["width"] == 200
        assert info["height"] == 150
        assert info["size"] == (200, 150)
        assert info["file_size"] > 0

    def test_validate_image_path(self, tmp_path: Path) -> None:
        """Test image path validation."""
        tmp_file = tmp_path / "test_image.jpg"
        # Save as JPEG instead of PNG for this test
        img = Image.new("RGB", (50, 50), color="green")
        img.save(tmp_file, "JPEG")

        assert validate_image_path(str(tmp_file)) is True
        assert validate_image_path("nonexistent.png") is False


class TestExecutionContextWithImages:
    def test_add_images_to_execution_context(self, tmp_path: Path) -> None:
        """Test adding images to an execution context."""
        # Create temporary test images using pytest's tmp_path
        img_path1 = tmp_path / "test_image1.png"
        img_path2 = tmp_path / "test_image2.png"

        create_test_image(img_path1, color="red")
        create_test_image(img_path2, color="blue")

        # Create an execution context
        toolbox = create_empty_toolbox()
        execution_context = ExecutionContext(
            toolbox, delay_initialization=True, support_images=True
        )

        # Load images and add them to the IMAGE database
        image1_base64 = load_image_as_base64(str(img_path1))
        image2_base64 = load_image_as_base64(str(img_path2))

        assert image1_base64 is not None
        assert image2_base64 is not None

        # Add images to the execution context
        img_id1 = execution_context.add_image(image1_base64)
        img_id2 = execution_context.add_image(image2_base64)

        # Verify image IDs are generated correctly
        assert img_id1 == ImageId(0)
        assert img_id2 == ImageId(1)

        # Create a message with the image IDs
        message = Message(
            sender=RoleType.USER,
            recipient=RoleType.AGENT,
            content="Analyze these two images",
            image_ids=[img_id1, img_id2],
        )

        # Add message to the execution context
        add_messages_to_execution_context(execution_context, [message])

        # Verify the IMAGE database contains both images
        image_db = execution_context.get_database(DatabaseNamespace.IMAGE)
        assert len(image_db) == 2
        assert image_db["image_id"][0] == 0
        assert image_db["image_id"][1] == 1
        assert image_db["image_content"][0] == image1_base64
        assert image_db["image_content"][1] == image2_base64

        # Verify the message was stored with image IDs
        sandbox_db = execution_context.get_database(DatabaseNamespace.SANDBOX)
        assert len(sandbox_db) == 1
        assert sandbox_db["image_ids"][0].to_list() == [0, 1]
        assert sandbox_db["content"][0] == "Analyze these two images"


class TestScenarioWithImageManipulationTool:
    """Test tools that manipulate images stored in the execution context."""

    def test_invert_colors_tool(self, tmp_path: Path) -> None:
        """Test a tool that loads an image, inverts colors, and returns a new image ID."""

        # Define a tool that inverts image colors
        def invert_colors(image_id: ImageId) -> ImageResult:
            """Invert the colors of an image.

            Args:
                image_id: The ID of the image to invert

            Returns:
                The ID of the new inverted image
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
            img = Image.open(BytesIO(image_bytes))

            # Invert colors
            inverted_img = ImageOps.invert(img.convert("RGB"))

            # Encode back to base64
            buffer = BytesIO()
            inverted_img.save(buffer, format="PNG")
            inverted_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Add inverted image to execution context
            new_image_id = execution_context.add_image(inverted_base64)

            return ImageResult(new_image_id)

        toolbox = create_empty_toolbox()

        # Create test image - small red square
        img_path = tmp_path / "test_image.png"
        create_test_image(img_path, color="red")

        # Create base scenario with image support
        base_scenario = create_base_scenario(
            toolbox=toolbox,
            add_agent_system_prompt=False,
            support_images=True,
        )
        # Clear tool_allow_list since we don't need end_conversation for this test
        base_scenario.starting_context.tool_allow_list = None

        # Create ScenarioExtension with initial images and user message reference the image
        scenario_extension = ScenarioExtension(
            name="test_invert_colors",
            base_scenario=base_scenario,
            image_paths=[str(img_path)],
            messages=[
                {
                    "sender": RoleType.USER,
                    "recipient": RoleType.AGENT,
                    "content": "Please process this image",
                    "image_ids": [ImageId(0)],
                }
            ],
        )

        # Get the extended scenario - this loads images and creates the message
        scenarios = scenario_extension.get_extended_scenario()
        scenario = scenarios["test_invert_colors"]
        execution_context = scenario.starting_context

        # Prepare scenario for delayed initialization
        scenario.prepare(execution_context)

        # Verify the scenario loaded the image correctly
        image_db = execution_context.get_database(DatabaseNamespace.IMAGE)
        assert len(image_db) == 1
        original_id = ImageId(image_db["image_id"][0])
        assert original_id == ImageId(0)

        # Verify the user message was created with the image
        sandbox_db = execution_context.get_database(
            DatabaseNamespace.SANDBOX, drop_sandbox_message_index=False
        )
        assert len(sandbox_db) == 1
        assert sandbox_db["sender"][0] == RoleType.USER
        assert sandbox_db["recipient"][0] == RoleType.AGENT
        assert sandbox_db["image_ids"][0].to_list() == [0]
        # Base scenario adds a system message so the user message has index=1
        expected_user_msg_idx = 1
        assert sandbox_db["sandbox_message_index"][0] == expected_user_msg_idx

        # The inital image should be at the user msg idx
        image_db_with_index = execution_context.get_database(
            DatabaseNamespace.IMAGE, drop_sandbox_message_index=False
        )
        assert image_db_with_index["sandbox_message_index"][0] == expected_user_msg_idx

        # Set execution context as current for the tool to use
        set_current_context(execution_context)

        # Call the tool
        image_result = invert_colors(original_id)

        # Verify the tool created a new image id and entry in the image database
        assert image_result.image_id == ImageId(1)
        image_db = execution_context.get_database(
            DatabaseNamespace.IMAGE, drop_sandbox_message_index=False
        )
        assert len(image_db) == 2
        assert image_db["image_id"][1] == 1
        assert image_db["sandbox_message_index"][1] == expected_user_msg_idx + 1

        # Verify the inverted image content is different from the original
        assert image_db["image_content"][0] != image_db["image_content"][1]
