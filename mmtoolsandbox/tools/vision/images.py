# Copyright © 2026 Apple Inc.

"""
A collection of core vision tools for image manipulation.
"""

import tempfile
from pathlib import Path

from PIL import ImageDraw, ImageEnhance
from typeguard import typechecked

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution_context import RoleType, get_current_context
from mmtoolsandbox.common.image_id import ImageId, ImageResult
from mmtoolsandbox.common.message_conversion import (
    Message,
    add_messages_to_execution_context,
)
from mmtoolsandbox.common.utils import register_as_tool
from mmtoolsandbox.toolbox.names import ToolboxName
from mmtoolsandbox.tools.vision.common import (
    load_image,
    map_normalized_box,
    store_image,
)


@register_as_tool(
    toolboxes={ToolboxName.FULL},
    visible_to=(RoleType.AGENT,),
    database_namespaces={DatabaseNamespace.IMAGE},
)
@typechecked
def save_image(
    image_id: ImageId,
) -> str:
    """Save an image to a temporary file and return the file path.

    Args:
        image_id: The ID of the image to save, e.g. `ImageId(1)`

    Returns:
        The absolute path to the saved image file as a string.

    Raises:
        ValueError: If the image with the given ID is not found.
    """
    image = load_image(image_id)

    # Determine file extension based on image format
    image_format = image.format if image.format else "PNG"
    extension_map = {
        "JPEG": ".jpg",
        "PNG": ".png",
        "GIF": ".gif",
        "BMP": ".bmp",
        "TIFF": ".tiff",
        "WEBP": ".webp",
    }
    extension = extension_map.get(image_format, ".png")

    # Create a temporary file with the appropriate extension
    # delete=False ensures the file persists after closing
    context = get_current_context()
    safe_dir = None
    if context.safety_guard and context.safety_guard.config.allowed_write_paths:
        safe_dir = context.safety_guard.config.allowed_write_paths[0]

    with tempfile.NamedTemporaryFile(
        suffix=extension, delete=False, dir=safe_dir
    ) as tmp_file:
        image.save(tmp_file, format=image_format)
        tmp_path = tmp_file.name

    # Return the absolute path as a string
    return f"Image saved to: {str(Path(tmp_path).absolute())}"


@register_as_tool(
    toolboxes={ToolboxName.FULL},
    visible_to=(RoleType.AGENT,),
    database_namespaces={DatabaseNamespace.IMAGE},
)
@typechecked
def view_image(
    image_id: ImageId,
) -> ImageResult:
    """View an image for extracting visual information from it.

    Args:
        image_id: The ID of the image to view.

    Returns:
        The image with matching id
    """
    # Note: The execution environment will add the image to the tool call result message
    # so that it can be processed by the assistant in the following turn.
    return ImageResult(image_id)


@register_as_tool(
    toolboxes={ToolboxName.FULL},
    visible_to=(RoleType.AGENT,),
    database_namespaces={DatabaseNamespace.IMAGE, DatabaseNamespace.SANDBOX},
)
@typechecked
def show_image_to_user(image_id: ImageId, message: str = "Here is the image") -> str:
    """Show an image to the user with an optional message.

    This tool allows the agent to send an image to the user, accompanied by a text message.
    This is useful for presenting visual results, asking for feedback on an image, or
    simply sharing visual information.

    The image must already exist in the sandbox (referenced by `image_id`). The message
    will be added to the conversation history as a message from the AGENT to the USER,
    containing both the text and the image.

    Args:
        image_id: The ID of the image to show. This ID must correspond to an existing
                  image in the sandbox.
        message: An optional text message to accompany the image. Defaults to
                 "Here is the image".

    Returns:
        A confirmation string indicating that the image has been sent to the user.

    Raises:
        ValueError: If the image with the given ID is not found.
    """
    # Verify image exists
    load_image(image_id)

    # Create message
    msg = Message(
        sender=RoleType.AGENT,
        recipient=RoleType.USER,
        content=message,
        image_ids=[image_id],
    )

    # Add to context
    add_messages_to_execution_context(get_current_context(), [msg])

    return "Image sent to user."


@register_as_tool(
    toolboxes={ToolboxName.FULL},
    visible_to=(RoleType.AGENT,),
    database_namespaces={DatabaseNamespace.IMAGE},
)
@typechecked
def draw_bbox(
    image_id: ImageId,
    left: int,
    upper: int,
    right: int,
    lower: int,
    color: str = "red",
    width: int = 5,
) -> ImageResult:
    """Draw a red bounding box on an image and return the new image with the box drawn.

    This tool draws a red rectangle on the image at the specified coordinates.
    The original image is not modified - a new image is created with the bounding box.

    Args:
        image_id: The ID of the image to draw on
        left: The left (x minimum) coordinate of the box in range [0, 1000]
        upper: The upper (y minimum) coordinate of the box in range [0, 1000]
        right: The right (x maximum) coordinate of the box in range [0, 1000]
        lower: The lower (y maximum) coordinate of the box in range [0, 1000]
        color: The color of the bounding box, by default "red"
        width: The width of the bounding box, by default 5

    Returns:
        The ID of the newly created image with the red bounding box drawn

    Raises:
        ValueError: If the image with the given ID is not found or coordinates
            are outside the range [0, 1000]
    """
    image = load_image(image_id)

    # Create a copy to avoid modifying the original
    image_with_bbox = image.copy()

    # Map normalized coordinates to pixel coordinates
    pixel_coords = map_normalized_box((left, upper, right, lower), image.size)

    # Draw the bounding box
    draw = ImageDraw.Draw(image_with_bbox)
    draw.rectangle(pixel_coords, outline=color, width=width)

    # Store the new image and return its ID
    new_image_id = store_image(image_with_bbox)

    return new_image_id


@register_as_tool(
    toolboxes={ToolboxName.FULL},
    visible_to=(RoleType.AGENT,),
    database_namespaces={DatabaseNamespace.IMAGE},
)
@typechecked
def crop_image(
    image_id: ImageId, left: int, upper: int, right: int, lower: int
) -> ImageResult:
    """Crop an image to the specified bounding box.

    This tool crops the image to the specified region and returns it as a new image.
    The zoomed region maintains its original resolution without upscaling.

    Args:
        image_id: The ID of the image to crop
        left: The left coordinate of the crop region in range [0, 1000]
        upper: The upper coordinate of the crop region in range [0, 1000]
        right: The right coordinate of the crop region in range [0, 1000]
        lower: The lower coordinate of the crop region in range [0, 1000]

    Returns:
        The cropped image

    Raises:
        ValueError: If the image with the given ID is not found or coordinates
            are outside the range [0, 1000]
    """
    image = load_image(image_id)

    # Map normalized coordinates to pixel coordinates
    pixel_coords = map_normalized_box((left, upper, right, lower), image.size)

    # Crop the image
    cropped_img = image.crop(pixel_coords)

    image_result = store_image(cropped_img)

    # Note: The execution environment will add the image to the tool call result message
    # so that it can be processed by the assistant in the following turn.
    return image_result


@register_as_tool(
    toolboxes={ToolboxName.FULL},
    visible_to=(RoleType.AGENT,),
    database_namespaces={DatabaseNamespace.IMAGE},
)
@typechecked
def get_image_size(image_id: ImageId) -> tuple[int, int]:
    """Get the size of an image.

    Args:
        image_id: The ID of the image to get the size of.

    Returns:
        A tuple containing the (width, height) of the image in pixels.

    Raises:
        ValueError: If the image with the given ID is not found.
    """
    image = load_image(image_id)
    return image.size


@register_as_tool(
    toolboxes={ToolboxName.FULL},
    visible_to=(RoleType.AGENT,),
    database_namespaces={DatabaseNamespace.IMAGE},
)
@typechecked
def resize_image(image_id: ImageId, width: int, height: int) -> ImageResult:
    """Resize an image to the specified dimensions.

    Args:
        image_id: The ID of the image to resize.
        width: The new width of the image in pixels.
        height: The new height of the image in pixels.

    Returns:
        The ID of the newly created resized image.

    Raises:
        ValueError: If the image with the given ID is not found.
    """
    image = load_image(image_id)
    resized_image = image.resize((width, height))
    image_result = store_image(resized_image)
    return image_result


@register_as_tool(
    toolboxes={ToolboxName.FULL},
    visible_to=(RoleType.AGENT,),
    database_namespaces={DatabaseNamespace.IMAGE},
)
@typechecked
def rotate_image(image_id: ImageId, angle: float, expand: bool = False) -> ImageResult:
    """Rotate an image by a specified angle.

    Args:
        image_id: The ID of the image to rotate.
        angle: The angle of rotation in degrees counter-clockwise.
        expand: If True, expands the output image to include all of the rotated image.
            If False (default), the output image is the same size as the input image.

    Returns:
        The ID of the newly created rotated image.

    Raises:
        ValueError: If the image with the given ID is not found.
    """
    image = load_image(image_id)
    rotated_image = image.rotate(angle, expand=expand)
    image_result = store_image(rotated_image)
    return image_result


@register_as_tool(
    toolboxes={ToolboxName.FULL},
    visible_to=(RoleType.AGENT,),
    database_namespaces={DatabaseNamespace.IMAGE},
)
@typechecked
def enhance_image(image_id: ImageId, factor: float) -> ImageResult:
    """Enhance the contrast of an image.

    Args:
        image_id: The ID of the image to enhance.
        factor: A floating point value controlling the enhancement.
            Factor 1.0 always returns a copy of the original image,
            lower factors mean less contrast,
            and higher values more. There are no restrictions on this value.

    Returns:
        The ID of the newly created enhanced image.

    Raises:
        ValueError: If the image with the given ID is not found.
    """
    image = load_image(image_id)
    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(factor)
    image_result = store_image(enhanced_image)
    return image_result


def annotate_image() -> None:
    raise NotImplementedError
