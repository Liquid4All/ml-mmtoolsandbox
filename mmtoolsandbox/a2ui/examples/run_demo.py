import argparse
import base64
import glob

# Import renderer and state directly via importlib to bypass the heavy
# a2ui/__init__.py (which pulls in execution_context → dill).
import importlib
import json
import os
import sys
import time
from typing import Any

# Add the project root to the python path.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, _PROJECT_ROOT)

_renderer = importlib.import_module("mmtoolsandbox.a2ui.renderer")
_state_mod = importlib.import_module("mmtoolsandbox.a2ui.state")
render_surface = _renderer.render_surface
get_ui_state = _state_mod.get_ui_state
UIState = _state_mod.UIState


def get_image_data_uri(image_filename: str) -> str:
    """Reads an image file and returns a Data URI."""
    image_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), image_filename)
    )
    try:
        with open(image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
            # Determine mime type based on extension
            ext = os.path.splitext(image_filename)[1].lower()
            mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
            return f"data:{mime_type};base64,{encoded_string}"
    except FileNotFoundError:
        print(f"Warning: Image file {image_filename} not found.")
        return ""


def process_images(obj: Any) -> Any:
    """Recursively find image URLs in the JSON object and replace local paths with Data URIs."""
    if isinstance(obj, dict):
        if obj.get("key") == "imageUrl" and "valueString" in obj:
            image_filename = obj["valueString"]
            # Check if it's a local file (not http/data) and not empty
            if (
                image_filename
                and not image_filename.startswith("http")
                and not image_filename.startswith("data:")
            ):
                data_uri = get_image_data_uri(image_filename)
                if data_uri:
                    obj["valueString"] = data_uri

        return {k: process_images(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [process_images(i) for i in obj]
    return obj


def render_json_file(json_path: str, output_dir: str | None = None) -> None:
    """Render a single JSON example file."""
    basename = os.path.splitext(os.path.basename(json_path))[0]
    print(f"\n--- Rendering: {basename} ---")

    try:
        with open(json_path, "r") as f:
            messages = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  Error loading {json_path}: {e}")
        return

    # Reset singleton state so files don't bleed into each other.
    UIState.reset()
    state = get_ui_state()
    render_count = 0

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for msg in messages:
        msg = process_images(msg)
        state.process_message(msg)

        if state.active_surface_id:
            surface = state.surfaces[state.active_surface_id]
            # Auto-detect root: if root_id is not set but a component with
            # id="root" exists, use it.  The numbered gallery examples follow
            # this convention without sending a beginRendering message.
            if surface.root_id is None and "root" in surface.components:
                surface.root_id = "root"
            image = render_surface(surface)
            if image:
                render_count += 1
                if output_dir:
                    out_path = os.path.join(
                        output_dir, f"{basename}_{render_count}.png"
                    )
                    image.save(out_path)
                    print(f"  Saved: {out_path}")
                else:
                    image.show()
                    time.sleep(1.0)

    if render_count == 0:
        print("  No surfaces rendered.")
    else:
        print(f"  Rendered {render_count} surface(s).")


def run_demo() -> None:
    parser = argparse.ArgumentParser(description="Run A2UI Demo")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a specific JSON example file to render.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Render all JSON examples in the examples directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Save rendered PNGs to this directory instead of displaying them.",
    )
    args = parser.parse_args()

    examples_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 50)
    print("A2UI Image Renderer Demo")
    print("=" * 50)

    if args.all:
        json_files = sorted(glob.glob(os.path.join(examples_dir, "*.json")))
        if not json_files:
            print("No JSON files found in examples directory.")
            return
        print(f"Found {len(json_files)} example(s).")
        for json_path in json_files:
            render_json_file(json_path, args.output_dir)
    elif args.file:
        json_path = os.path.abspath(args.file)
        render_json_file(json_path, args.output_dir)
    else:
        # Default: render booking_form.json
        default_path = os.path.join(examples_dir, "booking_form.json")
        render_json_file(default_path, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    run_demo()
