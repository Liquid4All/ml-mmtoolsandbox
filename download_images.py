"""Download and process original images."""

import base64
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image as PILImage
from tqdm import tqdm

DATASET_NAMES = [
    "omnidocbench",
    "hiertext",
    "screenspotpro",
    "docvqa",
    "infographicvqa",
    "chartmuseum",
    "chartqapro",
    "worldvqa",
]


def process_hiertext(output_dir: str) -> None:
    """Download HierText OCR test split from Open Images S3 and organize files.

    Original data stored at: https://github.com/google-research-datasets/hiertext?tab=readme-ov-file

    Args:
        output_dir: Base directory to save processed data.
    """
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    hiertext_dir = base_dir / "hiertext"
    hiertext_dir.mkdir(parents=True, exist_ok=True)

    tgz_path = hiertext_dir / "test.tgz"
    images_dir = hiertext_dir / "images"

    print("Downloading HierText test.tgz from S3...")
    subprocess.run(
        [
            "aws",
            "s3",
            "--no-sign-request",
            "cp",
            "s3://open-images-dataset/ocr/test.tgz",
            str(tgz_path),
        ],
        check=True,
    )

    print("Extracting test.tgz...")
    subprocess.run(
        ["tar", "-xzvf", str(tgz_path), "-C", hiertext_dir],
        check=True,
    )

    print("Creating target directory...")
    images_dir.mkdir(parents=True, exist_ok=True)

    extracted_dir = hiertext_dir / "test"
    if not extracted_dir.exists():
        raise RuntimeError("Expected 'test/' directory not found after extraction")

    target = images_dir / extracted_dir.name
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(extracted_dir), images_dir)
    print(f"HierText images saved to: {images_dir}")


def process_omnidocbench(output_dir: str, exclude_cn: bool = True) -> None:
    """Download OmniDocBench dataset and organize images.

    Args:
        output_dir: Base directory to save processed data.
        exclude_cn: If True, exclude Chinese and mixed language samples.
    """
    base_dir = Path(output_dir)
    images_dir = base_dir / "omnidocbench" / "images"
    metadata_path = base_dir / "omnidocbench" / "OmniDocBench.json"
    images_dir.mkdir(parents=True, exist_ok=True)
    local_repo_dir = None

    print("Downloading OmniDocBench dataset from Hugging Face...")
    repo_id = "opendatalab/OmniDocBench"
    local_repo_dir = snapshot_download(repo_id=repo_id, repo_type="dataset")
    print(f"Dataset downloaded to: {local_repo_dir}")

    if metadata_path.exists() or (
        local_repo_dir and (Path(local_repo_dir) / "OmniDocBench.json").exists()
    ):
        if local_repo_dir:
            downloaded_metadata_path = Path(local_repo_dir) / "OmniDocBench.json"
            if downloaded_metadata_path.exists():
                metadata_path = downloaded_metadata_path
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if exclude_cn:
            print("Filtering out Chinese and mixed language samples...")
            filtered_metadata = []
            excluded_count = 0
            for entry in metadata:
                if "page_info" in entry and "page_attribute" in entry["page_info"]:
                    page_attr = entry["page_info"]["page_attribute"]
                    if "language" in page_attr:
                        language = page_attr["language"].lower()
                        if "chinese" in language or "en_ch_mixed" in language:
                            excluded_count += 1
                            continue
                filtered_metadata.append(entry)
            print(
                f"Excluded {excluded_count} Chinese/mixed samples, keeping {len(filtered_metadata)} samples"
            )
            metadata = filtered_metadata
        src_images = Path(local_repo_dir) / "images"
        for idx, entry in enumerate(tqdm(metadata, desc="Processing images")):
            image_id = f"{idx:06d}"
            if "page_info" in entry and "image_path" in entry["page_info"]:
                image_name = entry["page_info"]["image_path"]
                img_path = src_images / image_name
                try:
                    img: PILImage.Image = PILImage.open(img_path)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    output_path = images_dir / f"{image_id}.jpg"
                    img.save(output_path, "JPEG", quality=95)
                except Exception as e:
                    print(
                        f"Warning: Failed to process image {image_name} for {image_id}: {e}"
                    )
    num_images = len(list(images_dir.glob("*.jpg")))
    print("\nCompleted!")
    print(f"Images saved: {num_images} files in {images_dir}")


def process_docvqa_infographicvqa(output_dir: str) -> None:
    """Download DocVQA and InfographicVQA dataset organize images.

    Args:
        output_dir: Base directory to save processed data.
    """

    def _download_and_save_images(
        dataset: str,
        sub_dataset: str,
        output_dir: str,
        split_name: str = "test",
    ) -> None:
        """Downloads dataset and save all images to a directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("Loading dataset...")
        ds = load_dataset(dataset, sub_dataset)
        print("Dataset loaded successfully. Available splits: %s", list(ds.keys()))

        total_images_saved = 0
        seen_hashes = set()

        assert split_name in ds

        split_data = ds[split_name]
        print("Processing split '%s' with %d samples", split_name, len(split_data))

        img_dir = output_path / sub_dataset / "images"
        img_dir.mkdir(exist_ok=True, parents=True)

        images_in_split = 0
        for idx, sample in enumerate(split_data):
            try:
                image = sample.get("image")
                if image is None:
                    print(
                        "No image found in sample %d of split '%s'",
                        idx,
                        split_name,
                    )
                    continue

                img_byte_arr = io.BytesIO()
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(img_byte_arr, format="JPEG", quality=95)
                image_hash = hashlib.sha256(img_byte_arr.getvalue()).hexdigest()

                if image_hash in seen_hashes:
                    continue
                seen_hashes.add(image_hash)

                question_id = sample.get(
                    "questionId",
                    sample.get("question_id", f"{split_name}_{idx}"),
                )
                filename = f"{question_id}.jpg"
                image_path = img_dir / filename

                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(image_path, "JPEG", quality=95)
                images_in_split += 1
                total_images_saved += 1

            except Exception as e:
                print(
                    "Error processing sample %d in split '%s': %s",
                    idx,
                    split_name,
                    e,
                )
                continue

        print("Completed split '%s': saved %d images", split_name, images_in_split)

        print("Download completed! Total images saved: %d", total_images_saved)
        print("Images saved to: %s", img_dir.absolute())

    _download_and_save_images(
        dataset="lmms-lab/DocVQA",
        sub_dataset="DocVQA",
        output_dir=output_dir,
        split_name="test",
    )

    _download_and_save_images(
        dataset="lmms-lab/DocVQA",
        sub_dataset="InfographicVQA",
        output_dir=output_dir,
        split_name="test",
    )

    output_path = Path(output_dir)
    os.rename(output_path / "InfographicVQA", output_path / "infographicvqa")
    os.rename(output_path / "DocVQA", output_path / "docvqa")


def process_chartmuseum_chartqapro(output_dir: str) -> None:
    """Download ChartMuseum and ChartQAPro datasets and organize images.

    Args:
        output_dir: Base directory to save processed data.
    """
    base_dir = Path(output_dir)

    # Download ChartMuseum using snapshot_download
    print("Downloading ChartMuseum dataset...")
    chartmuseum_dir = base_dir / "chartmuseum"

    local_repo = snapshot_download(
        repo_id="lytang/ChartMuseum",
        repo_type="dataset",
        local_dir=str(chartmuseum_dir),
        allow_patterns=["images/*"],
    )
    print(f"ChartMuseum downloaded to: {local_repo}")

    # Download ChartQAPro using load_dataset (images embedded in parquet as binary)
    print("\nDownloading ChartQAPro dataset...")
    chartqapro_dir = base_dir / "chartqapro"
    images_dir = chartqapro_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset - this reads the parquet with embedded images
    ds = load_dataset("ahmed-masry/ChartQAPro")

    # Extract and save images from the 'test' split
    print(f"Extracting {len(ds['test'])} images from ChartQAPro...")
    for idx, sample in enumerate(tqdm(ds["test"], desc="Extracting images")):
        image_data = sample["image"]

        # Handle both PIL Image and bytes formats
        if isinstance(image_data, bytes):
            image: PILImage.Image = PILImage.open(io.BytesIO(image_data))
        else:
            # Already a PIL Image (datasets library auto-decodes)
            image = image_data

        if image.mode != "RGB":
            image = image.convert("RGB")

        filename = f"chart_{idx:04d}.jpg"
        image.save(images_dir / filename, "JPEG", quality=95)

    print(f"ChartQAPro images saved to: {images_dir}")
    print("\nAll datasets downloaded successfully!")


def process_screenspotpro(output_dir: str) -> None:
    """Download ScreenSpot Pro dataset and organize images.

    Args:
        output_dir: Base directory to save processed data.
    """
    base_dir = Path(output_dir)
    images_dir = base_dir / "screenspotpro"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading ScreenSpot Pro dataset from Hugging Face...")
    snapshot_download(
        repo_id="likaixin/ScreenSpot-Pro",
        repo_type="dataset",
        local_dir=images_dir,
        allow_patterns=["images/*"],
        resume_download=True,
    )

    print("\nCompleted!")
    print(f"Images saved in {images_dir}")


def process_worldvqa(output_dir: str) -> None:
    """Download WorldVQA dataset and organize images.

    Original data: https://huggingface.co/datasets/moonshotai/WorldVQA

    Args:
        output_dir: Base directory to save processed data.
    """
    base_dir = Path(output_dir)
    images_dir = base_dir / "worldvqa" / "images"
    labels_dir = base_dir / "worldvqa" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading WorldVQA.tsv from Hugging Face...")
    tsv_path = hf_hub_download(
        repo_id="moonshotai/WorldVQA",
        filename="WorldVQA.tsv",
        repo_type="dataset",
    )

    csv.field_size_limit(sys.maxsize)

    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for sample in tqdm(reader, desc="Processing WorldVQA"):
            # Only keep English samples
            if sample["language"] != "non-zh":
                continue

            image = PILImage.open(io.BytesIO(base64.b64decode(sample["image"])))

            if image.mode != "RGB":
                image = image.convert("RGB")

            image_index = str(sample["index"])
            image.save(images_dir / f"{image_index}.jpg", "JPEG", quality=95)

            label = {
                "index": image_index,
                "question": sample["question"],
                "answer": sample["answer"],
                "difficulty": sample["difficulty"],
                "category": sample["category"],
            }
            with open(labels_dir / f"{image_index}.json", "w") as out:
                json.dump(label, out, indent=4)

    num_images = len(list(images_dir.glob("*.jpg")))
    print("\nCompleted!")
    print(f"Images saved: {num_images} files in {images_dir}")
    print(f"Labels saved: {len(list(labels_dir.glob('*.json')))} files in {labels_dir}")


def process_raw_datasets(output_dir: str) -> None:
    """Process raw datasets and save them into `output_dir`.

    Args:
        output_dir: Base directory to save processed data.
    """
    process_hiertext(output_dir)
    process_omnidocbench(output_dir)
    process_docvqa_infographicvqa(output_dir)
    process_chartmuseum_chartqapro(output_dir)
    process_screenspotpro(output_dir)
    process_worldvqa(output_dir)

    # Check that dataset is built corrected.
    base_dir = Path(output_dir)
    for dataset_name in DATASET_NAMES:
        dataset_dir = base_dir / dataset_name
        if not dataset_dir.exists():
            raise ValueError(f"Dataset {dataset_name} is not created correctly.")

    # Generate JSONL index of all datasets.
    base_dir = Path(output_dir)
    jsonl_path = base_dir / "all_images.jsonl"
    total_count = 0

    with open(jsonl_path, "w") as f:
        for dataset_name in DATASET_NAMES:
            images_dir = base_dir / dataset_name / "images"
            if not images_dir.exists():
                print(f"Warning: {images_dir} does not exist, skipping.")
                continue

            image_paths = sorted(images_dir.rglob("*"))
            image_paths = [
                p
                for p in image_paths
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]

            for image_path in image_paths:
                meta_data: dict[str, str] = {}
                if dataset_name == "worldvqa":
                    label_path = (
                        base_dir / "worldvqa" / "labels" / f"{image_path.stem}.json"
                    )
                    with open(label_path) as lf:
                        meta_data = json.load(lf)

                entry = {
                    "image_paths": [str(image_path.resolve())],
                    "meta_data": meta_data,
                    "dataset_type": dataset_name,
                }
                f.write(json.dumps(entry) + "\n")
                total_count += 1

    print(f"JSONL index created: {total_count} entries in {jsonl_path}")


if __name__ == "__main__":
    process_raw_datasets("data/")
