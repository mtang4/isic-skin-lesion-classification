"""
Download and prepare the ISIC 2019 skin lesion classification dataset.

The ISIC 2019 challenge dataset contains 25,331 dermoscopic images across 8 categories:
  0: MEL  - Melanoma
  1: NV   - Melanocytic nevus
  2: BCC  - Basal cell carcinoma
  3: AK   - Actinic keratosis
  4: BKL  - Benign keratosis
  5: DF   - Dermatofibroma
  6: VASC - Vascular lesion
  7: SCC  - Squamous cell carcinoma

Data source: ISIC Archive (https://challenge.isic-archive.com/data/#2019)

Usage:
    python scripts/download_data.py [--data_dir ./data] [--val_split 0.15] [--test_split 0.15]
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ISIC_2019_URLS = {
    "images": "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip",
    "labels": "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv",
}

CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]


def download_file(url: str, dest: Path) -> None:
    """Download a file with progress bar using urllib."""
    import urllib.request

    from tqdm import tqdm

    if dest.exists():
        print(f"  Already exists: {dest}")
        return

    print(f"  Downloading: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    response = urllib.request.urlopen(url)
    total_size = int(response.headers.get("Content-Length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=dest.name
    ) as pbar:
        while True:
            chunk = response.read(8192)
            if not chunk:
                break
            f.write(chunk)
            pbar.update(len(chunk))


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a zip file."""
    print(f"  Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    print(f"  Extracted to: {dest_dir}")


def prepare_splits(
    data_dir: Path,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> None:
    """
    Organize images into train/val/test splits with class subdirectories.

    Creates:
        data_dir/train/<class_name>/image.jpg
        data_dir/val/<class_name>/image.jpg
        data_dir/test/<class_name>/image.jpg
    """
    labels_path = data_dir / "ISIC_2019_Training_GroundTruth.csv"
    df = pd.read_csv(labels_path)

    # Convert one-hot to single label column
    df["label"] = df[CLASS_NAMES].idxmax(axis=1)
    df["label_idx"] = df[CLASS_NAMES].values.argmax(axis=1)

    print(f"\nDataset statistics:")
    print(f"  Total images: {len(df)}")
    print(f"  Class distribution:")
    for i, name in enumerate(CLASS_NAMES):
        count = (df["label_idx"] == i).sum()
        print(f"    {name:>4s} ({i}): {count:>6d} ({100 * count / len(df):.1f}%)")

    # Stratified split: first split off test, then split remainder into train/val
    train_val_df, test_df = train_test_split(
        df, test_size=test_split, stratify=df["label_idx"], random_state=seed
    )
    relative_val = val_split / (1 - test_split)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val,
        stratify=train_val_df["label_idx"],
        random_state=seed,
    )

    print(f"\n  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Find where images were extracted
    image_dir = data_dir / "ISIC_2019_Training_Input"
    if not image_dir.exists():
        # Some extractions put them in a nested directory
        candidates = list(data_dir.glob("**/ISIC_2019_Training_Input"))
        if candidates:
            image_dir = candidates[0]
        else:
            raise FileNotFoundError(
                f"Could not find extracted images in {data_dir}. "
                "Check that the zip file was extracted correctly."
            )

    # Organize into split/class directories
    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        split_dir = data_dir / split_name
        for class_name in CLASS_NAMES:
            (split_dir / class_name).mkdir(parents=True, exist_ok=True)

        for _, row in split_df.iterrows():
            image_id = row["image"]
            label = row["label"]
            src = image_dir / f"{image_id}.jpg"
            dst = split_dir / label / f"{image_id}.jpg"
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

    print(f"\n  Organized into: {data_dir}/{{train,val,test}}/<class>/")

    # Save split metadata for reproducibility
    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        split_df.to_csv(data_dir / f"{split_name}_metadata.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Download and prepare ISIC 2019 dataset")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to store data")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test_split", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download (use if data already downloaded)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        print("Step 1: Downloading ISIC 2019 dataset...")
        download_file(ISIC_2019_URLS["labels"], data_dir / "ISIC_2019_Training_GroundTruth.csv")
        download_file(ISIC_2019_URLS["images"], data_dir / "ISIC_2019_Training_Input.zip")

        print("\nStep 2: Extracting images...")
        zip_path = data_dir / "ISIC_2019_Training_Input.zip"
        if zip_path.exists():
            extract_zip(zip_path, data_dir)
    else:
        print("Skipping download (--skip_download).")

    print("\nStep 3: Preparing train/val/test splits...")
    prepare_splits(data_dir, args.val_split, args.test_split, args.seed)

    # Compute and save class weights for handling imbalance
    train_meta = pd.read_csv(data_dir / "train_metadata.csv")
    counts = np.array([
        (train_meta["label_idx"] == i).sum() for i in range(len(CLASS_NAMES))
    ])
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(CLASS_NAMES)
    weight_df = pd.DataFrame({"class": CLASS_NAMES, "count": counts, "weight": weights})
    weight_df.to_csv(data_dir / "class_weights.csv", index=False)
    print(f"\n  Class weights saved to {data_dir}/class_weights.csv")
    print(weight_df.to_string(index=False))

    print("\nDone! You can now run training with: python src/train.py")


if __name__ == "__main__":
    main()
