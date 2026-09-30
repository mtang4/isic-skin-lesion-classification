"""
PyTorch Dataset and transforms for ISIC 2019 skin lesion classification.

Uses DINOv2-compatible preprocessing (ImageNet normalization, 224x224 resize).
Includes augmentations tuned for dermoscopy images.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

# ImageNet stats (DINOv2 was pretrained with these)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Augmentations for training on dermoscopy images."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.25),
    ])


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Deterministic transforms for validation/test."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),  # Slight upscale before center crop
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class ISICSkinLesionDataset(Dataset):
    """
    ISIC Skin Lesion dataset organized as:
        data_dir/{split}/{class_name}/image.jpg

    Args:
        data_dir: Root data directory.
        split: One of 'train', 'val', 'test'.
        transform: Torchvision transforms to apply.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        transform: transforms.Compose | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        self.class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
        self.idx_to_class = {i: name for i, name in enumerate(CLASS_NAMES)}

        # Collect all image paths and labels
        split_dir = self.data_dir / split
        self.samples: list[tuple[Path, int]] = []
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((img_path, self.class_to_idx[class_name]))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No images found in {split_dir}. "
                "Run `python scripts/download_data.py` first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_counts(self) -> np.ndarray:
        """Return per-class sample counts."""
        labels = [label for _, label in self.samples]
        counts = np.bincount(labels, minlength=len(CLASS_NAMES))
        return counts

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for loss function."""
        counts = self.get_class_counts()
        weights = 1.0 / counts.astype(np.float64)
        weights = weights / weights.sum() * len(CLASS_NAMES)
        return torch.tensor(weights, dtype=torch.float32)


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 64,
    image_size: int = 224,
    num_workers: int = 4,
) -> dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders.

    Class imbalance is handled via weighted cross-entropy loss in the
    training script, NOT via oversampling here. Using both simultaneously
    over-corrects and causes the model to massively over-predict minority classes.
    """
    train_dataset = ISICSkinLesionDataset(
        data_dir, split="train", transform=get_train_transforms(image_size)
    )
    val_dataset = ISICSkinLesionDataset(
        data_dir, split="val", transform=get_eval_transforms(image_size)
    )
    test_dataset = ISICSkinLesionDataset(
        data_dir, split="test", transform=get_eval_transforms(image_size)
    )

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }
    return loaders
