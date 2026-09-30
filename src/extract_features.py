"""
Extract and cache DINOv2 [CLS] token embeddings for all images.

Runs each image through the frozen backbone exactly once and saves
the resulting feature vectors to disk. This eliminates redundant
backbone forward passes during linear probe training.

Output format (one .pt file per split):
    {
        "features": Tensor of shape (N, hidden_size),  # e.g. (17731, 768)
        "labels": Tensor of shape (N,),
        "paths": list of image paths (for debugging),
    }

Usage:
    python src/extract_features.py [--data_dir ./data] [--backbone facebook/dinov2-base]
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import Dinov2Model

from dataset import CLASS_NAMES, ISICSkinLesionDataset, get_eval_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract DINOv2 features")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--backbone", type=str, default="facebook/dinov2-base")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def get_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def extract_split(
    backbone: Dinov2Model,
    dataset: ISICSkinLesionDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict:
    """Extract [CLS] features for an entire dataset split."""
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_features = []
    all_labels = []
    all_paths = []

    for images, labels in tqdm(loader, desc="Extracting"):
        images = images.to(device, non_blocking=True)
        outputs = backbone(pixel_values=images)
        cls_tokens = outputs.last_hidden_state[:, 0]  # (B, hidden_size)
        all_features.append(cls_tokens.cpu())
        all_labels.append(labels)

    # Also collect paths for debugging
    all_paths = [str(p) for p, _ in dataset.samples]

    return {
        "features": torch.cat(all_features, dim=0),
        "labels": torch.cat(all_labels, dim=0),
        "paths": all_paths,
    }


def main():
    args = parse_args()
    device = get_device(args.device)
    data_dir = Path(args.data_dir)

    # Use deterministic eval transforms for all splits (including train).
    # Augmentation is not meaningful here — we extract one canonical
    # feature vector per image.
    transform = get_eval_transforms(args.image_size)

    print(f"Loading backbone: {args.backbone}")
    backbone = Dinov2Model.from_pretrained(args.backbone)
    backbone = backbone.to(device)
    backbone.eval()

    hidden_size = backbone.config.hidden_size
    print(f"  Hidden size: {hidden_size}")
    print(f"  Device: {device}\n")

    # Output directory
    feature_dir = data_dir / "features" / args.backbone.replace("/", "_")
    feature_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        print(f"Processing {split}...")
        dataset = ISICSkinLesionDataset(data_dir, split=split, transform=transform)
        print(f"  {len(dataset)} images")

        result = extract_split(
            backbone, dataset, args.batch_size, args.num_workers, device
        )

        out_path = feature_dir / f"{split}.pt"
        torch.save(result, out_path)
        print(f"  Saved: {out_path} — shape {result['features'].shape}\n")

    print("Done! Features cached. Now run:")
    print(f"  python src/train_cached.py --feature_dir {feature_dir}")


if __name__ == "__main__":
    main()
