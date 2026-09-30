"""
Train a linear classifier on pre-extracted DINOv2 features.

This is orders of magnitude faster than train.py because:
  - No image loading or decoding
  - No backbone forward pass (features are pre-computed)
  - Each "batch" is just a slice of a tensor already in memory

Typical runtime: <1 minute for 30 epochs on CPU, seconds on GPU.

Usage:
    python src/train_cached.py --feature_dir data/features/facebook_dinov2-base

    # Override defaults
    python src/train_cached.py --feature_dir data/features/facebook_dinov2-base \
        --epochs 50 --lr 1e-3 --dropout 0.1
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train linear head on cached features")
    parser.add_argument("--feature_dir", type=str, required=True, help="Path to extracted features")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--use_class_weights", action="store_true", default=True)
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument(
        "--weight_dampening", type=str, default="none",
        choices=["none", "sqrt", "log"],
        help="Dampen class weights: none=raw 1/freq, sqrt=sqrt(1/freq), log=log(1/freq)",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")
    return parser.parse_args()


def get_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_features(feature_dir: Path, device: torch.device) -> dict[str, TensorDataset]:
    """Load pre-extracted features into TensorDatasets, all in memory."""
    datasets = {}
    for split in ["train", "val", "test"]:
        data = torch.load(feature_dir / f"{split}.pt", map_location="cpu", weights_only=False)
        features = data["features"]  # (N, hidden_size)
        labels = data["labels"]      # (N,)
        datasets[split] = TensorDataset(features, labels)
        print(f"  {split}: {len(labels)} samples, feature dim {features.shape[1]}")
    return datasets


def compute_class_weights(dataset: TensorDataset, dampening: str = "none") -> torch.Tensor:
    """
    Compute class weights from a TensorDataset.

    Dampening controls how aggressively we correct for imbalance:
      - "none": raw inverse frequency (1/count). Strong correction.
      - "sqrt": sqrt of inverse frequency. Moderate correction.
      - "log":  log of inverse frequency. Gentle correction.
    """
    labels = dataset.tensors[1].numpy()
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float64)
    weights = 1.0 / counts

    if dampening == "sqrt":
        weights = np.sqrt(weights)
    elif dampening == "log":
        weights = np.log1p(weights)

    weights = weights / weights.sum() * len(CLASS_NAMES)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return {
        "loss": running_loss / n,
        "accuracy": (all_preds == all_labels).mean(),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(features)
        loss = criterion(logits, labels)

        running_loss += loss.item() * features.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return {
        "loss": running_loss / n,
        "accuracy": (all_preds == all_labels).mean(),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
    }


def main():
    args = parse_args()
    device = get_device(args.device)
    feature_dir = Path(args.feature_dir)
    use_class_weights = args.use_class_weights and not args.no_class_weights

    print(f"Device: {device}")
    print(f"\nLoading cached features from: {feature_dir}")
    datasets = load_features(feature_dir, device)

    hidden_size = datasets["train"].tensors[0].shape[1]
    num_classes = len(CLASS_NAMES)

    # Dataloaders — large batch size since these are just small tensors
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False),
    }

    # Classification head (same architecture as model.py, but standalone)
    model = nn.Sequential(
        nn.LayerNorm(hidden_size),
        nn.Dropout(args.dropout),
        nn.Linear(hidden_size, num_classes),
    )
    nn.init.xavier_uniform_(model[2].weight)
    nn.init.zeros_(model[2].bias)
    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters())
    print(f"\nClassifier: {trainable_params:,} parameters")

    # Loss — optionally weighted by inverse class frequency
    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(datasets["train"], dampening=args.weight_dampening).to(device)
        print(f"Class weights ({args.weight_dampening}): {class_weights.cpu().numpy().round(3)}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=args.label_smoothing
    )

    # Optimizer + cosine schedule with warmup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    def lr_lambda(epoch: int) -> float:
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Logging
    run_name = args.run_name or f"cached_linear_{time.strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = Path(args.checkpoint_dir) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(Path(args.log_dir) / run_name)

    # Training loop
    best_bal_acc = 0.0
    print(f"\nTraining for {args.epochs} epochs...\n")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_metrics = evaluate(model, loaders["val"], criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        for key in train_metrics:
            writer.add_scalar(f"train/{key}", train_metrics[key], epoch)
        for key in val_metrics:
            writer.add_scalar(f"val/{key}", val_metrics[key], epoch)
        writer.add_scalar("lr", current_lr, epoch)

        print(
            f"Epoch {epoch:>3d}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f}, BAcc: {train_metrics['balanced_accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, BAcc: {val_metrics['balanced_accuracy']:.4f}, "
            f"F1: {val_metrics['macro_f1']:.4f} | LR: {current_lr:.2e}"
        )

        if val_metrics["balanced_accuracy"] > best_bal_acc:
            best_bal_acc = val_metrics["balanced_accuracy"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "hidden_size": hidden_size,
                    "num_classes": num_classes,
                    "dropout": args.dropout,
                },
                checkpoint_dir / "best_head.pt",
            )
            print(f"  ↑ New best (BAcc: {best_bal_acc:.4f})")

    total_time = time.time() - t_start
    writer.close()

    print(f"\nTraining complete in {total_time:.1f}s")
    print(f"Best Val BAcc: {best_bal_acc:.4f}")
    print(f"Checkpoint: {checkpoint_dir / 'best_head.pt'}")
    print(f"\nTo evaluate:")
    print(f"  python src/evaluate_cached.py --feature_dir {feature_dir} --checkpoint {checkpoint_dir / 'best_head.pt'}")


if __name__ == "__main__":
    main()
