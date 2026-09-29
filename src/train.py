"""
Training script for ISIC 2019 skin lesion classification with DINOv2.

Usage:
    # Linear probe (fast, good baseline)
    python src/train.py --config configs/default.yaml

    # Fine-tune (better accuracy, slower)
    python src/train.py --config configs/default.yaml --mode fine_tune --epochs 15 --lr 1e-4

    # Override any config value via CLI
    python src/train.py --config configs/default.yaml --batch_size 32 --backbone facebook/dinov2-large
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import CLASS_NAMES, create_dataloaders
from model import build_model


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DINOv2 skin lesion classifier")
    parser.add_argument("--config", type=str, default="configs/default.yaml")

    # Allow overriding any config value
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["linear_probe", "fine_tune"], default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--backbone_lr", type=float, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()


def get_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None,
    epoch: int,
) -> dict[str, float]:
    """Train for one epoch. Returns metrics dict."""
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and device.type == "cuda":
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

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
    loader,
    criterion: nn.Module,
    device: torch.device,
    split: str = "val",
) -> dict[str, float]:
    """Evaluate on val/test set. Returns metrics dict."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc=f"Eval ({split})", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
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
    cfg = load_config(args.config)

    # Apply CLI overrides
    data_dir = args.data_dir or cfg["data"]["data_dir"]
    backbone = args.backbone or cfg["model"]["backbone"]
    mode = args.mode or cfg["model"]["mode"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    epochs = args.epochs or cfg["training"]["epochs"]
    lr = args.lr or cfg["training"]["lr"]
    backbone_lr = args.backbone_lr or cfg["training"]["backbone_lr"]
    image_size = args.image_size or cfg["data"]["image_size"]
    num_workers = cfg["data"]["num_workers"]
    label_smoothing = cfg["training"]["label_smoothing"]
    warmup_epochs = cfg["training"]["warmup_epochs"]
    weight_decay = cfg["training"]["weight_decay"]
    use_class_weights = cfg["training"]["use_class_weights"]
    checkpoint_dir = Path(cfg["output"]["checkpoint_dir"])
    log_dir = Path(cfg["output"]["log_dir"])

    run_name = args.run_name or f"{mode}_{backbone.split('/')[-1]}_{time.strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = checkpoint_dir / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    print(f"Device: {device}")

    # Data
    print("\nLoading data...")
    loaders = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        use_weighted_sampler=True,
    )

    # Model
    print("\nBuilding model...")
    model = build_model(
        backbone_name=backbone,
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
        mode=mode,
    )
    model = model.to(device)

    # Loss function with optional class weights
    class_weights = None
    if use_class_weights:
        class_weights = loaders["train"].dataset.get_class_weights().to(device)
        print(f"\nClass weights: {class_weights.cpu().numpy().round(3)}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=label_smoothing
    )

    # Optimizer
    if mode == "fine_tune":
        param_groups = model.get_param_groups(backbone_lr=backbone_lr, head_lr=lr)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            model.classifier.parameters(), lr=lr, weight_decay=weight_decay
        )

    # Learning rate scheduler with warmup
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision scaler (CUDA only)
    scaler = GradScaler() if device.type == "cuda" else None

    # TensorBoard
    writer = SummaryWriter(log_dir / run_name)

    # Training loop
    best_bal_acc = 0.0
    print(f"\nStarting training: {epochs} epochs, mode={mode}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, scaler, epoch
        )
        val_metrics = evaluate(model, loaders["val"], criterion, device, split="val")

        scheduler.step()
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[-1]["lr"]

        # Log to TensorBoard
        for key in train_metrics:
            writer.add_scalar(f"train/{key}", train_metrics[key], epoch)
        for key in val_metrics:
            writer.add_scalar(f"val/{key}", val_metrics[key], epoch)
        writer.add_scalar("lr", current_lr, epoch)

        # Print epoch summary
        print(
            f"Epoch {epoch:>3d}/{epochs} ({elapsed:.0f}s) | "
            f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
            f"BAcc: {train_metrics['balanced_accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
            f"BAcc: {val_metrics['balanced_accuracy']:.4f}, F1: {val_metrics['macro_f1']:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # Save best model
        if val_metrics["balanced_accuracy"] > best_bal_acc:
            best_bal_acc = val_metrics["balanced_accuracy"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": cfg,
                    "backbone": backbone,
                    "mode": mode,
                },
                checkpoint_dir / "best_model.pt",
            )
            print(f"  ↑ New best model saved (BAcc: {best_bal_acc:.4f})")

        # Save latest checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_metrics": val_metrics,
                "config": cfg,
            },
            checkpoint_dir / "latest.pt",
        )

    writer.close()
    print(f"\nTraining complete. Best Val BAcc: {best_bal_acc:.4f}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Logs: {log_dir / run_name}")
    print(f"\nTo evaluate: python src/evaluate.py --checkpoint {checkpoint_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
