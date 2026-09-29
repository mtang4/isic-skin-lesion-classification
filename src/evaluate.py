"""
Evaluate a trained skin lesion classifier on the test set.

Produces:
  - Per-class precision, recall, F1
  - Confusion matrix heatmap
  - Overall accuracy, balanced accuracy, macro F1

Usage:
    python src/evaluate.py --checkpoint checkpoints/<run>/best_model.pt [--data_dir ./data]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from dataset import CLASS_NAMES, ISICSkinLesionDataset, get_eval_transforms
from model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for plots/reports")
    return parser.parse_args()


def get_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path) -> None:
    """Plot and save a normalized confusion matrix."""
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=axes[0],
    )
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("Confusion Matrix (Counts)")

    # Normalized
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=axes[1],
    )
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Confusion Matrix (Normalized)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved to: {output_path}")


def plot_per_class_metrics(report_dict: dict, class_names: list[str], output_path: Path) -> None:
    """Bar chart of precision, recall, F1 per class."""
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, metric in enumerate(metrics):
        values = [report_dict[name][metric] for name in class_names]
        ax.bar(x + i * width, values, width, label=metric.capitalize())

    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Per-class metrics saved to: {output_path}")


@torch.no_grad()
def run_inference(
    model: nn.Module,
    dataset: ISICSkinLesionDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference, return (labels, predictions, probabilities)."""
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model.eval()

    all_labels, all_preds, all_probs = [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def main():
    args = parse_args()
    device = get_device(args.device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    backbone = ckpt.get("backbone", cfg["model"]["backbone"])
    mode = ckpt.get("mode", cfg["model"]["mode"])

    # Build model and load weights
    model = build_model(
        backbone_name=backbone,
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
        mode=mode,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load test data
    image_size = cfg["data"]["image_size"]
    test_dataset = ISICSkinLesionDataset(
        args.data_dir, split="test", transform=get_eval_transforms(image_size)
    )
    print(f"Test set: {len(test_dataset)} images\n")

    # Run inference
    labels, preds, probs = run_inference(model, test_dataset, args.batch_size, device)

    # Metrics
    accuracy = (preds == labels).mean()
    bal_acc = balanced_accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")

    # AUC (one-vs-rest)
    try:
        auc_ovr = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc_ovr = float("nan")

    print("=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"  Accuracy:          {accuracy:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Macro F1:          {macro_f1:.4f}")
    print(f"  Weighted F1:       {weighted_f1:.4f}")
    print(f"  AUC (OvR macro):   {auc_ovr:.4f}")
    print()

    # Per-class report
    report = classification_report(
        labels, preds, target_names=CLASS_NAMES, digits=4, output_dict=True,
    )
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))

    # Output directory for plots
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm, CLASS_NAMES, output_dir / "confusion_matrix.png")
    plot_per_class_metrics(report, CLASS_NAMES, output_dir / "per_class_metrics.png")

    # Save results to text file
    results_path = output_dir / "test_results.txt"
    with open(results_path, "w") as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Backbone: {backbone}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Test samples: {len(test_dataset)}\n\n")
        f.write(f"Accuracy:          {accuracy:.4f}\n")
        f.write(f"Balanced Accuracy: {bal_acc:.4f}\n")
        f.write(f"Macro F1:          {macro_f1:.4f}\n")
        f.write(f"Weighted F1:       {weighted_f1:.4f}\n")
        f.write(f"AUC (OvR macro):   {auc_ovr:.4f}\n\n")
        f.write(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))
    print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
