"""
DINOv2-based classifier for skin lesion classification.

Supports two modes:
  - linear_probe: Freeze backbone, only train the classification head.
  - fine_tune: Train backbone (with low LR) + classification head.
"""

import torch
import torch.nn as nn
from transformers import Dinov2Model


class DINOv2Classifier(nn.Module):
    """
    DINOv2 backbone with a classification head.

    Architecture:
        DINOv2 ViT → [CLS] token → LayerNorm → Dropout → Linear → num_classes

    Args:
        backbone_name: HuggingFace model name (e.g. 'facebook/dinov2-base').
        num_classes: Number of output classes.
        dropout: Dropout rate before the classifier.
        mode: 'linear_probe' (frozen backbone) or 'fine_tune'.
    """

    def __init__(
        self,
        backbone_name: str = "facebook/dinov2-base",
        num_classes: int = 8,
        dropout: float = 0.1,
        mode: str = "linear_probe",
    ):
        super().__init__()
        self.mode = mode

        # Load pretrained DINOv2
        self.backbone = Dinov2Model.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size  # 768 for base, 1024 for large

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier[2].weight)
        nn.init.zeros_(self.classifier[2].bias)

        # Freeze backbone for linear probing
        if mode == "linear_probe":
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.mode = "fine_tune"

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            pixel_values: Input images, shape (B, 3, H, W).

        Returns:
            Logits, shape (B, num_classes).
        """
        # DINOv2 forward pass — use the [CLS] token embedding
        if self.mode == "linear_probe":
            with torch.no_grad():
                outputs = self.backbone(pixel_values=pixel_values)
        else:
            outputs = self.backbone(pixel_values=pixel_values)

        cls_token = outputs.last_hidden_state[:, 0]  # (B, hidden_size)
        logits = self.classifier(cls_token)
        return logits

    def get_param_groups(self, backbone_lr: float, head_lr: float) -> list[dict]:
        """
        Get parameter groups with different learning rates.

        Useful for fine-tuning: backbone gets a lower LR, head gets a higher LR.
        """
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr},
            {"params": self.classifier.parameters(), "lr": head_lr},
        ]


def build_model(
    backbone_name: str = "facebook/dinov2-base",
    num_classes: int = 8,
    dropout: float = 0.1,
    mode: str = "linear_probe",
) -> DINOv2Classifier:
    """Build and return the model."""
    model = DINOv2Classifier(
        backbone_name=backbone_name,
        num_classes=num_classes,
        dropout=dropout,
        mode=mode,
    )

    # Log parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {backbone_name} ({mode})")
    print(f"  Total parameters:     {total_params:>12,}")
    print(f"  Trainable parameters: {trainable_params:>12,}")
    return model
