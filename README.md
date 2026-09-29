# Skin Lesion Classification with DINOv2

Classify dermoscopy images across 8 diagnostic categories using a [DINOv2](https://arxiv.org/abs/2304.07193) vision foundation model, trained on the [ISIC 2019 Challenge](https://challenge.isic-archive.com/data/#2019) dataset.

## Overview

This project fine-tunes (or linear-probes) Meta's DINOv2 Vision Transformer for skin lesion classification. DINOv2 was pretrained via self-supervised learning on 142M images and produces strong visual features out of the box — making it well-suited for medical imaging tasks where labeled data is limited.

**Architecture:** `DINOv2 ViT → [CLS] token → LayerNorm → Dropout → Linear → 8 classes`

### Classes

| Label | Description | ISIC 2019 Distribution |
|-------|-------------|----------------------|
| MEL   | Melanoma | ~18% |
| NV    | Melanocytic nevus | ~52% |
| BCC   | Basal cell carcinoma | ~14% |
| AK    | Actinic keratosis | ~3% |
| BKL   | Benign keratosis | ~10% |
| DF    | Dermatofibroma | ~1% |
| VASC  | Vascular lesion | ~1% |
| SCC   | Squamous cell carcinoma | ~2% |

## Quick Start (Google Colab)

The easiest way to run this project is on Google Colab with a free T4 GPU:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mtang4/isic-skin-lesion-classification/blob/main/notebooks/train_colab.ipynb)

The notebook clones this repo, installs dependencies, downloads the data, and runs training — all core logic lives in `src/`, not the notebook.

## Local Setup

```bash
# Clone the repo
git clone https://github.com/mtang4/isic-skin-lesion-classification.git
cd isic-skin-lesion-classification

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Download Data

Downloads the ISIC 2019 dataset (~9GB) and creates stratified train/val/test splits:

```bash
python scripts/download_data.py
```

This will:
1. Download 25,331 dermoscopy images + ground truth labels from the ISIC Archive
2. Create stratified 70/15/15 train/val/test splits
3. Organize images into `data/{train,val,test}/<class>/` directories
4. Compute class weights for handling the heavy class imbalance

To skip download if you already have the data:
```bash
python scripts/download_data.py --skip_download --data_dir ./data
```

## Training

### Linear Probe (recommended starting point)

Freezes the DINOv2 backbone and only trains the classification head. Fast and gives a strong baseline:

```bash
python src/train.py --config configs/default.yaml
```

### Fine-Tune

Trains the full model with a lower learning rate on the backbone:

```bash
python src/train.py --config configs/default.yaml --mode fine_tune --epochs 15 --lr 1e-4
```

### CLI Overrides

Any config value can be overridden via command line:

```bash
python src/train.py --config configs/default.yaml \
    --backbone facebook/dinov2-large \
    --batch_size 32 \
    --epochs 50
```

### Monitoring

Training logs are written to TensorBoard:

```bash
tensorboard --logdir logs/
```

## Evaluation

Evaluate a trained model on the held-out test set:

```bash
python src/evaluate.py --checkpoint checkpoints/<run_name>/best_model.pt
```

This produces:
- Overall accuracy, balanced accuracy, macro/weighted F1, AUC
- Per-class precision, recall, and F1 scores
- Confusion matrix (raw counts + normalized)
- Per-class bar chart

Results and plots are saved alongside the checkpoint.

## Project Structure

```
├── configs/
│   └── default.yaml            # Hyperparameters and paths
├── scripts/
│   └── download_data.py        # Dataset download and split preparation
├── src/
│   ├── dataset.py              # PyTorch Dataset, transforms, DataLoaders
│   ├── model.py                # DINOv2 backbone + classification head
│   ├── train.py                # Training loop with logging and checkpointing
│   └── evaluate.py             # Test evaluation with metrics and plots
├── requirements.txt
└── README.md
```

## Configuration

All hyperparameters live in `configs/default.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.backbone` | `facebook/dinov2-base` | DINOv2 variant (`-small`, `-base`, `-large`) |
| `model.mode` | `linear_probe` | `linear_probe` or `fine_tune` |
| `training.batch_size` | 64 | Batch size |
| `training.epochs` | 30 | Number of training epochs |
| `training.lr` | 1e-3 | Learning rate (classifier head) |
| `training.backbone_lr` | 1e-5 | Learning rate (backbone, fine-tune only) |
| `training.label_smoothing` | 0.1 | Label smoothing factor |
| `training.use_class_weights` | true | Weight loss by inverse class frequency |

## Handling Class Imbalance

ISIC 2019 is heavily skewed (NV is ~52% of samples, DF/VASC are ~1%). This project addresses it with:
- **Weighted random sampling** during training to oversample minority classes
- **Inverse-frequency class weights** in the cross-entropy loss
- **Balanced accuracy** as the primary model selection metric

## References

- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- [ISIC 2019 Challenge](https://challenge.isic-archive.com/data/#2019)
- [Skin Lesion Analysis Toward Melanoma Detection (ISBI 2017)](https://arxiv.org/abs/1710.05006)

## License

This project is for educational and research purposes. The ISIC 2019 dataset is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
