"""
src/models/image_model.py
─────────────────────────────────────────────────────────────────────────────
Image-based deepfake detection models.
Architectures: EfficientNet-B4, XceptionNet, ViT-Base
Trained on 140k Real/Fake Faces + CIFAKE datasets.
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
#  Attention modules
# ─────────────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.fc(self.gap(x).view(x.size(0), -1))
        mx  = self.fc(self.gmp(x).view(x.size(0), -1))
        attn = torch.sigmoid(avg + mx).view(x.size(0), x.size(1), 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn   = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        cat  = torch.cat([avg, mx], dim=1)
        attn = torch.sigmoid(self.bn(self.conv(cat)))
        return x * attn


# ─────────────────────────────────────────────────────────────────────────────
#  Classifiers
# ─────────────────────────────────────────────────────────────────────────────

class DeepfakeImageClassifier(nn.Module):
    """
    Flexible image classifier with pluggable backbone.
    Adds dual-attention head + custom classification neck.

    Architectures:
      - efficientnet_b4   (default, best accuracy/speed balance)
      - xception          (strong at detecting GAN artifacts)
      - vit_base_patch16_224  (strong at global structure anomalies)
    """

    SUPPORTED = {
        "efficientnet_b4":       {"features": 1792},
        "xception":              {"features": 2048},
        "vit_base_patch16_224":  {"features": 768},
    }

    def __init__(
        self,
        architecture: str = "efficientnet_b4",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.4,
    ):
        super().__init__()

        if architecture not in self.SUPPORTED:
            raise ValueError(f"Unsupported architecture: {architecture}. "
                             f"Choose from {list(self.SUPPORTED.keys())}")

        self.architecture = architecture
        self.num_features = self.SUPPORTED[architecture]["features"]
        self.is_vit       = "vit" in architecture

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            architecture,
            pretrained=pretrained,
            num_classes=0,          # remove original head
            global_pool="avg" if not self.is_vit else "",
        )

        # ── Attention (CNN only) ───────────────────────────────────────────────
        if not self.is_vit:
            in_ch = self._get_feature_channels()
            self.channel_att = ChannelAttention(in_ch)
            self.spatial_att = SpatialAttention()

        # ── Classification head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

        self._init_classifier_weights()
        logger.info(f"ImageClassifier: {architecture} | params={self._count_params():,}")

    def _get_feature_channels(self) -> int:
        """Get channel count at backbone's last conv layer."""
        try:
            return self.backbone.num_features
        except Exception:
            return self.SUPPORTED[self.architecture]["features"]

    def _init_classifier_weights(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_vit:
            features = self.backbone.forward_features(x)
            # ViT: take CLS token
            features = features[:, 0]
        else:
            # CNN: extract feature maps, apply attention, pool
            feat_maps = self.backbone.forward_features(x)
            feat_maps = self.channel_att(feat_maps)
            feat_maps = self.spatial_att(feat_maps)
            features  = F.adaptive_avg_pool2d(feat_maps, 1).flatten(1)

        logits = self.classifier(features)
        return logits


class EnsembleImageClassifier(nn.Module):
    """
    Ensemble of EfficientNet-B4 + Xception for higher robustness.
    Outputs averaged probabilities from both models.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.eff    = DeepfakeImageClassifier("efficientnet_b4",       num_classes, pretrained)
        self.xcep   = DeepfakeImageClassifier("xception",              num_classes, pretrained)
        self.weights = nn.Parameter(torch.tensor([0.55, 0.45]))    # learnable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits_eff  = F.softmax(self.eff(x), dim=-1)
        logits_xcep = F.softmax(self.xcep(x), dim=-1)
        w = F.softmax(self.weights, dim=0)
        return w[0] * logits_eff + w[1] * logits_xcep


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────

import os
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class FaceImageDataset(Dataset):
    """
    Generic face image dataset.
    Expects a list of (image_path, label) tuples.
    Applies strong augmentation during training to improve generalisation
    to unseen deepfake generators.
    """

    TRAIN_TRANSFORMS = A.Compose([
        A.Resize(299, 299),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.4),
        A.RandomBrightnessContrast(p=0.4),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
        A.HueSaturationValue(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    EVAL_TRANSFORMS = A.Compose([
        A.Resize(299, 299),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    def __init__(
        self,
        samples: list,          # [(path, label), ...]
        split: str = "train",
    ):
        self.samples   = samples
        self.transform = self.TRAIN_TRANSFORMS if split == "train" else self.EVAL_TRANSFORMS

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            img = np.array(Image.open(path).convert("RGB"))
        except Exception:
            img = np.zeros((299, 299, 3), dtype=np.uint8)
            label = 0

        augmented = self.transform(image=img)
        return augmented["image"], torch.tensor(label, dtype=torch.long)


def build_image_dataset(
    real_dir: str,
    fake_dirs: list,
    split_ratios: tuple = (0.8, 0.1, 0.1),
    seed: int = 42,
):
    """
    Build train/val/test splits from real and fake image directories.
    Returns three lists of (path, label) tuples.
    """
    import random

    def collect_images(directory: int, label: int):
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        files = []
        if os.path.exists(directory):
            for f in Path(directory).rglob("*"):
                if f.suffix.lower() in exts:
                    files.append((str(f), label))
        return files

    samples = collect_images(real_dir, 0)
    for fake_dir in fake_dirs:
        samples += collect_images(fake_dir, 1)

    random.seed(seed)
    random.shuffle(samples)

    n = len(samples)
    n_train = int(n * split_ratios[0])
    n_val   = int(n * split_ratios[1])

    return (
        samples[:n_train],
        samples[n_train:n_train + n_val],
        samples[n_train + n_val:],
    )


from pathlib import Path
