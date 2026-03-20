"""
src/models/video_model.py
─────────────────────────────────────────────────────────────────────────────
Video-based deepfake detection.
Architecture: EfficientNet-B4 (frame encoder) + BiLSTM (temporal reasoning)
Trained on FaceForensics++ / DFDC / Celeb-DF v2.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
#  Frame Encoder (CNN)
# ─────────────────────────────────────────────────────────────────────────────

class FrameEncoder(nn.Module):
    """
    EfficientNet-B4 backbone that encodes individual frames into
    fixed-size feature vectors. Weights are shared across all frames.
    """

    def __init__(self, pretrained: bool = True, feature_dim: int = 512):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        backbone_dim = self.backbone.num_features  # 1792 for B4

        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        feats = self.backbone(x)          # (B, backbone_dim)
        return self.projection(feats)     # (B, feature_dim)


# ─────────────────────────────────────────────────────────────────────────────
#  Temporal Attention
# ─────────────────────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """
    Self-attention over the temporal (frame) dimension.
    Helps the model focus on the most suspicious frames.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key   = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn = F.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        return (attn @ V).mean(dim=1)   # (B, H)  — temporal mean pooling


# ─────────────────────────────────────────────────────────────────────────────
#  Full Video Model
# ─────────────────────────────────────────────────────────────────────────────

class DeepfakeVideoClassifier(nn.Module):
    """
    End-to-end video deepfake detector.

    Pipeline:
      frames (B, T, C, H, W)
      → FrameEncoder (shared weights) → (B, T, feature_dim)
      → BiLSTM                        → (B, T, 2*lstm_hidden)
      → TemporalAttention             → (B, 2*lstm_hidden)
      → Classifier head               → (B, num_classes)
    """

    def __init__(
        self,
        feature_dim: int = 512,
        lstm_hidden: int = 512,
        lstm_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.5,
        pretrained: bool = True,
    ):
        super().__init__()

        self.encoder = FrameEncoder(pretrained=pretrained, feature_dim=feature_dim)

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        self.temporal_attn = TemporalAttention(lstm_hidden * 2)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),
        )

        self._init_weights()
        logger.info(f"VideoClassifier: EfficientNet-B4 + BiLSTM | "
                    f"params={sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def _init_weights(self) -> None:
        for name, p in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, C, H, W) → (B, T, feature_dim)"""
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)
        x = self.encoder(x)
        return x.view(B, T, -1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # ── Encode each frame independently ────────────────────────────────
        feat_seq = self.encode_frames(frames)               # (B, T, feature_dim)

        # ── Temporal modelling ────────────────────────────────────────────
        lstm_out, _ = self.lstm(feat_seq)                   # (B, T, 2*hidden)

        # ── Attend over time ──────────────────────────────────────────────
        context = self.temporal_attn(lstm_out)              # (B, 2*hidden)

        # ── Classify ──────────────────────────────────────────────────────
        return self.classifier(context)                     # (B, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class VideoFaceDataset(Dataset):
    """
    Loads sequences of face crops for a video clip.
    Expects metadata dict produced by preprocessing pipeline.
    """

    FRAME_TRANSFORMS = {
        "train": A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(var_limit=(5, 25), p=0.2),
            A.ImageCompression(quality_lower=70, quality_upper=100, p=0.4),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        "val": A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    }

    def __init__(
        self,
        metadata: Dict,
        split: str = "train",
        frames_per_clip: int = 32,
        face_key: str = "face_dir",
    ):
        self.metadata        = list(metadata.values())
        self.split           = split
        self.frames_per_clip = frames_per_clip
        self.face_key        = face_key
        self.transform       = self.FRAME_TRANSFORMS.get(split, self.FRAME_TRANSFORMS["val"])

    def __len__(self) -> int:
        return len(self.metadata)

    def _load_frames(self, face_dir: str) -> torch.Tensor:
        """Load and transform N frames from a face directory."""
        frame_paths = sorted(Path(face_dir).glob("*.jpg"))

        if len(frame_paths) == 0:
            return torch.zeros(self.frames_per_clip, 3, 224, 224)

        # Sample evenly distributed frames
        indices = np.linspace(0, len(frame_paths) - 1, self.frames_per_clip, dtype=int)
        selected = [frame_paths[i] for i in indices]

        frames = []
        for fp in selected:
            try:
                img = np.array(Image.open(fp).convert("RGB"))
                t   = self.transform(image=img)["image"]
                frames.append(t)
            except Exception:
                frames.append(torch.zeros(3, 224, 224))

        return torch.stack(frames)   # (T, C, H, W)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item      = self.metadata[idx]
        face_dir  = item.get(self.face_key, "")
        label     = item.get("label", 0)
        frames    = self._load_frames(face_dir)
        return frames, torch.tensor(label, dtype=torch.long)


def build_video_splits(
    metadata_paths: List[str],
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> Tuple[Dict, Dict, Dict]:
    """
    Merge metadata from multiple JSON files and split into train/val/test.
    """
    all_items = {}
    for path in metadata_paths:
        if os.path.exists(path):
            with open(path) as f:
                all_items.update(json.load(f))

    keys = list(all_items.keys())
    random.seed(seed)
    random.shuffle(keys)

    n       = len(keys)
    n_train = int(n * split_ratios[0])
    n_val   = int(n * split_ratios[1])

    train_meta = {k: all_items[k] for k in keys[:n_train]}
    val_meta   = {k: all_items[k] for k in keys[n_train:n_train + n_val]}
    test_meta  = {k: all_items[k] for k in keys[n_train + n_val:]}

    logger.info(f"Video splits — train: {len(train_meta)} | val: {len(val_meta)} | test: {len(test_meta)}")
    return train_meta, val_meta, test_meta
