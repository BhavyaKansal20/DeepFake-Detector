"""
src/models/audio_model.py
─────────────────────────────────────────────────────────────────────────────
Audio-based deepfake / voice-spoof detection.
Architecture: wav2vec2-base (feature extractor) + LCNN classifier head
Trained on ASVspoof 2019/2021 LA track.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Wav2Vec2Model, Wav2Vec2Config
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
#  LCNN (Light CNN) — effective for anti-spoofing
# ─────────────────────────────────────────────────────────────────────────────

class MaxFeatureMap2D(nn.Module):
    """Max-Feature-Map activation used in LCNN."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split channels in half, take element-wise max
        return torch.max(*x.chunk(2, dim=1))


class LCNNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size=3, stride=stride, padding=1)
        self.mfm  = MaxFeatureMap2D()
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.mfm(self.conv(x)))


class LCNN(nn.Module):
    """
    Light CNN adapted for spectrogram-based anti-spoofing.
    Input: (B, 1, F, T)  — single-channel spectrogram
    Output: (B, feature_dim)
    """

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            LCNNBlock(1,   32),
            nn.MaxPool2d(2, 2),
            LCNNBlock(32,  64),
            LCNNBlock(64,  64),
            nn.MaxPool2d(2, 2),
            LCNNBlock(64,  128),
            LCNNBlock(128, 128),
            nn.MaxPool2d(2, 2),
            LCNNBlock(128, 256),
            LCNNBlock(256, 256),
            nn.MaxPool2d(2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, F, T)
        h = self.net(x)
        h = self.pool(h).flatten(1)
        return self.proj(h)


# ─────────────────────────────────────────────────────────────────────────────
#  wav2vec2 Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────

class Wav2VecFeatureExtractor(nn.Module):
    """
    Uses pre-trained wav2vec2-base to extract contextualised frame-level
    representations, then reduces to clip-level with attention pooling.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        feature_dim: int = 256,
        freeze_feature_extractor: bool = True,
    ):
        super().__init__()
        self.w2v = Wav2Vec2Model.from_pretrained(model_name)

        if freeze_feature_extractor:
            # Freeze CNN feature extractor — only fine-tune transformer layers
            self.w2v.feature_extractor._freeze_parameters()

        hidden = self.w2v.config.hidden_size   # 768 for wav2vec2-base

        # Attention pooling over time
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.proj = nn.Sequential(
            nn.Linear(hidden, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (B, T_samples)
        outputs  = self.w2v(waveform, output_hidden_states=False)
        hidden   = outputs.last_hidden_state     # (B, T_frames, 768)

        # Attention pooling
        scores   = self.attn_pool(hidden)        # (B, T_frames, 1)
        weights  = F.softmax(scores, dim=1)
        pooled   = (hidden * weights).sum(dim=1) # (B, 768)

        return self.proj(pooled)                 # (B, feature_dim)


# ─────────────────────────────────────────────────────────────────────────────
#  Full Audio Model (Dual-Branch Fusion)
# ─────────────────────────────────────────────────────────────────────────────

class DeepfakeAudioClassifier(nn.Module):
    """
    Dual-branch audio deepfake detector:
      Branch A: wav2vec2 → contextual waveform features
      Branch B: LCNN     → spectrogram-based artefact features

    Both branches are fused and classified jointly.
    This dual approach covers both temporal and spectral forgery clues.
    """

    def __init__(
        self,
        wav2vec_model: str = "facebook/wav2vec2-base",
        num_classes: int = 2,
        feature_dim: int = 256,
        dropout: float = 0.4,
    ):
        super().__init__()

        # Branch A — waveform
        self.wav2vec_branch = Wav2VecFeatureExtractor(
            model_name=wav2vec_model,
            feature_dim=feature_dim,
        )

        # Branch B — spectrogram (MFCC / mel / LFCC)
        self.lcnn_branch = LCNN(feature_dim=feature_dim)

        # Fusion + Classifier
        fused_dim = feature_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),
        )

        self._init_weights()
        logger.info(
            f"AudioClassifier: wav2vec2 + LCNN dual-branch | "
            f"params={sum(p.numel() for p in self.parameters() if p.requires_grad):,}"
        )

    def _init_weights(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        waveform: torch.Tensor,        # (B, T_samples)
        spectrogram: torch.Tensor,     # (B, 1, F, T_frames)
    ) -> torch.Tensor:
        w2v_feat  = self.wav2vec_branch(waveform)      # (B, feature_dim)
        lcnn_feat = self.lcnn_branch(spectrogram)      # (B, feature_dim)
        fused     = torch.cat([w2v_feat, lcnn_feat], dim=1)
        return self.classifier(fused)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class AudioDeepfakeDataset(Dataset):
    """
    Loads pre-computed .npz files containing:
      waveform, mfcc, mel, lfcc
    Returns waveform + mel-spectrogram as model inputs.
    """

    def __init__(
        self,
        metadata: Dict,
        max_samples: int = 160000,  # 10s @ 16kHz
        augment: bool = False,
    ):
        self.samples    = list(metadata.values())
        self.max_samples = max_samples
        self.augment    = augment

    def __len__(self) -> int:
        return len(self.samples)

    def _pad_or_trim(self, arr: np.ndarray, length: int) -> np.ndarray:
        if len(arr) < length:
            return np.pad(arr, (0, length - len(arr)))
        return arr[:length]

    def _augment_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """Simple augmentations: speed perturbation, additive noise."""
        if np.random.random() < 0.3:
            # Additive Gaussian noise
            noise = np.random.randn(*waveform.shape).astype(np.float32) * 0.005
            waveform = np.clip(waveform + noise, -1.0, 1.0)
        if np.random.random() < 0.2:
            # Amplitude scaling
            scale = np.random.uniform(0.7, 1.3)
            waveform = np.clip(waveform * scale, -1.0, 1.0)
        return waveform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.samples[idx]
        try:
            npz      = np.load(item["npz_path"])
            waveform = npz["waveform"].astype(np.float32)
            mel      = npz["mel"].astype(np.float32)
        except Exception:
            waveform = np.zeros(self.max_samples, dtype=np.float32)
            mel      = np.zeros((80, 1000), dtype=np.float32)

        # Normalise mel
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        mel_tensor = torch.tensor(mel).unsqueeze(0)  # (1, F, T)

        # Pad/trim waveform
        waveform = self._pad_or_trim(waveform, self.max_samples)
        if self.augment:
            waveform = self._augment_waveform(waveform)
        wave_tensor = torch.tensor(waveform)

        label = torch.tensor(item.get("label", 0), dtype=torch.long)
        return wave_tensor, mel_tensor, label


def build_audio_splits(
    metadata_paths: List[str],
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Tuple[Dict, Dict, Dict]:
    """Merge audio metadata files and split."""
    import random

    all_items: Dict = {}
    for path in metadata_paths:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
                # ASVspoof has train/dev/eval nested; FakeAVCeleb is flat
                if "train" in data:
                    for split_data in data.values():
                        all_items.update(split_data)
                else:
                    all_items.update(data)

    keys = list(all_items.keys())
    random.seed(seed)
    random.shuffle(keys)

    n       = len(keys)
    n_train = int(n * split_ratios[0])
    n_val   = int(n * split_ratios[1])

    train = {k: all_items[k] for k in keys[:n_train]}
    val   = {k: all_items[k] for k in keys[n_train:n_train + n_val]}
    test  = {k: all_items[k] for k in keys[n_train + n_val:]}

    logger.info(f"Audio splits — train: {len(train)} | val: {len(val)} | test: {len(test)}")
    return train, val, test
