"""
src/train/train_audio.py
─────────────────────────────────────────────────────────────────────────────
Training pipeline for audio-based deepfake / voice spoof detection.
wav2vec2 + LCNN dual-branch model on ASVspoof 2019/2021.
─────────────────────────────────────────────────────────────────────────────
Usage:
  python -m src.train.train_audio --config configs/config.yaml
─────────────────────────────────────────────────────────────────────────────
"""

import os
import argparse
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
import yaml
from loguru import logger
from tqdm import tqdm

from src.models.audio_model import (
    DeepfakeAudioClassifier,
    AudioDeepfakeDataset,
    build_audio_splits,
)
from src.utils.helpers import (
    set_seed, get_device, setup_logger,
    compute_metrics, save_checkpoint, ensure_dir,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Equal Error Rate (key metric for anti-spoofing)
# ─────────────────────────────────────────────────────────────────────────────

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Equal Error Rate — standard ASVspoof metric."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[eer_idx] + fnr[eer_idx]) / 2)


# ─────────────────────────────────────────────────────────────────────────────
#  Epoch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    scheduler=None,
    is_train: bool = True,
) -> Tuple[float, Dict]:
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []
    desc = "Train" if is_train else "Val"

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for waveforms, spectrograms, labels in tqdm(loader, desc=desc, leave=False):
            waveforms    = waveforms.to(device)
            spectrograms = spectrograms.to(device)
            labels       = labels.to(device)

            with autocast():
                logits = model(waveforms, spectrograms)
                loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

            probs = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
            preds = probs.argmax(axis=1)

            total_loss  += loss.item()
            all_preds.extend(preds.tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.tolist())

    avg_loss = total_loss / max(len(loader), 1)
    y_true   = np.array(all_targets)
    y_pred   = np.array(all_preds)
    y_prob   = np.array(all_probs)

    metrics = compute_metrics(y_true, y_pred, y_prob=y_prob, verbose=False)
    try:
        metrics["eer"] = compute_eer(y_true, y_prob[:, 1])
    except Exception:
        metrics["eer"] = 1.0

    return avg_loss, metrics


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: Dict) -> None:
    set_seed(cfg["project"]["seed"])
    setup_logger()
    device = get_device()

    aud_cfg = cfg["audio_model"]
    out_dir = os.path.join(cfg["paths"]["models"], "audio_model")
    ensure_dir(out_dir)

    # ── Datasets ───────────────────────────────────────────────────────────────
    meta_paths = [
        os.path.join(cfg["paths"]["processed"], "asvspoof_metadata.json"),
        os.path.join(cfg["paths"]["processed"], "fakeavceleb_audio_metadata.json"),
    ]
    train_meta, val_meta, test_meta = build_audio_splits(
        meta_paths,
        split_ratios=(
            aud_cfg["train"]["batch_size"] and 0.8,  # always 0.8/0.1/0.1
            0.1, 0.1,
        ),
        seed=cfg["project"]["seed"],
    )

    train_ds = AudioDeepfakeDataset(train_meta, augment=True,
                                    max_samples=int(aud_cfg["sample_rate"] * aud_cfg["max_duration"]))
    val_ds   = AudioDeepfakeDataset(val_meta,   augment=False,
                                    max_samples=int(aud_cfg["sample_rate"] * aud_cfg["max_duration"]))
    test_ds  = AudioDeepfakeDataset(test_meta,  augment=False,
                                    max_samples=int(aud_cfg["sample_rate"] * aud_cfg["max_duration"]))

    train_loader = DataLoader(train_ds, batch_size=aud_cfg["train"]["batch_size"],
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=aud_cfg["train"]["batch_size"],
                              shuffle=False, num_workers=4, pin_memory=True)
    logger.info(f"Audio — Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DeepfakeAudioClassifier(
        wav2vec_model=aud_cfg["wav2vec_model"],
        num_classes=aud_cfg["num_classes"],
        feature_dim=256,
        dropout=0.4,
    ).to(device)

    # ── Optimiser — separate LR for pretrained wav2vec2 vs rest ───────────────
    w2v_params  = list(model.wav2vec_branch.w2v.parameters())
    rest_params = [p for p in model.parameters()
                   if not any(p is wp for wp in w2v_params)]

    optimizer = torch.optim.AdamW([
        {"params": w2v_params,  "lr": float(aud_cfg["train"]["lr"]) * 0.1},
        {"params": rest_params, "lr": float(aud_cfg["train"]["lr"])},
    ], weight_decay=float(aud_cfg["train"]["weight_decay"]))

    total_steps  = aud_cfg["train"]["epochs"] * len(train_loader)
    warmup_steps = int(aud_cfg["train"]["warmup_steps"])
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler    = GradScaler(enabled=aud_cfg["train"]["mixed_precision"])

    # ── Training loop ─────────────────────────────────────────────────────────
    best_eer   = 1.0
    no_improve = 0
    patience   = aud_cfg["train"]["early_stopping_patience"]

    for epoch in range(1, aud_cfg["train"]["epochs"] + 1):
        t0 = time.time()

        train_loss, train_m = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            scheduler=scheduler, is_train=True,
        )
        val_loss, val_m = run_epoch(
            model, val_loader, criterion, optimizer, device, scaler,
            is_train=False,
        )

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:03d}/{aud_cfg['train']['epochs']} | "
            f"TrainLoss={train_loss:.4f} Acc={train_m['accuracy']:.4f} EER={train_m.get('eer', 1):.4f} | "
            f"ValLoss={val_loss:.4f} Acc={val_m['accuracy']:.4f} EER={val_m.get('eer', 1):.4f} | "
            f"{elapsed:.1f}s"
        )

        current_eer = val_m.get("eer", 1.0)
        is_best     = current_eer < best_eer
        if is_best:
            best_eer   = current_eer
            no_improve = 0
        else:
            no_improve += 1

        save_checkpoint(
            model, optimizer, epoch, val_m,
            os.path.join(out_dir, f"checkpoint_epoch_{epoch:03d}.pt"),
            is_best=is_best,
        )

        if no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # ── Final test ────────────────────────────────────────────────────────────
    best_path = os.path.join(out_dir, "best_model.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)
    _, test_m   = run_epoch(model, test_loader, criterion, optimizer, device, scaler, is_train=False)
    logger.info(f"Audio Test Results: {test_m}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)
