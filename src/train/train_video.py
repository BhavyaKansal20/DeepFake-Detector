"""
src/train/train_video.py
─────────────────────────────────────────────────────────────────────────────
Training pipeline for video-based deepfake detection.
EfficientNet-B4 frame encoder + BiLSTM temporal model.
─────────────────────────────────────────────────────────────────────────────
Usage:
  python -m src.train.train_video --config configs/config.yaml
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
import yaml
from loguru import logger
from tqdm import tqdm

from src.models.video_model import (
    DeepfakeVideoClassifier,
    VideoFaceDataset,
    build_video_splits,
)
from src.utils.helpers import (
    set_seed, get_device, setup_logger,
    compute_metrics, save_checkpoint, ensure_dir,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    is_train: bool,
    grad_clip: float = 1.0,
) -> Tuple[float, Dict]:
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []
    desc = "Train" if is_train else "Val"

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for frames, labels in tqdm(loader, desc=desc, leave=False):
            frames = frames.to(device)          # (B, T, C, H, W)
            labels = labels.to(device)

            with autocast():
                logits = model(frames)
                loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

            probs = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
            preds = probs.argmax(axis=1)

            total_loss  += loss.item()
            all_preds.extend(preds.tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.tolist())

    avg_loss = total_loss / max(len(loader), 1)
    metrics  = compute_metrics(
        np.array(all_targets),
        np.array(all_preds),
        y_prob=np.array(all_probs),
        verbose=False,
    )
    return avg_loss, metrics


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: Dict) -> None:
    set_seed(cfg["project"]["seed"])
    setup_logger()
    device = get_device()

    vid_cfg = cfg["video_model"]
    out_dir = os.path.join(cfg["paths"]["models"], "video_model")
    ensure_dir(out_dir)

    # ── Build datasets ────────────────────────────────────────────────────────
    meta_paths = [
        os.path.join(cfg["paths"]["processed"], "ff_metadata.json"),
        os.path.join(cfg["paths"]["processed"], "dfdc_metadata.json"),
    ]

    train_meta, val_meta, test_meta = build_video_splits(
        meta_paths,
        split_ratios=(
            vid_cfg["dataset"]["train_split"],
            vid_cfg["dataset"]["val_split"],
            vid_cfg["dataset"]["test_split"],
        ),
        seed=cfg["project"]["seed"],
    )

    train_ds = VideoFaceDataset(train_meta, split="train",
                                frames_per_clip=vid_cfg["frames_per_clip"])
    val_ds   = VideoFaceDataset(val_meta,   split="val",
                                frames_per_clip=vid_cfg["frames_per_clip"])
    test_ds  = VideoFaceDataset(test_meta,  split="val",
                                frames_per_clip=vid_cfg["frames_per_clip"])

    # Video batches are heavy — keep batch_size small
    train_loader = DataLoader(
        train_ds,
        batch_size=vid_cfg["train"]["batch_size"],
        shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=vid_cfg["train"]["batch_size"],
        shuffle=False, num_workers=4, pin_memory=True,
    )
    logger.info(f"Video — Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = DeepfakeVideoClassifier(
        feature_dim=512,
        lstm_hidden=vid_cfg["lstm_hidden"],
        lstm_layers=vid_cfg["lstm_layers"],
        num_classes=vid_cfg["num_classes"],
        dropout=vid_cfg["dropout"],
        pretrained=vid_cfg["pretrained"],
    ).to(device)

    # ── Two-phase LR: lower LR for pre-trained backbone ───────────────────────
    backbone_params = list(model.encoder.backbone.parameters())
    new_params      = [p for p in model.parameters()
                       if not any(p is bp for bp in backbone_params)]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": float(vid_cfg["train"]["lr"]) * 0.1},
        {"params": new_params,      "lr": float(vid_cfg["train"]["lr"])},
    ], weight_decay=float(vid_cfg["train"]["weight_decay"]))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=vid_cfg["train"]["epochs"],
        eta_min=1e-7,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler    = GradScaler(enabled=vid_cfg["train"]["mixed_precision"])

    # ── Training loop ─────────────────────────────────────────────────────────
    best_auc   = 0.0
    no_improve = 0
    patience   = vid_cfg["train"]["early_stopping_patience"]

    for epoch in range(1, vid_cfg["train"]["epochs"] + 1):
        t0 = time.time()

        train_loss, train_m = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            is_train=True, grad_clip=float(vid_cfg["train"]["gradient_clip"])
        )
        scheduler.step()

        val_loss, val_m = run_epoch(
            model, val_loader, criterion, optimizer, device, scaler,
            is_train=False
        )

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:03d}/{vid_cfg['train']['epochs']} | "
            f"TrainLoss={train_loss:.4f} Acc={train_m['accuracy']:.4f} | "
            f"ValLoss={val_loss:.4f} Acc={val_m['accuracy']:.4f} "
            f"AUC={val_m.get('auc_roc', 0):.4f} | "
            f"{elapsed:.1f}s"
        )

        current_auc = val_m.get("auc_roc", val_m["accuracy"])
        is_best     = current_auc > best_auc
        if is_best:
            best_auc   = current_auc
            no_improve = 0
        else:
            no_improve += 1

        save_checkpoint(
            model, optimizer, epoch, val_m,
            os.path.join(out_dir, f"checkpoint_epoch_{epoch:03d}.pt"),
            is_best=is_best,
        )

        if no_improve >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    # ── Final test eval ───────────────────────────────────────────────────────
    best_path = os.path.join(out_dir, "best_model.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2)
    _, test_m   = run_epoch(model, test_loader, criterion, optimizer, device, scaler, False)
    logger.info(f"Video Test Results: {test_m}")


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
