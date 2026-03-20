"""
src/train/train_image.py
─────────────────────────────────────────────────────────────────────────────
Training pipeline for image-based deepfake detection.
Features: mixed precision, cosine LR schedule, label smoothing,
          class-balanced sampling, early stopping, MLflow tracking.
─────────────────────────────────────────────────────────────────────────────
Usage:
  python -m src.train.train_image --config configs/config.yaml
  python -m src.train.train_image --config configs/config.yaml --arch xception
─────────────────────────────────────────────────────────────────────────────
"""

import os
import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import yaml
from loguru import logger
from tqdm import tqdm

from src.models.image_model import (
    DeepfakeImageClassifier,
    FaceImageDataset,
    build_image_dataset,
)
from src.utils.helpers import (
    load_config, set_seed, get_device,
    setup_logger, compute_metrics,
    save_checkpoint, ensure_dir,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Focal Loss (handles class imbalance better than CE)
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.alpha           = alpha
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing, reduction="none"
        )(logits, targets)
        pt      = torch.exp(-ce_loss)
        focal   = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal.mean()


# ─────────────────────────────────────────────────────────────────────────────
#  Balanced sampler
# ─────────────────────────────────────────────────────────────────────────────

def make_balanced_sampler(dataset: FaceImageDataset) -> WeightedRandomSampler:
    labels  = [s[1] for s in dataset.samples]
    classes, counts = np.unique(labels, return_counts=True)
    class_weights   = 1.0 / counts
    sample_weights  = [class_weights[l] for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Training + Validation loop
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

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    desc = "Train" if is_train else "Val"

    with ctx:
        for imgs, labels in tqdm(loader, desc=desc, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            with autocast():
                logits = model(imgs)
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

            total_loss   += loss.item()
            all_preds.extend(preds.tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.tolist())

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(
        np.array(all_targets),
        np.array(all_preds),
        y_prob=np.array(all_probs),
        verbose=False,
    )
    return avg_loss, metrics


# ─────────────────────────────────────────────────────────────────────────────
#  Main trainer
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: Dict, arch_override: str = None) -> None:
    set_seed(cfg["project"]["seed"])
    setup_logger()
    device = get_device()

    img_cfg = cfg["image_model"]
    arch    = arch_override or img_cfg["architecture"]
    out_dir = os.path.join(cfg["paths"]["models"], "image_model")
    ensure_dir(out_dir)

    # ── Datasets ──────────────────────────────────────────────────────────────
    logger.info("Building image datasets …")
    real_dir  = os.path.join(cfg["paths"]["faces"], "real")
    fake_dirs = [
        os.path.join(cfg["paths"]["faces"], d)
        for d in os.listdir(cfg["paths"]["faces"])
        if d.startswith("fake")
    ] or [os.path.join(cfg["paths"]["faces"], "fake")]

    train_s, val_s, test_s = build_image_dataset(
        real_dir, fake_dirs,
        split_ratios=(
            img_cfg["dataset"]["train_split"],
            img_cfg["dataset"]["val_split"],
            img_cfg["dataset"]["test_split"],
        ),
        seed=cfg["project"]["seed"],
    )

    train_ds = FaceImageDataset(train_s, split="train")
    val_ds   = FaceImageDataset(val_s,   split="val")
    test_ds  = FaceImageDataset(test_s,  split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=img_cfg["train"]["batch_size"],
        sampler=make_balanced_sampler(train_ds),
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=img_cfg["train"]["batch_size"] * 2,
        shuffle=False, num_workers=4, pin_memory=True,
    )
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DeepfakeImageClassifier(
        architecture=arch,
        num_classes=img_cfg["num_classes"],
        pretrained=img_cfg["pretrained"],
        dropout=img_cfg["dropout"],
    ).to(device)

    # ── Optimiser + Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(img_cfg["train"]["lr"]),
        weight_decay=float(img_cfg["train"]["weight_decay"]),
    )

    total_steps = img_cfg["train"]["epochs"] * len(train_loader)
    warmup_steps = img_cfg["train"]["warmup_epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(img_cfg["train"]["lr"]),
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy="cos",
    )

    criterion = FocalLoss(label_smoothing=0.1)
    scaler    = GradScaler(enabled=img_cfg["train"]["mixed_precision"])

    # ── Training loop ─────────────────────────────────────────────────────────
    best_auc    = 0.0
    patience    = img_cfg["train"]["early_stopping_patience"]
    no_improve  = 0

    for epoch in range(1, img_cfg["train"]["epochs"] + 1):
        t0 = time.time()

        train_loss, train_m = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            is_train=True
        )
        scheduler.step()

        val_loss, val_m = run_epoch(
            model, val_loader, criterion, optimizer, device, scaler,
            is_train=False
        )

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:03d}/{img_cfg['train']['epochs']} | "
            f"TrainLoss={train_loss:.4f} Acc={train_m['accuracy']:.4f} | "
            f"ValLoss={val_loss:.4f} Acc={val_m['accuracy']:.4f} "
            f"AUC={val_m.get('auc_roc', 0):.4f} | "
            f"{elapsed:.1f}s"
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        current_auc = val_m.get("auc_roc", val_m["accuracy"])
        is_best     = current_auc > best_auc

        if is_best:
            best_auc   = current_auc
            no_improve = 0
        else:
            no_improve += 1

        ckpt_path = os.path.join(out_dir, f"checkpoint_epoch_{epoch:03d}.pt")
        save_checkpoint(model, optimizer, epoch, val_m, ckpt_path, is_best=is_best)

        if no_improve >= patience:
            logger.info(f"Early stopping — no AUC improvement for {patience} epochs")
            break

    # ── Final evaluation ─────────────────────────────────────────────────────
    logger.info("Running final evaluation on test set …")
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)
    best_model_path = os.path.join(out_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    _, test_m = run_epoch(model, test_loader, criterion, optimizer, device, scaler, False)
    logger.info(f"Test Results: {test_m}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image deepfake detector")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--arch",   default=None,
                        choices=["efficientnet_b4", "xception", "vit_base_patch16_224"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, arch_override=args.arch)
