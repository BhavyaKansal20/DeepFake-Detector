"""
src/utils/helpers.py
─────────────────────────────────────────────────────────────────────────────
Shared utilities: config loading, logging, seeding, metrics, file helpers.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import random
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import yaml
from loguru import logger
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, classification_report
)

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load YAML config and return as nested dict."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Seed everything for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Global seed set to {seed}")


# ─────────────────────────────────────────────────────────────────────────────
#  Device
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Auto-select best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU found — using CPU (training will be slow!)")
    return device


# ─────────────────────────────────────────────────────────────────────────────
#  Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(log_dir: str = "logs", name: str = "deepfake") -> None:
    """Configure loguru with file + console output."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = Path(log_dir) / f"{name}.log"
    logger.add(
        str(log_file),
        rotation="10 MB",
        retention="1 week",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{line} | {message}",
    )
    logger.info(f"Logger initialised → {log_file}")


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Compute and optionally print a full suite of classification metrics."""
    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, average="weighted")),
    }

    if y_prob is not None:
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        except Exception:
            metrics["auc_roc"] = 0.0

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    metrics.update({
        "true_positive_rate": float(tp / (tp + fn + 1e-8)),   # sensitivity
        "true_negative_rate": float(tn / (tn + fp + 1e-8)),   # specificity
        "false_positive_rate": float(fp / (fp + tn + 1e-8)),
        "false_negative_rate": float(fn / (fn + tp + 1e-8)),
        "precision": float(tp / (tp + fp + 1e-8)),
    })

    if verbose:
        _print_metrics_table(metrics)
        print(classification_report(y_true, y_pred, target_names=["REAL", "FAKE"]))

    return metrics


def _print_metrics_table(metrics: Dict[str, float]) -> None:
    table = Table(title="📊 Evaluation Metrics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    for k, v in metrics.items():
        color = "green" if v >= 0.9 else "yellow" if v >= 0.75 else "red"
        table.add_row(k.replace("_", " ").title(), f"[{color}]{v:.4f}[/{color}]")
    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
#  File helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dir(*paths: str) -> None:
    """Create directories if they don't exist."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


def file_hash(filepath: str, algo: str = "md5") -> str:
    """Return hex hash of a file (for deduplication)."""
    h = hashlib.new(algo)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_file_size_mb(filepath: str) -> float:
    return os.path.getsize(filepath) / (1024 * 1024)


def list_files(directory: str, extensions: List[str]) -> List[str]:
    """Recursively list all files with given extensions."""
    files = []
    for ext in extensions:
        files.extend(Path(directory).rglob(f"*{ext}"))
    return [str(f) for f in sorted(files)]


# ─────────────────────────────────────────────────────────────────────────────
#  Model checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    path: str,
    is_best: bool = False,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, path)
    if is_best:
        best_path = Path(path).parent / "best_model.pt"
        shutil.copy(path, best_path)
        logger.info(f"✅ New best model saved → {best_path}  |  metrics: {metrics}")


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[torch.nn.Module, int, Dict]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})
    logger.info(f"Loaded checkpoint from epoch {epoch}  |  {path}")
    return model, epoch, metrics


# ─────────────────────────────────────────────────────────────────────────────
#  Label utilities
# ─────────────────────────────────────────────────────────────────────────────

LABEL_MAP = {0: "REAL", 1: "FAKE"}
CONFIDENCE_LABELS = {
    "high":   lambda p: p >= 0.85 or p <= 0.15,
    "medium": lambda p: 0.65 <= p < 0.85 or 0.15 < p <= 0.35,
    "low":    lambda p: 0.35 < p < 0.65,
}


def probability_to_verdict(prob_fake: float, threshold: float = 0.5) -> Dict:
    """Convert raw fake-probability to a human-readable verdict dict."""
    label = "FAKE" if prob_fake >= threshold else "REAL"
    confidence = "HIGH" if abs(prob_fake - 0.5) > 0.35 else \
                 "MEDIUM" if abs(prob_fake - 0.5) > 0.15 else "LOW"
    return {
        "verdict": label,
        "confidence": confidence,
        "fake_probability": round(float(prob_fake), 4),
        "real_probability": round(float(1 - prob_fake), 4),
    }
