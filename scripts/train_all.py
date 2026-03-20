"""
scripts/train_all.py
─────────────────────────────────────────────────────────────────────────────
Master training script — trains all three models in sequence.
Usage:
  python scripts/train_all.py
  python scripts/train_all.py --skip-video  (skip slow video training)
  python scripts/train_all.py --model image  (train one model only)
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import time
import yaml
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def banner(title: str, color: str = "bold blue") -> None:
    console.print(Panel(f"🚀 {title}", style=color))


def train_image(cfg: dict) -> None:
    banner("Training IMAGE model (EfficientNet-B4 + Attention)", "bold cyan")
    t0 = time.time()
    from src.train.train_image import train
    train(cfg)
    console.print(f"[green]✅ Image model trained in {(time.time()-t0)/60:.1f} min[/green]")


def train_video(cfg: dict) -> None:
    banner("Training VIDEO model (EfficientNet-B4 + BiLSTM)", "bold magenta")
    t0 = time.time()
    from src.train.train_video import train
    train(cfg)
    console.print(f"[green]✅ Video model trained in {(time.time()-t0)/60:.1f} min[/green]")


def train_audio(cfg: dict) -> None:
    banner("Training AUDIO model (wav2vec2 + LCNN)", "bold yellow")
    t0 = time.time()
    from src.train.train_audio import train
    train(cfg)
    console.print(f"[green]✅ Audio model trained in {(time.time()-t0)/60:.1f} min[/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all deepfake detection models")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model",  choices=["image", "video", "audio", "all"], default="all")
    parser.add_argument("--skip-video", action="store_true",
                        help="Skip video model (slowest to train)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    total_start = time.time()
    console.print(Panel(
        "🔍 DeepfakeDetector — Full Training Pipeline\n"
        "Datasets: FaceForensics++ | DFDC | 140k Faces | ASVspoof 2019",
        style="bold white on blue"
    ))

    if args.model in ("image", "all"):
        train_image(cfg)

    if args.model in ("video", "all") and not args.skip_video:
        train_video(cfg)

    if args.model in ("audio", "all"):
        train_audio(cfg)

    elapsed = (time.time() - total_start) / 60
    console.print(Panel(
        f"✅ All models trained! Total time: {elapsed:.1f} minutes\n"
        f"Start the API: python -m api.main\n"
        f"Start the UI:  streamlit run ui/app.py",
        style="bold green"
    ))
