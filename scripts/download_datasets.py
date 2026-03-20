"""
scripts/download_datasets.py
─────────────────────────────────────────────────────────────────────────────
One-click dataset downloader.
Downloads and organises all required datasets from Kaggle + HuggingFace.

Requirements:
  pip install kaggle
  Set up ~/.kaggle/kaggle.json with your API key.
  (Get it from: https://www.kaggle.com/settings → API → Create New Token)

Usage:
  python scripts/download_datasets.py --dataset all
  python scripts/download_datasets.py --dataset image
  python scripts/download_datasets.py --dataset video
  python scripts/download_datasets.py --dataset audio
─────────────────────────────────────────────────────────────────────────────
"""

import os
import argparse
import subprocess
import zipfile
import shutil
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.panel import Panel

console = Console()

DATA_ROOT = "data/raw"


def run(cmd: list, desc: str = "") -> bool:
    console.print(f"[cyan]→ {desc}[/cyan]" if desc else f"[cyan]→ {' '.join(cmd)}[/cyan]")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def extract_zip(zip_path: str, dest: str) -> None:
    logger.info(f"Extracting {zip_path} → {dest}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    os.remove(zip_path)


# ─────────────────────────────────────────────────────────────────────────────
#  Image datasets
# ─────────────────────────────────────────────────────────────────────────────

def download_image_datasets() -> None:
    console.print(Panel("📥 Downloading Image Datasets", style="bold blue"))
    dest = os.path.join(DATA_ROOT, "images")
    os.makedirs(dest, exist_ok=True)

    # 1. 140k Real and Fake Faces
    logger.info("Downloading 140k Real and Fake Faces …")
    run(["kaggle", "datasets", "download", "-d", "xhlulu/140k-real-and-fake-faces",
         "-p", dest], desc="140k Real & Fake Faces")

    zip_path = os.path.join(dest, "140k-real-and-fake-faces.zip")
    if os.path.exists(zip_path):
        extract_zip(zip_path, os.path.join(dest, "140k_faces"))
        logger.info("✅ 140k faces downloaded")

    # 2. CIFAKE (AI-generated images)
    logger.info("Downloading CIFAKE …")
    run(["kaggle", "datasets", "download", "-d", "birdy654/cifake-real-and-ai-generated-synthetic-images",
         "-p", dest], desc="CIFAKE")

    zip_path = os.path.join(dest, "cifake-real-and-ai-generated-synthetic-images.zip")
    if os.path.exists(zip_path):
        extract_zip(zip_path, os.path.join(dest, "cifake"))
        logger.info("✅ CIFAKE downloaded")


# ─────────────────────────────────────────────────────────────────────────────
#  Video datasets
# ─────────────────────────────────────────────────────────────────────────────

def download_video_datasets() -> None:
    console.print(Panel("📥 Downloading Video Datasets", style="bold blue"))

    # FaceForensics++ — must be requested from their official form
    # https://github.com/ondyari/FaceForensics/tree/master/dataset
    console.print("""
[yellow]⚠️  FaceForensics++ requires manual access request.[/yellow]
1. Visit: https://github.com/ondyari/FaceForensics/tree/master/dataset
2. Fill out the access form
3. You will receive a download link/script
4. Run: python faceforensics_download_v4.py -d all -c c23 -t videos

Placing data at: data/raw/ff++/
""")

    # Celeb-DF v2
    console.print("""
[yellow]⚠️  Celeb-DF v2 requires manual download.[/yellow]
Visit: https://github.com/yuezunli/celeb-deepfakeforensics
Download: https://drive.google.com/file/d/1iLx76wsbi9itnkxSqz9BVBl4ZvnbIazj/view
Extract to: data/raw/celebdf/
""")

    # DFDC subset from Kaggle
    dest = os.path.join(DATA_ROOT, "dfdc")
    os.makedirs(dest, exist_ok=True)
    logger.info("Downloading DFDC sample (Kaggle) …")
    run(["kaggle", "competitions", "download", "-c", "deepfake-detection-challenge",
         "-p", dest, "--proxy", ""], desc="DFDC")
    logger.info("Note: Full DFDC is 470 GB. Download only the parts you need.")


# ─────────────────────────────────────────────────────────────────────────────
#  Audio datasets
# ─────────────────────────────────────────────────────────────────────────────

def download_audio_datasets() -> None:
    console.print(Panel("📥 Downloading Audio Datasets", style="bold blue"))
    dest = os.path.join(DATA_ROOT, "asvspoof2019")
    os.makedirs(dest, exist_ok=True)

    # ASVspoof 2019 from Kaggle mirror
    logger.info("Downloading ASVspoof 2019 …")
    run(["kaggle", "datasets", "download", "-d", "awsaf49/asvpoof-2019-dataset",
         "-p", dest], desc="ASVspoof 2019")

    zip_path = os.path.join(dest, "asvpoof-2019-dataset.zip")
    if os.path.exists(zip_path):
        extract_zip(zip_path, dest)
        logger.info("✅ ASVspoof 2019 downloaded")

    # FakeAVCeleb
    console.print("""
[yellow]⚠️  FakeAVCeleb requires request form.[/yellow]
Visit: https://github.com/DASH-Lab/FakeAVCeleb
Fill the form and download to: data/raw/fakeavceleb/
""")


# ─────────────────────────────────────────────────────────────────────────────
#  Create directory scaffold (for users without GPU who just want to explore)
# ─────────────────────────────────────────────────────────────────────────────

def create_sample_structure() -> None:
    """Create dummy data files for testing the pipeline without full datasets."""
    import numpy as np
    from PIL import Image as PILImage

    logger.info("Creating sample test data …")

    # Sample images
    for label in ["real", "fake"]:
        d = f"data/raw/images/sample/{label}"
        os.makedirs(d, exist_ok=True)
        for i in range(10):
            arr = np.random.randint(0, 255, (299, 299, 3), dtype=np.uint8)
            PILImage.fromarray(arr).save(f"{d}/sample_{i:04d}.jpg")

    # Sample audio
    import soundfile as sf
    for label in [0, 1]:
        d = f"data/raw/audio/sample/{'real' if label == 0 else 'fake'}"
        os.makedirs(d, exist_ok=True)
        for i in range(5):
            audio = np.random.randn(16000 * 3).astype(np.float32) * 0.1
            sf.write(f"{d}/sample_{i:04d}.wav", audio, 16000)

    logger.info("✅ Sample data created for pipeline testing")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download deepfake detection datasets")
    parser.add_argument("--dataset", choices=["image", "video", "audio", "sample", "all"],
                        default="all")
    args = parser.parse_args()

    os.makedirs(DATA_ROOT, exist_ok=True)

    if args.dataset in ("image", "all"):
        download_image_datasets()

    if args.dataset in ("video", "all"):
        download_video_datasets()

    if args.dataset in ("audio", "all"):
        download_audio_datasets()

    if args.dataset == "sample":
        create_sample_structure()

    console.print("\n[bold green]✅ Dataset download complete![/bold green]")
    console.print("Next step: [cyan]python -m src.preprocessing.extract_frames --dataset all[/cyan]")
