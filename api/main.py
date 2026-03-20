"""
api/main.py
─────────────────────────────────────────────────────────────────────────────
FastAPI backend for DeepfakeDetector.
Endpoints:
  POST /detect          — upload any file (image/video/audio) → verdict
  POST /detect/image    — image-only endpoint
  POST /detect/video    — video-only endpoint
  POST /detect/audio    — audio-only endpoint
  GET  /health          — health check
  GET  /models/status   — which models are loaded
─────────────────────────────────────────────────────────────────────────────
"""

import os
import time
import uuid
import tempfile
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

import aiofiles
import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger

from src.inference.detector import DeepfakeDetector


# ─────────────────────────────────────────────────────────────────────────────
#  Config + Global State
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_PATH  = os.environ.get("CONFIG_PATH", "configs/config.yaml")
MODELS_DIR   = os.environ.get("MODELS_DIR", "models")
UPLOAD_DIR   = os.environ.get("UPLOAD_DIR", "/tmp/deepfake_uploads")
MAX_FILE_MB  = int(os.environ.get("MAX_FILE_MB", "500"))

os.makedirs(UPLOAD_DIR, exist_ok=True)

detector: Optional[DeepfakeDetector] = None


# ─────────────────────────────────────────────────────────────────────────────
#  Lifespan (startup + shutdown)
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    logger.info("Loading DeepfakeDetector models …")
    try:
        detector = DeepfakeDetector(
            models_dir=MODELS_DIR,
            config_path=CONFIG_PATH,
        )
        logger.info("✅ Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.warning("API will start but detection endpoints won't work until models are trained.")
        detector = None
    yield
    logger.info("Shutting down …")


# ─────────────────────────────────────────────────────────────────────────────
#  App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Deepfake Detector API",
    description="Real-time deepfake detection for images, videos, and audio using deep learning.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
#  Pydantic Schemas
# ─────────────────────────────────────────────────────────────────────────────

class DetectionResult(BaseModel):
    verdict:          str                       # REAL / FAKE
    confidence:       str                       # HIGH / MEDIUM / LOW
    fake_probability: float
    real_probability: float
    modality:         str
    latency_ms:       Optional[float] = None
    detail:           Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    uptime_s: float


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

START_TIME = time.time()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
ALL_EXTS   = IMAGE_EXTS | VIDEO_EXTS | AUDIO_EXTS


async def save_upload(file: UploadFile, ext: str) -> str:
    """Save uploaded file to a temp path and return the path."""
    filename = f"{uuid.uuid4().hex}{ext}"
    path     = os.path.join(UPLOAD_DIR, filename)
    async with aiofiles.open(path, "wb") as f:
        content = await file.read()
        if len(content) > MAX_FILE_MB * 1024 * 1024:
            raise HTTPException(413, f"File too large (max {MAX_FILE_MB} MB)")
        await f.write(content)
    return path


def cleanup_file(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def run_detection_sync(path: str, forced_modality: Optional[str] = None) -> Dict:
    """Run detector synchronously (offloaded to thread pool)."""
    if detector is None:
        raise RuntimeError("Models not loaded. Please train models first.")

    if forced_modality == "image":
        return detector.image_detector.detect(path)
    elif forced_modality == "video":
        return detector.video_detector.detect(path)
    elif forced_modality == "audio":
        return detector.audio_detector.detect(path)
    else:
        return detector.detect(path)


# ─────────────────────────────────────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """System health check."""
    return {
        "status": "ok" if detector else "degraded",
        "models_loaded": {
            "image": detector.image_detector is not None if detector else False,
            "video": detector.video_detector is not None if detector else False,
            "audio": detector.audio_detector is not None if detector else False,
        },
        "uptime_s": round(time.time() - START_TIME, 1),
    }


@app.get("/models/status", tags=["System"])
async def models_status():
    """Detailed model status."""
    import torch
    return {
        "cuda_available": torch.cuda.is_available(),
        "device": str(detector.device) if detector else "N/A",
        "models": {
            "image": {"loaded": detector.image_detector is not None if detector else False},
            "video": {"loaded": detector.video_detector is not None if detector else False},
            "audio": {"loaded": detector.audio_detector is not None if detector else False},
        },
    }


@app.post("/detect", tags=["Detection"])
async def detect_any(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Auto-detect file type and run appropriate deepfake detector.
    Supports: images (jpg/png/webp), videos (mp4/avi/mov), audio (wav/mp3/flac).
    """
    ext = Path(file.filename or "file.jpg").suffix.lower()
    if ext not in ALL_EXTS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. "
                                 f"Supported: {sorted(ALL_EXTS)}")

    tmp_path = await save_upload(file, ext)
    background_tasks.add_task(cleanup_file, tmp_path)

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, run_detection_sync, tmp_path, None
        )
        if "error" in result:
            raise HTTPException(422, result["error"])
        return JSONResponse(content=result)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        logger.exception(f"Detection error: {e}")
        raise HTTPException(500, f"Detection failed: {str(e)}")


@app.post("/detect/image", tags=["Detection"])
async def detect_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Deepfake detection for images only."""
    ext = Path(file.filename or "img.jpg").suffix.lower()
    if ext not in IMAGE_EXTS:
        raise HTTPException(400, f"Not a supported image format: {ext}")
    if detector is None or detector.image_detector is None:
        raise HTTPException(503, "Image model not loaded")

    tmp_path = await save_upload(file, ext)
    background_tasks.add_task(cleanup_file, tmp_path)

    result = await asyncio.get_event_loop().run_in_executor(
        None, run_detection_sync, tmp_path, "image"
    )
    return JSONResponse(content=result)


@app.post("/detect/video", tags=["Detection"])
async def detect_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Deepfake detection for videos only."""
    ext = Path(file.filename or "vid.mp4").suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(400, f"Not a supported video format: {ext}")
    if detector is None or detector.video_detector is None:
        raise HTTPException(503, "Video model not loaded")

    tmp_path = await save_upload(file, ext)
    background_tasks.add_task(cleanup_file, tmp_path)

    result = await asyncio.get_event_loop().run_in_executor(
        None, run_detection_sync, tmp_path, "video"
    )
    return JSONResponse(content=result)


@app.post("/detect/audio", tags=["Detection"])
async def detect_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Deepfake / TTS / voice-clone detection for audio."""
    ext = Path(file.filename or "audio.wav").suffix.lower()
    if ext not in AUDIO_EXTS:
        raise HTTPException(400, f"Not a supported audio format: {ext}")
    if detector is None or detector.audio_detector is None:
        raise HTTPException(503, "Audio model not loaded")

    tmp_path = await save_upload(file, ext)
    background_tasks.add_task(cleanup_file, tmp_path)

    result = await asyncio.get_event_loop().run_in_executor(
        None, run_detection_sync, tmp_path, "audio"
    )
    return JSONResponse(content=result)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    uvicorn.run(
        "api.main:app",
        host=cfg["api"]["host"],
        port=cfg["api"]["port"],
        workers=cfg["api"]["workers"],
        reload=False,
        log_level="info",
    )
