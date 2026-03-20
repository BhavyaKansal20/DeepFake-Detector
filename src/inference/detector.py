"""
src/inference/detector.py
─────────────────────────────────────────────────────────────────────────────
Unified inference engine for all three modalities.
  - ImageDetector  : single image → verdict
  - VideoDetector  : video file   → per-frame + overall verdict
  - AudioDetector  : audio file   → verdict
  - DeepfakeDetector : auto-dispatches based on file type
─────────────────────────────────────────────────────────────────────────────
"""

import os
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from facenet_pytorch import MTCNN
from loguru import logger

from src.models.image_model import DeepfakeImageClassifier
from src.models.video_model import DeepfakeVideoClassifier
from src.models.audio_model import DeepfakeAudioClassifier
from src.preprocessing.audio_features import AudioPreprocessor
from src.utils.helpers import get_device, probability_to_verdict, load_config


# ─────────────────────────────────────────────────────────────────────────────
#  Shared transforms
# ─────────────────────────────────────────────────────────────────────────────

IMG_TRANSFORM = A.Compose([
    A.Resize(299, 299),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

FRAME_TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# ─────────────────────────────────────────────────────────────────────────────
#  Image Detector
# ─────────────────────────────────────────────────────────────────────────────

class ImageDetector:
    """Detects deepfake faces in static images."""

    def __init__(self, model_path: str, device: torch.device, cfg: dict):
        self.device = device
        self.cfg    = cfg

        self.model = DeepfakeImageClassifier(
            architecture=cfg["image_model"]["architecture"],
            num_classes=cfg["image_model"]["num_classes"],
            pretrained=False,
        ).to(device)

        ckpt = torch.load(model_path, map_location=device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.face_detector = MTCNN(
            image_size=299, margin=40, min_face_size=80,
            device=device, keep_all=False,
        )
        logger.info(f"ImageDetector loaded from {model_path}")

    @torch.no_grad()
    def detect(self, image: Union[str, np.ndarray, Image.Image]) -> Dict:
        """
        Detect deepfake in a single image.
        Input: file path, numpy array (H,W,3 RGB), or PIL Image.
        """
        t0 = time.time()

        # ── Load image ────────────────────────────────────────────────────────
        if isinstance(image, str):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = image

        img_np = np.array(pil_img)

        # ── Try to crop face, fall back to full image ─────────────────────────
        face_detected = False
        try:
            face_tensor = self.face_detector(pil_img)
            if face_tensor is not None:
                face_np = face_tensor.permute(1, 2, 0).numpy()
                face_np = ((face_np * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
                inp     = IMG_TRANSFORM(image=face_np)["image"]
                face_detected = True
        except Exception:
            pass

        if not face_detected:
            inp = IMG_TRANSFORM(image=img_np)["image"]

        # ── Inference ──────────────────────────────────────────────────────────
        inp    = inp.unsqueeze(0).to(self.device)
        logits = self.model(inp)
        probs  = F.softmax(logits, dim=-1).cpu().numpy()[0]

        result = probability_to_verdict(float(probs[1]))
        result.update({
            "modality":      "image",
            "face_detected": face_detected,
            "latency_ms":    round((time.time() - t0) * 1000, 1),
        })
        return result

    @torch.no_grad()
    def detect_batch(self, image_paths: List[str]) -> List[Dict]:
        """Run detection on a list of image paths."""
        results = []
        for path in image_paths:
            try:
                results.append(self.detect(path))
            except Exception as e:
                results.append({"error": str(e), "path": path})
        return results


# ─────────────────────────────────────────────────────────────────────────────
#  Video Detector
# ─────────────────────────────────────────────────────────────────────────────

class VideoDetector:
    """Detects deepfake videos using temporal face analysis."""

    def __init__(self, model_path: str, device: torch.device, cfg: dict):
        self.device           = device
        self.cfg              = cfg
        self.frames_per_clip  = cfg["video_model"]["frames_per_clip"]

        self.model = DeepfakeVideoClassifier(
            feature_dim=512,
            lstm_hidden=cfg["video_model"]["lstm_hidden"],
            lstm_layers=cfg["video_model"]["lstm_layers"],
            num_classes=cfg["video_model"]["num_classes"],
            dropout=0.0,     # no dropout at inference
            pretrained=False,
        ).to(device)

        ckpt = torch.load(model_path, map_location=device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.face_detector = MTCNN(
            image_size=224, margin=40, min_face_size=80,
            device=device, keep_all=False,
        )
        logger.info(f"VideoDetector loaded from {model_path}")

    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract evenly spaced frames from video."""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, max(total - 1, 0), self.frames_per_clip, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        cap.release()
        return frames

    def _process_frame(self, frame_rgb: np.ndarray) -> torch.Tensor:
        """Detect face and transform a single frame."""
        try:
            pil_img = Image.fromarray(frame_rgb)
            face    = self.face_detector(pil_img)
            if face is not None:
                face_np = face.permute(1, 2, 0).numpy()
                face_np = ((face_np * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
                return FRAME_TRANSFORM(image=face_np)["image"]
        except Exception:
            pass
        return FRAME_TRANSFORM(image=frame_rgb)["image"]

    @torch.no_grad()
    def detect(self, video_path: str) -> Dict:
        """Full video deepfake detection with per-clip analysis."""
        t0     = time.time()
        frames = self._extract_frames(video_path)

        if not frames:
            return {"error": "Could not extract frames", "video_path": video_path}

        # Process frames
        tensors = [self._process_frame(f) for f in frames]
        clip    = torch.stack(tensors).unsqueeze(0).to(self.device)  # (1, T, C, H, W)

        logits = self.model(clip)
        probs  = F.softmax(logits, dim=-1).cpu().numpy()[0]

        # Per-frame analysis (encode each frame individually for frame scores)
        frame_scores = []
        for f in frames:
            t = self._process_frame(f).unsqueeze(0).unsqueeze(0).to(self.device)
            # Single-frame inference (feed as 1-frame clip)
            enc = self.model.encode_frames(t.expand(-1, 1, -1, -1, -1))
            # Use first layer of LSTM
            with torch.no_grad():
                out, _ = self.model.lstm(enc)
                score  = F.softmax(self.model.classifier(out[:, 0, :]), dim=-1)
                frame_scores.append(float(score[0, 1].cpu()))

        result = probability_to_verdict(float(probs[1]))
        result.update({
            "modality":    "video",
            "num_frames":  len(frames),
            "frame_scores": [round(s, 4) for s in frame_scores],
            "peak_fake_frame": int(np.argmax(frame_scores)),
            "latency_ms":  round((time.time() - t0) * 1000, 1),
        })
        return result


# ─────────────────────────────────────────────────────────────────────────────
#  Audio Detector
# ─────────────────────────────────────────────────────────────────────────────

class AudioDetector:
    """Detects synthesised / voice-cloned audio."""

    def __init__(self, model_path: str, device: torch.device, cfg: dict):
        self.device    = device
        self.cfg       = cfg
        self.sr        = cfg["audio_model"]["sample_rate"]
        self.max_dur   = cfg["audio_model"]["max_duration"]
        self.max_samp  = int(self.sr * self.max_dur)

        self.model = DeepfakeAudioClassifier(
            wav2vec_model=cfg["audio_model"]["wav2vec_model"],
            num_classes=cfg["audio_model"]["num_classes"],
            dropout=0.0,
        ).to(device)

        ckpt = torch.load(model_path, map_location=device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.preprocessor = AudioPreprocessor(
            sample_rate=self.sr,
            max_duration=self.max_dur,
            n_mfcc=cfg["audio_model"]["n_mfcc"],
            n_fft=cfg["audio_model"]["n_fft"],
            hop_length=cfg["audio_model"]["hop_length"],
        )
        logger.info(f"AudioDetector loaded from {model_path}")

    @torch.no_grad()
    def detect(self, audio_path: str) -> Dict:
        """Detect voice deepfake in an audio file."""
        t0       = time.time()
        features = self.preprocessor.extract_all(audio_path)

        if features is None:
            return {"error": "Could not load audio", "audio_path": audio_path}

        waveform = torch.tensor(features["waveform"]).unsqueeze(0).to(self.device)

        mel      = torch.tensor(features["mel"])
        mel      = (mel - mel.mean()) / (mel.std() + 1e-8)
        mel      = mel.unsqueeze(0).unsqueeze(0).to(self.device)   # (1, 1, F, T)

        logits = self.model(waveform, mel)
        probs  = F.softmax(logits, dim=-1).cpu().numpy()[0]

        result = probability_to_verdict(float(probs[1]))
        result.update({
            "modality":   "audio",
            "duration_s": len(features["waveform"]) / self.sr,
            "latency_ms": round((time.time() - t0) * 1000, 1),
        })
        return result


# ─────────────────────────────────────────────────────────────────────────────
#  Unified Detector — auto-dispatch
# ─────────────────────────────────────────────────────────────────────────────

class DeepfakeDetector:
    """
    One-stop detector that automatically selects the right module
    based on file extension. Also handles video+audio jointly for
    multi-modal video files.
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    def __init__(self, models_dir: str = "models", config_path: str = "configs/config.yaml"):
        self.cfg    = load_config(config_path)
        self.device = get_device()
        self._load_models(models_dir)

    def _load_models(self, models_dir: str) -> None:
        def _path(subdir: str) -> Optional[str]:
            p = os.path.join(models_dir, subdir, "best_model.pt")
            return p if os.path.exists(p) else None

        img_path   = _path("image_model")
        vid_path   = _path("video_model")
        aud_path   = _path("audio_model")

        self.image_detector = ImageDetector(img_path, self.device, self.cfg) if img_path else None
        self.video_detector = VideoDetector(vid_path, self.device, self.cfg) if vid_path else None
        self.audio_detector = AudioDetector(aud_path, self.device, self.cfg) if aud_path else None

        loaded = sum(x is not None for x in [
            self.image_detector, self.video_detector, self.audio_detector
        ])
        logger.info(f"DeepfakeDetector: {loaded}/3 models loaded from {models_dir}")

    def detect(self, file_path: str) -> Dict:
        """Auto-detect file type and run appropriate detector(s)."""
        ext = Path(file_path).suffix.lower()

        if ext in self.IMAGE_EXTS:
            if self.image_detector is None:
                return {"error": "Image model not loaded"}
            return self.image_detector.detect(file_path)

        elif ext in self.VIDEO_EXTS:
            results = {}
            # Video (visual)
            if self.video_detector:
                results["video"] = self.video_detector.detect(file_path)
            # Extract audio and analyse too
            if self.audio_detector:
                audio_path = self._extract_audio(file_path)
                if audio_path:
                    results["audio"] = self.audio_detector.detect(audio_path)
                    os.remove(audio_path)
            return self._fuse_results(results)

        elif ext in self.AUDIO_EXTS:
            if self.audio_detector is None:
                return {"error": "Audio model not loaded"}
            return self.audio_detector.detect(file_path)

        else:
            return {"error": f"Unsupported file type: {ext}"}

    def _extract_audio(self, video_path: str) -> Optional[str]:
        """Extract audio from video to a temp WAV file."""
        import subprocess
        try:
            tmp = tempfile.mktemp(suffix=".wav")
            cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                   "-ar", "16000", "-ac", "1", tmp, "-y", "-loglevel", "error"]
            if subprocess.run(cmd, capture_output=True).returncode == 0:
                return tmp
        except Exception as e:
            logger.debug(f"Audio extract failed: {e}")
        return None

    def _fuse_results(self, results: Dict) -> Dict:
        """Weighted average fusion of video + audio results."""
        if not results:
            return {"error": "No results to fuse"}

        weights = self.cfg["fusion"]["weights"]
        prob_fake  = 0.0
        total_w    = 0.0

        if "video" in results and "error" not in results["video"]:
            p = results["video"]["fake_probability"]
            prob_fake += weights["video"] * p
            total_w   += weights["video"]

        if "audio" in results and "error" not in results["audio"]:
            p = results["audio"]["fake_probability"]
            prob_fake += weights["audio"] * p
            total_w   += weights["audio"]

        if total_w == 0:
            return results

        prob_fake /= total_w
        fused = probability_to_verdict(prob_fake)
        fused.update({
            "modality":       "video+audio",
            "component_results": results,
            "fusion_strategy": "weighted_average",
        })
        return fused
