"""
src/preprocessing/extract_frames.py
─────────────────────────────────────────────────────────────────────────────
Extract frames from videos at a target FPS, then detect + crop faces
using MTCNN. Supports FaceForensics++, DFDC, Celeb-DF formats.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from loguru import logger
from facenet_pytorch import MTCNN
import torch

from src.utils.helpers import ensure_dir, get_device, list_files


# ─────────────────────────────────────────────────────────────────────────────
#  Frame Extractor
# ─────────────────────────────────────────────────────────────────────────────

class FrameExtractor:
    """
    Extracts frames from video files at a controlled FPS.
    Saves as JPEG with optional quality setting.
    """

    def __init__(
        self,
        target_fps: int = 5,
        max_frames: int = 32,
        output_size: Tuple[int, int] = (224, 224),
        quality: int = 95,
    ):
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.output_size = output_size
        self.quality = quality

    def extract(self, video_path: str, output_dir: str) -> List[str]:
        """
        Extract frames from a single video.
        Returns list of saved frame paths.
        """
        ensure_dir(output_dir)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.warning(f"Cannot open video: {video_path}")
            return []

        video_fps   = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step  = max(1, int(video_fps / self.target_fps))

        saved_paths = []
        frame_idx   = 0
        saved_count = 0

        while cap.isOpened() and saved_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, self.output_size)
                out_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
                Image.fromarray(frame_resized).save(out_path, quality=self.quality)
                saved_paths.append(out_path)
                saved_count += 1

            frame_idx += 1

        cap.release()
        return saved_paths

    def extract_batch(
        self,
        video_dir: str,
        output_root: str,
        label: int,
        extensions: List[str] = [".mp4", ".avi", ".mov"],
    ) -> dict:
        """Extract frames from all videos in a directory."""
        video_files = list_files(video_dir, extensions)
        metadata = {}

        logger.info(f"Extracting frames from {len(video_files)} videos in {video_dir}")

        for video_path in tqdm(video_files, desc="Extracting frames"):
            vid_name  = Path(video_path).stem
            out_dir   = os.path.join(output_root, vid_name)
            frames    = self.extract(video_path, out_dir)
            metadata[vid_name] = {
                "video_path": video_path,
                "frame_dir":  out_dir,
                "num_frames": len(frames),
                "label":      label,
            }

        return metadata


# ─────────────────────────────────────────────────────────────────────────────
#  Face Cropper
# ─────────────────────────────────────────────────────────────────────────────

class FaceCropper:
    """
    Detects faces in frames using MTCNN and saves cropped face regions.
    Applies margin around the detected bounding box for context.
    """

    def __init__(
        self,
        image_size: int = 299,
        margin: int = 40,
        min_face_size: int = 80,
        device: Optional[torch.device] = None,
    ):
        self.image_size = image_size
        self.margin     = margin
        self.device     = device or get_device()
        self.detector   = MTCNN(
            image_size=image_size,
            margin=margin,
            min_face_size=min_face_size,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device,
            keep_all=False,
        )

    def crop_face(self, image_path: str, output_path: str) -> bool:
        """
        Detect and crop the primary face from an image.
        Returns True if a face was found and saved.
        """
        try:
            img  = Image.open(image_path).convert("RGB")
            face = self.detector(img)

            if face is None:
                return False

            # face is a normalised tensor — denorm and save
            face_np = face.permute(1, 2, 0).numpy()
            face_np = ((face_np * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(face_np).save(output_path, quality=95)
            return True

        except Exception as e:
            logger.debug(f"Face crop failed for {image_path}: {e}")
            return False

    def crop_batch(
        self,
        frames_root: str,
        output_root: str,
        metadata: dict,
    ) -> dict:
        """
        Crop faces from all frame directories described in metadata dict.
        Skips videos where fewer than 30% of frames yield a face.
        """
        ensure_dir(output_root)
        augmented_meta = {}

        for vid_name, info in tqdm(metadata.items(), desc="Cropping faces"):
            frame_dir  = info["frame_dir"]
            face_dir   = os.path.join(output_root, vid_name)
            ensure_dir(face_dir)

            frame_paths = sorted(Path(frame_dir).glob("*.jpg"))
            success = 0

            for fp in frame_paths:
                out_path = os.path.join(face_dir, fp.name)
                if self.crop_face(str(fp), out_path):
                    success += 1

            ratio = success / max(len(frame_paths), 1)
            if ratio >= 0.3:
                augmented_meta[vid_name] = {**info, "face_dir": face_dir, "face_ratio": ratio}
            else:
                logger.debug(f"Skipped {vid_name} — face detection ratio: {ratio:.2f}")

        logger.info(f"Face cropping complete: {len(augmented_meta)}/{len(metadata)} videos passed")
        return augmented_meta


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset-specific preprocessors
# ─────────────────────────────────────────────────────────────────────────────

class FF_Preprocessor:
    """
    FaceForensics++ preprocessing.
    Expected raw structure:
      data/raw/ff++/
        original_sequences/youtube/
        manipulated_sequences/Deepfakes/
        manipulated_sequences/Face2Face/
        ...
    """

    FAKE_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

    def __init__(self, cfg: dict):
        self.cfg        = cfg
        self.raw_root   = cfg["paths"]["raw"]
        self.out_frames = cfg["paths"]["frames"]
        self.out_faces  = cfg["paths"]["faces"]
        self.compression = cfg["video_model"]["dataset"]["compression"]
        self.extractor  = FrameExtractor(
            target_fps=5,
            max_frames=cfg["video_model"]["frames_per_clip"],
            output_size=(cfg["video_model"]["frame_size"],) * 2,
        )
        self.cropper = FaceCropper(image_size=cfg["image_model"]["input_size"])

    def run(self) -> None:
        metadata = {}

        # ── Real videos ──────────────────────────────────────────────────────
        real_dir = os.path.join(self.raw_root, "ff++", "original_sequences",
                                "youtube", self.compression, "videos")
        meta_real = self.extractor.extract_batch(
            real_dir, os.path.join(self.out_frames, "real"), label=0
        )
        metadata.update(meta_real)

        # ── Fake videos ──────────────────────────────────────────────────────
        for method in self.FAKE_METHODS:
            fake_dir = os.path.join(
                self.raw_root, "ff++", "manipulated_sequences",
                method, self.compression, "videos"
            )
            if not os.path.exists(fake_dir):
                logger.warning(f"FF++ method dir not found: {fake_dir}")
                continue
            meta_fake = self.extractor.extract_batch(
                fake_dir, os.path.join(self.out_frames, f"fake_{method}"), label=1
            )
            metadata.update(meta_fake)

        # ── Face cropping ─────────────────────────────────────────────────────
        face_meta = self.cropper.crop_batch(
            self.out_frames, self.out_faces, metadata
        )

        # ── Save metadata ─────────────────────────────────────────────────────
        meta_path = os.path.join(self.cfg["paths"]["processed"], "ff_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(face_meta, f, indent=2)
        logger.info(f"FF++ metadata saved → {meta_path}")


class DFDC_Preprocessor:
    """
    Facebook DFDC preprocessing.
    Reads the official metadata.json files per chunk.
    """

    def __init__(self, cfg: dict):
        self.cfg       = cfg
        self.raw_root  = os.path.join(cfg["paths"]["raw"], "dfdc")
        self.out_frames = cfg["paths"]["frames"]
        self.out_faces  = cfg["paths"]["faces"]
        self.extractor  = FrameExtractor(target_fps=3, max_frames=32)
        self.cropper    = FaceCropper()

    def run(self) -> None:
        metadata = {}

        chunk_dirs = sorted(Path(self.raw_root).glob("dfdc_train_part_*"))
        logger.info(f"Found {len(chunk_dirs)} DFDC chunks")

        for chunk in tqdm(chunk_dirs, desc="DFDC chunks"):
            meta_file = chunk / "metadata.json"
            if not meta_file.exists():
                continue
            with open(meta_file) as f:
                chunk_meta = json.load(f)

            for filename, info in chunk_meta.items():
                label     = 1 if info["label"] == "FAKE" else 0
                vid_path  = str(chunk / filename)
                vid_stem  = Path(filename).stem
                out_dir   = os.path.join(self.out_frames, "dfdc", vid_stem)
                frames    = self.extractor.extract(vid_path, out_dir)
                metadata[vid_stem] = {
                    "video_path": vid_path,
                    "frame_dir":  out_dir,
                    "num_frames": len(frames),
                    "label":      label,
                    "original":   info.get("original", ""),
                }

        face_meta = self.cropper.crop_batch(
            self.out_frames, self.out_faces, metadata
        )

        meta_path = os.path.join(self.cfg["paths"]["processed"], "dfdc_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(face_meta, f, indent=2)
        logger.info(f"DFDC metadata saved → {meta_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser(description="Preprocess deepfake video datasets")
    parser.add_argument("--dataset", choices=["ff++", "dfdc", "all"], default="all")
    parser.add_argument("--config",  default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.dataset in ("ff++", "all"):
        FF_Preprocessor(cfg).run()
    if args.dataset in ("dfdc", "all"):
        DFDC_Preprocessor(cfg).run()
