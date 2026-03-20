"""
src/preprocessing/audio_features.py
─────────────────────────────────────────────────────────────────────────────
Audio preprocessing for deepfake detection.
Extracts: raw waveforms (for wav2vec2), MFCCs, mel-spectrograms, LFCCs.
Supports ASVspoof 2019/2021 and FakeAVCeleb datasets.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import librosa
import soundfile as sf
from tqdm import tqdm
from loguru import logger

from src.utils.helpers import ensure_dir, list_files


# ─────────────────────────────────────────────────────────────────────────────
#  Audio Preprocessor
# ─────────────────────────────────────────────────────────────────────────────

class AudioPreprocessor:
    """
    Loads audio, resamples to target rate, pads/trims, and extracts
    multiple feature representations.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        max_duration: float = 10.0,          # seconds
        n_mfcc: int = 40,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 80,
        win_length: int = 400,
    ):
        self.sample_rate  = sample_rate
        self.max_samples  = int(sample_rate * max_duration)
        self.n_mfcc       = n_mfcc
        self.n_fft        = n_fft
        self.hop_length   = hop_length
        self.n_mels       = n_mels
        self.win_length   = win_length

    # ── Waveform loading ──────────────────────────────────────────────────────

    def load_waveform(self, audio_path: str) -> Optional[np.ndarray]:
        """Load and resample audio to mono waveform."""
        try:
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            waveform = self._pad_or_trim(waveform)
            return waveform.astype(np.float32)
        except Exception as e:
            logger.debug(f"Failed to load {audio_path}: {e}")
            return None

    def _pad_or_trim(self, waveform: np.ndarray) -> np.ndarray:
        """Pad with zeros or trim to max_samples."""
        if len(waveform) < self.max_samples:
            pad_len = self.max_samples - len(waveform)
            waveform = np.pad(waveform, (0, pad_len), mode="constant")
        else:
            waveform = waveform[: self.max_samples]
        return waveform

    # ── Feature extraction ────────────────────────────────────────────────────

    def extract_mfcc(self, waveform: np.ndarray) -> np.ndarray:
        """MFCC + delta + delta-delta → shape (3*n_mfcc, T)."""
        mfcc   = librosa.feature.mfcc(
            y=waveform, sr=self.sample_rate,
            n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length
        )
        delta  = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        return np.concatenate([mfcc, delta, delta2], axis=0).astype(np.float32)

    def extract_mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """Log-mel spectrogram → shape (n_mels, T)."""
        mel = librosa.feature.melspectrogram(
            y=waveform, sr=self.sample_rate,
            n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
        )
        return librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    def extract_lfcc(self, waveform: np.ndarray) -> np.ndarray:
        """
        Linear Frequency Cepstral Coefficients (LFCC).
        Used in ASVspoof baselines — effective for detecting TTS/VC artifacts.
        """
        # Linear filterbank via STFT
        stft  = np.abs(librosa.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)

        # Linear filterbank (40 filters)
        n_filters   = 40
        min_freq    = 0
        max_freq    = self.sample_rate / 2
        filter_edges = np.linspace(min_freq, max_freq, n_filters + 2)

        filterbank = np.zeros((n_filters, len(freqs)))
        for i in range(n_filters):
            low, mid, high = filter_edges[i], filter_edges[i + 1], filter_edges[i + 2]
            filterbank[i] = np.maximum(
                0,
                np.minimum(
                    (freqs - low) / (mid - low + 1e-8),
                    (high - freqs) / (high - mid + 1e-8),
                ),
            )

        linear_spec  = filterbank @ stft
        log_linear   = np.log(linear_spec + 1e-6)
        lfcc         = np.real(np.fft.dct(log_linear, axis=0))[:self.n_mfcc]
        return lfcc.astype(np.float32)

    def extract_all(self, audio_path: str) -> Optional[Dict[str, np.ndarray]]:
        """Extract all feature types from a single audio file."""
        waveform = self.load_waveform(audio_path)
        if waveform is None:
            return None
        return {
            "waveform": waveform,
            "mfcc":     self.extract_mfcc(waveform),
            "mel":      self.extract_mel_spectrogram(waveform),
            "lfcc":     self.extract_lfcc(waveform),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset-specific processors
# ─────────────────────────────────────────────────────────────────────────────

class ASVspoof_Preprocessor:
    """
    ASVspoof 2019 LA track preprocessing.
    Directory structure:
      data/raw/asvspoof2019/
        LA/
          ASVspoof2019_LA_train/flac/
          ASVspoof2019_LA_dev/flac/
          ASVspoof2019_LA_eval/flac/
          ASVspoof2019_LA_cm_protocols/
    """

    PROTOCOL_COLS = ["speaker", "filename", "env", "attack", "label"]

    def __init__(self, cfg: dict):
        self.cfg        = cfg
        self.raw_root   = os.path.join(cfg["paths"]["raw"], "asvspoof2019", "LA")
        self.out_dir    = cfg["paths"]["audio"]
        self.preprocessor = AudioPreprocessor(
            sample_rate  = cfg["audio_model"]["sample_rate"],
            max_duration = cfg["audio_model"]["max_duration"],
            n_mfcc       = cfg["audio_model"]["n_mfcc"],
            n_fft        = cfg["audio_model"]["n_fft"],
            hop_length   = cfg["audio_model"]["hop_length"],
        )
        ensure_dir(self.out_dir)

    def _parse_protocol(self, split: str) -> List[Dict]:
        """Parse ASVspoof protocol file."""
        protocol_map = {
            "train": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
            "dev":   "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
            "eval":  "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
        }
        protocol_path = os.path.join(self.raw_root, protocol_map[split])
        items = []
        with open(protocol_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    label = 0 if parts[4] == "bonafide" else 1
                    items.append({"filename": parts[1], "label": label, "attack": parts[3]})
        return items

    def process_split(self, split: str) -> Dict:
        """Process one split (train/dev/eval) and save .npz features."""
        items    = self._parse_protocol(split)
        flac_dir = os.path.join(self.raw_root, f"ASVspoof2019_LA_{split}", "flac")
        out_dir  = os.path.join(self.out_dir, split)
        ensure_dir(out_dir)

        metadata = {}
        logger.info(f"Processing ASVspoof2019 {split} split: {len(items)} files")

        for item in tqdm(items, desc=f"ASVspoof {split}"):
            audio_path = os.path.join(flac_dir, f"{item['filename']}.flac")
            if not os.path.exists(audio_path):
                continue

            features = self.preprocessor.extract_all(audio_path)
            if features is None:
                continue

            # Save compressed numpy archive
            npz_path = os.path.join(out_dir, f"{item['filename']}.npz")
            np.savez_compressed(npz_path, **features)

            metadata[item["filename"]] = {
                "npz_path":   npz_path,
                "audio_path": audio_path,
                "label":      item["label"],
                "attack":     item["attack"],
            }

        return metadata

    def run(self) -> None:
        all_meta = {}
        for split in ["train", "dev", "eval"]:
            try:
                meta = self.process_split(split)
                all_meta[split] = meta
                logger.info(f"ASVspoof {split}: {len(meta)} files processed")
            except FileNotFoundError as e:
                logger.warning(f"Skipping {split}: {e}")

        meta_path = os.path.join(self.cfg["paths"]["processed"], "asvspoof_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(all_meta, f, indent=2)
        logger.info(f"ASVspoof metadata saved → {meta_path}")


class FakeAVCeleb_Preprocessor:
    """
    FakeAVCeleb multi-modal preprocessing.
    Extracts audio tracks from video files for audio-only analysis.
    """

    def __init__(self, cfg: dict):
        self.cfg         = cfg
        self.raw_root    = os.path.join(cfg["paths"]["raw"], "fakeavceleb")
        self.out_dir     = os.path.join(cfg["paths"]["audio"], "fakeavceleb")
        self.preprocessor = AudioPreprocessor()
        ensure_dir(self.out_dir)

    def extract_audio_from_video(self, video_path: str, out_path: str) -> bool:
        """Use ffmpeg to extract audio track from video."""
        import subprocess
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            out_path, "-y", "-loglevel", "error"
        ]
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0

    def run(self) -> None:
        metadata = {}
        video_files = list_files(self.raw_root, [".mp4", ".avi"])
        logger.info(f"FakeAVCeleb: {len(video_files)} videos found")

        for video_path in tqdm(video_files, desc="FakeAVCeleb audio"):
            stem       = Path(video_path).stem
            label      = 0 if "real" in video_path.lower() else 1
            wav_path   = os.path.join(self.out_dir, f"{stem}.wav")
            npz_path   = os.path.join(self.out_dir, f"{stem}.npz")

            if not self.extract_audio_from_video(video_path, wav_path):
                continue

            features = self.preprocessor.extract_all(wav_path)
            if features is None:
                continue

            np.savez_compressed(npz_path, **features)
            metadata[stem] = {"npz_path": npz_path, "label": label}
            os.remove(wav_path)  # clean up intermediate wav

        meta_path = os.path.join(self.cfg["paths"]["processed"], "fakeavceleb_audio_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"FakeAVCeleb audio metadata saved → {meta_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["asvspoof", "fakeavceleb", "all"], default="all")
    parser.add_argument("--config",  default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.dataset in ("asvspoof", "all"):
        ASVspoof_Preprocessor(cfg).run()
    if args.dataset in ("fakeavceleb", "all"):
        FakeAVCeleb_Preprocessor(cfg).run()
