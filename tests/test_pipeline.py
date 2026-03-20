"""
tests/test_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Unit + integration tests for all components.
Run with:  pytest tests/ -v
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import tempfile
import pytest
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from unittest.mock import patch, MagicMock


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def cfg():
    import yaml
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def dummy_image_tensor():
    """Random (1, 3, 299, 299) image tensor."""
    return torch.randn(1, 3, 299, 299)


@pytest.fixture
def dummy_frame_tensor():
    """Random (1, 32, 3, 224, 224) video frames tensor."""
    return torch.randn(1, 32, 3, 224, 224)


@pytest.fixture
def dummy_audio():
    """16 kHz, 10s waveform + mel spectrogram."""
    wave = torch.randn(1, 160000)
    mel  = torch.randn(1, 1, 80, 1000)
    return wave, mel


@pytest.fixture
def temp_image_file():
    """Save a random RGB image to disk and return the path."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        path = f.name
    arr = np.random.randint(0, 255, (299, 299, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    yield path
    os.unlink(path)


@pytest.fixture
def temp_audio_file():
    """Save random float32 wav to disk."""
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    audio = np.random.randn(16000 * 5).astype(np.float32) * 0.1
    sf.write(path, audio, 16000)
    yield path
    os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_set_seed(self):
        from src.utils.helpers import set_seed
        set_seed(42)
        a = torch.randn(3)
        set_seed(42)
        b = torch.randn(3)
        assert torch.allclose(a, b)

    def test_get_device(self):
        from src.utils.helpers import get_device
        device = get_device()
        assert isinstance(device, torch.device)

    def test_probability_to_verdict_fake(self):
        from src.utils.helpers import probability_to_verdict
        r = probability_to_verdict(0.95)
        assert r["verdict"] == "FAKE"
        assert r["confidence"] == "HIGH"

    def test_probability_to_verdict_real(self):
        from src.utils.helpers import probability_to_verdict
        r = probability_to_verdict(0.05)
        assert r["verdict"] == "REAL"
        assert r["confidence"] == "HIGH"

    def test_probability_to_verdict_uncertain(self):
        from src.utils.helpers import probability_to_verdict
        r = probability_to_verdict(0.51)
        assert r["verdict"] == "FAKE"
        assert r["confidence"] == "LOW"

    def test_compute_metrics(self):
        from src.utils.helpers import compute_metrics
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        m = compute_metrics(y_true, y_pred, verbose=False)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)

    def test_ensure_dir(self, tmp_path):
        from src.utils.helpers import ensure_dir
        new_dir = str(tmp_path / "a" / "b" / "c")
        ensure_dir(new_dir)
        assert os.path.isdir(new_dir)


# ─────────────────────────────────────────────────────────────────────────────
#  Image Model
# ─────────────────────────────────────────────────────────────────────────────

class TestImageModel:
    def test_efficientnet_forward(self, dummy_image_tensor):
        from src.models.image_model import DeepfakeImageClassifier
        model = DeepfakeImageClassifier("efficientnet_b4", pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(dummy_image_tensor)
        assert out.shape == (1, 2), f"Expected (1,2), got {out.shape}"

    def test_vit_forward(self, dummy_image_tensor):
        from src.models.image_model import DeepfakeImageClassifier
        model = DeepfakeImageClassifier("vit_base_patch16_224", pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)

    def test_output_probabilities_sum_to_one(self, dummy_image_tensor):
        import torch.nn.functional as F
        from src.models.image_model import DeepfakeImageClassifier
        model = DeepfakeImageClassifier("efficientnet_b4", pretrained=False)
        model.eval()
        with torch.no_grad():
            logits = model(dummy_image_tensor)
            probs  = F.softmax(logits, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)

    def test_dataset_getitem(self, tmp_path):
        from src.models.image_model import FaceImageDataset
        # Create 4 dummy images
        samples = []
        for i in range(4):
            p = str(tmp_path / f"img_{i}.jpg")
            arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            Image.fromarray(arr).save(p)
            samples.append((p, i % 2))
        ds = FaceImageDataset(samples, split="val")
        img, label = ds[0]
        assert img.shape == (3, 299, 299)
        assert label.dtype == torch.int64

    def test_unsupported_architecture_raises(self):
        from src.models.image_model import DeepfakeImageClassifier
        with pytest.raises(ValueError):
            DeepfakeImageClassifier("resnet999")


# ─────────────────────────────────────────────────────────────────────────────
#  Video Model
# ─────────────────────────────────────────────────────────────────────────────

class TestVideoModel:
    def test_forward_shape(self, dummy_frame_tensor):
        from src.models.video_model import DeepfakeVideoClassifier
        model = DeepfakeVideoClassifier(
            feature_dim=512, lstm_hidden=256, lstm_layers=1,
            pretrained=False, dropout=0.0
        )
        model.eval()
        with torch.no_grad():
            out = model(dummy_frame_tensor)
        assert out.shape == (1, 2)

    def test_encode_frames_shape(self, dummy_frame_tensor):
        from src.models.video_model import DeepfakeVideoClassifier
        model = DeepfakeVideoClassifier(
            feature_dim=256, lstm_hidden=128, lstm_layers=1,
            pretrained=False, dropout=0.0
        )
        model.eval()
        with torch.no_grad():
            enc = model.encode_frames(dummy_frame_tensor)
        B, T, D = enc.shape
        assert B == 1 and T == 32 and D == 256

    def test_video_dataset_empty_dir(self, tmp_path):
        from src.models.video_model import VideoFaceDataset
        empty_meta = {
            "vid1": {"face_dir": str(tmp_path), "label": 0},
        }
        ds = VideoFaceDataset(empty_meta, split="val", frames_per_clip=4)
        frames, label = ds[0]
        assert frames.shape == (4, 3, 224, 224)


# ─────────────────────────────────────────────────────────────────────────────
#  Audio Model
# ─────────────────────────────────────────────────────────────────────────────

class TestAudioModel:
    def test_forward_shape(self, dummy_audio):
        from src.models.audio_model import DeepfakeAudioClassifier
        wave, mel = dummy_audio
        model = DeepfakeAudioClassifier(
            wav2vec_model="facebook/wav2vec2-base",
            num_classes=2, feature_dim=64, dropout=0.0
        )
        model.eval()
        with torch.no_grad():
            out = model(wave, mel)
        assert out.shape == (1, 2)

    def test_lcnn_forward(self):
        from src.models.audio_model import LCNN
        model = LCNN(feature_dim=128)
        model.eval()
        x = torch.randn(2, 1, 80, 400)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 128)

    def test_audio_dataset(self, tmp_path):
        from src.models.audio_model import AudioDeepfakeDataset
        # Create dummy .npz files
        meta = {}
        for i in range(3):
            npz_path = str(tmp_path / f"sample_{i}.npz")
            np.savez(
                npz_path,
                waveform=np.random.randn(160000).astype(np.float32),
                mel=np.random.randn(80, 1000).astype(np.float32),
                mfcc=np.random.randn(120, 500).astype(np.float32),
                lfcc=np.random.randn(40, 500).astype(np.float32),
            )
            meta[f"sample_{i}"] = {"npz_path": npz_path, "label": i % 2}
        ds = AudioDeepfakeDataset(meta)
        wave, mel, label = ds[0]
        assert wave.shape == (160000,)
        assert mel.shape[0] == 1
        assert label.dtype == torch.int64


# ─────────────────────────────────────────────────────────────────────────────
#  Audio Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

class TestAudioPreprocessor:
    def test_load_waveform(self, temp_audio_file):
        from src.preprocessing.audio_features import AudioPreprocessor
        proc = AudioPreprocessor(sample_rate=16000, max_duration=5.0)
        wave = proc.load_waveform(temp_audio_file)
        assert wave is not None
        assert len(wave) == 16000 * 5

    def test_extract_mfcc(self, temp_audio_file):
        from src.preprocessing.audio_features import AudioPreprocessor
        proc = AudioPreprocessor()
        wave = proc.load_waveform(temp_audio_file)
        mfcc = proc.extract_mfcc(wave)
        assert mfcc.shape[0] == 40 * 3   # mfcc + delta + delta2

    def test_extract_mel(self, temp_audio_file):
        from src.preprocessing.audio_features import AudioPreprocessor
        proc = AudioPreprocessor()
        wave = proc.load_waveform(temp_audio_file)
        mel  = proc.extract_mel_spectrogram(wave)
        assert mel.shape[0] == 80

    def test_extract_all_returns_all_keys(self, temp_audio_file):
        from src.preprocessing.audio_features import AudioPreprocessor
        proc    = AudioPreprocessor()
        result  = proc.extract_all(temp_audio_file)
        assert result is not None
        for key in ("waveform", "mfcc", "mel", "lfcc"):
            assert key in result


# ─────────────────────────────────────────────────────────────────────────────
#  Ensemble Fusion
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsembleFusion:
    def test_weighted_average_all_fake(self):
        from src.fusion.ensemble import EnsembleFusion
        fusion = EnsembleFusion(strategy="weighted_average")
        result = fusion.fuse({"image": 0.9, "video": 0.95, "audio": 0.88})
        assert result["verdict"] == "FAKE"
        assert result["confidence"] == "HIGH"

    def test_weighted_average_all_real(self):
        from src.fusion.ensemble import EnsembleFusion
        fusion = EnsembleFusion(strategy="weighted_average")
        result = fusion.fuse({"image": 0.05, "video": 0.08, "audio": 0.03})
        assert result["verdict"] == "REAL"

    def test_voting_fusion(self):
        from src.fusion.ensemble import EnsembleFusion
        fusion = EnsembleFusion(strategy="voting")
        result = fusion.fuse({"image": 0.9, "video": 0.2, "audio": 0.8})
        # 2/3 votes = FAKE
        assert result["verdict"] == "FAKE"

    def test_single_modality(self):
        from src.fusion.ensemble import EnsembleFusion
        fusion = EnsembleFusion(strategy="weighted_average")
        result = fusion.fuse({"video": 0.8})
        assert result["verdict"] == "FAKE"
        assert result["modality"] == "video"

    def test_empty_predictions(self):
        from src.fusion.ensemble import EnsembleFusion
        fusion = EnsembleFusion()
        result = fusion.fuse({})
        assert "error" in result

    def test_meta_classifier_trains(self):
        from src.fusion.ensemble import train_meta_classifier
        N = 100
        image = np.random.dirichlet([1, 1], N).astype(np.float32)
        video = np.random.dirichlet([1, 1], N).astype(np.float32)
        audio = np.random.dirichlet([1, 1], N).astype(np.float32)
        labels = np.random.randint(0, 2, N)
        meta = train_meta_classifier(image, video, audio, labels, epochs=5)
        probs = meta.predict_proba([image[0], video[0], audio[0]])
        assert len(probs) == 2
        assert abs(probs.sum() - 1.0) < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
#  API (mock detector)
# ─────────────────────────────────────────────────────────────────────────────

class TestAPI:
    @pytest.fixture(autouse=True)
    def setup_app(self):
        from fastapi.testclient import TestClient
        import api.main as api_module

        # Mock the detector
        mock_detector = MagicMock()
        mock_detector.detect.return_value = {
            "verdict": "FAKE",
            "confidence": "HIGH",
            "fake_probability": 0.93,
            "real_probability": 0.07,
            "modality": "image",
            "latency_ms": 42.0,
        }
        mock_detector.image_detector = MagicMock()
        mock_detector.image_detector.detect.return_value = {
            "verdict": "REAL",
            "confidence": "HIGH",
            "fake_probability": 0.03,
            "real_probability": 0.97,
            "modality": "image",
            "latency_ms": 30.0,
        }

        api_module.detector = mock_detector
        self.client = TestClient(api_module.app)

    def test_health_endpoint(self):
        r = self.client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data

    def test_models_status_endpoint(self):
        r = self.client.get("/models/status")
        assert r.status_code == 200

    def test_detect_image_endpoint(self, temp_image_file):
        with open(temp_image_file, "rb") as f:
            r = self.client.post(
                "/detect/image",
                files={"file": ("test.jpg", f, "image/jpeg")},
            )
        assert r.status_code == 200
        data = r.json()
        assert data["verdict"] in ("REAL", "FAKE")

    def test_detect_unsupported_type(self):
        r = self.client.post(
            "/detect",
            files={"file": ("test.xyz", b"data", "application/octet-stream")},
        )
        assert r.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
#  Integration: build_image_dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetBuilders:
    def test_build_image_dataset(self, tmp_path):
        from src.models.image_model import build_image_dataset

        # Create dummy real/fake dirs
        real_dir = tmp_path / "real"
        fake_dir = tmp_path / "fake"
        real_dir.mkdir()
        fake_dir.mkdir()

        for d, label in [(real_dir, "real"), (fake_dir, "fake")]:
            for i in range(20):
                arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                Image.fromarray(arr).save(str(d / f"{label}_{i}.jpg"))

        train, val, test = build_image_dataset(str(real_dir), [str(fake_dir)])
        assert len(train) + len(val) + len(test) == 40
        assert len(train) > len(val)

    def test_build_video_splits(self, tmp_path):
        from src.models.video_model import build_video_splits

        # Write a dummy metadata JSON
        meta = {str(i): {"face_dir": str(tmp_path), "label": i % 2}
                for i in range(20)}
        meta_path = str(tmp_path / "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        train, val, test = build_video_splits([meta_path])
        assert len(train) + len(val) + len(test) == 20
