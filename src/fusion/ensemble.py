"""
src/fusion/ensemble.py
─────────────────────────────────────────────────────────────────────────────
Ensemble strategies for combining predictions from image, video, and audio
models. Supports:
  1. Weighted Average  — configurable weights per modality
  2. Majority Voting   — hard vote across models
  3. Learned Meta      — small MLP trained on val-set logits
─────────────────────────────────────────────────────────────────────────────
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
#  Strategy 1 — Weighted Average
# ─────────────────────────────────────────────────────────────────────────────

def weighted_average_fusion(
    predictions: Dict[str, float],   # modality → fake probability
    weights: Dict[str, float],        # modality → weight
) -> float:
    """
    Fuse fake-probabilities from multiple modalities using weighted average.
    Only modalities present in predictions are included.
    """
    total_w = 0.0
    fused   = 0.0
    for modality, prob in predictions.items():
        w = weights.get(modality, 1.0)
        fused   += w * prob
        total_w += w
    return fused / max(total_w, 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
#  Strategy 2 — Majority Voting
# ─────────────────────────────────────────────────────────────────────────────

def majority_vote_fusion(
    predictions: Dict[str, float],
    threshold: float = 0.5,
) -> float:
    """
    Each modality casts a binary vote (FAKE if prob >= threshold).
    Returns 1.0 if majority vote is FAKE, 0.0 otherwise.
    """
    votes = [1 if p >= threshold else 0 for p in predictions.values()]
    majority = int(sum(votes) > len(votes) / 2)
    return float(majority)


# ─────────────────────────────────────────────────────────────────────────────
#  Strategy 3 — Learned Meta-Classifier
# ─────────────────────────────────────────────────────────────────────────────

class MetaClassifier(nn.Module):
    """
    Small MLP that learns optimal fusion weights from validation-set
    predictions of each base model.

    Input:  concatenated softmax probabilities from each model
    Output: final 2-class prediction
    """

    def __init__(self, n_models: int = 3, n_classes: int = 2):
        super().__init__()
        input_dim = n_models * n_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, probs_list: List[np.ndarray]) -> np.ndarray:
        """
        probs_list: list of [real_prob, fake_prob] arrays from each model.
        Returns: [real_prob, fake_prob] fused estimate.
        """
        x = torch.tensor(np.concatenate(probs_list)).unsqueeze(0).float()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1).numpy()[0]


def train_meta_classifier(
    image_preds:  np.ndarray,   # (N, 2) — val-set image model probs
    video_preds:  np.ndarray,   # (N, 2)
    audio_preds:  np.ndarray,   # (N, 2)
    labels:       np.ndarray,   # (N,)  — ground truth
    epochs: int   = 50,
    lr: float     = 1e-3,
) -> MetaClassifier:
    """
    Train the meta-classifier on held-out validation predictions.
    Call this after training all base models.
    """
    # Stack inputs
    X = np.concatenate([image_preds, video_preds, audio_preds], axis=1)  # (N, 6)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    model     = MetaClassifier(n_models=3, n_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            acc = (logits.argmax(1) == y).float().mean().item()
            logger.info(f"Meta-Classifier epoch {epoch+1}/{epochs} | loss={loss.item():.4f} acc={acc:.4f}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Unified Fusion Interface
# ─────────────────────────────────────────────────────────────────────────────

class EnsembleFusion:
    """
    Drop-in fusion layer that wraps all three strategies.
    Configure via the fusion section of config.yaml.
    """

    STRATEGIES = ("weighted_average", "voting", "learned_meta")

    def __init__(
        self,
        strategy: str = "weighted_average",
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.5,
        meta_model: Optional[MetaClassifier] = None,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from {self.STRATEGIES}")

        self.strategy   = strategy
        self.weights    = weights or {"image": 0.35, "video": 0.40, "audio": 0.25}
        self.threshold  = threshold
        self.meta_model = meta_model

        logger.info(f"EnsembleFusion: strategy={strategy} | weights={self.weights}")

    def fuse(
        self,
        predictions: Dict[str, float],   # modality → fake_probability
    ) -> Dict:
        """
        Fuse predictions from available modalities.
        Returns a result dict with verdict, confidence, and probabilities.
        """
        if not predictions:
            return {"error": "No predictions to fuse"}

        if self.strategy == "weighted_average":
            fake_prob = weighted_average_fusion(predictions, self.weights)

        elif self.strategy == "voting":
            fake_prob = majority_vote_fusion(predictions, self.threshold)

        elif self.strategy == "learned_meta":
            if self.meta_model is None:
                logger.warning("Meta model not trained — falling back to weighted average")
                fake_prob = weighted_average_fusion(predictions, self.weights)
            else:
                # Build full 6-dim input (zeros for missing modalities)
                probs_list = []
                for m in ["image", "video", "audio"]:
                    p = predictions.get(m, 0.5)
                    probs_list.append(np.array([1 - p, p]))
                fused_probs = self.meta_model.predict_proba(probs_list)
                fake_prob   = float(fused_probs[1])

        else:
            fake_prob = 0.5

        # Build result
        verdict    = "FAKE" if fake_prob >= self.threshold else "REAL"
        margin     = abs(fake_prob - 0.5)
        confidence = "HIGH" if margin > 0.35 else "MEDIUM" if margin > 0.15 else "LOW"

        return {
            "verdict":          verdict,
            "confidence":       confidence,
            "fake_probability": round(float(fake_prob), 4),
            "real_probability": round(float(1 - fake_prob), 4),
            "modality":         "+".join(sorted(predictions.keys())),
            "fusion_strategy":  self.strategy,
            "component_probs":  {k: round(v, 4) for k, v in predictions.items()},
        }

    @classmethod
    def from_config(cls, cfg: dict) -> "EnsembleFusion":
        fc = cfg.get("fusion", {})
        return cls(
            strategy=fc.get("strategy", "weighted_average"),
            weights=fc.get("weights", {}),
            threshold=fc.get("threshold", 0.5),
        )
