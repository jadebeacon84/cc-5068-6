"""A small logistic classifier trained and served fully locally.

The model is intentionally tiny and interpretable: standardized acoustic
features -> linear score -> sigmoid. Weights are plain JSON, so a reviewer can
inspect exactly what runs. It is *one* local baseline, not a universal VAD.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .features import FEATURE_NAMES


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


@dataclass
class LocalModel:
    weights: np.ndarray
    bias: float
    mean: np.ndarray
    std: np.ndarray
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    threshold: float = 0.5
    metadata: dict = field(default_factory=dict)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        z = (x - self.mean) / self.std
        return sigmoid(z @ self.weights + self.bias)

    def to_dict(self) -> dict:
        return {
            "model_type": "standardized_logistic_v1",
            "feature_names": list(self.feature_names),
            "weights": [float(v) for v in np.asarray(self.weights).ravel()],
            "bias": float(self.bias),
            "mean": [float(v) for v in np.asarray(self.mean).ravel()],
            "std": [float(v) for v in np.asarray(self.std).ravel()],
            "threshold": float(self.threshold),
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                     encoding="utf-8")
        return p

    @classmethod
    def from_dict(cls, d: dict) -> "LocalModel":
        return cls(
            weights=np.asarray(d["weights"], dtype=np.float64),
            bias=float(d["bias"]),
            mean=np.asarray(d["mean"], dtype=np.float64),
            std=np.asarray(d["std"], dtype=np.float64),
            feature_names=list(d.get("feature_names", FEATURE_NAMES)),
            threshold=float(d.get("threshold", 0.5)),
            metadata=dict(d.get("metadata", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "LocalModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0,
                 lr: float = 0.25, epochs: int = 600,
                 seed: int = 7) -> tuple[LocalModel, dict]:
    """Fit with batch gradient descent and balanced class weights."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-6
    z = (x - mean) / std

    n_pos = max(1.0, y.sum())
    n_neg = max(1.0, len(y) - n_pos)
    w_pos = len(y) / (2.0 * n_pos)
    w_neg = len(y) / (2.0 * n_neg)
    cw = np.where(y > 0.5, w_pos, w_neg)

    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.02, z.shape[1])
    b = 0.0
    last_loss = None
    for _ in range(epochs):
        p = sigmoid(z @ w + b)
        err = (p - y) * cw
        grad_w = z.T @ err / len(y) + l2 * w / len(y)
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
        eps = 1e-9
        last_loss = -np.mean(cw * (y * np.log(p + eps)
                                   + (1 - y) * np.log(1 - p + eps)))
    probs = sigmoid(z @ w + b)
    pred = probs >= 0.5
    train_acc = float((pred == y).mean())
    info = {"final_weighted_loss": float(last_loss), "train_accuracy": train_acc,
            "n_frames": int(len(y)), "n_speech_frames": int(y.sum())}
    model = LocalModel(weights=w, bias=b, mean=mean, std=std, metadata=info)
    return model, info


class HeuristicScorer:
    """Transparent multi-feature fallback used when no weights file exists.

    Every cue is acoustic, never raw amplitude alone. The score is a bounded
    combination of energy-above-floor, mid-band dominance, voicing and
    syllabic modulation, penalized by sub-bass wind ratio and sharp sparse
    transients.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
        above = np.clip(x[:, idx["above_floor_db"]] / 18.0, 0, 1)
        mid = np.clip((x[:, idx["mid_band_ratio"]] - 0.25) / 0.45, 0, 1)
        voice = np.clip((x[:, idx["voicing"]] - 0.18) / 0.4, 0, 1)
        mod = np.clip((x[:, idx["mod_2_8hz_ratio"]] - 0.12) / 0.35, 0, 1)
        low = x[:, idx["low_band_ratio"]]
        flat = x[:, idx["flatness"]]
        flux = x[:, idx["spectral_flux"]]
        score = (0.22 * above + 0.24 * mid + 0.27 * voice + 0.27 * mod)
        score *= 1.0 - 0.55 * np.clip((low - 0.55) / 0.35, 0, 1)
        score *= 1.0 - 0.35 * np.clip((flat - 0.45) / 0.4, 0, 1)
        score *= 1.0 - 0.25 * np.clip((flux - 0.12) / 0.25, 0, 1)
        return np.clip(score, 0.01, 0.99)
