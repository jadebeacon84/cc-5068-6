"""Frame-level voice-activity detection.

This module only decides *frames*. It knows nothing about turning frames into
intervals; see :mod:`vad_analyzer.segments`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .features import FrameFeatures, extract_features
from .model import HeuristicScorer, LocalModel

DEFAULT_WEIGHTS = Path(__file__).with_name("model_weights.json")


@dataclass
class FrameDecision:
    probs: np.ndarray          # speech-like probability per frame
    raw: np.ndarray            # raw boolean decision at `threshold`
    threshold: float
    backend: str
    features: FrameFeatures

    @property
    def times(self) -> np.ndarray:
        return self.features.times

    @property
    def start_samples(self) -> np.ndarray:
        return self.features.start_samples


def labels_for_frames(grid, active_spans: Sequence[tuple[int, int]],
                      n_total: int, overlap: float = 0.5) -> np.ndarray:
    """Build a boolean frame label from sample ``(start, end_exclusive)`` spans.

    A frame is positive when at least ``overlap`` fraction of its analysis
    window lies inside an active span.
    """
    y = np.zeros(grid.n_frames, dtype=bool)
    win = grid.win
    for i in range(grid.n_frames):
        a, b = grid.frame_span_samples(i, n_total)
        frame_len = max(1, min(win, n_total - a))
        hit = 0
        for s0, s1 in active_spans:
            lo = max(a, s0)
            hi = min(b if b > a else a + frame_len, s1)
            if hi > lo:
                hit += hi - lo
        if hit / float(frame_len) >= overlap:
            y[i] = True
    return y


class FrameDetector:
    def __init__(self, model: Optional[LocalModel] = None,
                 weights_path: Optional[Path] = DEFAULT_WEIGHTS) -> None:
        self.backend = "heuristic-multifeature-v1"
        self.model: Optional[LocalModel] = model
        if self.model is None and weights_path and Path(weights_path).exists():
            self.model = LocalModel.load(weights_path)
            self.backend = f"local-logistic-v1:{Path(weights_path).name}"
        self._heuristic = HeuristicScorer()

    def score_features(self, fx) -> np.ndarray:
        x = getattr(fx, "x", fx)
        if self.model is not None:
            return np.clip(self.model.predict_proba(x), 0.0, 1.0)
        return self._heuristic(x)

    def detect(self, audio, threshold: Optional[float] = None) -> FrameDecision:
        fx = extract_features(audio)
        probs = self.score_features(fx)
        if threshold is None:
            threshold = self.model.threshold if self.model is not None else 0.5
        raw = probs >= float(threshold)
        return FrameDecision(probs=probs, raw=raw, threshold=float(threshold),
                             backend=self.backend, features=fx)

    def redecode(self, decision: FrameDecision, threshold: float) -> FrameDecision:
        """Re-threshold existing scores without recomputing features."""
        return FrameDecision(probs=decision.probs,
                             raw=decision.probs >= threshold,
                             threshold=float(threshold),
                             backend=decision.backend,
                             features=decision.features)
