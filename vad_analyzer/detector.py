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
from .runtime_onnx import OnnxLogisticRuntime, runtime_available

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
    def __init__(self, model: Optional[object] = None,
                 weights_path: Optional[Path] = DEFAULT_WEIGHTS,
                 runtime: str = "auto") -> None:
        self.backend = "heuristic-multifeature-v1"
        self.model: Optional[object] = model
        self.reference_model: Optional[LocalModel] = None
        requested_runtime = str(runtime).lower()
        self.runtime = requested_runtime
        if self.model is None and weights_path and Path(weights_path).exists():
            local = LocalModel.load(weights_path)
            self.reference_model = local
            if requested_runtime in ("auto", "onnx") and runtime_available():
                try:
                    self.model = OnnxLogisticRuntime(
                        weights_path=Path(weights_path))
                    self.backend = self.model.backend
                    self.runtime = "onnx"
                except Exception:
                    if requested_runtime == "onnx":
                        raise
                    self.model = local
                    self.backend = f"local-logistic-v1:{Path(weights_path).name}"
                    self.runtime = "numpy-reference"
            else:
                if requested_runtime == "onnx":
                    raise RuntimeError(
                        "指定了 ONNX Runtime 迁移路径，但 onnxruntime 或 "
                        "local_logistic_v1.onnx 不可用；请先安装 "
                        "requirements.lock 中的锁定版本。")
                self.model = local
                self.backend = f"local-logistic-v1:{Path(weights_path).name}"
                self.runtime = ("numpy" if requested_runtime == "numpy"
                                else "numpy-reference")
        elif self.model is not None and hasattr(self.model, "backend"):
            self.backend = str(self.model.backend)
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
