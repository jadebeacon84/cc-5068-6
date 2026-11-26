"""High-level local analysis wiring the real execution chain together."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .audio import AudioBuffer, load_wav
from .cuts import CutConflict, LockedCut, evaluate_locked_cuts
from .detector import FrameDetector
from .labels import (MissedRegion, find_sidecar, load_annotation,
                     reference_sample_spans, subtract_detected)
from .segments import Interval, VadConfig, merge_decision


def config_for(detector: Optional[FrameDetector] = None,
               **overrides) -> VadConfig:
    det = detector or FrameDetector()
    kw = {}
    if det.model is not None:
        t = float(det.model.threshold)
        kw["enter_threshold"] = t
        kw["exit_threshold"] = round(max(0.12, t - 0.16), 3)
    kw.update({k: v for k, v in overrides.items() if v is not None})
    return VadConfig(**kw).validated()


@dataclass
class AnalysisResult:
    audio: AudioBuffer
    config: VadConfig
    intervals: list[Interval]
    detector: FrameDetector
    probs: list
    raw: list
    times: list
    annotation: Optional[dict] = None
    missed: list[MissedRegion] = field(default_factory=list)
    conflicts: list[CutConflict] = field(default_factory=list)
    backend: str = ""

    def summary(self) -> dict:
        speech = sum(iv.end_sample - iv.start_sample for iv in self.intervals)
        return {
            "audio_file": str(self.audio.path) if self.audio.path else None,
            "sample_rate": self.audio.sample_rate,
            "duration_sec": round(self.audio.duration, 3),
            "backend": self.backend,
            "config": self.config.to_dict(),
            "n_intervals": len(self.intervals),
            "voice_activity_sec": round(speech / self.audio.sample_rate, 3),
            "voice_activity_ratio": round(
                speech / max(1, len(self.audio.samples)), 4),
            "intervals": [iv.to_dict() for iv in self.intervals],
            "missed_regions": [m.to_dict() for m in self.missed],
            "cut_conflicts": [c.to_dict() for c in self.conflicts],
        }


def analyze_audio(audio: AudioBuffer,
                  config: Optional[VadConfig] = None,
                  detector: Optional[FrameDetector] = None,
                  annotation: Optional[dict] = None,
                  cuts: Optional[list[LockedCut]] = None,
                  missed_min_ms: int = 120) -> AnalysisResult:
    det = detector or FrameDetector()
    cfg = (config or config_for(det)).validated()
    decision = det.detect(audio, threshold=cfg.enter_threshold)
    intervals = merge_decision(decision, cfg)

    missed: list[MissedRegion] = []
    if annotation is not None:
        ref = reference_sample_spans(annotation, audio.sample_rate)
        missed = subtract_detected(ref, intervals, min_ms=missed_min_ms)

    conflicts: list[CutConflict] = []
    if cuts:
        conflicts = evaluate_locked_cuts(cuts, intervals)

    return AnalysisResult(
        audio=audio, config=cfg, intervals=intervals, detector=det,
        probs=decision.probs, raw=decision.raw, times=decision.times,
        annotation=annotation, missed=missed, conflicts=conflicts,
        backend=decision.backend)


def analyze_file(path: str | Path,
                 config: Optional[VadConfig] = None,
                 detector: Optional[FrameDetector] = None,
                 annotation_path: Optional[str | Path] = None,
                 use_sidecar: bool = True,
                 cuts: Optional[list[LockedCut]] = None) -> AnalysisResult:
    audio = load_wav(path)
    ann = None
    ap = Path(annotation_path) if annotation_path else (
        find_sidecar(path) if use_sidecar else None)
    if ap is not None and ap.exists():
        ann = load_annotation(ap)
    return analyze_audio(audio, config, detector, ann, cuts)
