"""Framework-free voice-interval kernel used by CLI, tests and the Tk window.

The GUI keeps widget state, but all interval recomputation, block-size
contracts and editor readback go through this object so a button cannot fake a
result independently of the VAD core.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .audio import AudioBuffer
from .cuts import CutConflict, LockedCut, evaluate_locked_cuts
from .detector import FrameDetector
from .exporter import ClipVerification, export_interval
from .features import extract_features
from .labels import MissedRegion, reference_sample_spans, subtract_detected
from .pipeline import AnalysisResult, config_for
from .segments import Interval, VadConfig, merge_decision
from .stream import StreamingVad


@dataclass
class HeadReadback:
    path: Path
    verification: ClipVerification
    interval: Interval

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "interval": self.interval.to_dict(),
            "verification": self.verification.to_dict(),
        }


class VoiceIntervalKernel:
    def __init__(self, detector: Optional[FrameDetector] = None,
                 config: Optional[VadConfig] = None) -> None:
        self.detector = detector or FrameDetector()
        self.config = (config or config_for(self.detector)).validated()

    def analyze(self, audio: AudioBuffer,
                config: Optional[VadConfig] = None,
                annotation: Optional[dict] = None,
                cuts: Optional[Sequence[LockedCut]] = None) -> AnalysisResult:
        from .pipeline import analyze_audio
        cfg = (config or self.config).validated()
        return analyze_audio(audio, cfg, self.detector, annotation,
                             list(cuts) if cuts else None)

    def recompute(self, audio: AudioBuffer, decision,
                  config: VadConfig,
                  annotation: Optional[dict] = None,
                  cuts: Optional[Sequence[LockedCut]] = None) -> AnalysisResult:
        """Re-merge cached scores; threshold sliders use this exact path."""
        cfg = config.validated()
        intervals = merge_decision(decision, cfg)
        missed: list[MissedRegion] = []
        if annotation is not None:
            refs = reference_sample_spans(annotation, audio.sample_rate)
            missed = subtract_detected(refs, intervals, min_ms=120)
        conflicts: list[CutConflict] = []
        if cuts:
            conflicts = evaluate_locked_cuts(list(cuts), intervals)
        return AnalysisResult(
            audio=audio, config=cfg, intervals=intervals,
            detector=self.detector, probs=decision.probs, raw=decision.raw,
            times=decision.times, annotation=annotation, missed=missed,
            conflicts=conflicts, backend=decision.backend)

    def detect(self, audio: AudioBuffer):
        return self.detector.detect(audio)

    def intervals_for_blocks(self, audio: AudioBuffer,
                             block_sizes: Sequence[int] = (256, 512),
                             config: Optional[VadConfig] = None
                             ) -> dict[int, list[Interval]]:
        cfg = (config or self.config).validated()
        out: dict[int, list[Interval]] = {}
        for block in block_sizes:
            block = int(block)
            if block <= 0:
                raise ValueError("block_size 必须为正整数")
            vad = StreamingVad(audio.sample_rate, cfg, self.detector)
            samples = audio.samples
            for a in range(0, len(samples), block):
                vad.feed(samples[a:a + block])
            vad.finish()
            out[block] = list(vad.confirmed)
        return out

    def block_contract(self, audio: AudioBuffer,
                       block_sizes: Sequence[int] = (256, 512),
                       config: Optional[VadConfig] = None) -> dict:
        cfg = (config or self.config).validated()
        decision = self.detector.detect(audio, threshold=cfg.enter_threshold)
        offline = merge_decision(decision, cfg)
        blocked = self.intervals_for_blocks(audio, block_sizes, cfg)

        def tuples(ivs):
            return [(int(x.start_sample), int(x.end_sample),
                     int(x.core_start_sample), int(x.core_end_sample))
                    for x in ivs]

        rows = {str(b): tuples(ivs) for b, ivs in blocked.items()}
        expected = tuples(offline)
        matches = {int(b): tuples(ivs) == expected
                   for b, ivs in blocked.items()}
        all_match = all(matches.values())
        return {
            "offline": expected,
            "blocks": rows,
            "matches_offline": matches,
            "all_match": all_match,
            "all_match_offline": all_match,
        }

    def inference_error(self, audio: AudioBuffer) -> dict:
        """Compare migrated ONNX scores with pinned NumPy reference weights."""
        if getattr(self.detector, "reference_model", None) is None:
            return {"comparable": False,
                    "reason": "detector has no pinned NumPy reference model"}
        fx = extract_features(audio)
        migrated = np.asarray(self.detector.model.predict_proba(fx.x),
                              dtype=np.float64)
        reference = np.asarray(
            self.detector.reference_model.predict_proba(fx.x),
            dtype=np.float64)
        err = np.abs(migrated - reference)
        thr = float(self.detector.reference_model.threshold)
        new_active = migrated >= thr
        old_active = reference >= thr
        first_new = int(np.argmax(new_active)) if new_active.any() else None
        first_old = int(np.argmax(old_active)) if old_active.any() else None
        return {
            "comparable": True,
            "threshold": thr,
            "n_frames": int(migrated.size),
            "max_abs_error": float(err.max(initial=0.0)),
            "mean_abs_error": float(err.mean() if err.size else 0.0),
            "first_active_frame_onnx": first_new,
            "first_active_frame_reference": first_old,
            "first_active_matches": first_new == first_old,
            "active_frame_count_onnx": int(new_active.sum()),
            "active_frame_count_reference": int(old_active.sum()),
            "active_counts_match": int(new_active.sum()) == int(old_active.sum()),
        }

    def export_and_read_back_head(self, audio: AudioBuffer,
                                  out_dir: str | Path,
                                  interval: Optional[Interval] = None
                                  ) -> HeadReadback:
        iv = interval or (self.analyze(audio).intervals or [None])[0]
        if iv is None:
            raise ValueError("没有可交给剪辑器的活动区间")
        path, verification = export_interval(audio, iv, Path(out_dir))
        if not verification.verified:
            raise RuntimeError(f"字头样本回读核验失败: {path}")
        source_head = int(audio.samples[iv.start_sample])
        if verification.first_sample_expected != source_head:
            raise RuntimeError("区间字头与源样本发生重复时间补偿偏移")
        return HeadReadback(path=path, verification=verification, interval=iv)


IntervalKernel = VoiceIntervalKernel
