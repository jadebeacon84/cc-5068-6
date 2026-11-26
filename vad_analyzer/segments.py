"""Interval merging: turns frame probabilities into time ranges.

Semantics (all times are sample-exact internally, seconds are derived):

``enter_threshold``
    A frame with probability >= this is *entry evidence* and opens/reenforces
    an interval.
``exit_threshold``
    While active, a frame below this is counted as exit evidence. Frames
    between ``exit_threshold`` and ``enter_threshold`` are weak/tail evidence:
    they extend the endpoint but still consume the gap budget.
``gap_tolerance_ms``
    An interval closes after this many milliseconds of consecutive frames
    below ``enter_threshold``. Short pauses at or below this value are bridged
    (one interval); longer pauses split.
``min_speech_ms``
    A candidate only survives when its span of *strong* (entry-threshold)
    evidence is at least this long and the strong-frame density clears
    ``min_strong_ratio``. Measuring strong evidence (not the tail-extended
    endpoint) keeps isolated clicks from qualifying.
``edge_pad_ms``
    Delivered intervals are extended by this much at both ends (clamped to the
    file) to keep breaths and masked tails audible. Padding never changes the
    detected core.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass
class VadConfig:
    enter_threshold: float = 0.55
    exit_threshold: float = 0.22
    min_speech_ms: int = 130
    min_strong_ratio: float = 0.35
    min_confidence: float = 0.0
    gap_tolerance_ms: int = 600
    edge_pad_ms: int = 90
    exit_tail_ms: int = 140
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    provisional_ms: int = 220

    def validated(self) -> "VadConfig":
        if not (0.0 <= self.enter_threshold <= 1.0):
            raise ValueError("enter_threshold 必须在 [0,1]")
        if not (0.0 <= self.exit_threshold <= 1.0):
            raise ValueError("exit_threshold 必须在 [0,1]")
        if self.exit_threshold > self.enter_threshold:
            raise ValueError("exit_threshold 不能高于 enter_threshold")
        if self.min_speech_ms < 0 or self.gap_tolerance_ms < 0 or self.edge_pad_ms < 0:
            raise ValueError("毫秒参数不能为负")
        if not (0.0 <= self.min_strong_ratio <= 1.0):
            raise ValueError("min_strong_ratio 必须在 [0,1]")
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError("min_confidence 必须在 [0,1]")
        if self.provisional_ms > self.gap_tolerance_ms:
            # provisional must be reportable before archive confirmation
            self.provisional_ms = max(0, self.gap_tolerance_ms - 40)
        return self

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Interval:
    start_sample: int
    end_sample: int                      # exclusive, delivered (padded)
    sample_rate: int
    core_start_sample: int = 0
    core_end_sample: int = 0
    confidence: float = 0.0
    confidence_low: float = 0.0
    strong_frame_ratio: float = 0.0
    bridged_gaps: int = 0
    index: Optional[int] = None
    label: str = "voice_activity"
    speaker: Optional[str] = None
    source: str = "confirmed"
    notes: str = ""

    @property
    def start(self) -> float:
        return self.start_sample / float(self.sample_rate)

    @property
    def end(self) -> float:
        return self.end_sample / float(self.sample_rate)

    @property
    def duration(self) -> float:
        return (self.end_sample - self.start_sample) / float(self.sample_rate)

    @property
    def core_start(self) -> float:
        return self.core_start_sample / float(self.sample_rate)

    @property
    def core_end(self) -> float:
        return self.core_end_sample / float(self.sample_rate)

    @property
    def confidence_level(self) -> str:
        if self.confidence >= 0.8:
            return "high"
        if self.confidence >= 0.6:
            return "medium"
        return "low"

    def to_dict(self) -> dict:
        d = {
            "index": self.index,
            "start_sec": round(self.start, 4),
            "end_sec": round(self.end, 4),
            "duration_sec": round(self.duration, 4),
            "start_sample": int(self.start_sample),
            "end_sample": int(self.end_sample),
            "core_start_sec": round(self.core_start, 4),
            "core_end_sec": round(self.core_end, 4),
            "core_start_sample": int(self.core_start_sample),
            "core_end_sample": int(self.core_end_sample),
            "confidence": round(self.confidence, 4),
            "confidence_low": round(self.confidence_low, 4),
            "confidence_level": self.confidence_level,
            "strong_frame_ratio": round(self.strong_frame_ratio, 3),
            "bridged_gaps": self.bridged_gaps,
            "label": self.label,
            "speaker": self.speaker,
            "source": self.source,
        }
        if self.notes:
            d["notes"] = self.notes
        return d


@dataclass
class CoreSegment:
    first_frame: int
    last_alive_frame: int
    last_strong_frame: int
    bridged_gaps: int = 0


class EvidenceFSM:
    """Frame-by-frame entry/exit state machine (shared by stream/offline)."""

    def __init__(self, sample_rate: int, hop_samples: int, config: VadConfig):
        self.sr = sample_rate
        self.hop = hop_samples
        self.cfg = config.validated()
        self.gap_frames = max(1, int(round(
            self.cfg.gap_tolerance_ms / (1000.0 * hop_samples / sample_rate))))
        self.provisional_frames = max(1, int(round(
            self.cfg.provisional_ms / (1000.0 * hop_samples / sample_rate))))
        self.tail_cap_frames = max(1, int(round(
            self.cfg.exit_tail_ms / (1000.0 * hop_samples / sample_rate))))
        self.open = False
        self.first = self.strong = self.alive = -1
        self.neg_run = 0
        self.bridges = 0
        self.provisional_sent = False

    def start_sample_of(self, frame: int) -> int:
        return frame * self.hop

    def add_frame(self, i: int, p: float) -> Optional[str]:
        """Feed one probability. Returns 'start' or 'close' event marker."""
        c = self.cfg
        if not self.open:
            if p >= c.enter_threshold:
                self.open = True
                self.first = self.strong = self.alive = i
                self.neg_run = 0
                self.bridges = 0
                self.provisional_sent = False
                return "start"
            return None

        if p >= c.enter_threshold:
            if self.neg_run > 0:
                self.bridges += 1
            self.strong = i
            self.alive = i
            self.neg_run = 0
        elif p >= c.exit_threshold:
            # Only count weak evidence as endpoint tail close to the last
            # strong frame; a long moderate-prob background stretch must not
            # drag the endpoint across the whole gap.
            if i - self.strong <= self.tail_cap_frames:
                self.alive = i
            self.neg_run += 1
        else:
            self.neg_run += 1

        if self.neg_run >= self.gap_frames:
            self.open = False
            self.provisional_sent = False
            return "close"
        return None

    def provisional_due(self) -> bool:
        if self.open and not self.provisional_sent \
                and self.neg_run >= self.provisional_frames \
                and self.neg_run < self.gap_frames:
            self.provisional_sent = True
            return True
        return False

    def retracted(self, p: float) -> bool:
        """True if strong evidence just cancelled a standing provisional."""
        return self.provisional_sent and p >= self.cfg.enter_threshold \
            and self.neg_run == 0

    def core(self) -> CoreSegment:
        return CoreSegment(self.first, max(self.alive, self.first),
                           max(self.strong, self.first), self.bridges)

    def flush(self) -> Optional[CoreSegment]:
        if self.open:
            self.open = False
            return self.core()
        return None


def _confidence(probs: np.ndarray, start_samples: np.ndarray,
                c0: int, c1: int, enter_thr: float) -> tuple[float, float, float]:
    m = (start_samples >= c0) & (start_samples < c1)
    seg = probs[m]
    if seg.size == 0:
        return 0.5, 0.5, 0.0
    return (float(seg.mean()), float(np.percentile(seg, 10)),
            float(np.mean(seg >= enter_thr)))


def cores_to_intervals(cores: Sequence[CoreSegment], probs: np.ndarray,
                       start_samples: np.ndarray, win_samples: int,
                       n_total: int, sr: int,
                       cfg: VadConfig) -> list[Interval]:
    cfg = cfg.validated()
    min_core = int(round(sr * cfg.min_speech_ms / 1000.0))
    pad = int(round(sr * cfg.edge_pad_ms / 1000.0))
    out: list[Interval] = []
    for k, core in enumerate(cores):
        c0 = int(start_samples[core.first_frame])
        alive_frame = min(core.last_alive_frame, len(start_samples) - 1)
        strong_frame = min(core.last_strong_frame, len(start_samples) - 1)
        alive_end = min(n_total, int(start_samples[alive_frame]) + win_samples)
        strong_end = min(n_total, int(start_samples[strong_frame]) + win_samples)
        # Minimum-speech gate uses *strong evidence*, not the tail-extended end,
        # so a transient click cannot borrow exit-tail frames to qualify.
        if strong_end - c0 < min_core:
            continue
        conf, low, strong_ratio = _confidence(probs, start_samples, c0, alive_end,
                                              cfg.enter_threshold)
        if strong_ratio < cfg.min_strong_ratio:
            continue
        if conf < cfg.min_confidence:
            continue
        s0 = max(0, c0 - pad)
        s1 = min(n_total, alive_end + pad)
        out.append(Interval(start_sample=s0, end_sample=s1, sample_rate=sr,
                            core_start_sample=c0, core_end_sample=strong_end,
                            confidence=conf, confidence_low=low,
                            strong_frame_ratio=strong_ratio,
                            bridged_gaps=core.bridged_gaps))
    # Union overlapping delivered intervals (large pad / tiny gap settings).
    merged: list[Interval] = []
    for iv in sorted(out, key=lambda v: v.start_sample):
        if merged and iv.start_sample <= merged[-1].end_sample:
            m = merged[-1]
            m.end_sample = max(m.end_sample, iv.end_sample)
            m.core_end_sample = max(m.core_end_sample, iv.core_end_sample)
            m.bridged_gaps += iv.bridged_gaps + 1
            conf, low, sr_ratio = _confidence(
                probs, start_samples, m.core_start_sample, m.core_end_sample,
                cfg.enter_threshold)
            m.confidence = conf
            m.confidence_low = low
            m.strong_frame_ratio = max(m.strong_frame_ratio, sr_ratio)
        else:
            merged.append(iv)
    for i, iv in enumerate(merged, 1):
        iv.index = i
    return merged


def merge_probabilities(probs, start_samples,
                        sr: int, hop_samples: int, win_samples: int,
                        n_total: int, cfg: VadConfig) -> list[Interval]:
    probs = np.asarray(probs, dtype=np.float64)
    start_samples = np.asarray(start_samples, dtype=np.int64)
    fsm = EvidenceFSM(sr, hop_samples, cfg)
    cores: list[CoreSegment] = []
    for i, p in enumerate(probs):
        ev = fsm.add_frame(i, float(p))
        if ev == "close":
            cores.append(fsm.core())
    tail = fsm.flush()
    if tail is not None:
        cores.append(tail)
    return cores_to_intervals(cores, probs, start_samples, win_samples,
                              n_total, sr, cfg)


def merge_decision(decision, cfg: VadConfig) -> list[Interval]:
    g = decision.features.grid
    return merge_probabilities(
        decision.probs, decision.start_samples, g.sample_rate, g.hop, g.win,
        g.n_total or int(decision.start_samples[-1] + g.win), cfg)
