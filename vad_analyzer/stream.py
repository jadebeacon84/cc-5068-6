"""Streaming / real-time voice-activity session.

Two endpoint notions are kept separate on purpose:

* ``provisional_end`` -- the monitor *may* have reached the end of speech after
  a short quiet stretch. It is advisory and can be retracted
  (``end_retracted``) when more entry evidence arrives.
* ``segment_confirmed`` -- emitted only after the full gap-tolerance evidence
  (or end-of-file), so an archive slicer can cut without later contradiction.

Locked editor cuts are handled elsewhere (:mod:`vad_analyzer.cuts`); this
session never rewrites them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .features import FeatureStreamer
from .segments import (EvidenceFSM, VadConfig, cores_to_intervals)


@dataclass
class StreamEvent:
    kind: str
    time_sec: float
    sample: int
    message: str = ""
    tentative_end_sec: Optional[float] = None
    tentative_end_sample: Optional[int] = None
    interval: Optional[dict] = None

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "time_sec": round(self.time_sec, 4),
             "sample": int(self.sample), "message": self.message}
        if self.tentative_end_sec is not None:
            d["tentative_end_sec"] = round(self.tentative_end_sec, 4)
            d["tentative_end_sample"] = int(self.tentative_end_sample)
        if self.interval is not None:
            d["interval"] = self.interval
        return d


class StreamingVad:
    def __init__(self, sample_rate: int, config: Optional[VadConfig] = None,
                 detector=None) -> None:
        self.sr = sample_rate
        self.cfg = (config or VadConfig()).validated()
        self.fx = FeatureStreamer(sample_rate, frame_ms=self.cfg.frame_ms,
                                  hop_ms=self.cfg.hop_ms)
        self.win = self.fx.win
        self.hop = self.fx.hop
        self.fsm = EvidenceFSM(sample_rate, self.hop, self.cfg)
        self.detector = detector
        self._buf = np.zeros(0, dtype=np.float32)
        self._fi = 0
        self._processed_samples = 0
        self.probs: list[float] = []
        self.start_samples: list[int] = []
        self.events: list[StreamEvent] = []
        self.confirmed = []
        self.finished = False

    def _score(self, v: np.ndarray) -> float:
        if self.detector is not None:
            return float(self.detector.score_features(v[None, :])[0])
        from .model import HeuristicScorer
        return float(HeuristicScorer()(v[None, :])[0])

    def _now(self, sample: int) -> float:
        return sample / float(self.sr)

    def _emit_close(self, core, at_sample: int, reason: str) -> Optional[StreamEvent]:
        probs = np.asarray(self.probs, dtype=np.float64)
        starts = np.asarray(self.start_samples, dtype=np.int64)
        ivs = cores_to_intervals([core], probs, starts, self.win,
                                 self._processed_samples, self.sr, self.cfg)
        if not ivs:
            ev = StreamEvent("short_dropped", self._now(at_sample), at_sample,
                              "候选短于 min_speech_ms，已作为误触发丢弃。")
            self.events.append(ev)
            return ev
        iv = ivs[0]
        iv.index = len(self.confirmed) + 1
        iv.source = "confirmed" if reason == "gap" else "eof_confirmed"
        self.confirmed.append(iv)
        ev = StreamEvent("segment_confirmed", self._now(at_sample), at_sample,
                         "间隙证据齐全，区间已确认，可归档切片。",
                         interval=iv.to_dict())
        self.events.append(ev)
        return ev

    def _handle_prob(self, i: int, p: float) -> list[StreamEvent]:
        out: list[StreamEvent] = []
        was_open = self.fsm.open
        was_provisional = self.fsm.provisional_sent
        marker = self.fsm.add_frame(i, p)

        if marker == "start":
            s = i * self.hop
            ev = StreamEvent("speech_started", self._now(s), s,
                             "检测到进入证据，语音活动开始。")
            self.events.append(ev)
            out.append(ev)

        # A retraction is only possible while the *same* open episode still has
        # a standing provisional. Once gap evidence closed/confirmed it, a
        # later strong frame is a new episode, never a retraction.
        if was_open and was_provisional and marker != "start" \
                and p >= self.cfg.enter_threshold and self.fsm.neg_run == 0:
            self.fsm.provisional_sent = False
            ev = StreamEvent(
                "end_retracted", self._now(i * self.hop), i * self.hop,
                "后续语音证据到达，撤回“可能结束”暂报，区间继续。")
            self.events.append(ev)
            out.append(ev)

        if self.fsm.provisional_due():
            end_frame = self.fsm.alive
            tend = min(self._processed_samples, end_frame * self.hop)
            ev = StreamEvent(
                "provisional_end", self._now(i * self.hop), i * self.hop,
                "短暂低证据：暂报可能结束，等待后续证据（可撤回）。",
                tentative_end_sec=self._now(tend),
                tentative_end_sample=int(tend))
            self.events.append(ev)
            out.append(ev)

        if marker == "close":
            core = self.fsm.core()
            ev = self._emit_close(core, i * self.hop, "gap")
            if ev is not None:
                out.append(ev)
        return out

    def feed(self, samples) -> list[StreamEvent]:
        if self.finished:
            raise RuntimeError("StreamingVad 已 finish，不能继续 feed")
        x = np.asarray(samples)
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        else:
            x = x.astype(np.float32, copy=False)
        self._buf = np.concatenate([self._buf, x])
        self._processed_samples += len(x)
        out: list[StreamEvent] = []
        while len(self._buf) >= self.win:
            frame = self._buf[:self.win]
            self._buf = self._buf[self.hop:]
            v = self.fx.process(frame)
            p = self._score(v)
            self.probs.append(p)
            self.start_samples.append(self._fi * self.hop)
            out.extend(self._handle_prob(self._fi, p))
            self._fi += 1
        return out

    def finish(self) -> list[StreamEvent]:
        out: list[StreamEvent] = []
        if self.finished:
            return out
        if 0 < len(self._buf) < self.win:
            v = self.fx.process(self._buf)  # zero-padded inside process()
            p = self._score(v)
            self.probs.append(p)
            self.start_samples.append(self._fi * self.hop)
            out.extend(self._handle_prob(self._fi, p))
            self._fi += 1
            self._buf = np.zeros(0, dtype=np.float32)
        core = self.fsm.flush()
        if core is not None:
            at = self._processed_samples
            ev = self._emit_close(core, at, "eof")
            if ev is not None:
                out.append(ev)
        end_ev = StreamEvent("finished", self._now(self._processed_samples),
                             self._processed_samples, "流式会话结束。")
        self.events.append(end_ev)
        out.append(end_ev)
        self.finished = True
        return out


def run_streaming(audio, config: Optional[VadConfig] = None, detector=None,
                  chunk_ms: float = 20.0) -> StreamingVad:
    """Convenience driver feeding an AudioBuffer in fixed real-time chunks."""
    vad = StreamingVad(audio.sample_rate, config, detector)
    chunk = max(1, int(audio.sample_rate * chunk_ms / 1000.0))
    s = audio.samples
    for a in range(0, len(s), chunk):
        vad.feed(s[a:a + chunk])
    vad.finish()
    return vad
