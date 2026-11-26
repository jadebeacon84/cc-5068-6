"""Locked editor cut points.

A locked interview cut belongs to the downstream editor: analysis must never
move, retract or rewrite it. When voice-activity evidence conflicts with a
locked cut, this module only produces a :class:`CutConflict` report.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass(frozen=True)
class LockedCut:
    cut_id: str
    sample: int
    sample_rate: int
    label: str = "interview-cut"
    note: str = ""

    @property
    def time_sec(self) -> float:
        return self.sample / float(self.sample_rate)

    def to_dict(self) -> dict:
        return {"cut_id": self.cut_id, "sample": int(self.sample),
                "time_sec": round(self.time_sec, 4),
                "sample_rate": self.sample_rate, "label": self.label,
                "locked": True, "note": self.note}


@dataclass
class CutConflict:
    cut_id: str
    kind: str
    severity: str           # "error" | "warning" | "info"
    message: str
    cut_sample: int
    cut_time_sec: float
    interval_index: Optional[int] = None
    evidence: dict = None

    def to_dict(self) -> dict:
        return {
            "cut_id": self.cut_id,
            "kind": self.kind,
            "severity": self.severity,
            "message": self.message,
            "cut_sample": int(self.cut_sample),
            "cut_time_sec": round(self.cut_time_sec, 4),
            "interval_index": self.interval_index,
            "evidence": self.evidence or {},
        }


def load_locked_cuts(path: str | Path, sample_rate: int) -> list[LockedCut]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    raw = data["cuts"] if isinstance(data, dict) else data
    cuts: list[LockedCut] = []
    for i, c in enumerate(raw, 1):
        if "sample" in c:
            sample = int(c["sample"])
        else:
            sample = int(round(float(c["time_sec"]) * sample_rate))
        cuts.append(LockedCut(
            cut_id=str(c.get("cut_id", f"locked-{i}")),
            sample=sample, sample_rate=sample_rate,
            label=str(c.get("label", "interview-cut")),
            note=str(c.get("note", ""))))
    return cuts


def evaluate_locked_cuts(cuts: Sequence[LockedCut], intervals: Sequence,
                         guard_ms: int = 140) -> list[CutConflict]:
    sr = cuts[0].sample_rate if cuts else (
        intervals[0].sample_rate if intervals else 16000)
    guard = int(sr * guard_ms / 1000.0)
    conflicts: list[CutConflict] = []
    for cut in cuts:
        s = cut.sample
        hit_core = None
        hit_padded = None
        near = None
        for iv in intervals:
            if iv.core_start_sample <= s < iv.core_end_sample:
                hit_core = iv
                break
            if iv.start_sample <= s < iv.end_sample:
                hit_padded = iv
            else:
                d = min(abs(s - iv.start_sample), abs(s - iv.end_sample))
                if d <= guard:
                    near = (iv, d)
        if hit_core is not None:
            iv = hit_core
            conflicts.append(CutConflict(
                cut.cut_id, "cut_through_voice_activity", "error",
                f"锁定切点 {cut.cut_id}（{cut.time_sec:.3f}s）落在活动区间 "
                f"#{iv.index} 的核心语音内，会切断可听语音；锁点不会被移动，"
                "请人工确认或调整剪辑。",
                s, cut.time_sec, iv.index,
                {"confidence": round(iv.confidence, 3),
                 "core_start_sec": round(iv.core_start, 3),
                 "core_end_sec": round(iv.core_end, 3)}))
        elif hit_padded is not None:
            iv = hit_padded
            conflicts.append(CutConflict(
                cut.cut_id, "cut_inside_safety_pad", "warning",
                f"锁定切点 {cut.cut_id}（{cut.time_sec:.3f}s）位于区间 "
                f"#{iv.index} 的边界扩展/尾音保留区内，直接切可能削掉尾音。",
                s, cut.time_sec, iv.index,
                {"interval_start_sec": round(iv.start, 3),
                 "interval_end_sec": round(iv.end, 3)}))
        elif near is not None:
            iv, d = near
            conflicts.append(CutConflict(
                cut.cut_id, "boundary_proximity", "info",
                f"锁定切点 {cut.cut_id} 距区间 #{iv.index} 边界仅 "
                f"{d / sr * 1000:.0f}ms，建议复核边界误差。",
                s, cut.time_sec, iv.index,
                {"distance_ms": round(d / sr * 1000, 1)}))
    return conflicts
