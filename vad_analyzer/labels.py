"""Reference annotations and set operations for missed-region reporting."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .errors import AnnotationError


@dataclass
class MissedRegion:
    start_sample: int
    end_sample: int
    sample_rate: int
    ref_index: int = 0

    @property
    def start(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end(self) -> float:
        return self.end_sample / self.sample_rate

    @property
    def duration(self) -> float:
        return (self.end_sample - self.start_sample) / self.sample_rate

    def to_dict(self) -> dict:
        return {
            "start_sec": round(self.start, 4),
            "end_sec": round(self.end, 4),
            "duration_sec": round(self.duration, 4),
            "start_sample": int(self.start_sample),
            "end_sample": int(self.end_sample),
            "ref_index": self.ref_index,
        }


def load_annotation(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise AnnotationError(f"标注文件不存在: {p}")
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise AnnotationError(f"标注 JSON 无法解析: {p.name} ({e})") from e
    if "segments" not in d or not isinstance(d["segments"], list):
        raise AnnotationError(f"标注文件缺少 segments 列表: {p.name}")
    return d


def find_sidecar(audio_path: str | Path) -> Optional[Path]:
    p = Path(audio_path)
    for cand in (p.with_suffix(".json"),
                 p.with_name(p.stem + ".annotation.json"),
                 p.parent / "annotations" / (p.stem + ".json")):
        if cand.exists():
            return cand
    return None


def reference_sample_spans(ann: dict, sample_rate: int,
                           kinds: Sequence[str] = ("speech", "voice")) -> list[tuple[int, int]]:
    """Phrase/turn-level reference spans, converted to samples."""
    spans: list[tuple[int, int]] = []
    for seg in ann["segments"]:
        kind = str(seg.get("kind", "speech")).lower()
        if kinds and kind not in kinds:
            continue
        if "start_sample" in seg and "end_sample" in seg:
            s0, s1 = int(seg["start_sample"]), int(seg["end_sample"])
        else:
            s0 = int(round(float(seg["start"]) * sample_rate))
            s1 = int(round(float(seg["end"]) * sample_rate))
        if s1 > s0:
            spans.append((s0, s1))
    return sorted(spans)


def active_sample_spans(ann: dict, sample_rate: int) -> Optional[list[tuple[int, int]]]:
    """Exact active spans used for frame metrics when the generator provides them."""
    raw = ann.get("active_spans_samples")
    if raw:
        return [(int(a), int(b)) for a, b in raw]
    if ann.get("active_spans_sec"):
        return [(int(round(a * sample_rate)), int(round(b * sample_rate)))
                for a, b in ann["active_spans_sec"]]
    return None


def union_spans(spans: Sequence[tuple[int, int]],
                join_gap_samples: int = 0) -> list[tuple[int, int]]:
    """Union overlapping (and optionally near-adjacent) sample spans."""
    if not spans:
        return []
    ordered = sorted((int(a), int(b)) for a, b in spans if b > a)
    out: list[list[int]] = [list(ordered[0])]
    for a, b in ordered[1:]:
        if a <= out[-1][1] + max(0, join_gap_samples):
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def subtract_detected(ref_spans: Sequence[tuple[int, int]],
                      detected: Sequence,
                      min_ms: int = 120) -> list[MissedRegion]:
    """Reference speech minus the union of delivered detected intervals."""
    if not detected:
        union: list[tuple[int, int]] = []
    else:
        ranges = sorted((max(0, iv.start_sample), iv.end_sample) for iv in detected)
        union = []
        for a, b in ranges:
            if union and a <= union[-1][1]:
                union[-1] = (union[-1][0], max(union[-1][1], b))
            else:
                union.append((a, b))

    out: list[MissedRegion] = []
    sr = detected[0].sample_rate if detected else 16000
    min_samples = int(sr * min_ms / 1000.0)
    for ri, (r0, r1) in enumerate(ref_spans):
        cursor = r0
        for u0, u1 in union:
            if u1 <= cursor:
                continue
            if u0 >= r1:
                break
            if u0 > cursor:
                gap_end = min(u0, r1)
                if gap_end - cursor >= min_samples:
                    out.append(MissedRegion(cursor, gap_end, sr, ri))
            cursor = max(cursor, u1)
            if cursor >= r1:
                break
        if cursor < r1 and r1 - cursor >= min_samples:
            out.append(MissedRegion(cursor, r1, sr, ri))
    return out


def save_annotation(path: str | Path, *, audio_file: str, sample_rate: int,
                    duration_sec: float, segments: list[dict],
                    annotator: str = "vad_analyzer", **extra) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "audio_file": audio_file,
        "sample_rate": sample_rate,
        "duration_sec": round(duration_sec, 4),
        "annotator": annotator,
        "segments": segments,
    }
    data.update(extra)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
