"""Reviewable VAD metrics.

Overall accuracy is reported but explicitly de-emphasized: long silences
dominate it. The operational numbers that matter for an editor are pooled
speech-duration miss rate, false-trigger duration/events, event recall and
boundary error, plus breakdowns by reference event duration and category.
"""
from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .audio import load_wav
from .detector import FrameDetector, labels_for_frames
from .labels import (active_sample_spans, load_annotation,
                     reference_sample_spans, union_spans)
from .pipeline import config_for
from .segments import VadConfig, merge_decision


def frame_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    yt = y_true.astype(bool)
    yp = y_pred.astype(bool)
    tp = int(np.sum(yt & yp))
    fn = int(np.sum(yt & ~yp))
    fp = int(np.sum(~yt & yp))
    tn = int(np.sum(~yt & ~yp))
    total = max(1, tp + fn + fp + tn)
    speech = max(1, tp + fn)
    nonspeech = max(1, tn + fp)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": round((tp + tn) / total, 4),
        "speech_frame_miss_rate": round(fn / speech, 4),
        "speech_frame_recall": round(tp / speech, 4),
        "nonspeech_false_positive_rate": round(fp / nonspeech, 4),
        "silence_frame_ratio": round((tn + fp) / total, 4),
    }


def iou(a0: int, a1: int, b0: int, b1: int) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def match_events(ref: list[tuple[int, int]], intervals,
                 iou_thr: float = 0.5):
    pairs = []
    for ri, (r0, r1) in enumerate(ref):
        for ii, iv in enumerate(intervals):
            v = iou(r0, r1, iv.start_sample, iv.end_sample)
            if v > 0:
                pairs.append((v, ri, ii))
    pairs.sort(reverse=True)
    m_ref, m_det = set(), set()
    matches = []
    for v, ri, ii in pairs:
        if ri in m_ref or ii in m_det:
            continue
        if v < iou_thr:
            continue
        m_ref.add(ri)
        m_det.add(ii)
        matches.append((ri, ii, v))
    unmatched_ref = [ri for ri in range(len(ref)) if ri not in m_ref]
    unmatched_det = [ii for ii in range(len(intervals)) if ii not in m_det]

    # Split/merge evidence from the raw overlap graph.
    ref_to_det: dict[int, set] = {}
    det_to_ref: dict[int, set] = {}
    for v, ri, ii in pairs:
        if v > 0.05:
            ref_to_det.setdefault(ri, set()).add(ii)
            det_to_ref.setdefault(ii, set()).add(ri)
    split_events = sum(1 for s in ref_to_det.values() if len(s) >= 2)
    merged_events = sum(1 for s in det_to_ref.values() if len(s) >= 2)
    return matches, unmatched_ref, unmatched_det, split_events, merged_events


def _mask(n: int, spans) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for a, b in spans:
        m[max(0, a):min(n, b)] = True
    return m


def coverage_counts(ref, intervals, cover: float = 0.5) -> tuple[int, int]:
    """Many-to-one event coverage: refs >=cover covered, dets >=cover supported."""
    covered = 0
    for r0, r1 in ref:
        cov = 0
        for iv in intervals:
            cov += max(0, min(r1, iv.end_sample) - max(r0, iv.start_sample))
        if cov / max(1, r1 - r0) >= cover:
            covered += 1
    supported = 0
    for iv in intervals:
        d0, d1 = iv.start_sample, iv.end_sample
        ov = 0
        for r0, r1 in ref:
            ov += max(0, min(d1, r1) - max(d0, r0))
        if ov / max(1, d1 - d0) >= cover:
            supported += 1
    return covered, supported


def _bucket(dur: float) -> str:
    if dur < 1.0:
        return "short_lt1s"
    if dur < 2.5:
        return "medium_1_2.5s"
    return "long_ge2.5s"


def evaluate_clip(wav_path: Path, ann: dict,
                  detector: Optional[FrameDetector] = None,
                  cfg: Optional[VadConfig] = None) -> tuple[dict, list[dict]]:
    det = detector or FrameDetector()
    c = cfg or config_for(det)
    audio = load_wav(wav_path)
    decision = det.detect(audio, threshold=c.enter_threshold)
    decision = det.redecode(decision, c.enter_threshold)
    intervals = merge_decision(decision, c)
    sr = audio.sample_rate
    n = len(audio.samples)

    phrase_ref = reference_sample_spans(ann, sr)
    exact = active_sample_spans(ann, sr) or phrase_ref
    # A VAD reports acoustic activity, not speaker diarization, so overlapping
    # reference turns are unioned for the primary activity-event matching.
    activity_ref = union_spans(phrase_ref)

    y_true = labels_for_frames(decision.features.grid, exact, n)
    fc = frame_confusion(y_true, decision.raw)

    active_mask = _mask(n, exact)
    phrase_mask = _mask(n, phrase_ref)
    det_mask = _mask(n, [(iv.start_sample, iv.end_sample) for iv in intervals])

    active_sec = active_mask.sum() / sr
    missed_sec = (active_mask & ~det_mask).sum() / sr
    outside_phrase_sec = (~phrase_mask & det_mask).sum() / sr
    det_speech_sec = det_mask.sum() / sr

    matches, um_ref, um_det, _, _ = match_events(
        activity_ref, intervals)
    # Phrase/turn graph kept only to evidence split vs merge behaviour.
    _, _, _, phrase_splits, phrase_merges = match_events(phrase_ref, intervals)
    cov_ref, cov_det = coverage_counts(activity_ref, intervals, 0.5)

    rows: list[dict] = []
    start_errs, end_errs, d_start_errs, d_end_errs = [], [], [], []
    bucket_acc: dict[str, dict] = {}
    match_by_ref = {ri: (ii, v) for ri, ii, v in matches}
    for ri, (r0, r1) in enumerate(activity_ref):
        dur = (r1 - r0) / sr
        b = _bucket(dur)
        ba = bucket_acc.setdefault(b, {"events": 0, "matched": 0,
                                       "span_samples": 0, "covered_samples": 0,
                                       "abs_start_ms": [], "abs_end_ms": []})
        ba["events"] += 1
        ba["span_samples"] += r1 - r0
        ba["covered_samples"] += int(det_mask[r0:r1].sum())
        row = {"clip": wav_path.name, "ref_index": ri,
               "start_sec": round(r0 / sr, 3), "end_sec": round(r1 / sr, 3),
               "duration_sec": round(dur, 3), "bucket": b,
               "matched": ri in match_by_ref}
        if ri in match_by_ref:
            ii, iouv = match_by_ref[ri]
            iv = intervals[ii]
            ba["matched"] += 1
            cs = (iv.core_start_sample - r0) / sr * 1000
            ce = (iv.core_end_sample - r1) / sr * 1000
            ds = (iv.start_sample - r0) / sr * 1000
            de = (iv.end_sample - r1) / sr * 1000
            start_errs.append(cs); end_errs.append(ce)
            d_start_errs.append(ds); d_end_errs.append(de)
            ba["abs_start_ms"].append(abs(cs))
            ba["abs_end_ms"].append(abs(ce))
            row.update({"det_index": iv.index, "iou": round(iouv, 3),
                        "core_start_err_ms": round(cs, 1),
                        "core_end_err_ms": round(ce, 1),
                        "delivered_start_err_ms": round(ds, 1),
                        "delivered_end_err_ms": round(de, 1)})
        rows.append(row)

    def pct(vals, q):
        if vals is None or len(vals) == 0:
            return None
        return round(float(np.percentile(np.asarray(vals), q)), 1)

    clip_metrics = {
        "clip": wav_path.name,
        "category": ann.get("category", ""),
        "snr_db": ann.get("snr_db"),
        "duration_sec": round(audio.duration, 2),
        "exact_voice_sec": round(active_sec, 2),
        "silence_ratio": fc["silence_frame_ratio"],
        "frame": fc,
        "operational": {
            "missed_voice_sec": round(missed_sec, 3),
            "voice_miss_rate": round(missed_sec / max(1e-9, active_sec), 4),
            "false_trigger_sec": round(outside_phrase_sec, 3),
            "false_trigger_ratio_of_clip": round(outside_phrase_sec / audio.duration, 4),
            "detected_voice_sec": round(det_speech_sec, 2),
        },
        "events": {
            "reference": len(activity_ref),
            "phrase_reference": len(phrase_ref),
            "overlap_unioned": max(0, len(phrase_ref) - len(activity_ref)),
            "detected": len(intervals),
            "matched": len(matches),
            "missed_events": len(um_ref),
            "false_events": len(um_det),
            "split_events": phrase_splits,
            "merged_events": phrase_merges,
            "coverage_covered_refs": cov_ref,
            "coverage_supported_dets": cov_det,
            "coverage_threshold": 0.5,
        },
        "boundary_core_ms": {
            "start_mae": pct(np.abs(start_errs), 50) if start_errs else None,
            "end_mae": pct(np.abs(end_errs), 50) if end_errs else None,
            "start_mean_abs": round(float(np.mean(np.abs(start_errs))), 1) if start_errs else None,
            "end_mean_abs": round(float(np.mean(np.abs(end_errs))), 1) if end_errs else None,
            "start_p95_abs": pct(np.abs(start_errs), 95),
            "end_p95_abs": pct(np.abs(end_errs), 95),
            "start_bias_signed": round(float(np.mean(start_errs)), 1) if start_errs else None,
            "end_bias_signed": round(float(np.mean(end_errs)), 1) if end_errs else None,
        },
        "boundary_delivered_ms": {
            "start_mean_abs": round(float(np.mean(np.abs(d_start_errs))), 1) if d_start_errs else None,
            "end_mean_abs": round(float(np.mean(np.abs(d_end_errs))), 1) if d_end_errs else None,
        },
        "duration_buckets": {
            b: {"events": v["events"], "matched": v["matched"],
                "event_recall": round(v["matched"] / max(1, v["events"]), 3),
                "span_miss_rate": round(
                    1 - v["covered_samples"] / max(1, v["span_samples"]), 3),
                "start_mae_ms": round(float(np.mean(v["abs_start_ms"])), 1)
                if v["abs_start_ms"] else None,
                "end_mae_ms": round(float(np.mean(v["abs_end_ms"])), 1)
                if v["abs_end_ms"] else None}
            for b, v in bucket_acc.items()
        },
        "n_intervals": len(intervals),
    }
    return clip_metrics, rows


def _bucket_rollup(rows: list[dict]) -> dict:
    tmp: dict[str, dict] = {}
    for r in rows:
        b = r.get("bucket") or _bucket(float(r.get("duration_sec", 0.0)))
        d = tmp.setdefault(b, {"events": 0, "matched": 0,
                               "start_abs": [], "end_abs": []})
        d["events"] += 1
        if r.get("matched"):
            d["matched"] += 1
            if r.get("core_start_err_ms") is not None:
                d["start_abs"].append(abs(float(r["core_start_err_ms"])))
            if r.get("core_end_err_ms") is not None:
                d["end_abs"].append(abs(float(r["core_end_err_ms"])))

    def mae(v):
        return round(float(np.mean(v)), 1) if v else None

    def p95(v):
        return round(float(np.percentile(np.asarray(v), 95)), 1) if v else None

    order = ["short_lt1s", "medium_1_2.5s", "long_ge2.5s"]
    out: dict = {}
    for b in order + [k for k in tmp if k not in order]:
        if b not in tmp:
            continue
        d = tmp[b]
        out[b] = {
            "events": d["events"], "matched": d["matched"],
            "event_recall": round(d["matched"] / max(1, d["events"]), 3),
            "start_mae_ms": mae(d["start_abs"]),
            "end_mae_ms": mae(d["end_abs"]),
            "start_p95_abs_ms": p95(d["start_abs"]),
            "end_p95_abs_ms": p95(d["end_abs"]),
        }
    return out


def _aggregate(clip_metrics: list[dict]) -> dict:
    agg: dict = {"n_clips": len(clip_metrics)}
    dur = sum(c["duration_sec"] for c in clip_metrics)
    voice = sum(c["exact_voice_sec"] for c in clip_metrics)
    miss = sum(c["operational"]["missed_voice_sec"] for c in clip_metrics)
    false_s = sum(c["operational"]["false_trigger_sec"] for c in clip_metrics)
    ev_ref = sum(c["events"]["reference"] for c in clip_metrics)
    ev_det = sum(c["events"]["detected"] for c in clip_metrics)
    ev_match = sum(c["events"]["matched"] for c in clip_metrics)
    ev_miss = sum(c["events"]["missed_events"] for c in clip_metrics)
    ev_false = sum(c["events"]["false_events"] for c in clip_metrics)
    cov_ref = sum(c["events"].get("coverage_covered_refs", 0) for c in clip_metrics)
    cov_det = sum(c["events"].get("coverage_supported_dets", 0) for c in clip_metrics)
    agg["total_duration_sec"] = round(dur, 1)
    agg["exact_voice_sec"] = round(voice, 1)
    agg["overall_silence_ratio"] = round(1 - voice / max(1e-9, dur), 3)
    agg["pooled_voice_miss_rate"] = round(miss / max(1e-9, voice), 4)
    agg["pooled_false_trigger_sec"] = round(false_s, 2)
    agg["false_trigger_sec_per_minute"] = round(false_s / max(1e-9, dur / 60), 2)
    agg["frame_accuracy_misleading_pooled"] = None
    tp = sum(c["frame"]["tp"] for c in clip_metrics)
    fp = sum(c["frame"]["fp"] for c in clip_metrics)
    fn = sum(c["frame"]["fn"] for c in clip_metrics)
    tn = sum(c["frame"]["tn"] for c in clip_metrics)
    agg["frame_pooled"] = {
        "accuracy": round((tp + tn) / max(1, tp + tn + fp + fn), 4),
        "speech_miss_rate": round(fn / max(1, tp + fn), 4),
        "nonspeech_fpr": round(fp / max(1, tn + fp), 4),
        "note": "accuracy is dominated by the high silence ratio; use miss/event metrics",
    }
    agg["events_pooled"] = {
        "reference": ev_ref, "detected": ev_det, "matched": ev_match,
        "missed_events": ev_miss, "false_events": ev_false,
        "recall": round(ev_match / max(1, ev_ref), 3),
        "precision": round(ev_match / max(1, ev_det), 3),
        "f1": round(2 * ev_match / max(1e-9, 2 * ev_match + ev_miss + ev_false), 3),
        "coverage_event_recall_many_to_one": round(cov_ref / max(1, ev_ref), 3),
        "coverage_event_precision_many_to_one": round(cov_det / max(1, ev_det), 3),
        "false_events_per_audio_minute": round(ev_false / max(1e-9, dur / 60), 3),
    }
    for key in ("start_mean_abs", "end_mean_abs"):
        vals = [c["boundary_core_ms"][key] for c in clip_metrics
                if c["boundary_core_ms"][key] is not None]
        agg[f"boundary_core_{key}_avg"] = round(float(np.mean(vals)), 1) if vals else None
    # Category rollup.
    cats: dict[str, dict] = {}
    for c in clip_metrics:
        k = c["category"] or "uncategorized"
        d = cats.setdefault(k, {"clips": 0, "miss_rate_sum": 0.0,
                                "missed_events": 0, "false_events": 0})
        d["clips"] += 1
        d["miss_rate_sum"] += c["operational"]["voice_miss_rate"]
        d["missed_events"] += c["events"]["missed_events"]
        d["false_events"] += c["events"]["false_events"]
    for k, d in cats.items():
        d["mean_voice_miss_rate"] = round(d["miss_rate_sum"] / d["clips"], 3)
        del d["miss_rate_sum"]
    agg["categories"] = cats
    return agg


def evaluate_dataset(eval_dir: str | Path,
                      out_dir: str | Path = "reports",
                      detector: Optional[FrameDetector] = None,
                      cfg: Optional[VadConfig] = None) -> dict:
    eval_dir = Path(eval_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fc_dir = out_dir / "failure_cases"
    det = detector or FrameDetector()

    clips = sorted(eval_dir.glob("*.wav"))
    metrics: list[dict] = []
    rows: list[dict] = []
    for wav in clips:
        ann_path = wav.with_suffix(".json")
        if not ann_path.exists():
            continue
        ann = load_annotation(ann_path)
        cm, cr = evaluate_clip(wav, ann, det, cfg)
        metrics.append(cm)
        rows.extend(cr)

    summary = _aggregate(metrics)
    summary["duration_buckets_global"] = _bucket_rollup(rows)
    summary["eval_dir"] = str(eval_dir)
    summary["backend"] = det.backend
    if cfg is not None:
        summary["config"] = cfg.to_dict()
    (out_dir / "metrics_summary.json").write_text(
        json.dumps({"summary": summary, "clips": metrics},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(out_dir / "event_metrics.csv", "w", newline="",
                  encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)

    _write_markdown(out_dir / "metrics_report.md", summary, metrics)
    _retain_failures(eval_dir, metrics, fc_dir)
    return {"summary": summary, "clips": metrics, "rows": rows}


def _is_failure(c: dict) -> tuple[bool, str]:
    reasons = []
    op = c["operational"]
    ev = c["events"]
    if op["voice_miss_rate"] >= 0.20:
        reasons.append(f"voice_miss_rate={op['voice_miss_rate']}")
    if ev["missed_events"] > 0:
        reasons.append(f"missed_events={ev['missed_events']}")
    if ev["false_events"] > 0:
        reasons.append(f"false_events={ev['false_events']}")
    if c["category"] == "low_snr_speech" and op["voice_miss_rate"] >= 0.35:
        reasons.append("low_snr_high_miss")
    return (bool(reasons), ";".join(reasons))


def _retain_failures(eval_dir: Path, metrics: list[dict], fc_dir: Path) -> None:
    if fc_dir.exists():
        shutil.rmtree(fc_dir)
    fc_dir.mkdir(parents=True, exist_ok=True)
    for c in metrics:
        fail, reason = _is_failure(c)
        low_snr = c["category"] == "low_snr_speech"
        if not fail and not low_snr:
            continue
        stem = Path(c["clip"]).stem
        dst = fc_dir / stem
        dst.mkdir(parents=True, exist_ok=True)
        for ext in (".wav", ".json"):
            src = eval_dir / f"{stem}{ext}"
            if src.exists():
                shutil.copy2(src, dst / src.name)
        (dst / "failure_report.json").write_text(json.dumps(
            {"reasons": reason or "retained low-SNR case", "metrics": c},
            ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(path: Path, s: dict, clips: list[dict]) -> None:
    L = []
    L.append("# 语音区间评估报告\n")
    L.append(f"- 后端：`{s['backend']}`")
    L.append(f"- 片段数：{s['n_clips']}，总时长：{s['total_duration_sec']}s，"
             f"其中精确语音：{s['exact_voice_sec']}s")
    L.append(f"- 长静音占比：**{s['overall_silence_ratio'] * 100:.1f}%**"
             "（因此总体准确率会被静音抬高，不能只看 accuracy）\n")
    fp = s["frame_pooled"]
    L.append("## 帧级（按语音时长复核）")
    L.append(f"- 总体准确率（仅供参考）：{fp['accuracy']}")
    L.append(f"- 语音帧漏判率：**{fp['speech_miss_rate']}**")
    L.append(f"- 非语音误帧率：{fp['nonspeech_fpr']}\n")
    L.append("## 区间/事件级（剪辑器真正使用的结果）")
    L.append(f"- 汇总语音时长漏检率：**{s['pooled_voice_miss_rate']}**")
    L.append(f"- 误触发总时长：{s['pooled_false_trigger_sec']}s"
             f"（{s['false_trigger_sec_per_minute']}s/分钟）")
    e = s["events_pooled"]
    L.append(f"- 参考事件 {e['reference']} / 检出 {e['detected']} / "
             f"匹配 {e['matched']}，漏事件 **{e['missed_events']}**，"
             f"误事件 **{e['false_events']}**")
    L.append(f"- 事件召回 {e['recall']} / 精确 {e['precision']} / F1 {e['f1']}"
             f"；误事件 {e['false_events_per_audio_minute']}/分钟")
    L.append(f"- 多对一覆盖率口径（允许合理桥接/重叠合段）：事件覆盖召回 "
             f"{e['coverage_event_recall_many_to_one']} / 区间有据精确 "
             f"{e['coverage_event_precision_many_to_one']}（覆盖阈值 0.5）")
    L.append(f"- 核心边界平均绝对误差：起点 {s.get('boundary_core_start_mean_abs')}ms"
             f" / 终点 {s.get('boundary_core_end_mean_abs')}ms\n")
    L.append("## 分类别")
    L.append("| 类别 | 片段 | 平均时长漏检率 | 漏事件 | 误事件 |")
    L.append("|---|---:|---:|---:|---:|")
    for k, d in s["categories"].items():
        L.append(f"| {k} | {d['clips']} | {d['mean_voice_miss_rate']} | "
                 f"{d['missed_events']} | {d['false_events']} |")
    L.append("\n## 各时长桶（按 union 后的活动事件）")
    L.append("| 时长桶 | 事件数 | 匹配 | 事件召回 | 起点MAE ms | 终点MAE ms | 终点p95 ms |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for b, d in s.get("duration_buckets_global", {}).items():
        L.append(f"| {b} | {d['events']} | {d['matched']} | {d['event_recall']} | "
                 f"{d['start_mae_ms']} | {d['end_mae_ms']} | {d['end_p95_abs_ms']} |")
    L.append("\n重叠的两人参考段已先 union 成声学活动区（VAD 不做说话人分离）；"
             "按原始短语/话轮统计的 split/merge 见 `metrics_summary.json`。")
    L.append("分事件明细（含 duration bucket、逐事件边界误差、区间时长漏检）"
             "见 `event_metrics.csv`。")
    path.write_text("\n".join(L), encoding="utf-8")
