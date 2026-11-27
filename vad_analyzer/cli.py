"""Command-line helper for voice-activity intervals.

The graphical "声音分析窗口 / 语音区间窗口" is the primary entry point
(``python -m vad_analyzer gui``); this CLI is for scripting, batch evaluation
and sample-exact clip export.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from ._env import WORKSPACE
from .errors import VadError


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vad-analyzer",
        description="本地语音活动区间分析器（帧判定 / 区间合并分离，导出可核验片段）")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("gui", help="打开图形声音分析窗口（主要入口）")

    a = sub.add_parser("analyze", help="分析一个 WAV 并输出语音活动区间")
    a.add_argument("audio", help="输入音频（建议 PCM WAV）")
    a.add_argument("--threshold", type=float, default=None,
                   help="进入阈值 enter_threshold（默认用本地模型选定值）")
    a.add_argument("--exit-threshold", type=float, default=None,
                   help="退出阈值（默认 threshold-0.16，下限 0.12）")
    a.add_argument("--min-speech-ms", type=int, default=None)
    a.add_argument("--min-strong-ratio", type=float, default=None)
    a.add_argument("--gap-ms", type=int, default=None, dest="gap_ms")
    a.add_argument("--pad-ms", type=int, default=None, dest="pad_ms")
    a.add_argument("--annotation", default=None, help="参考标注 JSON（用于漏检）")
    a.add_argument("--no-sidecar", action="store_true",
                   help="不自动读取同名 .json 标注")
    a.add_argument("--cuts", default=None, help="锁定切点 JSON（只报告冲突，不移动）")
    a.add_argument("--json-out", default=None, help="把完整结果写入 JSON")
    a.add_argument("--export-clips", default=None, metavar="DIR",
                   help="把每个活动区间导出为 WAV 并逐样本核验")

    g = sub.add_parser("generate-data", help="生成本地合成标注短录音与数据集")
    g.add_argument("--root", default=str(WORKSPACE / "data"))
    g.add_argument("--train-n", type=int, default=8)
    g.add_argument("--eval-n", type=int, default=4)
    g.add_argument("--val-n", type=int, default=6)
    g.add_argument("--no-batches", action="store_true")

    t = sub.add_parser("train", help="在标注数据集上训练本地逻辑回归权重")
    t.add_argument("--train-dir", default=str(WORKSPACE / "data" / "train"))

    e = sub.add_parser("evaluate", help="在独立评估集上计算可复核指标")
    e.add_argument("--eval-dir", default=str(WORKSPACE / "data" / "eval"))
    e.add_argument("--out", default=str(WORKSPACE / "reports"))

    d = sub.add_parser("demo", help="运行分岔/锁点/导出核验自检场景")
    d.add_argument("--out-dir", default=str(WORKSPACE / "exports" / "demo"))

    r = sub.add_parser(
        "runtime-check",
        help="核验 ONNX Runtime 迁移、时间映射、块大小契约和字头回读")
    r.add_argument("--json-out",
                   default=str(WORKSPACE / "reports" /
                               "runtime_migration_evidence.json"))
    r.add_argument("--head-dir",
                   default=str(WORKSPACE / "exports" / "runtime_migration"))
    r.add_argument("--no-validation", action="store_true",
                   help="跳过 validation 集阈值扫描，仅做样本级核验")
    return p


def _config_kwargs(args) -> dict:
    kw = {}
    if args.threshold is not None:
        kw["enter_threshold"] = args.threshold
        if args.exit_threshold is None:
            kw["exit_threshold"] = round(max(0.12, args.threshold - 0.16), 3)
    if args.exit_threshold is not None:
        kw["exit_threshold"] = args.exit_threshold
    if args.min_speech_ms is not None:
        kw["min_speech_ms"] = args.min_speech_ms
    if args.min_strong_ratio is not None:
        kw["min_strong_ratio"] = args.min_strong_ratio
    if args.gap_ms is not None:
        kw["gap_tolerance_ms"] = args.gap_ms
    if args.pad_ms is not None:
        kw["edge_pad_ms"] = args.pad_ms
    return kw


def _cmd_analyze(args) -> int:
    from .audio import load_wav
    from .cuts import load_locked_cuts
    from .detector import FrameDetector
    from .labels import find_sidecar, load_annotation
    from .pipeline import analyze_audio, config_for

    audio = load_wav(args.audio)
    det = FrameDetector()
    cfg = config_for(det, **_config_kwargs(args))

    ann = None
    ann_path = args.annotation
    if ann_path is None and not args.no_sidecar:
        side = find_sidecar(args.audio)
        ann_path = str(side) if side else None
    if ann_path:
        ann = load_annotation(ann_path)
        print(f"[标注] 使用参考标注：{ann_path}")

    cuts = None
    if args.cuts:
        cuts = load_locked_cuts(args.cuts, audio.sample_rate)
        print(f"[锁点] 载入 {len(cuts)} 个锁定切点（分析器不会移动它们）")

    res = analyze_audio(audio, cfg, det, ann, cuts)
    print(f"[后端] {res.backend}")
    print(f"[参数] enter={cfg.enter_threshold} exit={cfg.exit_threshold} "
          f"min_speech={cfg.min_speech_ms}ms strong_ratio={cfg.min_strong_ratio} "
          f"gap={cfg.gap_tolerance_ms}ms pad={cfg.edge_pad_ms}ms")
    print(f"\n活动区间 {len(res.intervals)} 个：")
    for iv in res.intervals:
        print(f"  #{iv.index:<2d} {iv.start:7.3f}-{iv.end:7.3f}s "
              f"samples[{iv.start_sample}:{iv.end_sample}] "
              f"conf={iv.confidence:.2f}/{iv.confidence_low:.2f} "
              f"strong={iv.strong_frame_ratio:.2f} level={iv.confidence_level}")

    if res.missed:
        print(f"\n漏检片段 {len(res.missed)} 个（参考语音 − 检出并集，真实区间差）：")
        for m in res.missed:
            print(f"  {m.start:7.3f}-{m.end:7.3f}s ({m.duration:.2f}s) "
                  f"samples[{m.start_sample}:{m.end_sample}]")
    elif ann is not None:
        print("\n漏检片段：无（参考语音已被检出区间并集覆盖）")

    if res.conflicts:
        print(f"\n锁定切点冲突 {len(res.conflicts)} 个（只报告，不修改锁点）：")
        for c in res.conflicts:
            print(f"  [{c.severity:7s}] {c.cut_id} @{c.cut_time_sec:.3f}s "
                  f"{c.kind}: {c.message}")

    verifications = []
    if args.export_clips:
        from .exporter import export_interval
        out_dir = Path(args.export_clips)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n导出并逐样本核验到：{out_dir}")
        for iv in res.intervals:
            path, ver = export_interval(audio, iv, out_dir)
            verifications.append(ver.to_dict())
            print(f"  #{iv.index:<2d} {path.name}")
            print(f"      samples {ver.requested_start_sample}:"
                  f"{ver.requested_end_sample} n={ver.expected_sample_count} "
                  f"verified={ver.verified} all_match={ver.all_samples_match}")
            print(f"      first exp/act={ver.first_sample_expected}/"
                  f"{ver.first_sample_actual}  last exp/act="
                  f"{ver.last_sample_expected}/{ver.last_sample_actual}")
            if not ver.verified:
                print("      !! 导出核验失败", file=sys.stderr)

    if args.json_out:
        payload = res.summary()
        payload["export_verifications"] = verifications
        Path(args.json_out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2,
                       default=_json_default), encoding="utf-8")
        print(f"\n[JSON] 完整结果已写入 {args.json_out}")
    return 0


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def _cmd_runtime_check(args) -> int:
    from .migration_evidence import collect_evidence
    report = collect_evidence(
        args.json_out, args.head_dir, run_validation=not args.no_validation)
    head = report["editor_head_readback"]["verification"]["verified"]
    contracts = all(m["block_contract"]["all_match_offline"]
                    for m in report["materials"])
    print(f"[runtime] {report['runtime']['backend']}")
    print(f"[head-readback] verified={head}")
    print(f"[block-contracts] all_match={contracts} blocks=256,512")
    print(f"[json] {args.json_out}")
    return 0 if head and contracts else 1


def _cmd_gui(_args) -> int:
    from .gui import main as gui_main
    return gui_main()


def _cmd_generate_data(args) -> int:
    from .datasets import (generate_batch, generate_labeled_short)
    root = Path(args.root)
    short = generate_labeled_short(root / "labeled_short")
    print(f"[data] labeled_short: {len(short)} clips -> {root/'labeled_short'}")
    if not args.no_batches:
        tr = generate_batch(root / "train", args.train_n, 1000, "train")
        print(f"[data] train: {len(tr['clips'])} clips -> {root/'train'}")
        va = generate_batch(root / "validation", args.val_n, 5000, "validation")
        print(f"[data] validation: {len(va['clips'])} clips -> {root/'validation'}")
        ev = generate_batch(root / "eval", args.eval_n, 9000, "eval")
        print(f"[data] eval: {len(ev['clips'])} clips -> {root/'eval'}")
    return 0


def _cmd_train(args) -> int:
    from .training import train_model
    _model, info = train_model(args.train_dir)
    print("[train] 完成：")
    for k, v in info.items():
        if k == "loss_history":
            continue
        print(f"  {k}: {v}")
    return 0


def _cmd_evaluate(args) -> int:
    from .evaluate import evaluate_dataset
    r = evaluate_dataset(args.eval_dir, args.out)
    s = r["summary"]
    print(f"[eval] clips={s['n_clips']} duration={s['total_duration_sec']}s "
          f"silence={s['overall_silence_ratio']}")
    print(f"[eval] pooled voice miss={s['pooled_voice_miss_rate']}  "
          f"false trigger={s['false_trigger_sec_per_minute']}s/min")
    e = s["events_pooled"]
    print(f"[eval] strict event P/R/F1={e['precision']}/{e['recall']}/{e['f1']}  "
          f"coverage recall={e['coverage_event_recall_many_to_one']}")
    print(f"[eval] report -> {Path(args.out)/'metrics_report.md'}")
    return 0


def _cmd_demo(args) -> int:
    import numpy as np
    from .audio import AudioBuffer
    from .cuts import evaluate_locked_cuts, LockedCut
    from .exporter import export_interval
    from .pipeline import config_for
    from .detector import FrameDetector
    from .scenarios import build_fork_pair

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    det = FrameDetector()
    cfg = config_for(det)
    fp = build_fork_pair()

    from .stream import run_streaming
    print("== 分岔场景：相同前缀，2.6s 处分岔 ==")
    for name, x in [("A_speech_resumes", fp.audio_a), ("B_room_tone", fp.audio_b)]:
        ab = AudioBuffer(samples=x, sample_rate=fp.sample_rate)
        st = run_streaming(ab, cfg, det)
        kinds = [e.kind for e in st.events]
        print(f"  {name}: provisional={kinds.count('provisional_end')} "
              f"retracted={kinds.count('end_retracted')} "
              f"confirmed={kinds.count('segment_confirmed')}")
        assert kinds.count("provisional_end") == 1
        if name.startswith("A"):
            assert kinds.count("end_retracted") == 1, "A 必须撤回暂报"
            assert len(st.confirmed) == 1, "A 撤回后应合并为单段"
        else:
            assert kinds.count("end_retracted") == 0, "B 不应撤回"
            assert st.confirmed[0].end_sample <= fp.divergence_sample + \
                int(0.12 * fp.sample_rate)

    print("== 导出首尾样本核验 ==")
    ab = AudioBuffer(samples=fp.audio_a, sample_rate=fp.sample_rate)
    st = run_streaming(ab, cfg, det)
    path, ver = export_interval(ab, st.confirmed[0], out_dir)
    print(f"  {path.name} verified={ver.verified} "
          f"first={ver.first_sample_actual} last={ver.last_sample_actual}")
    assert ver.verified

    print("== 锁定切点只报告冲突 ==")
    iv = st.confirmed[0]
    cut = LockedCut("L1", (iv.core_start_sample + iv.core_end_sample) // 2,
                    fp.sample_rate)
    before = cut.sample
    conflicts = evaluate_locked_cuts([cut], [iv])
    assert conflicts and conflicts[0].severity == "error"
    assert cut.sample == before, "锁点样本绝不能被修改"
    print(f"  {conflicts[0].cut_id}: {conflicts[0].kind} / "
          f"{conflicts[0].severity}；锁点仍在 {cut.sample}")
    print("DEMO_OK")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "gui":
            return _cmd_gui(args)
        if args.command == "analyze":
            return _cmd_analyze(args)
        if args.command == "generate-data":
            return _cmd_generate_data(args)
        if args.command == "train":
            return _cmd_train(args)
        if args.command == "evaluate":
            return _cmd_evaluate(args)
        if args.command == "demo":
            return _cmd_demo(args)
        if args.command == "runtime-check":
            return _cmd_runtime_check(args)
        parser.error(f"未知命令 {args.command}")
        return 2
    except VadError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"文件不存在：{e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
