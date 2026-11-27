"""Collect executable evidence for the VAD inference-runtime migration."""
from __future__ import annotations

import json
import platform
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from .audio import AudioBuffer, AudioPlayer, load_wav
from .detector import FrameDetector, labels_for_frames
from .evaluate import evaluate_dataset, frame_confusion
from .kernel import VoiceIntervalKernel
from .labels import active_sample_spans, load_annotation
from .pipeline import config_for
from .runtime_onnx import PINNED_RUNTIME_VERSION, installed_runtime_version

ROOT = Path(__file__).resolve().parent.parent
SHORT = ROOT / "data" / "labeled_short"


def _interval_rows(contract: dict) -> list[dict]:
    return [
        {"start_sample": s, "end_sample": e,
         "core_start_sample": cs, "core_end_sample": ce}
        for s, e, cs, ce in contract["offline"]
    ]


def _material_evidence(kernel: VoiceIntervalKernel, path: Path) -> dict:
    audio = load_wav(path)
    error = kernel.inference_error(audio)
    contract = kernel.block_contract(audio, block_sizes=(256, 512))
    return {
        "file": str(path.relative_to(ROOT)),
        "sample_rate": audio.sample_rate,
        "samples": len(audio.samples),
        "duration_sec": round(audio.duration, 3),
        "inference_error": error,
        "block_contract": {
            "block_sizes": [256, 512],
            "all_match_offline": contract["all_match"],
            "matches": {str(k): v for k, v in contract["matches_offline"].items()},
            "intervals": _interval_rows(contract),
        },
    }


def _generated_material_evidence(kernel: VoiceIntervalKernel) -> dict:
    original = load_wav(SHORT / "short_clean_A.wav")
    x = original.samples.astype(np.float32) / 32768.0
    quiet = AudioBuffer(
        np.clip(np.round(x * 0.12 * 32767), -32768, 32767).astype(np.int16),
        original.sample_rate)
    quiet_contract = kernel.block_contract(quiet, (256, 512))

    n = 80_000
    partial = AudioBuffer(original.samples[:n].copy(), original.sample_rate)
    partial_contract = kernel.block_contract(partial, (256, 512))
    return {
        "low_volume_gain": 0.12,
        "low_volume_peak_int16": int(np.max(np.abs(quiet.samples))),
        "low_volume_inference_error": kernel.inference_error(quiet),
        "low_volume_all_block_sizes_match": quiet_contract["all_match"],
        "low_volume_intervals": _interval_rows(quiet_contract),
        "partial_samples": n,
        "partial_mod_256": n % 256,
        "partial_mod_512": n % 512,
        "partial_all_block_sizes_match": partial_contract["all_match"],
        "partial_intervals": _interval_rows(partial_contract),
        "data_modification": "generated in memory; no data file was edited",
    }


def threshold_sweep_on_example(kernel: VoiceIntervalKernel) -> list[dict]:
    wav = SHORT / "interview_dialogue.wav"
    ann_path = SHORT / "interview_dialogue.json"
    audio = load_wav(wav)
    ann = load_annotation(ann_path)
    decision = kernel.detect(audio)
    exact = active_sample_spans(ann, audio.sample_rate)
    rows = []
    for enter in (0.300, 0.375, 0.450):
        cfg = config_for(
            kernel.detector, enter_threshold=enter,
            exit_threshold=round(max(0.12, enter - 0.16), 3))
        result = kernel.recompute(audio, decision, cfg, ann)
        y_true = labels_for_frames(decision.features.grid, exact,
                                   len(audio.samples))
        confusion = frame_confusion(y_true, decision.probs >= enter)
        missed_samples = sum(m.end_sample - m.start_sample
                             for m in result.missed)
        rows.append({
            "enter_threshold": enter,
            "exit_threshold": cfg.exit_threshold,
            "n_intervals": len(result.intervals),
            "interval_samples": sum(iv.end_sample - iv.start_sample
                                    for iv in result.intervals),
            "missed_regions": len(result.missed),
            "missed_samples": missed_samples,
            "missed_sec": round(missed_samples / audio.sample_rate, 3),
            "frame_metrics": confusion,
        })
    return rows


def validation_threshold_evidence(det: FrameDetector) -> list[dict]:
    val_dir = ROOT / "data" / "validation"
    rows = []
    for enter in (0.325, 0.375, 0.425):
        cfg = config_for(
            det, enter_threshold=enter,
            exit_threshold=round(max(0.12, enter - 0.16), 3))
        with tempfile.TemporaryDirectory(prefix="vad_threshold_") as td:
            result = evaluate_dataset(val_dir, Path(td), det, cfg)
        s = result["summary"]
        e = s["events_pooled"]
        rows.append({
            "enter_threshold": enter,
            "exit_threshold": cfg.exit_threshold,
            "n_clips": s["n_clips"],
            "pooled_voice_miss_rate": s["pooled_voice_miss_rate"],
            "false_trigger_sec_per_minute": s["false_trigger_sec_per_minute"],
            "frame_speech_miss_rate": s["frame_pooled"]["speech_miss_rate"],
            "frame_nonspeech_fpr": s["frame_pooled"]["nonspeech_fpr"],
            "event_recall": e["recall"],
            "event_precision": e["precision"],
            "event_f1": e["f1"],
            "coverage_event_recall": e["coverage_event_recall_many_to_one"],
        })
    return rows


def collect_evidence(json_out: str | Path = ROOT / "reports" /
                     "runtime_migration_evidence.json",
                     head_dir: str | Path = ROOT / "exports" /
                     "runtime_migration",
                     run_validation: bool = True) -> dict:
    det = FrameDetector(runtime="onnx")
    kernel = VoiceIntervalKernel(det)
    head_dir = Path(head_dir)
    head = kernel.export_and_read_back_head(
        load_wav(SHORT / "short_clean_A.wav"), head_dir)
    player = AudioPlayer()

    materials = [
        "short_clean_A.wav", "interview_dialogue.wav", "low_snr_case.wav",
        "wind_only.wav", "keyboard_only.wav", "music_only.wav",
    ]
    report = {
        "schema": "vad-runtime-migration-evidence-v1",
        "runtime": {
            "name": "onnxruntime",
            "installed_version": installed_runtime_version(),
            "pinned_version": PINNED_RUNTIME_VERSION,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "backend": det.backend,
            "identity": det.model.identity,
        },
        "old_to_new_time_contract": {
            "legacy_frame_timestamp": "center=i*hop+win//2",
            "migrated_output_timestamp": "start=i*hop",
            "mapping": "new_start=old_center-win//2=i*hop",
            "selected_live_compensation": "feature_side_preroll_once",
            "editor_offset_policy": "use interval.start_sample directly; "
                                    "do not apply AlignmentLedger twice",
        },
        "materials": [_material_evidence(kernel, SHORT / n) for n in materials],
        "generated_materials": _generated_material_evidence(kernel),
        "editor_head_readback": head.to_dict(),
        "playback": {
            "detected_backend": player.backend,
            "available": player.available(),
            "acoustic_playback_executed": False,
            "note": "Evidence collector never claims listening success in a "
                    "container without /dev/snd. GUI/player path should be "
                    "checked on a desktop or PulseAudio/PipeWire session.",
        },
        "threshold_example_check": threshold_sweep_on_example(kernel),
    }
    if run_validation:
        report["threshold_validation"] = validation_threshold_evidence(det)

    json_out = Path(json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    return report


def main() -> int:
    r = collect_evidence()
    print(json.dumps({
        "evidence": str(ROOT / "reports" / "runtime_migration_evidence.json"),
        "backend": r["runtime"]["backend"],
        "head_verified": r["editor_head_readback"]["verification"]["verified"],
        "materials": len(r["materials"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
