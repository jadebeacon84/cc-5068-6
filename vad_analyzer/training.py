"""Train the local logistic model from a generated/labeled clip folder."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .audio import load_wav
from .detector import DEFAULT_WEIGHTS, labels_for_frames
from .features import extract_features
from .labels import active_sample_spans, load_annotation
from .model import LocalModel, fit_logistic


def collect_folder(root: str | Path):
    root = Path(root)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    clips = 0
    for wav in sorted(root.glob("*.wav")):
        ann_path = wav.with_suffix(".json")
        if not ann_path.exists():
            continue
        ann = load_annotation(ann_path)
        audio = load_wav(wav)
        fx = extract_features(audio)
        spans = active_sample_spans(ann, audio.sample_rate)
        if spans is None:
            # Fall back to phrase segments as positive spans.
            spans = [(int(round(s["start"] * audio.sample_rate)),
                      int(round(s["end"] * audio.sample_rate)))
                     for s in ann["segments"] if s.get("kind", "speech") == "speech"]
        y = labels_for_frames(fx.grid, spans, len(audio.samples), overlap=0.45)
        xs.append(fx.x)
        ys.append(y.astype(float))
        clips += 1
    if not xs:
        raise FileNotFoundError(f"{root} 下没有找到 wav+json 训练样本")
    return np.concatenate(xs), np.concatenate(ys), clips


def train_model(train_dir: str | Path,
                out_weights: str | Path = DEFAULT_WEIGHTS) -> tuple[LocalModel, dict]:
    x, y, clips = collect_folder(train_dir)
    model, info = fit_logistic(x, y)
    # Pick an operating threshold favoring speech recall on the training data;
    # independent evaluation is what reports real generalization.
    p = model.predict_proba(x)
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.25, 0.75, 21):
        pred = p >= t
        tp = np.sum(pred & (y > 0.5))
        fn = np.sum(~pred & (y > 0.5))
        fp = np.sum(pred & (y < 0.5))
        rec = tp / max(1, tp + fn)
        prec = tp / max(1, tp + fp)
        score = 0.7 * rec + 0.3 * prec
        if score > best_score:
            best_score, best_t = score, float(t)
    model.threshold = round(best_t, 3)
    model.metadata.update({
        "trained_on_dir": str(train_dir),
        "n_clips": clips,
        "operating_threshold_note": "chosen on train for 0.7 recall + 0.3 precision",
        "operating_threshold_score": float(best_score),
    })
    model.save(out_weights)
    info["operating_threshold"] = model.threshold
    return model, info
