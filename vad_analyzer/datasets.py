"""Generation of the labeled short set, training set and independent eval set."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .audio import save_wav
from .scenarios import build_dialogue, build_fork_pair, build_interference
from .synth import SR, Scene
from . import synth

CATEGORIES = [
    "clean_speech", "noisy_speech", "low_snr_speech", "music_speech",
    "overlap_speech", "breath_gaps", "interf_wind", "interf_keyboard",
    "interf_music",
]


def _offset(spans, at_sr):
    return [(a + at_sr, b + at_sr) for a, b in spans]


def build_random_clip(seed: int, category: Optional[str] = None):
    rng = np.random.default_rng(seed)
    if category is None:
        category = CATEGORIES[int(rng.integers(0, len(CATEGORIES)))]
    dur = float(rng.uniform(4.5, 8.5))
    sc = Scene(dur)
    segments: list[dict] = []
    active: list[tuple[int, int]] = []
    snr = None

    is_interf = category.startswith("interf_")
    if is_interf:
        kind = category.split("_", 1)[1]
        x, _ = build_interference(kind, dur, seed=seed)
        return x, synth.annotation_from_scene(
            f"{category}_{seed}.wav", x, [], [], category=category, seed=seed)

    # Optional beds / noise depending on category.
    if category in ("noisy_speech", "low_snr_speech"):
        snr = float(rng.choice([8, 12, 16]) if category == "noisy_speech"
                    else rng.choice([-2, 2, 5]))
        noise_level = -19.0 - snr
        sc.add(synth.white_noise(dur, rng, noise_level))
    if category in ("music_speech",):
        sc.add(synth.music_bed(dur, rng, -25))
        snr = float(rng.choice([6, 10, 14]))
        sc.add(synth.white_noise(dur, rng, -19.0 - snr))
    if category == "breath_gaps":
        sc.add(synth.white_noise(dur, rng, -40))

    overlap = category == "overlap_speech"
    t = float(rng.uniform(0.2, 0.5))
    n_phrases = int(rng.integers(2, 5))
    for k in range(n_phrases):
        pd = float(rng.uniform(1.1, 2.4))
        f0 = float(rng.choice([rng.normal(128, 14), rng.normal(215, 18)]))
        level = -19.0 if category != "music_speech" else -20.0
        x, spans = synth.speech_phrase(pd, rng, f0=max(80, f0),
                                       level_dbfs=level)
        sc.add(x, t)
        real_start = t + (spans[0][0] / SR if spans else 0.0)
        real_end = t + (spans[-1][1] / SR if spans else pd)
        spk = "A" if f0 < 170 else "B"
        segments.append({"start": round(real_start, 3),
                         "end": round(real_end, 3),
                         "scheduled_start": round(t, 3),
                         "speaker": spk, "kind": "speech"})
        active.extend(_offset(spans, int(t * SR)))
        if overlap and k > 0:
            t += pd * float(rng.uniform(0.35, 0.6))   # overlapping turns
        else:
            gap = float(rng.uniform(0.18, 0.75))
            t += pd + gap
            if category == "breath_gaps" and rng.random() < 0.7:
                sc.add(synth.breath_sound(min(0.7, gap * 0.9), rng, -26),
                       t - gap + 0.05)
        if t > dur - 1.2:
            break

    i16 = sc.mix_i16()
    ann = synth.annotation_from_scene(
        f"{category}_{seed}.wav", i16, segments, active,
        category=category, seed=seed, snr_db=snr)
    return i16, ann


def write_clip(folder: Path, stem: str, i16: np.ndarray, ann: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    wav = folder / f"{stem}.wav"
    save_wav(wav, i16, SR)
    ann = dict(ann)
    ann["audio_file"] = wav.name
    (folder / f"{stem}.json").write_text(
        json.dumps(ann, ensure_ascii=False, indent=2), encoding="utf-8")
    return wav


def generate_labeled_short(root: str | Path) -> list[Path]:
    root = Path(root)
    out: list[Path] = []
    i16, ann = build_dialogue()
    out.append(write_clip(root, "interview_dialogue", i16, ann))

    fp = build_fork_pair()
    out.append(write_clip(root, "fork_A_speech_resumes", fp.audio_a, fp.annotation_a))
    out.append(write_clip(root, "fork_B_silence", fp.audio_b, fp.annotation_b))

    curated = [
        ("short_clean_A", "clean", 101),
        ("overlap_turns", "overlap_speech", 102),
        ("breath_and_pause", "breath_gaps", 103),
        ("low_snr_case", "low_snr_speech", 104),
        ("music_masked", "music_speech", 105),
        ("wind_only", "interf_wind", 106),
        ("keyboard_only", "interf_keyboard", 107),
        ("music_only", "interf_music", 108),
    ]
    for stem, cat, seed in curated:
        i16, ann = build_random_clip(seed, cat)
        out.append(write_clip(root, stem, i16, ann))
    return out


def generate_batch(root: str | Path, n_per_category: int, seed_base: int,
                   name: str) -> dict:
    root = Path(root)
    manifest = {"name": name, "sample_rate": SR, "clips": []}
    idx = 0
    for ci, cat in enumerate(CATEGORIES):
        for j in range(n_per_category):
            seed = seed_base + ci * 1000 + j
            i16, ann = build_random_clip(seed, cat)
            stem = f"{cat}_{j:02d}_{seed}"
            write_clip(root, stem, i16, ann)
            manifest["clips"].append({
                "stem": stem, "wav": f"{stem}.wav",
                "annotation": f"{stem}.json", "category": cat,
                "snr_db": ann.get("snr_db"), "seed": seed})
            idx += 1
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
