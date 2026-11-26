"""Curated scenarios that map directly to the acceptance checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import synth
from .synth import SR, Scene


def _spans_offset(spans, at: float) -> list[tuple[int, int]]:
    sh = int(at * SR)
    return [(a + sh, b + sh) for a, b in spans]


def build_dialogue(seed: int = 11):
    """Two-speaker interview-ish dialogue with breaths, masked tail, music."""
    rng = np.random.default_rng(seed)
    dur = 11.2
    sc = Scene(dur)
    segments: list[dict] = []
    active: list[tuple[int, int]] = []

    def phrase(at: float, d: float, f0: float, spk: str, text: str = "",
               level: float = -19.0):
        x, spans = synth.speech_phrase(d, rng, f0=f0, level_dbfs=level)
        sc.add(x, at)
        rs = at + (spans[0][0] / SR if spans else 0.0)
        re_ = at + (spans[-1][1] / SR if spans else d)
        segments.append({"start": round(rs, 3), "end": round(re_, 3),
                         "speaker": spk, "kind": "speech", "text": text})
        active.extend(_spans_offset(spans, at))

    # A first turn with an internal short pause (gap tolerance must bridge it).
    phrase(0.45, 2.05, 128.0, "A", "今天我们聊一下语音区间。")
    sc.add(synth.breath_sound(0.55, rng, -27), 2.55)

    # B turn.
    phrase(2.95, 1.75, 218.0, "B", "好的，我最关心剪辑点会不会切错。")

    # Brief overlap: A back-channel lands on B's tail, with breaths around it.
    sc.add(synth.breath_sound(0.35, rng, -30), 4.15)
    phrase(4.35, 0.62, 132.0, "A", "对。", level=-20)
    phrase(4.15, 0.85, 222.0, "B", "就是这样。", level=-20)
    sc.add(synth.breath_sound(0.5, rng, -27), 4.95)

    phrase(5.35, 1.35, 125.0, "A", "所以进入和退出证据要分开。")

    # B phrase with a weak final tail masked by a swelling music bed.
    phrase(6.9, 1.25, 215.0, "B", "尾音其实很轻。", level=-22)
    tail, tspans = synth.speech_phrase(0.55, rng, f0=210.0, level_dbfs=-27)
    sc.add(tail, 8.05)
    trs = 8.05 + (tspans[0][0] / SR if tspans else 0.0)
    tre = 8.05 + (tspans[-1][1] / SR if tspans else 0.55)
    segments.append({"start": round(trs, 3), "end": round(tre, 3),
                     "speaker": "B", "kind": "speech", "text": "嗯……"})
    active.extend(_spans_offset(tspans, 8.05))
    sc.add(synth.music_bed(3.6, rng, -22, swell_at=0.5), 7.7)
    sc.add(synth.breath_sound(0.7, rng, -28), 8.75)

    # Keyboard clicks in the music gap (interference, should not start a cut).
    sc.add(synth.keyboard_sound(0.7, rng, 3.0, -25), 9.0)

    phrase(9.55, 1.45, 128.0, "A", "这些干扰都不能只靠音量判断。", level=-19)

    i16 = sc.mix_i16()
    ann = synth.annotation_from_scene(
        "interview_dialogue.wav", i16, segments, active,
        scenario="two-speaker dialogue with breath overlap and music-masked tail")
    return i16, ann


@dataclass
class ForkPair:
    audio_a: np.ndarray
    audio_b: np.ndarray
    divergence_sample: int
    sample_rate: int
    annotation_a: dict
    annotation_b: dict

    @property
    def divergence_sec(self) -> float:
        return self.divergence_sample / SR


def build_fork_pair(seed: int = 23, total: float = 3.2,
                    divergence: float = 2.6):
    """Identical prefix; branch A has late speech, branch B only room tone."""
    rng = np.random.default_rng(seed)
    div_n = int(divergence * SR)
    total_n = int(total * SR)

    common_scene = Scene(divergence)
    segs_common: list[dict] = []
    active_common: list[tuple[int, int]] = []
    x, spans = synth.speech_phrase(1.85, rng, f0=150.0, level_dbfs=-19)
    common_scene.add(x, 0.25)
    crs = 0.25 + (spans[0][0] / SR if spans else 0.0)
    cre = 0.25 + (spans[-1][1] / SR if spans else 2.1)
    segs_common.append({"start": round(crs, 3), "end": round(cre, 3),
                       "speaker": "X", "kind": "speech",
                       "text": "相同前缀里的一句话。"})
    active_common.extend(_spans_offset(spans, 0.25))
    common_scene.add(synth.white_noise(divergence, rng, -42), 0.0)
    common = common_scene.mix()[:div_n]
    if len(common) < div_n:
        common = np.pad(common, (0, div_n - len(common)))

    tail_n = total_n - div_n
    # Branch A: immediately-voiced syllable right at the fork (retracts report).
    a_tail, a_spans = synth.speech_phrase((tail_n / SR) + 0.2,
                                          np.random.default_rng(seed + 1),
                                          f0=150.0, level_dbfs=-19)
    a_tail = a_tail[:tail_n]
    # Force a sharp voiced onset at the very first frames for determinism.
    tt = np.arange(tail_n) / SR
    onset = np.minimum(1.0, tt / 0.015)
    force = 0.18 * np.sin(2 * np.pi * 150 * tt) * onset
    a_tail = np.maximum(a_tail, force.astype(np.float32))
    audio_a_f = np.concatenate([common, a_tail])[:total_n]

    # Branch B: same room-tone level only.
    b_tail = synth.white_noise(tail_n / SR, np.random.default_rng(seed + 2), -42)
    b_tail = b_tail[:tail_n]
    audio_b_f = np.concatenate([common, b_tail])[:total_n]

    from .audio import pcm16_from_float
    a16 = pcm16_from_float(audio_a_f)
    b16 = pcm16_from_float(audio_b_f)
    assert np.array_equal(a16[:div_n], b16[:div_n]), "fork prefix must be identical"

    seg_a = segs_common + [{"start": divergence, "end": total, "speaker": "X",
                            "kind": "speech", "text": "分岔后的继续语音。"}]
    active_a = active_common + [(div_n + a, min(total_n, div_n + b))
                                for a, b in a_spans
                                if div_n + a < total_n]
    ann_a = synth.annotation_from_scene("fork_A_speech_resumes.wav", a16,
                                        seg_a, active_a, scenario="fork: late speech")
    ann_b = synth.annotation_from_scene(
        "fork_B_silence.wav", b16, segs_common, active_common,
        scenario="fork: no continuation")
    return ForkPair(a16, b16, div_n, SR, ann_a, ann_b)


def build_interference(kind: str, dur: float = 4.0, seed: int = 5):
    rng = np.random.default_rng(seed)
    sc = Scene(dur)
    if kind == "wind":
        sc.add(synth.wind_sound(dur, rng, -15))
    elif kind == "keyboard":
        sc.add(synth.keyboard_sound(dur, rng, 2.6, -23))
    elif kind == "music":
        sc.add(synth.music_bed(dur, rng, -20))
    elif kind == "breath":
        for at in np.arange(0.4, dur, 1.1):
            sc.add(synth.breath_sound(0.7, rng, -25), float(at))
    else:
        raise ValueError(kind)
    i16 = sc.mix_i16()
    ann = synth.annotation_from_scene(
        f"{kind}_only.wav", i16, [], [], category=f"interference_{kind}")
    return i16, ann
