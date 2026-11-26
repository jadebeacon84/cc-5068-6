"""Deterministic local synthesis of labeled acoustic scenes.

This is *test fixture* audio generated from first principles (no network, no
external model). It gives exact ground-truth spans and deliberately includes
non-speech loud sounds (wind, keyboard, music) so the detector cannot lean on
amplitude alone. The "voice" is a formant/harmonic + syllable-envelope model,
not a recording of a person; README states this limitation honestly.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .audio import pcm16_from_float

SR = 16000
VOWELS = {
    "a": (730.0, 1090.0, 2440.0),
    "i": (270.0, 2290.0, 3010.0),
    "u": (300.0, 870.0, 2240.0),
    "e": (530.0, 1840.0, 2480.0),
    "o": (570.0, 840.0, 2410.0),
}


def _n(dur: float) -> int:
    return max(1, int(round(dur * SR)))


def db_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    y = np.zeros(n, dtype=np.float32)
    y[:len(x)] = x
    return y


def scaled_rms(x: np.ndarray, target_dbfs: float) -> np.ndarray:
    r = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
    target = db_gain(target_dbfs)
    return (x * (target / r)).astype(np.float32)


def speech_phrase(dur: float, rng: np.random.Generator, f0: float = 130.0,
                  level_dbfs: float = -20.0,
                  brightness: float = 1.0) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Formant-shaped harmonic source with syllabic amplitude gate.

    Returns mono float and exact active sample spans.
    """
    n = _n(dur)
    tt = np.arange(n) / SR
    # Gentle declination + vibrato + slow random drift of fundamental.
    drift = np.cumsum(rng.normal(0.0, 0.4, n)) / SR
    drift = np.clip(drift, -8, 8)
    f0_contour = f0 * (1.0 - 0.06 * tt / max(dur, 0.1)) \
        + 3.5 * np.sin(2 * np.pi * 5.2 * tt) + drift
    phase = 2 * np.pi * np.cumsum(np.maximum(60.0, f0_contour)) / SR

    # Syllable amplitude train (~3.5-5 syllables/sec) with 1-2 short pauses.
    syl_rate = rng.uniform(3.4, 4.8)
    n_syl = max(1, int(dur * syl_rate))
    env = np.zeros(n, dtype=np.float32)
    pause_after = set(rng.choice(np.arange(1, max(2, n_syl)),
                                 size=min(2, max(0, n_syl // 4)),
                                 replace=False).tolist()) if n_syl > 3 else set()
    pos = 0.12 + rng.uniform(0, 0.1)
    for s in range(n_syl):
        sd = rng.uniform(0.12, 0.22)
        a0 = int(pos * SR)
        length = _n(sd * 1.7)
        if a0 < n:
            lt = np.arange(length) / SR
            attack = np.minimum(1.0, lt / rng.uniform(0.02, 0.05))
            decay = np.exp(-lt / rng.uniform(0.07, 0.13))
            pulse = attack * decay * rng.uniform(0.75, 1.0)
            end = min(n, a0 + length)
            env[a0:end] += pulse[:end - a0]
        pos += sd + rng.uniform(0.03, 0.09)
        if s in pause_after:
            pos += rng.uniform(0.16, 0.30)
    env = np.clip(env, 0, 1)
    # Phrase-level fade in/out.
    fi = _n(0.03); fo = _n(0.06)
    env[:fi] *= np.linspace(0, 1, fi)
    env[-fo:] *= np.linspace(1, 0, fo)

    # Move formants between two vowel targets for spectral dynamics.
    v1, v2 = rng.choice(list(VOWELS.values()), size=2)
    move = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.6, 1.4) * tt
                              + rng.uniform(0, 6))
    k_max = int(min(26, SR / 2 / max(70.0, f0)))
    sig = np.zeros(n, dtype=np.float64)
    kf = np.maximum(80.0, f0_contour)
    for k in range(1, k_max + 1):
        freq = k * kf
        tilt = 1.0 / (k ** (1.15 + 0.25 * brightness))
        form = np.zeros(n)
        for j in range(3):
            fj = v1[j] * (1 - move) + v2[j] * move
            bw = (90 + 60 * j) * (1.3 - 0.3 * brightness)
            form += rng.uniform(0.7, 1.2) * np.exp(-((freq - fj) / bw) ** 2)
        sig += tilt * (0.35 + form) * np.sin(k * phase + 0.3 * k)
    sig /= max(1e-6, np.abs(sig).max())
    breath_noise = rng.normal(0, 1, n) * env * 0.05
    sig = (sig * env + breath_noise).astype(np.float32)
    sig = scaled_rms(sig, level_dbfs)

    active = env > 0.10
    spans = _mask_spans(active)
    return sig, spans


def _mask_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        spans.append((i, j))
        i = j
    return spans


def breath_sound(dur: float, rng: np.random.Generator,
                 level_dbfs: float = -26.0) -> np.ndarray:
    n = _n(dur)
    x = rng.normal(0, 1, n).astype(np.float32)
    sp = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    # Band-pass around 700-2500 Hz with a soft peak.
    mask = np.exp(-((f - 1400) / 700) ** 2) + 0.25 * np.exp(-f / 400)
    sp *= mask
    y = np.fft.irfft(sp, n).astype(np.float32)
    tt = np.arange(n) / SR
    env = np.sin(np.pi * np.clip(tt / dur, 0, 1)) ** 1.5
    y = y * env
    return scaled_rms(y, level_dbfs)


def wind_sound(dur: float, rng: np.random.Generator,
               level_dbfs: float = -16.0) -> np.ndarray:
    n = _n(dur)
    white = rng.normal(0, 1, n).astype(np.float32)
    brown = np.cumsum(white)
    brown -= np.linspace(brown[0], brown[-1], n)
    brown /= np.abs(brown).max() + 1e-6
    # Heavy low-pass feel by smoothing.
    k = 40
    kernel = np.ones(k, dtype=np.float32) / k
    low = np.convolve(brown, kernel, mode="same")
    tt = np.arange(n) / SR
    lfo = 0.55 + 0.45 * (0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.1, 0.35) * tt
                                           + rng.uniform(0, 6)))
    gust = 0.75 + 0.25 * np.sin(2 * np.pi * 0.07 * tt + 2.0)
    y = (low * lfo * gust).astype(np.float32)
    return scaled_rms(y, level_dbfs)


def keyboard_sound(dur: float, rng: np.random.Generator,
                   clicks_per_sec: float = 2.2,
                   level_dbfs: float = -24.0) -> np.ndarray:
    n = _n(dur)
    y = np.zeros(n, dtype=np.float32)
    t = rng.uniform(0.05, 0.25)
    while t < dur - 0.05:
        cl = rng.uniform(0.008, 0.045)
        a = int(t * SR)
        m = _n(cl * 3)
        tt = np.arange(m) / SR
        body = np.sin(2 * np.pi * rng.uniform(1400, 2600) * tt) \
            * np.exp(-tt / rng.uniform(0.004, 0.012))
        snap = rng.normal(0, 1, m) * np.exp(-tt / 0.004)
        click = (0.7 * snap + 0.5 * body).astype(np.float32)
        end = min(n, a + m)
        y[a:end] += click[:end - a] * rng.uniform(0.5, 1.0)
        t += rng.exponential(1.0 / clicks_per_sec)
    return scaled_rms(y, level_dbfs)


def music_bed(dur: float, rng: np.random.Generator,
              level_dbfs: float = -24.0, swell_at: Optional[float] = None,
              tempo: float = 92.0) -> np.ndarray:
    """Simple tonal chord pad + pluck rhythm (a deliberate VAD hard negative)."""
    n = _n(dur)
    y = np.zeros(n, dtype=np.float32)
    chords = [ (220.0, 261.63, 329.63), (196.0, 246.94, 293.66),
               (174.61, 220.0, 261.63), (196.0, 246.94, 392.0) ]
    bar = 60.0 / tempo * 4.0
    t0 = 0.0
    ci = 0
    while t0 < dur:
        freqs = chords[ci % len(chords)]
        seg_dur = min(bar, dur - t0)
        m = _n(seg_dur)
        tt = np.arange(m) / SR
        seg = np.zeros(m, dtype=np.float32)
        for f in freqs:
            for h, amp in ((1, 1.0), (2, 0.34), (3, 0.16), (4, 0.08)):
                seg += amp / h * np.sin(2 * np.pi * f * h * tt
                                        + rng.uniform(0, 6))
        # Slow pad attack + rhythmic pluck pulses.
        pad_env = np.minimum(1.0, tt / 0.25) * np.exp(-0.15 * tt)
        beat = 60.0 / tempo
        pulse = np.zeros(m, dtype=np.float32)
        for b in np.arange(0, seg_dur, beat):
            bi = int(b * SR)
            L = min(_n(0.35), m - bi)
            if L > 0:
                lt = np.arange(L) / SR
                pulse[bi:bi + L] += 0.8 * np.exp(-lt / 0.12)
        seg = 0.5 * seg * pad_env + 0.25 * seg * np.minimum(1.0, pulse)
        a = int(t0 * SR)
        end = min(n, a + m)
        y[a:end] += seg[:end - a]
        t0 += bar
        ci += 1
    if swell_at is not None:
        tt = np.arange(n) / SR
        swell = 1.0 / (1.0 + np.exp(-(tt - swell_at) * 2.2))
        y *= (0.35 + 0.9 * swell)
    return scaled_rms(y, level_dbfs)


def white_noise(dur: float, rng: np.random.Generator,
                level_dbfs: float = -35.0) -> np.ndarray:
    return scaled_rms(rng.normal(0, 1, _n(dur)).astype(np.float32), level_dbfs)


class Scene:
    def __init__(self, dur: float) -> None:
        self.n = _n(dur)
        self.tracks: list[np.ndarray] = []

    def add(self, x: np.ndarray, at: float = 0.0,
            gain_db: float = 0.0) -> "Scene":
        track = np.zeros(self.n, dtype=np.float32)
        a = int(at * SR)
        if a < 0:
            x = x[-a:]
            a = 0
        if a >= self.n or len(x) == 0:
            self.tracks.append(track)
            return self
        x = _fit(x, self.n - a) * db_gain(gain_db)
        track[a:a + len(x)] += x
        self.tracks.append(track)
        return self

    def mix(self) -> np.ndarray:
        y = np.sum(np.stack(self.tracks), axis=0) if self.tracks \
            else np.zeros(self.n, dtype=np.float32)
        peak = float(np.abs(y).max())
        if peak > 0.98:
            y = y * (0.98 / peak)
        return y.astype(np.float32)

    def mix_i16(self) -> np.ndarray:
        return pcm16_from_float(self.mix())


def annotation_from_scene(audio_file: str, i16: np.ndarray,
                          segments: list[dict],
                          active_spans: Optional[list[tuple[int, int]]] = None,
                          **extra) -> dict:
    d = {
        "audio_file": audio_file,
        "sample_rate": SR,
        "duration_sec": round(len(i16) / SR, 4),
        "annotator": "synthetic-reference (machine exact labels; human-reviewable)",
        "segments": segments,
    }
    if active_spans:
        d["active_spans_samples"] = [[int(a), int(b)] for a, b in active_spans]
    d.update(extra)
    return d
