"""Frame-level acoustic features.

The feature set deliberately goes beyond RMS energy: spectral band ratios,
flatness, centroid/rolloff, temporal flux, voicing/autocorrelation and
syllabic-rate modulation. Wind, keyboard transients and tonal music therefore
have feature patterns different from speech-like activity even when equally
loud.

All temporal context (noise floor, modulation) is *causal* (past frames only),
so the offline extractor and the streaming extractor produce identical vectors
for the same frame sequence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

FEATURE_NAMES = [
    "log_rms",
    "above_floor_db",
    "zcr_rate",
    "low_band_ratio",
    "mid_band_ratio",
    "high_band_ratio",
    "centroid_norm",
    "flatness",
    "rolloff_norm",
    "spectral_flux",
    "voicing",
    "mod_2_8hz_ratio",
    "envelope_cv",
]


@dataclass
class FrameGrid:
    sample_rate: int
    win: int
    hop: int
    n_frames: int
    nfft: int
    n_total: int = 0

    def start_sample(self, i: int) -> int:
        return i * self.hop

    def frame_span_samples(self, i: int, n_total: int) -> tuple[int, int]:
        a = i * self.hop
        b = min(n_total, a + self.win)
        return a, b

    def time_of(self, i: int) -> float:
        return (i * self.hop) / float(self.sample_rate)


@dataclass
class FrameFeatures:
    x: np.ndarray            # (n_frames, len(FEATURE_NAMES))
    grid: FrameGrid
    rms: np.ndarray
    times: np.ndarray        # frame start, seconds
    start_samples: np.ndarray


def make_grid(sample_rate: int, frame_ms: float = 25.0,
              hop_ms: float = 10.0, n_total: int = 0) -> FrameGrid:
    win = max(64, int(round(sample_rate * frame_ms / 1000.0)))
    hop = max(16, int(round(sample_rate * hop_ms / 1000.0)))
    nfft = 1
    while nfft < win:
        nfft <<= 1
    n_frames = max(1, int(np.ceil(n_total / hop))) if n_total else 0
    return FrameGrid(sample_rate, win, hop, n_frames, nfft, n_total)


def _band_ratio(freqs: np.ndarray, power: np.ndarray, lo: float,
                hi: float) -> float:
    m = (freqs >= lo) & (freqs < hi)
    total = power.sum() + 1e-12
    return float(power[m].sum() / total)


def _voicing_strength(frame: np.ndarray, sr: int) -> float:
    x = frame - frame.mean()
    if np.sqrt((x ** 2).mean()) < 1e-4:
        return 0.0
    n = len(x)
    ac = np.correlate(x, x, mode="full")[n - 1:]
    if ac[0] <= 1e-12:
        return 0.0
    ac = ac / ac[0]
    lag_hi = max(2, int(sr / 65.0))    # lowest f0 ~65 Hz
    lag_lo = max(2, int(sr / 420.0))   # highest f0 ~420 Hz
    seg = ac[lag_lo:min(lag_hi + 1, len(ac))]
    if seg.size == 0:
        return 0.0
    return float(np.clip(seg.max(), 0.0, 1.0))


class FeatureStreamer:
    """Causal per-frame feature computer shared by offline/stream paths."""

    def __init__(self, sample_rate: int, frame_ms: float = 25.0,
                 hop_ms: float = 10.0, floor_win_frames: int = 120) -> None:
        self.sr = sample_rate
        self.grid = make_grid(sample_rate, frame_ms, hop_ms, 0)
        self.win = self.grid.win
        self.hop = self.grid.hop
        self.nfft = self.grid.nfft
        self._window = np.hanning(self.win).astype(np.float32)
        self._freqs = np.fft.rfftfreq(self.nfft, 1.0 / sample_rate)
        self._prev_mag: Optional[np.ndarray] = None
        self._rms_hist: list[float] = []
        self._env_hist: list[float] = []
        self.floor_win = floor_win_frames
        self.fps = 1000.0 / hop_ms
        self.mod_ctx = max(12, int(round(0.32 * self.fps)))

    def process(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float32)
        if len(raw) < self.win:
            padded = np.zeros(self.win, dtype=np.float32)
            padded[:len(raw)] = raw
            raw = padded

        rms = float(np.sqrt(np.mean(raw ** 2)) + 1e-12)
        zcr = float(np.mean(np.abs(np.diff(np.signbit(raw))))) if self.win > 1 else 0.0

        spec = np.fft.rfft(raw * self._window, n=self.nfft)
        mag = np.abs(spec).astype(np.float32) + 1e-10
        power = mag ** 2

        low_r = _band_ratio(self._freqs, power, 20.0, 300.0)
        mid_r = _band_ratio(self._freqs, power, 300.0, 3400.0)
        high_hi = min(9000.0, self.sr / 2 - 100)
        high_r = _band_ratio(self._freqs, power, 3400.0, high_hi) if high_hi > 3400 else 0.0
        total_p = power.sum() + 1e-12
        centroid = float((self._freqs * power).sum() / total_p)
        csum = np.cumsum(power)
        roll_idx = int(np.searchsorted(csum, 0.85 * csum[-1]))
        rolloff = float(self._freqs[min(roll_idx, len(self._freqs) - 1)])
        flatness = float(np.exp(np.mean(np.log(power + 1e-12)))
                         / (np.mean(power) + 1e-12))
        norm_mag = mag / (np.linalg.norm(mag) + 1e-12)
        if self._prev_mag is None:
            flux = 0.0
        else:
            diff = norm_mag - self._prev_mag
            flux = float(np.sqrt(np.mean(np.maximum(diff, 0.0) ** 2)))
        self._prev_mag = norm_mag
        voice = _voicing_strength(raw, self.sr)

        mid_mask = (self._freqs >= 300) & (self._freqs < 3400)
        mid_env = float(np.sqrt(power[mid_mask].sum() / self.nfft))

        self._rms_hist.append(rms)
        self._env_hist.append(mid_env)
        recent = np.asarray(self._rms_hist[-self.floor_win:], dtype=np.float32)
        floor_rms = max(float(np.percentile(recent, 15)), 1e-5)
        above_db = float(20.0 * np.log10(max(rms, 1e-6) / floor_rms))
        above_db = float(np.clip(above_db, -12.0, 40.0))

        env = np.asarray(self._env_hist[-(2 * self.mod_ctx):], dtype=np.float32)
        mod_ratio = 0.0
        env_cv = 0.0
        if env.size >= 16:
            e = env[-2 * self.mod_ctx:]
            e = e - e.mean()
            w = np.hanning(len(e)).astype(np.float32)
            sp = np.abs(np.fft.rfft(e * w)) ** 2
            mf = np.fft.rfftfreq(len(e), 1.0 / self.fps)
            band = (mf >= 2.0) & (mf <= 8.0)
            wide = (mf >= 0.5) & (mf <= 14.0)
            mod_ratio = float(sp[band].sum() / (sp[wide].sum() + 1e-10))
            tail = np.asarray(self._env_hist[-self.mod_ctx:], dtype=np.float32)
            env_cv = float(np.std(tail) / (abs(tail.mean()) + 1e-6))

        v = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        v[0] = np.log(rms + 1e-8)
        v[1] = above_db
        v[2] = zcr
        v[3] = low_r
        v[4] = mid_r
        v[5] = high_r
        v[6] = centroid / max(1.0, self.sr / 2.0)
        v[7] = min(1.0, flatness)
        v[8] = rolloff / max(1.0, self.sr / 2.0)
        v[9] = flux
        v[10] = voice
        v[11] = mod_ratio
        v[12] = env_cv
        return v


def extract_features_from_float(sig: np.ndarray, sr: int,
                                frame_ms: float = 25.0,
                                hop_ms: float = 10.0) -> FrameFeatures:
    sig = np.asarray(sig, dtype=np.float32)
    n_total = len(sig)
    grid = make_grid(sr, frame_ms, hop_ms, n_total)
    fs = FeatureStreamer(sr, frame_ms, hop_ms)
    X = np.zeros((grid.n_frames, len(FEATURE_NAMES)), dtype=np.float32)
    rms_arr = np.zeros(grid.n_frames, dtype=np.float32)
    for i in range(grid.n_frames):
        a = i * grid.hop
        raw = sig[a:min(n_total, a + grid.win)]
        X[i] = fs.process(raw)
        rms_arr[i] = fs._rms_hist[-1]
    times = np.arange(grid.n_frames) * grid.hop / float(sr)
    starts = np.arange(grid.n_frames) * grid.hop
    return FrameFeatures(x=X, grid=grid, rms=rms_arr, times=times,
                         start_samples=starts)


def extract_features(audio) -> FrameFeatures:
    """Accepts :class:`~vad_analyzer.audio.AudioBuffer`."""
    return extract_features_from_float(audio.float, audio.sample_rate)
