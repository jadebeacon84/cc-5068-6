"""Sample-exact clip export and first/last-sample verification."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .audio import AudioBuffer, load_wav, save_wav


@dataclass
class ClipVerification:
    path: str
    requested_start_sample: int
    requested_end_sample: int
    sample_rate: int
    expected_sample_count: int
    actual_sample_count: int
    length_match: bool
    all_samples_match: bool
    first_sample_expected: int
    first_sample_actual: int
    last_sample_expected: int
    last_sample_actual: int
    edges_match: bool
    sha256_source_slice: str
    sha256_exported_file: str

    @property
    def verified(self) -> bool:
        return (self.length_match and self.all_samples_match
                and self.edges_match)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["verified"] = self.verified
        return d


def _sha256_pcm(samples: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(samples).tobytes()).hexdigest()


def export_range(audio: AudioBuffer, start_sample: int, end_sample: int,
                 out_path: str | Path,
                 verify: bool = True) -> tuple[Path, Optional[ClipVerification]]:
    s0 = max(0, int(start_sample))
    s1 = min(len(audio.samples), int(end_sample))
    if s1 <= s0:
        raise ValueError(f"导出区间为空: [{start_sample},{end_sample})")
    slice_i16 = np.ascontiguousarray(audio.samples[s0:s1])
    path = save_wav(out_path, slice_i16, audio.sample_rate)
    report = verify_export(path, audio, s0, s1) if verify else None
    return path, report


def verify_export(clip_path: str | Path, source: AudioBuffer,
                  start_sample: int, end_sample: int) -> ClipVerification:
    """Re-open the exported WAV and compare it against the source slice."""
    p = Path(clip_path)
    reloaded = load_wav(p)
    expected = np.ascontiguousarray(source.samples[start_sample:end_sample])
    actual = reloaded.samples
    n_expected = int(end_sample - start_sample)

    length_match = len(actual) == n_expected
    if length_match:
        all_match = bool(np.array_equal(expected, actual))
    else:
        n = min(len(actual), n_expected)
        all_match = bool(np.array_equal(expected[:n], actual[:n])) and False

    first_exp = int(expected[0]) if n_expected else 0
    last_exp = int(expected[-1]) if n_expected else 0
    first_act = int(actual[0]) if len(actual) else 0
    last_act = int(actual[-1]) if len(actual) else 0
    edges_match = (first_exp == first_act and last_exp == last_act
                   and length_match)

    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)

    return ClipVerification(
        path=str(p),
        requested_start_sample=int(start_sample),
        requested_end_sample=int(end_sample),
        sample_rate=source.sample_rate,
        expected_sample_count=n_expected,
        actual_sample_count=int(len(actual)),
        length_match=length_match,
        all_samples_match=all_match,
        first_sample_expected=first_exp,
        first_sample_actual=first_act,
        last_sample_expected=last_exp,
        last_sample_actual=last_act,
        edges_match=edges_match,
        sha256_source_slice=_sha256_pcm(expected),
        sha256_exported_file=h.hexdigest(),
    )


def export_interval(audio: AudioBuffer, interval, out_dir: str | Path,
                    name_prefix: str = "clip") -> tuple[Path, ClipVerification]:
    out_dir = Path(out_dir)
    idx = interval.index if interval.index is not None else 0
    fname = (f"{name_prefix}_{idx:02d}_"
             f"{interval.start:.3f}-{interval.end:.3f}.wav")
    path, report = export_range(audio, interval.start_sample,
                                interval.end_sample, out_dir / fname)
    assert report is not None
    return path, report
