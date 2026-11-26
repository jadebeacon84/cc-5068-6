"""Audio I/O and clip playback.

The canonical in-memory sample type is mono signed 16-bit integer. Analysis
runs on a float32 view, while export copies integer samples so exported clips
can be verified sample-for-sample against the source file.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .errors import AudioReadError, PlaybackUnavailable


@dataclass
class AudioBuffer:
    samples: np.ndarray  # int16, mono
    sample_rate: int
    path: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.samples.dtype != np.int16:
            raise TypeError("AudioBuffer.samples must be int16")
        if self.samples.ndim != 1:
            raise ValueError("AudioBuffer expects mono samples")

    @property
    def duration(self) -> float:
        return len(self.samples) / float(self.sample_rate)

    @property
    def float(self) -> np.ndarray:
        return self.samples.astype(np.float32) / 32768.0

    def time_to_sample(self, t: float) -> int:
        return int(round(t * self.sample_rate))

    def sample_to_time(self, i: int) -> float:
        return i / float(self.sample_rate)


def load_wav(path: str | os.PathLike) -> AudioBuffer:
    p = Path(path)
    if not p.exists():
        raise AudioReadError(f"音频文件不存在: {p}")
    try:
        with wave.open(str(p), "rb") as wf:
            nch = wf.getnchannels()
            sw = wf.getsampwidth()
            sr = wf.getframerate()
            n = wf.getnframes()
            raw = wf.readframes(n)
    except wave.Error as e:
        raise AudioReadError(f"无法读取 WAV（仅支持 PCM WAV）: {p.name} ({e})") from e

    if sw == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.int16)
    elif sw == 1:
        u8 = np.frombuffer(raw, dtype=np.uint8)
        data = ((u8.astype(np.int16) - 128) << 8)
    elif sw == 4:
        i32 = np.frombuffer(raw, dtype="<i4")
        data = np.clip(i32 >> 16, -32768, 32767).astype(np.int16)
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        i32 = (b[:, 0].astype(np.int32)
               | (b[:, 1].astype(np.int32) << 8)
               | (b[:, 2].astype(np.int32) << 16))
        neg = i32 & 0x800000
        i32 = i32 - (neg << 1)
        data = np.clip(i32 >> 8, -32768, 32767).astype(np.int16)
    else:
        raise AudioReadError(f"不支持的采样位宽 {sw * 8}-bit: {p.name}")

    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1).astype(np.int16)
    if data.size == 0:
        raise AudioReadError(f"音频为空: {p.name}")
    return AudioBuffer(samples=np.ascontiguousarray(data), sample_rate=sr, path=p)


def save_wav(path: str | os.PathLike, samples: np.ndarray, sample_rate: int) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if samples.dtype != np.int16:
        samples = np.clip(samples, -32768, 32767).astype(np.int16)
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(np.ascontiguousarray(samples).tobytes())
    return p


def pcm16_from_float(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)


class AudioPlayer:
    """Best-effort local clip playback.

    Tries python audio packages first, then common command line players.
    Every failure is reported as :class:`PlaybackUnavailable` so the GUI can
    show it instead of pretending playback worked.
    """

    CLI_CANDIDATES = ("paplay", "aplay", "ffplay", "play", "afplay")

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self.backend = self._detect_backend()

    def _detect_backend(self) -> str:
        try:
            import sounddevice  # noqa: F401
            return "sounddevice"
        except Exception:
            pass
        try:
            import simpleaudio  # noqa: F401
            return "simpleaudio"
        except Exception:
            pass
        for cmd in self.CLI_CANDIDATES:
            if shutil.which(cmd):
                return f"cmd:{cmd}"
        if os.name == "nt":
            return "winsound"
        return "unavailable"

    def available(self) -> bool:
        return self.backend != "unavailable"

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None

    def play_buffer(self, audio: AudioBuffer, start_sample: int = 0,
                    end_sample: Optional[int] = None) -> Path:
        self.stop()
        end_sample = len(audio.samples) if end_sample is None else end_sample
        start_sample = max(0, start_sample)
        end_sample = min(len(audio.samples), end_sample)
        if end_sample <= start_sample:
            raise PlaybackUnavailable("播放区间为空。")
        clip = audio.samples[start_sample:end_sample]
        tmp = Path(tempfile.gettempdir()) / "vad_playback_clip.wav"
        save_wav(tmp, clip, audio.sample_rate)
        self.play_file(tmp)
        return tmp

    def play_file(self, path: str | os.PathLike) -> None:
        p = Path(path)
        if not p.exists():
            raise PlaybackUnavailable(f"待播放文件不存在: {p}")
        if self.backend == "unavailable":
            raise PlaybackUnavailable(
                "本机未找到可用的音频播放后端（paplay/aplay/ffplay/play 或 "
                "sounddevice/simpleaudio 均不可用）。片段已导出为 WAV，"
                "可直接用系统播放器审听。"
            )
        if self.backend == "sounddevice":
            try:
                import sounddevice as sd
                with wave.open(str(p), "rb") as wf:
                    sr = wf.getframerate()
                    x = np.frombuffer(wf.readframes(wf.getnframes()),
                                      dtype=np.int16)
                sd.stop()
                sd.play(x, sr)
                return
            except Exception as e:  # device missing etc.
                raise PlaybackUnavailable(f"sounddevice 播放失败: {e}") from e
        if self.backend == "simpleaudio":
            try:
                import simpleaudio as sa
                w = sa.WaveObject.from_wave_file(str(p))
                self._play_obj = w.play()
                return
            except Exception as e:
                raise PlaybackUnavailable(f"simpleaudio 播放失败: {e}") from e
        if self.backend == "winsound":
            import winsound  # type: ignore
            winsound.PlaySound(str(p), winsound.SND_FILENAME | winsound.SND_ASYNC)
            return
        cmd = self.backend.split(":", 1)[1]
        args = [cmd]
        if cmd == "ffplay":
            args += ["-nodisp", "-autoexit", "-loglevel", "quiet"]
        if cmd == "afplay":
            args = [cmd]
        args.append(str(p))
        try:
            self._proc = subprocess.Popen(
                args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except OSError as e:
            raise PlaybackUnavailable(f"无法启动播放器 {cmd}: {e}") from e
