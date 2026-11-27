"""ONNX Runtime backend and one-shot clock/alignment ledger.

The migrated graph keeps the old model identity, but its output contract is
explicitly "frame start".  The legacy feature windows are centered at
``i*hop + win//2``; callers convert that center to the new start coordinate by
subtracting the same half window exactly once.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from .model import LocalModel

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_ONNX_PATH = PACKAGE_DIR / "local_logistic_v1.onnx"
DEFAULT_WEIGHTS_PATH = PACKAGE_DIR / "model_weights.json"

RUNTIME_NAME = "onnxruntime"
PINNED_RUNTIME_VERSION = "1.23.2"
MIGRATION_CONTRACT_VERSION = "onnx-frame-start-v1"
ONNX_OUTPUT_TIMING = "frame_start"

CompensationSide = Literal["feature_pad", "output_trim"]
InputKind = Literal["file_zero", "live_mid_stride"]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def runtime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return DEFAULT_ONNX_PATH.exists()


def installed_runtime_version() -> Optional[str]:
    try:
        import onnxruntime as ort
        return str(ort.__version__)
    except Exception:
        return None


@dataclass(frozen=True)
class InputClock:
    """Capture clock for local sample zero.

    File analysis uses origin zero and phase zero.  A live fragment supplies
    the absolute capture-clock sample of its first sample.  When that first
    sample is in the middle of a stride, ``phase_samples=stride//2`` and the
    first frame-start grid point is ``phase_samples`` before local sample 0.
    """

    sample_rate: int
    kind: InputKind = "file_zero"
    capture_origin_sample: int = 0
    stride_samples: int = 0
    phase_samples: int = 0

    @classmethod
    def file_zero(cls, sample_rate: int, hop_samples: int) -> "InputClock":
        return cls(sample_rate=sample_rate, kind="file_zero",
                   capture_origin_sample=0, stride_samples=int(hop_samples),
                   phase_samples=0)

    @classmethod
    def live_mid_stride(cls, sample_rate: int,
                        capture_origin_sample: int,
                        stride_samples: int) -> "InputClock":
        stride = max(1, int(stride_samples))
        return cls(sample_rate=sample_rate, kind="live_mid_stride",
                   capture_origin_sample=int(capture_origin_sample),
                   stride_samples=stride, phase_samples=stride // 2)

    @property
    def origin_sec(self) -> float:
        return self.capture_origin_sample / float(self.sample_rate)

    def frame_start_local_sample(self, frame_index: int) -> int:
        return int(frame_index) * self.stride_samples - self.phase_samples

    def frame_start_clock_sample(self, frame_index: int) -> int:
        return (self.capture_origin_sample
                + self.frame_start_local_sample(frame_index))

    def clock_to_local_sample(self, clock_sample: int) -> int:
        return int(clock_sample) - self.capture_origin_sample


@dataclass(frozen=True)
class AlignmentLedger:
    """Single owner of the file/live and padding offset.

    Mapping from a post-compensation output coordinate to a source-local
    sample is ``output_sample + output_left_trim - feature_left_pad``.  The
    selected compensation owner prevents both sides from subtracting the same
    lead offset.
    """

    clock: InputClock
    compensation: CompensationSide = "feature_pad"
    feature_left_pad: int = 0
    feature_right_pad: int = 0
    output_left_trim: int = 0
    output_right_trim: int = 0
    block_size_samples: int = 0
    total_source_samples: int = 0

    def __post_init__(self) -> None:
        if self.compensation == "feature_pad" and self.output_left_trim:
            raise ValueError("feature_pad 已补偿字头，output 不能再次左裁")
        if self.compensation == "output_trim" and self.feature_left_pad:
            raise ValueError("output_trim 已补偿字头，feature 不能再次左补")

    @property
    def source_head_offset_samples(self) -> int:
        """Offset applied exactly once before handing samples to the editor."""
        return self.output_left_trim - self.feature_left_pad

    def output_to_source_sample(self, output_sample: int) -> int:
        return int(output_sample) + self.source_head_offset_samples

    def output_to_clock_sample(self, output_sample: int) -> int:
        return (self.clock.capture_origin_sample
                + self.output_to_source_sample(output_sample))

    def editor_range(self, start_output_sample: int,
                     end_output_sample: int) -> tuple[int, int]:
        s0 = self.output_to_source_sample(start_output_sample)
        s1 = self.output_to_source_sample(end_output_sample)
        lo, hi = 0, int(self.total_source_samples)
        return max(lo, s0), min(hi, s1)

    def to_dict(self) -> dict:
        return {
            "contract": MIGRATION_CONTRACT_VERSION,
            "compensation": self.compensation,
            "feature_left_pad": int(self.feature_left_pad),
            "feature_right_pad": int(self.feature_right_pad),
            "output_left_trim": int(self.output_left_trim),
            "output_right_trim": int(self.output_right_trim),
            "source_head_offset_samples": int(self.source_head_offset_samples),
            "block_size_samples": int(self.block_size_samples),
            "total_source_samples": int(self.total_source_samples),
            "clock": {
                "kind": self.clock.kind,
                "sample_rate": self.clock.sample_rate,
                "capture_origin_sample": self.clock.capture_origin_sample,
                "stride_samples": self.clock.stride_samples,
                "phase_samples": self.clock.phase_samples,
            },
        }


def make_alignment(total_source_samples: int,
                   clock: InputClock,
                   block_size_samples: int,
                   compensation: CompensationSide = "feature_pad"
                   ) -> AlignmentLedger:
    """Plan tail block padding and live mid-stride lead compensation.

    ``feature_pad`` prepends the live phase and zero-pads the final partial
    block; its output is not time-trimmed.  ``output_trim`` performs only the
    unavoidable physical block fill and assigns ownership to output cropping;
    no feature-side lead pad is added.
    """
    total = max(0, int(total_source_samples))
    block = max(1, int(block_size_samples))
    right_pad = (-total) % block
    lead = max(0, int(clock.phase_samples)) if clock.kind == "live_mid_stride" else 0
    if compensation == "feature_pad":
        return AlignmentLedger(
            clock=clock, compensation=compensation,
            feature_left_pad=lead, feature_right_pad=right_pad,
            block_size_samples=block, total_source_samples=total)
    # Output ownership: do not prepend phase.  The first output coordinate is
    # treated as the padded grid origin and cropped by the phase once.
    return AlignmentLedger(
        clock=clock, compensation=compensation,
        feature_left_pad=0, feature_right_pad=right_pad,
        output_left_trim=lead, output_right_trim=0,
        block_size_samples=block, total_source_samples=total)


def legacy_center_to_new_start_sample(legacy_frame_index: int,
                                      hop_samples: int,
                                      win_samples: int) -> int:
    """Convert old center timestamp to the migrated frame-start timestamp."""
    center = int(legacy_frame_index) * int(hop_samples) + int(win_samples) // 2
    return center - int(win_samples) // 2


@dataclass
class OnnxLogisticRuntime:
    model_path: Path = DEFAULT_ONNX_PATH
    weights_path: Path = DEFAULT_WEIGHTS_PATH
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    threshold: float = 0.5
    feature_names: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    runtime_version: str = ""
    model_sha256: str = ""
    weights_sha256_expected: str = ""
    weights_sha256_actual: str = ""
    session: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        import onnxruntime as ort

        self.runtime_version = str(ort.__version__)
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"ONNX VAD 模型不存在: {self.model_path}")
        self.model_sha256 = sha256_file(self.model_path)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=so,
            providers=list(self.providers))
        meta = self.session.get_modelmeta().custom_metadata_map or {}
        self.metadata = dict(meta)
        names = meta.get("feature_names", "")
        self.feature_names = [x for x in names.split(",") if x]
        self.threshold = float(meta.get("operating_threshold", 0.5))

        if self.weights_path.exists():
            local = LocalModel.load(self.weights_path)
            if not self.feature_names:
                self.feature_names = list(local.feature_names)
            self.threshold = float(local.threshold)
            self.weights_sha256_actual = sha256_file(self.weights_path)
            expected = meta.get("source_weights_sha256", "")
            self.weights_sha256_expected = expected
            if expected and expected != self.weights_sha256_actual:
                raise RuntimeError(
                    "VAD 模型身份校验失败：ONNX 元数据中的权重哈希与 "
                    f"{self.weights_path.name} 不一致。")
        expected_layout = "raw_features[N,13]float32"
        if meta.get("feature_layout") and meta["feature_layout"] != expected_layout:
            raise RuntimeError(
                f"不支持的 VAD 特征布局: {meta['feature_layout']}")
        if meta.get("output_timing") not in ("", ONNX_OUTPUT_TIMING):
            raise RuntimeError(
                "VAD 输出时间契约不是 frame_start: "
                f"{meta.get('output_timing')}")

    @property
    def input_name(self) -> str:
        return self.session.get_inputs()[0].name

    @property
    def output_name(self) -> str:
        return self.session.get_outputs()[0].name

    @property
    def backend(self) -> str:
        identity = self.metadata.get("model_identity", "unknown")
        return (f"{RUNTIME_NAME}-{self.runtime_version}:"
                f"{MIGRATION_CONTRACT_VERSION}:{identity}:"
                f"{Path(self.model_path).name}@sha256:{self.model_sha256[:12]}")

    @property
    def identity(self) -> dict:
        return {
            "runtime": RUNTIME_NAME,
            "runtime_version": self.runtime_version,
            "pinned_runtime_version": PINNED_RUNTIME_VERSION,
            "contract_version": MIGRATION_CONTRACT_VERSION,
            "model_identity": self.metadata.get("model_identity"),
            "model_file": Path(self.model_path).name,
            "model_sha256": self.model_sha256,
            "weights_file": Path(self.weights_path).name,
            "weights_sha256": self.weights_sha256_actual,
            "source_weights_sha256_in_graph": self.weights_sha256_expected,
            "feature_layout": self.metadata.get("feature_layout"),
            "normalization": self.metadata.get("normalization"),
            "output_timing": self.metadata.get("output_timing"),
            "threshold": self.threshold,
            "providers": list(self.session.get_providers()),
        }

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != len(self.feature_names or range(13)):
            raise ValueError(
                f"ONNX VAD 期望 [N,13] float32，实际形状 {arr.shape}")
        out = self.session.run(
            [self.output_name], {self.input_name: np.ascontiguousarray(arr)})[0]
        return np.clip(np.asarray(out, dtype=np.float64).reshape(-1), 0.0, 1.0)
