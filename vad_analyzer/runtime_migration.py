"""Public compatibility surface for the inference-runtime migration."""
from __future__ import annotations

from .kernel import IntervalKernel, VoiceIntervalKernel
from .migration_evidence import collect_evidence
from .runtime_export import export_onnx_model
from .runtime_onnx import (
    MIGRATION_CONTRACT_VERSION,
    ONNX_OUTPUT_TIMING,
    PINNED_RUNTIME_VERSION,
    RUNTIME_NAME,
    AlignmentLedger,
    InputClock,
    OnnxLogisticRuntime,
    installed_runtime_version,
    legacy_center_to_new_start_sample,
    make_alignment,
    runtime_available,
    sha256_file,
)

__all__ = [
    "MIGRATION_CONTRACT_VERSION",
    "ONNX_OUTPUT_TIMING",
    "PINNED_RUNTIME_VERSION",
    "RUNTIME_NAME",
    "AlignmentLedger",
    "InputClock",
    "IntervalKernel",
    "OnnxLogisticRuntime",
    "VoiceIntervalKernel",
    "collect_evidence",
    "export_onnx_model",
    "installed_runtime_version",
    "legacy_center_to_new_start_sample",
    "make_alignment",
    "runtime_available",
    "sha256_file",
]
