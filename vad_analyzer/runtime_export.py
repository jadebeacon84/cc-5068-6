"""Build-time exporter for the ONNX Runtime VAD backend.

The ``onnx`` package is only required to rebuild the checked-in artifact.
Normal inference uses :class:`vad_analyzer.runtime_onnx.OnnxLogisticRuntime`
and ``onnxruntime``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

DEFAULT_WEIGHTS_PATH = Path(__file__).with_name("model_weights.json")
DEFAULT_ONNX_PATH = Path(__file__).with_name("local_logistic_v1.onnx")
ONNX_FRAME_LAYOUT = "raw_features[N,13]float32"
ONNX_OUTPUT_TIMING = "frame_start"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_onnx_model(weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
                      onnx_path: str | Path = DEFAULT_ONNX_PATH):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weights_path = Path(weights_path)
    onnx_path = Path(onnx_path)
    import json
    d = json.loads(weights_path.read_text(encoding="utf-8"))

    mean = np.asarray(d["mean"], dtype=np.float32).reshape(1, -1)
    std = np.asarray(d["std"], dtype=np.float32).reshape(1, -1)
    w = np.asarray(d["weights"], dtype=np.float32).reshape(-1, 1)
    b = np.asarray([float(d["bias"])], dtype=np.float32)

    X = helper.make_tensor_value_info(
        "raw_features", TensorProto.FLOAT, ["N", mean.shape[1]])
    Y = helper.make_tensor_value_info(
        "probabilities", TensorProto.FLOAT, ["N", 1])

    nodes = [
        helper.make_node("Constant", [], ["mean"],
                         value=numpy_helper.from_array(mean, "mean_value")),
        helper.make_node("Constant", [], ["std"],
                         value=numpy_helper.from_array(std, "std_value")),
        helper.make_node("Constant", [], ["weights"],
                         value=numpy_helper.from_array(w, "weights_value")),
        helper.make_node("Constant", [], ["bias"],
                         value=numpy_helper.from_array(b, "bias_value")),
        helper.make_node("Sub", ["raw_features", "mean"], ["centered"]),
        helper.make_node("Div", ["centered", "std"], ["normalized"]),
        helper.make_node("Gemm", ["normalized", "weights", "bias"],
                         ["logits"], transB=0, alpha=1.0, beta=1.0),
        helper.make_node("Sigmoid", ["logits"], ["probabilities"]),
    ]
    graph = helper.make_graph(
        nodes, "local_logistic_v1_normalized_sigmoid", [X], [Y])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)],
        producer_name="vad_analyzer",
        model_version=1)
    model.doc_string = (
        "Identity-preserving ONNX Runtime export of local-logistic-v1. "
        "Input is raw 13-feature rows; source mean/std normalization is baked "
        "into the graph. Output rows use the new backend frame-start clock.")

    props = {
        "model_identity": "local-logistic-v1",
        "source_weights_file": weights_path.name,
        "source_weights_sha256": sha256_file(weights_path),
        "feature_layout": ONNX_FRAME_LAYOUT,
        "feature_names": ",".join(d.get("feature_names", [])),
        "normalization": "baked_in_graph_source_mean_std",
        "output_timing": ONNX_OUTPUT_TIMING,
        "operating_threshold": str(d.get("threshold", 0.5)),
    }
    for key, value in props.items():
        p = model.metadata_props.add()
        p.key = key
        p.value = value

    onnx.checker.check_model(model)
    # Keep the tiny artifact loadable by the pinned ONNX Runtime release.
    model.ir_version = 10
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(model.SerializeToString())
    return onnx_path
