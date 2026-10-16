"""ONNX export and int8 quantization of a gate checkpoint, for faster CPU inference.

    python -m laya_gate.onnx_export checkpoints/v2   # writes <checkpoint>/onnx/model{,.int8}.onnx

The graph takes the five collated inputs (`input_ids`, `attention_mask`, `marker_pos`,
`marker_mask`, `qtype`) and returns the option logits, so tokenization, truncation and
temperature scaling stay in `SwedishPreferenceGate` (`backend="onnx"` or `"onnx-int8"`).
int8 uses ONNX Runtime dynamic quantization with its default operator set (int8 weights,
including the embedding table; activations quantized per batch at run time).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

INPUTS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
ONNX_DIR = "onnx"
FILES = {"onnx": "model.onnx", "onnx-int8": "model.int8.onnx"}


class _LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args):
        return self.model(*args)[0]


def export(checkpoint: str | Path, out_dir: str | Path | None = None, opset: int = 17) -> Path:
    """Export the fp32 graph and its dynamically quantized int8 version; returns the folder."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    from .predictor import SwedishPreferenceGate, collate

    gate = SwedishPreferenceGate(checkpoint, device="cpu", dtype=torch.float32)
    out = Path(out_dir) if out_dir else gate.model_dir / ONNX_DIR
    out.mkdir(parents=True, exist_ok=True)
    # one input at the full length limit and one short, padded together: tracing sees the
    # long-sequence attention path and padding; the varying axes are declared dynamic below
    long_text = (
        "Växter tar upp ljus med klorofyll och bygger socker av koldioxid och vatten. " * 200
    )
    items = [
        gate.encode("Hur fungerar fotosyntesen?", long_text, long_text),
        gate.encode("Vad är huvudstaden i Sverige?", "Stockholm.", "Huvudstaden är Göteborg."),
    ]
    batch = collate(items, gate.tok.pad_token_id)
    dyn = {
        "input_ids": {0: "batch", 1: "tokens"},
        "attention_mask": {0: "batch", 1: "tokens"},
        "marker_pos": {0: "batch", 1: "options"},
        "marker_mask": {0: "batch", 1: "options"},
        "qtype": {0: "batch"},
        "logits": {0: "batch", 1: "options"},
    }
    fp32 = out / FILES["onnx"]
    torch.onnx.export(
        _LogitsOnly(gate.model).eval(),
        tuple(batch[k] for k in INPUTS),
        str(fp32),
        input_names=list(INPUTS),
        output_names=["logits"],
        dynamic_axes=dyn,
        opset_version=opset,
        dynamo=False,
    )
    quantize_dynamic(str(fp32), str(out / FILES["onnx-int8"]), weight_type=QuantType.QInt8)
    return out


class OnnxForward:
    """Drop-in for `forward_logits` backed by an ONNX Runtime CPU session."""

    def __init__(self, path: str | Path, threads: int | None = None):
        import onnxruntime as ort

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    def __call__(self, batch: dict[str, torch.Tensor]) -> np.ndarray:
        feed = {k: batch[k].numpy() for k in INPUTS}
        return self.session.run(None, feed)[0]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    print(f"exported to {export(args.checkpoint, args.out)}")


if __name__ == "__main__":
    main()
