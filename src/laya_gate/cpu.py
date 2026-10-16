"""CPU inference options: latency and what each costs in accuracy.

    python -m laya_gate.cpu --checkpoint checkpoints/v3

Variants, all batch 1 on CPU:
- torch fp32 (the v1 benchmark's setting)
- torch dynamic int8 (`torch.ao.quantization.quantize_dynamic` on the encoder's nn.Linear)
- ONNX Runtime fp32 and dynamic int8 (`python -m laya_gate.onnx_export`)
- ONNX int8 with a 512-token input limit instead of 1024

Latency is `gate.logits([item])` (collate + forward + logits to host) over the same val_eval
pairs for every variant; tokenization is excluded, as in the GPU benchmark. Accuracy is the
preference question on the probe (both orders, order-averaged) and on the first N original
val_eval pairs, plus the mean absolute change in P(A) against torch fp32.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .probe import PROBE_DIR, summarize
from .utils import QID, SPLITS_DIR, read_jsonl, update_metrics


def timed(gate, items, runs: int, warmup: int) -> list[float]:
    for i in range(warmup):
        gate.logits([items[i % len(items)]], batch_size=1)
    out = []
    for i in range(runs):
        t = time.perf_counter()
        gate.logits([items[i % len(items)]], batch_size=1)
        out.append((time.perf_counter() - t) * 1000)
    return out


def p_a(gate, rows) -> np.ndarray:
    items = [gate.encode(r["prompt"], r["candidate_a"], r["candidate_b"]) for r in rows]
    return gate.probabilities(gate.logits(items, batch_size=1), QID)[:, 0]


def make_gate(checkpoint: Path, variant: str, threads: int):
    from .predictor import SwedishPreferenceGate

    if variant.startswith("onnx"):
        backend = "onnx" if variant == "onnx-fp32" else "onnx-int8"
        max_len = 512 if variant.endswith("512") else None
        return SwedishPreferenceGate(checkpoint, backend=backend, threads=threads, max_len=max_len)
    gate = SwedishPreferenceGate(checkpoint, device="cpu", dtype=torch.float32)
    if variant == "torch-int8":
        # encoder only: nn.TransformerEncoderLayer in the head reads `linear.weight` as a
        # tensor, which a dynamically quantized Linear does not provide
        gate.model.encoder = torch.ao.quantization.quantize_dynamic(
            gate.model.encoder, {torch.nn.Linear}, dtype=torch.qint8
        )
    return gate


def main(argv=None) -> None:
    from .onnx_export import FILES, ONNX_DIR, export

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default="checkpoints/v2")
    parser.add_argument("--threads", type=int, default=torch.get_num_threads())
    parser.add_argument("--latency-items", type=int, default=50)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--n-original", type=int, default=300)
    parser.add_argument("--n-probe", type=int, default=None, help="debug: first N probe pairs")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["torch-fp32", "torch-int8", "onnx-fp32", "onnx-int8", "onnx-int8-512"],
    )
    args = parser.parse_args(argv)
    ckpt = Path(args.checkpoint)
    torch.set_num_threads(args.threads)
    if any(v.startswith("onnx") for v in args.variants) and not all(
        (ckpt / ONNX_DIR / f).exists() for f in FILES.values()
    ):
        t = time.perf_counter()
        export(ckpt)
        print(f"onnx export + quantization: {time.perf_counter() - t:.0f}s", flush=True)

    probe = [
        {
            "prompt": p["sv_prompt"],
            "candidate_a": p["sv_chosen"],
            "candidate_b": p["sv_rejected"],
            "label": 0,
        }
        for p in read_jsonl(PROBE_DIR / "probe_pairs.jsonl")[: args.n_probe]
    ]
    probe_rev = [
        {**r, "candidate_a": r["candidate_b"], "candidate_b": r["candidate_a"]} for r in probe
    ]
    original = read_jsonl(SPLITS_DIR / "val_eval.jsonl")
    lat_rows, acc_rows = original[: args.latency_items], original[: args.n_original]

    out: dict[str, Any] = {
        "checkpoint": str(ckpt),
        "cpu": _cpu_name(),
        "threads": args.threads,
        "method": "batch 1; collate + forward + logits to host; tokenization excluded; "
        f"{args.runs} runs over the first {args.latency_items} val_eval pairs",
        "files_mb": {
            v: round((ckpt / ONNX_DIR / f).stat().st_size / 2**20)
            for v, f in FILES.items()
            if (ckpt / ONNX_DIR / f).exists()
        },
        "variants": {},
    }
    if (ckpt / "model.safetensors").exists():
        out["files_mb"]["torch (safetensors)"] = round(
            (ckpt / "model.safetensors").stat().st_size / 2**20
        )
    ref = None
    for variant in args.variants:
        t0 = time.perf_counter()
        gate = make_gate(ckpt, variant, args.threads)
        items = [gate.encode(r["prompt"], r["candidate_a"], r["candidate_b"]) for r in lat_rows]
        lat = timed(gate, items, args.runs, warmup=5)
        pf, pr = p_a(gate, probe), 1 - p_a(gate, probe_rev)
        po = p_a(gate, acc_rows)
        labels = np.array([r["label"] for r in acc_rows])
        res = {
            "latency_ms": {
                "p50": float(np.median(lat)),
                "p95": float(np.percentile(lat, 95)),
                "mean": float(np.mean(lat)),
            },
            "mean_input_tokens": float(np.mean([len(i["ids"]) for i in items])),
            "probe_accuracy": summarize((pf + pr) / 2 > 0.5),
            "original_accuracy": summarize((po > 0.5) == (labels == 0)),
            "seconds_total": None,
        }
        probs = np.r_[pf, pr, po]
        if ref is None:
            ref = probs
        res["mean_abs_delta_p_vs_torch_fp32"] = float(np.abs(probs - ref).mean())
        res["max_abs_delta_p_vs_torch_fp32"] = float(np.abs(probs - ref).max())
        res["seconds_total"] = time.perf_counter() - t0
        out["variants"][variant] = res
        print(
            variant, json.dumps({k: res[k] for k in ("latency_ms", "probe_accuracy")}), flush=True
        )
        del gate

    # thread scaling for the fastest full-length variant
    best = min(
        (v for v in out["variants"] if not v.endswith("512")),
        key=lambda v: out["variants"][v]["latency_ms"]["p50"],
    )
    sweep = {}
    for n in sorted({4, 8, 16, args.threads}):
        torch.set_num_threads(n)
        gate = make_gate(ckpt, best, n)
        items = [gate.encode(r["prompt"], r["candidate_a"], r["candidate_b"]) for r in lat_rows]
        sweep[n] = float(np.median(timed(gate, items, min(args.runs, 50), warmup=3)))
        del gate
    out["thread_sweep"] = {"variant": best, "p50_ms_by_threads": sweep}
    torch.set_num_threads(args.threads)
    update_metrics("cpu_inference", out)
    print(json.dumps(out, indent=2))


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return os.uname().machine


if __name__ == "__main__":
    main()
