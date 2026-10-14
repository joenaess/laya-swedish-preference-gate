"""Latency, throughput, positional consistency and margin coverage of the calibrated gate.

Everything here is measured on `val_eval`, the half of validation that neither checkpoint
selection nor the temperature fit used.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from .predictor import SwedishPreferenceGate, collate, forward_logits
from .utils import (
    CACHE_DIR,
    CHECKPOINT_DIR,
    DEFAULT_MARGIN,
    LATENCY_TARGET_MS,
    MAX_LEN,
    SPLITS_DIR,
    percentiles,
    read_jsonl,
    update_metrics,
)

MARGINS = [round(0.05 * i, 2) for i in range(10)]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_forward(gate: SwedishPreferenceGate, items, runs: int, warmup: int) -> list[float]:
    """Milliseconds per batch-1 forward pass (host->device copy + model + logits to host)."""
    for i in range(warmup):
        forward_logits(
            gate.model, collate([items[i % len(items)]], gate.tok.pad_token_id), gate.device
        ).cpu()
    times = []
    for i in range(runs):
        batch = collate([items[i % len(items)]], gate.tok.pad_token_id)
        _sync(gate.device)
        t = time.perf_counter()
        forward_logits(gate.model, batch, gate.device).cpu()
        _sync(gate.device)
        times.append((time.perf_counter() - t) * 1000)
    return times


def time_tokenization(gate: SwedishPreferenceGate, rows) -> list[float]:
    times = []
    for r in rows:
        t = time.perf_counter()
        gate.encode(r["prompt"], r["candidate_a"], r["candidate_b"])
        times.append((time.perf_counter() - t) * 1000)
    return times


def time_end_to_end(gate: SwedishPreferenceGate, rows, warmup: int) -> list[float]:
    for r in rows[:warmup]:
        gate.predict_preference(r["prompt"], r["candidate_a"], r["candidate_b"])
    times = []
    for r in rows:
        _sync(gate.device)
        t = time.perf_counter()
        gate.predict_preference(r["prompt"], r["candidate_a"], r["candidate_b"])
        _sync(gate.device)
        times.append((time.perf_counter() - t) * 1000)
    return times


def throughput(gate: SwedishPreferenceGate, items, batch_size: int) -> dict[str, float]:
    gate.logits(items[: max(batch_size, 8)], batch_size=batch_size)  # warm-up
    _sync(gate.device)
    t = time.perf_counter()
    gate.logits(items, batch_size=batch_size)
    _sync(gate.device)
    dt = time.perf_counter() - t
    return {
        "batch_size": batch_size,
        "pairs": len(items),
        "seconds": dt,
        "pairs_per_second": len(items) / dt,
    }


def max_length_item(gate: SwedishPreferenceGate, rows):
    """A real validation pair whose encoded input fills the full 1024-token window."""
    for r in rows:
        item = gate.encode(r["prompt"], r["candidate_a"], r["candidate_b"])
        if len(item["ids"]) >= MAX_LEN - 2:
            return item
    return max(
        (gate.encode(r["prompt"], r["candidate_a"], r["candidate_b"]) for r in rows),
        key=lambda it: len(it["ids"]),
    )


def positional_and_margin(gate: SwedishPreferenceGate, rows) -> tuple[dict, dict]:
    """Score each pair in both orders. Returns (metrics, per-pair p_a for the judge step)."""
    labels = np.array([r["label"] for r in rows])
    fwd = [gate.encode(r["prompt"], r["candidate_a"], r["candidate_b"]) for r in rows]
    rev = [gate.encode(r["prompt"], r["candidate_b"], r["candidate_a"]) for r in rows]
    p_fwd = gate.probabilities(gate.logits(fwd))[:, 0]  # P(stored candidate_a wins)
    p_rev = gate.probabilities(gate.logits(rev))[:, 0]  # P(stored candidate_b wins)

    # P(chosen wins) in each presentation order.
    pc_fwd = np.where(labels == 0, p_fwd, 1 - p_fwd)
    pc_rev = np.where(labels == 0, 1 - p_rev, p_rev)
    # Instances with chosen in slot A: forward order when label 0, reversed order when label 1.
    pc_chosen_a = np.where(labels == 0, pc_fwd, pc_rev)
    pc_chosen_b = np.where(labels == 0, pc_rev, pc_fwd)

    winner_fwd = np.where(p_fwd >= 0.5, 0, 1)  # index of the stored candidate that wins
    winner_rev = np.where(p_rev >= 0.5, 1, 0)
    both = np.concatenate([pc_fwd, pc_rev])

    def acc(p):
        return float((p > 0.5).mean())

    margin_rows = []
    for m in MARGINS:
        confident = np.abs(both - 0.5) >= m
        margin_rows.append(
            {
                "margin": m,
                "coverage": float(confident.mean()),
                "accuracy_on_covered": float((both[confident] > 0.5).mean())
                if confident.any()
                else None,
            }
        )
    confident = np.abs(both - 0.5) >= DEFAULT_MARGIN
    metrics = {
        "split": "val_eval",
        "n_pairs": len(rows),
        "positional": {
            "consistent_fraction": float((winner_fwd == winner_rev).mean()),
            "accuracy_original_order": acc(pc_fwd),
            "accuracy_swapped_order": acc(pc_rev),
            "accuracy_both_orders": acc(both),
            "accuracy_chosen_in_a": acc(pc_chosen_a),
            "accuracy_chosen_in_b": acc(pc_chosen_b),
            "accuracy_gap_a_minus_b": acc(pc_chosen_a) - acc(pc_chosen_b),
            "accuracy_order_averaged_prob": acc((pc_fwd + pc_rev) / 2),
            "mean_p_slot_a": float(np.concatenate([p_fwd, p_rev]).mean()),
        },
        "margin": {
            "default_margin": DEFAULT_MARGIN,
            "coverage": float(confident.mean()),
            "accuracy_on_covered": float((both[confident] > 0.5).mean())
            if confident.any()
            else None,
            "accuracy_all": acc(both),
            "sweep": margin_rows,
            "note": "over both presentation orders (2 x n_pairs instances)",
        },
    }
    per_pair = {
        r["id"]: {"label": int(r["label"]), "p_a_fwd": float(a), "p_b_rev": float(b)}
        for r, a, b in zip(rows, p_fwd, p_rev, strict=True)
    }
    return metrics, per_pair


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--cpu-runs", type=int, default=30)
    args = parser.parse_args(argv)

    rows = read_jsonl(SPLITS_DIR / "val_eval.jsonl")
    out: dict = {"split": "val_eval"}

    gate = SwedishPreferenceGate(CHECKPOINT_DIR)
    out["temperature"] = gate.temperature
    quality, per_pair = positional_and_margin(gate, rows)
    out.update(quality)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "gate_val_eval.json").write_text(json.dumps(per_pair), encoding="utf-8")

    sample = rows[: args.runs]
    items = [gate.encode(r["prompt"], r["candidate_a"], r["candidate_b"]) for r in sample]
    lengths = [len(it["ids"]) for it in items]
    latency: dict = {
        "method": (
            f"batch size 1, {args.warmup} warm-up passes, then {args.runs} timed passes over "
            f"{len(sample)} distinct val_eval pairs; torch.cuda.synchronize() before and after "
            "each timed pass; forward = host->device copy + model + logits back to host; "
            "tokenization = token-level truncation + laya sequence building"
        ),
        "input_tokens": percentiles(lengths),
    }
    if gate.device.type == "cuda":
        latency["device"] = torch.cuda.get_device_name(0)
        latency["dtype"] = str(gate.dtype)
        latency["gpu_forward_ms"] = percentiles(time_forward(gate, items, args.runs, args.warmup))
        long_item = max_length_item(gate, rows)
        latency["gpu_forward_ms_max_length"] = {
            "input_tokens": len(long_item["ids"]),
            **percentiles(time_forward(gate, [long_item], args.runs, args.warmup)),
        }
        latency["gpu_end_to_end_ms"] = percentiles(time_end_to_end(gate, sample, args.warmup))
        latency["throughput_gpu"] = [
            throughput(
                gate,
                [gate.encode(r["prompt"], r["candidate_a"], r["candidate_b"]) for r in rows[:512]],
                bs,
            )
            for bs in (1, 8, 32)
        ]
        med = latency["gpu_forward_ms"]["p50"]
        latency["target_ms"] = LATENCY_TARGET_MS
        latency["target_met"] = med < LATENCY_TARGET_MS
        latency["target_criterion"] = "median GPU forward pass at batch size 1 on real inputs"
        latency["p95_under_target"] = latency["gpu_forward_ms"]["p95"] < LATENCY_TARGET_MS
    else:
        latency["gpu"] = "not measured: no CUDA device"
        latency["target_met"] = None
    latency["tokenization_ms"] = percentiles(time_tokenization(gate, sample))
    del gate
    torch.cuda.empty_cache()

    cpu_gate = SwedishPreferenceGate(CHECKPOINT_DIR, device="cpu")
    cpu_items = items[: args.cpu_runs]
    latency["cpu_forward_ms"] = percentiles(time_forward(cpu_gate, cpu_items, args.cpu_runs, 3))
    latency["cpu_threads"] = torch.get_num_threads()
    latency["cpu_dtype"] = str(cpu_gate.dtype)
    out["latency"] = latency
    update_metrics("benchmark", out)
    print(json.dumps(out, indent=2)[:4000])


if __name__ == "__main__":
    main()
