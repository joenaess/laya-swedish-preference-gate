"""Post-hoc temperature scaling.

`T` is fitted on the calibration half of validation (`val_calib`) and every reported
calibration number is measured on the other half (`val_eval`), which neither training,
checkpoint selection nor the fit has seen.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from .utils import (
    CACHE_DIR,
    CHECKPOINT_DIR,
    ECE_BINS,
    ECE_BINS_PATH,
    ECE_TARGET,
    SPLITS_DIR,
    read_jsonl,
    update_metrics,
)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def nll(logits: np.ndarray, labels: Sequence[int], temperature: float = 1.0) -> float:
    p = softmax(logits, temperature)
    y = np.asarray(labels)
    if y.ndim == 2:  # soft targets
        return float(-np.mean((y * np.log(np.clip(p, 1e-12, 1.0))).sum(-1)))
    return float(-np.mean(np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1.0))))


def fit_temperature(logits: np.ndarray, labels: Sequence[int], max_iter: int = 200) -> float:
    """Single scalar T > 0 minimizing NLL, by L-BFGS over log T (so T = exp(log_T) > 0).

    `labels` are class indices, or an `[n, k]` array of soft target distributions.
    """
    x = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    y_arr = np.asarray(labels)
    y = torch.as_tensor(y_arr, dtype=torch.float64 if y_arr.ndim == 2 else torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(x / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def ece(
    probs: np.ndarray, labels: Sequence[int], n_bins: int = ECE_BINS
) -> tuple[float, list[dict[str, Any]]]:
    """Expected calibration error of the top-1 confidence, with `n_bins` equal-width bins."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    conf = probs.max(axis=-1)
    correct = (probs.argmax(axis=-1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total, bins = 0.0, []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        n = int(mask.sum())
        row = {"lower": float(lo), "upper": float(hi), "count": n}
        if n:
            row["confidence"] = float(conf[mask].mean())
            row["accuracy"] = float(correct[mask].mean())
            total += n / len(conf) * abs(row["confidence"] - row["accuracy"])
        else:
            row["confidence"] = row["accuracy"] = None
        bins.append(row)
    return float(total), bins


def score_split(gate, rows) -> tuple[np.ndarray, np.ndarray]:
    items = [gate.encode(r["prompt"], r["candidate_a"], r["candidate_b"]) for r in rows]
    return gate.logits(items), np.array([r["label"] for r in rows])


def main(argv=None) -> None:
    from laya.common import TEMP_MAX, TEMP_MIN

    from .predictor import TEMPERATURE_FILE, SwedishPreferenceGate

    gate = SwedishPreferenceGate(CHECKPOINT_DIR, temperature=1.0)
    out: dict[str, Any] = {}
    data = {}
    for split in ("val_calib", "val_eval"):
        logits, labels = score_split(gate, read_jsonl(SPLITS_DIR / f"{split}.jsonl"))
        data[split] = (logits, labels)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE_DIR / f"logits_{split}.npz", logits=logits, labels=labels)

    t = fit_temperature(*data["val_calib"])
    report, bins_out = {}, {}
    for split, (logits, labels) in data.items():
        pre_ece, pre_bins = ece(softmax(logits), labels)
        post_ece, post_bins = ece(softmax(logits, t), labels)
        report[split] = {
            "n": len(labels),
            "accuracy": float((logits.argmax(-1) == labels).mean()),
            "ece_pre": pre_ece,
            "ece_post": post_ece,
            "nll_pre": nll(logits, labels),
            "nll_post": nll(logits, labels, t),
        }
        bins_out[split] = {"pre": pre_bins, "post": post_bins}

    (CHECKPOINT_DIR / TEMPERATURE_FILE).write_text(
        json.dumps({"temperature": t, "fitted_on": "val_calib", "n": len(data["val_calib"][1])})
        + "\n",
        encoding="utf-8",
    )
    # Also write it where `laya.load` looks, so the checkpoint is calibrated there too.
    cfg_path = CHECKPOINT_DIR / "rl_agent_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["temperature"] = [t, 1.0, 1.0]
    cfg["temperature_by_options"] = {}
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    ev = report["val_eval"]
    out = {
        "method": "temperature scaling, single scalar T = exp(log_T), L-BFGS on NLL",
        "fit_split": "val_calib",
        "eval_split": "val_eval",
        "n_bins": ECE_BINS,
        "temperature": t,
        "temperature_within_laya_clamp": bool(TEMP_MIN <= t <= TEMP_MAX),
        "ece_pre": ev["ece_pre"],
        "ece_post": ev["ece_post"],
        "nll_pre": ev["nll_pre"],
        "nll_post": ev["nll_post"],
        "accuracy": ev["accuracy"],
        "target_ece": ECE_TARGET,
        "target_met": ev["ece_post"] <= ECE_TARGET,
        "per_split": report,
        "note": "headline ECE/NLL are on val_eval (held out from the fit); val_calib numbers "
        "are in-sample for the fit and therefore optimistic",
    }
    update_metrics("calibration", out)
    ECE_BINS_PATH.write_text(json.dumps(bins_out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_split"}, indent=2))


if __name__ == "__main__":
    main()
