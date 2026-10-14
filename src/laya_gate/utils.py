"""Shared constants, paths and the `results/metrics.json` store."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
METRICS_PATH = RESULTS_DIR / "metrics.json"
ECE_BINS_PATH = RESULTS_DIR / "ece_bins.json"
JUDGE_CACHE_PATH = RESULTS_DIR / "judge_cache.jsonl"
CACHE_DIR = ROOT / "cache"
SPLITS_DIR = CACHE_DIR / "splits"
CHECKPOINT_DIR = ROOT / "checkpoints" / "best_model"

BASE_MODEL_REPO = "convaiinnovations/laya"
BASE_MODEL_SUBFOLDER = "multilingual"
DATASET_REPO = "jonasaise/dpo-swedish-interpreted"
GITHUB_REPO = "joenaess/laya-swedish-preference-gate"
DEFAULT_HF_REPO = "joenaess/laya-swedish-preference-gate"

MAX_LEN = 1024
HEAD_MAX_LEN = 256
SEED = 20261014
DEFAULT_MARGIN = 0.2
ECE_BINS = 15
ECE_TARGET = 0.085
LATENCY_TARGET_MS = 35.0

QID = "preference"
QUESTION = {
    QID: {
        "type": "choice",
        "instructions": "Which candidate response better answers the prompt?",
        "criteria": {
            "candidate_a": "response A is superior",
            "candidate_b": "response B is superior",
        },
    }
}
OPTIONS = ("candidate_a", "candidate_b")


def load_metrics() -> dict[str, Any]:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}


def update_metrics(section: str, data: Any) -> None:
    """Replace one top-level section of `results/metrics.json`, keeping the others."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()
    metrics[section] = data
    tmp = METRICS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, METRICS_PATH)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def percentiles(values, ps=(50, 95)) -> dict[str, float]:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    out = {f"p{p}": float(np.percentile(arr, p)) for p in ps}
    out["mean"] = float(arr.mean())
    out["max"] = float(arr.max())
    out["n"] = int(arr.size)
    return out
