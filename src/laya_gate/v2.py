"""v2 gate: `preference` + `naturalness`, trained on the original pairs and fluent-vs-fluent pairs.

    python -m laya_gate.v2 train       # fine-tune from the base multilingual checkpoint
    python -m laya_gate.v2 calibrate   # one temperature per question, fitted on val_calib
    python -m laya_gate.v2 evaluate    # val_eval of both sources, the probe, cascade, latency

Training data:
- original pairs (`jonasaise/dpo-swedish-interpreted`): idiomatic chosen vs. deliberately poorly
  translated rejected. Targets: preference = chosen, naturalness = chosen.
- fluent pairs (`laya_gate.fluent`): both answers natural Swedish, different quality.
  Targets: preference = higher-rated answer, naturalness = 50/50.
Every pair is encoded in both presentation orders; each epoch draws one order per item.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from .calibrate import ece, fit_temperature, nll
from .data import state_room_for, truncate_triplet
from .fluent import FLUENT_DIR
from .judge import (
    JUDGE_CACHE_PATH,
    cache_key,
    cascade_table,
    judge_correct,
    load_cache,
    recommend,
)
from .probe import PROBE_DIR, bootstrap_ci, summarize
from .train import Hyper, _collate_train, choose_memory_config, save
from .utils import (
    BASE_MODEL_SUBFOLDER,
    DEFAULT_MARGIN,
    NATURALNESS_QID,
    QID,
    QUESTIONS_FILE,
    QUESTIONS_V2,
    ROOT,
    SPLITS_DIR,
    read_jsonl,
    update_metrics,
)

V2_DIR = ROOT / "checkpoints" / "v2"
SOURCES = ("original", "fluent")
JUDGE_MODEL = "kimi-k3"


# --------------------------------------------------------------------------- data


def load_rows(split: str) -> dict[str, list[dict[str, Any]]]:
    return {
        "original": read_jsonl(SPLITS_DIR / f"{split}.jsonl"),
        "fluent": read_jsonl(FLUENT_DIR / f"{split}.jsonl"),
    }


def targets(source: str, qid: str, label: int) -> list[float]:
    """Target over (candidate_a, candidate_b) in the stored order; label 1 = B is chosen."""
    if source == "fluent" and qid == NATURALNESS_QID:
        return [0.5, 0.5]
    return [1.0, 0.0] if label == 0 else [0.0, 1.0]


def encode_both_orders(tok, internal, row, room, max_len, head_max_len) -> dict[str, Any]:
    """Token ids for every question in both orders: {"fwd"|"rev": {qid: (ids, markers)}}."""
    from laya.common import build_sequence

    out = {}
    for order, (a, b) in (
        ("fwd", (row["candidate_a"], row["candidate_b"])),
        ("rev", (row["candidate_b"], row["candidate_a"])),
    ):
        state, _, _ = truncate_triplet(tok, row["prompt"], a, b, room)
        out[order] = {}
        for qid, q in internal.items():
            ids, markers = build_sequence(tok, state, q, max_len, head_max_len)
            out[order][qid] = (np.asarray(ids, dtype=np.int32), markers)
    return out


def build_units(tok, rows_by_source, internal, room, max_len, head_max_len):
    """One unit per (pair, question) with both-order encodings and the stored-order target."""
    units = []
    for source, rows in rows_by_source.items():
        for row in rows:
            enc = encode_both_orders(tok, internal, row, room, max_len, head_max_len)
            for qid in internal:
                units.append(
                    {
                        "source": source,
                        "qid": qid,
                        "id": row["id"],
                        "label": row["label"],
                        "target": targets(source, qid, row["label"]),
                        "fwd": enc["fwd"][qid],
                        "rev": enc["rev"][qid],
                    }
                )
    return units


def as_item(unit, order: str) -> dict[str, Any]:
    """A training item; single-response units (v3) have one encoding and no order to flip."""
    if unit.get("single"):
        ids, markers = unit["fwd"]
        return {
            "ids": ids.tolist(),
            "markers": markers,
            "qtype": unit["qtype"],
            "target": unit["target"],
        }
    ids, markers = unit[order]
    target = unit["target"] if order == "fwd" else unit["target"][::-1]
    return {"ids": ids.tolist(), "markers": markers, "qtype": 0, "target": target}


def batch_loss_k(model, batch, device):
    """Soft-target cross-entropy over every option the item has (2 for pairwise questions,
    5 for the quality score); padded options are masked to -1e4 by the model."""
    args = [
        batch[k].to(device)
        for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
    ]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits, _ = model(*args)
    logits = logits.float()
    target = batch["target"].to(device)
    loss = -(target * torch.log_softmax(logits, -1)).sum(-1).mean()
    return loss, logits


# ------------------------------------------------------------------------- train


@torch.no_grad()
def eval_units(model, units, pad_id, device, batch_size=32) -> dict[str, dict[str, float]]:
    """Loss and accuracy per (source, question), stored order; MAE for the quality score."""
    model.eval()
    order = sorted(range(len(units)), key=lambda i: len(units[i]["fwd"][0]))
    sums: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
    for s in range(0, len(order), batch_size):
        chunk = [units[i] for i in order[s : s + batch_size]]
        batch = _collate_train([as_item(u, "fwd") for u in chunk], pad_id)
        _, logits = batch_loss_k(model, batch, device)
        logp = torch.log_softmax(logits, -1).cpu().numpy()
        for u, lp in zip(chunk, logp, strict=True):
            k = len(u["target"])
            lp = lp[:k]
            key = (u["source"], u["qid"])
            loss = -float(np.dot(u["target"], lp))
            hard = max(u["target"]) > 0.5
            correct = float(np.argmax(lp) == u["label"]) if hard else float("nan")
            err = (
                abs(float(np.exp(lp) @ np.arange(1, k + 1)) - u["rating"])
                if "rating" in u
                else float("nan")
            )
            sums.setdefault(key, []).append((loss, correct, err))
    out = {}
    for (src, qid), vals in sums.items():
        arr = np.array(vals)
        out[f"{src}/{qid}"] = {
            "loss": float(arr[:, 0].mean()),
            "acc": float(np.nanmean(arr[:, 1])) if not np.isnan(arr[:, 1]).all() else None,
            "n": len(vals),
        }
        if not np.isnan(arr[:, 2]).all():
            out[f"{src}/{qid}"]["mae"] = float(np.nanmean(arr[:, 2]))
    out["mean_loss"] = float(np.mean([v["loss"] for v in out.values()]))
    return out


def setup_training(args):
    """Seeded base model, tokenizer and config on the training device."""
    from laya.train import load_checkpoint

    hp = Hyper(epochs=args.epochs)
    random.seed(hp.seed)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, cfg = load_checkpoint(BASE_MODEL_SUBFOLDER)
    model.to(device)
    return hp, device, model, tok, cfg


def train(args) -> None:
    from laya.train import to_internal

    hp, device, model, tok, cfg = setup_training(args)
    internal = {qid: to_internal(qid, q) for qid, q in QUESTIONS_V2.items()}
    room = state_room_for(tok, QUESTIONS_V2, hp.max_len, hp.head_max_len)

    t = time.perf_counter()
    data = {}
    for split in ("train", "val_calib"):
        rows = load_rows(split)
        if args.limit:
            rows = {k: v[: args.limit] for k, v in rows.items()}
        data[split] = build_units(tok, rows, internal, room, hp.max_len, hp.head_max_len)
    print(
        {k: len(v) for k, v in data.items()},
        f"encoded in {time.perf_counter() - t:.0f}s",
        flush=True,
    )
    run_training(hp, device, model, tok, cfg, data, QUESTIONS_V2, V2_DIR, "v2_training", "v2")


def run_training(hp, device, model, tok, cfg, data, questions, out_dir, section, recipe) -> None:
    """Train on `data["train"]` units, keep the epoch with the lowest mean val_calib loss."""
    from transformers import get_cosine_schedule_with_warmup

    pad = tok.pad_token_id
    train_units = data["train"]
    probe_items = [as_item(u, "fwd") for u in train_units]
    hp = choose_memory_config(model, probe_items, pad, device, hp)
    del probe_items
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    micro_steps = math.ceil(len(train_units) / hp.micro_batch)
    total_updates = math.ceil(micro_steps / hp.grad_accum) * hp.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(hp.warmup_frac * total_updates), total_updates
    )
    history, best = [], {"loss": float("inf"), "epoch": None}
    t_start = time.perf_counter()
    for epoch in range(hp.epochs):
        t_epoch = time.perf_counter()
        model.train()
        rng = random.Random(hp.seed + epoch)
        order = list(range(len(train_units)))
        rng.shuffle(order)
        total, n = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, s in enumerate(range(0, len(order), hp.micro_batch)):
            chunk = [
                as_item(train_units[i], rng.choice(("fwd", "rev")))
                for i in order[s : s + hp.micro_batch]
            ]
            loss, _ = batch_loss_k(model, _collate_train(chunk, pad), device)
            window = min(hp.grad_accum, micro_steps - (step // hp.grad_accum) * hp.grad_accum)
            (loss / window).backward()
            total += loss.item()
            n += 1
            if (step + 1) % hp.grad_accum == 0 or step + 1 == micro_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (step + 1) % 200 == 0:
                print(
                    f"epoch {epoch + 1} step {step + 1}/{micro_steps} loss {total / n:.4f} "
                    f"{time.perf_counter() - t_epoch:.0f}s",
                    flush=True,
                )
        val = eval_units(model, data["val_calib"], pad, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": total / n,
            "val_calib": val,
            "epoch_seconds": time.perf_counter() - t_epoch,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val["mean_loss"] < best["loss"]:
            best = {"loss": val["mean_loss"], "epoch": epoch + 1}
            save(model, tok, cfg, hp, out_dir, {"best_epoch": epoch + 1, "recipe": recipe})
            (out_dir / QUESTIONS_FILE).write_text(json.dumps(questions, indent=2) + "\n")
            print(f"saved best {recipe} checkpoint (epoch {epoch + 1})", flush=True)

    update_metrics(
        section,
        {
            "init": f"convaiinnovations/laya (subfolder={BASE_MODEL_SUBFOLDER})",
            "questions": questions,
            "hyperparameters": asdict(hp),
            "order_resampling": "each pair encoded in both orders; one order drawn per item "
            "per epoch",
            "train_units": len(train_units),
            "train_units_by_source_question": _count(train_units),
            "optimizer_updates": total_updates,
            "history": history,
            "best_epoch": best["epoch"],
            "selection": "lowest mean val_calib loss over (source, question) groups",
            "total_seconds": time.perf_counter() - t_start,
            "per_epoch_seconds": [h["epoch_seconds"] for h in history],
            "peak_gpu_memory_allocated_gib": torch.cuda.max_memory_allocated() / 2**30
            if device.type == "cuda"
            else None,
            "hardware": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        },
    )


def _count(units) -> dict[str, int]:
    out: dict[str, int] = {}
    for u in units:
        key = f"{u['source']}/{u['qid']}"
        out[key] = out.get(key, 0) + 1
    return out


# --------------------------------------------------------------------- calibrate


def gate_probs(gate, rows, qid, raw: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """P(stored candidate_a wins) in the stored order and P(stored candidate_a wins) when
    shown reversed (i.e. 1 - P(slot A) of the swapped input)."""
    fwd = [gate.encode(r["prompt"], r["candidate_a"], r["candidate_b"], qid) for r in rows]
    rev = [gate.encode(r["prompt"], r["candidate_b"], r["candidate_a"], qid) for r in rows]
    lf, lr = gate.logits(fwd, batch_size=8), gate.logits(rev, batch_size=8)
    if raw:
        return lf, lr
    return gate.probabilities(lf, qid)[:, 0], 1 - gate.probabilities(lr, qid)[:, 0]


def fit_pairwise_temperatures(gate) -> dict[str, dict[str, Any]]:
    """Temperatures for `preference` (original + fluent) and `naturalness` (original), fitted
    on val_calib in both orders. `gate` must be loaded with every temperature at 1.0."""
    from laya.common import TEMP_MAX, TEMP_MIN

    calib = load_rows("val_calib")
    fits = {}
    for qid, sources in ((QID, SOURCES), (NATURALNESS_QID, ("original",))):
        logits, labels = [], []
        for src in sources:
            lf, lr = gate_probs(gate, calib[src], qid, raw=True)
            y = np.array([r["label"] for r in calib[src]])
            logits += [lf, lr[:, ::-1]]  # reversed input, columns mapped back to stored order
            labels += [y, y]
        logits, labels = np.concatenate(logits), np.concatenate(labels)
        t = fit_temperature(logits, labels)
        fits[qid] = {
            "temperature": t,
            "fit_on": f"val_calib {'+'.join(sources)}, both orders",
            "n": int(len(labels)),
            "nll_pre": nll(logits, labels),
            "nll_post": nll(logits, labels, t),
            "within_laya_clamp": bool(TEMP_MIN <= t <= TEMP_MAX),
        }
    return fits


def uncalibrated_gate(path):
    from .predictor import SwedishPreferenceGate

    gate = SwedishPreferenceGate(path, temperature=1.0)
    gate.temperatures = {q: 1.0 for q in gate.temperatures}
    return gate


def save_temperatures(path, fits: dict[str, dict[str, Any]], section: str) -> None:
    from .predictor import TEMPERATURE_FILE

    temps = {q: f["temperature"] for q, f in fits.items()}
    (path / TEMPERATURE_FILE).write_text(
        json.dumps({"temperature": temps[QID], "temperatures": temps}, indent=2) + "\n"
    )
    cfg_path = path / "rl_agent_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["temperature"], cfg["temperature_by_options"] = [temps[QID], 1.0, 1.0], {}
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    update_metrics(section, fits)
    print(json.dumps(fits, indent=2))


def calibrate(args) -> None:
    fits = fit_pairwise_temperatures(uncalibrated_gate(V2_DIR))
    save_temperatures(V2_DIR, fits, "v2_calibration")


# ---------------------------------------------------------------------- evaluate


def pair_metrics(p_fwd, p_rev, labels) -> dict[str, Any]:
    """Accuracy/ECE for P(stored A wins) in both orders; label 1 = stored B is chosen."""
    labels = np.asarray(labels)
    pc_fwd = np.where(labels == 0, p_fwd, 1 - p_fwd)  # P(chosen) per order
    pc_rev = np.where(labels == 0, p_rev, 1 - p_rev)
    both = np.concatenate([pc_fwd, pc_rev])
    avg = (pc_fwd + pc_rev) / 2
    e, _ = ece(np.stack([1 - both, both], axis=1), np.ones(len(both), dtype=int))
    chosen_a = np.where(labels == 0, pc_fwd, pc_rev)
    chosen_b = np.where(labels == 0, pc_rev, pc_fwd)
    margin_ok = np.abs(both - 0.5) >= DEFAULT_MARGIN
    e_avg, _ = ece(np.stack([1 - avg, avg], axis=1), np.ones(len(avg), dtype=int))
    avg_ok = np.abs(avg - 0.5) >= DEFAULT_MARGIN
    return {
        "n_pairs": int(len(labels)),
        "accuracy_both_orders": summarize(both > 0.5),
        "accuracy_order_averaged_prob": summarize(avg > 0.5),
        "accuracy_chosen_in_a": float((chosen_a > 0.5).mean()),
        "accuracy_chosen_in_b": float((chosen_b > 0.5).mean()),
        "positional_consistency": float(((p_fwd > 0.5) == (p_rev > 0.5)).mean()),
        "ece_both_orders": e,
        "mean_confidence": float(np.maximum(both, 1 - both).mean()),
        "default_margin": {
            "margin": DEFAULT_MARGIN,
            "coverage": float(margin_ok.mean()),
            "accuracy_on_covered": float((both[margin_ok] > 0.5).mean())
            if margin_ok.any()
            else None,
        },
        # same definitions as the v1 probe metrics (order-averaged probability per pair)
        "ece_order_averaged": e_avg,
        "default_margin_order_averaged": {
            "margin": DEFAULT_MARGIN,
            "coverage": float(avg_ok.mean()),
            "accuracy_on_covered": float((avg[avg_ok] > 0.5).mean()) if avg_ok.any() else None,
            "accuracy_on_routed": float((avg[~avg_ok] > 0.5).mean()) if (~avg_ok).any() else None,
        },
    }


def neutrality(p_fwd, p_rev) -> dict[str, float]:
    """For pairs that should be a tie: how far from 0.5 the order-averaged probability is."""
    avg = (np.asarray(p_fwd) + np.asarray(p_rev)) / 2
    return {
        "mean_abs_dev_from_0.5": float(np.abs(avg - 0.5).mean()),
        "fraction_within_0.2": float((np.abs(avg - 0.5) < 0.2).mean()),
    }


def evaluate(args) -> None:
    from .benchmark import time_forward
    from .predictor import SwedishPreferenceGate

    gate = SwedishPreferenceGate(args.checkpoint or V2_DIR)
    out: dict[str, Any] = {"checkpoint": args.section, "temperatures": gate.temperatures}
    ev = load_rows("val_eval")

    for src in SOURCES:
        rows, labels = ev[src], [r["label"] for r in ev[src]]
        pf, pr = gate_probs(gate, rows, QID)
        out[f"{src}/preference"] = pair_metrics(pf, pr, labels)
        if not gate.has_naturalness:
            continue
        nf, nr = gate_probs(gate, rows, NATURALNESS_QID)
        out[f"{src}/naturalness"] = (
            pair_metrics(nf, nr, labels) if src == "original" else neutrality(nf, nr)
        )

    probe = read_jsonl(PROBE_DIR / "probe_pairs.jsonl")
    prows = [
        {
            "prompt": p["sv_prompt"],
            "candidate_a": p["sv_chosen"],
            "candidate_b": p["sv_rejected"],
            "label": 0,
            "id": p["id"],
        }
        for p in probe
    ]
    pf, pr = gate_probs(gate, prows, QID)
    pm = pair_metrics(pf, pr, [0] * len(prows))
    conf = np.maximum((pf + pr) / 2, 1 - (pf + pr) / 2)
    ok = (pf + pr) / 2 > 0.5
    pm["by_confidence"] = [
        {
            "confidence": [lo, min(hi, 1.0)],
            "fraction": float(m.mean()),
            "accuracy": float(ok[m].mean()) if m.any() else None,
        }
        for lo, hi in ((0.5, 0.7), (0.7, 0.9), (0.9, 1.01))
        for m in [(conf >= lo) & (conf < hi)]
    ]
    pm["accuracy_order_averaged_prob"]["ci95"] = bootstrap_ci(ok.astype(float))
    out["probe/preference"] = pm
    if gate.has_naturalness:
        nf, nr = gate_probs(gate, prows, NATURALNESS_QID)
        out["probe/naturalness"] = neutrality(nf, nr)

    # cascade on the probe, with the cached Kimi-k3 verdicts from the probe run
    cache = load_cache(JUDGE_CACHE_PATH)
    gate_p, judge_ok = [], []
    for p, a, b in zip(probe, pf, pr, strict=True):
        for order, gp, (x, y) in (
            ("AB", a, (p["sv_chosen"], p["sv_rejected"])),
            ("BA", b, (p["sv_rejected"], p["sv_chosen"])),
        ):
            v = cache.get(cache_key(JUDGE_MODEL, p["sv_prompt"], x, y), {}).get("verdict")
            gate_p.append(gp)
            judge_ok.append(judge_correct(v, 0, order))

    items = [
        gate.encode(r["prompt"], r["candidate_a"], r["candidate_b"]) for r in ev["original"][:200]
    ]
    lat = time_forward(gate, items, runs=200, warmup=20)
    gate_ms = float(np.median(lat))
    judge_ms = _judge_median_ms()
    table = cascade_table(np.array(gate_p), judge_ok, gate_ms, judge_ms)
    out["probe/cascade"] = {
        "table": table,
        "recommended": recommend(table),
        "judge_accuracy": float(np.mean([j for j in judge_ok if j is not None])),
    }
    out["latency_gpu_forward_ms"] = {
        "p50": gate_ms,
        "p95": float(np.percentile(lat, 95)),
        "n": len(lat),
    }
    update_metrics(args.section, out)
    print(json.dumps({k: v for k, v in out.items() if k != "probe/cascade"}, indent=2)[:6000])
    print(json.dumps(out["probe/cascade"]["recommended"], indent=2))


def _judge_median_ms() -> float:
    from .utils import load_metrics

    j = load_metrics().get("judge")
    return float(j["latency_ms"]["p50"]) if isinstance(j, dict) else float("nan")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["train", "calibrate", "evaluate"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="debug: N rows per source")
    parser.add_argument(
        "--checkpoint", default=None, help="evaluate: another checkpoint, e.g. the v1 model"
    )
    parser.add_argument("--section", default="v2_eval", help="evaluate: metrics.json section")
    args = parser.parse_args(argv)
    {"train": train, "calibrate": calibrate, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
