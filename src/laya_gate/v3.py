"""v3 gate: v2's pairwise questions plus two single-response questions.

    python -m laya_gate.v3 train       # fine-tune from the base multilingual checkpoint
    python -m laya_gate.v3 calibrate   # one temperature per question, fitted on val_calib
    python -m laya_gate.v3 evaluate    # v2's pairwise evaluation + single-response metrics

Single-response questions (state = {"prompt", "response"}):
- quality (`score`, levels 1-5): fluent responses; soft target from the UltraFeedback GPT-4
  rating (1.0-5.0 in steps of 0.25), split between the two nearest levels.
- natural_swedish (`choice`): original chosen = natural, original rejected = translationese,
  every fluent response = natural (both sides of a fluent pair are natural Swedish).

Comparing N responses by quality takes N forward passes instead of N(N-1) ordered pairs.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import math
import time
from typing import Any

import numpy as np

from . import v2
from .calibrate import ece, fit_temperature, nll
from .data import state_room_for, truncate_state
from .fluent import FLUENT_DIR
from .probe import PROBE_DIR, summarize
from .utils import (
    NATURAL_SINGLE_QID,
    POINTWISE_QUESTIONS,
    QUALITY_LEVELS,
    QUALITY_QID,
    QUESTIONS_V2,
    QUESTIONS_V3,
    ROOT,
    read_jsonl,
    update_metrics,
    write_jsonl,
)

V3_DIR = ROOT / "checkpoints" / "v3"
RATINGS_PATH = FLUENT_DIR / "ratings.jsonl"


# --------------------------------------------------------------------------- data


def load_ratings() -> dict[str, tuple[float, float]]:
    """UltraFeedback (chosen_rating, rejected_rating) per fluent pair id, cached locally."""
    if not RATINGS_PATH.exists():
        from .probe import eligible_pairs

        write_jsonl(
            RATINGS_PATH,
            [
                {"id": r["id"], "chosen": r["chosen_rating"], "rejected": r["rejected_rating"]}
                for r in eligible_pairs()
            ],
        )
    return {r["id"]: (r["chosen"], r["rejected"]) for r in read_jsonl(RATINGS_PATH)}


def rating_target(rating: float) -> list[float]:
    """Soft target over levels 1..5: a rating between two levels is split linearly."""
    x = min(max(rating, 1.0), 5.0) - 1.0
    lo = int(math.floor(x))
    w = x - lo
    t = [0.0] * len(QUALITY_LEVELS)
    t[lo] = 1.0 - w
    if w > 0:
        t[lo + 1] = w
    return t


def single_rows(rows_by_source, ratings) -> list[dict[str, Any]]:
    """One row per (response, question) with its target."""
    out = []
    for row in rows_by_source.get("original", []):
        chosen, rejected = (
            (row["candidate_a"], row["candidate_b"])
            if row["label"] == 0
            else (row["candidate_b"], row["candidate_a"])
        )
        for resp, natural in ((chosen, True), (rejected, False)):
            out.append(_row("original", NATURAL_SINGLE_QID, row, resp, natural=natural))
    for row in rows_by_source.get("fluent", []):
        chosen, rejected = (
            (row["candidate_a"], row["candidate_b"])
            if row["label"] == 0
            else (row["candidate_b"], row["candidate_a"])
        )
        r_chosen, r_rejected = ratings[row["id"]]
        for resp, rating in ((chosen, r_chosen), (rejected, r_rejected)):
            out.append(_row("fluent", QUALITY_QID, row, resp, rating=rating))
            out.append(_row("fluent", NATURAL_SINGLE_QID, row, resp, natural=True))
    return out


def _row(source, qid, row, response, natural=None, rating=None) -> dict[str, Any]:
    base = {"source": source, "qid": qid, "id": row["id"], "prompt": row["prompt"]}
    if qid == QUALITY_QID:
        target = rating_target(rating)
        return {
            **base,
            "response": response,
            "target": target,
            "label": int(np.argmax(target)),
            "rating": rating,
        }
    return {
        **base,
        "response": response,
        "target": [1.0, 0.0] if natural else [0.0, 1.0],
        "label": 0 if natural else 1,
    }


def build_single_units(tok, rows, internal, room, max_len, head_max_len):
    from laya.common import QTYPES, build_sequence

    units = []
    for r in rows:
        state, _, _ = truncate_state(tok, r["prompt"], {"response": r["response"]}, room)
        q = internal[r["qid"]]
        ids, markers = build_sequence(tok, state, q, max_len, head_max_len)
        enc = (np.asarray(ids, dtype=np.int32), markers)
        unit = {k: r[k] for k in ("source", "qid", "id", "label", "target")}
        unit.update(single=True, qtype=QTYPES[q["t"]], fwd=enc, rev=enc)
        if "rating" in r:
            unit["rating"] = r["rating"]
        units.append(unit)
    return units


# ------------------------------------------------------------------------- train


def train(args) -> None:
    from laya.train import to_internal

    hp, device, model, tok, cfg = v2.setup_training(args)
    internal = {qid: to_internal(qid, q) for qid, q in QUESTIONS_V3.items()}
    pair_internal = {q: internal[q] for q in QUESTIONS_V2}
    room = state_room_for(tok, QUESTIONS_V3, hp.max_len, hp.head_max_len)
    ratings = load_ratings()

    t = time.perf_counter()
    data = {}
    for split in ("train", "val_calib"):
        rows = v2.load_rows(split)
        if args.limit:
            rows = {k: v[: args.limit] for k, v in rows.items()}
        pairs = v2.build_units(tok, rows, pair_internal, room, hp.max_len, hp.head_max_len)
        singles = build_single_units(
            tok, single_rows(rows, ratings), internal, room, hp.max_len, hp.head_max_len
        )
        data[split] = pairs + singles
    print(
        {k: len(v) for k, v in data.items()},
        f"encoded in {time.perf_counter() - t:.0f}s",
        flush=True,
    )
    v2.run_training(hp, device, model, tok, cfg, data, QUESTIONS_V3, V3_DIR, "v3_training", "v3")


# --------------------------------------------------------------------- calibrate


def single_logits(gate, rows) -> np.ndarray:
    items = [gate.encode_single(r["prompt"], r["response"], r["qid"]) for r in rows]
    return gate.logits(items, batch_size=16)


def calibrate(args) -> None:
    from laya.common import TEMP_MAX, TEMP_MIN

    gate = v2.uncalibrated_gate(V3_DIR)
    fits = v2.fit_pairwise_temperatures(gate)
    rows = single_rows(v2.load_rows("val_calib"), load_ratings())
    for qid in POINTWISE_QUESTIONS:
        sel = [r for r in rows if r["qid"] == qid]
        logits = single_logits(gate, sel)
        labels = (
            np.array([r["target"] for r in sel])
            if qid == QUALITY_QID
            else np.array([r["label"] for r in sel])
        )
        t = fit_temperature(logits, labels)
        fits[qid] = {
            "temperature": t,
            "fit_on": "val_calib "
            + ("fluent responses, soft rating targets" if qid == QUALITY_QID else "responses")
            + f" ({', '.join(sorted({r['source'] for r in sel}))})",
            "n": len(sel),
            "nll_pre": nll(logits, labels),
            "nll_post": nll(logits, labels, t),
            "within_laya_clamp": bool(TEMP_MIN <= t <= TEMP_MAX),
        }
    v2.save_temperatures(V3_DIR, fits, "v3_calibration")


# ---------------------------------------------------------------------- evaluate


def p_a_beats_b(pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """P(level_a > level_b) + 0.5 P(tie) for independent level distributions [n, 5]."""
    cdf_b = np.cumsum(pb, axis=1)
    below = np.concatenate([np.zeros((len(pb), 1)), cdf_b[:, :-1]], axis=1)  # P(b < level)
    return (pa * below).sum(1) + 0.5 * (pa * pb).sum(1)


def quality_pair_metrics(gate, prompts, chosen, rejected) -> dict[str, Any]:
    """Pairwise accuracy of the single-response quality score (chosen vs rejected)."""
    pc = np.array([s["probabilities"] for s in _quality(gate, prompts, chosen)])
    pr = np.array([s["probabilities"] for s in _quality(gate, prompts, rejected)])
    levels = np.asarray(QUALITY_LEVELS, dtype=float)
    ec, er = pc @ levels, pr @ levels
    p_win = p_a_beats_b(pc, pr)
    ok = (ec > er).astype(float)
    e, _ = ece(np.stack([1 - p_win, p_win], axis=1), np.ones(len(p_win), dtype=int))
    return {
        "n_pairs": len(ok),
        "accuracy_expected_score": summarize(ec > er),
        "ties": int((ec == er).sum()),
        "ece_p_chosen_beats_rejected": e,
        "mean_score_chosen": float(ec.mean()),
        "mean_score_rejected": float(er.mean()),
        "expected": (ec, er),
    }


def _quality(gate, prompts, responses):
    out = gate.score_quality_batch(prompts, responses, batch_size=16)
    return [
        {"score": o["score"], "probabilities": [o["probabilities"][lvl] for lvl in QUALITY_LEVELS]}
        for o in out
    ]


def roc_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a random positive > random negative), ties counted half."""
    from sklearn.metrics import roc_auc_score

    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))


def evaluate(args) -> None:
    from scipy.stats import pearsonr, spearmanr

    from .benchmark import time_forward
    from .predictor import SwedishPreferenceGate

    # pairwise questions: exactly v2's evaluation, on the v3 checkpoint
    v2.evaluate(argparse.Namespace(checkpoint=V3_DIR, section="v3_eval"))

    gate = SwedishPreferenceGate(V3_DIR)
    ev = v2.load_rows("val_eval")
    ratings = load_ratings()
    out: dict[str, Any] = {"temperatures": gate.temperatures}

    # quality: fluent val_eval responses against their UltraFeedback ratings
    fl = ev["fluent"]
    chosen = [r["candidate_a"] if r["label"] == 0 else r["candidate_b"] for r in fl]
    rejected = [r["candidate_b"] if r["label"] == 0 else r["candidate_a"] for r in fl]
    prompts = [r["prompt"] for r in fl]
    q = quality_pair_metrics(gate, prompts, chosen, rejected)
    ec, er = q.pop("expected")
    rat = np.r_[[ratings[r["id"]][0] for r in fl], [ratings[r["id"]][1] for r in fl]]
    pred = np.r_[ec, er]
    q["spearman_vs_rating"] = float(spearmanr(pred, rat).statistic)
    q["pearson_vs_rating"] = float(pearsonr(pred, rat).statistic)
    q["mae_vs_rating"] = float(np.abs(pred - rat).mean())
    out["fluent/quality"] = q

    # quality on the probe (reasoning-on translations, never trained on)
    probe = read_jsonl(PROBE_DIR / "probe_pairs.jsonl")
    pp = [p["sv_prompt"] for p in probe]
    qp = quality_pair_metrics(
        gate, pp, [p["sv_chosen"] for p in probe], [p["sv_rejected"] for p in probe]
    )
    ec, er = qp.pop("expected")
    prat = np.r_[[p["chosen_rating"] for p in probe], [p["rejected_rating"] for p in probe]]
    qp["spearman_vs_rating"] = float(spearmanr(np.r_[ec, er], prat).statistic)
    out["probe/quality"] = qp

    # natural_swedish: original val_eval (natural chosen vs translationese rejected)
    org = ev["original"]
    nat = [r["candidate_a"] if r["label"] == 0 else r["candidate_b"] for r in org]
    trn = [r["candidate_b"] if r["label"] == 0 else r["candidate_a"] for r in org]
    op = [r["prompt"] for r in org]
    p_nat = np.array([x["p_natural"] for x in gate.score_naturalness_batch(nat, op, 16)])
    p_trn = np.array([x["p_natural"] for x in gate.score_naturalness_batch(trn, op, 16)])
    probs = np.r_[p_nat, p_trn]
    labels = np.r_[np.zeros(len(p_nat), int), np.ones(len(p_trn), int)]  # 0 = natural
    e, _ = ece(np.stack([probs, 1 - probs], axis=1), labels)
    out["original/natural_swedish"] = {
        "n_responses": len(probs),
        "auc": roc_auc(p_nat, p_trn),
        "accuracy": summarize((probs >= 0.5) == (labels == 0)),
        "ece": e,
        "natural_flagged_translationese": float((p_nat < 0.5).mean()),
        "translationese_passed_as_natural": float((p_trn >= 0.5).mean()),
        "within_pair_natural_scores_higher": float((p_nat > p_trn).mean()),
    }
    # fluent and probe responses are all natural: how many does it flag?
    flu = chosen + rejected
    p_flu = np.array([x["p_natural"] for x in gate.score_naturalness_batch(flu, prompts * 2, 16)])
    probe_resp = [p["sv_chosen"] for p in probe] + [p["sv_rejected"] for p in probe]
    p_pro = np.array([x["p_natural"] for x in gate.score_naturalness_batch(probe_resp, pp * 2, 16)])
    out["fluent/natural_swedish"] = {
        "n_responses": len(p_flu),
        "flagged_translationese": float((p_flu < 0.5).mean()),
        "mean_p_natural": float(p_flu.mean()),
    }
    out["probe/natural_swedish"] = {
        "n_responses": len(p_pro),
        "flagged_translationese": float((p_pro < 0.5).mean()),
        "mean_p_natural": float(p_pro.mean()),
    }

    # single-response forward latency (GPU, batch 1)
    items = [gate.encode_single(p, r, QUALITY_QID) for p, r in zip(prompts, chosen, strict=True)]
    lat = time_forward(gate, items[:200], runs=200, warmup=20)
    out["latency_gpu_forward_ms_single_response"] = {
        "p50": float(np.median(lat)),
        "p95": float(np.percentile(lat, 95)),
        "n": len(lat),
    }
    update_metrics("v3_pointwise", out)
    print(json.dumps(out, indent=2))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["train", "calibrate", "evaluate"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="debug: N rows per source")
    args = parser.parse_args(argv)
    {"train": train, "calibrate": calibrate, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
