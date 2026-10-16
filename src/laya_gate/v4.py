"""v4 ablation: which extra training data makes the gate better at judging Swedish?

A 2x2 design, every arm fine-tuned from the base multilingual checkpoint with the v3 recipe
(v3 data and questions, 1 epoch, same seed); only the added data differs:

    control   v3 data
    europarl  + Europarl: professional human translation > Kimi-k3 machine translation
    native    + DaLAJ-GED train (corrected > learner sentence) + SweFAQ train (own answer >
                other answer on the same FAQ page)
    both      + both
    control at a second seed, for the seed-to-seed noise floor

    python -m laya_gate.v4 prepare                  # training data for the added sources
    python -m laya_gate.v4 run --arm europarl       # train, calibrate, evaluate one arm
    python -m laya_gate.v4 run --arm control --seed 20261015
    python -m laya_gate.v4 compare                  # paired differences against control

Controls in the data: the Europarl negatives are Kimi-k3 translations only, so the OPUS-MT
benchmarks (liu-nlp, FLORES+) measure transfer rather than detection of one MT system; the
Europarl pairs are length-balanced (the human side is longer in exactly half), because "the
longer one wins" alone scores above chance on liu-nlp; near-identical pairs are dropped; the
Europarl evaluation paragraphs and every Swedish-original paragraph are excluded. DaLAJ and
SweFAQ training come from their train splits only (SweFAQ splits share no FAQ page).

Every arm is scored on the same held-out items; `compare` reports each arm's difference from
control with a paired bootstrap 95% CI, next to the control-vs-control (seed) difference.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import asyncio
import difflib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import external_eval as X
from . import liu_eval, v2, v3
from .probe import PROBE_DIR, bootstrap_ci
from .utils import (
    CACHE_DIR,
    METRICS_PATH,
    NATURALNESS_QID,
    QID,
    QUESTIONS_V2,
    QUESTIONS_V3,
    ROOT,
    SEED,
    load_metrics,
    read_jsonl,
    update_metrics,
    write_jsonl,
)

V4_DIR = CACHE_DIR / "v4"
ARMS = {
    "control": (),
    "europarl": ("europarl",),
    "native": ("dalaj", "swefaq"),
    "both": ("europarl", "dalaj", "swefaq"),
}
EP_TRAIN = 12000  # Europarl pairs after filtering
EP_POOL = 18000  # paragraphs translated, before filtering
FAQ_NEGATIVES = 4  # other answers per SweFAQ question
NEAR_DUPLICATE = 0.95  # character-level similarity above which a pair is dropped


def tag(arm: str, seed: int) -> str:
    return arm if seed == SEED else f"{arm}-s{seed}"


def checkpoint_dir(arm: str, seed: int) -> Path:
    return ROOT / "checkpoints" / f"v4-{tag(arm, seed)}"


# --------------------------------------------------------------------------- data


def as_pair(rid: str, prompt: str, ref: str, other: str, rng: random.Random) -> dict[str, Any]:
    """A training row in the v2 layout with a random slot for the reference (label 1 = B)."""
    label = int(rng.random() < 0.5)
    a, b = (ref, other) if label == 0 else (other, ref)
    return {"id": rid, "prompt": prompt, "candidate_a": a, "candidate_b": b, "label": label}


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def length_balanced(pairs: list[dict[str, Any]], n: int, rng: random.Random) -> list[dict]:
    """`n` pairs, half with the reference longer and half with it shorter (ties dropped)."""
    longer = [p for p in pairs if len(p["ref"]) > len(p["other"])]
    shorter = [p for p in pairs if len(p["ref"]) < len(p["other"])]
    k = min(n // 2, len(longer), len(shorter))
    return rng.sample(longer, k) + rng.sample(shorter, k)


def europarl_train(rows, exclude: set[str], seed: int = SEED) -> list[dict[str, Any]]:
    """A fixed-seed pool of English-original paragraphs not used for evaluation."""
    pool = sorted(
        (r for r in rows if r["original"] == "EN" and r["id"] not in exclude),
        key=lambda r: r["id"],
    )
    return random.Random(seed).sample(pool, min(EP_POOL, len(pool)))


def europarl_pairs(rows, rng: random.Random) -> tuple[list[dict], dict[str, Any]]:
    """(human, Kimi-k3 MT) pairs: near duplicates dropped, then length-balanced."""
    cand, dropped = [], {"no_mt": 0, "near_duplicate": 0}
    for r in rows:
        human, mt = " ".join(r["sv"]), r.get("mt_kimi")
        if not mt:
            dropped["no_mt"] += 1
            continue
        if similarity(X._norm(human), X._norm(mt)) > NEAR_DUPLICATE:
            dropped["near_duplicate"] += 1
            continue
        cand.append({"id": r["id"], "en": " ".join(r["en"]), "ref": human, "other": mt})
    kept = length_balanced(cand, EP_TRAIN, rng)
    stats = {
        "paragraphs_translated": len(rows),
        "dropped": dropped,
        "candidates": len(cand),
        "kept": len(kept),
        "ref_longer_in_candidates": float(np.mean([len(p["ref"]) > len(p["other"]) for p in cand])),
        "ref_longer_in_kept": float(np.mean([len(p["ref"]) > len(p["other"]) for p in kept])),
    }
    return kept, stats


def faq_train(rows, rng: random.Random) -> list[dict[str, Any]]:
    """Own answer vs up to FAQ_NEGATIVES other answers from the same page."""
    pairs, _ = X.swefaq_pairs(rows, "train")
    by_q: dict[str, list] = {}
    for p in pairs:
        by_q.setdefault(p["question_id"], []).append(p)
    out = []
    for qid in sorted(by_q):
        ps = by_q[qid]
        out += rng.sample(ps, min(FAQ_NEGATIVES, len(ps)))
    return out


def prepare(args) -> None:
    from datasets import load_dataset

    V4_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    stats: dict[str, Any] = {}

    dalaj, dropped = X.dalaj_pairs(load_dataset(X.SUPERLIM, "dalaj-ged-superlim", split="train"))
    write_jsonl(
        V4_DIR / "dalaj_naturalness.jsonl",
        [as_pair(p["id"], "", p["ref"], p["other"], rng) for p in dalaj],
    )
    stats["dalaj"] = {"pairs": len(dalaj), "dropped": dropped, "question": NATURALNESS_QID}

    faq = faq_train(load_dataset(X.SUPERLIM, "swefaq", split="train"), rng)
    write_jsonl(
        V4_DIR / "swefaq_preference.jsonl",
        [as_pair(p["id"], p["prompt"], p["ref"], p["other"], rng) for p in faq],
    )
    stats["swefaq"] = {
        "pairs": len(faq),
        "questions": len({p["question_id"] for p in faq}),
        "question": QID,
    }

    pool_path = V4_DIR / "europarl_pool.jsonl"
    if pool_path.exists():
        pool = read_jsonl(pool_path)
    else:
        t = time.perf_counter()
        paras, _ = X.europarl_paragraphs()
        exclude = {r["id"] for r in read_jsonl(X.EXT_DIR / "europarl.jsonl")}
        pool = europarl_train(paras, exclude)
        print(f"europarl pool: {len(pool)} paragraphs in {time.perf_counter() - t:.0f}s")
    missing = [v for v in ("BASE_URL", "API_KEY") if not os.environ.get(v)]
    if missing:
        stats["europarl"] = f"skipped: {missing[0]} not set (Kimi-k3 translation needed)"
    else:
        stats["europarl_mt"] = asyncio.run(X._kimi_translate(pool, args.concurrency))
        write_jsonl(pool_path, pool)
        ep, ep_stats = europarl_pairs(pool, rng)
        write_jsonl(
            V4_DIR / "europarl_preference.jsonl",
            [as_pair(p["id"], X.SOURCE_PROMPT + p["en"], p["ref"], p["other"], rng) for p in ep],
        )
        write_jsonl(
            V4_DIR / "europarl_naturalness.jsonl",
            [as_pair(p["id"], "", p["ref"], p["other"], rng) for p in ep],
        )
        stats["europarl"] = {
            **ep_stats,
            "questions": {QID: "English source in the prompt", NATURALNESS_QID: "no prompt"},
            "negatives": "Kimi-k3 translations (reasoning off); OPUS-MT never used in training",
        }
    _set_v4("data", stats)
    print(json.dumps(stats, indent=2))


# ------------------------------------------------------------------------- train

SOURCE_FILES = {
    "europarl": (
        ("europarl_preference.jsonl", QID),
        ("europarl_naturalness.jsonl", NATURALNESS_QID),
    ),
    "dalaj": (("dalaj_naturalness.jsonl", NATURALNESS_QID),),
    "swefaq": (("swefaq_preference.jsonl", QID),),
}


def extra_units(tok, sources, internal, room, max_len, head_max_len) -> list[dict[str, Any]]:
    units = []
    for source in sources:
        for fname, qid in SOURCE_FILES[source]:
            rows = read_jsonl(V4_DIR / fname)
            units += v2.build_units(
                tok, {source: rows}, {qid: internal[qid]}, room, max_len, head_max_len
            )
    return units


def train(arm: str, seed: int) -> None:
    import torch
    from laya.train import load_checkpoint, to_internal

    from .data import state_room_for
    from .train import Hyper
    from .utils import BASE_MODEL_SUBFOLDER

    hp = Hyper(epochs=1, seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, cfg = load_checkpoint(BASE_MODEL_SUBFOLDER)
    model.to(device)

    internal = {qid: to_internal(qid, q) for qid, q in QUESTIONS_V3.items()}
    pair_internal = {q: internal[q] for q in QUESTIONS_V2}
    room = state_room_for(tok, QUESTIONS_V3, hp.max_len, hp.head_max_len)
    ratings = v3.load_ratings()
    t = time.perf_counter()
    data = {}
    for split in ("train", "val_calib"):
        rows = v2.load_rows(split)
        data[split] = v2.build_units(
            tok, rows, pair_internal, room, hp.max_len, hp.head_max_len
        ) + v3.build_single_units(
            tok, v3.single_rows(rows, ratings), internal, room, hp.max_len, hp.head_max_len
        )
    data["train"] += extra_units(tok, ARMS[arm], internal, room, hp.max_len, hp.head_max_len)
    print({k: len(v) for k, v in data.items()}, f"encoded in {time.perf_counter() - t:.0f}s")
    section = f"_v4_training_{tag(arm, seed)}"
    v2.run_training(
        hp, device, model, tok, cfg, data, QUESTIONS_V3, checkpoint_dir(arm, seed), section, "v4"
    )
    _move(section, tag(arm, seed), "training")


# --------------------------------------------------------------- metrics storage


def _write(metrics: dict[str, Any]) -> None:
    tmp = METRICS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, METRICS_PATH)


def _set_v4(key: str, value: Any) -> None:
    v4 = load_metrics().get("v4") or {}
    v4[key] = value
    update_metrics("v4", v4)


def _move(section: str, arm_tag: str, key: str) -> None:
    """Move a top-level section written by a shared helper into `v4.arms.<tag>.<key>`."""
    m = load_metrics()
    value = m.pop(section, None)
    m.setdefault("v4", {}).setdefault("arms", {}).setdefault(arm_tag, {})[key] = value
    _write(m)


# --------------------------------------------------------------------- evaluate


def pair_wins(gate, qid, prompts, ref, other) -> np.ndarray:
    """Per pair: 1 if the reference wins on order-averaged P, else 0."""
    pf, pr = X.pairwise(gate, qid, prompts, ref, other)
    return ((pf + pr) / 2 > 0.5).astype(float)


def natural_scores(gate, texts, prompts) -> np.ndarray:
    return np.array([x["p_natural"] for x in gate.score_naturalness_batch(texts, prompts, 32)])


def quality_scores(gate, prompts, texts) -> np.ndarray:
    return np.array([x["score"] for x in gate.score_quality_batch(prompts, texts, 32)])


def single_wins(gate, kind, prompts, ref, other) -> np.ndarray:
    if kind == "natural_swedish":
        return X.versus(natural_scores(gate, ref, prompts), natural_scores(gate, other, prompts))
    return X.versus(quality_scores(gate, prompts, ref), quality_scores(gate, prompts, other))


def test_suite() -> list[dict[str, Any]]:
    """Every held-out test as (name, kind, prompts, ref, other); kind is a question id."""
    tests = []

    def add(name, kind, prompts, ref, other, group):
        tests.append(
            {
                "name": name,
                "kind": kind,
                "prompts": prompts,
                "ref": ref,
                "other": other,
                "group": group,
            }
        )

    ev = v2.load_rows("val_eval")
    for src in ("original", "fluent"):
        rows = ev[src]
        ch = [r["candidate_a"] if r["label"] == 0 else r["candidate_b"] for r in rows]
        rj = [r["candidate_b"] if r["label"] == 0 else r["candidate_a"] for r in rows]
        add(f"{src}/preference", QID, [r["prompt"] for r in rows], ch, rj, "in-domain")
    probe = read_jsonl(PROBE_DIR / "probe_pairs.jsonl")
    add(
        "probe/preference",
        QID,
        [p["sv_prompt"] for p in probe],
        [p["sv_chosen"] for p in probe],
        [p["sv_rejected"] for p in probe],
        "in-domain",
    )

    liu = liu_eval.load_rows()
    mask = liu_eval.comparison_masks(liu)["opus"]
    liu = [r for r, m in zip(liu, mask, strict=True) if m]
    human = [liu_eval.strip_punct(r["swedish_human"]) for r in liu]
    opus = [liu_eval.strip_punct(r["swedish_opus"]) for r in liu]
    none = [""] * len(liu)
    ctx0 = [liu_eval.make_prompt(r["english_source"], r["context"], 0) for r in liu]
    add("liu/naturalness", NATURALNESS_QID, none, human, opus, "native vs MT")
    add("liu/natural_swedish", "natural_swedish", none, human, opus, "native vs MT")
    add("liu/preference+source", QID, ctx0, human, opus, "native vs MT")

    from .mtqa import PAIRS_PATH, PROMPT

    fl = [r for r in read_jsonl(PAIRS_PATH) if X._norm(r["mt"]) != X._norm(r["human"])]
    fl_prompts = [PROMPT + r["en"] for r in fl]
    fh, fm = [r["human"] for r in fl], [r["mt"] for r in fl]
    add("flores/naturalness", NATURALNESS_QID, [""] * len(fl), fh, fm, "professional vs MT")
    add("flores/natural_swedish", "natural_swedish", [""] * len(fl), fh, fm, "professional vs MT")
    add("flores/preference+source", QID, fl_prompts, fh, fm, "professional vs MT")

    sets = X.load_sets()
    d = sets["dalaj"]
    dr, do = [p["ref"] for p in d], [p["other"] for p in d]
    add("dalaj/naturalness", NATURALNESS_QID, [""] * len(d), dr, do, "grammar")
    add("dalaj/natural_swedish", "natural_swedish", [""] * len(d), dr, do, "grammar")
    f = sets["swefaq"]
    fp, fr, fo = [p["prompt"] for p in f], [p["ref"] for p in f], [p["other"] for p in f]
    add("swefaq/preference", QID, fp, fr, fo, "relevance")
    add("swefaq/quality", "quality", fp, fr, fo, "relevance")
    if "gerlangmod" in sets:
        g = sets["gerlangmod"]
        gr, go = [p["ref"] for p in g], [p["other"] for p in g]
        add("gerlangmod/naturalness", NATURALNESS_QID, [""] * len(g), gr, go, "grammar")
        add("gerlangmod/natural_swedish", "natural_swedish", [""] * len(g), gr, go, "grammar")
    for name in ("wmt24pp/opus", "wmt24pp/kimi"):
        if name in sets:
            w = sets[name]
            wr, wo = [p["ref"] for p in w], [p["other"] for p in w]
            add(f"{name}/preference+source", QID, [p["prompt"] for p in w], wr, wo, "WMT24++")
            add(f"{name}/naturalness", NATURALNESS_QID, [""] * len(w), wr, wo, "WMT24++")
    for name in ("europarl_en/opus", "europarl_en/kimi", "europarl_sv/opus", "europarl_sv/kimi"):
        e = sets[name]
        er, eo = [p["ref"] for p in e], [p["other"] for p in e]
        add(f"{name}/preference+source", QID, [p["prompt"] for p in e], er, eo, "Europarl")
        add(f"{name}/naturalness", NATURALNESS_QID, [""] * len(e), er, eo, "Europarl")
    return tests


def run_tests(gate) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    wins, guards, extra = {}, {}, {}
    for t in test_suite():
        if t["kind"] in (QID, NATURALNESS_QID):
            wins[t["name"]] = pair_wins(gate, t["kind"], t["prompts"], t["ref"], t["other"])
        else:
            wins[t["name"]] = single_wins(gate, t["kind"], t["prompts"], t["ref"], t["other"])
        print(f"  {t['name']}: {wins[t['name']].mean():.3f}", flush=True)
    # SweFAQ: rank every answer on the page by quality
    questions = read_jsonl(X.EXT_DIR / "swefaq_questions.jsonl")
    keys, qs, cs = [], [], []
    for q in questions:
        for j, c in enumerate(q["candidates"]):
            keys.append(f"{q['id']}|{j}")
            qs.append(q["question"])
            cs.append(c)
    scores = dict(zip(keys, quality_scores(gate, qs, cs), strict=True))
    top1 = []
    for q in questions:
        s = np.array([scores[f"{q['id']}|{j}"] for j in range(len(q["candidates"]))])
        top1.append(float((s >= s[q["gold"]]).sum() == 1))
    wins["swefaq/quality_top1"] = np.array(top1)
    # ScaLA-sv (EuroEval's split): natural_swedish as a correct/incorrect classifier
    scala_path = X.EXT_DIR / "scala_test.jsonl"
    if scala_path.exists():
        test, val = read_jsonl(scala_path), read_jsonl(X.EXT_DIR / "scala_val.jsonl")
        pt = natural_scores(gate, [r["text"] for r in test], [""] * len(test))
        pv = natural_scores(gate, [r["text"] for r in val], [""] * len(val))
        yt = np.array([r["correct"] for r in test])
        yv = np.array([r["correct"] for r in val])
        thr = max(np.unique(pv), key=lambda t: X.mcc(yv, pv >= t))
        wins["scala/accuracy@0.5"] = ((pt >= 0.5) == yt).astype(float)
        wins["scala/accuracy@val"] = ((pt >= thr) == yt).astype(float)
        extra["scala"] = {
            "mcc@0.5": X.mcc(yt, pt >= 0.5),
            "mcc@val_threshold": X.mcc(yt, pt >= thr),
            "val_threshold": float(thr),
            "auc": v3.roc_auc(pt[yt], pt[~yt]),
            "flagged_incorrect": float((pt < 0.5).mean()),
            "n": int(len(yt)),
        }
    # guard: natural Swedish responses wrongly flagged as translationese (lower is better)
    ev = v2.load_rows("val_eval")["fluent"]
    resp = [r["candidate_a"] for r in ev] + [r["candidate_b"] for r in ev]
    guards["fluent_flagged_translationese"] = (
        natural_scores(gate, resp, [r["prompt"] for r in ev] * 2) < 0.5
    ).astype(float)
    return wins, guards, extra


def summary(v: np.ndarray) -> dict[str, Any]:
    return {"mean": float(v.mean()), "ci95": bootstrap_ci(v), "n": int(v.size)}


def evaluate(arm: str, seed: int) -> None:
    t, path = tag(arm, seed), checkpoint_dir(arm, seed)
    # the standard v3 evaluation (ECE, margins, cascade, latency, single-response metrics)
    v3.evaluate(argparse.Namespace(checkpoint=path, section=f"_v4_{t}"))
    _move(f"_v4_{t}_eval", t, "standard_eval")
    _move(f"_v4_{t}_pointwise", t, "pointwise_eval")
    score_suite(path, t)


def score_suite(path: Path, t: str) -> None:
    """The held-out test suite for one checkpoint, stored per item and summarized."""
    from .predictor import SwedishPreferenceGate

    V4_DIR.mkdir(parents=True, exist_ok=True)
    gate = SwedishPreferenceGate(path)
    wins, guards, extra = run_tests(gate)
    np.savez(V4_DIR / f"wins_{t}.npz", **wins, **{f"guard:{k}": v for k, v in guards.items()})
    m = load_metrics()
    entry = m.setdefault("v4", {}).setdefault("arms", {}).setdefault(t, {})
    entry["tests"] = {k: summary(v) for k, v in wins.items()}
    entry["guards"] = {k: summary(v) for k, v in guards.items()}
    entry["classifiers"] = extra
    _write(m)


def calibrate(arm: str, seed: int) -> None:
    t = tag(arm, seed)
    v3.calibrate(argparse.Namespace(checkpoint=checkpoint_dir(arm, seed), section=f"_v4_cal_{t}"))
    _move(f"_v4_cal_{t}", t, "calibration")


def run(args) -> None:
    path = checkpoint_dir(args.arm, args.seed)
    if not (path / "model.safetensors").exists() or args.retrain:
        train(args.arm, args.seed)
    calibrate(args.arm, args.seed)
    evaluate(args.arm, args.seed)


# ---------------------------------------------------------------------- compare


def paired_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = SEED) -> dict:
    """Mean of a - b over the same items with a paired bootstrap 95% CI."""
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": float(d.mean()), "ci95": [float(lo), float(hi)], "n": int(len(d))}


def compare(args) -> None:
    files = sorted(V4_DIR.glob("wins_*.npz"))
    wins = {f.stem.removeprefix("wins_"): dict(np.load(f)) for f in files}
    if "control" not in wins:
        print("compare: control arm not evaluated yet")
        return
    base = wins["control"]
    out: dict[str, Any] = {
        "against": "control",
        "metric": "per-item difference in the share of items where the reference wins "
        "(guards: share flagged, lower is better); paired bootstrap over items",
        "arms": {},
    }
    for t, w in wins.items():
        if t == "control":
            continue
        out["arms"][t] = {k: paired_diff(w[k], base[k]) for k in base if k in w}
    _set_v4("comparison", out)
    for t, r in out["arms"].items():
        print(t)
        for k, v in r.items():
            sig = "*" if v["ci95"][0] > 0 or v["ci95"][1] < 0 else " "
            print(f"  {sig} {k:42s} {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["prepare", "run", "reference", "compare"])
    parser.add_argument("--arm", choices=sorted(ARMS), default="control")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--concurrency", type=int, default=64)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        _set_v4(
            "design",
            {
                "arms": {k: list(v) for k, v in ARMS.items()},
                "recipe": "v3 data and questions, 1 epoch, from the base checkpoint",
                "europarl_train_pairs_target": EP_TRAIN,
                "swefaq_negatives_per_question": FAQ_NEGATIVES,
                "near_duplicate_similarity": NEAR_DUPLICATE,
                "seed": SEED,
            },
        )
        run(args)
    elif args.command == "reference":
        score_suite(v3.V3_DIR, "v3")  # the published v3 (2 epochs), scored on the same suite
    else:
        compare(args)


if __name__ == "__main__":
    main()
