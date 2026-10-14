"""Fluent-vs-fluent training data for the v2 gate.

UltraFeedback pairs with a clear GPT-4 rating gap are rewritten into natural Swedish by
Kimi-k3, both answers with the same instructions (`probe.TRANSLATE_SYSTEM`), so the two sides
differ in quality, not in style. Generation runs with Kimi-k3's reasoning switched off
(about 4.6x fewer tokens; fidelity was checked against reasoning-on translations).

Pairs sampled for the probe are never used, and pairs whose Swedish prompt nearly duplicates
a probe prompt or a validation prompt of the original dataset are dropped.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import os
import random
import sys
import time
from typing import Any

import numpy as np

from .judge import Judge, discover_model, judge_correct, make_client, quiet_http_logs
from .probe import PROBE_DIR, eligible_pairs, sample_pairs, to_swedish, translate_all
from .utils import CACHE_DIR, SEED, SPLITS_DIR, read_jsonl, update_metrics, write_jsonl

FLUENT_DIR = CACHE_DIR / "fluent"
TRANSLATIONS = FLUENT_DIR / "translations_nothink.jsonl"
FIDELITY_TRANSLATIONS = FLUENT_DIR / "fidelity.jsonl"
NOTHINK = {"extra_body": {"chat_template_kwargs": {"thinking": False}}}
MODE = "nothink"
OVERLAP_THRESHOLD = 0.8


def sample_training_pairs(n: int) -> tuple[list[dict[str, Any]], int]:
    """`n` eligible UltraFeedback pairs, disjoint from the 300-pair probe sample."""
    probe_ids = {p["id"] for p in sample_pairs(300)[0]}
    rows = [r for r in eligible_pairs() if r["id"] not in probe_ids]
    rng = random.Random(SEED + 1)
    picked = rng.sample(rows, min(n, len(rows)))
    for p in picked:  # hide which answer is better from the translator
        p["chosen_first"] = rng.random() < 0.5
    return picked, len(rows)


def near_duplicates(prompts: list[str], references: list[str]) -> np.ndarray:
    """True where a prompt's char 3-5-gram TF-IDF cosine to any reference is >= threshold."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    vec.fit(references + prompts)
    sims = (vec.transform(prompts) @ vec.transform(references).T).max(axis=1)
    return np.asarray(sims.todense()).ravel() >= OVERLAP_THRESHOLD


def build_splits(pairs: list[dict[str, Any]], seed: int = SEED) -> dict[str, list[dict]]:
    """Symmetrize (chosen in slot B for exactly half), then stratified 85 / 7.5 / 7.5."""
    from sklearn.model_selection import train_test_split

    rng = random.Random(seed)
    flags = [i < len(pairs) // 2 for i in range(len(pairs))]
    rng.shuffle(flags)
    rows = []
    for p, swap in zip(pairs, flags, strict=True):
        a, b = (p["sv_rejected"], p["sv_chosen"]) if swap else (p["sv_chosen"], p["sv_rejected"])
        rows.append(
            {
                "id": p["id"],
                "prompt": p["sv_prompt"],
                "candidate_a": a,
                "candidate_b": b,
                "label": int(swap),
                "margin": p["margin"],
                "source": p["source"],
            }
        )
    labels = [r["label"] for r in rows]
    train, val = train_test_split(rows, test_size=0.15, stratify=labels, random_state=seed)
    calib, ev = train_test_split(
        val, test_size=0.5, stratify=[r["label"] for r in val], random_state=seed
    )
    return {"train": train, "val_calib": calib, "val_eval": ev}


async def generate(pairs, chunk: int, concurrencies: list[int]) -> tuple[list, list[dict]]:
    """Translate in chunks. The first chunks try each concurrency; the rest use the fastest."""
    client = make_client()
    model = await discover_model(client)
    rows, runs = [], []
    chunks = [pairs[i : i + chunk] for i in range(0, len(pairs), chunk)]
    for k, part in enumerate(chunks):
        if k < len(concurrencies):
            conc = concurrencies[k]
        else:
            measured = [r for r in runs if r["live"] >= 0.8 * chunk]
            conc = (
                max(measured, key=lambda r: r["requests_per_second"])["concurrency"]
                if measured
                else concurrencies[-1]
            )
        t = time.perf_counter()
        got, st = await translate_all(
            client,
            model,
            part,
            conc,
            cache_path=TRANSLATIONS,
            extra=NOTHINK,
            mode=MODE,
            progress_every=0,
        )
        wall = time.perf_counter() - t
        live = [g for g in got if g.get("usage") and g.get("latency_s")]
        run = {
            "chunk": k,
            "concurrency": conc,
            "requests": len(part),
            "live": st["live"],
            "failed": st["failed"],
            "parse_failed": st["parse_failed"],
            "wall_seconds": wall,
            "requests_per_second": st["live"] / wall if st["live"] else None,
            "output_tokens_per_second": sum(g["usage"].get("completion_tokens") or 0 for g in live)
            / wall
            if live
            else None,
            "latency_p50_s": float(np.median([g["latency_s"] for g in live])) if live else None,
        }
        runs.append(run)
        print(json.dumps(run), flush=True)
        rows.extend(got)
    return rows, runs, model


async def fidelity_check(n: int) -> dict[str, Any]:
    """Re-translate the first `n` probe pairs with reasoning off and compare to reasoning on.

    Both versions are judged by Kimi-k3 in both orders; if reasoning-off translations kept the
    quality gap, the judge should recover the UltraFeedback label about as often on both.
    """
    client = make_client()
    model = await discover_model(client)
    probe = read_jsonl(PROBE_DIR / "probe_pairs.jsonl")[:n]
    rows, st = await translate_all(
        client,
        model,
        probe,
        64,
        cache_path=FIDELITY_TRANSLATIONS,
        extra=NOTHINK,
        mode=MODE,
        progress_every=0,
    )
    by = {r["id"]: r.get("translation") for r in rows}
    pairs = [(p, to_swedish(p, by[p["id"]])) for p in probe if by.get(p["id"])]

    def sim(key):
        return float(
            np.mean([difflib.SequenceMatcher(None, a[key], b[key]).ratio() for a, b in pairs])
        )

    def len_ratio(key):
        return float(np.mean([len(b[key]) / len(a[key]) for a, b in pairs]))

    judge = Judge(client, model, concurrency=64)
    jobs = []
    for tag, idx in (("reasoning_on", 0), ("reasoning_off", 1)):
        for pair in pairs:
            p, key = pair[idx], f"{pair[idx]['id']}|{tag}"
            jobs.append((key, "AB", p["sv_prompt"], p["sv_chosen"], p["sv_rejected"]))
            jobs.append((key, "BA", p["sv_prompt"], p["sv_rejected"], p["sv_chosen"]))
    verdicts = await judge.run(jobs, progress_every=0)
    acc = {}
    for tag in ("reasoning_on", "reasoning_off"):
        # match by job, not by the returned pair_id: cache hits keep the id they were stored with
        oks = [
            judge_correct(v["verdict"], 0, job[1])
            for job, v in zip(jobs, verdicts, strict=True)
            if job[0].endswith("|" + tag)
        ]
        acc[tag] = float(np.mean([o for o in oks if o is not None]))
    tokens = [r["usage"]["completion_tokens"] for r in rows if r.get("usage")]
    return {
        "n_pairs": len(pairs),
        "translation_failures": st["failed"] + st["parse_failed"],
        "reasoning_off_completion_tokens_median": float(np.median(tokens)) if tokens else None,
        "mean_char_similarity_to_reasoning_on": {
            "chosen": sim("sv_chosen"),
            "rejected": sim("sv_rejected"),
        },
        "mean_length_ratio_off_over_on": {
            "chosen": len_ratio("sv_chosen"),
            "rejected": len_ratio("sv_rejected"),
        },
        "judge_accuracy_both_orders": acc,
        "judge": model,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=12000)
    parser.add_argument("--chunk", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[64, 128])
    parser.add_argument(
        "--fidelity", type=int, metavar="N", help="only run the reasoning-off fidelity check"
    )
    args = parser.parse_args(argv)
    quiet_http_logs()
    missing = [v for v in ("BASE_URL", "API_KEY") if not os.environ.get(v)]
    if missing:
        print(f"fluent data skipped: {missing[0]} not set")
        update_metrics("fluent_data", f"skipped: {missing[0]} not set")
        sys.exit(0)

    if args.fidelity:
        out = asyncio.run(fidelity_check(args.fidelity))
        update_metrics("fluent_fidelity", out)
        print(json.dumps(out, indent=2))
        return

    pairs, n_pool = sample_training_pairs(args.n)
    rows, runs, model = asyncio.run(generate(pairs, args.chunk, args.concurrency))
    by = {r["id"]: r.get("translation") for r in rows}
    done = [to_swedish(p, by[p["id"]]) for p in pairs if by.get(p["id"])]

    probe_prompts = [r["sv_prompt"] for r in read_jsonl(PROBE_DIR / "probe_pairs.jsonl")]
    val_prompts = [
        r["prompt"]
        for s in ("val_calib", "val_eval")
        for r in read_jsonl(SPLITS_DIR / f"{s}.jsonl")
    ]
    dup = near_duplicates([p["sv_prompt"] for p in done], probe_prompts + val_prompts)
    kept = [p for p, d in zip(done, dup, strict=True) if not d]
    splits = build_splits(kept)
    for name, part in splits.items():
        write_jsonl(FLUENT_DIR / f"{name}.jsonl", part)

    out = {
        "source": "argilla/ultrafeedback-binarized-preferences-cleaned",
        "translator": f"{model}, reasoning off, prompt = probe.TRANSLATE_SYSTEM",
        "eligible_pool_excluding_probe": n_pool,
        "requested": len(pairs),
        "translated": len(done),
        "translation_failures": len(pairs) - len(done),
        "dropped_near_duplicate_prompts": int(dup.sum()),
        "kept": len(kept),
        "splits": {k: len(v) for k, v in splits.items()},
        "label_balance_frac_chosen_in_b": {
            k: float(np.mean([r["label"] for r in v])) for k, v in splits.items()
        },
        "margin_median": float(np.median([p["margin"] for p in kept])),
        "throughput_runs": runs,
    }
    update_metrics("fluent_data", out)
    print(json.dumps({k: v for k, v in out.items() if k != "throughput_runs"}, indent=2))


if __name__ == "__main__":
    main()
