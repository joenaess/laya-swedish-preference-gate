"""Fluent-vs-fluent probe: can the gate rank two answers that are *both* natural Swedish?

The training data contrasts idiomatic Swedish (chosen) with deliberately poor translation
(rejected), so held-out accuracy there mostly measures translationese detection. This probe
holds style constant: original UltraFeedback pairs with a clear GPT-4 rating gap are rewritten
by Kimi-k3 into natural Swedish, *both* answers with the same instructions, preserving their
content (including mistakes). The label is the original rating: chosen = higher-rated answer.

Scorers: the gate (both orders), Kimi-k3 as judge (both orders), the TF-IDF style baseline
trained on the gate's training split, and "longer answer wins".
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import time
from typing import Any

import numpy as np

from .judge import (
    Judge,
    JudgeError,
    discover_model,
    is_retryable,
    judge_correct,
    make_client,
    quiet_http_logs,
    sanitize,
)
from .utils import (
    CACHE_DIR,
    CHECKPOINT_DIR,
    DEFAULT_MARGIN,
    SEED,
    SPLITS_DIR,
    read_jsonl,
    update_metrics,
)

SOURCE = "argilla/ultrafeedback-binarized-preferences-cleaned"
PROBE_DIR = CACHE_DIR / "probe"
TRANSLATIONS = PROBE_DIR / "translations.jsonl"
MIN_MARGIN = 1.0
MAX_CHARS = 3500

TRANSLATE_SYSTEM = (
    "You are a professional English-to-Swedish translator and localizer. You will receive a "
    "user prompt and two assistant responses as JSON. Rewrite all three in natural, idiomatic "
    "Swedish, as a skilled native Swedish writer would phrase them. Avoid literal translation, "
    "anglicisms and translationese in BOTH responses equally.\n\n"
    "Strict fidelity rules:\n"
    "- Preserve each response's content exactly: the same facts, the same mistakes and "
    "inaccuracies, the same omissions, the same reasoning, the same level of detail.\n"
    "- Do not fix, improve, shorten, extend or weaken either response. A weak response must "
    "stay exactly as weak in substance; only its language should become natural Swedish.\n"
    "- Keep formatting (lists, headings, code blocks), numbers, names and code unchanged; "
    "translate comments and prose only.\n"
    "- Do not adapt facts to Sweden (no changing units, places or institutions).\n\n"
    'Reply with JSON only: {"prompt": "...", "response_1": "...", "response_2": "..."}'
)


# ------------------------------------------------------------------- sampling


def _content(messages) -> str:
    """The assistant turn of an UltraFeedback chat-message list."""
    for m in reversed(messages):
        if m["role"] == "assistant":
            return m["content"]
    return ""


def sample_pairs(n: int, seed: int = SEED) -> tuple[list[dict[str, Any]], int]:
    from datasets import load_dataset

    ds = load_dataset(SOURCE, split="train")
    rows = []
    for i, r in enumerate(ds):
        chosen, rejected = _content(r["chosen"]).strip(), _content(r["rejected"]).strip()
        margin = r["chosen-rating"] - r["rejected-rating"]
        total = len(r["prompt"]) + len(chosen) + len(rejected)
        if (
            margin >= MIN_MARGIN
            and total <= MAX_CHARS
            and chosen
            and rejected
            and chosen != rejected
            and r["prompt"].strip()
        ):
            rows.append(
                {
                    "id": f"uf-{i}",
                    "source": r["source"],
                    "prompt": r["prompt"].strip(),
                    "chosen": chosen,
                    "rejected": rejected,
                    "margin": float(margin),
                    "chosen_rating": float(r["chosen-rating"]),
                    "rejected_rating": float(r["rejected-rating"]),
                }
            )
    rng = random.Random(seed)
    picked = rng.sample(rows, n)
    for p in picked:  # hide which answer is better from the translator
        p["chosen_first"] = rng.random() < 0.5
    return picked, len(rows)


# ---------------------------------------------------------------- translation


def _parse_translation(content: str | None) -> dict[str, str] | None:
    if not content:
        return None
    text = content.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    keys = ("prompt", "response_1", "response_2")
    if not all(isinstance(obj.get(k), str) and obj[k].strip() for k in keys):
        return None
    return {k: obj[k].strip() for k in keys}


def _key(model: str, pair: dict[str, Any]) -> str:
    payload = json.dumps([model, TRANSLATE_SYSTEM, pair["id"], pair["chosen_first"]])
    return hashlib.sha256(payload.encode()).hexdigest()


async def translate_all(client, model: str, pairs, concurrency: int, max_retries: int = 3):
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    cache = {r["key"]: r for r in read_jsonl(TRANSLATIONS)} if TRANSLATIONS.exists() else {}
    sem, lock = asyncio.Semaphore(concurrency), asyncio.Lock()
    stats = {"live": 0, "failed": 0, "parse_failed": 0, "done": 0}

    async def one(p):
        key = _key(model, p)
        if key in cache and cache[key].get("translation"):
            return cache[key]
        first, second = (
            (p["chosen"], p["rejected"]) if p["chosen_first"] else (p["rejected"], p["chosen"])
        )
        user = json.dumps(
            {"prompt": p["prompt"], "response_1": first, "response_2": second}, ensure_ascii=False
        )
        async with sem:
            for attempt in range(max_retries + 1):
                t = time.perf_counter()
                try:
                    resp = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=model,
                            temperature=0,
                            max_tokens=16384,
                            messages=[
                                {"role": "system", "content": TRANSLATE_SYSTEM},
                                {"role": "user", "content": user},
                            ],
                        ),
                        timeout=900,
                    )
                    break
                except Exception as exc:  # noqa: BLE001 - sanitized below
                    if attempt < max_retries and is_retryable(exc):
                        await asyncio.sleep(2.0 * 2**attempt)
                        continue
                    stats["failed"] += 1
                    return {
                        "key": key,
                        "id": p["id"],
                        "translation": None,
                        "error": sanitize(f"{type(exc).__name__}"),
                    }
        stats["live"] += 1
        content = resp.choices[0].message.content
        tr = _parse_translation(content)
        if tr is None:
            stats["parse_failed"] += 1
        row = {
            "key": key,
            "id": p["id"],
            "translation": tr,
            "latency_s": time.perf_counter() - t,
            "finish_reason": resp.choices[0].finish_reason,
            "usage": resp.usage.model_dump() if resp.usage else {},
        }
        async with lock:
            with open(TRANSLATIONS, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats["done"] += 1
        if stats["done"] % 25 == 0:
            print(f"translate: {stats['done']} live done", flush=True)
        return row

    rows = await asyncio.gather(*(one(p) for p in pairs))
    return rows, stats


def to_swedish(pair: dict[str, Any], tr: dict[str, str]) -> dict[str, Any]:
    """Map response_1/response_2 back to chosen/rejected."""
    c, r = ("response_1", "response_2") if pair["chosen_first"] else ("response_2", "response_1")
    return {**pair, "sv_prompt": tr["prompt"], "sv_chosen": tr[c], "sv_rejected": tr[r]}


# ------------------------------------------------------------------- scoring


def bootstrap_ci(correct: np.ndarray, n_boot: int = 2000, seed: int = SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct), (n_boot, len(correct)))
    means = correct[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(correct) -> dict[str, Any]:
    arr = np.asarray(correct, dtype=float)
    return {"accuracy": float(arr.mean()), "ci95": bootstrap_ci(arr), "n": int(arr.size)}


def overlap_flags(probe_prompts: list[str], threshold: float = 0.8) -> tuple[np.ndarray, list]:
    """Character n-gram TF-IDF cosine to the nearest training prompt (Swedish vs Swedish)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    train_prompts = [r["prompt"] for r in read_jsonl(SPLITS_DIR / "train.jsonl")]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    t = vec.fit_transform(train_prompts)
    q = vec.transform(probe_prompts)
    sims = np.asarray((q @ t.T).max(axis=1).todense()).ravel()
    return sims >= threshold, sims.tolist()


async def amain(args) -> int:
    quiet_http_logs()
    missing = [v for v in ("BASE_URL", "API_KEY") if not os.environ.get(v)]
    if missing:
        msg = f"skipped: {missing[0]} not set"
        print(f"probe {msg}")
        update_metrics("probe", msg)
        return 0

    pairs, n_eligible = sample_pairs(args.n)
    client = make_client()
    model = await discover_model(client)
    t0 = time.perf_counter()
    rows, tstats = await translate_all(client, model, pairs, args.concurrency)
    print(f"translation: {tstats} in {time.perf_counter() - t0:.0f}s", flush=True)
    by_id = {r["id"]: r for r in rows}
    probe = [
        to_swedish(p, by_id[p["id"]]["translation"])
        for p in pairs
        if by_id[p["id"]].get("translation")
    ]
    with open(PROBE_DIR / "probe_pairs.jsonl", "w", encoding="utf-8") as f:
        for p in probe:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # --- gate, both orders ------------------------------------------------------
    from .data import fit_tfidf_baseline
    from .predictor import SwedishPreferenceGate

    gate = SwedishPreferenceGate(CHECKPOINT_DIR)
    fwd = [gate.encode(p["sv_prompt"], p["sv_chosen"], p["sv_rejected"]) for p in probe]
    rev = [gate.encode(p["sv_prompt"], p["sv_rejected"], p["sv_chosen"]) for p in probe]
    p_fwd = gate.probabilities(gate.logits(fwd, batch_size=8))[:, 0]  # P(chosen), chosen in A
    p_rev = 1 - gate.probabilities(gate.logits(rev, batch_size=8))[:, 0]  # P(chosen), chosen in B
    p_avg = (p_fwd + p_rev) / 2
    truncated = float(np.mean([it["truncated"] for it in fwd]))
    conf = np.maximum(p_avg, 1 - p_avg)
    del gate

    # --- baselines ----------------------------------------------------------------
    predict_b = fit_tfidf_baseline(read_jsonl(SPLITS_DIR / "train.jsonl"))
    tfidf_ok = np.array(
        [not predict_b(p["sv_chosen"], p["sv_rejected"]) for p in probe]
    )  # candidate_a = chosen, so "B better" is wrong
    longer_ok = np.array([len(p["sv_chosen"]) > len(p["sv_rejected"]) for p in probe])
    longer_ok_en = np.array([len(p["chosen"]) > len(p["rejected"]) for p in probe])
    len_ratio_en = np.log([len(p["chosen"]) / len(p["rejected"]) for p in probe])
    len_ratio_sv = np.log([len(p["sv_chosen"]) / len(p["sv_rejected"]) for p in probe])

    # --- Kimi-k3 as judge, both orders -------------------------------------------
    judge = Judge(client, model, concurrency=args.concurrency)
    jobs = []
    for p in probe:
        jobs.append((p["id"], "AB", p["sv_prompt"], p["sv_chosen"], p["sv_rejected"]))
        jobs.append((p["id"], "BA", p["sv_prompt"], p["sv_rejected"], p["sv_chosen"]))
    verdicts = await judge.run(jobs)
    vmap = {(v["pair_id"], v["order"]): v.get("verdict") for v in verdicts}
    # label 0: chosen is the stored candidate_a (AB order shows chosen first)
    j_ab = [judge_correct(vmap[(p["id"], "AB")], 0, "AB") for p in probe]
    j_ba = [judge_correct(vmap[(p["id"], "BA")], 0, "BA") for p in probe]
    j_inst = [x for x in j_ab + j_ba if x is not None]
    j_pair_both = [a for a, b in zip(j_ab, j_ba, strict=True) if a is not None and a == b]

    flags, sims = overlap_flags([p["sv_prompt"] for p in probe])
    gate_ok = p_avg > 0.5
    margins = np.array([p["margin"] for p in probe])

    def by_mask(mask):
        mask = np.asarray(mask)
        return summarize(gate_ok[mask]) if mask.any() else None

    from .calibrate import ece

    probe_ece, _ = ece(np.stack([1 - p_avg, p_avg], axis=1), np.ones(len(probe), dtype=int))
    margin_ok = np.abs(p_avg - 0.5) >= DEFAULT_MARGIN
    conf_bins = []
    for lo, hi in ((0.5, 0.7), (0.7, 0.9), (0.9, 1.01)):
        m = (conf >= lo) & (conf < hi)
        conf_bins.append(
            {
                "confidence": [lo, min(hi, 1.0)],
                "fraction": float(m.mean()),
                "accuracy": float(gate_ok[m].mean()) if m.any() else None,
            }
        )

    out = {
        "source": SOURCE,
        "design": (
            f"{len(probe)} UltraFeedback pairs (seed {SEED}, GPT-4 rating margin >= {MIN_MARGIN}, "
            f"<= {MAX_CHARS} English chars) rewritten into natural Swedish by kimi-k3 with "
            "identical instructions for both answers, content and errors preserved; label = "
            "higher-rated answer"
        ),
        "n_requested": args.n,
        "n_eligible_pool": n_eligible,
        "n_pairs": len(probe),
        "translation": {**tstats, "model": model},
        "fraction_truncated_for_gate": truncated,
        "length_ratio_correlation_en_sv": float(np.corrcoef(len_ratio_en, len_ratio_sv)[0, 1]),
        "gate": {
            "order_averaged": summarize(gate_ok),
            "chosen_in_a": summarize(p_fwd > 0.5),
            "chosen_in_b": summarize(p_rev > 0.5),
            "positional_consistency": float(np.mean((p_fwd > 0.5) == (p_rev > 0.5))),
            "mean_confidence": float(conf.mean()),
            "ece_order_averaged": probe_ece,
            "default_margin": {
                "margin": DEFAULT_MARGIN,
                "coverage": float(margin_ok.mean()),
                "accuracy_on_covered": float(gate_ok[margin_ok].mean())
                if margin_ok.any()
                else None,
                "accuracy_on_routed": float(gate_ok[~margin_ok].mean())
                if (~margin_ok).any()
                else None,
            },
            "by_confidence": conf_bins,
            "by_margin": {
                "1.0-<2.0": by_mask((margins >= 1.0) & (margins < 2.0)),
                ">=2.0": by_mask(margins >= 2.0),
            },
            "excluding_likely_training_overlap": by_mask(~flags),
        },
        "judge": {
            "order_averaged": summarize(j_inst),
            "both_orders_correct_fraction": float(
                np.mean([a is True and b is True for a, b in zip(j_ab, j_ba, strict=True)])
            ),
            "positional_consistency": float(len(j_pair_both) / len(probe)),
            "parse_failures": int(sum(v is None for v in j_ab + j_ba)),
        },
        "baselines": {
            "tfidf_style_model": summarize(tfidf_ok),
            "longer_wins_swedish": summarize(longer_ok),
            "longer_wins_english_original": summarize(longer_ok_en),
        },
        "gate_judge_agreement": float(
            np.mean(
                [
                    (bool(g) == bool(j))
                    for g, j in zip(np.r_[p_fwd > 0.5, p_rev > 0.5], j_ab + j_ba, strict=True)
                    if j is not None
                ]
            )
        ),
        "likely_training_overlap": {
            "rule": "char 3-5-gram TF-IDF cosine >= 0.8 to the nearest training prompt",
            "n_flagged": int(flags.sum()),
            "max_similarity_median": float(np.median(sims)),
        },
    }
    update_metrics("probe", out)
    print(json.dumps(out, indent=2))
    return 0


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args(argv)
    try:
        code = asyncio.run(amain(args))
    except JudgeError as exc:
        print(f"probe failed: {exc}", file=sys.stderr)
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
