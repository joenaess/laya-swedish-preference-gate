"""Evaluate the gates on human-annotated English-to-Swedish translationese minimal pairs.

    python -m laya_gate.liu_eval gate     # v1/v2/v3, every question, the paper's prompt setups
    python -m laya_gate.liu_eval judge    # Kimi-k3 as a naturalness judge (needs BASE_URL, API_KEY)

Data: `liu-nlp/translationese-opensubtitles` (Kunz, Jarochenko & Bollmann, "A Dataset for
Probing Translationese Preferences in English-to-Swedish Translation", arXiv:2603.08450;
Apache-2.0). 600 OpenSubtitles lines, each with an OPUS-MT translation, a GPT translation, a
human-written idiomatic alternative, error tags for the OPUS translation, a flag for whether
the GPT translation was judged as acceptable as the human one, and 10 preceding English lines.
No Kimi-k3 model touched this data.

Setups follow the paper's `translationese_eval.py`: `no_prompt` scores the Swedish sentences
alone; `context_N` puts the paper's translation instruction with N preceding English lines in
the state's `prompt` field. Trailing punctuation is stripped from every candidate, as in the
paper (the human alternatives often lack the final period the MT output has); an unstripped
`no_prompt` run is kept as a sensitivity check. "Accuracy" is the share of pairs where the
human alternative is preferred, the paper's metric, so its LLM results are directly comparable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import string
import sys
import unicodedata
import urllib.request
from typing import Any

import numpy as np

from .probe import summarize
from .utils import CACHE_DIR, ROOT, update_metrics

DATASET = "liu-nlp/translationese-opensubtitles"
PAPER = "arXiv:2603.08450 (Kunz, Jarochenko & Bollmann)"
PAPER_RESULTS_URL = (
    "https://raw.githubusercontent.com/jekunz/translationese/main/translationese_eval_results.json"
)
LIU_DIR = CACHE_DIR / "liu"
CONTEXT_LEVELS: list[int | None] = [None, 0, 1, 2, 3, 4, 5, 10]
TAG_SETUPS = ("no_prompt", "context_10")  # the paper's Figure 3 setups
OTHERS = ("opus", "gpt")


def setup_name(n_ctx: int | None) -> str:
    return "no_prompt" if n_ctx is None else f"context_{n_ctx}"


def strip_punct(text: str) -> str:
    """The paper's normalization: drop trailing punctuation and spaces."""
    return text.strip().rstrip(string.punctuation + " ").strip()


def make_prompt(english: str, context: list[str], n_ctx: int | None) -> str:
    """The paper's translation instruction, without its trailing "Översättning: " cue (the
    candidate goes in its own slot of the Laya state)."""
    if n_ctx is None:
        return ""
    if n_ctx == 0:
        return f"Översätt följande mening till svenska:\n{english}"
    ctx = " ".join(context[-n_ctx:])
    return (
        "Översätt följande mening till svenska med hänsyn till kontexten:\n"
        f"Kontext: {ctx}\nMening: {english}"
    )


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split()).casefold()


def load_rows() -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="train")
    return [{"id": i, **r} for i, r in enumerate(ds)]


def comparison_masks(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Which pairs enter each comparison: identical candidates (after stripping) are dropped;
    `gpt_unacceptable` keeps the pairs where annotators judged GPT worse than the human."""
    h = [_norm(strip_punct(r["swedish_human"])) for r in rows]
    o = [_norm(strip_punct(r["swedish_opus"])) for r in rows]
    g = [_norm(strip_punct(r["swedish_gpt"])) for r in rows]
    diff_o = np.array([a != b for a, b in zip(h, o, strict=True)])
    diff_g = np.array([a != b for a, b in zip(h, g, strict=True)])
    unacc = np.array([not r["gpt_acceptable"] for r in rows])
    return {"opus": diff_o, "gpt": diff_g, "gpt_unacceptable": diff_g & unacc}


def summarize_pref(p_human: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Share of pairs where the human alternative wins (P > 0.5, or a strictly higher score)."""
    p = p_human[mask]
    return {**summarize(p > 0.5), "mean_p_human": float(p.mean()) if len(p) else None}


def tag_breakdown(rows, p_human_vs_opus: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Per OPUS error tag: share of pairs where OPUS is preferred (the paper's Figure 3)."""
    out: dict[str, Any] = {}
    for tag in sorted({t for r in rows for t in r["error_tags"]}):
        idx = np.array([tag in r["error_tags"] for r in rows]) & mask
        if idx.any():
            out[tag] = {
                "n": int(idx.sum()),
                "opus_preferred": float((p_human_vs_opus[idx] <= 0.5).mean()),
            }
    return out


def length_baseline(rows, masks) -> dict[str, Any]:
    """Trivial baseline: the longer candidate (characters, stripped) is the human one."""
    hl = np.array([len(strip_punct(r["swedish_human"])) for r in rows])
    out = {}
    for comp, other in (("opus", "opus"), ("gpt", "gpt"), ("gpt_unacceptable", "gpt")):
        ol = np.array([len(strip_punct(r[f"swedish_{other}"])) for r in rows])
        m = masks[comp]
        score = np.where(hl > ol, 1.0, np.where(hl < ol, 0.0, 0.5))[m]
        out[comp] = {"accuracy_longer_wins": float(score.mean()), "n": int(m.sum())}
    return out


def punct_stats(rows) -> dict[str, float]:
    def ends(key):
        return float(np.mean([r[key].strip()[-1:] in string.punctuation for r in rows]))

    return {k: ends(f"swedish_{k}") for k in ("human", "opus", "gpt")}


# -------------------------------------------------------------------------- gate


def _pairwise(gate, qid, prompts, human, other) -> np.ndarray:
    """P(human wins), averaged over both slot orders."""
    fwd = [gate.encode(p, h, o, qid) for p, h, o in zip(prompts, human, other, strict=True)]
    rev = [gate.encode(p, o, h, qid) for p, h, o in zip(prompts, human, other, strict=True)]
    pf = gate.probabilities(gate.logits(fwd, 32), qid)[:, 0]
    pr = 1 - gate.probabilities(gate.logits(rev, 32), qid)[:, 0]
    return (pf + pr) / 2


def _versus(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Single-response scores as a pseudo-probability: 1 if a > b, 0 if a < b, 0.5 on ties."""
    return np.where(a > b, 1.0, np.where(a < b, 0.0, 0.5))


def score_gate(gate, rows, n_ctx: int | None, strip: bool = True) -> dict[str, dict]:
    """P(human preferred) vs each MT system, per question, for one prompt setup."""
    f = strip_punct if strip else str.strip
    prompts = [make_prompt(r["english_source"], r["context"], n_ctx) for r in rows]
    texts = {k: [f(r[f"swedish_{k}"]) for r in rows] for k in ("human", *OTHERS)}
    out: dict[str, dict] = {}
    qids = ["preference"] + (["naturalness"] if gate.has_naturalness else [])
    for qid in qids:
        out[qid] = {o: _pairwise(gate, qid, prompts, texts["human"], texts[o]) for o in OTHERS}
    if gate.has_pointwise:
        nat = {
            k: np.array([x["p_natural"] for x in gate.score_naturalness_batch(v, prompts, 32)])
            for k, v in texts.items()
        }
        qual = {
            k: np.array([x["score"] for x in gate.score_quality_batch(prompts, v, 32)])
            for k, v in texts.items()
        }
        out["natural_swedish"] = {o: _versus(nat["human"], nat[o]) for o in OTHERS}
        out["quality"] = {o: _versus(qual["human"], qual[o]) for o in OTHERS}
        out["natural_swedish_p"] = nat  # kept for flag rates
    return out


def gate_eval(args) -> None:
    from .predictor import SwedishPreferenceGate
    from .v3 import V3_DIR

    rows = load_rows()
    masks = comparison_masks(rows)
    out: dict[str, Any] = {
        "dataset": DATASET,
        "paper": PAPER,
        "n_pairs": len(rows),
        "n_compared": {k: int(v.sum()) for k, v in masks.items()},
        "metric": "share of pairs where the human alternative is preferred (order-averaged "
        "P > 0.5 for pairwise questions; strictly higher score for single-response questions, "
        "ties count 0.5); trailing punctuation stripped as in the paper",
        "candidates_ending_in_punctuation": punct_stats(rows),
        "length_baseline": length_baseline(rows, masks),
        "models": {},
    }
    checkpoints = {
        "v1": ROOT / "checkpoints" / "best_model",
        "v2": ROOT / "checkpoints" / "v2",
        "v3": V3_DIR,
    }
    for name, path in checkpoints.items():
        if not (path / "model.safetensors").exists():
            out["models"][name] = "not measured: checkpoint missing"
            continue
        gate = SwedishPreferenceGate(path)
        res: dict[str, Any] = {"setups": {}, "error_tags": {}, "unstripped_no_prompt": {}}
        for n_ctx in CONTEXT_LEVELS:
            s = setup_name(n_ctx)
            scores = score_gate(gate, rows, n_ctx)
            nat_p = scores.pop("natural_swedish_p", None)
            for qid, by_other in scores.items():
                q = res["setups"].setdefault(qid, {})
                q[s] = {
                    comp: summarize_pref(by_other["gpt" if comp != "opus" else "opus"], m)
                    for comp, m in masks.items()
                }
                if s in TAG_SETUPS:
                    res["error_tags"].setdefault(qid, {})[s] = tag_breakdown(
                        rows, by_other["opus"], masks["opus"]
                    )
            if nat_p is not None and s == "no_prompt":
                res["natural_swedish_flagged_translationese"] = {
                    k: float((v < 0.5).mean()) for k, v in nat_p.items()
                }
            print(f"{name} {s} done", flush=True)
        raw = score_gate(gate, rows, None, strip=False)
        raw.pop("natural_swedish_p", None)
        for qid, by_other in raw.items():
            res["unstripped_no_prompt"][qid] = {
                comp: summarize_pref(by_other["gpt" if comp != "opus" else "opus"], m)
                for comp, m in masks.items()
            }
        out["models"][name] = res
        del gate
    out["paper_reference"] = paper_reference()
    update_metrics("liu_translationese", {**_existing_judge(), **out})
    print(json.dumps(_headline(out), indent=2))


def _existing_judge() -> dict[str, Any]:
    """Keep a previous judge run when the gate evaluation is rerun."""
    from .utils import load_metrics

    prev = load_metrics().get("liu_translationese") or {}
    return {"judge": prev["judge"]} if "judge" in prev else {}


def _headline(out: dict[str, Any]) -> dict[str, Any]:
    head = {}
    for name, res in out["models"].items():
        if isinstance(res, dict):
            for qid, setups in res["setups"].items():
                for s in ("no_prompt", "context_0", "context_10"):
                    head[f"{name}/{qid}/{s}"] = {
                        c: round(v["accuracy"], 3) for c, v in setups[s].items()
                    }
    return head


def paper_reference() -> dict[str, Any]:
    """The paper's published LLM results (its GitHub JSON): per setup, the best of its models
    and the AI Sweden Llama-3-8B base model, for Human > OPUS and Human > GPT."""
    try:
        with urllib.request.urlopen(PAPER_RESULTS_URL, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - reference numbers are optional
        return {
            "status": f"not measured: could not fetch the paper's results ({type(exc).__name__})"
        }
    out: dict[str, Any] = {"source": PAPER_RESULTS_URL, "n_models": len(data), "setups": {}}
    for n_ctx in CONTEXT_LEVELS:
        s = setup_name(n_ctx)
        row: dict[str, Any] = {}
        for comp in OTHERS:
            key = f"accuracy {comp}"
            scored = [(m, r[s][key]) for m, r in data.items() if s in r and key in r[s]]
            if scored:
                best = max(scored, key=lambda x: x[1])
                row[comp] = {
                    "best_model": best[0],
                    "best_accuracy": float(best[1]),
                    "median_accuracy": float(np.median([x[1] for x in scored])),
                    "ai_sweden_llama3_8b": data.get("AI-Sweden-Models/Llama-3-8B", {})
                    .get(s, {})
                    .get(key),
                }
        out["setups"][s] = row
    return out


# ------------------------------------------------------------------------- judge

JUDGE_SYSTEM = {
    "no_prompt": (
        "You are an expert in the Swedish language. You will be given two Swedish versions, A "
        "and B, of the same sentence. Decide which version is more natural and idiomatic "
        "Swedish, the way a native speaker would say it. Do not let the order of the versions "
        "or their length by itself influence your decision.\n\n"
        'Reply with JSON only, exactly one of: {"preference": "A"} or {"preference": "B"}'
    ),
    "context_10": (
        "You are an expert English-to-Swedish translator. You will be given an English "
        "sentence from a film or TV subtitle, the subtitle lines that precede it, and two "
        "Swedish translations, A and B. Decide which translation is more natural and idiomatic "
        "Swedish while conveying the meaning of the sentence in its context. Do not let the "
        "order of the translations or their length by itself influence your decision.\n\n"
        'Reply with JSON only, exactly one of: {"preference": "A"} or {"preference": "B"}'
    ),
}


def judge_user_message(setup: str, row: dict[str, Any], a: str, b: str) -> str:
    if setup == "no_prompt":
        return f"## Version A\n{a}\n\n## Version B\n{b}"
    ctx = "\n".join(row["context"][-10:])
    return (
        f"## Preceding subtitle lines (English)\n{ctx}\n\n## English sentence\n"
        f"{row['english_source']}\n\n## Translation A\n{a}\n\n## Translation B\n{b}"
    )


def judge_scores(verdicts: dict[tuple[int, str, str], str | None], n: int, other: str, setup):
    """Per pair: share of the two orders in which the human version won (0, 0.5 or 1), or NaN
    when either order failed to parse; plus per-order consistency."""
    score, consistent = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        hab = verdicts.get((i, other, setup + "|HA"))  # human in slot A
        hba = verdicts.get((i, other, setup + "|AH"))  # human in slot B
        if hab is None or hba is None:
            continue
        wins = (hab == "A") + (hba == "B")
        score[i] = wins / 2
        consistent[i] = float(wins != 1)
    return score, consistent


async def _judge_run(args) -> dict[str, Any]:
    from .judge import Judge, discover_model, make_client, quiet_http_logs

    class NaturalnessJudge(Judge):
        async def _call(self, prompt: str, a: str, b: str):
            setup, _, body = prompt.partition("\n")
            return await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM[setup]},
                        {"role": "user", "content": body},
                    ],
                    temperature=0,
                    max_tokens=self.max_tokens,
                ),
                timeout=self.timeout,
            )

    quiet_http_logs()
    rows = load_rows()
    client = make_client()
    model = await discover_model(client)
    judge = NaturalnessJudge(
        client, model, cache_path=LIU_DIR / "judge_cache.jsonl", concurrency=args.concurrency
    )
    jobs, keys = [], []
    for setup in JUDGE_SYSTEM:
        for r in rows:
            h = strip_punct(r["swedish_human"])
            for other in OTHERS:
                o = strip_punct(r[f"swedish_{other}"])
                for order, (a, b) in (("HA", (h, o)), ("AH", (o, h))):
                    # the setup rides in the first line of `prompt` so it is part of the cache key
                    prompt = f"{setup}\n{judge_user_message(setup, r, a, b)}"
                    jobs.append((f"{r['id']}|{other}|{setup}", order, prompt, a, b))
                    keys.append((r["id"], other, f"{setup}|{order}"))
    results = await judge.run(jobs, progress_every=200)
    verdicts = {k: v["verdict"] for k, v in zip(keys, results, strict=True)}
    masks = comparison_masks(rows)
    lat = [v["latency_s"] for v in results if v.get("latency_s") is not None]
    toks = [
        v["usage"]["completion_tokens"]
        for v in results
        if (v.get("usage") or {}).get("completion_tokens")
    ]
    out: dict[str, Any] = {
        "model": model,
        "hardware": "self-hosted Kimi-k3 on 8× NVIDIA B300 (HGX)",
        "requests": len(jobs),
        "live_requests": judge.live_requests,
        "failed_requests": judge.failures,
        "parse_failures": int(sum(v["verdict"] is None for v in results)),
        "latency_s_median": float(np.median(lat)) if lat else None,
        "completion_tokens_median": float(np.median(toks)) if toks else None,
        "metric": "per pair, the share of the two slot orders in which the human version won; "
        "a pair counts as human-preferred when both orders pick it (0.5 = split)",
        "setups": {},
    }
    for setup in JUDGE_SYSTEM:
        res = {}
        for comp, m in masks.items():
            other = "gpt" if comp != "opus" else "opus"
            score, cons = judge_scores(verdicts, len(rows), other, setup)
            ok = m & ~np.isnan(score)
            res[comp] = {
                "human_preferred_both_orders": float((score[ok] == 1).mean()),
                "mt_preferred_both_orders": float((score[ok] == 0).mean()),
                "mean_score": float(score[ok].mean()),
                "positional_consistency": float(cons[ok].mean()),
                "n": int(ok.sum()),
            }
            if comp == "opus":
                tags = {}
                for tag in sorted({t for r in rows for t in r["error_tags"]}):
                    idx = np.array([tag in r["error_tags"] for r in rows]) & ok
                    if idx.any():
                        tags[tag] = {
                            "n": int(idx.sum()),
                            "opus_preferred": float((score[idx] < 0.5).mean()),
                        }
                res["error_tags"] = tags
        out["setups"][setup] = res
    return out


def judge_eval(args) -> None:
    missing = [v for v in ("BASE_URL", "API_KEY") if not os.environ.get(v)]
    if missing:
        print(f"judge skipped: {missing[0]} not set")
        sys.exit(0)
    from .utils import load_metrics

    res = asyncio.run(_judge_run(args))
    prev = load_metrics().get("liu_translationese") or {}
    update_metrics("liu_translationese", {**prev, "judge": res})
    print(json.dumps({k: v for k, v in res.items() if k != "setups"}, indent=2))
    for s, r in res["setups"].items():
        print(
            s,
            {
                c: round(v["human_preferred_both_orders"], 3)
                for c, v in r.items()
                if c != "error_tags"
            },
        )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["gate", "judge"])
    parser.add_argument("--concurrency", type=int, default=64)
    args = parser.parse_args(argv)
    {"gate": gate_eval, "judge": judge_eval}[args.command](args)


if __name__ == "__main__":
    main()
