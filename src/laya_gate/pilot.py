"""Pilot: can gate scores of a model's Swedish output predict its EuroEval Swedish results?

    python -m laya_gate.pilot generate   # small open LLMs: FLORES+ en->sv + Swedish prompts
    python -m laya_gate.pilot score      # gate scores per output (several checkpoints)
    python -m laya_gate.pilot analyze    # model-level correlations with the EuroEval leaderboard

Each model (small enough for one 12 GB GPU, with Swedish results on the EuroEval leaderboard)
translates the same 200 FLORES+ devtest sentences into Swedish and answers the same 100 Swedish
prompts, greedy, thinking off. The gate scores every output; model-level means are correlated
(Spearman, bootstrap CI over models) with the EuroEval Swedish rank (lower is better) and task
scores. Baselines that need no gate: chrF against the FLORES+ reference, log parameter count,
and the share of answers written in Swedish. With about 20 models this is a pilot: CIs are wide.

Leaderboard: `EuroEval/leaderboards` (GitHub), `leaderboards/swedish_all.csv` and
`swedish_all_simplified.csv`, fetched at run time and stored in `cache/pilot/`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .utils import CACHE_DIR, ROOT, SEED, read_jsonl, update_metrics, write_jsonl

PILOT_DIR = CACHE_DIR / "pilot"
LEADERBOARD = "https://raw.githubusercontent.com/EuroEval/leaderboards/main/leaderboards/"
TASKS = (
    "Sentiment classification",
    "Named entity recognition",
    "Linguistic acceptability",
    "Reading comprehension",
    "Summarization",
    "Knowledge",
    "Common sense reasoning",
)
MODELS = (
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "LiquidAI/LFM2-350M",
    "LiquidAI/LFM2-700M",
    "LiquidAI/LFM2-1.2B",
    "google/gemma-3-270m-it",
    "NbAiLab/borealis-270m-instruct-preview",
    "meta-llama/Llama-3.2-1B-Instruct",
    "google/gemma-3-1b-it",
    "NbAiLab/borealis-1b-instruct-preview",
    "yulan-team/YuLan-Mini-Instruct",
    "Qwen/Qwen3.5-0.8B",
    "Unbabel/Tower-Plus-2B",
    "utter-project/EuroMoE-2.6B-A0.6B-Instruct-Preview",
    "Qwen/Qwen3.5-2B",
    "ibm-granite/granite-4.0-micro",
    "ibm-granite/granite-4.0-h-micro",
    "deepcogito/cogito-v1-preview-llama-3B",
    "NbAiLab/borealis-4b-instruct-preview",
    "google/gemma-3-4b-it",
    "mii-llm/nesso-4B",
    "tencent/Hunyuan-4B-Instruct",
)
N_TRANSLATE, N_PROMPTS = 200, 100
TRANSLATE_INSTRUCTION = (
    "Översätt följande text från engelska till svenska. Svara endast med översättningen.\n\n"
)
TRANSLATE_PROMPT = "Översätt följande text till svenska:\n"  # the gate's prompt, as in mtqa

# ------------------------------------------------------------------- leaderboard


def _name(cell: str) -> str:
    m = re.search(r">([^<]+)</a>", cell)
    return (m.group(1) if m else re.sub(r"<[^>]+>", "", cell)).strip()


def _num(cell: str) -> float | None:
    v = cell.split("@@")[-1] if "@@" in cell else cell
    try:
        return float(v)
    except ValueError:
        return None


def load_leaderboard() -> dict[str, dict[str, Any]]:
    """EuroEval Swedish leaderboard rows by model name: rank, task scores, parameters."""
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    tables = {}
    for name in ("swedish_all.csv", "swedish_all_simplified.csv"):
        path = PILOT_DIR / name
        if not path.exists():
            urllib.request.urlretrieve(LEADERBOARD + name, path)  # noqa: S310 - fixed URL
        tables[name] = list(csv.reader(path.open(encoding="utf-8")))
    simple = tables["swedish_all_simplified.csv"]
    meta = {_name(r[0]): dict(zip(simple[0], r, strict=False)) for r in simple[1:]}
    full = tables["swedish_all.csv"]
    head = [re.sub(r"<[^>]+>", "", h).strip() for h in full[0]]
    cols = {t: head.index(t) for t in TASKS if t in head}
    out = {}
    for r in full[2:]:
        name = _name(r[0])
        row = {"rank": _num(r[2]), **{t: _num(r[i]) for t, i in cols.items()}}
        params = (meta.get(name) or {}).get("parameters")
        row["parameters"] = float(params) if params and params.replace(".", "").isdigit() else None
        out[name] = row
    return out


def leaderboard_row(board: dict[str, dict], model: str) -> dict[str, Any] | None:
    """The leaderboard row for a model id; "(val)" entries are matched too. Rows without a
    valid rank (EuroEval lists incomplete evaluations with rank 100) are skipped."""
    for key in (model, f"{model} (val)"):
        row = board.get(key)
        if row and row.get("rank") is not None and row["rank"] < 100:
            return {**row, "leaderboard_name": key}
    return None


# --------------------------------------------------------------------- prompts


def translation_items(n: int = N_TRANSLATE) -> list[dict[str, str]]:
    from .mtqa import load_flores

    rows = [r for r in load_flores() if r["id"].startswith("devtest-")]
    return sorted(rows, key=lambda r: int(r["id"].split("-")[1]))[:n]


def prompt_items(n: int = N_PROMPTS, seed: int = SEED) -> list[dict[str, str]]:
    """A fixed-seed sample of the probe's Swedish prompts (short enough for small models)."""
    import random

    from .probe import PROBE_DIR

    rows = [
        {"id": p["id"], "prompt": p["sv_prompt"]}
        for p in read_jsonl(PROBE_DIR / "probe_pairs.jsonl")
        if len(p["sv_prompt"].split()) <= 120
    ]
    return random.Random(seed).sample(sorted(rows, key=lambda r: r["id"]), n)


# ------------------------------------------------------------------- generation

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def clean_output(text: str) -> str:
    text = _THINK.sub("", text)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def _chat(tok, user: str) -> str:
    msgs = [{"role": "user", "content": user}]
    for kwargs in ({"enable_thinking": False}, {}):
        try:
            return tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, **kwargs
            )
        except Exception:  # noqa: BLE001,S112 - template without that kwarg, or no template
            continue
    return user + "\n"


def generate_model(model_id: str, items: list[dict], max_new: dict[str, int], bs: int = 16):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    model = model.to("cuda").eval()
    out = []
    for kind in ("translate", "answer"):
        sel = [it for it in items if it["kind"] == kind]
        sel_sorted = sorted(sel, key=lambda it: len(it["user"]))
        for s in range(0, len(sel_sorted), bs):
            chunk = sel_sorted[s : s + bs]
            enc = tok(
                [_chat(tok, it["user"]) for it in chunk],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to("cuda")
            with torch.inference_mode():
                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new[kind],
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            for it, t in zip(chunk, texts, strict=True):
                out.append({**it, "output": clean_output(t)})
    del model
    torch.cuda.empty_cache()
    return out


def _free_gpu() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate(args) -> None:
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    items = [
        {
            "kind": "translate",
            "id": r["id"],
            "en": r["en"],
            "reference": r["human"],
            "user": TRANSLATE_INSTRUCTION + r["en"],
        }
        for r in translation_items()
    ] + [
        {"kind": "answer", "id": r["id"], "prompt": r["prompt"], "user": r["prompt"]}
        for r in prompt_items()
    ]
    status_path = PILOT_DIR / "generate_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    for model_id in args.models or MODELS:
        path = PILOT_DIR / "outputs" / f"{model_id.replace('/', '__')}.jsonl"
        if path.exists():
            status.setdefault(model_id, "cached")
            continue
        t = time.perf_counter()
        try:
            rows = generate_model(
                model_id, items, {"translate": 256, "answer": 384}, args.batch_size
            )
        except Exception as exc:  # noqa: BLE001 - one model failing must not stop the pilot
            status[model_id] = f"failed: {type(exc).__name__}: {str(exc)[:200]}"
            _free_gpu()
            print(model_id, status[model_id], flush=True)
            continue
        write_jsonl(path, rows)
        status[model_id] = f"{time.perf_counter() - t:.0f}s"
        print(model_id, status[model_id], flush=True)
    status_path.write_text(json.dumps(status, indent=2) + "\n")


# ----------------------------------------------------------------------- scores

_WORD = re.compile(r"[a-zåäöéü]+")
_SV = (
    "och att det som en är på för med av den till har inte om de jag ett men så kan ska vi "
    "eller du sig från vid när var mycket också även"
)
_EN = (
    "the and to of a is in that it for with as on are be this you not or by from at have "
    "can your if will an"
)
SV_WORDS, EN_WORDS = frozenset(_SV.split()), frozenset(_EN.split())


def is_swedish(text: str) -> bool:
    """Crude language check: more Swedish than English function words, and at least one."""
    words = _WORD.findall(text.lower())
    sv = sum(w in SV_WORDS for w in words)
    en = sum(w in EN_WORDS for w in words)
    return sv > en and sv >= 1


def chrf(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    """Sentence chrF (character n-grams 1..n without spaces, F-beta), 0-100."""

    def grams(s, k):
        s = s.replace(" ", "")
        return Counter(s[i : i + k] for i in range(len(s) - k + 1))

    precs, recs = [], []
    for k in range(1, n + 1):
        h, r = grams(hyp, k), grams(ref, k)
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precs.append(overlap / sum(h.values()))
        recs.append(overlap / sum(r.values()))
    if not precs:
        return 0.0
    p, r = float(np.mean(precs)), float(np.mean(recs))
    if p + r == 0:
        return 0.0
    return 100 * (1 + beta**2) * p * r / (beta**2 * p + r)


def gate_checkpoints() -> dict[str, Path]:
    from .v3 import V3_DIR

    out = {"v3": V3_DIR}
    for arm in ("both", "grammar", "adequacy"):
        path = ROOT / "checkpoints" / f"v4-{arm}"
        if (path / "model.safetensors").exists():
            out[f"v4-{arm}"] = path
    return out


def score(args) -> None:
    from .external_eval import pairwise
    from .predictor import SwedishPreferenceGate

    files = sorted((PILOT_DIR / "outputs").glob("*.jsonl"))
    outputs = {f.stem.replace("__", "/"): read_jsonl(f) for f in files}
    per_item: dict[str, dict[str, Any]] = {m: {} for m in outputs}
    for m, rows in outputs.items():
        tr = [r for r in rows if r["kind"] == "translate"]
        an = [r for r in rows if r["kind"] == "answer"]
        per_item[m]["chrf"] = [chrf(r["output"], r["reference"]) for r in tr]
        per_item[m]["swedish_answers"] = [float(is_swedish(r["output"])) for r in an]
        per_item[m]["answer_words"] = [len(r["output"].split()) for r in an]
    for name, path in gate_checkpoints().items():
        gate = SwedishPreferenceGate(path)
        for m, rows in outputs.items():
            tr = [r for r in rows if r["kind"] == "translate"]
            an = [r for r in rows if r["kind"] == "answer"]
            prompts = [TRANSLATE_PROMPT + r["en"] for r in tr]
            hyp, ref = [r["output"] or "-" for r in tr], [r["reference"] for r in tr]
            pf, pr = pairwise(gate, "preference", prompts, hyp, ref)
            d = per_item[m]
            d[f"{name}/translation_vs_reference"] = ((pf + pr) / 2).tolist()
            nf, nr = pairwise(gate, "naturalness", [""] * len(tr), hyp, ref)
            d[f"{name}/naturalness_vs_reference"] = ((nf + nr) / 2).tolist()
            ap, ao = [r["prompt"] for r in an], [r["output"] or "-" for r in an]
            d[f"{name}/answer_quality"] = [x["score"] for x in gate.score_quality_batch(ap, ao, 16)]
            d[f"{name}/answer_natural"] = [
                x["p_natural"] for x in gate.score_naturalness_batch(ao, ap, 16)
            ]
            if gate.has_grammar:
                d[f"{name}/answer_grammatical"] = [
                    x["p_correct"] for x in gate.score_grammar_batch(ao, None, 16)
                ]
                d[f"{name}/translation_grammatical"] = [
                    x["p_correct"] for x in gate.score_grammar_batch(hyp, None, 16)
                ]
            print(name, m, "scored", flush=True)
        del gate
    (PILOT_DIR / "scores.json").write_text(json.dumps(per_item) + "\n")


# ---------------------------------------------------------------------- analyze


def spearman_ci(x, y, n_boot: int = 2000, seed: int = SEED) -> dict[str, Any]:
    from scipy.stats import spearmanr

    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if len(set(x[i])) > 1 and len(set(y[i])) > 1:
            boots.append(spearmanr(x[i], y[i]).statistic)
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (math.nan, math.nan)
    return {"rho": rho, "ci95": [float(lo), float(hi)], "n_models": int(len(x))}


def _rank_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    ra, rb = rankdata(a), rankdata(b)
    return ra - np.polyval(np.polyfit(rb, ra, 1), rb)


def partial_spearman_ci(x, y, z, n_boot: int = 2000, seed: int = SEED) -> dict[str, Any]:
    """Spearman correlation of x and y controlling for z (ranks residualized on z's ranks),
    with a bootstrap CI over models."""
    from scipy.stats import spearmanr

    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    ok = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[ok], y[ok], z[ok]
    rho = float(spearmanr(_rank_residual(x, z), _rank_residual(y, z)).statistic)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if len(set(z[i])) > 2 and len(set(x[i])) > 1 and len(set(y[i])) > 1:
            boots.append(
                spearmanr(_rank_residual(x[i], z[i]), _rank_residual(y[i], z[i])).statistic
            )
    lo, hi = np.nanpercentile(boots, [2.5, 97.5]) if boots else (math.nan, math.nan)
    return {"rho": rho, "ci95": [float(lo), float(hi)], "n_models": int(len(x))}


def analyze(args) -> None:
    board = load_leaderboard()
    per_item = json.loads((PILOT_DIR / "scores.json").read_text())
    models, rows = [], []
    for m, d in per_item.items():
        lb = leaderboard_row(board, m)
        if lb is None:
            continue
        feats = {k: float(np.mean(v)) for k, v in d.items()}
        feats["log_parameters"] = math.log10(lb["parameters"]) if lb.get("parameters") else math.nan
        models.append(m)
        rows.append({"model": m, "leaderboard": lb, "features": feats})
    targets = {"euroeval_rank (negated)": [-r["leaderboard"]["rank"] for r in rows]}
    for t in ("Linguistic acceptability", "Summarization", "Reading comprehension", "Knowledge"):
        targets[t] = [r["leaderboard"].get(t) for r in rows]
    feature_names = sorted({k for r in rows for k in r["features"]})
    corr = {
        t: {
            f: spearman_ci(
                [r["features"].get(f, math.nan) for r in rows],
                [math.nan if v is None else v for v in ys],
            )
            for f in feature_names
        }
        for t, ys in targets.items()
    }
    size = [r["features"]["log_parameters"] for r in rows]
    partial = {
        f: partial_spearman_ci(
            [r["features"].get(f, math.nan) for r in rows], targets["euroeval_rank (negated)"], size
        )
        for f in feature_names
        if f != "log_parameters"
    }
    # independent adequacy check: per translation, does the gate agree with chrF?
    from scipy.stats import spearmanr

    item_level = {}
    chrf_all = [x for m in per_item for x in per_item[m]["chrf"]]
    for key in sorted({k for d in per_item.values() for k in d}):
        if key.endswith("/translation_vs_reference"):
            vals = [x for m in per_item for x in per_item[m][key]]
            item_level[key] = {
                "spearman_with_chrf": float(spearmanr(vals, chrf_all).statistic),
                "n_translations": len(vals),
            }
    status = json.loads((PILOT_DIR / "generate_status.json").read_text())
    out = {
        "design": {
            "translation_items": N_TRANSLATE,
            "translation_source": "FLORES+ devtest, first 200 by id; gate prompt = 'Översätt "
            "följande text till svenska:' + English; reference = professional translation",
            "answer_prompts": N_PROMPTS,
            "answer_source": "probe Swedish prompts (Kimi-k3 translations of UltraFeedback "
            "prompts), fixed-seed sample of those with at most 120 words",
            "decoding": "greedy, thinking off where the chat template allows; 256 / 384 new tokens",
            "leaderboard": "EuroEval/leaderboards swedish_all.csv (rank: lower is better)",
            "features": "model-level means; translation_vs_reference = order-averaged P(model "
            "translation beats the reference) under `preference` with the English source",
        },
        "generation": status,
        "models": rows,
        "spearman": corr,
        "translation_item_level": item_level,
        "partial_spearman_rank_given_size": partial,
    }
    update_metrics("pilot_euroeval", out)
    print(json.dumps(item_level, indent=2))
    for t, fs in corr.items():
        print(f"\n{t}")
        for f, c in sorted(fs.items(), key=lambda x: -abs(x[1]["rho"])):
            print(f"  {f:42s} rho {c['rho']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["generate", "score", "analyze"])
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    {"generate": generate, "score": score, "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    main()
