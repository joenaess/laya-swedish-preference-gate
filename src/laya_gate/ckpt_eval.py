"""Can the gate monitor Swedish during pretraining, instead of a large LLM judge?

    python -m laya_gate.ckpt_eval prepare   # held-out Swedish prefixes, FLORES+ few-shot items
    python -m laya_gate.ckpt_eval run       # every checkpoint: generations + LM-based signals
    python -m laya_gate.ckpt_eval score     # gate signals on the generations
    python -m laya_gate.ckpt_eval judge     # Kimi-k3 reference on a sample (BASE_URL, API_KEY)
    python -m laya_gate.ckpt_eval analyze   # which signal tracks the judge and training progress

Two checkpoint series stand in for "a model during pretraining": the Goldfish Swedish models
(`goldfish-models/swe_latn_{5,10,100,1000}mb`, the same architecture trained on more Swedish
data) and intermediate checkpoints of `swiss-ai/Apertus-8B-2509` (a multilingual model; branches
from 210B to 15T training tokens, loaded in 4-bit NF4). Each checkpoint continues 200 held-out
Swedish Wikipedia openings (greedy, 64 new tokens) and translates 100 FLORES+ devtest sentences
5-shot. Signals that need no gate: bits per byte on held-out Swedish text, minimal-pair accuracy
from log-probabilities (MultiBLiMP, BLiMPPI, GerLangMod, DaLAJ), chrF, repetition and the share
of Swedish output. The gate scores the same generations; Kimi-k3 compares a sample with the
human reference in both orders as the "large judge" reference.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import sys
import time
from typing import Any

import numpy as np

from .utils import CACHE_DIR, SEED, read_jsonl, update_metrics, write_jsonl

CKPT_DIR = CACHE_DIR / "ckpt"
GOLDFISH = [(f"goldfish-models/swe_latn_{mb}mb", None, float(mb)) for mb in (5, 10, 100, 1000)]
APERTUS_REPO = "swiss-ai/Apertus-8B-2509"
APERTUS = [
    (APERTUS_REPO, branch, tokens)
    for branch, tokens in (
        ("step50000-tokens210B", 210),
        ("step200000-tokens840B", 840),
        ("step500000-tokens2100B", 2100),
        ("step950000-tokens3990B", 3990),
        ("step1700000-tokens7232B", 7232),
        ("step2300000-tokens12272B", 12272),
        ("step2627139-tokens15T", 15000),
    )
]
N_PREFIX, N_TRANSLATE, N_SHOT, N_JUDGE = 200, 100, 5, 50
MINIMAL_PAIRS = {"multiblimp": None, "blimppi": None, "gerlangmod": None, "dalaj": 500}


def label(repo: str, revision: str | None) -> str:
    if repo.startswith("goldfish"):
        return "goldfish-" + repo.rsplit("_", 1)[-1]
    return "apertus-" + (revision or "main").split("-tokens")[-1]


def series(repo: str) -> str:
    return "goldfish" if repo.startswith("goldfish") else "apertus"


# -------------------------------------------------------------------------- data

_SENT = re.compile(r"(?<=[.!?])\s+")


def split_prefix(paragraph: str) -> tuple[str, str] | None:
    """(first sentence, the rest up to ~60 words) for paragraphs with at least two sentences."""
    parts = _SENT.split(paragraph)
    if len(parts) < 2 or len(parts[0].split()) < 8:
        return None
    rest = " ".join(" ".join(parts[1:]).split()[:60])
    return (parts[0], rest) if len(rest.split()) >= 15 else None


def prepare(args) -> None:
    from .mtqa import load_flores
    from .v4 import V4_DIR, wikipedia_paragraphs

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    used = {
        " ".join((r["candidate_a"] if r["label"] == 0 else r["candidate_b"]).split()).casefold()
        for r in read_jsonl(V4_DIR / "wikipedia_naturalness.jsonl")
    }
    paras = [
        p for p in wikipedia_paragraphs(n=8 * N_PREFIX, seed=SEED + 99) if p.casefold() not in used
    ]
    items = []
    for i, p in enumerate(paras):
        sp = split_prefix(p)
        if sp:
            items.append({"id": f"wiki-{i}", "prefix": sp[0], "reference": sp[1], "paragraph": p})
        if len(items) >= N_PREFIX:
            break
    write_jsonl(CKPT_DIR / "prefixes.jsonl", items)
    flores = load_flores()
    dev = sorted((r for r in flores if r["id"].startswith("dev-")), key=lambda r: r["id"])
    test = sorted(
        (r for r in flores if r["id"].startswith("devtest-")),
        key=lambda r: int(r["id"].split("-")[1]),
    )[:N_TRANSLATE]
    shots = random.Random(SEED).sample(dev, N_SHOT)
    write_jsonl(CKPT_DIR / "flores_shots.jsonl", shots)
    write_jsonl(CKPT_DIR / "flores_test.jsonl", test)
    print(f"{len(items)} prefixes, {len(test)} translation items, {len(shots)} shots")


def few_shot_prompt(shots, english: str) -> str:
    blocks = [f"English: {s['en']}\nSwedish: {s['human']}" for s in shots]
    return "\n\n".join(blocks + [f"English: {english}\nSwedish:"])


def minimal_pairs() -> dict[str, list[dict[str, Any]]]:
    from .external_eval import load_sets

    sets = load_sets()
    out = {}
    for name, cap in MINIMAL_PAIRS.items():
        if name in sets:
            ps = sets[name]
            out[name] = random.Random(SEED).sample(ps, cap) if cap and len(ps) > cap else ps
    return out


# ------------------------------------------------------------------------ model


def load_lm(repo: str, revision: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    except TypeError:
        # Goldfish ships an Albert tokenizer config that transformers 5 cannot build; its
        # tokenizer.json loads directly
        from pathlib import Path

        from huggingface_hub import hf_hub_download
        from transformers import PreTrainedTokenizerFast

        special = json.loads(
            Path(hf_hub_download(repo, "special_tokens_map.json", revision=revision)).read_text()
        )
        names = {
            k: (v["content"] if isinstance(v, dict) else v)
            for k, v in special.items()
            if k in ("bos_token", "eos_token", "unk_token", "pad_token")
        }
        tok = PreTrainedTokenizerFast(
            tokenizer_file=hf_hub_download(repo, "tokenizer.json", revision=revision), **names
        )
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if series(repo) == "apertus":
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            repo,
            revision=revision,
            quantization_config=quant,
            device_map="cuda",
            dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(repo, revision=revision).to("cuda")
    return model.eval(), tok


def max_context(model) -> int:
    cfg = model.config
    return int(getattr(cfg, "n_positions", None) or getattr(cfg, "max_position_embeddings", 2048))


def generate(model, tok, prompts: list[str], max_new: int, stop_newline: bool, bs: int = 8):
    import torch

    ctx = max_context(model)
    out = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    s = 0
    while s < len(order):
        idx = order[s : s + bs]
        enc = tok(
            [prompts[i] for i in idx],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=ctx - max_new,
        ).to("cuda")
        try:
            with torch.inference_mode():
                gen = model.generate(
                    **enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id
                )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs //= 2
            continue
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        for i, t in zip(idx, texts, strict=True):
            t = t.strip()
            out[i] = t.split("\n")[0].strip() if stop_newline else " ".join(t.split())
        s += len(idx)
    return out


def sequence_logprobs(model, tok, texts: list[str], bs: int = 8) -> list[tuple[float, int]]:
    """(sum of log-probabilities, number of scored tokens) per text; the first token is
    conditioned on BOS (or EOS when the tokenizer has no BOS)."""
    import torch

    start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    ctx = max_context(model)
    res = []
    for s in range(0, len(texts), bs):
        ids = [
            [start] + tok(t, add_special_tokens=False)["input_ids"][: ctx - 1]
            for t in texts[s : s + bs]
        ]
        width = max(len(x) for x in ids)
        pad = tok.pad_token_id
        inp = torch.tensor([x + [pad] * (width - len(x)) for x in ids], device="cuda")
        att = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids], device="cuda")
        with torch.inference_mode():
            logits = model(input_ids=inp, attention_mask=att).logits.float()
        logp = torch.log_softmax(logits[:, :-1], dim=-1)
        tgt = inp[:, 1:]
        tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * att[:, 1:]
        for row, x in zip(tok_lp, ids, strict=True):
            res.append((float(row.sum()), len(x) - 1))
    return res


def repetition(text: str) -> float:
    """Share of word 3-grams that repeat an earlier 3-gram in the same text."""
    w = text.split()
    grams = [tuple(w[i : i + 3]) for i in range(len(w) - 2)]
    if not grams:
        return 0.0
    return 1 - len(set(grams)) / len(grams)


def run_checkpoint(repo, revision, prefixes, shots, test, pairs) -> dict[str, Any]:
    t = time.perf_counter()
    model, tok = load_lm(repo, revision)
    cont = generate(model, tok, [p["prefix"] for p in prefixes], 64, stop_newline=False)
    trans = generate(
        model, tok, [few_shot_prompt(shots, r["en"]) for r in test], 128, stop_newline=True
    )
    texts = [p["paragraph"] for p in prefixes]
    lps = sequence_logprobs(model, tok, texts)
    nbytes = sum(len(x.encode("utf-8")) for x in texts)
    bpb = -sum(lp for lp, _ in lps) / (nbytes * math.log(2))
    mp = {}
    for name, ps in pairs.items():
        good = sequence_logprobs(model, tok, [p["ref"] for p in ps])
        bad = sequence_logprobs(model, tok, [p["other"] for p in ps])
        mp[name] = float(np.mean([g[0] > b[0] for g, b in zip(good, bad, strict=True)]))
    del model
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    return {
        "continuations": cont,
        "translations": trans,
        "bits_per_byte": bpb,
        "minimal_pairs": mp,
        "seconds": time.perf_counter() - t,
    }


def run(args) -> None:
    prefixes = read_jsonl(CKPT_DIR / "prefixes.jsonl")
    shots, test = (
        read_jsonl(CKPT_DIR / "flores_shots.jsonl"),
        read_jsonl(CKPT_DIR / "flores_test.jsonl"),
    )
    pairs = minimal_pairs()
    out_dir = CKPT_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for repo, revision, progress in GOLDFISH + APERTUS:
        name = label(repo, revision)
        path = out_dir / f"{name}.json"
        if path.exists():
            continue
        try:
            res = run_checkpoint(repo, revision, prefixes, shots, test, pairs)
        except Exception as exc:  # noqa: BLE001 - one checkpoint failing must not stop the run
            print(f"{name} failed: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            continue
        res.update(repo=repo, revision=revision, series=series(repo), progress=progress, label=name)
        path.write_text(json.dumps(res, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            f"{name}: bpb {res['bits_per_byte']:.3f} {res['minimal_pairs']} {res['seconds']:.0f}s",
            flush=True,
        )
        if series(repo) == "apertus" and args.delete_after:
            # each Apertus branch is a full copy of the weights; drop them once used
            from huggingface_hub import scan_cache_dir

            info = scan_cache_dir()
            revs = [r.commit_hash for rp in info.repos if rp.repo_id == repo for r in rp.revisions]
            if revs:
                info.delete_revisions(*revs).execute()


# ------------------------------------------------------------------------- score


def load_outputs() -> list[dict[str, Any]]:
    outs = [json.loads(p.read_text()) for p in sorted((CKPT_DIR / "outputs").glob("*.json"))]
    return sorted(outs, key=lambda o: (o["series"], o["progress"]))


def score(args) -> None:
    from .external_eval import pairwise
    from .pilot import TRANSLATE_PROMPT, chrf, is_swedish
    from .predictor import SwedishPreferenceGate
    from .publish import V4_CHECKPOINT_DIR

    prefixes = read_jsonl(CKPT_DIR / "prefixes.jsonl")
    test = read_jsonl(CKPT_DIR / "flores_test.jsonl")
    gate = SwedishPreferenceGate(V4_CHECKPOINT_DIR)
    scores = {}
    for o in load_outputs():
        cont = [c or "-" for c in o["continuations"]]
        trans = [t or "-" for t in o["translations"]]
        pre = [p["prefix"] for p in prefixes]
        ref = [p["reference"] for p in prefixes]
        d: dict[str, Any] = {
            "repetition": float(np.mean([repetition(c) for c in cont])),
            "swedish_share": float(np.mean([is_swedish(c) for c in cont])),
            "continuation_chrf": float(
                np.mean([chrf(c, r) for c, r in zip(cont, ref, strict=True)])
            ),
            "translation_chrf": float(
                np.mean([chrf(t, r["human"]) for t, r in zip(trans, test, strict=True)])
            ),
            "bits_per_byte": o["bits_per_byte"],
            **{f"minimal_pairs/{k}": v for k, v in o["minimal_pairs"].items()},
            "minimal_pairs/mean": float(np.mean(list(o["minimal_pairs"].values()))),
        }
        d["gate/continuation_natural"] = float(
            np.mean([x["p_natural"] for x in gate.score_naturalness_batch(cont, pre, 16)])
        )
        d["gate/continuation_grammatical"] = float(
            np.mean([x["p_correct"] for x in gate.score_grammar_batch(cont, None, 16)])
        )
        pf, pb = pairwise(gate, "naturalness", [""] * len(cont), cont, ref, batch_size=16)
        d["gate/continuation_vs_human"] = float(((pf + pb) / 2).mean())
        prompts = [TRANSLATE_PROMPT + r["en"] for r in test]
        tf, tb = pairwise(gate, "preference", prompts, trans, [r["human"] for r in test], 16)
        d["gate/translation_vs_reference"] = float(((tf + tb) / 2).mean())
        scores[o["label"]] = d
        print(o["label"], {k: round(v, 3) for k, v in d.items()}, flush=True)
    (CKPT_DIR / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")


# ------------------------------------------------------------------------- judge

CONTINUATION_SYSTEM = (
    "You are an expert in the Swedish language. You will be given the opening sentence of a "
    "Swedish encyclopedia text and two possible continuations, A and B. Decide which continuation "
    "is better Swedish: grammatical, natural, coherent and fitting the opening. Do not let the "
    "order or the length by itself influence your decision.\n\n"
    'Reply with JSON only, exactly one of: {"preference": "A"} or {"preference": "B"}'
)


async def _judge(args) -> dict[str, Any]:
    from .external_eval import JUDGE_SYSTEMS
    from .judge import Judge, discover_model, make_client, quiet_http_logs

    systems = {
        "continuation": CONTINUATION_SYSTEM,
        "translation": JUDGE_SYSTEMS["translation_generic"],
    }

    class SetupJudge(Judge):
        async def _call(self, prompt: str, a: str, b: str):
            setup, _, body = prompt.partition("\n")
            return await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": systems[setup]},
                        {"role": "user", "content": body},
                    ],
                    temperature=0,
                    max_tokens=self.max_tokens,
                ),
                timeout=self.timeout,
            )

    quiet_http_logs()
    client = make_client()
    judge = SetupJudge(
        client,
        await discover_model(client),
        cache_path=CKPT_DIR / "judge_cache.jsonl",
        concurrency=args.concurrency,
    )
    prefixes = read_jsonl(CKPT_DIR / "prefixes.jsonl")[:N_JUDGE]
    test = read_jsonl(CKPT_DIR / "flores_test.jsonl")[:N_JUDGE]
    jobs, index = [], []
    for o in load_outputs():
        for i, p in enumerate(prefixes):
            m, h = o["continuations"][i] or "-", p["reference"]
            body = f"## Opening\n{p['prefix']}\n\n"
            for order, (a, b) in (("MA", (m, h)), ("MB", (h, m))):
                jobs.append(
                    (
                        f"{o['label']}|c{i}",
                        order,
                        f"continuation\n{body}## Continuation A\n{a}\n\n## Continuation B\n{b}",
                        a,
                        b,
                    )
                )
                index.append((o["label"], "continuation", i, order))
        for i, r in enumerate(test):
            m, h = o["translations"][i] or "-", r["human"]
            for order, (a, b) in (("MA", (m, h)), ("MB", (h, m))):
                body = f"## English\n{r['en']}\n\n## Translation A\n{a}\n\n## Translation B\n{b}"
                jobs.append((f"{o['label']}|t{i}", order, f"translation\n{body}", a, b))
                index.append((o["label"], "translation", i, order))
    res = await judge.run(jobs, progress_every=200)
    wins: dict[tuple[str, str], list[float]] = {}
    verdicts = {k: r.get("verdict") for k, r in zip(index, res, strict=True)}
    for (lab, setup, i, order), v in verdicts.items():
        if order != "MA":
            continue
        v2 = verdicts.get((lab, setup, i, "MB"))
        if v is None or v2 is None:
            continue
        wins.setdefault((lab, setup), []).append(((v == "A") + (v2 == "B")) / 2)
    return {
        "requests": len(jobs),
        "parse_failures": int(sum(r.get("verdict") is None for r in res)),
        "per_checkpoint": {
            f"{lab}/{setup}": {"model_beats_reference": float(np.mean(w)), "n": len(w)}
            for (lab, setup), w in wins.items()
        },
        "metric": "share of the two slot orders in which Kimi-k3 prefers the checkpoint's output "
        "to the human reference (continuation of a Wikipedia opening, FLORES+ translation)",
    }


def judge(args) -> None:
    missing = [v for v in ("BASE_URL", "API_KEY") if not os.environ.get(v)]
    if missing:
        print(f"judge skipped: {missing[0]} not set")
        sys.exit(0)
    res = asyncio.run(_judge(args))
    (CKPT_DIR / "judge.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "per_checkpoint"}, indent=2))


# -------------------------------------------------------- checkpoint vs final


async def _versus_final(args) -> dict[str, Any]:
    """Each checkpoint against the last one of its series on the same items: Kimi-k3 (both
    orders) and the gate (order-averaged), and their per-item agreement."""
    from .external_eval import JUDGE_SYSTEMS, pairwise
    from .judge import Judge, discover_model, make_client, quiet_http_logs
    from .pilot import TRANSLATE_PROMPT, chrf
    from .predictor import SwedishPreferenceGate
    from .publish import V4_CHECKPOINT_DIR

    systems = {
        "continuation": CONTINUATION_SYSTEM,
        "translation": JUDGE_SYSTEMS["translation_generic"],
    }

    class SetupJudge(Judge):
        async def _call(self, prompt: str, a: str, b: str):
            setup, _, body = prompt.partition("\n")
            return await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": systems[setup]},
                        {"role": "user", "content": body},
                    ],
                    temperature=0,
                    max_tokens=self.max_tokens,
                ),
                timeout=self.timeout,
            )

    quiet_http_logs()
    client = make_client()
    judge = SetupJudge(
        client,
        await discover_model(client),
        cache_path=CKPT_DIR / "judge_cache.jsonl",
        concurrency=args.concurrency,
    )
    prefixes = read_jsonl(CKPT_DIR / "prefixes.jsonl")[:N_JUDGE]
    test = read_jsonl(CKPT_DIR / "flores_test.jsonl")[:N_JUDGE]
    outs = load_outputs()
    final = {s: [o for o in outs if o["series"] == s][-1] for s in ("goldfish", "apertus")}
    gate = SwedishPreferenceGate(V4_CHECKPOINT_DIR)
    jobs, index, gate_p, chrf_diff = [], [], {}, {}
    for o in outs:
        f = final[o["series"]]
        if o is f:
            continue
        setups = [("continuation", prefixes)] + (
            [("translation", test)] if o["series"] == "apertus" else []
        )
        for setup, items in setups:
            key = "continuations" if setup == "continuation" else "translations"
            mine = [o[key][i] or "-" for i in range(len(items))]
            ref = [f[key][i] or "-" for i in range(len(items))]
            if setup == "continuation":
                pf, pb = pairwise(gate, "naturalness", [""] * len(items), mine, ref, batch_size=16)
            else:
                pr = [TRANSLATE_PROMPT + r["en"] for r in items]
                pf, pb = pairwise(gate, "preference", pr, mine, ref, batch_size=16)
                chrf_diff[(o["label"], setup)] = [
                    chrf(a, r["human"]) - chrf(b, r["human"])
                    for a, b, r in zip(mine, ref, items, strict=True)
                ]
            gate_p[(o["label"], setup)] = ((pf + pb) / 2).tolist()
            for i, it in enumerate(items):
                a, b = mine[i], ref[i]
                if setup == "continuation":
                    body = f"## Opening\n{it['prefix']}\n\n"
                    fmt = "## Continuation A\n{}\n\n## Continuation B\n{}"
                else:
                    body = f"## English\n{it['en']}\n\n"
                    fmt = "## Translation A\n{}\n\n## Translation B\n{}"
                for order, (x, y) in (("MA", (a, b)), ("MB", (b, a))):
                    jobs.append(
                        (
                            f"{o['label']}|vsfinal|{setup}{i}",
                            order,
                            f"{setup}\n{body}{fmt.format(x, y)}",
                            x,
                            y,
                        )
                    )
                    index.append((o["label"], setup, i, order))
    res = await judge.run(jobs, progress_every=300)
    verdicts = {k: r.get("verdict") for k, r in zip(index, res, strict=True)}
    out = {}
    for (lab, setup), gp in gate_p.items():
        kimi = []
        for i in range(len(gp)):
            v1, v2 = verdicts.get((lab, setup, i, "MA")), verdicts.get((lab, setup, i, "MB"))
            kimi.append(np.nan if v1 is None or v2 is None else ((v1 == "A") + (v2 == "B")) / 2)
        k, g = np.array(kimi), np.array(gp)
        decided = ~np.isnan(k) & (k != 0.5)
        entry = {
            "kimi_checkpoint_wins": float(np.nanmean(k)),
            "gate_checkpoint_wins": float((g > 0.5).mean()),
            "gate_mean_p": float(g.mean()),
            "n": int(len(g)),
            "n_kimi_decided": int(decided.sum()),
            "gate_agrees_with_kimi": float(((g > 0.5) == (k > 0.5))[decided].mean())
            if decided.any()
            else None,
        }
        if (lab, setup) in chrf_diff:
            cd = np.array(chrf_diff[(lab, setup)])
            ok = decided & (cd != 0)
            entry["chrf_agrees_with_kimi"] = (
                float(((cd > 0) == (k > 0.5))[ok].mean()) if ok.any() else None
            )
        out[f"{lab}/{setup}"] = entry
    return {"versus": {s: final[s]["label"] for s in final}, "pairs": out}


def versus_final(args) -> None:
    missing = [v for v in ("BASE_URL", "API_KEY") if not os.environ.get(v)]
    if missing:
        print(f"skipped: {missing[0]} not set")
        sys.exit(0)
    res = asyncio.run(_versus_final(args))
    (CKPT_DIR / "versus_final.json").write_text(json.dumps(res, indent=2) + "\n")
    for k, v in res["pairs"].items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})


# ----------------------------------------------------------------------- analyze


def analyze(args) -> None:
    from scipy.stats import spearmanr

    outs = load_outputs()
    scores = json.loads((CKPT_DIR / "scores.json").read_text())
    j = (
        json.loads((CKPT_DIR / "judge.json").read_text())
        if (CKPT_DIR / "judge.json").exists()
        else {}
    )
    J = j.get("per_checkpoint", {})
    labels = [o["label"] for o in outs if o["label"] in scores]
    signals = sorted({k for s in scores.values() for k in s})
    lower_better = {"bits_per_byte", "repetition"}

    def vec(f):
        return np.array([scores[lab][f] * (-1 if f in lower_better else 1) for lab in labels])

    refs = {
        name: np.array(
            [J.get(f"{lab}/{name}", {}).get("model_beats_reference", np.nan) for lab in labels]
        )
        for name in ("continuation", "translation")
    }
    progress = {
        s: [
            (o["label"], math.log10(o["progress"]))
            for o in outs
            if o["series"] == s and o["label"] in scores
        ]
        for s in ("goldfish", "apertus")
    }
    table = {}
    for f in signals:
        x = vec(f)
        row = {}
        for name, y in refs.items():
            ok = ~np.isnan(y)
            if ok.sum() >= 4:
                row[f"spearman_with_kimi_{name}"] = float(spearmanr(x[ok], y[ok]).statistic)
        for s, items in progress.items():
            if len(items) >= 3:
                xs = [scores[lab][f] * (-1 if f in lower_better else 1) for lab, _ in items]
                row[f"spearman_with_progress_{s}"] = float(
                    spearmanr(xs, [p for _, p in items]).statistic
                )
        table[f] = row
    out = {
        "checkpoints": [
            {
                "label": o["label"],
                "series": o["series"],
                "progress": o["progress"],
                "revision": o["revision"],
                **scores.get(o["label"], {}),
                **{
                    f"kimi/{k}": J.get(f"{o['label']}/{k}", {}).get("model_beats_reference")
                    for k in refs
                },
            }
            for o in outs
        ],
        "signals": table,
        "note": "signals oriented so that higher is better (bits per byte and repetition negated); "
        "progress = log10 of Swedish training data (Goldfish, MB) or training tokens (Apertus, B)",
        "judge": {k: v for k, v in j.items() if k != "per_checkpoint"},
    }
    vf = CKPT_DIR / "versus_final.json"
    if vf.exists():
        out["versus_final"] = json.loads(vf.read_text())
    update_metrics("checkpoint_signals", out)
    print(json.dumps(table, indent=2))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "command", choices=["prepare", "run", "score", "judge", "versus-final", "analyze"]
    )
    parser.add_argument("--delete-after", action="store_true")
    parser.add_argument("--concurrency", type=int, default=48)
    args = parser.parse_args(argv)
    {
        "prepare": prepare,
        "run": run,
        "score": score,
        "judge": judge,
        "versus-final": versus_final,
        "analyze": analyze,
    }[args.command](args)


if __name__ == "__main__":
    main()
