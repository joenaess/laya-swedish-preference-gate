"""arena_sv: human preference votes on Swedish LLM answers, as a held-out test.

    python -m laya_gate.arena_eval build   # collect, filter and deduplicate the Swedish battles
    python -m laya_gate.arena_eval gate    # v3, v4, v4.1: preference, quality, ties
    python -m laya_gate.arena_eval judge   # Kimi-k3 on the same pairs (needs BASE_URL, API_KEY)

Chatbot Arena users write a prompt, read two anonymous answers and vote for the better one (or
tie / both bad). Sources: `lmarena-ai/arena-human-preference-140k` (CC BY 4.0, `language ==
"sv"`), `lmarena-ai/arena-human-preference-100k` (`language == "Swedish"`) and
`lmarena-ai/arena-human-preference-55k` (Apache-2.0, no language field). Kept: single-turn
battles whose prompt and both answers read as Swedish, deduplicated across sources. Decisive
votes are the test; ties and both-bad votes check calibration (P should stay near 0.5). Never
used for training.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections import Counter
from typing import Any

import numpy as np

from .probe import bootstrap_ci, summarize
from .utils import CACHE_DIR, ROOT, load_metrics, read_jsonl, update_metrics, write_jsonl

ARENA_DIR = CACHE_DIR / "arena"
PAIRS_PATH = ARENA_DIR / "arena_sv.jsonl"
SOURCES = {
    "arena-140k": "lmarena-ai/arena-human-preference-140k",
    "arena-100k": "lmarena-ai/arena-human-preference-100k",
    "arena-55k": "lmarena-ai/arena-human-preference-55k",
}


def _text(content: Any) -> str:
    """Message content as plain text (140k stores a list of typed parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text") or "" for p in content if isinstance(p, dict))
    return ""


def split_turns(conversation: list[dict]) -> tuple[list[str], list[str]]:
    """(user turns, assistant turns) from a list of role/content messages."""
    users = [_text(m.get("content")) for m in conversation if m.get("role") == "user"]
    bots = [_text(m.get("content")) for m in conversation if m.get("role") == "assistant"]
    return users, bots


def label_from_winner(winner: str) -> str:
    """'a', 'b', 'tie' or 'both_bad' from the sources' winner strings."""
    w = str(winner).lower()
    if w == "model_a":
        return "a"
    if w == "model_b":
        return "b"
    if "bad" in w:
        return "both_bad"
    return "tie"


def battle(source, rid, users, a_turns, b_turns, label, model_a, model_b) -> dict[str, Any] | None:
    """A single-turn battle record, or None when it is multi-turn or incomplete."""
    if len(users) != 1 or len(a_turns) != 1 or len(b_turns) != 1:
        return None
    prompt, a, b = (x.strip() for x in (users[0], a_turns[0], b_turns[0]))
    if not (prompt and a and b) or a == b:
        return None
    return {
        "id": f"{source}-{rid}",
        "source": source,
        "prompt": prompt,
        "response_a": a,
        "response_b": b,
        "label": label,
        "model_a": model_a,
        "model_b": model_b,
    }


def from_140k(row) -> dict[str, Any] | None:
    users, a = split_turns(row["conversation_a"])
    _, b = split_turns(row["conversation_b"])
    return battle(
        "arena-140k",
        row["id"],
        users,
        a,
        b,
        label_from_winner(row["winner"]),
        row["model_a"],
        row["model_b"],
    )


def from_100k(row) -> dict[str, Any] | None:
    users, a = split_turns(row["conversation_a"])
    _, b = split_turns(row["conversation_b"])
    return battle(
        "arena-100k",
        row["question_id"],
        users,
        a,
        b,
        label_from_winner(row["winner"]),
        row["model_a"],
        row["model_b"],
    )


def from_55k(row) -> dict[str, Any] | None:
    def turns(s):
        try:
            return [t or "" for t in json.loads(s)]
        except (json.JSONDecodeError, TypeError):
            return []

    if row["winner_model_a"]:
        label = "a"
    elif row["winner_model_b"]:
        label = "b"
    else:
        label = "tie"  # the 55k set does not separate "both bad" from ties
    return battle(
        "arena-55k",
        row["id"],
        turns(row["prompt"]),
        turns(row["response_a"]),
        turns(row["response_b"]),
        label,
        row["model_a"],
        row["model_b"],
    )


def is_swedish_battle(r: dict[str, Any]) -> bool:
    from .pilot import is_swedish

    return all(is_swedish(r[k]) for k in ("prompt", "response_a", "response_b"))


def dedup_key(r: dict[str, Any]) -> str:
    norm = [" ".join(r[k].split()) for k in ("prompt", "response_a", "response_b")]
    payload = json.dumps([norm[0], sorted(norm[1:])], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build(args) -> None:
    from datasets import load_dataset

    stats: dict[str, Any] = {}
    kept, seen = [], set()
    for source, repo in SOURCES.items():
        ds = load_dataset(repo, split="train")
        if source == "arena-140k":
            ds = ds.filter(lambda r: r["language"] == "sv")
            conv = from_140k
        elif source == "arena-100k":
            ds = ds.filter(lambda r: r["language"] == "Swedish")
            conv = from_100k
        else:
            conv = from_55k
        s = Counter()
        for row in ds:
            r = conv(row)
            if r is None:
                s["multi_turn_or_incomplete"] += 1
                continue
            if not is_swedish_battle(r):
                s["not_swedish"] += 1
                continue
            key = dedup_key(r)
            if key in seen:
                s["duplicate"] += 1
                continue
            seen.add(key)
            kept.append(r)
            s[f"kept_{r['label']}"] += 1
        stats[source] = dict(s)
        print(source, dict(s), flush=True)
    write_jsonl(PAIRS_PATH, kept)
    labels = Counter(r["label"] for r in kept)
    update_metrics(
        "arena_sv_data",
        {"sources": SOURCES, "by_source": stats, "battles": len(kept), "labels": dict(labels)},
    )
    print(len(kept), dict(labels))


# -------------------------------------------------------------------------- gate


def gate_checkpoints() -> dict[str, Any]:
    from .v3 import V3_DIR

    out = {"v3": V3_DIR}
    for name, arm in (("v4", "adequacy"), ("v4.1", "v41"), ("v4.2a", "v42a"), ("v4.2b", "v42b")):
        path = ROOT / "checkpoints" / f"v4-{arm}"
        if (path / "model.safetensors").exists():
            out[name] = path
    return out


def decisive_metrics(p_a: np.ndarray, labels: list[str]) -> dict[str, Any]:
    """Accuracy of order-averaged P(A wins) against decisive human votes."""
    y = np.array([lab == "a" for lab in labels])
    correct = (p_a > 0.5) == y
    conf = np.abs(p_a - 0.5) >= 0.2
    return {
        **summarize(correct),
        "confident_coverage": float(conf.mean()),
        "confident_accuracy": float(correct[conf].mean()) if conf.any() else None,
    }


def tie_metrics(p_a: np.ndarray) -> dict[str, Any]:
    dev = np.abs(p_a - 0.5)
    return {
        "n": int(len(p_a)),
        "mean_abs_dev_from_0.5": float(dev.mean()) if len(dev) else None,
        "fraction_within_0.2": float((dev < 0.2).mean()) if len(dev) else None,
    }


def length_baseline(rows) -> dict[str, Any]:
    y = np.array([r["label"] == "a" for r in rows])
    la = np.array([len(r["response_a"]) for r in rows])
    lb = np.array([len(r["response_b"]) for r in rows])
    longer_a = np.where(la > lb, 1.0, np.where(la < lb, 0.0, 0.5))
    score = np.where(y, longer_a, 1 - longer_a)
    return {
        "accuracy_longer_wins": float(score.mean()),
        "ci95": bootstrap_ci(score),
        "n": len(rows),
    }


def gate_eval(args) -> None:
    from .external_eval import pairwise, versus
    from .predictor import SwedishPreferenceGate

    rows = read_jsonl(PAIRS_PATH)
    dec = [r for r in rows if r["label"] in ("a", "b")]
    ties = [r for r in rows if r["label"] == "tie"]
    bad = [r for r in rows if r["label"] == "both_bad"]
    out: dict[str, Any] = {
        "battles": len(rows),
        "decisive": len(dec),
        "ties": len(ties),
        "both_bad": len(bad),
        "by_source": dict(Counter(r["source"] for r in dec)),
        "length_baseline": length_baseline(dec),
        "metric": "decisive votes: share where the gate's order-averaged P(A wins) > 0.5 agrees "
        "with the human vote; ties / both bad: distance of P from 0.5",
        "models": {},
    }
    per_item: dict[str, Any] = {r["id"]: {"label": r["label"]} for r in rows}
    for name, path in gate_checkpoints().items():
        gate = SwedishPreferenceGate(path)
        res: dict[str, Any] = {}
        for subset, sel in (("decisive", dec), ("ties", ties), ("both_bad", bad)):
            if not sel:
                continue
            pr = [r["prompt"] for r in sel]
            a, b = [r["response_a"] for r in sel], [r["response_b"] for r in sel]
            pf, pb = pairwise(gate, "preference", pr, a, b, batch_size=8)
            p_a = (pf + pb) / 2
            qa = np.array([x["score"] for x in gate.score_quality_batch(pr, a, 8)])
            qb = np.array([x["score"] for x in gate.score_quality_batch(pr, b, 8)])
            for r, p, x, z in zip(sel, p_a, qa, qb, strict=True):
                per_item[r["id"]][name] = {
                    "p_a": float(p),
                    "quality_a": float(x),
                    "quality_b": float(z),
                }
            if subset == "decisive":
                labels = [r["label"] for r in sel]
                y = np.array([lab == "a" for lab in labels])
                q_win = versus(qa, qb)
                res["preference"] = {
                    **decisive_metrics(p_a, labels),
                    "positional_consistency": float(((pf > 0.5) == (pb > 0.5)).mean()),
                }
                q_correct = np.where(y, q_win, 1 - q_win)
                res["quality"] = {
                    "accuracy": float(q_correct.mean()),
                    "ci95": bootstrap_ci(q_correct),
                    "n": int(len(q_correct)),
                }
                trunc = [
                    gate.encode(p, x, z)["truncated"] for p, x, z in zip(pr, a, b, strict=True)
                ]
                res["fraction_truncated"] = float(np.mean(trunc))
                la = np.array([len(x) for x in a])
                lb = np.array([len(x) for x in b])
                hl = (la > lb) == y  # the human picked the longer answer
                pc = ((p_a > 0.5) == y).astype(float)
                res["length"] = {
                    "gate_picks_longer": float(((p_a > 0.5) == (la > lb)).mean()),
                    "preference_accuracy_human_picked_longer": float(pc[hl].mean()),
                    "preference_accuracy_human_picked_shorter": float(pc[~hl].mean()),
                    "quality_accuracy_human_picked_longer": float(q_correct[hl].mean()),
                    "quality_accuracy_human_picked_shorter": float(q_correct[~hl].mean()),
                    "n_human_picked_shorter": int((~hl).sum()),
                }
                res["by_source"] = {
                    s: summarize(((p_a > 0.5) == y)[np.array([r["source"] == s for r in sel])])
                    for s in sorted({r["source"] for r in sel})
                }
            else:
                res[subset] = tie_metrics(p_a)
        out["models"][name] = res
        print(
            name, json.dumps({k: v for k, v in res.items() if k != "by_source"})[:600], flush=True
        )
        del gate
    write_jsonl(ARENA_DIR / "gate_scores.jsonl", [{"id": k, **v} for k, v in per_item.items()])
    prev = load_metrics().get("arena_sv") or {}
    update_metrics("arena_sv", {**({"judge": prev["judge"]} if "judge" in prev else {}), **out})


# ------------------------------------------------------------------------- judge


async def _judge_run(args) -> dict[str, Any]:
    from .judge import Judge, discover_model, make_client, quiet_http_logs

    quiet_http_logs()
    rows = [r for r in read_jsonl(PAIRS_PATH) if r["label"] in ("a", "b")]
    client = make_client()
    model = await discover_model(client)
    judge = Judge(
        client, model, cache_path=ARENA_DIR / "judge_cache.jsonl", concurrency=args.concurrency
    )
    jobs = []
    for r in rows:
        jobs.append((r["id"], "AB", r["prompt"], r["response_a"], r["response_b"]))
        jobs.append((r["id"], "BA", r["prompt"], r["response_b"], r["response_a"]))
    res = await judge.run(jobs, progress_every=100)
    score = []
    for i, r in enumerate(rows):
        ab, ba = res[2 * i]["verdict"], res[2 * i + 1]["verdict"]
        if ab is None or ba is None:
            score.append(np.nan)
            continue
        a_wins = (ab == "A") + (ba == "B")  # times response_a won, 0..2
        human_a = r["label"] == "a"
        score.append((a_wins if human_a else 2 - a_wins) / 2)
    s = np.array(score)
    ok = ~np.isnan(s)
    human_longer = np.array(
        [(len(r["response_a"]) > len(r["response_b"])) == (r["label"] == "a") for r in rows]
    )
    kimi_models = sorted(
        {m for r in rows for m in (r["model_a"], r["model_b"]) if "kimi" in m.lower()}
    )
    lat = [x["latency_s"] for x in res if x.get("latency_s")]
    return {
        "model": model,
        "hardware": "self-hosted Kimi-k3 on 8× NVIDIA B300 (HGX)",
        "requests": len(jobs),
        "parse_failures": int(sum(x["verdict"] is None for x in res)),
        "accuracy": float(s[ok].mean()),
        "agrees_both_orders": float((s[ok] == 1).mean()),
        "ci95": bootstrap_ci(s[ok]),
        "positional_consistency": float((s[ok] != 0.5).mean()),
        "n": int(ok.sum()),
        "latency_s_median": float(np.median(lat)) if lat else None,
        "accuracy_human_picked_longer": float(s[ok & human_longer].mean()),
        "accuracy_human_picked_shorter": float(s[ok & ~human_longer].mean()),
        "kimi_models_in_battles": kimi_models,
        "metric": "per battle, the share of the two slot orders in which Kimi-k3 agrees with the "
        "human vote (split verdicts count 0.5)",
    }


def judge_eval(args) -> None:
    missing = [v for v in ("BASE_URL", "API_KEY") if not os.environ.get(v)]
    if missing:
        print(f"judge skipped: {missing[0]} not set")
        sys.exit(0)
    res = asyncio.run(_judge_run(args))
    prev = load_metrics().get("arena_sv") or {}
    update_metrics("arena_sv", {**prev, "judge": res})
    print(json.dumps(res, indent=2))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["build", "gate", "judge"])
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args(argv)
    {"build": build, "gate": gate_eval, "judge": judge_eval}[args.command](args)


if __name__ == "__main__":
    main()
