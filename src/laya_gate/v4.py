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
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from . import external_eval as X
from . import liu_eval, v2, v3
from .probe import PROBE_DIR, bootstrap_ci
from .utils import (
    CACHE_DIR,
    GRAMMAR_QID,
    METRICS_PATH,
    NATURAL_SINGLE_QID,
    NATURALNESS_QID,
    QID,
    QUESTIONS_V2,
    QUESTIONS_V3,
    QUESTIONS_V4,
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
    "grammar": ("europarl", "dalaj", "swefaq", "dalaj_grammar"),
    "adequacy": ("europarl", "dalaj", "swefaq", "dalaj_grammar", "adequacy"),
    "v41": ("europarl", "dalaj", "swefaq", "dalaj_grammar", "adequacy", "registers", "hard_source"),
}
V41 = ("europarl", "dalaj", "swefaq", "dalaj_grammar", "adequacy", "registers", "hard_source")
ARMS["v42a"] = (*V41, "length_balance")
ARMS["v42b"] = (*V41, "length_balance", "length_pairs")
ARMS["v43"] = (*V41, "wikipedia")
WIKIPEDIA_PARAGRAPHS = 4000
REGISTER_PER_SOURCE = 1000
LENGTH_PROMPTS = 3000
LENGTH_RATINGS = {"good": 4.5, "bad": 2.0}  # `quality` targets for the length-controlled answers
HARD_SOURCE_PAIRS = 1500
ADEQUACY_PER_TYPE = 3000
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


def grammar_rows(pairs: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    """Single-response `grammatical` rows from (corrected, learner) pairs in the v2 layout:
    the corrected sentence is `correct`, the learner sentence `incorrect`."""
    rows = []
    for p in pairs:
        ref = p["candidate_a"] if p["label"] == 0 else p["candidate_b"]
        other = p["candidate_b"] if p["label"] == 0 else p["candidate_a"]
        for text, ok in ((ref, True), (other, False)):
            rows.append(
                {
                    "source": source,
                    "qid": GRAMMAR_QID,
                    "id": p["id"],
                    "prompt": "",
                    "response": text,
                    "target": [1.0, 0.0] if ok else [0.0, 1.0],
                    "label": 0 if ok else 1,
                }
            )
    return rows


def shuffle_neighbours(text: str, rng: random.Random, rate: float = 0.3) -> str:
    """Swap about `rate` of adjacent word pairs (keeps every word, breaks the order)."""
    words = text.split()
    i = 0
    while i < len(words) - 1:
        if rng.random() < rate:
            words[i], words[i + 1] = words[i + 1], words[i]
            i += 2
        else:
            i += 1
    return " ".join(words)


def adequacy_data(pool, used: set[str], rng: random.Random) -> dict[str, list[dict]]:
    """Negatives that need the source: from Europarl paragraphs not in the main pairs.

    `preference` (English source in the prompt): the correct human translation beats the human
    translation of another paragraph, a half translation, a word-shuffled translation and the
    untranslated English; the correct Kimi-k3 translation also beats another paragraph's human
    translation (adequacy over style). `naturalness`: Swedish beats the English original.
    Single-response: the English original is neither natural nor grammatical Swedish."""
    paras = [r for r in pool if r["id"] not in used and r.get("mt_kimi")]
    rng.shuffle(paras)
    n = min(ADEQUACY_PER_TYPE, len(paras))
    sel = paras[:n]

    def other_para(i):
        return " ".join(sel[(i + 1 + rng.randrange(n - 1)) % n]["sv"])

    pref, nat, single = [], [], []
    for i, r in enumerate(sel):
        human, en, mt = " ".join(r["sv"]), " ".join(r["en"]), r["mt_kimi"]
        prompt = X.SOURCE_PROMPT + en
        words = human.split()
        half = " ".join(words[: max(1, len(words) // 2)])
        for kind, ref, other in (
            ("wrong_source", human, other_para(i)),
            ("mt_vs_wrong_source", mt, other_para(i)),
            ("half", human, half),
            ("shuffled", human, shuffle_neighbours(human, rng)),
            ("untranslated", human, en),
        ):
            if X._norm(ref) != X._norm(other):
                pref.append(as_pair(f"{r['id']}-{kind}", prompt, ref, other, rng))
        nat.append(as_pair(f"{r['id']}-sv-vs-en", "", human, en, rng))
        for qid in (NATURAL_SINGLE_QID, GRAMMAR_QID):
            single.append(
                {
                    "source": "adequacy",
                    "qid": qid,
                    "id": f"{r['id']}-en",
                    "prompt": "",
                    "response": en,
                    "target": [0.0, 1.0],
                    "label": 1,
                }
            )
    return {"preference": pref, "naturalness": nat, "single": single}


_PUNCT_RE = __import__("re").compile(r"[^\w\s-]")


def lower_no_punct(text: str) -> str:
    return " ".join(_PUNCT_RE.sub(" ", text.lower()).split())


def corrupt(text: str, rng: random.Random) -> str:
    """ScaLA-style corruption: delete one word or swap two neighbouring words."""
    words = text.split()
    if len(words) < 4:
        return text
    i = rng.randrange(1, len(words) - 1)
    if rng.random() < 0.5:
        out = words[:i] + words[i + 1 :]
    else:
        out = words[: i - 1] + [words[i], words[i - 1]] + words[i + 1 :]
    return " ".join(out)


def first_sentences(text: str, n: int = 3, max_words: int = 120) -> str:
    import re

    parts = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    return " ".join(" ".join(parts[:n]).split()[:max_words])


def register_texts(rng: random.Random) -> dict[str, list[str]]:
    """Correct Swedish from training-only sources in four registers (no UD treebank, so
    ScaLA-sv, GerLangMod-sv, BLiMPPI and MultiBLiMP stay unseen)."""
    from datasets import load_dataset

    faq = load_dataset(X.SUPERLIM, "swefaq", split="train")
    answers = sorted({a for r in faq for a in r["candidate_answers"]})
    dalaj = load_dataset(X.SUPERLIM, "dalaj-ged-superlim", split="train")
    textbook = sorted(
        r["sentence"]
        for r in dalaj
        if str(r["label"]) == "0" and r["meta"]["data_source"].startswith("COCTAILL")
    )
    from .fluent import FLUENT_DIR

    fluent = sorted(
        {
            r[k]
            for r in read_jsonl(FLUENT_DIR / "train.jsonl")
            for k in ("candidate_a", "candidate_b")
        }
    )
    pool = read_jsonl(V4_DIR / "europarl_pool.jsonl")
    used = {r["id"] for r in read_jsonl(V4_DIR / "europarl_preference.jsonl")}
    used |= {r["id"].rsplit("-", 1)[0] for r in read_jsonl(V4_DIR / "adequacy_naturalness.jsonl")}
    europarl = sorted(" ".join(r["sv"]) for r in pool if r["id"] not in used)
    out = {}
    for name, texts in (
        ("swefaq_answers", answers),
        ("textbook_sentences", textbook),
        ("fluent_answers", fluent),
        ("europarl", europarl),
    ):
        texts = [first_sentences(t) for t in texts if len(t.split()) >= 6]
        out[name] = rng.sample(texts, min(REGISTER_PER_SOURCE, len(texts)))
    return out


def registers_data(texts: dict[str, list[str]], rng: random.Random) -> dict[str, list[dict]]:
    """Correct text vs a corrupted copy: `naturalness` pairs (no prompt) and single-response
    `grammatical` rows for both sides; a fifth is lowercased without punctuation."""
    pairs, single = [], []
    for source, items in texts.items():
        for i, text in enumerate(items):
            bad = corrupt(text, rng)
            if X._norm(bad) == X._norm(text):
                continue
            if rng.random() < 0.2:
                text, bad = lower_no_punct(text), lower_no_punct(bad)
            rid = f"reg-{source}-{i}"
            pairs.append(as_pair(rid, "", text, bad, rng))
            for resp, ok in ((text, True), (bad, False)):
                single.append(
                    {
                        "source": "registers",
                        "qid": GRAMMAR_QID,
                        "id": rid,
                        "prompt": "",
                        "response": resp,
                        "target": [1.0, 0.0] if ok else [0.0, 1.0],
                        "label": 0 if ok else 1,
                    }
                )
    return {"naturalness": pairs, "single": single}


def session_neighbour(ids: list[str]) -> dict[str, str]:
    """For each Europarl paragraph id, the closest other paragraph of the same sitting."""
    by_session: dict[str, list[tuple[int, str]]] = {}
    for pid in ids:
        sess, _, num = pid.rpartition(".")
        if num.isdigit():
            by_session.setdefault(sess, []).append((int(num), pid))
    out = {}
    for items in by_session.values():
        items.sort()
        for j, (_, pid) in enumerate(items):
            if len(items) > 1:
                k = j + 1 if j + 1 < len(items) else j - 1
                out[pid] = items[k][1]
    return out


def hard_source_data(pool, rng: random.Random) -> list[dict[str, Any]]:
    """Wrong-source negatives from the same sitting (same topic, similar wording): the correct
    human or Kimi-k3 translation beats the human translation of a neighbouring paragraph."""
    by_id = {r["id"]: r for r in pool if r.get("mt_kimi")}
    nb = session_neighbour(sorted(by_id))
    ids = sorted(nb)
    rng.shuffle(ids)
    out = []
    for i, pid in enumerate(ids[:HARD_SOURCE_PAIRS]):
        r, other = by_id[pid], " ".join(by_id[nb[pid]]["sv"])
        ref = " ".join(r["sv"]) if i % 2 == 0 else r["mt_kimi"]
        if X._norm(ref) != X._norm(other):
            kind = "human" if i % 2 == 0 else "mt"
            out.append(
                as_pair(f"{pid}-hard_{kind}", X.SOURCE_PROMPT + " ".join(r["en"]), ref, other, rng)
            )
    return out


def wikiedits_exclusions() -> tuple[set[str], set[str]]:
    """Titles of the WikiEdits pages and normalized openings of their texts, so the Wikipedia
    training paragraphs never contain a WikiEdits test text (in either version)."""
    from urllib.parse import parse_qs, unquote, urlparse

    titles, openings = set(), set()
    for p in read_jsonl(X.EXT_DIR / "wikiedits.jsonl"):
        q = parse_qs(urlparse(p.get("url", "")).query)
        if q.get("title"):
            titles.add(unquote(q["title"][0]).replace("_", " ").casefold())
        for t in (p["ref"], p["other"]):
            openings.add(X._norm(t)[:60])
    return titles, openings


# Lsjbot and similar template text (geography and species stubs), half of a random sample
_BOT_TEXT = __import__("re").compile(
    r"Omgivningarna runt|Trakten runt|runt .{0,40} är det (ganska |mycket )?(tät|gles)befolkat|"
    r"Årsmedeltemperaturen|Medelnederbörden|invånare per kvadratkilometer|"
    r"Inga underarter finns listade|enligt Catalogue of Life|Klimatet i området är|"
    r"ligger .{0,40} meter över havet|Terrängen runt|Den högsta punkten i närheten",
    __import__("re").I,
)


def wikipedia_paragraphs(n: int = WIKIPEDIA_PARAGRAPHS, seed: int = SEED) -> list[str]:
    """Plain prose paragraphs (40-150 words) from Swedish Wikipedia (20231101 dump), skipping
    every WikiEdits page and any paragraph that opens like a WikiEdits text."""
    from datasets import load_dataset

    titles, openings = wikiedits_exclusions()
    ds = load_dataset("wikimedia/wikipedia", "20231101.sv", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    out = []
    for art in ds:
        if art["title"].casefold() in titles:
            continue
        for para in art["text"].split("\n"):
            para = " ".join(para.split())
            words = len(para.split())
            if not 40 <= words <= 150 or "|" in para or para[-1:] not in ".!?":
                continue
            if X._norm(para)[:60] in openings or _BOT_TEXT.search(para):
                continue
            out.append(para)
            break  # one paragraph per article, for variety
        if len(out) >= n:
            break
    return out


def prepare_wikipedia() -> dict[str, Any]:
    rng = random.Random(SEED + 43)
    paras = wikipedia_paragraphs()
    data = registers_data({"wikipedia": paras}, rng)
    for row in data["single"]:
        row["source"] = "wikipedia"
    write_jsonl(V4_DIR / "wikipedia_naturalness.jsonl", data["naturalness"])
    write_jsonl(V4_DIR / "wikipedia_single.jsonl", data["single"])
    stats = {
        "paragraphs": len(paras),
        "pairs": len(data["naturalness"]),
        "single": len(data["single"]),
        "source": "wikimedia/wikipedia 20231101.sv (CC BY-SA 3.0), WikiEdits pages and "
        "bot-generated stub text excluded",
    }
    v4 = load_metrics().get("v4") or {}
    v4.setdefault("data", {})["wikipedia"] = stats
    update_metrics("v4", v4)
    return stats


def prepare_v41() -> dict[str, Any]:
    rng = random.Random(SEED + 41)
    reg = registers_data(register_texts(rng), rng)
    hard = hard_source_data(read_jsonl(V4_DIR / "europarl_pool.jsonl"), rng)
    write_jsonl(V4_DIR / "registers_naturalness.jsonl", reg["naturalness"])
    write_jsonl(V4_DIR / "registers_single.jsonl", reg["single"])
    write_jsonl(V4_DIR / "hard_source_preference.jsonl", hard)
    stats = {
        "registers": {
            "pairs": len(reg["naturalness"]),
            "single": len(reg["single"]),
            "by_source": dict(Counter(p["id"].split("-")[1] for p in reg["naturalness"])),
        },
        "hard_source": {"pairs": len(hard)},
    }
    v4 = load_metrics().get("v4") or {}
    v4.setdefault("data", {}).update(stats)
    update_metrics("v4", v4)
    return stats


LENGTH_SYSTEM = (
    "You write Swedish answers for a dataset that teaches a reward model to judge answer quality "
    "independently of length. For the user's prompt, write four answers in natural, fluent, "
    "idiomatic Swedish:\n"
    "- good_long: complete, correct and helpful, about 150-250 words.\n"
    "- good_short: equally correct and helpful but concise, about 30-70 words.\n"
    "- bad_long: as long and as fluent as good_long, but clearly worse: it contains a factual "
    "error, or misreads or misses the core of the question. Not absurd, not obviously sloppy.\n"
    "- bad_short: as concise and fluent as good_short, with the same kind of flaw.\n"
    "Reply with JSON only: "
    '{"good_long": "...", "good_short": "...", "bad_long": "...", "bad_short": "..."}'
)
LENGTH_KEYS = ("good_long", "good_short", "bad_long", "bad_short")


def parse_length_answers(content: str | None) -> dict[str, str] | None:
    import re

    if not content:
        return None
    text = re.sub(r"<think>.*?</think>", "", content, flags=re.S).split("</think>")[-1]
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not all(isinstance(obj.get(k), str) and obj[k].strip() for k in LENGTH_KEYS):
        return None
    out = {k: " ".join(obj[k].split()) for k in LENGTH_KEYS}
    if not (
        len(out["good_short"]) < len(out["bad_long"])
        and len(out["bad_short"]) < len(out["good_long"])
    ):
        return None  # the length contrast must hold both ways
    return out


def length_prompts(n: int = LENGTH_PROMPTS, seed: int = SEED) -> list[dict[str, str]]:
    from .fluent import FLUENT_DIR

    rows = [
        r for r in read_jsonl(FLUENT_DIR / "train.jsonl") if 5 <= len(r["prompt"].split()) <= 150
    ]
    rows = sorted({r["id"]: r for r in rows}.values(), key=lambda r: r["id"])
    return [
        {"id": r["id"], "prompt": r["prompt"]}
        for r in random.Random(seed).sample(rows, min(n, len(rows)))
    ]


async def _generate_length_answers(prompts, concurrency: int) -> list[dict[str, Any]]:
    import hashlib

    from .fluent import NOTHINK
    from .judge import discover_model, is_retryable, make_client, quiet_http_logs

    quiet_http_logs()
    client = make_client()
    model = await discover_model(client)
    path = V4_DIR / "length_gen.jsonl"
    cache = {r["key"]: r for r in read_jsonl(path)} if path.exists() else {}
    sem, lock = asyncio.Semaphore(concurrency), asyncio.Lock()

    async def one(p):
        key = hashlib.sha256(json.dumps([model, LENGTH_SYSTEM, p["prompt"]]).encode()).hexdigest()
        if key in cache and cache[key].get("answers"):
            return cache[key]
        async with sem:
            for attempt in range(4):
                try:
                    resp = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=model,
                            temperature=0.7,
                            max_tokens=4096,
                            messages=[
                                {"role": "system", "content": LENGTH_SYSTEM},
                                {"role": "user", "content": p["prompt"]},
                            ],
                            **NOTHINK,
                        ),
                        timeout=600,
                    )
                    break
                except Exception as exc:  # noqa: BLE001 - only the class name is kept
                    if attempt < 3 and is_retryable(exc):
                        await asyncio.sleep(2.0 * 2**attempt)
                        continue
                    return {"key": key, "id": p["id"], "answers": None, "error": type(exc).__name__}
        row = {
            "key": key,
            "id": p["id"],
            "prompt": p["prompt"],
            "answers": parse_length_answers(resp.choices[0].message.content),
        }
        async with lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    return await asyncio.gather(*(one(p) for p in prompts))


async def _judge_length_pairs(gen, concurrency: int) -> dict[tuple[str, str], tuple]:
    """Kimi-k3 (standard judge prompt, both orders) on good-vs-bad pairs of opposite length."""
    from .judge import Judge, discover_model, make_client, quiet_http_logs

    quiet_http_logs()
    client = make_client()
    judge = Judge(
        client,
        await discover_model(client),
        cache_path=V4_DIR / "length_judge_cache.jsonl",
        concurrency=concurrency,
    )
    jobs, index = [], []
    for g in gen:
        for good, bad in (("good_short", "bad_long"), ("good_long", "bad_short")):
            a, b = g["answers"][good], g["answers"][bad]
            jobs += [(g["id"], "AB", g["prompt"], a, b), (g["id"], "BA", g["prompt"], b, a)]
            index.append((g["id"], good))
    res = await judge.run(jobs, progress_every=500)
    return {k: (res[2 * i]["verdict"], res[2 * i + 1]["verdict"]) for i, k in enumerate(index)}


def prepare_length() -> dict[str, Any]:
    """Length-controlled preference pairs (good short vs bad long, good long vs bad short),
    kept when Kimi-k3 prefers the good answer in both orders."""
    if not all(os.environ.get(v) for v in ("BASE_URL", "API_KEY")):
        raise SystemExit("length pairs need BASE_URL and API_KEY (Kimi-k3)")
    gen = asyncio.run(_generate_length_answers(length_prompts(), 64))
    gen = [g for g in gen if g.get("answers")]
    verdicts = asyncio.run(_judge_length_pairs(gen, 64))
    rng = random.Random(SEED + 42)
    pref, single, kept = [], [], Counter()
    for g in gen:
        for good, bad in (("good_short", "bad_long"), ("good_long", "bad_short")):
            if verdicts.get((g["id"], good)) != ("A", "B"):
                kept["judge_disagreed"] += 1
                continue
            kept[good] += 1
            a = g["answers"]
            pref.append(as_pair(f"{g['id']}-{good}", g["prompt"], a[good], a[bad], rng))
            for key, rating in ((good, LENGTH_RATINGS["good"]), (bad, LENGTH_RATINGS["bad"])):
                target = v3.rating_target(rating)
                single.append(
                    {
                        "source": "length_pairs",
                        "qid": v3.QUALITY_QID,
                        "id": f"{g['id']}-{key}",
                        "prompt": g["prompt"],
                        "response": a[key],
                        "target": target,
                        "label": int(np.argmax(target)),
                        "rating": rating,
                    }
                )
    write_jsonl(V4_DIR / "length_preference.jsonl", pref)
    write_jsonl(V4_DIR / "length_quality.jsonl", single)
    stats = {
        "prompts": len(gen),
        "pairs": len(pref),
        "single": len(single),
        "kept": dict(kept),
        "preferred_longer": float(
            np.mean(
                [
                    len(r["candidate_a"] if r["label"] == 0 else r["candidate_b"])
                    > len(r["candidate_b"] if r["label"] == 0 else r["candidate_a"])
                    for r in pref
                ]
            )
        ),
    }
    v4 = load_metrics().get("v4") or {}
    v4.setdefault("data", {})["length_pairs"] = stats
    update_metrics("v4", v4)
    return stats


def length_test_prompts(n: int = 300, seed: int = SEED + 300) -> list[dict[str, str]]:
    """Prompts for the held-out length-controlled test: fluent train prompts not used for the
    length-pair training data."""
    from .fluent import FLUENT_DIR

    used = {p["id"] for p in length_prompts()}
    rows = [
        r
        for r in read_jsonl(FLUENT_DIR / "train.jsonl")
        if 5 <= len(r["prompt"].split()) <= 150 and r["id"] not in used
    ]
    rows = sorted({r["id"]: r for r in rows}.values(), key=lambda r: r["id"])
    return [{"id": r["id"], "prompt": r["prompt"]} for r in random.Random(seed).sample(rows, n)]


def prepare_length_test() -> dict[str, Any]:
    """300 new prompts, four answers each, the two length-crossed pairs kept when Kimi-k3 agrees
    in both orders: a held-out test of whether a gate judges quality independently of length."""
    gen = asyncio.run(_generate_length_answers(length_test_prompts(), 64))
    gen = [g for g in gen if g.get("answers")]
    verdicts = asyncio.run(_judge_length_pairs(gen, 64))
    pairs = []
    for g in gen:
        for good, bad in (("good_short", "bad_long"), ("good_long", "bad_short")):
            if verdicts.get((g["id"], good)) == ("A", "B"):
                a = g["answers"]
                pairs.append(
                    {
                        "id": f"{g['id']}-{good}",
                        "prompt": g["prompt"],
                        "ref": a[good],
                        "other": a[bad],
                        "kind": good,
                    }
                )
    write_jsonl(X.EXT_DIR / "length_test.jsonl", pairs)
    stats = {
        "prompts": len(gen),
        "pairs": len(pairs),
        "by_kind": dict(Counter(p["kind"] for p in pairs)),
    }
    data = load_metrics().get("external_data") or {}
    data["length_test"] = stats
    update_metrics("external_data", data)
    return stats


def base_length_stats() -> dict[str, Any]:
    """How often the chosen answer is the longer one in the base preference sources."""
    rows = v2.load_rows("train")
    out = {}
    for src, rs in rows.items():
        flags = [chosen_is_longer(r) for r in rs]
        out[src] = {
            "chosen_longer": float(np.mean([f is True for f in flags])),
            "n": len(rs),
        }
    v4 = load_metrics().get("v4") or {}
    v4.setdefault("data", {})["base_length_bias"] = out
    update_metrics("v4", v4)
    return out


def chosen_is_longer(row: dict[str, Any]) -> bool | None:
    """Whether the chosen answer (label 0 = candidate_a) has more characters; None on a tie."""
    ch = row["candidate_a"] if row["label"] == 0 else row["candidate_b"]
    rj = row["candidate_b"] if row["label"] == 0 else row["candidate_a"]
    return None if len(ch) == len(rj) else len(ch) > len(rj)


def balance_rows(rows: list[dict[str, Any]], rng: random.Random) -> tuple[list[dict], dict]:
    """Drop pairs where the chosen answer is longer until it is longer in exactly half."""
    longer = [r for r in rows if chosen_is_longer(r) is True]
    shorter = [r for r in rows if chosen_is_longer(r) is False]
    keep_longer = rng.sample(longer, min(len(longer), len(shorter)))
    kept = keep_longer + shorter
    return kept, {
        "before": len(rows),
        "chosen_longer_before": len(longer),
        "chosen_shorter": len(shorter),
        "after": len(kept),
    }


def prepare_adequacy() -> dict[str, Any]:
    pool = read_jsonl(V4_DIR / "europarl_pool.jsonl")
    used = {r["id"] for r in read_jsonl(V4_DIR / "europarl_preference.jsonl")}
    data = adequacy_data(pool, used, random.Random(SEED + 7))
    write_jsonl(V4_DIR / "adequacy_preference.jsonl", data["preference"])
    write_jsonl(V4_DIR / "adequacy_naturalness.jsonl", data["naturalness"])
    write_jsonl(V4_DIR / "adequacy_single.jsonl", data["single"])
    stats = {k: len(v) for k, v in data.items()}
    stats["by_kind"] = dict(Counter(p["id"].split("-")[-1] for p in data["preference"]))
    v4 = load_metrics().get("v4") or {}
    v4.setdefault("data", {})["adequacy"] = stats
    update_metrics("v4", v4)
    return stats


def extra_units(tok, sources, internal, room, max_len, head_max_len) -> list[dict[str, Any]]:
    units = []
    for source in sources:
        if source == "wikipedia":
            rows = read_jsonl(V4_DIR / "wikipedia_naturalness.jsonl")
            units += v2.build_units(
                tok,
                {source: rows},
                {NATURALNESS_QID: internal[NATURALNESS_QID]},
                room,
                max_len,
                head_max_len,
            )
            rows = read_jsonl(V4_DIR / "wikipedia_single.jsonl")
            units += v3.build_single_units(tok, rows, internal, room, max_len, head_max_len)
            continue
        if source == "length_balance":
            continue  # applied to the base pairs in train()
        if source == "length_pairs":
            rows = read_jsonl(V4_DIR / "length_preference.jsonl")
            units += v2.build_units(
                tok, {source: rows}, {QID: internal[QID]}, room, max_len, head_max_len
            )
            rows = read_jsonl(V4_DIR / "length_quality.jsonl")
            units += v3.build_single_units(tok, rows, internal, room, max_len, head_max_len)
            continue
        if source == "registers":
            rows = read_jsonl(V4_DIR / "registers_naturalness.jsonl")
            units += v2.build_units(
                tok,
                {source: rows},
                {NATURALNESS_QID: internal[NATURALNESS_QID]},
                room,
                max_len,
                head_max_len,
            )
            rows = read_jsonl(V4_DIR / "registers_single.jsonl")
            units += v3.build_single_units(tok, rows, internal, room, max_len, head_max_len)
            continue
        if source == "hard_source":
            rows = read_jsonl(V4_DIR / "hard_source_preference.jsonl")
            units += v2.build_units(
                tok, {source: rows}, {QID: internal[QID]}, room, max_len, head_max_len
            )
            continue
        if source == "adequacy":
            for fname, qid in (
                ("adequacy_preference.jsonl", QID),
                ("adequacy_naturalness.jsonl", NATURALNESS_QID),
            ):
                rows = read_jsonl(V4_DIR / fname)
                units += v2.build_units(
                    tok, {source: rows}, {qid: internal[qid]}, room, max_len, head_max_len
                )
            rows = read_jsonl(V4_DIR / "adequacy_single.jsonl")
            units += v3.build_single_units(tok, rows, internal, room, max_len, head_max_len)
            continue
        if source == "dalaj_grammar":
            rows = grammar_rows(read_jsonl(V4_DIR / "dalaj_naturalness.jsonl"), source)
            units += v3.build_single_units(tok, rows, internal, room, max_len, head_max_len)
            continue
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

    questions = QUESTIONS_V4 if "dalaj_grammar" in ARMS[arm] else QUESTIONS_V3
    if "adequacy" in ARMS[arm] and not (V4_DIR / "adequacy_preference.jsonl").exists():
        print(prepare_adequacy())
    if "registers" in ARMS[arm] and not (V4_DIR / "registers_single.jsonl").exists():
        print(prepare_v41())
    if "length_pairs" in ARMS[arm] and not (V4_DIR / "length_preference.jsonl").exists():
        print(prepare_length())
    if "wikipedia" in ARMS[arm] and not (V4_DIR / "wikipedia_single.jsonl").exists():
        print(prepare_wikipedia())
    internal = {qid: to_internal(qid, q) for qid, q in questions.items()}
    pair_internal = {q: internal[q] for q in QUESTIONS_V2}
    room = state_room_for(tok, questions, hp.max_len, hp.head_max_len)
    ratings = v3.load_ratings()
    t = time.perf_counter()
    data = {}
    balance_stats = {}
    for split in ("train", "val_calib"):
        rows = v2.load_rows(split)
        if split == "train" and "length_balance" in ARMS[arm]:
            # `preference` sees a length-balanced subset; `naturalness` keeps every pair
            rng = random.Random(seed + 7)
            pref_rows = {}
            for src, rs in rows.items():
                pref_rows[src], balance_stats[src] = balance_rows(rs, rng)
            units = v2.build_units(
                tok, pref_rows, {QID: internal[QID]}, room, hp.max_len, hp.head_max_len
            ) + v2.build_units(
                tok,
                rows,
                {NATURALNESS_QID: internal[NATURALNESS_QID]},
                room,
                hp.max_len,
                hp.head_max_len,
            )
        else:
            units = v2.build_units(tok, rows, pair_internal, room, hp.max_len, hp.head_max_len)
        data[split] = units + v3.build_single_units(
            tok, v3.single_rows(rows, ratings), internal, room, hp.max_len, hp.head_max_len
        )
    if balance_stats:
        print("length balance", balance_stats, flush=True)
    data["train"] += extra_units(tok, ARMS[arm], internal, room, hp.max_len, hp.head_max_len)
    print({k: len(v) for k, v in data.items()}, f"encoded in {time.perf_counter() - t:.0f}s")
    section = f"_v4_training_{tag(arm, seed)}"
    v2.run_training(
        hp, device, model, tok, cfg, data, questions, checkpoint_dir(arm, seed), section, "v4"
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
    lt_path = X.EXT_DIR / "length_test.jsonl"
    if lt_path.exists():
        lt = read_jsonl(lt_path)
        for kind, label in (("good_short", "short_correct"), ("good_long", "long_correct")):
            sel = [p for p in lt if p["kind"] == kind]
            pr, rf, ot = (
                [p["prompt"] for p in sel],
                [p["ref"] for p in sel],
                [p["other"] for p in sel],
            )
            add(f"length_test/{label}/preference", QID, pr, rf, ot, "length")
            add(f"length_test/{label}/quality", "quality", pr, rf, ot, "length")
    for name in ("blimppi", "multiblimp", "wikiedits"):
        if name in sets:
            mp = sets[name]
            mr, mo = [p["ref"] for p in mp], [p["other"] for p in mp]
            add(f"{name}/naturalness", NATURALNESS_QID, [""] * len(mp), mr, mo, "grammar")
            add(f"{name}/natural_swedish", "natural_swedish", [""] * len(mp), mr, mo, "grammar")
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
    if gate.has_grammar:
        grammar_tests(gate, wins, guards, extra)
    # guard: natural Swedish responses wrongly flagged as translationese (lower is better)
    ev = v2.load_rows("val_eval")["fluent"]
    resp = [r["candidate_a"] for r in ev] + [r["candidate_b"] for r in ev]
    guards["fluent_flagged_translationese"] = (
        natural_scores(gate, resp, [r["prompt"] for r in ev] * 2) < 0.5
    ).astype(float)
    return wins, guards, extra


def p_correct(gate, texts) -> np.ndarray:
    return np.array([x["p_correct"] for x in gate.score_grammar_batch(texts, None, 32)])


def grammar_tests(gate, wins, guards, extra) -> None:
    """The `grammatical` question: pairwise via scores, ScaLA-sv as a classifier, and how
    often good human Swedish is flagged as incorrect (lower is better)."""
    sets = X.load_sets()
    for name in ("dalaj", "gerlangmod", "blimppi", "multiblimp", "wikiedits"):
        if name in sets:
            ps = sets[name]
            wins[f"{name}/grammatical"] = X.versus(
                p_correct(gate, [p["ref"] for p in ps]), p_correct(gate, [p["other"] for p in ps])
            )
    test, val = (
        read_jsonl(X.EXT_DIR / "scala_test.jsonl"),
        read_jsonl(X.EXT_DIR / "scala_val.jsonl"),
    )
    pt, pv = p_correct(gate, [r["text"] for r in test]), p_correct(gate, [r["text"] for r in val])
    yt, yv = np.array([r["correct"] for r in test]), np.array([r["correct"] for r in val])
    thr = max(np.unique(pv), key=lambda t: X.mcc(yv, pv >= t))
    wins["scala/grammatical_accuracy@0.5"] = ((pt >= 0.5) == yt).astype(float)
    extra["scala_grammatical"] = {
        "mcc@0.5": X.mcc(yt, pt >= 0.5),
        "mcc@val_threshold": X.mcc(yt, pt >= thr),
        "val_threshold": float(thr),
        "auc": v3.roc_auc(pt[yt], pt[~yt]),
        "by_corruption": {
            str(c): float(
                ((pt >= 0.5) == yt)[np.array([r["corruption"] == c for r in test])].mean()
            )
            for c in sorted({r["corruption"] for r in test}, key=str)
        },
        "n": int(len(yt)),
    }
    human = {
        "flores_reference": [r["human"] for r in read_jsonl(_flores_path())],
        "wmt24pp_reference": [p["ref"] for p in sets.get("wmt24pp/opus", [])],
        "europarl_human": [p["ref"] for p in sets.get("europarl_en/kimi", [])],
        "wikiedits_edited": [p["ref"] for p in sets.get("wikiedits", [])],
        "fluent_responses": [
            r[k] for r in v2.load_rows("val_eval")["fluent"] for k in ("candidate_a", "candidate_b")
        ],
    }
    for k, texts in human.items():
        if texts:
            guards[f"{k}_flagged_incorrect"] = (p_correct(gate, texts) < 0.5).astype(float)


def _flores_path():
    from .mtqa import PAIRS_PATH

    return PAIRS_PATH


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
    t, path = tag(arm, seed), checkpoint_dir(arm, seed)
    v3.calibrate(argparse.Namespace(checkpoint=path, section=f"_v4_cal_{t}"))
    _move(f"_v4_cal_{t}", t, "calibration")
    if "dalaj_grammar" in ARMS[arm]:
        calibrate_grammar(path, t)


def calibrate_grammar(path: Path, t: str) -> None:
    """Temperature for `grammatical`, fitted on DaLAJ-GED dev pairs (both sentences)."""
    from datasets import load_dataset
    from laya.common import TEMP_MAX, TEMP_MIN

    from .calibrate import fit_temperature, nll
    from .predictor import TEMPERATURE_FILE

    gate = v2.uncalibrated_gate(path)
    pairs, _ = X.dalaj_pairs(load_dataset(X.SUPERLIM, "dalaj-ged-superlim", split="dev"))
    texts = [p["ref"] for p in pairs] + [p["other"] for p in pairs]
    labels = np.r_[np.zeros(len(pairs), int), np.ones(len(pairs), int)]
    logits = gate.logits([gate.encode_single("", x, GRAMMAR_QID) for x in texts], 32)
    temp = fit_temperature(logits, labels)
    fit = {
        "temperature": temp,
        "fit_on": "DaLAJ-GED dev (corrected and learner sentences)",
        "n": int(len(labels)),
        "nll_pre": nll(logits, labels),
        "nll_post": nll(logits, labels, temp),
        "within_laya_clamp": bool(TEMP_MIN <= temp <= TEMP_MAX),
    }
    tf = path / TEMPERATURE_FILE
    saved = json.loads(tf.read_text())
    saved.setdefault("temperatures", {})[GRAMMAR_QID] = temp
    tf.write_text(json.dumps(saved, indent=2) + "\n")
    m = load_metrics()
    m["v4"]["arms"][t]["calibration"][GRAMMAR_QID] = fit
    _write(m)
    print(json.dumps(fit, indent=2))


def run(args) -> None:
    path = checkpoint_dir(args.arm, args.seed)
    if not (path / "model.safetensors").exists() or args.retrain:
        train(args.arm, args.seed)
    calibrate(args.arm, args.seed)
    evaluate(args.arm, args.seed)


# ----------------------------------------------------------------------- sanity

SANITY_GARBAGE_MODEL = "LiquidAI/LFM2-350M"  # lowest chrF in the EuroEval pilot


def sanity_sets() -> dict[str, list[dict[str, str]]]:
    """FLORES+ devtest (the pilot's 200 sentences): the reference translation against four
    degraded alternatives, with the English source in the prompt. The reference should win."""
    from .pilot import PILOT_DIR, TRANSLATE_PROMPT, translation_items

    items = translation_items()
    n = len(items)
    garbage_path = PILOT_DIR / "outputs" / f"{SANITY_GARBAGE_MODEL.replace('/', '__')}.jsonl"
    garbage = {r["id"]: r["output"] for r in read_jsonl(garbage_path) if r["kind"] == "translate"}

    def pair(i, other):
        r = items[i]
        return {"prompt": TRANSLATE_PROMPT + r["en"], "ref": r["human"], "other": other}

    def first_half(text: str) -> str:
        words = text.split()
        return " ".join(words[: max(1, len(words) // 2)])

    return {
        "wrong_source": [pair(i, items[(i + 1) % n]["human"]) for i in range(n)],
        "garbage_mt": [
            pair(i, garbage[items[i]["id"]]) for i in range(n) if garbage.get(items[i]["id"])
        ],
        "untranslated": [pair(i, items[i]["en"]) for i in range(n)],
        "first_half_only": [pair(i, first_half(items[i]["human"])) for i in range(n)],
    }


def sanity(args) -> None:
    """Every checkpoint on the sanity sets: share of pairs where the reference wins."""
    from .predictor import SwedishPreferenceGate

    sets = sanity_sets()
    ckpts = {"v3": v3.V3_DIR}
    for t in (
        "control",
        "control-s20261015",
        "europarl",
        "native",
        "both",
        "grammar",
        "adequacy",
        "v41",
        "v42a",
        "v42b",
        "v43",
    ):
        path = ROOT / "checkpoints" / f"v4-{t}"
        if (path / "model.safetensors").exists():
            ckpts[t] = path
    out: dict[str, Any] = {
        "data": "FLORES+ devtest, the pilot's 200 sentences; reference translation vs a "
        "degraded alternative; English source in the prompt for `preference`",
        "garbage_model": SANITY_GARBAGE_MODEL,
        "n": {k: len(v) for k, v in sets.items()},
        "checkpoints": {},
    }
    for name, path in ckpts.items():
        gate = SwedishPreferenceGate(path)
        res = {}
        for sname, pairs in sets.items():
            pr = [p["prompt"] for p in pairs]
            ref, oth = [p["ref"] for p in pairs], [p["other"] for p in pairs]
            res[f"{sname}/preference+source"] = summary(pair_wins(gate, QID, pr, ref, oth))
            res[f"{sname}/naturalness"] = summary(
                pair_wins(gate, NATURALNESS_QID, [""] * len(pairs), ref, oth)
            )
            if gate.has_grammar:
                res[f"{sname}/grammatical"] = summary(
                    X.versus(p_correct(gate, ref), p_correct(gate, oth))
                )
        out["checkpoints"][name] = res
        print(name, {k: round(v["mean"], 3) for k, v in res.items()}, flush=True)
        del gate
    _set_v4("sanity", out)


# -------------------------------------------------------------------- baselines

MINIMAL_PAIR_SETS = ("dalaj", "gerlangmod", "blimppi", "multiblimp", "wikiedits")


def baselines(args) -> None:
    """Model-free references on the minimal-pair sets: longer wins, unigram frequency
    (Europarl-SV word statistics) and a Europarl-SV word trigram (mean log-probability)."""
    from . import ngram

    sets = X.load_sets()
    tri = ngram.load()
    freq = X.load_stats()
    total, vocab = sum(freq.values()), len(freq)
    out: dict[str, Any] = {
        "trigram": f"stupid backoff over {tri.total:,} Europarl-SV tokens",
        "sets": {},
    }
    for name in MINIMAL_PAIR_SETS:
        if name not in sets:
            continue
        ps = sets[name]
        ref, oth = [p["ref"] for p in ps], [p["other"] for p in ps]
        length = X.versus(np.array([len(x) for x in ref]), np.array([len(x) for x in oth]))
        uni = X.versus(
            np.array([X.unigram_logprob(x, freq, total, vocab) for x in ref]),
            np.array([X.unigram_logprob(x, freq, total, vocab) for x in oth]),
        )
        trig = X.versus(
            np.array([tri.mean_logprob(x) for x in ref]),
            np.array([tri.mean_logprob(x) for x in oth]),
        )
        out["sets"][name] = {
            "n": len(ps),
            "longer_wins": summary(length),
            "unigram_frequency": summary(uni),
            "trigram": summary(trig),
        }
        np.save(V4_DIR / f"trigram_wins_{name}.npy", trig)
    _set_v4("baselines", out)
    print(json.dumps(out, indent=2))


def rescore(args) -> None:
    """Re-run the held-out suite (not training or the standard evaluation) for existing arms,
    e.g. after tests were added."""
    for t in args.tags:
        path = v3.V3_DIR if t == "v3" else ROOT / "checkpoints" / f"v4-{t}"
        if (path / "model.safetensors").exists():
            score_suite(path, t)
            print(f"rescored {t}", flush=True)


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
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "prepare-length",
            "prepare-length-test",
            "run",
            "reference",
            "compare",
            "sanity",
            "baselines",
            "rescore",
        ],
    )
    parser.add_argument("--arm", choices=sorted(ARMS), default="control")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--tags", nargs="*", default=["v3", "control", "adequacy"])
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
    elif args.command == "sanity":
        sanity(args)
    elif args.command == "prepare-length":
        print(json.dumps(base_length_stats(), indent=2))
        print(json.dumps(prepare_length(), indent=2))
    elif args.command == "prepare-length-test":
        print(json.dumps(prepare_length_test(), indent=2))
    elif args.command == "baselines":
        baselines(args)
    elif args.command == "rescore":
        rescore(args)
    elif args.command == "reference":
        score_suite(v3.V3_DIR, "v3")  # the published v3 (2 epochs), scored on the same suite
    else:
        compare(args)


if __name__ == "__main__":
    main()
