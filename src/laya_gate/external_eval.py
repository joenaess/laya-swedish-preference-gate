"""External Swedish benchmarks: grammaticality, answer relevance, translation, context length.

    python -m laya_gate.external_eval prepare    # DaLAJ + SweFAQ pairs, Europarl sample, OPUS-MT
    python -m laya_gate.external_eval translate  # Kimi-k3 EN->SV for the Europarl sample
    python -m laya_gate.external_eval gate       # v1/v2/v3 on every set, plus baselines
    python -m laya_gate.external_eval judge      # Kimi-k3 on a fixed sample of each set
    python -m laya_gate.external_eval length     # 1024 vs 2048 vs 4096 tokens on val_eval

Every set is a list of pairs `(prompt, ref, other)`, where `ref` is the version the benchmark
expects to win; the metric is the share of pairs in which the gate prefers `ref`. None of this
data was used to train any gate.

- `dalaj` (Superlim 2, DaLAJ-GED test, CC BY 4.0): a Swedish learner sentence with one error vs
  the same sentence with that error corrected (rebuilt from the error span and its correction).
  Native error types: M morphology, S syntax, L lexical, O orthography, P punctuation.
- `swefaq` (Superlim 2, SweFAQ dev + test, CC BY-SA 4.0): a Swedish public-agency FAQ question,
  its own answer vs every other answer from the same FAQ page. Natively written Swedish; the
  only difference between candidates is relevance.
- `europarl_en` (Språkbanken Europarl sv/en): a speech given in English, its professional
  Swedish translation vs a machine translation of the English (OPUS-MT, Kimi-k3).
- `europarl_sv`: a speech given in Swedish (the few tagged `speakerlang="SV"`), the native
  Swedish vs a machine translation of its English version (Swedish -> English -> Swedish).

OPUS-MT was trained on OPUS, which includes Europarl, so its Europarl output may be memorized;
Kimi-k3's training data is unknown. Both are reported separately.
"""

from __future__ import annotations

import argparse
import asyncio
import bz2
import hashlib
import html
import json
import os
import random
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .probe import bootstrap_ci, summarize
from .utils import CACHE_DIR, ROOT, SEED, load_metrics, read_jsonl, update_metrics, write_jsonl

SUPERLIM = "sbx/superlim-2"
EUROPARL_URL = "https://spraakbanken.gu.se/resurser/meningsmangder/europarl-{lang}.xml.bz2"
STATS_URL = "https://svn.spraakbanken.gu.se/sb-arkiv/pub/frekvens/stats_EUROPARL-SV.txt.zip"
MT_MODEL = "Helsinki-NLP/opus-mt-en-sv"
EXT_DIR = CACHE_DIR / "external"
EP_DIR = CACHE_DIR / "europarl"
SETS = ("dalaj", "swefaq", "europarl_en", "europarl_sv")
MT_SYSTEMS = ("opus", "kimi")
EP_SAMPLE = 1000
SOURCE_PROMPT = "Översätt följande text till svenska:\n"
DALAJ_TYPES = {
    "M": "morphology",
    "S": "syntax",
    "L": "lexical",
    "O": "orthography",
    "P": "punctuation",
}


def checkpoints() -> dict[str, Path]:
    from .v3 import V3_DIR

    return {
        "v1": ROOT / "checkpoints" / "best_model",
        "v2": ROOT / "checkpoints" / "v2",
        "v3": V3_DIR,
    }


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


# ------------------------------------------------------------------------- DaLAJ

_SPACE_BEFORE = re.compile(r"\s+([,.;:!?])")


def apply_correction(sentence: str, start: int, stop: int, correction: str) -> str:
    """Replace `sentence[start:stop]` with `correction`, keeping word boundaries on insertion
    and dropping a space left before punctuation."""
    left, right = sentence[:start], sentence[stop:]
    if correction and correction[0].isalnum() and left[-1:].isalnum():
        correction = " " + correction
    if correction and correction[-1].isalnum() and right[:1].isalnum():
        correction = correction + " "
    out = left + correction + right
    return _SPACE_BEFORE.sub(r"\1", " ".join(out.split()))


def dalaj_pairs(rows) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """One pair per incorrect sentence: (corrected, learner). Rows whose correction cannot be
    rebuilt (missing span, or a correction that leaves the text unchanged) are dropped."""
    pairs, dropped = [], Counter()
    for i, r in enumerate(rows):
        if str(r["label"]) != "1":
            continue
        meta = r["meta"]
        span, cp = meta["error_span"], meta["confusion_pair"]
        if span["start"] is None or cp["correction"] is None:
            dropped["no_span"] += 1
            continue
        s = r["sentence"]
        fixed = apply_correction(s, span["start"], span["stop"], cp["correction"])
        if _norm(fixed) == _norm(s):
            dropped["unchanged"] += 1
            continue
        types = [t.strip() for t in meta["error_label"].split(",") if t.strip()]
        pairs.append(
            {
                "id": f"dalaj-{i}",
                "prompt": "",
                "ref": fixed,
                "other": " ".join(s.split()),
                "types": types,
                "education_level": meta.get("education_level", ""),
            }
        )
    return pairs, dict(dropped)


# ------------------------------------------------------------------------ SweFAQ


def swefaq_pairs(rows, split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pairs (own answer, other answer of the same page) and per-question candidate lists."""
    pairs, questions = [], []
    for i, r in enumerate(rows):
        cands = [" ".join(c.split()) for c in r["candidate_answers"]]
        gold = int(r["label"])
        qid = f"swefaq-{split}-{i}"
        questions.append({"id": qid, "question": r["question"], "candidates": cands, "gold": gold})
        for j, c in enumerate(cands):
            if j != gold and _norm(c) != _norm(cands[gold]):
                pairs.append(
                    {
                        "id": f"{qid}-{j}",
                        "question_id": qid,
                        "prompt": r["question"],
                        "ref": cands[gold],
                        "other": c,
                    }
                )
    return pairs, questions


_WORD = re.compile(r"\w+", re.UNICODE)


def overlap_score(question: str, answer: str) -> float:
    """Lexical baseline: share of the question's word types found in the answer."""
    q = set(_WORD.findall(question.casefold()))
    a = set(_WORD.findall(answer.casefold()))
    return len(q & a) / max(1, len(q))


# ---------------------------------------------------------------------- Europarl

_LANG = re.compile(r'speakerlang="([^"]*)"')
_ID = re.compile(r'id="([^"]*)"')


def iter_paragraphs(lines, link_tag: str = "linken"):
    """`(paragraph id, speakerlang or None, [sentence tokens])` from Språkbanken's line-per-
    element Europarl XML; the id is the English paragraph's (`<linken id=...>`)."""
    lang, pid, sents, cur = None, None, [], None
    open_tag, close_tag = f"<{link_tag} ", f"</{link_tag}>"
    for line in lines:
        if line.startswith("<text"):
            m = _LANG.search(line)
            lang = m.group(1) if m else None
        elif line.startswith(open_tag):
            pid, sents = _ID.search(line).group(1), []
        elif line.startswith("<sentence"):
            cur = []
        elif line.startswith("<w ") and cur is not None:
            cur.append(html.unescape(line[line.index(">") + 1 : line.rindex("</w>")]))
        elif line.startswith("</sentence>") and cur is not None:
            sents.append(cur)
            cur = None
        elif line.startswith(close_tag) and pid is not None:
            yield pid, lang, sents
            pid = None


_COLON_S = re.compile(r"\b([A-ZÅÄÖ]{2,}): (s|n|t|ns|en|et)\b")


def fix_colon_genitive(text: str) -> str:
    """Moses splits Swedish abbreviation inflections ("EU:s", "FN:s"); join them again."""
    return _COLON_S.sub(r"\1:\2", text)


def _detok(lang: str):
    from sacremoses import MosesDetokenizer

    md = MosesDetokenizer(lang=lang)
    if lang == "sv":
        return lambda sents: [fix_colon_genitive(md.detokenize(s)) for s in sents]
    return lambda sents: [md.detokenize(s) for s in sents]


def _bz2_lines(path: Path):
    with bz2.open(path, "rt", encoding="utf-8") as f:
        yield from f


def _download(url: str, path: Path) -> Path:
    import urllib.request

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed public URL
        os.replace(tmp, path)
    return path


_LANG_MARK = re.compile(r"\([A-Z]{2}\)")


def keep_paragraph(sv: list[str], en: list[str]) -> bool:
    """Same sentence count (1-4), 12-150 Swedish words, character ratio within 0.7-1.5, no
    interpreter language marks such as "(EL)"."""
    if not 1 <= len(sv) == len(en) <= 4:
        return False
    if any(_LANG_MARK.search(x) for x in sv + en):
        return False
    n_words = sum(len(s.split()) for s in sv)
    ratio = sum(map(len, sv)) / max(1, sum(map(len, en)))
    return 12 <= n_words <= 150 and 0.7 <= ratio <= 1.5


def europarl_paragraphs() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """English- and Swedish-original paragraphs aligned with the other language."""
    sv_path = _download(EUROPARL_URL.format(lang="sv"), EP_DIR / "europarl-sv.xml.bz2")
    en_path = _download(EUROPARL_URL.format(lang="en"), EP_DIR / "europarl-en.xml.bz2")
    detok_sv, detok_en = _detok("sv"), _detok("en")
    sv, langs = {}, Counter()
    for pid, lang, sents in iter_paragraphs(_bz2_lines(sv_path)):
        langs[lang or "untagged"] += 1
        if lang in ("EN", "SV") and pid not in sv:
            sv[pid] = (lang, sents)
    en = {}
    for pid, _lang, sents in iter_paragraphs(_bz2_lines(en_path)):
        if pid in sv and pid not in en:
            en[pid] = sents
    rows, seen = [], set()
    for pid, (lang, sv_sents) in sv.items():
        if pid not in en:
            continue
        s, e = detok_sv(sv_sents), detok_en(en[pid])
        key = _norm(" ".join(s))
        if keep_paragraph(s, e) and key not in seen:
            seen.add(key)
            rows.append({"id": f"ep-{pid}", "original": lang, "sv": s, "en": e})
    stats = {
        "paragraphs_by_speakerlang": dict(langs.most_common()),
        "aligned_kept": dict(Counter(r["original"] for r in rows)),
    }
    return rows, stats


def europarl_sample(rows, n: int = EP_SAMPLE, seed: int = SEED) -> list[dict[str, Any]]:
    """Every Swedish-original paragraph, plus a fixed-seed sample of `n` English-original ones."""
    en = sorted((r for r in rows if r["original"] == "EN"), key=lambda r: r["id"])
    rng = random.Random(seed)
    return [r for r in rows if r["original"] == "SV"] + rng.sample(en, min(n, len(en)))


def opus_translate(rows, batch_size: int = 32) -> None:
    """Sentence-by-sentence OPUS-MT (beam 4) of each paragraph's English side, in place."""
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = MarianTokenizer.from_pretrained(MT_MODEL)
    model = MarianMTModel.from_pretrained(MT_MODEL).to(device).eval()
    flat = [(i, s) for i, r in enumerate(rows) for s in r["en"]]
    out: dict[int, list[str]] = defaultdict(list)
    for start in range(0, len(flat), batch_size):
        chunk = flat[start : start + batch_size]
        enc = tok([s for _, s in chunk], return_tensors="pt", padding=True, truncation=True)
        with torch.inference_mode():
            gen = model.generate(**enc.to(device), num_beams=4, max_new_tokens=256)
        for (i, _), t in zip(chunk, tok.batch_decode(gen, skip_special_tokens=True), strict=True):
            out[i].append(t)
    for i, r in enumerate(rows):
        r["mt_opus"] = " ".join(out[i])


def load_stats() -> dict[str, int]:
    """Word-form frequencies from Språkbanken's Europarl-SV statistics (summed over POS)."""
    path = _download(STATS_URL, EP_DIR / "stats_EUROPARL-SV.txt.zip")
    freq: Counter = Counter()
    with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as f:
        for raw in f:
            parts = raw.decode("utf-8", "replace").rstrip("\n").split("\t")
            if len(parts) >= 4 and parts[3].isdigit():
                freq[parts[0].casefold()] += int(parts[3])
    return dict(freq)


def unigram_logprob(text: str, freq: dict[str, int], total: int, vocab: int) -> float:
    """Mean add-one-smoothed unigram log-probability per word (higher = more typical)."""
    words = _WORD.findall(text.casefold())
    if not words:
        return float("-inf")
    return float(np.mean([np.log((freq.get(w, 0) + 1) / (total + vocab)) for w in words]))


def prepare(args) -> None:
    from datasets import load_dataset

    EXT_DIR.mkdir(parents=True, exist_ok=True)
    dalaj, dropped = dalaj_pairs(load_dataset(SUPERLIM, "dalaj-ged-superlim", split="test"))
    write_jsonl(EXT_DIR / "dalaj.jsonl", dalaj)
    faq_pairs, faq_q = [], []
    for split in ("dev", "test"):
        p, q = swefaq_pairs(load_dataset(SUPERLIM, "swefaq", split=split), split)
        faq_pairs += p
        faq_q += q
    write_jsonl(EXT_DIR / "swefaq.jsonl", faq_pairs)
    write_jsonl(EXT_DIR / "swefaq_questions.jsonl", faq_q)
    t = time.perf_counter()
    paras, ep_stats = europarl_paragraphs()
    sample = europarl_sample(paras, args.n_europarl)
    print(f"europarl: {len(paras)} aligned paragraphs in {time.perf_counter() - t:.0f}s")
    opus_translate(sample)
    write_jsonl(EXT_DIR / "europarl.jsonl", sample)
    load_stats()
    train_dalaj = load_dataset(SUPERLIM, "dalaj-ged-superlim", split="train")
    summary = {
        "train_splits": {
            "dalaj_sentences": len(train_dalaj),
            "dalaj_incorrect": sum(str(x) == "1" for x in train_dalaj["label"]),
            "swefaq_questions": len(load_dataset(SUPERLIM, "swefaq", split="train")),
        },
        "dalaj": {"n_pairs": len(dalaj), "dropped": dropped},
        "swefaq": {"n_questions": len(faq_q), "n_pairs": len(faq_pairs)},
        "europarl": {**ep_stats, "sample": dict(Counter(r["original"] for r in sample))},
    }
    update_metrics("external_data", summary)
    print(json.dumps(summary, indent=2))


# ------------------------------------------------------------ Kimi-k3 translation

KIMI_MT_SYSTEM = (
    "You are a professional translator. Translate the user's English text, an excerpt from a "
    "European Parliament debate, into Swedish. Reply with the Swedish translation only, "
    "without comments, quotation marks or notes."
)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def clean_translation(content: str | None) -> str | None:
    """The translation without reasoning blocks or wrapping quotes; None if empty/unfinished."""
    if not content:
        return None
    text = _THINK.sub("", content)
    if "<think>" in text.lower():
        return None
    text = text.split("</think>")[-1].strip().strip('"“”').strip()
    return " ".join(text.split()) or None


async def _kimi_translate(rows, concurrency: int) -> dict[str, Any]:
    from .fluent import NOTHINK
    from .judge import discover_model, is_retryable, make_client, quiet_http_logs, sanitize

    quiet_http_logs()
    client = make_client()
    model = await discover_model(client)
    path = EXT_DIR / "kimi_mt.jsonl"
    cache = {r["key"]: r for r in read_jsonl(path)} if path.exists() else {}
    sem, lock = asyncio.Semaphore(concurrency), asyncio.Lock()
    stats = Counter()

    async def one(r):
        text = " ".join(r["en"])
        key = hashlib.sha256(json.dumps([model, KIMI_MT_SYSTEM, text]).encode()).hexdigest()
        if key in cache and cache[key].get("translation"):
            return cache[key]
        async with sem:
            for attempt in range(4):
                t = time.perf_counter()
                try:
                    resp = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=model,
                            temperature=0,
                            max_tokens=4096,
                            messages=[
                                {"role": "system", "content": KIMI_MT_SYSTEM},
                                {"role": "user", "content": text},
                            ],
                            **NOTHINK,
                        ),
                        timeout=600,
                    )
                    break
                except Exception as exc:  # noqa: BLE001 - sanitized below
                    if attempt < 3 and is_retryable(exc):
                        await asyncio.sleep(2.0 * 2**attempt)
                        continue
                    stats["failed"] += 1
                    return {"key": key, "translation": None, "error": sanitize(type(exc).__name__)}
        stats["live"] += 1
        row = {
            "key": key,
            "id": r["id"],
            "translation": clean_translation(resp.choices[0].message.content),
            "latency_s": time.perf_counter() - t,
            "usage": resp.usage.model_dump() if resp.usage else {},
        }
        async with lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    results = await asyncio.gather(*(one(r) for r in rows))
    for r, res in zip(rows, results, strict=True):
        r["mt_kimi"] = res.get("translation")
    lat = [x["latency_s"] for x in results if x.get("latency_s")]
    return {
        "model": model,
        "reasoning": "off",
        "requests": len(rows),
        "live_requests": stats["live"],
        "failed": stats["failed"],
        "empty": sum(r["mt_kimi"] is None for r in rows),
        "latency_s_median": float(np.median(lat)) if lat else None,
    }


def translate(args) -> None:
    missing = [v for v in ("BASE_URL", "API_KEY") if not os.environ.get(v)]
    if missing:
        print(f"kimi translation skipped: {missing[0]} not set")
        sys.exit(0)
    rows = read_jsonl(EXT_DIR / "europarl.jsonl")
    res = asyncio.run(_kimi_translate(rows, args.concurrency))
    write_jsonl(EXT_DIR / "europarl.jsonl", rows)
    data = load_metrics().get("external_data") or {}
    update_metrics("external_data", {**data, "kimi_mt": res})
    print(json.dumps(res, indent=2))


# ---------------------------------------------------------------------- the sets


def europarl_pairs(rows, original: str, mt: str) -> list[dict[str, Any]]:
    """(human Swedish, MT Swedish) pairs; the prompt carries the English text."""
    out = []
    for r in rows:
        other = r.get(f"mt_{mt}")
        human = " ".join(r["sv"])
        if r["original"] != original or not other or _norm(other) == _norm(human):
            continue
        en = " ".join(r["en"])
        out.append(
            {"id": f"{r['id']}-{mt}", "prompt": SOURCE_PROMPT + en, "ref": human, "other": other}
        )
    return out


def load_sets() -> dict[str, list[dict[str, Any]]]:
    """Every (set/condition) list of pairs that has data on disk."""
    sets: dict[str, list[dict[str, Any]]] = {
        "dalaj": read_jsonl(EXT_DIR / "dalaj.jsonl"),
        "swefaq": read_jsonl(EXT_DIR / "swefaq.jsonl"),
    }
    ep = read_jsonl(EXT_DIR / "europarl.jsonl")
    for original, name in (("EN", "europarl_en"), ("SV", "europarl_sv")):
        for mt in MT_SYSTEMS:
            pairs = europarl_pairs(ep, original, mt)
            if pairs:
                sets[f"{name}/{mt}"] = pairs
    return sets


# -------------------------------------------------------------------------- gate


def pairwise(gate, qid: str, prompts, ref, other, batch_size: int = 32):
    """(P(ref wins) with ref in slot A, P(ref wins) with ref in slot B)."""
    fwd = [gate.encode(p, a, b, qid) for p, a, b in zip(prompts, ref, other, strict=True)]
    rev = [gate.encode(p, b, a, qid) for p, a, b in zip(prompts, ref, other, strict=True)]
    pf = gate.probabilities(gate.logits(fwd, batch_size), qid)[:, 0]
    pr = 1 - gate.probabilities(gate.logits(rev, batch_size), qid)[:, 0]
    return pf, pr


def pair_summary(pf: np.ndarray, pr: np.ndarray, groups=None) -> dict[str, Any]:
    avg = (pf + pr) / 2
    out = {
        **summarize(avg > 0.5),
        "mean_p_ref": float(avg.mean()),
        "positional_consistency": float(((pf > 0.5) == (pr > 0.5)).mean()),
        "confident_ref": float((avg >= 0.7).mean()),
        "confident_other": float((avg <= 0.3).mean()),
    }
    if groups:
        out["by_group"] = {
            g: {"n": int(m.sum()), "accuracy": float((avg[m] > 0.5).mean())}
            for g, m in groups.items()
            if m.any()
        }
    return out


def versus(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 if a > b, 0 if a < b, 0.5 on ties."""
    return np.where(a > b, 1.0, np.where(a < b, 0.0, 0.5))


def point_summary(score_ref: np.ndarray, score_other: np.ndarray, groups=None) -> dict[str, Any]:
    win = versus(score_ref, score_other)
    out = {
        "accuracy": float(win.mean()),
        "ci95": bootstrap_ci(win),
        "n": int(win.size),
        "mean_ref": float(score_ref.mean()),
        "mean_other": float(score_other.mean()),
    }
    if groups:
        out["by_group"] = {
            g: {"n": int(m.sum()), "accuracy": float(win[m].mean())}
            for g, m in groups.items()
            if m.any()
        }
    return out


def dalaj_groups(pairs) -> dict[str, np.ndarray]:
    return {
        f"{t} ({name})": np.array([t in p["types"] for p in pairs])
        for t, name in DALAJ_TYPES.items()
    }


def faq_ranking(questions, scores: dict[str, float]) -> dict[str, Any]:
    """Top-1 accuracy and MRR when each question's candidates are ranked by `scores`
    (keyed by `question id|candidate index`); ties rank pessimistically."""
    top1, rr = [], []
    for q in questions:
        s = np.array([scores[f"{q['id']}|{j}"] for j in range(len(q["candidates"]))])
        rank = 1 + int((s > s[q["gold"]]).sum() + (s == s[q["gold"]]).sum() - 1)
        top1.append(rank == 1)
        rr.append(1 / rank)
    chance = float(np.mean([1 / len(q["candidates"]) for q in questions]))
    return {
        "top1": summarize(top1),
        "mrr": float(np.mean(rr)),
        "top1_chance": chance,
        "n_questions": len(questions),
    }


def strict_top1(pairs, p_ref: np.ndarray, questions) -> dict[str, Any]:
    """Share of questions whose own answer wins every pairwise comparison on its page."""
    wins: dict[str, list[bool]] = defaultdict(list)
    for p, v in zip(pairs, p_ref, strict=True):
        wins[p["question_id"]].append(bool(v > 0.5))
    ok = [all(wins[q["id"]]) for q in questions if wins[q["id"]]]
    return summarize(ok)


def baselines(sets, questions) -> dict[str, Any]:
    """Length, word-overlap and unigram-frequency baselines, scored like the gate."""
    out: dict[str, Any] = {}
    freq = load_stats()
    total, vocab = sum(freq.values()), len(freq)

    def lp(t):
        return unigram_logprob(t, freq, total, vocab)

    for name, pairs in sets.items():
        ref = [p["ref"] for p in pairs]
        oth = [p["other"] for p in pairs]
        b: dict[str, Any] = {
            "longer_wins": float(
                versus(np.array(list(map(len, ref))), np.array(list(map(len, oth)))).mean()
            ),
            "shorter_wins": float(
                versus(np.array(list(map(len, oth))), np.array(list(map(len, ref)))).mean()
            ),
        }
        lr, lo = np.array(list(map(lp, ref))), np.array(list(map(lp, oth)))
        b["unigram_frequency_wins"] = float(versus(lr, lo).mean())
        if name.startswith("europarl"):
            b["unigram_frequency_note"] = "the human side is part of the frequency corpus"
        if name == "swefaq":
            ov_r = np.array([overlap_score(p["prompt"], p["ref"]) for p in pairs])
            ov_o = np.array([overlap_score(p["prompt"], p["other"]) for p in pairs])
            b["word_overlap_wins"] = float(versus(ov_r, ov_o).mean())
            scores = {
                f"{q['id']}|{j}": overlap_score(q["question"], c)
                for q in questions
                for j, c in enumerate(q["candidates"])
            }
            b["word_overlap_ranking"] = faq_ranking(questions, scores)
        out[name] = b
    return out


def score_model(gate, sets, questions) -> dict[str, Any]:
    res: dict[str, Any] = {}
    qids = ["preference"] + (["naturalness"] if gate.has_naturalness else [])
    for name, pairs in sets.items():
        prompts = [p["prompt"] for p in pairs]
        ref, oth = [p["ref"] for p in pairs], [p["other"] for p in pairs]
        groups = dalaj_groups(pairs) if name == "dalaj" else None
        conds = {"with_prompt": prompts}
        if name.startswith("europarl"):
            conds["no_prompt"] = [""] * len(pairs)
        r: dict[str, Any] = {}
        for cond, pr in conds.items():
            for qid in qids:
                pf, pb = pairwise(gate, qid, pr, ref, oth)
                key = qid if cond == "with_prompt" else f"{qid}/no_prompt"
                r[key] = pair_summary(pf, pb, groups)
                if name == "swefaq" and qid == "preference":
                    r[key]["strict_top1"] = strict_top1(pairs, (pf + pb) / 2, questions)
            if gate.has_pointwise:
                suffix = "" if cond == "with_prompt" else "/no_prompt"
                nat_r = np.array([x["p_natural"] for x in gate.score_naturalness_batch(ref, pr)])
                nat_o = np.array([x["p_natural"] for x in gate.score_naturalness_batch(oth, pr)])
                r[f"natural_swedish{suffix}"] = {
                    **point_summary(nat_r, nat_o, groups),
                    "flagged_translationese": {
                        "ref": float((nat_r < 0.5).mean()),
                        "other": float((nat_o < 0.5).mean()),
                    },
                }
                if name == "swefaq" and not suffix:
                    continue  # quality on SweFAQ is scored per candidate below
                q_r = np.array([x["score"] for x in gate.score_quality_batch(pr, ref)])
                q_o = np.array([x["score"] for x in gate.score_quality_batch(pr, oth)])
                r[f"quality{suffix}"] = point_summary(q_r, q_o, groups)
        if name == "swefaq" and gate.has_pointwise:
            keys, qs, cs = [], [], []
            for q in questions:
                for j, c in enumerate(q["candidates"]):
                    keys.append(f"{q['id']}|{j}")
                    qs.append(q["question"])
                    cs.append(c)
            scores = dict(
                zip(keys, (x["score"] for x in gate.score_quality_batch(qs, cs)), strict=True)
            )
            gold = {q["id"]: q["gold"] for q in questions}
            qp = [scores[f"{p['question_id']}|{p['id'].rsplit('-', 1)[1]}"] for p in pairs]
            qr = [scores[f"{p['question_id']}|{gold[p['question_id']]}"] for p in pairs]
            r["quality"] = {
                **point_summary(np.array(qr), np.array(qp)),
                "ranking": faq_ranking(questions, scores),
            }
        res[name] = r
        print(f"  {name}: done", flush=True)
    return res


def gate_eval(args) -> None:
    from .predictor import SwedishPreferenceGate

    sets = load_sets()
    questions = read_jsonl(EXT_DIR / "swefaq_questions.jsonl")
    out: dict[str, Any] = {
        "metric": "share of pairs where the reference (corrected sentence, own FAQ answer, "
        "human translation) is preferred; pairwise questions average P over both slot orders "
        "(> 0.5 wins), single-response questions compare scores (ties 0.5)",
        "n_pairs": {k: len(v) for k, v in sets.items()},
        "baselines": baselines(sets, questions),
        "models": {},
    }
    for name, path in checkpoints().items():
        if not (path / "model.safetensors").exists():
            out["models"][name] = "not measured: checkpoint missing"
            continue
        print(name, flush=True)
        gate = SwedishPreferenceGate(path)
        out["models"][name] = score_model(gate, sets, questions)
        del gate
    prev = load_metrics().get("external_eval") or {}
    update_metrics("external_eval", {**{k: prev[k] for k in ("judge",) if k in prev}, **out})
    print(json.dumps(_headline(out), indent=2))


def _headline(out) -> dict[str, Any]:
    head = {}
    for m, res in out["models"].items():
        if isinstance(res, dict):
            for s, r in res.items():
                head[f"{m}/{s}"] = {q: round(v["accuracy"], 3) for q, v in r.items()}
    return head


# ------------------------------------------------------------------------- judge

JUDGE_SYSTEMS = {
    "grammar": (
        "You are an expert in the Swedish language. You will be given two versions, A and B, "
        "of the same Swedish sentence. They differ in at most a few words or punctuation "
        "marks. Decide which version is correct, natural Swedish (grammar, word choice, "
        "spelling, punctuation). Do not let the order of the versions influence your "
        "decision.\n\n"
        'Reply with JSON only, exactly one of: {"preference": "A"} or {"preference": "B"}'
    ),
    "translation": (
        "You are an expert English-to-Swedish translator. You will be given an English text "
        "from a European Parliament debate and two Swedish translations, A and B. Decide which "
        "translation is better: faithful to the English and written in natural, idiomatic "
        "Swedish. Do not let the order of the translations or their length by itself "
        "influence your decision.\n\n"
        'Reply with JSON only, exactly one of: {"preference": "A"} or {"preference": "B"}'
    ),
    "naturalness": (
        "You are an expert in the Swedish language. You will be given two Swedish versions, A "
        "and B, of the same text. Decide which version is more natural and idiomatic Swedish, "
        "the way a native speaker would put it. Do not let the order of the versions or their "
        "length by itself influence your decision.\n\n"
        'Reply with JSON only, exactly one of: {"preference": "A"} or {"preference": "B"}'
    ),
}
# set -> [(judge setup, gate question it is compared with, use the prompt)]
JUDGE_PLAN = {
    "dalaj": [("grammar", "naturalness", False)],
    "swefaq": [("preference", "preference", True)],
    "europarl_en": [("translation", "preference", True), ("naturalness", "naturalness", False)],
    "europarl_sv": [("translation", "preference", True), ("naturalness", "naturalness", False)],
}


def judge_sample(pairs, n: int, seed: int = SEED) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    return sorted(rng.sample(pairs, min(n, len(pairs))), key=lambda p: p["id"])


def judge_body(setup: str, p: dict[str, Any], a: str, b: str) -> str:
    from .judge import user_message

    if setup == "preference":
        return user_message(p["prompt"], a, b)
    if setup == "translation":
        en = p["prompt"].removeprefix(SOURCE_PROMPT)
        return f"## English\n{en}\n\n## Translation A\n{a}\n\n## Translation B\n{b}"
    return f"## Version A\n{a}\n\n## Version B\n{b}"


def score_verdicts(v_ref_a: list, v_ref_b: list) -> tuple[np.ndarray, np.ndarray]:
    """Per pair, the share of the two orders in which ref won (NaN if either failed to parse),
    and whether both orders agreed on a winner."""
    score = np.full(len(v_ref_a), np.nan)
    cons = np.full(len(v_ref_a), np.nan)
    for i, (x, y) in enumerate(zip(v_ref_a, v_ref_b, strict=True)):
        if x is None or y is None:
            continue
        wins = (x == "A") + (y == "B")
        score[i], cons[i] = wins / 2, float(wins != 1)
    return score, cons


async def _judge_run(args) -> dict[str, Any]:
    from .judge import SYSTEM_PROMPT, Judge, discover_model, make_client, quiet_http_logs
    from .predictor import SwedishPreferenceGate

    systems = {**JUDGE_SYSTEMS, "preference": SYSTEM_PROMPT}

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
    model = await discover_model(client)
    judge = SetupJudge(
        client, model, cache_path=EXT_DIR / "judge_cache.jsonl", concurrency=args.concurrency
    )
    sets = load_sets()
    samples = {name: judge_sample(pairs, args.n) for name, pairs in sets.items()}
    jobs, index = [], []
    for name, pairs in samples.items():
        for setup, _qid, _use in JUDGE_PLAN[name.split("/")[0]]:
            for i, p in enumerate(pairs):
                for order, (a, b) in (
                    ("RA", (p["ref"], p["other"])),
                    ("RB", (p["other"], p["ref"])),
                ):
                    body = judge_body(setup, p, a, b)
                    # the setup rides in the first line of `prompt` so it is part of the cache key
                    jobs.append((f"{p['id']}|{setup}", order, f"{setup}\n{body}", a, b))
                    index.append((name, setup, i, order))
    print(f"judge: {len(jobs)} requests", flush=True)
    results = await judge.run(jobs, progress_every=250)
    verdicts: dict[tuple[str, str, str], list] = {}
    for (name, setup, i, order), r in zip(index, results, strict=True):
        lst = verdicts.setdefault((name, setup, order), [None] * len(samples[name]))
        lst[i] = r.get("verdict")
    lat = [r["latency_s"] for r in results if r.get("latency_s") is not None]
    out: dict[str, Any] = {
        "model": model,
        "hardware": "self-hosted Kimi-k3 on 8× NVIDIA B300 (HGX)",
        "requests": len(jobs),
        "live_requests": judge.live_requests,
        "failed_requests": judge.failures,
        "parse_failures": int(sum(r.get("verdict") is None for r in results)),
        "latency_s_median": float(np.median(lat)) if lat else None,
        "metric": "per pair, the share of the two slot orders in which the reference won; "
        "accuracy counts pairs where both orders pick it, split pairs count 0.5",
        "sets": {},
    }
    v3 = checkpoints()["v3"]
    gate = SwedishPreferenceGate(v3) if (v3 / "model.safetensors").exists() else None
    for name, pairs in samples.items():
        res: dict[str, Any] = {"n_pairs": len(pairs)}
        for setup, qid, use_prompt in JUDGE_PLAN[name.split("/")[0]]:
            score, cons = score_verdicts(
                verdicts[(name, setup, "RA")], verdicts[(name, setup, "RB")]
            )
            ok = ~np.isnan(score)
            entry: dict[str, Any] = {
                "accuracy": float(score[ok].mean()),
                "ref_preferred_both_orders": float((score[ok] == 1).mean()),
                "other_preferred_both_orders": float((score[ok] == 0).mean()),
                "positional_consistency": float(cons[ok].mean()),
                "ci95": bootstrap_ci(score[ok]),
                "n": int(ok.sum()),
            }
            if gate is not None and (qid != "naturalness" or gate.has_naturalness):
                prompts = [p["prompt"] if use_prompt else "" for p in pairs]
                pf, pb = pairwise(
                    gate, qid, prompts, [p["ref"] for p in pairs], [p["other"] for p in pairs]
                )
                g = ((pf + pb) / 2 > 0.5).astype(float)
                j = score > 0.5
                decided = ok & (score != 0.5)
                entry["v3_same_sample"] = {
                    "question": qid,
                    "accuracy": float(g[ok].mean()),
                    "agreement_with_judge": float((g[decided] == j[decided]).mean()),
                }
            res[setup] = entry
        out["sets"][name] = res
    return out


def judge_eval(args) -> None:
    missing = [v for v in ("BASE_URL", "API_KEY") if not os.environ.get(v)]
    if missing:
        print(f"judge skipped: {missing[0]} not set")
        sys.exit(0)
    res = asyncio.run(_judge_run(args))
    prev = load_metrics().get("external_eval") or {}
    update_metrics("external_eval", {**prev, "judge": res})
    print(json.dumps(res, indent=2))


# ------------------------------------------------------------------ context length


def length_eval(args) -> None:
    """The DPO validation pairs at 1024, 2048 and 4096 input tokens, without retraining."""
    from .benchmark import time_forward
    from .predictor import SwedishPreferenceGate
    from .v2 import load_rows

    rows = load_rows("val_eval")
    args.max_lens = sorted(args.max_lens)
    out: dict[str, Any] = {
        "data": "val_eval (original DPO pairs and fluent pairs), both slot orders, preference",
        "max_lens": args.max_lens,
        "note": "all checkpoints were trained at 1024 tokens; longer inputs are unseen lengths",
        "models": {},
    }
    for name, path in checkpoints().items():
        if not (path / "model.safetensors").exists():
            out["models"][name] = "not measured: checkpoint missing"
            continue
        res: dict[str, Any] = {}
        trunc_1024: dict[str, np.ndarray] = {}
        for max_len in args.max_lens:
            gate = SwedishPreferenceGate(path, max_len=max_len)
            r: dict[str, Any] = {}
            for source, src_rows in rows.items():
                prompts = [x["prompt"] for x in src_rows]
                chosen = [
                    x["candidate_a"] if x["label"] == 0 else x["candidate_b"] for x in src_rows
                ]
                rejected = [
                    x["candidate_b"] if x["label"] == 0 else x["candidate_a"] for x in src_rows
                ]
                trunc = np.array(
                    [
                        gate.encode(p, c, j)["truncated"]
                        for p, c, j in zip(prompts, chosen, rejected, strict=True)
                    ]
                )
                if max_len == min(args.max_lens):
                    trunc_1024[source] = trunc
                pf, pb = pairwise(gate, "preference", prompts, chosen, rejected, batch_size=8)
                win = (pf + pb) / 2 > 0.5
                long = trunc_1024[source]
                r[source] = {
                    "accuracy": summarize(win),
                    "accuracy_truncated_at_shortest": summarize(win[long]) if long.any() else None,
                    "accuracy_fits_at_shortest": summarize(win[~long]) if (~long).any() else None,
                    "fraction_truncated": float(trunc.mean()),
                    "positional_consistency": float(((pf > 0.5) == (pb > 0.5)).mean()),
                }
            if gate.device.type == "cuda":
                src = rows["original"]
                items = [gate.encode(x["prompt"], x["candidate_a"], x["candidate_b"]) for x in src]
                longest = sorted(items, key=lambda it: len(it["ids"]))[-50:]
                lat = time_forward(gate, longest, runs=100, warmup=10)
                r["latency_gpu_forward_ms_longest_inputs"] = {
                    "input_tokens_median": float(np.median([len(i["ids"]) for i in longest])),
                    "p50": float(np.median(lat)),
                    "p95": float(np.percentile(lat, 95)),
                    "n": len(lat),
                }
            res[str(max_len)] = r
            print(f"{name} max_len={max_len}: done", flush=True)
            del gate
        out["models"][name] = res
    update_metrics("context_length", out)
    print(json.dumps(out, indent=2))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["prepare", "translate", "gate", "judge", "length"])
    parser.add_argument("--n-europarl", type=int, default=EP_SAMPLE)
    parser.add_argument("--n", type=int, default=300, help="judge pairs per set")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--max-lens", type=int, nargs="+", default=[1024, 2048, 4096])
    args = parser.parse_args(argv)
    {
        "prepare": prepare,
        "translate": translate,
        "gate": gate_eval,
        "judge": judge_eval,
        "length": length_eval,
    }[args.command](args)


if __name__ == "__main__":
    main()
