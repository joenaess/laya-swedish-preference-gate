"""Dataset loading, symmetrization, token-level truncation and splitting.

`jonasaise/dpo-swedish-interpreted` ships one `train` split as 301 parquet files with two
schemas: most files carry `prompt, chosen, rejected, target_language, dataset_source,
is_safety_focused`, a minority `prompt, chosen, rejected, language`. The dataset card declares
only the second, so `datasets.load_dataset` fails with a cast error. We read the parquet files
directly and keep the columns both schemas share.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
from collections.abc import Sequence
from typing import Any

import numpy as np

from .utils import (
    BASE_MODEL_REPO,
    BASE_MODEL_SUBFOLDER,
    DATASET_REPO,
    HEAD_MAX_LEN,
    MAX_LEN,
    QID,
    QUESTION,
    SEED,
    SPLITS_DIR,
    update_metrics,
    write_jsonl,
)

# --------------------------------------------------------------------------- loading


def pair_id(prompt: str, chosen: str, rejected: str) -> str:
    h = hashlib.sha256("\x00".join((prompt, chosen, rejected)).encode("utf-8"))
    return h.hexdigest()[:16]


def load_raw_pairs(repo_id: str = DATASET_REPO) -> list[dict[str, str]]:
    """All (prompt, chosen, rejected) pairs, read file by file to survive the mixed schema."""
    import pandas as pd
    from huggingface_hub import snapshot_download

    root = snapshot_download(repo_id, repo_type="dataset", allow_patterns=["data/*.parquet"])
    frames = []
    for path in sorted(glob.glob(f"{root}/data/*.parquet")):
        df = pd.read_parquet(path)
        lang_col = "target_language" if "target_language" in df else "language"
        frames.append(
            df[["prompt", "chosen", "rejected"]].assign(language=df[lang_col].astype(str))
        )
    df = pd.concat(frames, ignore_index=True)
    return [
        {
            "id": pair_id(r.prompt, r.chosen, r.rejected),
            "prompt": r.prompt,
            "chosen": r.chosen,
            "rejected": r.rejected,
            "language": r.language,
        }
        for r in df.itertuples(index=False)
    ]


def clean_pairs(pairs: Sequence[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Drop pairs that cannot carry a preference: empty fields, identical responses, duplicates."""
    kept, seen = [], set()
    dropped = {"empty_field": 0, "identical_responses": 0, "duplicate": 0}
    for p in pairs:
        if not all(p[k] and p[k].strip() for k in ("prompt", "chosen", "rejected")):
            dropped["empty_field"] += 1
        elif p["chosen"].strip() == p["rejected"].strip():
            dropped["identical_responses"] += 1
        elif p["id"] in seen:
            dropped["duplicate"] += 1
        else:
            seen.add(p["id"])
            kept.append(p)
    return kept, dropped


# --------------------------------------------------------------------- symmetrization


def symmetrize(pairs: Sequence[dict[str, str]], seed: int = SEED) -> list[dict[str, Any]]:
    """Place `chosen` in slot A for a random half of the pairs and in slot B for the rest.

    `label` is 0 when candidate_a is the chosen response and 1 when candidate_b is.
    Exactly half (rounded down) get label 1, at random positions, so the split is balanced.
    """
    rng = random.Random(seed)
    flags = [i < len(pairs) // 2 for i in range(len(pairs))]
    rng.shuffle(flags)
    out = []
    for p, swap in zip(pairs, flags, strict=True):
        a, b = (p["rejected"], p["chosen"]) if swap else (p["chosen"], p["rejected"])
        out.append(
            {
                "id": p["id"],
                "prompt": p["prompt"],
                "candidate_a": a,
                "candidate_b": b,
                "label": int(swap),
            }
        )
    return out


def make_state(prompt: str, candidate_a: str, candidate_b: str) -> dict[str, str]:
    return {"prompt": prompt, "candidate_a": candidate_a, "candidate_b": candidate_b}


# ------------------------------------------------------------------------- truncation


def internal_question():
    from laya.train import to_internal

    return to_internal(QID, QUESTION[QID])


def state_room(tok, max_len: int = MAX_LEN, head_max_len: int = HEAD_MAX_LEN) -> int:
    """State tokens that fit after the question head and the closing [SEP]."""
    from laya.common import state_room as _room

    return _room(tok, internal_question(), max_len, head_max_len)


def count_state_tokens(tok, state: dict[str, str]) -> int:
    from laya.common import encode_text, serialize_state

    text = serialize_state(state).replace(tok.mask_token, " ")
    return len(encode_text(tok, text, add_special_tokens=False)["input_ids"])


def _offsets(tok, text: str) -> list[tuple[int, int]]:
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return [tuple(o) for o in enc["offset_mapping"]]


def head_tokens(text: str, offsets: list[tuple[int, int]], n: int) -> str:
    """The text covered by the first `n` tokens. Cuts fall on token boundaries, which are
    always character boundaries, so a multi-byte character such as å is never split."""
    if n >= len(offsets):
        return text
    if n <= 0:
        return ""
    return text[: offsets[n - 1][1]]


def tail_tokens(text: str, offsets: list[tuple[int, int]], n: int) -> str:
    """The text covered by the last `n` tokens."""
    if n >= len(offsets):
        return text
    if n <= 0:
        return ""
    return text[offsets[len(offsets) - n][0] :]


def waterfill(lengths: Sequence[int], budget: int) -> list[int]:
    """Max-min fair allocation: short fields keep everything, long fields share the rest equally."""
    alloc = [0] * len(lengths)
    remaining = list(range(len(lengths)))
    left = max(0, budget)
    while remaining:
        share = left // len(remaining)
        short = [i for i in remaining if lengths[i] <= share]
        if not short:
            for j, i in enumerate(remaining):
                alloc[i] = share + (1 if j < left - share * len(remaining) else 0)
            return alloc
        for i in short:
            alloc[i] = lengths[i]
            left -= lengths[i]
            remaining.remove(i)
    return alloc


def truncate_triplet(
    tok,
    prompt: str,
    candidate_a: str,
    candidate_b: str,
    room: int,
    min_response_tokens: int = 16,
    max_iter: int = 20,
) -> tuple[dict[str, str], bool, int]:
    """Fit the serialized state into `room` tokens.

    Keeps the tail of the prompt (most recent context) and the head of each response. The
    model reads the fields JSON-serialized, where escapes (newlines, control characters) can
    cost far more tokens than the raw text, so the budget is water-filled over each field's
    *serialized* token count and mapped back to a cut in the field's own tokens. A short field
    is never cut, and each response keeps at least `min_response_tokens` of its own tokens.
    The serialized state is re-measured after every cut, shrinking until it fits.

    Returns `(state, truncated, n_state_tokens)`.
    """
    fields = (prompt, candidate_a, candidate_b)
    state = make_state(*fields)
    n_full = count_state_tokens(tok, state)
    if n_full <= room:
        return state, False, n_full

    offs = [_offsets(tok, t) for t in fields]
    raw = [len(o) for o in offs]
    ser = [
        max(1, count_state_tokens(tok, {"": t}) - count_state_tokens(tok, {"": ""})) for t in fields
    ]
    floors = [0] + [min(r, min_response_tokens) for r in raw[1:]]
    overhead = count_state_tokens(tok, make_state("", "", ""))
    budget = room - overhead
    if budget < 3 * min_response_tokens:
        raise ValueError(f"state room {room} leaves no space for the responses")
    for _ in range(max_iter):
        alloc = waterfill(ser, budget)
        keep = [
            min(r, max(f, int(a * r / s)))
            for r, s, a, f in zip(raw, ser, alloc, floors, strict=True)
        ]
        state = make_state(
            tail_tokens(prompt, offs[0], keep[0]),
            head_tokens(candidate_a, offs[1], keep[1]),
            head_tokens(candidate_b, offs[2], keep[2]),
        )
        n = count_state_tokens(tok, state)
        if n <= room:
            return state, True, n
        # Token merges at field edges can still overshoot; shrink from what was allocated.
        budget = min(budget, sum(alloc)) - (n - room)
    raise RuntimeError("could not fit the state into the token budget")


# --------------------------------------------------------------------------- encoding


def build_items(tok, examples: Sequence[dict[str, Any]], room: int | None = None):
    """Truncate and tokenize examples into laya training items, plus truncation stats."""
    from laya.train import encode_state, make_item

    q = internal_question()
    room = state_room(tok) if room is None else room
    items, stats = [], []
    for ex in examples:
        state, truncated, n_state = truncate_triplet(
            tok, ex["prompt"], ex["candidate_a"], ex["candidate_b"], room
        )
        target = [1.0, 0.0] if ex["label"] == 0 else [0.0, 1.0]
        item, reason = make_item(tok, q, target, encode_state(tok, state, MAX_LEN), HEAD_MAX_LEN)
        if item is None:
            raise ValueError(f"question cannot be encoded: {reason}")
        item["label"] = ex["label"]
        item["id"] = ex["id"]
        items.append(item)
        stats.append({"truncated": truncated, "state_tokens": n_state})
    return items, stats


def full_sequence_tokens(tok, ex: dict[str, Any], head_len: int) -> int:
    """Length the model input would have without truncation: head + state + [SEP]."""
    state = make_state(ex["prompt"], ex["candidate_a"], ex["candidate_b"])
    return head_len + count_state_tokens(tok, state) + 1


# ------------------------------------------------------------------------------ split


def split_examples(examples: Sequence[dict[str, Any]], val_frac: float = 0.15, seed: int = SEED):
    """Stratified 85/15 train/val split; val is then halved (stratified) into calib/eval.

    Pairs that share a prompt are kept on the training side so no prompt leaks across splits.
    """
    from sklearn.model_selection import train_test_split

    labels = [e["label"] for e in examples]
    train, val = train_test_split(
        list(examples), test_size=val_frac, stratify=labels, random_state=seed
    )
    train_prompts = {e["prompt"] for e in train}
    leaked = [e for e in val if e["prompt"] in train_prompts]
    val = [e for e in val if e["prompt"] not in train_prompts]
    train = train + leaked
    calib, evaluation = train_test_split(
        val, test_size=0.5, stratify=[e["label"] for e in val], random_state=seed
    )
    return {"train": train, "val_calib": calib, "val_eval": evaluation}, len(leaked)


# ------------------------------------------------------------------------- baselines


def shortcut_baselines(train, evaluation, top_k: int = 15) -> dict[str, Any]:
    """Cheap non-neural baselines on val_eval, to show how much of the task is lexical.

    - longer response wins
    - logistic regression on the TF-IDF difference (candidate_a - candidate_b)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    y_tr = np.array([r["label"] for r in train])
    y_ev = np.array([r["label"] for r in evaluation])
    longer_b = np.array([len(r["candidate_b"]) > len(r["candidate_a"]) for r in evaluation])
    vec = TfidfVectorizer(min_df=5, max_features=50_000, sublinear_tf=True)
    vec.fit([r[k] for r in train for k in ("candidate_a", "candidate_b")])

    def feats(rows):
        a = vec.transform([r["candidate_a"] for r in rows])
        return a - vec.transform([r["candidate_b"] for r in rows])

    clf = LogisticRegression(max_iter=2000, C=4.0).fit(feats(train), y_tr)
    names, w = vec.get_feature_names_out(), clf.coef_[0]
    return {
        "split": "val_eval",
        "longer_response_wins_accuracy": float((longer_b.astype(int) == y_ev).mean()),
        "tfidf_logreg_accuracy": float((clf.predict(feats(evaluation)) == y_ev).mean()),
        "words_favouring_chosen": [str(x) for x in names[np.argsort(w)[:top_k]]],
        "words_favouring_rejected": [str(x) for x in names[np.argsort(w)[-top_k:][::-1]]],
        "note": "label 1 = candidate_b chosen, so negative weights favour candidate_a; lists "
        "are mapped to chosen/rejected accordingly",
    }


# ------------------------------------------------------------------------------- main


def fetch_licenses() -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    out = {}
    for repo, kind in ((BASE_MODEL_REPO, "model"), (DATASET_REPO, "dataset")):
        info = api.repo_info(repo, repo_type=kind)
        card = info.card_data.to_dict() if info.card_data else {}
        out[repo] = card.get("license")
    permissive = {"apache-2.0", "mit", "bsd-3-clause", "cc-by-4.0"}
    out["redistribution_allowed"] = all(
        (v or "").lower() in permissive for k, v in out.items() if k != "redistribution_allowed"
    )
    out["derivative_license"] = "apache-2.0"
    return out


def load_tokenizer():
    from laya.agent import _fix_tokenizer_config
    from laya.train import resolve_checkpoint_dir
    from transformers import AutoTokenizer

    model_dir = resolve_checkpoint_dir(BASE_MODEL_SUBFOLDER)
    _fix_tokenizer_config(model_dir)
    return AutoTokenizer.from_pretrained(f"{model_dir}/tokenizer")


def main(argv: list[str] | None = None) -> None:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="debug: use only N pairs")
    args = parser.parse_args(argv)
    console = Console()

    raw = load_raw_pairs()
    pairs, dropped = clean_pairs(raw)
    if args.limit:
        pairs = pairs[: args.limit]
    console.print(f"[bold]data[/] raw={len(raw)} kept={len(pairs)} dropped={dropped}")

    examples = symmetrize(pairs)
    splits, n_leaked = split_examples(examples)

    tok = load_tokenizer()
    room = state_room(tok)
    from laya.common import build_head

    head_len = len(build_head(tok, internal_question(), HEAD_MAX_LEN)[0])

    full_lens, truncated = [], 0
    for name, rows in splits.items():
        for ex in rows:
            n_full = full_sequence_tokens(tok, ex, head_len)
            full_lens.append(n_full)
            truncated += n_full > MAX_LEN
        write_jsonl(SPLITS_DIR / f"{name}.jsonl", rows)

    def balance(rows):
        return float(np.mean([r["label"] for r in rows])) if rows else None

    stats = {
        "source": DATASET_REPO,
        "schema_note": "two parquet schemas in one split; read file-by-file, "
        "columns prompt/chosen/rejected (+ language or target_language)",
        "raw_pairs": len(raw),
        "dropped": dropped,
        "total_pairs": len(examples),
        "languages": dict(
            zip(*np.unique([p["language"] for p in pairs], return_counts=True), strict=True)
        ),
        "splits": {k: len(v) for k, v in splits.items()},
        "train_size": len(splits["train"]),
        "val_size": len(splits["val_calib"]) + len(splits["val_eval"]),
        "val_pairs_moved_to_train_for_shared_prompt": n_leaked,
        "label_balance_frac_chosen_in_b": {k: balance(v) for k, v in splits.items()},
        "max_len": MAX_LEN,
        "head_max_len": HEAD_MAX_LEN,
        "question_head_tokens": head_len,
        "state_room_tokens": room,
        "fraction_truncated": truncated / len(full_lens),
        "n_truncated": int(truncated),
        "token_length_untruncated": {
            "median": float(np.median(full_lens)),
            "p95": float(np.percentile(full_lens, 95)),
            "max": int(max(full_lens)),
        },
        "seed": SEED,
        "licenses": fetch_licenses(),
        "baselines": shortcut_baselines(splits["train"], splits["val_eval"]),
        "construction": "per the dataset card, kimi-k3 localized the prompt and chosen "
        "response and deliberately generated a rejected response with poor translation, "
        "English loan words or US-centric content",
    }
    stats["languages"] = {str(k): int(v) for k, v in stats["languages"].items()}
    update_metrics("data", stats)
    console.print_json(json.dumps({k: v for k, v in stats.items() if k != "licenses"}))


if __name__ == "__main__":
    main()
