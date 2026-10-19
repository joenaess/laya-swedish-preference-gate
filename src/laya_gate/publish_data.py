"""Publish the derived v4 training and evaluation pairs as a Hugging Face dataset.

    python -m laya_gate.publish_data            # dry run: build the configs, print the card
    python -m laya_gate.publish_data --upload   # push to the Hub (public), needs HF_TOKEN

One config per source, each under the license of the text it contains. `huggingface_hub` reads
HF_TOKEN from the environment; it is never passed, printed or written. Pilot outputs of the 22
small LLMs are not included (their model licenses differ); only their scores are in
`results/metrics.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from . import external_eval as X
from .utils import GITHUB_REPO, read_jsonl, update_metrics
from .v4 import V4_DIR

DATA_REPO = "jonasaise/laya-swedish-gate-v4-data"
EUROPARL_CITATION = (
    "Koehn, Philipp (2005). Europarl: A Parallel Corpus for Statistical Machine Translation. "
    "MT Summit X. Swedish-English files enriched and distributed by Språkbanken "
    "(https://doi.org/10.23695/mdp4-2p72), CC BY 4.0."
)


def _pairs_v2(rows: list[dict[str, Any]], question: str) -> list[dict[str, Any]]:
    """v2-layout rows (candidate_a/b + label) as reference/other pairs."""
    out = []
    for r in rows:
        ref = r["candidate_a"] if r["label"] == 0 else r["candidate_b"]
        other = r["candidate_b"] if r["label"] == 0 else r["candidate_a"]
        out.append(
            {
                "id": r["id"],
                "question": question,
                "prompt": r["prompt"],
                "preferred": ref,
                "rejected": other,
            }
        )
    return out


def configs() -> dict[str, dict[str, Any]]:
    """name -> {splits: {split: rows}, license, description}."""
    out: dict[str, dict[str, Any]] = {}
    out["europarl_human_vs_kimi"] = {
        "license": "cc-by-4.0",
        "description": "12,000 English-original Europarl paragraphs: professional Swedish "
        "translation (preferred) vs Kimi-k3 machine translation (rejected), length-balanced. "
        "`preference` rows carry the English source in the prompt; `naturalness` rows have no "
        "prompt. Training data of the +Europarl arms. On its own this pairing teaches 'the "
        "less literal side wins' (see the README's sanity tests); pair it with "
        "`europarl_adequacy`.",
        "splits": {
            "train": _pairs_v2(read_jsonl(V4_DIR / "europarl_preference.jsonl"), "preference")
            + _pairs_v2(read_jsonl(V4_DIR / "europarl_naturalness.jsonl"), "naturalness")
        },
    }
    if (V4_DIR / "adequacy_preference.jsonl").exists():
        out["europarl_adequacy"] = {
            "license": "cc-by-4.0",
            "description": "Source-dependent negatives from Europarl paragraphs not in "
            "`europarl_human_vs_kimi`: the correct translation vs another paragraph's "
            "translation, a half translation, a word-shuffled translation, or the untranslated "
            "English (kind in the id suffix).",
            "splits": {
                "train": _pairs_v2(read_jsonl(V4_DIR / "adequacy_preference.jsonl"), "preference")
                + _pairs_v2(read_jsonl(V4_DIR / "adequacy_naturalness.jsonl"), "naturalness")
            },
        }
    ep = read_jsonl(X.EXT_DIR / "europarl.jsonl")
    out["europarl_eval"] = {
        "license": "cc-by-4.0",
        "description": "Evaluation paragraphs (never trained on): every usable Swedish-original "
        "speech and 1,000 English-original ones, with the Swedish and English sides and "
        "OPUS-MT (opus-mt-en-sv) and Kimi-k3 translations of the English.",
        "splits": {
            "test": [
                {
                    "id": r["id"],
                    "original_language": r["original"],
                    "swedish": " ".join(r["sv"]),
                    "english": " ".join(r["en"]),
                    "mt_opus": r.get("mt_opus"),
                    "mt_kimi": r.get("mt_kimi"),
                }
                for r in ep
            ]
        },
    }
    dalaj_train = _pairs_v2(read_jsonl(V4_DIR / "dalaj_naturalness.jsonl"), "naturalness")
    dalaj_test = [
        {
            "id": p["id"],
            "question": "naturalness",
            "prompt": "",
            "preferred": p["ref"],
            "rejected": p["other"],
            "error_types": ",".join(p["types"]),
        }
        for p in read_jsonl(X.EXT_DIR / "dalaj.jsonl")
    ]
    for r in dalaj_train:
        r["error_types"] = ""
    out["dalaj_pairs"] = {
        "license": "cc-by-4.0",
        "description": "DaLAJ-GED (Superlim 2) learner sentences paired with the same sentence "
        "with its error corrected (rebuilt from the annotated span; word-order corrections "
        "that cannot be rebuilt are dropped). Train from the train split, test from the test "
        "split.",
        "splits": {"train": dalaj_train, "test": dalaj_test},
    }
    faq_test = read_jsonl(X.EXT_DIR / "swefaq.jsonl")
    out["swefaq_pairs"] = {
        "license": "cc-by-sa-4.0",
        "description": "SweFAQ (Superlim 2): a question with its own answer (preferred) vs "
        "another answer from the same FAQ page. Train: up to 4 negatives per train question. "
        "Test: every negative for dev + test questions.",
        "splits": {
            "train": _pairs_v2(read_jsonl(V4_DIR / "swefaq_preference.jsonl"), "preference"),
            "test": [
                {
                    "id": p["id"],
                    "question": "preference",
                    "prompt": p["prompt"],
                    "preferred": p["ref"],
                    "rejected": p["other"],
                }
                for p in faq_test
            ],
        },
    }
    wmt = read_jsonl(X.EXT_DIR / "wmt24pp.jsonl")
    out["wmt24pp_en_sv_mt"] = {
        "license": "apache-2.0",
        "description": "WMT24++ en->sv (google/wmt24pp, segments not marked bad): English "
        "source, post-edited Swedish reference, OPUS-MT and Kimi-k3 translations.",
        "splits": {
            "test": [
                {
                    "id": r["id"],
                    "domain": r["domain"],
                    "english": " ".join(r["en"]),
                    "reference": " ".join(r["sv"]),
                    "mt_opus": r.get("mt_opus"),
                    "mt_kimi": r.get("mt_kimi"),
                }
                for r in wmt
            ]
        },
    }
    return out


def dataset_card(cfgs: dict[str, dict[str, Any]]) -> str:
    yaml_cfgs = "\n".join(
        f"- config_name: {name}\n  data_files:\n"
        + "\n".join(f"  - split: {s}\n    path: {name}/{s}-*" for s in c["splits"])
        for name, c in cfgs.items()
    )
    rows = "\n".join(
        f"| `{name}` | {', '.join(f'{s} {len(v):,}' for s, v in c['splits'].items())} | "
        f"{c['license']} | {c['description']} |"
        for name, c in cfgs.items()
    )
    return f"""---
language: [sv, en]
license: other
license_name: per-config
license_details: "Each config keeps the license of its source text: see the table."
pretty_name: Laya Swedish gate v4 training and evaluation pairs
task_categories: [text-classification, translation]
tags: [preference, swedish, translationese, grammaticality, europarl, superlim]
configs:
{yaml_cfgs}
---

# Laya Swedish gate v4: training and evaluation pairs

The derived pairs behind the v4 ablation of the
[Laya Swedish preference gate](https://huggingface.co/jonasaise/laya-swedish-preference-gate)
(code and full results: [github.com/{GITHUB_REPO}](https://github.com/{GITHUB_REPO})). Each row
has a `preferred` and a `rejected` text for one question (`preference`: which response or
translation is better, with the prompt or English source in `prompt`; `naturalness`: which is
more natural Swedish, no prompt).

| Config | Rows | License | Contents |
|---|---|---|---|
{rows}

Machine translations: `Helsinki-NLP/opus-mt-en-sv` (OPUS-MT) and Kimi-k3 (reasoning off), served
on a self-hosted endpoint.

## Sources and licenses

- **Europarl** sv/en: {EUROPARL_CITATION}
- **DaLAJ-GED** and **SweFAQ**: [Superlim 2](https://huggingface.co/datasets/sbx/superlim-2)
  (Språkbanken), DaLAJ-GED CC BY 4.0, SweFAQ CC BY-SA 4.0. Cite the Superlim 2 paper and the
  individual datasets as listed on the Superlim card.
- **WMT24++**: [google/wmt24pp](https://huggingface.co/datasets/google/wmt24pp), Apache-2.0.

## Caveats

- `europarl_human_vs_kimi` on its own is a shortcut trap: the human side is always the less
  literal one, and a gate trained only on it learns "less source-like wins" (it then prefers a
  fluent translation of the wrong sentence). The README documents the sanity tests that showed
  this and the `europarl_adequacy` negatives added to fix it.
- Europarl's "Swedish-original" paragraphs rely on its `speakerlang` tag; only 1,587 of
  1,162,185 Swedish paragraphs carry `speakerlang="SV"`.
"""


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo", default=os.environ.get("HF_DATA_REPO_ID", DATA_REPO))
    args = parser.parse_args(argv)
    cfgs = configs()
    card = dataset_card(cfgs)
    summary = {
        name: {"license": c["license"], "rows": {s: len(v) for s, v in c["splits"].items()}}
        for name, c in cfgs.items()
    }
    print(json.dumps(summary, indent=2))
    if not args.upload:
        print(card)
        return
    if not os.environ.get("HF_TOKEN"):
        print("data upload skipped: HF_TOKEN not set")
        sys.exit(0)
    from datasets import Dataset
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=False, exist_ok=True)
    for name, c in cfgs.items():
        for split, rows in c["splits"].items():
            Dataset.from_list(rows).push_to_hub(
                args.repo, config_name=name, split=split, private=False
            )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
        commit_message="Dataset card",
    )
    info = api.repo_info(args.repo, repo_type="dataset")
    out = {
        "status": "uploaded",
        "repo_id": args.repo,
        "url": f"https://huggingface.co/datasets/{args.repo}",
        "visibility": "private" if info.private else "public",
        "configs": summary,
    }
    update_metrics("publish_data", out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
