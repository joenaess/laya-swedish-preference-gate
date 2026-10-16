"""Naturalness checking for machine translation, on data no Kimi-k3 model touched.

    python -m laya_gate.mtqa translate   # FLORES+ English -> Swedish with opus-mt-en-sv (cached)
    python -m laya_gate.mtqa evaluate    # score human vs machine translations with v1/v2/v3

FLORES+ (`openlanguagedata/flores_plus`, dev + devtest) pairs each English sentence with a
professional Swedish translation. The same English sentence is machine-translated with
`Helsinki-NLP/opus-mt-en-sv`. A naturalness checker should rate the professional translation as
more natural than the machine translation. Both are translations of the same sentence, so length
and content are matched; neither was written by Kimi-k3. The professional version is itself a
translation, not native Swedish, so this measures "professional vs MT", a harder contrast than
the deliberate translationese the gate was trained on.
"""

from __future__ import annotations

import argparse
import json
import time
import unicodedata
from typing import Any

import numpy as np

from .probe import bootstrap_ci, summarize
from .utils import CACHE_DIR, ROOT, read_jsonl, update_metrics, write_jsonl

FLORES = "openlanguagedata/flores_plus"
MT_MODEL = "Helsinki-NLP/opus-mt-en-sv"
MT_DIR = CACHE_DIR / "mtqa"
PAIRS_PATH = MT_DIR / "flores_en_sv_mt.jsonl"
PROMPT = "Översätt följande text till svenska:\n"


def load_flores() -> list[dict[str, str]]:
    """English and Swedish FLORES+ sentences aligned by id (dev + devtest)."""
    from huggingface_hub import hf_hub_download

    rows = []
    for split in ("dev", "devtest"):
        sides = {}
        for lang in ("eng_Latn", "swe_Latn"):
            path = hf_hub_download(FLORES, f"{split}/{lang}.jsonl", repo_type="dataset")
            sides[lang] = {r["id"]: r["text"] for r in read_jsonl(path)}
        for i, en in sides["eng_Latn"].items():
            if i in sides["swe_Latn"]:
                rows.append({"id": f"{split}-{i}", "en": en, "human": sides["swe_Latn"][i]})
    return rows


def translate(args) -> None:
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    rows = load_flores()
    device = torch.device(args.device)
    tok = MarianTokenizer.from_pretrained(MT_MODEL)
    model = MarianMTModel.from_pretrained(MT_MODEL).to(device).eval()
    t = time.perf_counter()
    for s in range(0, len(rows), args.batch_size):
        chunk = rows[s : s + args.batch_size]
        enc = tok([r["en"] for r in chunk], return_tensors="pt", padding=True, truncation=True)
        with torch.inference_mode():
            gen = model.generate(**enc.to(device), num_beams=4, max_new_tokens=256)
        for r, out in zip(chunk, tok.batch_decode(gen, skip_special_tokens=True), strict=True):
            r["mt"] = out
    write_jsonl(PAIRS_PATH, rows)
    print(f"translated {len(rows)} sentences in {time.perf_counter() - t:.0f}s on {device}")


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split()).casefold()


def _p_natural(gate, texts, prompts) -> np.ndarray:
    return np.array([x["p_natural"] for x in gate.score_naturalness_batch(texts, prompts, 32)])


def evaluate(args) -> None:
    from .predictor import SwedishPreferenceGate
    from .v3 import V3_DIR, roc_auc

    rows = read_jsonl(PAIRS_PATH)
    kept = [r for r in rows if _norm(r["mt"]) != _norm(r["human"])]
    prompts = [PROMPT + r["en"] for r in kept]
    human, mt = [r["human"] for r in kept], [r["mt"] for r in kept]
    out: dict[str, Any] = {
        "source": f"{FLORES} (dev + devtest), professional English->Swedish translations",
        "mt_system": MT_MODEL,
        "n_sentences": len(rows),
        "n_identical_dropped": len(rows) - len(kept),
        "n_pairs": len(kept),
        "prompt": "'" + PROMPT.strip() + "' + English source",
        "mean_chars": {
            "human": float(np.mean([len(x) for x in human])),
            "mt": float(np.mean([len(x) for x in mt])),
        },
        "models": {},
    }

    def pairwise(gate, qid):
        """P(human wins) per pair, averaged over both slot orders."""
        fwd = [gate.encode(p, h, m, qid) for p, h, m in zip(prompts, human, mt, strict=True)]
        rev = [gate.encode(p, m, h, qid) for p, h, m in zip(prompts, human, mt, strict=True)]
        pf = gate.probabilities(gate.logits(fwd, 16), qid)[:, 0]
        pr = 1 - gate.probabilities(gate.logits(rev, 16), qid)[:, 0]
        avg = (pf + pr) / 2
        return {
            "p_human_wins_mean": float(avg.mean()),
            "human_preferred": summarize(avg > 0.5),
            "positional_consistency": float(((pf > 0.5) == (pr > 0.5)).mean()),
            "confident_human": float((avg >= 0.7).mean()),
            "confident_mt": float((avg <= 0.3).mean()),
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
        res: dict[str, Any] = {"preference": pairwise(gate, "preference")}
        if gate.has_naturalness:
            res["naturalness"] = pairwise(gate, "naturalness")
        if gate.has_pointwise:
            for cond, ctx in (("with_source", prompts), ("no_prompt", [""] * len(kept))):
                ph, pm = _p_natural(gate, human, ctx), _p_natural(gate, mt, ctx)
                better = (ph > pm).astype(float)
                res[f"natural_swedish/{cond}"] = {
                    "auc_human_vs_mt": roc_auc(ph, pm),
                    "within_pair_human_higher": {
                        "accuracy": float(better.mean()),
                        "ci95": bootstrap_ci(better),
                        "n": len(better),
                    },
                    "mean_p_natural": {"human": float(ph.mean()), "mt": float(pm.mean())},
                    "flagged_translationese": {
                        "human": float((ph < 0.5).mean()),
                        "mt": float((pm < 0.5).mean()),
                    },
                }
        out["models"][name] = res
        del gate
    update_metrics("mt_naturalness", out)
    print(json.dumps(out, indent=2))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["translate", "evaluate"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    {"translate": translate, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
