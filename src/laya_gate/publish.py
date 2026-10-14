"""Upload the calibrated checkpoint to the Hugging Face Hub with a model card.

The repo is created private; `--public` makes it public after a successful upload.

Runs only when HF_TOKEN is set, the licenses allow redistribution, every test passed and
post-calibration ECE meets the target. `huggingface_hub` reads HF_TOKEN from the environment
itself; the token is never passed explicitly, printed or written to disk.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from . import readme
from .utils import (
    CHECKPOINT_DIR,
    DEFAULT_HF_REPO,
    ECE_TARGET,
    GITHUB_REPO,
    METRICS_PATH,
    load_metrics,
    update_metrics,
)

UPLOAD_FILES = ("model.safetensors", "rl_agent_config.json", "temperature.json")
UPLOAD_DIRS = ("encoder", "tokenizer")


def blockers(m: dict) -> list[str]:
    out = []
    if not os.environ.get("HF_TOKEN"):
        out.append("HF_TOKEN not set")
    if not readme.get(m, "data.licenses.redistribution_allowed"):
        out.append("licenses do not clearly permit redistribution")
    tests = m.get("tests")
    if not (isinstance(tests, dict) and tests.get("all_passed")):
        out.append("not all tests passed (or tests were not run)")
    ece = readme.get(m, "calibration.ece_post")
    if not (isinstance(ece, float) and ece <= ECE_TARGET):
        out.append(f"post-calibration ECE {ece} is not <= {ECE_TARGET}")
    if not (CHECKPOINT_DIR / "model.safetensors").exists():
        out.append("no trained checkpoint")
    return out


def model_card(m: dict, repo_id: str) -> str:
    g = readme.get
    lic = g(m, "data.licenses", {})
    acc, ece = g(m, "calibration.accuracy"), g(m, "calibration.ece_post")
    lat = g(m, "benchmark.latency.gpu_forward_ms.p50")
    base = g(m, "data.baselines.tfidf_logreg_accuracy")
    header = f"""---
language: [sv]
license: apache-2.0
base_model: convaiinnovations/laya
datasets: [jonasaise/dpo-swedish-interpreted]
pipeline_tag: text-classification
tags: [preference-model, reward-model, dpo, swedish, laya, mmbert, llm-as-judge]
model-index:
- name: laya-swedish-preference-gate
  results:
  - task: {{type: text-classification, name: Pairwise preference}}
    dataset: {{type: jonasaise/dpo-swedish-interpreted, name: dpo-swedish-interpreted (held-out val_eval)}}
    metrics:
    - {{type: accuracy, value: {acc if acc is not None else "null"}}}
    - {{type: ece, value: {ece if ece is not None else "null"}, name: Expected calibration error (15 bins)}}
---
"""
    return f"""{header}
# Laya Swedish Preference Gate

**A ~{readme.num(lat, 0)} ms pairwise judge for Swedish text.** Give it a Swedish prompt and two
candidate responses; it returns a calibrated probability that the first one is better, from a
single forward pass of a fine-tuned [Laya](https://github.com/NandhaKishorM/laya) decision model
(mmBERT-base backbone). It is built as a "System 1" filter: settle the confident comparisons in
milliseconds and escalate the uncertain ones to a slow LLM judge ("System 2").

> **What it actually learned.** The training data pairs an idiomatic, culturally localized
> Swedish answer (chosen) with a *deliberately* poor translation (rejected: literal phrasing,
> English loan words, US-centric content). The gate is therefore, first and foremost, an
> **authentic-Swedish vs. translationese detector**. It reaches {readme.pct(acc)} on held-out
> pairs, but a bag-of-words baseline reaches {readme.pct(base)} on the same pairs. Its ability to
> rank two *equally fluent* answers by helpfulness or correctness has **not** been measured.
> See [Limitations](#limitations).

Code, pipeline and full results: [github.com/{GITHUB_REPO}](https://github.com/{GITHUB_REPO}).

## Results at a glance

{readme.results_table(m)}

## Model details

| | |
|---|---|
| Model type | Laya decision model: bidirectional encoder + typed decision head, `choice` primitive with two options |
| Backbone | `jhu-clsp/mmBERT-base` (ModernBERT, 22 layers, hidden 768, 256k vocab) |
| Fine-tuned from | `convaiinnovations/laya`, subfolder `multilingual` |
| Parameters | ~322M |
| Language | Swedish (sv) |
| Input | prompt + two candidate responses, up to 1024 tokens in total (longer inputs are truncated, see below) |
| Output | calibrated P(candidate A is better), plus the opposite probability |
| Calibration | temperature scaling, T = {readme.num(g(m, "calibration.temperature"))} (stored in `temperature.json` and `rl_agent_config.json`) |
| Weights | `model.safetensors` (fp16), loadable with this repo's package or plain `laya` |
| License | Apache-2.0 |

## How to use

### Install

```bash
pip install "git+https://github.com/{GITHUB_REPO}"
# or: uv pip install "git+https://github.com/{GITHUB_REPO}"
```

This installs `laya_gate` with `laya`, `torch` and `transformers`. The weights download from this
repository on first use. A CUDA GPU is recommended (≈{readme.num(lat, 0)} ms per pair); CPU works
(≈{readme.num(g(m, "benchmark.latency.cpu_forward_ms.p50"), 0)} ms per pair).

### Score one pair

```python
from laya_gate.predictor import SwedishPreferenceGate

gate = SwedishPreferenceGate("{repo_id}")  # defaults: margin=0.2, CUDA if available

result = gate.predict_preference(
    prompt="Hur fungerar fotosyntesen?",
    chosen_candidate="Fotosyntesen är den process där växter med hjälp av solljus omvandlar "
                     "koldioxid och vatten till socker och syre.",
    rejected_candidate="Fotosyntes är en process som involverar växter och baserat på ljus de "
                       "producerar energi, vänligen notera att detta är viktigt.",
)
print(result)
# {{'preference': 'candidate_a', 'p_chosen': 0.98, 'is_valid_margin': True,
#  'probabilities': {{'candidate_a': 0.98, 'candidate_b': 0.02}}}}
```

| Key | Type | Meaning |
|---|---|---|
| `preference` | `"candidate_a"` / `"candidate_b"` | the response the model prefers |
| `p_chosen` | float in [0, 1] | calibrated probability that `candidate_a` (your `chosen_candidate`) is better |
| `is_valid_margin` | bool | `abs(p_chosen - 0.5) >= margin`; with the default `margin=0.2`, true when either side has ≥ 0.70 |
| `probabilities` | dict | both option probabilities (sum to 1) |

`chosen_candidate` / `rejected_candidate` are just slot A and slot B: the names describe your
hypothesis, not ground truth.

### Score many pairs (batched)

```python
results = gate.predict_preference_batch(
    prompts=["Vad är huvudstaden i Sverige?", "Skriv en mening om hösten."],
    chosen_candidates=["Huvudstaden är Stockholm.", "Löven blir röda och gula."],
    rejected_candidates=["Huvudstaden av Sverige är Stockholm City.", "Höst är en säsong."],
    batch_size=8,   # 8 was the fastest batch size on a 12 GB GPU at up to 1024 tokens/pair
)
```

### Remove residual position bias (score both orders)

```python
ab = gate.predict_preference(prompt, resp_1, resp_2)["p_chosen"]
ba = gate.predict_preference(prompt, resp_2, resp_1)["p_chosen"]
p_resp_1_better = (ab + (1 - ba)) / 2
```

On held-out pairs, averaging both orders raised accuracy from
{readme.pct(g(m, "benchmark.positional.accuracy_both_orders"))} to
{readme.pct(g(m, "benchmark.positional.accuracy_order_averaged_prob"))}.

### Best-of-N reranking

```python
from itertools import permutations

def best_of_n(gate, prompt, candidates):
    pairs = list(permutations(range(len(candidates)), 2))        # every ordered pair
    res = gate.predict_preference_batch(
        [prompt] * len(pairs),
        [candidates[i] for i, _ in pairs],
        [candidates[j] for _, j in pairs],
    )
    score = [0.0] * len(candidates)
    for (i, j), r in zip(pairs, res):
        score[i] += r["p_chosen"]
        score[j] += 1 - r["p_chosen"]
    return max(range(len(candidates)), key=score.__getitem__), score
```

### Cascade: gate first, LLM judge only when unsure

```python
def judge_pair(prompt, a, b, llm_judge, gate=gate):
    r = gate.predict_preference(prompt, a, b)
    if r["is_valid_margin"]:
        return r["preference"], "gate"
    return llm_judge(prompt, a, b), "llm"     # your own (slow) LLM-as-judge call
```

With the measured numbers (see the cascade table under Evaluation), the default
`margin=0.2` routes {readme.pct(_routed_at(m, 0.2))} of comparisons to the LLM.

### Without this package (plain `laya`)

The checkpoint uses the standard Laya layout, and the fitted temperature is in its config:

```python
import laya

agent = laya.load("{repo_id}")
question = {{"preference": {{
    "type": "choice",
    "instructions": "Which candidate response better answers the prompt?",
    "criteria": {{"candidate_a": "response A is superior",
                  "candidate_b": "response B is superior"}},
}}}}
state = {{"prompt": "Hur fungerar fotosyntesen?", "candidate_a": "...", "candidate_b": "..."}}
answer = agent.predict_batch([state], question)[0]["answers"]["preference"]
print(answer["choice"], answer["probabilities"])
```

The question text and option labels must match exactly what is shown above (the model was
trained on them). **Caveat:** plain `laya` truncates an over-long state from the end, which can
cut `candidate_b` away entirely. `SwedishPreferenceGate` truncates each field fairly
(the tail of the prompt, the head of each response), so prefer it whenever inputs can exceed
~1000 tokens.

## Intended use

- Fast first-pass ranking or filtering of Swedish LLM outputs where **naturalness and
  localization** matter: catching translationese, English loan-word overuse, literal phrasing
  or US-centric answers.
- The cheap first stage of an LLM-as-judge cascade, routing low-margin pairs onward.
- Flagging suspicious pairs in Swedish preference datasets.

**Out of scope:** fact-checking; safety or toxicity classification; ranking two fluent, native
answers by helpfulness (not evaluated); languages other than Swedish; any high-stakes decision
without human review.

## Training data

{readme.dataset_section(m)}

## Training procedure

{readme.training_section(m)}

## Evaluation

All evaluation uses **val_eval**, 1,118 pairs that were never used for training, checkpoint
selection or the temperature fit.

### Calibration

{readme.calibration_section(m)}

### Latency and throughput

{readme.latency_section(m)}

{readme.positional_section(m)}

### System 1 vs System 2 (Kimi-k3 judge)

{readme.judge_section(m)}

## Limitations

{readme.limitations({**m, "publish": {"status": "uploaded"}})}

## Files

| File | Contents |
|---|---|
| `model.safetensors` | decision-model weights (encoder + head), fp16 |
| `rl_agent_config.json` | Laya config: `max_len` 1024, `head_max_len` 256, fitted temperature |
| `temperature.json` | the fitted temperature, read by `SwedishPreferenceGate` |
| `encoder/config.json` | ModernBERT (mmBERT-base) encoder config |
| `tokenizer/` | tokenizer files |
| `results/metrics.json` | every number on this card, from the training/evaluation run |

## Citation and acknowledgements

- Laya decision models and training recipe: [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya), base checkpoint [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya).
- Encoder: [`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base).
- Data: [`jonasaise/dpo-swedish-interpreted`](https://huggingface.co/datasets/jonasaise/dpo-swedish-interpreted), derived from `argilla/ultrafeedback-binarized-preferences-cleaned`.
- Judge model for the comparison: Kimi-k3, self-hosted (8× NVIDIA B300, HGX).

```bibtex
@software{{laya_swedish_preference_gate,
  title  = {{Laya Swedish Preference Gate}},
  author = {{Lind, Jonas}},
  year   = {{2026}},
  url    = {{https://huggingface.co/{repo_id}}}
}}
```

## License

Apache-2.0. Base model license: {lic.get("convaiinnovations/laya", "see repo")}; dataset license:
{lic.get("jonasaise/dpo-swedish-interpreted", "see repo")}. The dataset is derived from
UltraFeedback; check the upstream terms before commercial use.
"""


def _routed_at(m: dict, threshold: float):
    for r in readme.get(m, "judge.cascade", []) or []:
        if abs(r["threshold"] - threshold) < 1e-9:
            return r["routed_fraction"]
    return None


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--public", action="store_true", help="make the repo public after a successful upload"
    )
    args = parser.parse_args(argv)
    m = load_metrics()
    problems = blockers(m)
    if problems:
        reason = "upload skipped: " + "; ".join(problems)
        print(reason)
        update_metrics("publish", {"status": "skipped", "reason": reason})
        readme.main()
        sys.exit(0)

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    repo_id = os.environ.get("HF_REPO_ID", DEFAULT_HF_REPO)
    api = HfApi()  # token from HF_TOKEN in the environment
    try:
        api.create_repo(repo_id, private=True, exist_ok=True, repo_type="model")
        info = api.repo_info(repo_id)
    except HfHubHTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        reason = (
            f"upload skipped: the Hub refused to create `{repo_id}` (HTTP {status}); the "
            "HF_TOKEN in use cannot write to that namespace. Set HF_REPO_ID to a namespace the "
            "token can write to, or use a token for the intended account, and rerun "
            "`python -m laya_gate.publish`."
        )
        print(reason)
        update_metrics("publish", {"status": "skipped", "reason": reason})
        readme.main()
        sys.exit(0)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for name in UPLOAD_FILES:
            shutil.copy2(CHECKPOINT_DIR / name, stage / name)
        for name in UPLOAD_DIRS:
            shutil.copytree(CHECKPOINT_DIR / name, stage / name)
        (stage / "results").mkdir()
        shutil.copy2(METRICS_PATH, stage / "results" / "metrics.json")
        (stage / "README.md").write_text(model_card(m, repo_id), encoding="utf-8")
        for f in stage.rglob("*"):  # nothing from the judge cache or the environment
            if f.is_file():
                text = f.read_bytes()
                for var in ("BASE_URL", "API_KEY", "HF_TOKEN"):
                    val = os.environ.get(var)
                    if val and val.encode() in text:
                        raise SystemExit(f"refusing to upload: {f.name} contains {var}")
        commit = api.upload_folder(
            folder_path=str(stage),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload Laya Swedish preference gate checkpoint",
        )

    if args.public:
        api.update_repo_settings(repo_id, private=False)
        info = api.repo_info(repo_id)
    out = {
        "status": "uploaded",
        "repo_id": repo_id,
        "url": f"https://huggingface.co/{repo_id}",
        "visibility": "private" if info.private else "public",
        "commit": getattr(commit, "oid", None),
        "files": sorted([*UPLOAD_FILES, *UPLOAD_DIRS, "results/metrics.json", "README.md"]),
    }
    update_metrics("publish", out)
    readme.main()  # adds the Hub link and Hub quickstart to the GitHub README
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
