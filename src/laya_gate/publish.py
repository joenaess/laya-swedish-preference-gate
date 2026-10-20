"""Upload the calibrated checkpoint to the Hugging Face Hub with a model card.

The repo is created private; `--public` makes it public after a successful upload.
`--checkpoint v2` uploads the v2 checkpoint (preference + naturalness questions) with its own
card; `--tag-previous v1` first tags the revision currently on the Hub so it stays loadable
with `revision="v1"`. `--checkpoint v3` adds the single-response questions (quality score,
natural Swedish) and the int8 ONNX graph for CPU inference.

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
    QUESTIONS_FILE,
    load_metrics,
    update_metrics,
)

UPLOAD_FILES = ("model.safetensors", "rl_agent_config.json", "temperature.json")
UPLOAD_DIRS = ("encoder", "tokenizer")
V2_CHECKPOINT_DIR = CHECKPOINT_DIR.parent / "v2"
V3_CHECKPOINT_DIR = CHECKPOINT_DIR.parent / "v3"
V3_ONNX_FILES = ("onnx/model.int8.onnx",)
V4_CHECKPOINT_DIR = CHECKPOINT_DIR.parent / f"v4-{readme.RELEASE_ARM}"
V4_DATA_REPO = "jonasaise/laya-swedish-gate-v4-data"


def blockers(m: dict, which: str = "v1") -> list[str]:
    out = []
    if not os.environ.get("HF_TOKEN"):
        out.append("HF_TOKEN not set")
    if not readme.get(m, "data.licenses.redistribution_allowed"):
        out.append("licenses do not clearly permit redistribution")
    tests = m.get("tests")
    if not (isinstance(tests, dict) and tests.get("all_passed")):
        out.append("not all tests passed (or tests were not run)")
    if which == "v4":
        sanity = readme.get(m, f"v4.sanity.checkpoints.{readme.RELEASE_ARM}", {})
        for test in ("garbage_mt", "first_half_only", "untranslated"):
            v = readme.get(sanity, f"{test}/preference+source.mean")
            if not (isinstance(v, float) and v >= 0.9):
                out.append(f"v4 sanity test {test} at {v} (needs >= 0.9)")
    if which in ("v2", "v3", "v4"):
        if readme.get(m, "fluent_data.source_license") not in ("mit", "apache-2.0", "cc-by-4.0"):
            out.append("license of the fluent-pair source data not recorded as permissive")
        for src in ("original", "fluent"):
            ece = (
                readme.get(
                    m,
                    f"v4.arms.{readme.RELEASE_ARM}.standard_eval.{src}/preference.ece_both_orders",
                )
                if which == "v4"
                else readme.get(m, f"{which}_eval.{src}/preference.ece_both_orders")
            )
            if not (isinstance(ece, float) and ece <= ECE_TARGET):
                out.append(f"{which} {src} val_eval ECE {ece} is not <= {ECE_TARGET}")
    else:
        ece = readme.get(m, "calibration.ece_post")
        if not (isinstance(ece, float) and ece <= ECE_TARGET):
            out.append(f"post-calibration ECE {ece} is not <= {ECE_TARGET}")
    if not (checkpoint_dir(which) / "model.safetensors").exists():
        out.append(f"no trained {which} checkpoint")
    return out


def checkpoint_dir(which: str) -> Path:
    return {"v2": V2_CHECKPOINT_DIR, "v3": V3_CHECKPOINT_DIR, "v4": V4_CHECKPOINT_DIR}.get(
        which, CHECKPOINT_DIR
    )


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
> pairs, but a bag-of-words baseline reaches {readme.pct(base)} on the same pairs.
> {_probe_line(m)}
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
answers by helpfulness (see the fluent-vs-fluent probe below: well behind an LLM judge, and
over-confident); languages other than Swedish; any high-stakes decision without human review.

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

### Fluent-vs-fluent probe (does it judge quality?)

{readme.probe_section(m)}

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


def v2_results_table(m: dict) -> str:
    g, pct, num, ms = readme.get, readme.pct, readme.num, readme.ms
    ev, v1 = m.get("v2_eval", {}), m.get("v1_eval", {})

    def acc(d, key, kind="accuracy_both_orders"):
        return g(d, f"{key}.{kind}.accuracy")

    def ci(d, key):
        lo = g(d, f"{key}.accuracy_order_averaged_prob.ci95.0")
        hi = g(d, f"{key}.accuracy_order_averaged_prob.ci95.1")
        return f"{pct(acc(d, key, 'accuracy_order_averaged_prob'))} [{pct(lo)}, {pct(hi)}]"

    lat = g(ev, "latency_gpu_forward_ms", {})
    ece_o = g(ev, "original/preference.ece_both_orders")
    ece_f = g(ev, "fluent/preference.ece_both_orders")
    rows = [
        (
            "Original held-out pairs: idiomatic vs translationese (one call)",
            pct(acc(v1, "original/preference")),
            pct(acc(ev, "original/preference")),
            "—",
        ),
        (
            "Fluent held-out pairs: both natural Swedish (one call)",
            pct(acc(v1, "fluent/preference")),
            pct(acc(ev, "fluent/preference")),
            "—",
        ),
        (
            "**Probe**: 299 fluent pairs, both orders averaged [95% CI]",
            ci(v1, "probe/preference"),
            ci(ev, "probe/preference"),
            f"Kimi-k3 judge: {pct(g(m, 'probe.judge.order_averaged.accuracy'))}",
        ),
        (
            "ECE, original / fluent held-out (one call, 15 bins)",
            f"{num(g(v1, 'original/preference.ece_both_orders'))} / "
            f"{num(g(v1, 'fluent/preference.ece_both_orders'))}",
            f"{num(ece_o)} / {num(ece_f)}",
            f"{readme.mark(isinstance(ece_o, float) and isinstance(ece_f, float) and max(ece_o, ece_f) <= ECE_TARGET)} ≤ {ECE_TARGET}",
        ),
        (
            "ECE on the probe (both orders averaged)",
            num(g(v1, "probe/preference.ece_order_averaged")),
            num(g(ev, "probe/preference.ece_order_averaged")),
            "—",
        ),
        (
            f"Probe accuracy on pairs passing margin {readme.DEFAULT_MARGIN} (share passing)",
            f"{pct(g(v1, 'probe/preference.default_margin.accuracy_on_covered'))} "
            f"({pct(g(v1, 'probe/preference.default_margin.coverage'))})",
            f"{pct(g(ev, 'probe/preference.default_margin.accuracy_on_covered'))} "
            f"({pct(g(ev, 'probe/preference.default_margin.coverage'))})",
            "—",
        ),
        (
            "Naturalness question: idiomatic vs translationese (one call)",
            "—",
            pct(acc(ev, "original/naturalness")),
            "—",
        ),
        (
            "GPU forward, batch 1, median / p95",
            f"{ms(g(v1, 'latency_gpu_forward_ms.p50'))} / {ms(g(v1, 'latency_gpu_forward_ms.p95'))}",
            f"{ms(lat.get('p50'))} / {ms(lat.get('p95'))}",
            f"{readme.mark(isinstance(lat.get('p50'), float) and lat['p50'] < readme.LATENCY_TARGET_MS)} < {readme.LATENCY_TARGET_MS:.0f} ms",
        ),
    ]
    out = ["| Metric | v1 | **v2 (this revision)** | Reference / target |", "|---|---|---|---|"]
    out += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    return "\n".join(out)


def model_card_v2(m: dict, repo_id: str) -> str:
    g, pct, num = readme.get, readme.pct, readme.num
    ev = m.get("v2_eval", {})
    lic = g(m, "data.licenses", {})
    probe_acc = g(ev, "probe/preference.accuracy_order_averaged_prob.accuracy")
    lat = g(ev, "latency_gpu_forward_ms.p50")
    rec = g(ev, "probe/cascade.recommended", {})
    t_pref = g(m, "v2_calibration.preference.temperature")
    t_nat = g(m, "v2_calibration.naturalness.temperature")
    header = f"""---
language: [sv]
license: apache-2.0
base_model: convaiinnovations/laya
datasets: [jonasaise/dpo-swedish-interpreted, argilla/ultrafeedback-binarized-preferences-cleaned]
pipeline_tag: text-classification
tags: [preference-model, reward-model, dpo, swedish, laya, mmbert, llm-as-judge]
model-index:
- name: laya-swedish-preference-gate
  results:
  - task: {{type: text-classification, name: Pairwise preference (fluent Swedish pairs)}}
    dataset: {{type: argilla/ultrafeedback-binarized-preferences-cleaned, name: Fluent-vs-fluent probe (299 pairs translated to Swedish)}}
    metrics:
    - {{type: accuracy, value: {probe_acc if probe_acc is not None else "null"}}}
    - {{type: ece, value: {g(ev, "probe/preference.ece_order_averaged") or "null"}, name: Expected calibration error (15 bins)}}
---
"""
    return f"""{header}
# Laya Swedish Preference Gate (v2)

**A ~{num(lat, 0)} ms pairwise judge for Swedish text.** Give it a Swedish prompt and two
candidate responses; it returns a calibrated probability that the first one is better, from a
single forward pass of a fine-tuned [Laya](https://github.com/NandhaKishorM/laya) decision model
(mmBERT-base backbone). It is built as a "System 1" filter: settle the confident comparisons in
milliseconds and escalate the uncertain ones to a slow LLM judge ("System 2").

v2 answers two questions:

- **preference**: which response better answers the prompt. On 299 held-out pairs where *both*
  answers are fluent Swedish and only quality differs, it is right {pct(probe_acc)} of the time
  (v1: {pct(g(m, "v1_eval.probe/preference.accuracy_order_averaged_prob.accuracy"))}; Kimi-k3 as
  judge: {pct(g(m, "probe.judge.order_averaged.accuracy"))}) and its confidence is calibrated
  there too.
- **naturalness**: which response is written in more natural, idiomatic Swedish (the
  translationese detector that v1 effectively was, now as an explicit question).

> **v1 is still available:** `SwedishPreferenceGate("{repo_id}", revision="v1")`. v1 is
> slightly better on the original translationese-vs-idiomatic pairs but much weaker and
> over-confident on fluent pairs; see the comparison below.

Code, pipeline and full results: [github.com/{GITHUB_REPO}](https://github.com/{GITHUB_REPO}).

## Results at a glance

{v2_results_table(m)}

"One call" scores each pair once, in the stored order and in the swapped order, and counts both;
"both orders averaged" averages the two probabilities per pair (more robust, two calls). CPU latency
was measured for v1 only ({num(g(m, "benchmark.latency.cpu_forward_ms.p50"), 0)} ms per pair); v2
has the same architecture and size.

## How to use

### Install

```bash
pip install "git+https://github.com/{GITHUB_REPO}"
# or: uv pip install "git+https://github.com/{GITHUB_REPO}"
```

### Which answer is better?

```python
from laya_gate.predictor import SwedishPreferenceGate

gate = SwedishPreferenceGate("{repo_id}")  # margin=0.2, CUDA if available

result = gate.predict_preference(
    prompt="Hur fungerar fotosyntesen?",
    chosen_candidate="Växter tar upp ljus med klorofyll och använder energin för att bygga "
                     "socker av koldioxid och vatten. Syre bildas som en restprodukt.",
    rejected_candidate="Växter gör mat av solljus.",
)
print(result)
# {{'preference': 'candidate_a', 'p_chosen': <float>, 'is_valid_margin': <bool>,
#  'probabilities': {{'candidate_a': <float>, 'candidate_b': <float>}}}}
```

| Key | Type | Meaning |
|---|---|---|
| `preference` | `"candidate_a"` / `"candidate_b"` | the response the model prefers |
| `p_chosen` | float in [0, 1] | calibrated probability that `candidate_a` (your `chosen_candidate`) is better |
| `is_valid_margin` | bool | `abs(p_chosen - 0.5) >= margin`; with the default `margin=0.2`, true when either side has ≥ 0.70 |
| `probabilities` | dict | both option probabilities (sum to 1) |

`chosen_candidate` / `rejected_candidate` are just slot A and slot B: the names describe your
hypothesis, not ground truth. Batched: `gate.predict_preference_batch(prompts, chosen_candidates,
rejected_candidates, batch_size=8)`.

### Which answer reads more naturally?

```python
r = gate.predict_naturalness(
    prompt="Vad ska jag göra i Stockholm en regnig dag?",
    candidate_a="Gå på Fotografiska eller Vasamuseet, och fika på något mysigt kafé.",
    candidate_b="Du kan besöka ett museum, vänligen notera att det är en bra aktivitet för regn.",
)
print(r)  # {{'more_natural': 'candidate_a', 'p_a_more_natural': <float>, 'probabilities': {{...}}}}
```

Batched: `gate.predict_naturalness_batch(prompts, candidates_a, candidates_b)`. When both answers
are fluent, this question stays near 0.5 (within 0.2 of 0.5 for
{pct(ev.get("probe/naturalness", {}).get("fraction_within_0.2"))} of probe pairs), so a confident
naturalness verdict is a sign that one side is translationese.

### Score both orders (recommended for fluent answers)

```python
ab = gate.predict_preference(prompt, resp_1, resp_2)["p_chosen"]
ba = gate.predict_preference(prompt, resp_2, resp_1)["p_chosen"]
p_resp_1_better = (ab + (1 - ba)) / 2
```

The model picks the same winner in both orders for
{pct(g(ev, "probe/preference.positional_consistency"))} of probe pairs; averaging removes the rest
of the position effect.

### Cascade: gate first, LLM judge only when unsure

```python
def judge_pair(prompt, a, b, llm_judge, gate=gate, margin=0.2):
    r = gate.predict_preference(prompt, a, b)
    if abs(r["p_chosen"] - 0.5) >= margin:
        return r["preference"], "gate"
    return llm_judge(prompt, a, b), "llm"     # your own (slow) LLM-as-judge call
```

On the probe, with Kimi-k3 as the judge, see the cascade table below: higher margins route more
pairs to the judge and get closer to its accuracy. The pre-registered rule (cheapest threshold
within 0.5 points of the best) picks {num(rec.get("threshold"), 2)}, routing
{pct(rec.get("routed_fraction"))} for {pct(rec.get("cascade_accuracy"))}.

### Pin a version

```python
gate_v1 = SwedishPreferenceGate("{repo_id}", revision="v1")   # first release
```

With plain `laya`: `laya.load("{repo_id}")` opens the checkpoint; the two question definitions it
was trained on are in `questions.json` and must be used verbatim.

## Intended use

- Fast first-pass ranking of Swedish LLM outputs (best-of-N, A/B comparisons of model versions),
  with low-margin pairs escalated to an LLM judge or a human.
- Detecting translationese, English loan-word overuse, literal phrasing or US-centric answers
  (naturalness question).
- Flagging suspicious pairs in Swedish preference datasets.

**Out of scope:** fact-checking; safety or toxicity classification; languages other than
Swedish; any high-stakes decision without human review. On fluent pairs it is still
{num(100 * ((g(m, "probe.judge.order_averaged.accuracy") or 0) - (probe_acc or 0)), 1)} percentage
points behind an LLM judge.

## Training data

1. **Original pairs** (`jonasaise/dpo-swedish-interpreted`): idiomatic, localized Swedish
   (chosen) vs a deliberately poor translation (rejected). Used for both questions.

{readme.dataset_section(m)}

2. **Fluent pairs** (new in v2): UltraFeedback pairs with a clear GPT-4 rating gap, both answers
   rewritten into natural Swedish by Kimi-k3 with identical instructions, so the two sides differ
   in quality, not style. Preference target = the higher-rated answer; naturalness target =
   50/50.

{readme.generation_section(m)}

## Training, calibration and evaluation

{readme.v2_section(m, intro=False)}

Fitted temperatures: preference T = {num(t_pref)}, naturalness T = {num(t_nat)} (in
`temperature.json`).

## Limitations

- **Still behind an LLM judge on fluent pairs**, and the probe has only 299 pairs (see the CI).
- **Translated, not native, data.** All training and test text was written or rewritten into
  Swedish by Kimi-k3. Native Swedish preferences were not measured.
- **The judge is also the translator.** Kimi-k3 generated all Swedish text and is the reference
  judge, so agreement with it is not independent evidence.
- **Pairwise preference, not fact-checking.** A fluent wrong answer can beat a terse correct one.
- **1024-token window.** Long prompts keep their tail and long answers their head; differences
  late in long answers are invisible.
- **Labels are GPT-4 ratings (UltraFeedback) and Kimi-k3's construction**, not human judgements.
- **Single seed, single run.**

## Files

| File | Contents |
|---|---|
| `model.safetensors` | decision-model weights (encoder + head) |
| `rl_agent_config.json` | Laya config: `max_len` 1024, `head_max_len` 256 |
| `questions.json` | the two `choice` questions (`preference`, `naturalness`) |
| `temperature.json` | fitted temperature per question, read by `SwedishPreferenceGate` |
| `encoder/config.json` | ModernBERT (mmBERT-base) encoder config |
| `tokenizer/` | tokenizer files |
| `results/metrics.json` | every number on this card |

## Citation and acknowledgements

- Laya decision models and training recipe: [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya), base checkpoint [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya).
- Encoder: [`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base).
- Data: [`jonasaise/dpo-swedish-interpreted`](https://huggingface.co/datasets/jonasaise/dpo-swedish-interpreted) and [`argilla/ultrafeedback-binarized-preferences-cleaned`](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned) (UltraFeedback).
- Translation and judge model: Kimi-k3, self-hosted (8× NVIDIA B300, HGX).

```bibtex
@software{{laya_swedish_preference_gate,
  title  = {{Laya Swedish Preference Gate}},
  author = {{Lind, Jonas}},
  year   = {{2026}},
  url    = {{https://huggingface.co/{repo_id}}}
}}
```

## License

Apache-2.0. Base model license: {lic.get("convaiinnovations/laya", "see repo")}; dataset licenses:
`jonasaise/dpo-swedish-interpreted` {lic.get("jonasaise/dpo-swedish-interpreted", "see repo")},
`argilla/ultrafeedback-binarized-preferences-cleaned` {g(m, "fluent_data.source_license", "see repo")}.
Check the upstream UltraFeedback terms before commercial use.
"""


def v3_results_table(m: dict) -> str:
    g, pct, num, ms = readme.get, readme.pct, readme.num, readme.ms
    ev, v2, pw = m.get("v3_eval", {}), m.get("v2_eval", {}), m.get("v3_pointwise", {})
    oa, ci = readme._oa, readme._ci
    ece_o = g(ev, "original/preference.ece_both_orders")
    ece_f = g(ev, "fluent/preference.ece_both_orders")
    lat = g(ev, "latency_gpu_forward_ms", {})
    rows = [
        (
            "Original held-out pairs: idiomatic vs translationese",
            pct(oa(v2, "original/preference")),
            pct(oa(ev, "original/preference")),
            "—",
        ),
        (
            "Fluent held-out pairs: both natural Swedish",
            pct(oa(v2, "fluent/preference")),
            pct(oa(ev, "fluent/preference")),
            "—",
        ),
        (
            "**Probe**: 299 fluent pairs, pairwise preference [95% CI]",
            ci(v2, "probe/preference"),
            ci(ev, "probe/preference"),
            f"Kimi-k3 judge: {pct(g(m, 'probe.judge.order_averaged.accuracy'))}",
        ),
        (
            "**Probe**: higher quality score wins (one pass per answer) [95% CI]",
            "—",
            ci(pw, "probe/quality", "accuracy_expected_score"),
            "—",
        ),
        (
            "Preference ECE, original / fluent held-out (one call, 15 bins)",
            f"{num(g(v2, 'original/preference.ece_both_orders'))} / "
            f"{num(g(v2, 'fluent/preference.ece_both_orders'))}",
            f"{num(ece_o)} / {num(ece_f)}",
            f"{readme.mark(isinstance(ece_o, float) and isinstance(ece_f, float) and max(ece_o, ece_f) <= ECE_TARGET)} ≤ {ECE_TARGET}",
        ),
        (
            "Natural-Swedish check, AUC on original responses",
            "—",
            num(g(pw, "original/natural_swedish.auc")),
            "—",
        ),
        (
            "GPU forward, batch 1, median: pair / single response",
            ms(g(v2, "latency_gpu_forward_ms.p50")),
            f"{ms(lat.get('p50'))} / {ms(g(pw, 'latency_gpu_forward_ms_single_response.p50'))}",
            f"{readme.mark(isinstance(lat.get('p50'), float) and lat['p50'] < readme.LATENCY_TARGET_MS)} < {readme.LATENCY_TARGET_MS:.0f} ms",
        ),
        (
            "CPU, batch 1, median (torch fp32 / ONNX int8)",
            "—",
            f"{ms(g(m, 'cpu_inference.variants.torch-fp32.latency_ms.p50'))} / "
            f"{ms(g(m, 'cpu_inference.variants.onnx-int8.latency_ms.p50'))}",
            "—",
        ),
    ]
    out = ["| Metric | v2 | **v3 (this revision)** | Reference / target |", "|---|---|---|---|"]
    out += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    return "\n".join(out)


def model_card_v3(m: dict, repo_id: str) -> str:
    g, pct, num, ms = readme.get, readme.pct, readme.num, readme.ms
    ev, pw = m.get("v3_eval", {}), m.get("v3_pointwise", {})
    lic = g(m, "data.licenses", {})
    probe_acc = readme._oa(ev, "probe/preference")
    q_acc = g(pw, "probe/quality.accuracy_expected_score.accuracy")
    judge = g(m, "probe.judge.order_averaged.accuracy")
    rec = g(ev, "probe/cascade.recommended", {})
    temps = ", ".join(
        f"{q} T = {num(f.get('temperature'))}" for q, f in m.get("v3_calibration", {}).items()
    )
    header = f"""---
language: [sv]
license: apache-2.0
base_model: convaiinnovations/laya
datasets: [jonasaise/dpo-swedish-interpreted, argilla/ultrafeedback-binarized-preferences-cleaned]
pipeline_tag: text-classification
tags: [preference-model, reward-model, dpo, swedish, laya, mmbert, llm-as-judge]
model-index:
- name: laya-swedish-preference-gate
  results:
  - task: {{type: text-classification, name: Pairwise preference (fluent Swedish pairs)}}
    dataset: {{type: argilla/ultrafeedback-binarized-preferences-cleaned, name: Fluent-vs-fluent probe (299 pairs translated to Swedish)}}
    metrics:
    - {{type: accuracy, value: {probe_acc if probe_acc is not None else "null"}}}
    - {{type: ece, value: {g(ev, "probe/preference.ece_order_averaged") or "null"}, name: Expected calibration error (15 bins)}}
---
"""
    return f"""{header}
# Laya Swedish Preference Gate (v3)

**A ~{num(g(ev, "latency_gpu_forward_ms.p50"), 0)} ms judge for Swedish LLM output.** One forward
pass of a fine-tuned [Laya](https://github.com/NandhaKishorM/laya) decision model (mmBERT-base
backbone) answers one of four questions. It is built as a "System 1" filter: settle the
confident cases in milliseconds on a GPU and escalate the uncertain ones to a slow LLM judge
("System 2").

| Question | Input | Output |
|---|---|---|
| `preference` | prompt + two responses | calibrated P(response A is better) |
| `naturalness` | prompt + two responses | P(A is more natural, idiomatic Swedish) |
| `quality` (new) | prompt + one response | 1–5 score |
| `natural_swedish` (new) | one response (prompt optional) | P(natural Swedish vs translationese) |

On 299 held-out pairs where *both* answers are fluent Swedish, the pairwise question picks the
better answer {pct(probe_acc)} of the time and comparing quality scores {pct(q_acc)} of the time
(Kimi-k3 as judge: {pct(judge)}).

> **Earlier releases:** `revision="v2"` (pairwise questions only) and `revision="v1"` (first
> release, a translationese detector). See the comparison below.

Code, pipeline and full results: [github.com/{GITHUB_REPO}](https://github.com/{GITHUB_REPO}).

## Results at a glance

{v3_results_table(m)}

Pairwise accuracies average the probabilities of both presentation orders. "One call" ECE
scores each order separately.

## How to use

### Install

```bash
pip install "git+https://github.com/{GITHUB_REPO}"
```

### Which answer is better?

```python
from laya_gate.predictor import SwedishPreferenceGate

gate = SwedishPreferenceGate("{repo_id}")  # margin=0.2, CUDA if available

result = gate.predict_preference(
    prompt="Hur fungerar fotosyntesen?",
    chosen_candidate="Växter tar upp ljus med klorofyll och använder energin för att bygga "
                     "socker av koldioxid och vatten. Syre bildas som en restprodukt.",
    rejected_candidate="Växter gör mat av solljus.",
)
# {{'preference': 'candidate_a', 'p_chosen': <float>, 'is_valid_margin': <bool>,
#  'probabilities': {{'candidate_a': <float>, 'candidate_b': <float>}}}}
```

`p_chosen` is the calibrated probability that `candidate_a` (your `chosen_candidate`) is
better; the argument names are slot A and slot B, not ground truth. `is_valid_margin` is
`abs(p_chosen - 0.5) >= margin` (default 0.2). For fluent answers, score both orders and
average: `(p_ab + (1 - p_ba)) / 2`. Batched:
`gate.predict_preference_batch(prompts, chosen_candidates, rejected_candidates)`.

### Best-of-N with one pass per candidate

```python
ranking = gate.rank_by_quality("Hur fungerar fotosyntesen?", [cand_1, cand_2, cand_3])
# [(index, score), ...] best first; gate.score_quality(prompt, response) -> {{'score', 'probabilities'}}
```

The score ranks well (Spearman {num(g(pw, "fluent/quality.spearman_vs_rating"), 2)} against GPT-4
ratings), but the "A beats B" probability derived from two scores is **not** calibrated (ECE
{num(g(pw, "probe/quality.ece_p_chosen_beats_rejected"), 2)} on the probe). Use
`predict_preference` when you need a calibrated margin, e.g. for a cascade.

### Is this natural Swedish?

```python
gate.score_naturalness("Du kan besöka ett museum, vänligen notera att det är en bra aktivitet.")
# {{'label': 'natural' | 'translationese', 'p_natural': <float>, 'is_valid_margin': <bool>}}
gate.predict_naturalness(prompt, candidate_a, candidate_b)   # pairwise version
```

It catches Kimi-k3-style translationese (AUC {num(g(pw, "original/natural_swedish.auc"))}) but
**not** ordinary machine translation: on professional vs `opus-mt-en-sv` translations of FLORES+
sentences its AUC is {num(g(m, "mt_naturalness.models.v3.natural_swedish/no_prompt.auc_human_vs_mt"))}.

### Cascade: gate first, LLM judge only when unsure

```python
def judge_pair(prompt, a, b, llm_judge, gate=gate, margin=0.3):
    r = gate.predict_preference(prompt, a, b)
    if abs(r["p_chosen"] - 0.5) >= margin:
        return r["preference"], "gate"
    return llm_judge(prompt, a, b), "llm"     # your own (slow) LLM-as-judge call
```

On the probe with Kimi-k3 as the judge, threshold {num(rec.get("threshold"), 2)} routes
{pct(rec.get("routed_fraction"))} of instances and reaches {pct(rec.get("cascade_accuracy"))},
above either model alone ({pct(g(ev, "probe/cascade.table.0.cascade_accuracy"))} gate,
{pct(g(ev, "probe/cascade.judge_accuracy"))} judge).

### CPU

```python
gate = SwedishPreferenceGate("{repo_id}", device="cpu")   # recommended on CPU
```

A pair takes {ms(g(m, "cpu_inference.variants.torch-fp32.latency_ms.p50"))} (median) on a laptop
i9; the millisecond figures above need a GPU.

`backend="onnx-int8"` ({ms(g(m, "cpu_inference.variants.onnx-int8.latency_ms.p50"))}) is
**lossy**: it costs about
{num(100 * ((g(m, "cpu_inference.variants.torch-fp32.probe_accuracy.accuracy") or 0) - (g(m, "cpu_inference.variants.onnx-int8.probe_accuracy.accuracy") or 0)), 1)}
points on the probe on average, but single predictions can move by up to
{num(g(m, "cpu_inference.variants.onnx-int8.max_abs_delta_p_vs_torch_fp32"), 2)} in probability
(a clear 0.94 can become 0.50). Use it only for coarse ranking, never for margins or cascades.

### Pin a version

```python
SwedishPreferenceGate("{repo_id}", revision="v2")
SwedishPreferenceGate("{repo_id}", revision="v1")
```

With plain `laya`: `laya.load("{repo_id}")` opens the checkpoint; the four question definitions
it was trained on are in `questions.json` and must be used verbatim.

## Intended use

- Fast first-pass ranking of Swedish LLM outputs (best-of-N, A/B comparisons of model versions),
  with low-margin pairs escalated to an LLM judge or a human.
- Flagging translationese (literal phrasing, English loan words, US-centric answers) in
  LLM-generated Swedish.
- Flagging suspicious pairs in Swedish preference datasets.

**Out of scope:** fact-checking; machine-translation quality estimation; safety or toxicity
classification; languages other than Swedish; high-stakes decisions without human review.

## Training data

1. **Original pairs** (`jonasaise/dpo-swedish-interpreted`): idiomatic, localized Swedish
   (chosen) vs a deliberately poor translation (rejected).
2. **Fluent pairs**: UltraFeedback pairs with a clear GPT-4 rating gap, both answers rewritten
   into natural Swedish by Kimi-k3 with identical instructions, so the two sides differ in
   quality, not style. Each fluent response also carries its GPT-4 rating for the quality
   question.

{readme.generation_section(m)}

## Training, calibration and evaluation

{readme.v3_section(m)}

Fitted temperatures (in `temperature.json`): {temps}.

## Limitations

- **Still behind an LLM judge on fluent pairs**, and the probe has only 299 pairs (see the CIs).
- **Naturalness is specific to Kimi-k3-style translationese**; it does not detect machine
  translation in general.
- **Translated, not native, data.** All Swedish text was written or rewritten by Kimi-k3, which
  is also the reference judge, so agreement with it is not independent evidence.
- **Pairwise preference, not fact-checking.** A fluent wrong answer can beat a terse correct one.
- **1024-token window.** Long prompts keep their tail and long answers their head.
- **Quality scores are not calibrated as pairwise probabilities.**
- **Labels are GPT-4 ratings (UltraFeedback) and Kimi-k3's construction**, not human judgements.
- **Single seed, single run.**

## Files

| File | Contents |
|---|---|
| `model.safetensors` | decision-model weights (encoder + head) |
| `rl_agent_config.json` | Laya config: `max_len` 1024, `head_max_len` 256 |
| `questions.json` | the four questions |
| `temperature.json` | fitted temperature per question, read by `SwedishPreferenceGate` |
| `onnx/model.int8.onnx` | int8 ONNX graph for `backend="onnx-int8"` (lossy, see CPU) |
| `encoder/config.json` | ModernBERT (mmBERT-base) encoder config |
| `tokenizer/` | tokenizer files |
| `results/metrics.json` | every number on this card |

## Citation and acknowledgements

- Laya decision models and training recipe: [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya), base checkpoint [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya).
- Encoder: [`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base).
- Data: [`jonasaise/dpo-swedish-interpreted`](https://huggingface.co/datasets/jonasaise/dpo-swedish-interpreted) and [`argilla/ultrafeedback-binarized-preferences-cleaned`](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned) (UltraFeedback); FLORES+ and `Helsinki-NLP/opus-mt-en-sv` for the machine-translation test.
- Translation and judge model: Kimi-k3, self-hosted (8× NVIDIA B300, HGX).

```bibtex
@software{{laya_swedish_preference_gate,
  title  = {{Laya Swedish Preference Gate}},
  author = {{Lind, Jonas}},
  year   = {{2026}},
  url    = {{https://huggingface.co/{repo_id}}}
}}
```

## License

Apache-2.0. Base model license: {lic.get("convaiinnovations/laya", "see repo")}; dataset licenses:
`jonasaise/dpo-swedish-interpreted` {lic.get("jonasaise/dpo-swedish-interpreted", "see repo")},
`argilla/ultrafeedback-binarized-preferences-cleaned` {g(m, "fluent_data.source_license", "see repo")}.
Check the upstream UltraFeedback terms before commercial use.
"""


def model_card_v4(m: dict, repo_id: str) -> str:
    g, pct, num = readme.get, readme.pct, readme.num
    A = g(m, "v4.arms", {})
    S = g(m, "v4.sanity.checkpoints", {})
    P = g(m, "pilot_euroeval", {})
    lic = g(m, "data.licenses", {})
    R, NAME = readme.RELEASE_ARM, readme.RELEASE_NAME
    v4 = g(A, R, {})

    def t(arm, test):
        return g(A, f"{arm}.tests.{test}.mean")

    rows = [
        ("Fluent-vs-fluent probe, `preference`", "probe/preference"),
        ("DPO validation, `preference`", "original/preference"),
        ("liu-nlp native vs OPUS-MT, `naturalness`", "liu/naturalness"),
        ("FLORES+ human vs OPUS-MT, `preference` with source", "flores/preference+source"),
        ("WMT24++ human vs OPUS-MT, `preference` with source", "wmt24pp/opus/preference+source"),
        ("WMT24++ human vs Kimi-k3, `preference` with source", "wmt24pp/kimi/preference+source"),
        ("DaLAJ corrected vs learner, `naturalness`", "dalaj/naturalness"),
        ("GerLangMod-sv original vs verb misplaced, `naturalness`", "gerlangmod/naturalness"),
        ("SweFAQ own answer vs other, `preference`", "swefaq/preference"),
        ("BLiMPPI-sv polarity items, `naturalness`", "blimppi/naturalness"),
        ("MultiBLiMP-sv agreement, `naturalness`", "multiblimp/naturalness"),
    ]
    table = "\n".join(
        f"| {label} | {pct(t('v3', k))} | {pct(t('adequacy', k))} | **{pct(t(R, k))}** |"
        for label, k in rows
    )
    mcc_v3 = readme._scala_mcc(A, "v3")
    mcc_v4 = readme._scala_mcc(A, R)
    mcc_v40 = readme._scala_mcc(A, "adequacy")
    sanity = "\n".join(
        f"| {label} | {pct(g(S, f'v3.{k}.mean'))} | {pct(g(S, f'grammar.{k}.mean'))} | "
        f"{pct(g(S, f'adequacy.{k}.mean'))} | **{pct(g(S, f'{R}.{k}.mean'))}** |"
        for k, label in readme.SANITY_ROWS
    )
    q = g(P, "spearman.euroeval_rank (negated).v4-v41/answer_quality", {})
    qp = g(P, "partial_spearman_rank_given_size.v4-v41/answer_quality", {})
    temps = ", ".join(
        f"{k} T = {num(f.get('temperature'))}" for k, f in v4.get("calibration", {}).items()
    )
    probe_v4 = t(R, "probe/preference")
    header = f"""---
language: [sv]
license: apache-2.0
base_model: convaiinnovations/laya
datasets:
- jonasaise/dpo-swedish-interpreted
- argilla/ultrafeedback-binarized-preferences-cleaned
- sbx/superlim-2
- {V4_DATA_REPO}
pipeline_tag: text-classification
tags: [preference-model, reward-model, swedish, laya, mmbert, llm-as-judge, grammaticality, translation-quality]
model-index:
- name: laya-swedish-preference-gate
  results:
  - task: {{type: text-classification, name: Linguistic acceptability (ScaLA-sv, EuroEval split)}}
    dataset: {{type: mteb/scala_sv_classification, name: ScaLA-sv test}}
    metrics:
    - {{type: mcc, value: {mcc_v4 if mcc_v4 is not None else "null"}}}
  - task: {{type: text-classification, name: Pairwise preference (fluent Swedish pairs)}}
    dataset: {{type: argilla/ultrafeedback-binarized-preferences-cleaned, name: Fluent-vs-fluent probe (299 pairs translated to Swedish)}}
    metrics:
    - {{type: accuracy, value: {probe_v4 if probe_v4 is not None else "null"}}}
---
"""
    return f"""{header}
# Laya Swedish Preference Gate ({NAME})

**A ~{num(g(v4, "standard_eval.latency_gpu_forward_ms.p50"), 0)} ms judge for Swedish text.** One
forward pass of a fine-tuned [Laya](https://github.com/NandhaKishorM/laya) decision model
(mmBERT-base backbone) answers one of five questions, as a "System 1" filter in front of a slow
LLM judge.

| Question | Input | Output |
|---|---|---|
| `preference` | prompt (or English source) + two responses | calibrated P(response A is better) |
| `naturalness` | two responses (prompt optional) | P(A is more natural, idiomatic Swedish) |
| `quality` | prompt + one response | 1–5 score |
| `natural_swedish` | one response | P(natural Swedish vs translationese) |
| `grammatical` (new) | one text | P(grammatically correct); compare two texts, see limitations |

> **Earlier releases:** `revision="v4"`, `revision="v3"`, `revision="v2"`, `revision="v1"`.

Code, the full ablation and every number below: [github.com/{GITHUB_REPO}](https://github.com/{GITHUB_REPO}).
Training and evaluation pairs: [`{V4_DATA_REPO}`](https://huggingface.co/datasets/{V4_DATA_REPO}).

## What changed from v3

v4 and {NAME} add training data chosen in a controlled ablation (same recipe, seed and
hyperparameters; only the data differs; all tests held out):

- **DaLAJ-GED** (Superlim 2): learner sentences vs the same sentence corrected, for
  `naturalness` and the new `grammatical` question.
- **SweFAQ** (Superlim 2): a question's own answer vs other answers from the same FAQ page,
  for `preference`.
- **Europarl** (Språkbanken): professional Swedish translations vs Kimi-k3 machine
  translations of English speeches, plus *adequacy negatives* (another paragraph's
  translation, a half translation, a word-shuffled translation, the untranslated English).
- **{NAME} only:** correct text from four registers against corrupted copies (a word deleted or
  two neighbours swapped), and wrong-source negatives from the adjacent paragraph of the same
  debate, to reduce `grammatical` over-flagging and fluent-but-wrong translations winning.

| Held-out test (share where the reference wins) | v3 | v4 | {NAME} |
|---|---|---|---|
{table}
| ScaLA-sv MCC (EuroEval's metric), single-response | {num(mcc_v3)} | {num(mcc_v40)} | **{num(mcc_v4)}** |

Calibration: {temps}. ECE on the DPO validation pairs
{num(g(v4, "standard_eval.original/preference.ece_both_orders"), 4)}.

**A trap we hit on the way.** Human-vs-MT pairs alone teach a gate that "the candidate least like
the source wins": the intermediate arms preferred a fluent translation of the wrong sentence, half
translations and nonsense. The adequacy negatives fix it (FLORES+ devtest, English source in the
prompt, share where the reference wins):

| The reference against … | v3 | without adequacy negatives | v4 | **{NAME}** |
|---|---|---|---|---|
{sanity}

On {200 * len(g(P, "models", []))} translations by {len(g(P, "models", []))} LLMs, {NAME}'s translation score agrees with chrF at Spearman
{num(g(P, "translation_item_level.v4-v41/translation_vs_reference.spearman_with_chrf"), 2)}
per translation (v3: {num(g(P, "translation_item_level.v3/translation_vs_reference.spearman_with_chrf"), 2)}).

## Pilot: a quick proxy for EuroEval Swedish

Scoring 100 Swedish answers per model with `quality` and averaging, over {len(g(P, "models", []))} open LLMs with
Swedish results on the [EuroEval leaderboard](https://euroeval.com): Spearman with the Swedish
rank {num(q.get("rho"), 2)} [{num(g(q, "ci95.0"), 2)}, {num(g(q, "ci95.1"), 2)}],
{num(qp.get("rho"), 2)} [{num(g(qp, "ci95.0"), 2)}, {num(g(qp, "ci95.1"), 2)}] after controlling for
model size. A pilot (models up to 11B parameters, 6 of them in 4-bit), not a calibrated
predictor.

## Quickstart

```bash
uv pip install "git+https://github.com/{GITHUB_REPO}"
```

```python
from laya_gate.predictor import SwedishPreferenceGate

gate = SwedishPreferenceGate("{repo_id}")

r = gate.predict_preference(
    prompt="Hur fungerar fotosyntesen?",
    chosen_candidate="Fotosyntesen är den process där växter omvandlar ljus till energi.",
    rejected_candidate="Det är när solen lyser på träd och gör magi.",
)
# {{"preference": "candidate_a", "p_chosen": <float>, "is_valid_margin": <bool>, ...}}

# translation quality: put the English source in the prompt
t = gate.predict_preference(
    prompt="Översätt följande text till svenska:\nThe meeting was postponed.",
    chosen_candidate="Mötet sköts upp.",
    rejected_candidate="Mötet ställdes in.",
)

s = gate.score_quality(prompt="Hur fungerar fotosyntesen?", response="Växter gör mat av solljus.")
g = gate.score_grammar("Människor är snälla där och alla varor är billiga.")
a, b = (gate.score_grammar(x)["p_correct"] for x in ("Hon har bott här.", "Hon har bo här."))
```

## Limitations

- **`grammatical` over-flags as a stand-alone verdict.** At 0.5 it marks
  {pct(g(v4, "guards.fluent_responses_flagged_incorrect.mean"))} of fluent LLM answers and
  {pct(g(v4, "guards.wmt24pp_reference_flagged_incorrect.mean"))} of WMT24++ references as
  incorrect (it was trained on single learner sentences). Compare two versions of a text instead.
  On lowercased text without punctuation (GerLangMod-sv) it is at chance
  ({pct(t(R, "gerlangmod/grammatical"))}); v4 was at
  {pct(t("adequacy", "gerlangmod/grammatical"))}.
- **Wrong-content translations still win sometimes:** v4 prefers a fluent translation of a
  different sentence in {pct(1 - (g(S, f"{R}.wrong_source/preference+source.mean") or 0))} of
  FLORES+ pairs (v3: {pct(1 - (g(S, "v3.wrong_source/preference+source.mean") or 0))}).
- **Pairwise preference, not fact-checking;** 1,024-token window; one seed per training arm.
- **Kimi-k3 is not a neutral reference for its own output:** on WMT24++ it prefers the human
  translation over OPUS-MT in
  {pct(g(m, "external_eval.judge.sets.wmt24pp/opus.translation_generic.accuracy"))} of pairs but
  over its own translation in only
  {pct(g(m, "external_eval.judge.sets.wmt24pp/kimi.translation_generic.accuracy"))}.

## Training data and licenses

Weights: **Apache-2.0**. Base model `convaiinnovations/laya`: {lic.get("convaiinnovations/laya", "see repo")}.
Training data:
- `jonasaise/dpo-swedish-interpreted` ({lic.get("jonasaise/dpo-swedish-interpreted", "see repo")}) and
  `argilla/ultrafeedback-binarized-preferences-cleaned` ({g(m, "fluent_data.source_license", "see repo")}).
- Europarl Swedish-English: Koehn (2005), *Europarl: A Parallel Corpus for Statistical Machine
  Translation*, MT Summit X; enriched and distributed by Språkbanken,
  https://doi.org/10.23695/mdp4-2p72, CC BY 4.0.
- DaLAJ-GED (CC BY 4.0) and SweFAQ (CC BY-SA 4.0) from Superlim 2
  ([`sbx/superlim-2`](https://huggingface.co/datasets/sbx/superlim-2)), Språkbanken.
- Machine translations by `Helsinki-NLP/opus-mt-en-sv` (evaluation only) and Kimi-k3.

Evaluation only: liu-nlp/translationese-opensubtitles (Apache-2.0), FLORES+, WMT24++
(Apache-2.0), ScaLA-sv (`mteb/scala_sv_classification`), GerLangMod (Apache-2.0), BLiMPPI
(arXiv:2610.13607; Swedish subset from UD SweLL/Talbanken CC BY-SA 4.0 and LinES CC BY-NC-SA
4.0), MultiBLiMP (`jumelet/multiblimp`, CC BY 4.0).

```bibtex
@software{{laya_swedish_preference_gate,
  title  = {{Laya Swedish Preference Gate}},
  author = {{Lind, Jonas}},
  year   = {{2026}},
  url    = {{https://huggingface.co/{repo_id}}}
}}
```
"""


def _probe_line(m: dict) -> str:
    summary = readme.probe_summary(m)
    if summary:
        return (
            f"**Measured on fluent pairs:** {summary}. Treat its confidence as meaningful "
            "only for naturalness/translationese."
        )
    return (
        "Its ability to rank two *equally fluent* answers by helpfulness or correctness has "
        "**not** been measured."
    )


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
    parser.add_argument("--checkpoint", choices=["v1", "v2", "v3", "v4"], default="v1")
    parser.add_argument(
        "--tag-previous",
        metavar="TAG",
        help="tag the revision currently on the Hub (e.g. v1) before uploading over it",
    )
    args = parser.parse_args(argv)
    m = load_metrics()
    which = args.checkpoint
    src_dir = checkpoint_dir(which)
    problems = blockers(m, which)
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
        tagged = None
        if args.tag_previous:
            refs = api.list_repo_refs(repo_id)
            if any(t.name == args.tag_previous for t in refs.tags):
                print(f"tag {args.tag_previous} already exists; leaving it")
            else:
                api.create_tag(repo_id, tag=args.tag_previous, revision=info.sha)
                print(f"tagged {info.sha[:8]} as {args.tag_previous}")
            tagged = {"tag": args.tag_previous, "revision": info.sha}
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
        files = list(UPLOAD_FILES)
        if which in ("v2", "v3", "v4"):
            files.append(QUESTIONS_FILE)
        if which == "v3":
            files += [f for f in V3_ONNX_FILES if (src_dir / f).exists()]
        for name in files:
            (stage / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_dir / name, stage / name)
        for name in UPLOAD_DIRS:
            shutil.copytree(src_dir / name, stage / name)
        (stage / "results").mkdir()
        shutil.copy2(METRICS_PATH, stage / "results" / "metrics.json")
        card = {"v2": model_card_v2, "v3": model_card_v3, "v4": model_card_v4}.get(
            which, model_card
        )(m, repo_id)
        (stage / "README.md").write_text(card, encoding="utf-8")
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
            commit_message=f"Upload Laya Swedish preference gate checkpoint ({which})",
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
        "checkpoint": which,
        "release": readme.RELEASE_NAME if which == "v4" else which,
        "previous_revision_tag": tagged,
        "files": sorted([*files, *UPLOAD_DIRS, "results/metrics.json", "README.md"]),
    }
    update_metrics("publish", out)
    readme.main()  # adds the Hub link and Hub quickstart to the GitHub README
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
