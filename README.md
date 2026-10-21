# Laya Swedish Preference Gate

A fast pairwise judge for Swedish text. Give it a Swedish prompt and two candidate responses
and it returns a calibrated probability that the first is better, in one forward pass of a
fine-tuned [Laya](https://github.com/NandhaKishorM/laya) decision model (mmBERT-base backbone).
It is meant as a "System 1" filter: it settles the easy comparisons in milliseconds and hands
the uncertain ones to a slower "System 2" LLM judge. It was trained on
[`jonasaise/dpo-swedish-interpreted`](https://huggingface.co/datasets/jonasaise/dpo-swedish-interpreted),
where the preferred answer is idiomatic, localized Swedish and the rejected one is a
deliberately poor translation. What it has learned is mainly to tell **authentic Swedish from
translationese**, and it is compared against self-hosted Kimi-k3 on 8× NVIDIA B300 (HGX).

Weights: [huggingface.co/jonasaise/laya-swedish-preference-gate](https://huggingface.co/jonasaise/laya-swedish-preference-gate) (public).

A **v2** checkpoint (`python -m laya_gate.v2`, see [v2](#v2-training-on-fluent-pairs-too)) adds 9921 fluent-vs-fluent training pairs and a separate naturalness question. On the probe it reaches 75.9% (95% CI 70.9%–80.6%) vs 64.2% for v1, with ECE 0.033 instead of 0.230, so its margin routes mistakes to the judge again: with about 32% of probe instances routed to Kimi-k3, the cascade reaches 84.8% with v2 (33.1% routed) vs 73.7% with v1. The cost is 98.7% → 97.6% on the original held-out pairs (both orders averaged), and the probe is still 11.4 percentage points behind Kimi-k3. v2 is not on the Hub yet.

A **v3** checkpoint (`python -m laya_gate.v3`, see [v3](#v3-single-response-questions)) keeps v2's two pairwise questions and adds two single-response questions: a 1–5 **quality** score and a **natural Swedish vs translationese** check. On the probe its pairwise preference reaches 80.3% [75.6%, 84.6%] (v2: 75.9%), and ranking the two answers by their quality scores reaches 82.9% [78.9%, 87.0%] with one forward pass per answer (Kimi-k3: 87.3%). On the original held-out pairs v3 is back to 98.9% (v2: 97.6%). On human vs machine translations of FLORES+ sentences, no version is useful (see [the machine-translation test](#machine-translation-test-a-negative-result)). v3 is not on the Hub yet.

## Results at a glance

| Metric | Value | Target |
|---|---|---|
| Validation accuracy (val_eval, held out) | 98.4% | — |
| ECE before calibration | 0.013 | — |
| ECE after calibration | 0.006 | ✅ target ≤ 0.085 |
| Fitted temperature T | 1.967 | — |
| GPU latency, batch 1 (median / p95) | 9.9 ms / 11.2 ms | ✅ target < 35 ms (median) |
| Gate accuracy on judge sample (order-averaged) | 98.6% | — |
| Kimi-k3 accuracy on judge sample (order-averaged) | 95.5% | — |

All numbers are generated from [`results/metrics.json`](results/metrics.json) by
`python -m laya_gate.readme`.

## Installation

```bash
git clone https://github.com/joenaess/laya-swedish-preference-gate.git
cd laya-swedish-preference-gate
uv sync --extra dev
```

Requires Python ≥ 3.11. A CUDA GPU is optional for inference and needed for training.

## Quickstart

The fine-tuned weights are on the Hugging Face Hub at [`jonasaise/laya-swedish-preference-gate`](https://huggingface.co/jonasaise/laya-swedish-preference-gate); `SwedishPreferenceGate` downloads them on first use.

```python
from laya_gate.predictor import SwedishPreferenceGate

gate = SwedishPreferenceGate(checkpoint_path="jonasaise/laya-swedish-preference-gate")
result = gate.predict_preference(
    prompt="Hur fungerar fotosyntesen?",
    chosen_candidate="Fotosyntesen är den process där växter omvandlar ljusenergi till kemisk energi...",
    rejected_candidate="Det är när solen lyser på träd och gör magi.",
)
print(result)
# {"preference": "candidate_a", "p_chosen": <float>, "is_valid_margin": <bool>,
#  "probabilities": {"candidate_a": <float>, "candidate_b": <float>}}
```

`chosen_candidate` is shown to the model as `candidate_a` and `rejected_candidate` as
`candidate_b`; the names describe your hypothesis, not ground truth. `p_chosen` is the
temperature-calibrated probability that `candidate_a` is the better response.
`is_valid_margin` is `True` when `|p_chosen − 0.5| ≥ margin`; the constructor argument `margin`
defaults to **0.2** (i.e. calibrated probability ≥ 0.7 for
either side). The extra `probabilities` key holds both option probabilities (they sum to 1).

Batched scoring (length-sorted internally, results returned in input order):

```python
results = gate.predict_preference_batch(
    prompts=["Vad är huvudstaden i Sverige?", "Skriv en mening om hösten."],
    chosen_candidates=["Stockholm.", "Löven blir röda och gula."],
    rejected_candidates=["Göteborg är huvudstaden.", "Höst."],
    batch_size=32,
)
```

The Hub checkpoint is **v4.1**, which also answers *which response is written in more
natural, idiomatic Swedish*:

```python
r = gate.predict_naturalness(
    prompt="Vad ska jag göra i Stockholm en regnig dag?",
    candidate_a="Gå på Fotografiska eller Vasamuseet, och fika på något mysigt kafé.",
    candidate_b="Du kan besöka ett museum, vänligen notera att det är en bra aktivitet för regn.",
)
# {"more_natural": "candidate_a", "p_a_more_natural": <float>, "probabilities": {...}}
```

v4 and v4.1 add a single-response grammar check (use it to compare two versions of a text; as a
stand-alone verdict it flags too much good Swedish) and judges translations with the English
source in the prompt:

```python
g = gate.score_grammar("Människor är snälla där och alla varor är billiga.")
# {"label": "correct" | "incorrect", "p_correct": <float>, "is_valid_margin": <bool>}
t = gate.predict_preference(
    prompt="Översätt följande text till svenska:
The meeting was postponed.",
    chosen_candidate="Mötet sköts upp.",
    rejected_candidate="Mötet ställdes in.",
)
s = gate.score_quality(prompt="Hur fungerar fotosyntesen?", response="Växter gör mat av solljus.")
```

Earlier releases stay loadable: `revision="v4"`, `revision="v3"`, `revision="v2"` and
`revision="v1"`.

Also `device="cpu"`, `dtype=torch.float32`, `temperature=<float>`, `margin=<float>` and
`revision=<Hub branch or tag>` are accepted. The checkpoint is in the standard Laya layout, so `laya.load(path)` opens it too
(the fitted temperature is stored in its `rl_agent_config.json`).

## How it works

- **`choice` primitive.** Each pair becomes a Laya state
  `{"prompt", "candidate_a", "candidate_b"}` and one `choice` question with options
  `candidate_a: response A is superior` and `candidate_b: response B is superior`. The encoder
  reads `[CLS] question [SEP] [MASK] option A [MASK] option B [SEP] state [SEP]`, and the
  decision head scores each `[MASK]`: one forward pass, no generation.
- **Symmetrization.** With a fixed seed, exactly half of the pairs put the chosen response in
  slot A and half in slot B, so the model cannot learn a positional shortcut.
- **Truncation.** Inputs are limited to 1024 tokens. When a pair is longer, it is cut at
  token boundaries using the model's own tokenizer: the *tail* of the prompt (the most recent
  context) and the *head* of each response are kept. The budget is shared max-min fairly in
  serialized-token space, so short fields are untouched and neither response is cut to
  nothing.
- **Temperature scaling.** A single temperature `T` is fitted on a calibration half of the
  validation set and applied as `softmax(logits / T)`. It changes confidence, not the
  predicted winner.
- **Pipeline.** `./run.sh` runs data → train → calibrate → benchmark → judge → probe →
  fluent pairs → v2 and v3 train / calibrate / evaluate → CPU benchmark →
  machine-translation test → tests → readme → publish.

## Detailed results

### Dataset

Source: [`jonasaise/dpo-swedish-interpreted`](https://huggingface.co/datasets/jonasaise/dpo-swedish-interpreted).
The repository's single `train` split is stored as parquet files with two different
schemas (the dataset card declares only one), so `datasets.load_dataset` fails with a cast
error; `data.py` reads the files one by one and keeps `prompt`, `chosen`, `rejected`.

| | |
|---|---|
| Raw pairs | 14986 |
| Dropped (empty / identical / duplicate) | {"empty_field": 0, "identical_responses": 0, "duplicate": 0} |
| Train / val_calib / val_eval | 12750 / 1118 / 1118 |
| Val pairs moved to train (prompt also in train) | 12 |
| Fraction with chosen in slot B (train / calib / eval) | 0.500 / 0.499 / 0.498 |
| Pairs truncated to fit 1024 tokens | 6336 (42.3%) |
| Untruncated input length, tokens (median / p95 / max) | 858.0 / 2423.75 / 11276 |
| Question head / state budget, tokens | 32 / 991 |

**How the pairs were made.** The dataset card says kimi-k3 localized each UltraFeedback prompt
and chosen response into idiomatic Swedish and deliberately generated the rejected response
with poor translation, excessive English loan words or US-centric content. The preference
signal is therefore mostly *authentic vs. translated Swedish*. Two non-neural baselines on
val_eval show how lexical it is:

| Baseline (val_eval) | Accuracy |
|---|---|
| Longer response wins | 71.3% |
| Logistic regression on TF-IDF(A) − TF-IDF(B) | 98.4% |

Words the TF-IDF model associates with the *chosen* side: *alltså*, *gärna*, *utifrån*, *innebär*, *stället*, *svenska*, *exempel*, *sverige*, *samt*, *däremot*.
With the *rejected* side: *denna*, *detta*, *konfidens*, *såsom*, *kommer*, *av*, *baserat*, *inkluderar*, *hjälpa*, *är*.

### Training

Base: `convaiinnovations/laya (subfolder=multilingual)` (backbone `jhu-clsp/mmBERT-base`), adapted from the
reference recipe in [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) (`laya.train`).

| Hyperparameter | Value |
|---|---|
| Precision | bfloat16 autocast |
| Micro-batch × accumulation (effective) | 8 × 2 (16) |
| OOM fallback used | none (gradient checkpointing: False) |
| Learning rate / schedule | 2e-05, linear warmup 10% of 2391 updates, cosine decay |
| Optimizer | AdamW, weight decay 0.01, grad-norm clip 1.0 |
| Epochs | 3 |
| Loss | soft cross-entropy over the two choice options |
| Sequence budget | max_len 1024, head_max_len 256 |

| Epoch | Train loss | Val loss (calib / eval) | Val acc (calib / eval) | Time |
|---|---|---|---|---|
| 1 | 0.3167 | 0.1283 / 0.1201 | 97.4% / 97.5% | 10.2 min |
| 2 **(best)** | 0.0360 | 0.0772 / 0.0651 | 97.9% / 98.5% | 10.5 min |
| 3 | 0.0017 | 0.0968 / 0.0781 | 98.3% / 98.6% | 10.6 min |

- Zero-shot base checkpoint (before fine-tuning): val_eval accuracy 51.6%, loss 1.238.
- Best epoch: **2**, chosen by lowest loss on the val_calib half; val_eval is never used for selection.
- Training wall-clock: 31.4 min total
  (10.2, 10.5, 10.6 min per epoch, including per-epoch validation).
- Peak GPU memory: 10.73 GiB allocated / 10.84 GiB reserved.
- Hardware: NVIDIA GeForce RTX 4080 Laptop GPU, PyTorch 2.14.1+cu130.

### Calibration

Temperature scaling with a single scalar `T = exp(log_T)`, fitted by L-BFGS on the NLL of the
**val_calib** half; all headline numbers below are on the **val_eval** half, which neither the
fit, checkpoint selection nor training saw. ECE uses 15 equal-width bins over
the top-1 confidence.

| | Before | After |
|---|---|---|
| ECE (val_eval) | 0.013 | 0.006 |
| NLL (val_eval) | 0.066 | 0.046 |
| ECE (val_calib, in-sample for the fit) | 0.016 | 0.009 |

Fitted temperature: **T = 1.967**
(T > 1: the raw logits were over-confident).
Target ECE ≤ 0.085: ✅.

Reliability table (val_eval; empty bins omitted; for two options the top-1 confidence is ≥ 0.5):

| Confidence bin | n (pre) | Conf. / acc. (pre) | n (post) | Conf. / acc. (post) |
|---|---|---|---|---|
| 0.467–0.533 | 0 | — | 1 | 0.520 / 1.000 |
| 0.533–0.600 | 1 | 0.539 / 1.000 | 1 | 0.579 / 0.000 |
| 0.600–0.667 | 1 | 0.651 / 0.000 | 5 | 0.643 / 0.600 |
| 0.667–0.733 | 1 | 0.731 / 0.000 | 5 | 0.709 / 0.600 |
| 0.733–0.800 | 4 | 0.767 / 0.750 | 7 | 0.781 / 0.286 |
| 0.800–0.867 | 4 | 0.846 / 0.750 | 7 | 0.833 / 0.857 |
| 0.867–0.933 | 7 | 0.915 / 0.143 | 12 | 0.906 / 0.917 |
| 0.933–1.000 | 1100 | 0.999 / 0.993 | 1080 | 0.993 / 0.994 |

### Latency and throughput

Method: batch size 1, 20 warm-up passes, then 200 timed passes over 200 distinct val_eval pairs; torch.cuda.synchronize() before and after each timed pass; forward = host->device copy + model + logits back to host; tokenization = token-level truncation + laya sequence building.
Inputs are real validation pairs (median 878 tokens, p95 1024).
GPU: NVIDIA GeForce RTX 4080 Laptop GPU (torch.bfloat16 weights). CPU: 24 threads, torch.float32.

| Measurement (batch 1) | Median | p95 | n |
|---|---|---|---|
| GPU forward pass | 9.9 ms | 11.2 ms | 200 |
| GPU forward, full 1024-token input (1024 tokens) | 11.4 ms | 11.6 ms | 200 |
| Tokenization + truncation (CPU) | 1.5 ms | 9.2 ms | 200 |
| GPU end-to-end `predict_preference` | 11.9 ms | 21.1 ms | 200 |
| CPU forward pass | 606.8 ms | 760.3 ms | 30 |

| Throughput (length-sorted batches, 512 val_eval pairs) | Pairs/s | Per pair |
|---|---|---|
| GPU, batch 1 | 106.3 pairs/s | 9.4 ms/pair |
| GPU, batch 8 | 119.9 pairs/s | 8.3 ms/pair |
| GPU, batch 32 | 100.0 pairs/s | 10.0 ms/pair |

Batch 32 is slower than batch 8: at up to 1024 tokens per pair, 32-pair batches push the 12 GB laptop GPU to its memory limit (the CUDA allocator logged failed allocations and retries), so throughput peaks at batch 8 on this card.

Target: median GPU forward pass < 35 ms at batch 1 → ✅
(p95 under target: ✅).

### Positional consistency

Every val_eval pair (1118) scored in both orders (A/B and B/A).

| | |
|---|---|
| Same winner in both orders | 98.4% |
| Accuracy with chosen in slot A | 98.0% |
| Accuracy with chosen in slot B | 98.9% |
| Gap (A − B) | -0.9 pp |
| Mean P(slot A wins) | 0.494 |
| Accuracy, averaging P over both orders | 98.7% |

### Accuracy at the margin threshold

At the default `margin=0.2`, **99.4%** of
predictions pass `is_valid_margin`, with accuracy **98.6%** on that
subset (vs 98.5% on all predictions; both orders counted).

| Margin | Fraction passing | Accuracy on passing |
|---|---|---|
| 0.00 | 100.0% | 98.5% |
| 0.05 | 100.0% | 98.5% |
| 0.10 | 99.8% | 98.5% |
| 0.15 | 99.7% | 98.6% |
| 0.20 | 99.4% | 98.6% |
| 0.25 | 99.1% | 98.8% |
| 0.30 | 98.5% | 99.0% |
| 0.35 | 98.0% | 99.1% |
| 0.40 | 97.6% | 99.1% |
| 0.45 | 95.2% | 99.5% |

### System 1 vs System 2

Judge: self-hosted Kimi-k3 on 8× NVIDIA B300 (HGX), model id `kimi-k3`, temperature 0, reasoning allowed
(only the final answer is parsed). Sample: 500 fixed-seed pairs from val_eval,
each queried in both orders (1000 requests, concurrency
16). The judge sees the full, untruncated texts; the gate sees
its 1024-token window.

| | Gate (System 1) | Kimi-k3 (System 2) |
|---|---|---|
| Accuracy, order A/B | 98.8% | 95.6% |
| Accuracy, order B/A | 98.4% | 95.4% |
| Accuracy, order-averaged | 98.6% | 95.5% |
| Positional consistency | 98.8% | 98.2% |
| Median latency per pair | 11.9 ms | 9531.2 ms (p95 34626.1 ms) |

- Gate–judge agreement: **94.7%** of instances. On the 53
  disagreements the gate is right 79.2% of the time and the judge
  20.8%.
- Judge parse failures: 0 (0.00%); request failures after retries: 0.
  Judge accuracy counting parse failures as wrong: 95.5%.
- Judge throughput: 0.99 requests/s at concurrency 16.
  Tokens used: 1770631 total (1272393 prompt, 498238 completion, reasoning included).
- Latency ratio (judge median / gate end-to-end median): **802×**.

### Cascade (System 1 → System 2)

An instance goes to the judge when `|p_chosen − 0.5| < threshold`; otherwise the gate's answer
stands. Expected latency = gate latency + routed fraction × judge median latency. Routed
instances without a parseable judge verdict keep the gate's answer.

| Threshold | Routed to judge | Cascade accuracy | Expected latency / pair |
|---|---|---|---|
| **0.00** | **0.0%** | **98.6%** | **11.9 ms** |
| 0.05 | 0.0% | 98.6% | 11.9 ms |
| 0.10 | 0.2% | 98.6% | 30.9 ms |
| 0.15 | 0.3% | 98.6% | 40.5 ms |
| 0.20 | 0.7% | 98.6% | 78.6 ms |
| 0.25 | 1.1% | 98.6% | 116.7 ms |
| 0.30 | 1.6% | 98.9% | 164.4 ms |
| 0.35 | 2.3% | 98.7% | 231.1 ms |
| 0.40 | 2.7% | 98.7% | 269.2 ms |
| 0.45 | 5.2% | 99.1% | 507.5 ms |

Recommended threshold (**bold**): **0.00** — lowest routed fraction within 0.5% of the best cascade accuracy in the sweep:
0.0% routed, 98.6% accuracy.

### Label-noise findings

Rule: judge prefers the rejected response in both orders AND gate's order-averaged P(chosen) <= 0.30. **1** of 500 sampled pairs
(0.2%) meet it. Their ids are in `results/metrics.json`
(`judge.label_noise.pair_ids`). These are candidates for mislabeled or ambiguous pairs, not
confirmed errors: both "judges" may share a bias, for example toward longer answers.

### Fluent-vs-fluent probe (does it judge quality?)

The held-out numbers above come from the training distribution, where "chosen" is idiomatic
Swedish and "rejected" is a deliberately poor translation. This probe asks the harder question:
**can the gate pick the better answer when both are natural Swedish?**

Design: 299 UltraFeedback pairs (seed 20261014, GPT-4 rating margin >= 1.0, <= 3500 English chars) rewritten into natural Swedish by kimi-k3 with identical instructions for both answers, content and errors preserved; label = higher-rated answer. Source: [`argilla/ultrafeedback-binarized-preferences-cleaned`](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned).
299 of 300 sampled pairs translated cleanly (translation parse
failures: 1). Sanity checks: the log length ratio between
the two answers correlates 0.998 between English and Swedish
(content length preserved), and the TF-IDF style model that scores
98.4% on the training distribution drops to
64.2% here (the translationese signal is largely gone).
5.0% of probe inputs were truncated for the gate.

| Scorer (both orders) | Accuracy [95% bootstrap CI] |
|---|---|
| **Gate** (order-averaged probability) | **64.2% [58.9%, 69.6%]** |
| Gate, better answer in slot A | 58.9% [53.2%, 64.5%] |
| Gate, better answer in slot B | 70.6% [65.6%, 75.6%] |
| **Kimi-k3 judge** | **87.3% [84.6%, 89.8%]** |
| TF-IDF style model (trained on the gate's training split) | 64.2% [58.5%, 69.6%] |
| Longer Swedish answer wins | 57.5% [51.8%, 63.2%] |

- Gate positional consistency drops to 82.3% (vs
  98.4% in-distribution); Kimi-k3's is
  94.0%. Gate–judge agreement: 61.0%.
- **Calibration does not transfer:** mean confidence 0.872, ECE
  0.230 (vs 0.006 in-distribution).
  At the default `margin=0.2`, 83.6% of probe pairs pass
  as confident, with only 66.8% accuracy, so a cascade would keep
  most of its mistakes instead of routing them to the judge.
- By GPT-4 rating gap: 59.8% [52.3%, 68.2%] for gaps 1.0–2.0, 67.7% [60.5%, 74.9%] for gaps ≥ 2.0.
- Excluding the 32 probe prompts that likely overlap training prompts
  (char 3-5-gram TF-IDF cosine >= 0.8 to the nearest training prompt): 62.5% [56.6%, 68.2%].

| Gate confidence | Share of probe pairs | Accuracy |
|---|---|---|
| 0.5–0.7 | 16.4% | 51.0% |
| 0.7–0.9 | 18.4% | 56.4% |
| 0.9–1.0 | 65.2% | 69.7% |

**Takeaway:** the gate carries a modest quality signal (above chance) but is far behind an
LLM judge on fluent pairs and is badly over-confident there. Use it for naturalness and
translationese; do not use its confidence to skip the judge on fluent answers.

### v2: training on fluent pairs too

**Why:** the probe showed that v1 mostly detects translationese. v2 keeps the original
data and adds pairs where *both* answers are fluent Swedish and only quality differs, so the
preference head has to learn more than style. A second `choice` question, **naturalness**
("Which candidate response is written in more natural, idiomatic Swedish?"), keeps the
translationese skill available explicitly: it is trained on the original pairs (natural vs
translationese) and on fluent pairs with a 50/50 target (both are natural, so it should not
pick a side).

#### Training data generation

12000 more UltraFeedback pairs (disjoint from the probe; pool of
23244) were rewritten into natural Swedish with the probe's
translation prompt, by kimi-k3, reasoning off, prompt = probe.TRANSLATE_SYSTEM. 11988 translated cleanly
(12 failures); 316 were
dropped because their Swedish prompt nearly duplicates a probe or validation prompt (char n-gram
cosine ≥ 0.8), leaving **11672** pairs, split
train 9921 / val_calib 875 / val_eval 876 with the better answer in slot B
for 50.0% of train pairs.

Before generating, the first 100 probe pairs were re-translated with reasoning off (median 518 completion tokens per pair) and compared with the reasoning-on translations. Lengths are preserved (off/on length ratio 0.996 for the better answer, 1.000 for the worse), character similarity is 0.80 / 0.78, and Kimi-k3 recovers the UltraFeedback label on 86.5% of reasoning-off pairs vs 89.0% of reasoning-on pairs (both orders). The small drop was accepted for training data; the probe itself stays reasoning-on.

Throughput against the self-hosted Kimi-k3 endpoint (8× NVIDIA B300, HGX), 1,000 requests per
chunk, reasoning off: concurrency 64: 1.15–1.79 req/s over 11 chunk(s), concurrency 128: 1.44–1.44 req/s over 1 chunk(s). Concurrency 128 was slower than 64 (longer queueing, more
parse failures), so the remaining chunks ran at 64. Total generation wall-clock:
2.3 h. Variation between chunks at the same concurrency reflects other load on
the shared endpoint.

| Chunk | Concurrency | Requests/s | Output tokens/s | Median latency | Failed / parse failures |
|---|---|---|---|---|---|
| 0 | 64 | 1.77 | 1161 | 33.2 s | 0 / 0 |
| 1 | 128 | 1.44 | 990 | 50.2 s | 0 / 2 |
| 2 | 64 | 1.22 | 820 | 32.7 s | 0 / 1 |
| 3 | 64 | 1.15 | 798 | 34.0 s | 0 / 2 |
| 4 | 64 | 1.79 | 1158 | 32.9 s | 0 / 0 |
| 5 | 64 | 1.74 | 1175 | 34.5 s | 0 / 0 |
| 6 | 64 | 1.77 | 1161 | 33.9 s | 0 / 2 |
| 7 | 64 | 1.45 | 972 | 33.1 s | 0 / 1 |
| 8 | 64 | 1.43 | 942 | 33.2 s | 0 / 1 |
| 9 | 64 | 1.41 | 945 | 33.9 s | 0 / 2 |
| 10 | 64 | 1.27 | 863 | 34.1 s | 0 / 1 |
| 11 | 64 | 1.76 | 1158 | 33.8 s | 0 / 0 |

#### Training

Initialized from the base `convaiinnovations/laya` multilingual checkpoint (not from v1), same
recipe as v1 (lr 2e-05, micro-batch 8 × accumulation
2, warmup + cosine, bf16), 3 epochs over
45342 (pair, question) units; each pair encoded in both orders; one order drawn per item per epoch. Best epoch:
**1** (lowest mean val_calib loss over (source, question) groups). Total 108.7 min on
NVIDIA GeForce RTX 4080 Laptop GPU; peak GPU memory 10.73 GiB.

Accuracy is n/a for fluent/naturalness, whose target is 50/50.

| Epoch | Train loss | original/preference loss / acc | original/naturalness loss / acc | fluent/preference loss / acc | fluent/naturalness loss / acc | Time |
|---|---|---|---|---|---|---|
| 1 | 0.421 | 0.055 / 97.8% | 0.066 / 98.1% | 0.551 / 75.3% | 0.748 / n/a | 35.9 min |
| 2 | 0.266 | 0.046 / 98.7% | 0.042 / 98.7% | 0.700 / 76.5% | 0.743 / n/a | 36.4 min |
| 3 | 0.197 | 0.088 / 98.6% | 0.046 / 98.6% | 1.322 / 73.9% | 0.755 / n/a | 36.4 min |

Temperatures were fitted per question on val_calib, both orders:
`preference` T = 1.446 (NLL 0.280 → 0.265, fit on val_calib original+fluent, both orders), `naturalness` T = 0.575 (NLL 0.066 → 0.057, fit on val_calib original, both orders).

#### v1 vs v2

Both checkpoints were scored with the same code (`python -m laya_gate.v2 evaluate`, v1 via
`--checkpoint checkpoints/best_model --section v1_eval`), on the held-out val_eval halves and the
299-pair probe, each pair in both orders.

| Metric | v1 | v2 |
|---|---|---|
| Original val_eval, preference (order-averaged) | 98.7% | 97.6% |
| Original val_eval, ECE (order-averaged) | 0.009 | 0.023 |
| Fluent val_eval, preference (order-averaged) | 63.4% | 78.8% |
| Fluent val_eval, ECE (order-averaged) | 0.224 | 0.024 |
| **Probe** (reasoning-on), preference [95% CI] | 64.2% [58.9%, 69.6%] | 75.9% [70.9%, 80.6%] |
| Probe, ECE (order-averaged) | 0.230 | 0.033 |
| Probe, positional consistency | 82.3% | 85.3% |
| Probe, share passing margin 0.2 | 83.6% | 65.6% |
| Probe, accuracy on pairs passing the margin | 66.8% | 85.7% |
| Probe, accuracy on pairs routed to the judge | 51.0% | 57.3% |
| GPU forward, batch 1, median | 10.3 ms | 11.0 ms |

For reference, Kimi-k3 scores 87.3% on the probe.

**Naturalness head (v2 only).** On original val_eval it picks the idiomatic answer over the
translationese one 98.1% of the time (ECE
0.017). On fluent pairs, where both answers are natural, its
order-averaged probability stays within 0.2 of 0.5 for 85.4% of
fluent val_eval pairs and 83.3% of probe pairs (mean distance from
0.5: 0.095 and 0.105).

**Probe cascade.** Gate first, Kimi-k3 (cached reasoning-on verdicts from the probe run, judge
alone 87.3%) for instances with `|p − 0.5| < threshold`. Each order is
a separate instance here, so threshold 0 differs slightly from the order-averaged accuracy above.

| Threshold | v1 routed / accuracy | v2 routed / accuracy | v2 expected latency |
|---|---|---|---|
| 0.00 | 0.0% / 64.7% | 0.0% / 75.9% | 11.0 ms |
| 0.05 | 2.8% / 65.4% | 7.9% / 78.8% | 760.1 ms |
| 0.10 | 4.2% / 65.6% | 17.1% / 81.3% | 1636.7 ms |
| 0.15 | 5.5% / 66.1% | 24.7% / 83.8% | 2369.9 ms |
| 0.20 | 7.2% / 66.7% | 33.1% / 84.8% | 3166.8 ms |
| 0.25 | 9.4% / 67.1% | 38.8% / 85.1% | 3708.7 ms |
| 0.30 | 11.9% / 68.9% | 49.5% / 86.6% | 4728.8 ms |
| 0.35 | 16.2% / 70.4% | 59.5% / 86.6% | 5685.1 ms |
| 0.40 | 21.6% / 71.1% | 72.6% / 87.3% | 6928.3 ms |
| **0.45** (recommended) | 31.9% / 73.7% | 88.0% / 88.1% | 8394.7 ms |

Recommended for v2 (lowest routed fraction within 0.5% of the best cascade accuracy in the sweep): threshold 0.45, routing
88.0% for 88.1%; for v1 the same rule gives
0.45 → 31.9% routed, 73.7%.

### v3: single-response questions

**Why:** comparing N candidates pairwise costs N(N−1) ordered forward passes. v3 adds two
questions over a *single* response (state `{"prompt", "response"}`):

- **quality** (`score`, levels 1–5): trained on fluent responses with a soft target from the
  UltraFeedback GPT-4 rating, split between the two nearest levels. The expected level under the
  calibrated distribution is the score.
- **natural_swedish** (`choice`: natural / translationese): original chosen = natural, original
  rejected = translationese, every fluent response = natural.

The pairwise `preference` and `naturalness` questions are trained exactly as in v2.

#### Training

Initialized from the base multilingual checkpoint; lr 2e-05, micro-batch
8 × accumulation 2, warmup + cosine, bf16,
OOM fallback: none. 2 epochs over
110526 units (original/preference 12750 · original/naturalness 12750 · fluent/preference 9921 · fluent/naturalness 9921 · original/natural_swedish 25500 · fluent/quality 19842 · fluent/natural_swedish 19842). Best epoch: **1**
(lowest mean val_calib loss over (source, question) groups); as with v2, the fluent preference loss rises in epoch 2. Total
154.4 min on NVIDIA GeForce RTX 4080 Laptop GPU; peak GPU memory
10.73 GiB. Quality accuracy is exact-level (5-way).

| Epoch | Train loss | fluent/natural_swedish loss / acc | original/natural_swedish loss / acc | original/preference loss / acc | original/naturalness loss / acc | fluent/quality loss / acc | fluent/preference loss / acc | fluent/naturalness loss / acc | Time |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.509 | 0.105 / 95.8% | 0.163 / 94.8% | 0.034 / 98.9% | 0.044 / 98.7% | 1.370 / 43.7% | 0.583 / 77.3% | 0.764 / n/a | 78.5 min |
| 2 | 0.322 | 0.124 / 97.0% | 0.194 / 95.2% | 0.040 / 99.1% | 0.048 / 99.1% | 1.471 / 44.9% | 0.830 / 75.9% | 0.739 / n/a | 75.9 min |

Temperatures, fitted per question on val_calib: `preference` T = 1.941 (NLL 0.275 → 0.233), `naturalness` T = 0.746 (NLL 0.042 → 0.041), `quality` T = 1.286 (NLL 1.370 → 1.358), `natural_swedish` T = 1.482 (NLL 0.137 → 0.124).

#### v2 vs v3 on the pairwise questions

Order-averaged accuracy (both orders scored, probabilities averaged), same evaluation code
(`python -m laya_gate.v3 evaluate`).

| Metric | v2 | v3 |
|---|---|---|
| Original val_eval, preference | 97.6% | 98.9% |
| Original val_eval, naturalness | 98.1% | 98.8% |
| Original val_eval, preference ECE | 0.023 | 0.014 |
| Original, accuracy with chosen in A | 97.8% | 98.8% |
| Original, accuracy with chosen in B | 97.0% | 98.7% |
| Fluent val_eval, preference | 78.8% | 78.8% |
| Fluent val_eval, preference ECE | 0.024 | 0.033 |
| **Probe**, preference [95% CI] | 75.9% [70.9%, 80.6%] | 80.3% [75.6%, 84.6%] |
| Probe, preference ECE | 0.033 | 0.041 |
| Probe, positional consistency | 85.3% | 86.3% |
| GPU forward, batch 1, median (pair) | 11.0 ms | 9.8 ms |

The probe gain (75.9% → 80.3%) is inside
overlapping 95% CIs, so it is suggestive, not established.

#### Quality score

Comparing two answers by their scores (one forward pass each):

| Metric | Fluent val_eval (876 pairs) | Probe (299 pairs) |
|---|---|---|
| Accuracy (higher score = chosen) [95% CI] | 79.1% [76.5%, 82.0%] | 82.9% [78.9%, 87.0%] |
| Spearman vs GPT-4 rating | 0.570 | 0.557 |
| Mean score chosen / rejected | 3.77 / 3.14 | 3.80 / 3.12 |
| ECE of P(chosen scores higher) | 0.107 | 0.139 |

Mean absolute error against the rating on fluent val_eval: 0.86
levels. The ranking is useful; the derived "A beats B" probability is **not** well calibrated
(ECE above 0.085), so use the pairwise question when you need a calibrated margin.
GPU forward for one response: 6.0 ms median.

#### Natural-Swedish check

On original val_eval responses: AUC 0.993, accuracy
93.5% [92.4%, 94.5%], ECE 0.021. It lets
12.2% of deliberate translationese pass as natural and
flags 0.9% of the idiomatic answers; within a pair, the
idiomatic answer gets the higher P(natural) 98.8% of
the time. On fluent responses (all natural) it flags 3.9%
(fluent val_eval) and 2.2% (probe).

#### Probe cascade (v3 preference, Kimi-k3 judge)

Kimi-k3 alone: 87.3%. Each order is a separate instance.

| Threshold | Routed to judge | Cascade accuracy | Expected latency |
|---|---|---|---|
| 0.00 | 0.0% | 79.1% | 9.8 ms |
| 0.05 | 9.5% | 82.6% | 918.3 ms |
| 0.10 | 17.7% | 83.4% | 1699.3 ms |
| 0.15 | 23.7% | 85.3% | 2273.0 ms |
| 0.20 | 31.3% | 88.1% | 2990.3 ms |
| 0.25 | 39.1% | 88.6% | 3739.4 ms |
| **0.30** (recommended) | 46.8% | 88.8% | 4472.6 ms |
| 0.35 | 56.4% | 89.1% | 5381.1 ms |
| 0.40 | 72.2% | 87.8% | 6895.2 ms |
| 0.45 | 88.5% | 87.6% | 8441.2 ms |

Recommended (lowest routed fraction within 0.5% of the best cascade accuracy in the sweep): threshold 0.30, routing
46.8% for 88.8%.

### CPU inference

Checkpoint `checkpoints/v3` on 13th Gen Intel(R) Core(TM) i9-13980HX, 24 threads.
Method: batch 1; collate + forward + logits to host; tokenization excluded; 100 runs over the first 50 val_eval pairs. Probe accuracy is the preference question, both orders averaged;
"original" is the first 300 original val_eval pairs. ΔP is the change in P(A) against torch fp32.

| Variant | Latency median / p95 | Mean input tokens | Probe accuracy | Original accuracy | Mean / max ΔP |
|---|---|---|---|---|---|
| `torch-fp32` | 506.7 ms / 611.5 ms | 720 | 80.3% | 99.0% | 0.000 / 0.000 |
| `torch-int8` | 486.5 ms / 572.0 ms | 720 | 77.6% | 99.0% | 0.073 / 0.596 |
| `onnx-fp32` | 827.2 ms / 1095.5 ms | 720 | 80.3% | 99.0% | 0.000 / 0.000 |
| `onnx-int8` | 448.5 ms / 595.5 ms | 720 | 77.9% | 99.0% | 0.057 / 0.538 |
| `onnx-int8-512` | 147.7 ms / 151.9 ms | 421 | 75.9% | 97.7% | 0.086 / 0.740 |

`torch-int8` is `torch.ao.quantization.quantize_dynamic` on the encoder's linear layers; the ONNX
variants come from `python -m laya_gate.onnx_export` (int8 = ONNX Runtime dynamic
quantization); `-512` caps the input at 512 tokens instead of 1024. Thread sweep for
`onnx-int8` (median): 4 threads 559.6 ms, 8 threads 447.2 ms, 16 threads 580.5 ms, 24 threads 736.7 ms. The sweep's 24-thread median (736.7 ms) differs from the table's (448.5 ms) for the same setting, so expect large run-to-run variance on this hybrid P/E-core laptop CPU; treat the CPU numbers as rough. Files: onnx 1228 MB, onnx-int8 318 MB, torch (safetensors) 614 MB.

No CPU variant comes close to the 35 ms target: the GPU is what makes the gate
a millisecond-scale filter. int8 saves little time at 1024 tokens and costs a few probe points;
the 512-token cap is the only large speed-up, at a larger accuracy cost.

### Machine-translation test (a negative result)

Does the naturalness skill transfer to data no Kimi-k3 model touched? 1989
sentences from openlanguagedata/flores_plus (dev + devtest), professional English->Swedish translations are paired with a machine translation of the same English
sentence by `Helsinki-NLP/opus-mt-en-sv` (20 identical pairs dropped).
Length and content are matched (mean 129 vs
127 characters). Pairwise questions use the prompt
'Översätt följande text till svenska:' + English source; "human preferred" is order-averaged.

| Model | Question | Human translation preferred [95% CI] | Positional consistency | AUC |
|---|---|---|---|---|
| v1 | `preference` (pairwise) | 45.8% [43.6%, 47.9%] | 50.0% | — |
| v2 | `preference` (pairwise) | 42.6% [40.4%, 44.7%] | 47.1% | — |
| v2 | `naturalness` (pairwise) | 44.8% [42.7%, 47.0%] | 54.4% | — |
| v3 | `preference` (pairwise) | 48.2% [46.1%, 50.4%] | 57.4% | — |
| v3 | `naturalness` (pairwise) | 49.1% [47.0%, 51.3%] | 58.7% | — |
| v3 | `natural_swedish/with_source` (single response) | 53.3% [51.2%, 55.6%] | — | 0.520 |
| v3 | `natural_swedish/no_prompt` (single response) | 57.9% [55.8%, 60.0%] | — | 0.554 |

**No version separates professional from machine translation usefully.** The pairwise
questions are at chance or lean toward the MT output; the best signal, v3's single-response
check without a prompt, is above chance but weak (AUC 0.554). The professional reference is itself a
translation, so this is a harder contrast than the deliberate translationese in the training
data, but the result means the gate's "naturalness" is specific to Kimi-k3-style translationese
(literal phrasing, English loan words, US-centric content). Do not use it as a general
machine-translation quality estimator.

### Human-annotated translationese benchmark (a negative result)

An external, human-annotated test: [`liu-nlp/translationese-opensubtitles`](https://huggingface.co/datasets/liu-nlp/translationese-opensubtitles)
(arXiv:2603.08450 (Kunz, Jarochenko & Bollmann); Apache-2.0). 600 English subtitle lines from OpenSubtitles, each
with an OPUS-MT translation, a GPT translation and an idiomatic alternative written by a native
speaker, plus error tags for the OPUS translation and the 10 preceding English lines. No Kimi-k3
model touched it. `python -m laya_gate.liu_eval gate|judge`.

**Method.** The setups are the paper's: `no_prompt` scores the Swedish sentences alone;
`context_N` puts the paper's instruction ("Översätt följande mening till svenska …") with N
preceding English lines in the state's `prompt` field. Trailing punctuation is stripped from
every candidate as in the paper: only 39% of the human versions end in
punctuation vs 98% of the OPUS ones, so without stripping the final period
would be a giveaway. The metric is the paper's: the share of pairs where the human version is
preferred (gate: order-averaged P > 0.5; single-response questions: strictly higher score;
Kimi-k3: preferred in both slot orders). Pairs whose candidates are identical after stripping
are dropped (513 of 600 GPT pairs remain). The
paper's LLM numbers rank sentences by mean log-probability, the gates and Kimi-k3 classify, so
the comparison is of outcomes, not of methods. All 8 setups are in `metrics.json`; the table shows
three. Kimi-k3's "10 lines" setup is a judge prompt with the same context, not the paper's
instruction.

**Human vs OPUS-MT** (600 pairs; 50% = chance, 95% CI about ±4 points):

| Model | No prompt | Instruction, no context | Instruction + 10 lines |
|---|---|---|---|
| Paper: best of its 16 LLMs | 58.3% (gemma-3-12b-it) | 48.7% (gemma-3-12b-it) | 55.7% (gemma-3-12b-it) |
| Paper: median of its 16 LLMs | 49.8% | 39.8% | 44.3% |
| Paper: AI Sweden Llama-3-8B | 56.7% | 43.0% | 50.7% |
| Longer candidate wins (no model) | 62.3% | 62.3% | 62.3% |
| v1 `preference` | 42.8% | 34.3% | 39.5% |
| v2 `preference` | 43.5% | 35.2% | 36.2% |
| v2 `naturalness` | 44.7% | 34.2% | 37.0% |
| v3 `preference` | 45.0% | 37.3% | 40.5% |
| v3 `naturalness` | 44.3% | 39.0% | 40.0% |
| v3 `natural_swedish` | 51.3% | 45.8% | 48.0% |
| v3 `quality` | 50.3% | 39.5% | 42.5% |
| Kimi-k3 judge (both orders agree) | 57.5% | — | 70.7% |

**Every gate prefers the OPUS-MT translationese**, as the paper's LLMs mostly do, and adding the
English source makes it worse (v3 `preference`: 45.0% →
37.3%), the paper's main finding repeated in a classifier.
The best gate result, v3's single-response `natural_swedish` check without a prompt
(51.3%; unstripped 54.5%), is at chance, and no gate beats the trivial
"longer candidate wins" rule. Kimi-k3 does: 70.7% with context, above every
LLM in the paper.

**Human vs GPT translation** (all non-identical pairs / only pairs where annotators judged GPT
worse than the human):

| Model | No prompt, all | No prompt, GPT worse | 10 lines, all | 10 lines, GPT worse |
|---|---|---|---|---|
| Longer candidate wins | 56.9% | 55.1% | — | — |
| v3 `preference` | 45.6% | 46.9% | 40.7% | 38.4% |
| v3 `naturalness` | 44.8% | 44.0% | 39.4% | 39.3% |
| v3 `natural_swedish` | 45.8% | 44.6% | 44.8% | 44.0% |
| v3 `quality` | 47.4% | 47.5% | 47.2% | 43.4% |
| Kimi-k3 judge | 36.8% | 43.1% | 34.5% | 41.3% |

The paper's GPT column counts identical pairs as losses, so it is not comparable and is omitted.
Kimi-k3 prefers the GPT translation overall (the human version wins in both orders for
36.8% / 34.5% of pairs) and is split roughly
evenly on the pairs annotators judged GPT worse (41.3% human
vs 41.6% GPT with context),
consistent with the paper's finding that larger LLMs lean toward LLM-produced translations.

**Per OPUS error tag:** share of pairs where OPUS-MT is preferred (lower is better; tags sorted
by frequency):

| Tag | n | v3 `naturalness` | v3 `natural_swedish` | v3 `preference`, 10 lines | Kimi-k3 | Kimi-k3, 10 lines |
|---|---|---|---|---|---|---|
| SEM | 276 | 57% | 49% | 59% | 27% | 22% |
| BET | 215 | 52% | 52% | 60% | 19% | 9% |
| PR | 164 | 53% | 51% | 59% | 22% | 27% |
| LF | 142 | 52% | 51% | 52% | 21% | 8% |
| DIR | 124 | 48% | 40% | 69% | 16% | 22% |
| SAK | 105 | 70% | 50% | 72% | 18% | 13% |
| GR | 98 | 57% | 48% | 58% | 18% | 14% |
| SL | 34 | 44% | 41% | 74% | 18% | 9% |
| ST | 32 | 62% | 56% | 59% | 31% | 19% |
| ADD | 25 | 56% | 64% | 44% | 40% | 8% |
| ID | 25 | 52% | 48% | 56% | 16% | 12% |

Tags: SEM semantic shift, BET loss of meaning, PR lexical preference, LF incorrect word, DIR direct
translation, SAK missing words, GR grammar, SL slang, ST domain style, ID idiom, ADD added words.
The gate prefers OPUS most where OPUS *omits* words (SAK), even with the English source in the
prompt, so it does not check the translation against the source. The Kimi-k3 columns count
pairs where it picks OPUS in *both* orders (a split verdict counts for neither side); in every
category that happens far less often than the gate's OPUS preference.

**What this means.** The gate's "naturalness" is a detector for the kind of translationese Kimi-k3
wrote into the training data, not for the translationese real MT systems produce. Improving that
needs training pairs with real MT output against native or idiomatic Swedish; this benchmark
should stay held out to measure it.

### Grammar, relevance and Europarl (external benchmarks)

Three more external tests, none used for training, scored with v1, v2 and v3
(`python -m laya_gate.external_eval`). The metric is always the share of pairs in which the
*reference* side wins: the corrected sentence, the question's own answer, the human Swedish.
Pairwise questions average P over both slot orders. Single-response questions compare the two
scores. Kimi-k3 sees a fixed-seed sample of 300 pairs per set in
both orders (9000 requests, 0 parse failures,
median 20.8 s, reasoning on); its accuracy counts a split verdict
as half.

- **DaLAJ-GED** ([Superlim 2](https://huggingface.co/datasets/sbx/superlim-2), test split, CC BY
  4.0): Swedish learner sentences with exactly one error, paired with the same sentence with
  that error corrected, rebuilt from the annotated span (181
  word-order corrections that cannot be rebuilt are dropped). Prompt: none. Natively written
  Swedish; the only difference is the error.
- **SweFAQ** (Superlim 2, dev + test, CC BY-SA 4.0): 219 questions
  from Swedish public-agency FAQ pages; each question's own answer is paired with every other
  answer on the same page. Natively written; the only difference is relevance.
- **Europarl sv/en** ([Språkbanken](https://spraakbanken.gu.se/en/resources/europarl-sven)):
  paragraphs aligned by id, kept when both sides have 1-4 sentences and similar length. Only
  1,587 of 1,162,185 Swedish paragraphs are tagged as *spoken in Swedish*
  (`speakerlang="SV"`), so almost all Swedish Europarl text is itself translation. Two sets:
  English-original speeches (a fixed sample of 1000), with their
  professional Swedish translation against machine translation of the English; and every usable
  Swedish-original speech (1022), native Swedish against a machine
  translation of its English version (a round trip). Machine translation: `opus-mt-en-sv`
  (sentence by sentence) and Kimi-k3 (reasoning off, median 3.1
  s). OPUS-MT's training data includes Europarl; Kimi-k3's is unknown. Prompt: the English
  text with "Översätt följande text till svenska:".

| Set | Pairs | v1 `preference` | v3 `preference` | v3 `naturalness` | v3 `natural_swedish` | v3 `quality` | Baseline | Kimi-k3 |
|---|---|---|---|---|---|---|---|---|
| DaLAJ: corrected vs learner sentence | 1971 | 50.8% | 60.8% | 55.9% | 68.8% | 63.6% | frequency 65.1% | 90.3% |
| SweFAQ: own answer vs other answer | 3240 | 56.7% | 60.3% | 55.8% | 56.1% | 66.8% | overlap 79.7% | 99.0% |
| Europarl EN-original: human vs OPUS-MT | 994 | 43.4% | 40.6% | 41.8% | 48.0% | 36.7% | longer 64.8% | 34.3% |
| Europarl EN-original: human vs Kimi-k3 MT | 995 | 40.1% | 38.7% | 40.0% | 47.8% | 33.8% | longer 61.2% | 17.3% |
| Europarl SV-original: native vs OPUS-MT round trip | 1008 | 46.4% | 40.8% | 43.8% | 51.5% | 36.4% | longer 50.8% | 25.2% |
| Europarl SV-original: native vs Kimi-k3 round trip | 1010 | 45.8% | 37.2% | 39.9% | 48.3% | 32.5% | longer 49.6% | 13.2% |

Baselines: DaLAJ, the sentence with the higher mean unigram log-frequency in Språkbanken's
Europarl-SV word statistics; SweFAQ, the answer sharing more of the question's words; Europarl,
the longer side. Kimi-k3 prompts: a grammar prompt for DaLAJ, the standard judge prompt for
SweFAQ, a translation-quality prompt (English shown) for Europarl. 95% CIs for the gate are about
±2 points (DaLAJ, SweFAQ) and ±3 points (Europarl) and are in `metrics.json`.

**Grammar errors barely register.** v1 is at chance on DaLAJ
(50.8%) and v3's pairwise questions reach
55.9% (`naturalness`) and
60.8% (`preference`). v3's single-response `natural_swedish` check is best at 68.8%,
close to the unigram-frequency baseline (65.1%), while
Kimi-k3 reaches 90.3%. By error type (v3; a sentence can carry two):

| Error type | Pairs | `natural_swedish` | `naturalness` | `preference` |
|---|---|---|---|---|
| M (morphology) | 764 | 72.3% | 59.9% | 63.9% |
| S (syntax) | 402 | 70.4% | 39.3% | 45.5% |
| L (lexical) | 446 | 64.8% | 62.1% | 64.3% |
| O (orthography) | 344 | 82.4% | 79.4% | 81.7% |
| P (punctuation) | 179 | 46.1% | 36.3% | 50.8% |

Spelling errors are the easiest. Punctuation errors are at or below chance, and syntax errors
are below chance for the pairwise question.

**Relevance is weak too.** On SweFAQ the gate does worse than a bag-of-words overlap count
(79.7%); Kimi-k3 is near perfect. Ranking every answer on the
page by v3's `quality` score puts the right one first for 20.1% of
questions (MRR 0.385; chance 9.1%; word overlap
32.9%, MRR 0.488). With pairwise `preference`, the
right answer beats every other answer on its page for 16.0% of questions. On natively
written Swedish where style is held constant, the gate mostly cannot tell a relevant answer
from an irrelevant one.

**Europarl is not a usable "human beats machine" test.** Share of pairs in which the human side
wins:

| Set | v3 `preference`, no prompt | v3 `natural_swedish`, no prompt | Kimi-k3, translation | Kimi-k3, naturalness | Kimi-k3 picks MT in both orders |
|---|---|---|---|---|---|
| Europarl EN-original: human vs OPUS-MT | 51.5% | 51.5% | 34.3% | 56.2% | 59.7% |
| Europarl EN-original: human vs Kimi-k3 MT | 49.4% | 49.9% | 17.3% | 37.0% | 75.7% |
| Europarl SV-original: native vs OPUS-MT round trip | 50.9% | 55.1% | 25.2% | 50.3% | 70.7% |
| Europarl SV-original: native vs Kimi-k3 round trip | 49.3% | 50.1% | 13.2% | 35.2% | 82.3% |

Under the translation prompt Kimi-k3 prefers the machine translation in most pairs, most
strongly against its own output (self-preference), and even for native Swedish speeches against
their round trip. Under the naturalness prompt it is split against OPUS-MT and still prefers its
own output. For those, the translation prompt is confounded: the English was translated *from*
the Swedish, so a back-translation follows it more closely. The 1997-2006 parliamentary Swedish
is often loose or dated ("skall"), so modern MT plausibly reads as at least as good. With the English
source shown, the gate also prefers the MT (v3 `preference`
38.7% vs Kimi-k3 MT). Without it, it is at chance.
Europarl's human side therefore cannot serve as the "better" label for naturalness or
preference training.

**What this means for training.** The gate's two weak spots on natively written Swedish are
grammar and relevance, and both have usable training data in Superlim: DaLAJ-GED train
(35,581 sentences, 18,109
of them with an error) and SweFAQ train (781 questions,
with hard negatives from the same page). Europarl is not worth training on as a
preference signal.

### Context length (1024 vs longer windows)

mmBERT's position embeddings go to 8,192 tokens, but Laya and all three gates were
trained at 1024. Does a longer window help without retraining? Each
checkpoint scored `val_eval` (both orders, `preference`) at 1024, 2048, 4096 tokens
(`python -m laya_gate.external_eval length`). "Truncated" columns are the pairs that do not fit
in 1024 tokens (489 original pairs for v3). Latency is the batch-1 GPU
forward pass on the 50 longest original pairs at that limit (bf16, `sdpa` attention).

| Model | Max tokens | Original truncated | Original acc. | on truncated | Fluent acc. | on truncated | Input tokens | Forward p50 / p95 |
|---|---|---|---|---|---|---|---|---|
| v1 | 1024 | 41.9% | 98.7% | 99.1% | 63.4% | 72.0% | 1024 | 10.9 ms / 11.9 ms |
| v1 | 2048 | 10.4% | 98.9% | 99.6% | 63.4% | 72.0% | 2048 | 26.0 ms / 31.4 ms |
| v1 | 4096 | 0.0% | 99.0% | 99.8% | 63.4% | 72.0% | 2970 | 50.1 ms / 76.0 ms |
| v2 | 1024 | 42.2% | 97.6% | 98.5% | 78.8% | 86.3% | 1017 | 11.1 ms / 11.7 ms |
| v2 | 2048 | 10.4% | 97.6% | 98.5% | 78.7% | 84.3% | 2041 | 26.7 ms / 31.8 ms |
| v2 | 4096 | 0.0% | 97.6% | 98.5% | 78.7% | 84.3% | 2970 | 51.0 ms / 75.8 ms |
| v3 | 1024 | 43.7% | 98.9% | 99.8% | 78.8% | 84.3% | 977 | 11.0 ms / 11.6 ms |
| v3 | 2048 | 11.2% | 98.9% | 99.8% | 78.8% | 84.3% | 2001 | 26.4 ms / 31.6 ms |
| v3 | 4096 | 0.0% | 98.9% | 99.8% | 78.8% | 84.3% | 2970 | 50.7 ms / 75.8 ms |

**No meaningful gain.** Seeing the rest of long pairs changes accuracy by under a point on the
original pairs and not at all, or slightly downward, on the fluent ones. The truncated pairs
are already *easier* than the rest, because the first 1024 tokens settle
them. Forward latency more than doubles at 2048, where the p95 forward pass alone leaves no
room for tokenization within the 35 ms target; at 4096 the forward pass alone exceeds it. The window
stays at 1024; fine-tuning at a longer length was not tried.

### v4 ablation: Europarl, DaLAJ and SweFAQ as training data

Which extra training data makes the gate a better judge of Swedish? A controlled ablation
(`python -m laya_gate.v4`): every arm fine-tunes the base checkpoint with the v3 recipe (v3 data
and questions, 1 epoch, the same seed and hyperparameters); only the added data differs. A second
control seed measures seed-to-seed noise.

The derived training and evaluation pairs are on the Hub: [jonasaise/laya-swedish-gate-v4-data](https://huggingface.co/datasets/jonasaise/laya-swedish-gate-v4-data) (one config per source, each under its source's license).

- **+Europarl**: 12,000 English-original speeches (none from the Europarl test
  sample), professional Swedish translation > Kimi-k3 machine translation, as `preference`
  (English source in the prompt) and `naturalness` (no prompt). The negatives are Kimi-k3's
  only, so OPUS-MT never appears in training. Length-balanced: the human side is longer in
  exactly half the pairs (57.3% before balancing).
- **+DaLAJ/SweFAQ**: 16,365 DaLAJ-GED train pairs (corrected > learner
  sentence, `naturalness`) and 3,074 SweFAQ train pairs (own answer >
  another answer on the same FAQ page, `preference`; no page shared with the test questions).
- **+both**: both. **+grammar**: +both plus a new single-response question, `grammatical`
  (correct / contains an error), trained on both sentences of every DaLAJ pair; its
  temperature is fitted on DaLAJ dev.
- **v4 (+adequacy)**: +grammar plus negatives that can only be told apart with the source,
  built from 3,000 further Europarl paragraphs: the
  correct translation beats another paragraph's human translation, a half translation, a
  word-shuffled translation and the untranslated English, and Kimi-k3's correct translation
  beats another paragraph's human translation; the English is neither natural nor grammatical
  Swedish (14989 `preference`, 3000 `naturalness`
  and 6000 single-response rows).

All tests are held out from every arm: DaLAJ and SweFAQ dev/test splits; ScaLA-sv is EuroEval's
own test split (`mteb/scala_sv_classification`); GerLangMod-sv the native Talbanken test file
(lowercased, no punctuation, as distributed); WMT24++ en→sv is EuroEval's translation set
(`google/wmt24pp`, 960 post-edited segments with OPUS-MT and Kimi-k3 translations). Each cell is
the share of items where the reference side wins (corrected sentence, own answer, human Swedish).
**\*** marks a difference from control whose paired-bootstrap 95% CI excludes zero; the two
control seeds differ by up to 4.2 points on a single
test, so smaller differences are noise even when starred.

| Test | v3 (published) | control | control, seed 2 | +Europarl | +DaLAJ/SweFAQ | +both | +both +grammar | v4: +grammar +adequacy | **v4.1**: v4 +registers +hard sources | v4.3: v4.1 +Wikipedia (not released) |
|---|---|---|---|---|---|---|---|---|---|---|
| **In-domain (DPO validation, fluent pairs, probe)** | | | | | | | | | | |
| original/preference | 98.9% | 98.9% | 99.0% | 98.7% | 99.0% | 99.0% | 98.7% | 98.7% | 98.8% | 98.7% |
| fluent/preference | 78.7% | 77.7% | 76.6% | 74.5% * | 77.6% | 77.7% | 79.1% | 77.4% | 76.6% | 77.1% |
| probe/preference | 79.9% | 78.3% | 77.9% | 76.9% | 78.9% | 76.9% | 79.3% | 78.6% | 79.3% | 81.9% |
| **Native speaker vs OPUS-MT (liu-nlp, human-annotated)** | | | | | | | | | | |
| liu/naturalness | 44.3% | 43.5% | 44.3% | 58.8% * | 47.3% | 53.7% * | 52.7% * | 52.2% * | 51.0% * | 52.3% * |
| liu/natural_swedish | 51.6% | 49.5% | 52.8% | 52.2% | 48.4% | 53.0% | 48.5% | 52.2% | 51.4% | 48.5% |
| liu/preference+source | 37.3% | 36.5% | 34.7% | 60.8% * | 39.5% | 50.3% * | 66.0% * | 53.0% * | 57.5% * | 54.8% * |
| **Professional translation vs MT (FLORES+, WMT24++)** | | | | | | | | | | |
| flores/naturalness | 58.1% * | 55.5% | 56.8% | 58.5% | 59.5% * | 61.0% * | 58.2% | 65.2% * | 62.0% * | 62.9% * |
| flores/preference+source | 48.5% | 47.2% | 46.2% | 73.3% * | 53.0% * | 77.0% * | 76.1% * | 73.8% * | 71.9% * | 71.4% * |
| wmt24pp/opus/naturalness | 63.1% | 64.5% | 64.2% | 66.9% | 68.9% * | 66.7% | 63.1% | 68.5% * | 68.5% * | 69.1% * |
| wmt24pp/opus/preference+source | 54.3% | 51.6% | 55.8% * | 76.3% * | 62.1% * | 74.9% * | 71.5% * | 74.6% * | 72.8% * | 73.2% * |
| wmt24pp/kimi/preference+source | 43.1% | 40.1% | 42.5% | 66.7% * | 41.7% | 71.8% * | 72.9% * | 58.9% * | 59.8% * | 58.3% * |
| **Grammar (DaLAJ, GerLangMod-sv, ScaLA-sv)** | | | | | | | | | | |
| dalaj/naturalness | 55.9% * | 60.4% | 57.1% * | 52.2% * | 90.8% * | 87.7% * | 88.6% * | 89.7% * | 88.0% * | 89.5% * |
| dalaj/natural_swedish | 68.8% | 66.8% | 65.6% | 61.8% * | 78.3% * | 74.4% * | 84.6% * | 83.6% * | 81.3% * | 83.8% * |
| dalaj/grammatical | — | — | — | — | — | — | 83.2% | 83.2% | 81.4% | 82.1% |
| gerlangmod/naturalness | 83.3% | 85.1% | 81.6% * | 32.5% * | 94.5% * | 89.2% * | 93.6% * | 95.5% * | 95.7% * | 96.4% * |
| gerlangmod/grammatical | — | — | — | — | — | — | 88.0% | 47.8% | 86.3% | 90.8% |
| blimppi/naturalness | 40.7% * | 13.6% | — | — | — | — | — | 86.4% * | 89.8% * | 91.5% * |
| blimppi/grammatical | — | — | — | — | — | — | — | 84.7% | 71.2% | 67.8% |
| multiblimp/naturalness | 83.1% | 80.1% | — | — | — | — | — | 97.0% * | 97.5% * | 96.0% * |
| multiblimp/grammatical | — | — | — | — | — | — | — | 94.0% | 95.3% | 93.3% |
| wikiedits/naturalness | 51.4% | 52.6% | — | — | — | — | — | — | 56.6% | 61.0% * |
| wikiedits/grammatical | — | — | — | — | — | — | — | — | 53.5% | 57.9% |
| **Relevance (SweFAQ)** | | | | | | | | | | |
| swefaq/preference | 60.3% | 60.8% | 60.0% | 54.6% * | 88.7% * | 87.5% * | 87.7% * | 88.9% * | 88.6% * | 88.3% * |
| swefaq/quality | 66.7% | 67.9% | 64.9% * | 65.2% * | 76.6% * | 68.0% | 71.7% * | 72.8% * | 71.3% * | 71.0% * |
| swefaq/quality_top1 | 20.1% | 16.9% | 16.4% | 17.8% | 27.9% * | 20.1% | 27.4% * | 20.1% | 23.3% * | 26.0% * |
| **Europarl (in-distribution for the Europarl arms)** | | | | | | | | | | |
| europarl_en/kimi/preference+source | 38.7% | 35.9% | 39.8% * | 85.4% * | 35.6% | 87.8% * | 87.3% * | 77.4% * | 75.9% * | 74.6% * |
| europarl_sv/kimi/preference+source | 37.2% * | 33.7% | 35.7% | 88.8% * | 36.2% | 91.0% * | 90.0% * | 83.6% * | 82.8% * | 82.2% * |
| ScaLA-sv MCC (EuroEval metric), best single-response question | 0.146 | 0.117 | 0.100 | 0.055 | 0.167 | 0.114 | 0.318 | 0.341 | 0.466 | 0.513 |
| Fluent LLM answers flagged as translationese (lower is better) | 3.9% | 4.5% | 3.9% | 4.9% | 4.1% | 3.8% | 3.4% | 3.3% | 3.9% | 3.4% |
| ECE, DPO validation `preference` | 0.0108 | 0.0120 | 0.0138 | 0.0151 | 0.0139 | 0.0157 | 0.0102 | 0.0142 | 0.0164 | 0.0139 |
| Training units / wall-clock | 2 epochs | 110,526 / 1.3 h | 110,526 / 1.3 h | 134,526 / 1.5 h | 129,965 / 1.5 h | 153,965 / 1.7 h | 186,695 / 2.0 h | 210,684 / 2.1 h | 223,521 / 2.8 h | 235,521 / 2.3 h |

#### Sanity tests: a shortcut in the Europarl pairs

FLORES+ devtest (the pilot's 200 sentences), the professional reference against a degraded
alternative, English source in the prompt for `preference`; share of pairs where the reference
wins (`python -m laya_gate.v4 sanity`):

| The reference translation against … | v3 (published) | control | control, seed 2 | +Europarl | +DaLAJ/SweFAQ | +both | +both +grammar | v4: +grammar +adequacy | **v4.1**: v4 +registers +hard sources | v4.3: v4.1 +Wikipedia (not released) |
|---|---|---|---|---|---|---|---|---|---|---|
| the reference of a *different* sentence | 91.0% | 90.5% | 94.0% | 2.5% | 97.5% | 0.5% | 0.0% | 78.0% | 89.5% | 86.5% |
| LFM2-350M's translation (pilot's lowest chrF) | 97.0% | 99.0% | 99.0% | 57.5% | 98.5% | 49.0% | 3.5% | 98.0% | 97.5% | 98.0% |
| only the first half of the reference | 72.0% | 81.0% | 85.5% | 59.5% | 99.5% | 14.5% | 7.5% | 100.0% | 100.0% | 100.0% |
| the untranslated English | 20.5% | 14.5% | 12.5% | 99.5% | 47.5% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| the untranslated English (`naturalness`, no prompt) | 87.5% | 82.0% | 83.5% | 56.5% | 87.5% | 45.5% | 38.5% | 99.5% | 99.5% | 99.5% |

**Every arm trained on the Europarl pairs alone learned "the candidate least like the English
source wins".** The human side of a Europarl pair is always the less literal one, so that rule
fits the training pairs perfectly. It also explains part of the large liu-nlp, FLORES+ and
WMT24++ gains of +Europarl, +both and +grammar: human translations are less literal than MT
there too. These arms pick a fluent translation of the *wrong* sentence over the right one
(the reference wins in 2.5% to
0.0% of pairs); +both and +grammar also lose
to half translations and to LFM2's nonsense; and on the pilot's 4,400 real model translations
their translation score is uncorrelated or negatively correlated with chrF (Spearman per translation:
+both -0.08, +grammar -0.19; v3 0.30).

**The adequacy negatives remove the shortcut.** v4 picks the reference over a half translation,
garbage MT and untranslated English in 100.0%,
98.0% and
100.0% of pairs, and over another sentence's
reference in 78.0% (v3:
91.0%). These tests were built the same way as the
adequacy negatives, so they show the fix took, not independent generalization. The independent
check is the pilot: v4's translation score agrees with chrF per translation at
0.38, the highest of any checkpoint. What remains of the translation gains
after the fix is smaller but still clear: FLORES+ `preference` 47.2%
→ 73.8%, WMT24++ against OPUS-MT
51.6% → 74.6%
and against Kimi-k3 40.1% →
58.9%, liu-nlp
36.5% → 53.0%.

**DaLAJ and SweFAQ fix the two weak spots they target**, in every arm that includes them:
grammar 60.4% → 89.7%
(DaLAJ) and 85.1% →
95.5% (GerLangMod-sv), relevance
60.8% → 88.9%
(SweFAQ). The single-response `grammatical` question raises ScaLA-sv's MCC from
0.117 (control) to 0.341 (v4).

**Costs.** As a stand-alone yes/no check, `grammatical` flags much good Swedish as incorrect at
threshold 0.5: 40.4% of fluent LLM answers,
59.9% of WMT24++ references,
42.2% of FLORES+ references (it was trained
on single learner sentences). It is useful for comparing two versions of a text, not as a
verdict on one. On GerLangMod-sv (lowercased, no punctuation) v4's `grammatical` drops to
47.8% from 88.0%
with +grammar. In-domain accuracy, calibration and latency stay at the control level. Each arm was
trained once.

#### v4.1: correct text from more registers, harder wrong-source negatives

v4.1 adds two sources to v4 aimed at its two weak spots
(`python -m laya_gate.v4 run --arm v41`), none from the UD treebanks behind ScaLA-sv,
GerLangMod-sv, BLiMPPI and MultiBLiMP:

- **Registers**: 3,779 correct texts from four training-only sources
  (SweFAQ train answers, DaLAJ's correct textbook sentences, the fluent training answers, unused
  Europarl paragraphs), each against a corrupted copy (a word deleted or two neighbours swapped,
  as in ScaLA), as `naturalness` pairs and `grammatical` single responses; a fifth lowercased
  without punctuation.
- **Hard wrong source**: 1,500 pairs where the correct human or Kimi-k3
  translation beats the translation of the *adjacent* paragraph of the same sitting.

Against v4 on the same items: the `grammatical` question's ScaLA-sv MCC rises from
0.341 to
0.466, GerLangMod-sv
from 47.8% to 86.3%,
and over-flagging of good Swedish falls (fluent LLM answers
40.4% →
25.2%, FLORES+ references
42.2% →
26.7%, WMT24++ references
59.9% →
47.9%), though it is still far from a
reliable stand-alone verdict. The wrong-source sanity test goes from
78.0% to
89.5% (v3:
91.0%). The cost is 1-3 points on several tests
(FLORES+, DaLAJ, SweFAQ `quality`), within the seed noise. v4.1 is the release.

**Minimal pairs need n-gram baselines** (BLiMPPI's lesson, arXiv:2610.13607). Share of pairs where
the grammatical side wins, for model-free baselines and v4.1: the longer side, the more frequent
words (Europarl-SV statistics), a word trigram model trained on
23,875,904 Europarl-SV tokens
(`python -m laya_gate.ngram build`, `python -m laya_gate.v4 baselines`):

| Set | Pairs | Longer wins | Unigram frequency | Trigram | v4.1 `naturalness` | v4.1 `grammatical` |
|---|---|---|---|---|---|---|
| dalaj | 1971 | 64.4% | 65.1% | 80.5% | 88.0% | 81.4% |
| gerlangmod | 844 | 50.0% | 50.1% | 90.9% | 95.7% | 86.3% |
| blimppi | 59 | 93.2% | 89.8% | 93.2% | 89.8% | 71.2% |
| multiblimp | 201 | 5.2% | 99.3% | 98.8% | 97.5% | 95.3% |
| wikiedits | 664 | 51.3% | 45.9% | 49.8% | 56.6% | 53.5% |

BLiMPPI (polarity items: the licenser is removed) and MultiBLiMP (subject-verb number agreement;
the wrong side uses a plural verb form, archaic in modern Swedish) are new held-out tests here.
Both are nearly solved by length or the trigram model: in BLiMPPI the grammatical sentence is
the longer one in nearly every pair, in MultiBLiMP the shorter. The gate's lead over the trigram
is clearest on DaLAJ and GerLangMod-sv.

**WikiEdits and v4.3 (not released).** [`lang-uk/WikiEdits-MultiGEC`](https://huggingface.co/datasets/lang-uk/WikiEdits-MultiGEC)
(MIT) holds human edits of Swedish Wikipedia text (September 2024 - May 2025). Keeping edits that
change words but not numbers and do not rewrite more than a fifth of the text leaves
664 pairs from 318 pages, a native, multi-sentence test of
"the edited version is better" that no baseline solves (trigram
49.8%, longer wins
51.3%).
Many edits are stylistic rather than grammatical, so this is a hard test. v4.3 adds to v4.1
4,000 Swedish Wikipedia paragraphs (`wikimedia/wikipedia` 20231101.sv,
CC BY-SA 3.0; every WikiEdits page and bot-generated stub text excluded), each against a
corrupted copy, to teach that native encyclopedic text is correct. Against v4.1: WikiEdits
`naturalness` 56.6% → 61.0%,
`grammatical` 53.5% → 57.9%,
ScaLA-sv MCC 0.466 →
0.513; but over-flagging did
not fall (fluent LLM answers flagged 25.2%
→ 27.8%, edited Wikipedia text
25.2% →
23.5%), and the wrong-source sanity test
slipped (89.5% →
86.5%). A better grammar ranker, not a fix for
over-flagging; v4.1 stays the release.

### Pilot: predicting EuroEval Swedish results

Can the gate act as a quick predictor of EuroEval's Swedish results? A pilot
(`python -m laya_gate.pilot`): 28 open LLMs (up to 11B parameters; 6 loaded in 4-bit NF4
to fit a 12 GB GPU) that have Swedish results on the
[EuroEval leaderboard](https://euroeval.com) (overall Swedish rank 1.94
for `speakleash/Bielik-11B-v2.3-Instruct` to 4.06 for `HuggingFaceTB/SmolLM2-360M-Instruct`; lower is
better) each translate the same 200 FLORES+ devtest sentences
into Swedish and answer the same 100 Swedish prompts (greedy,
thinking off). The gate scores every output; the model-level means are correlated with the
EuroEval Swedish rank (Spearman, bootstrap 95% CI over models). Model size alone predicts rank
well on this range, so the second column controls for it (ranks residualized on log parameter
count).

| Model-level score | Spearman with EuroEval Swedish rank | controlling for size |
|---|---|---|
| v4.1 `quality` of the model's answers | 0.66 [0.33, 0.85] | 0.63 [0.25, 0.81] |
| v4 `quality` of the model's answers | 0.73 [0.44, 0.88] | 0.68 [0.29, 0.84] |
| v3 `quality` of the model's answers | 0.73 [0.43, 0.89] | 0.66 [0.29, 0.84] |
| v4.1 P(model translation beats the reference) | 0.72 [0.40, 0.90] | 0.42 [-0.08, 0.78] |
| same, +grammar arm (before the adequacy fix) | -0.31 [-0.62, 0.09] | -0.20 [-0.55, 0.24] |
| v4.1 `natural_swedish` of the answers | 0.27 [-0.13, 0.61] | 0.34 [-0.10, 0.66] |
| v4.1 `grammatical` of the answers | 0.19 [-0.21, 0.54] | 0.04 [-0.43, 0.48] |
| chrF against the FLORES+ reference (no gate) | 0.83 [0.60, 0.93] | 0.48 [-0.02, 0.79] |
| log parameter count (no gate) | 0.83 [0.64, 0.92] | — |
| share of answers in Swedish (no gate) | -0.01 [-0.35, 0.34] | -0.26 [-0.61, 0.14] |

**The single-response `quality` score of a model's Swedish answers tracks the leaderboard, and
not only through model size.** No reference answers are needed. It correlates with the task scores too (linguistic acceptability 0.65, summarization 0.73, reading comprehension 0.53, knowledge 0.59). The
translation score only becomes predictive once the Europarl shortcut is removed (the +grammar
arm's score correlates negatively with rank). v4.1's `quality` correlation is slightly lower than
v4's and v3's, well within the intervals. EuroEval ran full precision; on the 22 models run in
bf16 alone the v4.1 `quality` correlation is 0.68, so the 4-bit models do not drive
it. 8 further models are missing: they ran out of GPU memory next to another process
on the shared GPU, except `google/gemma-3n-E4B-it`, which needs Pillow for its vision tower
(`BSC-LT/salamandra-7b-instruct`, `NbAiLab/borealis-12b-instruct-preview`, `Qwen/Qwen3.5-9B`, `google/gemma-3-12b-it`, `google/gemma-3n-E4B-it`, `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`, `tencent/Hunyuan-7B-Instruct`). With 28 models the intervals are still wide; a pilot,
not a calibrated predictor. The models' outputs are not redistributed (their licenses differ); the
scores are in `metrics.json`.

### Human preference on Swedish answers (Chatbot Arena)

The only human-judged test of *answer quality* in Swedish we could find without new annotation:
[Chatbot Arena](https://lmarena.ai) votes, where users write a prompt, read two anonymous answers and
vote for the better one (`python -m laya_gate.arena_eval`). Sources:
`lmarena-ai/arena-human-preference-140k` (CC BY 4.0), `-100k`, `-55k` (Apache-2.0) and
`lmsys/chatbot_arena_conversations` (gated); single-turn
battles whose prompt and both answers read as Swedish, deduplicated: 203
battles, 129 decisive votes, 39 ties and
35 "both bad". Never used for training. No Kimi model appears in these battles,
so Kimi-k3 is not judging its own output. 38.0% of the pairs
exceed the gate's 1,024-token window. With 129 votes the 95% CI is about ±8 points.

| Agreement with the human vote | All decisive | Human picked the longer answer | Human picked the shorter (46) |
|---|---|---|---|
| v3 `preference` | 62.8% [55%, 71%] | 77.1% | 37.0% |
| v3 `quality` | 65.9% [57%, 74%] | 85.5% | 30.4% |
| v4 `preference` | 62.0% [53%, 71%] | 74.7% | 39.1% |
| v4 `quality` | 64.3% [56%, 72%] | 80.7% | 34.8% |
| v4.1 `preference` | 58.9% [50%, 67%] | 69.9% | 39.1% |
| v4.1 `quality` | 60.5% [52%, 69%] | 74.7% | 34.8% |
| v4.2a `preference` | 63.6% [56%, 72%] | 74.7% | 43.5% |
| v4.2a `quality` | 62.8% [54%, 71%] | 77.1% | 37.0% |
| v4.2b `preference` | 62.0% [53%, 70%] | 71.1% | 45.7% |
| v4.2b `quality` | 62.8% [54%, 71%] | 77.1% | 37.0% |
| v4.3 `preference` | 58.1% [50%, 66%] | 71.1% | 34.8% |
| v4.3 `quality` | 59.7% [51%, 68%] | 77.1% | 28.3% |
| Longer answer wins (no model) | 64.3% [56%, 72%] | 100% | 0% |
| Kimi-k3 judge (both orders) | 74.8% [68%, 82%] | 75.3% | 73.9% |

**On human votes about Swedish answers, the gate mostly measures length.** People preferred the
longer answer in 64.3% of these battles, and the gate picks the longer
answer about as often (v3 72%, v4 70%, v4.1 67%, v4.2a 68%, v4.2b 65%, v4.3 69%). Where people
preferred the shorter answer, the gate agrees with them less often than a coin flip. Kimi-k3 agrees
with the human vote equally often either way: it judges content, the gate does not. The v4 and v4.1
training (grammar, relevance, translation) did not help here; their accuracy is a few points below
v3's, within the noise. Ties are not kept near 0.5 either (share within 0.2 of 0.5:
v3 56%, v4 26%, v4.1 36%, v4.2a 44%, v4.2b 46%, v4.3 36%).

What this means: the gate is a useful fast check for Swedish *form* (grammar, translationese,
translation adequacy, relevance on FAQ pages), not yet for which of two fluent answers people
prefer. Improving that needs human preference data on Swedish answers, which is scarce: this set is
small and must stay a test set.

**Can training remove the length bias? (v4.2, not released.)** Both base preference sources
favour length: the chosen answer is the longer one in
70.6% of the original DPO pairs and
62.2% of the fluent pairs. Two arms on top of v4.1 (`python -m laya_gate.v4 run --arm v42a|v42b`):
**v4.2a** drops "longer wins" pairs from the base `preference` data until it is 50/50; **v4.2b**
also adds 5,417 length-controlled pairs: for 2901 training
prompts Kimi-k3 wrote a correct and a flawed answer (factual error or missing the point), each
short and long, and the pairs pit *short correct vs long flawed* and *long correct vs short
flawed*, kept when Kimi-k3 picks the correct one in both orders (with `quality` targets 4.5 vs 2).
A held-out test built the same way from 289 other prompts
(539 pairs) and arena_sv measure it:

| Model | Length test: correct is shorter, `preference` | … `quality` | Length test: correct is longer | arena_sv `preference` | arena_sv, human picked shorter |
|---|---|---|---|---|---|
| v3 | 27.8% | 25.9% | 87.2% | 62.8% | 37.0% |
| v4.1 (published) | 28.6% | 23.7% | 84.2% | 58.9% | 39.1% |
| v4.2a: length-balanced base pairs | 29.3% | 21.8% | 83.2% | 63.6% | 43.5% |
| v4.2b: + length-controlled pairs | 77.4% | 81.2% | 85.3% | 62.0% | 45.7% |

The gates before v4.2b pick a long flawed answer over a short correct one about three times in
four. v4.2b learns to judge these pairs independently of length, but the gain barely carries over
to the human votes, and it costs elsewhere (SweFAQ `quality` and grammar over-flagging get
worse; the EuroEval pilot correlation drops). Balancing the base data alone (v4.2a) does not
touch the bias. Synthetic "correct vs flawed" pairs do not teach what people prefer in Swedish
answers; that needs human preference data.

### Tests

`pytest -v --cov=src/laya_gate (all markers; skips listed)`: **171 passed, 0 failed,
0 errors, 0 skipped** of 171.
Coverage: **44.7%** overall — `__init__.py` 100.0% · `arena_eval.py` 37.4% · `benchmark.py` 63.2% · `calibrate.py` 64.3% · `cpu.py` 0.0% · `data.py` 70.9% · `external_eval.py` 38.5% · `fluent.py` 20.6% · `judge.py` 79.3% · `liu_eval.py` 36.6% · `mtqa.py` 0.0% · `ngram.py` 61.0% · `onnx_export.py` 87.2% · `pilot.py` 37.1% · `predictor.py` 81.1% · `probe.py` 46.0% · `publish.py` 57.7% · `publish_data.py` 0.0% · `readme.py` 69.6% · `testing.py` 0.0% · `train.py` 37.3% · `utils.py` 100.0% · `v2.py` 29.4% · `v3.py` 18.0% · `v4.py` 28.3%.


## Limitations

- **The labels encode translation quality, by construction.** Per the dataset card, kimi-k3 localized each prompt and chosen response into idiomatic Swedish and *deliberately* wrote the rejected response as poor translation (literal phrasing, English loan words, US-centric content). A TF-IDF logistic regression reaches 98.4% on the same held-out pairs, so most of the gate's accuracy is detecting translationese. Measured directly: on 299 pairs where *both* answers are natural Swedish and only quality differs, the gate is right 64.2% of the time (95% CI 58.9%–69.6%) versus 87.3% for Kimi-k3, and its calibration breaks down (ECE 0.230). Do not use it as a general helpfulness or correctness judge for two *fluent* Swedish answers.
- **The judge generated the data.** Kimi-k3 wrote both sides of every pair, so its agreement with the labels measures self-consistency, not independent quality judgement.
- **Pairwise preference, not fact-checking.** The gate predicts which of two responses the dataset labels as chosen. It does not verify factual correctness, and a fluent wrong answer can beat a terse correct one.
- **1024-token window.** 42.3% of pairs exceed it; the gate then sees only the end of the prompt and the start of each response, so differences late in long answers are invisible to it. Scoring at 2048 or 4096 tokens without retraining did not help (see Context length) and costs latency.
- **Kimi-k3 is a reference, not ground truth.** The judge is a strong but imperfect model; agreement with it is not correctness.
- **One training run per arm.** Two seeds of the v4 control arm differ by up to 4.2 points on a single test; smaller differences between arms are noise.
- **Naturalness does not transfer to machine translation.** On professional vs `opus-mt-en-sv` translations of FLORES+ sentences no version separates the two usefully (AUC ≤ 0.56, see the machine-translation test); it detects Kimi-k3-style translationese, not machine translation in general. The same holds on the human-annotated translationese benchmark (liu-nlp/translationese-opensubtitles): every gate prefers the OPUS-MT translation over the native speaker's alternative.
- **Grammar and relevance are weak spots.** On natively written Swedish the gate barely prefers a corrected sentence over a learner error (DaLAJ, best v3 question 68.8%) and loses to a word-overlap baseline at picking a question's own FAQ answer (SweFAQ, v3 `quality` 66.8% vs 79.7%); Kimi-k3 gets 90.3% and 99.0%.
- **v4.1's `grammatical` question over-flags.** At threshold 0.5 it marks 25.2% of fluent LLM answers and 47.9% of WMT24++ references as incorrect; use it to compare two versions of a text, not as a verdict on one.
- **Translation preference is source-sensitive only after the adequacy fix.** Europarl human-vs-MT pairs alone teach 'the less literal side wins' (see the v4 sanity tests); v4.1 still picks a fluent translation of the wrong sentence in 10.5% of FLORES+ pairs.
- **Answer preference largely tracks length.** On 129 human Chatbot Arena votes on Swedish answers the gate agrees with people 58.9% (v4.1), close to 'the longer answer wins' (64.3%); Kimi-k3 reaches 74.8%.
- **CPU is slow.** A pair takes 506.7 ms (median) on CPU with torch fp32; int8 and ONNX do not get anywhere near the GPU's millisecond latency.

## Conclusion

The fine-tuned gate reaches 98.4% accuracy on held-out pairs (the zero-shot base checkpoint: 51.6%), with calibrated probabilities (ECE 0.006, meeting the ≤ 0.085 target) and a median single-pair GPU forward pass of 9.9 ms (meeting the 35 ms target). It picks the same winner in both presentation orders for 98.4% of pairs. When it is confident (margin ≥ 0.2, 99.4% of predictions) it is right 98.6% of the time, so its confidence is usable for routing. That accuracy should be read next to the lexical baseline: a TF-IDF logistic regression scores 98.4% on the same pairs, because the dataset contrasts idiomatic Swedish with deliberately poor translation. The gate does not measurably beat that baseline (+0.0 points); what it adds is calibrated confidence, order-robustness and a single model that also reads the prompt. The gate is a strong translationese detector. It is a weak quality judge: on 299 pairs where *both* answers are natural Swedish and only quality differs, the gate is right 64.2% of the time (95% CI 58.9%–69.6%) versus 87.3% for Kimi-k3, and its calibration breaks down (ECE 0.230).

Against these labels Kimi-k3 is *less* accurate than the gate (95.5% vs 98.6%) and 802× slower, which is expected: the labels mark translationese, which the gate was trained on, while the judge was also asked about helpfulness and correctness, and it generated these pairs itself. The judge still helps where the gate is unsure: routing the 5.2% least-confident instances (threshold 0.45) lifts accuracy from 98.6% to 99.1% at an expected 507.5 ms per pair. The pre-registered rule (cheapest threshold within 0.5 points of the best) picks 0.00, i.e. 0.0% routed, because the gain is that small. For its intended System 1 role on data like this, settling confident comparisons in milliseconds and escalating the rest, the gate is fit for purpose (latency target met, ECE target met, confident-subset accuracy 98.6% vs judge 95.5%). It should not be the only judge for high-stakes decisions.

A **v2** checkpoint (`python -m laya_gate.v2`, see [v2](#v2-training-on-fluent-pairs-too)) adds 9921 fluent-vs-fluent training pairs and a separate naturalness question. On the probe it reaches 75.9% (95% CI 70.9%–80.6%) vs 64.2% for v1, with ECE 0.033 instead of 0.230, so its margin routes mistakes to the judge again: with about 32% of probe instances routed to Kimi-k3, the cascade reaches 84.8% with v2 (33.1% routed) vs 73.7% with v1. The cost is 98.7% → 97.6% on the original held-out pairs (both orders averaged), and the probe is still 11.4 percentage points behind Kimi-k3. v2 is not on the Hub yet.

A **v3** checkpoint (`python -m laya_gate.v3`, see [v3](#v3-single-response-questions)) keeps v2's two pairwise questions and adds two single-response questions: a 1–5 **quality** score and a **natural Swedish vs translationese** check. On the probe its pairwise preference reaches 80.3% [75.6%, 84.6%] (v2: 75.9%), and ranking the two answers by their quality scores reaches 82.9% [78.9%, 87.0%] with one forward pass per answer (Kimi-k3: 87.3%). On the original held-out pairs v3 is back to 98.9% (v2: 97.6%). On human vs machine translations of FLORES+ sentences, no version is useful (see [the machine-translation test](#machine-translation-test-a-negative-result)). v3 is not on the Hub yet.

**v4.1** closes the two gaps the external benchmarks exposed: grammar (DaLAJ 55.9% → 88.0%, ScaLA-sv MCC 0.146 → 0.466) and relevance (SweFAQ 60.3% → 88.6%), and judges translations better (FLORES+ 48.5% → 71.9%), with in-domain accuracy, calibration and latency about the same as v3's. Getting there exposed a trap worth knowing: human-vs-MT pairs alone teach a gate to reward whatever is least like the source; source-dependent negatives fixed it. Its `grammatical` question is useful for comparing two texts but over-flags good Swedish as a stand-alone verdict. In a 22-model pilot, the gate's quality score of a model's Swedish answers tracks the EuroEval Swedish rank (Spearman 0.66, 0.63 after controlling for model size), which makes it a plausible cheap proxy for screening Swedish models, to be validated on more and larger models.

**Overall.** Across v1 to v4.3 one lesson repeats: the gate learns exactly the contrast its pairs encode, including unintended ones (translationese in v1, "least like the source wins" from Europarl, length from UltraFeedback-style data). Each was found only with a held-out test built to expose it, so sanity tests and model-free baselines (length, n-gram) are part of every result here. As a fast System 1 check of Swedish *form* the released v4.1 is solid: grammar (DaLAJ 88.0%, ScaLA-sv MCC 0.466), translation adequacy, translationese and relevance, at about 10 ms per pair. As a judge of which of two fluent Swedish answers people prefer it is not: on human Chatbot Arena votes it agrees 58.9%, near "the longer answer wins" (64.3%), while Kimi-k3 reaches 74.8%. Synthetic length-controlled data fixes the length bias in-distribution but not on human votes, so the remaining gap needs human preference data on Swedish answers (or a larger model with world knowledge). Used that way, as a form filter with uncertain or content-level decisions sent to an LLM judge, it is useful; its model-level `quality` score also tracks EuroEval Swedish rank (Spearman 0.63 after controlling for size), a cheap screening signal.

## Usage ideas

- **Best-of-N reranking:** generate N Swedish candidates from an LLM, run a pairwise knockout
  or round-robin with `predict_preference_batch` (score both orders and average), and keep the
  winner; N=8 round-robin is 56 ordered pairs, one batched call. With v3, `rank_by_quality`
  scores each candidate once (N forward passes) instead.
- **LLM-as-judge cascade:** score every pair with the gate and only send pairs with
  `is_valid_margin == False` to an expensive judge such as Kimi-k3. In our measurement the recommended threshold 0.00 routes 0.0% at 98.6% accuracy, and the most accurate setting (0.45) routes 5.2% for 99.1% at 507.5 ms expected per pair (see the cascade table).
- **DPO data curation:** score a Swedish preference dataset in both orders and flag pairs where
  the gate confidently prefers the rejected response; if an LLM judge agrees, drop or relabel the
  pair (the label-noise check above flagged 1 such pair(s) in the judge sample).
- **Checkpoint A/B evaluation:** answer a fixed Swedish prompt set with two model versions and
  report the order-averaged win rate of the new one, to catch regressions between releases.
- **Active learning:** send the lowest-margin pairs to human annotators first; they are the ones
  where a label changes the model most.
- **Production monitoring:** compare live responses against a reference answer per prompt and
  track the daily share where the reference is confidently preferred, to detect quality drift.

## Citation and acknowledgements

- **Laya:** decision model, `choice` primitive and training recipe:
  [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya), checkpoint
  [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) (`multilingual`).
- **mmBERT:** multilingual ModernBERT encoder
  [`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base).
- **Dataset:** [`jonasaise/dpo-swedish-interpreted`](https://huggingface.co/datasets/jonasaise/dpo-swedish-interpreted),
  derived from `argilla/ultrafeedback-binarized-preferences-cleaned`.
- **Judge:** Kimi-k3 (Moonshot AI), served on a self-hosted endpoint (8× NVIDIA B300, HGX).

```bibtex
@software{laya_swedish_preference_gate,
  title  = {Laya Swedish Preference Gate},
  author = {Lind, Jonas},
  year   = {2026},
  url    = {https://github.com/joenaess/laya-swedish-preference-gate}
}
```

## License

Code and fine-tuned weights: **Apache-2.0**. The base model `convaiinnovations/laya` is
licensed apache-2.0; the dataset `jonasaise/dpo-swedish-interpreted` is
licensed mit (licenses read from the Hub on
this run). Both permit redistributing a fine-tuned derivative with attribution.
The dataset is derived from UltraFeedback; check the upstream terms before commercial use.
