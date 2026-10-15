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

A **v2** checkpoint (`python -m laya_gate.v2`, see [v2](#v2-training-on-fluent-pairs-too)) adds 9921 fluent-vs-fluent training pairs and a separate naturalness question. On the probe it reaches 75.9% (95% CI 70.9%–80.6%) vs 64.2% for v1, with ECE 0.033 instead of 0.230, so its margin routes mistakes to the judge again: with about 32% of probe instances routed to Kimi-k3, the cascade reaches 84.8% with v2 (33.1% routed) vs 73.7% with v1. The cost is 98.7% → 97.6% on the original held-out pairs (both orders averaged), and the probe is still 11.4 percentage points behind Kimi-k3. v2 is the default checkpoint on the Hub; v1 stays available as `revision="v1"`.

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

The Hub checkpoint is **v2**, which also answers a second question, *which response is
written in more natural, idiomatic Swedish*:

```python
r = gate.predict_naturalness(
    prompt="Vad ska jag göra i Stockholm en regnig dag?",
    candidate_a="Gå på Fotografiska eller Vasamuseet, och fika på något mysigt kafé.",
    candidate_b="Du kan besöka ett museum, vänligen notera att det är en bra aktivitet för regn.",
)
# {"more_natural": "candidate_a", "p_a_more_natural": <float>, "probabilities": {...}}

gate_v1 = SwedishPreferenceGate("jonasaise/laya-swedish-preference-gate", revision="v1")  # the first release
```

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
  fluent pairs → v2 train / calibrate / evaluate → tests → readme → publish.

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

### Tests

`pytest -v --cov=src/laya_gate (all markers; skips listed)`: **108 passed, 0 failed,
0 errors, 0 skipped** of 108.
Coverage: **62.3%** overall — `__init__.py` 100.0% · `benchmark.py` 63.2% · `calibrate.py` 63.8% · `data.py` 70.2% · `fluent.py` 20.6% · `judge.py` 79.3% · `predictor.py` 99.2% · `probe.py` 46.0% · `publish.py` 66.2% · `readme.py` 95.1% · `testing.py` 0.0% · `train.py` 37.3% · `utils.py` 100.0% · `v2.py` 30.1%.


## Limitations

- **The labels encode translation quality, by construction.** Per the dataset card, kimi-k3 localized each prompt and chosen response into idiomatic Swedish and *deliberately* wrote the rejected response as poor translation (literal phrasing, English loan words, US-centric content). A TF-IDF logistic regression reaches 98.4% on the same held-out pairs, so most of the gate's accuracy is detecting translationese. Measured directly: on 299 pairs where *both* answers are natural Swedish and only quality differs, the gate is right 64.2% of the time (95% CI 58.9%–69.6%) versus 87.3% for Kimi-k3, and its calibration breaks down (ECE 0.230). Do not use it as a general helpfulness or correctness judge for two *fluent* Swedish answers.
- **The judge generated the data.** Kimi-k3 wrote both sides of every pair, so its agreement with the labels measures self-consistency, not independent quality judgement.
- **Pairwise preference, not fact-checking.** The gate predicts which of two responses the dataset labels as chosen. It does not verify factual correctness, and a fluent wrong answer can beat a terse correct one.
- **1024-token window.** 42.3% of pairs exceed it; the gate then sees only the end of the prompt and the start of each response, so differences late in long answers are invisible to it.
- **Kimi-k3 is a reference, not ground truth.** The judge is a strong but imperfect model; agreement with it is not correctness.
- **Single seed, single run.** Numbers come from one training run; seed-to-seed variance was not measured.

## Conclusion

The fine-tuned gate reaches 98.4% accuracy on held-out pairs (the zero-shot base checkpoint: 51.6%), with calibrated probabilities (ECE 0.006, meeting the ≤ 0.085 target) and a median single-pair GPU forward pass of 9.9 ms (meeting the 35 ms target). It picks the same winner in both presentation orders for 98.4% of pairs. When it is confident (margin ≥ 0.2, 99.4% of predictions) it is right 98.6% of the time, so its confidence is usable for routing. That accuracy should be read next to the lexical baseline: a TF-IDF logistic regression scores 98.4% on the same pairs, because the dataset contrasts idiomatic Swedish with deliberately poor translation. The gate does not measurably beat that baseline (+0.0 points); what it adds is calibrated confidence, order-robustness and a single model that also reads the prompt. The gate is a strong translationese detector. It is a weak quality judge: on 299 pairs where *both* answers are natural Swedish and only quality differs, the gate is right 64.2% of the time (95% CI 58.9%–69.6%) versus 87.3% for Kimi-k3, and its calibration breaks down (ECE 0.230).

Against these labels Kimi-k3 is *less* accurate than the gate (95.5% vs 98.6%) and 802× slower, which is expected: the labels mark translationese, which the gate was trained on, while the judge was also asked about helpfulness and correctness, and it generated these pairs itself. The judge still helps where the gate is unsure: routing the 5.2% least-confident instances (threshold 0.45) lifts accuracy from 98.6% to 99.1% at an expected 507.5 ms per pair. The pre-registered rule (cheapest threshold within 0.5 points of the best) picks 0.00, i.e. 0.0% routed, because the gain is that small. For its intended System 1 role on data like this, settling confident comparisons in milliseconds and escalating the rest, the gate is fit for purpose (latency target met, ECE target met, confident-subset accuracy 98.6% vs judge 95.5%). It should not be the only judge for high-stakes decisions.

A **v2** checkpoint (`python -m laya_gate.v2`, see [v2](#v2-training-on-fluent-pairs-too)) adds 9921 fluent-vs-fluent training pairs and a separate naturalness question. On the probe it reaches 75.9% (95% CI 70.9%–80.6%) vs 64.2% for v1, with ECE 0.033 instead of 0.230, so its margin routes mistakes to the judge again: with about 32% of probe instances routed to Kimi-k3, the cascade reaches 84.8% with v2 (33.1% routed) vs 73.7% with v1. The cost is 98.7% → 97.6% on the original held-out pairs (both orders averaged), and the probe is still 11.4 percentage points behind Kimi-k3. v2 is the default checkpoint on the Hub; v1 stays available as `revision="v1"`.

## Usage ideas

- **Best-of-N reranking:** generate N Swedish candidates from an LLM, run a pairwise knockout
  or round-robin with `predict_preference_batch` (score both orders and average), and keep the
  winner; N=8 round-robin is 56 ordered pairs, one batched call.
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
