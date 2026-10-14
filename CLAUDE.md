# CLAUDE.md — Laya Swedish Preference Gate

## Project Overview

Build, test, fine-tune, document, and publish a Swedish preference discriminator using `laya-multilingual`. The model acts as a fast "System 1" preference evaluator trained on `jonasaise/dpo-swedish-interpreted`. A self-hosted Kimi-k3 endpoint acts as the "System 2" LLM judge for comparison and cascade evaluation.

**Target environment:** Pop!_OS, NVIDIA RTX 4080 Laptop GPU (12GB VRAM)

**Key goals:**
- Sub-35 ms single forward-pass latency on GPU (batch size 1)
- Calibrated output with ECE ≤ 0.085
- Gate vs. Kimi-k3 judge comparison and a System 1 → System 2 cascade evaluation
- All results documented in `README.md` with a conclusion and usage ideas
- Code published to GitHub as `joenaess/laya-swedish-preference-gate`
- Model weights published to Hugging Face Hub (conditional, see "Hugging Face Upload")

---

## Secrets and Environment Variables

A `.env` file already exists in the project root. It contains:

| Variable   | Purpose                                                          |
|------------|------------------------------------------------------------------|
| `HF_TOKEN` | Hugging Face write token (used by `huggingface_hub` for upload)  |
| `API_KEY`  | API key for the self-hosted Kimi-k3 endpoint                     |
| `BASE_URL` | Base URL of the self-hosted Kimi-k3 endpoint (OpenAI-compatible) |

**Rules — these override everything else in this file:**

1. **Never open, read, `cat`, `grep`, or print `.env`.** Refer to variables by name only.
2. **Never log, print, or echo secret values**, including in error messages, tracebacks, debug output, or test output. Mask them if a library logs request headers.
3. **Never write `BASE_URL`'s value** into the README, model card, `results/`, code defaults, or commit messages. The endpoint is internal infrastructure. Describe it only as "a self-hosted Kimi-k3 endpoint (8× NVIDIA B300, HGX)".
4. **Load variables only via `uv run --env-file .env ...`.** Code reads them from `os.environ`. Never hardcode them and never add a fallback default value.
5. **`.env` must never be committed.** Before *any* `git add`, run `git check-ignore -v .env` and stop if it does not report `.env` as ignored.
6. If a required variable is missing, skip the dependent step, report which variable name was missing, and continue with the rest of the pipeline.

---

## Ground Rules

- **Never invent numbers.** Every metric in the README, model card, and final report must come from an actual run and be traceable to `results/metrics.json`. If a step fails or a metric is missing, write "not measured" and explain why.
- If a target (latency, ECE) is not met, report it honestly in the README. Do not tune the measurement to hit the target.

---

## Before Writing Code

Do these first and stop to report if any don't match the spec:

0. **Protect `.env` immediately.** Create or update `.gitignore` (contents below) before running `uv init` or any git command. Confirm with `git check-ignore -v .env` once git is initialized.
1. The `laya` package (≥0.3.3) installs from PyPI and exposes the `choice` primitive.
2. `convaiinnovations/laya` with `subfolder="multilingual"` loads via `transformers`.
3. Inspect the schema of `jonasaise/dpo-swedish-interpreted` (column names, splits, whether prompts/responses are plain strings or chat-message lists). Adapt `data.py` to the actual schema.
4. Read the reference recipe at https://github.com/NandhaKishorM/laya before implementing `train.py`.
5. Record the licenses of the base model (`convaiinnovations/laya`) and the dataset (`jonasaise/dpo-swedish-interpreted`). These determine the license of the fine-tuned model and whether it may be redistributed.
6. **Verify the Kimi-k3 endpoint** with a small Python script run via `uv run --env-file .env`:
   - Call `GET {BASE_URL}/models` with the OpenAI client to discover the served model id. Determine whether `BASE_URL` already ends in `/v1`; do not double it.
   - Send one tiny chat completion and confirm a response arrives.
   - Check whether the model returns separate reasoning content (e.g. a `reasoning_content` field or `<think>` blocks) and note how to extract the final answer.
   - Use the discovered model id unless `KIMI_MODEL` is set in the environment.
   - Print only the model id, status, and latency — never the URL or key.

---

## Commands

```bash
# Setup (the folder already exists and contains .env)
cd laya-swedish-preference-gate
uv init --lib
# uv init --lib scaffolds src/laya_swedish_preference_gate/ — delete it and
# create src/laya_gate/ instead, then replace pyproject.toml with the one below.
rm -rf src/laya_swedish_preference_gate
mkdir -p src/laya_gate tests results
uv sync --extra dev
git check-ignore -v .env   # must print a match

# Lint & format
uv run ruff check . --fix
uv run ruff format .

# Tests (fast, offline, no GPU, no secrets required)
uv run pytest -v --cov=src/laya_gate -m "not gpu and not network and not judge"

# Full test suite (CUDA + Hub access + trained checkpoint + Kimi endpoint)
uv run --env-file .env pytest -v --cov=src/laya_gate --cov-report=term-missing

# End-to-end pipeline
chmod +x run.sh
./run.sh
```

`run.sh` must use `set -euo pipefail` and invoke every Python step via `uv run --env-file .env python -m laya_gate.<module>`. Steps in order: data → train → calibrate → benchmark → judge → tests → readme → publish. The judge and publish steps exit cleanly (code 0, with a message) if their required variables are missing.

---

## `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "laya-swedish-preference-gate"
version = "0.1.0"
description = "Sub-35ms Swedish DPO preference discriminator using Laya"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "laya>=0.3.3",
    "torch>=2.2.0",
    "transformers>=4.40.0",
    "datasets>=2.18.0",
    "huggingface_hub>=0.23.0",
    "openai>=1.30.0",
    "scikit-learn>=1.4.0",
    "numpy>=1.26.0",
    "pydantic>=2.6.0",
    "rich>=13.7.0",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.3.0",
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.23.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/laya_gate"]

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "gpu: requires a CUDA device",
    "network: requires Hugging Face Hub access",
    "judge: requires the live Kimi-k3 endpoint (API_KEY, BASE_URL)",
]
```

---

## Directory Structure

```text
laya-swedish-preference-gate/
├── .env                    # Secrets — pre-existing, never read, never committed
├── .gitignore
├── pyproject.toml
├── README.md               # Project docs + all results (see "README Requirements")
├── results/
│   ├── metrics.json        # Single source of truth for all reported numbers
│   ├── ece_bins.json       # Per-bin calibration data (pre and post)
│   └── judge_cache.jsonl   # Cached Kimi-k3 verdicts (gitignored)
├── src/
│   └── laya_gate/
│       ├── __init__.py
│       ├── data.py         # Dataset parser, truncation & pairing
│       ├── train.py        # Fine-tuning loop on mmBERT-base backbone
│       ├── calibrate.py    # Post-hoc temperature scaling (ECE optimization)
│       ├── benchmark.py    # Latency + positional-consistency evaluation
│       ├── judge.py        # Kimi-k3 System 2 judge + cascade simulation
│       ├── predictor.py    # Sub-35ms inference engine
│       ├── readme.py       # Renders README.md from results/metrics.json
│       └── publish.py      # Hugging Face Hub upload + model card
├── tests/
│   ├── conftest.py         # Shared fixtures (tiny synthetic dataset, skip logic)
│   ├── test_data.py        # Data pipeline & context truncation unit tests
│   ├── test_calibrate.py   # Optimization & ECE unit tests
│   ├── test_judge.py       # Judge parsing, caching, cascade math (mocked client)
│   └── test_predictor.py   # API & forward pass integration tests
└── run.sh                  # End-to-end execution script
```

`.gitignore` must include at least:

```text
# Secrets
.env
.env.*

# Python / tooling
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.coverage
htmlcov/

# Model artifacts
checkpoints/
*.safetensors
*.bin
*.pt

# Judge cache (contains full dataset text and raw model outputs)
results/judge_cache.jsonl
```

Never commit model weights (GitHub rejects files over 100MB). `results/metrics.json` and `results/ece_bins.json` are small and **are** committed.

---

## Implementation Specs

### 1. Data Pipeline — `src/laya_gate/data.py`

- Load `jonasaise/dpo-swedish-interpreted` via Hugging Face `datasets`.
- Extract Swedish DPO pairs: `prompt`, `chosen`, `rejected` (map from the actual column names found during verification).
- Format samples for Laya's `choice` primitive:
  - **State:** `{"prompt": prompt, "candidate_a": resp_a, "candidate_b": resp_b}`
  - **Question primitive:** `choice` with options:
    `{"candidate_a": "response A is superior", "candidate_b": "response B is superior"}`
- **Symmetrize:** randomly swap order (50% A = chosen, 50% B = chosen) to eliminate positional bias. Use a fixed seed for reproducibility.
- **Token budget:** Laya checkpoints use a 1024-token context. If combined prompt + responses exceed 1024 tokens, truncate at the token level using the model's tokenizer (never by slicing bytes or raw strings). Keep the tail of the prompt (most recent context) and the head of each response. Allocate the budget so neither response is truncated to nothing.
- **Split:** 85% train / 15% validation, stratified by label, fixed seed.
- Record dataset stats to `results/metrics.json`: total pairs, train/val sizes, label balance, fraction of samples truncated, token-length distribution (median, p95, max).

### 2. Fine-Tuning — `src/laya_gate/train.py`

- **Base checkpoint:** `convaiinnovations/laya` with `subfolder="multilingual"` (mmBERT-base backbone, 256k vocab). Chosen to handle Swedish characters (å, ä, ö) without subword fragmentation.
- **Reference recipe:** adapt from https://github.com/NandhaKishorM/laya
- **Hyperparameters (RTX 4080 Laptop, 12GB VRAM):**
  - Precision: `bfloat16` (supported natively on Ada Lovelace; prefer over fp16)
  - Batch size: 8 per device, gradient accumulation = 2 (effective batch 16)
  - If OOM at 1024 tokens: enable gradient checkpointing first; if still OOM, use batch 4 × accumulation 4 (keep effective batch 16)
  - Learning rate: 2e-5, linear warmup (10% of steps), cosine decay
  - Save best checkpoint by validation loss; also log validation accuracy
- Record to `results/metrics.json`: final hyperparameters actually used (including any OOM fallbacks), per-epoch train/val loss and val accuracy, best epoch, total and per-epoch wall-clock time, peak GPU memory.

### 3. Calibration — `src/laya_gate/calibrate.py`

- Raw logits are miscalibrated (reference: base weights show ECE ≈ 0.466). Measure and log ECE on the validation set *before* calibration for the fine-tuned model.
- Fit a single scalar temperature `T > 0` via temperature scaling: minimize NLL with L-BFGS on validation logits. Parameterize as `T = exp(log_T)` to enforce positivity.
- Verify ECE drops to **≤ 0.085**. Use 15 equal-width bins.
- Save `T` as `temperature.json` in the checkpoint directory alongside the model weights.
- Record to `results/metrics.json`: pre/post ECE, pre/post NLL, fitted `T`. Write per-bin data (confidence, accuracy, count) to `results/ece_bins.json`.
- Caveat to document: if `T` is fitted and evaluated on the same validation set, post-calibration ECE is optimistic. If the dataset is large enough, split validation into a calibration half and an evaluation half and report ECE on the held-out half.

### 4. Inference — `src/laya_gate/predictor.py`

Expose a `SwedishPreferenceGate` class:

```python
from laya_gate.predictor import SwedishPreferenceGate

gate = SwedishPreferenceGate(checkpoint_path="checkpoints/best_model")
result = gate.predict_preference(
    prompt="Hur fungerar fotosyntesen?",
    chosen_candidate="Fotosyntesen är den process där...",
    rejected_candidate="Det är när solen lyser på träd och gör magi.",
)
# Returns: {"preference": "candidate_a", "p_chosen": 0.942, "is_valid_margin": True}
```

- `checkpoint_path` accepts either a local directory or a Hugging Face Hub repo id.
- `chosen_candidate` is presented as `candidate_a`, `rejected_candidate` as `candidate_b`. The argument names describe the caller's hypothesis, not ground truth.
- `p_chosen` is the temperature-calibrated probability that `candidate_a` is preferred.
- `is_valid_margin` is True when the prediction is confident. The threshold must be a constructor argument with a documented default (document it in the README).
- Load `temperature.json` from the checkpoint directory automatically.
- Also expose a batched method (`predict_preference_batch`).

### 5. Benchmarking — `src/laya_gate/benchmark.py`

- **Latency:** must be < 35 ms per single forward pass on GPU at batch size 1. Run ≥10 warm-up passes, call `torch.cuda.synchronize()` before and after timing, and report median and p95 over ≥100 runs. Report tokenization time separately from model forward time. Also report CPU latency and GPU throughput at batch sizes 1, 8, 32.
- **Positional consistency:** on the validation set, score each pair in both orders (A/B and B/A). Report the fraction of pairs where the prediction flips consistently, and the accuracy when `chosen` is in position A vs position B. A large gap means residual positional bias.
- **Confidence vs accuracy:** report validation accuracy on the subset where `is_valid_margin` is True, and the fraction of samples that pass the margin.
- Write all results to `results/metrics.json`.

### 6. System 2 Judge — `src/laya_gate/judge.py`

Uses the self-hosted Kimi-k3 endpoint as a strong LLM judge on a held-out sample, to compare against the gate and simulate a cascade.

**Client:**
- `openai.AsyncOpenAI(base_url=os.environ["BASE_URL"], api_key=os.environ["API_KEY"])`. If either variable is missing, skip the judge step and record `"judge": "skipped: <VARIABLE_NAME> not set"` in `metrics.json`.
- Model id: `KIMI_MODEL` env var if set, otherwise the id discovered via `/models`.
- Concurrency via `asyncio.Semaphore` (default 16, configurable via `--concurrency`). Per-request timeout, exponential-backoff retries (max 3) on timeouts, 429, and 5xx.
- Set `temperature=0`. If the model emits reasoning, allow enough `max_tokens` for it to finish and extract only the final answer.
- Catch and re-raise client errors with a sanitized message so the URL and key never appear in logs or tracebacks.

**Sample:**
- Fixed-seed sample of N validation pairs (default 500, configurable via `--n`), taken from the evaluation half if validation was split for calibration.
- Query every pair in **both orders** (A/B and B/A), i.e. 2N requests.

**Prompt:**
- System prompt in English instructing the model to judge which Swedish response better answers the Swedish prompt (helpfulness, correctness, fluency in Swedish), and to reply with JSON only: `{"preference": "A"}` or `{"preference": "B"}`.
- Parse robustly (strip code fences and reasoning blocks). Record parse failures as a metric; never silently coerce them to a label.

**Cache:**
- Append every verdict to `results/judge_cache.jsonl`, keyed by a hash of (model id, prompt, candidate order). On rerun, reuse cached verdicts and only query missing ones. Store per-request latency and token usage.

**Metrics to record in `results/metrics.json`:**
- Judge accuracy vs dataset labels (both orders, and order-averaged)
- Judge positional consistency (fraction of pairs with the same winner in both orders)
- Gate accuracy on the exact same sample
- Gate–judge agreement rate, and accuracy on pairs where they disagree
- Judge latency per request (median, p95), throughput (requests/s at the configured concurrency), total tokens, parse failure rate
- **Cascade simulation:** sweep the gate's margin threshold across a range (e.g. 0.0 to 0.45 in steps of 0.05). For each threshold, compute the fraction routed to the judge (`is_valid_margin == False`), combined cascade accuracy (gate answer when confident, judge answer otherwise), and expected mean latency per pair (gate latency + routed fraction × judge latency). Identify the threshold that best trades accuracy against the routed fraction and note it in the README.
- **Label-noise check:** count pairs where both the gate and the judge confidently disagree with the dataset label. Save their ids (not their text) to `metrics.json`, since they are candidates for mislabeled data.

**Tests (`tests/test_judge.py`):** all offline with a mocked client. Cover JSON parsing (clean, fenced, with reasoning blocks, malformed), cache hit/miss behavior, retry logic, cascade math on synthetic inputs, and that no secret value appears in raised exceptions. Mark any live-endpoint test with `@pytest.mark.judge`.

---

## Testing Requirements

Use `pytest`. All tests must pass before committing. Unit tests must run offline on CPU, without `.env`, using small synthetic fixtures. Mark tests that need the real dataset with `@pytest.mark.network`, tests that need CUDA with `@pytest.mark.gpu`, and tests that need the live Kimi endpoint with `@pytest.mark.judge` (all auto-skip when unavailable).

**`tests/test_data.py`**
- Label balance is ~50/50 in both train and val splits after randomization (assert within a tolerance, e.g. ±2%, not exact equality).
- Inputs > 1024 tokens are truncated to ≤ 1024 tokens.
- Truncated text containing å, ä, ö decodes to valid UTF-8 with no replacement characters (`\ufffd`).
- Neither response is empty after truncation.

**`tests/test_calibrate.py`**
- Temperature fitting on synthetic overconfident logits recovers `T > 1`.
- ECE decreases after calibration.
- Fitted `T` is always positive.

**`tests/test_judge.py`**
- See section 6.

**`tests/test_predictor.py`**
- Single and batched prediction work on CPU, and on GPU when available.
- The two option probabilities (`candidate_a`, `candidate_b`) sum to 1.0 (within 1e-6).
- `p_chosen` is in [0, 1] and the returned dict matches the documented schema (keys and types).
- Batched and single predictions agree for the same inputs.
- Skip cleanly if no trained checkpoint exists.

Record test count, pass/fail/skip counts, and coverage (overall and per module) to `results/metrics.json`.

---

## README Requirements

`src/laya_gate/readme.py` renders `README.md` after all pipeline steps have completed. All numbers must be read from `results/metrics.json`. The endpoint URL must never appear. Sections, in order:

1. **Title and one-paragraph summary:** what the model does, in plain language.
2. **Results at a glance:** a table with validation accuracy, pre/post-calibration ECE, fitted temperature, GPU latency (median/p95), gate vs Kimi-k3 accuracy on the judge sample, and whether each target was met (✅/❌).
3. **Installation:** `uv`-based setup.
4. **Quickstart:** the `SwedishPreferenceGate` example, loading from the Hugging Face Hub if uploaded, otherwise from a local checkpoint. Include a batched example.
5. **How it works:** the `choice` primitive framing, symmetrization, truncation strategy, temperature scaling. Keep it short.
6. **Detailed results:**
   - Dataset statistics (sizes, truncation rate, token-length distribution)
   - Training: hyperparameters actually used, per-epoch loss/accuracy table, best epoch, training time, peak GPU memory, hardware
   - Calibration: pre/post ECE and NLL, a per-bin reliability table from `ece_bins.json`, and the same-set fitting caveat if applicable
   - Latency and throughput table (GPU and CPU, batch sizes 1/8/32), with the measurement method described
   - Positional consistency results
   - Accuracy at the default margin threshold and the fraction of samples passing it
   - **System 1 vs System 2:** gate vs Kimi-k3 accuracy, agreement rate, both models' positional consistency, judge latency and throughput, and a latency ratio (judge median / gate median). Describe the judge hardware as "self-hosted Kimi-k3 on 8× NVIDIA B300 (HGX)".
   - **Cascade table:** threshold → routed fraction → cascade accuracy → expected mean latency, with the recommended threshold highlighted
   - **Label-noise findings:** the number of suspected mislabeled pairs and what that implies about the dataset
   - Test results and coverage
7. **Limitations:** at minimum, that the dataset is interpreted/translated rather than natively written Swedish preferences, that the model judges pairwise preference rather than factual correctness, the 1024-token limit, that Kimi-k3 is itself an imperfect reference rather than ground truth, and any target that was not met.
8. **Conclusion:** 1–2 paragraphs assessing whether the model is fit for its intended "System 1" role, based on the actual numbers, including how much of the judge's accuracy the cascade recovers at what fraction of the cost. Be candid about weaknesses.
9. **Usage ideas:** concrete ways to use the gate, each with a sentence on how. Include at least:
   - **Best-of-N reranking:** generate N Swedish candidates from an LLM and use pairwise comparisons to pick the best
   - **LLM-as-judge cascade:** use the gate as a cheap first pass and send only low-margin pairs to an expensive LLM judge such as Kimi-k3 (cite the measured cascade results)
   - **DPO data curation:** filter or flag noisy pairs in Swedish preference datasets where the gate (and judge) strongly disagree with the label
   - **Checkpoint A/B evaluation:** compare outputs of two Swedish model versions on a fixed prompt set to track regressions
   - **Active learning:** route low-confidence pairs to human annotators to build better training data
   - **Production monitoring:** score live responses against a reference answer to detect quality drift
10. **Citation and acknowledgements:** Laya (link to the reference repo), mmBERT, the dataset, and Kimi-k3 as the judge model.
11. **License:** based on the licenses recorded during verification.

---

## Hugging Face Upload — `src/laya_gate/publish.py`

Upload only if **all** of these are true. Otherwise skip, and state in the README and final report why it was skipped:

- `HF_TOKEN` is set in the environment (loaded from `.env` via `uv run --env-file .env`)
- The base model and dataset licenses permit redistribution of a fine-tuned derivative
- All tests pass and post-calibration ECE ≤ 0.085

Steps:

1. Repo id: `joenaess/laya-swedish-preference-gate` (override via `HF_REPO_ID` env var).
2. Upload the best checkpoint, tokenizer files, `temperature.json`, and `results/metrics.json` using `huggingface_hub.HfApi().upload_folder()`. `huggingface_hub` reads `HF_TOKEN` from the environment automatically; do not pass the token explicitly, on the command line, or write it to disk. Do not run `hf auth login`, which would persist the token to `~/.cache/huggingface`.
3. Do **not** upload `results/judge_cache.jsonl` or anything that contains the endpoint URL.
4. Create the repo as **private** first. Report that it's private; the user makes it public after reviewing.
5. Generate a model card (`README.md` in the HF repo) with YAML metadata:
   ```yaml
   language: [sv]
   license: <from verification>
   base_model: convaiinnovations/laya
   datasets: [jonasaise/dpo-swedish-interpreted]
   pipeline_tag: text-classification
   tags: [preference-model, reward-model, dpo, swedish, laya]
   ```
   The model card body contains the summary, results table (including the System 1 vs System 2 comparison), quickstart, limitations, and a link to the GitHub repo.
6. After a successful upload, add the Hub link and a Hub-based quickstart to the GitHub README.

---

## Git & Deployment

Prerequisite: `gh auth status` must show an authenticated session for `joenaess`.

Order of operations: train → calibrate → benchmark → judge → test → write README → (optional) HF upload → update README with Hub link → secret scan → commit → push.

```bash
# uv init already ran `git init`; this is a no-op if the repo exists
git init

# Secret safety checks — stop if any fail
git check-ignore -v .env                      # must report .env as ignored
git add .
git diff --cached --name-only | grep -E '(^|/)\.env' && exit 1   # .env must not be staged
git diff --cached --name-only | grep -E '\.(safetensors|bin|pt)$' && exit 1   # no weights
# Scan staged content for secrets without printing their values:
uv run --env-file .env python -c "
import os, subprocess, sys
staged = subprocess.run(['git', 'diff', '--cached'], capture_output=True, text=True).stdout
leaks = [k for k in ('HF_TOKEN', 'API_KEY', 'BASE_URL') if os.environ.get(k) and os.environ[k] in staged]
if leaks:
    print('Secret value found in staged changes for:', ', '.join(leaks))
    sys.exit(1)
print('No secret values found in staged changes.')
"

git commit -m "feat: Swedish DPO preference gate using Laya, with results and docs"

gh repo create joenaess/laya-swedish-preference-gate \
    --public \
    --source=. \
    --remote=origin \
    --push
```

**Final report must include:**
- Final commit hash and GitHub URL
- Hugging Face repo URL and visibility, or the reason the upload was skipped
- Test coverage percentage (overall and per module)
- Training wall-clock time (total and per epoch)
- Pre- and post-calibration ECE, fitted temperature
- Inference latency (median and p95, GPU, batch size 1)
- Gate vs Kimi-k3 accuracy, agreement rate, recommended cascade threshold and its routed fraction
- Which targets were met and which were not
- Confirmation that the secret scan passed
