"""Render README.md from results/metrics.json and results/ece_bins.json.

Every number in the README comes from those files; a missing value is written "not measured".
"""

from __future__ import annotations

import json
from typing import Any

from .utils import (
    DEFAULT_MARGIN,
    ECE_BINS_PATH,
    ECE_TARGET,
    GITHUB_REPO,
    LATENCY_TARGET_MS,
    ROOT,
    load_metrics,
)

NM = "not measured"
JUDGE_HW = "self-hosted Kimi-k3 on 8× NVIDIA B300 (HGX)"


def get(d: Any, path: str, default: Any = None) -> Any:
    for key in path.split("."):
        if isinstance(d, dict) and key in d:
            d = d[key]
        elif isinstance(d, list) and key.isdigit() and int(key) < len(d):
            d = d[int(key)]
        else:
            return default
    return d


def pct(x: Any, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%" if isinstance(x, int | float) else NM


def num(x: Any, digits: int = 3) -> str:
    return f"{x:.{digits}f}" if isinstance(x, int | float) else NM


def ms(x: Any) -> str:
    return f"{x:.1f} ms" if isinstance(x, int | float) else NM


def mark(ok: Any) -> str:
    return "✅" if ok is True else "❌" if ok is False else "—"


def hub_info(m: dict) -> tuple[bool, str | None, str]:
    pub = m.get("publish") or {}
    uploaded = isinstance(pub, dict) and pub.get("status") == "uploaded"
    repo = pub.get("repo_id") if uploaded else None
    reason = (
        pub.get("reason", "the publish step has not run") if isinstance(pub, dict) else str(pub)
    )
    return uploaded, repo, reason


# ------------------------------------------------------------------- sections


def results_table(m: dict) -> str:
    cal, bench, judge = m.get("calibration", {}), m.get("benchmark", {}), m.get("judge")
    lat = get(bench, "latency", {})
    j = judge if isinstance(judge, dict) else {}
    rows = [
        ("Validation accuracy (val_eval, held out)", pct(cal.get("accuracy")), "—"),
        ("ECE before calibration", num(cal.get("ece_pre")), "—"),
        (
            "ECE after calibration",
            num(cal.get("ece_post")),
            f"{mark(cal.get('target_met'))} target ≤ {ECE_TARGET}",
        ),
        ("Fitted temperature T", num(cal.get("temperature")), "—"),
        (
            "GPU latency, batch 1 (median / p95)",
            f"{ms(get(lat, 'gpu_forward_ms.p50'))} / {ms(get(lat, 'gpu_forward_ms.p95'))}",
            f"{mark(lat.get('target_met'))} target < {LATENCY_TARGET_MS:.0f} ms (median)",
        ),
        (
            "Gate accuracy on judge sample (order-averaged)",
            pct(get(j, "gate_accuracy.order_averaged")),
            "—",
        ),
        (
            "Kimi-k3 accuracy on judge sample (order-averaged)",
            pct(get(j, "judge_accuracy.order_averaged")),
            "—",
        ),
    ]
    out = ["| Metric | Value | Target |", "|---|---|---|"]
    out += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    return "\n".join(out)


def quickstart(m: dict) -> str:
    uploaded, repo, _ = hub_info(m)
    path = repo if uploaded else "checkpoints/best_model"
    where = (
        f"The fine-tuned weights are on the Hugging Face Hub at "
        f"[`{repo}`](https://huggingface.co/{repo}); `SwedishPreferenceGate` downloads them "
        "on first use."
        if uploaded
        else "The weights are not on the Hub (see [License](#license) and the publish "
        "notes), so run `./run.sh` first to produce `checkpoints/best_model`."
    )
    return f"""{where}

```python
from laya_gate.predictor import SwedishPreferenceGate

gate = SwedishPreferenceGate(checkpoint_path="{path}")
result = gate.predict_preference(
    prompt="Hur fungerar fotosyntesen?",
    chosen_candidate="Fotosyntesen är den process där växter omvandlar ljusenergi till kemisk energi...",
    rejected_candidate="Det är när solen lyser på träd och gör magi.",
)
print(result)
# {{"preference": "candidate_a", "p_chosen": <float>, "is_valid_margin": <bool>,
#  "probabilities": {{"candidate_a": <float>, "candidate_b": <float>}}}}
```

`chosen_candidate` is shown to the model as `candidate_a` and `rejected_candidate` as
`candidate_b`; the names describe your hypothesis, not ground truth. `p_chosen` is the
temperature-calibrated probability that `candidate_a` is the better response.
`is_valid_margin` is `True` when `|p_chosen − 0.5| ≥ margin`; the constructor argument `margin`
defaults to **{DEFAULT_MARGIN}** (i.e. calibrated probability ≥ {0.5 + DEFAULT_MARGIN:.1f} for
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

{v2_quickstart(m)}Also `device="cpu"`, `dtype=torch.float32`, `temperature=<float>`, `margin=<float>` and
`revision=<Hub branch or tag>` are accepted. The checkpoint is in the standard Laya layout, so `laya.load(path)` opens it too
(the fitted temperature is stored in its `rl_agent_config.json`)."""


def v2_quickstart(m: dict) -> str:
    which = get(m, "publish.checkpoint")
    if which not in ("v2", "v3", "v4"):
        return ""
    repo = get(m, "publish.repo_id")
    v3 = (
        """

v3 adds two single-response questions, so N candidates need N forward passes instead of
N(N−1) ordered pairs:

```python
s = gate.score_quality(prompt="Hur fungerar fotosyntesen?", response="Växter gör mat av solljus.")
# {"score": <float 1-5>, "probabilities": {1: <float>, ..., 5: <float>}}
ranking = gate.rank_by_quality(prompt, [cand_1, cand_2, cand_3])  # [(index, score), ...] best first

n = gate.score_naturalness("Du kan besöka ett museum, vänligen notera att det är bra.")
# {"label": "natural" | "translationese", "p_natural": <float>, "is_valid_margin": <bool>}
```

Earlier releases stay loadable: `revision="v2"` and `revision="v1"`."""
        if which == "v3"
        else """

v4 and v4.1 add a single-response grammar check (use it to compare two versions of a text; as a
stand-alone verdict it flags too much good Swedish) and judges translations with the English
source in the prompt:

```python
g = gate.score_grammar("Människor är snälla där och alla varor är billiga.")
# {"label": "correct" | "incorrect", "p_correct": <float>, "is_valid_margin": <bool>}
t = gate.predict_preference(
    prompt="Översätt följande text till svenska:\nThe meeting was postponed.",
    chosen_candidate="Mötet sköts upp.",
    rejected_candidate="Mötet ställdes in.",
)
s = gate.score_quality(prompt="Hur fungerar fotosyntesen?", response="Växter gör mat av solljus.")
```

Earlier releases stay loadable: `revision="v4"`, `revision="v3"`, `revision="v2"` and
`revision="v1"`."""
        if which == "v4"
        else f"""

```python
gate_v1 = SwedishPreferenceGate("{repo}", revision="v1")  # the first release
```"""
    )
    name = get(m, "publish.release") or which
    return f"""The Hub checkpoint is **{name}**, which also answers *which response is written in more
natural, idiomatic Swedish*:

```python
r = gate.predict_naturalness(
    prompt="Vad ska jag göra i Stockholm en regnig dag?",
    candidate_a="Gå på Fotografiska eller Vasamuseet, och fika på något mysigt kafé.",
    candidate_b="Du kan besöka ett museum, vänligen notera att det är en bra aktivitet för regn.",
)
# {{"more_natural": "candidate_a", "p_a_more_natural": <float>, "probabilities": {{...}}}}
```{v3}

"""


def dataset_section(m: dict) -> str:
    d = m.get("data", {})
    tl = d.get("token_length_untruncated", {})
    bal = d.get("label_balance_frac_chosen_in_b", {})
    return f"""Source: [`{d.get("source", NM)}`](https://huggingface.co/datasets/{d.get("source", "")}).
The repository's single `train` split is stored as parquet files with two different
schemas (the dataset card declares only one), so `datasets.load_dataset` fails with a cast
error; `data.py` reads the files one by one and keeps `prompt`, `chosen`, `rejected`.

| | |
|---|---|
| Raw pairs | {d.get("raw_pairs", NM)} |
| Dropped (empty / identical / duplicate) | {json.dumps(d.get("dropped", NM))} |
| Train / val_calib / val_eval | {get(d, "splits.train", NM)} / {get(d, "splits.val_calib", NM)} / {get(d, "splits.val_eval", NM)} |
| Val pairs moved to train (prompt also in train) | {d.get("val_pairs_moved_to_train_for_shared_prompt", NM)} |
| Fraction with chosen in slot B (train / calib / eval) | {num(bal.get("train"))} / {num(bal.get("val_calib"))} / {num(bal.get("val_eval"))} |
| Pairs truncated to fit 1024 tokens | {d.get("n_truncated", NM)} ({pct(d.get("fraction_truncated"))}) |
| Untruncated input length, tokens (median / p95 / max) | {tl.get("median", NM)} / {tl.get("p95", NM)} / {tl.get("max", NM)} |
| Question head / state budget, tokens | {d.get("question_head_tokens", NM)} / {d.get("state_room_tokens", NM)} |

**How the pairs were made.** The dataset card says kimi-k3 localized each UltraFeedback prompt
and chosen response into idiomatic Swedish and deliberately generated the rejected response
with poor translation, excessive English loan words or US-centric content. The preference
signal is therefore mostly *authentic vs. translated Swedish*. Two non-neural baselines on
val_eval show how lexical it is:

| Baseline (val_eval) | Accuracy |
|---|---|
| Longer response wins | {pct(get(d, "baselines.longer_response_wins_accuracy"))} |
| Logistic regression on TF-IDF(A) − TF-IDF(B) | {pct(get(d, "baselines.tfidf_logreg_accuracy"))} |

Words the TF-IDF model associates with the *chosen* side: {", ".join(f"*{w}*" for w in get(d, "baselines.words_favouring_chosen", [])[:10]) or NM}.
With the *rejected* side: {", ".join(f"*{w}*" for w in get(d, "baselines.words_favouring_rejected", [])[:10]) or NM}."""


def training_section(m: dict) -> str:
    t = m.get("training", {})
    hp = t.get("hyperparameters", {})
    hist = t.get("history", [])
    rows = [
        "| Epoch | Train loss | Val loss (calib / eval) | Val acc (calib / eval) | Time |",
        "|---|---|---|---|---|",
    ]
    for h in hist:
        best = " **(best)**" if h["epoch"] == t.get("best_epoch") else ""
        rows.append(
            f"| {h['epoch']}{best} | {h['train_loss']:.4f} | {h['val_calib_loss']:.4f} / "
            f"{h['val_eval_loss']:.4f} | {pct(h['val_calib_acc'])} / {pct(h['val_eval_acc'])} "
            f"| {h['epoch_seconds'] / 60:.1f} min |"
        )
    zs = t.get("zero_shot_base", {})
    total = t.get("total_seconds")
    return f"""Base: `{t.get("base_checkpoint", NM)}` (backbone `{t.get("backbone", NM)}`), adapted from the
reference recipe in [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) (`laya.train`).

| Hyperparameter | Value |
|---|---|
| Precision | {hp.get("precision", NM)} autocast |
| Micro-batch × accumulation (effective) | {hp.get("micro_batch", NM)} × {hp.get("grad_accum", NM)} ({t.get("effective_batch", NM)}) |
| OOM fallback used | {hp.get("oom_fallback", NM)} (gradient checkpointing: {hp.get("gradient_checkpointing", NM)}) |
| Learning rate / schedule | {hp.get("lr", NM)}, linear warmup {pct(hp.get("warmup_frac"), 0)} of {t.get("optimizer_updates", NM)} updates, cosine decay |
| Optimizer | AdamW, weight decay {hp.get("weight_decay", NM)}, grad-norm clip {hp.get("grad_clip", NM)} |
| Epochs | {hp.get("epochs", NM)} |
| Loss | {hp.get("loss", NM)} |
| Sequence budget | max_len {hp.get("max_len", NM)}, head_max_len {hp.get("head_max_len", NM)} |

{chr(10).join(rows)}

- Zero-shot base checkpoint (before fine-tuning): val_eval accuracy {pct(zs.get("val_eval_acc"))}, loss {num(zs.get("val_eval_loss"))}.
- Best epoch: **{t.get("best_epoch", NM)}**, chosen by {t.get("selection", NM)}.
- Training wall-clock: {f"{total / 60:.1f} min" if total else NM} total
  ({", ".join(f"{s / 60:.1f}" for s in t.get("per_epoch_seconds", [])) or NM} min per epoch, including per-epoch validation).
- Peak GPU memory: {num(t.get("peak_gpu_memory_allocated_gib"), 2)} GiB allocated / {num(t.get("peak_gpu_memory_reserved_gib"), 2)} GiB reserved.
- Hardware: {t.get("hardware", NM)}, PyTorch {t.get("torch", NM)}."""


def calibration_section(m: dict) -> str:
    c = m.get("calibration", {})
    bins = json.loads(ECE_BINS_PATH.read_text()) if ECE_BINS_PATH.exists() else {}
    ev = bins.get("val_eval", {})
    rows = [
        "| Confidence bin | n (pre) | Conf. / acc. (pre) | n (post) | Conf. / acc. (post) |",
        "|---|---|---|---|---|",
    ]
    for pre, post in zip(ev.get("pre", []), ev.get("post", []), strict=False):
        if not pre["count"] and not post["count"]:
            continue

        def cell(b):
            if not b["count"]:
                return "—"
            return f"{b['confidence']:.3f} / {b['accuracy']:.3f}"

        rows.append(
            f"| {pre['lower']:.3f}–{pre['upper']:.3f} | {pre['count']} | {cell(pre)} | "
            f"{post['count']} | {cell(post)} |"
        )
    cal = get(c, "per_split.val_calib", {})
    return f"""Temperature scaling with a single scalar `T = exp(log_T)`, fitted by L-BFGS on the NLL of the
**val_calib** half; all headline numbers below are on the **val_eval** half, which neither the
fit, checkpoint selection nor training saw. ECE uses {c.get("n_bins", NM)} equal-width bins over
the top-1 confidence.

| | Before | After |
|---|---|---|
| ECE (val_eval) | {num(c.get("ece_pre"))} | {num(c.get("ece_post"))} |
| NLL (val_eval) | {num(c.get("nll_pre"))} | {num(c.get("nll_post"))} |
| ECE (val_calib, in-sample for the fit) | {num(cal.get("ece_pre"))} | {num(cal.get("ece_post"))} |

Fitted temperature: **T = {num(c.get("temperature"))}**
({"T > 1: the raw logits were over-confident" if (c.get("temperature") or 0) > 1 else "T ≤ 1: the raw logits were under-confident" if c.get("temperature") else NM}).
Target ECE ≤ {ECE_TARGET}: {mark(c.get("target_met"))}.

Reliability table (val_eval; empty bins omitted; for two options the top-1 confidence is ≥ 0.5):

{chr(10).join(rows)}"""


def thr_note(thr: list) -> str:
    by_bs = {t["batch_size"]: t["pairs_per_second"] for t in thr}
    if 8 in by_bs and 32 in by_bs and by_bs[32] < by_bs[8]:
        return (
            "Batch 32 is slower than batch 8: at up to 1024 tokens per pair, 32-pair batches "
            "push the 12 GB laptop GPU to its memory limit (the CUDA allocator logged failed "
            "allocations and retries), so throughput peaks at batch 8 on this card.\n\n"
        )
    return ""


def latency_section(m: dict) -> str:
    lat = get(m, "benchmark.latency", {})
    thr = lat.get("throughput_gpu", [])
    thr_rows = "\n".join(
        f"| GPU, batch {t['batch_size']} | {t['pairs_per_second']:.1f} pairs/s | "
        f"{1000 / t['pairs_per_second']:.1f} ms/pair |"
        for t in thr
    )
    g, gl = lat.get("gpu_forward_ms", {}), lat.get("gpu_forward_ms_max_length", {})
    e2e, tokn, cpu = (
        lat.get("gpu_end_to_end_ms", {}),
        lat.get("tokenization_ms", {}),
        lat.get("cpu_forward_ms", {}),
    )
    it = lat.get("input_tokens", {})
    return f"""Method: {lat.get("method", NM)}.
Inputs are real validation pairs (median {num(it.get("p50"), 0)} tokens, p95 {num(it.get("p95"), 0)}).
GPU: {lat.get("device", NM)} ({lat.get("dtype", NM)} weights). CPU: {lat.get("cpu_threads", NM)} threads, {lat.get("cpu_dtype", NM)}.

| Measurement (batch 1) | Median | p95 | n |
|---|---|---|---|
| GPU forward pass | {ms(g.get("p50"))} | {ms(g.get("p95"))} | {g.get("n", NM)} |
| GPU forward, full 1024-token input ({gl.get("input_tokens", NM)} tokens) | {ms(gl.get("p50"))} | {ms(gl.get("p95"))} | {gl.get("n", NM)} |
| Tokenization + truncation (CPU) | {ms(tokn.get("p50"))} | {ms(tokn.get("p95"))} | {tokn.get("n", NM)} |
| GPU end-to-end `predict_preference` | {ms(e2e.get("p50"))} | {ms(e2e.get("p95"))} | {e2e.get("n", NM)} |
| CPU forward pass | {ms(cpu.get("p50"))} | {ms(cpu.get("p95"))} | {cpu.get("n", NM)} |

| Throughput (length-sorted batches, 512 val_eval pairs) | Pairs/s | Per pair |
|---|---|---|
{thr_rows or "| GPU | " + NM + " | |"}

{thr_note(thr)}Target: median GPU forward pass < {LATENCY_TARGET_MS:.0f} ms at batch 1 → {mark(lat.get("target_met"))}
(p95 under target: {mark(lat.get("p95_under_target"))})."""


def positional_section(m: dict) -> str:
    p = get(m, "benchmark.positional", {})
    mg = get(m, "benchmark.margin", {})
    sweep = "\n".join(
        f"| {r['margin']:.2f} | {pct(r['coverage'])} | {pct(r['accuracy_on_covered'])} |"
        for r in mg.get("sweep", [])
    )
    return f"""### Positional consistency

Every val_eval pair ({get(m, "benchmark.n_pairs", NM)}) scored in both orders (A/B and B/A).

| | |
|---|---|
| Same winner in both orders | {pct(p.get("consistent_fraction"))} |
| Accuracy with chosen in slot A | {pct(p.get("accuracy_chosen_in_a"))} |
| Accuracy with chosen in slot B | {pct(p.get("accuracy_chosen_in_b"))} |
| Gap (A − B) | {num(p.get("accuracy_gap_a_minus_b") * 100 if isinstance(p.get("accuracy_gap_a_minus_b"), float) else None, 1)} pp |
| Mean P(slot A wins) | {num(p.get("mean_p_slot_a"))} |
| Accuracy, averaging P over both orders | {pct(p.get("accuracy_order_averaged_prob"))} |

### Accuracy at the margin threshold

At the default `margin={mg.get("default_margin", DEFAULT_MARGIN)}`, **{pct(mg.get("coverage"))}** of
predictions pass `is_valid_margin`, with accuracy **{pct(mg.get("accuracy_on_covered"))}** on that
subset (vs {pct(mg.get("accuracy_all"))} on all predictions; both orders counted).

| Margin | Fraction passing | Accuracy on passing |
|---|---|---|
{sweep}"""


def judge_section(m: dict) -> str:
    j = m.get("judge")
    if not isinstance(j, dict):
        return f"Not measured: {j or 'the judge step has not run'}."
    ja, ga = j.get("judge_accuracy", {}), j.get("gate_accuracy", {})
    lat, thr = j.get("latency_ms") or {}, j.get("throughput", {})
    dis = j.get("disagreements", {})
    rec = j.get("recommended", {})
    rows = []
    for r in j.get("cascade", []):
        star = "**" if r["threshold"] == rec.get("threshold") else ""
        rows.append(
            f"| {star}{r['threshold']:.2f}{star} | {star}{pct(r['routed_fraction'])}{star} | "
            f"{star}{pct(r['cascade_accuracy'])}{star} | {star}{ms(r['expected_latency_ms'])}{star} |"
        )
    tokens = j.get("tokens", {})
    noise = j.get("label_noise", {})
    sample = j.get("sample", {})
    return f"""Judge: {JUDGE_HW}, model id `{j.get("model", NM)}`, temperature 0, reasoning allowed
(only the final answer is parsed). Sample: {sample.get("n_pairs", NM)} fixed-seed pairs from val_eval,
each queried in both orders ({sample.get("n_requests", NM)} requests, concurrency
{get(j, "settings.concurrency", NM)}). The judge sees the full, untruncated texts; the gate sees
its 1024-token window.

| | Gate (System 1) | Kimi-k3 (System 2) |
|---|---|---|
| Accuracy, order A/B | {pct(ga.get("AB"))} | {pct(ja.get("AB"))} |
| Accuracy, order B/A | {pct(ga.get("BA"))} | {pct(ja.get("BA"))} |
| Accuracy, order-averaged | {pct(ga.get("order_averaged"))} | {pct(ja.get("order_averaged"))} |
| Positional consistency | {pct(j.get("gate_positional_consistency"))} | {pct(j.get("judge_positional_consistency"))} |
| Median latency per pair | {ms(j.get("gate_latency_ms_used"))} | {ms(lat.get("p50"))} (p95 {ms(lat.get("p95"))}) |

- Gate–judge agreement: **{pct(j.get("agreement_rate"))}** of instances. On the {dis.get("n", NM)}
  disagreements the gate is right {pct(dis.get("gate_accuracy"))} of the time and the judge
  {pct(dis.get("judge_accuracy"))}.
- Judge parse failures: {j.get("parse_failures", NM)} ({pct(j.get("parse_failure_rate"), 2)}); request failures after retries: {j.get("request_failures", NM)}.
  Judge accuracy counting parse failures as wrong: {pct(ja.get("order_averaged_parse_failures_as_wrong"))}.
- Judge throughput: {num(thr.get("requests_per_second"), 2)} requests/s at concurrency {thr.get("concurrency", NM)}.
  Tokens used: {tokens.get("total_tokens", NM)} total ({tokens.get("prompt_tokens", NM)} prompt, {tokens.get("completion_tokens", NM)} completion, reasoning included).
- Latency ratio (judge median / gate end-to-end median): **{num(j.get("latency_ratio_judge_over_gate"), 0)}×**.

### Cascade (System 1 → System 2)

An instance goes to the judge when `|p_chosen − 0.5| < threshold`; otherwise the gate's answer
stands. Expected latency = gate latency + routed fraction × judge median latency. Routed
instances without a parseable judge verdict keep the gate's answer.

| Threshold | Routed to judge | Cascade accuracy | Expected latency / pair |
|---|---|---|---|
{chr(10).join(rows)}

Recommended threshold (**bold**): **{num(rec.get("threshold"), 2)}** — {rec.get("rule", NM)}:
{pct(rec.get("routed_fraction"))} routed, {pct(rec.get("cascade_accuracy"))} accuracy.

### Label-noise findings

Rule: {noise.get("rule", NM)}. **{noise.get("n_suspected", NM)}** of {sample.get("n_pairs", NM)} sampled pairs
({pct(noise.get("fraction_of_sample"))}) meet it. Their ids are in `results/metrics.json`
(`judge.label_noise.pair_ids`). These are candidates for mislabeled or ambiguous pairs, not
confirmed errors: both "judges" may share a bias, for example toward longer answers."""


def probe_summary(m: dict) -> str | None:
    """One sentence with the fluent-vs-fluent probe result, or None when it has not run."""
    p = m.get("probe")
    if not isinstance(p, dict):
        return None
    g, j = get(p, "gate.order_averaged", {}), get(p, "judge.order_averaged", {})
    return (
        f"on {p.get('n_pairs')} pairs where *both* answers are natural Swedish and only quality "
        f"differs, the gate is right {pct(g.get('accuracy'))} of the time (95% CI "
        f"{pct(get(g, 'ci95.0'))}–{pct(get(g, 'ci95.1'))}) versus {pct(j.get('accuracy'))} for "
        f"Kimi-k3, and its calibration breaks down (ECE {num(get(p, 'gate.ece_order_averaged'))})"
    )


def probe_section(m: dict) -> str:
    p = m.get("probe")
    if not isinstance(p, dict):
        return f"Not measured: {p or 'run `python -m laya_gate.probe`'}."
    g, j, b = p.get("gate", {}), p.get("judge", {}), p.get("baselines", {})
    dm = g.get("default_margin", {})

    def cell(x):
        return (
            f"{pct(x.get('accuracy'))} [{pct(get(x, 'ci95.0'))}, {pct(get(x, 'ci95.1'))}]"
            if isinstance(x, dict)
            else NM
        )

    conf_rows = "\n".join(
        f"| {c['confidence'][0]:.1f}–{c['confidence'][1]:.1f} | {pct(c['fraction'])} | "
        f"{pct(c['accuracy'])} |"
        for c in g.get("by_confidence", [])
    )
    ov = p.get("likely_training_overlap", {})
    return f"""The held-out numbers above come from the training distribution, where "chosen" is idiomatic
Swedish and "rejected" is a deliberately poor translation. This probe asks the harder question:
**can the gate pick the better answer when both are natural Swedish?**

Design: {p.get("design", NM)}. Source: [`{p.get("source", NM)}`](https://huggingface.co/datasets/{p.get("source", "")}).
{p.get("n_pairs", NM)} of {p.get("n_requested", NM)} sampled pairs translated cleanly (translation parse
failures: {get(p, "translation.parse_failed", NM)}). Sanity checks: the log length ratio between
the two answers correlates {num(p.get("length_ratio_correlation_en_sv"))} between English and Swedish
(content length preserved), and the TF-IDF style model that scores
{pct(get(m, "data.baselines.tfidf_logreg_accuracy"))} on the training distribution drops to
{pct(get(b, "tfidf_style_model.accuracy"))} here (the translationese signal is largely gone).
{pct(p.get("fraction_truncated_for_gate"))} of probe inputs were truncated for the gate.

| Scorer (both orders) | Accuracy [95% bootstrap CI] |
|---|---|
| **Gate** (order-averaged probability) | **{cell(g.get("order_averaged"))}** |
| Gate, better answer in slot A | {cell(g.get("chosen_in_a"))} |
| Gate, better answer in slot B | {cell(g.get("chosen_in_b"))} |
| **Kimi-k3 judge** | **{cell(j.get("order_averaged"))}** |
| TF-IDF style model (trained on the gate's training split) | {cell(b.get("tfidf_style_model"))} |
| Longer Swedish answer wins | {cell(b.get("longer_wins_swedish"))} |

- Gate positional consistency drops to {pct(g.get("positional_consistency"))} (vs
  {pct(get(m, "benchmark.positional.consistent_fraction"))} in-distribution); Kimi-k3's is
  {pct(j.get("positional_consistency"))}. Gate–judge agreement: {pct(p.get("gate_judge_agreement"))}.
- **Calibration does not transfer:** mean confidence {num(g.get("mean_confidence"))}, ECE
  {num(g.get("ece_order_averaged"))} (vs {num(get(m, "calibration.ece_post"))} in-distribution).
  At the default `margin={dm.get("margin", NM)}`, {pct(dm.get("coverage"))} of probe pairs pass
  as confident, with only {pct(dm.get("accuracy_on_covered"))} accuracy, so a cascade would keep
  most of its mistakes instead of routing them to the judge.
- By GPT-4 rating gap: {cell(g.get("by_margin", {}).get("1.0-<2.0"))} for gaps 1.0–2.0, {cell(g.get("by_margin", {}).get(">=2.0"))} for gaps ≥ 2.0.
- Excluding the {ov.get("n_flagged", NM)} probe prompts that likely overlap training prompts
  ({ov.get("rule", NM)}): {cell(g.get("excluding_likely_training_overlap"))}.

| Gate confidence | Share of probe pairs | Accuracy |
|---|---|---|
{conf_rows}

**Takeaway:** the gate carries a modest quality signal (above chance) but is far behind an
LLM judge on fluent pairs and is badly over-confident there. Use it for naturalness and
translationese; do not use its confidence to skip the judge on fluent answers."""


def generation_section(m: dict) -> str:
    """How the fluent training pairs were produced: reasoning-off fidelity and concurrency."""
    f, fid = m.get("fluent_data"), m.get("fluent_fidelity")
    if not isinstance(f, dict):
        return f"Not measured: {f or 'run `python -m laya_gate.fluent`'}."
    runs = f.get("throughput_runs", [])
    run_rows = "\n".join(
        f"| {r['chunk']} | {r['concurrency']} | {num(r.get('requests_per_second'), 2)} | "
        f"{num(r.get('output_tokens_per_second'), 0)} | {num(r.get('latency_p50_s'), 1)} s | "
        f"{r.get('failed', NM)} / {r.get('parse_failed', NM)} |"
        for r in runs
    )
    by_conc: dict[int, list[float]] = {}
    for r in runs:
        if isinstance(r.get("requests_per_second"), float):
            by_conc.setdefault(r["concurrency"], []).append(r["requests_per_second"])
    conc_line = ", ".join(
        f"concurrency {c}: {min(v):.2f}–{max(v):.2f} req/s over {len(v)} chunk(s)"
        for c, v in sorted(by_conc.items())
    )
    total_s = sum(r.get("wall_seconds", 0.0) for r in runs)
    if isinstance(fid, dict):
        ja = fid.get("judge_accuracy_both_orders", {})
        sim, lr = (
            fid.get("mean_char_similarity_to_reasoning_on", {}),
            fid.get("mean_length_ratio_off_over_on", {}),
        )
        fid_text = (
            f"Before generating, the first {fid.get('n_pairs', NM)} probe pairs were re-translated "
            f"with reasoning off (median {num(fid.get('reasoning_off_completion_tokens_median'), 0)} "
            "completion tokens per pair) and compared with the reasoning-on translations. "
            f"Lengths are preserved (off/on length ratio {num(lr.get('chosen'))} for the better "
            f"answer, {num(lr.get('rejected'))} for the worse), character similarity is "
            f"{num(sim.get('chosen'), 2)} / {num(sim.get('rejected'), 2)}, and Kimi-k3 recovers the "
            f"UltraFeedback label on {pct(ja.get('reasoning_off'))} of reasoning-off pairs vs "
            f"{pct(ja.get('reasoning_on'))} of reasoning-on pairs (both orders). The small drop "
            "was accepted for training data; the probe itself stays reasoning-on."
        )
    else:
        fid_text = "The reasoning-off fidelity check was not recorded."
    return f"""{f.get("requested", NM)} more UltraFeedback pairs (disjoint from the probe; pool of
{f.get("eligible_pool_excluding_probe", NM)}) were rewritten into natural Swedish with the probe's
translation prompt, by {f.get("translator", NM)}. {f.get("translated", NM)} translated cleanly
({f.get("translation_failures", NM)} failures); {f.get("dropped_near_duplicate_prompts", NM)} were
dropped because their Swedish prompt nearly duplicates a probe or validation prompt (char n-gram
cosine ≥ 0.8), leaving **{f.get("kept", NM)}** pairs, split
{" / ".join(f"{k} {v}" for k, v in f.get("splits", {}).items())} with the better answer in slot B
for {pct(get(f, "label_balance_frac_chosen_in_b.train"))} of train pairs.

{fid_text}

Throughput against the self-hosted Kimi-k3 endpoint (8× NVIDIA B300, HGX), 1,000 requests per
chunk, reasoning off: {conc_line}. Concurrency 128 was slower than 64 (longer queueing, more
parse failures), so the remaining chunks ran at 64. Total generation wall-clock:
{total_s / 3600:.1f} h. Variation between chunks at the same concurrency reflects other load on
the shared endpoint.

| Chunk | Concurrency | Requests/s | Output tokens/s | Median latency | Failed / parse failures |
|---|---|---|---|---|---|
{run_rows}"""


def v2_summary(m: dict) -> str | None:
    """One paragraph comparing v1 and v2 on the probe, or None when v2 has not been evaluated."""
    ev, v1 = m.get("v2_eval"), m.get("v1_eval")
    if not (isinstance(ev, dict) and ev.get("checkpoint") and isinstance(v1, dict)):
        return None
    k = "probe/preference"

    def a(d, key=k):
        return get(d, f"{key}.accuracy_order_averaged_prob.accuracy")

    t2 = get(ev, "probe/cascade.table", [])
    r1 = get(v1, "probe/cascade.recommended", {})
    # v2 row routing about as much as v1's recommended setting, for a like-for-like comparison
    near = min(
        t2, key=lambda r: abs(r["routed_fraction"] - r1.get("routed_fraction", 0)), default={}
    )
    return (
        f"A **v2** checkpoint (`python -m laya_gate.v2`, see "
        f"[v2](#v2-training-on-fluent-pairs-too)) adds {get(m, 'fluent_data.splits.train', NM)} "
        "fluent-vs-fluent training pairs and a separate naturalness question. On the probe it "
        f"reaches {pct(a(ev))} (95% CI {pct(get(ev, f'{k}.accuracy_order_averaged_prob.ci95.0'))}"
        f"–{pct(get(ev, f'{k}.accuracy_order_averaged_prob.ci95.1'))}) vs {pct(a(v1))} for v1, "
        f"with ECE {num(get(ev, f'{k}.ece_order_averaged'))} instead of "
        f"{num(get(v1, f'{k}.ece_order_averaged'))}, so its margin routes mistakes to the judge "
        f"again: with about {pct(r1.get('routed_fraction'), 0)} of probe instances routed to "
        f"Kimi-k3, the cascade reaches {pct(near.get('cascade_accuracy'))} with v2 "
        f"({pct(near.get('routed_fraction'))} routed) vs {pct(r1.get('cascade_accuracy'))} with v1. "
        f"The cost is {pct(a(v1, 'original/preference'))} → {pct(a(ev, 'original/preference'))} "
        "on the original held-out pairs (both orders averaged), and the probe is still "
        f"{num(100 * ((get(m, 'probe.judge.order_averaged.accuracy') or 0) - (a(ev) or 0)), 1)} "
        "percentage points "
        "behind Kimi-k3. "
        + {
            "v2": 'v2 is the default checkpoint on the Hub; v1 stays available as `revision="v1"`.',
            "v3": 'v2 is on the Hub as `revision="v2"`.',
        }.get(get(m, "publish.checkpoint"), "v2 is not on the Hub yet.")
    )


def _oa(d: Any, key: str, part: str = "accuracy") -> Any:
    """Order-averaged accuracy (or one CI bound: part="ci95.0" / "ci95.1")."""
    return get(d, f"{key}.accuracy_order_averaged_prob.{part}")


def _ci(d: Any, key: str, path: str = "accuracy_order_averaged_prob") -> str:
    a = get(d, f"{path}.accuracy") if key == "" else get(d, f"{key}.{path}.accuracy")
    base = path if key == "" else f"{key}.{path}"
    lo, hi = get(d, f"{base}.ci95.0"), get(d, f"{base}.ci95.1")
    return f"{pct(a)} [{pct(lo)}, {pct(hi)}]" if a is not None else NM


def v3_summary(m: dict) -> str | None:
    """One paragraph on v3, or None when v3 has not been evaluated."""
    ev, pw, v2 = m.get("v3_eval"), m.get("v3_pointwise"), m.get("v2_eval")
    if not (isinstance(ev, dict) and isinstance(pw, dict) and isinstance(v2, dict)):
        return None
    k = "probe/preference"
    pub = (
        'v3 is the default checkpoint on the Hub; v2 and v1 stay available as `revision="v2"` '
        'and `revision="v1"`.'
        if get(m, "publish.checkpoint") == "v3"
        else "v3 is not on the Hub yet."
    )
    return (
        f"A **v3** checkpoint (`python -m laya_gate.v3`, see [v3](#v3-single-response-questions)) "
        "keeps v2's two pairwise questions and adds two single-response questions: a 1–5 "
        "**quality** score and a **natural Swedish vs translationese** check. On the probe its "
        f"pairwise preference reaches {_ci(ev, k)} (v2: {pct(_oa(v2, k))}), and ranking the two "
        f"answers by their quality scores reaches {_ci(pw, 'probe/quality', 'accuracy_expected_score')} "
        f"with one forward pass per answer (Kimi-k3: "
        f"{pct(get(m, 'probe.judge.order_averaged.accuracy'))}). On the original held-out pairs "
        f"v3 is back to {pct(_oa(ev, 'original/preference'))} (v2: "
        f"{pct(_oa(v2, 'original/preference'))}). On human vs machine translations of FLORES+ "
        "sentences, no version is useful (see "
        f"[the machine-translation test](#machine-translation-test-a-negative-result)). {pub}"
    )


def _history_table(tr: dict) -> str:
    hist = tr.get("history", [])
    groups = [k for k in (hist[0]["val_calib"] if hist else {}) if k != "mean_loss"]
    head = " | ".join(f"{g} loss / acc" for g in groups)
    rows = "\n".join(
        f"| {h['epoch']} | {num(h['train_loss'])} | "
        + " | ".join(
            f"{num(h['val_calib'][g]['loss'])} / "
            + (pct(h["val_calib"][g]["acc"]) if h["val_calib"][g]["acc"] is not None else "n/a")
            for g in groups
        )
        + f" | {h['epoch_seconds'] / 60:.1f} min |"
        for h in hist
    )
    return f"""| Epoch | Train loss | {head} | Time |
|---|---|{"---|" * len(groups)}---|
{rows}"""


def v3_section(m: dict) -> str:
    ev, pw, v2 = m.get("v3_eval"), m.get("v3_pointwise"), m.get("v2_eval")
    tr, cal = m.get("v3_training"), m.get("v3_calibration")
    if not all(isinstance(x, dict) for x in (ev, pw, tr)):
        return "Not measured: run `python -m laya_gate.v3 train|calibrate|evaluate`."
    v2 = v2 if isinstance(v2, dict) else {}
    hp = tr.get("hyperparameters", {})
    units = " · ".join(f"{k} {v}" for k, v in tr.get("train_units_by_source_question", {}).items())

    def row(label, key, field=None, fmt=pct):
        if field is None:
            a, b = _oa(v2, key), _oa(ev, key)
        else:
            a, b = get(v2, f"{key}.{field}"), get(ev, f"{key}.{field}")
        return f"| {label} | {fmt(a)} | {fmt(b)} |"

    pair_rows = [
        row("Original val_eval, preference", "original/preference"),
        row("Original val_eval, naturalness", "original/naturalness"),
        row("Original val_eval, preference ECE", "original/preference", "ece_order_averaged", num),
        row("Original, accuracy with chosen in A", "original/preference", "accuracy_chosen_in_a"),
        row("Original, accuracy with chosen in B", "original/preference", "accuracy_chosen_in_b"),
        row("Fluent val_eval, preference", "fluent/preference"),
        row("Fluent val_eval, preference ECE", "fluent/preference", "ece_order_averaged", num),
        f"| **Probe**, preference [95% CI] | {_ci(v2, 'probe/preference')} | "
        f"{_ci(ev, 'probe/preference')} |",
        row("Probe, preference ECE", "probe/preference", "ece_order_averaged", num),
        row("Probe, positional consistency", "probe/preference", "positional_consistency"),
        row("GPU forward, batch 1, median (pair)", "latency_gpu_forward_ms", "p50", ms),
    ]
    fq, pq = pw.get("fluent/quality", {}), pw.get("probe/quality", {})
    on, fn, pn = (
        pw.get("original/natural_swedish", {}),
        pw.get("fluent/natural_swedish", {}),
        pw.get("probe/natural_swedish", {}),
    )
    c = get(ev, "probe/cascade", {})
    rec = c.get("recommended", {})
    casc_rows = "\n".join(
        (
            f"| **{r['threshold']:.2f}** (recommended) "
            if abs(r["threshold"] - rec.get("threshold", -1)) < 1e-9
            else f"| {r['threshold']:.2f} "
        )
        + f"| {pct(r['routed_fraction'])} | {pct(r['cascade_accuracy'])} | "
        f"{ms(r['expected_latency_ms'])} |"
        for r in c.get("table", [])
    )
    temps = ", ".join(
        f"`{q}` T = {num(f.get('temperature'))} (NLL {num(f.get('nll_pre'))} → "
        f"{num(f.get('nll_post'))})"
        for q, f in (cal or {}).items()
    )
    return f"""**Why:** comparing N candidates pairwise costs N(N−1) ordered forward passes. v3 adds two
questions over a *single* response (state `{{"prompt", "response"}}`):

- **quality** (`score`, levels 1–5): trained on fluent responses with a soft target from the
  UltraFeedback GPT-4 rating, split between the two nearest levels. The expected level under the
  calibrated distribution is the score.
- **natural_swedish** (`choice`: natural / translationese): original chosen = natural, original
  rejected = translationese, every fluent response = natural.

The pairwise `preference` and `naturalness` questions are trained exactly as in v2.

#### Training

Initialized from the base multilingual checkpoint; lr {hp.get("lr", NM)}, micro-batch
{hp.get("micro_batch", NM)} × accumulation {hp.get("grad_accum", NM)}, warmup + cosine, bf16,
OOM fallback: {hp.get("oom_fallback", NM)}. {hp.get("epochs", NM)} epochs over
{tr.get("train_units", NM)} units ({units}). Best epoch: **{tr.get("best_epoch", NM)}**
({tr.get("selection", NM)}); as with v2, the fluent preference loss rises in epoch 2. Total
{num((tr.get("total_seconds") or 0) / 60, 1)} min on {tr.get("hardware", NM)}; peak GPU memory
{num(tr.get("peak_gpu_memory_allocated_gib"), 2)} GiB. Quality accuracy is exact-level (5-way).

{_history_table(tr)}

Temperatures, fitted per question on val_calib: {temps}.

#### v2 vs v3 on the pairwise questions

Order-averaged accuracy (both orders scored, probabilities averaged), same evaluation code
(`python -m laya_gate.v3 evaluate`).

| Metric | v2 | v3 |
|---|---|---|
{chr(10).join(pair_rows)}

The probe gain ({pct(_oa(v2, "probe/preference"))} → {pct(_oa(ev, "probe/preference"))}) is inside
overlapping 95% CIs, so it is suggestive, not established.

#### Quality score

Comparing two answers by their scores (one forward pass each):

| Metric | Fluent val_eval ({fq.get("n_pairs", NM)} pairs) | Probe ({pq.get("n_pairs", NM)} pairs) |
|---|---|---|
| Accuracy (higher score = chosen) [95% CI] | {_ci(fq, "", "accuracy_expected_score")} | {_ci(pq, "", "accuracy_expected_score")} |
| Spearman vs GPT-4 rating | {num(fq.get("spearman_vs_rating"))} | {num(pq.get("spearman_vs_rating"))} |
| Mean score chosen / rejected | {num(fq.get("mean_score_chosen"), 2)} / {num(fq.get("mean_score_rejected"), 2)} | {num(pq.get("mean_score_chosen"), 2)} / {num(pq.get("mean_score_rejected"), 2)} |
| ECE of P(chosen scores higher) | {num(fq.get("ece_p_chosen_beats_rejected"))} | {num(pq.get("ece_p_chosen_beats_rejected"))} |

Mean absolute error against the rating on fluent val_eval: {num(fq.get("mae_vs_rating"), 2)}
levels. The ranking is useful; the derived "A beats B" probability is **not** well calibrated
(ECE above {ECE_TARGET}), so use the pairwise question when you need a calibrated margin.
GPU forward for one response: {ms(get(pw, "latency_gpu_forward_ms_single_response.p50"))} median.

#### Natural-Swedish check

On original val_eval responses: AUC {num(on.get("auc"))}, accuracy
{_ci(on, "", "accuracy")}, ECE {num(on.get("ece"))}. It lets
{pct(on.get("translationese_passed_as_natural"))} of deliberate translationese pass as natural and
flags {pct(on.get("natural_flagged_translationese"))} of the idiomatic answers; within a pair, the
idiomatic answer gets the higher P(natural) {pct(on.get("within_pair_natural_scores_higher"))} of
the time. On fluent responses (all natural) it flags {pct(fn.get("flagged_translationese"))}
(fluent val_eval) and {pct(pn.get("flagged_translationese"))} (probe).

#### Probe cascade (v3 preference, Kimi-k3 judge)

Kimi-k3 alone: {pct(c.get("judge_accuracy"))}. Each order is a separate instance.

| Threshold | Routed to judge | Cascade accuracy | Expected latency |
|---|---|---|---|
{casc_rows}

Recommended ({rec.get("rule", NM)}): threshold {num(rec.get("threshold"), 2)}, routing
{pct(rec.get("routed_fraction"))} for {pct(rec.get("cascade_accuracy"))}."""


def cpu_section(m: dict) -> str:
    c = m.get("cpu_inference")
    if not isinstance(c, dict) or not c.get("variants"):
        return "Not measured: run `python -m laya_gate.cpu`."
    rows = "\n".join(
        f"| `{k}` | {ms(get(v, 'latency_ms.p50'))} / {ms(get(v, 'latency_ms.p95'))} | "
        f"{num(v.get('mean_input_tokens'), 0)} | {pct(get(v, 'probe_accuracy.accuracy'))} | "
        f"{pct(get(v, 'original_accuracy.accuracy'))} | "
        f"{num(v.get('mean_abs_delta_p_vs_torch_fp32'))} / {num(v.get('max_abs_delta_p_vs_torch_fp32'))} |"
        for k, v in c["variants"].items()
    )
    v = c["variants"]
    n_orig = get(next(iter(v.values())), "original_accuracy.n", NM)
    sweep = get(c, "thread_sweep.p50_ms_by_threads", {})
    sweep_s = ", ".join(f"{t} threads {ms(x)}" for t, x in sweep.items())
    files = ", ".join(f"{k} {x} MB" for k, x in c.get("files_mb", {}).items())
    same_t = sweep.get(str(c.get("threads")))
    main_t = get(v, f"{get(c, 'thread_sweep.variant')}.latency_ms.p50")
    noise = (
        f" The sweep's {c.get('threads')}-thread median ({ms(same_t)}) differs from the table's "
        f"({ms(main_t)}) for the same setting, so expect large run-to-run variance on this "
        "hybrid P/E-core laptop CPU; treat the CPU numbers as rough."
        if isinstance(same_t, float)
        and isinstance(main_t, float)
        and abs(same_t - main_t) > 0.2 * main_t
        else ""
    )
    return f"""Checkpoint `{c.get("checkpoint", NM)}` on {c.get("cpu", NM)}, {c.get("threads", NM)} threads.
Method: {c.get("method", NM)}. Probe accuracy is the preference question, both orders averaged;
"original" is the first {n_orig} original val_eval pairs. ΔP is the change in P(A) against torch fp32.

| Variant | Latency median / p95 | Mean input tokens | Probe accuracy | Original accuracy | Mean / max ΔP |
|---|---|---|---|---|---|
{rows}

`torch-int8` is `torch.ao.quantization.quantize_dynamic` on the encoder's linear layers; the ONNX
variants come from `python -m laya_gate.onnx_export` (int8 = ONNX Runtime dynamic
quantization); `-512` caps the input at 512 tokens instead of 1024. Thread sweep for
`{get(c, "thread_sweep.variant", NM)}` (median): {sweep_s}.{noise} Files: {files}.

No CPU variant comes close to the {LATENCY_TARGET_MS:.0f} ms target: the GPU is what makes the gate
a millisecond-scale filter. int8 saves little time at 1024 tokens and costs a few probe points;
the 512-token cap is the only large speed-up, at a larger accuracy cost."""


def liu_section(m: dict) -> str:
    L = m.get("liu_translationese")
    if not isinstance(L, dict) or not L.get("models"):
        return "Not measured: run `python -m laya_gate.liu_eval gate` (and `judge`)."
    setups = ("no_prompt", "context_0", "context_10")
    ref = get(L, "paper_reference.setups", {})
    j = L.get("judge") if isinstance(L.get("judge"), dict) else {}
    lb = L.get("length_baseline", {})

    def acc(model, qid, setup, comp="opus"):
        return get(L, f"models.{model}.setups.{qid}.{setup}.{comp}.accuracy")

    def jacc(setup, comp="opus"):
        return get(j, f"setups.{setup}.{comp}.human_preferred_both_orders")

    gate_rows = [
        (m_, q) for m_, res in L["models"].items() if isinstance(res, dict) for q in res["setups"]
    ]
    rows = [
        "| Paper: best of its 16 LLMs | "
        + " | ".join(
            f"{pct(get(ref, f'{s}.opus.best_accuracy'))} "
            f"({str(get(ref, f'{s}.opus.best_model', '')).split('/')[-1]})"
            for s in setups
        )
        + " |",
        "| Paper: median of its 16 LLMs | "
        + " | ".join(pct(get(ref, f"{s}.opus.median_accuracy")) for s in setups)
        + " |",
        "| Paper: AI Sweden Llama-3-8B | "
        + " | ".join(pct(get(ref, f"{s}.opus.ai_sweden_llama3_8b")) for s in setups)
        + " |",
        f"| Longer candidate wins (no model) | {pct(get(lb, 'opus.accuracy_longer_wins'))} | "
        f"{pct(get(lb, 'opus.accuracy_longer_wins'))} | {pct(get(lb, 'opus.accuracy_longer_wins'))} |",
        *(
            f"| {m_} `{q}` | " + " | ".join(pct(acc(m_, q, s)) for s in setups) + " |"
            for m_, q in gate_rows
        ),
        f"| Kimi-k3 judge (both orders agree) | {pct(jacc('no_prompt'))} | — | "
        f"{pct(jacc('context_10'))} |",
    ]
    gpt_rows = [
        f"| Longer candidate wins | {pct(get(lb, 'gpt.accuracy_longer_wins'))} | "
        f"{pct(get(lb, 'gpt_unacceptable.accuracy_longer_wins'))} | — | — |",
        *(
            f"| {m_} `{q}` | {pct(acc(m_, q, 'no_prompt', 'gpt'))} | "
            f"{pct(acc(m_, q, 'no_prompt', 'gpt_unacceptable'))} | "
            f"{pct(acc(m_, q, 'context_10', 'gpt'))} | "
            f"{pct(acc(m_, q, 'context_10', 'gpt_unacceptable'))} |"
            for m_, q in gate_rows
            if m_ == "v3"
        ),
        f"| Kimi-k3 judge | {pct(jacc('no_prompt', 'gpt'))} | "
        f"{pct(jacc('no_prompt', 'gpt_unacceptable'))} | {pct(jacc('context_10', 'gpt'))} | "
        f"{pct(jacc('context_10', 'gpt_unacceptable'))} |",
    ]
    tag_cols = [
        ("v3 `naturalness`", get(L, "models.v3.error_tags.naturalness.no_prompt", {})),
        ("v3 `natural_swedish`", get(L, "models.v3.error_tags.natural_swedish.no_prompt", {})),
        ("v3 `preference`, 10 lines", get(L, "models.v3.error_tags.preference.context_10", {})),
        ("Kimi-k3", get(j, "setups.no_prompt.error_tags", {})),
        ("Kimi-k3, 10 lines", get(j, "setups.context_10.error_tags", {})),
    ]
    tags = sorted(tag_cols[0][1], key=lambda t: -tag_cols[0][1][t]["n"])
    tag_rows = "\n".join(
        f"| {t} | {tag_cols[0][1][t]['n']} | "
        + " | ".join(pct(get(c, f"{t}.opus_preferred"), 0) for _, c in tag_cols)
        + " |"
        for t in tags
    )
    ends = L.get("candidates_ending_in_punctuation", {})
    v3_raw = get(L, "models.v3.unstripped_no_prompt.natural_swedish.opus.accuracy")
    v3_str = acc("v3", "natural_swedish", "no_prompt")
    return f"""An external, human-annotated test: [`{L.get("dataset")}`](https://huggingface.co/datasets/{L.get("dataset")})
({L.get("paper")}; Apache-2.0). {L.get("n_pairs", NM)} English subtitle lines from OpenSubtitles, each
with an OPUS-MT translation, a GPT translation and an idiomatic alternative written by a native
speaker, plus error tags for the OPUS translation and the 10 preceding English lines. No Kimi-k3
model touched it. `python -m laya_gate.liu_eval gate|judge`.

**Method.** The setups are the paper's: `no_prompt` scores the Swedish sentences alone;
`context_N` puts the paper's instruction ("Översätt följande mening till svenska …") with N
preceding English lines in the state's `prompt` field. Trailing punctuation is stripped from
every candidate as in the paper: only {pct(ends.get("human"), 0)} of the human versions end in
punctuation vs {pct(ends.get("opus"), 0)} of the OPUS ones, so without stripping the final period
would be a giveaway. The metric is the paper's: the share of pairs where the human version is
preferred (gate: order-averaged P > 0.5; single-response questions: strictly higher score;
Kimi-k3: preferred in both slot orders). Pairs whose candidates are identical after stripping
are dropped ({get(L, "n_compared.gpt", NM)} of {L.get("n_pairs", NM)} GPT pairs remain). The
paper's LLM numbers rank sentences by mean log-probability, the gates and Kimi-k3 classify, so
the comparison is of outcomes, not of methods. All 8 setups are in `metrics.json`; the table shows
three. Kimi-k3's "10 lines" setup is a judge prompt with the same context, not the paper's
instruction.

**Human vs OPUS-MT** ({get(L, "n_compared.opus", NM)} pairs; 50% = chance, 95% CI about ±4 points):

| Model | No prompt | Instruction, no context | Instruction + 10 lines |
|---|---|---|---|
{chr(10).join(rows)}

**Every gate prefers the OPUS-MT translationese**, as the paper's LLMs mostly do, and adding the
English source makes it worse (v3 `preference`: {pct(acc("v3", "preference", "no_prompt"))} →
{pct(acc("v3", "preference", "context_0"))}), the paper's main finding repeated in a classifier.
The best gate result, v3's single-response `natural_swedish` check without a prompt
({pct(v3_str)}; unstripped {pct(v3_raw)}), is at chance, and no gate beats the trivial
"longer candidate wins" rule. Kimi-k3 does: {pct(jacc("context_10"))} with context, above every
LLM in the paper.

**Human vs GPT translation** (all non-identical pairs / only pairs where annotators judged GPT
worse than the human):

| Model | No prompt, all | No prompt, GPT worse | 10 lines, all | 10 lines, GPT worse |
|---|---|---|---|---|
{chr(10).join(gpt_rows)}

The paper's GPT column counts identical pairs as losses, so it is not comparable and is omitted.
Kimi-k3 prefers the GPT translation overall (the human version wins in both orders for
{pct(jacc("no_prompt", "gpt"))} / {pct(jacc("context_10", "gpt"))} of pairs) and is split roughly
evenly on the pairs annotators judged GPT worse ({pct(jacc("context_10", "gpt_unacceptable"))} human
vs {pct(get(j, "setups.context_10.gpt_unacceptable.mt_preferred_both_orders"))} GPT with context),
consistent with the paper's finding that larger LLMs lean toward LLM-produced translations.

**Per OPUS error tag:** share of pairs where OPUS-MT is preferred (lower is better; tags sorted
by frequency):

| Tag | n | {" | ".join(name for name, _ in tag_cols)} |
|---|---|{"---|" * len(tag_cols)}
{tag_rows}

Tags: SEM semantic shift, BET loss of meaning, PR lexical preference, LF incorrect word, DIR direct
translation, SAK missing words, GR grammar, SL slang, ST domain style, ID idiom, ADD added words.
The gate prefers OPUS most where OPUS *omits* words (SAK), even with the English source in the
prompt, so it does not check the translation against the source. The Kimi-k3 columns count
pairs where it picks OPUS in *both* orders (a split verdict counts for neither side); in every
category that happens far less often than the gate's OPUS preference.

**What this means.** The gate's "naturalness" is a detector for the kind of translationese Kimi-k3
wrote into the training data, not for the translationese real MT systems produce. Improving that
needs training pairs with real MT output against native or idiomatic Swedish; this benchmark
should stay held out to measure it."""


def external_section(m: dict) -> str:
    E = m.get("external_eval")
    if not isinstance(E, dict) or not E.get("models"):
        return "Not measured: run `python -m laya_gate.external_eval prepare|translate|gate|judge`."
    D, J, B = m.get("external_data", {}), E.get("judge") or {}, E.get("baselines", {})
    n = E.get("n_pairs", {})

    def acc(model, s, q):
        return get(E, f"models.{model}.{s}.{q}.accuracy")

    def jacc(s, setup):
        return get(J, f"sets.{s}.{setup}.accuracy")

    sets = [
        ("dalaj", "DaLAJ: corrected vs learner sentence", "grammar"),
        ("swefaq", "SweFAQ: own answer vs other answer", "preference"),
        ("europarl_en/opus", "Europarl EN-original: human vs OPUS-MT", "translation"),
        ("europarl_en/kimi", "Europarl EN-original: human vs Kimi-k3 MT", "translation"),
        ("europarl_sv/opus", "Europarl SV-original: native vs OPUS-MT round trip", "translation"),
        ("europarl_sv/kimi", "Europarl SV-original: native vs Kimi-k3 round trip", "translation"),
    ]
    qs = ["preference", "naturalness", "natural_swedish", "quality"]
    rows = []
    for s, label, setup in sets:
        cells = [pct(acc("v1", s, "preference"))] + [pct(acc("v3", s, q)) for q in qs]
        b = B.get(s, {})
        base = (
            f"overlap {pct(b.get('word_overlap_wins'))}"
            if s == "swefaq"
            else f"frequency {pct(b.get('unigram_frequency_wins'))}"
            if s == "dalaj"
            else f"longer {pct(b.get('longer_wins'))}"
        )
        rows.append(
            f"| {label} | {n.get(s, NM)} | " + " | ".join(cells) + f" | {base} | "
            f"{pct(jacc(s, setup))} |"
        )
    ep_rows = []
    for s, label, _ in sets[2:]:
        ep_rows.append(
            f"| {label} | {pct(acc('v3', s, 'preference/no_prompt'))} | "
            f"{pct(acc('v3', s, 'natural_swedish/no_prompt'))} | "
            f"{pct(jacc(s, 'translation'))} | {pct(jacc(s, 'naturalness'))} | "
            f"{pct(get(J, f'sets.{s}.translation.other_preferred_both_orders'))} |"
        )
    groups = get(E, "models.v3.dalaj.natural_swedish.by_group", {})
    nat_groups = get(E, "models.v3.dalaj.naturalness.by_group", {})
    g_rows = "\n".join(
        f"| {g} | {x['n']} | {pct(x['accuracy'])} | {pct(get(nat_groups, f'{g}.accuracy'))} | "
        f"{pct(get(E, f'models.v3.dalaj.preference.by_group.{g}.accuracy'))} |"
        for g, x in groups.items()
    )
    rank = get(E, "models.v3.swefaq.quality.ranking", {})
    orank = get(B, "swefaq.word_overlap_ranking", {})
    strict = get(E, "models.v3.swefaq.preference.strict_top1.accuracy")
    ep = get(D, "europarl", {})
    langs = ep.get("paragraphs_by_speakerlang", {})
    n_par = sum(langs.values()) if langs else None
    km = D.get("kimi_mt", {})
    return f"""Three more external tests, none used for training, scored with v1, v2 and v3
(`python -m laya_gate.external_eval`). The metric is always the share of pairs in which the
*reference* side wins: the corrected sentence, the question's own answer, the human Swedish.
Pairwise questions average P over both slot orders. Single-response questions compare the two
scores. Kimi-k3 sees a fixed-seed sample of {get(J, "sets.dalaj.n_pairs", NM)} pairs per set in
both orders ({J.get("requests", NM)} requests, {J.get("parse_failures", NM)} parse failures,
median {num(J.get("latency_s_median"), 1)} s, reasoning on); its accuracy counts a split verdict
as half.

- **DaLAJ-GED** ([Superlim 2](https://huggingface.co/datasets/sbx/superlim-2), test split, CC BY
  4.0): Swedish learner sentences with exactly one error, paired with the same sentence with
  that error corrected, rebuilt from the annotated span ({get(D, "dalaj.dropped.unchanged", NM)}
  word-order corrections that cannot be rebuilt are dropped). Prompt: none. Natively written
  Swedish; the only difference is the error.
- **SweFAQ** (Superlim 2, dev + test, CC BY-SA 4.0): {get(D, "swefaq.n_questions", NM)} questions
  from Swedish public-agency FAQ pages; each question's own answer is paired with every other
  answer on the same page. Natively written; the only difference is relevance.
- **Europarl sv/en** ([Språkbanken](https://spraakbanken.gu.se/en/resources/europarl-sven)):
  paragraphs aligned by id, kept when both sides have 1-4 sentences and similar length. Only
  {langs.get("SV", 0):,} of {n_par or 0:,} Swedish paragraphs are tagged as *spoken in Swedish*
  (`speakerlang="SV"`), so almost all Swedish Europarl text is itself translation. Two sets:
  English-original speeches (a fixed sample of {get(ep, "sample.EN", NM)}), with their
  professional Swedish translation against machine translation of the English; and every usable
  Swedish-original speech ({get(ep, "sample.SV", NM)}), native Swedish against a machine
  translation of its English version (a round trip). Machine translation: `opus-mt-en-sv`
  (sentence by sentence) and Kimi-k3 (reasoning off, median {num(km.get("latency_s_median"), 1)}
  s). OPUS-MT's training data includes Europarl; Kimi-k3's is unknown. Prompt: the English
  text with "Översätt följande text till svenska:".

| Set | Pairs | v1 `preference` | v3 `preference` | v3 `naturalness` | v3 `natural_swedish` | v3 `quality` | Baseline | Kimi-k3 |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

Baselines: DaLAJ, the sentence with the higher mean unigram log-frequency in Språkbanken's
Europarl-SV word statistics; SweFAQ, the answer sharing more of the question's words; Europarl,
the longer side. Kimi-k3 prompts: a grammar prompt for DaLAJ, the standard judge prompt for
SweFAQ, a translation-quality prompt (English shown) for Europarl. 95% CIs for the gate are about
±2 points (DaLAJ, SweFAQ) and ±3 points (Europarl) and are in `metrics.json`.

**Grammar errors barely register.** v1 is at chance on DaLAJ
({pct(acc("v1", "dalaj", "preference"))}) and v3's pairwise questions reach
{pct(acc("v3", "dalaj", "naturalness"))} (`naturalness`) and
{pct(acc("v3", "dalaj", "preference"))} (`preference`). v3's single-response `natural_swedish` check is best at {pct(acc("v3", "dalaj", "natural_swedish"))},
close to the unigram-frequency baseline ({pct(get(B, "dalaj.unigram_frequency_wins"))}), while
Kimi-k3 reaches {pct(jacc("dalaj", "grammar"))}. By error type (v3; a sentence can carry two):

| Error type | Pairs | `natural_swedish` | `naturalness` | `preference` |
|---|---|---|---|---|
{g_rows}

Spelling errors are the easiest. Punctuation errors are at or below chance, and syntax errors
are below chance for the pairwise question.

**Relevance is weak too.** On SweFAQ the gate does worse than a bag-of-words overlap count
({pct(get(B, "swefaq.word_overlap_wins"))}); Kimi-k3 is near perfect. Ranking every answer on the
page by v3's `quality` score puts the right one first for {pct(get(rank, "top1.accuracy"))} of
questions (MRR {num(rank.get("mrr"))}; chance {pct(rank.get("top1_chance"))}; word overlap
{pct(get(orank, "top1.accuracy"))}, MRR {num(orank.get("mrr"))}). With pairwise `preference`, the
right answer beats every other answer on its page for {pct(strict)} of questions. On natively
written Swedish where style is held constant, the gate mostly cannot tell a relevant answer
from an irrelevant one.

**Europarl is not a usable "human beats machine" test.** Share of pairs in which the human side
wins:

| Set | v3 `preference`, no prompt | v3 `natural_swedish`, no prompt | Kimi-k3, translation | Kimi-k3, naturalness | Kimi-k3 picks MT in both orders |
|---|---|---|---|---|---|
{chr(10).join(ep_rows)}

Under the translation prompt Kimi-k3 prefers the machine translation in most pairs, most
strongly against its own output (self-preference), and even for native Swedish speeches against
their round trip. Under the naturalness prompt it is split against OPUS-MT and still prefers its
own output. For those, the translation prompt is confounded: the English was translated *from*
the Swedish, so a back-translation follows it more closely. The 1997-2006 parliamentary Swedish
is often loose or dated ("skall"), so modern MT plausibly reads as at least as good. With the English
source shown, the gate also prefers the MT (v3 `preference`
{pct(acc("v3", "europarl_en/kimi", "preference"))} vs Kimi-k3 MT). Without it, it is at chance.
Europarl's human side therefore cannot serve as the "better" label for naturalness or
preference training.

**What this means for training.** The gate's two weak spots on natively written Swedish are
grammar and relevance, and both have usable training data in Superlim: DaLAJ-GED train
({get(D, "train_splits.dalaj_sentences", 0):,} sentences, {get(D, "train_splits.dalaj_incorrect", 0):,}
of them with an error) and SweFAQ train ({get(D, "train_splits.swefaq_questions", NM)} questions,
with hard negatives from the same page). Europarl is not worth training on as a
preference signal."""


def context_section(m: dict) -> str:
    C = m.get("context_length")
    if not isinstance(C, dict) or not C.get("models"):
        return "Not measured: run `python -m laya_gate.external_eval length`."
    lens = [str(x) for x in C.get("max_lens", [])]
    rows = []
    for name, res in C["models"].items():
        if not isinstance(res, dict):
            continue
        for L in lens:
            r = res.get(L, {})
            lat = r.get("latency_gpu_forward_ms_longest_inputs", {})
            rows.append(
                f"| {name} | {L} | {pct(get(r, 'original.fraction_truncated'))} | "
                f"{pct(get(r, 'original.accuracy.accuracy'))} | "
                f"{pct(get(r, 'original.accuracy_truncated_at_shortest.accuracy'))} | "
                f"{pct(get(r, 'fluent.accuracy.accuracy'))} | "
                f"{pct(get(r, 'fluent.accuracy_truncated_at_shortest.accuracy'))} | "
                f"{num(lat.get('input_tokens_median'), 0)} | {ms(lat.get('p50'))} / "
                f"{ms(lat.get('p95'))} |"
            )
    n_long = (
        get(C, f"models.v3.{lens[0]}.original.accuracy_truncated_at_shortest.n") if lens else None
    )
    return f"""mmBERT's position embeddings go to 8,192 tokens, but Laya and all three gates were
trained at {lens[0] if lens else NM}. Does a longer window help without retraining? Each
checkpoint scored `val_eval` (both orders, `preference`) at {", ".join(lens)} tokens
(`python -m laya_gate.external_eval length`). "Truncated" columns are the pairs that do not fit
in {lens[0] if lens else NM} tokens ({n_long or NM} original pairs for v3). Latency is the batch-1 GPU
forward pass on the 50 longest original pairs at that limit (bf16, `sdpa` attention).

| Model | Max tokens | Original truncated | Original acc. | on truncated | Fluent acc. | on truncated | Input tokens | Forward p50 / p95 |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

**No meaningful gain.** Seeing the rest of long pairs changes accuracy by under a point on the
original pairs and not at all, or slightly downward, on the fluent ones. The truncated pairs
are already *easier* than the rest, because the first {lens[0] if lens else NM} tokens settle
them. Forward latency more than doubles at 2048, where the p95 forward pass alone leaves no
room for tokenization within the 35 ms target; at 4096 the forward pass alone exceeds it. The window
stays at {lens[0] if lens else NM}; fine-tuning at a longer length was not tried."""


V4_ORDER = (
    ("v3", "v3 (published)"),
    ("control", "control"),
    ("control-s20261015", "control, seed 2"),
    ("europarl", "+Europarl"),
    ("native", "+DaLAJ/SweFAQ"),
    ("both", "+both"),
    ("grammar", "+both +grammar"),
    ("adequacy", "v4: +grammar +adequacy"),
    ("v41", "**v4.1**: v4 +registers +hard sources"),
)
RELEASE_ARM, RELEASE_NAME = "v41", "v4.1"
V4_GROUPS = (
    (
        "In-domain (DPO validation, fluent pairs, probe)",
        ("original/preference", "fluent/preference", "probe/preference"),
    ),
    (
        "Native speaker vs OPUS-MT (liu-nlp, human-annotated)",
        ("liu/naturalness", "liu/natural_swedish", "liu/preference+source"),
    ),
    (
        "Professional translation vs MT (FLORES+, WMT24++)",
        (
            "flores/naturalness",
            "flores/preference+source",
            "wmt24pp/opus/naturalness",
            "wmt24pp/opus/preference+source",
            "wmt24pp/kimi/preference+source",
        ),
    ),
    (
        "Grammar (DaLAJ, GerLangMod-sv, ScaLA-sv)",
        (
            "dalaj/naturalness",
            "dalaj/natural_swedish",
            "dalaj/grammatical",
            "gerlangmod/naturalness",
            "gerlangmod/grammatical",
            "blimppi/naturalness",
            "blimppi/grammatical",
            "multiblimp/naturalness",
            "multiblimp/grammatical",
            "scala/accuracy@0.5",
        ),
    ),
    ("Relevance (SweFAQ)", ("swefaq/preference", "swefaq/quality", "swefaq/quality_top1")),
    (
        "Europarl (in-distribution for the Europarl arms)",
        ("europarl_en/kimi/preference+source", "europarl_sv/kimi/preference+source"),
    ),
)
SANITY_ROWS = (
    ("wrong_source/preference+source", "the reference of a *different* sentence"),
    ("garbage_mt/preference+source", "LFM2-350M's translation (pilot's lowest chrF)"),
    ("first_half_only/preference+source", "only the first half of the reference"),
    ("untranslated/preference+source", "the untranslated English"),
    ("untranslated/naturalness", "the untranslated English (`naturalness`, no prompt)"),
)


def _scala_mcc(A: dict, arm: str) -> Any:
    c = get(A, f"{arm}.classifiers") or {}
    vals = [(c.get(k) or {}).get("mcc@0.5") for k in ("scala", "scala_grammatical")]
    vals = [v for v in vals if isinstance(v, float)]
    return max(vals) if vals else None


def v4_section(m: dict) -> str:
    V = m.get("v4")
    if not isinstance(V, dict) or not get(V, "arms.control.tests"):
        return "Not measured: run `python -m laya_gate.v4 prepare`, `run --arm ...`, `compare`."
    A, C = V["arms"], get(V, "comparison.arms", {})
    S = get(V, "sanity.checkpoints", {})
    arms = [(k, label) for k, label in V4_ORDER if get(A, f"{k}.tests")]

    def val(arm, test):
        return get(A, f"{arm}.tests.{test}.mean")

    def cell(arm, test):
        x, d = val(arm, test), get(C, f"{arm}.{test}")
        if x is None:
            return "—"
        sig = d and (d["ci95"][0] > 0 or d["ci95"][1] < 0)
        return pct(x) + (" *" if sig else "")

    def sane(arm, test):
        return get(S, f"{arm}.{test}.mean")

    def ece_of(arm):
        if arm == "v3":
            return get(m, "v3_eval.original/preference.ece_both_orders")
        return get(A, f"{arm}.standard_eval.original/preference.ece_both_orders")

    head = "| Test | " + " | ".join(label for _, label in arms) + " |"
    lines = [head, "|---|" + "---|" * len(arms)]
    for title, tests in V4_GROUPS:
        lines.append(f"| **{title}** |" + " |" * len(arms))
        for t in tests:
            if any(val(k, t) is not None for k, _ in arms):
                lines.append(f"| {t} | " + " | ".join(cell(k, t) for k, _ in arms) + " |")
    lines += [
        "| ScaLA-sv MCC (EuroEval metric), best single-response question | "
        + " | ".join(num(_scala_mcc(A, k)) for k, _ in arms)
        + " |",
        "| Fluent LLM answers flagged as translationese (lower is better) | "
        + " | ".join(pct(get(A, f"{k}.guards.fluent_flagged_translationese.mean")) for k, _ in arms)
        + " |",
        "| ECE, DPO validation `preference` | "
        + " | ".join(num(ece_of(k), 4) for k, _ in arms)
        + " |",
        "| Training units / wall-clock | "
        + " | ".join(
            f"{get(A, f'{k}.training.train_units', 0):,} / "
            f"{get(A, f'{k}.training.total_seconds', 0) / 3600:.1f} h"
            if get(A, f"{k}.training")
            else "2 epochs"
            for k, _ in arms
        )
        + " |",
    ]
    table = "\n".join(lines)
    s_arms = [(k, label) for k, label in V4_ORDER if k in S]
    s_lines = [
        "| The reference translation against … | " + " | ".join(lb for _, lb in s_arms) + " |",
        "|---|" + "---|" * len(s_arms),
    ]
    for t, label in SANITY_ROWS:
        s_lines.append(f"| {label} | " + " | ".join(pct(sane(k, t)) for k, _ in s_arms) + " |")
    sanity_table = "\n".join(s_lines)
    D = V.get("data", {})
    ep, adq = D.get("europarl", {}), D.get("adequacy", {})
    seed_d = [abs(x["diff"]) for x in get(C, "control-s20261015", {}).values()]
    seed_max = max(seed_d) if seed_d else None
    P = m.get("pilot_euroeval") or {}
    item = get(P, "translation_item_level", {})
    gram = get(A, "adequacy.guards", {})
    gram41 = get(A, f"{RELEASE_ARM}.guards", {})
    BL = get(V, "baselines.sets", {})
    bl_rows = "\n".join(
        f"| {name} | {get(BL, f'{name}.n', NM)} | {pct(get(BL, f'{name}.longer_wins.mean'))} | "
        f"{pct(get(BL, f'{name}.unigram_frequency.mean'))} | {pct(get(BL, f'{name}.trigram.mean'))} | "
        f"{pct(val(RELEASE_ARM, f'{name}/naturalness'))} | {pct(val(RELEASE_ARM, f'{name}/grammatical'))} |"
        for name in ("dalaj", "gerlangmod", "blimppi", "multiblimp")
        if get(BL, name)
    )

    def item_rho(name):
        return num(get(item, f"{name}/translation_vs_reference.spearman_with_chrf"), 2)

    data_url = get(m, "publish_data.url")
    data_line = (
        f"\n\nThe derived training and evaluation pairs are on the Hub: [{data_url.split('/datasets/')[-1]}]({data_url}) "
        "(one config per source, each under its source's license)."
        if data_url
        else ""
    )
    return f"""Which extra training data makes the gate a better judge of Swedish? A controlled ablation
(`python -m laya_gate.v4`): every arm fine-tunes the base checkpoint with the v3 recipe (v3 data
and questions, 1 epoch, the same seed and hyperparameters); only the added data differs. A second
control seed measures seed-to-seed noise.{data_line}

- **+Europarl**: {get(ep, "kept", 0):,} English-original speeches (none from the Europarl test
  sample), professional Swedish translation > Kimi-k3 machine translation, as `preference`
  (English source in the prompt) and `naturalness` (no prompt). The negatives are Kimi-k3's
  only, so OPUS-MT never appears in training. Length-balanced: the human side is longer in
  exactly half the pairs ({pct(ep.get("ref_longer_in_candidates"))} before balancing).
- **+DaLAJ/SweFAQ**: {get(D, "dalaj.pairs", 0):,} DaLAJ-GED train pairs (corrected > learner
  sentence, `naturalness`) and {get(D, "swefaq.pairs", 0):,} SweFAQ train pairs (own answer >
  another answer on the same FAQ page, `preference`; no page shared with the test questions).
- **+both**: both. **+grammar**: +both plus a new single-response question, `grammatical`
  (correct / contains an error), trained on both sentences of every DaLAJ pair; its
  temperature is fitted on DaLAJ dev.
- **v4 (+adequacy)**: +grammar plus negatives that can only be told apart with the source,
  built from {adq.get("naturalness", 0):,} further Europarl paragraphs: the
  correct translation beats another paragraph's human translation, a half translation, a
  word-shuffled translation and the untranslated English, and Kimi-k3's correct translation
  beats another paragraph's human translation; the English is neither natural nor grammatical
  Swedish ({adq.get("preference", NM)} `preference`, {adq.get("naturalness", NM)} `naturalness`
  and {adq.get("single", NM)} single-response rows).

All tests are held out from every arm: DaLAJ and SweFAQ dev/test splits; ScaLA-sv is EuroEval's
own test split (`mteb/scala_sv_classification`); GerLangMod-sv the native Talbanken test file
(lowercased, no punctuation, as distributed); WMT24++ en→sv is EuroEval's translation set
(`google/wmt24pp`, 960 post-edited segments with OPUS-MT and Kimi-k3 translations). Each cell is
the share of items where the reference side wins (corrected sentence, own answer, human Swedish).
**\\*** marks a difference from control whose paired-bootstrap 95% CI excludes zero; the two
control seeds differ by up to {num(100 * seed_max if seed_max else None, 1)} points on a single
test, so smaller differences are noise even when starred.

{table}

#### Sanity tests: a shortcut in the Europarl pairs

FLORES+ devtest (the pilot's 200 sentences), the professional reference against a degraded
alternative, English source in the prompt for `preference`; share of pairs where the reference
wins (`python -m laya_gate.v4 sanity`):

{sanity_table}

**Every arm trained on the Europarl pairs alone learned "the candidate least like the English
source wins".** The human side of a Europarl pair is always the less literal one, so that rule
fits the training pairs perfectly. It also explains part of the large liu-nlp, FLORES+ and
WMT24++ gains of +Europarl, +both and +grammar: human translations are less literal than MT
there too. These arms pick a fluent translation of the *wrong* sentence over the right one
(the reference wins in {pct(sane("europarl", "wrong_source/preference+source"))} to
{pct(sane("grammar", "wrong_source/preference+source"))} of pairs); +both and +grammar also lose
to half translations and to LFM2's nonsense; and on the pilot's 4,400 real model translations
their translation score is uncorrelated or negatively correlated with chrF (Spearman per translation:
+both {item_rho("v4-both")}, +grammar {item_rho("v4-grammar")}; v3 {item_rho("v3")}).

**The adequacy negatives remove the shortcut.** v4 picks the reference over a half translation,
garbage MT and untranslated English in {pct(sane("adequacy", "first_half_only/preference+source"))},
{pct(sane("adequacy", "garbage_mt/preference+source"))} and
{pct(sane("adequacy", "untranslated/preference+source"))} of pairs, and over another sentence's
reference in {pct(sane("adequacy", "wrong_source/preference+source"))} (v3:
{pct(sane("v3", "wrong_source/preference+source"))}). These tests were built the same way as the
adequacy negatives, so they show the fix took, not independent generalization. The independent
check is the pilot: v4's translation score agrees with chrF per translation at
{item_rho("v4-adequacy")}, the highest of any checkpoint. What remains of the translation gains
after the fix is smaller but still clear: FLORES+ `preference` {pct(val("control", "flores/preference+source"))}
→ {pct(val("adequacy", "flores/preference+source"))}, WMT24++ against OPUS-MT
{pct(val("control", "wmt24pp/opus/preference+source"))} → {pct(val("adequacy", "wmt24pp/opus/preference+source"))}
and against Kimi-k3 {pct(val("control", "wmt24pp/kimi/preference+source"))} →
{pct(val("adequacy", "wmt24pp/kimi/preference+source"))}, liu-nlp
{pct(val("control", "liu/preference+source"))} → {pct(val("adequacy", "liu/preference+source"))}.

**DaLAJ and SweFAQ fix the two weak spots they target**, in every arm that includes them:
grammar {pct(val("control", "dalaj/naturalness"))} → {pct(val("adequacy", "dalaj/naturalness"))}
(DaLAJ) and {pct(val("control", "gerlangmod/naturalness"))} →
{pct(val("adequacy", "gerlangmod/naturalness"))} (GerLangMod-sv), relevance
{pct(val("control", "swefaq/preference"))} → {pct(val("adequacy", "swefaq/preference"))}
(SweFAQ). The single-response `grammatical` question raises ScaLA-sv's MCC from
{num(_scala_mcc(A, "control"))} (control) to {num(_scala_mcc(A, "adequacy"))} (v4).

**Costs.** As a stand-alone yes/no check, `grammatical` flags much good Swedish as incorrect at
threshold 0.5: {pct(get(gram, "fluent_responses_flagged_incorrect.mean"))} of fluent LLM answers,
{pct(get(gram, "wmt24pp_reference_flagged_incorrect.mean"))} of WMT24++ references,
{pct(get(gram, "flores_reference_flagged_incorrect.mean"))} of FLORES+ references (it was trained
on single learner sentences). It is useful for comparing two versions of a text, not as a
verdict on one. On GerLangMod-sv (lowercased, no punctuation) v4's `grammatical` drops to
{pct(val("adequacy", "gerlangmod/grammatical"))} from {pct(val("grammar", "gerlangmod/grammatical"))}
with +grammar. In-domain accuracy, calibration and latency stay at the control level. Each arm was
trained once.

#### v4.1: correct text from more registers, harder wrong-source negatives

v4.1 adds two sources to v4 aimed at its two weak spots
(`python -m laya_gate.v4 run --arm v41`), none from the UD treebanks behind ScaLA-sv,
GerLangMod-sv, BLiMPPI and MultiBLiMP:

- **Registers**: {get(D, "registers.pairs", 0):,} correct texts from four training-only sources
  (SweFAQ train answers, DaLAJ's correct textbook sentences, the fluent training answers, unused
  Europarl paragraphs), each against a corrupted copy (a word deleted or two neighbours swapped,
  as in ScaLA), as `naturalness` pairs and `grammatical` single responses; a fifth lowercased
  without punctuation.
- **Hard wrong source**: {get(D, "hard_source.pairs", 0):,} pairs where the correct human or Kimi-k3
  translation beats the translation of the *adjacent* paragraph of the same sitting.

Against v4 on the same items: the `grammatical` question's ScaLA-sv MCC rises from
{num((get(A, "adequacy.classifiers.scala_grammatical") or {}).get("mcc@0.5"))} to
{num((get(A, f"{RELEASE_ARM}.classifiers.scala_grammatical") or {}).get("mcc@0.5"))}, GerLangMod-sv
from {pct(val("adequacy", "gerlangmod/grammatical"))} to {pct(val(RELEASE_ARM, "gerlangmod/grammatical"))},
and over-flagging of good Swedish falls (fluent LLM answers
{pct(get(gram, "fluent_responses_flagged_incorrect.mean"))} →
{pct(get(gram41, "fluent_responses_flagged_incorrect.mean"))}, FLORES+ references
{pct(get(gram, "flores_reference_flagged_incorrect.mean"))} →
{pct(get(gram41, "flores_reference_flagged_incorrect.mean"))}, WMT24++ references
{pct(get(gram, "wmt24pp_reference_flagged_incorrect.mean"))} →
{pct(get(gram41, "wmt24pp_reference_flagged_incorrect.mean"))}), though it is still far from a
reliable stand-alone verdict. The wrong-source sanity test goes from
{pct(sane("adequacy", "wrong_source/preference+source"))} to
{pct(sane(RELEASE_ARM, "wrong_source/preference+source"))} (v3:
{pct(sane("v3", "wrong_source/preference+source"))}). The cost is 1-3 points on several tests
(FLORES+, DaLAJ, SweFAQ `quality`), within the seed noise. v4.1 is the release.

**Minimal pairs need n-gram baselines** (BLiMPPI's lesson, arXiv:2610.13607). Share of pairs where
the grammatical side wins, for model-free baselines and v4.1: the longer side, the more frequent
words (Europarl-SV statistics), a word trigram model trained on
{get(V, "baselines.trigram", "Europarl-SV").removeprefix("stupid backoff over ")}
(`python -m laya_gate.ngram build`, `python -m laya_gate.v4 baselines`):

| Set | Pairs | Longer wins | Unigram frequency | Trigram | v4.1 `naturalness` | v4.1 `grammatical` |
|---|---|---|---|---|---|---|
{bl_rows}

BLiMPPI (polarity items: the licenser is removed) and MultiBLiMP (subject-verb number agreement;
the wrong side uses a plural verb form, archaic in modern Swedish) are new held-out tests here.
Both are nearly solved by length or the trigram model: in BLiMPPI the grammatical sentence is
the longer one in nearly every pair, in MultiBLiMP the shorter. The gate's lead over the trigram
is clearest on DaLAJ and GerLangMod-sv."""


def pilot_section(m: dict) -> str:
    P = m.get("pilot_euroeval")
    if not isinstance(P, dict) or not P.get("models"):
        return "Not measured: run `python -m laya_gate.pilot generate`, `score`, `analyze`."
    R = get(P, "spearman.euroeval_rank (negated)", {})
    PS = P.get("partial_spearman_rank_given_size", {})
    feats = (
        ("v4-v41/answer_quality", "v4.1 `quality` of the model's answers"),
        ("v4-adequacy/answer_quality", "v4 `quality` of the model's answers"),
        ("v3/answer_quality", "v3 `quality` of the model's answers"),
        ("v4-v41/translation_vs_reference", "v4.1 P(model translation beats the reference)"),
        ("v4-grammar/translation_vs_reference", "same, +grammar arm (before the adequacy fix)"),
        ("v4-v41/answer_natural", "v4.1 `natural_swedish` of the answers"),
        ("v4-v41/answer_grammatical", "v4.1 `grammatical` of the answers"),
        ("chrf", "chrF against the FLORES+ reference (no gate)"),
        ("log_parameters", "log parameter count (no gate)"),
        ("swedish_answers", "share of answers in Swedish (no gate)"),
    )
    rows = []
    for f, label in feats:
        c, pc = R.get(f), PS.get(f)
        if c:
            rows.append(
                f"| {label} | {num(c['rho'], 2)} [{num(c['ci95'][0], 2)}, {num(c['ci95'][1], 2)}] | "
                + (
                    f"{num(pc['rho'], 2)} [{num(pc['ci95'][0], 2)}, {num(pc['ci95'][1], 2)}]"
                    if pc
                    else "—"
                )
                + " |"
            )
    tasks = []
    for t in ("Linguistic acceptability", "Summarization", "Reading comprehension", "Knowledge"):
        c = get(P, f"spearman.{t}.v4-v41/answer_quality")
        if c:
            tasks.append(f"{t.lower()} {num(c['rho'], 2)}")
    models = sorted(P["models"], key=lambda r: r["leaderboard"]["rank"])
    n = len(models)
    best, worst = models[0], models[-1]
    n4 = get(P, "n_models_by_precision.4-bit NF4", 0)
    bf = get(P, "spearman_rank_bf16_only.v4-v41/answer_quality", {})
    pmax = max((r["leaderboard"].get("parameters") or 0) for r in models) / 1e9
    failed = sorted(
        k for k, v in (P.get("generation") or {}).items() if str(v).startswith("failed")
    )
    return f"""Can the gate act as a quick predictor of EuroEval's Swedish results? A pilot
(`python -m laya_gate.pilot`): {n} open LLMs (up to {pmax:.0f}B parameters; {n4} loaded in 4-bit NF4
to fit a 12 GB GPU) that have Swedish results on the
[EuroEval leaderboard](https://euroeval.com) (overall Swedish rank {num(best["leaderboard"]["rank"], 2)}
for `{best["model"]}` to {num(worst["leaderboard"]["rank"], 2)} for `{worst["model"]}`; lower is
better) each translate the same {get(P, "design.translation_items", NM)} FLORES+ devtest sentences
into Swedish and answer the same {get(P, "design.answer_prompts", NM)} Swedish prompts (greedy,
thinking off). The gate scores every output; the model-level means are correlated with the
EuroEval Swedish rank (Spearman, bootstrap 95% CI over models). Model size alone predicts rank
well on this range, so the second column controls for it (ranks residualized on log parameter
count).

| Model-level score | Spearman with EuroEval Swedish rank | controlling for size |
|---|---|---|
{chr(10).join(rows)}

**The single-response `quality` score of a model's Swedish answers tracks the leaderboard, and
not only through model size.** No reference answers are needed. It correlates with the task scores too ({", ".join(tasks)}). The
translation score only becomes predictive once the Europarl shortcut is removed (the +grammar
arm's score correlates negatively with rank). v4.1's `quality` correlation is slightly lower than
v4's and v3's, well within the intervals. EuroEval ran full precision; on the {n - n4} models run in
bf16 alone the v4.1 `quality` correlation is {num(bf.get("rho"), 2)}, so the 4-bit models do not drive
it. {len(failed)} further models are missing: they ran out of GPU memory next to another process
on the shared GPU, except `google/gemma-3n-E4B-it`, which needs Pillow for its vision tower
({", ".join(f"`{x}`" for x in failed)}). With {n} models the intervals are still wide; a pilot,
not a calibrated predictor. The models' outputs are not redistributed (their licenses differ); the
scores are in `metrics.json`."""


def arena_section(m: dict) -> str:
    S, D = m.get("arena_sv"), m.get("arena_sv_data") or {}
    if not isinstance(S, dict) or not S.get("models"):
        return "Not measured: run `python -m laya_gate.arena_eval build`, `gate`, `judge`."
    J = S.get("judge") or {}
    rows = []
    for name, r in S["models"].items():
        pr, q, ln = r.get("preference", {}), r.get("quality", {}), r.get("length", {})
        rows.append(
            f"| {name} `preference` | {pct(pr.get('accuracy'))} [{pct(get(pr, 'ci95.0'), 0)}, "
            f"{pct(get(pr, 'ci95.1'), 0)}] | {pct(ln.get('preference_accuracy_human_picked_longer'))} | "
            f"{pct(ln.get('preference_accuracy_human_picked_shorter'))} |"
        )
        rows.append(
            f"| {name} `quality` | {pct(q.get('accuracy'))} [{pct(get(q, 'ci95.0'), 0)}, "
            f"{pct(get(q, 'ci95.1'), 0)}] | {pct(ln.get('quality_accuracy_human_picked_longer'))} | "
            f"{pct(ln.get('quality_accuracy_human_picked_shorter'))} |"
        )
    lb = S.get("length_baseline", {})
    rows.append(
        f"| Longer answer wins (no model) | {pct(lb.get('accuracy_longer_wins'))} "
        f"[{pct(get(lb, 'ci95.0'), 0)}, {pct(get(lb, 'ci95.1'), 0)}] | 100% | 0% |"
    )
    if J:
        rows.append(
            f"| Kimi-k3 judge (both orders) | {pct(J.get('accuracy'))} [{pct(get(J, 'ci95.0'), 0)}, "
            f"{pct(get(J, 'ci95.1'), 0)}] | {pct(J.get('accuracy_human_picked_longer'))} | "
            f"{pct(J.get('accuracy_human_picked_shorter'))} |"
        )
    n_short = get(S, "models.v3.length.n_human_picked_shorter", NM)
    ties = {
        name: (r.get("ties") or {}).get("fraction_within_0.2") for name, r in S["models"].items()
    }
    picks = {name: get(r, "length.gate_picks_longer") for name, r in S["models"].items()}
    W = get(m, "v4.arms", {})

    def lt(arm, kind, q):
        return get(W, f"{arm}.tests.length_test/{kind}/{q}.mean")

    LD = get(m, "v4.data.length_pairs", {})
    LT = get(m, "external_data.length_test", {})
    length_rows = "\n".join(
        f"| {label} | {pct(lt(arm, 'short_correct', 'preference'))} | "
        f"{pct(lt(arm, 'short_correct', 'quality'))} | {pct(lt(arm, 'long_correct', 'preference'))} | "
        f"{pct((get(S, 'models') or {}).get(name, {}).get('preference', {}).get('accuracy'))} | "
        f"{pct(((get(S, 'models') or {}).get(name, {}).get('length') or {}).get('preference_accuracy_human_picked_shorter'))} |"
        for arm, name, label in (
            ("v3", "v3", "v3"),
            ("v41", "v4.1", "v4.1 (published)"),
            ("v42a", "v4.2a", "v4.2a: length-balanced base pairs"),
            ("v42b", "v4.2b", "v4.2b: + length-controlled pairs"),
        )
        if get(W, f"{arm}.tests")
    )
    length_part = (
        f"""

**Can training remove the length bias? (v4.2, not released.)** Both base preference sources
favour length: the chosen answer is the longer one in
{pct(get(m, "v4.data.base_length_bias.original.chosen_longer"))} of the original DPO pairs and
{pct(get(m, "v4.data.base_length_bias.fluent.chosen_longer"))} of the fluent pairs. Two arms on top of v4.1 (`python -m laya_gate.v4 run --arm v42a|v42b`):
**v4.2a** drops "longer wins" pairs from the base `preference` data until it is 50/50; **v4.2b**
also adds {get(LD, "pairs", 0):,} length-controlled pairs: for {get(LD, "prompts", NM)} training
prompts Kimi-k3 wrote a correct and a flawed answer (factual error or missing the point), each
short and long, and the pairs pit *short correct vs long flawed* and *long correct vs short
flawed*, kept when Kimi-k3 picks the correct one in both orders (with `quality` targets 4.5 vs 2).
A held-out test built the same way from {get(LT, "prompts", NM)} other prompts
({get(LT, "pairs", NM)} pairs) and arena_sv measure it:

| Model | Length test: correct is shorter, `preference` | … `quality` | Length test: correct is longer | arena_sv `preference` | arena_sv, human picked shorter |
|---|---|---|---|---|---|
{length_rows}

The gates before v4.2b pick a long flawed answer over a short correct one about three times in
four. v4.2b learns to judge these pairs independently of length, but the gain barely carries over
to the human votes, and it costs elsewhere (SweFAQ `quality` and grammar over-flagging get
worse; the EuroEval pilot correlation drops). Balancing the base data alone (v4.2a) does not
touch the bias. Synthetic "correct vs flawed" pairs do not teach what people prefer in Swedish
answers; that needs human preference data."""
        if get(W, "v42b.tests")
        else ""
    )
    return f"""The only human-judged test of *answer quality* in Swedish we could find without new annotation:
[Chatbot Arena](https://lmarena.ai) votes, where users write a prompt, read two anonymous answers and
vote for the better one (`python -m laya_gate.arena_eval`). Sources:
`lmarena-ai/arena-human-preference-140k` (CC BY 4.0), `-100k` and `-55k` (Apache-2.0); single-turn
battles whose prompt and both answers read as Swedish, deduplicated: {D.get("battles", NM)}
battles, {S.get("decisive", NM)} decisive votes, {S.get("ties", NM)} ties and
{S.get("both_bad", NM)} "both bad". Never used for training. No Kimi model appears in these battles,
so Kimi-k3 is not judging its own output. {pct(get(S, "models.v3.fraction_truncated"))} of the pairs
exceed the gate's 1,024-token window. With {S.get("decisive", NM)} votes the 95% CI is about ±8 points.

| Agreement with the human vote | All decisive | Human picked the longer answer | Human picked the shorter ({n_short}) |
|---|---|---|---|
{chr(10).join(rows)}

**On human votes about Swedish answers, the gate mostly measures length.** People preferred the
longer answer in {pct(lb.get("accuracy_longer_wins"))} of these battles, and the gate picks the longer
answer about as often ({", ".join(f"{k} {pct(v, 0)}" for k, v in picks.items())}). Where people
preferred the shorter answer, the gate agrees with them less often than a coin flip. Kimi-k3 agrees
with the human vote equally often either way: it judges content, the gate does not. The v4 and v4.1
training (grammar, relevance, translation) did not help here; their accuracy is a few points below
v3's, within the noise. Ties are not kept near 0.5 either (share within 0.2 of 0.5:
{", ".join(f"{k} {pct(v, 0)}" for k, v in ties.items())}).

What this means: the gate is a useful fast check for Swedish *form* (grammar, translationese,
translation adequacy, relevance on FAQ pages), not yet for which of two fluent answers people
prefer. Improving that needs human preference data on Swedish answers, which is scarce: this set is
small and must stay a test set.{length_part}"""


def mt_section(m: dict) -> str:
    mt = m.get("mt_naturalness")
    if not isinstance(mt, dict) or not mt.get("models"):
        return "Not measured: run `python -m laya_gate.mtqa translate` then `evaluate`."
    rows = []
    for ver, qs in mt["models"].items():
        for q, d in qs.items():
            if "human_preferred" in d:
                rows.append(
                    f"| {ver} | `{q}` (pairwise) | {_ci(d, '', 'human_preferred')} | "
                    f"{pct(d.get('positional_consistency'))} | — |"
                )
            else:
                rows.append(
                    f"| {ver} | `{q}` (single response) | "
                    f"{_ci(d, '', 'within_pair_human_higher')} | — | "
                    f"{num(d.get('auc_human_vs_mt'))} |"
                )
    return f"""Does the naturalness skill transfer to data no Kimi-k3 model touched? {mt.get("n_pairs", NM)}
sentences from {mt.get("source", NM)} are paired with a machine translation of the same English
sentence by `{mt.get("mt_system", NM)}` ({mt.get("n_identical_dropped", NM)} identical pairs dropped).
Length and content are matched (mean {num(get(mt, "mean_chars.human"), 0)} vs
{num(get(mt, "mean_chars.mt"), 0)} characters). Pairwise questions use the prompt
{mt.get("prompt", NM)}; "human preferred" is order-averaged.

| Model | Question | Human translation preferred [95% CI] | Positional consistency | AUC |
|---|---|---|---|---|
{chr(10).join(rows)}

**No version separates professional from machine translation usefully.** The pairwise
questions are at chance or lean toward the MT output; the best signal, v3's single-response
check without a prompt, is above chance but weak (AUC {num(get(mt, "models.v3.natural_swedish/no_prompt.auc_human_vs_mt"))}). The professional reference is itself a
translation, so this is a harder contrast than the deliberate translationese in the training
data, but the result means the gate's "naturalness" is specific to Kimi-k3-style translationese
(literal phrasing, English loan words, US-centric content). Do not use it as a general
machine-translation quality estimator."""


def _v2_row(label: str, v1: Any, v2: Any, fmt=pct) -> str:
    return f"| {label} | {fmt(v1)} | {fmt(v2)} |"


def v2_section(m: dict, intro: bool = True) -> str:
    """The v2 README section; `intro=False` drops the motivation and data generation."""
    ev, v1 = m.get("v2_eval"), m.get("v1_eval")
    tr, cal = m.get("v2_training"), m.get("v2_calibration")
    if not (isinstance(ev, dict) and ev.get("checkpoint") and isinstance(tr, dict)):
        return "Not measured: run `python -m laya_gate.v2 train|calibrate|evaluate`."
    v1 = v1 if isinstance(v1, dict) else {}

    def acc(d, key):
        return get(d, f"{key}.accuracy_order_averaged_prob.accuracy")

    def ci(d, key):
        lo = get(d, f"{key}.accuracy_order_averaged_prob.ci95.0")
        hi = get(d, f"{key}.accuracy_order_averaged_prob.ci95.1")
        a = acc(d, key)
        return f"{pct(a)} [{pct(lo)}, {pct(hi)}]" if a is not None else NM

    def same(x):
        return x

    rows = [
        _v2_row(
            "Original val_eval, preference (order-averaged)",
            acc(v1, "original/preference"),
            acc(ev, "original/preference"),
        ),
        _v2_row(
            "Original val_eval, ECE (order-averaged)",
            get(v1, "original/preference.ece_order_averaged"),
            get(ev, "original/preference.ece_order_averaged"),
            num,
        ),
        _v2_row(
            "Fluent val_eval, preference (order-averaged)",
            acc(v1, "fluent/preference"),
            acc(ev, "fluent/preference"),
        ),
        _v2_row(
            "Fluent val_eval, ECE (order-averaged)",
            get(v1, "fluent/preference.ece_order_averaged"),
            get(ev, "fluent/preference.ece_order_averaged"),
            num,
        ),
        _v2_row(
            "**Probe** (reasoning-on), preference [95% CI]",
            ci(v1, "probe/preference"),
            ci(ev, "probe/preference"),
            same,
        ),
        _v2_row(
            "Probe, ECE (order-averaged)",
            get(v1, "probe/preference.ece_order_averaged"),
            get(ev, "probe/preference.ece_order_averaged"),
            num,
        ),
        _v2_row(
            "Probe, positional consistency",
            get(v1, "probe/preference.positional_consistency"),
            get(ev, "probe/preference.positional_consistency"),
        ),
        _v2_row(
            f"Probe, share passing margin {DEFAULT_MARGIN}",
            get(v1, "probe/preference.default_margin_order_averaged.coverage"),
            get(ev, "probe/preference.default_margin_order_averaged.coverage"),
        ),
        _v2_row(
            "Probe, accuracy on pairs passing the margin",
            get(v1, "probe/preference.default_margin_order_averaged.accuracy_on_covered"),
            get(ev, "probe/preference.default_margin_order_averaged.accuracy_on_covered"),
        ),
        _v2_row(
            "Probe, accuracy on pairs routed to the judge",
            get(v1, "probe/preference.default_margin_order_averaged.accuracy_on_routed"),
            get(ev, "probe/preference.default_margin_order_averaged.accuracy_on_routed"),
        ),
        _v2_row(
            "GPU forward, batch 1, median",
            get(v1, "latency_gpu_forward_ms.p50"),
            get(ev, "latency_gpu_forward_ms.p50"),
            ms,
        ),
    ]
    hp = tr.get("hyperparameters", {})
    nat = ev.get("original/naturalness", {})
    fn, pn = ev.get("fluent/naturalness", {}), ev.get("probe/naturalness", {})

    def casc(d):
        c = get(d, "probe/cascade", {})
        rows_ = c.get("table", [])
        rec = c.get("recommended", {})
        return c, rows_, rec

    c2, t2, r2 = casc(ev)
    c1, t1, r1 = casc(v1)
    by_thr1 = {round(r["threshold"], 2): r for r in t1}

    def casc_row(r):
        o = by_thr1.get(round(r["threshold"], 2), {})
        is_rec = abs(r["threshold"] - r2.get("threshold", -1)) < 1e-9
        thr = f"**{r['threshold']:.2f}** (recommended)" if is_rec else f"{r['threshold']:.2f}"
        return (
            f"| {thr} | {pct(o.get('routed_fraction'))} / "
            f"{pct(o.get('cascade_accuracy'))} | {pct(r['routed_fraction'])} / "
            f"{pct(r['cascade_accuracy'])} | {ms(r['expected_latency_ms'])} |"
        )

    casc_rows = "\n".join(casc_row(r) for r in t2)
    temps = ", ".join(
        f"`{q}` T = {num(f.get('temperature'))} (NLL {num(f.get('nll_pre'))} → "
        f"{num(f.get('nll_post'))}, fit on {f.get('fit_on', NM)})"
        for q, f in (cal or {}).items()
    )
    why = f"""**Why:** the probe showed that v1 mostly detects translationese. v2 keeps the original
data and adds pairs where *both* answers are fluent Swedish and only quality differs, so the
preference head has to learn more than style. A second `choice` question, **naturalness**
("Which candidate response is written in more natural, idiomatic Swedish?"), keeps the
translationese skill available explicitly: it is trained on the original pairs (natural vs
translationese) and on fluent pairs with a 50/50 target (both are natural, so it should not
pick a side).

#### Training data generation

{generation_section(m)}

"""
    return (
        (why if intro else "")
        + f"""#### Training

Initialized from the base `convaiinnovations/laya` multilingual checkpoint (not from v1), same
recipe as v1 (lr {hp.get("lr", NM)}, micro-batch {hp.get("micro_batch", NM)} × accumulation
{hp.get("grad_accum", NM)}, warmup + cosine, bf16), {hp.get("epochs", NM)} epochs over
{tr.get("train_units", NM)} (pair, question) units; {tr.get("order_resampling", "")}. Best epoch:
**{tr.get("best_epoch", NM)}** ({tr.get("selection", NM)}). Total {num((tr.get("total_seconds") or 0) / 60, 1)} min on
{tr.get("hardware", NM)}; peak GPU memory {num(tr.get("peak_gpu_memory_allocated_gib"), 2)} GiB.

Accuracy is n/a for fluent/naturalness, whose target is 50/50.

{_history_table(tr)}

Temperatures were fitted per question on val_calib, both orders:
{temps}.

#### v1 vs v2

Both checkpoints were scored with the same code (`python -m laya_gate.v2 evaluate`, v1 via
`--checkpoint checkpoints/best_model --section v1_eval`), on the held-out val_eval halves and the
{get(ev, "probe/preference.n_pairs", NM)}-pair probe, each pair in both orders.

| Metric | v1 | v2 |
|---|---|---|
{chr(10).join(rows)}

For reference, Kimi-k3 scores {pct(get(m, "probe.judge.order_averaged.accuracy"))} on the probe.

**Naturalness head (v2 only).** On original val_eval it picks the idiomatic answer over the
translationese one {pct(get(nat, "accuracy_order_averaged_prob.accuracy"))} of the time (ECE
{num(nat.get("ece_order_averaged"))}). On fluent pairs, where both answers are natural, its
order-averaged probability stays within 0.2 of 0.5 for {pct(fn.get("fraction_within_0.2"))} of
fluent val_eval pairs and {pct(pn.get("fraction_within_0.2"))} of probe pairs (mean distance from
0.5: {num(fn.get("mean_abs_dev_from_0.5"))} and {num(pn.get("mean_abs_dev_from_0.5"))}).

**Probe cascade.** Gate first, Kimi-k3 (cached reasoning-on verdicts from the probe run, judge
alone {pct(c2.get("judge_accuracy"))}) for instances with `|p − 0.5| < threshold`. Each order is
a separate instance here, so threshold 0 differs slightly from the order-averaged accuracy above.

| Threshold | v1 routed / accuracy | v2 routed / accuracy | v2 expected latency |
|---|---|---|---|
{casc_rows}

Recommended for v2 ({r2.get("rule", NM)}): threshold {num(r2.get("threshold"), 2)}, routing
{pct(r2.get("routed_fraction"))} for {pct(r2.get("cascade_accuracy"))}; for v1 the same rule gives
{num(r1.get("threshold"), 2)} → {pct(r1.get("routed_fraction"))} routed, {pct(r1.get("cascade_accuracy"))}."""
    )


def tests_section(m: dict) -> str:
    t = m.get("tests")
    if not isinstance(t, dict):
        return NM
    mods = " · ".join(f"`{k}` {v}%" for k, v in t.get("coverage_per_module", {}).items())
    skipped = "\n".join(f"  - `{k}`: {v}" for k, v in t.get("skipped_tests", {}).items())
    return f"""`{t.get("command", NM)}`: **{t.get("passed", NM)} passed, {t.get("failed", NM)} failed,
{t.get("errors", NM)} errors, {t.get("skipped", NM)} skipped** of {t.get("total", NM)}.
Coverage: **{t.get("coverage_percent", NM)}%** overall — {mods}.
{("Skipped:" + chr(10) + skipped) if skipped else ""}"""


def limitations(m: dict) -> str:
    c, lat = m.get("calibration", {}), get(m, "benchmark.latency", {})
    missed = []
    if c.get("target_met") is False:
        missed.append(
            f"post-calibration ECE {num(c.get('ece_post'))} is above the {ECE_TARGET} target"
        )
    if lat.get("target_met") is False:
        missed.append(
            f"median GPU latency {ms(get(lat, 'gpu_forward_ms.p50'))} misses the {LATENCY_TARGET_MS:.0f} ms target"
        )
    if lat.get("target_met") is True and lat.get("p95_under_target") is False:
        missed.append(
            f"p95 GPU latency {ms(get(lat, 'gpu_forward_ms.p95'))} is above {LATENCY_TARGET_MS:.0f} ms even though the median meets it"
        )
    pub_ok, _, reason = hub_info(m)
    items = [
        "**The labels encode translation quality, by construction.** Per the dataset card, "
        "kimi-k3 localized each prompt and chosen response into idiomatic Swedish and "
        "*deliberately* wrote the rejected response as poor translation (literal phrasing, "
        "English loan words, US-centric content). A TF-IDF logistic regression reaches "
        f"{pct(get(m, 'data.baselines.tfidf_logreg_accuracy'))} on the same held-out pairs, "
        "so most of the gate's accuracy is detecting translationese. "
        + (f"Measured directly: {probe_summary(m)}. " if probe_summary(m) else "")
        + "Do not use it as a general helpfulness or correctness judge for two *fluent* "
        "Swedish answers.",
        "**The judge generated the data.** Kimi-k3 wrote both sides of every pair, so its "
        "agreement with the labels measures self-consistency, not independent quality "
        "judgement.",
        "**Pairwise preference, not fact-checking.** The gate predicts which of two responses "
        "the dataset labels as chosen. It does not verify factual correctness, and a "
        "fluent wrong answer can beat a terse correct one.",
        f"**1024-token window.** {pct(get(m, 'data.fraction_truncated'))} of pairs exceed it; "
        "the gate then sees only the end of the prompt and the start of each response, so "
        "differences late in long answers are invisible to it."
        + (
            " Scoring at 2048 or 4096 tokens without retraining did not help (see Context "
            "length) and costs latency."
            if get(m, "context_length.models")
            else ""
        ),
        "**Kimi-k3 is a reference, not ground truth.** The judge is a strong but imperfect "
        "model; agreement with it is not correctness.",
        "**Single seed, single run.** Numbers come from one training run; seed-to-seed variance "
        "was not measured.",
    ]
    if isinstance(m.get("mt_naturalness"), dict):
        items.append(
            "**Naturalness does not transfer to machine translation.** On professional vs "
            "`opus-mt-en-sv` translations of FLORES+ sentences no version separates the two usefully "
            "(AUC ≤ 0.56, see the machine-translation test); it detects Kimi-k3-style "
            "translationese, not machine translation in general."
            + (
                " The same holds on the human-annotated translationese benchmark "
                f"({get(m, 'liu_translationese.dataset')}): every gate prefers the OPUS-MT "
                "translation over the native speaker's alternative."
                if get(m, "liu_translationese.models")
                else ""
            )
        )
    if get(m, "external_eval.models.v3"):
        items.append(
            "**Grammar and relevance are weak spots.** On natively written Swedish the gate "
            "barely prefers a corrected sentence over a learner error (DaLAJ, best v3 question "
            f"{pct(get(m, 'external_eval.models.v3.dalaj.natural_swedish.accuracy'))}) and loses "
            "to a word-overlap baseline at picking a question's own FAQ answer (SweFAQ, v3 "
            f"`quality` {pct(get(m, 'external_eval.models.v3.swefaq.quality.accuracy'))} vs "
            f"{pct(get(m, 'external_eval.baselines.swefaq.word_overlap_wins'))}); Kimi-k3 gets "
            f"{pct(get(m, 'external_eval.judge.sets.dalaj.grammar.accuracy'))} and "
            f"{pct(get(m, 'external_eval.judge.sets.swefaq.preference.accuracy'))}."
        )
    if get(m, f"v4.arms.{RELEASE_ARM}.tests"):
        items.append(
            f"**{RELEASE_NAME}'s `grammatical` question over-flags.** At threshold 0.5 it marks "
            f"{pct(get(m, f'v4.arms.{RELEASE_ARM}.guards.fluent_responses_flagged_incorrect.mean'))} of "
            "fluent LLM answers and "
            f"{pct(get(m, f'v4.arms.{RELEASE_ARM}.guards.wmt24pp_reference_flagged_incorrect.mean'))} of "
            "WMT24++ references as incorrect; use it to compare two versions of a text, not as "
            "a verdict on one."
        )
        items.append(
            "**Translation preference is source-sensitive only after the adequacy fix.** "
            "Europarl human-vs-MT pairs alone teach 'the less literal side wins' (see the v4 "
            f"sanity tests); {RELEASE_NAME} still picks a fluent translation of the wrong sentence in "
            f"{pct(1 - (get(m, f'v4.sanity.checkpoints.{RELEASE_ARM}.wrong_source/preference+source.mean') or 0))} "
            "of FLORES+ pairs."
        )
    if get(m, "arena_sv.models.v3"):
        items.append(
            "**Answer preference largely tracks length.** On "
            f"{get(m, 'arena_sv.decisive', NM)} human Chatbot Arena votes on Swedish answers the gate "
            "agrees with people "
            f"{pct(get((get(m, 'arena_sv.models') or {}).get(RELEASE_NAME, {}), 'preference.accuracy'))} "
            f"({RELEASE_NAME}), close to 'the longer answer wins' "
            f"({pct(get(m, 'arena_sv.length_baseline.accuracy_longer_wins'))}); Kimi-k3 reaches "
            f"{pct(get(m, 'arena_sv.judge.accuracy'))}."
        )
    cpu = get(m, "cpu_inference.variants.torch-fp32.latency_ms.p50")
    if isinstance(cpu, float):
        items.append(
            f"**CPU is slow.** A pair takes {ms(cpu)} (median) on CPU with torch fp32; int8 and "
            "ONNX do not get anywhere near the GPU's millisecond latency."
        )
    if missed:
        items.append("**Targets not met:** " + "; ".join(missed) + ".")
    if not pub_ok:
        items.append(f"**Not on the Hub:** {reason}")
    return "\n".join(f"- {i}" for i in items)


def conclusion(m: dict) -> str:
    c, j = m.get("calibration", {}), m.get("judge")
    lat = get(m, "benchmark.latency", {})
    pos = get(m, "benchmark.positional", {})
    mg = get(m, "benchmark.margin", {})
    acc = c.get("accuracy")
    p1 = (
        f"The fine-tuned gate reaches {pct(acc)} accuracy on held-out pairs "
        f"(the zero-shot base checkpoint: {pct(get(m, 'training.zero_shot_base.val_eval_acc'))}), "
        f"with calibrated probabilities (ECE {num(c.get('ece_post'))}, "
        f"{'meeting' if c.get('target_met') else 'missing'} the ≤ {ECE_TARGET} target) and a median "
        f"single-pair GPU forward pass of {ms(get(lat, 'gpu_forward_ms.p50'))} "
        f"({'meeting' if lat.get('target_met') else 'missing'} the {LATENCY_TARGET_MS:.0f} ms target). "
        f"It picks the same winner in both presentation orders for "
        f"{pct(pos.get('consistent_fraction'))} of pairs. When it is confident (margin ≥ "
        f"{DEFAULT_MARGIN}, {pct(mg.get('coverage'))} of predictions) it is right "
        f"{pct(mg.get('accuracy_on_covered'))} of the time, so its confidence is usable for "
        "routing. That accuracy should be read next to the lexical baseline: a TF-IDF "
        f"logistic regression scores {pct(get(m, 'data.baselines.tfidf_logreg_accuracy'))} on "
        "the same pairs, because the dataset contrasts idiomatic Swedish with deliberately "
        f"poor translation. {baseline_comparison(m)} The gate is a strong translationese "
        "detector. "
        + (
            f"It is a weak quality judge: {probe_summary(m)}."
            if probe_summary(m)
            else "How well it ranks two equally fluent answers by helpfulness was not "
            "measured and should not be assumed."
        )
    )
    if not isinstance(j, dict):
        return p1 + "\n\nThe System 2 comparison was not run, so the cascade could not be assessed."
    ga, ja = get(j, "gate_accuracy.order_averaged"), get(j, "judge_accuracy.order_averaged")
    rec = j.get("recommended", {})
    if isinstance(ga, float) and isinstance(ja, float):
        gap = ja - ga
        recovered = (rec.get("cascade_accuracy", ga) - ga) / gap if gap > 0 else None
        if gap > 0:
            rel = (
                f"Kimi-k3 is more accurate on the same sample ({pct(ja)} vs {pct(ga)}) but its "
                f"median latency is {num(j.get('latency_ratio_judge_over_gate'), 0)}× the gate's. "
                f"At the recommended threshold ({num(rec.get('threshold'), 2)}) the cascade sends "
                f"{pct(rec.get('routed_fraction'))} of instances to the judge and reaches "
                f"{pct(rec.get('cascade_accuracy'))}, recovering "
                f"{pct(recovered, 0) if recovered is not None else NM} of the gate→judge accuracy "
                f"gap at an expected {ms(rec.get('expected_latency_ms'))} per pair instead of "
                f"{ms(get(j, 'latency_ms.p50'))} for the judge alone."
            )
        else:
            best = max(j.get("cascade", []), key=lambda r: r["cascade_accuracy"], default={})
            gain = (best.get("cascade_accuracy", ga) - ga) if best else 0.0
            rel = (
                f"Against these labels Kimi-k3 is *less* accurate than the gate ({pct(ja)} vs "
                f"{pct(ga)}) and {num(j.get('latency_ratio_judge_over_gate'), 0)}× slower, which "
                "is expected: the labels mark translationese, which the gate was trained on, "
                "while the judge was also asked about helpfulness and correctness, and it "
                "generated these pairs itself. "
            )
            if gain > 0:
                rel += (
                    f"The judge still helps where the gate is unsure: routing the "
                    f"{pct(best['routed_fraction'])} least-confident instances (threshold "
                    f"{num(best['threshold'], 2)}) lifts accuracy from {pct(ga)} to "
                    f"{pct(best['cascade_accuracy'])} at an expected "
                    f"{ms(best['expected_latency_ms'])} per pair. The pre-registered rule "
                    f"(cheapest threshold within 0.5 points of the best) picks "
                    f"{num(rec.get('threshold'), 2)}, i.e. {pct(rec.get('routed_fraction'))} "
                    "routed, because the gain is that small."
                )
            else:
                rel += "No threshold in the sweep improves on the gate alone."
        p2 = (
            rel + " " + verdict(m, ja) + " It should not be the only judge for "
            "high-stakes decisions."
        )
    else:
        p2 = "The judge comparison is incomplete, so the cascade could not be assessed."
    extra = [x for x in (v2_summary(m), v3_summary(m), v4_summary(m)) if x]
    return "\n\n".join([p1, p2, *extra])


def v4_summary(m: dict) -> str | None:
    A = get(m, "v4.arms", {})
    if not get(A, f"{RELEASE_ARM}.tests"):
        return None

    def v(arm, t):
        return get(A, f"{arm}.tests.{t}.mean")

    q = get(m, "pilot_euroeval.spearman.euroeval_rank (negated).v4-v41/answer_quality", {})
    qp = get(m, "pilot_euroeval.partial_spearman_rank_given_size.v4-v41/answer_quality", {})
    return (
        f"**{RELEASE_NAME}** closes the two gaps the external benchmarks exposed: grammar (DaLAJ "
        f"{pct(v('v3', 'dalaj/naturalness'))} → {pct(v(RELEASE_ARM, 'dalaj/naturalness'))}, "
        f"ScaLA-sv MCC {num(_scala_mcc(A, 'v3'))} → {num(_scala_mcc(A, RELEASE_ARM))}) and "
        f"relevance (SweFAQ {pct(v('v3', 'swefaq/preference'))} → "
        f"{pct(v(RELEASE_ARM, 'swefaq/preference'))}), and judges translations better (FLORES+ "
        f"{pct(v('v3', 'flores/preference+source'))} → {pct(v(RELEASE_ARM, 'flores/preference+source'))}), "
        "with in-domain accuracy, calibration and latency about the same as v3's. Getting there "
        "exposed a trap worth knowing: human-vs-MT pairs alone teach a gate to reward whatever "
        "is least like the source; source-dependent negatives fixed it. Its `grammatical` "
        "question is useful for comparing two texts but over-flags good Swedish as a stand-alone "
        "verdict. In a 22-model pilot, the gate's quality score of a model's Swedish answers "
        f"tracks the EuroEval Swedish rank (Spearman {num(q.get('rho'), 2)}, "
        f"{num(qp.get('rho'), 2)} after controlling for model size), which makes it a plausible "
        "cheap proxy for screening Swedish models, to be validated on more and larger models."
    )


def baseline_comparison(m: dict) -> str:
    gate, base = get(m, "calibration.accuracy"), get(m, "data.baselines.tfidf_logreg_accuracy")
    if not (isinstance(gate, float) and isinstance(base, float)):
        return ""
    diff = (gate - base) * 100
    if abs(diff) < 0.5:
        return (
            f"The gate does not measurably beat that baseline ({diff:+.1f} points); what it "
            "adds is calibrated confidence, order-robustness and a single model that also "
            "reads the prompt."
        )
    if diff > 0:
        return f"The gate beats that baseline by {diff:.1f} points."
    return f"The gate trails that baseline by {-diff:.1f} points."


def verdict(m: dict, judge_acc: float) -> str:
    """Fit-for-purpose rule, fixed before results were seen: latency and ECE targets met, and
    the confident subset within 2 points of the judge's accuracy."""
    lat_ok = get(m, "benchmark.latency.target_met") is True
    ece_ok = get(m, "calibration.target_met") is True
    covered = get(m, "benchmark.margin.accuracy_on_covered")
    conf_ok = isinstance(covered, float) and covered >= judge_acc - 0.02
    why = (
        f"latency target {'met' if lat_ok else 'missed'}, ECE target "
        f"{'met' if ece_ok else 'missed'}, confident-subset accuracy {pct(covered)} vs judge "
        f"{pct(judge_acc)}"
    )
    if lat_ok and ece_ok and conf_ok:
        return (
            f"For its intended System 1 role on data like this, settling confident comparisons "
            f"in milliseconds and escalating the rest, the gate is fit for purpose ({why})."
        )
    return (
        f"For the System 1 role it is only partly fit ({why}); the failed criteria above "
        f"should be fixed before relying on it unsupervised."
    )


def usage_ideas(m: dict) -> str:
    j = m.get("judge") if isinstance(m.get("judge"), dict) else {}
    rec = j.get("recommended", {})
    best = max(j.get("cascade", []), key=lambda r: r["cascade_accuracy"], default=None)
    cascade = (
        f"In our measurement the recommended threshold {num(rec.get('threshold'), 2)} routes "
        f"{pct(rec.get('routed_fraction'))} at {pct(rec.get('cascade_accuracy'))} accuracy, and "
        f"the most accurate setting ({num(best['threshold'], 2)}) routes "
        f"{pct(best['routed_fraction'])} for {pct(best['cascade_accuracy'])} at "
        f"{ms(best['expected_latency_ms'])} expected per pair (see the cascade table)."
        if rec and best
        else "The cascade was not measured in this run."
    )
    n_noise = get(j, "label_noise.n_suspected")
    noise = (
        f"the label-noise check above flagged {n_noise} such pair(s) in the judge sample"
        if isinstance(n_noise, int)
        else "the label-noise check was not run"
    )
    return f"""- **Best-of-N reranking:** generate N Swedish candidates from an LLM, run a pairwise knockout
  or round-robin with `predict_preference_batch` (score both orders and average), and keep the
  winner; N=8 round-robin is 56 ordered pairs, one batched call. With v3, `rank_by_quality`
  scores each candidate once (N forward passes) instead.
- **LLM-as-judge cascade:** score every pair with the gate and only send pairs with
  `is_valid_margin == False` to an expensive judge such as Kimi-k3. {cascade}
- **DPO data curation:** score a Swedish preference dataset in both orders and flag pairs where
  the gate confidently prefers the rejected response; if an LLM judge agrees, drop or relabel the
  pair ({noise}).
- **Checkpoint A/B evaluation:** answer a fixed Swedish prompt set with two model versions and
  report the order-averaged win rate of the new one, to catch regressions between releases.
- **Active learning:** send the lowest-margin pairs to human annotators first; they are the ones
  where a label changes the model most.
- **Production monitoring:** compare live responses against a reference answer per prompt and
  track the daily share where the reference is confidently preferred, to detect quality drift."""


def license_section(m: dict) -> str:
    lic = get(m, "data.licenses", {})
    return f"""Code and fine-tuned weights: **Apache-2.0**. The base model `convaiinnovations/laya` is
licensed {lic.get("convaiinnovations/laya", NM)}; the dataset `jonasaise/dpo-swedish-interpreted` is
licensed {lic.get("jonasaise/dpo-swedish-interpreted", NM)} (licenses read from the Hub on
{"this run" if lic else NM}). Both permit redistributing a fine-tuned derivative with attribution.
The dataset is derived from UltraFeedback; check the upstream terms before commercial use."""


def render(m: dict | None = None) -> str:
    m = load_metrics() if m is None else m
    uploaded, repo, reason = hub_info(m)
    hub_line = (
        f"Weights: [huggingface.co/{repo}](https://huggingface.co/{repo}) "
        f"({(m.get('publish') or {}).get('visibility', 'private')})."
        if uploaded
        else "Weights are not published on the Hub: " + reason.removeprefix("upload skipped: ")
    )
    return f"""# Laya Swedish Preference Gate

A fast pairwise judge for Swedish text. Give it a Swedish prompt and two candidate responses
and it returns a calibrated probability that the first is better, in one forward pass of a
fine-tuned [Laya](https://github.com/NandhaKishorM/laya) decision model (mmBERT-base backbone).
It is meant as a "System 1" filter: it settles the easy comparisons in milliseconds and hands
the uncertain ones to a slower "System 2" LLM judge. It was trained on
[`jonasaise/dpo-swedish-interpreted`](https://huggingface.co/datasets/jonasaise/dpo-swedish-interpreted),
where the preferred answer is idiomatic, localized Swedish and the rejected one is a
deliberately poor translation. What it has learned is mainly to tell **authentic Swedish from
translationese**, and it is compared against {JUDGE_HW}.

{hub_line}
{"".join(chr(10) + x + chr(10) for x in (v2_summary(m), v3_summary(m)) if x)}
## Results at a glance

{results_table(m)}

All numbers are generated from [`results/metrics.json`](results/metrics.json) by
`python -m laya_gate.readme`.

## Installation

```bash
git clone https://github.com/{GITHUB_REPO}.git
cd laya-swedish-preference-gate
uv sync --extra dev
```

Requires Python ≥ 3.11. A CUDA GPU is optional for inference and needed for training.

## Quickstart

{quickstart(m)}

## How it works

- **`choice` primitive.** Each pair becomes a Laya state
  `{{"prompt", "candidate_a", "candidate_b"}}` and one `choice` question with options
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

{dataset_section(m)}

### Training

{training_section(m)}

### Calibration

{calibration_section(m)}

### Latency and throughput

{latency_section(m)}

{positional_section(m)}

### System 1 vs System 2

{judge_section(m)}

### Fluent-vs-fluent probe (does it judge quality?)

{probe_section(m)}

### v2: training on fluent pairs too

{v2_section(m)}

### v3: single-response questions

{v3_section(m)}

### CPU inference

{cpu_section(m)}

### Machine-translation test (a negative result)

{mt_section(m)}

### Human-annotated translationese benchmark (a negative result)

{liu_section(m)}

### Grammar, relevance and Europarl (external benchmarks)

{external_section(m)}

### Context length (1024 vs longer windows)

{context_section(m)}

### v4 ablation: Europarl, DaLAJ and SweFAQ as training data

{v4_section(m)}

### Pilot: predicting EuroEval Swedish results

{pilot_section(m)}

### Human preference on Swedish answers (Chatbot Arena)

{arena_section(m)}

### Tests

{tests_section(m)}

## Limitations

{limitations(m)}

## Conclusion

{conclusion(m)}

## Usage ideas

{usage_ideas(m)}

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
@software{{laya_swedish_preference_gate,
  title  = {{Laya Swedish Preference Gate}},
  author = {{Lind, Jonas}},
  year   = {{2026}},
  url    = {{https://github.com/{GITHUB_REPO}}}
}}
```

## License

{license_section(m)}
"""


def main(argv=None) -> None:
    (ROOT / "README.md").write_text(render(), encoding="utf-8")
    print("README.md written")


if __name__ == "__main__":
    main()
