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

Also `device="cpu"`, `dtype=torch.float32`, `temperature=<float>` and `margin=<float>` are
accepted. The checkpoint is in the standard Laya layout, so `laya.load(path)` opens it too
(the fitted temperature is stored in its `rl_agent_config.json`)."""


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
        f"An experimental **v2** checkpoint (`python -m laya_gate.v2`, see "
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
        "behind Kimi-k3. v2 is not on the Hub yet."
    )


def _v2_row(label: str, v1: Any, v2: Any, fmt=pct) -> str:
    return f"| {label} | {fmt(v1)} | {fmt(v2)} |"


def v2_section(m: dict) -> str:
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
    hist = tr.get("history", [])
    groups = [k for k in (hist[0]["val_calib"] if hist else {}) if k != "mean_loss"]
    head = " | ".join(f"{g} loss / acc" for g in groups)
    hist_rows = "\n".join(
        f"| {h['epoch']} | {num(h['train_loss'])} | "
        + " | ".join(
            f"{num(h['val_calib'][g]['loss'])} / {pct(h['val_calib'][g]['acc'])}" for g in groups
        )
        + f" | {h['epoch_seconds'] / 60:.1f} min |"
        for h in hist
    )
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
    return f"""**Why:** the probe showed that v1 mostly detects translationese. v2 keeps the original
data and adds pairs where *both* answers are fluent Swedish and only quality differs, so the
preference head has to learn more than style. A second `choice` question, **naturalness**
("Which candidate response is written in more natural, idiomatic Swedish?"), keeps the
translationese skill available explicitly: it is trained on the original pairs (natural vs
translationese) and on fluent pairs with a 50/50 target (both are natural, so it should not
pick a side).

#### Training data generation

{generation_section(m)}

#### Training

Initialized from the base `convaiinnovations/laya` multilingual checkpoint (not from v1), same
recipe as v1 (lr {hp.get("lr", NM)}, micro-batch {hp.get("micro_batch", NM)} × accumulation
{hp.get("grad_accum", NM)}, warmup + cosine, bf16), {hp.get("epochs", NM)} epochs over
{tr.get("train_units", NM)} (pair, question) units; {tr.get("order_resampling", "")}. Best epoch:
**{tr.get("best_epoch", NM)}** ({tr.get("selection", NM)}). Total {num((tr.get("total_seconds") or 0) / 60, 1)} min on
{tr.get("hardware", NM)}; peak GPU memory {num(tr.get("peak_gpu_memory_allocated_gib"), 2)} GiB.

| Epoch | Train loss | {head} | Time |
|---|---|{"---|" * len(groups)}---|
{hist_rows}

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
        "differences late in long answers are invisible to it.",
        "**Kimi-k3 is a reference, not ground truth.** The judge is a strong but imperfect "
        "model; agreement with it is not correctness.",
        "**Single seed, single run.** Numbers come from one training run; seed-to-seed variance "
        "was not measured.",
    ]
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
    v2 = v2_summary(m)
    return p1 + "\n\n" + p2 + (f"\n\n{v2}" if v2 else "")


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
  winner; N=8 round-robin is 56 ordered pairs, one batched call.
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
{(chr(10) + v2_summary(m) + chr(10)) if v2_summary(m) else ""}
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
- **Pipeline.** `./run.sh` runs data → train → calibrate → benchmark → judge → tests →
  readme → publish.

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
