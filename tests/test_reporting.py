"""README rendering, publish gating, metrics store, benchmark and training helpers (offline)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from laya_gate import publish, readme, utils
from laya_gate.predictor import SwedishPreferenceGate

ROWS = [
    {
        "id": "a",
        "prompt": "Hur mår du?",
        "candidate_a": "Bra, tack!",
        "candidate_b": "Ok.",
        "label": 0,
    },
    {
        "id": "b",
        "prompt": "Vad heter Sveriges huvudstad?",
        "candidate_a": "Oslo.",
        "candidate_b": "Stockholm.",
        "label": 1,
    },
    {
        "id": "c",
        "prompt": "Skriv om hösten.",
        "candidate_a": "Löven faller.",
        "candidate_b": "Höst är en årstid som kommer efter sommaren.",
        "label": 1,
    },
]


def synthetic_metrics() -> dict:
    hist = [
        {
            "epoch": 1,
            "train_loss": 0.3,
            "val_calib_loss": 0.1,
            "val_eval_loss": 0.1,
            "val_calib_acc": 0.97,
            "val_eval_acc": 0.97,
            "epoch_seconds": 600.0,
        }
    ]
    return {
        "data": {
            "source": utils.DATASET_REPO,
            "splits": {"train": 10, "val_calib": 2, "val_eval": 2},
            "fraction_truncated": 0.4,
            "baselines": {
                "tfidf_logreg_accuracy": 0.98,
                "words_favouring_chosen": ["gärna"],
                "words_favouring_rejected": ["vänligen"],
            },
            "licenses": {
                "convaiinnovations/laya": "apache-2.0",
                "jonasaise/dpo-swedish-interpreted": "mit",
                "redistribution_allowed": True,
            },
        },
        "training": {
            "history": hist,
            "best_epoch": 1,
            "per_epoch_seconds": [600.0],
            "total_seconds": 600.0,
            "hyperparameters": {"micro_batch": 8},
        },
        "calibration": {
            "accuracy": 0.98,
            "ece_pre": 0.02,
            "ece_post": 0.01,
            "temperature": 2.0,
            "target_met": True,
        },
        "benchmark": {
            "positional": {"consistent_fraction": 0.98, "accuracy_gap_a_minus_b": -0.01},
            "margin": {
                "coverage": 0.99,
                "accuracy_on_covered": 0.985,
                "sweep": [{"margin": 0.2, "coverage": 0.99, "accuracy_on_covered": 0.985}],
            },
            "latency": {
                "gpu_forward_ms": {"p50": 10.0, "p95": 11.0, "n": 200},
                "gpu_end_to_end_ms": {"p50": 12.0},
                "target_met": True,
                "p95_under_target": True,
                "throughput_gpu": [
                    {"batch_size": 8, "pairs_per_second": 120.0},
                    {"batch_size": 32, "pairs_per_second": 100.0},
                ],
            },
        },
        "judge": {
            "model": "kimi-k3",
            "gate_accuracy": {"order_averaged": 0.98},
            "judge_accuracy": {"order_averaged": 0.95},
            "latency_ms": {"p50": 6000.0},
            "cascade": [
                {
                    "threshold": 0.0,
                    "routed_fraction": 0.0,
                    "cascade_accuracy": 0.98,
                    "expected_latency_ms": 12.0,
                }
            ],
            "recommended": {
                "threshold": 0.0,
                "routed_fraction": 0.0,
                "cascade_accuracy": 0.98,
                "expected_latency_ms": 12.0,
                "rule": "r",
            },
            "label_noise": {"n_suspected": 1, "pair_ids": ["x"]},
        },
        "tests": {"all_passed": True, "passed": 10, "coverage_per_module": {"data.py": 90.0}},
    }


# --------------------------------------------------------------------- readme


def test_render_empty_metrics_says_not_measured():
    text = readme.render({})
    assert "not measured" in text
    assert "None" not in text.replace("NoneType", "")
    for heading in (
        "## Results at a glance",
        "## Limitations",
        "## Conclusion",
        "## Usage ideas",
        "## License",
    ):
        assert heading in text


def test_render_full_metrics(monkeypatch):
    monkeypatch.setenv("BASE_URL", "https://secret.internal/v1")
    text = readme.render(synthetic_metrics())
    assert "98.0%" in text and "✅" in text
    assert "secret.internal" not in text
    assert "Batch 32 is slower than batch 8" in text
    assert "fit for purpose" in text  # all three criteria met in the synthetic metrics


def test_verdict_partial_when_a_target_fails():
    m = synthetic_metrics()
    m["calibration"]["target_met"] = False
    assert "only partly fit" in readme.verdict(m, 0.95)


def test_render_mentions_hub_after_upload():
    m = {
        **synthetic_metrics(),
        "publish": {"status": "uploaded", "repo_id": "me/model", "visibility": "private"},
    }
    text = readme.render(m)
    assert "huggingface.co/me/model" in text and 'checkpoint_path="me/model"' in text


def test_formatters():
    assert readme.pct(0.5) == "50.0%" and readme.pct(None) == readme.NM
    assert readme.ms(1.234) == "1.2 ms" and readme.num("x") == readme.NM
    assert readme.get({"a": [{"b": 1}]}, "a.0.b") == 1 and readme.get({}, "a.b", 3) == 3


# -------------------------------------------------------------------- publish


def test_publish_blockers(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    m = synthetic_metrics()
    assert "HF_TOKEN not set" in publish.blockers(m)
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    m["calibration"]["ece_post"] = 0.2
    m["tests"]["all_passed"] = False
    problems = " ".join(publish.blockers(m))
    assert "ECE" in problems and "tests" in problems


def test_model_card_metadata():
    import yaml

    card = publish.model_card(synthetic_metrics(), "me/model")
    meta = yaml.safe_load(card.split("---")[1])
    assert meta["language"] == ["sv"] and meta["license"] == "apache-2.0"
    assert meta["base_model"] == "convaiinnovations/laya"
    assert meta["datasets"] == ["jonasaise/dpo-swedish-interpreted"]
    assert meta["pipeline_tag"] == "text-classification"
    assert {"preference-model", "reward-model", "dpo", "swedish", "laya"} <= set(meta["tags"])
    assert meta["model-index"][0]["results"][0]["metrics"][0]["value"] == 0.98
    assert 'SwedishPreferenceGate("me/model")' in card and 'laya.load("me/model")' in card
    for section in (
        "## How to use",
        "## Intended use",
        "## Training data",
        "## Evaluation",
        "## Limitations",
        "## License",
    ):
        assert section in card


def test_publish_skips_and_records_reason(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    recorded = {}
    monkeypatch.setattr(publish, "load_metrics", lambda: {})
    monkeypatch.setattr(publish, "update_metrics", lambda k, v: recorded.update({k: v}))
    monkeypatch.setattr(publish.readme, "main", lambda: None)
    with pytest.raises(SystemExit) as exc:
        publish.main([])
    assert exc.value.code == 0
    assert recorded["publish"]["status"] == "skipped"
    assert "HF_TOKEN" in recorded["publish"]["reason"]


# ---------------------------------------------------------------------- utils


def test_metrics_store_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(utils, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(utils, "METRICS_PATH", tmp_path / "metrics.json")
    utils.update_metrics("a", {"x": 1})
    utils.update_metrics("b", 2)
    utils.update_metrics("a", {"x": 3})
    assert utils.load_metrics() == {"a": {"x": 3}, "b": 2}
    utils.write_jsonl(tmp_path / "r.jsonl", [{"å": "ä"}])
    assert utils.read_jsonl(tmp_path / "r.jsonl") == [{"å": "ä"}]
    p = utils.percentiles([1, 2, 3, 4])
    assert p["p50"] == 2.5 and p["n"] == 4 and p["max"] == 4


# -------------------------------------------------------------- benchmark + train


@pytest.fixture(scope="module")
def tiny_gate(tiny_checkpoint):
    return SwedishPreferenceGate(tiny_checkpoint, device="cpu")


def test_positional_and_margin(tiny_gate):
    from laya_gate.benchmark import positional_and_margin

    metrics, per_pair = positional_and_margin(tiny_gate, ROWS)
    pos = metrics["positional"]
    assert 0 <= pos["consistent_fraction"] <= 1
    assert pos["accuracy_both_orders"] == pytest.approx(
        (pos["accuracy_chosen_in_a"] + pos["accuracy_chosen_in_b"]) / 2
    )
    assert set(per_pair) == {"a", "b", "c"}
    assert len(metrics["margin"]["sweep"]) == 10
    assert metrics["margin"]["sweep"][0]["coverage"] == 1.0


def test_timing_helpers(tiny_gate):
    from laya_gate import benchmark as B

    items = [tiny_gate.encode(r["prompt"], r["candidate_a"], r["candidate_b"]) for r in ROWS]
    assert len(B.time_forward(tiny_gate, items, runs=3, warmup=1)) == 3
    assert len(B.time_tokenization(tiny_gate, ROWS)) == 3
    assert len(B.time_end_to_end(tiny_gate, ROWS, warmup=1)) == 3
    thr = B.throughput(tiny_gate, items, batch_size=2)
    assert thr["pairs"] == 3 and thr["pairs_per_second"] > 0
    assert len(B.max_length_item(tiny_gate, ROWS)["ids"]) > 0


def test_train_loss_and_eval(tiny_checkpoint, tiny_tokenizer):
    import torch

    from laya_gate import train as T
    from laya_gate.data import build_items
    from laya_gate.predictor import load_model

    model, tok, _ = load_model(tiny_checkpoint)
    model.float()
    items, _ = build_items(tok, ROWS, room=150)
    encoded = T.encode_all(tok, items)
    loss, logits = T.batch_loss(
        model, T._collate_train(encoded, tok.pad_token_id), torch.device("cpu")
    )
    assert logits.shape == (3, 2) and loss.item() > 0
    val_loss, acc = T.evaluate(model, encoded, tok.pad_token_id, torch.device("cpu"))
    assert np.isfinite(val_loss) and 0 <= acc <= 1


def test_calibrate_score_split(tiny_gate):
    from laya_gate.calibrate import score_split

    logits, labels = score_split(tiny_gate, ROWS)
    assert logits.shape == (3, 2) and list(labels) == [0, 1, 1]
    json.dumps(logits.tolist())


def test_probe_section_renders_and_is_optional():
    assert readme.probe_summary({}) is None
    assert "Not measured" in readme.probe_section({})
    m = synthetic_metrics()
    acc = {"accuracy": 0.64, "ci95": [0.59, 0.7], "n": 299}
    m["probe"] = {
        "n_pairs": 299,
        "gate": {
            "order_averaged": acc,
            "ece_order_averaged": 0.23,
            "by_margin": {"1.0-<2.0": acc, ">=2.0": acc},
            "default_margin": {"margin": 0.2, "coverage": 0.84, "accuracy_on_covered": 0.67},
            "by_confidence": [{"confidence": [0.9, 1.0], "fraction": 0.65, "accuracy": 0.7}],
        },
        "judge": {"order_averaged": {"accuracy": 0.87, "ci95": [0.85, 0.9], "n": 598}},
    }
    section = readme.probe_section(m)
    assert "64.0% [59.0%, 70.0%] for gaps 1.0–2.0" in section
    assert "0.9–1.0 | 65.0% | 70.0%" in section
    text = readme.render(m)
    assert "Measured directly: on 299 pairs" in text and "weak quality judge" in text
    assert "**Measured on fluent pairs:**" in publish.model_card(m, "me/model")


def _v2_metrics() -> dict:
    m = synthetic_metrics()
    acc = {"accuracy": 0.8, "ci95": [0.75, 0.85], "n": 299}
    pm = {
        "n_pairs": 299,
        "accuracy_order_averaged_prob": acc,
        "positional_consistency": 0.9,
        "ece_order_averaged": 0.05,
        "default_margin_order_averaged": {
            "margin": 0.2,
            "coverage": 0.6,
            "accuracy_on_covered": 0.9,
            "accuracy_on_routed": 0.65,
        },
    }
    table = [
        {
            "threshold": t,
            "routed_fraction": t,
            "cascade_accuracy": 0.8 + t / 10,
            "expected_latency_ms": 10 + t,
        }
        for t in (0.0, 0.1, 0.2)
    ]
    cascade = {"table": table, "recommended": {**table[1], "rule": "r"}, "judge_accuracy": 0.87}
    ev = {
        "checkpoint": "v2_eval",
        "original/preference": pm,
        "fluent/preference": pm,
        "probe/preference": pm,
        "original/naturalness": pm,
        "fluent/naturalness": {"mean_abs_dev_from_0.5": 0.1, "fraction_within_0.2": 0.85},
        "probe/naturalness": {"mean_abs_dev_from_0.5": 0.12, "fraction_within_0.2": 0.8},
        "probe/cascade": cascade,
        "latency_gpu_forward_ms": {"p50": 10.0, "p95": 11.0},
    }
    m["v2_eval"], m["v1_eval"] = ev, {**ev, "checkpoint": "v1_eval"}
    val = {"original/preference": {"loss": 0.1, "acc": 0.98, "n": 10}, "mean_loss": 0.1}
    m["v2_training"] = {
        "hyperparameters": {"lr": 2e-5, "micro_batch": 8, "grad_accum": 2, "epochs": 1},
        "train_units": 40,
        "order_resampling": "one order per item per epoch",
        "history": [{"epoch": 1, "train_loss": 0.3, "val_calib": val, "epoch_seconds": 60.0}],
        "best_epoch": 1,
        "total_seconds": 60.0,
    }
    m["v2_calibration"] = {"preference": {"temperature": 2.0, "nll_pre": 0.5, "nll_post": 0.4}}
    m["fluent_data"] = {
        "requested": 10,
        "kept": 9,
        "splits": {"train": 7, "val_calib": 1, "val_eval": 1},
        "throughput_runs": [
            {
                "chunk": 0,
                "concurrency": 64,
                "requests_per_second": 1.8,
                "output_tokens_per_second": 1100.0,
                "latency_p50_s": 33.0,
                "failed": 0,
                "parse_failed": 0,
                "wall_seconds": 600.0,
            },
            {
                "chunk": 1,
                "concurrency": 128,
                "requests_per_second": 1.4,
                "output_tokens_per_second": 990.0,
                "latency_p50_s": 50.0,
                "failed": 0,
                "parse_failed": 2,
                "wall_seconds": 700.0,
            },
        ],
    }
    m["fluent_fidelity"] = {
        "n_pairs": 100,
        "judge_accuracy_both_orders": {"reasoning_on": 0.89, "reasoning_off": 0.865},
    }
    return m


def test_v2_section_renders_and_is_optional():
    assert "Not measured" in readme.v2_section({})
    assert "Not measured" in readme.generation_section({})
    m = _v2_metrics()
    section = readme.v2_section(m)
    assert "| **Probe** (reasoning-on), preference [95% CI] | 80.0% [75.0%, 85.0%] |" in section
    assert "**0.10** (recommended)" in section
    assert "concurrency 64: 1.80–1.80 req/s" in section and "concurrency 128" in section
    assert "86.5% of reasoning-off pairs vs 89.0%" in section
    assert "### v2: training on fluent pairs too" in readme.render(m)


def test_pair_metrics_order_averaged_fields():
    from laya_gate import v2

    m = v2.pair_metrics(np.array([0.9, 0.6]), np.array([0.8, 0.5]), [0, 0])
    dm = m["default_margin_order_averaged"]
    assert dm["coverage"] == 0.5 and dm["accuracy_on_covered"] == 1.0
    assert 0.0 <= m["ece_order_averaged"] <= 1.0


def test_model_card_v2_and_blockers(monkeypatch, tmp_path):
    m = _v2_metrics()
    m["v2_eval"]["original/preference"]["ece_both_orders"] = 0.01
    m["v2_eval"]["fluent/preference"]["ece_both_orders"] = 0.2
    m["data"]["licenses"]["redistribution_allowed"] = True
    m["tests"] = {"all_passed": True}
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setattr(publish, "V2_CHECKPOINT_DIR", tmp_path)
    problems = publish.blockers(m, "v2")
    assert any("fluent val_eval ECE" in p for p in problems)
    assert any("source data" in p for p in problems)
    assert any("no trained v2 checkpoint" in p for p in problems)
    card = publish.model_card_v2(m, "me/model")
    assert card.startswith("---\nlanguage: [sv]")
    assert 'revision="v1"' in card and "predict_naturalness" in card
    assert "80.0% [75.0%, 85.0%]" in card


def test_readme_mentions_v2_on_hub():
    m = _v2_metrics()
    m["publish"] = {"status": "uploaded", "repo_id": "me/model", "checkpoint": "v2"}
    text = readme.render(m)
    assert "v2 is the default checkpoint on the Hub" in text
    assert 'SwedishPreferenceGate("me/model", revision="v1")' in text


def _v3_metrics() -> dict:
    m = _v2_metrics()
    pair = {
        "accuracy_order_averaged_prob": {"accuracy": 0.85, "ci95": [0.8, 0.9]},
        "ece_both_orders": 0.02,
        "ece_order_averaged": 0.03,
        "positional_consistency": 0.9,
    }
    m["v3_eval"] = {
        "original/preference": dict(pair),
        "fluent/preference": dict(pair),
        "probe/preference": dict(pair),
        "probe/cascade": {
            "table": [
                {
                    "threshold": 0.0,
                    "routed_fraction": 0.0,
                    "cascade_accuracy": 0.85,
                    "expected_latency_ms": 10.0,
                },
                {
                    "threshold": 0.3,
                    "routed_fraction": 0.4,
                    "cascade_accuracy": 0.9,
                    "expected_latency_ms": 900.0,
                },
            ],
            "recommended": {"threshold": 0.3, "routed_fraction": 0.4, "cascade_accuracy": 0.9},
            "judge_accuracy": 0.88,
        },
        "latency_gpu_forward_ms": {"p50": 10.0, "p95": 11.0},
    }
    m["v3_pointwise"] = {
        "probe/quality": {
            "n_pairs": 299,
            "accuracy_expected_score": {"accuracy": 0.83, "ci95": [0.79, 0.87]},
            "ece_p_chosen_beats_rejected": 0.14,
        },
        "original/natural_swedish": {"auc": 0.99, "accuracy": {"accuracy": 0.93}},
        "latency_gpu_forward_ms_single_response": {"p50": 6.0},
    }
    m["v3_training"] = {"history": [], "hyperparameters": {"lr": 2e-5}, "train_units": 10}
    m["mt_naturalness"] = {
        "n_pairs": 5,
        "models": {
            "v3": {
                "preference": {
                    "human_preferred": {"accuracy": 0.48, "ci95": [0.46, 0.5]},
                    "positional_consistency": 0.57,
                },
                "natural_swedish/no_prompt": {
                    "auc_human_vs_mt": 0.55,
                    "within_pair_human_higher": {"accuracy": 0.58, "ci95": [0.56, 0.6]},
                },
            }
        },
    }
    return m


def test_v3_sections_render_and_are_optional():
    assert "Not measured" in readme.v3_section({})
    assert "Not measured" in readme.cpu_section({})
    assert "Not measured" in readme.mt_section({})
    m = _v3_metrics()
    section = readme.v3_section(m)
    assert (
        "| **Probe**, preference [95% CI] | 80.0% [75.0%, 85.0%] | 85.0% [80.0%, 90.0%] |"
        in section
    )
    assert "**0.30** (recommended)" in section and "83.0% [79.0%, 87.0%]" in section
    mt = readme.mt_section(m)
    assert "| v3 | `preference` (pairwise) | 48.0% [46.0%, 50.0%] | 57.0% | — |" in mt
    assert "AUC 0.550" in mt
    text = readme.render(m)
    assert "### v3: single-response questions" in text and "v3 is not on the Hub yet." in text


def test_model_card_v3_and_blockers(monkeypatch, tmp_path):
    m = _v3_metrics()
    m["v3_eval"]["fluent/preference"]["ece_both_orders"] = 0.2
    m["data"]["licenses"]["redistribution_allowed"] = True
    m["tests"] = {"all_passed": True}
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setattr(publish, "V3_CHECKPOINT_DIR", tmp_path)
    problems = publish.blockers(m, "v3")
    assert any("v3 fluent val_eval ECE" in p for p in problems)
    assert any("no trained v3 checkpoint" in p for p in problems)
    card = publish.model_card_v3(m, "me/model")
    assert card.startswith("---\nlanguage: [sv]")
    assert 'revision="v2"' in card and "rank_by_quality" in card and "onnx-int8" in card
    assert "82.9%" not in card and "83.0% [79.0%, 87.0%]" in card


def test_readme_mentions_v3_on_hub():
    m = _v3_metrics()
    m["publish"] = {"status": "uploaded", "repo_id": "me/model", "checkpoint": "v3"}
    text = readme.render(m)
    assert "v3 is the default checkpoint on the Hub" in text
    assert 'v2 is on the Hub as `revision="v2"`' in text
    assert "rank_by_quality" in text
