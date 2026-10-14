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
