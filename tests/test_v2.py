"""v2 helpers: targets, both-order encoding, metrics bookkeeping, multi-question predictor."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from laya_gate import v2
from laya_gate.predictor import SwedishPreferenceGate
from laya_gate.utils import NATURALNESS_QID, QID, QUESTIONS_V2

ROW = {
    "id": "x",
    "prompt": "Fråga?",
    "candidate_a": "Svar A är längre och bättre.",
    "candidate_b": "Svar B.",
    "label": 1,
}


def test_targets():
    assert v2.targets("original", QID, 0) == [1.0, 0.0]
    assert v2.targets("original", NATURALNESS_QID, 1) == [0.0, 1.0]
    assert v2.targets("fluent", QID, 1) == [0.0, 1.0]
    assert v2.targets("fluent", NATURALNESS_QID, 0) == [0.5, 0.5]


def test_build_units_both_orders(tiny_tokenizer):
    from laya.train import to_internal

    internal = {q: to_internal(q, d) for q, d in QUESTIONS_V2.items()}
    units = v2.build_units(
        tiny_tokenizer,
        {"original": [ROW], "fluent": [ROW]},
        internal,
        room=150,
        max_len=256,
        head_max_len=64,
    )
    assert len(units) == 4
    u = next(u for u in units if u["source"] == "original" and u["qid"] == QID)
    fwd, rev = v2.as_item(u, "fwd"), v2.as_item(u, "rev")
    assert fwd["target"] == [0.0, 1.0] and rev["target"] == [1.0, 0.0]  # flips with the order
    assert fwd["ids"] != rev["ids"]
    pref = next(u for u in units if u["source"] == "original" and u["qid"] == QID)["fwd"][0]
    nat = next(u for u in units if u["source"] == "original" and u["qid"] == NATURALNESS_QID)
    assert not np.array_equal(pref, nat["fwd"][0])  # different question heads


def test_pair_metrics_order_bookkeeping():
    # label 0: chosen is stored A. Perfect model: P(stored A) = 0.9 in both orders.
    m = v2.pair_metrics(np.array([0.9, 0.2]), np.array([0.9, 0.2]), [0, 1])
    assert m["accuracy_both_orders"]["accuracy"] == 1.0
    assert m["positional_consistency"] == 1.0
    # A model that always prefers slot A: right only when chosen is in slot A.
    m = v2.pair_metrics(np.array([0.9, 0.9]), np.array([0.1, 0.1]), [0, 1])
    assert m["accuracy_chosen_in_a"] == 1.0 and m["accuracy_chosen_in_b"] == 0.0
    assert m["positional_consistency"] == 0.0


def test_neutrality():
    n = v2.neutrality(np.array([0.5, 0.9]), np.array([0.5, 0.7]))
    assert n["mean_abs_dev_from_0.5"] == pytest.approx(0.15)
    assert n["fraction_within_0.2"] == 0.5


def test_predictor_with_questions_and_temperatures(tiny_checkpoint, tmp_path):
    ckpt = tmp_path / "v2"
    shutil.copytree(tiny_checkpoint, ckpt)
    (ckpt / "questions.json").write_text(json.dumps(QUESTIONS_V2))
    (ckpt / "temperature.json").write_text(
        json.dumps({"temperature": 2.0, "temperatures": {QID: 2.0, NATURALNESS_QID: 3.0}})
    )
    gate = SwedishPreferenceGate(ckpt, device="cpu")
    assert gate.has_naturalness and gate.temperatures == {QID: 2.0, NATURALNESS_QID: 3.0}
    r = gate.predict_naturalness("Fråga?", "Ett svar.", "Ett annat svar.")
    assert set(r) == {"more_natural", "p_a_more_natural", "probabilities"}
    assert sum(r["probabilities"].values()) == pytest.approx(1.0)
    logits = np.array([[1.0, 0.0]])
    assert gate.probabilities(logits, NATURALNESS_QID)[0, 0] < gate.probabilities(logits)[0, 0]
    p = gate.predict_preference("Fråga?", "Ett svar.", "Ett annat svar.")
    assert set(p) == {"preference", "p_chosen", "is_valid_margin", "probabilities"}


def test_v1_checkpoint_has_no_naturalness(tiny_checkpoint):
    gate = SwedishPreferenceGate(tiny_checkpoint, device="cpu")
    assert not gate.has_naturalness
    with pytest.raises(ValueError):
        gate.predict_naturalness("a", "b", "c")
