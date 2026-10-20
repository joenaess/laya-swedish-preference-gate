"""Offline tests for the arena_sv test set (synthetic rows)."""

import json

import numpy as np

from laya_gate import arena_eval as A

SV = "Det är en bra fråga och jag ska försöka svara så tydligt som möjligt."


def test_label_from_winner():
    assert A.label_from_winner("model_a") == "a"
    assert A.label_from_winner("model_b") == "b"
    assert A.label_from_winner("tie") == "tie"
    assert A.label_from_winner("tie (bothbad)") == "both_bad"
    assert A.label_from_winner("both_bad") == "both_bad"


def test_split_turns_handles_both_content_layouts():
    conv_140k = [
        {"role": "user", "content": [{"type": "text", "text": "Hej"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hej!"}]},
    ]
    conv_100k = [{"role": "user", "content": "Hej"}, {"role": "assistant", "content": "Hej!"}]
    assert A.split_turns(conv_140k) == (["Hej"], ["Hej!"]) == A.split_turns(conv_100k)


def test_battle_keeps_single_turn_only():
    ok = A.battle("s", 1, ["fråga"], ["svar a"], ["svar b"], "a", "m1", "m2")
    assert ok["prompt"] == "fråga" and ok["label"] == "a"
    assert A.battle("s", 2, ["q1", "q2"], ["a1", "a2"], ["b1", "b2"], "a", "m1", "m2") is None
    assert A.battle("s", 3, ["fråga"], ["samma"], ["samma"], "a", "m1", "m2") is None


def test_from_55k_parses_json_turns_and_ties():
    row = {
        "id": 7,
        "model_a": "x",
        "model_b": "y",
        "prompt": json.dumps([SV]),
        "response_a": json.dumps([SV + " A"]),
        "response_b": json.dumps([SV + " B"]),
        "winner_model_a": 0,
        "winner_model_b": 0,
        "winner_tie": 1,
    }
    r = A.from_55k(row)
    assert r["label"] == "tie" and r["response_a"].endswith("A")
    assert A.is_swedish_battle(r)


def test_dedup_key_ignores_answer_order_and_whitespace():
    r1 = {"prompt": "fråga", "response_a": "ett  svar", "response_b": "annat svar"}
    r2 = {"prompt": "fråga ", "response_a": "annat svar", "response_b": "ett svar"}
    assert A.dedup_key(r1) == A.dedup_key(r2)


def test_metrics():
    p = np.array([0.9, 0.2, 0.6, 0.45])
    m = A.decisive_metrics(p, ["a", "b", "b", "b"])
    assert m["accuracy"] == 0.75 and m["confident_coverage"] == 0.5
    assert m["confident_accuracy"] == 1.0
    t = A.tie_metrics(np.array([0.5, 0.8]))
    assert t["fraction_within_0.2"] == 0.5
    rows = [
        {"label": "a", "response_a": "lång " * 5, "response_b": "kort"},
        {"label": "b", "response_a": "lång " * 5, "response_b": "kort"},
    ]
    assert A.length_baseline(rows)["accuracy_longer_wins"] == 0.5
