"""Offline tests for the v4 ablation helpers (synthetic rows, no GPU, no network)."""

import random

import numpy as np
import pytest

from laya_gate import v4
from laya_gate.utils import SEED


def test_as_pair_places_reference_in_labelled_slot():
    rng = random.Random(0)
    rows = [v4.as_pair(str(i), "p", "REF", "OTHER", rng) for i in range(200)]
    for r in rows:
        ref_slot = r["candidate_a"] if r["label"] == 0 else r["candidate_b"]
        assert ref_slot == "REF"
    assert 0.4 < np.mean([r["label"] for r in rows]) < 0.6


def test_length_balanced_is_exactly_half_and_drops_ties():
    rng = random.Random(0)
    pairs = [{"ref": "x" * 10, "other": "x" * 5}] * 30 + [{"ref": "x", "other": "xx"}] * 8
    pairs += [{"ref": "xx", "other": "yy"}] * 5  # ties
    kept = v4.length_balanced(pairs, 100, rng)
    assert len(kept) == 16
    assert sum(len(p["ref"]) > len(p["other"]) for p in kept) == 8


def test_europarl_pairs_drop_near_duplicates_and_missing_mt():
    rows = [
        {
            "id": "a",
            "sv": ["Vi måste agera nu för att skydda miljön i hela unionen."],
            "en": ["We must act now."],
            "mt_kimi": "Vi måste agera nu för att skydda miljön i hela unionen!",
        },
        {"id": "b", "sv": ["Det är en lång mening här."], "en": ["Long."], "mt_kimi": "Kort."},
        {
            "id": "c",
            "sv": ["Kort."],
            "en": ["Short."],
            "mt_kimi": "En betydligt längre översättning.",
        },
        {"id": "d", "sv": ["Ingen."], "en": ["None."], "mt_kimi": None},
    ]
    kept, stats = v4.europarl_pairs(rows, random.Random(0))
    assert stats["dropped"] == {"no_mt": 1, "near_duplicate": 1}
    assert stats["candidates"] == 2 and stats["kept"] == 2
    assert stats["ref_longer_in_kept"] == pytest.approx(0.5)
    assert {p["id"] for p in kept} == {"b", "c"}


def test_europarl_train_excludes_eval_and_swedish_originals():
    rows = [{"id": f"e{i}", "original": "EN"} for i in range(10)] + [{"id": "s", "original": "SV"}]
    pool = v4.europarl_train(rows, exclude={"e0", "e1"})
    ids = {r["id"] for r in pool}
    assert ids == {f"e{i}" for i in range(2, 10)}


def test_faq_train_caps_negatives_per_question():
    rows = [
        {"question": "Q1?", "candidate_answers": [f"A{i}" for i in range(10)], "label": 3},
        {"question": "Q2?", "candidate_answers": ["B0", "B1"], "label": 0},
    ]
    pairs = v4.faq_train(rows, random.Random(0))
    by_q = {}
    for p in pairs:
        by_q.setdefault(p["question_id"], []).append(p)
    assert len(by_q["swefaq-train-0"]) == v4.FAQ_NEGATIVES
    assert len(by_q["swefaq-train-1"]) == 1
    assert all(p["ref"] == "A3" for p in by_q["swefaq-train-0"])


def test_tag_and_checkpoint_dir():
    assert v4.tag("control", SEED) == "control"
    assert v4.tag("control", SEED + 1) == f"control-s{SEED + 1}"
    assert v4.checkpoint_dir("both", SEED).name == "v4-both"


def test_paired_diff():
    a, b = np.array([1.0, 1, 1, 0]), np.array([0.0, 1, 1, 0])
    d = v4.paired_diff(a, b)
    assert d["diff"] == pytest.approx(0.25) and d["n"] == 4
    assert d["ci95"][0] <= 0.25 <= d["ci95"][1]
    same = v4.paired_diff(a, a)
    assert same["diff"] == 0 and same["ci95"] == [0.0, 0.0]


def test_arms_and_sources_are_consistent():
    assert set(v4.ARMS) == {"control", "europarl", "native", "both"}
    for sources in v4.ARMS.values():
        assert all(s in v4.SOURCE_FILES for s in sources)
    assert set(v4.ARMS["both"]) == set(v4.ARMS["europarl"]) | set(v4.ARMS["native"])
