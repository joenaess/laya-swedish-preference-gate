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
    assert set(v4.ARMS) == {
        "control",
        "europarl",
        "native",
        "both",
        "grammar",
        "adequacy",
        "v41",
        "v42a",
        "v42b",
    }
    for sources in v4.ARMS.values():
        extra = (
            "dalaj_grammar",
            "adequacy",
            "registers",
            "hard_source",
            "length_balance",
            "length_pairs",
        )
        assert all(s in v4.SOURCE_FILES or s in extra for s in sources)
    assert set(v4.ARMS["both"]) == set(v4.ARMS["europarl"]) | set(v4.ARMS["native"])


def test_grammar_rows_label_corrected_as_correct():
    pairs = [
        {"id": "d1", "candidate_a": "Hon är snäll.", "candidate_b": "Hon är snälla.", "label": 0},
        {
            "id": "d2",
            "candidate_a": "Vi har bo här.",
            "candidate_b": "Vi har bott här.",
            "label": 1,
        },
    ]
    rows = v4.grammar_rows(pairs, "dalaj_grammar")
    got = {(r["response"], r["label"]) for r in rows}
    assert got == {
        ("Hon är snäll.", 0),
        ("Hon är snälla.", 1),
        ("Vi har bott här.", 0),
        ("Vi har bo här.", 1),
    }
    assert all(r["qid"] == v4.GRAMMAR_QID and r["prompt"] == "" for r in rows)
    assert all(r["target"] == ([1.0, 0.0] if r["label"] == 0 else [0.0, 1.0]) for r in rows)
    assert set(v4.ARMS["grammar"]) == set(v4.ARMS["both"]) | {"dalaj_grammar"}


def test_adequacy_data_needs_the_source():
    pool = [
        {
            "id": f"p{i}",
            "sv": [f"Detta är mening nummer {i} om miljön och unionen."],
            "en": [f"This is sentence {i} about the environment."],
            "mt_kimi": f"Det här är mening {i} om miljön.",
        }
        for i in range(10)
    ]
    d = v4.adequacy_data(pool, {"p0"}, random.Random(0))
    assert {r["id"].split("-")[0] for r in d["preference"]} <= {f"p{i}" for i in range(1, 10)}
    for r in d["preference"]:
        ref = r["candidate_a"] if r["label"] == 0 else r["candidate_b"]
        pid = r["id"].split("-")[0]
        assert pid.removeprefix("p") in ref  # the winner always translates this paragraph
        assert r["prompt"].endswith(f"sentence {pid.removeprefix('p')} about the environment.")
    assert all(s["label"] == 1 and s["response"].startswith("This is") for s in d["single"])
    kinds = {r["id"].split("-")[-1] for r in d["preference"]}
    assert kinds == {"wrong_source", "mt_vs_wrong_source", "half", "shuffled", "untranslated"}


def test_shuffle_neighbours_keeps_words():
    text = "en två tre fyra fem sex sju åtta"
    out = v4.shuffle_neighbours(text, random.Random(1))
    assert sorted(out.split()) == sorted(text.split()) and out != text


def test_corrupt_and_lower_no_punct():
    rng = random.Random(0)
    text = "Detta innebär att det är kommissionens tolkning."
    bad = v4.corrupt(text, rng)
    assert bad != text and abs(len(bad.split()) - len(text.split())) <= 1
    assert v4.corrupt("Hej då", rng) == "Hej då"  # too short to corrupt
    assert v4.lower_no_punct("Hej, Världen! Det är EU:s tur.") == "hej världen det är eu s tur"


def test_first_sentences_caps_sentences_and_words():
    text = "En. Två. Tre. Fyra."
    assert v4.first_sentences(text, n=2) == "En. Två."
    assert len(v4.first_sentences("ord " * 500).split()) == 120


def test_session_neighbour_stays_in_the_same_sitting():
    ids = ["ep-ep-00-01-17.2", "ep-ep-00-01-17.5", "ep-ep-00-01-17.9", "ep-ep-01-02-03.1"]
    nb = v4.session_neighbour(ids)
    assert nb["ep-ep-00-01-17.2"] == "ep-ep-00-01-17.5"
    assert nb["ep-ep-00-01-17.9"] == "ep-ep-00-01-17.5"
    assert "ep-ep-01-02-03.1" not in nb  # alone in its sitting


def test_registers_data_pairs_correct_with_corrupted():
    texts = {"src": ["Detta är en ganska lång och helt korrekt svensk mening."] * 5}
    d = v4.registers_data(texts, random.Random(3))
    assert len(d["naturalness"]) == 5 and len(d["single"]) == 10
    for r in d["naturalness"]:
        ref = r["candidate_a"] if r["label"] == 0 else r["candidate_b"]
        assert ref.lower().replace(".", "").split() == (
            ["detta", "är", "en", "ganska", "lång", "och", "helt", "korrekt", "svensk", "mening"]
        )
    assert {s["label"] for s in d["single"]} == {0, 1}


def test_parse_length_answers_requires_all_keys_and_crossed_lengths():
    long_a, short_a = "lång " * 40, "kort svar."
    good = {
        "good_long": long_a,
        "good_short": short_a,
        "bad_long": long_a + "x",
        "bad_short": "fel.",
    }
    import json as _json

    text = "<think>plan</think>```json\n" + _json.dumps(good) + "\n```"
    assert v4.parse_length_answers(text)["good_short"] == "kort svar."
    assert v4.parse_length_answers('{"good_long": "a"}') is None
    swapped = {**good, "good_short": long_a + "y"}  # short answer longer than bad_long
    assert v4.parse_length_answers(_json.dumps(swapped)) is None
    assert v4.parse_length_answers(None) is None


def test_balance_rows_makes_chosen_longer_half():
    longer = {"candidate_a": "en lång text", "candidate_b": "kort", "label": 0}
    shorter = {"candidate_a": "kort", "candidate_b": "en lång text", "label": 0}
    tie = {"candidate_a": "abc", "candidate_b": "xyz", "label": 1}
    kept, st = v4.balance_rows([longer] * 7 + [shorter] * 3 + [tie], random.Random(0))
    assert st == {"before": 11, "chosen_longer_before": 7, "chosen_shorter": 3, "after": 6}
    assert sum(v4.chosen_is_longer(r) for r in kept) == 3
    assert v4.chosen_is_longer(tie) is None
