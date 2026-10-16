"""Offline tests for the translationese minimal-pair evaluation (synthetic rows, no network)."""

import numpy as np

from laya_gate import liu_eval


def _rows():
    base = {"english_source": "Let's go.", "context": [f"line {i}" for i in range(10)]}
    return [
        {
            **base,
            "id": 0,
            "swedish_human": "Nu går vi",
            "swedish_opus": "Låt oss gå.",
            "swedish_gpt": "Nu går vi!",
            "error_tags": ["DIR", "BET"],
            "gpt_acceptable": True,
        },
        {
            **base,
            "id": 1,
            "swedish_human": "Men varför?",
            "swedish_opus": "Och för vad?",
            "swedish_gpt": "Och för vad?",
            "error_tags": ["DIR"],
            "gpt_acceptable": False,
        },
        {
            **base,
            "id": 2,
            "swedish_human": "Vila lite",
            "swedish_opus": "Ta det lugnt.",
            "swedish_gpt": "Du ska vila",
            "error_tags": ["SEM"],
            "gpt_acceptable": False,
        },
    ]


def test_strip_punct_matches_paper():
    assert liu_eval.strip_punct("Låt oss gå. ") == "Låt oss gå"
    assert liu_eval.strip_punct("Vad?!") == "Vad"
    assert liu_eval.strip_punct("Nu går vi") == "Nu går vi"


def test_make_prompt_setups():
    ctx = [f"line {i}" for i in range(10)]
    assert liu_eval.make_prompt("Hi.", ctx, None) == ""
    assert liu_eval.make_prompt("Hi.", ctx, 0) == "Översätt följande mening till svenska:\nHi."
    p = liu_eval.make_prompt("Hi.", ctx, 2)
    assert "Kontext: line 8 line 9\nMening: Hi." in p and "Översättning" not in p


def test_comparison_masks_drop_identical_and_select_unacceptable():
    m = liu_eval.comparison_masks(_rows())
    assert m["opus"].tolist() == [True, True, True]
    assert m["gpt"].tolist() == [False, True, True]  # "Nu går vi!" == human after stripping
    assert m["gpt_unacceptable"].tolist() == [False, True, True]


def test_summaries_and_tags():
    rows = _rows()
    p = np.array([0.9, 0.2, 0.5])
    s = liu_eval.summarize_pref(p, np.array([True, True, True]))
    assert abs(s["accuracy"] - 1 / 3) < 1e-9 and s["n"] == 3
    tags = liu_eval.tag_breakdown(rows, p, np.array([True, True, True]))
    assert tags["DIR"] == {"n": 2, "opus_preferred": 0.5}
    assert tags["SEM"] == {"n": 1, "opus_preferred": 1.0}  # a 0.5 tie is not a human win


def test_versus_ties():
    out = liu_eval._versus(np.array([1.0, 0.0, 0.5]), np.array([0.0, 1.0, 0.5]))
    assert out.tolist() == [1.0, 0.0, 0.5]


def test_length_baseline():
    out = liu_eval.length_baseline(_rows(), liu_eval.comparison_masks(_rows()))
    # every human version here is shorter than the OPUS one (after stripping punctuation)
    assert out["opus"] == {"accuracy_longer_wins": 0.0, "n": 3}
    # row 0 is dropped for GPT (identical); rows 1-2: human 10 < 11, 9 < 11
    assert out["gpt"] == {"accuracy_longer_wins": 0.0, "n": 2}


def test_judge_scores_orders_and_failures():
    v = {
        (0, "opus", "no_prompt|HA"): "A",  # human in A, picked: win
        (0, "opus", "no_prompt|AH"): "B",  # human in B, picked: win
        (1, "opus", "no_prompt|HA"): "A",
        (1, "opus", "no_prompt|AH"): "A",  # split: position-driven
        (2, "opus", "no_prompt|HA"): None,  # parse failure
        (2, "opus", "no_prompt|AH"): "B",
    }
    score, cons = liu_eval.judge_scores(v, 3, "opus", "no_prompt")
    assert score[0] == 1.0 and score[1] == 0.5 and np.isnan(score[2])
    assert cons[0] == 1.0 and cons[1] == 0.0
