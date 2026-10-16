"""Offline tests for the external benchmarks (synthetic rows, no network, no GPU)."""

import numpy as np
import pytest

from laya_gate import external_eval as X


def _dalaj_row(sentence, label, start=None, stop=None, correction=None, types=""):
    return {
        "sentence": sentence,
        "label": label,
        "meta": {
            "confusion_pair": {"correction": correction, "incorrect_span": None},
            "error_span": {"start": start, "stop": stop},
            "error_label": types,
        },
    }


def test_apply_correction_replaces_and_inserts_with_word_boundaries():
    assert X.apply_correction("Människor är snäll där.", 13, 18, "snälla") == (
        "Människor är snälla där."
    )
    # insertion of a word at the start of the next word keeps a space on both sides
    assert X.apply_correction("Är de verkligen viktigaste?", 16, 16, "det") == (
        "Är de verkligen det viktigaste?"
    )
    # inserted punctuation does not leave a space before it
    assert X.apply_correction("vid deras byrå där", 14, 14, ",") == "vid deras byrå, där"
    assert X.apply_correction("Hej  då .", 8, 9, "!") == "Hej då!"


def test_dalaj_pairs_rebuild_and_drop_unchanged():
    rows = [
        _dalaj_row("Människor är snäll där.", 1, 13, 18, "snälla", "M"),
        _dalaj_row("även Du kan.", 1, 0, 4, "även", "S"),  # word-order fix: not rebuildable
        _dalaj_row("Allt är bra.", 0),  # correct sentences are not pairs on their own
        _dalaj_row("Fel här", "1", None, None, None, "L"),
    ]
    pairs, dropped = X.dalaj_pairs(rows)
    assert len(pairs) == 1
    assert pairs[0]["ref"] == "Människor är snälla där."
    assert pairs[0]["other"] == "Människor är snäll där."
    assert pairs[0]["types"] == ["M"] and pairs[0]["prompt"] == ""
    assert dropped == {"unchanged": 1, "no_span": 1}
    groups = X.dalaj_groups(pairs)
    assert groups["M (morphology)"].tolist() == [True]
    assert groups["S (syntax)"].tolist() == [False]


def test_swefaq_pairs_and_ranking():
    rows = [
        {
            "question": "Hur söker jag?",
            "candidate_answers": ["Du söker här.", "Nej.", "Ja."],
            "label": 0,
        },
        {
            "question": "Vad kostar det?",
            "candidate_answers": ["Gratis.", "Det kostar 100 kr."],
            "label": "1",
        },
    ]
    pairs, questions = X.swefaq_pairs(rows, "test")
    assert [p["id"] for p in pairs] == ["swefaq-test-0-1", "swefaq-test-0-2", "swefaq-test-1-0"]
    assert [p["ref"] for p in pairs] == ["Du söker här.", "Du söker här.", "Det kostar 100 kr."]
    scores = {
        "swefaq-test-0|0": 0.9,
        "swefaq-test-0|1": 0.1,
        "swefaq-test-0|2": 0.2,
        "swefaq-test-1|0": 0.8,
        "swefaq-test-1|1": 0.5,
    }
    r = X.faq_ranking(questions, scores)
    assert r["top1"]["accuracy"] == pytest.approx(0.5)
    assert r["mrr"] == pytest.approx((1 + 0.5) / 2)
    assert r["top1_chance"] == pytest.approx((1 / 3 + 1 / 2) / 2)
    # a tie with the gold answer ranks pessimistically
    tied = {**scores, "swefaq-test-1|0": 0.5}
    assert X.faq_ranking(questions, tied)["top1"]["accuracy"] == pytest.approx(0.5)
    st = X.strict_top1(pairs, np.array([0.9, 0.6, 0.4]), questions)
    assert st["accuracy"] == pytest.approx(0.5) and st["n"] == 2


def test_overlap_score():
    assert X.overlap_score("Hur söker jag bidrag?", "Du söker bidrag här.") == pytest.approx(0.5)
    assert X.overlap_score("", "x") == 0.0


XML = """<corpus id="europarl-sv">
<text date="2000-01-17">
<paragraph>
<linkde id="ep-1">
<linken id="ep-1">
<sentence id="s1">
<w pos="NN">Återupptagande</w>
<w pos="PP">av</w>
<w pos="NN">sessionen</w>
</sentence>
</linken>
</linkde>
</paragraph>
</text>
<text speaker="X" speakerlang="SV" date="2000-01-17">
<paragraph>
<linken id="ep-2">
<sentence id="s2">
<w pos="PN">Vi</w>
<w pos="VB">tackar</w>
<w pos="MAD">&amp;</w>
<w pos="MAD">.</w>
</sentence>
<sentence id="s3">
<w pos="PN">Bra</w>
</sentence>
</linken>
</paragraph>
</text>
</corpus>
"""


def test_iter_paragraphs_reads_ids_speakerlang_and_tokens():
    paras = list(X.iter_paragraphs(XML.splitlines(keepends=True)))
    assert paras == [
        ("ep-1", None, [["Återupptagande", "av", "sessionen"]]),
        ("ep-2", "SV", [["Vi", "tackar", "&", "."], ["Bra"]]),
    ]


def test_keep_paragraph_filters():
    sv = ["Det här är en ganska lång svensk mening med många ord i sig."]
    en = ["This is a fairly long English sentence with many words in it."]
    assert X.keep_paragraph(sv, en)
    assert not X.keep_paragraph(sv, en + ["Extra."])  # sentence counts differ
    assert not X.keep_paragraph(["Kort."], ["Short."])  # too few words
    assert not X.keep_paragraph(["(EL) " + sv[0]], en)  # interpreter language mark
    assert not X.keep_paragraph(sv, [en[0] * 3])  # length ratio


def test_europarl_sample_keeps_all_swedish_originals_and_is_deterministic():
    rows = [{"id": f"en{i}", "original": "EN"} for i in range(50)] + [
        {"id": "sv0", "original": "SV"}
    ]
    a, b = X.europarl_sample(rows, 10), X.europarl_sample(rows, 10)
    assert a == b and len(a) == 11 and a[0]["id"] == "sv0"


def test_europarl_pairs_drop_identical_and_missing_mt():
    rows = [
        {
            "id": "ep-1",
            "original": "EN",
            "sv": ["Hej", "då."],
            "en": ["Hi", "bye."],
            "mt_opus": "Hej då.",
        },
        {"id": "ep-2", "original": "EN", "sv": ["Ja."], "en": ["Yes."], "mt_opus": "Japp."},
        {"id": "ep-3", "original": "EN", "sv": ["Nej."], "en": ["No."], "mt_opus": None},
        {
            "id": "ep-4",
            "original": "SV",
            "sv": ["Tack."],
            "en": ["Thanks."],
            "mt_opus": "Tack så mycket.",
        },
    ]
    pairs = X.europarl_pairs(rows, "EN", "opus")
    assert [p["id"] for p in pairs] == ["ep-2-opus"]
    assert pairs[0]["prompt"] == X.SOURCE_PROMPT + "Yes." and pairs[0]["ref"] == "Ja."
    assert [p["id"] for p in X.europarl_pairs(rows, "SV", "opus")] == ["ep-4-opus"]


def test_unigram_logprob_prefers_frequent_words():
    freq = {"och": 100, "att": 80, "probelm": 0, "problem": 20}
    total, vocab = sum(freq.values()), len(freq)
    assert X.unigram_logprob("problem", freq, total, vocab) > X.unigram_logprob(
        "probelm", freq, total, vocab
    )
    assert X.unigram_logprob("", freq, total, vocab) == float("-inf")


def test_clean_translation():
    assert X.clean_translation('<think>hmm</think>\n"Hej  där."') == "Hej där."
    assert X.clean_translation("hmm</think>Hej.") == "Hej."
    assert X.clean_translation("<think>never finished") is None
    assert X.clean_translation("") is None and X.clean_translation(None) is None


def test_score_verdicts_order_and_parse_failures():
    # ref in slot A in the first list, in slot B in the second
    score, cons = X.score_verdicts(["A", "A", "B", None], ["B", "A", "A", "B"])
    assert score[:3].tolist() == [1.0, 0.5, 0.0] and np.isnan(score[3])
    assert cons[:3].tolist() == [1.0, 0.0, 1.0] and np.isnan(cons[3])


def test_versus_and_summaries():
    assert X.versus(np.array([2, 1, 1]), np.array([1, 2, 1])).tolist() == [1.0, 0.0, 0.5]
    s = X.point_summary(np.array([2.0, 1.0]), np.array([1.0, 1.0]))
    assert s["accuracy"] == pytest.approx(0.75) and s["n"] == 2
    p = X.pair_summary(np.array([0.9, 0.2]), np.array([0.8, 0.6]), {"g": np.array([True, False])})
    assert p["accuracy"] == pytest.approx(0.5)
    assert p["positional_consistency"] == pytest.approx(0.5)
    assert p["by_group"]["g"] == {"n": 1, "accuracy": 1.0}


def test_judge_plan_covers_every_set_and_setup():
    for name in X.SETS:
        for setup, qid, _ in X.JUDGE_PLAN[name]:
            assert setup in X.JUDGE_SYSTEMS or setup == "preference"
            assert qid in ("preference", "naturalness")
    p = {"prompt": X.SOURCE_PROMPT + "Hello.", "ref": "Hej.", "other": "Hallå."}
    body = X.judge_body("translation", p, "Hej.", "Hallå.")
    assert body.startswith("## English\nHello.") and X.SOURCE_PROMPT not in body
    assert X.judge_body("naturalness", p, "a", "b") == "## Version A\na\n\n## Version B\nb"


def test_fix_colon_genitive_only_joins_abbreviations():
    assert (
        X.fix_colon_genitive("EU: s medlemsstater och FN: s råd")
        == "EU:s medlemsstater och FN:s råd"
    )
    assert X.fix_colon_genitive("frågan: en fråga om punkt 3: en sak") == (
        "frågan: en fråga om punkt 3: en sak"
    )
