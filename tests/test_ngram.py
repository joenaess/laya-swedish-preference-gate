"""Offline tests for the trigram baseline (tiny synthetic corpus)."""

from laya_gate import ngram


def _model():
    corpus = ["hon har bott här länge ."] * 20 + ["vi har bott där länge ."] * 10
    return ngram.Trigram.from_sentences(corpus)


def test_tokens_lowercase_and_split_punctuation():
    assert ngram.tokens("Hej, Världen!") == ["<s>", "<s>", "hej", ",", "världen", "!", "</s>"]


def test_trigram_prefers_seen_word_order():
    m = _model()
    assert m.mean_logprob("hon har bott här länge .") > m.mean_logprob("hon bott har här länge .")
    assert m.mean_logprob("hon har bott här länge .") > m.mean_logprob("hon har bo här länge .")


def test_probabilities_are_valid():
    m = _model()
    for u, v, w in (("<s>", "<s>", "hon"), ("har", "bott", "här"), ("x", "y", "okänt")):
        assert 0 < m.score_token(u, v, w) <= 1
