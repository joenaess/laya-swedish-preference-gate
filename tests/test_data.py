"""Data pipeline: symmetrization, splitting and token-level truncation."""

from __future__ import annotations

import pytest
from conftest import SWEDISH

from laya_gate.data import (
    build_items,
    clean_pairs,
    head_tokens,
    make_state,
    split_examples,
    state_room,
    symmetrize,
    tail_tokens,
    truncate_triplet,
    waterfill,
)
from laya_gate.utils import HEAD_MAX_LEN, MAX_LEN


def _sequence_length(tok, state, max_len, head_max_len):
    from laya.common import build_sequence

    from laya_gate.data import internal_question

    seq, _markers, trunc = build_sequence(
        tok, state, internal_question(), max_len, head_max_len, return_truncation_stats=True
    )
    return len(seq), trunc


def test_label_balance_in_splits(synthetic_pairs):
    splits, _ = split_examples(symmetrize(synthetic_pairs))
    for name, rows in splits.items():
        frac_b = sum(r["label"] for r in rows) / len(rows)
        assert abs(frac_b - 0.5) <= 0.02, name


def test_split_sizes_and_disjoint(synthetic_pairs):
    splits, _ = split_examples(symmetrize(synthetic_pairs))
    n = sum(len(v) for v in splits.values())
    assert n == len(synthetic_pairs)
    assert abs(len(splits["train"]) / n - 0.85) < 0.02
    ids = [set(r["id"] for r in v) for v in splits.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_shared_prompts_stay_in_train(synthetic_pairs):
    pairs = synthetic_pairs[:200]
    for p in pairs[100:]:
        p["prompt"] = "samma fråga"
    splits, _ = split_examples(symmetrize(pairs))
    val_prompts = {r["prompt"] for k in ("val_calib", "val_eval") for r in splits[k]}
    train_prompts = {r["prompt"] for r in splits["train"]}
    assert not val_prompts & train_prompts


def test_symmetrize_is_seeded_and_preserves_content(synthetic_pairs):
    a, b = symmetrize(synthetic_pairs, seed=1), symmetrize(synthetic_pairs, seed=1)
    assert a == b
    for ex, p in zip(a, synthetic_pairs, strict=True):
        chosen = ex["candidate_b"] if ex["label"] else ex["candidate_a"]
        rejected = ex["candidate_a"] if ex["label"] else ex["candidate_b"]
        assert (chosen, rejected) == (p["chosen"], p["rejected"])


def test_clean_pairs_drops_unusable():
    rows = [
        {"id": "1", "prompt": "a", "chosen": "x", "rejected": "y"},
        {"id": "1", "prompt": "a", "chosen": "x", "rejected": "y"},
        {"id": "2", "prompt": " ", "chosen": "x", "rejected": "y"},
        {"id": "3", "prompt": "a", "chosen": "x", "rejected": " x "},
    ]
    kept, dropped = clean_pairs(rows)
    assert [r["id"] for r in kept] == ["1"]
    assert dropped == {"empty_field": 1, "identical_responses": 1, "duplicate": 1}


def test_waterfill():
    assert waterfill([10, 500, 500], 300) == [10, 145, 145]
    assert waterfill([10, 20, 30], 100) == [10, 20, 30]
    assert sum(waterfill([400, 400, 400], 301)) == 301


def test_head_and_tail_cut_on_token_boundaries(tiny_tokenizer):
    text = "Åsa åt ärtsoppa på Öland."
    offs = [
        tuple(o)
        for o in tiny_tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)[
            "offset_mapping"
        ]
    ]
    for n in range(len(offs) + 1):
        h, t = head_tokens(text, offs, n), tail_tokens(text, offs, n)
        assert text.startswith(h) and text.endswith(t)
        h.encode("utf-8"), t.encode("utf-8")  # always valid strings


def test_short_input_is_not_truncated(tiny_tokenizer):
    state, truncated, _ = truncate_triplet(tiny_tokenizer, "Hej?", "Hej!", "Tja.", room=200)
    assert not truncated
    assert state == make_state("Hej?", "Hej!", "Tja.")


@pytest.mark.parametrize("max_len", [256, 1024])
def test_long_input_truncated_within_budget(tiny_tokenizer, max_len):
    tok = tiny_tokenizer
    long = SWEDISH * 40
    head = 64
    room = state_room(tok, max_len, head)
    full = make_state(long, long, long)
    assert _sequence_length(tok, full, 10**6, head)[0] > max_len  # really over budget
    state, truncated, n = truncate_triplet(tok, long, long, long, room)
    assert truncated and n <= room
    length, trunc = _sequence_length(tok, state, max_len, head)
    assert length <= max_len
    assert trunc["state_tokens_dropped"] == 0  # our truncation did it, not laya's clamp


def test_truncation_keeps_prompt_tail_and_response_heads(tiny_tokenizer):
    prompt = "START " + SWEDISH * 20 + " SLUTFRÅGA"
    resp = "BÖRJAN " + SWEDISH * 20 + " SLUT"
    state, truncated, _ = truncate_triplet(tiny_tokenizer, prompt, resp, resp, room=300)
    assert truncated
    assert state["prompt"].endswith("SLUTFRÅGA") and not state["prompt"].startswith("START")
    assert state["candidate_a"].startswith("BÖRJAN") and not state["candidate_a"].endswith("SLUT")


def test_truncated_swedish_text_is_valid_utf8(tiny_tokenizer):
    tok = tiny_tokenizer
    text = "åäöÅÄÖ " * 600
    state, truncated, _ = truncate_triplet(tok, text, text, text, room=200)
    assert truncated
    for value in state.values():
        assert "�" not in value
        value.encode("utf-8")
        ids = tok(value, add_special_tokens=False)["input_ids"]
        assert "�" not in tok.decode(ids)


def test_neither_response_empty_after_truncation(tiny_tokenizer):
    tok = tiny_tokenizer
    huge_prompt = SWEDISH * 200
    for a, b in [(SWEDISH * 50, "Ja."), (SWEDISH * 50, SWEDISH * 80), ("Nej.", SWEDISH * 90)]:
        state, _, n = truncate_triplet(tok, huge_prompt, a, b, room=180)
        assert n <= 180
        assert state["candidate_a"].strip() and state["candidate_b"].strip()


def test_escape_heavy_response_is_not_emptied(tiny_tokenizer):
    """Control characters cost many tokens once JSON-escaped; the other fields must survive."""
    tok = tiny_tokenizer
    weird = "Svar:" + "\u000b" * 3000 + " slut"
    state, truncated, n = truncate_triplet(tok, "Fråga?", SWEDISH * 3, weird, room=200)
    assert truncated and n <= 200
    assert state["candidate_a"].strip() and state["candidate_b"].strip()


def test_build_items_targets(tiny_tokenizer):
    examples = symmetrize(
        [
            {"id": str(i), "prompt": "Fråga?", "chosen": "Bra svar.", "rejected": "Dåligt."}
            for i in range(6)
        ]
    )
    items, stats = build_items(tiny_tokenizer, examples, room=150)
    for it, ex in zip(items, examples, strict=True):
        assert it["target"] == ([1.0, 0.0] if ex["label"] == 0 else [0.0, 1.0])
        assert it["k"] == 2
    assert not any(s["truncated"] for s in stats)


@pytest.mark.network
def test_real_tokenizer_truncates_to_1024():
    from laya_gate.data import load_tokenizer

    tok = load_tokenizer()
    room = state_room(tok)
    long = "Det här är en mycket lång text om åäö och svensk grammatik. " * 400
    state, truncated, _ = truncate_triplet(tok, long, long, long, room)
    length, trunc = _sequence_length(tok, state, MAX_LEN, HEAD_MAX_LEN)
    assert truncated and length <= MAX_LEN and trunc["state_tokens_dropped"] == 0
    assert all("�" not in v for v in state.values())


@pytest.mark.network
def test_load_raw_pairs_schema():
    from laya_gate.data import load_raw_pairs

    pairs = load_raw_pairs()
    assert len(pairs) > 10_000
    assert {"id", "prompt", "chosen", "rejected", "language"} <= set(pairs[0])
    assert all(isinstance(pairs[0][k], str) for k in ("prompt", "chosen", "rejected"))
