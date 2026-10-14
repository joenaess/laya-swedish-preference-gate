"""Fluent-vs-fluent probe helpers (offline)."""

from __future__ import annotations

import numpy as np
import pytest

from laya_gate import probe as P


@pytest.mark.parametrize(
    "content",
    [
        '{"prompt": "Hej?", "response_1": "Ett.", "response_2": "Två."}',
        '```json\n{"prompt": "Hej?", "response_1": "Ett.", "response_2": "Två."}\n```',
        'tänker...</think>\n{"prompt": "Hej?", "response_1": "Ett.", "response_2": "Två."}',
    ],
)
def test_parse_translation_valid(content):
    assert P._parse_translation(content) == {
        "prompt": "Hej?",
        "response_1": "Ett.",
        "response_2": "Två.",
    }


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        "inte json",
        '{"prompt": "x", "response_1": "y"}',
        '{"prompt": "x", "response_1": "y", "response_2": "  "}',
        '{"prompt": ',
    ],
)
def test_parse_translation_invalid(content):
    assert P._parse_translation(content) is None


def test_to_swedish_maps_back_to_chosen():
    tr = {"prompt": "F", "response_1": "R1", "response_2": "R2"}
    base = {"id": "x", "chosen": "c", "rejected": "r"}
    first = P.to_swedish({**base, "chosen_first": True}, tr)
    assert (first["sv_chosen"], first["sv_rejected"]) == ("R1", "R2")
    second = P.to_swedish({**base, "chosen_first": False}, tr)
    assert (second["sv_chosen"], second["sv_rejected"]) == ("R2", "R1")
    assert second["sv_prompt"] == "F"


def test_translation_key_depends_on_order():
    p = {"id": "uf-1", "chosen_first": True}
    assert P._key("m", p) != P._key("m", {**p, "chosen_first": False})
    assert P._key("m", p) == P._key("m", dict(p))


def test_bootstrap_ci_brackets_mean():
    correct = np.array([1] * 70 + [0] * 30, dtype=float)
    lo, hi = P.bootstrap_ci(correct)
    assert lo < 0.7 < hi
    assert lo > 0.55 and hi < 0.85
    s = P.summarize(correct)
    assert s["accuracy"] == pytest.approx(0.7) and s["n"] == 100


def test_content_takes_assistant_turn():
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    assert P._content(msgs) == "a"
    assert P._content([{"role": "user", "content": "q"}]) == ""


def test_translate_all_forwards_request_options_and_caches_by_mode(tmp_path):
    import asyncio
    import json
    from types import SimpleNamespace

    seen = []

    async def create(**kwargs):
        seen.append(kwargs)
        content = json.dumps({"prompt": "F", "response_1": "A", "response_2": "B"})
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")
            ],
            usage=SimpleNamespace(model_dump=lambda: {"completion_tokens": 3}),
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    pairs = [{"id": "uf-1", "prompt": "p", "chosen": "c", "rejected": "r", "chosen_first": True}]
    extra = {"extra_body": {"chat_template_kwargs": {"thinking": False}}}
    cache = tmp_path / "t.jsonl"

    rows, _ = asyncio.run(
        P.translate_all(
            client, "m", pairs, 1, cache_path=cache, extra=extra, mode="nothink", progress_every=0
        )
    )
    assert seen[0]["extra_body"] == extra["extra_body"]
    assert rows[0]["translation"]["response_1"] == "A"
    asyncio.run(
        P.translate_all(
            client, "m", pairs, 1, cache_path=cache, extra=extra, mode="nothink", progress_every=0
        )
    )
    assert len(seen) == 1  # cache hit for the same mode
    asyncio.run(P.translate_all(client, "m", pairs, 1, cache_path=cache, progress_every=0))
    assert len(seen) == 2 and "extra_body" not in seen[1]  # other mode, not shared
