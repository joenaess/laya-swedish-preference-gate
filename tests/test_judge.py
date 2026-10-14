"""Judge parsing, caching, retries, secret hygiene and cascade math, all with a mocked client."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import httpx2 as httpx
import numpy as np
import openai
import pytest

from laya_gate import judge as J

FAKE_URL = "https://kimi.internal.example:8443/v1"
FAKE_KEY = "sk-test-secret-0123456789"


@pytest.fixture()
def secrets(monkeypatch):
    monkeypatch.setenv("BASE_URL", FAKE_URL)
    monkeypatch.setenv("API_KEY", FAKE_KEY)


def _response(content: str, finish: str = "stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)],
        usage=SimpleNamespace(
            model_dump=lambda: {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        ),
    )


def _status_error(cls, status: int):
    req = httpx.Request(
        "POST", f"{FAKE_URL}/chat/completions", headers={"Authorization": f"Bearer {FAKE_KEY}"}
    )
    resp = httpx.Response(status, request=req)
    return cls(f"error from {FAKE_URL} with key {FAKE_KEY}", response=resp, body=None)


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return _response(outcome)


def _judge(client, tmp_path, **kw):
    return J.Judge(client, "kimi-k3", cache_path=tmp_path / "cache.jsonl", base_delay=0.0, **kw)


# --------------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    "content, expected",
    [
        ('{"preference": "A"}', "A"),
        ('{"preference":"b"}', "B"),
        ('```json\n{"preference": "B"}\n```', "B"),
        ('<think>A looks better... {"preference": "A"}</think>\n{"preference": "B"}', "B"),
        ('lots of reasoning</think>{"preference": "A"}', "A"),
        ('Final answer: {"preference": "A"}', "A"),
    ],
)
def test_parse_valid(content, expected):
    assert J.parse_verdict(content) == expected


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        "A",
        "Response A is better.",
        '{"preference": "C"}',
        '{"preference": "A", ',
        '{"winner": "A"}',
        '<think>still thinking {"preference": "A"}',
        '{"preference": "A"} {"preference": "B"}',
    ],
)
def test_parse_malformed_returns_none(content):
    assert J.parse_verdict(content) is None


# ----------------------------------------------------------------------- cache


def test_cache_miss_then_hit(tmp_path):
    client = FakeClient(['{"preference": "A"}'])
    jobs = [("p1", "AB", "Fråga", "svar a", "svar b"), ("p1", "BA", "Fråga", "svar b", "svar a")]

    async def go():
        return await _judge(client, tmp_path).run(jobs, progress_every=0)

    first = asyncio.run(go())
    assert client.calls == 2 and [r["verdict"] for r in first] == ["A", "A"]
    second = asyncio.run(go())  # new Judge, same cache file
    assert client.calls == 2
    assert [r["verdict"] for r in second] == ["A", "A"]
    assert len((tmp_path / "cache.jsonl").read_text().splitlines()) == 2


def test_cache_key_depends_on_order_and_model():
    k = J.cache_key("m", "p", "a", "b")
    assert k != J.cache_key("m", "p", "b", "a")
    assert k != J.cache_key("other", "p", "a", "b")
    assert k == J.cache_key("m", "p", "a", "b")


def test_cache_records_usage_and_latency(tmp_path):
    client = FakeClient(['{"preference": "B"}'])
    row = asyncio.run(_judge(client, tmp_path).judge("p", "AB", "q", "a", "b"))
    assert row["usage"]["total_tokens"] == 15
    assert row["latency_s"] >= 0
    assert row["verdict"] == "B"


# --------------------------------------------------------------------- retries


def test_retries_on_429_and_5xx_then_succeeds(tmp_path, secrets):
    client = FakeClient(
        [
            _status_error(openai.RateLimitError, 429),
            _status_error(openai.InternalServerError, 503),
            '{"preference": "A"}',
        ]
    )
    row = asyncio.run(_judge(client, tmp_path).judge("p", "AB", "q", "a", "b"))
    assert client.calls == 3 and row["verdict"] == "A"


def test_retries_on_timeout(tmp_path):
    client = FakeClient([TimeoutError(), '{"preference": "B"}'])
    row = asyncio.run(_judge(client, tmp_path).judge("p", "AB", "q", "a", "b"))
    assert client.calls == 2 and row["verdict"] == "B"


def test_gives_up_after_max_retries(tmp_path, secrets):
    client = FakeClient([_status_error(openai.InternalServerError, 500)])
    j = _judge(client, tmp_path)
    with pytest.raises(J.JudgeError):
        asyncio.run(j.request("q", "a", "b"))
    assert client.calls == 1 + j.max_retries


def test_no_retry_on_client_error(tmp_path, secrets):
    client = FakeClient([_status_error(openai.BadRequestError, 400)])
    with pytest.raises(J.JudgeError):
        asyncio.run(_judge(client, tmp_path).request("q", "a", "b"))
    assert client.calls == 1


def test_failed_request_is_not_cached(tmp_path, secrets):
    client = FakeClient([_status_error(openai.AuthenticationError, 401)])
    j = _judge(client, tmp_path)
    row = asyncio.run(j.judge("p", "AB", "q", "a", "b"))
    assert row["verdict"] is None and "error" in row
    assert not (tmp_path / "cache.jsonl").exists()


# ---------------------------------------------------------------------- secrets


def test_errors_never_contain_secrets(tmp_path, secrets):
    client = FakeClient([_status_error(openai.AuthenticationError, 401)])
    with pytest.raises(J.JudgeError) as info:
        asyncio.run(_judge(client, tmp_path).request("q", "a", "b"))
    text = str(info.value) + repr(info.value)
    assert FAKE_URL not in text and FAKE_KEY not in text and "kimi.internal" not in text
    assert info.value.__cause__ is None and info.value.__suppress_context__


def test_sanitize_masks_env_values_and_urls(secrets):
    msg = f"connect to {FAKE_URL}/models failed with key={FAKE_KEY}"
    clean = J.sanitize(msg)
    assert FAKE_KEY not in clean and FAKE_URL not in clean and "https://" not in clean


def test_missing_env_skips_cleanly(monkeypatch, tmp_path):
    monkeypatch.delenv("BASE_URL", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    recorded = {}
    monkeypatch.setattr(J, "update_metrics", lambda k, v: recorded.update({k: v}))
    code = asyncio.run(J.amain(SimpleNamespace(n=2, concurrency=1)))
    assert code == 0 and recorded["judge"].startswith("skipped: ")


# ----------------------------------------------------------------------- cascade


def test_judge_correct_handles_orders():
    assert J.judge_correct("A", 0, "AB") is True
    assert J.judge_correct("A", 0, "BA") is False
    assert J.judge_correct("B", 0, "BA") is True
    assert J.judge_correct("A", 1, "BA") is True
    assert J.judge_correct(None, 1, "AB") is None


def test_gate_p_chosen():
    assert J.gate_p_chosen({"label": 0, "p_a_fwd": 0.8, "p_b_rev": 0.3}) == (0.8, 0.7)
    assert J.gate_p_chosen({"label": 1, "p_a_fwd": 0.8, "p_b_rev": 0.3}) == pytest.approx(
        (0.2, 0.3)
    )


def test_cascade_math():
    gate_p = np.array([0.95, 0.10, 0.55, 0.45])  # right, wrong, right(low), wrong(low)
    judge_ok = [False, True, False, True]
    rows = J.cascade_table(
        gate_p, judge_ok, gate_ms=10.0, judge_ms=1000.0, thresholds=[0.0, 0.1, 0.5]
    )
    t0, t1, t5 = rows
    assert t0["routed_fraction"] == 0 and t0["cascade_accuracy"] == 0.5
    assert t0["expected_latency_ms"] == 10.0
    # 0.1 routes the two low-margin instances; the judge fixes #4 and breaks #3.
    assert t1["routed_fraction"] == 0.5 and t1["cascade_accuracy"] == 0.5
    assert t1["expected_latency_ms"] == pytest.approx(510.0)
    # 0.5 routes everything: judge accuracy.
    assert t5["routed_fraction"] == 1.0 and t5["cascade_accuracy"] == 0.5


def test_cascade_falls_back_to_gate_without_verdict():
    rows = J.cascade_table(np.array([0.52, 0.48]), [None, None], 1.0, 100.0, thresholds=[0.4])
    assert rows[0]["cascade_accuracy"] == 0.5
    assert rows[0]["routed_without_verdict"] == 2


def test_recommend_picks_cheapest_near_best():
    rows = [
        {"threshold": 0.0, "routed_fraction": 0.0, "cascade_accuracy": 0.70},
        {"threshold": 0.1, "routed_fraction": 0.2, "cascade_accuracy": 0.778},
        {"threshold": 0.2, "routed_fraction": 0.5, "cascade_accuracy": 0.78},
        {"threshold": 0.3, "routed_fraction": 0.8, "cascade_accuracy": 0.781},
    ]
    assert J.recommend(rows)["threshold"] == 0.1


def test_evaluate_end_to_end_synthetic():
    rows = [{"id": "x", "label": 0}, {"id": "y", "label": 1}]
    gate = {
        "x": {"label": 0, "p_a_fwd": 0.9, "p_b_rev": 0.2},
        "y": {"label": 1, "p_a_fwd": 0.9, "p_b_rev": 0.15},
    }
    verdicts = [
        {"pair_id": "x", "order": "AB", "verdict": "A"},
        {"pair_id": "x", "order": "BA", "verdict": "B"},
        {"pair_id": "y", "order": "AB", "verdict": "A"},
        {"pair_id": "y", "order": "BA", "verdict": "B"},
    ]
    out = J.evaluate(rows, verdicts, gate, 10.0, 1000.0)
    assert out["judge_accuracy"]["order_averaged"] == 0.5
    assert out["gate_accuracy"]["order_averaged"] == 0.5
    assert out["judge_positional_consistency"] == 1.0
    assert out["agreement_rate"] == 1.0
    assert out["label_noise"]["pair_ids"] == ["y"]


# -------------------------------------------------------------------- live endpoint


@pytest.mark.judge
def test_live_endpoint_returns_a_verdict(tmp_path):
    async def go():
        client = J.make_client()
        model = await J.discover_model(client)
        j = J.Judge(client, model, cache_path=tmp_path / "live.jsonl", concurrency=1)
        return await j.judge("live", "AB", "Vad är 2 + 2?", "2 + 2 = 4.", "2 + 2 = 5.")

    row = asyncio.run(go())
    assert row["verdict"] == "A"
    assert os.environ["BASE_URL"] not in str(row)
