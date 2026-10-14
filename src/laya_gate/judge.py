"""Kimi-k3 as a System 2 LLM judge, compared with the gate, plus a cascade simulation.

The endpoint is self-hosted infrastructure: its URL and key are read from the environment
(`BASE_URL`, `API_KEY`) and never printed, logged, cached or written to results.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .utils import (
    CACHE_DIR,
    DEFAULT_MARGIN,
    JUDGE_CACHE_PATH,
    SEED,
    SPLITS_DIR,
    load_metrics,
    percentiles,
    read_jsonl,
    update_metrics,
)

SECRET_VARS = ("BASE_URL", "API_KEY", "HF_TOKEN")
THRESHOLDS = [round(0.05 * i, 2) for i in range(10)]  # 0.00 ... 0.45

SYSTEM_PROMPT = (
    "You are an expert evaluator of Swedish-language assistant responses. You will be given "
    "a user prompt written in Swedish and two candidate responses, labelled A and B. Decide "
    "which response better answers the prompt, judging helpfulness, correctness, and how "
    "fluent and natural the Swedish is. Do not let the order of the responses or their length "
    "by itself influence your decision.\n\n"
    'Reply with JSON only, exactly one of: {"preference": "A"} or {"preference": "B"}'
)


def user_message(prompt: str, a: str, b: str) -> str:
    return f"## Prompt\n{prompt}\n\n## Response A\n{a}\n\n## Response B\n{b}"


# ------------------------------------------------------------------- secrets


class JudgeError(RuntimeError):
    """A judge request failed. The message never contains the endpoint URL or key."""


_URL = re.compile(r"https?://\S+")


def sanitize(text: str) -> str:
    """Remove every secret value (and any URL) from `text`."""
    for var in SECRET_VARS:
        value = os.environ.get(var)
        if value:
            text = text.replace(value, f"<{var}>")
            text = text.replace(value.rstrip("/"), f"<{var}>")
    return _URL.sub("<url>", text)


def quiet_http_logs() -> None:
    """httpx logs full request URLs at INFO; keep it and openai at WARNING."""
    for name in ("httpx", "httpx2", "httpcore", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ------------------------------------------------------------------- parsing

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?", re.IGNORECASE)
_JSON_OBJ = re.compile(r"\{[^{}]*\}")
_PREF = re.compile(r'"preference"\s*:\s*"\s*([AB])\s*"', re.IGNORECASE)


def parse_verdict(content: str | None) -> str | None:
    """`"A"`, `"B"`, or None when the reply holds no unambiguous verdict.

    Strips reasoning blocks (closed, or an unclosed `<think>` up to `</think>`) and code
    fences, then reads the last JSON object with a `preference` key. Never guesses.
    """
    if not content:
        return None
    text = _THINK.sub("", content)
    if "</think>" in text.lower():
        text = text[text.lower().rindex("</think>") + len("</think>") :]
    elif "<think>" in text.lower():
        return None  # reasoning never finished
    text = _FENCE.sub("", text)
    verdicts = []
    for obj in _JSON_OBJ.findall(text):
        try:
            value = json.loads(obj).get("preference")
        except (json.JSONDecodeError, AttributeError):
            m = _PREF.search(obj)
            value = m.group(1) if m else None
        if isinstance(value, str) and value.strip().upper() in ("A", "B"):
            verdicts.append(value.strip().upper())
    if not verdicts or len(set(verdicts)) > 1:
        return None
    return verdicts[-1]


# --------------------------------------------------------------------- cache


def cache_key(model: str, prompt: str, a: str, b: str) -> str:
    payload = json.dumps([model, prompt, a, b], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out = {}
    for row in read_jsonl(path):
        out[row["key"]] = row
    return out


# ---------------------------------------------------------------------- client


def _status(exc: BaseException) -> int | None:
    return getattr(exc, "status_code", None)


def is_retryable(exc: BaseException) -> bool:
    import openai

    if isinstance(exc, openai.APITimeoutError | asyncio.TimeoutError | openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIConnectionError):
        return True
    status = _status(exc)
    return status is not None and (status == 429 or status >= 500)


@dataclass
class Judge:
    client: Any
    model: str
    cache_path: Path = JUDGE_CACHE_PATH
    concurrency: int = 16
    timeout: float = 600.0
    max_retries: int = 3
    base_delay: float = 2.0
    max_tokens: int = 16384
    cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    live_requests: int = 0
    failures: int = 0

    def __post_init__(self) -> None:
        self.cache = load_cache(self.cache_path)
        self._sem = asyncio.Semaphore(self.concurrency)
        self._lock = asyncio.Lock()

    async def _append(self, row: dict[str, Any]) -> None:
        async with self._lock:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.cache[row["key"]] = row

    async def _call(self, prompt: str, a: str, b: str):
        return await asyncio.wait_for(
            self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message(prompt, a, b)},
                ],
                temperature=0,
                max_tokens=self.max_tokens,
            ),
            timeout=self.timeout,
        )

    async def request(self, prompt: str, a: str, b: str) -> tuple[Any, float]:
        """One completion with exponential-backoff retries; raises a sanitized `JudgeError`."""
        attempt = 0
        while True:
            t = time.perf_counter()
            try:
                resp = await self._call(prompt, a, b)
                return resp, time.perf_counter() - t
            except Exception as exc:  # noqa: BLE001 - re-raised sanitized below
                if attempt < self.max_retries and is_retryable(exc):
                    await asyncio.sleep(self.base_delay * 2**attempt + random.random() * 0.1)
                    attempt += 1
                    continue
                msg = sanitize(f"{type(exc).__name__} (status={_status(exc)}): {exc}")
                raise JudgeError(
                    f"judge request failed after {attempt + 1} attempt(s): {msg}"
                ) from None

    async def judge(self, pair_id: str, order: str, prompt: str, a: str, b: str):
        key = cache_key(self.model, prompt, a, b)
        if key in self.cache:
            return self.cache[key]
        async with self._sem:
            try:
                resp, latency = await self.request(prompt, a, b)
            except JudgeError as exc:
                self.failures += 1
                return {
                    "key": key,
                    "pair_id": pair_id,
                    "order": order,
                    "verdict": None,
                    "error": str(exc),
                    "cached": False,
                }
        self.live_requests += 1
        choice = resp.choices[0]
        content = choice.message.content
        usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
        row = {
            "key": key,
            "pair_id": pair_id,
            "order": order,
            "model": self.model,
            "verdict": parse_verdict(content),
            "content": content,
            "finish_reason": choice.finish_reason,
            "latency_s": latency,
            "usage": {
                k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            },
        }
        await self._append(row)
        return row

    async def run(self, jobs: Sequence[tuple[str, str, str, str, str]], progress_every=50):
        done = 0

        async def one(job):
            nonlocal done
            res = await self.judge(*job)
            done += 1
            if progress_every and done % progress_every == 0:
                print(
                    f"judge: {done}/{len(jobs)} (live {self.live_requests}, "
                    f"failed {self.failures})",
                    flush=True,
                )
            return res

        return await asyncio.gather(*(one(j) for j in jobs))


def make_client():
    from openai import AsyncOpenAI

    base = os.environ["BASE_URL"].rstrip("/")
    return AsyncOpenAI(base_url=base, api_key=os.environ["API_KEY"], max_retries=0)


async def discover_model(client) -> str:
    if os.environ.get("KIMI_MODEL"):
        return os.environ["KIMI_MODEL"]
    try:
        models = await client.models.list()
    except Exception as exc:  # noqa: BLE001
        raise JudgeError(sanitize(f"model discovery failed: {type(exc).__name__}")) from None
    return models.data[0].id


# ---------------------------------------------------------------- evaluation


def gate_p_chosen(entry: dict[str, Any]) -> tuple[float, float]:
    """Gate P(chosen wins) in the forward (AB) and reversed (BA) orders."""
    if entry["label"] == 0:
        return entry["p_a_fwd"], 1 - entry["p_b_rev"]
    return 1 - entry["p_a_fwd"], entry["p_b_rev"]


def judge_correct(verdict: str | None, label: int, order: str) -> bool | None:
    """Whether a verdict picks the chosen response. `order` AB = stored order, BA = swapped."""
    if verdict is None:
        return None
    chosen_slot = ("A" if label == 0 else "B") if order == "AB" else ("B" if label == 0 else "A")
    return verdict == chosen_slot


def cascade_table(
    gate_p: np.ndarray,
    judge_ok: Sequence[bool | None],
    gate_ms: float,
    judge_ms: float,
    thresholds: Sequence[float] = THRESHOLDS,
) -> list[dict[str, float]]:
    """Per threshold: route an instance to the judge when `|p - 0.5| < threshold`.

    `gate_p` is the gate's P(chosen) per instance; `judge_ok` whether the judge was right
    (None = no parseable verdict; such a routed instance keeps the gate's answer).
    """
    gate_p = np.asarray(gate_p, dtype=float)
    gate_ok = gate_p > 0.5
    rows = []
    for thr in thresholds:
        routed = np.abs(gate_p - 0.5) < thr
        final = gate_ok.copy()
        fallback = 0
        for i in np.flatnonzero(routed):
            if judge_ok[i] is None:
                fallback += 1
            else:
                final[i] = judge_ok[i]
        frac = float(routed.mean())
        rows.append(
            {
                "threshold": float(thr),
                "routed_fraction": frac,
                "cascade_accuracy": float(final.mean()),
                "expected_latency_ms": gate_ms + frac * judge_ms,
                "routed_without_verdict": fallback,
            }
        )
    return rows


def recommend(rows: list[dict[str, float]], tolerance: float = 0.005) -> dict[str, float]:
    """Lowest threshold whose accuracy is within `tolerance` of the best in the sweep."""
    best = max(r["cascade_accuracy"] for r in rows)
    for r in sorted(rows, key=lambda r: r["routed_fraction"]):
        if r["cascade_accuracy"] >= best - tolerance:
            return {
                **r,
                "rule": f"lowest routed fraction within {tolerance:.1%} of the best "
                "cascade accuracy in the sweep",
            }
    return rows[-1]


def evaluate(rows, verdicts, gate, gate_ms, judge_ms) -> dict[str, Any]:
    by_key = {(v["pair_id"], v["order"]): v for v in verdicts}
    inst_gate_p, inst_judge_ok, inst_gate_ok = [], [], []
    per_order = {"AB": {"judge": [], "gate": []}, "BA": {"judge": [], "gate": []}}
    consistent, noise_ids = [], []
    for r in rows:
        p_ab, p_ba = gate_p_chosen(gate[r["id"]])
        oks = {}
        for order, p in (("AB", p_ab), ("BA", p_ba)):
            ok = judge_correct(by_key[(r["id"], order)]["verdict"], r["label"], order)
            oks[order] = ok
            inst_gate_p.append(p)
            inst_judge_ok.append(ok)
            inst_gate_ok.append(p > 0.5)
            per_order[order]["judge"].append(ok)
            per_order[order]["gate"].append(p > 0.5)
        if None not in oks.values():
            consistent.append(oks["AB"] == oks["BA"])
        gate_avg = (p_ab + p_ba) / 2
        if oks["AB"] is False and oks["BA"] is False and gate_avg <= 0.5 - DEFAULT_MARGIN:
            noise_ids.append(r["id"])

    def acc(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None

    def acc_strict(xs):
        return float(np.mean([bool(x) for x in xs]))

    parsed = [i for i, ok in enumerate(inst_judge_ok) if ok is not None]
    agree = [inst_gate_ok[i] == inst_judge_ok[i] for i in parsed]
    disagree = [i for i in parsed if inst_gate_ok[i] != inst_judge_ok[i]]
    table = cascade_table(np.array(inst_gate_p), inst_judge_ok, gate_ms, judge_ms)
    return {
        "judge_accuracy": {
            "AB": acc(per_order["AB"]["judge"]),
            "BA": acc(per_order["BA"]["judge"]),
            "order_averaged": acc(inst_judge_ok),
            "order_averaged_parse_failures_as_wrong": acc_strict(inst_judge_ok),
        },
        "judge_positional_consistency": acc(consistent),
        "gate_accuracy": {
            "AB": acc(per_order["AB"]["gate"]),
            "BA": acc(per_order["BA"]["gate"]),
            "order_averaged": acc(inst_gate_ok),
        },
        "gate_positional_consistency": float(
            np.mean(
                [
                    (gate_p_chosen(gate[r["id"]])[0] > 0.5)
                    == (gate_p_chosen(gate[r["id"]])[1] > 0.5)
                    for r in rows
                ]
            )
        ),
        "agreement_rate": float(np.mean(agree)) if agree else None,
        "disagreements": {
            "n": len(disagree),
            "gate_accuracy": acc([inst_gate_ok[i] for i in disagree]),
            "judge_accuracy": acc([inst_judge_ok[i] for i in disagree]),
        },
        "cascade": table,
        "recommended": recommend(table),
        "label_noise": {
            "rule": f"judge prefers the rejected response in both orders AND gate's "
            f"order-averaged P(chosen) <= {0.5 - DEFAULT_MARGIN:.2f}",
            "n_suspected": len(noise_ids),
            "fraction_of_sample": len(noise_ids) / len(rows),
            "pair_ids": noise_ids,
        },
    }


# --------------------------------------------------------------------- main


async def amain(args) -> int:
    quiet_http_logs()
    missing = [v for v in ("BASE_URL", "API_KEY") if not os.environ.get(v)]
    if missing:
        msg = f"skipped: {missing[0]} not set"
        print(f"judge {msg}")
        update_metrics("judge", msg)
        return 0
    gate_path = CACHE_DIR / "gate_val_eval.json"
    if not gate_path.exists():
        print("judge skipped: run `python -m laya_gate.benchmark` first", file=sys.stderr)
        return 1
    gate = json.loads(gate_path.read_text(encoding="utf-8"))

    rows = read_jsonl(SPLITS_DIR / "val_eval.jsonl")
    sample = random.Random(SEED).sample(rows, min(args.n, len(rows)))
    client = make_client()
    model = await discover_model(client)
    judge = Judge(client, model, concurrency=args.concurrency)
    jobs = []
    for r in sample:
        jobs.append((r["id"], "AB", r["prompt"], r["candidate_a"], r["candidate_b"]))
        jobs.append((r["id"], "BA", r["prompt"], r["candidate_b"], r["candidate_a"]))
    cached_before = sum(cache_key(model, *j[2:]) in judge.cache for j in jobs)
    t = time.perf_counter()
    verdicts = await judge.run(jobs)
    wall = time.perf_counter() - t

    ok_rows = [v for v in verdicts if "latency_s" in v]
    lat_ms = [v["latency_s"] * 1000 for v in ok_rows]
    previous = load_metrics().get("judge")
    previous = previous if isinstance(previous, dict) else {}
    if judge.live_requests:
        thr = {
            "requests_per_second": judge.live_requests / wall,
            "wall_seconds": wall,
            "live_requests": judge.live_requests,
            "concurrency": args.concurrency,
        }
    else:
        thr = previous.get("throughput", {"note": "all verdicts cached; not re-measured"})

    bench = load_metrics().get("benchmark", {}).get("latency", {})
    gate_ms = bench.get("gpu_end_to_end_ms", {}).get("p50")
    if gate_ms is None:
        gate_ms = bench.get("cpu_forward_ms", {}).get("p50", 0.0)
    judge_ms = float(np.median(lat_ms)) if lat_ms else float("nan")

    usage = [v.get("usage") or {} for v in ok_rows]
    failures = [v for v in verdicts if v.get("verdict") is None]
    out = {
        "model": model,
        "endpoint": "self-hosted Kimi-k3 (8x NVIDIA B300, HGX)",
        "sample": {
            "split": "val_eval",
            "n_pairs": len(sample),
            "n_requests": len(jobs),
            "seed": SEED,
        },
        "settings": {
            "temperature": 0,
            "max_tokens": judge.max_tokens,
            "concurrency": args.concurrency,
            "max_retries": judge.max_retries,
            "timeout_s": judge.timeout,
            "input": "full untruncated prompt and responses",
        },
        "cached_before_run": cached_before,
        "request_failures": judge.failures,
        "parse_failures": len([v for v in failures if "error" not in v]),
        "parse_failure_rate": len([v for v in failures if "error" not in v]) / len(jobs),
        "latency_ms": percentiles(lat_ms) if lat_ms else None,
        "throughput": thr,
        "tokens": {
            k: int(sum((u.get(k) or 0) for u in usage))
            for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "gate_latency_ms_used": gate_ms,
        "latency_ratio_judge_over_gate": judge_ms / gate_ms if gate_ms else None,
        **evaluate(sample, verdicts, gate, gate_ms, judge_ms),
    }
    update_metrics("judge", out)
    printable = {k: v for k, v in out.items() if k not in ("cascade", "label_noise")}
    print(json.dumps(printable, indent=2))
    return 0


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args(argv)
    try:
        code = asyncio.run(amain(args))
    except JudgeError as exc:
        print(f"judge failed: {exc}", file=sys.stderr)
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
