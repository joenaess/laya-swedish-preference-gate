"""Inference engine: `SwedishPreferenceGate`.

One forward pass of the Laya decision model scores both candidates through the `choice`
primitive; a fitted temperature turns the two logits into a calibrated probability.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import state_room_for, truncate_triplet
from .utils import (
    CHECKPOINT_DIR,
    DEFAULT_MARGIN,
    HEAD_MAX_LEN,
    MAX_LEN,
    NATURALNESS_QID,
    OPTIONS,
    QID,
    QUESTION,
    QUESTIONS_FILE,
)

TEMPERATURE_FILE = "temperature.json"


def resolve_checkpoint(checkpoint_path: str | os.PathLike, revision: str | None = None) -> Path:
    """A local directory, or a Hugging Face Hub repo id (at `revision`) downloaded to the cache."""
    path = Path(checkpoint_path)
    if path.is_dir():
        return path
    if str(checkpoint_path).startswith(("/", "./", "../")):
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_path}")
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            str(checkpoint_path),
            revision=revision,
            allow_patterns=[
                "rl_agent_config.json",
                "model.safetensors",
                "tokenizer/*",
                "encoder/*",
                TEMPERATURE_FILE,
                QUESTIONS_FILE,
            ],
        )
    )


def load_model(model_dir: Path):
    """`(model, tokenizer, cfg)` from a checkpoint in the layout `laya.load` reads."""
    from laya.agent import _fix_tokenizer_config
    from laya.common import build_model
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    _fix_tokenizer_config(str(model_dir))
    cfg = json.loads((model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))
    model = build_model(cfg, encoder_dir=str(model_dir / "encoder"), pretrained=False)
    model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True)
    return model, tok, cfg


def collate(items: Sequence[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    from laya.common import collate_items

    batch = collate_items([list(items)], pad_id)
    keys = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
    return {k: batch[k] for k in keys}


@torch.inference_mode()
def forward_logits(model, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    """Raw option logits `[n, 2]` (float32) for a collated batch."""
    args = [batch[k].to(device, non_blocking=True) for k in batch]
    logits, _act = model(*args)
    return logits[:, : len(OPTIONS)].float()


class SwedishPreferenceGate:
    """Fast pairwise preference judge for Swedish responses.

    Args:
        checkpoint_path: local checkpoint directory or Hugging Face Hub repo id.
        margin: `is_valid_margin` is True when `|p_chosen - 0.5| >= margin`. The default 0.2
            accepts predictions with calibrated probability >= 0.7 for either candidate.
        device: "cuda", "cpu", ... Defaults to CUDA when available.
        dtype: weights/activation dtype. Defaults to bfloat16 on CUDA and float32 elsewhere.
        temperature: override the fitted `preference` temperature (otherwise read from
            `temperature.json`; 1.0 when the checkpoint has none).
        revision: Hub branch, tag or commit (e.g. "v1" for the first release); ignored for
            local directories.

    A checkpoint that ships `questions.json` with a `naturalness` question (v2) also supports
    `predict_naturalness` / `predict_naturalness_batch`.
    """

    def __init__(
        self,
        checkpoint_path: str | os.PathLike = CHECKPOINT_DIR,
        margin: float = DEFAULT_MARGIN,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        temperature: float | None = None,
        revision: str | None = None,
    ):
        if not 0.0 <= margin < 0.5:
            raise ValueError("margin must be in [0, 0.5)")
        self.margin = float(margin)
        self.model_dir = resolve_checkpoint(checkpoint_path, revision)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if dtype is None:
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.dtype = dtype
        model, self.tok, self.cfg = load_model(self.model_dir)
        self.model = model.to(self.device, dtype=dtype).eval()
        self.max_len = int(self.cfg.get("max_len", MAX_LEN))
        self.head_max_len = int(self.cfg.get("head_max_len", HEAD_MAX_LEN))
        q_path = self.model_dir / QUESTIONS_FILE
        self.questions = (
            json.loads(q_path.read_text(encoding="utf-8")) if q_path.exists() else dict(QUESTION)
        )
        from laya.train import to_internal

        self._internal = {qid: to_internal(qid, q) for qid, q in self.questions.items()}
        self.room = state_room_for(self.tok, self.questions, self.max_len, self.head_max_len)

        temp_path = self.model_dir / TEMPERATURE_FILE
        saved = json.loads(temp_path.read_text(encoding="utf-8")) if temp_path.exists() else {}
        self.calibrated = temperature is not None or bool(saved)
        self.temperatures = {qid: 1.0 for qid in self.questions}
        self.temperatures.update(saved.get("temperatures", {}))
        if "temperature" in saved:
            self.temperatures[QID] = saved["temperature"]
        if temperature is not None:
            self.temperatures[QID] = temperature
        self.temperature = float(self.temperatures[QID])
        if any(t <= 0 for t in self.temperatures.values()):
            raise ValueError("temperature must be positive")

    # ------------------------------------------------------------------ encoding

    def encode(
        self, prompt: str, candidate_a: str, candidate_b: str, qid: str = QID
    ) -> dict[str, Any]:
        """Token-level truncation, then the `choice` sequence laya trains on."""
        from laya.common import build_sequence

        if qid not in self._internal:
            raise ValueError(f"this checkpoint has no {qid!r} question")
        state, truncated, _ = truncate_triplet(
            self.tok, prompt, candidate_a, candidate_b, self.room
        )
        ids, markers = build_sequence(
            self.tok, state, self._internal[qid], self.max_len, self.head_max_len
        )
        return {"ids": ids, "markers": markers, "qtype": 0, "truncated": truncated}

    def logits(self, items: Sequence[dict[str, Any]], batch_size: int = 32) -> np.ndarray:
        """Raw (uncalibrated) logits `[n, 2]`; batches are length-sorted, order is restored."""
        order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
        out = np.zeros((len(items), len(OPTIONS)), dtype=np.float32)
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            batch = collate([items[i] for i in idx], self.tok.pad_token_id)
            out[idx] = forward_logits(self.model, batch, self.device).cpu().numpy()
        return out

    def probabilities(self, logits: np.ndarray, qid: str = QID) -> np.ndarray:
        t = self.temperature if qid == QID else self.temperatures[qid]
        z = np.asarray(logits, dtype=np.float64) / t
        z = z - z.max(axis=-1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=-1, keepdims=True)

    def _result(self, probs: np.ndarray) -> dict[str, Any]:
        p_a = float(probs[0])
        return {
            "preference": OPTIONS[0] if p_a >= 0.5 else OPTIONS[1],
            "p_chosen": p_a,
            "is_valid_margin": bool(abs(p_a - 0.5) >= self.margin),
            "probabilities": {OPTIONS[0]: p_a, OPTIONS[1]: float(probs[1])},
        }

    # --------------------------------------------------------------------- public

    def predict_preference(
        self, prompt: str, chosen_candidate: str, rejected_candidate: str
    ) -> dict[str, Any]:
        """Score one pair. `chosen_candidate` is shown as candidate_a, `rejected_candidate` as
        candidate_b; `p_chosen` is the calibrated probability that candidate_a is preferred."""
        item = self.encode(prompt, chosen_candidate, rejected_candidate)
        logits = self.logits([item], batch_size=1)
        return self._result(self.probabilities(logits)[0])

    def predict_preference_batch(
        self,
        prompts: Sequence[str],
        chosen_candidates: Sequence[str],
        rejected_candidates: Sequence[str],
        batch_size: int = 32,
    ) -> list[dict[str, Any]]:
        if not len(prompts) == len(chosen_candidates) == len(rejected_candidates):
            raise ValueError("prompts and candidate lists must have the same length")
        items = [
            self.encode(p, a, b)
            for p, a, b in zip(prompts, chosen_candidates, rejected_candidates, strict=True)
        ]
        probs = self.probabilities(self.logits(items, batch_size=batch_size))
        return [self._result(p) for p in probs]

    @property
    def has_naturalness(self) -> bool:
        return NATURALNESS_QID in self._internal

    def _natural_result(self, probs: np.ndarray) -> dict[str, Any]:
        p_a = float(probs[0])
        return {
            "more_natural": OPTIONS[0] if p_a >= 0.5 else OPTIONS[1],
            "p_a_more_natural": p_a,
            "probabilities": {OPTIONS[0]: p_a, OPTIONS[1]: float(probs[1])},
        }

    def predict_naturalness(self, prompt: str, candidate_a: str, candidate_b: str):
        """Which response reads as more natural, idiomatic Swedish (v2 checkpoints only)."""
        return self.predict_naturalness_batch([prompt], [candidate_a], [candidate_b], 1)[0]

    def predict_naturalness_batch(
        self,
        prompts: Sequence[str],
        candidates_a: Sequence[str],
        candidates_b: Sequence[str],
        batch_size: int = 32,
    ) -> list[dict[str, Any]]:
        if not len(prompts) == len(candidates_a) == len(candidates_b):
            raise ValueError("prompts and candidate lists must have the same length")
        items = [
            self.encode(p, a, b, qid=NATURALNESS_QID)
            for p, a, b in zip(prompts, candidates_a, candidates_b, strict=True)
        ]
        probs = self.probabilities(self.logits(items, batch_size=batch_size), NATURALNESS_QID)
        return [self._natural_result(p) for p in probs]
