"""`SwedishPreferenceGate`: schema, probabilities, batching. Runs on a tiny offline checkpoint
and, when present, on the trained one (CPU, and GPU when available)."""

from __future__ import annotations

import json

import pytest

from laya_gate.predictor import SwedishPreferenceGate

PROMPT = "Hur fungerar fotosyntesen?"
GOOD = "Fotosyntesen är den process där växter omvandlar ljusenergi till kemisk energi."
BAD = "Det är när solen lyser på träd och gör magi."
CASES = [
    (PROMPT, GOOD, BAD),
    ("Vad är huvudstaden i Sverige?", "Stockholm.", "Göteborg är huvudstaden."),
    ("Skriv en mening om hösten.", "Löven blir röda och gula.", "Höst."),
]


def _check_schema(res):
    assert set(res) == {"preference", "p_chosen", "is_valid_margin", "probabilities"}
    assert res["preference"] in ("candidate_a", "candidate_b")
    assert isinstance(res["p_chosen"], float) and 0.0 <= res["p_chosen"] <= 1.0
    assert isinstance(res["is_valid_margin"], bool)
    probs = res["probabilities"]
    assert set(probs) == {"candidate_a", "candidate_b"}
    assert probs["candidate_a"] + probs["candidate_b"] == pytest.approx(1.0, abs=1e-6)
    assert probs["candidate_a"] == res["p_chosen"]
    assert (res["preference"] == "candidate_a") == (res["p_chosen"] >= 0.5)


def _check_gate(gate, atol):
    single = [gate.predict_preference(*c) for c in CASES]
    batch = gate.predict_preference_batch(*zip(*CASES, strict=True), batch_size=2)
    for s, b in zip(single, batch, strict=True):
        _check_schema(s)
        _check_schema(b)
        assert s["p_chosen"] == pytest.approx(b["p_chosen"], abs=atol)


# --------------------------------------------------------------- tiny checkpoint


@pytest.fixture(scope="module")
def tiny_gate(tiny_checkpoint):
    return SwedishPreferenceGate(tiny_checkpoint, device="cpu")


def test_tiny_single_and_batch_cpu(tiny_gate):
    _check_gate(tiny_gate, atol=1e-5)


def test_temperature_loaded_from_checkpoint(tiny_gate):
    assert tiny_gate.calibrated and tiny_gate.temperature == 1.5


def test_temperature_scales_probabilities(tiny_checkpoint):
    import numpy as np

    g1 = SwedishPreferenceGate(tiny_checkpoint, device="cpu", temperature=1.0)
    logits = np.array([[2.0, 0.0]])
    assert g1.probabilities(logits)[0, 0] > 0.88
    g2 = SwedishPreferenceGate(tiny_checkpoint, device="cpu", temperature=4.0)
    assert 0.5 < g2.probabilities(logits)[0, 0] < 0.65


def test_margin_semantics(tiny_checkpoint):
    import numpy as np

    gate = SwedishPreferenceGate(tiny_checkpoint, device="cpu", margin=0.2)
    assert gate._result(np.array([0.71, 0.29]))["is_valid_margin"] is True
    assert gate._result(np.array([0.69, 0.31]))["is_valid_margin"] is False
    assert gate._result(np.array([0.2, 0.8]))["is_valid_margin"] is True


def test_invalid_arguments(tiny_checkpoint):
    with pytest.raises(ValueError):
        SwedishPreferenceGate(tiny_checkpoint, device="cpu", margin=0.5)
    with pytest.raises(ValueError):
        SwedishPreferenceGate(tiny_checkpoint, device="cpu", temperature=-1.0)
    gate = SwedishPreferenceGate(tiny_checkpoint, device="cpu")
    with pytest.raises(ValueError):
        gate.predict_preference_batch(["a"], ["b", "c"], ["d"])


def test_missing_temperature_defaults_to_one(tiny_checkpoint, tmp_path):
    import shutil

    copy = tmp_path / "ckpt"
    shutil.copytree(tiny_checkpoint, copy)
    (copy / "temperature.json").unlink()
    gate = SwedishPreferenceGate(copy, device="cpu")
    assert gate.temperature == 1.0 and not gate.calibrated


def test_long_inputs_fit_the_window(tiny_gate):
    long = "Det här är en lång mening med åäö. " * 200
    item = tiny_gate.encode(long, long, long)
    assert item["truncated"] and len(item["ids"]) <= tiny_gate.max_len
    _check_schema(tiny_gate.predict_preference(long, long, "Kort."))


def test_missing_local_path_raises():
    with pytest.raises(FileNotFoundError):
        SwedishPreferenceGate("./does-not-exist", device="cpu")


@pytest.mark.gpu
def test_tiny_gpu(tiny_checkpoint):
    import torch

    gate = SwedishPreferenceGate(tiny_checkpoint, device="cuda", dtype=torch.float32)
    _check_gate(gate, atol=1e-4)


# ------------------------------------------------------------- trained checkpoint


def test_trained_cpu(trained_checkpoint):
    gate = SwedishPreferenceGate(trained_checkpoint, device="cpu")
    _check_gate(gate, atol=1e-4)
    assert json.loads((trained_checkpoint / "temperature.json").read_text())["temperature"] > 0


@pytest.mark.gpu
def test_trained_gpu(trained_checkpoint):
    gate = SwedishPreferenceGate(trained_checkpoint, device="cuda")
    # bf16 with different padding in a batch moves logits slightly.
    _check_gate(gate, atol=2e-2)


def test_resolve_checkpoint_passes_revision(monkeypatch, tmp_path):
    import huggingface_hub

    from laya_gate import predictor

    seen = {}

    def fake_download(repo_id, revision=None, allow_patterns=None):
        seen.update(repo_id=repo_id, revision=revision, patterns=allow_patterns)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    assert predictor.resolve_checkpoint("me/model", revision="v1") == tmp_path
    assert seen["repo_id"] == "me/model" and seen["revision"] == "v1"
    assert "questions.json" in seen["patterns"]
