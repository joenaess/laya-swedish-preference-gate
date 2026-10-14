"""Temperature scaling and ECE."""

from __future__ import annotations

import numpy as np
import pytest

from laya_gate.calibrate import ece, fit_temperature, nll, softmax


def _synthetic(scale: float, n: int = 4000, seed: int = 0):
    """Labels drawn from calibrated logits; the model reports them multiplied by `scale`."""
    rng = np.random.default_rng(seed)
    true_margin = rng.normal(0.0, 1.5, n)
    p_b = 1 / (1 + np.exp(-true_margin))
    labels = (rng.random(n) < p_b).astype(int)
    logits = np.stack([np.zeros(n), true_margin * scale], axis=1)
    return logits, labels


def test_overconfident_logits_recover_t_above_one():
    logits, labels = _synthetic(scale=3.0)
    t = fit_temperature(logits, labels)
    assert t > 1.0
    assert t == pytest.approx(3.0, rel=0.15)


def test_ece_and_nll_decrease_after_calibration():
    logits, labels = _synthetic(scale=4.0)
    t = fit_temperature(logits, labels)
    pre, _ = ece(softmax(logits), labels)
    post, _ = ece(softmax(logits, t), labels)
    assert post < pre
    assert nll(logits, labels, t) < nll(logits, labels)


@pytest.mark.parametrize("scale", [0.2, 1.0, 5.0])
def test_temperature_always_positive(scale):
    logits, labels = _synthetic(scale=scale, n=500, seed=1)
    assert fit_temperature(logits, labels) > 0


def test_underconfident_logits_give_t_below_one():
    logits, labels = _synthetic(scale=0.3)
    assert fit_temperature(logits, labels) < 1.0


def test_ece_bins_are_complete():
    logits, labels = _synthetic(scale=1.0, n=1000)
    value, bins = ece(softmax(logits), labels, n_bins=15)
    assert len(bins) == 15
    assert sum(b["count"] for b in bins) == 1000
    assert 0.0 <= value <= 1.0
    assert bins[0]["lower"] == 0.0 and bins[-1]["upper"] == 1.0


def test_ece_zero_for_perfect_predictions():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]] * 50)
    labels = np.array([0, 1] * 50)
    assert ece(probs, labels)[0] == pytest.approx(0.0)
