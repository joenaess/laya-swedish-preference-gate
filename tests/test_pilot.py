"""Offline tests for the EuroEval pilot helpers."""

import pytest

from laya_gate import pilot


def test_chrf_bounds_and_order():
    ref = "Katten sitter på mattan."
    assert pilot.chrf(ref, ref) == pytest.approx(100.0)
    assert pilot.chrf("", ref) == 0.0
    close = pilot.chrf("Katten sitter på en matta.", ref)
    far = pilot.chrf("Hunden springer i parken.", ref)
    assert 0 < far < close < 100


def test_is_swedish():
    assert pilot.is_swedish("Det är en fin dag och vi går ut.")
    assert not pilot.is_swedish("It is a nice day and we go out.")
    assert not pilot.is_swedish("12345")


def test_clean_output_strips_reasoning():
    assert pilot.clean_output("<think>hmm</think>\nHej!") == "Hej!"
    assert pilot.clean_output("plan</think>Svar") == "Svar"


def test_leaderboard_parsing_helpers():
    cell = "<a href='https://hf.co/x/y'>x/y (val)</a>"
    assert pilot._name(cell) == "x/y (val)"
    assert pilot._num("2.37@@2.37") == 2.37
    assert pilot._num("?") is None
    board = {"a/b": {"rank": 100.0}, "a/b (val)": {"rank": 2.0}, "c/d": {"rank": 3.0}}
    assert pilot.leaderboard_row(board, "a/b")["leaderboard_name"] == "a/b (val)"
    assert pilot.leaderboard_row(board, "c/d")["rank"] == 3.0
    assert pilot.leaderboard_row(board, "e/f") is None


def test_spearman_ci():
    r = pilot.spearman_ci([1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12])
    assert r["rho"] == pytest.approx(1.0) and r["n_models"] == 6
    r = pilot.spearman_ci([1, 2, float("nan"), 4], [1, 2, 3, 4])
    assert r["n_models"] == 3


def test_partial_spearman_removes_a_shared_cause():
    import numpy as np

    rng = np.random.default_rng(0)
    z = np.arange(30, dtype=float)
    x = z + rng.normal(0, 4, 30)  # x and y both just follow z
    y = z + rng.normal(0, 4, 30)
    assert pilot.spearman_ci(x, y)["rho"] > 0.6
    assert abs(pilot.partial_spearman_ci(x, y, z)["rho"]) < 0.4
