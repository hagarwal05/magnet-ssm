"""Metric definitions, checked against the upstream tutorial-3 saved outputs.

``tutorials/tutorial-3/Output/{pred,meas}.csv`` are the predicted-region rows the upstream
notebook wrote (21 sequences each for warm-ups 100/300/500/700/900, ragged widths).
Re-scoring them must reproduce the numbers the notebook printed: mean 16.13 %, p95 33.27 %,
and the per-warm-up means.
"""
import numpy as np
import pytest

from conftest import ref_path
from src import metrics as M


def _ragged(path):
    rows = [np.array([float(x) for x in line.split(",")]) for line in open(path).read().splitlines()]
    return rows


@pytest.fixture(scope="module")
def upstream_rows():
    pred = _ragged(ref_path("tutorials/tutorial-3/Output/pred.csv"))
    meas = _ragged(ref_path("tutorials/tutorial-3/Output/meas.csv"))
    L = 1000
    warm = np.array([L - len(r) for r in meas])
    P = np.zeros((len(meas), L))
    Mm = np.zeros((len(meas), L))
    for i, (p, m) in enumerate(zip(pred, meas)):
        P[i, warm[i]:] = p
        Mm[i, warm[i]:] = m
    return P, Mm, warm


def test_sequence_nrmse_reproduces_tutorial3(upstream_rows):
    P, Mm, warm = upstream_rows
    seq = M.sequence_nrmse(P, Mm, warm)
    assert seq.shape == (105,)
    assert np.mean(seq) == pytest.approx(16.127178, abs=1e-5)
    assert np.percentile(seq, 95) == pytest.approx(33.270399, abs=1e-5)
    printed = {100: 18.74, 300: 20.67, 500: 12.25, 700: 15.57, 900: 13.40}
    for w, want in printed.items():
        assert np.mean(seq[warm == w]) == pytest.approx(want, abs=0.005)


def test_grading_definition_differs(upstream_rows):
    """The organizers' RMS(err)/RMS(meas) is not the same number; both must be reported."""
    P, Mm, warm = upstream_rows
    rms = M.sequence_rms_error(P, Mm, warm)
    assert np.mean(rms) == pytest.approx(13.826399, abs=1e-5)
    assert np.percentile(rms, 95) == pytest.approx(28.311386, abs=1e-5)


def test_sequence_nrmse_matches_notebook_loop():
    rng = np.random.default_rng(0)
    N, L = 12, 50
    meas = rng.normal(size=(N, L)) * 10
    pred = meas + rng.normal(size=(N, L))
    warm = rng.integers(5, 45, size=N)
    got = M.sequence_nrmse(pred, meas, warm)
    for i in range(N):
        p, m = pred[i, warm[i]:], meas[i, warm[i]:]
        rmse = np.sqrt(np.mean((p - m) ** 2))
        norm = np.mean(np.abs(m)) if np.mean(np.abs(m)) != 0 else 1e-8
        assert got[i] == pytest.approx(rmse / norm * 100)


def test_energy_matches_notebook_formula():
    rng = np.random.default_rng(1)
    B = np.cumsum(rng.normal(size=(8, 100)), axis=1)
    Hm = rng.normal(size=(8, 100))
    Hp = Hm + 0.1 * rng.normal(size=(8, 100))
    dB = np.diff(B, axis=1)
    e = lambda H: np.abs(np.sum(0.5 * (H[:, 1:] + H[:, :-1]) * dB, axis=1))
    want = np.sqrt(np.mean((e(Hp) - e(Hm)) ** 2)) / np.mean(np.abs(e(Hm))) * 100
    assert M.energy_nrmse(B, Hp, Hm) == pytest.approx(want)


def test_energy_uses_full_span():
    """The tutorial-3 energy integrates all 1000 steps, warm-up included: it equals the
    trapezoid sum over the whole row, not over the predicted region."""
    rng = np.random.default_rng(2)
    B = np.cumsum(rng.normal(size=(1, 30)), axis=1)
    H = rng.normal(size=(1, 30))
    full = abs(np.sum((H[0, 1:] + H[0, :-1]) / 2 * np.diff(B[0])))
    tail = abs(np.sum((H[0, 11:] + H[0, 10:-1]) / 2 * np.diff(B[0, 10:])))
    assert M.total_energy(B, H)[0] == pytest.approx(full)
    assert M.total_energy(B, H)[0] != pytest.approx(tail)


def test_energy_relative_error_only_counts_predicted_pairs():
    B = np.arange(10.0)[None, :]
    Hm = np.zeros((1, 10))
    Hp = Hm.copy()
    Hp[0, :4] = 100.0           # error confined to the warm-up must not count
    loss = np.array([10.0])
    assert M.energy_relative_error(B, Hp, Hm, np.array([5]), loss=loss)[0] == 0.0
    Hp[0, 7] = 2.0              # pairs (6,7) and (7,8): mid diffs 1 and 1, dB = 1 each
    assert M.energy_relative_error(B, Hp, Hm, np.array([5]), loss=loss)[0] == pytest.approx(20.0)


def test_hazen_percentile_matches_matlab_prctile():
    v = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    # MATLAB: prctile([1 2 3 4 10], 95) = 10 (95% of n=5 -> position 5.25 > n, clamps to max)
    assert M.summarize(v, "hazen")["p95"] == pytest.approx(10.0)
    assert M.summarize(v, "linear")["p95"] == pytest.approx(8.8)


def test_drift_curve_shapes():
    meas = np.ones((3, 30))
    pred = meas.copy()
    pred[:, 20:] = 2.0
    d = M.drift_curve(pred, meas, np.array([10, 10, 25]))
    assert len(d["step"]) == 20
    assert d["n"][0] == 3 and d["n"][-1] == 2
    # step 1: two sequences are still exact (index 10), the third is already off (index 25)
    assert d["abs_err_pct"][0] == pytest.approx(100.0 / 3)
    assert d["abs_err_pct"][-1] == pytest.approx(100.0)


def test_rejects_empty_prediction_region():
    with pytest.raises(ValueError):
        M.sequence_nrmse(np.zeros((1, 5)), np.ones((1, 5)), np.array([5]))
