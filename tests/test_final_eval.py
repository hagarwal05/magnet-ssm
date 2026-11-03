"""Final-test (materials A to E) data and the organizers' scoring, using the files under
data/Evaluation Tools/ (committed through Git LFS). Skipped when the data is absent."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import metrics as M
from src.data import find_test_set

ROOT = Path(__file__).resolve().parents[1]
GT = ROOT / "data" / "Evaluation Tools" / "Ground Truth Data"
MATERIALS = {"A": ("3C92", 1260), "B": ("3C95", 945), "C": ("FEC007", 1116), "D": ("FEC014", 1242), "E": ("T37", 1080)}


def _need_data():
    probe = GT / "Material A - 3C92" / "3C92_Testing_True_B_seq.csv"
    if not probe.exists() or probe.stat().st_size < 1000:  # absent, or an un-pulled LFS pointer
        pytest.skip("final-test data not present (git lfs pull)")


@pytest.mark.parametrize("letter", sorted(MATERIALS))
def test_final_sets_load(letter):
    _need_data()
    name, n = MATERIALS[letter]
    ts = find_test_set(GT, name)
    assert ts.B.shape == (n, 1000) and ts.H_true is not None and ts.loss is not None
    warm = set(np.unique(ts.warmups).tolist())
    assert warm == ({1} if letter == "C" else {100, 500, 900})
    assert np.all(np.diff(ts.warmups) >= 0)  # one contiguous block per warm-up


def test_perfect_prediction_scores_zero():
    _need_data()
    ts = find_test_set(GT, "3C92")
    s, _ = M.evaluate_predictions(ts.B, ts.H_true, ts.H_true, ts.warmups, loss=ts.loss)
    assert s["official_seq_score"] == 0.0 and s["official_energy_score"] == 0.0


def test_official_score_aggregation():
    """Per-block Hazen p95, averaged over blocks (Evaluation_Read.m)."""
    v = np.concatenate([np.arange(1, 21), 10 * np.arange(1, 21)])
    w = np.repeat([100, 900], 20)
    want = np.mean([np.percentile(np.arange(1, 21), 95, method="hazen"),
                    np.percentile(10 * np.arange(1, 21), 95, method="hazen")])
    assert M.official_score(v, w) == pytest.approx(want)


def test_per_material_leaderboard_matches_published_averages():
    table = ROOT / "results" / "reference" / "leaderboard_mc2_per_material.csv"
    df = pd.read_csv(table)
    pub = pd.read_csv(ROOT / "results" / "reference" / "leaderboard_mc2.csv").set_index("team")
    avg = df.groupby("team")[["seq_err_p95_pct", "energy_err_p95_pct"]].mean()
    teams = avg.index.difference(["Tokyo"])  # README shows a placeholder 800 for Tokyo
    assert len(teams) == 31
    np.testing.assert_allclose(avg.loc[teams, "seq_err_p95_pct"], pub.loc[teams, "seq_err_p95_pct"], atol=0.006)
    np.testing.assert_allclose(avg.loc[teams, "energy_err_p95_pct"], pub.loc[teams, "energy_err_p95_pct"], atol=0.006)
