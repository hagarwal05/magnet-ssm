import numpy as np
import pandas as pd
import pytest

from conftest import ref_path
from src import data as D
from src.normalize import NormParams


def test_load_training_h5_demo_file():
    ws = D.load_training_h5(ref_path("tutorials/tutorial-1/3C90_Training.h5"))
    # the demo file holds 7 buckets x 2500 windows of 80 steps
    assert ws.seq_B.shape == (17500, 80)
    assert ws.seq_H.shape == (17500, 80)
    assert ws.B_next.shape == ws.H_next.shape == ws.T.shape == (17500,)
    assert set(np.unique(ws.T)) <= {25.0, 50.0, 70.0}


def test_fitted_norm_close_to_tutorial1_file():
    """tutorial-1's JSON was produced by fitting on this very file (in float32)."""
    ws = D.load_training_h5(ref_path("tutorials/tutorial-1/3C90_Training.h5"))
    ours = ws.fit_norm(tie=False).to_dict()
    ref = NormParams.from_json(ref_path("tutorials/tutorial-1/Normalization_Params.json")).to_dict()
    for k in ref:
        assert ours[k] == pytest.approx(ref[k], rel=1e-3, abs=1e-6), k


def test_tied_norm_shares_statistics():
    ws = D.load_training_h5(ref_path("tutorials/tutorial-1/3C90_Training.h5"))
    p = ws.fit_norm(tie=True)
    assert (p.mean_Scal, p.std_Scal) == (p.mean_B, p.std_B)
    assert (p.mean_out, p.std_out) == (p.mean_H, p.std_H)
    x = np.array([1.0, -2.0])
    np.testing.assert_allclose(p.out_to_H_space(x), x)


def test_constant_feature_normalizes_to_zero():
    """Single-temperature data (e.g. tutorial-2's N87 bucket) must not produce NaN."""
    rng = np.random.default_rng(0)
    ws = D.WindowSet(rng.normal(size=(50, 80)), rng.normal(size=(50, 80)), rng.normal(size=50),
                     rng.normal(size=50), np.full(50, 25.0), np.zeros(50, dtype=np.int64), np.full(50, "X"))
    p = ws.fit_norm()
    assert p.std_T == 1.0
    tensors = ws.to_tensors(p)
    assert all(np.isfinite(t.numpy()).all() for t in tensors)
    assert float(tensors[3].abs().max()) == 0.0


def test_warmup_lengths():
    H = np.ones((3, 10))
    H[0, 4:] = np.nan
    H[1, 0:] = np.nan
    np.testing.assert_array_equal(D.warmup_lengths(H), [4, 0, 10])


def _raw_bucket(n, L, material="X", bucket=1, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.arange(n) + 1000 * bucket
    return D.RawBucket(material, bucket, rng.normal(size=(n, L)), rng.normal(size=(n, L)),
                       np.full(n, 25.0), ids)


def test_raw_split_is_by_sequence():
    bks = [_raw_bucket(20, 300, bucket=i) for i in (1, 2)]
    tr, va, te = D.split_raw(bks, (0.8, 0.1, 0.1), seed=3)
    ids = [np.concatenate([b.seq_id for b in part]) for part in (tr, va, te)]
    assert sum(len(i) for i in ids) == 40
    assert not (set(ids[0]) & set(ids[1]) or set(ids[0]) & set(ids[2]) or set(ids[1]) & set(ids[2]))
    w = D.windows_from_raw(tr, per_seq=5)
    assert set(np.unique(w.group)) == set(ids[0])


def test_windows_from_raw_alignment():
    b = _raw_bucket(1, 200)
    w = D.windows_from_raw([b], per_seq=10, seed=0)
    for k in range(len(w)):
        # locate the target index from the stored next B (windows are stored as float32)
        m = int(np.flatnonzero(b.B[0].astype(np.float32) == w.B_next[k])[0])
        np.testing.assert_allclose(w.seq_B[k], b.B[0, m - 80:m], rtol=1e-6)
        np.testing.assert_allclose(w.seq_H[k], b.H[0, m - 80:m], rtol=1e-6)
        assert w.H_next[k] == pytest.approx(b.H[0, m], rel=1e-6)


def test_make_test_segments_layout():
    bks = [_raw_bucket(4, 1500, bucket=1)]
    ts = D.make_test_segments(bks, seg_len=1000, warmups=(100, 500, 900), seed=0)
    assert ts.B.shape == (12, 1000)
    np.testing.assert_array_equal(ts.warmups, np.repeat([100, 500, 900], 4))
    # each warm-up block repeats the same segments
    np.testing.assert_array_equal(ts.B[:4], ts.B[4:8])
    assert np.isnan(ts.H_padded[0, 100]) and not np.isnan(ts.H_padded[0, 99])


def test_csv_test_layout(tmp_path):
    rng = np.random.default_rng(0)
    B, H = rng.normal(size=(3, 20)), rng.normal(size=(3, 20))
    Hp = H.copy()
    Hp[:, 5:] = np.nan
    pd.DataFrame(B).to_csv(tmp_path / "A_True_B_seq.csv", header=False, index=False)
    pd.DataFrame(Hp).to_csv(tmp_path / "A_Padded_H_seq.csv", header=False, index=False)
    pd.DataFrame(np.full((3, 1), 25.0)).to_csv(tmp_path / "A_True_T.csv", header=False, index=False)
    ts = D.find_test_set(tmp_path, "A")
    assert ts.H_true is None
    np.testing.assert_array_equal(ts.warmups, [5, 5, 5])
    pd.DataFrame(H).to_csv(tmp_path / "A_True_H_seq.csv", header=False, index=False)
    assert D.find_test_set(tmp_path, "A").H_true is not None


def test_raw_cache_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    for k, arr in (("B", rng.normal(size=(2, 50))), ("H", rng.normal(size=(2, 50))), ("T", np.array([[25.0], [70.0]]))):
        pd.DataFrame(arr).to_csv(tmp_path / f"M_1_{k}.csv", header=False, index=False)
    a = D.load_raw_material(tmp_path, "M", cache=tmp_path / "cache")
    b = D.load_raw_material(tmp_path, "M", cache=tmp_path / "cache")
    assert (tmp_path / "cache" / "M_1.h5").exists()
    np.testing.assert_allclose(a[0].B, b[0].B, rtol=1e-6)
    np.testing.assert_array_equal(b[0].T, [25.0, 70.0])
