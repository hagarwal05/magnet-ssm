"""End-to-end smoke tests: train -> checkpoint -> evaluate, on tiny synthetic data, plus the
experiment drivers in miniature. These check plumbing, not accuracy."""
import numpy as np
import pandas as pd
import pytest

from src import data as D
from src.checkpoint import load_checkpoint
from src.evaluate import main as evaluate_main
from src.train import TrainConfig, train


def _write_raw(root, material, n=12, L=1300, seed=0, gain=15.0):
    rng = np.random.default_rng(seed)
    t = np.arange(L)
    for i in (1, 2):
        period = rng.uniform(40, 300, size=(n, 1))
        B = 0.1 * np.sin(2 * np.pi * t[None, :] / period + rng.uniform(0, 6, size=(n, 1)))
        H = gain * np.tanh(B / 0.05) + 0.2 * rng.normal(size=(n, L))
        T = rng.choice([25.0, 50.0, 70.0], size=(n, 1))
        for k, arr in (("B", B), ("H", H), ("T", T)):
            pd.DataFrame(arr).to_csv(root / f"{material}_{i}_{k}.csv", header=False, index=False)


@pytest.fixture
def raw_dir(tmp_path):
    for k, m in enumerate(("S1", "S2", "TG")):
        _write_raw(tmp_path, m, seed=k, gain=15.0 + 5 * k)
    return tmp_path


def test_train_checkpoint_evaluate(raw_dir, tmp_path):
    sp = D.split_raw(D.load_raw_material(raw_dir, "S1"), (0.8, 0.1, 0.1), seed=1)
    tr, va = D.windows_from_raw(sp[0], per_seq=32), D.windows_from_raw(sp[1], per_seq=32)
    cfg = TrainConfig(model="ssm", model_config={"d_model": 4, "n_layers": 1}, epochs=3, batch_size=256, device="cpu")
    res = train(cfg, tr, va, tmp_path / "run", verbose=False)
    model, norm, ck = load_checkpoint(tmp_path / "run" / "model.pt")
    assert ck["n_params"] == res["n_params"]
    assert norm == res["norm"]

    ts = D.make_test_segments(sp[2], seg_len=1000, warmups=(100, 900), name="S1")
    out = tmp_path / "eval"
    # write the segments as an HDF5 pair and run the CLI on them
    import h5py
    with h5py.File(tmp_path / "S1_Testing_padded.h5", "w") as f:
        f["B_seq"], f["H_seq"], f["T"] = ts.B, ts.H_padded, ts.T.reshape(-1, 1)
    with h5py.File(tmp_path / "S1_Testing_true.h5", "w") as f:
        f["B_seq"], f["H_seq"], f["T"] = ts.B, ts.H_true, ts.T.reshape(-1, 1)
    rc = evaluate_main(["--checkpoint", str(tmp_path / "run" / "model.pt"), "--test-dir", str(tmp_path),
                        "--materials", "S1", "--out", str(out), "--save-pred"])
    assert rc == 0
    assert (out / "summary.json").exists() and (out / "H_S1.csv").exists()
    pred = np.loadtxt(out / "H_S1.csv", delimiter=",")
    assert pred.shape == ts.B.shape and np.all(np.isfinite(pred))


def test_rollout_validation_selection(raw_dir, tmp_path):
    from src.train import make_val_segments
    sp = D.split_raw(D.load_raw_material(raw_dir, "S1"), (0.8, 0.1, 0.1), seed=1)
    cfg = TrainConfig(model="official_lstm", epochs=4, batch_size=256, rollout_val=True,
                      rollout_val_every=2, rollout_val_max_seqs=4, device="cpu")
    res = train(cfg, D.windows_from_raw(sp[0], per_seq=16), D.windows_from_raw(sp[1], per_seq=16),
                tmp_path / "r", val_segments=make_val_segments(sp[1], cfg), verbose=False)
    assert res["selection"] == "val_rollout_p95"
    assert res["best_epoch"] in (2, 4)
    hist = pd.read_csv(tmp_path / "r" / "history.csv")
    assert hist["val_rollout_p95"].notna().sum() == 2


def test_cache_flag_disables_cache():
    import argparse
    from src.train import add_train_args, cache_from_args
    ap = argparse.ArgumentParser()
    add_train_args(ap)
    assert cache_from_args(ap.parse_args(["--cache", ""])) is None
    assert str(cache_from_args(ap.parse_args([]))) == "data/cache"


def test_size_sweep_driver(raw_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # any stray relative write lands in tmp, not the repo
    from src.experiments import size_sweep
    out = tmp_path / "sweep"
    args = ["--raw-dir", str(raw_dir), "--materials", "S1", "--preset", "smoke", "--epochs", "2",
            "--batch-size", "512", "--windows-per-seq", "16", "--device", "cpu", "--out", str(out),
            "--cache", ""]
    assert size_sweep.main(args) == 0
    df = pd.read_csv(out / "size_sweep.csv")
    assert len(df) == 3 and set(df["model"]) == {"ssm", "official_lstm"}
    assert (out / "error_vs_params_S1.png").exists()
    assert size_sweep.main(args) == 0          # resume: nothing new to run
    assert len(pd.read_csv(out / "size_sweep.csv")) == 3


def test_cross_material_driver(raw_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from src.experiments import cross_material
    out = tmp_path / "xm"
    args = ["--raw-dir", str(raw_dir), "--sources", "S1", "S2", "--targets", "TG",
            "--d-model", "4", "--n-layers", "1", "--epochs", "2", "--finetune-epochs", "2",
            "--finetune-seqs", "3", "--batch-size", "512", "--windows-per-seq", "16",
            "--device", "cpu", "--out", str(out), "--cache", ""]
    assert cross_material.main(args) == 0
    df = pd.read_csv(out / "cross_material.csv")
    assert set(df["condition"]) == {"zero_shot", "finetune_3", "in_domain"}
    assert (out / "cross_material_seq_nrmse_p95.png").exists()
