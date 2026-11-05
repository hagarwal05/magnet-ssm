"""Multi-step (rollout) training: the training-time rollout must be the same computation as
the test-time rollout in src.rollout, or we would optimise something we never score."""
import numpy as np
import pytest
import torch

from src import data as D
from src.models import build_model
from src.normalize import NormParams
from src.rollout import rollout
from src.train import TrainConfig, current_horizon, rollout_loss, train

# tied normalization (mean/std of out == H, Scal == B), as all our models use
NORM = NormParams(0.01, 0.1, -0.5, 16.0, -0.5, 16.0, 0.01, 0.1, 44.7, 19.1)


def _bucket(n=6, L=400, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(L)
    B = 0.1 * np.sin(2 * np.pi * t[None, :] / rng.uniform(40, 200, size=(n, 1)))
    H = 15 * np.tanh(B / 0.05) + rng.normal(size=(n, L))
    return D.RawBucket("X", 1, B.astype(np.float32).astype(np.float64), H.astype(np.float32).astype(np.float64),
                       rng.choice([25.0, 50.0, 70.0], size=n), np.arange(n))


@pytest.mark.parametrize("name,cfg", [("ssm", {"d_model": 4, "n_layers": 1, "d_state": 8}), ("official_lstm", None)])
def test_training_rollout_equals_test_rollout(name, cfg):
    torch.manual_seed(0)
    model = build_model(name, cfg).eval()
    K = 30
    ch = D.chunks_from_raw([_bucket()], horizon=K, per_seq=2, seed=3)
    B_win, B_next, H, T = ch.to_tensors(NORM)
    # predictions from the training-time loop
    preds, hist = [], H[:, :80].unsqueeze(-1)
    with torch.no_grad():
        for j in range(K):
            t = 80 + j
            y = model(B_win[:, t - 80:t].unsqueeze(-1), hist, B_next[:, t:t + 1], T)
            preds.append(y[:, 0])
            hist = torch.cat((hist[:, 1:], y.unsqueeze(1)), dim=1)
    train_pred = NORM.denorm_out(torch.stack(preds, 1).numpy().astype(np.float64))
    # the same chunks through the evaluation rollout (warm-up = 80 true steps)
    Hp = ch.H.astype(np.float64).copy()
    Hp[:, 80:] = np.nan
    ts = D.TestSet(ch.B.astype(np.float64), Hp, ch.T, ch.H.astype(np.float64))
    test_pred = rollout(model, ts, NORM).H_pred[:, 80:]
    # Step 1 sees identical inputs (up to float32 rounding of the normalization, ~1e-7) and
    # must agree exactly; later steps feed predictions back, so that rounding compounds on an
    # untrained (unstable) model. 1% of the signal scale over 30 steps is still far below
    # any logic error (a shifted index or wrong feedback unit gives O(100%) differences).
    np.testing.assert_allclose(train_pred[:, 0], test_pred[:, 0], atol=1e-5)
    scale = np.mean(np.abs(test_pred))
    assert np.max(np.abs(train_pred - test_pred)) < 0.01 * scale


def test_k1_rollout_loss_equals_next_step_mse():
    torch.manual_seed(0)
    model = build_model("ssm", {"d_model": 4, "n_layers": 1, "d_state": 8}).eval()
    ch = D.chunks_from_raw([_bucket()], horizon=1, per_seq=4, seed=1)
    B_win, B_next, H, T = ch.to_tensors(NORM)
    with torch.no_grad():
        y = model(B_win[:, :80].unsqueeze(-1), H[:, :80].unsqueeze(-1), B_next[:, 80:81], T)
        want = float(((y[:, 0] - H[:, 80]) ** 2).mean())
        got = float(rollout_loss(model, B_win, B_next, H, T, 80, 1))
    assert got == pytest.approx(want, rel=1e-6)


def test_gradients_flow_through_rollout():
    torch.manual_seed(0)
    model = build_model("ssm", {"d_model": 4, "n_layers": 1, "d_state": 8})
    ch = D.chunks_from_raw([_bucket()], horizon=8, per_seq=2, seed=1)
    loss = rollout_loss(model, *ch.to_tensors(NORM), 80, 8)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_horizon_curriculum():
    cfg = TrainConfig(horizon=33, horizon_warmup_epochs=4)
    assert [current_horizon(cfg, e) for e in range(6)] == [9, 17, 25, 33, 33, 33]
    assert current_horizon(TrainConfig(horizon=16), 0) == 16


def test_multistep_train_end_to_end(tmp_path):
    tr_b, va_b = _bucket(seed=0), _bucket(seed=1)
    cfg = TrainConfig(model="ssm", model_config={"d_model": 4, "n_layers": 1, "d_state": 8},
                      epochs=2, batch_size=8, horizon=10, device="cpu")
    res = train(cfg, D.chunks_from_raw([tr_b], 10, per_seq=4), D.chunks_from_raw([va_b], 10, per_seq=2),
                tmp_path, verbose=False, norm_source=D.windows_from_raw([tr_b], per_seq=16))
    assert res["best_epoch"] in (1, 2)
    import pandas as pd
    assert list(pd.read_csv(tmp_path / "history.csv")["horizon"]) == [10, 10]
    with pytest.raises(ValueError):  # horizon > 1 with next-step data is a config error
        train(TrainConfig(horizon=5, epochs=1, device="cpu"), D.windows_from_raw([tr_b], per_seq=4),
              D.windows_from_raw([va_b], per_seq=4), tmp_path / "x", verbose=False)
