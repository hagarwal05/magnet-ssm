"""Rollout indexing, checked against a line-by-line transcription of the upstream notebook's
``get_dataset`` + ``Test`` loop (tutorials/tutorial-3/Demo_LSTM_Evaluation.ipynb), including
the warm-up < 80 padding branch that the upstream demo data never exercises."""
import numpy as np
import pytest
import torch

from src.data import TestSet
from src.models import TutorialLSTM
from src.normalize import NormParams
from src.rollout import rollout

NORM = NormParams(0.01, 0.1, -0.2, 16.0, 0.1, 16.2, 0.02, 0.097, 44.7, 19.1)


def notebook_rollout(net, B, H_mem, H_true, T, data_length, norm):
    """Transcription of upstream get_dataset() + Test() for ONE equal-warm-up group."""
    data_length_test = H_mem.shape[1]
    if data_length > data_length_test:
        B_ftr = B[:, data_length_test:]
        H_ftr = H_true[:, data_length_test:]
        pad_len = data_length - data_length_test
        B_seq = np.concatenate([np.tile(B[:, [0]], (1, pad_len)), B[:, :data_length_test]], axis=1)
        H_seq = np.concatenate([np.tile(H_mem[:, [0]], (1, pad_len)), H_mem[:, :data_length_test]], axis=1)
    else:
        B_ftr = B[:, data_length_test:]
        H_ftr = H_true[:, data_length_test:]
        B_seq = B[:, data_length_test - data_length:data_length_test]
        H_seq = H_mem[:, data_length_test - data_length:data_length_test]
    in_B = B_seq.copy()
    in_H = H_seq.copy()
    in_B_next = in_B.copy()
    B_scal = B_ftr[:, [0]]
    T_scal = T.copy()
    for i in range(B_ftr.shape[1] - 1):
        in_B_next = np.roll(in_B_next, -1, axis=1)
        in_B_next[:, -1] = B_ftr[:, i]
        B_scal = np.vstack([B_scal, B_ftr[:, [i + 1]]])
        T_scal = np.vstack([T_scal, T])
        in_B = np.vstack([in_B, in_B_next])
        in_H = np.vstack([in_H, np.zeros_like(H_seq)])
    B_scal = norm.norm_Scal(B_scal)
    T_scal = norm.norm_T(T_scal)
    in_B = norm.norm_B(in_B)
    in_H = norm.norm_H(in_H)
    in_B = torch.from_numpy(in_B).float().view(-1, data_length, 1)
    in_H = torch.from_numpy(in_H).float().view(-1, data_length, 1)
    B_scal = torch.from_numpy(B_scal).float().view(-1, 1)
    T_scal = torch.from_numpy(T_scal).float().view(-1, 1)
    n = len(B)
    out_pred = torch.tensor([])
    previous_in_H = None
    with torch.no_grad():
        for s in range(0, in_B.shape[0], n):
            b_in_B, b_in_H, b_B, b_T = in_B[s:s + n], in_H[s:s + n], B_scal[s:s + n], T_scal[s:s + n]
            if previous_in_H is None:
                outputs = net(b_in_B, b_in_H, b_B, b_T)
            else:
                b_in_H = np.roll(previous_in_H.cpu(), -1, axis=1)
                b_in_H[:, -1] = out_pred[:, -1].cpu().numpy().reshape(-1, 1)
                b_in_H = torch.from_numpy(b_in_H).float()
                outputs = net(b_in_B, b_in_H, b_B, b_T)
            out_pred = torch.cat((out_pred, outputs), dim=1)
            previous_in_H = b_in_H
    return norm.denorm_out(out_pred.numpy().astype(np.float64))


def _synthetic_set(n, L, warmups, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(L)
    B = 0.1 * np.sin(2 * np.pi * t[None, :] / rng.uniform(40, 200, size=(n, 1))) + 0.01 * rng.normal(size=(n, L))
    H = 15 * np.tanh(B / 0.05) + rng.normal(size=(n, L))
    T = rng.choice([25.0, 50.0, 70.0], size=(n, 1))
    w = np.repeat(warmups, n // len(warmups))
    Hp = np.where(t[None, :] < w[:, None], H, np.nan)
    return B, H, Hp, T, w


@pytest.mark.parametrize("warmups", [(30, 100, 150), (80,), (1,)])
def test_rollout_matches_notebook_transcription(warmups):
    torch.manual_seed(0)
    net = TutorialLSTM().eval()
    B, H, Hp, T, w = _synthetic_set(6 * len(warmups), 200, warmups)
    ours = rollout(net, TestSet(B, Hp, T, H), NORM).H_pred
    for wk in warmups:
        sel = w == wk
        ref = notebook_rollout(net, B[sel], Hp[sel][:, :wk], H[sel], T[sel], 80, NORM)
        np.testing.assert_allclose(ours[sel][:, wk:], ref, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(ours[sel][:, :wk], H[sel][:, :wk])


def test_rollout_batches_do_not_mix():
    """Splitting a group across batches must not change results."""
    torch.manual_seed(0)
    net = TutorialLSTM().eval()
    B, H, Hp, T, _ = _synthetic_set(10, 150, (100,))
    ts = TestSet(B, Hp, T, H)
    a = rollout(net, ts, NORM).H_pred
    b = rollout(net, ts, NORM, max_batch=3).H_pred
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


def test_rollout_never_sees_future_H():
    """Corrupting the true H after the warm-up must not change predictions."""
    torch.manual_seed(0)
    net = TutorialLSTM().eval()
    B, H, Hp, T, _ = _synthetic_set(4, 150, (100,))
    a = rollout(net, TestSet(B, Hp, T), NORM).H_pred
    H2 = H.copy()
    H2[:, 100:] = 1e6
    b = rollout(net, TestSet(B, Hp, T, H2), NORM).H_pred
    np.testing.assert_array_equal(a, b)
