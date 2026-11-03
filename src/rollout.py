"""Autoregressive rollout over 1000-step test segments with a variable warm-up.

This mirrors ``get_dataset`` + ``Test`` in tutorials/tutorial-3/Demo_LSTM_Evaluation.ipynb:

* Each sequence's warm-up ``w`` is the index of its first NaN in the padded H.
  Sequences are processed in groups of equal warm-up (the notebook's segmentation).
* Predicting index ``t`` (t = w, w+1, ..., L-1) uses the B window ``B[t-W : t]``, the next B
  ``B[t]``, temperature, and an H window of the ``W`` most recent H values (true H inside
  the warm-up, the model's own predictions afterwards).
* If ``w < W`` the windows are left-padded with each sequence's first value, as the
  notebook does (relevant for Material C, which provides only H(1) = 0).
* Model inputs are normalized in float64 and cast to float32; the model runs in float32.

One upstream quirk is reproduced by default: the notebook writes the model output, which is
in ``out``-normalized units, straight into the H window, which is in ``H``-normalized units.
``convert_feedback=True`` converts it properly instead. The two are identical when
``mean_out == mean_H`` and ``std_out == std_H``, which holds for every model trained here
(``WindowSet.fit_norm(tie=True)``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import WINDOW, TestSet
from .normalize import NormParams


@dataclass
class RolloutResult:
    H_pred: np.ndarray   # (N, L) true warm-up prefix + predictions, denormalized
    warmups: np.ndarray  # (N,)
    nonfinite: int       # number of predicted values that were NaN/inf (diverged rollouts)


def _contiguous_groups(warmups: np.ndarray) -> list[tuple[int, int]]:
    change = np.flatnonzero(np.diff(warmups)) + 1
    starts = np.concatenate([[0], change])
    stops = np.concatenate([change, [len(warmups)]])
    return list(zip(starts.tolist(), stops.tolist()))


@torch.no_grad()
def rollout(model: torch.nn.Module, test: TestSet, norm: NormParams, window: int = WINDOW,
            device: str | torch.device = "cpu", convert_feedback: bool = False,
            max_batch: int = 8192) -> RolloutResult:
    """Predict the NaN part of every padded H sequence autoregressively."""
    model.eval()
    device = torch.device(device)
    B, H_pad, T = test.B, test.H_padded, test.T
    N, L = B.shape
    warmups = test.warmups
    if np.any(warmups >= L):
        raise ValueError("some sequences have no NaN region to predict")
    if np.any(np.isnan(B)):
        raise ValueError("B contains NaN; the full excitation must be known")

    H_pred = H_pad.copy()
    nonfinite = 0

    def f32(x):
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)

    for g0, g1 in _contiguous_groups(warmups):
        w = int(warmups[g0])
        for b0 in range(g0, g1, max_batch):
            b1 = min(g1, b0 + max_batch)
            Bg, Hg, Tg = B[b0:b1], H_pad[b0:b1], T[b0:b1]
            n = b1 - b0
            pad = max(0, window - w)
            if w == 0:
                # No H history at all: assume zero (the Webinar-6 convention for Material C).
                Hg = np.zeros_like(Hg)
            B_ext = np.concatenate([np.repeat(Bg[:, :1], pad, axis=1), Bg], axis=1)
            H_hist = np.concatenate([np.repeat(Hg[:, :1], pad, axis=1), Hg[:, :w]], axis=1)[:, -window:]

            Bn = f32(norm.norm_B(B_ext))                    # (n, pad + L)
            Bs = f32(norm.norm_Scal(Bg))                    # (n, L)
            Tn = f32(norm.norm_T(Tg)).reshape(n, 1)
            hist = f32(norm.norm_H(H_hist)).reshape(n, window, 1)

            preds = torch.empty((n, L - w), dtype=torch.float32, device=device)
            for k, t in enumerate(range(w, L)):
                seq_B = Bn[:, pad + t - window: pad + t].unsqueeze(-1)
                y = model(seq_B, hist, Bs[:, t:t + 1], Tn)  # (n, 1), out-normalized
                preds[:, k] = y[:, 0]
                fb = norm.out_to_H_space(y) if convert_feedback else y
                hist = torch.cat((hist[:, 1:, :], fb.reshape(n, 1, 1)), dim=1)

            y_np = norm.denorm_out(preds.cpu().numpy().astype(np.float64))
            nonfinite += int(np.sum(~np.isfinite(y_np)))
            H_pred[b0:b1, w:] = y_np

    return RolloutResult(H_pred=H_pred, warmups=warmups, nonfinite=nonfinite)
