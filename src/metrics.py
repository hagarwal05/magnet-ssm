"""Evaluation metrics, transcribed from the upstream code.

All functions take full-length arrays of shape (N, L) plus the per-sequence warm-up
length ``warmups`` (N,). ``H_pred`` holds the true warm-up prefix followed by the model's
predictions, the same layout ``Demo_LSTM_Evaluation.ipynb`` builds (``all_y_pred``).

Primary metrics (the project's headline, from tutorials/tutorial-3/Demo_LSTM_Evaluation.ipynb)
  * ``sequence_nrmse``: per sequence, over the predicted region only,
    RMSE / mean(|H_meas|) * 100. Headline = 95th percentile (``np.percentile``, linear).
  * ``energy_nrmse``: one number for the whole set. Energy = |sum(H_mid * dB)| over the
    FULL sequence (warm-up prefix included).

Secondary metrics (the organizers' grading definitions: Webinar-6 slides,
Pre_Evaluation_Rules.pdf, Test/Error_Distribution_Plot_Template.m)
  * ``sequence_rms_error``: RMS(H_pred - H_meas) / RMS(H_meas) * 100, predicted region.
  * ``energy_relative_error``: per sequence, (integral of dB*H_pred - integral of dB*H_meas)
    over the predicted segment, divided by |integral of dB*H_meas| over the full excitation
    (the file's ``Loss`` field when present, else the measured segment itself).

See docs/reference_audit.md for why both families are reported.
"""
from __future__ import annotations

import numpy as np

PERCENTILE = 95


def region_mask(warmups, length: int) -> np.ndarray:
    """Boolean (N, L) mask that is True on the predicted region (step index >= warm-up)."""
    warmups = np.asarray(warmups).reshape(-1, 1)
    return np.arange(length)[None, :] >= warmups


def _region_counts(mask: np.ndarray) -> np.ndarray:
    n = mask.sum(axis=1)
    if np.any(n == 0):
        raise ValueError(
            f"{int(np.sum(n == 0))} sequence(s) have no predicted region (warm-up == length); "
            "drop them before computing metrics"
        )
    return n


def sequence_nrmse(H_pred, H_meas, warmups) -> np.ndarray:
    """Per-sequence NRMSE [%] exactly as tutorial-3: rmse / mean(|meas|), predicted region."""
    H_pred = np.asarray(H_pred, dtype=np.float64)
    H_meas = np.asarray(H_meas, dtype=np.float64)
    m = region_mask(warmups, H_meas.shape[1])
    n = _region_counts(m)
    rmse = np.sqrt(np.where(m, (H_pred - H_meas) ** 2, 0.0).sum(axis=1) / n)
    norm = np.where(m, np.abs(H_meas), 0.0).sum(axis=1) / n
    norm = np.where(norm != 0, norm, 1e-8)
    return rmse / norm * 100.0


def sequence_rms_error(H_pred, H_meas, warmups) -> np.ndarray:
    """Per-sequence relative error [%] as graded: RMS(pred - meas) / RMS(meas), predicted region."""
    H_pred = np.asarray(H_pred, dtype=np.float64)
    H_meas = np.asarray(H_meas, dtype=np.float64)
    m = region_mask(warmups, H_meas.shape[1])
    n = _region_counts(m)
    rms_err = np.sqrt(np.where(m, (H_pred - H_meas) ** 2, 0.0).sum(axis=1) / n)
    rms_meas = np.sqrt(np.where(m, H_meas ** 2, 0.0).sum(axis=1) / n)
    rms_meas = np.where(rms_meas != 0, rms_meas, 1e-8)
    return rms_err / rms_meas * 100.0


def total_energy(B, H) -> np.ndarray:
    """Open-loop energy density |sum(H_mid * dB)| per sequence [J/m^3] (tutorial-3)."""
    B = np.asarray(B, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    dB = np.diff(B, axis=1)
    H_mid = (H[:, :-1] + H[:, 1:]) / 2
    return np.abs(np.sum(H_mid * dB, axis=1))


def energy_nrmse(B, H_pred, H_meas) -> float:
    """Set-level energy NRMSE [%] exactly as tutorial-3 (full span, warm-up included)."""
    e_pred = total_energy(B, H_pred)
    e_meas = total_energy(B, H_meas)
    return float(np.sqrt(np.mean((e_pred - e_meas) ** 2)) / np.mean(np.abs(e_meas)) * 100.0)


def energy_relative_error(B, H_pred, H_meas, warmups, loss=None) -> np.ndarray:
    """Per-sequence signed energy error [%] following the official grading definition.

    Numerator: trapezoidal integral of dB * (H_pred - H_meas) over the predicted segment
    (pairs of samples that both lie in the predicted region, as the MATLAB template does).
    Denominator: |loss| when given (``Loss`` = the full-excitation core loss stored with the
    official test files), otherwise |signed measured energy of the whole segment|.
    """
    B = np.asarray(B, dtype=np.float64)
    H_pred = np.asarray(H_pred, dtype=np.float64)
    H_meas = np.asarray(H_meas, dtype=np.float64)
    dB = np.diff(B, axis=1)
    pair_mask = region_mask(warmups, B.shape[1] - 1)
    mid_diff = ((H_pred[:, :-1] + H_pred[:, 1:]) - (H_meas[:, :-1] + H_meas[:, 1:])) / 2
    num = np.where(pair_mask, mid_diff * dB, 0.0).sum(axis=1)
    if loss is None:
        denom = np.sum((H_meas[:, :-1] + H_meas[:, 1:]) / 2 * dB, axis=1)
    else:
        denom = np.asarray(loss, dtype=np.float64).reshape(-1)
    denom = np.abs(denom)
    denom = np.where(denom != 0, denom, 1e-8)
    return num / denom * 100.0


def summarize(values, method: str = "linear") -> dict:
    """Mean, 95th percentile and max of |values|.

    ``method="linear"`` is NumPy's default (what tutorial-3 uses). MATLAB's ``prctile``
    (used by the organizers' plotting templates) corresponds to ``method="hazen"``.
    """
    v = np.abs(np.asarray(values, dtype=np.float64))
    return {
        "mean": float(np.mean(v)),
        "p95": float(np.percentile(v, PERCENTILE, method=method)),
        "max": float(np.max(v)),
        "n": int(v.size),
    }


def drift_curve(H_pred, H_meas, warmups) -> dict:
    """Error growth with horizon: for k steps after the warm-up ends, the mean over sequences
    of |pred - meas| / mean(|meas| over that sequence's predicted region), in percent."""
    H_pred = np.asarray(H_pred, dtype=np.float64)
    H_meas = np.asarray(H_meas, dtype=np.float64)
    warmups = np.asarray(warmups).reshape(-1)
    N, L = H_meas.shape
    horizon = L - int(warmups.min())
    total = np.zeros(horizon)
    count = np.zeros(horizon, dtype=np.int64)
    for i in range(N):
        w = int(warmups[i])
        meas = H_meas[i, w:]
        if meas.size == 0:
            continue
        scale = np.mean(np.abs(meas)) or 1e-8
        err = np.abs(H_pred[i, w:] - meas) / scale * 100.0
        total[: err.size] += err
        count[: err.size] += 1
    with np.errstate(invalid="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return {"step": np.arange(1, horizon + 1), "abs_err_pct": mean, "n": count}


def evaluate_predictions(B, H_meas, H_pred, warmups, loss=None, percentile_method: str = "linear") -> tuple[dict, dict]:
    """All metrics for one test set. Returns (summary, per_sequence arrays)."""
    warmups = np.asarray(warmups).reshape(-1)
    seq = sequence_nrmse(H_pred, H_meas, warmups)
    seq_rms = sequence_rms_error(H_pred, H_meas, warmups)
    ene_rel = energy_relative_error(B, H_pred, H_meas, warmups, loss=loss)

    s_seq = summarize(seq, percentile_method)
    s_rms = summarize(seq_rms, percentile_method)
    s_ene = summarize(ene_rel, percentile_method)
    summary = {
        "n_sequences": int(seq.size),
        # headline family (tutorial-3 definitions)
        "seq_nrmse_p95": s_seq["p95"],
        "seq_nrmse_mean": s_seq["mean"],
        "seq_nrmse_max": s_seq["max"],
        "energy_nrmse": energy_nrmse(B, H_pred, H_meas),
        # official grading family
        "seq_rms_p95": s_rms["p95"],
        "seq_rms_mean": s_rms["mean"],
        "energy_rel_p95": s_ene["p95"],
        "energy_rel_mean": s_ene["mean"],
        "energy_rel_denominator": "loss" if loss is not None else "segment",
        "percentile_method": percentile_method,
        "by_warmup": {},
    }
    for w in sorted(np.unique(warmups)):
        sel = warmups == w
        summary["by_warmup"][int(w)] = {
            "n": int(sel.sum()),
            "seq_nrmse_mean": float(np.mean(seq[sel])),
            "seq_nrmse_p95": float(np.percentile(seq[sel], PERCENTILE, method=percentile_method)),
            "seq_rms_p95": float(np.percentile(seq_rms[sel], PERCENTILE, method=percentile_method)),
        }
    per_seq = {
        "warmup": warmups,
        "seq_nrmse": seq,
        "seq_rms": seq_rms,
        "energy_rel": ene_rel,
        "energy_pred": total_energy(B, H_pred),
        "energy_meas": total_energy(B, H_meas),
    }
    return summary, per_seq
