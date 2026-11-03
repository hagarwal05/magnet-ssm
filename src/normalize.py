"""Normalization parameters (the upstream ``Normalization_Params.json`` scheme).

Five (mean, std) pairs z-score the model inputs and target:
``B`` / ``H`` for the 80-step history windows, ``Scal`` for the next-B input,
``out`` for the next-H target, and ``T`` for temperature.

The parameters are part of the model. A weight file is only meaningful with the
normalization it was trained with, and the two upstream JSON files are not
interchangeable. Our checkpoints embed their parameters (``checkpoint.py``); the
pairings for the upstream reference weights are pinned in ``reference.py``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

FIELDS = (
    "mean_B", "std_B",
    "mean_H", "std_H",
    "mean_out", "std_out",
    "mean_Scal", "std_Scal",
    "mean_T", "std_T",
)


@dataclass(frozen=True)
class NormParams:
    mean_B: float
    std_B: float
    mean_H: float
    std_H: float
    mean_out: float
    std_out: float
    mean_Scal: float
    std_Scal: float
    mean_T: float
    std_T: float

    # ---- construction / serialization -------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict) -> "NormParams":
        missing = [k for k in FIELDS if k not in d]
        if missing:
            raise KeyError(f"normalization parameters missing fields: {missing}")
        return cls(**{k: float(d[k]) for k in FIELDS})

    @classmethod
    def from_json(cls, path: str | Path) -> "NormParams":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def fit(cls, seq_B, seq_H, B_next, H_next, T) -> "NormParams":
        """Statistics computed the way tutorial-1 ``Demo_LSTM_Training`` computes them:
        a global mean and unbiased std over every element of each array.

        (The notebook reduces the window arrays in float32; we reduce in float64, which
        changes the values only in the ~7th significant digit.)
        """

        def mean_std(x):
            x = np.asarray(x, dtype=np.float64)
            std = float(x.std(ddof=1)) if x.size > 1 else 0.0
            # A constant input (e.g. a single-temperature training set) has std 0; use 1 so
            # the feature becomes 0 instead of NaN.
            return float(x.mean()), (std if std > 1e-12 and np.isfinite(std) else 1.0)

        mB, sB = mean_std(seq_B)
        mH, sH = mean_std(seq_H)
        mo, so = mean_std(H_next)
        ms, ss = mean_std(B_next)
        mT, sT = mean_std(T)
        return cls(mB, sB, mH, sH, mo, so, ms, ss, mT, sT)

    # ---- transforms (work on numpy arrays and torch tensors alike) ---------------------

    def norm_B(self, x):
        return (x - self.mean_B) / self.std_B

    def norm_H(self, x):
        return (x - self.mean_H) / self.std_H

    def norm_Scal(self, x):
        return (x - self.mean_Scal) / self.std_Scal

    def norm_T(self, x):
        return (x - self.mean_T) / self.std_T

    def norm_out(self, x):
        return (x - self.mean_out) / self.std_out

    def denorm_out(self, y):
        return y * self.std_out + self.mean_out

    def out_to_H_space(self, y):
        """Re-express an ``out``-normalized prediction in ``H``-window units."""
        return self.norm_H(self.denorm_out(y))

    def H_to_out_space(self, h):
        """Re-express an ``H``-window value in ``out``-normalized units."""
        return self.norm_out(h * self.std_H + self.mean_H)
