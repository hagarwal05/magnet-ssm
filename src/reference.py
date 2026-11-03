"""Pinned upstream reference artifacts (minjiechen/magnetchallenge-2 @ ab4913c).

Each entry pairs a weight file with the architecture and the normalization file it belongs
to, plus SHA-256 hashes so a swapped file is caught instead of silently degrading results.
Targets are numbers printed in the upstream notebooks' saved outputs.

Paths are relative to the reference clone (``third_party/magnetchallenge-2``; create it with
``scripts/fetch_reference.sh``). See docs/reference_audit.md for how each pairing was
established.
"""
from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = PROJECT_ROOT / "third_party" / "magnetchallenge-2"

PRETEST_MATERIALS = ("3C90", "3C94", "3E6", "3F4", "77", "78", "N27", "N30", "N49", "N87")


@dataclass(frozen=True)
class ReferenceModel:
    key: str
    model_name: str
    weights: str
    weights_sha256: str
    norm: Optional[str]              # None = pairing not published in the repo
    norm_sha256: Optional[str]
    test_padded: Optional[str] = None
    test_true: Optional[str] = None
    targets: dict = field(default_factory=dict)
    source: str = ""


TUTORIAL3 = ReferenceModel(
    key="tutorial3",
    model_name="tutorial_lstm",
    weights="tutorials/tutorial-3/Model_LSTM.sd",
    weights_sha256="121d2175cc7548de6e46631950c51be1fea234e0caa755333e8a446abfff2feb",
    norm="tutorials/tutorial-3/Normalization_Params.json",
    norm_sha256="3bee88f0206507d57f789ca22a22e32db90569a9dcef36c35235b705808a6e72",
    test_padded="tutorials/tutorial-3/3C90_Testing_padded.h5",
    test_true="tutorials/tutorial-3/3C90_Testing_true.h5",
    # Saved outputs of tutorials/tutorial-3/Demo_LSTM_Evaluation.ipynb (cells 12-13 and the
    # histogram annotation). That notebook's Net prints "Number of parameters: 2399".
    targets={
        "seq_nrmse_mean": 16.13,
        "seq_nrmse_p95": 33.27,
        "energy_nrmse": 39.77666323876542,
        "by_warmup_mean": {100: 18.74, 300: 20.67, 500: 12.25, 700: 15.57, 900: 13.40},
        "n_params": 2399,
    },
    source="tutorials/tutorial-3/Demo_LSTM_Evaluation.ipynb",
)

TUTORIAL1_PRINCETON = ReferenceModel(
    key="tutorial1_princeton",
    model_name="tutorial_lstm",
    weights="tutorials/tutorial-1/Model_LSTM_Princeton.sd",
    weights_sha256="11ebfa56ef5844cb70eb94a082103616b44b5e3e7d3cad5a22c294725997d203",
    norm="tutorials/tutorial-1/Normalization_Params.json",
    norm_sha256="61a417b272d43d79e040d52d8ee80136c6b2a793d182a6b96ebe56b9a534b35c",
    targets={"n_params": 2399},
    source="tutorials/tutorial-1/Demo_LSTM_Testing.ipynb",
)

_PRETEST_SHA = {
    "3C90": "fd3e18d3a3592ee2c1030e29f6625eb0c9cdecc3f929ebc93734d553d1b9e0f7",
    "3C94": "a739348e8ac12926ee8122825b178b2ccfa7b6c1079255ef594dd3025ef5efc9",
    "3E6": "fd811849bda99b8e76933c4e74f2b17dd9e1aa961a13c0e85a9863f6f3038584",
    "3F4": "9f0ebd1d7f5a0234a3905c483a6fd44a858a6c057b1906b01aa61c9ab1714b71",
    "77": "622de15c0f7d1ea7ec99f859e6b1136fe2bb3a151b39f56d5f4981b59eae9ec0",
    "78": "effcea10f213fa9b5f398c43220123e742fa059b076e7caae770229b78ccdbc0",
    "N27": "9595747ceceb166a88562cbb9ba7c60e8c4126439eceee2db8778501258aeaf4",
    "N30": "8e056b5641e5de515d8ed509180d0ebb1c00de9663f60450d4b939c77684fdd9",
    "N49": "7f1e981ce996080872545a8c6c8e0caa238f2c51aeaf50bd2ffe48767d5f670d",
    "N87": "f971514ef5d8d2428d26547f228d62ea306bb689969396befcb266f2a8aaf1ce",
}

# The ten 771-parameter pretest models. Their normalization files are NOT in the repo
# (Test/EvaluationExample.ipynb reads Normalization_Params.json from a private path), so
# norm=None and evaluate.py requires an explicit --norm for them.
PRETEST = {
    m: ReferenceModel(
        key=f"pretest_{m}",
        model_name="official_lstm",
        weights=f"Pretest/Pretest_Models/Model_LSTM_{m}.sd",
        weights_sha256=_PRETEST_SHA[m],
        norm=None,
        norm_sha256=None,
        targets={"n_params": 771},
        source="Test/EvaluationExample.ipynb, Pretest/PreEvaluationExample.ipynb",
    )
    for m in PRETEST_MATERIALS
}

REFERENCES = {r.key: r for r in (TUTORIAL3, TUTORIAL1_PRINCETON, *PRETEST.values())}


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(rel: str, root: Path = REFERENCE_ROOT) -> Path:
    p = root / rel
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run scripts/fetch_reference.sh first")
    return p


def check_hash(path: Path, expected: Optional[str], strict: bool = True) -> None:
    if expected is None:
        return
    got = sha256(path)
    if got != expected:
        msg = f"{path}: sha256 {got[:12]}... does not match the pinned {expected[:12]}..."
        if strict:
            raise ValueError(msg + " (pass strict=False / --allow-hash-mismatch to proceed)")
        warnings.warn(msg)
