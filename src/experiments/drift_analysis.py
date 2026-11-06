"""Within-sequence error vs forecast horizon, for our models and for the MC2 submissions.

The official score aggregates per warm-up block, and the w100 / w500 / w900 blocks are
disjoint sets of sequences, so comparing block scores says nothing about how one forecast
evolves. This analysis instead follows the same sequences along the forecast: it uses the
w100 block (100 known steps, 900 predicted) of materials A, B, D, E (C has a 1-step warm-up)
and reports, per span of steps after the boundary, the p95 and median over sequences of

    RMS(H_pred - H_meas) over the span / RMS(H_meas) over the whole predicted region

(a fixed per-sequence denominator, so the windows are comparable), averaged over materials.
A per-window-denominator variant checks that the growth is not an artefact of that choice.

Method by reviewer-teams (literature-review team, 2026-11-06); reproduced independently by
the lead and by winner-others. Inputs: submission-format H_Material <L>.csv files (ours are
written by src.experiments.final_eval; teams' are in data/Competition Result/, Git LFS).

  python -m src.experiments.drift_analysis --out results/drift
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import find_test_set
from ..score_submission import GT_DIR, MATERIALS, TEAMS_DIR, read_prediction

MATS = ("A", "B", "D", "E")
TEAMS = ("Hefei", "Fuzhou", "Southeast1", "Eindhoven", "Bristol", "SiegenPaderborn", "Tsinghua",
         "XJTU1", "Sydney", "NITC", "Tianjin", "Gatech", "Cambridge", "Leuven")
OURS = {"S4D-205": "ssm_s", "S4D-1001": "ssm_m", "LSTM-771": "official_lstm"}
WINDOWS = [(100 * k, 100 * (k + 1)) for k in range(9)]           # steps after the boundary
EARLY = [(0, 1), (0, 10), (10, 25), (25, 50), (50, 100), (800, 900)]


def span_errors(pred: np.ndarray, H: np.ndarray, spans, per_window_den: bool = False) -> np.ndarray:
    """(n_seq, n_spans) relative RMS error per span. pred, H: predicted region only."""
    den = np.sqrt((H ** 2).mean(1))
    out = []
    for a, b in spans:
        num = np.sqrt(((pred[:, a:b] - H[:, a:b]) ** 2).mean(1))
        d = np.sqrt((H[:, a:b] ** 2).mean(1)) if per_window_den else den
        out.append(num / np.where(d > 0, d, 1e-12))
    return np.stack(out, 1)


def profile(pred_dir: Path, spans, tests: dict, per_window_den: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """p95 and median over sequences per span, each averaged over MATS."""
    p95, med = [], []
    for L in MATS:
        ts = tests[L]
        sel = ts.warmups == 100
        P = read_prediction(pred_dir, L)[sel, 100:]
        e = span_errors(P, ts.H_true[sel, 100:], spans, per_window_den)
        p95.append(np.percentile(e, 95, axis=0) * 100)
        med.append(np.median(e, axis=0) * 100)
    return np.mean(p95, 0), np.mean(med, 0)


def sources(runs_root: Path, seeds=(1, 2, 3), tag: str = "_hn0.1_ms32") -> dict:
    src = {t: TEAMS_DIR / t / "Result" for t in TEAMS}
    for label, key in OURS.items():
        for s in seeds:
            src[f"{label} s{s}"] = runs_root / f"seed{s}" / f"{key}{tag}" / "Result"
    return src


def table(src: dict, spans, tests: dict, per_window_den: bool = False) -> pd.DataFrame:
    rows = []
    for name, d in src.items():
        p95, med = profile(d, spans, tests, per_window_den)
        row = {"source": name}
        for (a, b), x, m in zip(spans, p95, med):
            row[f"p95_{a + 1}-{b}"] = x
            row[f"med_{a + 1}-{b}"] = m
        rows.append(row)
    df = pd.DataFrame(rows)
    # seed means for our models
    for label in OURS:
        sub = df[df.source.str.startswith(label + " s")]
        if len(sub):
            mean = sub.drop(columns="source").mean().to_dict()
            df = pd.concat([df, pd.DataFrame([{"source": f"{label} (seed mean)", **mean}])], ignore_index=True)
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path.home() / "magnet-data" / "runs" / "seeds3",
                    help="our seeds3 prediction root (written by final_eval)")
    ap.add_argument("--out", type=Path, default=Path("results/drift"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    tests = {L: find_test_set(GT_DIR, MATERIALS[L]) for L in MATS}
    src = sources(args.runs)

    win = table(src, WINDOWS, tests)
    win["growth_9_over_1"] = win["p95_801-900"] / win["p95_1-100"]
    win["added_pts_9_minus_1"] = win["p95_801-900"] - win["p95_1-100"]
    win.to_csv(args.out / "windows.csv", index=False, float_format="%.2f")

    early = table(src, EARLY, tests)
    early.to_csv(args.out / "early_steps.csv", index=False, float_format="%.2f")

    chk = table(src, WINDOWS, tests, per_window_den=True)
    chk["growth_9_over_1"] = chk["p95_801-900"] / chk["p95_1-100"]
    chk.to_csv(args.out / "windows_per_window_denominator.csv", index=False, float_format="%.2f")

    show = ["source", "p95_1-100", "p95_401-500", "p95_801-900", "growth_9_over_1", "added_pts_9_minus_1"]
    print(win[show].round(2).to_string(index=False))
    print(early[["source", "p95_1-1", "p95_1-10", "p95_51-100"]].round(2).to_string(index=False))
    print(f"wrote {args.out}/windows.csv, early_steps.csv, windows_per_window_denominator.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
