"""Score a submitted prediction folder (``H_Material {A..E}.csv``, full 1000 steps per row)
against the A to E ground truth, exactly as the organizers' Evaluation_Read.m does, and
compare with the published per-material scores.

  python -m src.score_submission --team Hefei
  python -m src.score_submission --pred-dir results/eval/<run> --name ours   # our own H_*.csv

Our predictions use the same layout: ``src.evaluate --save-pred`` writes ``H_<material>.csv``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .data import find_test_set

ROOT = Path(__file__).resolve().parents[1]
GT_DIR = ROOT / "data" / "Evaluation Tools" / "Ground Truth Data"
TEAMS_DIR = ROOT / "data" / "Competition Result"
LEADERBOARD = ROOT / "results" / "reference" / "leaderboard_mc2_per_material.csv"
MATERIALS = {"A": "3C92", "B": "3C95", "C": "FEC007", "D": "FEC014", "E": "T37"}


def read_prediction(pred_dir: Path, letter: str) -> np.ndarray:
    for name in (f"H_Material {letter}.csv", f"H_{MATERIALS[letter]}.csv", f"H_Material_{letter}.csv"):
        p = pred_dir / name
        if p.exists():
            if p.stat().st_size < 1000 and p.read_text(errors="ignore").startswith("version https://git-lfs"):
                raise FileNotFoundError(f"{p} is an LFS pointer; run: git lfs pull --include=\"{p.parent}/*\"")
            return pd.read_csv(p, header=None, dtype=np.float64).to_numpy()
    raise FileNotFoundError(f"no prediction for Material {letter} in {pred_dir}")


def score_material(pred: np.ndarray, letter: str) -> dict:
    ts = find_test_set(GT_DIR, MATERIALS[letter])
    if pred.shape != ts.H_true.shape:
        raise ValueError(f"Material {letter}: prediction {pred.shape} vs ground truth {ts.H_true.shape}")
    w = ts.warmups
    seq = M.sequence_rms_error(pred, ts.H_true, w)
    ene = M.energy_relative_error(ts.B, pred, ts.H_true, w, loss=ts.loss)
    return {"material": letter, "material_name": MATERIALS[letter], "n": len(pred),
            "seq_score": M.official_score(seq, w), "energy_score": M.official_score(ene, w),
            "nonfinite": int(np.sum(~np.isfinite(pred)))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--team", help="folder name under data/Competition Result/")
    src.add_argument("--pred-dir", type=Path, help="any folder of H_Material {A..E}.csv")
    ap.add_argument("--name", default=None, help="label for --pred-dir")
    ap.add_argument("--materials", nargs="+", default=list(MATERIALS))
    ap.add_argument("--tol", type=float, default=0.01, help="max |ours - published| in % points")
    args = ap.parse_args(argv)

    pred_dir = TEAMS_DIR / args.team / "Result" if args.team else args.pred_dir
    label = args.team or args.name or pred_dir.name
    published = pd.read_csv(LEADERBOARD) if LEADERBOARD.exists() else None
    rows, fails = [], []
    for letter in args.materials:
        r = score_material(read_prediction(pred_dir, letter), letter)
        if args.team and published is not None:
            p = published[(published.team == args.team) & (published.material == letter)]
            if len(p):
                r["published_seq"] = float(p.seq_err_p95_pct.iloc[0])
                r["published_energy"] = float(p.energy_err_p95_pct.iloc[0])
                ok = (abs(r["seq_score"] - r["published_seq"]) <= args.tol * max(1, r["published_seq"] / 100)
                      and abs(r["energy_score"] - r["published_energy"]) <= args.tol * max(1, r["published_energy"] / 100))
                r["match"] = ok
                if not ok:
                    fails.append(letter)
        rows.append(r)
        msg = f"  {label} {letter} ({r['material_name']}): seq {r['seq_score']:8.4f}%  energy {r['energy_score']:8.4f}%"
        if "published_seq" in r:
            msg += f"   published {r['published_seq']:8.4f}% / {r['published_energy']:8.4f}%  {'OK' if r['match'] else 'MISMATCH'}"
        print(msg)
    df = pd.DataFrame(rows)
    print(f"  {label} mean over {len(df)} materials: seq {df.seq_score.mean():.4f}%  energy {df.energy_score.mean():.4f}%")
    if fails:
        print(f"MISMATCH on materials {fails}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
