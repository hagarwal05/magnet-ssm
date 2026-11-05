"""Leaderboard-comparable run: train one model per final-test material (A to E) on that
material's training data, roll out on the official A-E test set, and score exactly as the
organizers did (Evaluation_Read.m; verified against Result_Final.xlsx).

This mirrors the competition protocol: each team trained per material with whatever data the
organizers released for it (Material A: 105 sequences; B: frequency buckets 1-5 only), then
predicted the official test inputs. Model selection uses a held-out 10% of the training
sequences, never the test set.

Outputs:
  <out>/final_eval.csv              one row per (model, material): official scores + params
  <out>/summary.csv                 mean over A-E per model next to the leaderboard
  <out>/error_vs_params.{png,pdf}   ours vs all 32 teams, same test set and metric
  <out>/models/<model>_<L>/         checkpoint, history, config (KB-sized)
  <runs>/<model>/Result/H_Material <L>.csv   submission-format predictions (large, not in git)

  python -m src.experiments.final_eval --models official_lstm ssm_s ssm_m --device mps
"""
from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .. import data as D
from ..evaluate import evaluate_model
from ..models import build_model, count_parameters
from ..plots import plot_error_vs_params
from ..score_submission import GT_DIR, LEADERBOARD, MATERIALS, score_material
from ..train import (TrainConfig, add_model_args, add_train_args, make_val_segments, train,
                     train_config_from_args)
from ..utils import pick_device
from .common import ResultsTable

MODELS = {
    "official_lstm": ("official_lstm", {}),
    "tutorial_lstm": ("tutorial_lstm", {}),
    "ssm_s": ("ssm", {"d_model": 4, "n_layers": 1, "d_state": 8}),
    "ssm_m": ("ssm", {"d_model": 8, "n_layers": 2, "d_state": 16}),
    "ssm_l": ("ssm", {"d_model": 16, "n_layers": 3, "d_state": 16}),
}
LABELS = {"official_lstm": "Dual-LSTM (official arch.)", "tutorial_lstm": "Dual-LSTM (tutorial arch.)",
          "ssm_s": "S4D small", "ssm_m": "S4D medium", "ssm_l": "S4D large"}


def run_one(key: str, letter: str, cfg: TrainConfig, packed: Path, out: Path, runs: Path, device,
            max_windows: int) -> dict:
    name, mcfg = MODELS[key]
    test = D.find_test_set(GT_DIR, MATERIALS[letter])
    buckets = D.load_raw_material(packed, letter)
    tr_b, va_b = D.split_raw(buckets, (0.9, 0.1), seed=cfg.split_seed)
    # Same window budget for every material: dense sampling where sequences are few
    # (Material A: ~95 training sequences), sparse where they are many.
    n_tr = sum(len(b.T) for b in tr_b)
    per_seq = max(8, int(np.ceil(max_windows / max(n_tr, 1))))
    tr = D.windows_from_raw(tr_b, per_seq=per_seq, seed=cfg.split_seed)
    va = D.windows_from_raw(va_b, per_seq=per_seq, seed=cfg.split_seed + 1)
    test_warmups = tuple(int(w) for w in np.unique(test.warmups))
    vseg = make_val_segments(va_b, cfg, warmups=test_warmups) if cfg.rollout_val else None
    run_dir = runs / key / f"material_{letter}"
    t0 = time.time()
    res = train(replace(cfg, model=name, model_config=mcfg), tr, va, run_dir, val_segments=vseg,
                verbose=False, meta={"material": letter, "protocol": "final_eval"})
    train_s = time.time() - t0

    ev = evaluate_model(res["model"].to(device), res["norm"], test, device=device)
    sub = runs / key / "Result"
    sub.mkdir(parents=True, exist_ok=True)
    np.savetxt(sub / f"H_Material {letter}.csv", ev["H_pred"], delimiter=",", fmt="%.7g")
    # score the written CSV with the leaderboard scorer, exactly as a submission would be
    official = score_material(pd.read_csv(sub / f"H_Material {letter}.csv", header=None).to_numpy(), letter)

    keep = out / "models" / f"{key}_{letter}"
    keep.mkdir(parents=True, exist_ok=True)
    for f in ("model.pt", "history.csv", "config.json"):
        shutil.copy2(run_dir / f, keep / f)
    s = ev["summary"]
    return {
        "model": key, "arch": name, "material": letter, "material_name": MATERIALS[letter],
        "n_params": res["n_params"], "official_seq": official["seq_score"],
        "official_energy": official["energy_score"], "seq_nrmse_p95": s["seq_nrmse_p95"],
        "seq_rms_mean": s["seq_rms_mean"], "nonfinite": official["nonfinite"],
        "n_train_seqs": int(n_tr), "n_train_windows": len(tr), "windows_per_seq": per_seq,
        "val_warmups": "/".join(map(str, test_warmups)),
        "best_epoch": res["best_epoch"], "selection": res["selection"],
        "epochs": cfg.epochs, "train_minutes": round(train_s / 60, 1),
        # fixed columns so every material fits one table (C has a single 1-step warm-up)
        **{f"seq_w{w}": s["by_warmup"].get(w, {}).get("seq_rms_p95", float("nan")) for w in (1, 100, 500, 900)},
    }


def summarize(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    lb = pd.read_csv(LEADERBOARD)
    complete = df.groupby("model").filter(lambda g: g["material"].nunique() == 5)
    ours = (complete.groupby("model")
            .agg(n_params=("n_params", "first"), seq=("official_seq", "mean"), energy=("official_energy", "mean"))
            .reset_index().assign(source="ours"))
    teams = (lb.groupby("team").agg(n_params=("params", "mean"), seq=("seq_err_p95_pct", "mean"),
                                    energy=("energy_err_p95_pct", "mean")).reset_index()
             .rename(columns={"team": "model"}).assign(source="leaderboard"))
    table = pd.concat([ours, teams]).sort_values("seq").reset_index(drop=True)
    table["rank_seq"] = np.arange(1, len(table) + 1)
    table.to_csv(out / "summary.csv", index=False, float_format="%.4f")
    if len(ours):
        sweep = ours.rename(columns={"seq": "official_seq"}).assign(
            model=lambda d: d["model"].map(lambda k: MODELS[k][0]))
        lb_avg = pd.read_csv(LEADERBOARD.parent / "leaderboard_mc2.csv")
        plot_error_vs_params(sweep, out / "error_vs_params", metric="official_seq", leaderboard=lb_avg,
                             test_label="A–E official test", same_test_set=True)
    return table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--materials", nargs="+", default=list(MATERIALS), choices=list(MATERIALS))
    ap.add_argument("--packed", type=Path, default=Path("data/packed"))
    ap.add_argument("--out", type=Path, default=Path("results/final_AtoE"))
    ap.add_argument("--runs", type=Path, default=Path.home() / "magnet-data" / "runs" / "final_AtoE",
                    help="large outputs (predictions); outside the repo by default")
    ap.add_argument("--max-windows", type=int, default=500_000,
                    help="training windows per material (sets windows per sequence)")
    add_model_args(ap)  # only the defaults are used; --models picks each architecture
    add_train_args(ap)
    args = ap.parse_args(argv)

    cfg = train_config_from_args(args)
    device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    table = ResultsTable(args.out / "final_eval.csv", ("model", "material"))
    print(f"device {device}; epochs {cfg.epochs} (lr decay x{cfg.decay_ratio} every {cfg.decay_epoch}); "
          f"selection {'rollout p95 every %d' % cfg.rollout_val_every if cfg.rollout_val else 'val MSE'}", flush=True)
    for key in args.models:
        n = count_parameters(build_model(*MODELS[key][:1], MODELS[key][1] or None))
        for L in args.materials:
            if table.has(model=key, material=L):
                print(f"[skip] {key} {L}", flush=True)
                continue
            print(f"[run ] {key} ({n} params) on Material {L} ({MATERIALS[L]})  {time.strftime('%H:%M')}", flush=True)
            row = run_one(key, L, cfg, args.packed, args.out, args.runs, device, args.max_windows)
            table.add(row)
            print(f"       official seq {row['official_seq']:7.2f}%  energy {row['official_energy']:6.2f}%  "
                  f"(best epoch {row['best_epoch']}, {row['train_minutes']} min)", flush=True)
            summarize(pd.read_csv(table.path), args.out)
    t = summarize(pd.read_csv(table.path), args.out)
    print(t[t.source == "ours"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
