"""RQ1: accuracy vs model size.

Trains every (architecture, width, depth, seed) combination on the same sequence-level
split of each material, scores each one by autoregressive rollout on the held-out
sequences, and writes one tidy row per run to ``<out>/size_sweep.csv``. The table can be
resumed: re-running skips rows that are already present.

The dual-LSTM baselines are retrained on the identical split as fixed-size references.

  # quick proof-of-life on CPU/MPS (a few minutes)
  python -m src.experiments.size_sweep --raw-dir data/raw --materials 3C90 --preset smoke

  # the real sweep (GPU)
  python -m src.experiments.size_sweep --raw-dir data/raw --materials 3C90 N87 3F4 \
      --preset full --seeds 1 2 3 --device cuda
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .. import data as D
from ..models import SSMConfig, build_model, count_parameters
from ..plots import plot_error_vs_params
from ..train import (TrainConfig, add_model_args, add_train_args, cache_from_args, load_material_splits,
                     make_val_segments, train, train_config_from_args)
from ..utils import pick_device
from .common import ResultsTable, score

PRESETS = {
    # (d_model, n_layers) grid; d_state and head are held fixed within a preset
    "smoke": {"grid": [(4, 1), (8, 2)], "d_state": 8, "head_hidden": 8, "baselines": ["official_lstm"]},
    "full": {"grid": list(itertools.product([4, 6, 8, 12, 16, 24, 32], [1, 2, 3])), "d_state": 16,
             "head_hidden": 8, "baselines": ["official_lstm", "tutorial_lstm"]},
}


def ssm_configs(preset: dict) -> list[dict]:
    return [SSMConfig(d_model=w, n_layers=l, d_state=preset["d_state"], head_hidden=preset["head_hidden"]).to_dict()
            for w, l in preset["grid"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--materials", nargs="+", required=True)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--max-seqs-per-bucket", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("results/size_sweep"))
    ap.add_argument("--leaderboard", type=Path, default=Path("results/reference/leaderboard_mc2.csv"))
    add_model_args(ap)  # only the defaults are used; the grid sets width/depth per run
    add_train_args(ap)
    args = ap.parse_args(argv)
    if args.preset == "smoke" and args.epochs == TrainConfig().epochs:
        args.epochs = 20

    base_cfg = train_config_from_args(args)
    preset = PRESETS[args.preset]
    runs = [("ssm", c) for c in ssm_configs(preset)] + [(b, {}) for b in preset["baselines"]]
    table = ResultsTable(args.out / "size_sweep.csv", ("material", "model", "config_id", "seed"))
    device = pick_device(args.device)

    for material in args.materials:
        sp = load_material_splits(args.raw_dir, material, base_cfg.split, base_cfg.split_seed,
                                  args.max_seqs_per_bucket, cache_from_args(args))
        tr = D.windows_from_raw(sp[0], per_seq=base_cfg.windows_per_seq, seed=base_cfg.split_seed)
        va = D.windows_from_raw(sp[1], per_seq=base_cfg.windows_per_seq, seed=base_cfg.split_seed + 1)
        test = D.make_test_segments(sp[2], seed=base_cfg.split_seed + 2, name=material)
        val_segs = make_val_segments(sp[1], base_cfg) if base_cfg.rollout_val else None
        print(f"== {material}: {len(tr)} train windows, {len(va)} val windows, {len(test)} test segments")

        for (name, mcfg), seed in itertools.product(runs, args.seeds):
            config_id = "-".join(f"{k}{v}" for k, v in sorted(mcfg.items()) if k in ("d_model", "n_layers", "d_state")) or "fixed"
            if table.has(material=material, model=name, config_id=config_id, seed=seed):
                continue
            n_params = count_parameters(build_model(name, mcfg or None))
            cfg = replace(base_cfg, model=name, model_config=mcfg, seed=seed)
            run_dir = args.out / "runs" / material / f"{name}_{config_id}_s{seed}"
            print(f"-- {name} {config_id} seed {seed} ({n_params} params)")
            res = train(cfg, tr, va, run_dir, val_segments=val_segs, verbose=False)
            row = {"material": material, "model": name, "config_id": config_id, "seed": seed,
                   "n_params": n_params, **{k: mcfg.get(k, "") for k in ("d_model", "n_layers", "d_state", "head_hidden")},
                   "best_epoch": res["best_epoch"], "selection": res["selection"],
                   "train_seconds": round(res["train_seconds"], 1), "epochs": cfg.epochs,
                   **score(res["model"].to(device), res["norm"], test, device=device, out_dir=run_dir)}
            table.add(row)
            print(f"   seq NRMSE p95 {row['seq_nrmse_p95']:.2f}%  RMS-err p95 {row['seq_rms_p95']:.2f}%  "
                  f"energy {row['energy_nrmse']:.2f}%")

    df = pd.read_csv(table.path)
    lb = pd.read_csv(args.leaderboard) if args.leaderboard.exists() else None
    for material in args.materials:
        d = df[df["material"] == material]
        if len(d):
            plot_error_vs_params(d, args.out / f"error_vs_params_{material}", metric="seq_rms_p95",
                                 leaderboard=lb, test_label=f"{material} held-out sequences")
    agg = (df.groupby(["material", "model", "config_id", "n_params"])
             [["seq_nrmse_p95", "seq_rms_p95", "energy_nrmse"]].agg(["mean", "std", "count"]))
    agg.to_csv(args.out / "size_sweep_summary.csv")
    print(f"wrote {table.path}, {args.out / 'size_sweep_summary.csv'} and plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
