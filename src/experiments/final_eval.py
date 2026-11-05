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
  <out>/models/<model>_s<seed>_<L>/ checkpoint, history, config (KB-sized)
  <runs>/seed<seed>/<model>/Result/H_Material <L>.csv   submission-format predictions
                                    (large, outside the repo)

Training has up to two stages: next-step training (with optional --h-noise), then an
optional multi-step stage (--ms-horizon K > 1) that continues from the selected stage-1
checkpoint with a K-step rollout loss. Stage 2 is kept only if it beats stage 1 on the same
validation rollout. --seeds runs several model-init seeds on the same data split;
summary.csv averages over seeds.

  python -m src.experiments.final_eval --models official_lstm ssm_s ssm_m --device mps
  python -m src.experiments.final_eval --models ssm_s --h-noise 0.05 --ms-horizon 32 \
      --seeds 1 2 3 --tag _ms32 --rollout-val --device mps --out results/final_AtoE_ms
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
            max_windows: int, ms: dict | None = None, tag: str | None = None) -> dict:
    """``ms`` (multi-step stage, optional): dict(horizon, epochs, chunks, lr, warmup_epochs).
    Stage 1 is next-step training (``cfg``); stage 2 continues from its selected checkpoint
    with a K-step rollout loss and the same normalization and validation selection."""
    name, mcfg = MODELS[key]
    tag = tag or key
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
    run_dir = runs / tag / f"material_{letter}"
    t0 = time.time()
    res = train(replace(cfg, model=name, model_config=mcfg), tr, va, run_dir, val_segments=vseg,
                verbose=False, meta={"material": letter, "protocol": "final_eval", "stage": "next_step"})
    stage1 = {"stage1_best_epoch": res["best_epoch"], "stage1_val_p95": res["best_score"]}
    if ms:
        K = int(ms["horizon"])
        cps = max(2, int(np.ceil(ms["chunks"] / max(n_tr, 1))))
        tr_c = D.chunks_from_raw(tr_b, horizon=K, per_seq=cps, seed=cfg.split_seed + 10)
        va_c = D.chunks_from_raw(va_b, horizon=K, per_seq=max(1, cps // 4), seed=cfg.split_seed + 11)
        cfg2 = replace(cfg, model=name, model_config=mcfg, horizon=K, epochs=int(ms["epochs"]),
                       lr=float(ms["lr"]), decay_epoch=max(1, int(ms["epochs"]) // 2),
                       horizon_warmup_epochs=int(ms.get("warmup_epochs", 0)),
                       batch_size=int(ms.get("batch_size", 512)), rollout_val_every=1)
        init = {k: v.detach().cpu() for k, v in res["model"].state_dict().items()}
        res2 = train(cfg2, tr_c, va_c, run_dir / "multistep", val_segments=vseg, norm=res["norm"],
                     init_state=init, verbose=False,
                     meta={"material": letter, "protocol": "final_eval", "stage": "multistep"})
        # keep stage 2 only if it beats the stage-1 checkpoint on the same validation segments
        if res2["best_score"] <= res["best_score"]:
            res, run_dir = res2, run_dir / "multistep"
        stage1.update(ms_horizon=K, ms_chunks=len(tr_c), ms_best_epoch=res2["best_epoch"],
                      ms_val_p95=res2["best_score"], kept_stage="multistep" if res is res2 else "next_step")
    train_s = time.time() - t0

    ev = evaluate_model(res["model"].to(device), res["norm"], test, device=device)
    sub = runs / tag / "Result"
    sub.mkdir(parents=True, exist_ok=True)
    np.savetxt(sub / f"H_Material {letter}.csv", ev["H_pred"], delimiter=",", fmt="%.7g")
    # score the written CSV with the leaderboard scorer, exactly as a submission would be
    official = score_material(pd.read_csv(sub / f"H_Material {letter}.csv", header=None).to_numpy(), letter)

    keep = out / "models" / f"{tag}_s{cfg.seed}_{letter}"
    keep.mkdir(parents=True, exist_ok=True)
    for f in ("model.pt", "history.csv", "config.json"):
        shutil.copy2(run_dir / f, keep / f)
    if ms and (runs / tag / f"material_{letter}" / "multistep" / "history.csv").exists():
        shutil.copy2(runs / tag / f"material_{letter}" / "multistep" / "history.csv", keep / "history_multistep.csv")
    s = ev["summary"]
    return {
        "model": tag, "arch": name, "material": letter, "material_name": MATERIALS[letter],
        "seed": cfg.seed, "h_noise": cfg.h_noise, **stage1,
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
    if "seed" not in df.columns:
        df = df.assign(seed=1)
    # a (model, seed) counts once it has all five materials; then average over seeds
    complete = df.groupby(["model", "seed"]).filter(lambda g: g["material"].nunique() == 5)
    per_seed = (complete.groupby(["model", "seed"])
                .agg(n_params=("n_params", "first"), arch=("arch", "first"),
                     seq=("official_seq", "mean"), energy=("official_energy", "mean")).reset_index())
    ours = (per_seed.groupby("model")
            .agg(n_params=("n_params", "first"), arch=("arch", "first"), n_seeds=("seed", "nunique"),
                 seq=("seq", "mean"), seq_std=("seq", "std"), seq_min=("seq", "min"), seq_max=("seq", "max"),
                 energy=("energy", "mean"), energy_std=("energy", "std"))
            .reset_index().assign(source="ours"))
    teams = (lb.groupby("team").agg(n_params=("params", "mean"), seq=("seq_err_p95_pct", "mean"),
                                    energy=("energy_err_p95_pct", "mean")).reset_index()
             .rename(columns={"team": "model"}).assign(source="leaderboard"))
    table = pd.concat([ours, teams]).sort_values("seq").reset_index(drop=True)
    table["rank_seq"] = np.arange(1, len(table) + 1)
    table.to_csv(out / "summary.csv", index=False, float_format="%.4f")
    if len(per_seed):
        # one point per seed so the plot's min-max bars show the seed spread
        sweep = per_seed.rename(columns={"seq": "official_seq"}).assign(model=lambda d: d["arch"])
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
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="model-init seeds; the data split stays fixed (--split-seed)")
    ap.add_argument("--tag", default="", help="suffix for the model label, e.g. '_ms32'")
    ms = ap.add_argument_group("multi-step stage (off unless --ms-horizon > 1)")
    ms.add_argument("--ms-horizon", type=int, default=1)
    ms.add_argument("--ms-epochs", type=int, default=12)
    ms.add_argument("--ms-chunks", type=int, default=30_000, help="training chunks per material")
    ms.add_argument("--ms-lr", type=float, default=3e-4)
    ms.add_argument("--ms-warmup-epochs", type=int, default=4, help="horizon curriculum 1 -> K")
    ms.add_argument("--ms-batch-size", type=int, default=512)
    add_model_args(ap)  # only the defaults are used; --models picks each architecture
    add_train_args(ap)
    args = ap.parse_args(argv)

    base = train_config_from_args(args)
    device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    table = ResultsTable(args.out / "final_eval.csv", ("model", "seed", "material"))
    ms_cfg = (dict(horizon=args.ms_horizon, epochs=args.ms_epochs, chunks=args.ms_chunks, lr=args.ms_lr,
                   warmup_epochs=args.ms_warmup_epochs, batch_size=args.ms_batch_size)
              if args.ms_horizon > 1 else None)
    print(f"device {device}; stage 1: {base.epochs} epochs next-step, h_noise {base.h_noise}; "
          f"stage 2: {('K=%d, %d epochs' % (args.ms_horizon, args.ms_epochs)) if ms_cfg else 'off'}; "
          f"selection {'rollout p95' if base.rollout_val else 'val MSE'}; seeds {args.seeds or [base.seed]}",
          flush=True)
    for seed in (args.seeds or [base.seed]):
        cfg = replace(base, seed=seed)
        for key in args.models:
            tag = key + args.tag
            n = count_parameters(build_model(*MODELS[key][:1], MODELS[key][1] or None))
            for L in args.materials:
                if table.has(model=tag, seed=seed, material=L):
                    print(f"[skip] {tag} s{seed} {L}", flush=True)
                    continue
                print(f"[run ] {tag} ({n} params) seed {seed} Material {L} ({MATERIALS[L]})  "
                      f"{time.strftime('%H:%M')}", flush=True)
                row = run_one(key, L, cfg, args.packed, args.out, args.runs / f"seed{seed}", device,
                              args.max_windows, ms=ms_cfg, tag=tag)
                table.add(row)
                kept = f", kept {row['kept_stage']}" if "kept_stage" in row else ""
                print(f"       official seq {row['official_seq']:7.2f}%  energy {row['official_energy']:6.2f}%  "
                      f"({row['train_minutes']} min{kept})", flush=True)
                summarize(pd.read_csv(table.path), args.out)
    t = summarize(pd.read_csv(table.path), args.out)
    print(t[t.source == "ours"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
