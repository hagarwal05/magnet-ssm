"""RQ2: generalization to materials the model never trained on.

For each architecture and seed:
  1. ``pooled``: train once on the training splits of the SOURCE materials (one shared
     normalization fitted on the pooled source data).
  2. ``zero_shot``: score that model, unchanged, on each TARGET material's held-out test
     sequences.
  3. ``finetune_k``: starting from the pooled model, fine-tune on k training sequences of the
     target (``--finetune-seqs``, lower lr, fewer epochs; source normalization kept), then
     score on the same held-out sequences. This mirrors the final test, where new materials
     came with very little data (Material A: 105 training sequences).
  4. ``in_domain``: an identical model trained only on the target's own training split.
     This is the upper reference.

Every condition is scored on the same held-out test segments of the target, built from the
same sequence-level split. Results go to ``<out>/cross_material.csv`` (resumable).

  python -m src.experiments.cross_material --raw-dir data/raw \
      --sources 3C90 3C94 3F4 N27 N49 N87 --targets 3E6 77 N30 \
      --d-model 8 --n-layers 2 --seeds 1 2 3 --device cuda
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .. import data as D
from ..checkpoint import load_checkpoint
from ..models import count_parameters
from ..plots import plot_cross_material
from ..train import (TrainConfig, add_model_args, add_train_args, cache_from_args, load_material_splits,
                     make_val_segments, train, train_config_from_args)
from ..utils import pick_device
from .common import ResultsTable, score


def first_k_sequences(buckets: list[D.RawBucket], k: int, seed: int) -> list[D.RawBucket]:
    """k sequences drawn across buckets (frequencies) as evenly as possible."""
    rng = np.random.default_rng(seed)
    pool = [(bi, r) for bi, b in enumerate(buckets) for r in range(len(b.T))]
    order = rng.permutation(len(pool))
    # round-robin over buckets so small k still covers several frequencies
    by_bucket: dict[int, list[int]] = {}
    for i in order:
        by_bucket.setdefault(pool[i][0], []).append(pool[i][1])
    picked: dict[int, list[int]] = {bi: [] for bi in by_bucket}
    while sum(len(v) for v in picked.values()) < min(k, len(pool)):
        for bi, rows in by_bucket.items():
            if rows and sum(len(v) for v in picked.values()) < k:
                picked[bi].append(rows.pop())
    return [buckets[bi].rows(np.sort(rows)) for bi, rows in picked.items() if rows]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--finetune-seqs", type=int, nargs="*", default=[10, 100])
    ap.add_argument("--finetune-epochs", type=int, default=50)
    ap.add_argument("--finetune-lr", type=float, default=3e-4)
    ap.add_argument("--skip-in-domain", action="store_true")
    ap.add_argument("--max-seqs-per-bucket", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("results/cross_material"))
    add_model_args(ap)
    add_train_args(ap)
    args = ap.parse_args(argv)
    overlap = set(args.sources) & set(args.targets)
    if overlap:
        ap.error(f"materials in both --sources and --targets: {sorted(overlap)}")

    base = train_config_from_args(args)
    cache = cache_from_args(args)
    device = pick_device(args.device)
    table = ResultsTable(args.out / "cross_material.csv",
                         ("model", "config_id", "seed", "target", "condition"))
    config_id = "-".join(f"{k}{v}" for k, v in sorted(base.model_config.items())
                         if k in ("d_model", "n_layers", "d_state")) or "fixed"

    splits = {m: load_material_splits(args.raw_dir, m, base.split, base.split_seed, args.max_seqs_per_bucket, cache)
              for m in [*args.sources, *args.targets]}
    tests = {m: D.make_test_segments(splits[m][2], seed=base.split_seed + 2, name=m) for m in args.targets}
    w = lambda bks, s: D.windows_from_raw(bks, per_seq=base.windows_per_seq, seed=s)

    for seed in args.seeds:
        cfg = replace(base, seed=seed)
        tag = f"{cfg.model}_{config_id}_s{seed}"
        common = {"model": cfg.model, "config_id": config_id, "seed": seed,
                  "sources": "+".join(args.sources)}

        # 1-2: pooled source model, zero-shot on each target
        pooled_dir = args.out / "runs" / tag / "pooled"
        if (pooled_dir / "model.pt").exists():
            pooled_model, pooled_norm, _ = load_checkpoint(pooled_dir / "model.pt")
        else:
            src_tr = D.WindowSet.concat([w(splits[m][0], cfg.split_seed) for m in args.sources])
            src_va = D.WindowSet.concat([w(splits[m][1], cfg.split_seed + 1) for m in args.sources])
            val_segs = make_val_segments([b for m in args.sources for b in splits[m][1]], cfg) if cfg.rollout_val else None
            print(f"== pooled {tag}: {len(src_tr)} windows from {args.sources}")
            r = train(cfg, src_tr, src_va, pooled_dir, val_segments=val_segs, verbose=False,
                      meta={"sources": args.sources})
            pooled_model, pooled_norm = r["model"], r["norm"]
        n_params = count_parameters(pooled_model)

        for tgt in args.targets:
            if not table.has(**common, target=tgt, condition="zero_shot"):
                row = score(pooled_model.to(device), pooled_norm, tests[tgt], device, args.out / "runs" / tag / f"zero_shot_{tgt}")
                table.add({**common, "target": tgt, "condition": "zero_shot", "finetune_seqs": 0,
                           "n_params": n_params, **row})
                print(f"   {tgt} zero-shot: seq NRMSE p95 {row['seq_nrmse_p95']:.2f}%")

            # 3: few-shot fine-tuning from the pooled weights
            for k in args.finetune_seqs:
                cond = f"finetune_{k}"
                if table.has(**common, target=tgt, condition=cond):
                    continue
                few = first_k_sequences(splits[tgt][0], k, seed)
                ft_cfg = replace(cfg, epochs=args.finetune_epochs, lr=args.finetune_lr,
                                 decay_epoch=max(1, args.finetune_epochs // 2), rollout_val=False, patience=0)
                ft_dir = args.out / "runs" / tag / f"{cond}_{tgt}"
                r = train(ft_cfg, w(few, cfg.split_seed), w(splits[tgt][1], cfg.split_seed + 1), ft_dir,
                          norm=pooled_norm,
                          init_state={k_: v.cpu() for k_, v in pooled_model.state_dict().items()},
                          verbose=False, meta={"finetune_from": str(pooled_dir), "target": tgt, "k": k})
                row = score(r["model"].to(device), r["norm"], tests[tgt], device, ft_dir)
                table.add({**common, "target": tgt, "condition": cond, "finetune_seqs": k,
                           "n_params": n_params, **row})
                print(f"   {tgt} fine-tuned on {k} seqs: seq NRMSE p95 {row['seq_nrmse_p95']:.2f}%")

            # 4: in-domain reference
            if not args.skip_in_domain and not table.has(**common, target=tgt, condition="in_domain"):
                in_dir = args.out / "runs" / tag / f"in_domain_{tgt}"
                val_segs = make_val_segments(splits[tgt][1], cfg) if cfg.rollout_val else None
                r = train(cfg, w(splits[tgt][0], cfg.split_seed), w(splits[tgt][1], cfg.split_seed + 1), in_dir,
                          val_segments=val_segs, verbose=False)
                row = score(r["model"].to(device), r["norm"], tests[tgt], device, in_dir)
                table.add({**common, "target": tgt, "condition": "in_domain", "finetune_seqs": "",
                           "n_params": n_params, **row})
                print(f"   {tgt} in-domain: seq NRMSE p95 {row['seq_nrmse_p95']:.2f}%")

    df = pd.read_csv(table.path)
    df["finetune_seqs"] = pd.to_numeric(df["finetune_seqs"], errors="coerce")
    for metric in ("seq_nrmse_p95", "seq_rms_p95"):
        plot_cross_material(df, args.out / f"cross_material_{metric}", metric=metric, model=base.model)
    (df.groupby(["model", "config_id", "target", "condition"])[["seq_nrmse_p95", "seq_rms_p95", "energy_nrmse"]]
       .agg(["mean", "std", "count"]).to_csv(args.out / "cross_material_summary.csv"))
    print(f"wrote {table.path} and plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
