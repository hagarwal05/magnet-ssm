"""Teacher-forced next-step training (the tutorial-1 recipe), for any registered model.

Data sources (combine freely):
  --train-h5 FILE...          windowed tutorial-1 files (e.g. the in-repo 3C90 demo file)
  --raw-dir DIR --materials   raw Dropbox CSVs (<M>_<i>_{B,H,T}.csv); split by SEQUENCE
                              into train/val/test so overlapping windows never leak

Defaults mirror tutorials/tutorial-1/Demo_LSTM_Training.ipynb: Adam, lr 1e-3 decayed x0.6
every 50 epochs, 200 epochs, batch 2048, MSE on the normalized next H, seed 1, and an
80/10/10 split. Model selection uses the best validation loss, or with ``--rollout-val``
the best 95th-percentile sequence error of a real autoregressive rollout on validation
segments (raw data only). Teacher-forced MSE is a weak proxy for long-horizon drift.

Examples:
  python -m src.train --model official_lstm --train-h5 third_party/magnetchallenge-2/tutorials/tutorial-1/3C90_Training.h5
  python -m src.train --model ssm --d-model 8 --n-layers 2 --raw-dir data/raw --materials 3C90 --rollout-val
"""
from __future__ import annotations

import argparse
import copy
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import data as D
from . import metrics as M
from .checkpoint import save_checkpoint
from .models import SSMConfig, build_model, count_parameters
from .normalize import NormParams
from .rollout import rollout
from .utils import pick_device, seed_everything, write_json


@dataclass
class TrainConfig:
    model: str = "ssm"
    model_config: dict = field(default_factory=dict)
    epochs: int = 200
    batch_size: int = 2048
    lr: float = 1e-3
    decay_epoch: int = 50
    decay_ratio: float = 0.6
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    h_noise: float = 0.0          # std of Gaussian noise added to the normalized H window
    split: tuple = (0.8, 0.1, 0.1)
    seed: int = 1                 # model init / batch order
    split_seed: int = 1           # data split + window sampling (fixed across a sweep)
    norm_mode: str = "tied"       # "tied" (ours) or "untied" (tutorial-1 exactly)
    windows_per_seq: int = 64     # raw data only
    rollout_val: bool = False     # raw data only
    rollout_val_every: int = 10
    rollout_val_max_seqs: int = 200
    patience: int = 0             # epochs without improvement before stopping; 0 = never
    device: str = "auto"


def _batched_mse(model, tensors, batch_size) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for sB, sH, bN, T, y in D.iterate_batches(tensors, batch_size, shuffle=False):
            out = model(sB, sH, bN, T)
            total += float(((out - y) ** 2).sum())
            n += y.shape[0]
    return total / max(n, 1)


def train(cfg: TrainConfig, train_set: D.WindowSet, val_set: D.WindowSet,
          out_dir: Path, val_segments: Optional[D.TestSet] = None,
          norm: Optional[NormParams] = None, init_state: Optional[dict] = None,
          meta: Optional[dict] = None, verbose: bool = True) -> dict:
    """Train one model. Returns a result dict and writes model.pt, history.csv, config.json."""
    seed_everything(cfg.seed)
    device = pick_device(cfg.device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    norm = norm or train_set.fit_norm(tie=(cfg.norm_mode == "tied"))
    tr = train_set.to_tensors(norm, device)
    va = val_set.to_tensors(norm, device)
    for name, ts in (("train", tr), ("val", va)):
        if not all(torch.isfinite(t).all() for t in ts):
            raise ValueError(f"{name} tensors contain NaN/inf after normalization ({norm})")

    model = build_model(cfg.model, cfg.model_config or None).to(device)
    if init_state is not None:
        model.load_state_dict(init_state, strict=True)
    n_params = count_parameters(model)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()
    gen = torch.Generator().manual_seed(cfg.seed)

    if verbose:
        print(f"[train] {cfg.model} {cfg.model_config or ''} params={n_params} device={device} "
              f"train={len(train_set)} val={len(val_set)}")

    history, best = [], {"score": float("inf"), "epoch": -1, "state": None}
    t_start, since_best = time.time(), 0
    for epoch in range(cfg.epochs):
        t0 = time.time()
        for g in opt.param_groups:
            g["lr"] = cfg.lr * (cfg.decay_ratio ** (epoch // cfg.decay_epoch))
        model.train()
        run, n = 0.0, 0
        for sB, sH, bN, T, y in D.iterate_batches(tr, cfg.batch_size, shuffle=True, generator=gen):
            if cfg.h_noise > 0:
                sH = sH + cfg.h_noise * torch.randn_like(sH)
            opt.zero_grad()
            loss = loss_fn(model(sB, sH, bN, T), y)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            run += loss.item() * y.shape[0]
            n += y.shape[0]
        row = {"epoch": epoch + 1, "lr": opt.param_groups[0]["lr"], "train_mse": run / n,
               "val_mse": _batched_mse(model, va, cfg.batch_size)}

        score = row["val_mse"]
        do_rollout = (cfg.rollout_val and val_segments is not None
                      and ((epoch + 1) % cfg.rollout_val_every == 0 or epoch + 1 == cfg.epochs))
        if do_rollout:
            res = rollout(model, val_segments, norm, device=device)
            seq = M.sequence_nrmse(res.H_pred, val_segments.H_true, res.warmups)
            row["val_rollout_p95"] = float(np.percentile(seq, 95)) if np.all(np.isfinite(seq)) else float("inf")
            row["val_rollout_mean"] = float(np.mean(seq))
            score = row["val_rollout_p95"]
        row["seconds"] = time.time() - t0
        history.append(row)

        selects_on_rollout = cfg.rollout_val and val_segments is not None
        if (not selects_on_rollout or do_rollout) and score < best["score"]:
            best = {"score": score, "epoch": epoch + 1,
                    "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}
            since_best = 0
        elif not selects_on_rollout or do_rollout:
            since_best += cfg.rollout_val_every if selects_on_rollout else 1

        if verbose and (epoch == 0 or (epoch + 1) % 10 == 0 or do_rollout or epoch + 1 == cfg.epochs):
            extra = f" rollout p95 {row['val_rollout_p95']:.2f}%" if do_rollout else ""
            print(f"  epoch {epoch + 1:4d}  train {row['train_mse']:.3e}  val {row['val_mse']:.3e}{extra}  "
                  f"({row['seconds']:.1f}s)")
        if cfg.patience and since_best >= cfg.patience:
            if verbose:
                print(f"  early stop at epoch {epoch + 1} (best {best['epoch']})")
            break

    if best["state"] is None:
        raise RuntimeError(f"no epoch produced a finite validation score (last row: {history[-1]}); "
                           "check the inputs and normalization for NaN/inf")
    model.load_state_dict(best["state"])
    model.cpu()
    result = {
        "n_params": n_params,
        "best_epoch": best["epoch"],
        "best_score": best["score"],
        "selection": "val_rollout_p95" if (cfg.rollout_val and val_segments is not None) else "val_mse",
        "train_seconds": time.time() - t_start,
        "final_val_mse": history[-1]["val_mse"],
    }
    save_checkpoint(out_dir / "model.pt", model, cfg.model, norm,
                    meta={"train_config": asdict(cfg), "result": result, **(meta or {})})
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    write_json(out_dir / "config.json", {"train_config": asdict(cfg), "result": result,
                                         "norm": norm.to_dict(), **(meta or {})})
    if verbose:
        print(f"[train] best epoch {best['epoch']} ({result['selection']} = {best['score']:.4g}); "
              f"saved {out_dir / 'model.pt'}")
    return {"model": model, "norm": norm, **result}


# -----------------------------------------------------------------------------------------
# Data assembly shared by the CLI and the experiments
# -----------------------------------------------------------------------------------------

@dataclass
class Prepared:
    train: D.WindowSet
    val: D.WindowSet
    test_windows: Optional[D.WindowSet]
    val_segments: Optional[D.TestSet]
    test_segments: dict  # material -> TestSet built from held-out sequences
    description: dict


def load_material_splits(raw_dir: Path, material: str, split=(0.8, 0.1, 0.1), split_seed: int = 1,
                         max_seqs_per_bucket: Optional[int] = None,
                         cache: Optional[Path] = None) -> list[list[D.RawBucket]]:
    """Sequence-level train/val/test split of one material's raw data. Deterministic in
    ``split_seed`` so every model in an experiment sees the same held-out sequences."""
    buckets = D.load_raw_material(raw_dir, material, cache=cache)
    if max_seqs_per_bucket:
        rng = np.random.default_rng(split_seed)
        buckets = [b.rows(np.sort(rng.choice(len(b.T), min(len(b.T), max_seqs_per_bucket), replace=False)))
                   for b in buckets]
    return D.split_raw(buckets, split, seed=split_seed)


def prepare_data(cfg: TrainConfig, train_h5: Sequence[Path] = (), raw_dir: Optional[Path] = None,
                 materials: Sequence[str] = (), max_seqs_per_bucket: Optional[int] = None,
                 test_starts_per_seq: int = 1, cache: Optional[Path] = None) -> Prepared:
    parts_tr, parts_va, parts_te = [], [], []
    val_buckets, test_segments, desc = [], {}, {"sources": []}
    s = cfg.split_seed

    for p in train_h5:
        ws = D.load_training_h5(p)
        idx = D.split_indices(len(ws), cfg.split, seed=s, groups=ws.group)
        parts_tr.append(ws.subset(idx[0]))
        parts_va.append(ws.subset(idx[1]))
        if len(cfg.split) > 2 and len(idx[2]):
            parts_te.append(ws.subset(idx[2]))
        desc["sources"].append({"windowed_h5": str(p), "n_windows": len(ws),
                                "split": "random over windows (source sequences unknown)"})

    if raw_dir is not None:
        for m in materials:
            sp = load_material_splits(raw_dir, m, cfg.split, s, max_seqs_per_bucket, cache)
            parts_tr.append(D.windows_from_raw(sp[0], per_seq=cfg.windows_per_seq, seed=s))
            parts_va.append(D.windows_from_raw(sp[1], per_seq=cfg.windows_per_seq, seed=s + 1))
            val_buckets.extend(sp[1])
            if len(sp) > 2 and sp[2]:
                test_segments[m] = D.make_test_segments(sp[2], starts_per_seq=test_starts_per_seq,
                                                        seed=s + 2, name=m)
            desc["sources"].append({
                "raw_material": m,
                "n_sequences": [int(sum(len(b.T) for b in s)) for s in sp],
                "split": "by sequence, stratified by frequency bucket",
            })

    if not parts_tr:
        raise ValueError("no training data: pass --train-h5 and/or --raw-dir with --materials")
    val_segments = make_val_segments(val_buckets, cfg) if cfg.rollout_val and val_buckets else None

    cat = lambda xs: D.WindowSet.concat(xs) if len(xs) > 1 else xs[0]
    return Prepared(cat(parts_tr), cat(parts_va), cat(parts_te) if parts_te else None,
                    val_segments, test_segments, desc)


def make_val_segments(buckets: Sequence[D.RawBucket], cfg: TrainConfig) -> D.TestSet:
    """Rollout-validation segments, capped at ``rollout_val_max_seqs`` (kept warm-up-sorted)."""
    vs = D.make_test_segments(buckets, seed=cfg.split_seed + 3, name="val")
    if len(vs) > cfg.rollout_val_max_seqs:
        keep = np.random.default_rng(cfg.split_seed).choice(len(vs), cfg.rollout_val_max_seqs, replace=False)
        keep = keep[np.argsort(vs.warmups[keep], kind="stable")]
        vs = D.TestSet(vs.B[keep], vs.H_padded[keep], vs.T[keep], vs.H_true[keep], name="val")
    return vs


def add_model_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", default="ssm", choices=["ssm", "official_lstm", "tutorial_lstm"])
    d = SSMConfig()
    ap.add_argument("--d-model", type=int, default=d.d_model)
    ap.add_argument("--d-state", type=int, default=d.d_state)
    ap.add_argument("--n-layers", type=int, default=d.n_layers)
    ap.add_argument("--head-hidden", type=int, default=d.head_hidden)
    ap.add_argument("--residual-head", action="store_true")


def model_config_from_args(args) -> dict:
    if args.model != "ssm":
        return {}
    return SSMConfig(d_model=args.d_model, d_state=args.d_state, n_layers=args.n_layers,
                     head_hidden=args.head_hidden, residual_head=args.residual_head).to_dict()


def add_train_args(ap: argparse.ArgumentParser) -> None:
    c = TrainConfig()
    ap.add_argument("--epochs", type=int, default=c.epochs)
    ap.add_argument("--batch-size", type=int, default=c.batch_size)
    ap.add_argument("--lr", type=float, default=c.lr)
    ap.add_argument("--decay-epoch", type=int, default=c.decay_epoch)
    ap.add_argument("--decay-ratio", type=float, default=c.decay_ratio)
    ap.add_argument("--h-noise", type=float, default=c.h_noise)
    ap.add_argument("--norm-mode", choices=["tied", "untied"], default=c.norm_mode)
    ap.add_argument("--windows-per-seq", type=int, default=c.windows_per_seq)
    ap.add_argument("--rollout-val", action="store_true")
    ap.add_argument("--rollout-val-every", type=int, default=c.rollout_val_every)
    ap.add_argument("--patience", type=int, default=c.patience)
    ap.add_argument("--seed", type=int, default=c.seed)
    ap.add_argument("--split-seed", type=int, default=c.split_seed)
    ap.add_argument("--device", default=c.device)
    ap.add_argument("--cache", default="data/cache",
                    help="HDF5 cache folder for raw CSVs ('' to disable)")


def train_config_from_args(args) -> TrainConfig:
    return TrainConfig(
        model=args.model, model_config=model_config_from_args(args), epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, decay_epoch=args.decay_epoch,
        decay_ratio=args.decay_ratio, h_noise=args.h_noise, norm_mode=args.norm_mode,
        windows_per_seq=args.windows_per_seq, rollout_val=args.rollout_val,
        rollout_val_every=args.rollout_val_every, patience=args.patience, seed=args.seed,
        split_seed=args.split_seed, device=args.device,
    )


def cache_from_args(args) -> Optional[Path]:
    # Check the raw string: Path("") is ".", which would silently cache into the cwd.
    return Path(args.cache) if args.cache else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-h5", type=Path, nargs="*", default=[])
    ap.add_argument("--raw-dir", type=Path)
    ap.add_argument("--materials", nargs="*", default=[])
    ap.add_argument("--max-seqs-per-bucket", type=int, default=None,
                    help="subsample raw sequences (quick local runs)")
    ap.add_argument("--out", type=Path, default=None)
    add_model_args(ap)
    add_train_args(ap)
    args = ap.parse_args(argv)

    cfg = train_config_from_args(args)
    prep = prepare_data(cfg, args.train_h5, args.raw_dir, args.materials, args.max_seqs_per_bucket,
                        cache=cache_from_args(args))
    tag = args.out or Path("results") / "runs" / f"{cfg.model}_{'-'.join(args.materials) or 'h5'}_s{cfg.seed}"
    res = train(cfg, prep.train, prep.val, tag, val_segments=prep.val_segments,
                meta={"data": prep.description})

    if prep.test_windows is not None:
        te = prep.test_windows.to_tensors(res["norm"])
        print(f"[train] held-out windows teacher-forced MSE: {_batched_mse(res['model'], te, cfg.batch_size):.4e}")
    for m, ts in prep.test_segments.items():
        rr = rollout(res["model"], ts, res["norm"])
        s, _ = M.evaluate_predictions(ts.B, ts.H_true, rr.H_pred, rr.warmups)
        print(f"[train] {m} held-out sequences: seq NRMSE p95 {s['seq_nrmse_p95']:.2f}%  "
              f"mean {s['seq_nrmse_mean']:.2f}%  energy {s['energy_nrmse']:.2f}%  (n={s['n_sequences']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
