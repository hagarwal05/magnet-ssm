"""Run a model over test sets and report metrics plus parameter count.

Three ways to specify the model:

  # 1) A pinned upstream reference (weights, normalization and test files checked by hash)
  python -m src.evaluate --reference tutorial3

  # 2) One of our checkpoints (normalization embedded) over test sets found under a folder
  python -m src.evaluate --checkpoint results/runs/ssm_3C90/model.pt \
      --test-dir data/pretest --materials 3C90 N87

  # 3) A bare upstream state dict + architecture + an explicit normalization file
  python -m src.evaluate --weights third_party/.../Model_LSTM_N87.sd --model official_lstm \
      --norm data/pretest/N87/Normalization_Params.json --test-dir data/pretest --materials N87

Outputs go to ``--out`` (default results/eval/<tag>/): ``summary.json``, one
``per_sequence_<M>.csv`` per material, ``drift_<M>.csv``, and optionally the predicted H in
the submission layout (``--save-pred`` writes ``H_<M>.csv``, full 1000 steps per row).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import metrics as M
from .checkpoint import load_checkpoint, load_state_dict_file
from .data import TestSet, find_test_set, load_test_h5
from .models import build_model, count_parameters
from .normalize import NormParams
from .reference import REFERENCES, check_hash, resolve
from .rollout import rollout
from .utils import pick_device, seed_everything, write_json


def evaluate_model(model: torch.nn.Module, norm: NormParams, test: TestSet, device="cpu",
                   convert_feedback: bool = False, percentile_method: str = "linear") -> dict:
    """Roll out over one test set and compute every metric."""
    if test.H_true is None:
        raise ValueError(f"{test.name}: no ground-truth H; cannot score (use --save-pred only)")
    t0 = time.time()
    res = rollout(model, test, norm, device=device, convert_feedback=convert_feedback)
    elapsed = time.time() - t0
    if res.nonfinite:
        # Diverged rollouts: score them as-is would give NaN; make that loud instead.
        print(f"  WARNING: {res.nonfinite} non-finite predictions in {test.name}")
    summary, per_seq = M.evaluate_predictions(test.B, test.H_true, res.H_pred, res.warmups,
                                              loss=test.loss, percentile_method=percentile_method)
    summary.update(material=test.name, n_params=count_parameters(model),
                   rollout_seconds=round(elapsed, 2), nonfinite_predictions=res.nonfinite)
    drift = M.drift_curve(res.H_pred, test.H_true, res.warmups)
    return {"summary": summary, "per_seq": per_seq, "drift": drift, "H_pred": res.H_pred}


def check_targets(summary: dict, targets: dict, tol: float = 0.02) -> list[str]:
    """Compare against published numbers (upstream prints 2 decimals, so the tolerance
    covers rounding plus float32 differences across platforms). Returns failures."""
    fails = []
    for key in ("seq_nrmse_mean", "seq_nrmse_p95", "energy_nrmse", "n_params"):
        if key in targets:
            got, want = summary[key], targets[key]
            ok = abs(got - want) <= (0 if key == "n_params" else tol)
            print(f"  {'OK  ' if ok else 'FAIL'} {key:16s} got {got:10.4f}  target {want}")
            if not ok:
                fails.append(key)
    for w, want in targets.get("by_warmup_mean", {}).items():
        got = summary["by_warmup"].get(int(w), {}).get("seq_nrmse_mean", float("nan"))
        ok = abs(got - want) <= tol
        print(f"  {'OK  ' if ok else 'FAIL'} warm-up {w:<4d} mean got {got:8.4f}  target {want}")
        if not ok:
            fails.append(f"by_warmup_{w}")
    return fails


def _write_outputs(out: Path, name: str, result: dict, save_pred: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result["per_seq"]).to_csv(out / f"per_sequence_{name}.csv", index_label="seq")
    d = result["drift"]
    pd.DataFrame({"step_after_warmup": d["step"], "abs_err_pct": d["abs_err_pct"], "n": d["n"]}
                 ).to_csv(out / f"drift_{name}.csv", index=False)
    if save_pred:
        np.savetxt(out / f"H_{name}.csv", result["H_pred"], delimiter=",")


def _print_summary(s: dict) -> None:
    print(f"  {s['material']}: n={s['n_sequences']}  params={s['n_params']}")
    print(f"    seq NRMSE   p95 {s['seq_nrmse_p95']:7.2f}%   mean {s['seq_nrmse_mean']:7.2f}%   (tutorial-3 def.)")
    print(f"    energy NRMSE    {s['energy_nrmse']:7.2f}%                     (tutorial-3 def.)")
    print(f"    seq RMS err p95 {s['seq_rms_p95']:7.2f}%   mean {s['seq_rms_mean']:7.2f}%   (grading def.)")
    print(f"    energy rel. p95 {s['energy_rel_p95']:7.2f}%   mean {s['energy_rel_mean']:7.2f}%   "
          f"(grading def., denominator={s['energy_rel_denominator']})")
    print(f"    OFFICIAL score  seq {s['official_seq_score']:7.2f}%   energy {s['official_energy_score']:7.2f}%   "
          f"(per-warm-up Hazen p95, mean over blocks; = leaderboard per-material number)")
    for w, v in s["by_warmup"].items():
        print(f"      warm-up {w:4d}: n={v['n']:5d}  mean {v['seq_nrmse_mean']:6.2f}%  p95 {v['seq_nrmse_p95']:6.2f}%")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--reference", choices=sorted(REFERENCES), help="pinned upstream reference")
    src.add_argument("--checkpoint", type=Path, help="checkpoint written by src.train")
    src.add_argument("--weights", type=Path, help="bare state dict (.sd); needs --model and --norm")
    ap.add_argument("--model", choices=["official_lstm", "tutorial_lstm"], help="architecture for --weights")
    ap.add_argument("--norm", type=Path, help="normalization JSON (required with --weights; "
                                              "required for pretest references)")
    ap.add_argument("--test-dir", type=Path, help="folder searched for each material's test set")
    ap.add_argument("--test-padded", type=Path, help="explicit padded test HDF5")
    ap.add_argument("--test-true", type=Path, help="explicit true test HDF5")
    ap.add_argument("--materials", nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cpu", help="cpu (default, bit-stable), cuda, mps or auto")
    ap.add_argument("--convert-feedback", action="store_true",
                    help="convert fed-back predictions from out- to H-normalized units "
                         "(upstream does not; irrelevant for our tied-normalization models)")
    ap.add_argument("--percentile", default="linear", choices=["linear", "hazen"],
                    help="linear = NumPy/tutorial-3 (default); hazen = MATLAB prctile")
    ap.add_argument("--save-pred", action="store_true")
    ap.add_argument("--allow-hash-mismatch", action="store_true")
    args = ap.parse_args(argv)

    seed_everything(1)
    device = pick_device(args.device)
    targets, tag = {}, None
    tests: list[TestSet] = []

    if args.reference:
        ref = REFERENCES[args.reference]
        tag = ref.key
        w_path = resolve(ref.weights)
        check_hash(w_path, ref.weights_sha256, strict=not args.allow_hash_mismatch)
        if ref.norm is not None:
            if args.norm is not None:
                ap.error(f"{ref.key} has a pinned normalization file; do not pass --norm")
            n_path = resolve(ref.norm)
            check_hash(n_path, ref.norm_sha256, strict=not args.allow_hash_mismatch)
        elif args.norm is None:
            ap.error(f"{ref.key}: the matching normalization file is not published in the upstream "
                     "repo. Pass --norm with the file that came with these weights (see "
                     "docs/reference_audit.md)")
        else:
            n_path = args.norm
        model = build_model(ref.model_name)
        model.load_state_dict(load_state_dict_file(w_path), strict=True)
        norm = NormParams.from_json(n_path)
        targets = ref.targets
        if ref.test_padded and not (args.test_dir or args.test_padded):
            tests.append(load_test_h5(resolve(ref.test_padded), resolve(ref.test_true)))
    elif args.checkpoint:
        if args.norm is not None:
            ap.error("--norm is not allowed with --checkpoint (the checkpoint carries its own)")
        model, norm, ck = load_checkpoint(args.checkpoint)
        tag = args.checkpoint.parent.name
    else:
        if args.model is None or args.norm is None:
            ap.error("--weights requires --model and --norm")
        model = build_model(args.model)
        model.load_state_dict(load_state_dict_file(args.weights), strict=True)
        norm = NormParams.from_json(args.norm)
        tag = args.weights.stem

    if args.test_padded:
        tests.append(load_test_h5(args.test_padded, args.test_true))
    if args.test_dir:
        if not args.materials:
            ap.error("--test-dir needs --materials")
        tests.extend(find_test_set(args.test_dir, m) for m in args.materials)
    if not tests:
        ap.error("no test set given (use --test-dir/--materials or --test-padded/--test-true)")

    model = model.to(device)
    out = args.out or Path("results") / "eval" / tag
    print(f"model: {type(model).__name__}, {count_parameters(model)} trainable parameters; device {device}")
    summaries, failures = [], []
    for test in tests:
        res = evaluate_model(model, norm, test, device=device, convert_feedback=args.convert_feedback,
                             percentile_method=args.percentile)
        _print_summary(res["summary"])
        _write_outputs(out, test.name, res, args.save_pred)
        summaries.append(res["summary"])
        if targets:
            print("  correctness gate vs published numbers:")
            failures += check_targets(res["summary"], targets)

    write_json(out / "summary.json", {"model": type(model).__name__, "results": summaries,
                                      "gate_failures": failures if targets else None})
    print(f"wrote {out}/")
    if targets and failures:
        print(f"GATE FAILED: {failures}")
        return 1
    if targets:
        print("GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
