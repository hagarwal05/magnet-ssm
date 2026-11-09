"""Step-1 diagnostics of early error compounding (LITERATURE_REVIEW.md §9 item 1, contrast case).

No training. Uses saved predictions and checkpoints from earlier runs. Three questions:

  D1  Did the multi-step stage (K=32 rollout fine-tune) change the early-step curve?
      For each seeds3 run, both checkpoints exist (stage 1 = next-step + H-noise, and
      stage 2 = multistep/). Roll out both on the official test and compare the error over
      steps 1, 1–10, 11–25, 26–50, 51–100, and windows 1 / 9.
  D2  Did H-history noise change it? Compare saved predictions from phase 1 (σ = 0),
      phase 2 (σ = 0.05) and seeds3 seed 1 (σ = 0.1 + stage 2) for the same architectures.
  D3  How much does each prediction lean on the most recent fed-back H values? Input-gradient
      sensitivity of the next-H prediction to each position of the 80-step H window, versus
      the B window, at real test windows.

Error metric: as src.experiments.drift_analysis (w100 block of A, B, D, E; RMS error over
the span / RMS(H_meas) over the predicted region; p95 and median over sequences; mean over
materials).

  python -m src.experiments.diagnostics --out results/diagnostics
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..checkpoint import load_checkpoint
from ..data import TestSet, find_test_set
from ..rollout import rollout
from ..score_submission import GT_DIR, MATERIALS
from ..utils import pick_device
from .drift_analysis import EARLY, MATS, WINDOWS, span_errors

RUNS = Path.home() / "magnet-data" / "runs"
SPANS = EARLY[:-1] + [WINDOWS[0], WINDOWS[-1]]  # 1, 1–10, 11–25, 26–50, 51–100, win1, win9
KEYS = {"ssm_s": "S4D-205", "ssm_m": "S4D-1001", "official_lstm": "LSTM-771"}


def span_label(a: int, b: int) -> str:
    return f"{a + 1}" if b - a == 1 else f"{a + 1}-{b}"


def summarize(preds: dict, tests: dict) -> dict:
    """preds[L] = full (N, 1000) predictions; returns {span: (p95, median)} averaged over MATS."""
    p95, med = [], []
    for L in MATS:
        ts = tests[L]
        sel = ts.warmups == 100
        e = span_errors(preds[L][sel, 100:], ts.H_true[sel, 100:], SPANS)
        p95.append(np.percentile(e, 95, axis=0) * 100)
        med.append(np.median(e, axis=0) * 100)
    p95, med = np.mean(p95, 0), np.mean(med, 0)
    return {span_label(a, b): (p, m) for (a, b), p, m in zip(SPANS, p95, med)}


def w100_only(ts: TestSet) -> TestSet:
    sel = ts.warmups == 100
    return TestSet(ts.B[sel], ts.H_padded[sel], ts.T[sel], ts.H_true[sel], name=ts.name)


def roll(ckpt: Path, tests: dict, device) -> dict:
    """Roll out a checkpoint on the w100 block only (that is all D1 needs; much faster)."""
    model, norm, _ = load_checkpoint(ckpt)
    model = model.to(device)
    out = {}
    for L in MATS:
        sub = w100_only(tests[L])
        full = np.full_like(tests[L].H_true, np.nan)
        full[tests[L].warmups == 100] = rollout(model, sub, norm, device=device).H_pred
        out[L] = full
    return out


def read_preds(result_dir: Path) -> dict:
    from ..score_submission import read_prediction
    return {L: read_prediction(result_dir, L) for L in MATS}


def d1_stage_comparison(tests: dict, device) -> pd.DataFrame:
    rows = []
    for key, label in KEYS.items():
        for s in (1, 2, 3):
            base = RUNS / "seeds3" / f"seed{s}" / f"{key}_hn0.1_ms32"
            for L in MATS:
                pass  # per-material checkpoints, handled below
            # stage-1 and stage-2 checkpoints are per material
            for stage, sub in (("stage1", ""), ("stage2", "multistep")):
                preds = {}
                ok = True
                for L in MATS:
                    ck = base / f"material_{L}" / sub / "model.pt"
                    if not ck.exists():
                        ok = False
                        break
                    model, norm, _ = load_checkpoint(ck)
                    model = model.to(device)
                    full = np.full_like(tests[L].H_true, np.nan)
                    full[tests[L].warmups == 100] = rollout(model, w100_only(tests[L]), norm,
                                                            device=device).H_pred
                    preds[L] = full
                if not ok:
                    continue
                summ = summarize(preds, tests)
                rows.append({"model": label, "seed": s, "stage": stage,
                             **{f"p95_{k}": v[0] for k, v in summ.items()},
                             **{f"med_{k}": v[1] for k, v in summ.items()}})
                print(f"  D1 {label} s{s} {stage}: steps 1-10 p95 {summ['1-10'][0]:5.1f}  "
                      f"win1 {summ['1-100'][0]:5.1f}  win9 {summ['801-900'][0]:5.1f}", flush=True)
    return pd.DataFrame(rows)


def d2_noise_comparison(tests: dict) -> pd.DataFrame:
    sets = []
    for key, label in KEYS.items():
        sets += [(label, "sigma 0 (phase 1)", RUNS / "final_AtoE" / key / "Result"),
                 (label, "sigma 0.05 (phase 2)", RUNS / "final_AtoE_hnoise" / key / "Result"),
                 (label, "sigma 0.1 + stage 2 (seeds3 s1)", RUNS / "seeds3" / "seed1" / f"{key}_hn0.1_ms32" / "Result")]
    for sig in ("0.02", "0.05", "0.1"):
        sets.append(("S4D-205", f"sweep sigma {sig} + stage 2", RUNS / "sweep_noise" / "seed1" / f"ssm_s_hn{sig}_ms32" / "Result"))
    rows = []
    for label, setting, d in sets:
        if not (d / "H_Material A.csv").exists():
            continue
        summ = summarize(read_preds(d), tests)
        rows.append({"model": label, "setting": setting,
                     **{f"p95_{k}": v[0] for k, v in summ.items()},
                     **{f"med_{k}": v[1] for k, v in summ.items()}})
    return pd.DataFrame(rows)


def d3_sensitivity(tests: dict, device, n_windows: int = 512, seed: int = 0) -> pd.DataFrame:
    """Mean |d y / d input| per window position, for H and B channels, at real test windows.

    Windows are taken from the w100 block at the warm-up boundary (positions 20..99 of the
    true history), i.e. the exact input the model sees for its first prediction. Gradients
    are w.r.t. normalized inputs, so H and B sensitivities are on the same scale.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for key, label in KEYS.items():
        for s in (1, 2, 3):
            for L in MATS:
                ck = RUNS / "seeds3" / f"seed{s}" / f"{key}_hn0.1_ms32" / f"material_{L}"
                ck = ck / "multistep" / "model.pt" if (ck / "multistep" / "model.pt").exists() else ck / "model.pt"
                # use the stage the final model actually kept, per final_eval.csv
                model, norm, _ = load_checkpoint(ck)
                model = model.to(device).eval()
                ts = tests[L]
                sel = np.flatnonzero(ts.warmups == 100)
                pick = rng.choice(sel, size=min(n_windows, len(sel)), replace=False)
                Bw = norm.norm_B(ts.B[pick, 20:100])
                Hw = norm.norm_H(ts.H_true[pick, 20:100])
                bN = norm.norm_Scal(ts.B[pick, 100:101])
                T = norm.norm_T(ts.T[pick]).reshape(-1, 1)
                t = lambda x: torch.tensor(np.asarray(x, dtype=np.float32), device=device)
                sB = t(Bw).unsqueeze(-1).requires_grad_(True)
                sH = t(Hw).unsqueeze(-1).requires_grad_(True)
                y = model(sB, sH, t(bN), t(T))
                y.sum().backward()
                gH = sH.grad.abs().mean(0).squeeze(-1).cpu().numpy()   # (80,)
                gB = sB.grad.abs().mean(0).squeeze(-1).cpu().numpy()
                rows.append({"model": label, "seed": s, "material": L,
                             "H_last1": gH[-1], "H_last1_share_of_H": gH[-1] / gH.sum(),
                             "H_last8_share_of_H": gH[-8:].sum() / gH.sum(),
                             "H_total": gH.sum(), "B_total": gB.sum(),
                             "H_over_B": gH.sum() / max(gB.sum(), 1e-12),
                             **{f"gH_{i}": gH[i] for i in range(80)}})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("results/diagnostics"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--skip", nargs="*", default=[], choices=["d1", "d2", "d3"])
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    tests = {L: find_test_set(GT_DIR, MATERIALS[L]) for L in MATS}

    if "d2" not in args.skip:
        d2 = d2_noise_comparison(tests)
        d2.to_csv(args.out / "d2_noise.csv", index=False, float_format="%.3f")
        print(d2[["model", "setting", "p95_1", "p95_1-10", "p95_51-100", "p95_1-100", "p95_801-900"]]
              .round(1).to_string(index=False), flush=True)
    if "d3" not in args.skip:
        d3 = d3_sensitivity(tests, device)
        d3.to_csv(args.out / "d3_sensitivity.csv", index=False, float_format="%.5g")
        print(d3.groupby("model")[["H_last1_share_of_H", "H_last8_share_of_H", "H_over_B"]]
              .mean().round(3).to_string(), flush=True)
    if "d1" not in args.skip:
        d1 = d1_stage_comparison(tests, device)
        d1.to_csv(args.out / "d1_stage.csv", index=False, float_format="%.3f")
    print(f"wrote {args.out}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
