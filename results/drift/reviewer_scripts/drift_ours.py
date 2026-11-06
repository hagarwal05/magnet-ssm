"""drift.py metric (fixed per-sequence denominator, p95 over sequences, mean over A,B,D,E) on our
seeds3 predictions, plus the per-window-denominator variant. reviewer-teams."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/Users/agarwalh1/Library/CloudStorage/OneDrive-Leidos-LeidosCorpUS/Desktop/CMSC798")
sys.path.insert(0, str(ROOT))
from src.data import find_test_set
GT = ROOT / "data" / "Evaluation Tools" / "Ground Truth Data"
MATS = {"A": "3C92", "B": "3C95", "D": "FEC014", "E": "T37"}
cache = {L: find_test_set(GT, n) for L, n in MATS.items()}
runs = dict(a.split("=") for a in open("/tmp/mc2lit/rt/ours_args.txt").read().split())
agg = {}
for k, d in runs.items():
    fixed, perwin, med = [], [], []
    for L, ts in cache.items():
        pred = pd.read_csv(Path(d) / f"H_Material {L}.csv", header=None, dtype=np.float64).to_numpy()
        sel = ts.warmups == 100; P, H = pred[sel, 100:], ts.H_true[sel, 100:]
        we = np.stack([np.sqrt(((P - H)[:, 100*j:100*(j+1)] ** 2).mean(1)) for j in range(9)], 1)
        wm = np.stack([np.sqrt((H[:, 100*j:100*(j+1)] ** 2).mean(1)) for j in range(9)], 1)
        den = np.sqrt((H ** 2).mean(1, keepdims=True))
        fixed.append(np.percentile(we / den, 95, axis=0) * 100)
        perwin.append(np.percentile(we / wm, 95, axis=0) * 100)
        med.append(np.median(we / den, axis=0) * 100)
    f, p, m = np.mean(fixed, 0), np.mean(perwin, 0), np.mean(med, 0)
    print(f"{k:22s} fixed-den p95: " + " ".join(f"{x:5.1f}" for x in f[[0,2,4,6,8]]) + f"  ratio {f[8]/f[0]:.2f} | median w1 {m[0]:.1f} w9 {m[8]:.1f} | per-win p95 w1 {p[0]:.1f} w9 {p[8]:.1f}")
    agg.setdefault(k.rsplit("_s", 1)[0], []).append(f)
for k, v in agg.items():
    v = np.mean(v, 0); print(f"MEAN {k:17s} win1 {v[0]:.1f} win3 {v[2]:.1f} win5 {v[4]:.1f} win7 {v[6]:.1f} win9 {v[8]:.1f}  ratio {v[8]/v[0]:.2f}")
