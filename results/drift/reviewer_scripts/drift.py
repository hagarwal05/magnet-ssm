"""Within-sequence horizon test (reviewer-teams): w100 block only (900 predicted steps).
For each sequence, RMS error in 100-step windows after the boundary, divided by RMS(H_meas)
over the whole predicted region (fixed per-sequence denominator). Report median and p95 over
sequences, averaged over materials A, B, D, E."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/Users/agarwalh1/Library/CloudStorage/OneDrive-Leidos-LeidosCorpUS/Desktop/CMSC798")
sys.path.insert(0, str(ROOT))
from src.data import find_test_set
GT = ROOT / "data" / "Evaluation Tools" / "Ground Truth Data"
MATS = {"A": "3C92", "B": "3C95", "D": "FEC014", "E": "T37"}
teams = {t: ROOT / "data/Competition Result" / t / "Result" for t in ["Hefei", "Fuzhou", "SiegenPaderborn", "Bristol", "Gatech", "Cambridge"]}
for t in ["Southeast1", "Eindhoven", "Tsinghua", "XJTU1", "Sydney", "NITC", "Tianjin", "Leuven"]:
    teams[t] = Path(f"/tmp/mc2lit/others/subs/{t}/Result")
ours = sys.argv[1:]  # optional extra dirs label=path
for o in ours:
    k, v = o.split("="); teams[k] = Path(v)
cache = {L: find_test_set(GT, n) for L, n in MATS.items()}
print(f"{'team':16s} " + " ".join(f"win{k}" for k in range(1, 10)) + "   (p95 over seqs, mean over A,B,D,E)   | median win1 / win9")
for t, d in teams.items():
    p95s, meds = [], []
    for L, ts in cache.items():
        pred = pd.read_csv(d / f"H_Material {L}.csv", header=None, dtype=np.float64).to_numpy()
        sel = ts.warmups == 100
        P, H = pred[sel, 100:], ts.H_true[sel, 100:]
        den = np.sqrt((H ** 2).mean(1, keepdims=True))
        e = (P - H) ** 2
        win = np.stack([np.sqrt(e[:, 100*k:100*(k+1)].mean(1)) for k in range(9)], 1) / den
        p95s.append(np.percentile(win, 95, axis=0) * 100); meds.append(np.median(win, axis=0) * 100)
    p = np.mean(p95s, 0); m = np.mean(meds, 0)
    print(f"{t:16s} " + " ".join(f"{x:5.1f}" for x in p) + f"   | {m[0]:5.1f} / {m[8]:5.1f}")
