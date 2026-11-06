"""Robustness of drift.py (reviewer-teams): (1) is RMS(H_meas) stationary across the 9 windows?
(2) does the growth survive a per-window denominator (window error / window RMS(H_meas))?"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/Users/agarwalh1/Library/CloudStorage/OneDrive-Leidos-LeidosCorpUS/Desktop/CMSC798")
sys.path.insert(0, str(ROOT))
from src.data import find_test_set
GT = ROOT / "data" / "Evaluation Tools" / "Ground Truth Data"
MATS = {"A": "3C92", "B": "3C95", "D": "FEC014", "E": "T37"}
cache = {L: find_test_set(GT, n) for L, n in MATS.items()}
print("(1) window RMS(H_meas) / region RMS, median over sequences, per material (w100 block)")
for L, ts in cache.items():
    sel = ts.warmups == 100; H = ts.H_true[sel, 100:]
    den = np.sqrt((H ** 2).mean(1, keepdims=True))
    r = np.stack([np.sqrt((H[:, 100*k:100*(k+1)] ** 2).mean(1)) for k in range(9)], 1) / den
    print(f"  {L}: " + " ".join(f"{x:.2f}" for x in np.median(r, 0)) + "   mean win1 %.2f win9 %.2f" % (r[:, 0].mean(), r[:, 8].mean()))
teams = {t: ROOT / "data/Competition Result" / t / "Result" for t in ["Hefei", "Fuzhou", "SiegenPaderborn", "Bristol"]}
for t in ["Southeast1", "Eindhoven", "Tsinghua", "XJTU1", "Sydney", "NITC"]:
    teams[t] = Path(f"/tmp/mc2lit/others/subs/{t}/Result")
print("(2) per-window denominator: median over sequences of window RMS err / window RMS meas, mean over A,B,D,E; and p95")
for t, d in teams.items():
    med, p95 = [], []
    for L, ts in cache.items():
        pred = pd.read_csv(d / f"H_Material {L}.csv", header=None, dtype=np.float64).to_numpy()
        sel = ts.warmups == 100; P, H = pred[sel, 100:], ts.H_true[sel, 100:]
        we = np.stack([np.sqrt(((P - H)[:, 100*k:100*(k+1)] ** 2).mean(1)) for k in range(9)], 1)
        wm = np.stack([np.sqrt((H[:, 100*k:100*(k+1)] ** 2).mean(1)) for k in range(9)], 1)
        r = we / wm
        med.append(np.median(r, 0) * 100); p95.append(np.percentile(r, 95, axis=0) * 100)
    m, p = np.mean(med, 0), np.mean(p95, 0)
    print(f"  {t:16s} median win1 {m[0]:5.1f} win9 {m[8]:5.1f} ({m[8]/m[0]:.2f}x) | p95 win1 {p[0]:5.1f} win9 {p[8]:5.1f} ({p[8]/p[0]:.2f}x)")
