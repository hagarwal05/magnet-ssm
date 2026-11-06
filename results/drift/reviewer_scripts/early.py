"""Early-step error (reviewer-teams): RMS err over steps [a,b) after the boundary / RMS(H_meas) over
the whole predicted region; w100 block; p95 and median over sequences; mean over A,B,D,E."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/Users/agarwalh1/Library/CloudStorage/OneDrive-Leidos-LeidosCorpUS/Desktop/CMSC798")
sys.path.insert(0, str(ROOT))
from src.data import find_test_set
GT = ROOT / "data" / "Evaluation Tools" / "Ground Truth Data"
MATS = {"A": "3C92", "B": "3C95", "D": "FEC014", "E": "T37"}
cache = {L: find_test_set(GT, n) for L, n in MATS.items()}
R = Path.home() / "magnet-data/runs/seeds3"
runs = {t: ROOT / "data/Competition Result" / t / "Result" for t in ["Hefei", "SiegenPaderborn", "Fuzhou"]}
for t in ["Tianjin", "NITC", "Tsinghua"]:
    runs[t] = Path(f"/tmp/mc2lit/others/subs/{t}/Result")
for m in ["ssm_s", "ssm_m", "official_lstm"]:
    for s in (1, 2, 3):
        runs[f"ours_{m}_s{s}"] = R / f"seed{s}" / f"{m}_hn0.1_ms32" / "Result"
spans = [(0, 1), (0, 10), (10, 25), (25, 50), (50, 100), (800, 900)]
print("p95 (median) per span of steps after boundary:", spans)
acc = {}
for k, d in runs.items():
    p, md = [], []
    for L, ts in cache.items():
        pred = pd.read_csv(d / f"H_Material {L}.csv", header=None, dtype=np.float64).to_numpy()
        sel = ts.warmups == 100; P, H = pred[sel, 100:], ts.H_true[sel, 100:]
        den = np.sqrt((H ** 2).mean(1))
        v = np.stack([np.sqrt(((P - H)[:, a:b] ** 2).mean(1)) / den for a, b in spans], 1)
        p.append(np.percentile(v, 95, axis=0) * 100); md.append(np.median(v, axis=0) * 100)
    p, md = np.mean(p, 0), np.mean(md, 0)
    key = k.rsplit("_s", 1)[0] if k.startswith("ours") else k
    acc.setdefault(key, []).append((p, md))
for k, v in acc.items():
    p = np.mean([x[0] for x in v], 0); md = np.mean([x[1] for x in v], 0)
    print(f"{k:18s} " + "  ".join(f"{a:5.1f} ({b:4.1f})" for a, b in zip(p, md)))
