"""Independent per-warm-up re-scoring (reviewer-teams)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path("/Users/agarwalh1/Library/CloudStorage/OneDrive-Leidos-LeidosCorpUS/Desktop/CMSC798")
sys.path.insert(0, str(ROOT))
from src import metrics as M
from src.data import find_test_set
GT = ROOT / "data" / "Evaluation Tools" / "Ground Truth Data"
MATS = {"A": "3C92", "B": "3C95", "C": "FEC007", "D": "FEC014", "E": "T37"}
team, pred_dir = sys.argv[1], Path(sys.argv[2])
blocks = {}
offs = []
for L, name in MATS.items():
    ts = find_test_set(GT, name)
    pred = pd.read_csv(pred_dir / f"H_Material {L}.csv", header=None, dtype=np.float64).to_numpy()
    assert pred.shape == ts.H_true.shape, (pred.shape, ts.H_true.shape)
    w = ts.warmups
    # sanity: does the submission keep the true warm-up prefix?
    seq = M.sequence_rms_error(pred, ts.H_true, w)
    off = M.official_score(seq, w)
    offs.append(off)
    line = f"{team} {L}: official {off:.2f}"
    for wu in np.unique(w):
        sel = w == wu
        p95 = np.percentile(np.abs(seq[sel]), 95, method="hazen")
        line += f" | w{wu} n={sel.sum()} p95 {p95:.2f} mean {seq[sel].mean():.2f}"
        blocks.setdefault(int(wu), []).append(p95)
    pooled = (np.mean(seq), np.percentile(seq, 95), np.percentile(seq, 95, method="hazen"))
    line += f" | pooled mean {pooled[0]:.2f} p95lin {pooled[1]:.2f} p95hazen {pooled[2]:.2f}"
    print(line)
print(f"{team} official mean A-E {np.mean(offs):.2f}")
for wu, v in sorted(blocks.items()):
    print(f"{team} block w{wu}: n_mat={len(v)} mean {np.mean(v):.2f}")
