"""Shared pieces for the experiment drivers."""
from __future__ import annotations

import csv
from pathlib import Path

from .. import data as D
from ..evaluate import evaluate_model

SUMMARY_KEYS = ("seq_nrmse_p95", "seq_nrmse_mean", "energy_nrmse", "seq_rms_p95", "seq_rms_mean",
                "energy_rel_p95", "energy_rel_mean", "n_sequences", "nonfinite_predictions")


def score(model, norm, test: D.TestSet, device="cpu", out_dir: Path | None = None) -> dict:
    """Roll out + all metrics; optionally store the per-sequence and drift tables."""
    res = evaluate_model(model, norm, test, device=device)
    if out_dir is not None:
        import pandas as pd

        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(res["per_seq"]).to_csv(out_dir / f"per_sequence_{test.name}.csv", index_label="seq")
        d = res["drift"]
        pd.DataFrame({"step_after_warmup": d["step"], "abs_err_pct": d["abs_err_pct"], "n": d["n"]}
                     ).to_csv(out_dir / f"drift_{test.name}.csv", index=False)
    row = {k: res["summary"][k] for k in SUMMARY_KEYS}
    for w, v in res["summary"]["by_warmup"].items():
        row[f"seq_nrmse_p95_w{w}"] = v["seq_nrmse_p95"]
    return row


class ResultsTable:
    """Append-only CSV that lets an interrupted experiment resume where it stopped."""

    def __init__(self, path: Path, key_fields: tuple[str, ...]):
        self.path = Path(path)
        self.key_fields = key_fields
        self.done: set[tuple] = set()
        if self.path.exists():
            with open(self.path) as f:
                for r in csv.DictReader(f):
                    self.done.add(tuple(str(r[k]) for k in key_fields))

    def has(self, **key) -> bool:
        return tuple(str(key[k]) for k in self.key_fields) in self.done

    def add(self, row: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        if not new:
            with open(self.path) as f:
                header = next(csv.reader(f))
            missing = [k for k in row if k not in header]
            if missing:
                raise ValueError(f"{self.path}: new columns {missing}; start a fresh table")
        else:
            header = list(row)
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header, restval="", lineterminator="\n")
            if new:
                w.writeheader()
            w.writerow(row)
        self.done.add(tuple(str(row[k]) for k in self.key_fields))
