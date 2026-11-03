"""Static figures for results/ and the paper (matplotlib, PNG + PDF).

Every figure is drawn from a tidy CSV that is written next to it; the CSV doubles as the
figure's table view. Palette: categorical slots 1-3 of the reference palette (validated
all-pairs: worst CVD dE 9.2, normal-vision dE 24.0), a muted gray for context marks, and a
single-hue blue ramp for ordered series (validated ``--ordinal``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # blue, orange, aqua
ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]


def _style(ax, title: str, subtitle: Optional[str] = None):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9, width=0.8)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    import textwrap

    lines = textwrap.wrap(subtitle, 78) if subtitle else []
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold",
                 pad=8 + 13 * len(lines))
    if lines:
        ax.text(0, 1.02, "\n".join(lines), transform=ax.transAxes, color=INK_2, fontsize=8.5,
                va="bottom", linespacing=1.3)


def _legend(ax, below: bool = False, **kw):
    """Legend in secondary ink. ``below=True`` puts it under the axes, clear of the data."""
    if below:
        kw = {"loc": "upper left", "bbox_to_anchor": (0, -0.16), "ncol": 1, **kw}
    return ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, **kw)


def _save(fig, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_error_vs_params(sweep: pd.DataFrame, out: Path, metric: str = "seq_rms_p95",
                         leaderboard: Optional[pd.DataFrame] = None,
                         test_label: str = "our held-out split",
                         label_teams: Sequence[str] = ("Hefei", "SiegenPaderborn", "Gatech", "NTU2")) -> None:
    """RQ1: error vs parameter count (log-log). Leaderboard points are context only and are
    labelled with their own test set."""
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    if leaderboard is not None and len(leaderboard):
        lb = leaderboard[leaderboard["seq_err_p95_pct"] < 700]  # drop the 800 / 836 placeholders
        ax.scatter(lb["avg_params"], lb["seq_err_p95_pct"], s=26, color=MUTED, alpha=0.55,
                   linewidths=0, label="MC2 leaderboard (materials A–E)", zorder=2)
        for _, r in lb[lb["team"].isin(label_teams)].iterrows():
            ax.annotate(r["team"], (r["avg_params"], r["seq_err_p95_pct"]), xytext=(4, 3),
                        textcoords="offset points", fontsize=7.5, color=MUTED)
    families = [f for f in ("ssm", "official_lstm", "tutorial_lstm") if f in set(sweep["model"])]
    names = {"ssm": "S4D (ours)", "official_lstm": "Dual-LSTM 771 (retrained)",
             "tutorial_lstm": "Dual-LSTM 2,399 (retrained)"}
    for k, fam in enumerate(families):
        d = sweep[sweep["model"] == fam].groupby("n_params", as_index=False)[metric].agg(["mean", "min", "max"])
        d = d.reset_index() if "n_params" not in d.columns else d
        color = SERIES[k % len(SERIES)]
        ax.errorbar(d["n_params"], d["mean"], yerr=[d["mean"] - d["min"], d["max"] - d["mean"]],
                    fmt="o", ms=6, color=color, mec=SURFACE, mew=1.5, elinewidth=1, capsize=0,
                    label=f"{names.get(fam, fam)} — {test_label}", zorder=3)
        if fam == "ssm" and len(d) > 1:
            front = d.sort_values("n_params")
            front = front[front["mean"] <= front["mean"].cummin()]
            ax.plot(front["n_params"], front["mean"], color=color, lw=1.5, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Trainable parameters per material")
    ax.set_ylabel("95th-pct sequence error [%]")
    _style(ax, "Accuracy vs model size",
           f"metric: {metric}; markers = mean over seeds, bars = min–max. "
           "Leaderboard and our points use different test sets.")
    _legend(ax, below=True)
    _save(fig, out)


def plot_drift(curves: dict, out: Path, title: str = "Autoregressive drift") -> None:
    """Error vs steps after the warm-up. ``curves``: label -> DataFrame from drift_<M>.csv."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6 + 0.25 * len(curves)))
    for k, (label, df) in enumerate(curves.items()):
        ok = df["n"] >= 5
        ax.plot(df.loc[ok, "step_after_warmup"], df.loc[ok, "abs_err_pct"], lw=1.5,
                color=SERIES[k % len(SERIES)], label=label)
    ax.set_xlabel("Steps after warm-up (62.5 ns each)")
    ax.set_ylabel("Mean |H_pred − H_meas| / mean|H_meas| [%]")
    _style(ax, title, "averaged over sequences still in their prediction window (n ≥ 5)")
    if len(curves) > 1:
        _legend(ax, below=True)
    _save(fig, out)


def plot_error_histogram(per_seq: pd.DataFrame, out: Path, column: str = "seq_nrmse",
                         title: str = "Sequence error distribution") -> None:
    """The official histogram (ratio of sequences per bin) with mean and 95th percentile."""
    v = np.abs(per_seq[column].to_numpy())
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.hist(v, bins=40, weights=np.full(v.size, 1.0 / v.size), color=SERIES[0],
            edgecolor=SURFACE, linewidth=1.0)
    mean, p95 = float(np.mean(v)), float(np.percentile(v, 95))
    for x, lab in ((mean, f"mean {mean:.2f}%"), (p95, f"95th pct {p95:.2f}%")):
        ax.axvline(x, color=INK_2, lw=1)
        ax.annotate(lab, (x, 1), xycoords=("data", "axes fraction"), xytext=(4, -12),
                    textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xlabel(f"{column} [%]")
    ax.set_ylabel("Ratio of sequences")
    _style(ax, title, f"n = {v.size} sequences, max {v.max():.1f}%")
    _save(fig, out)


def figures_for_eval(eval_dirs: Sequence[Path], labels: Optional[Sequence[str]] = None,
                     out: Optional[Path] = None) -> list[Path]:
    """Histogram per eval dir and one combined drift plot per material."""
    eval_dirs = [Path(d) for d in eval_dirs]
    labels = list(labels or [d.name for d in eval_dirs])
    out = Path(out or eval_dirs[0])
    written = []
    drifts: dict[str, dict] = {}
    for d, lab in zip(eval_dirs, labels):
        for f in sorted(d.glob("per_sequence_*.csv")):
            mat = f.stem.removeprefix("per_sequence_")
            p = d / f"hist_seq_nrmse_{mat}"
            plot_error_histogram(pd.read_csv(f), p, title=f"Sequence error, {mat} — {lab}")
            written.append(p.with_suffix(".png"))
            dr = d / f"drift_{mat}.csv"
            if dr.exists():
                drifts.setdefault(mat, {})[lab] = pd.read_csv(dr)
    for mat, curves in drifts.items():
        p = out / f"drift_{mat}"
        plot_drift(dict(list(curves.items())[:3]), p, title=f"Autoregressive drift, {mat}")
        written.append(p.with_suffix(".png"))
    return written


def plot_cross_material(df: pd.DataFrame, out: Path, metric: str = "seq_nrmse_p95",
                        model: str = "ssm") -> None:
    """RQ2: per held-out material, zero-shot vs best fine-tuned vs in-domain (dot plot)."""
    d = df[df["model"] == model].copy()
    if d.empty:
        return
    d["cond"] = np.where(d["condition"].str.startswith("finetune"), "fine-tuned", d["condition"])
    ft = d[d["cond"] == "fine-tuned"]
    if not ft.empty:
        kmax = ft["finetune_seqs"].max()
        d = d[(d["cond"] != "fine-tuned") | (d["finetune_seqs"] == kmax)]
    order = [c for c in ("zero_shot", "fine-tuned", "in_domain") if c in set(d["cond"])]
    labels = {"zero_shot": "zero-shot (never seen)",
              "fine-tuned": f"fine-tuned on {int(ft['finetune_seqs'].max()) if not ft.empty else 0} sequences",
              "in_domain": "trained on this material"}
    mats = sorted(d["target"].unique())
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    x = np.arange(len(mats))
    offs = np.linspace(-0.18, 0.18, len(order)) if len(order) > 1 else [0.0]
    for k, c in enumerate(order):
        g = d[d["cond"] == c].groupby("target")[metric].agg(["mean", "min", "max"]).reindex(mats)
        ax.errorbar(x + offs[k], g["mean"], yerr=[g["mean"] - g["min"], g["max"] - g["mean"]], fmt="o",
                    ms=6, color=SERIES[k], mec=SURFACE, mew=1.5, elinewidth=1, label=labels[c])
    ax.set_xticks(x, mats)
    ax.set_xlabel("Held-out material")
    ax.set_ylabel(f"{metric} [%]")
    _style(ax, "Generalization to unseen materials", f"model: {model}; mean over seeds, bars = min–max")
    _legend(ax, below=True)
    _save(fig, out)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Figures from src.evaluate output folders.")
    ap.add_argument("eval_dirs", nargs="+", type=Path)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    for p in figures_for_eval(args.eval_dirs, args.labels, args.out):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
