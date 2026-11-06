# Within-sequence error vs forecast horizon (2026-11-06)

`python -m src.experiments.drift_analysis --out results/drift` (src/experiments/drift_analysis.py).
The method came from reviewer-teams on the literature-review team, and the lead and
winner-others reproduced it independently. This module rewrites it against repo paths.

**Why:** the official score aggregates per warm-up block, and the w100 / w500 / w900 blocks are
disjoint sets of sequences. Flat block scores therefore say nothing about how a single forecast
evolves. This analysis follows the same sequences along the forecast.

**Method:** w100 block (100 known steps, 900 predicted) of materials A, B, D, E. For each sequence
and each span of steps after the boundary: RMS(H_pred − H_meas) over the span divided by
RMS(H_meas) over the whole predicted region. Then p95 (and median) over sequences, averaged over
the four materials. `windows_per_window_denominator.csv` uses a per-window denominator instead,
as a robustness check. Our predictions are from the 3-seed run (`results/seeds3/`, files under
`~/magnet-data/runs/seeds3/`). Teams' are their official submissions (`data/Competition Result/`).

## p95 error per 100-step window (%), first vs last window

| source | window 1 | window 9 | ratio | added (pts) |
|---|---|---|---|---|
| Hefei | 6.8 | 11.2 | 1.64 | +4.4 |
| Fuzhou | 6.9 | 11.3 | 1.64 | +4.4 |
| Southeast1 | 12.4 | 16.6 | 1.34 | +4.2 |
| Eindhoven | 15.0 | 17.2 | 1.15 | +2.2 |
| Bristol | 14.6 | 20.0 | 1.37 | +5.4 |
| SiegenPaderborn | 10.5 | 24.6 | 2.33 | +14.0 |
| Tsinghua | 13.8 | 31.9 | 2.32 | +18.2 |
| XJTU1 | 12.2 | 39.1 | 3.22 | +26.9 |
| Sydney | 17.1 | 32.4 | 1.89 | +15.3 |
| NITC | 10.7 | 39.6 | 3.70 | +28.9 |
| Tianjin | 17.2 | 34.5 | 2.01 | +17.4 |
| Gatech | 21.2 | 32.6 | 1.54 | +11.5 |
| Cambridge | 38.7 | 32.2 | 0.83 | −6.4 |
| Leuven | 19.8 | 31.7 | 1.60 | +11.9 |
| **S4D-205 (ours, seed mean)** | **37.7** | **61.2** | 1.62 | **+23.4** |
| **S4D-1,001 (ours, seed mean)** | **34.0** | **72.4** | 2.13 | **+38.4** |
| **Dual-LSTM 771 (ours, seed mean)** | **46.0** | **67.6** | 1.47 | **+21.6** |

S4D-1,001 seed 3 reaches 107.0 % in window 9, against 59.0 / 51.3 % for seeds 1 / 2.

## Early steps (p95 %, mean over A/B/D/E)

| source | step 1 | steps 1–10 | steps 51–100 |
|---|---|---|---|
| Hefei | 6.8 | 5.9 | 7.8 |
| SiegenPaderborn | 4.7 | 8.4 | 12.7 |
| NITC | 8.2 | 5.3 | 13.6 |
| Tianjin | 16.7 | 16.0 | 18.2 |
| **S4D-205 (ours)** | **17.8** | **31.8** | **41.5** |
| **S4D-1,001 (ours)** | **15.7** | **31.1** | **37.1** |
| **Dual-LSTM 771 (ours)** | **17.8** | **40.5** | **50.9** |

The full tables are `windows.csv` and `early_steps.csv` (all 14 teams, every window and span, p95
and median, per seed).

## Reading

- **One-step error is not our gap.** Our first-step error (15.7–17.8 %) matches Tianjin
  (16.7 %), the submission closest to our inference scheme (one-step autoregressive, 80-step
  window [transcript]).
- **Our error doubles within ~10 fed-back steps** (31–41 %); Tianjin's does not (16.0 %).
- **That is not the whole gap.** 41–61 % of our window-9 gap to Hefei is present by step 10,
  and the rest accrues over the remaining steps: we add 22–38 points, at or above the
  closed-loop / AR teams (+14 to +29) and far above the non-AR teams (+2.2 to +5.4). Our ratio
  looks mild only because window 1 is already high.
- **Robustness:** with a per-window denominator, growth persists (Hefei 1.52×, Bristol 1.40×,
  SiegenPaderborn 2.37×, Tsinghua 2.56×, NITC 3.91×).
- **Attribution [inferred]:** fast early compounding of fed-back errors (exposure bias) is a main
  error source, alongside long-horizon accumulation. The contrast with Tianjin suggests the cause
  is how errors re-enter our model rather than the AR scheme itself. It is a single comparison,
  and Tianjin also differs in architecture, loss and augmentation.
