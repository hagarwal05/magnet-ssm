# Step-1 diagnostics of early error compounding (2026-11-09)

No new training. Every result re-reads saved predictions and checkpoints from earlier runs:
`results/final_AtoE` (σ = 0), `final_AtoE_hnoise` (σ = 0.05), `sweep_noise` and `seeds3`, with
files under `~/magnet-data/runs/`. Code: `src/experiments/diagnostics.py`, for D1 rollouts,
D2 and the D3 gradients. The extra analyses below (selection, sensitivity correlation, loop gain,
offset decay) were run inline from the same module's functions, and their CSVs are in this
folder. The error metric is the one in `results/drift/`: w100 block of A, B, D, E; RMS error over
the span divided by RMS(H_meas) over the predicted region; p95 over sequences, averaged over
the four materials.

The diagnosis being probed (RESEARCH_LOG §5.6): our first predicted step matches Tianjin, but
our error doubles within about 10 fed-back steps and keeps growing.

## D1: did the multi-step stage (K = 32) change things? (`d1_stage.csv`, `d1_selection.csv`)

Stage 1 and stage 2 checkpoints were both rolled out for every seeds3 run.

| p95 %, seed mean | steps 1–10, stage 1 | steps 1–10, stage 2 | window 9, stage 1 | window 9, stage 2 |
|---|---|---|---|---|
| S4D-205 | 32.2 | 31.8 | 88.4 | 58.6 |
| S4D-1,001 | 31.6 | 30.3 | 79.7 | 58.1 |
| LSTM-771 | 41.0 | 40.4 | 72.5 | 67.1 |

- **Stage 2 does nothing for the first 10 steps** (≤ 1.3 points on average). It cuts the late
  windows sharply instead (S4D window 9: −22 to −30 points).
- **Checkpoint selection threw away part of that gain.** The rule keeps stage 2 only if it beats
  stage 1 on validation. Stage 1 was kept 11 times, and in 7 of those stage 2 was better on test.
  One case is large: S4D-1,001 seed 3 B scores 50.6 % with the kept stage 1 vs 24.6 % with stage 2.
  Mean official A–E score (seed mean):

  | | always stage 1 | always stage 2 | as kept | best of the two (test oracle) |
  |---|---|---|---|---|
  | S4D-205 | 60.0 | **52.2** | 52.4 | 52.2 |
  | S4D-1,001 | 52.5 | **48.9** | 50.8 | 48.1 |
  | LSTM-771 | 68.5 | **63.1** | 64.0 | 61.6 |

  "Always use stage 2" would have scored better than the selection rule for all three models.
  For S4D-1,001 it gives 48.9 % instead of 50.8 %. The validation set (200 segments) is too
  noisy to choose between the two stages. These numbers come from the test set, so they diagnose
  the rule; they must not be used to pick a reported model.

## D2: did H-history noise change it? (`d2_noise.csv`)

Saved predictions, same architectures, p95 %:

| model | setting | step 1 | steps 1–10 | window 1 | window 9 |
|---|---|---|---|---|---|
| S4D-205 | σ 0 (phase 1) | 11.9 | 41.8 | 66.5 | 212.5 |
| | σ 0.05 (phase 2) | 12.9 | 30.7 | 37.6 | 92.2 |
| | σ 0.1 + stage 2 (seeds3 s1) | 17.5 | 28.5 | 34.1 | 57.6 |
| S4D-1,001 | σ 0 | 9.6 | 63.6 | 324.9 | 1135.4 |
| | σ 0.05 | 10.0 | 31.0 | 36.6 | 143.7 |
| | σ 0.1 + stage 2 | 15.6 | 31.4 | 34.0 | 59.0 |
| LSTM-771 | σ 0 | 12.4 | 46.9 | 70.4 | 401.5 |
| | σ 0.05 | 15.4 | 45.1 | 53.5 | 63.1 |
| | σ 0.1 + stage 2 | 18.9 | 43.6 | 48.6 | 66.6 |

- **Noise trades one-step accuracy for stability.** Without noise, our first-step error is 9.6–12.4 %.
  That is better than Tianjin's 16.7 %, but the rollout then explodes. Noise lifts step 1 to
  15.6–18.9 % and cuts the explosion. The sweep shows the same monotone pattern for S4D-205:
  step 1 goes 11.2 → 13.0 → 17.5 % for σ = 0.02 → 0.05 → 0.1.
- So "our one-step error matches Tianjin" is partly a product of the noise. Our noise-free models
  are better at one step and worse at compounding.
- **Noise helps steps 1–10 much less than it helps later windows**, and it does nothing for the
  LSTM's first 10 steps (46.9 → 43.6).

## D3: how do the models use fed-back H? (`d3_sensitivity.csv`, `d3_vs_compounding.csv`, `d3_offset_decay.csv`)

The models are the kept stage-2 checkpoints, at real test windows at the warm-up boundary.

**Loop gain (finite difference).** Shift the whole 80-step H window by ε, and the next prediction
moves by 0.93 ε for all three models (range 0.83–0.98 across 36 runs). An error in the fed-back
history is passed on almost undamped. The last H value alone carries 0.27–0.39 ε of that.

**Offset decay (closed loop).** Start a real rollout with the true history offset by 0.5 sd and
track how much of the offset survives:

| fraction retained after k steps | 1 | 10 | 25 | 50 | 100 | 300 |
|---|---|---|---|---|---|---|
| LSTM-771 | 0.93 | 0.77 | 0.61 | 0.44 | 0.26 | 0.06 |
| S4D-205 | 0.92 | 0.85 | 0.77 | 0.67 | 0.51 | 0.20 |
| S4D-1,001 | 0.93 | 0.87 | 0.81 | 0.73 | 0.60 | 0.28 |

- **Our models integrate fed-back H instead of re-deriving H from B.** Any error made in the first
  steps is carried for hundreds of steps. The S4D models keep it longest (about 50 % left after
  100 steps). The physical relation that should pull H back toward the value set by B is weak.
- Gradient shares agree: the H window carries about half the total input sensitivity
  (H / B ≈ 0.51–0.57). The LSTM puts 89 % of its H sensitivity on the last 8 values, the
  S4D models 42–50 %.
- **Across runs, sensitivity does not predict the size of the early doubling** (Spearman ρ of
  H-last-1 or H-last-8 share vs steps-1–10 / step-1 ratio: −0.29 to +0.48 within each
  architecture, n = 12 each). The persistence is shared by all runs; how much it hurts varies
  with the material and the seed.

## What this means for the plan

1. **The compounding mechanism is a near-unit feedback gain on fed-back H**, plus slow decay of
   offsets [measured]. Removing the feedback (a non-autoregressive model, step 2 of the plan) goes
   straight at it. Two other fixes act on the same gain: anchoring H (feed back fewer H values),
   and a loss that rewards pulling H back toward its B-driven value.
2. **The multi-step stage should always be kept.** It never hurts steps 1–10 and strongly helps
   late windows; the validation rule was the problem. Next runs should drop the keep-if-better
   rule, or use a larger validation set.
3. **Noise has a cost**: higher one-step error. A non-AR model needs no H noise, so it could keep
   the noise-free 9.6–12.4 % first-step error [inferred].
4. Nothing here points to architecture capacity as the problem. Bigger S4D models compound
   *more slowly* but carry offsets *longer*.
