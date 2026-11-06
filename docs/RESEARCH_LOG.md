# Research log: state-space models for MagNet Challenge 2

**This is the master record of the project.** Methods, setup, every experiment and its exact
numbers, and every decision with its reason, kept so paper sections can be written from it.

Conventions:

- Numbers are copied from committed result files (cited per entry), never from memory.
- "Official score" = the organizers' metric and aggregation (section 3.2), on the official
  A–E test set: directly comparable to the leaderboard.
- Every entry carries a status tag: **[result]** = finished and committed; **[running]**;
  **[preliminary]** = single seed or partial; **[caveat]** = a limit the paper must state.
- Newest experiment entries go at the end of section 5. Do not edit old entries; append
  corrections as new dated notes.

Related docs: `docs/reference_audit.md` (verification of upstream claims, in more depth),
`data/README.md` (data handling), `results/*/README.md` (per-run detail), `PROJECT_CONTEXT.md`
(original framing).

---

## 1. Problem and research questions

Task (MagNet Challenge 2): given the past B(t) and H(t) of a magnetic core, the upcoming
B'(t) and the temperature T, predict the upcoming H'(t). Sampled at 16 MHz. The model must be
frequency-, time-step- and initial-state-agnostic.

- **RQ1, efficiency:** can a state-space model (S4D) match competitive accuracy with few
  parameters? Measured as official error vs parameter count, against the 32-team
  leaderboard.
- **RQ2, generalization:** how well does it transfer to materials it never trained on?
  (Planned: train on the 10 known materials, test on held-out ones.)

## 2. Data

Source: MagNetX Database (Dropbox, downloaded 2026-11-04), stored outside git in
`~/magnet-data/`, packed losslessly with `src.pack_data` into `data/packed/` (section 4.1).

| set | content | use |
|---|---|---|
| Training Data (10 known materials) | 3C90 17,269 / 3C94 10,413 / 3E6 10,583 / 3F4 11,100 / 77 11,909 / 78 10,726 / N27 12,844 / N30 12,105 / N49 7,190 / N87 12,617 sequences. 7 frequency buckets each (50–800 kHz; **N49: buckets 1–5 only**). T ∈ {25, 50, 70} °C. | RQ2 (planned) |
| Final Evaluation / Training (A–E) | A 105 / B 6,669 / C 7,990 / D 9,530 / E 6,445 sequences (**B: buckets 1–5 only**) | training for the A–E protocol |
| Final Evaluation / Testing (A–E) + ground truth | A (3C92) 1,260 / B (3C95) 945 / C (FEC007) 1,116 / D (FEC014) 1,242 / E (T37) 1,080 segments of 1,000 steps. Warm-ups 100 / 500 / 900 steps (equal thirds); **C: warm-up of 1 step (H(1) = 0)** | official test |
| Pretest (10 known materials) | 6,300 segments each (N49: 4,500), warm-ups 100 / 500 / 900 | **not usable as a clean test (see below)** |

Facts established while handling the data:

- **[caveat] Pretest segments are cut from the training sequences.** 58 / 60 sampled pretest
  segments of 3C90 and of N87 appear verbatim inside that material's training rows. Scores on
  the pretest set are therefore inflated, and we do not report them as test results. The A–E
  test segments do not appear in the A–E training data (0 / 60 for each material).
- **[caveat] Material C test segments start demagnetized** (median |H(0)| = 0.001 A/m,
  |B(0)| = 0.0064 T); C training sequences do not (median |H(0)| = 6.93 A/m). C is an
  out-of-distribution starting state by design ("no past H" challenge).
- Sequence counts differ from the organizers' April 2025 slide (e.g. 3C90 17,269 vs 13,587).
  We use the counts in `data/packed/manifest.json`.
- All measured values are float32-representable (round-trip error ~5e-15, CSV rounding only).

## 3. Evaluation

### 3.1 Protocol

Autoregressive rollout (`src/rollout.py`), mirroring the upstream evaluator: each test
sequence is seeded with its true H over the warm-up (read per sequence from the NaN
position); the model predicts the next H from an 80-step window of past B and past H, the
next B and T; each prediction is fed back into the H window, up to step 1,000. Warm-ups
shorter than 80 are left-padded with the first value; a warm-up of 0 uses zero history.

### 3.2 Metrics

Two families are implemented (`src/metrics.py`). Only the official family is comparable to
the leaderboard.

| | official (leaderboard) | tutorial-3 (upstream notebook) |
|---|---|---|
| sequence error, per sequence | RMS(H_pred − H_meas) / RMS(H_meas), predicted region | RMSE / mean(\|H_meas\|), predicted region |
| energy error | per sequence: (∫dB·H_pred − ∫dB·H_meas over predicted segment) / core loss of the full excitation (`Loss` field) | one RMS over sequences of \|full-span energy\| |
| aggregation | 95th percentile (MATLAB `prctile`, Hazen) **within each warm-up block, averaged over blocks**; then mean over A–E | 95th percentile over all sequences (NumPy linear) |

Source of the official definition: `data/Evaluation Tools/Evaluation_Read.m` (the organizers'
scoring script), Webinar-6 slides, `Pre_Evaluation_Rules.pdf`.

### 3.3 Verification of the evaluation harness **[result]**

1. **Upstream reproduction gate.** The tutorial-3 model, weights and normalization file,
   run through our data layer, rollout and metrics, reproduce every number printed in the
   upstream notebook: sequence 33.27 % p95 / 16.13 % mean, energy 39.78 %, and all five
   per-warm-up means. Predictions match the notebook's saved CSV point by point (< 1e-3
   relative). `tests/test_reproduction.py`; `python -m src.evaluate --reference tutorial3`.
2. **Leaderboard reproduction.** Re-scoring four teams' submitted predictions with our scorer
   reproduces their published per-material scores to 4 decimals: Hefei 9.1727 % / 1.5654 %,
   Gatech 38.7600 / 7.9701, Cambridge 37.0127 / 9.2391, SiegenPaderborn 20.4316 / 3.4051
   (sequence / energy, mean over A–E). `tests/test_final_eval.py`; `src/score_submission.py`.
3. **Leaderboard table.** Per-material scores extracted from the organizers' workbook
   (`Result_Final.xlsx`) average to the README leaderboard within 0.005 for 31 / 32 teams
   (Tokyo's C entry is ~1.2e7 %, shown as 800 in the README).
4. Our own results are re-scored independently from the written submission CSVs before being
   reported; all matched.

### 3.4 Findings about the upstream reference material **[caveat]**

- The "official baseline 33.27 %" figure in the project brief belongs to the **2,399-param**
  tutorial-3 model (LSTM hidden 12, ReLU), not the 771-param one; its notebook prints
  "Number of parameters: 2399".
- The ten 771-param pretest weight files have **no published normalization file** (the
  evaluator reads one from a private path; the pretest download has none). With the
  repo's JSON files they score 51.6 % / 63.6 % (tutorial-3 metric, demo set), matching no
  target. The 771-param architecture is therefore **retrained** on our splits as the
  small-model baseline.
- The upstream rollout writes `out`-normalized predictions into an `H`-normalized window (a
  unit mismatch). It is reproduced for the gate; our models use tied normalization, so it
  cannot affect them.

## 4. Methods

### 4.1 Data pipeline

- `src.pack_data`: reads the per-material zips in place and writes one float32 HDF5 per
  material (shuffle + gzip). 33.5 GB of CSV becomes 5.8 GB of training data, with no loss
  (verified bucket by bucket against the CSVs for N87). A manifest records sequence counts
  and sha256 per file.
- Splits are **by sequence** (never by window), stratified by frequency bucket.
- Next-step samples: 80-step windows of B and H, plus the next B, T and the target next H.
  For the A–E protocol, windows are sampled uniformly from each training sequence with a
  fixed budget of 500,000 windows per material (dense for A's ~95 sequences, sparse for
  others).
- Normalization: z-scoring with statistics fitted on the training split. **Tied** mode
  (ours): the next-B input shares B's statistics and the target shares H's, so predictions
  can be fed back with no change of units. Constant features (single temperature) get std 1.

### 4.2 Models (exact trainable parameter counts)

| name | architecture | params | breakdown |
|---|---|---|---|
| Dual-LSTM, official arch. | two LSTM(1, 6) over the B and H windows; last states ++ [B_next, T] → Linear 14→14, Tanh, 14→8, Tanh, 8→1 | 771 | LSTM 216 + 216, head 339 |
| Dual-LSTM, tutorial arch. | two LSTM(1, 12); ReLU head 26→26→8→4→1 | 2,399 | LSTM 720 + 720, head 959 |
| S4D small | d_model 4, 1 layer, state 8, head hidden 8 | 205 | encoder 12, blocks 120, norm 8, head 65 |
| S4D medium | d_model 8, 2 layers, state 16 | 1,001 | encoder 24, blocks 864, norm 16, head 97 |
| S4D large | d_model 16, 3 layers, state 16 | 3,601 | encoder 48, blocks 3,360, norm 32, head 161 |

S4D model (`src/models/ssm.py`): the window is a 2-channel sequence (B_t, H_t), lifted
pointwise to d_model channels, then L residual blocks (pre-LayerNorm → diagonal SSM
convolution with D skip → GELU → GLU channel mixing). Each channel has d_state/2
complex-conjugate modes with S4D-Lin initialization (A = −1/2 + iπn), a learned step size
(init 0.01–0.1) and zero-order-hold discretization, following the minimal S4D reference
(state-spaces/s4, Apache-2.0). It is written in real arithmetic and applied as a causal
depthwise convolution, so it runs on CPU, CUDA and Apple MPS. The head reads the last-step
features ++ [B_next, T] through a Tanh MLP. Tests verify the kernel against an explicit
recurrence and check causality.

### 4.3 Training

- **Stage 1, next-step (teacher-forced):** MSE on the normalized next H. Adam, lr 1e-3
  decayed ×0.6 every 25 epochs, 80 epochs, batch 2,048, gradient clip 1.0.
- **H-history noise (`--h-noise σ`):** Gaussian noise of std σ (normalized units) added to
  the H window of each training batch, to expose the model to imperfect fed-back history.
- **Stage 2, multi-step (`--ms-horizon K`), added 2026-11-05:** continues from the stage-1
  checkpoint with a K-step autoregressive rollout loss: starting from the true H over the
  first 80 steps of a chunk, predict K steps, feeding predictions back, MSE over all K
  steps, backpropagated through the whole rollout. The horizon ramps 1 → K over the first 4
  epochs. 12 epochs, lr 3e-4, batch 512, 30,000 chunks per material, same noise and
  normalization as stage 1. **Stage 2 is kept only if it beats stage 1 on the validation
  rollout.** A test verifies the training-time rollout is the same computation as the
  evaluation rollout (`tests/test_multistep.py`).
- **Model selection:** the checkpoint with the lowest 95th-percentile rollout error on 200
  validation segments (from the held-out 10 % of training sequences, using the test set's
  warm-ups), evaluated every 5 epochs in stage 1 and every epoch in stage 2. The test set
  is never used for selection.
- Seeds: `--seeds` varies model init and batch order on a fixed data split (split seed 1).

### 4.4 Compute

MacBook Pro M3 Pro (18 GB), PyTorch 2.15 on Apple MPS, Python 3.12. Next-step step times
(batch 2,048): LSTM-771 12.5 ms, S4D-205 30 ms, S4D-1001 73 ms. Multi-step: 8.3 / 13.9 /
35 µs per predicted step. Per-material wall clock (stage 1, 80 epochs): about 3.5 / 9 / 23
min.

## 5. Experiments (chronological)

### 5.1 Pipeline proof-of-life on demo data, 2026-11-02/03 **[result, not a finding]**

Trained on the upstream 3C90 demo file (17,500 windows), scored on the 105-segment tutorial-3
set (tutorial-3 metric): 771-param LSTM 655 % p95, S4D 1,001 params 1,145 %. Upstream's own
demo-trained weights score 176 %. Conclusion: a tiny dataset gives unstable rollouts; this
only validated the plumbing. (`README.md`, `results/eval/`.)

### 5.2 Phase 1: official A–E protocol, next-step only, 2026-11-05 **[result, 1 seed]**

Setup: section 4.3 stage 1, no noise. Official sequence error (%):

| model | params | A | B | C | D | E | **mean** | rank / 35 |
|---|---|---|---|---|---|---|---|---|
| Dual-LSTM | 771 | 257.1 | 207.9 | 170.9 | 85.1 | 94.5 | **163.1** | 25 |
| S4D small | 205 | 114.4 | 278.1 | 709.0 | 109.8 | 77.1 | **257.7** | 29 |
| S4D medium | 1,001 | 987.8 | 902.9 | 659.4 | 614.3 | 283.9 | **689.6** | 33 |

Energy (mean): 49.0 / 29.8 / 104.0. Rank is among the 32 teams plus our 3 models. Source:
`results/final_AtoE/` (commit 7f6e162).

Observations: next-step training MSE ends at 1e-4 to 6e-3 (normalized), while validation
rollout error swings 10–100× between checkpoints 5 epochs apart (LSTM on B: 65 % to 4,610 %).
**The dominant failure is autoregressive drift, not fitting.** Selection is noisy (S4D small
on C: 47 % validation vs 709 % test). The larger S4D drifted more on every material.

### 5.3 Phase 2: phase 1 + H-history noise σ = 0.05, 2026-11-05 **[result, 1 seed]**

The only change from 5.2 is `--h-noise 0.05`. Official sequence error (%):

| model | params | A | B | C | D | E | **mean** | phase 1 | rank / 35 |
|---|---|---|---|---|---|---|---|---|---|
| S4D small | 205 | 85.0 | 51.5 | 79.1 | 36.2 | 53.0 | **61.0** | 257.7 | 22 |
| S4D medium | 1,001 | 89.5 | 94.4 | 43.3 | 38.8 | 64.9 | **66.2** | 689.6 | 23 |
| Dual-LSTM | 771 | 106.2 | 27.2 | 76.0 | 37.5 | 91.6 | **67.7** | 163.1 | 25 |

Energy (mean): S4D small 17.0, LSTM 19.2, S4D medium 22.2. Source:
`results/final_AtoE_hnoise/` (commit 8f7846f).

Findings:

- **Noise on the H history reduces the error 2.4× (LSTM) to 10× (S4D medium).** This is
  direct evidence that exposure bias / drift was the main error source.
- With noise, the 205-param S4D has the lowest mean among our models, beating the stock
  2,399-param LSTM of NTU2 (75.1 %) with 12× fewer parameters. The spread among our three
  models (61–68 %) is small next to per-material variation. **[preliminary]** With one seed
  this ordering is not established.
- Reference points (same metric and test): SiegenPaderborn 20.4 % (325 params), Cambridge
  37.0 % (1,182), Gatech 38.8 % (83), Hefei 9.2 % (6,564, winner). We are about 3× from the
  small-model frontier.

### 5.4 Noise-level sweep with multi-step stage, 2026-11-05 **[result, 1 seed]**

S4D small (205 params), one seed, σ ∈ {0.02, 0.05, 0.1}, each followed by stage 2 (K = 32,
section 4.3). Purpose: pick σ for the 3-seed run, and test stage 2. Source:
`results/sweep_noise/` (08:22–11:22).

Official sequence error (%):

| setting | A | B | C | D | E | **mean** | energy |
|---|---|---|---|---|---|---|---|
| σ 0.02 + stage 2 | 100.0 | 23.1 | 76.9 | 46.9 | 56.0 | 60.6 | 22.0 |
| σ 0.05 + stage 2 | 85.0 | 25.7 | 79.1 | 34.6 | 53.0 | 55.5 | 17.4 |
| **σ 0.1 + stage 2** | 83.7 | 24.0 | 51.7 | 38.5 | 46.0 | **48.8** | **12.3** |
| phase 2: σ 0.05, stage 1 only (5.3) | 85.0 | 51.5 | 79.1 | 36.2 | 53.0 | 61.0 | 17.0 |

Stage 2 was kept on 5 / 5 materials at σ 0.02, 2 / 5 at σ 0.05 (B, D) and 3 / 5 at σ 0.1
(C, D, E). Per-material keep decisions and validation numbers are in
`results/sweep_noise/README.md`.

Findings:

- **The multi-step stage improves the official score, shown in a controlled comparison.**
  At σ = 0.05 with the same seed as phase 2, the only difference is stage 2. Where stage 2
  was kept, the test score improved (B 51.5 → 25.7 %, D 36.2 → 34.6 %). Where it was not
  kept, scores are identical to phase 2 (A, C, E), which confirms the comparison is exact.
  Mean 61.0 → 55.5 %.
- **The keep-if-better-on-validation rule produced no test regression** in the controlled
  comparison.
- **More noise helped on average** (60.6 → 55.5 → 48.8 %), mostly through C (76.9 → 51.7 %)
  and E (56.0 → 46.0 %). σ = 0.1 is the largest value tried; the optimum may be higher.
  **[preliminary]** single seed.
- Multi-step training stabilizes the validation rollout. Material A, σ 0.02: within
  117–126 % over all 12 stage-2 epochs, against 123–435 % across stage-1 checkpoints. But
  stage 2 improves the selected checkpoint by only 3–14 % on validation. K = 32 is short
  next to the 900-step test horizon.
- **[caveat]** Validation does not always rank test outcomes. C at σ 0.02: 33.8 %
  validation, 76.9 % test.

**Decision:** σ = 0.1 + stage 2 for the 3-seed run. Chosen on validation (mean selected
validation p95 49.9 % vs 52.0 / 59.0 %), not on test; it is also best on test.

### 5.5 Three seeds per model at σ 0.1 + stage 2, 2026-11-05/06 **[result, 3 seeds]**

{S4D small 205, Dual-LSTM 771, S4D medium 1,001} × seeds {1, 2, 3}, setting from 5.4, same
split (seeds vary init and batch order). 45 runs, 11:31 → 00:22, including a ~1 h pause with
the lid closed. Source: `results/seeds3/`; launch script `scripts/runs/seeds3.sh`.

Official sequence error (%):

| model | params | seed 1 | seed 2 | seed 3 | **mean ± sd** | energy mean ± sd | rank / 35 |
|---|---|---|---|---|---|---|---|
| S4D medium | 1,001 | 49.0 | 46.8 | 56.6 | **50.8 ± 5.1** | 16.0 ± 0.8 | 19 |
| S4D small | 205 | 48.8 | 53.2 | 55.2 | **52.4 ± 3.3** | 19.6 ± 6.6 | 21 |
| Dual-LSTM | 771 | 69.4 | 57.7 | 64.8 | **64.0 ± 5.9** | 14.9 ± 2.5 | 24 |

Paired over the 15 (material, seed) pairs, exact two-sided sign-flip test:

| comparison | seed-mean difference | better on | p |
|---|---|---|---|
| S4D small − Dual-LSTM | −11.6 pts | 14 / 15 | 0.0011 |
| S4D medium − Dual-LSTM | −13.2 pts | 12 / 15 | 0.014 |
| S4D medium − S4D small | −1.6 pts | 11 / 15 | 0.61 |

Per-material means and spreads are in `results/seeds3/README.md`.

Findings:

- **RQ1 evidence: the S4D beats the matched-recipe LSTM baseline with fewer parameters.**
  The 205-param S4D is better than the 771-param Dual-LSTM on every seed (by 4.5–20.6 pts)
  and on 14 / 15 material–seed pairs (p = 0.001), under identical data, training, selection
  and scoring.
- **S4D size does not matter in this range:** 1,001 vs 205 params differ by 1.6 pts
  (p = 0.61).
- **[caveat] Energy error does not follow sequence error:** the LSTM has the lowest mean
  energy error (14.9 %) vs 16.0 / 19.6 % for the S4D models. The S4D advantage is on the
  waveform metric.
- **Position:** S4D medium 50.8 % ranks 19 / 35; S4D small 52.4 % ranks 21. Both are ahead of
  NTU2 (75.1 %, 2,399 params), Aalborg2 (59.5 %, 415,753) and ASU (54.1 %, 18,337). The
  small-model frontier is still about 2.5× better: SiegenPaderborn 20.4 % (325 params),
  Cambridge 37.0 % (1,182), Gatech 38.8 % (83).
- **Material A** (~95 training sequences) is the weak point for every model (92–102 %) and
  the noisiest across seeds.
- **[caveat] Single-seed numbers overstated the S4D:** the sweep's 48.8 % for S4D small was
  its best seed; the 3-seed mean is 52.4 %. Seed spread (sd 3–6 pts) is as large as the
  gains between the sweep settings in 5.4, so those σ differences are not established.

Progress of the S4D small mean across the day: 257.7 % (5.2) → 61.0 % (5.3, noise) →
52.4 ± 3.3 % (5.5, noise 0.1 + multi-step, 3 seeds).

## 6. Decisions log

| date | decision | reason |
|---|---|---|
| 11-02 | Mirror the upstream rollout and metrics exactly before any new model | comparability; the reproduction gate proves correctness |
| 11-02 | Report both metric families; headline the official one | tutorial-3 is verifiable against printed numbers; only the official one is leaderboard-comparable |
| 11-03 | Retrain the 771-param baseline instead of using the pretest weights | no published normalization for those weights |
| 11-03 | Keep team submissions in the repo (LFS) | they let us verify our scorer exactly against the leaderboard |
| 11-04 | Keep the 23 GB database out of git; pack per machine | exceeds the free LFS quota; packing is lossless and 4× smaller |
| 11-05 | Evaluate on the official A–E test, not pretest | pretest segments overlap the training sequences |
| 11-05 | Select checkpoints by validation rollout error, not next-step loss | next-step loss does not predict rollout quality (5.2) |
| 11-05 | Add H-history noise (phase 2), then multi-step training | phase 1 showed drift dominates |
| 11-05 | Keep stage 2 only if it beats stage 1 on validation | guard against a regression from the extra stage |
| 11-05 | σ = 0.1 + stage 2 for the 3-seed run | lowest validation error in the sweep (5.4); chosen without using test scores |

## 7. Open caveats for the paper

- Seed variance is now measured for one setting (5.5: sd 3–6 pts on the mean). The sweep
  (5.4) and phases 1–2 are single-seed, so differences smaller than ~6 pts there are not
  established.
- Only one training recipe tuned (none of lr, epochs, window budget swept).
- Model selection uses 200 validation segments; selected-checkpoint transfer to test is
  inconsistent (5.2).
- Pretest overlap means known-material results need a held-out split we construct ourselves
  (`src.experiments.cross_material` builds one by sequence) for RQ2.
- Material A has ~95 training sequences; its results carry the largest uncertainty.
- Compute is a laptop GPU; the planned cluster runs may change the time budget per model.

## 8. Reproducing

```bash
scripts/setup_env.sh && source ~/.venvs/magnet-ssm/bin/activate
scripts/fetch_reference.sh
python -m src.pack_data --src ~/magnet-data/"MagNetX Database" --out data/packed
python -m pytest -q                                  # 71 tests
python -m src.evaluate --reference tutorial3         # reproduction gate
python -m src.score_submission --team Hefei          # leaderboard scorer check
# phase 2 (section 5.3):
python -m src.experiments.final_eval --models official_lstm ssm_s ssm_m \
  --epochs 80 --decay-epoch 25 --batch-size 2048 --max-windows 500000 \
  --rollout-val --rollout-val-every 5 --h-noise 0.05 --device mps --out results/final_AtoE_hnoise
```

Exact commands for every run are in `scripts/runs/*.sh` (copies of the launch scripts) and in
each run's `config.json` (stored with its checkpoint under `results/<run>/models/`).

## 9. Keeping this log current

Append an entry to section 5 for every experiment that runs on real data: date, the exact
change from the previous run, a results table with the official scores, the source folder
and commit, and the observations that follow from the numbers. Add a row to section 6 for
every design decision, and to section 7 for every new caveat.
