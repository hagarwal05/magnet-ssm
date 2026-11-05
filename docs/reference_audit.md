# Reference audit: what the upstream repo actually supports

Written 2026-11-03 while building this codebase. Every statement below was checked against
`minjiechen/magnetchallenge-2` at commit `ab4913c` (cloned to `third_party/`) or by running
our code. Several findings contradict `PROJECT_CONTEXT.md`; those are marked **[differs]**
and need a decision from Harshil before the paper relies on them.

## 1. The reproduction gate numbers belong to the 2,399-parameter model  [differs]

`PROJECT_CONTEXT.md` lists "official baseline (~771 params): 33.27% p95, 16.13% mean, 39.78%
energy". Those three numbers come from the saved outputs of
`tutorials/tutorial-3/Demo_LSTM_Evaluation.ipynb`. That notebook defines
`Net` with **`LSTM(1, 12)` + ReLU (the 2,399-parameter tutorial model)**, loads
`tutorial-3/Model_LSTM.sd`, and prints `Number of parameters: 2399`.

Running our pipeline:

| weights | arch | normalization | seq p95 | seq mean | energy |
|---|---|---|---|---|---|
| `tutorial-3/Model_LSTM.sd` | 2,399 | tutorial-3 | **33.27** | **16.13** | **39.78** |
| `Pretest/Pretest_Models/Model_LSTM_3C90.sd` | 771 | tutorial-3 | 51.59 | 17.69 | 48.71 |
| `Pretest/Pretest_Models/Model_LSTM_3C90.sd` | 771 | tutorial-1 | 63.61 | 26.40 | 49.76 |

Only the first row matches the targets, and it matches every printed number (mean, p95,
energy, all five per-warm-up means) to rounding. Element-wise, our predictions match the
notebook's saved `Output/pred.csv` within 1e-3 relative (`tests/test_reproduction.py`).

So the correctness gate is reproduced, and it validates our data layer, normalization
handling, rollout and metrics. But it is the **2,399-parameter** model's result, not the
771-parameter one.

## 2. The 771-parameter models have no published normalization or result on this test set

- The ten `Pretest/Pretest_Models/Model_LSTM_<M>.sd` files are the 771-parameter
  architecture (all ten load strictly into `OfficialLSTM`).
- `Test/EvaluationExample.ipynb` and `Pretest/PreEvaluationExample.ipynb` read
  `Normalization_Params.json` from private paths (`C:/Users/Labadmin/...`). Neither notebook
  prints any metric. The matching normalization files are not in the repo.
- Neither JSON in the repo reproduces anything with these weights (table above).
- The pretest evaluation files (>4,500 segments per material) live in the Dropbox
  `MagNet_Challenge_2_Pretest` folder, not in the repo.

Consequence: the 771-parameter baseline cannot be gated from the repo alone.
`src/reference.py` therefore pins `norm=None` for these weights, and `evaluate.py` refuses to
run them without an explicit `--norm`. If the Dropbox pretest folder ships per-material
normalization files, pass them with `--norm`. Otherwise, use the **retrained** 771-parameter
model (same architecture, trained by `src.train` on our split) as the small-model reference.
That is a fair, like-for-like baseline and needs no missing file.

## 3. The tutorial-3 metric is not the grading metric  [differs]

`PROJECT_CONTEXT.md` says to use the tutorial-3 formulas "so our numbers line up with the
published leaderboard". The organizers' grading definitions (Webinar-6 slides p.7,
`Pre_Evaluation_Rules.pdf`, `Test/Error_Distribution_Plot_Template.m`) are different:

| | tutorial-3 notebook | grading (leaderboard) |
|---|---|---|
| sequence error | RMSE / **mean(abs(meas))** | RMS(err) / **RMS(meas)** |
| energy error | one RMS over all sequences of abs(full-span energy) | per-sequence (predicted-segment energy error) / (full-excitation core loss), then p95 |
| percentile | NumPy linear | MATLAB `prctile` (Hazen) |

On the tutorial-3 data the two sequence metrics give p95 33.27% vs 28.31%. The energy
metrics are not comparable at all: the "Avg. 95% Energy Error" column is the 95th
percentile of the second definition.

What the code does: it reports **both families** for every evaluation (`seq_nrmse_*` and
`energy_nrmse` follow tutorial-3, `seq_rms_*` and `energy_rel_*` follow grading). The
tutorial-3 family stays the headline only because it is the one we can verify against
printed numbers. Any leaderboard comparison should use `official_seq_score` and
`official_energy_score` (section 7), which also apply the organizers' per-block
aggregation. The leaderboard
energy column needs the `Loss` field (full-excitation core loss), which ships with the
official test files. Without it, `energy_rel_*` falls back to the segment's own measured
energy, and the output says so (`energy_rel_denominator`).

## 4. The leaderboard is on materials A to E only

The README's ranking table averages the five **final-test** materials (A to E), scored by
the organizers. A number from the ten known materials, or from our own held-out split, is a
different test set. The RQ1 plot shows the leaderboard as grey context points and says so in
its subtitle. A like-for-like comparison requires running on the A to E test CSVs from the
Dropbox final-evaluation folder (`src.evaluate --test-dir ... --materials A B C D E`) with
the grading metric.

Also from Webinar 6: Material A has 105 training sequences ("tiny data"), Material C
provides only H(1) = 0 as warm-up ("no past H"), and the warm-ups there are 10/50/90% known
(100/500/900 steps). The 100/300/500/700/900 mix is the tutorial-3 demo file only. The
rollout reads the warm-up per sequence, so all of these work, including warm-up = 1 (left
padding) and warm-up = 0 (zero history). The A to E ground truth is now committed (section 7),
so this comparison can be run.

## 5. Other details confirmed

- Window 80; the history window B[t-80:t] uses B *before* t, and B[t] is the separate
  "next B" input; the H window is fed back with the raw model output (see 6).
- The training HDF5 demo file holds 7 x 2,500 windows (17,500 total).
- Fitting normalization on the demo file reproduces `tutorial-1/Normalization_Params.json`
  to about 1e-3 relative (float32 vs float64 reduction), so that file came from this data.
- `tutorial-1/Model_LSTM_Princeton.sd` with the tutorial-1 normalization scores 87.6% p95 on
  the tutorial-3 test set, and the demo-trained `Model_LSTM_New.sd` scores 175.6%.

## 6. A latent upstream bug we reproduce on purpose

The upstream rollout writes the model's output, which is in `out`-normalized units, directly
into the H window, which is in `H`-normalized units. The two differ slightly for the
upstream JSON files. `rollout()` reproduces this by default, which is required to hit the
gate. Our own models are trained with **tied** normalization (mean_out = mean_H,
std_out = std_H, mean_Scal = mean_B, std_Scal = std_B), so the bug cannot affect them.
`--convert-feedback` does the conversion explicitly.

## 7. Final-test ground truth (added 2026-11-03, commit 4c3060d)

`data/Evaluation Tools/` (from the "all teams' results" Dropbox folder) holds the A to E
ground truth and the organizers' scoring script `Evaluation_Read.m`:

| | material | segments | warm-ups | notes |
|---|---|---|---|---|
| A | 3C92 | 1,260 | 100 / 500 / 900 (420 each) | "tiny data": 105 training sequences |
| B | 3C95 | 945 | 100 / 500 / 900 (315 each) | "frequency" challenge |
| C | FEC007 | 1,116 | 1 | "no past H": padded H(1) = 0 A/m while the measured H(1) is within ±0.21 A/m |
| D | FEC014 | 1,242 | 100 / 500 / 900 (414 each) | "flux density" challenge |
| E | T37 | 1,080 | 100 / 500 / 900 (360 each) | "special material" |

All five load through `find_test_set`, with `True_H_seq` and `Loss` present. The
padded/true consistency check now allows a 1-step warm-up to differ from the measurement
(Material C), and still rejects any longer mismatch.

`Evaluation_Read.m` settles the aggregation question from section 3. The per-material score
is MATLAB `prctile(|err|, 95)` **within each warm-up block, averaged over blocks**, not a
single p95 over all sequences. `metrics.official_score` implements it.
`results/reference/leaderboard_mc2_per_material.csv` was extracted from
`data/Result_Final.xlsx`, and its A to E means reproduce the README table within 0.005 for
31 of 32 teams (Tokyo's C entry is ~1.2e7 %, shown as 800 in the README).

**Grading-path sanity check (not a result of ours).** The upstream tutorial-3 weights
(2,399 params, trained on 3C90 only) applied zero-shot to A to E score official seq
99.4 / 63.8 / 58.4 / 86.2 / 145.2 % (mean 90.6 %) and energy mean 18.4 %
(`results/eval/final_AtoE_tutorial3_weights/`). NTU2 submitted the same architecture trained
by themselves and scored 75.11 % / 27.43 %. The same order of magnitude is what you'd expect
for a 3C90-only model, which suggests the grading path is wired correctly. It is not an
exact cross-check: NTU2's weights are not available.

**Exact cross-check (2026-11-03, commit 15866e7 added `data/Competition Result/`).** Re-scoring
teams' submitted `H_Material {A..E}.csv` with `src.score_submission` reproduces their
published per-material scores to 4 decimals for Hefei (seq 9.1727 % / energy 1.5654 % mean),
Gatech (38.7600 / 7.9701), Cambridge (37.0127 / 9.2391) and SiegenPaderborn
(20.4316 / 3.4051). Covered by `tests/test_final_eval.py`. Our grading path is therefore
verified end to end against the organizers' own numbers, not just sanity-checked.

The folder holds 32 teams' predictions (5 x `H_Material *.csv` each, ~2.9 GB in LFS) and
the organizers' histogram PNGs. Model files over GitHub's limit were not pushed. Only the
predictions are used. Pull one team with
`git lfs pull --include="data/Competition Result/<Team>/Result/*"`.

## 8. The MagNetX Database download (2026-11-05)

Downloaded to the project root as `MagNetX Database/` (git-ignored). It matches its outer zip
file for file (146 files, identical sizes), and all 10 training zips pass `unzip -t`.

- **Training Data:** per-material zips of `<M>_<i>_{B,H,T}.csv`. 9 materials have all 7
  frequency buckets; **N49 has buckets 1–5 only** (no 500 / 800 kHz). Values are
  float32-representable, so `src.pack_data` stores them losslessly as float32 HDF5. N87 was
  verified bucket by bucket against the CSVs.
- **Sequence counts are higher than the organizers' slide** for every known material (e.g.
  3C90 17,269 vs 13,587; 3E6 10,583 vs 7,407). The exception is N49 (7,190 vs 7,266). The
  slide is dated April 2025 and the zips were re-uploaded on April 8 (N49 on June 11). Use
  the manifest counts, not the slide.
- **A–E training:** A has 105 sequences across all 7 buckets, as stated. **B has buckets 1–5
  only.** B–E counts are a little below the slide (B 6,669 vs 6,758; E 6,445 vs 6,901).
- **Pretest:** `<M>_Testing_{Padded,True}.h5` for all 10 materials. 6,300 segments each
  (N49 4,500), warm-ups 100 / 500 / 900, `Loss` included. **No normalization files or model
  weights**, so the 771-parameter pretest models still can't be gated (section 2). The
  pretest sets are a second, much larger test set for the known materials (63k segments vs
  the 105 demo segments).

## 9. What `PROJECT_CONTEXT.md` says exists but is not in this folder

The context says the pipeline, harness, scaffold and an SSM "overfits one batch" already
exist. This folder contained only `PROJECT_CONTEXT.md`, and no project code or `.h5` data
was found elsewhere on this Mac. Everything here was written fresh. The Dropbox data is not
present (Dropbox links refused connections from here).
