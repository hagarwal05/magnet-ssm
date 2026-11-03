# Reference numbers

## `leaderboard_mc2.csv`

Transcribed verbatim from the "[News #13] Preliminary ranking of 32 teams" table in the
upstream `README.md` (minjiechen/magnetchallenge-2 @ ab4913c). One row per team, averaged
over the five final-test materials A to E.

## `leaderboard_mc2_per_material.csv`

Extracted from the organizers' workbook `data/Result_Final.xlsx` (sheets `P95Error` and
`Parameters`). One row per team and material, with the official sequence score, energy score
and parameter count. The mean over A to E reproduces the README averages within 0.005 for 31
of 32 teams (`tests/test_final_eval.py`). The exception is Tokyo: its Material C entry is
about 1.2e7 %, which the README shows as a placeholder 800.

Materials: A = 3C92, B = 3C95, C = FEC007, D = FEC014, E = T37.

## How these numbers are computed (data/Evaluation Tools/Evaluation_Read.m)

1. Sequence error per sequence: RMS(H_pred - H_meas) / RMS(H_meas) over the predicted region.
2. Energy error per sequence: predicted-segment energy error divided by the file's `Loss`
   (full-excitation core loss).
3. Within each warm-up block (100 / 500 / 900 steps; Material C has a single 1-step block),
   take MATLAB `prctile(|err|, 95)`, then average over the blocks. That is the per-material
   number.
4. "Avg. 95%" = mean of the per-material numbers over A to E.

`src.evaluate` reports this as `official_seq_score` / `official_energy_score`. Those are
the only numbers directly comparable to these tables, and only when computed on the A to E
test files in `data/Evaluation Tools/Ground Truth Data/`.
