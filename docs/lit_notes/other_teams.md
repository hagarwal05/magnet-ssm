# Literature notes: other notable MagNet Challenge 2 teams

Author: winner-others (literature-review team). Scope: every notable MC2 team except
SiegenPaderborn / RHINO-MAG, which is in `docs/lit_notes/siegen_paderborn.md`.
Last updated 2026-11-06.

Conventions:

- **"official"** = the leaderboard metric: per-sequence RMS(H_pred − H_meas)/RMS(H_meas) over
  the predicted region, 95th percentile (Hazen) **within each warm-up block**, averaged over
  blocks, then over materials A–E (RESEARCH_LOG §3.2). Many papers report other aggregations
  (mean, pooled p95, other datasets). The aggregation is stated for every number below.
- **[measured]** = computed by us (winner-others) on 2026-11-06. We re-scored the team's
  submitted CSVs (`data/Competition Result/<team>/Result`) with our scorer (`src/metrics.py`,
  `src/score_submission.py`). For every team we re-scored, our official per-material score
  reproduces the published one to 2 decimals. Raw output: `/tmp/mc2lit/others/warmup_breakdown.txt`.
- **[inferred]** = our interpretation, not stated by the source.
- **[transcript]** = from the auto-generated YouTube captions of the team's 5-minute APEC-week
  presentation (Princeton Power Electronics channel, Feb 9–12 2026). Captions are machine
  transcribed, so technical terms are garbled ("RSTM" = LSTM, "pray operator" = play
  operator). Captions are only paraphrased here, and claims that rest only on them are marked.
- Leaderboard numbers: `results/reference/leaderboard_mc2_per_material.csv` and
  https://minjiechen.github.io/magnetchallenge-2/ (News #13). Seq rank = organizers' published
  "Avg. Seq. Error Ranking" among 32 entries. It equals the rank of each team's mean
  per-material rank, **not** of the A–E mean error. That holds only when per-material ranks
  are computed over all 32 entries *including* Princeton (32/32 match; 21/31 without
  Princeton; rank of mean error matches 12/32) [measured by reviewer-teams, review_log T18].
  Per-material ranks quoted in team sections ("1st on C", etc.) are among the 31 competing
  teams, Princeton excluded.
- Material key (organizers / MagLearn 2 Table I): A = 3C92 (105 training sequences, "tiny
  data"), B = 3C95 (frequency gap), C = FEC007 (no past H: H(1) = 0, 999 steps to predict),
  D = FEC014 (flux-density gap), E = T37 ("special material") (Webinar-6 slides,
  `/tmp/mc2lit/pdf/Webinar-6-2025.txt`).

## 0. Awards and sources (context for all sections)

**Final awards** (APEC 2026 keynote slide 16, `docs/apeckeynote.pdf` on the MC2 site,
https://minjiechen.github.io/magnetchallenge-2/docs/apeckeynote.pdf):

- Performance Track: 1. Siegen & Paderborn, 2. Hefei & CityU & Xiamen, 3. Sydney.
- Innovation Track: 1. Georgia Tech, 2. Cambridge, 3. Silicon Austria Labs.
- Software Engineering: Hefei & CityU & Xiamen.
- Honorable mentions: Tianjin, Zhejiang, UESTC, Southeast1, Fuzhou, Eindhoven, Bristol, ASU,
  Leuven, Tsinghua, Mizzou.

**Award citations** (organizers' image "MagNet Challenge 2 Winners (March 10th, 2026)",
https://github.com/minjiechen/magnetchallenge-2/blob/main/img/winners.jpg, viewed by us):

- Performance: Siegen & Paderborn "for well-balanced model size, model performance, and
  Pareto-front exploration". HFUT-CITYUHK-XMU (= Hefei) "for excellent modeling strategy and
  model accuracy". Sydney "for balanced performance and modeling strategy".
- Innovation: Georgia Tech "for innovations on mechanical analogy for magnetic systems and
  outstanding performance". Cambridge "for innovations on hybrid data-driven and analytical
  methods". Silicon Austria Labs "for compact and simple feedforward neural network
  implementation".
- Software Engineering: HFUT-CITYUHK-XMU "for deployment-ready web GUI and clear code
  instruction".
- Honorable mentions: "excellent performance" for Southeast1, Fuzhou, Eindhoven and Bristol;
  "balanced size and performance" for Tianjin, Zhejiang and UESTC; "new methods exploration"
  for ASU, Leuven, Tsinghua and Mizzou.
- Judges: Sullivan, Perreault, Maksimović, Hui, Biela, Wilkowski, Chen.

**Judging criteria.** The MC2 site lists five: accuracy, model size, explainability, novelty
and software quality. The Performance Track was therefore not decided by error alone. Hefei
has the lowest error but placed 2nd behind SiegenPaderborn (seq rank 6, size rank 3).

**Primary sources found (per team):**

| team | write-up | code | talk |
|---|---|---|---|
| Hefei (HFUT-CityUHK-XMU) | TPEL 2026 paper (LSTM-DAO) + TechRxiv preprint | inference + training/fine-tune repos | yes |
| Fuzhou | none found (4-page report not public) | none | yes |
| Sydney | arXiv 2608.02965 (PI-HNO) | none for MC2 (magnet-engine is MC1) | yes |
| Gatech (GT & GM) | TPEL 2026 letter (abstract only accessible) | none | yes |
| Cambridge | none found | none | yes |
| Bristol (MagLearn 2) | arXiv 2610.03504 + IPEC 2026 paper | MagLearn2 repo (D, E only) | yes |
| Southeast1 | none found | none | yes |
| Eindhoven | none found | none | yes |
| Tsinghua | none found | none | yes |
| SAL | none found (prize page only) | none | yes |

The 4-page final reports (due Jan 30 2026) are not public. The organizers' Pretest/Results PDFs
(github.com/minjiechen/magnetchallenge-2/tree/main/Pretest/Results) are self-reported results on
the 10 known materials with "no influence on the final evaluation". All but Fuzhou's are plot
images without text. They describe no methods.

## 1. Measured: how each team's error depends on warm-up length and on forecast horizon [measured]

### 1.1 Between-sequence comparison: error by warm-up block

Mean over materials of the per-block 95th-percentile sequence error (%). Columns w100/w500/w900
average materials A, B, D, E; Material C has a single block (warm-up 1). **The three blocks are
disjoint sets of test sequences** (0 shared B rows; review_log T55), so this table compares
different sequences. It does *not* show how error evolves along one forecast; §1.2 does.

| team | params (avg) | official | w100 | w500 | w900 | C (w1) | trend w100→w900 |
|---|---|---|---|---|---|---|---|
| Hefei | 6,564 | 9.17 | 9.86 | 6.85 | 7.88 | 13.07 | ↓ then ↑ |
| Fuzhou | 63,022 | 12.42 | 9.36 | 11.32 | 16.67 | 12.29 | **↑ (worse with more history)** |
| Southeast1 | 22,197 | 14.51 | 13.87 | 12.18 | 16.90 | 15.29 | ↓ then ↑ |
| Eindhoven | 81,893 | 17.71 | 16.00 | 16.65 | 20.67 | 17.45 | ↑ |
| Tsinghua | 2,177,283 | 19.93 | 25.81 | 15.61 | 16.45 | 22.48 | ↓ |
| SiegenPaderborn | 325 | 20.43 | 18.72 | 19.02 | 18.64 | 27.00 | flat |
| XJTU1 | 1,350,913 | 22.48 | 29.50 | 23.09 | 17.12 | 19.44 | **↓ (+12.4 at w100)** |
| Bristol | 860,929 | 25.22 | 18.44 | 16.90 | 16.59 | 56.88 | flat (↓ slightly) |
| Sydney | 4,777 | 24.63 | 25.82 | 23.72 | 22.38 | 27.22 | ↓ |
| NITC (16-step chunked AR) | 347,237 | 24.85 | 29.27 | 22.96 | 16.86 | 32.15 | **↓ (+12.4 at w100)** |
| Tianjin (one-step AR) | 7,939–38,305 | 32.56 | 25.34 | 25.28 | 30.95 | 54.06 | ↑ |
| Leuven (diffusion) | 111,985 | 34.54 | 28.06 | 23.45 | 20.55 | 76.64 | ↓ |
| Cambridge | 1,182 | 37.01 | 35.00 | 24.36 | 26.54 | 70.53 | ↓ then ↑ |
| Gatech | 83 | 38.76 | 26.44 | 24.41 | 30.98 | 84.68 | ↓ then ↑ |
| **ours, S4D 205** (3 seeds) | 205 | 52.4 | 50.5 | 49.4 | 52.8 | 56.9 | flat |
| **ours, S4D 1,001** (3 seeds) | 1,001 | 50.8 | 56.2* | 49.0 | 49.0 | 47.0 | flat* |

Our rows: seed means of `seq_w100/500/900` and `seq_w1` (Material C) in
`results/seeds3/final_eval.csv`. Those columns are the per-block p95 of sequence RMS error
written by `src/experiments/final_eval.py` (`by_warmup[..]["seq_rms_p95"]`, NumPy linear
percentile). The team rows use Hazen. The two percentile methods differ by ≤ 0.16 points on
pooled per-material errors and by up to 0.93 points per block (SiegenPaderborn A w500)
[measured; review_log T15]. That is immaterial to the comparisons here.
\* S4D-1,001's w100 = 56.2 comes from one cell: seed 3 on material B scores 103.7 %, against
~20 % for the other seeds. Its per-seed w100 means are 49.2 / 45.3 / 74.1. Without that cell
the profile is flat, like S4D-205's (review_log T3).

Cross-check: our SiegenPaderborn per-block values agree exactly with the organizers'
host-provided per-block results stored in the RHINO-MAG repo
(`rhmag/utils/final_data_evaluation.py`, `SIEGEN_PB_HOST_PROVIDED_RESULTS`; relayed by
winner-siegen). Per material w100/w500/w900: A 21.62/21.78/26.42, B 15.17/17.71/19.10,
D 20.51/16.76/11.33, E 17.56/19.83/17.71. This independently validates the re-scoring.

Caveat on "flat": the columns average over materials, and per-material trends can go in
opposite directions. For SiegenPaderborn, p95 rises with known history on A and B but falls on
D (20.51 → 11.33). Hefei falls on D (8.72 → 4.25) and E (11.25 → 8.49) but is non-monotonic on
A (13.67 / 10.95 / 13.24) [measured; per-material values in `warmup_breakdown.txt`]. The
"trend" column and "flat" in this subsection refer to the material average.

What the between-sequence table shows:

- For the top 7 (ranks 1–7), w100 and w900 block averages differ by at most 9.4 points, in
  both directions: w100 − w900 is Tsinghua +9.4, Hefei +2.0, Bristol +1.9, SiegenPaderborn +0.1,
  Southeast1 −3.0, Eindhoven −4.7 and Fuzhou −7.3. **Only SiegenPaderborn and Bristol are flat**
  across all three blocks: max/min block ratio 1.02 and 1.11, against Eindhoven 1.29,
  Southeast1 1.39, Hefei 1.44, Tsinghua 1.65 and Fuzhou 1.78 (review_log T70). Outside the top
  7: Sydney +3.4, XJTU1 and NITC +12.4 each. **None of this shows how error evolves along a
  forecast** (§1.2). The blocks are different sequences, and the w900 block is penalized by the
  metric itself (next bullet).
- Part of the w900 penalty is a property of the metric [measured]: with only 100 predicted
  steps, the denominator RMS(H_meas) over the predicted region can be small. Its 5th
  percentile, relative to the full-sequence RMS, is 0.21–0.32 at w900 vs 0.81–0.89 at w100
  (materials A, B, D, E). A short window that happens to fall near an H zero crossing inflates
  the relative error, and the p95 picks up exactly those cases [inferred].
- **Material C (no H history)** separates the teams. It is 12–17 % for Fuzhou, Hefei,
  Southeast1 and Eindhoven, but 57–85 % for Bristol, Cambridge and Gatech. Gatech explains
  theirs as a wrong demagnetized initial state (§5) [transcript]. Bristol, Cambridge and Gatech
  ranked 17th, 20th and 24th on C (of 31 competing teams, Princeton excluded). Tianjin (54.1)
  and Leuven (76.6) also fail on C.

### 1.2 Within-sequence drift: error vs steps since the boundary [measured]

Test: the w100 block only (the same sequences throughout, 900 predicted steps). For each
sequence, RMS error in each 100-step window after the boundary is divided by RMS(H_meas) over
the whole predicted region, a fixed per-sequence denominator. The table gives the p95 over
sequences, averaged over A, B, D, E (%). Script `/tmp/mc2lit/rt/drift.py`, written by
reviewer-teams (review_log T55). We re-ran it and got identical numbers
(`/tmp/mc2lit/others/drift_rerun.txt`). Window 1 = steps 1–100 after the boundary; window 9 =
steps 801–900.

| team | rank | how predicted H is produced | win1 | win3 | win5 | win7 | win9 | win9/win1 |
|---|---|---|---|---|---|---|---|---|
| Hefei | 1 | non-AR, whole sequence (code) | 6.8 | 9.5 | 10.4 | 10.2 | 11.2 | 1.65 |
| Fuzhou | 2 | seq2seq (transcript; feedback not stated) | 6.9 | 9.4 | 10.4 | 9.6 | 11.3 | 1.64 |
| Southeast1 | 3 | analytical + BiLSTM residual (transcript) | 12.4 | 13.5 | 14.9 | 15.5 | 16.6 | 1.34 |
| Eindhoven | 4 | encoder–decoder (transcript; feedback not stated) | 15.0 | 15.4 | 16.6 | 17.3 | 17.2 | 1.15 |
| Tsinghua | 5 | B only, BiLSTM (transcript) | 13.8 | 22.4 | 27.9 | 28.3 | 31.9 | 2.31 |
| SiegenPaderborn | 6 | closed-loop GRU state (code) | 10.5 | 16.4 | 19.2 | 22.2 | 24.6 | 2.34 |
| Bristol | 7 | non-AR encoder–decoder (code + paper) | 14.6 | 19.4 | 19.4 | 18.4 | 20.0 | 1.37 |
| XJTU1 | 8 | unknown | 12.1 | 23.7 | 30.2 | 35.1 | 39.1 | 3.23 |
| Sydney | 9 | non-AR, decoder trained on 100-step targets (paper) | 17.1 | 22.3 | 27.5 | 30.4 | 32.4 | 1.89 |
| NITC | 10 | AR, 16-step chunks (transcript) | 10.7 | 23.9 | 30.0 | 35.9 | 39.6 | 3.70 |
| Tianjin | 13 | AR, one step (transcript) | 17.2 | 24.2 | 26.8 | 29.0 | 34.5 | 2.01 |
| Leuven | 12 | non-AR diffusion (transcript) | 19.8 | 28.1 | 30.2 | 30.9 | 31.7 | 1.60 |
| Cambridge | 14 | B only, no H (transcript) | 38.6 | 37.3 | 35.9 | 33.4 | 32.2 | 0.83 |
| Gatech | 17 | physics ODE, state from warm-up (transcript) | 21.2 | 27.1 | 29.3 | 30.6 | 32.6 | 1.54 |
| **ours, S4D 205** (3-seed mean) | — | AR, one step, 80-step window, K = 32 rollout stage | 37.7 | 50.1 | 55.2 | 57.1 | 61.2 | 1.62 |
| **ours, S4D 1,001** (3-seed mean) | — | same | 34.0 | 46.5 | 54.0 | 56.7 | 72.4 | 2.13* |
| **ours, Dual-LSTM 771** (3-seed mean) | — | same | 46.0 | 60.7 | 63.6 | 65.7 | 67.6 | 1.47 |

Our rows use the seeds3 test predictions saved at
`~/magnet-data/runs/seeds3/seed{1,2,3}/{ssm_s,ssm_m,official_lstm}_hn0.1_ms32/Result/` (45
CSVs; re-scoring them reproduces `results/seeds3/final_eval.csv` exactly, review_log T75).
Script `/tmp/mc2lit/rt/drift_ours.py` (reviewer-teams). We re-ran it and got identical numbers
(`/tmp/mc2lit/others/drift_ours_rerun.txt`). \* S4D-1,001 seed 3 alone reaches 107.0 % at
window 9; the other two seeds end at 59.0 and 51.3 (ratios 1.73, 1.58).

**Early steps** (`/tmp/mc2lit/rt/early.py`, reviewer-teams; re-run identical,
`/tmp/mc2lit/others/early_rerun.txt`). Same normalization (fixed region denominator, w100
block, mean over A, B, D, E); p95 over sequences, median in brackets (%):

| model | step 1 | steps 1–10 | 11–25 | 26–50 | 51–100 | 801–900 |
|---|---|---|---|---|---|---|
| Hefei | 6.8 (1.4) | 5.9 (1.9) | 5.9 (2.1) | 6.9 (2.4) | 7.8 (2.8) | 11.2 (3.2) |
| Fuzhou | 8.9 (2.4) | 6.0 (2.3) | 5.6 (1.8) | 6.5 (2.1) | 7.9 (2.6) | 11.3 (3.3) |
| SiegenPaderborn | 4.6 (0.3) | 8.4 (1.5) | 8.9 (2.4) | 10.5 (3.2) | 12.7 (4.1) | 24.6 (7.2) |
| Tsinghua | 8.0 (1.9) | 8.5 (2.3) | 10.4 (2.7) | 13.0 (3.3) | 16.0 (4.1) | 31.9 (6.4) |
| NITC | 8.2 (0.8) | 5.3 (1.3) | 6.7 (2.0) | 9.4 (2.7) | 13.6 (4.1) | 39.6 (10.3) |
| Tianjin (one-step AR) | 16.7 (2.7) | 16.0 (3.8) | 15.8 (4.1) | 17.2 (5.2) | 18.2 (6.2) | 34.5 (11.4) |
| ours, S4D 205 | 17.8 (3.2) | 31.8 (7.9) | 34.3 (10.8) | 38.3 (12.9) | 41.5 (16.0) | 61.2 (24.6) |
| ours, S4D 1,001 | 15.7 (2.9) | 31.1 (7.1) | 31.6 (9.8) | 34.8 (11.5) | 37.1 (14.1) | 72.4 (20.8) |
| ours, Dual-LSTM 771 | 17.8 (2.5) | 40.5 (7.8) | 40.0 (13.7) | 45.0 (16.8) | 50.9 (20.2) | 67.6 (24.9) |

**Robustness** (`/tmp/mc2lit/rt/drift_check.py`, reviewer-teams; re-run identical,
`/tmp/mc2lit/others/drift_check_rerun.txt`). The measured signal is near-stationary across
windows: median window RMS(H_meas) is 0.80–0.95 of the region RMS for every material and
window (review_log T74). The growth also survives a per-window denominator (window error /
window RMS(H_meas)). p95 ratio window 9 / window 1: Hefei 1.52, Fuzhou 1.36, Eindhoven 1.22,
Southeast1 1.25, Bristol 1.40, Sydney 2.12, SiegenPaderborn 2.37, Tsinghua 2.56, XJTU1 3.40,
NITC 3.91. So the growth is not an artefact of the fixed denominator.

What this shows:

- **Error grows with steps since the boundary for every top-10 team** (1.15–3.70× from window
  1 to window 9 at p95). The flat block averages in §1.1 hide this, because they compare
  different sequences and are confounded by the w900 denominator effect. Only Cambridge,
  which never uses H, starts high and falls (38.6 → 32.2). That fits a model with an initial
  offset and no H calibration [inferred].
- **Growth is smallest for the non-AR teams that also use the known H and train over the full
  horizon**: Hefei (code), Bristol (code + paper) and Southeast1 (transcript) grow ≤ 1.65×.
  Fuzhou and Eindhoven (1.64, 1.15) look the same, but their designs are unverified. Not
  feeding H back is not enough by itself. Tsinghua never reads H yet grows 2.31×, and Sydney
  is non-AR but its decoder is trained only on 100-step targets and evaluated on up to 900; it
  grows 1.89× [inferred: train/test horizon mismatch]. Cambridge (B only) falls, 0.83×. The
  closed-loop and autoregressive teams grow 2.0–3.7×: SiegenPaderborn 2.34, Tianjin 2.01, NITC
  3.70; XJTU1 (3.23) has an unknown design. **This is the best evidence in these notes for
  "train as you infer over the full horizon", but it is correlational** (different
  architectures, data handling and model sizes).
- Low window-1 error does not imply a good long horizon. NITC starts at 10.7 (similar to
  SiegenPaderborn's 10.5) but ends at 39.6, the highest in the top 10.
- **Where we stand (deciding test run, review_log T75–T76).** Neither branch of the decision
  rule from revision 2 fits. Our window-1 p95 is 34–46 %: below our 50 % official score, but
  2–7× the top teams' 7–15 %. Our ratio (1.47–2.13×) is low only because window 1 is already
  high. In absolute terms we add 22–38 points over the forecast (S4D-205 +23.5, S4D-1,001
  +38.4, LSTM +21.6). That is at or above the closed-loop/AR teams (SiegenPaderborn +14.1,
  Tianjin +17.3, NITC +28.9) and far above the non-AR ones (Hefei +4.4, Bristol +5.4,
  Southeast1 +4.2). The early-step table separates the effects:
  - *One-step accuracy is not the gap.* At the first predicted step our p95 is 15.7–17.8 %
    (median 2.5–3.2), the same as Tianjin, the closest one-step AR analog (16.7, median 2.7).
    Hefei is 6.8 (median 1.4).
  - *The gap opens within the first ~10 fed-back steps.* Over steps 1–10 our p95 doubles to
    31–41 % (median 7–8), while Tianjin stays at 16.0 (median 3.8) and Hefei and NITC are at
    5.9 and 5.3. Over steps 51–100 we are at 37–51 % against Tianjin's 18.2.
  - *Later growth contributes about as much.* Comparing window 9 with steps 1–10: S4D-205
    goes from 31.8 to 61.2, against Hefei's 5.9 → 11.2. The share of our window-9 gap to Hefei
    already present by step 10 is 52 % (S4D-205), 41 % (S4D-1,001) and 61 % (LSTM). Our growth
    after step 10 (+29 for S4D-205) also exceeds Tianjin's (+18).
  **Reading [inferred]:** our error is not a one-step accuracy problem. About half of our gap to
  Hefei is present by step 10, and the rest accrues over the remaining steps. The early
  doubling (exposure bias within the first ~10 rollout steps) is what distinguishes us from
  Tianjin at equal one-step error. Neither H-noise (which gave the 2.4–10× gains in
  RESEARCH_LOG §5.3) nor the K = 32 rollout stage (§5.4; e.g. S4D-205 at σ 0.05: 61.0 → 55.5 %;
  kept for 34 of 45 model–material–seed runs in seeds3, `kept_stage` in `final_eval.csv`)
  stopped this compounding. Tianjin, with the same one-step accuracy and a similar window
  length (its 80-step window is our reading of the captions), does not compound this way. That
  suggests the cause is in how errors re-enter our model, not in the AR scheme as such. It is
  a single comparison, and Tianjin also differs in architecture, loss and augmentation.

## 2. Hefei (HFUT-CityUHK-XMU): LSTM-DAO. Performance 2nd, Software Engineering award, official 9.17 % (rank 1)

**Citation.** Y. Wang, Y. Liu, L. Mo, W. Wang, S. Ren, G. Cai, S. Liu, H. Wu, M. Ma, C. Q.
Jiang, "Framework of LSTM-Based Direction-Dependent Asymmetric Operator Model for B-H
Prediction Under Transient Excitations," *IEEE Trans. Power Electron.*, vol. 41, no. 12,
pp. 21230–21243, 2026, doi:10.1109/TPEL.2026.3715566. Preprint: TechRxiv
doi:10.36227/techrxiv.177162964.49399026/v1 (posted 2026-02-20, CC BY 4.0 per Crossref; the
unversioned DOI returns 404). The full text returned 403 to us, so only the abstract was
read. Abstract:
https://scholars.cityu.edu.hk/en/publications/framework-of-lstm-based-direction-dependent-asymmetric-operator-m/.
Code: https://github.com/ouihon/Mag-LSTM-DAO-Inference (competition inference: model,
weights, normalization) and https://github.com/ouihon/LSTM-DAO-Inference (design app with
`train_finetune.py`). Talk: https://www.youtube.com/watch?v=-erftMlwTno.

**Architecture** (from `model.py`, `inference.py`, `utils.py` in Mag-LSTM-DAO-Inference):

- Inputs per time step, over the **whole 1,000-step sequence at once**: normalized B, ΔB, Δ²B,
  H with unknown positions set to 0, and an H-known mask. That is 5 channels.
- One unidirectional LSTM layer (hidden 26–40). At every step an MLP head reads the LSTM output
  concatenated with the normalized temperature: Linear(h+1 → 16 or 20), ReLU, Linear → 6K.
- The head emits per-step parameters for **K = 3–4 "direction-dependent asymmetric operators"
  (DAO)**: radius r, bias b and weight w, each with separate values for rising (ΔB ≥ 0) and
  falling B. Constraints: r_down = softplus + 1e-4, r_up = r_down + softplus (so r_up ≥ r_down),
  and w = softplus normalized to sum 1 over K for each direction.
- Each operator computes `Y = min(max(B − b, cummax(B − b − r)), cummin(B − b + r))` along time.
  The output is **H = Σ_k w_k Y_k**, i.e. H is a convex combination of clipped, shifted copies of
  B in normalized units. That is a learned, time-varying Prandtl–Ishlinskii-style output map.
- The model returns H only for t ≥ split point. Known H is copied into the output.
- **Non-autoregressive:** predicted H is never fed back. The LSTM sees known H (masked) and the
  full future B in one causal pass, and the operator turns its outputs into H.
- [inferred, checked by us numerically] The cummax/cummin form is not the classical play
  operator's recursion: it is the envelope of all past values, not a reversal-tracking state.
  Within one sequence, with fixed r and b, it saturates to a constant once B has swept one way.
  Because r, b and w are re-emitted by the LSTM at every step, the memory effectively lives in
  the LSTM, and the operator acts as a structured, bounded output nonlinearity with a
  direction-dependent (hysteretic) switch. The paper's physics framing should be read with
  this in mind.

**Parameter counts** (from MATERIAL_CONFIGS, verified by our count of the formula; they match the
leaderboard exactly):

| material | K | LSTM hidden | MLP hidden | params |
|---|---|---|---|---|
| A, B | 4 | 40 | 16 | 8,600 |
| C | 3 | 26 | 16 | 4,186 |
| D | 3 | 30 | 16 | 5,258 |
| E | 4 | 32 | 20 | 6,176 |

The LSTM is 81–87 % of the parameters: A/B 87.4 % (7,520 + 1,080 head), C 82.0 %, D 84.4 %,
E 80.8 %.

**Hyperparameter selection.** The captions say "parallel basis optimization" over number of
operators, LSTM size and MLP size [transcript, 02:27]; we read it as Bayesian optimization
[inferred]. The paper's abstract says "Pareto-front optimization is
employed to select suitable configurations". A team member said that below about 1,000
parameters "the accuracy start[s] to decrease dramatically". The Pareto screening used "1 %"
of the data (the speaker's figure; the interviewer said "1 to 10 %"), and full-data training
takes "weeks or even months" [transcript, 10:20–11:31].

**Training objective.** The talk says a loss term on energy is added: "we add the energy loss
as one of the loss[es] to train the model" [transcript, 06:40]. **Caveat:** the configs
embedded in the A and C competition checkpoints record `Loss_select = 'rmse'`; B/D/E carry no
config. The energy-loss claim is therefore transcript-only and not reflected in the
checkpoints we can read (review_log T26). The released `train_finetune.py` (design-app
fine-tuning, not necessarily the competition script) offers MSE, RMSE or a squared relative
energy-error loss. It always trains with split point 1 (only H(0) known), i.e. the Material C
setting. Fine-tuning can freeze the LSTM and optionally the first head layer, i.e. transfer
learning from a base model. The training is sequence-level, with the whole forecast in one
pass, so it is free of teacher forcing [inferred from the architecture].

**Warm-up / Material C.** The H-mask channel makes one network handle any warm-up; known H
enters as input values. For C, split point = 1 (only H(1) = 0 known) (`utils.py`). Checkpoint
metadata (read from the pickle with `pickletools`): A = `A_exp7`, epoch 146; C = `C_exp2`,
epoch 139; B, D, E = `All_in_one_B/D/E`, epochs 200/150/200. A's embedded config has
`KNOWN_POINTS = [100, 500, 900]`, so **A was trained on all three warm-ups jointly**. C's
has `KNOWN_POINTS = [1]`, matching its test condition; both use loss `rmse` (review_log T7).
The `All_in_one` names suggest the same joint training for B, D and E [inferred from names;
no config embedded]. A and C were separate experiment runs, not separate per-warm-up models.

**Data strategy for A / transfer.** Not described in sources we could read. The app supports
base-model fine-tuning with frozen layers. Whether the competition A model was fine-tuned from
a known material is not established.

**Reported numbers.**
- Official (leaderboard): A 12.62, B 5.55, C 13.07, D 5.95, E 8.67, mean **9.17 %**; energy
  1.57 %. 1st on A and B, 2nd on C, D and E. Seq rank 1, energy rank 1, size rank 10.
- Paper abstract, 10 known materials (its own protocol): "average sequence root-mean-square
  error between 1.82 % and 5.30 %" and "weighted relative mean error for energy loss between
  2.11 % and 3.16 %". These are means, not the official p95.
- Pooled over warm-ups [measured]: per-material mean 5.84 / 2.41 / 5.76 / 2.55 / 3.66 %.
- Per warm-up [measured]: w100 9.86, w500 6.85, w900 7.88, C 13.07 (§1.1). Within-sequence
  drift: p95 6.8 % in window 1 → 11.2 % in window 9 (1.65×), the lowest absolute level of any
  team (§1.2).
- One remark: "the inference interval must be consistent with that used during training"
  [transcript, 02:45], i.e. trained and evaluated forecast lengths should match.

**Applicability to us.**
- *Strongest single lesson* [inferred]: a non-autoregressive, whole-sequence formulation with a
  known-H mask removes exposure bias by construction. Its within-sequence error still grows
  (1.65× over 900 steps), but from the lowest base and less than any closed-loop team (§1.2).
  Our S4D can do the same: a causal SSM over the full 1,000-step
  sequence with channels (B, ΔB, Δ²B, H·mask, mask), trained with a sequence loss over the
  predicted region. This is natural for S4D (convolution mode over 1,000 steps).
- *Structured output head*: replace our Tanh MLP head with a small bounded/direction-dependent
  output map (e.g. K = 3 DAO-like terms). This is cheap in parameters (head ≈ 1 k params here).
  The effect of the DAO head versus a plain head is not shown in sources we could read, so
  test it with an ablation.
- *ΔB and Δ²B as explicit inputs* (also used by Sydney, Cambridge, Gatech, SAL, Tsinghua).
- RQ1: Hefei sits at ~4–9 k params; their own sweep says accuracy falls fast below ~1 k. That
  is consistent with our 205 ≈ 1,001-param tie (both are below the knee) [inferred].

## 3. Fuzhou University: dual-attention LSTM seq2seq. Honorable mention, official 12.42 % (rank 2)

**Citation.** No public paper or code found (searched English and Chinese; Fuzhou's university
news covers their MC1 2nd place, not MC2). Sources: talk
https://www.youtube.com/watch?v=pHzgWAF_6Co [transcript] and pretest PDFs
`Pretest/Results/Fuzhou.pdf`, `FuzhouXGboost.pdf` on the MC2 GitHub.

**Method** [transcript]:
- Tried 10+ classical ML regressors (XGBoost, LightGBM, SVR, RF, decision trees, stacking,
  bagging). XGBoost and LightGBM were best, but they generalized weakly at extreme values.
- Compared five learning strategies: sequence-to-sequence, point-to-point, transfer learning,
  domain adaptation and "PI" (probably physics-informed). **Seq2seq was best**, i.e. learning
  the whole input→output sequence mapping end to end.
- Compared backbones (LSTM, PatchTST-like, SegNet, Transformer, Informer, TimeMixer). Attention
  helped on the long input sequence. The final model is a **"dual-attention sequence-to-sequence"
  network combining an LSTM with a dual attention module**.
- Three areas of optimization: input features (material-specific variation features),
  architecture, and loss. The loss uses **smooth-L1 instead of MSE plus "simple physical
  constraints"** to fit peaks and troughs better, reportedly reducing error by 1–2 %. A
  temperature-derived feature, described as "temperature divided by H", is fed as an input
  channel; the transcript is unclear here.
- Five models per material with different hyperparameters (ensemble or selection is unclear).
  Errors are largest near H ≈ 0 (small values).
- Future work: few-/zero-shot strategies and test-time adaptation.

**Params:** 57,409–69,441 per material (avg 63,022; leaderboard). C used 57,409, but the
speaker says C's parameter count was "relatively higher than others", which conflicts with
the leaderboard.

**Reported numbers.** Official A 16.38, B 8.74, C 12.29 (1st on C), D 13.10, E 11.57, mean
**12.42 %**; energy 1.91 %. Seq rank 2, size rank 16. [measured]: w100 9.36, w500 11.32,
w900 16.67, so error *rises* with warm-up length (shorter forecast). Pooled means: A 7.04,
B 3.76, C 5.52, D 4.50, E 4.56 %.
Pretest (self-reported, 10 known materials, not comparable): the XGBoost version reports
better sequence error than their NN version, e.g. 3C90 p95 6.88/7.46/10.05 % vs
12.37/14.57/25.26 % for 10/50/90 % known (`/tmp/mc2lit/FuzhouXGboost.txt`, `Fuzhou.txt`).

**Applicability.**
- [inferred] Error rising with warm-up length suggests their seq2seq struggled with *short*
  forecasts (100 steps), possibly from fixed-length output alignment or the normalization
  window. It confirms that one model across warm-ups is not automatic; train on all three
  splits (Bristol and Hefei do).
- Smooth-L1 plus peak emphasis targets exactly what a p95 metric punishes (peaks). This is
  cheap for us to try.
- Best on Material C with a direct seq2seq. If the decoder does not re-read its own outputs
  (not stated), this is another case of a non-autoregressive mapping handling the no-history
  case well [inferred].

## 4. University of Sydney: PI-HNO. Performance 3rd, official 24.63 % (rank 9)

**Citation.** Y. Zhu, Q. Huang, S. Li, Y. Li, G. Lei, J. Zhu, "A Physics-Informed Hybrid Neural
Operator for Transient Magnetization Prediction in Power Magnetics," arXiv:2608.02965 (v3,
2026-08-18; UTS and University of Sydney), https://arxiv.org/abs/2608.02965. This is the
arXiv-listed title. The v3 PDF/HTML body title adds "High Frequency" ("…in High Frequency
Power Magnetics"). Identified as the
MC2 model because its "4777 trainable parameters per model" equals Sydney's leaderboard count,
and the talk (https://www.youtube.com/watch?v=8Y3p8dFIU-Q; presenter transcribed as "Choj
Huang", likely co-first author Qiujie Huang [inferred]) describes the same architecture. No MC2 code found. github.com/moetomg/magnet-engine and
mag-net-hub `sydney.py` are Sydney's **MC1 core-loss** model, not this one.

**Architecture** (paper §III, §IV-C):
- *Local branch:* a history GRU (hidden 8, 1 layer) reads the most recent **100** samples of
  (B, H) before the boundary. Its final state is added to MLP(1–8–8) projections of temperature
  and of the normalized start position:
  h_init = h_hist + h_temp + h_pos. A future GRU (hidden 8) is initialized with h_init and
  reads B(t), ΔB(t) and a learned future-position embedding at every predicted step.
- *Global branch:* a 1-layer Transformer (width 8, 2 heads, FFN 64) over the **complete
  1,000-step B sequence** plus the broadcast start position, mean-pooled. They motivate
  attention as a data-driven analog of the Preisach weighted superposition (attention map ≈
  Preisach plane).
- *Fusion:* u(t) = h_local(t) + v_global; H = MLP(8–128–1, dropout 0.1). The additive fusion
  is motivated by H = H_hys + H_dyn (loss separation). The GRU update is motivated as a
  discretized first-order relaxation ODE (eddy-current dynamics).
- **Non-autoregressive:** H is never fed back. The future GRU is driven by B only.
- Warm-up handling [transcript, 03:49–04:17]: "standardize everything into a simple 100-point
  window". A longer history is truncated to the most recent 100. A shorter one is padded with
  the initial state (this covers C's single known value).

**Training:** each 1,000-step sequence is cut into five non-overlapping 200-step segments: 100
history + 100 target, with the full B sequence given to the global branch. Loss: smooth-L1 on
normalized H plus 0.1 × L_phys, where L_phys = |W_pred − W_meas|/max(|W_meas|, ε) +
0.01·max(0, −W_pred) and W = trapezoidal ∫H dB over the forecast. AdamW, one-cycle LR (peak
1e-2), weight decay 5e-4, grad clip 0.5, EMA 0.999, batch 64, up to 1,000 epochs, early stop
patience 80. Low-B_pp sequences are added for materials with weak validation at low B_pp.
- [inferred] It trains on 100-step targets but is evaluated on up to 900-step forecasts,
  relying on the future GRU to extrapolate. The per-block averages look similar (w100 25.8 vs
  w900 22.4), but within the same sequences its p95 grows from 17.1 % (window 1) to 32.4 %
  (window 9), 1.89× (§1.2). That is consistent with the train/test horizon mismatch.

**Ablations and baselines** (paper Tables III–IV; their own 14-material MagNetX protocol,
mean/p95 seq error over pooled ratios 0.1/0.5/0.9, **not** the official A–E metric). The last
two rows are Table III baselines:

| variant | params | seq mean | seq p95 | energy p95 |
|---|---|---|---|---|
| full PI-HNO | 4,777 | 13.40 | 31.39 | 7.60 |
| A1 − global (Transformer) branch | 2,241 | 17.77 | 42.19 | 9.61 |
| A2 − local (GRU) branch | 3,817 | 20.42 | 49.10 | 12.50 |
| A3 − ΔB input | 4,777 | 17.53 | 40.09 | 12.66 |
| A4 − future GRU | 4,665 | 19.01 | 43.65 | 12.18 |
| A5 − temp/pos fed per step instead of initial state | 4,649 | 24.83 | 75.68 | 12.49 |
| plain GRU (matched 4,758 params) | 4,758 | 18.44 | 46.17 | 16.62 |
| MagLearn2 (860,929) | 860,929 | 12.88 | 28.47 | 16.46 |

The energy term is not ablated separately. All variants are single runs (no seeds or variance
reported; the paper's "fixed random seed" refers to data sampling) on Sydney's own protocol.
A3's main effect is on energy (p95 7.60 → 12.66); its sequence p95 rises 31.39 → 40.09.

**Reported numbers.** Official A 37.98, B 17.12, C 27.22, D 21.24, E 19.56, mean **24.63 %**;
energy 5.44 %. Seq rank 9, size rank 8. [measured] w100 25.82, w500 23.72, w900 22.38;
within-sequence 17.1 → 32.4 % (§1.2).

**Applicability.**
- **A5 is a cheap idea worth testing:** A5 (T and start position fed per step to the future
  GRU instead of initializing h) raised p95 from 31.4 to 75.7 %. It is a single run on their own
  protocol, and it moves two inputs at once (temperature and start position). The paper itself
  concludes only that "placement is important". Our S4D currently feeds T only to the head.
  Conditioning the SSM's initial state (or the B/H lifting) on T and on the warm-up length is
  cheap to try [inferred].
- Two-stream "history encoder → state; future B drives the dynamics" design. The ΔB input
  matters (A3, mainly for energy). They use a short, fixed history window (100 samples).
- Matched-size plain GRU 46.2 % p95 vs PI-HNO 31.4 % (their protocol): structure beat raw
  capacity at ~5 k params. That is relevant RQ1 evidence.
- Physics loss: an energy-consistency term in the loss helps the energy metric. That matters
  for us because our energy error trails our sequence ranking (RESEARCH_LOG §5.5 caveat).
- Talk Q&A [transcript, 07:36–08:44]: a participant transcribed as "Wilm/William", most likely
  Wilhelm Kirchgässner of SiegenPaderborn [inferred], argued that a GRU
  suits this problem over an LSTM because its single state is easy to initialize from the last
  ground-truth H, and it has fewer parameters. Organizer comment: top teams split "one path for
  long-term behavior and one for instantaneous response".

## 5. Georgia Tech (GT & GM): mechanical-analogy physics model. Innovation 1st, 83 params, official 38.76 %

**Citation.** P. Chauhan, Y. Mo, D. Li, L. Chang, X. Xian, B. Ge, "Dynamic Magnetic Modeling
With Magnetization, Hysteresis, Eddy, and Displacement Currents," *IEEE Trans. Power
Electron.*, vol. 41, no. 12, pp. 20869–20873, 2026 (letter), doi:10.1109/TPEL.2026.3712771
(Crossref metadata; open access, hybrid, per OpenAlex; the licence is not confirmed). Only the
abstract was accessible to us (verbatim via OpenAlex and Semantic Scholar); IEEE Xplore
blocked automated access. Talk:
https://www.youtube.com/watch?v=0-705NtutoE. News:
https://ece.gatech.edu/news/2026/04/georgia-tech-team-takes-home-top-prize-magnet-challenge
(two provisional patents filed). No code.

**Method** (abstract + [transcript]):
- Map B → displacement, H → force, dB/dt → velocity, d²B/dt² → acceleration. Decompose H into
  four vector-valued component forces:
  1. **Spring** = ideal magnetization (saturation curve). Options include tanh and the Langevin
     function plus a linear term for the approach to saturation. An "effective α" reduces the
     slope at high B, and an "effective B" with a **wipe-out property like Preisach** is used.
  2. **Slider** = hysteresis via domain-wall pinning. The novelty: **static and dynamic
     friction are decoupled**, where previous models force them equal.
  3. **Viscous damper** = eddy current ("damper is to spring as magnetic inductance is to
     reluctance").
  4. **Inertia/mass** ∝ d²B/dt². The abstract calls it "one more type of component that seems
     not being reported previously", attributed to displacement current: parasitic winding
     capacitance or the dielectric property of the ferrite. It "further improves the modeling
     accuracy at turning points" [transcript, 04:07–04:21; not in the abstract].
- Initial state: the model needs each spring's initial stretch. For A, B, D and E it is found
  from the known segment. For C they set it to zero (demagnetized), which "was a mistake".
  The organizers agreed C is "no history", not "zero state". A rerun with the same strategy as
  the other materials improved C, but that rerun is not in the leaderboard. In training they
  sidestep initial states by prepending a quarter of a periodic wave [transcript].
- Parameter identification by fitting (they report about 10 s for single-sequence training and
  under a day for multi-sequence B→H training). Inference takes about 20 ms per 1,000 points.
  It is vectorial, so the model can extend to FEA, and it runs both B→H and H→B. Planned next
  step: a small NN in place of the 2-degree-of-freedom saturation function.

**Params:** 87 / 81 / 78 / 90 / 81 (mean 83; size rank 1).

**Reported numbers.** Official A 34.19, B 20.31, C 84.68, D 34.88, E 19.74, mean **38.76 %**;
energy 7.97 %. Without C, the A/B/D/E mean is 27.28 % [measured from the leaderboard CSV].
[measured] w100 26.44, w500 24.41, w900 30.98, C 84.68; within-sequence 21.2 → 32.6 % (§1.2).

**Applicability.**
- Physics prior for a tiny model: an explicit additive decomposition H = f_spring(B_eff) +
  f_slider(history) + c₁·dB/dt + c₂·d²B/dt². The last two terms are the same "rate-dependent"
  correction that Cambridge (c₁Ḃ + c₂B̈) and Sydney (ΔB input, GRU relaxation) use. We can add
  a linear skip from (B, ΔB, Δ²B) to the output (≈ 3 params), as Tsinghua does for B.
- **Initial-state lesson for C:** do not assume a demagnetized start. Material C test segments
  start near H = 0, but the material is not necessarily demagnetized ("no past H", not "zero
  state"). RESEARCH_LOG §2 already records the C distribution mismatch. Our zero-history
  handling of C should be checked against this [inferred].
- Shows that ~83 physics parameters reach ~27 % on A/B/D/E, better than our 205–1,001-param
  S4D (≈ 50 %). That sets the RQ1 bar for "small" [inferred].
- For background-lit (physics priors): Gatech's spring/slider/damper/inertia mapping, the
  Preisach-like wipe-out of "effective B", and decoupled static/dynamic pinning friction. Also
  Sydney's attention-as-Preisach-plane and GRU-as-relaxation-ODE arguments (§4), and Hefei's
  direction-dependent operator head (§2).

## 6. University of Cambridge: physics-inspired compact DeepONet. Innovation 2nd, 1,182 params, official 37.01 %

**Citation.** No paper or code found. Source: talk https://www.youtube.com/watch?v=2AkU2jc_aUU
[transcript].

**Method** [transcript]:
- **H = H_static + H_dynamic**, two parallel modules.
- *Static hysteresis:* a **compact DeepONet** (branch and trunk nets, inner product). The
  branch input is an **11-dimensional hand-built memory state** instead of the full B history:
  the four reversal points of B nearest the current B, a "loop closure" (relative position of B
  in the current minor loop), historical max and min B, temperature, the initial field, and
  "reversal density" (count of reversals in the last 100 B samples, a pseudo-frequency). The
  trunk input is the current B and a reference field (median H at that B and T from training
  data). The model is described as sampling-frequency free.
- *Dynamic part:* from an equivalent RLC view of the two-winding measurement at high dV/dt,
  H_dyn = C₁·Ḃ + C₂·B̈, with C₁ and C₂ produced by an MLP. A gate driven by reversal density
  turns the term off at low frequency.
- **H is never used**, in training or inference. B → H directly, which makes the model stable
  over long horizons. They acknowledge short-forecast instability because the known H is
  unused, and plan an H-based DC-offset calibration as future work.
- 1,182 params; 60.9 ms per sequence. Self-reported average NRMSE ~11.04 % (metric not
  specified).

**Reported numbers.** Official A 45.12, B 16.26, C 70.53, D 27.22, E 25.94, mean **37.01 %**;
energy 9.24 %. Seq rank 14, size rank 4. [measured] w100 35.00, w500 24.36, w900 26.54, C 70.53.
Its w100 block is its worst on the material average (+10.6 / +8.5 over w500 / w900). XJTU1 and
NITC (+12.4) have larger w100 penalties. Per material, the penalty comes from D (46.9 vs
19.6 / 15.2) and E (34.9 vs 17.6 / 25.3); on A the w100 block is its *best* (40.2 vs 45.5 /
49.7) (review_log T44). Within the same w100 sequences its error *falls* with horizon (38.6 %
in window 1 → 32.2 % in window 9; §1.2), the only team where it does. That is consistent with
an initial offset that the model cannot correct, because it never uses the known H [inferred].

**Applicability.**
- Hand-crafted Preisach-style memory features (nearest reversal points, loop-closure
  fraction, running extrema) are cheap inputs that encode hysteresis memory explicitly. They
  could be added as extra channels to our S4D input [inferred].
- Cautionary result: ignoring the known H costs a lot (C 70.5 %; mean far behind teams that use
  H). Use H for initial-state calibration.

## 7. University of Bristol: MagLearn 2 (LSTM seq2seq). Honorable mention, official 25.22 % (rank 7)

**Citation.** J. Yang, B. Cui, Y. Huo, L. Zhang, A. Martinez, S. Liu, J. Wang, "MagLearn 2:
High-Fidelity, Saturation-Aware, and History-Efficient Sequence-to-sequence Modeling of
Transient B–H Behaviour Under PWM Excitation," arXiv:2610.03504 (Oct 2026; submitted to TPEL),
https://arxiv.org/abs/2610.03504. Preliminary version: B. Cui et al., "MagLearn 2: A
Short-Window Seq2Seq Framework…," IPEC-Nagasaki 2026 (ECCE Asia). Code (deployment for
Materials D and E only): https://github.com/JunWang-Bristol/MagLearn2. Talk:
https://www.youtube.com/watch?v=W7GEu5IXt68. Predecessor: MagLearn (MC1),
https://github.com/JunWang-Bristol/MagLearn.

**Architecture** (paper §III, `Deployment/predict.py`):
- Encoder LSTM (3 layers, hidden 128, dropout 0.2) over the history [B_t, H_t]. Its final
  (h, c) initializes a decoder LSTM (3 × 128).
- A condition vector [temperature, effective PWM frequency] passes through an MLP (2 → 256 →
  128) and a second MLP (128 → 128 → 128).
- Decoder input per step: [h_enc (repeated), condition embedding, B_future(t)] → Linear → H.
  It is **non-autoregressive**: the decoder never sees predicted H.
- 860,929 params (Table V, H128-L3; the same figure as the leaderboard). Checkpoint ~9.9 MB.
- Effective PWM frequency f_eff = (# prominent maxima + minima of B)/length, computed from the
  history, or from the full B if the history is too short (Material C).
- **Saturation-aware auxiliary MLP** (0.91 % of the parameters): replaces predictions where
  |H_pred| ≥ H_sat. The paper only defines H_sat; the talk says the threshold was set by hand
  per material "based on experience" [transcript, 10:22]. In the talk the team said it "did
  not make a big difference" to the final result, because the round-2 data has little
  saturation [transcript, 10:59–11:16].

**Training:** a fixed 1,000-sample window slides along the long raw sequences (data
augmentation). Each window is used in all three history:forecast splits (1:9, 5:5, 9:1) with
equal mixing, so **one model serves all warm-ups**. Loss: **MSE with the top 5 % of pointwise
squared errors trimmed** (quantile trimming). The paper's stated mechanism is to reduce "the
influence of rare high-error points (often associated with saturation events or
under-represented patterns)". The talk gives the motive as improving the 95th-percentile
error [transcript, 04:58]. Its effect is not ablated. Adam,
lr 5e-4, batch 128. A 60:20:20 split is made by waveform before windowing (no leakage). The
best checkpoint is chosen by equally weighted validation over the three splits.

**Transfer / small data (Material A):** "performance-driven base-model selection". Each of the
10 known-material models gets a ~100-epoch adaptation run on the target. The best on target
validation is fine-tuned end to end at lr 1e-4 to 3e-4. N87 was chosen for B, C and D, and N27
for A and E; material similarity did not predict the choice. For A (3C92) they used stronger
augmentation: "available historical segments with 500 and 900 samples" were *interpolated to
1,000 samples* as extra trajectories, with a sliding stride of 100. The paper does not say
where these segments come from; the test-set warm-ups are one possibility [inferred]. For C the
history is fixed to 1 sample, using a "ratio-matched" base model.

**Interpretability: history length.** Occlusion of the oldest H-history samples shows error
grows only when the most recent history is removed. The median error-growth threshold is
~560 ns, about **9 samples at 16 MHz** [inferred: 560 ns × 16 MHz]. The range is 187.5 ns (3
samples) to 2.31 µs (37 samples). "More than ~90 % of the preceding history [is] removable in
most cases" (Table VI, §V-B). Note that it masks H-history only, with the full B kept. **These
thresholds mark where error growth turns rapid, not where error stays small.** Table VI's
other columns give the H-history needed to keep the mean error under 20 %: 2.19–43.06 µs,
i.e. ~35–690 samples (T37: ~357 steps) (review_log T32).

**Model size sweep (N87, own protocol):** H64-L1 84 k … H192-L3 1.93 M. Gains flatten after
H128-L3 (mean 3.07 / p95 6.60 %). H160-L3 has the lowest errors (p95 6.28 %), but H128-L3 was
chosen for its smaller size; H192-L3 is slightly worse (Fig. 7, Table V).

**Conditioning ablation (Table IV, own protocol):** T + f_eff gives mean 3.62 / p95 8.36 %;
T only 5.40 / 13.44; f_eff only 3.74 / 9.26; none 4.47 / 11.16. Temperature alone was worse
than no conditioning in their test.

**What the "2.2 % average RMSE" is.** The abstract claims the "lowest reported average RMSE to
date on the sequence-prediction task, at 2.2 %". This is the **mean** relative RMSE of the
**single best material, FEC014 (Material D)**: 2.24 % mean / 5.66 % p95 (Table VII). It is not
an average over materials and not the official metric. We checked it [measured]: pooling all D
test sequences across warm-ups, Bristol's submitted CSV gives exactly mean 2.24 / p95 5.66 %. A,
D and E reproduce Table VII exactly (A 8.16/18.22, E 3.95/8.61). **B and C in Table VII (3.30/8.36
and 5.79/13.29) do not match the submission** (we get 9.68/38.71 and 27.83/56.72). The paper
presents all five materials as independent-evaluation results and never says B or C were
retrained. One explanation is that B and C in Table VII come from a later model; another is
that they were scored on the team's own split test set [inferred; the paper does not say].
The team said B and C "broke" in the final evaluation through overfitting [transcript,
07:27]. Table VII also mixes aggregations: its "Baseline [19]" column (95.35 / 43.40 / 96.95 /
40.70 / 57.84) equals Princeton's *official* per-block scores, while the team's own columns
are pooled p95 (review_log T33). The paper's "15-material average
95th-percentile 9.44 %" is a mix of own-split (10 known materials) and A–E numbers. On the
official metric Bristol's MC2 entry is 25.22 % (rank 7): 1st on D (5.78) and E (8.46), 3rd on A
(18.23), 21st on B (36.77), 17th on C (56.88). Sydney's PI-HNO paper also re-trains MagLearn2 on
14 materials under its own protocol (mean 12.88 / p95 28.47 %), which is yet another number.

**Applicability.**
- Most relevant: (i) **train one model on all three warm-up splits** with validation averaged
  over splits. (ii) **Quantile-trimmed MSE** is intended to improve p95 (untested): drop the top 5 %
  pointwise errors from the loss. That is cheap, but the trimming could also discard
  saturation peaks; they handle those separately. (iii) **Base-model selection by target
  validation, then full fine-tune**, which addresses RQ2 and Material A. (iv) For A, **resample
  shorter histories to 1,000 samples** as augmentation.
- The occlusion result does **not** settle whether our 80-step window is long enough. The
  most recent ~3–37 samples carry the sharpest information, but keeping mean error under 20 %
  needed ~35–690 samples of H-history in their model (Table VI). 80 steps sits inside that
  range. A window-length sweep on our model would settle it [inferred].
- f_eff from B is a cheap scalar conditioning feature that helped more than temperature in
  their ablation.
- Caveat for our paper: cite MagLearn 2's numbers with their aggregation. The 2.2 % is a
  best-material mean.

## 8. Southeast University 1 (Cheng): vector-magnetic-circuit analytical model + BiLSTM residual. Honorable mention, official 14.51 % (rank 3)

**Citation.** No paper found. Talk: https://www.youtube.com/watch?v=6REL90As66U [transcript].
The team builds on the group's vector magnetic circuit theory (M. Cheng's group; MC1 entry
used it in the frequency domain).

**Method** [transcript]:
- Analytical lumped model, the magnetic-circuit dual of an RLC circuit: reluctance R
  (nonlinear in B) plus two new components ("magductance" etc.) carrying hysteresis and eddy
  current. The B–H relation of a periodic sequence uses only 6 parameters. Parameters are
  obtained by curve fitting (C held constant; R and C fitted, since (H, B, ∫B) lies close to a
  plane) and retrieved by **table lookup** keyed on flux amplitude. The fit does not depend on
  frequency.
- A **BiLSTM residual corrector**: it predicts H_true − H_analytical from physically motivated
  features. It is bidirectional because the analytical H for the whole sequence is known in
  advance. Errors concentrate after large, rapid H changes, and local peaks are important
  history.
- Inference: generate the last 900 values (999 for C) analytically, correct them with the
  BiLSTM, then apply an **offset calibration to the known H** for the 100/500/900 warm-up
  cases.

**Params:** 22,007–22,417 (avg 22,197). **Reported:** official A 22.49, B 10.18, C 15.29, D 12.63,
E 11.96, mean **14.51 %**; energy 2.69 %. Seq rank 3 (3rd or 4th on every material).
[measured] w100 13.87, w500 12.18, w900 16.90, C 15.29.

**Applicability.** A physics baseline plus a learned residual is a strong combination for
generalization across materials. It finished 3rd with ~22 k params, and its per-material
*ranks* are the most uniform of the top teams (3rd–4th on every material). By the spread of
the scores themselves it is not: its coefficient of variation over A–E is 0.33, 6th of the top
10 (Fuzhou 0.22, SiegenPaderborn 0.22, NITC 0.20; review_log T61). Its within-sequence drift is
among the smallest (1.34×, §1.2). Offset calibration of the prediction to the last
known H is an inference-time trick that we could apply to any model [inferred].

## 9. Eindhoven (TU/e): "MAD", MLP-only frequency-decomposed encoder. Honorable mention, official 17.71 % (rank 4)

**Citation.** No paper found (the TU/e group also authored "Magnetic Hysteresis Modeling with
Neural Operators", arXiv:2407.03261, a different work). Talk:
https://www.youtube.com/watch?v=6pHZcLsHtgQ [transcript].

**Method** [transcript]: an encoder–decoder made **only of dense (MLP) residual blocks with a
bottleneck**. They cite work questioning whether time series need attention or SSMs at all. The
signal is **decomposed into a low-frequency "trend" and a high-frequency "hysteresis" part**,
each with its own residual branch. A **variational latent with a distribution loss**
(VAE-style) is meant to smooth the latent space for generalization and "zero-shot" use on
unseen materials (not fully validated, per the team). The history input is 100 samples.
~100 k params (≈ 400 KB); 37 ms per 1,000 steps; linear complexity. They argue for FLOPs
instead of parameter count as the efficiency metric.

**Params** 81,893. **Reported:** official A 31.31, B 11.24, C 17.45, D 13.91, E 14.64, mean
**17.71 %**; energy 2.38 % (energy rank 3 on the published table). Seq rank 4. The talk says
"ranks the first for the sequence error", which does not match the published ranking.
[measured] w100 16.00, w500 16.65, w900 20.67, C 17.45.

**Applicability.** Trend/hysteresis frequency decomposition is a cheap structural prior. A
plain MLP over a 100-sample history plus future B ranked 4th. That is a reminder that our gap
to the top is not about lacking an exotic sequence model [inferred].

## 10. Tsinghua: LLM-agent-designed BiLSTM with linear B skip and grouped pretraining. Honorable mention, official 19.93 % (rank 5)

**Citation.** No paper found. Talk: https://www.youtube.com/watch?v=bW8KJCIxRa8 [transcript].

**Method** [transcript]: inputs B, ΔB and a third channel (the captions say "B and DB and T";
whether "T" means time or temperature is ambiguous); **no past H is used at all**, which makes C work
with no special handling. A BiLSTM, then a dense GELU head, plus a **linear skip from B to the
output** ("approximates the linear permeability contribution"). Pretraining on **groups of
similar known materials** (three base models: e.g. N27/N30/N49/N87; 3C90/3E6/…; 77/78), then
fine-tuning on each target. The best base was chosen per material (base 1 for A, B, C; base 2
for D, E). Pretraining on a subset matched pretraining on all materials (4.17 % vs 4.06 % on
N49) at 1/5 of the time (5 h vs 25 h). Inference uses a sliding window plus "bias correction".
The architecture came from an LLM agent ("FM agent", a self-evolving code-generating agent)
guided by physics prompts; on N49 it went from ~55 % to ~4 % error and ~3.5 % after tuning
(own metric).

**Params** 2,177,283 (size rank 29). **Reported:** official A 29.29, B 12.69, C 22.48, D 19.28,
E 15.89, mean **19.93 %**. Seq rank 5. [measured] w100 25.81, w500 15.61, w900 16.45, C 22.48.
This is the largest w100 penalty among the top 7 (XJTU1 and NITC are larger, +12.4). Within
the same w100 sequences its p95 grows 13.8 → 31.9 % (2.31×, §1.2). That fits a model that never
reads H: only the "bias correction" can use the known history, so an offset at the boundary
is never corrected later in the forecast [inferred].

**Applicability.** (i) A linear B → H skip is a 2-parameter prior worth adding. (ii) Pretraining
on grouped similar materials, then fine-tuning, is RQ2-relevant and cheap. (iii) Even without H,
a bidirectional B-only model reached rank 5 and 22.5 % on C. Its error still grows 2.3× over
900 steps (§1.2), so a B-only model is not immune to horizon-dependent error; the cost of
ignoring the known H shows up as a growing error, not a constant one [inferred]. Note: a bidirectional pass over the future B is allowed here because the whole
future B is given.

## 11. Silicon Austria Labs (SAL): feature-based FFN with short-rollout training. Innovation 3rd, official 39.48 %

**Citation.** Prize page (above); talk https://www.youtube.com/watch?v=asrbcj6cSSk [transcript].

**Method** [transcript]: an early version fitted Preisach per sequence on the history, then an
LSTM; it was abandoned once histories became short (100 samples or 1 sample). The final model
is a **feed-forward net, one-step autoregressive**, on 38 features: 10 material descriptors
from training data (max H, H at 0.9·B_max, retentivity, …) and 28 sequence features
(aggregates of past B, H and T, and of future B). **The last H value was deliberately excluded:
XGBoost importance showed over-reliance on it, which made errors accumulate in rollout.** The
loss combines NRMSE, energy and extra weight near high H. **Rollout training through 20 steps**
with backprop (multi-step loss), to make the model robust to its own errors. A was the weak
material (88.5 %; "under-represented conditions").

**Params** 3,713. **Reported:** official mean 39.48 % (A 88.55, B 18.28, C 28.93, D 31.72,
E 29.92).

**Applicability.** This is the closest published analog to our recipe (one-step autoregressive +
multi-step rollout loss). It lands closer to us (39.5 vs our 50.8–52.4 %) than to the top teams,
with the same failure on A [inferred].
The "drop the last H to stop drift" finding is a more drastic version of our H-noise.

## 12. Remaining notable entries: NITC, XJTU1, Tianjin, Leuven

None of these four has a public paper or code. All per-warm-up numbers below are [measured]
and reproduce the published per-material scores (§1.1). Within-sequence drift for all four is
in §1.2.

**NITC (National Institute of Technology Calicut, Muhammed): mixture-of-experts LSTM, chunked
autoregressive. Official 24.85 % (rank 10), 347,237 params.** Talk:
https://www.youtube.com/watch?v=nFvaGZiIaTA [transcript]. Inputs are B, H, dB/dt (central
difference) and T, min–max normalized per feature and per material. Windows hold 64 past steps
of (B, dB/dt, H) plus the next 16 steps of B and dB/dt as "future forcing". Each expert has
three LSTM encoders (H, B excitation, dB/dt rate) and a 64-unit future-driven LSTM decoder over
the 16 future steps. The decoder output is refined by 4-head cross-attention (decoder =
query, encoders = keys/values). A temperature-adaptive gating network (LSTM + dense, softmax)
weights the experts. Loss: a weighted sum of MSE, a first-derivative (Sobolev) term, a peak
hinge loss on extremes, a per-cycle energy loss and a relative RMS loss; the weights were set
experimentally. **Inference is autoregressive in 16-step chunks.** Their ablation says that
using fewer of the 16 predicted steps per call (e.g. 4 or 1) gave "further deviation from the
ground truth" [transcript, 12:24, 16:03]. C is
handled by padding with the initial value. Self-reported R² 0.93–0.99. Per block: w100 29.27,
w500 22.96, w900 16.86, C 32.15. This is the largest w100 penalty in the top 10, tied with XJTU1.
Within the same w100 sequences: 10.7 % in window 1 → 39.6 % in window 9 (3.70×), the steepest
growth we measured (§1.2).
- Applicability: the strongest autoregressive entry we found, and a direct data point for
  us. Predicting a multi-step chunk per call (fewer feedback events over 900 steps) helped in
  their own ablation [transcript]. It still has the steepest within-sequence drift of any team
  we measured. Its first 100 steps are as good as SiegenPaderborn's, and the error then grows
  almost 4× (§1.2). The
  five-term loss (Sobolev + peak hinge + energy + relative RMS) is a cheap recipe to test.

**XJTU1 (Xi'an Jiaotong University, Wei): method unknown. Official 22.48 % (rank 8),
1,350,913 params.** There is no talk on the Princeton channel (it is listed under
"Unconfirmed" for the Feb presentations), no paper and no code. The pretest PDF
(`Pretest/Results/XJTU.pdf`) holds plots only. Per block: w100 29.50, w500 23.09, w900 17.12,
C 19.44. It is good on C (5th) despite the w100 penalty. Within-sequence: 12.1 → 39.1 % (3.23×,
§1.2), a growth pattern like NITC's [inferred: possibly autoregressive, unverified].

**Tianjin University: dual-encoder LSTM, one-step autoregressive. Official 32.56 % (rank 13),
7,939–38,305 params. Honorable mention "for balanced size and performance".** Talk:
https://www.youtube.com/watch?v=Ddc0bGuUC10 [transcript]. This is the entry most similar to
our setup. **80-step windows** ("short sequences of 80"; the unit is cut off in the captions,
read as steps [inferred]); parallel LSTM encoders for B and for the H history,
cross-attention between them, feature fusion for temperature, and a small 1D-CNN for the rate
of change of B. One-step prediction is fed back through a sliding window. Data: **symmetry
augmentation** "exploiting the symmetry of magnetic loops" doubles the training set [inferred:
(B, H) → (−B, −H)], with frequency- and temperature-balanced sampling. Key methodological
remark: tuning the *architecture* for better one-step accuracy caused "error explosion" in
rollout, while tuning only the *loss and learning rate* for one-step accuracy did not.
Loss: absolute error where |H| is near zero and relative error elsewhere. They found that a
pure relative loss makes the model predict H ≈ 0. Hyperparameters were tuned with Optuna. A
and C used a different, larger model: their two-encoder design needs an H history, which C
lacks [transcript, Q&A]. Per block: w100 25.34, w500 25.28, w900 30.95, C 54.06; A/B/D/E mean
27.2 % [measured from the leaderboard CSV]. Within-sequence: 17.2 → 34.5 % (2.01×, §1.2).
- Applicability: (i) the closest analog to our S4D recipe (80-step window, one-step AR). Its
  A/B/D/E level (27 %) shows that a carefully tuned one-step AR model can be roughly twice as
  good as ours, so autoregression alone does not explain our 50 % [inferred]. It still drifts
  (2.01×), so a one-step AR model is not drift-free either. (ii) **Sign-flip symmetry
  augmentation** is free and should help Material A [inferred; Tianjin reports only that it
  doubles the data]. (iii) The hybrid
  absolute/relative loss near H = 0 matches Fuzhou's observation that errors concentrate
  near zero. (iv) Their warning matches our RESEARCH_LOG §5.2: next-step accuracy does not
  predict rollout quality.

**KU Leuven: denoising diffusion (DDPM) over the missing H. Official 34.54 % (rank 12),
111,985 params. Honorable mention "for new methods exploration".** Talk:
https://www.youtube.com/watch?v=PYCW11p2ENk [transcript]. The task is framed as inpainting
the unknown part of H. A 1D-CNN encoder–decoder (~112 k params) with MLP embeddings of
material, time and temperature is trained to predict the added noise. Inference takes about
1,000 denoising iterations from pure noise. It is non-autoregressive. Weak at low frequency
(large offsets). Per block: w100 28.06, w500 23.45, w900 20.55, C 76.64. Within-sequence:
19.8 → 31.7 % (1.60×, §1.2).
- Applicability: little, beyond one more non-autoregressive data point. Inference cost
  (~1,000 network calls per sequence) is the opposite of our goal.

## 13. Other related work found (not MC2 entries, or not identified)

- **Res-FNO** (KU Leuven): Z. Guo, X. Shen, R. V. Sabariego, arXiv:2604.04150. An FNO plus a
  multi-scale ResNet path, with dB/dt as an input. Evaluated on MC1 tier-2 materials (79, 3C92,
  T37, 3C95, ML95S) and MC2 3C90 minor loops, not on the MC2 A–E protocol. Leuven's MC2 talk
  presented a DDPM (§12), so Res-FNO is separate work by the same department [inferred].
- **PRSA-GRU** (Southeast University, Wei Xu's group): Y. Hu, Y. Zhang, J. Wang, W. Xu,
  "Physics-Reconstructed Self-Attention GRU Framework for Data-Efficient Transient Hysteresis
  Modeling of Power Magnetics," *IEEE Trans. Instrum. Meas.*, vol. 75, art. 2521111, 2026,
  doi:10.1109/TIM.2026.3742024. The abstract mentions a "two-stage physics reconstruction" in a
  decoupled architecture. Per OpenAlex it has an inverse Jiles–Atherton core and ~7 k params
  and is evaluated on 3C90/N30/N87 (review_log T52). The Southeast University affiliation and
  the ~7 k params (Southeast2 averages 6,913) suggest it is Southeast2's entry (official
  248.88 %) [inferred]. Not read beyond the abstract.
- **MIST** (C. Yao, C. Gong, "Cross-Material Support Transfer for Core-Loss Prediction Under
  Waveform Covariate Shift," arXiv:2609.31659). This is a **core-loss (scalar) task**, most
  likely MagNet Challenge 1's final test [inferred: the abstract's waveform classes
  sinusoidal/triangular/trapezoidal are MC1's; MC2 test inputs are PWM transients]. One
  2,784-param model is trained jointly on five materials with FiLM material-identity
  conditioning and no fine-tuning. It reports five-seed D p95 12.38 ± 0.92 % vs the best
  submission's 15.9 %. Not comparable with any MC2 sequence number. It is listed only because
  joint multi-material training with material identity (and no fine-tuning) is a transfer
  idea relevant to RQ2 [inferred].
- **Kirchgässner equivalent circuit** (arXiv:2611.02222) and RHINO-MAG: covered in
  `siegen_paderborn.md` §7. They are not repeated here.

## 14. Cross-team synthesis

### 14.1 Patterns that separate the top teams from the rest

1. **[Correlational pattern across teams; no team published a controlled AR-vs-non-AR ablation
   on the official metric. MagLearn 2 and Cambridge only assert that AR accumulates error.]
   No top-7 entry is known to feed predicted H back as an *input*, and none is known to be a
   one-step teacher-forced model rolled out at test time. SiegenPaderborn does feed predicted H
   back, through its hidden state.** Of the top 7:
   - *Non-autoregressive in H, verified from code or paper:* Hefei (code: whole-sequence causal
     LSTM with an H mask) and Bristol (code + paper: encoder → decoder driven by future B only).
   - *Non-autoregressive, from transcripts only:* Tsinghua (B only) and Southeast1 (analytical
     model + BiLSTM residual + offset calibration).
   - *Unverified:* Fuzhou ("direct" seq2seq) and Eindhoven (encoder–decoder from a 100-sample
     history). The transcripts do not say whether the decoder re-reads its own outputs.
   - *Closed-loop:* SiegenPaderborn (rank 6). Predicted H is hidden-state element 0 and is fed
     back through the state at every step. It is warmed up by state injection, then trained free-running
     over 128–156 steps with full BPTT and no teacher forcing (siegen_paderborn.md §1).
   That is 4 non-AR (2 verified, 2 transcript), 1 closed-loop and 2 unknown. "Train as you
   infer" is not universal either: Sydney (rank 9, non-AR) trains on 100-step targets and
   infers up to 900 steps (§4), and its within-sequence error grows 1.89× (§1.2).
   The autoregressive entries we know are spread across a wide range: NITC with 16-step chunks
   24.9 % (rank 10); Tianjin one-step 32.6 % (rank 13); SAL one-step with a 20-step rollout loss
   39.5 %; ours with K = 32 50.8–64.0 %; NTU2's stock 2,399-param tutorial LSTM 75.1 %
   (RESEARCH_LOG §5.3). **So autoregression is associated with larger drift but does not
   determine the score.** Tianjin's one-step, 80-step-window dual-LSTM reaches 27 % on A/B/D/E,
   about half our error, under the same inference scheme. NITC's own ablation (longer chunks
   are better) and Tianjin's remark (architecture tuned for one-step accuracy "explodes" in
   rollout) point the same way [transcript].
2. **Error grows along the forecast for every top-10 team, least for the non-AR teams that use
   the known H and train over the full horizon [measured, §1.2].** Within the same w100
   sequences, p95 error from window 1 to window 9 grows 1.15–1.65× for Hefei, Fuzhou,
   Southeast1, Eindhoven and Bristol (Fuzhou and Eindhoven unverified), and 1.89–3.70× for
   SiegenPaderborn, Tsinghua, Sydney, XJTU1 and NITC (Tianjin 2.01×). Tsinghua (never reads H)
   and Sydney (non-AR, trained on 100-step targets) show that avoiding H feedback alone is not
   enough. The growth survives a per-window denominator. The per-block averages (§1.1) cannot
   show any of this: the blocks are different sequences, the w900 block is penalized by a small
   denominator, and even so the top-7 blocks differ by up to 9.4 points in both directions
   (only SiegenPaderborn and Bristol are flat). This is the strongest evidence here for "train
   as you infer over the full horizon", and it is still correlational.
   **Our own models (deciding test run, §1.2):** our first-step p95 (15.7–17.8 %) matches
   Tianjin's (16.7 %), but ours doubles within the first ~10 fed-back steps (31–41 %) while
   Tianjin's does not (16.0 %). **Reading [inferred]:** about half of our gap to Hefei is
   present by step 10, and the rest accrues over the remaining steps. The early doubling is what
   distinguishes us from Tianjin at equal one-step error.
3. **They use the known H to set the initial state, but not as a recurrent input.** Hefei
   (masked H channel), Sydney (history GRU → h_init), Bristol (encoder (h, c)), Southeast1
   (offset calibration to known H), SiegenPaderborn (warm-up of the GRU state). Material C
   (one known H value) is where state initialization matters most. Five entries reach ≥ 54 %
   there: Gatech 84.7 % (zero initial state, [transcript]), Leuven 76.6 % (diffusion),
   Cambridge 70.5 % (no H at all), Bristol 56.9 % (1-sample history with a ratio-matched base
   model) and Tianjin 54.1 % (two-encoder design needs an H history [transcript]). Tsinghua's
   B-only model does not collapse (22.5 %), so "no H" alone is not the cause [inferred from the
   leaderboard].
4. **Explicit rate features ΔB and Δ²B (or dB/dt, d²B/dt²)** appear in Hefei, Sydney (ΔB;
   single-run ablation A3: sequence p95 +8.7 points, energy p95 7.60 → 12.66), Gatech (damper +
   inertia), Cambridge (C₁Ḃ + C₂B̈), SAL, NITC, Tsinghua and Res-FNO. The physics rationale given is eddy current plus a capacitive or
   displacement-current term at fast transitions.
5. **One model trained on all warm-up splits, with validation averaged over splits** (Bristol
   explicitly; Hefei's A checkpoint config lists `KNOWN_POINTS = [100, 500, 900]`, and its B/D/E
   checkpoints are named "all in one"). Hefei warns that the trained and evaluated forecast
   intervals must match [transcript].
6. **Losses aimed at the evaluation metric:** quantile-trimmed MSE for p95 (Bristol); smooth-L1
   plus peak/trough emphasis (Fuzhou, Sydney); energy terms (Hefei, Sydney, SAL, NITC);
   Sobolev (derivative) and peak-hinge terms (NITC); absolute error near H = 0 with relative
   error elsewhere (Tianjin). Sydney's L_phys is the only one with a precise published form.
   Two teams independently report trouble near H ≈ 0 (Fuzhou: largest errors there; Tianjin:
   a relative loss collapses predictions to 0).
7. **Transfer from the 10 known materials for A–E:** Bristol (pick the base by target
   validation, then full fine-tune), Tsinghua (grouped-material pretraining), Hefei (fine-tune
   with frozen LSTM supported in the app). No team reports a controlled no-transfer ablation on
   Material A that we could read.
8. **Accuracy versus size:** Hefei's 4–9 k params beat models of 60 k–2 M. Hefei's own sweep says
   accuracy falls "dramatically" below ~1 k params; Bristol's flattens beyond 860 k. Structured
   models dominate the small end: Gatech 83, SiegenPaderborn 325, Kirchgässner circuit 32–656
   (siegen_paderborn.md).

### 14.2 Ranked techniques to consider (for our S4D)

Evidence strength: **A** = controlled ablation with repeat runs, or on the official metric;
**B** = a paper's single-run ablation on its own protocol, a code-verified design of a top team,
or several top teams agree; **C** = transcript or single team only. No technique below
reaches A.

| rank | technique | teams | evidence | why for us / cost |
|---|---|---|---|---|
| 1 | **Train exactly as we infer, over the full horizon.** (a) Non-autoregressive: run the causal S4D over all 1,000 steps with channels (B, ΔB, Δ²B, H·mask, mask) and a sequence loss over the predicted region, with no H feedback. (b) Alternatively, keep feedback but train closed-loop over the whole forecast (≫ our K = 32), as SiegenPaderborn does. A smaller step in the same direction is (c) predicting multi-step chunks per call (NITC: 16) | (a) Hefei (code), Bristol (code + paper), Tsinghua, Southeast1 (transcript), Fuzhou (unverified); (b) SiegenPaderborn; (c) NITC | B (within-sequence drift is smallest for the non-AR teams that use known H and train over the full horizon, §1.2; correlational; no AR-vs-non-AR ablation published; NITC's chunk ablation is transcript-only) | **Measured on our models (§1.2):** our first-step error matches Tianjin's (p95 ≈ 16–18 %), but it doubles within ~10 fed-back steps (31–41 %), and about half our window-9 gap to Hefei is present by step 10, the rest accruing later. So early compounding of fed-back errors is a main error source [inferred]. Option (a) removes the feedback path that compounds. (b) trains against the compounding directly; our K = 32 stage was not enough, so the horizon must be longer, or the loss must weight the first ~10 rollout steps. S4D convolution mode handles L = 1,000 cheaply; (a) leaves params unchanged. Tianjin (§12) shows a one-step AR model *can* avoid early compounding (16.0 % over steps 1–10) and reach ~27 % on A/B/D/E, so how fed-back errors re-enter our model matters as much as the AR scheme. |
| 2 | **Inject T and warm-up/boundary context into the initial state** instead of per-step or head inputs | Sydney (A5: feeding T and start position per step to the future GRU instead of initializing h raised p95 31.4 → 75.7 %) | B (single-run ablation, own protocol; moves T and position together) | Near-zero params. Our T currently enters only the head. |
| 3 | **ΔB, Δ²B input channels + linear skip from (B, ΔB, Δ²B) to the output** | Hefei, Sydney (A3), Gatech, Cambridge, Tsinghua, NITC | B (Sydney A3 is a single run; its main effect is on energy, 7.60 → 12.66 % p95; several top teams use these inputs) | +2–3 input channels, ~3 skip params. Physically motivated (eddy and displacement current). |
| 4 | **Train on all three warm-up splits; select checkpoints by validation averaged over splits** | Bristol, Hefei | B | Our selection already uses test-like warm-ups; the training windows should match too. |
| 5 | **p95-targeted loss**: quantile-trimmed MSE (drop top 5 %) or smooth-L1 + peak weighting; add Sydney's energy term (λ = 0.1); treat the H ≈ 0 region with absolute error | Bristol, Fuzhou, Sydney, NITC, Tianjin | B (MagLearn 2 and Fuzhou claim gains; Sydney's energy term not ablated) | Few lines of code. Should also help our weak energy error. |
| 6 | **Transfer and augmentation for A (and RQ2)**: pretrain on known materials (or groups), choose the base by short target-validation runs, then full fine-tune at lower LR; augment A by resampling shorter segments to full length and by sign-flip symmetry (B, H) → (−B, −H) | Bristol, Tsinghua, Hefei; symmetry: Tianjin | B (MagLearn 2: 3C92 rank 3 with N27 base; no from-scratch ablation shown); symmetry C (transcript) | A is our worst material (92–102 %). Directly RQ2. The symmetry flip costs nothing. |
| 7 | **Structured, bounded output head** (direction-dependent operator mixture, K = 3–4) | Hefei | C (the DAO-vs-plain-head effect is not shown in sources we could read) | ~1 k params. Ablate against our MLP head. |
| 8 | **Initial-state handling for C**: do not assume a demagnetized state; treat C as "no history", not "zero state" | Gatech (self-reported mistake), organizers' clarification in the Gatech Q&A | C (transcript) | Our rollout left-pads C's 80-step window with H(1) = 0, i.e. an all-zero H history (`src/rollout.py`). Our training windows always carry true H history, and C's training sequences do not start at H ≈ 0 (RESEARCH_LOG §2), so the model never sees this input during training [inferred]. Hefei, Fuzhou, Southeast1 and Eindhoven reach 12–17 % on C. Train with 1-step warm-ups (as Hefei's split point 1, SiegenPaderborn's past length 1, Bristol's ratio-matched base) instead of fixing the state. |
| 9 | **Hand-crafted hysteresis memory features** (nearest reversal points, loop-closure fraction, running extrema, reversal density / f_eff) | Cambridge, Bristol (f_eff ablation), SAL | B (Bristol Table IV) / C | Cheap extra channels or scalar conditioning. |
| 10 | **Physics backbone + learned residual** (analytical or Gatech-style model, NN predicts the residual) | Southeast1, Gatech (planned) | C | Bigger change. Southeast1 ranks 3rd–4th on every material and has low drift (1.34×, §1.2). |
| 11 | **Inference-time offset calibration** of predicted H to the last known H | Southeast1, Cambridge (planned) | C | Zero training cost; easy A/B test on validation. |
| 12 | Trend/hysteresis frequency decomposition; VAE latent for generalization | Eindhoven | C | Lower priority. |

What not to copy blindly: MagLearn 2's 2.2 % headline (best-material mean, §7); the pretest PDFs
(self-reported, known-material overlap; RESEARCH_LOG §2); the bidirectional models of Tsinghua
and Southeast1, which are fine here because the whole future B is given, but are not causal
simulators.
