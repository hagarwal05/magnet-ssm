# Transient magnetics modeling with state-space neural networks

CMSC798 independent research (UMD, Fall 2026). Harshil Agarwal, advised by Dr. Mohammad
Nayeem Teli. Built on MagNet Challenge 2 (MC2): given the past B(t) and H(t) of a magnetic
core, the upcoming B'(t) and the temperature T, predict the upcoming H'(t), sampled at
16 MHz. We train small S4D state-space models and a size-matched LSTM under the official
protocol, then score them on the official A–E test set with the organizers' metric, so every
number is directly comparable to the 32-team leaderboard.

- **RQ1, efficiency:** can an S4D model match competitive accuracy with few parameters
  (official error vs parameter count)?
- **RQ2, generalization:** how well does it transfer to materials it never trained on?
  **Not started yet.**

## Current results (3 seeds, 2026-11-06)

Official A–E sequence error (p95 per warm-up block, averaged over blocks, then over A–E;
lower is better). Training: next-step stage with H-history noise σ = 0.1, then a K = 32
multi-step stage kept only if it beats stage 1 on validation. Seeds vary init and batch
order on a fixed split.

| model | params | sequence error (%) | energy error (%) | rank / 35 |
|---|---|---|---|---|
| **S4D small (ours)** | 205 | **52.4 ± 3.3** | 19.6 ± 6.6 | 21 |
| **S4D medium (ours)** | 1,001 | **50.8 ± 5.1** | 16.0 ± 0.8 | 19 |
| Dual-LSTM, official arch., retrained by us | 771 | 64.0 ± 5.9 | 14.9 ± 2.5 | 24 |
| SiegenPaderborn (Performance Track winner) | 325 | 20.4 | 3.4 | 6 |
| Hefei (lowest error) | 6,564 | 9.2 | 1.6 | 1 |

Rank is among the 32 teams plus our three models. **Takeaway:** the S4D beats the
matched-recipe LSTM with fewer parameters (S4D-205 vs LSTM: −11.6 pts, better on 14 / 15
material–seed pairs, exact sign-flip p = 0.0011), but we are still about 2.5× behind the
small-model frontier. Also:

- The two S4D sizes are not distinguishable (−1.6 pts, p = 0.61).
- The S4D advantage is on waveform error only; the LSTM has the lowest energy error.
- Material A (105 training sequences) is the weak point for every model (92–102 %).

Per-material and per-seed tables, paired tests and the error-vs-params plot:
`results/seeds3/README.md`. Earlier stages (`docs/RESEARCH_LOG.md` §5.2–5.4) took the S4D
small mean from 257.7 % (next-step only) to 61.0 % (H-history noise), then to 52.4 %.

## Diagnosis: where the error comes from

**Symptom** (`results/drift/README.md`, `docs/RESEARCH_LOG.md` §5.6): our first-step error
(15.7–17.8 % p95) matches Tianjin's (16.7 %), the closest team to our one-step autoregressive
scheme, but ours doubles within ~10 fed-back steps (31–41 %) and keeps growing (+22 to +38
points over the forecast). This rests on one comparison team, which also differs in
architecture, loss and augmentation.

**Mechanism** (`results/diagnostics/README.md`, `docs/RESEARCH_LOG.md` §5.7, no new training):

- **Near-unit feedback gain:** shifting the fed-back H window by ε moves the next prediction by
  0.93 ε, so our models carry an early error forward instead of correcting it from B. Half of an
  offset is still there after 100 steps for the S4D models.
- **The multi-step stage barely changes steps 1–10** (≤ 1.3 pts), but it cuts late-window
  error by 22–30 points for the S4D. The keep-if-better-on-validation rule dropped it in 11 of
  45 runs, and in 7 of those it would have helped on test.
- **H-history noise trades one-step accuracy for stability:** step-1 error rises from 9.6–12.4 %
  without noise to 15.6–18.9 % with σ = 0.1.

## Next steps

Ranked and sourced in `docs/LITERATURE_REVIEW.md` §9.

1. ~~Diagnostics on the saved predictions~~ **done** (2026-11-09, `results/diagnostics/`).
2. **Train as we infer:** a non-autoregressive S4D over the whole 1,000-step sequence, with no
   H noise (§9 item 1a; RESEARCH_LOG §6, 11-09), plus a Material C fix: train C under its
   test condition, a single known H value (§9 item 7).
3. **Material A:** longer training on all its data (§9 item 4).
4. **Size-matched GRU baseline:** the winner's GRU in our harness at 209 and 1,029
   parameters, next to our 205 and 1,001 (§9 item 8).
5. **RQ2:** cross-material transfer (§9 item 9).

## Documentation map

| file | what it is |
|---|---|
| `docs/RESEARCH_LOG.md` | **Master record:** methods, data, evaluation, every experiment with its numbers, decisions (§6), open caveats (§7) |
| `docs/LITERATURE_REVIEW.md` | Fact-checked literature review, 115 references; §1–8 for the paper, §9 internal implications. Generated: edit `docs/lit_notes/_source/LITERATURE_REVIEW.keyed.md`, then run `python docs/lit_notes/_source/writer_refs.py` |
| `docs/lit_notes/` | Researchers' notes files and `review_log.md`, the fact-check log |
| `docs/reference_audit.md` | What the upstream repo actually supports, including where it differs from `PROJECT_CONTEXT.md` |
| `data/README.md` | Data download, packing, Git LFS |
| `results/*/README.md` | Per-run setup and results (one folder per experiment) |
| `PROJECT_CONTEXT.md` | Original project framing |

## Setup

```bash
scripts/setup_env.sh                 # venv at ~/.venvs/magnet-ssm (outside OneDrive)
source ~/.venvs/magnet-ssm/bin/activate
scripts/fetch_reference.sh           # upstream repo -> third_party/ (pinned commit)
python -m pytest -q                  # 71 tests, about 1 min on an M3 Pro
```

## Data

Details in `data/README.md`.

- **Reference data in git (LFS, ~3.2 GB):** A–E ground truth, the organizers' scoring
  script, the leaderboard workbook and every team's submitted predictions. Install git-lfs
  and run `git lfs install` before cloning, or `git lfs pull` after. To skip the 2.9 GB of
  team submissions: `git config lfs.fetchexclude "data/Competition Result/**"`, then pull a
  team when needed: `git lfs pull --include="data/Competition Result/Hefei/Result/*"`.
- **Training database (not in git, 23 GB):** download the MagNetX Database from Dropbox on
  each machine and pack it losslessly into `data/packed/` (~5 GB, about 10 min):
  `python -m src.pack_data --src ~/magnet-data/"MagNetX Database" --out data/packed`.
- **Don't use the pretest sets as a test:** their segments are cut from the training
  sequences. We report only the A–E test.

## Correctness gate

```bash
python -m src.evaluate --reference tutorial3    # reproduction gate
python -m src.score_submission --team Hefei     # leaderboard scorer check
```

The gate reproduces the upstream tutorial-3 notebook exactly: seq NRMSE p95 33.27 %, mean
16.13 %, energy 39.78 %, and all five per-warm-up means (`GATE PASSED`, exit code 0). Weight
and normalization files are SHA-256 pinned, and the wrong normalization file makes the gate
fail (tested). That gate model has 2,399 parameters, not 771 (see the audit). The scorer
reproduces the published per-material scores of Hefei, Gatech, Cambridge and
SiegenPaderborn to 4 decimals (e.g. Hefei 9.1727 % / 1.5654 %).

## Layout

```
src/
  data.py              HDF5/CSV loaders (windowed training, raw sequences, test sets), splits
  pack_data.py         MagNetX Database zips -> data/packed/ (float32 HDF5 + manifest)
  normalize.py         Normalization_Params.json scheme; tied mode for our models
  metrics.py           tutorial-3 metrics + official grading metrics, drift curve
  rollout.py           variable-warm-up autoregressive rollout (mirrors the upstream loop)
  models/
    baseline_lstm.py   OfficialLSTM (771) and TutorialLSTM (2,399); load upstream .sd strictly
    ssm.py             S4D model; width/depth/state configurable
  train.py             next-step training, H-history noise, multi-step stage, rollout selection
  evaluate.py          metrics + param count for references, checkpoints, or bare .sd files
  score_submission.py  score any H_Material {A..E}.csv folder exactly as Evaluation_Read.m
  checkpoint.py        checkpoints embed their normalization
  reference.py         pinned upstream artifacts and published targets
  plots.py             histogram, drift, error-vs-params, cross-material figures
  experiments/
    final_eval.py      official A–E protocol: train per material, roll out, score (headline)
    drift_analysis.py  within-sequence error vs forecast horizon, ours and 14 teams
    diagnostics.py     early-compounding diagnostics: stage 1 vs 2, noise, H sensitivity
    size_sweep.py      RQ1 sweep on known materials
    cross_material.py  RQ2: zero-shot, few-shot fine-tune, in-domain reference
tests/                 metrics vs upstream outputs, rollout vs the notebook, SSM kernel vs
                       recurrence, multi-step vs eval rollout, scorer vs leaderboard, pipeline
scripts/runs/          exact launch scripts of every real-data run
results/
  seeds3/ drift/ diagnostics/   current results (RESEARCH_LOG §5.5–5.7)
  final_AtoE/ final_AtoE_hnoise/ sweep_noise/   earlier stages (§5.2–5.4)
  reference/           leaderboard tables (transcribed and from Result_Final.xlsx)
docs/                  research log, literature review, reference audit
paper/main.tex         IEEEtran draft, structure only
```

## Common commands

```bash
# Official A-E protocol, the 3-seed run (exact flags: scripts/runs/seeds3.sh)
python -m src.experiments.final_eval --models ssm_s official_lstm ssm_m --seeds 1 2 3 \
  --tag _hn0.1_ms32 --epochs 80 --decay-epoch 25 --batch-size 2048 --max-windows 500000 \
  --rollout-val --rollout-val-every 5 --h-noise 0.1 \
  --ms-horizon 32 --ms-epochs 12 --ms-chunks 30000 --ms-warmup-epochs 4 \
  --device mps --out results/seeds3 --runs ~/magnet-data/runs/seeds3

# Within-sequence drift analysis (reads the --runs predictions above and the teams'
# submissions in data/Competition Result/, which must be pulled from LFS)
python -m src.experiments.drift_analysis --runs ~/magnet-data/runs/seeds3 --out results/drift

# Early-compounding diagnostics: stage 1 vs 2, noise levels, H sensitivity (no training)
python -m src.experiments.diagnostics --out results/diagnostics

# Score a team's submission, or our own predictions written by final_eval
python -m src.score_submission --team SiegenPaderborn
python -m src.score_submission --pred-dir ~/magnet-data/runs/seeds3/seed1/ssm_s_hn0.1_ms32/Result --name s4d_small_s1

# Evaluate a single checkpoint on A-E and write submission-style H_<M>.csv
python -m src.evaluate --checkpoint <model.pt> --test-dir "data/Evaluation Tools/Ground Truth Data" \
  --materials 3C92 3C95 FEC007 FEC014 T37 --save-pred

# Train one model on packed raw sequences (split by sequence)
python -m src.train --model ssm --d-model 8 --n-layers 2 --raw-dir data/packed --materials 3C90 --rollout-val

# RQ1 sweep on known materials (smoke runs only so far) / RQ2 driver (not yet run)
python -m src.experiments.size_sweep --raw-dir data/packed --materials 3C90 N87 3F4 --preset full --seeds 1 2 3 --device cuda
python -m src.experiments.cross_material --raw-dir data/packed --sources 3C90 3C94 3F4 N27 N49 N87 \
    --targets 3E6 77 N30 --d-model 8 --n-layers 2 --seeds 1 2 3 --device cuda

# Figures from any evaluation folder
python -m src.plots results/eval/<tag>
```

The training drivers (`final_eval`, `size_sweep`, `cross_material`) are resumable. Large
predictions go to `--runs`, outside the repo; checkpoints and configs go under
`results/<run>/models/`.

## Known limits

Full list in `docs/RESEARCH_LOG.md` §7.

- Seed variance is measured for one setting only (sd 3–6 pts), so smaller single-seed
  differences from the earlier stages are not established. One recipe, not tuned.
- Model selection uses 200 validation segments and transfers to test inconsistently.
- Material A has 105 training sequences (98 in our split): the largest uncertainty.
- The upstream 771-param weights have no published normalization, so we retrain that
  architecture as the baseline. Compute so far is a laptop GPU (Apple MPS).
