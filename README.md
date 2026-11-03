# Transient magnetics modeling with state-space neural networks

CMSC798 independent research (UMD, Fall 2026). Harshil Agarwal, advised by Dr. Mohammad
Nayeem Teli. Built on MagNet Challenge 2: predict the H-field response of a magnetic core
over time from past B/H, the upcoming B, and temperature, with
`H'(t) = f(B(t), H(t), B'(t), T)`.

Research questions: **RQ1**: can an S4D state-space model match accuracy with far fewer
parameters (error vs size)? **RQ2**: how well does it generalize to materials it never
trained on?

Read `PROJECT_CONTEXT.md` for the framing and `docs/reference_audit.md` for what was
verified against upstream, **including where it differs from the context file**.

## Setup

```bash
scripts/setup_env.sh                 # venv at ~/.venvs/magnet-ssm (outside OneDrive)
source ~/.venvs/magnet-ssm/bin/activate
scripts/fetch_reference.sh           # upstream repo -> third_party/ (pinned commit)
python -m pytest -q                  # 51 tests, ~30 s on CPU
```

Full data goes under `data/` and is committed through Git LFS (see `data/README.md`).
Install git-lfs and run `git lfs install` before cloning on a machine that needs the data,
or run `git lfs pull` after cloning. Without git-lfs, data files arrive as small
pointer stubs.

## Correctness gate

```bash
python -m src.evaluate --reference tutorial3
```

Reproduces the upstream tutorial-3 notebook exactly: seq NRMSE p95 33.27%, mean 16.13%,
energy 39.78%, and all five per-warm-up means (`GATE PASSED`, exit code 0). Weight and
normalization files are SHA-256 pinned, and the wrong normalization file makes the gate
fail (tested).

## Layout

```
src/
  data.py            HDF5/CSV loaders (windowed training, raw sequences, test sets), splits
  normalize.py       Normalization_Params.json scheme; tied mode for our models
  metrics.py         tutorial-3 metrics (headline) + official grading metrics, drift curve
  rollout.py         variable-warm-up autoregressive rollout (mirrors the upstream loop)
  models/
    baseline_lstm.py OfficialLSTM (771) and TutorialLSTM (2,399); load upstream .sd strictly
    ssm.py           S4D model; width/depth/state configurable
  train.py           teacher-forced training (tutorial-1 recipe), optional rollout-based selection
  evaluate.py        metrics + param count for references, checkpoints, or bare .sd files
  checkpoint.py      checkpoints embed their normalization
  reference.py       pinned upstream artifacts and published targets
  plots.py           histogram, drift, error-vs-params, cross-material figures
  experiments/
    size_sweep.py    RQ1
    cross_material.py RQ2: zero-shot, few-shot fine-tune, in-domain reference
tests/               metrics vs upstream outputs, rollout vs a transcription of the notebook,
                     SSM kernel vs recurrence, end-to-end pipeline and drivers
results/reference/   leaderboard table (transcribed)
docs/                reference audit, paper notes
```

## Common commands

```bash
# Train (any model) on the windowed demo file or on raw sequences (split by sequence)
python -m src.train --model ssm --d-model 8 --n-layers 2 --raw-dir data/raw --materials 3C90 --rollout-val
python -m src.train --model official_lstm --train-h5 third_party/magnetchallenge-2/tutorials/tutorial-1/3C90_Training.h5

# Evaluate a checkpoint on official test files and write submission-style H_<M>.csv
python -m src.evaluate --checkpoint results/runs/<run>/model.pt --test-dir data/final --materials A B C D E --save-pred

# RQ1 / RQ2
python -m src.experiments.size_sweep --raw-dir data/raw --materials 3C90 N87 3F4 --preset full --seeds 1 2 3 --device cuda
python -m src.experiments.cross_material --raw-dir data/raw --sources 3C90 3C94 3F4 N27 N49 N87 \
    --targets 3E6 77 N30 --d-model 8 --n-layers 2 --seeds 1 2 3 --device cuda

# Figures from any evaluation folder
python -m src.plots results/eval/<tag>
```

Experiment tables are resumable: re-running skips rows already in the CSV.

## Status (2026-11-03)

Done and verified locally:

- Reproduction gate passes (tutorial-3, 2,399-parameter model; see the audit for why
  this is not the 771-parameter model).
- All 51 tests pass on CPU, and the SSM overfit test also passes on Apple MPS.
- End-to-end runs on the small real data that is available without Dropbox (results below).

Not done:

- Full training, the RQ1 sweep and RQ2 need the Dropbox raw data and a GPU.
- The 771-parameter gate needs normalization files that are not in the upstream repo.
- No accuracy result exists yet. The numbers below are plumbing checks on tiny data, not findings.

### Proof-of-life numbers (tiny data, NOT results)

3C90 demo: 17.5k windows, 200 epochs on CPU. Rollout on the 105 tutorial-3 test sequences:

| model | params | seq NRMSE p95 | mean | energy |
|---|---|---|---|---|
| upstream tutorial-3 weights (gate) | 2,399 | 33.27% | 16.13% | 39.78% |
| upstream `Model_LSTM_Princeton.sd` | 2,399 | 87.60% | 31.60% | 85.55% |
| upstream demo-trained `Model_LSTM_New.sd` | 2,399 | 175.62% | 76.02% | 80.87% |
| ours, OfficialLSTM retrained on demo | 771 | 655.12% | 170.75% | 81.84% |
| ours, S4D d8 L2 trained on demo | 1,001 | 1145.46% | 375.42% | 118.72% |

N87 smoke sweep (`results/size_sweep_smoke_N87b5/`): 128 raw sequences from one frequency
bucket at 25 °C, split by sequence, 40 epochs, one seed. Grading-definition seq error p95 on
65 held-out segments: S4D 205 params 227.9%, S4D 745 params 104.8%, OfficialLSTM 771 params
77.7%.

Both of our demo-trained models drift badly over the long horizon. Upstream's own
demo-trained model does too (175.6%), so 17.5k teacher-forced windows are not enough to
learn a stable rollout. Next-step teacher-forced MSE on held-out windows is low (about 6e-3
to 7e-3 normalized) while rollout error is large. That gap is the drift problem the
proposal flags, and `--rollout-val` exists so model selection uses rollout error instead.
