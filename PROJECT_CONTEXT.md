# Project Context for Transient Magnetics Modeling with State-Space Neural Networks

> A machine-learning research project built on the open-source MagNet Challenge 2 dataset.
> This file is the shared context brief and single source of truth. Any agent (including Claude Code) should read it first.

---

## How to use this file

This file carries both the research framing and the concrete technical details needed to build the code and run the analysis. Treat the decisions here as settled unless Harshil says otherwise. Facts in the technical reference were verified by inspecting the actual repository files, so prefer them over assumptions. Confirm current progress with Harshil before assuming a step is finished, and never claim results that do not exist.

---

## People and course

- Student. Harshil Agarwal
- Advisor. Dr. Mohammad Nayeem Teli
- Course. CMSC798, Independent Research, University of Maryland, Department of Computer Science
- Term. Fall 2026

---

## One-paragraph summary

We predict how magnetic materials respond to electrical excitation over time, using machine learning trained on real measured data from Princeton's open MagNet project. The task comes from MagNet Challenge 2, which asks for the full time-domain magnetic response during transients, not just a single steady-state loss number. The challenge is over, but all data and results are public, so we run our own study aimed at a short publishable paper. The chosen angle is to test a state-space neural network (Mamba or S4), a model family the competing teams barely used, and to measure both its accuracy-vs-size trade-off and how well it generalizes to materials it never trained on.

---

## The domain problem

Magnetic cores (inductors and transformers) sit inside almost every power converter, such as chargers, EV inverters, and data-center supplies. They waste energy as heat, called **core loss**. Predicting core loss accurately is a long-standing, hard problem in power electronics. The classical tools (the Steinmetz equation and its variants) are crude approximations that break down under realistic, non-sinusoidal conditions. Princeton's MagNet project measures large amounts of real data on actual materials and releases it openly so machine learning can replace those formulas.

---

## The two MagNet challenges (important distinction)

- **Challenge 1 (2023), steady-state core loss.** Given a repeating waveform, a material, and a temperature, predict the single core-loss number. Repo `PrincetonUniversity/magnet`.
- **Challenge 2 (2025), transient behavior.** This project's target. Predict the full magnetic response during transients (arbitrary, non-repeating excitation). The model must be **frequency-agnostic, time-step-agnostic, and initial-state-agnostic**. Repo `minjiechen/magnetchallenge-2`.

The deliverable is a function

```
H'(t) = f( B(t), H(t), B'(t), T )
```

Given the past B-field and H-field history and the upcoming B-field plus temperature, predict the upcoming H-field.

---

## Why this project targets Challenge 2

- It is a sequence-to-sequence problem, which is modern deep learning and leans on a machine-learning strength rather than deep power-electronics knowledge.
- Its three agnostic constraints map onto live ML themes (generalization, zero-shot transfer, dynamical-system learning).
- It concluded recently with thin published literature, which is where a solo researcher can add something new.
- All final data and every team's results are public and released for future publications.

---

## Research questions

1. **Efficiency.** Can a state-space model match top-tier accuracy with far fewer parameters? Measured by comparing error and model size against the public leaderboard.
2. **Generalization.** How well does it generalize to materials it never trained on? Measured by training on the known materials and testing on the unseen ones.

---

## Scope (one semester)

**In scope**
- One SSM architecture, built and trained.
- A subset of materials, not all 15.
- Both research questions answered.
- Comparison against the baseline and the leaderboard.
- A short workshop-style paper draft.

**Out of scope**
- Beating the overall winner.
- All 15 materials at full scale.
- Hardware experiments or collecting new data.
- Multiple competing architectures.
- A full journal submission this term.

The realistic win condition is a clean, honest, novel result on a subset, publishable as a workshop paper or preprint.

---

# Technical reference (verified from the repository)

Everything below was confirmed by inspecting the files in `minjiechen/magnetchallenge-2`.

## Task framing in code

The baseline treats the problem as **next-step autoregressive prediction**. At each step the model sees an 80-step window of past B and past H, the next B value, and temperature, and predicts the next H value. The 80 steps are the model's **input window**, not the warm-up length.

At test time each 1,000-step sequence is seeded with a **variable warm-up** of true H, one of 100, 300, 500, 700, or 900 steps, with 21 sequences at each length. The model then rolls its own predictions forward over the remaining steps, so the prediction horizon is up to **900 steps**. The evaluation notebook groups sequences by warm-up length using NaN-run-length segmentation. This matters for `rollout.py`, because autoregressive drift over 900 steps is a much harder regime than over a short horizon.

## Data format

### Training file (one HDF5 per material, for example `3C90_Training.h5`)

Keys are grouped into 7 frequency buckets, indexed `1` through `7`.

| Key pattern | Shape | Meaning |
| --- | --- | --- |
| `B_seq_f_{i}` | (2500, 80) | past B window (input history) |
| `H_seq_f_{i}` | (2500, 80) | past H window (input history) |
| `B_scal_{i}` | (2500, 1) | the next B value (the upcoming input to condition on) |
| `H_scal_{i}` | (2500, 1) | the next H value (the training target) |
| `T_{i}` | (2500, 1) | temperature for those samples |

Each bucket holds 2,500 windows, so one material's demo file has about 17,500 samples after concatenating the 7 buckets. Values are float64.

### Test file (padded and true variants, for example `3C90_Testing_padded.h5`)

| Key | Shape | Meaning |
| --- | --- | --- |
| `B_seq` | (N, 1000) | full B excitation sequence |
| `H_seq` | (N, 1000) | H sequence. In the padded file the warm-up prefix is the true H seed and the rest is to be predicted (filled with NaN or padding). The true file holds the full measured H for scoring. |
| `T` | (N, 1) | temperature per sequence |

The warm-up length is **not** fixed at 80. It is one of 100, 300, 500, 700, or 900 true steps per sequence. The 80-step figure is only the model's input window. Read the warm-up length per sequence from the data (the NaN run length), do not hardcode it.

### Full dataset versus the in-repo demo files

The `.h5` files committed inside the repo (for example `tutorials/tutorial-1/3C90_Training.h5`) are small demo subsets for quick testing. The full Transient database lives in the Dropbox folder linked below. In the full set, each of the 10 existing materials has more than 4,500 randomly sampled segments of 1,000 steps each, and each of the 5 final materials (A to E) has around 1,000 such segments. Use the demo files for fast local checks and the Dropbox data for real training and evaluation.

### Final test set (materials A to E)

The official submission was 5 CSV files of predictions plus the parameter count per material. This matters only as the evaluation shape to mirror, since the challenge is concluded.

## Normalization parameters

**Warning. There are two different `Normalization_Params.json` files with different values, and they are not interchangeable.** The one in `tutorials/tutorial-1` has `mean_B` around `6.2e-4`. The one in `tutorials/tutorial-3` has `mean_B` around `-1.9e-11`, and every field differs. The tutorial-3 file is the one that pairs with the official `Model_LSTM.sd` weights used by the evaluator. Using the wrong file silently degrades predictions. Always pin the normalization file to the weights being loaded.

The tutorial-1 values are below for reference. Inputs and target are standardized before the model and de-standardized after.

```json
{
  "mean_B": 0.000621893210336566, "std_B": 0.09532535076141357,
  "mean_H": -0.14776034653186798, "std_H": 16.165433883666992,
  "mean_out": -0.10919976234436035, "std_out": 16.259883880615234,
  "mean_Scal": 0.0007534750059608726, "std_Scal": 0.09643163314234662,
  "mean_T": 44.75714285714286, "std_T": 19.12826442376272
}
```

`B` and `H` normalize the sequence windows, `Scal` normalizes the next-B input, `out` normalizes the H target, and `T` normalizes temperature.

## Baselines (there are two, and the graders used the smaller one)

**Important.** Two baseline architectures exist in the repo.

- **Official baseline (what the evaluator runs).** Defined in `Test/EvaluationExample.ipynb`. Two `LSTM(1, 6)` encoders with `Tanh`, plus a 3-layer projector. About **771 parameters**. This matches all ten weight files in `Pretest/Pretest_Models/*.sd`. This is the honest "small model" reference point, and the one to compare against.
- **Tutorial baseline (larger, do not treat as the official one).** The `Net` class below, about **2,399 parameters** with `ReLU`. It appears only in the tutorials. One leaderboard team (NTU2) submitted this stock model unchanged and scored 75.11%, which is near the bottom, so a bigger stock model is not better here.

Mirror the 771-parameter official baseline as the primary reference, and keep the 2,399-parameter one only as a secondary sanity check.

The tutorial `Net` is defined in `tutorials/tutorial-1/Demo_LSTM_Training.ipynb` as `class Net(nn.Module)`.

- `lstm_B = nn.LSTM(1, 12, num_layers=1, batch_first=True)`
- `lstm_H = nn.LSTM(1, 12, num_layers=1, batch_first=True)`
- `projector = Linear(26, 26) -> ReLU -> Linear(26, 8) -> ReLU -> Linear(8, 4) -> ReLU -> Linear(4, 1)`
- forward. Run each LSTM over its 80-step window, take the last hidden state (12 dims each), concatenate `[B_scal, T, h_B, h_H]` (2 + 12 + 12 = 26 dims), pass through the projector, output the next H scalar.

This tutorial model has about **2,399** trainable parameters (hidden size 12). It is the larger of the two baselines. The official 771-parameter model above (hidden size 6, Tanh, 3-layer projector) is the one the graders actually ran and the one to beat.

Pretrained state dicts are provided as `.sd` files (for example `Pretest/Pretest_Models/Model_LSTM_3C90.sd` and `tutorials/tutorial-1/Model_LSTM_Princeton.sd`).

## Evaluation metrics (exact, from the evaluation notebooks)

Two metrics, both reported as percentages, matching the official "sequence relative error" and "energy relative error".

### Sequence error (per-sequence NRMSE)

```python
mse  = np.mean((pred - meas) ** 2)
rmse = np.sqrt(mse)
norm = np.mean(np.abs(meas)) if np.mean(np.abs(meas)) != 0 else 1e-8
nrmse = (rmse / norm) * 100   # percent
```

The **leaderboard headline is the 95th percentile** of the per-sequence NRMSE, not the mean. The README column is "Avg. 95% Seq. Error [%]". Report the 95th percentile as the primary number so comparisons are like-for-like. Quoting our mean against their 95th percentile would overstate our result by roughly two times. Keep the mean as a secondary diagnostic only.

**Span.** Sequence error is computed over the **predicted region only** (after the warm-up), per sequence.

### Energy error

Energy per unit volume is the open-loop integral of H over dB for each sequence.

```python
def compute_total_energy(B, H):
    dB = np.diff(B, axis=1)
    H_mid = 0.5 * (H[:, 1:] + H[:, :-1])
    energy_density = np.sum(H_mid * dB, axis=1)   # J/m^3 per sequence
    return np.abs(energy_density)

energy_pred = compute_total_energy(B, y_pred)
energy_meas = compute_total_energy(B, y_meas)
energy_nrmse = np.sqrt(np.mean((energy_pred - energy_meas) ** 2)) / np.mean(np.abs(energy_meas)) * 100
```

**Span.** Energy error integrates over the **full 1,000 steps including the warm-up prefix**, unlike sequence error which uses only the predicted region. The two metrics cover different spans, which was confirmed by reproducing the exact baseline numbers. Using the predicted region alone for energy does not reproduce the target.

Report the sequence NRMSE 95th percentile as the headline, the energy NRMSE, and the parameter count. Use these exact definitions so our numbers line up with the published leaderboard.

## Verified reproduction targets (correctness gate)

Before building anything new, reproduce the official baseline on the real test files. Match these numbers to prove the pipeline, normalization pairing, metric spans, and rollout are all correct.

- Official baseline (about 771 params). Sequence error 95th percentile about **33.27%**, mean about **16.13%**, energy error about **39.78%**.
- Tutorial baseline (about 2,399 params, as submitted by NTU2). About **75.11%** 95th percentile.
- Best team (Hefei). About **9.17%** 95th percentile.

If the reproduced numbers do not match, something in the pipeline is wrong and must be fixed before trusting any new-model result.

## Repository tutorials map (what to reuse)

- `tutorials/tutorial-1`. PyTorch LSTM next-step training and testing, the demo data, and the normalization file. This is the main path to mirror for our SSM.
- `tutorials/tutorial-2`. A physics and ODE approach in JAX (uses `equinox`, `diffrax`, `optax`, `jax`). Alternate reference for a physics-informed angle, not our main path.
- `tutorials/tutorial-3`. End-to-end autoregressive evaluation demo (`Demo_LSTM_Evaluation.ipynb`) with padded and true test files. This is where the metric code and the rollout live.
- `tutorials/tools`. MATLAB scripts that turn raw CSV into the HDF5 training and test layout.
- `Pretest` and `Test`. Official rules, save templates, and the evaluation notebook used for ranking.
- `docs`. Slides and the IEEE report template.

## Dependencies

- Main path (ours). `torch`, `h5py`, `numpy`, and for a state-space layer either `mamba-ssm` or a compact S4 or S5 implementation. `matplotlib` for plots.
- Physics ODE path (tutorial-2 only, optional reference). Python 3.12 with `jax==0.5.2`, `equinox`, `diffrax`, `optimistix`, `optax`, plus `numpy`, `scipy`, `pandas`.
- Compute. A single modern GPU is enough. Google Colab free or Pro handles this scale, and the organizers offered to cover Colab Pro for teams.

---

## Suggested repository layout for our project

```
magnet-ssm/
  PROJECT_CONTEXT.md        # this file
  README.md
  requirements.txt
  data/                     # HDF5 files (gitignored; pulled from Dropbox)
  src/
    data.py                 # HDF5 loading, windowing, normalization, train/val split
    normalize.py            # load Normalization_Params.json, standardize/destandardize
    models/
      baseline_lstm.py      # faithful reimplementation of Net for a sanity check
      ssm.py                # the state-space model (Mamba/S4/S5) under study
    train.py                # training loop (teacher forcing on next-step H)
    rollout.py              # autoregressive rollout for long test sequences
    metrics.py              # sequence NRMSE and energy NRMSE (exact formulas above)
    evaluate.py             # run a trained model over test files, report metrics + param count
    experiments/
      cross_material.py     # train on known materials, test on unseen ones (RQ2)
      size_sweep.py         # vary SSM width/depth, record error vs params (RQ1)
  notebooks/                # exploration only
  results/                  # metrics tables, plots, leaderboard comparison
  paper/                    # IEEE TPEL style draft
```

---

## Concrete tasks Claude Code can take on

1. **Reproduce the baseline.** Implement the official 771-parameter model from `Test/EvaluationExample.ipynb` (two `LSTM(1, 6)` with Tanh, 3-layer projector) in `src/models/baseline_lstm.py`, load the `Pretest/Pretest_Models/*.sd` weights with the **tutorial-3 normalization file**, run `evaluate.py`, and confirm the numbers match the reproduction targets above (about 33.27% 95th-pct sequence, 39.78% energy). This validates the pipeline, the normalization pairing, and the metric spans end to end.
2. **Build the data layer.** Write `data.py` to read the `B_seq_f_i`, `H_seq_f_i`, `B_scal_i`, `H_scal_i`, `T_i` groups, concatenate the 7 buckets, standardize with the normalization JSON, and yield `(seq_B[B,80,1], seq_H[B,80,1], next_B[B,1], T[B,1]) -> next_H[B,1]` batches.
3. **Implement metrics exactly.** Put the sequence NRMSE (mean and 95th percentile) and the energy NRMSE in `metrics.py`, matching the formulas above, with a small test that reproduces the notebook numbers on the sample data.
4. **Write the rollout.** `rollout.py` reads each sequence's warm-up length (100, 300, 500, 700, or 900), seeds from that many true H steps, and predicts forward to 1,000, feeding predictions back as history. It must handle a horizon up to 900 steps and stay frequency-, step-, and initial-state-agnostic. Watch for autoregressive drift over long horizons and report it.
5. **Add the SSM.** `src/models/ssm.py` with a state-space sequence model that consumes the B and H history (and conditions on next B and T) and predicts next H. Start from a known-good S4, S5, or Mamba block, keep parameter count low, and expose width and depth as config so the size sweep can vary them.
6. **Run the experiments.** RQ1 as a size sweep (error vs params, to plot against the leaderboard corner). RQ2 as cross-material generalization (train on a subset of known materials, test on held-out ones). Save tidy CSV and plots in `results/`.

### Guardrails for code work
- Mirror the baseline exactly before changing anything, so comparisons are fair.
- Do not invent dataset paths or numbers. Use the Dropbox data for the full set and the repo demo files for quick tests.
- Keep the normalization identical to the reference so metrics stay comparable.
- Watch for autoregressive drift in long rollouts and report it honestly.
- Do not claim leaderboard-beating results. The goal is the small-and-accurate corner and strong generalization, demonstrated honestly.

---

## Scoreboard facts (for comparison)

All leaderboard sequence errors are **95th percentile** ("Avg. 95% Seq. Error [%]").

- Best accuracy was Hefei, about **9.17%** with roughly 6,500 parameters.
- Official baseline is about **33.27%** at about 771 parameters.
- NTU2 submitted the stock 2,399-parameter tutorial model and scored **75.11%**.
- Model sizes across teams ranged from 83 parameters (Gatech) to about 13,000,000 (Hangzhou).
- Bigger did not mean better. Judging also rewarded explainability and novelty.

---

## Timeline and current progress

Phases from mid-September to mid-December, about 12 weeks. State as of late October.

1. Problem and literature. (done)
2. Data, pipeline, and evaluation harness. (done, including the correctness-gate reproduction and the five harness corrections)
3. Build the SSM and run local tests. (in progress. Architecture coded in pure PyTorch, forward pass verified, overfits one batch on the Mac. Full training is GPU-gated.)
4. GPU training and size sweep for RQ1. (planned, GPU-gated)
5. Cross-material generalization for RQ2 and analysis. (planned, GPU-gated)
6. Write-up and final talk. (early to mid December)

### What is done locally (no GPU needed)
- Dataset secured and migrated to the Mac (the corporate-network Dropbox block was worked around on a separate machine).
- Model-agnostic data pipeline with correct normalization pairing.
- Evaluation harness with the 95th-percentile sequence metric, the full-span energy metric, and the variable-warm-up autoregressive rollout.
- Official 771-parameter baseline reproduced as the correctness gate.
- Repo scaffold and paper skeleton.
- SSM architecture implemented and smoke-tested (overfits one batch).

### What is blocked on a CUDA GPU
- Full training of the SSM across materials.
- The size sweep (RQ1) and cross-material generalization runs (RQ2).

### Compute plan
- Primary. UMD Nexus cluster (UMIACS), requested through the advisor Dr. Teli, who already uses it. CUDA, so `mamba-ssm` installs there if ever needed.
- Fallback. Google Colab (free or Pro) for quick runs.
- The target models are tiny, so compute is a scheduling step, not a hardware hunt. A pure-PyTorch SSM also runs on the Mac for small-scale proof-of-life.

---

## Quick-reference numbers

| Item | Value |
| --- | --- |
| Training materials | 10 (3C90, 3C94, 3E6, 3F4, 77, 78, N27, N30, N49, N87) |
| Test materials (held out) | 5 (A to E) |
| Sampling rate | 16 MHz |
| Training buckets per material | 7 frequency groups of 2,500 windows each |
| Model input window | 80 steps |
| Test segment length | 1,000 steps |
| Warm-up length | variable, one of 100 / 300 / 500 / 700 / 900 true steps |
| Prediction horizon | up to 900 steps (autoregressive) |
| Official baseline | two LSTM(1,6) Tanh + 3-layer projector, about 771 params |
| Tutorial baseline | two LSTM(1,12) ReLU + 4-layer MLP, about 2,399 params |
| Headline metric | sequence NRMSE 95th percentile, percent (energy NRMSE secondary) |
| Sequence-error span | predicted region only |
| Energy-error span | full 1,000 steps including warm-up |
| Reproduction target (official baseline) | ~33.27% seq 95th pct, ~39.78% energy |
| Best team | Hefei, about 9.17% seq 95th pct, about 6,500 params |
| Size range across teams | 83 to about 13,000,000 params |
| Teams | 32 |

---

## Links Claude Code can fetch

Agents with web access can retrieve these directly. The default branch is `main`, so the raw links below resolve to file contents.

### Datasets (Dropbox)
- Main MagNet Challenge 2 data (10 existing materials plus the final evaluation folder). `https://www.dropbox.com/scl/fo/o07x7fa6d5elev19pzd1w/AISuBEWf8LQVszFeoaTlqY4?rlkey=8ec7mqukfr9no61h193zb4qdb&dl=0`
- All 32 teams' submitted results and final evaluation data, released for cross-checking and future publications. `https://www.dropbox.com/scl/fo/en5z0mef5cub7q574cknf/AA7moyzEJDOC398WwhvLzJs?rlkey=n9r8jqk6azh461tmxy1ixr3vm&dl=0`

Note. Dropbox folder links often need a browser to enumerate and download files, so a plain fetch may not list contents. If a direct pull fails, Harshil can download the folder and place the `.h5` files under `data/`.

### Repositories
- Challenge 2, this project's target. `https://github.com/minjiechen/magnetchallenge-2`
- Challenge 2 discussion board (questions, clarifications). `https://github.com/minjiechen/magnetchallenge-2/discussions`
- Challenge 1 platform and database. `https://github.com/PrincetonUniversity/magnet`
- Challenge 1 competition repo. `https://github.com/minjiechen/magnetchallenge`
- Challenge website with news feed and leaderboard. `https://minjiechen.github.io/magnetchallenge/`
- Princeton MagNet web app and database. `https://mag-net.princeton.edu/`

### Key repo files (raw, directly fetchable)
- Main README with news and ranking. `https://raw.githubusercontent.com/minjiechen/magnetchallenge-2/main/README.md`
- Official final-test rules. `https://raw.githubusercontent.com/minjiechen/magnetchallenge-2/main/Test/Readme.md`
- Baseline training notebook. `https://raw.githubusercontent.com/minjiechen/magnetchallenge-2/main/tutorials/tutorial-1/Demo_LSTM_Training.ipynb`
- Baseline testing notebook. `https://raw.githubusercontent.com/minjiechen/magnetchallenge-2/main/tutorials/tutorial-1/Demo_LSTM_Testing.ipynb`
- Autoregressive evaluation notebook (metrics and rollout). `https://raw.githubusercontent.com/minjiechen/magnetchallenge-2/main/tutorials/tutorial-3/Demo_LSTM_Evaluation.ipynb`
- Normalization parameters JSON. `https://raw.githubusercontent.com/minjiechen/magnetchallenge-2/main/tutorials/tutorial-1/Normalization_Params.json`
- Tutorial 1 readme (data description). `https://raw.githubusercontent.com/minjiechen/magnetchallenge-2/main/tutorials/tutorial-1/Readme.md`
- Physics ODE path readme (tutorial 2). `https://raw.githubusercontent.com/minjiechen/magnetchallenge-2/main/tutorials/tutorial-2/README.md`
- Physics ODE requirements. `https://raw.githubusercontent.com/minjiechen/magnetchallenge-2/main/tutorials/tutorial-2/requirements.txt`

To fetch any other file, swap the path on `https://raw.githubusercontent.com/minjiechen/magnetchallenge-2/main/<path>`.

### Related reference code (SSMs and prior MagNet tooling)
- MagNetX, a transient MagNet reimplementation. `https://github.com/PaulShuk/MagNetX`
- magnet-engine, modeling tooling. `https://github.com/moetomg/magnet-engine`
- mag-net-hub, pretrained MagNet models. `https://github.com/upb-lea/mag-net-hub`
- Awesome time-series papers (background for sequence models). `https://github.com/TSCenter/awesome-time-series-papers`

### Papers (IEEE, may be paywalled; use for citations)
- "Why MagNet" and "How MagNet" and related MagNet Challenge papers are on IEEE Xplore. Example entries include `https://ieeexplore.ieee.org/document/10169101`, `https://ieeexplore.ieee.org/document/10232863`, and `https://ieeexplore.ieee.org/document/10232911`. The report format is the IEEE TPEL PDF template included under `docs` in the repo.

### Webinar recordings (method walkthroughs, optional)
- The challenge ran recorded webinars on YouTube. Search "MagNet Challenge" on YouTube, or see the links listed in the repo tutorial-5 and tutorial-6 readmes.

---

## Glossary

- **Core loss.** Energy wasted as heat inside a magnetic core.
- **B-field.** Magnetic flux density, the driving input.
- **H-field.** Magnetic field strength, the response being predicted.
- **Transient.** Behavior during non-repeating, changing excitation.
- **Steady state.** Behavior under a clean repeating waveform.
- **Steinmetz equation.** The classical core-loss formula that ML aims to replace.
- **SSM.** State-space model, an efficient sequence model family (S4, S5, Mamba).
- **Autoregressive.** Predicting one step at a time, feeding each prediction back as input.
- **NRMSE.** Normalized root mean squared error, the sequence metric here.
- **Zero-shot or generalization.** Performing on materials never seen during training.