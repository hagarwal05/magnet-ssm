# SiegenPaderborn (MC2 Performance Track, 1st place): RHINO-MAG

Notes by `winner-siegen`, 2026-11-06. Sources read in full: the paper (arXiv HTML, cached at
`/tmp/mc2lit/rhino.txt`; PDF at `/tmp/mc2reports/rhino_mag.pdf`), the repository at commit
`aec7e99` (2026-10-02, cloned to `/tmp/mc2lit/RHINO-MAG`), the pretest histogram report
(`/tmp/mc2reports/PaderbornSiegen.pdf`), and the MC2 organizers' material (README leaderboard
in `third_party/magnetchallenge-2/README.md`, `Slides-2025`, APEC keynote).

Conventions:
- **[inferred]** marks my own deductions.
- **[computed]** marks arithmetic I did on published numbers.
- "Repo" paths are relative to the RHINO-MAG repository root.
- "Paper" means arXiv 2603.29745v2.

The ablation notebooks under `dev/documentation/ablation_study/` were committed **without
saved outputs** (0 output cells in each). Ablation results therefore come from the paper's
Figs. 9–10. The notebooks give only the experimental setup.

## Citation

- H. Vater, O. Schweins, L. Hölsch, W. Kirchgässner, T. Piepenbrock, O. Wallscheid,
  "RHINO-MAG: Recursive H-Field Inference based on Observed Magnetic Flux Density under
  Dynamic Excitation," arXiv:2603.29745 [eess.SY]: v1 31 Mar 2026 (per the arXiv abstract
  page, recorded in `review_log.md` §0), v2 9 Sep 2026 (the version read here). Accepted to *IEEE
  Transactions on Power Electronics*, DOI 10.1109/TPEL.2026.3733311 (early access; repo
  `README.md` citation block). https://arxiv.org/abs/2603.29745
- Code: https://github.com/upb-lea/RHINO-MAG (MIT license, "Copyright (c) 2026 Paderborn
  University - Department LEA"), JAX + Equinox 0.12.1 + Optax 0.2.4 (`pyproject.toml`).
- Pretest report: "MagNet Challenge 2025 Pretest Evaluation Results by Team Siegen and
  Paderborn, Germany." Same authors plus Marvin Meyer.
- Follow-up by a RHINO-MAG co-author (in this file's scope, per `winner-others`):
  W. Kirchgässner (Beckhoff Automation), "A Data-Driven, Gain-Scheduled Equivalent Circuit
  for the B–H Behavior of Power Ferrites," arXiv:2611.02222, submitted 1 Nov 2026.
  https://arxiv.org/abs/2611.02222. Code: https://github.com/wkirgsn/ferrite-bh-circuit
  (MIT). Read from the arXiv PDF. Cited below as "K26".
- DOIs of the classical works RHINO-MAG cites, resolved on Crossref/OpenAlex by
  `background-lit`:
  - Jiles & Atherton 1986: 10.1016/0304-8853(86)90066-1
  - Preisach 1935: 10.1007/BF01349418
  - Zirka et al. 2012 [45]: 10.1063/1.4747915
  - Mörée & Leijon 2023 [20]: 10.3390/en16093908
  - Benabou et al. 2003 [25]: 10.1016/S0304-8853(02)01463-4
  - Roussel et al. 2022 [49]: 10.1103/PhysRevLett.128.204801
  - HARDCORE [12]: 10.1109/TPEL.2024.3488174
  - RHINO-MAG does not cite dynamic JA (Jiles, J. Appl. Phys. 76(10):5849–5855, 1994,
    10.1063/1.358399). It is listed here only for cross-reference with `background.md` §2.5.
- Organizer sources cited in these notes. All are in `minjiechen/magnetchallenge-2` under
  `docs/` at commit `ab4913c` (2026-08-03). The local clone in `third_party/` is a sparse
  checkout without `docs/`. Text copies are in `/tmp/mc2lit/pdf/`.
  - **`Slides-2025`**: S. Wang, H. Kwon, H. Li, T. Guillod, M. Chen, C. R. Sullivan (MagNet
    2025 Organizing Team), "MagNet Challenge 2: 'From Steady-State to Transient Models!'",
    IEEE PELS, Kickoff Meeting, February 25, 2025. `docs/Slides-2025.pdf`.
    - "Evaluation Criteria" is printed slide 17 (PDF page 18). It covers performance,
      size, novelty and generality, and states "Final Winners selected by the Academic
      Advisory Committee": Sullivan, Perreault, Hui, Biela, Maksimovic.
    - "Award Structure (TBD)" is slide 20 (PDF page 21).
  - **`Handbook-2025`**: "2025 PELS MagNet Challenge 2 (MagNet 2025)" handbook, IEEE PELS /
    Princeton / Dartmouth, header "(Updated on January 30th, 2024)" [sic].
    `docs/Handbook-2025.pdf`, p. 4 of 10. Quote: "the judging committee will evaluate the
    model's accuracy, size, generality, and novelty."
  - **APEC keynote**: M. Chen (Princeton), C. R. Sullivan (Dartmouth), "MagNet Challenge:
    people · magnetics · AI", APEC 2026 keynote slides. `docs/apeckeynote.pdf` (README
    News #17).
    - The deck gives no presentation date. Its slide 17 announces the MagNet Award Ceremony
      for "Tuesday, March 24, 1:30PM-3:00PM, Grand Hyatt Hotel."
    - The winners list ("MagNet Challenge 2 Winners", Performance / Innovation /
      Software-Engineering tracks and honorable mentions) is printed slide 16 (PDF page 16).
  - **`Webinar-6-2025`**: H. Kwon, S. Wang, H. Li, T. Guillod, M. Chen, C. R. Sullivan,
    "MagNet Challenge 2", Tutorial Session 6, Oct. 30th, 2025. Slide 7 gives the A–E
    training and testing sequence counts.
  - **Winners image**: README News #16 → `img/winners.jpg`, "MagNet Challenge 2 Winners
    (March 10th, 2026)". I did not view it myself; its content is recorded in the reviewer's
    `review_log.md` §0, including the Performance-Track citation for Siegen & Paderborn:
    "well-balanced model size, model performance, and Pareto-front exploration".
- Hosted inference: `mag-net-hub` (https://github.com/upb-lea/mag-net-hub, MIT) serves
  `siegen_paderborn` ("GRU with 8 hidden states … RHINO-MAG"), `kirchgaessner2026` (32
  parameters per material) and `kirchgaessner2026_mlp16` (656) through
  `mh.sequence.SequenceModel`. Optional past-B/past-H windows are used for warm-up. The page
  lists no accuracy figures.

## Summary

1. Per material, they train a single-layer **GRU with 8 hidden units (325 parameters).** The
   hidden unit `g[0]` *is* the normalized H prediction. H is never a model input (Paper
   Sec. III-A1).
2. **Warm-up by state injection.** Over the known segment, `g[0]` is overwritten with the
   true H at every step and the other 7 units evolve freely. Prediction then runs closed-loop
   from that state. Training uses exactly this procedure: 28 warm-up steps, then a 128–156-step
   free-running rollout with full BPTT. There is **no teacher forcing**.
3. Inputs are only B~, dB~, d²B~ and T~. Normalization is max-abs. The switching frequency is
   deliberately not used. The loss is a ΔB-weighted RMSE normalized by each sequence's RMS(H),
   so it targets the per-sequence relative error that the metric scores.
4. **The headline 8.02 % SRE / 1.07 % NERE is the *mean* over test sequences. The leaderboard's
   20.43 % / 3.41 % is the *95th percentile*.** The models are the same, but Table V's
   numbers come from re-generated predictions, not from the submitted outputs. Scored under
   the official aggregation, the submission itself gives mean SRE **7.96 %** and p95
   **20.43 %**; that is the like-for-like pair (`review_log.md` P-rows, reviewer re-score).
5. The evaluated physics-structured variants were all worse than or no better than the plain
   GRU: JA, GRU parameterizing JA, JA plus residual GRU, a JA-regularized PINN, and
   magnetization/permeability/vector-field GRUs.
   - Preisach got only a preliminary reverse-direction test (B from H, Table II).
   - LLG was never evaluated; it was judged infeasible at about 6,000 micromagnetic steps per
     sample.
   - The GRU-P wins at small sizes but does not reach Hefei at larger sizes (Fig. 12).
   - A later 32-parameter Prandtl–Ishlinskii equivalent circuit by co-author Kirchgässner
     (K26) roughly matches it (official p95 SRE 19.4 % vs 20.4 %).
   - K26 also reports that a GRU8 *replica*, which K26 calls "weaker than the published
     network," fails at frequency extrapolation.

## Method details

### 1. Architecture, parameter count, and hidden-state initialization

**GRU cell** (Paper eq. 10; repo `rhmag/models/RNN.py` class `GRU`, wrapping
`eqx.nn.GRUCell`). The cell uses the standard GRU gates (z, r, candidate) plus an extra bias
`b_n` inside the reset product:
`g~ = tanh(W x + b + r ⊙ (U g + b_n))`, `g_k = g~ + z ⊙ (g_{k-1} − g~)`.

**Variants evaluated** (Paper Sec. III, Table III):

| name | output / prediction function | warm-up (inverse of output) | outcome |
|---|---|---|---|
| **GRU-P** (submitted) | `H~_k = g_k[0]` (eq. 16) | `g'_k = [H~_k ; g_k[1:]]` (eq. 14) | best at small sizes (Fig. 12) |
| GRU-M (magnetization) | `H~_k = tanh(B~_k − g_k[0])` (eq. 18). The physical form `H = B/μ0 − g[0]·H_max` (eq. 17) was "very poor and training is unstable" | `g'_k[0] = B~_k − atanh(H~_k)` (eq. 19) | worse than GRU-P |
| GRU-L (linear in B) | `H~_k = g_k[0]·B~_k`, i.e. the GRU predicts the normalized inverse permeability (eqs. 20–21) | `g'_k[0] = H~_k / B~_k` (eq. 22) | worse. Cannot represent a B–H time shift |
| LSTM-P | `H~_k = h_k[0]`, same injection as GRU-P (eq. 25). The cell state c starts at 0 | same | slightly less parameter-efficient than GRU-P at most sizes |
| JA (static, inverse) | explicit-Euler inverse JA: `dH/dt = (dB/dt)(1/μ0)[1 − (dM/dH)/(1+dM/dH)]` (eqs. 26–31). 5 parameters, sigmoid-wrapped with heuristic scales (eq. 32) | none | far worse (Fig. 12) |
| GRU-JADP | GRU outputs the 5 JA parameters each step (eq. 33) | cannot be warmed up this way | worse |
| JA + residual GRU (eq. 34) | sum of JA and GRU increments | — | unstable, needed float64, worse. Dropped |
| PINN with JA regularizer (eqs. 36–38) | GRU-P plus `λ·L_JA` (code weight `physics_weight_lambda=1e-6`, repo `rhmag/model_setup.py`) | GRU-P | dropped. "JA is likely incapable of providing significant help" |
| Preisach (differentiable, after Roussel et al. 2022) | tanh hysteron, T = 1e-3 (eqs. 39–42) | — | preliminary test only (Table II) |
| LLG (NeuralMag) | — | — | infeasible: ~6000 micromagnetic steps per 62.5 ns sample (Sec. III-E1) |
| GRU-V (vector-field state) | `H~ = B~ − Σ g(i,j)[0]` (eq. 51) | cannot be warmed up | worse |

**Recurrent, not windowed.** The model consumes one feature vector `x_k` per step and keeps
its state recurrently. No B or H window is used (repo `GRU.__call__` uses `jax.lax.scan`).

**Hidden size and parameter count.** The submission is `GRU8`: hidden size 8, input size 4
(repo `rhmag/runners/final_model_training.py`, `model_types=["GRU8"]`). The 325 parameters
break down as follows [computed from code]:
- GRU cell: `weight_ih` 3·8·4 = 96, `weight_hh` 3·8·8 = 192, `bias` 24, `bias_n` 8. That is
  **320** (field list in equinox 0.12.1 `GRUCell`).
- Plus **3** normalization scalars B_max, H_max, T_max. These are non-static float fields of
  `Normalizer` and are counted by `count_model_parameters` with filter `eqx.is_array_like`.
- Plus **2** feature-normalization maxima. `n_params` adds `len(normalizer.norm_fe_max)`
  (repo `rhmag/model_interfaces/model_interface.py` lines 76–80, 120–124).
- 320 + 3 + 2 = **325**.

The general formula is `3h² + 16h + 5` [computed]. It reproduces every size on the Fig. 6
axis: h = 2/4/6/8/10/16/32 → 49/117/209/325/465/1029/3589. It also matches the wall-clock
models GRU32 = 3,589 and GRU64 = 13,317 (Paper Sec. IV-C). So **the 325 counts the
normalization constants as parameters**, consistent with the MC2 rule "number of
material-specific parameters that need to be kept in the model" (`Slides-2025`, Evaluation
Criteria).

K26 calls the same model "a GRU with eight hidden states and 336 parameters per material"
(K26 Sec. I) and gives no derivation. 336 is exactly what a PyTorch-style GRU(4 → 8) gives:
96 + 192 + separate `bias_ih` and `bias_hh` of 24 each [inferred]. That convention has 16
more biases than Equinox's `bias` (3h) + `bias_n` (h), and it omits the normalization
constants. Parameter counts for the "same" model therefore depend on the counting
convention.

**Hidden-state initialization from the warm-up** (Paper eqs. 12–16, Fig. 3; repo
`RNNwInterface._warmup`, `GRU.warmup_call`, `GRU.construct_init_hidden`):

1. `g'_{k0} = [H~_{k0}, 0, …, 0]`.
2. For k = k0+1 … k1−1: `g_k = GRU(x_k, g'_{k−1})`, then `g'_k = [H~_k ; g_k[1:]]`. The true H
   overwrites unit 0 at every step. In code: `rnn_out.at[0].set(out_true_t)`.
3. Prediction for k = k1 … k2: `g_k = GRU(x_k, g_{k−1})` and `Ĥ~_k = g_k[0]`. The state loops
   back with no injection.
4. Gradients flow through `g[1:]` during the warm-up, but not through the injected H values
   (Sec. III-A1).
5. If the known H has length 1, as for Material C, the loop is skipped and the state is
   `[H~_last, 0…0]` (repo `RNNwInterface.normalized_call`, `if warmup and H_past.shape[1] > 1`).
6. The authors argue this lets the free units adapt to the hysteresis state "without
   accumulation of prediction errors" and supports any warm-up length.

The prediction is bounded [inferred]. A GRU state is a convex combination of the previous
state and a tanh value, so it cannot grow beyond its starting magnitude or 1. H is max-abs
normalized by the *training* maximum, so an injected test value H~ can exceed 1. The bound is
therefore |Ĥ~| ≤ max(1, max|H~_injected|), and a rollout cannot blow up numerically.

### 2. Feature engineering and normalization

**Normalization** (Paper eq. 5; repo `rhmag/data_management.py` `Normalizer`,
`MaterialSet.normalize`):
- Divide by the max-abs over the material's training set for H, B and T. There is no mean
  subtraction.
- Each derived feature is also divided by its own training max-abs (`norm_fe_max`).
- An optional tanh transform of H exists (`transform_H`) but is **off** in the submission
  (`transform_H=False`).

**Submitted input vector, 4 features** (`disable_features="reduce"`; repo
`rhmag/model_setup.py` `setup_featurize`; `RNNwInterface._prepare_model_input`):
`x_k = [B~_k, T~, ΔB~_k, Δ²B~_k]`.
- `ΔB` is `jnp.gradient`, a central difference. `Δ²B` is `jnp.gradient(jnp.gradient(·))`.
  Both are computed over the concatenated past+future B, then sliced
  (`rhmag/features/features_jax.py`).
- Time is normalized with τ' = 1 s instead of 62.5 ns, "for improved numerical stability"
  (Sec. II-A2).
- The central difference at step k uses B_{k+1}, i.e. one step of B lookahead [inferred from
  code]. This is legitimate, since the future B is given.

**Not used:**
- The switching frequency f_sw, "explicitly not" extracted, to keep the model usable for
  non-fixed-frequency waveforms (Sec. II-A).
- **H is not an input.** It enters only through state injection.

**"Full" feature set**, tested and dropped (Sec. II-A2: "dynamic averaging … did not improve
the model performance"; repo `compute_fe_single`): the reduced set plus an 11-sample moving
average of B with odd-reflect padding, and `pwm = sign(dB/dt)`. A `time_shift`-ed B feature
also exists in code but is set to 0. The feature-ablation notebook (`ablation_features.ipynb`
comparing `final-f32` (full) with `ablation-default-f32` (reduced)) has no outputs, and the
paper has no figure for it.

### 3. Training

**Loss** (Paper eqs. 7–8; repo `rhmag/losses.py` `adapted_RMS_trajectory_based`):
`L' = sqrt( mean_k[(H~_k − Ĥ~_k)² · |B~_k − B~_{k−1}|] ) · H_max / RMS(H_{0:k3})`.
- The RMS is taken over the **full original sequence** the subsequence was cut from
  (`H_RMS` precomputed per sequence at load).
- The result is averaged over the batch.
- The ΔB weight brings in NERE's energy weighting. The RMS(H) normalization stops
  low-frequency, saturated sequences (large H) from dominating (Sec. II-B).

**Loss ablation** (Sec. IV-B, Fig. 9; 5 models per loss per material; repo
`rhmag/runners/ablations/loss_ablation_study.py`). The four losses compared were: MSE,
adapted RMS, "only H_RMS⁻¹" (eq. 52) and "only ΔB" (eq. 7).
- Both modifications improve on MSE.
- Combining them "is not clearly better or worse than either isolated version." The authors'
  explanation: both "reduc[e] emphasis on the strongly saturated cases."
- In Fig. 9 the MSE points generally sit higher than the adapted variants, most visibly for
  NERE on FEC007/FEC014. On 3C92, one adapted-RMS model is an outlier above all MSE models
  (visual reading; no table is given).

**Rollout / truncated-BPTT length. This is closed-loop training.** Each training sample is a
random subsequence. The model is warmed up on the first `past_size` steps, then predicts the
rest free-running, and the loss covers the predicted part (repo
`rhmag/training/jax_routine.py` `batched_step`, `single_batch_step`). The lengths differ
between paper and code:
- **Paper (Table VI):** subsequence length l = 156 = past 28 + future 128. For FEC007: 129 =
  1 + 128.
- **Code (`final_model_training.py`):** `tbptt_size=156`, `past_size=28` (1 for C). The loader
  slices `tbptt_size + past_size` samples, so the code's free-running horizon is **156**
  (total 184).
- See Open questions. Either way the horizon is 128–156 steps. Test horizons go up to 900.

**Epoch construction** (repo `train_epoch`, `sample_random_indices`):
- Each training sequence is cut into non-overlapping chunks of length tbptt+past, starting at
  a random offset.
- Each sequence also contributes a chunk at index 0, a chunk ending at the sequence end, and
  one more chunk. That extra chunk is either the next regular chunk or a random one,
  depending on the offset.
- All chunks from all frequency buckets are shuffled together and batched. A final partial
  batch is dropped.
- Batch size is 512.

**Curriculum:** none in the submission. `tbptt_size_start` exists (short sequences first) but
is `None` (`final_model_training.py`). Noise on B (`noise_on_data`) exists but is 0.0.

**Optimizer** (Table VI; repo `rhmag/model_setup.py` `setup_experiment`):
- Adam with lr 1e-3. `optax.exponential_decay` with `transition_begin=2000`,
  `transition_steps=1e6`, `decay_rate=0.1`, `end_value=1e-4`.
- Gradient sanitization: `jnp.nan_to_num(g, nan=0.0, posinf=1.0, neginf=-1.0)` in
  `make_step`, so NaN gradients become 0 and ±inf become ±1. There is no norm clipping.
- float32 (`-f32` in the model file names).

**Epochs and data per material** (Table VI, Table V; `final_model_training.py`):

| material | train seq (Table V) | epochs | past (warm-up) length | notes |
|---|---|---|---|---|
| A 3C92 | 105 | **10,000** | 28 | small data |
| B 3C95 | 6,758 | 1,500 | 28 | 80 / 500 kHz missing from training |
| C FEC007 | 8,158 | 1,500 | **1** | test warm-up is one step |
| D FEC014 | 9,536 | 1,500 | 28 | missing B values in training (Table IV) |
| E T37 | 6,901 | **2,500** | 28 | high-permeability filter material |

- `use_all_data=True`: the submitted models train on **all** provided training data, with
  **no validation split and no early stopping** (Table VI: "no early stopping"). The final
  epoch's weights are kept (`train_model`: `final_model = model` when `val_set is None`).
- Five seeds were trained per material ([12, 53, 66, 105, 6]). The submitted files are the
  `seed12` ones (`final_model_training.py` docstring; `data/models/*_final-*_seed12.eqx`).

**Material A (tiny data).** No special method:
- Same architecture and loss, trained on all 105 sequences for 10,000 epochs.
- The Pareto runner used 25,000 epochs for A's larger GRUs
  (`rhmag/runners/pareto_front_training.py`).
- No transfer or pretraining went into the submission. The paper states "No cross-material
  modeling is evaluated within the scope of this work" (Sec. IV) and lists transfer learning
  to data-poor materials as future work (Sec. V).
- Result: **23.27 % p95 SRE on A** (host value), with 325 parameters and no transfer.

**Material C (no history).**
- Trained with `past_size=1`: every training subsequence starts from a state built from one
  true H value, which matches the test condition.
- Training subsequences start at random points of magnetized training data. The test starts
  from H(1) = 0 [inferred; our log §2 notes C test segments start demagnetized, unlike the
  training data].
- Result: 27.00 % p95 SRE, their worst material.

**Transfer / pretraining in the repo (unreported):**
- `rhmag/model_setup.py` `setup_dataset("X")` pools **the five evaluation materials A–E**, not
  the ten pretest materials, into one training set.
- `rhmag/runners/train_GRU_w_pretraining.py` fine-tunes pooled "X" checkpoints
  (`X_GRU{8,16,32}_pretraining_*`) per material: A 7,500 epochs, others 500.
- `dev/model_tests/hv-crossmaterial_posttraining.ipynb` builds a "frankenstein" model: pooled
  weights plus the target material's normalizer.
- No outputs or results are published for any of these.
- `rhmag/runners/all_materials_all_data_models.py` trains GRU8 on all 14 materials except A.
  Per-material GRU8 weights for all 10 pretest materials ship in `data/models/`.

### 4. Pareto front (size vs accuracy)

- **Fig. 6 (pre-submission model selection).** GRU-P at 49, 117, 209, 325, 465, 1,029 and
  3,589 parameters. Errors are averaged over seeds and measured by cross-validation on a
  70/15/15 split of each material's training data (`setup_dataset`, `split_into_train_val_test`).
  - All four metrics fall roughly monotonically with size on all five materials, with
    diminishing returns.
  - 325 (h = 8) was chosen "to balance model size and accuracy" (Sec. IV-A).
  - No numeric table is given, so I do not read values off the log-scale plot.
- **Fig. 12 (post-challenge, official test data, p95 SRE and NERE).** Seeded trials per model
  type and material; the median line runs over sizes.
  - GRU-P has the best accuracy for its size at small sizes. LSTM-P is close but slightly
    worse "for most of the considered sizes."
  - GRU-L, GRU-V, GRU-M, GRU-JADP and JA are worse.
  - External teams are plotted from the organizers' results. **Hefei is better than any GRU-P
    at larger sizes** (Sec. IV-D).
  - The authors leave open whether the gap is architecture or training.
- **Wall-clock (Sec. IV-C, Fig. 11).** 100 warm-up + 900 prediction steps, b ≈ 3.16 × 10⁵
  sequences in parallel, RTX 6000 Ada: GRU8 (325) 0.110 s vs GRU64 (13,317) 1.722 s, ≈ 15.65×
  faster. For tiny batches the CPU is faster than the GPU.

### 5. What their ablations show mattered most

1. **Anchoring the state to the true H** is the biggest effect (Fig. 10, 5 models per setup).
   - "Init zeros" (GRU from a zero state, no H injection; repo `GRUwZeroInit`) "produces
     poor prediction results." In Fig. 10 its SRE points sit above the H-injection variants
     on all five materials (e.g. 3C92 p95 roughly 0.6–0.7 vs about 0.25; visual reading, no
     table given).
   - Exception: on 3C92 one default seq–seq model is an outlier (p95 ≈ 0.85, SRE_avg ≈
     0.26), above every init-zeros point. Fig. 9 shows a matching 3C92 outlier for the
     default adapted-RMS loss.
   - Both are probably the same default-configuration seed [inferred]. That matters because
     the submitted seed (12) was chosen with no validation set (§3).
   - Among the H-injecting variants, a full-sequence warm-up ("seq–seq") vs a one-step
     initialization `[H~_{k1−1}, 0]` in training ("one–seq") or in both training and
     inference ("one–one") makes **no significant difference**, except a slight advantage of
     seq–seq on FEC014, the most hysteretic material.
   - For FEC007 there is no warm-up, so the three H-injecting setups "are essentially
     identical" there (Sec. IV-B). The comparison is informative on 4 materials.
   - So **one true H value plus closed-loop training is nearly as good as a long H history on
     3 of the 4 materials with a warm-up** (FEC014 the exception).
2. **Loss shape** (Fig. 9): each modification beats MSE, and neither stacks on the other
   (§3).
3. **Physics structure**, a negative result: no evaluated JA, magnetization, permeability or
   vector-field structure helped (Sec. III, Fig. 12, Sec. V). Preisach got only a preliminary
   reverse-direction test (Table II). LLG was judged infeasible and never evaluated.
4. **Features**: derivatives of B are kept. Dynamic averaging and PWM did not help (Sec. II-A2;
   no figure).
5. **Size**: there are diminishing returns past h = 8 at the scales they chose (Fig. 6).

Set up in code but **not reported**:
- `GRUwInputH8`: H fed as an *input* instead of injected into the state
  (`rhmag/runners/ablations/init_ablation_study.py`, `init_type="H_as_input"`). This is the
  closest analog to our design. Its notebook cell is commented out.
- `GRULinearOut8`: zero init with a linear readout.

### 6. Their metrics vs the leaderboard metric: why 8.02 % vs 20.43 %

**Definitions.** SRE (eq. 1) = RMS(Ĥ − H) / RMS(H) over the predicted segment, per sequence.
NERE (eq. 2) = (Σ ΔB·Ĥ − Σ ΔB·H over the predicted part) / Σ ΔB·H over the full
sequence. The code takes the absolute value (`rhmag/metrics.py` `nere`). These match the official
per-sequence definitions (our `RESEARCH_LOG.md` §3.2).

**Aggregation.** Both numbers use the same structure (repo `rhmag/utils/final_data_evaluation.py`
`evaluate_test_scenarios`, `reduce_metrics`, `update_pareto_df`):
1. A per-sequence error.
2. A statistic within each warm-up block (100 / 500 / 900 known; one block for C).
3. A mean over blocks.
4. A mean over materials A–E.

The only difference is the statistic in step 2: `np.mean` gives "Avg." and
`np.percentile(·, 95)` gives "95th-per.". The organizers' leaderboard uses the 95th
percentile (`Slides-2025`: "Model performance: 95th percentile error on RMS error in B-H
sequence"; README column "Avg. 95% Seq. Error").

Caveat: this is the structure of the repo's evaluation code. The paper does not say how
Table V was aggregated.
- For the mean it makes no difference: the three warm-up blocks are equal thirds of each
  test set (`RESEARCH_LOG.md` §2), so a pooled mean equals the block-averaged mean.
- For the p95, neither a pooled nor a block-averaged percentile on the submitted outputs
  reproduces Table V, because Table V uses different predictions (see below).

**The headline averages tie out exactly** [computed]. The per-material values agree only
approximately, as explained below.

| quantity | source | A | B | C | D | E | mean A–E |
|---|---|---|---|---|---|---|---|
| mean SRE (%) | Paper Table V | 10.98 | 6.51 | 9.40 | 6.04 | 7.16 | **8.02** = abstract / README |
| mean NERE (%) | Paper Table V | 0.65 | 1.23 | 1.82 | 1.09 | 0.56 | **1.07** = abstract / README |
| p95 SRE (%) | Paper Table V | 22.99 | 17.46 | 27.63 | 16.20 | 18.38 | 20.53 |
| p95 SRE (%) | organizers (repo `rhmag/utils/provided_final_results.py`) | 23.27 | 17.32 | 27.00 | 16.20 | 18.37 | **20.43** = leaderboard |
| p95 NERE (%) | organizers (same file) | 2.08 | 3.93 | 4.85 | 4.48 | 1.69 | **3.41** = leaderboard |
| p95 SRE (%), block mean | organizers' per-block values (`SIEGEN_PB_HOST_PROVIDED_RESULTS` in `final_data_evaluation.py`) | (21.62+21.78+26.42)/3 = 23.27 | 17.33 | 27.00 | 16.20 | 18.37 | 20.43 |
| mean SRE (%), block mean | same | 11.01 | 6.45 | 9.27 | 6.02 | 7.06 | 7.96 |

So **8.02 % is a mean-over-sequences SRE and 20.43 % is the 95th-percentile SRE**, for the
same 325-parameter models on the same official A–E test set. They are not computed from the
same predictions, though (see below). The like-for-like pair on the submitted outputs is
**7.96 % mean vs 20.43 % p95**, a ratio of about 2.6×. The per-sequence error distribution
is heavily right-skewed, as their pretest histograms show: e.g. 3C90 at 10 % known has mean
8.04 %, p95 21.50 % and max 87.56 %; 78 at 90 % known has max 566 % (pretest report pp. 2,
7).

An independent re-evaluation agrees for four of the five materials. K26 (Table II) ran the
published GRU8 weights on the official A–E test set. Mean SRE matches Table V for A, B, D and
E (10.97, 6.51, 6.04, 7.16) and p95 SRE is within 0.3 points of the organizers' values (23.37,
17.58, 16.25, 18.50). **For C (FEC007) it does not match:** K26 gets mean 8.05 / p95 22.65,
against Table V's 9.40 / 27.63 and the organizers' 27.00.
- Possible causes are a different model file or a different handling of the 1-step warm-up
  [inferred].
- This is not specific to C: for every material A–E the repo ships both a `final-` and a
  non-final GRU8 file (`data/models/`).

**Table V does not come from the submitted outputs** (reviewer-ssm's re-score of the
submitted CSVs; `review_log.md`, `siegen_paderborn.md` rows):
- *Means:* the submission's means are 11.01 / 6.45 / 9.27 / 6.02 / 7.06 (A–E mean 7.96 %).
  Table V's are 10.98 / 6.51 / 9.40 / 6.04 / 7.16. A percentile convention cannot change a
  mean, so the difference must come from different predictions.
- *p95:* no percentile convention applied to the submitted CSVs reproduces Table V's p95.
  NumPy-linear pooled gives 23.81 / 17.03 / 26.99 / 16.52 / 17.75, and block-averaged gives
  22.83 / 17.22 / 26.99 / 16.17 / 18.29. Table V has 22.99 / 17.46 / 27.63 / 16.20 / 18.38.
- *K26:* Table V's means for A, B, D and E equal K26's re-run of the published weights
  (10.97 / 6.51 / 6.04 / 7.16).

So Table V was computed from **re-generated predictions** of the published models, not from
the submission. Why the re-run differs from the submitted outputs is not stated. Plausible
causes are float precision (the evaluation notebooks enable float64, the models were trained
in float32) or other changes in the inference path [inferred]. The headline 8.02 % comes from
that re-run, not from the organizers' sheet.

**Why they won with 20.43 % while Hefei had 9.17 %.** The Performance Track was judged on
error *and* model size, and winners were "selected by the Academic Advisory Committee"
(`Slides-2025`, Evaluation Criteria). SiegenPaderborn ranks 6th in sequence error, 6th in
energy error and 3rd in size. Hefei ranks 1st, 1st and 10th (README leaderboard). The
published results are Siegen & Paderborn 1st, Hefei & CityU & Xiamen 2nd, Sydney 3rd (APEC
keynote slide 16, "MagNet Challenge 2 Winners"). The award citation on the organizers'
winners image reads "well-balanced model size, model performance, and Pareto-front
exploration" (as recorded in `review_log.md` §0). No published scoring formula combines the
criteria [as far as the sources I read show].

### 7. Follow-up from the same group: a 32-parameter equivalent circuit (K26)

Kirchgässner is a RHINO-MAG co-author. His K26 paper asks whether "the information the GRU
extracts from the data can be expressed as a governing equation" (Sec. I). All details below
are from the arXiv PDF of 2611.02222.

**Model.** A difference equation at the 62.5 ns step (eqs. 10–14, Fig. 3) with five parts:
- **Measurement correction**: `u = B + τ1·Ḃ + τ2·B̈`, central differences in samples.
- **Six viscous play operators**: a Prandtl–Ishlinskii bank with log-equidistant widths and a
  shared rate α < 1 (rate-dependent).
- **Two first-order relaxations** (magnetic after-effect).
- **Mean-field feedback** of H_{n−1} into every operator input, "as the effective field of
  the Jiles–Atherton model … written in the B-driven direction."
- **Output**: `H = h_s(u) + Σ w_i p_i + H_off`, where `h_s` is a linear + cubic +
  exponential-knee reversible curve. A first-order lag `H^m = η H^m_{n−1} + (1−η) H` then
  models the measured H.

Every coefficient is constant or linear in temperature. That gives **32 parameters per
material** (Table I): 5 for the measurement chain and 27 for the material. The model has an
exact SPICE realization with positive elements apart from the feedback sources (Sec. IV-A),
and it can be run at any time step (Sec. IV-B).

**Warm-up state estimator.** This is separate from the model and has **33 learned parameters
per material, not counted in the 32** (Sec. V). It has three parts:
- A prior that places each play state within its band from the first sample's level and
  direction (eq. 20).
- Learned gains that distribute the initial mismatch across states, followed by a
  constant-gain Luenberger observer over the warm-up.
- A Bayesian-ridge amplitude correction fitted on an open-loop pass over the warm-up.

**Training** (Sec. VI):
- Adam with a one-cycle schedule (peak lr 2e-2), 20,000 steps.
- Each step draws **768 random 1000-sample windows per material**, with a random warm-up
  `k ∈ {100, 500, 900}` "as in the evaluation."
- **Loss = mean of SRE + NERE of the predicted part**, i.e. the evaluation metric itself,
  summed over materials.
- So training covers the full test horizon (up to 900 closed-loop steps), unlike RHINO-MAG's
  128–156 [inferred from the two papers].
- A fused Triton GPU kernel with a hand-derived adjoint makes this feasible: about 400×
  faster than a PyTorch loop. One training run of all 15 materials takes about 1 h for the
  classical circuit on one AMD RX 9070 XT. The scheduled circuits take about 1 h (h = 16)
  and 2 h (h = 64).

**Gain-scheduled variant** (Sec. IV-D):
- Per material, an MLP with 2 hidden layers of h tanh units maps
  `(u, asinh(Ḃ), asinh(B̈), ϑ)` to 16 per-step shifts of the circuit coefficients. The map is
  symmetrized so that H(−B) = −H(B).
- The MLP is stateless; the memory stays in the play states.
- h = 16 gives 656 parameters per material; h = 64 gives 5,552.

**Evaluation protocol.**
- Training materials: 70/15/15 split by sequence, stratified by temperature, with 300 fixed
  test windows per frequency file.
- A–E: the official MC2 test set.
- The published RHINO-MAG GRU8 models were evaluated "on exactly the same windows" (Sec. III).
- "Mean test SRE" is the per-window mean averaged over scenarios, **not** the p95.

Results (Tables II, III, V; Fig. 5). The GRU8 row pairs two different sources:
- K26's abstract compares mean SRE 7.33 % (circuit) with 7.77 % for "the challenge-winning
  recurrent network with 336 parameters." That is K26's own evaluation, with K26's
  parameter count.
- K26 Sec. VII-A compares official p95 SRE "19.4 % vs 20.4 %" with "the challenge winner
  RHINO-MAG" (winner of the Performance Track). Here 20.4 % is the organizers' leaderboard
  value, which the leaderboard lists at 325 parameters.
- Both numbers describe the same submitted model. Only the parameter-count convention
  differs (see §1).

| model | params / material | mean test SRE, 15 mat. (%) | official p95 SRE / NERE, A–E (%) |
|---|---|---|---|
| RHINO-MAG GRU8 (published weights) | 336 per K26 = 325 per team / leaderboard (same model) | 7.77 (K26's own evaluation) | 20.4 / 3.4 (organizers' leaderboard, listed there at 325 params) |
| classical circuit (10 seeds) | 32 (+33 estimator) | 7.33 ± 0.03 | 19.43 ± 0.22 / 5.12 ± 0.19 |
| MLP16-scheduled (5 seeds) | 656 (+33) | 5.25 ± 0.04 | 13.03 ± 0.17 / 1.91 ± 0.14 |
| MLP64-scheduled (5 seeds) | 5,552 (+33) | 4.80 ± 0.03 | 12.09 ± 0.14 / 1.61 ± 0.01 |

- The classical circuit beats GRU8 on mean SRE on 12 of 15 materials. It is behind on N87,
  **3C92 (Material A: 12.34 vs 10.97 % mean SRE; p95 NERE 10.71 vs 3.83 %)** and FEC014
  (Table II).
- The author attributes the 3C92 deficit to "only 15 training sequences per frequency," so
  "the saturation knee is poorly determined at 70 °C" (Sec. VII-A).
- MLP64 "would rank second of 33 … by both metrics"; only Hefei is better.
- **Measurement lag in H.** The fit found a lag in the recorded H. Without the explicit lag,
  the 3C92 and FEC014 fits form an output autoregression with coefficients 0.59 and 0.51
  (Sec. IV-C).
- **Ablation** (Table IV, mean SRE Δ against 7.30 %): removing the measurement correction of
  B costs +5.32 points. Rate-independent plays cost +3.68, the mean-field feedback +0.79, the
  saturation knee +0.61, the relaxations +0.38, the cubic term +0.29 and the H lag +0.16
  (3C92 +1.5).
- **Extrapolation and data efficiency** (Table V). This used a GRU8 *replica* trained with
  K26's pipeline ("weaker than the published network"), on 3C94/N87/3F4/77, restricted /
  full-data SRE %:

  | training restriction | classical | GRU8 replica |
  |---|---|---|
  | f ≤ 320 kHz → test at 500/800 kHz | 10.2 / 5.3 | **54.5** / 6.0 (up to 106 % on 77) |
  | 25, 50 °C → test at 70 °C | 9.3 / 5.6 | 16.4 / 7.9 |
  | 25, 70 °C → test at 50 °C | 7.6 / 6.8 | 16.5 / 10.1 |
  | 10 % of sequences | 6.5 / 6.4 | 9.4 / 9.6 |
  | 2 % of sequences | 6.5 / 6.4 | 9.1 / 9.6 |
- **Specimen transfer** (Sec. IV-C). The fitted circuit was applied to MC1 recordings of the
  same ten materials on other cores. Refitting only the 5 measurement parameters recovers
  two thirds of the gap between no refit (10.8 % SRE) and a full refit (7.1 %). Adding the 14
  amplitude parameters recovers 84 %. The 13 dynamic parameters add 0.6 points: "the dynamics
  carry over almost unchanged."
- Disclosure: the acknowledgment states that Claude Opus 5.5 "assisted in every facet of this
  work," including models, experiments and writing.

## Reported results

**Submitted GRU-P, 325 parameters, official A–E test** (Paper Table V; organizers' values
from repo `rhmag/utils/provided_final_results.py`):

| | 3C92 (A) | 3C95 (B) | FEC007 (C) | FEC014 (D) | T37 (E) | mean |
|---|---|---|---|---|---|---|
| train / test sequences | 105 / 1,260 | 6,758 / 945 | 8,158 / 1,116 | 9,536 / 1,242 | 6,901 / 1,080 | |
| mean SRE % (Table V) | 10.98 | 6.51 | 9.40 | 6.04 | 7.16 | 8.02 |
| p95 SRE % (official) | 23.27 | 17.32 | 27.00 | 16.20 | 18.37 | **20.43** |
| mean NERE % (Table V) | 0.65 | 1.23 | 1.82 | 1.09 | 0.56 | 1.07 |
| p95 NERE % (official) | 2.08 | 3.93 | 4.85 | 4.48 | 1.69 | **3.41** |
| model size on disk | 3 kB | 3 kB | 3 kB | 3 kB | 3 kB | |

**Official per-warm-up-block p95 SRE %** (`SIEGEN_PB_HOST_PROVIDED_RESULTS`):
- A: 21.62 / 21.78 / 26.42 for 10 / 50 / 90 % known.
- B: 15.17 / 17.71 / 19.10.
- C: 27.00 (one block).
- D: 20.51 / 16.76 / 11.33.
- E: 17.56 / 19.83 / 17.71.
- On A and B the shortest predicted segment (90 % known, 100 steps) has the *highest* p95.
  The relative error over short segments is noisier [inferred].
- Mean over A, B, D, E per block [computed]: 18.72 / 19.02 / 18.64. `winner-others` got
  the same three values independently by re-scoring the submitted predictions
  (`other_teams.md` §1), which confirms the per-block organizer values above.
- The flat average hides opposite per-material trends. Going from 10 % to 90 % known, the
  p95 rises on A (21.62 → 26.42) and B (15.17 → 19.10) but falls on D (20.51 → 11.33).
- For comparison, `winner-others` re-scored Hefei at 9.86 / 6.85 / 7.88 on the same blocks
  and materials (`other_teams.md` §1). Hefei's lead is therefore present at every warm-up
  length, not only at short warm-ups.

**Pretest report** (per-material GRU on the ten known materials, pretest segments; SRE %,
blocks 10 / 50 / 90 % known):

| material | mean SRE | p95 SRE | p95 NERE |
|---|---|---|---|
| 3C90 | 8.04 / 7.40 / 5.85 | 21.50 / 20.51 / 18.36 | 5.09 / 3.07 / 0.86 |
| 3C94 | 7.99 / 7.73 / 6.40 | 19.90 / 20.22 / 19.46 | 4.47 / 2.95 / 0.89 |
| 3E6 | 9.60 / 9.59 / 9.29 | 19.21 / 20.35 / 25.26 | 2.86 / 1.70 / 0.41 |
| 3F4 | 5.75 / 5.04 / 3.82 | 15.37 / 14.26 / 12.31 | 7.35 / 4.40 / 1.18 |
| 77 | 9.55 / 8.81 / 6.75 | 25.49 / 24.69 / 20.68 | 3.46 / 2.21 / 0.57 |
| 78 | 8.25 / 7.64 / 6.24 | 21.99 / 20.56 / 18.49 | 3.84 / 2.37 / 0.67 |
| N27 | 8.66 / 8.60 / 7.21 | 26.28 / 27.19 / 23.37 | 3.27 / 2.10 / 0.63 |
| N30 | 9.01 / 8.95 / 7.46 | 21.16 / 21.67 / 22.32 | 2.69 / 1.84 / 0.51 |
| N49 | 15.29 / 14.12 / 9.76 | 35.81 / 33.75 / 28.92 | 38.96 / 21.46 / 4.05 |
| N87 | 7.55 / 7.10 / 5.59 | 19.07 / 18.25 / 17.22 | 4.43 / 2.44 / 0.63 |

**[caveat]** Our log (§2) found that pretest segments are cut from the training sequences, so
these numbers may be inflated. The report does not say which model variant produced them.

**Preisach vs GRU-P, preliminary** (Table II): 3C90, 320 kHz, 25 °C, predicting **B from H**
(the reverse direction). Preisach (328 params) vs GRU-P (248 params): MSE 0.0081 vs 0.0005,
MAE 0.0534 vs 0.0143, worst-case error 0.6763 vs 0.2417.

**Against us** (official p95 SRE %, same scorer; ours are 3-seed means from
`results/seeds3/README.md`):

| model | params | A | B | C | D | E | mean |
|---|---|---|---|---|---|---|---|
| SiegenPaderborn GRU-P | 325 | 23.27 | 17.32 | 27.00 | 16.20 | 18.37 | 20.43 |
| our S4D small | 205 | 91.6 | 24.4 | 57.3 | 35.6 | 53.3 | 52.4 |
| our S4D medium | 1,001 | 99.0 | 31.6 | 47.2 | 30.6 | 45.7 | 50.8 |
| gap, S4D small − GRU-P | | 68.3 | 7.1 | 30.3 | 19.4 | 34.9 | 32.0 |

[computed] **Material A alone accounts for 43 % of our gap** to the winner (68.3 of 160.0
summed points). B is nearly closed.

## Applicability to our project

Ranked by expected value. "Expected impact" is my judgment [inferred] unless it is plain
arithmetic.

1. **Make closed-loop rollout the main training regime, at a horizon of about 128–160
   steps.**
   - Theirs: every gradient step is a 28-step H-anchored warm-up followed by a 128–156-step
     free-running prediction, for 1,500–10,000 epochs, and no teacher forcing.
   - Ours: 80 teacher-forced epochs, then a 12-epoch stage 2 at K = 32 (`RESEARCH_LOG.md`
     §4.3). Our log already shows drift is the dominant error (§5.2) and that stage 2 helps
     where it is kept (§5.4).
   - Change: raise K to 128–160 and make stage 2 the long stage, with many more epochs and the
     stage-1 length cut.
   - The co-author follow-up (K26 Sec. VI) goes further: it trains on full 1000-sample
     windows with a random warm-up k ∈ {100, 500, 900}, i.e. the exact test condition with
     rollouts up to 900 steps, and uses SRE + NERE as the loss. If compute allows, sampling
     the test's warm-up lengths at the full horizon is the end point of this change.
   - Impact: high. It attacks our main failure mode, and it is the regime under which a
     325-parameter model reaches 20 %.
   - Effort: medium. Rollout cost grows about 4–5× per sample. Our sliding-window S4D costs
     O(window) per predicted step, while running the diagonal SSM in recurrent mode would make
     each step O(1) [inferred].
2. **Material A: train like they did before concluding the data is too small.**
   - They reach 23.27 % on A with 105 sequences, no transfer, 10,000 epochs, all data and no
     early stopping. We are at 92–102 % for every model.
   - That suggests our A failure is the training regime (drift / teacher forcing), not data
     scarcity [inferred].
   - Change: A-specific long closed-loop training (item 1) on all sequences.
   - Impact: by arithmetic, each 10 points cut on A lowers the A–E mean by 2 points. Closing A
     to ~30 % would remove about 12 of our 32-point gap.
   - Effort: low (config).
3. **Swap the loss for their per-sequence relative loss in the rollout stage**: RMSE weighted
   by |ΔB|, divided by RMS(H) of the full source sequence.
   - The official metric is a per-sequence *relative* error. Our MSE on z-scored H
     over-weights large-amplitude (saturated, low-frequency) sequences.
   - Their ablation: either modification alone beats MSE.
   - Impact: moderate, on p95 especially. Effort: low (we need the source sequence's RMS(H)
     per chunk, which we can precompute).
   - Alternative: train directly on per-window SRE (+ NERE) as K26 does (Sec. VI).
4. **Shrink or remove the H input window. Anchor H instead of feeding it back.**
   - Their warm-up ablation shows one true H value is nearly as good as a long H history on
     3 of the 4 materials with a warm-up (FEC014 the exception; FEC007 has no warm-up).
   - They never feed H as an input. Our design feeds 80 past H values back in, which is
     where exposure bias enters [inferred].
   - Cheap test first: keep the 80-step B window but give the model only the last few H values
     (1–8).
   - Larger change: an S4D in recurrent mode with one state/output channel overwritten by the
     true H during warm-up (their eq. 14 applied to an SSM).
   - Impact: possibly high for drift but uncertain. Effort: low for the window test, high for
     the recurrent redesign.
   - Their unreported `GRUwInputH8` ablation is the direct analog. We could run it with their
     code to learn whether H-as-input itself hurts.
5. **Bound the output.**
   - Their prediction lives inside a GRU state, so it stays within max(1, max|injected H~|)
     in max-abs units (§1).
     Our head is unbounded, so a drifting rollout can run away and land in the 95th-percentile
     tail.
   - Change: a max-abs-normalized H target with a `tanh` (or clamp) at the output.
   - Impact: tail (p95) reduction, uncertain. Effort: low.
6. **Material C: train under the test condition.**
   - They use `past_size=1` for C: every training rollout starts from one known H.
   - Change: in our stage 2 for C, build the H window by left-padding the single starting
     value, exactly as at test time, rather than using 80 true values.
   - Impact: moderate on C (our 47–57 % vs their 27 %). Effort: low.
   - Neither approach covers the demagnetized H(0) = 0 start of the C test [inferred]. Starting
     some rollouts from near-zero H could.
7. **Add ΔB and Δ²B as explicit inputs** (central differences, max-abs normalized).
   - Our S4D can in principle learn derivatives from the B window, so the gain is uncertain.
   - Supporting evidence from K26's ablation (Table IV): the second-order "measurement
     correction" `u = B + τ1 Ḃ + τ2 B̈` is the single most important element. Removing it
     costs +5.3 points of mean SRE, and the GRU then wins on all 15 materials. This
     corrects acquisition dynamics, not material physics.
   - Impact: low–moderate. Effort: low.
8. **RQ1 framing and fairness.**
   - Report our mean SRE alongside the p95. Compare our p95 (50.8 / 52.4 %) with their
     **20.43 %**, never with 8.02 %.
   - For a mean-to-mean comparison, use their submission's official-aggregation mean of
     **7.96 %**, not Table V's 8.02 %, which comes from re-generated predictions (§6).
   - Count normalization constants the way they do: their 325 includes 5. Add our
     normalization statistics to our counts, or state the difference.
   - Size-matched references from their formula `3h² + 16h + 5`: GRU-P h = 6 has **209**
     parameters (vs our 205) and h = 16 has **1,029** (vs our 1,001).
   - Re-implementing GRU-P in our PyTorch harness (about 50 lines) at those two sizes would
     give a strong, size-matched baseline under identical data and scoring. Effort: medium.
     Value for the paper: high.
9. **RQ2 (unseen materials).**
   - The paper does no cross-material evaluation and names transfer learning as future work.
     That leaves an open gap our RQ2 can fill.
   - Usable assets in their repo:
     - Per-material GRU8 weights for all ten pretest materials (`data/models/`).
     - A pooled-pretraining and fine-tuning path (`train_GRU_w_pretraining.py`).
     - The "frankenstein" idea (`hv-crossmaterial_posttraining.ipynb`): pretrained weights
       plus the target material's max-abs normalizer, which needs only B_max, H_max and T_max
       from the new material.
   - That last idea is a cheap zero-shot baseline for RQ2. Effort: medium.
   - K26 gives two RQ2-relevant reference points.
     - **Frequency extrapolation is the GRU's weak spot**: trained on f ≤ 320 kHz, a GRU8
       replica reaches 54.5 % SRE at 500/800 kHz vs 10.2 % for the circuit (Table V). We
       should report our S4D under the same restriction.
     - **Most cross-specimen difference sits in a few parameters**: the measurement chain
       and the amplitudes, while the dynamics carry over (Sec. IV-C). For RQ2 this suggests
       fine-tuning only input/output affine layers or normalization on a new material, with
       the recurrent core frozen [inferred]. That is a few-parameter adaptation with a
       physical reading. Effort: medium.
10. **A 32-parameter physics baseline now exists and is open source.**
    - K26's classical circuit reaches 19.4 % official p95 SRE with 32 parameters per material
      (plus 33 for the warm-up estimator). That is below RHINO-MAG's 20.4 %, which the
      leaderboard lists at 325 parameters and K26 counts as 336.
    - For a fair plot, use one counting convention for every point, or label each point with
      its source's count.
    - For RQ1 this moves the small-model frontier. Our accuracy-vs-parameters plot should
      include it, along with MLP16 (13.0 %, 656) and MLP64 (12.1 %, 5,552).
    - Effort: low (numbers only).

## Open questions

- **Rollout horizon:** paper Table VI says future 128 (subsequence 156). The code passes
  `tbptt_size=156` with `past_size=28`, i.e. future 156. Which produced the submission?
- **Seed selection:** five seeds were trained per material with no validation set; the
  submitted files are all `seed12`. Was seed 12 chosen on some criterion or simply the first
  seed?
- **Table V vs official numbers (partly resolved):** it is not a percentile-method effect.
  Table V was computed from re-generated predictions, not from the submitted outputs (§6).
  Open: why do the re-generated predictions differ from the submission? Candidates are
  precision, the model file, or the inference path. Was the C model the same?
- **Outlier seed:** is the 3C92 outlier in Figs. 9 and 10 the same default-config seed, and
  is it seed 12, the submitted one?
- **Sequence counts (partly resolved):** Table V's training counts (105 / 6,758 / 8,158 /
  9,536 / 6,901) are identical to the organizers' slide (Webinar-6-2025, "Tutorial Session 6,
  Oct. 30th, 2025," slide 7). They differ from the files we downloaded (B 6,669, C 7,990,
  D 9,530, E 6,445; `RESEARCH_LOG.md` §2). Table V's *test* counts (1,260 / 945 / 1,116 /
  1,242 / 1,080) match our test files, not that slide (1,260 / 963 / 1,119 / 1,242 / 1,089).
  So the training counts in Table V may be copied from the slide rather than counted
  [inferred].
- **FEC014's missing B values** are listed as a challenge (Table IV), but I found no NaN
  handling in `rhmag/data_management.py`. How were they treated?
- **Unreported experiments:** H-as-input (`GRUwInputH8`), linear readout (`GRULinearOut8`),
  full vs reduced features, and pooled-A–E pretraining with fine-tuning all have code but no
  published results.
- **Pretest report:** which model produced it, and was it scored on pretest segments that
  overlap training data (our §2 caveat)?
- **Tail composition:** which sequences make up their p95 tail (frequency, amplitude, warm-up
  block)? Their 90 %-known block has the highest p95 on A and B, which suggests short-segment
  noise in the relative error.
- **Judging:** what weighting of error and size did the committee use for the Performance
  Track?
- **K26 vs Table V on C:** K26 re-scores the published GRU8 on FEC007 at mean SRE 8.05 % /
  p95 22.65 %, against Table V's 9.40 / 27.63 % and the organizers' 27.00 %. Which C model
  file and warm-up handling did each evaluation use?
- **K26 parameter counting:** K26 counts the GRU8 as 336 parameters (my inference: PyTorch
  bias convention, normalization excluded) and its own circuit as 32, excluding the 33-parameter
  warm-up estimator. Under the MC2 rule ("material-specific parameters that need to be kept"),
  is the circuit 32 or 65? Is RHINO-MAG 320, 325 or 336?
- **K26 maturity:** it is an arXiv preprint (1 Nov 2026), not peer-reviewed, with an
  acknowledged heavy LLM contribution. The extrapolation comparison (Table V) uses a GRU
  replica that K26 itself calls weaker than the published model.
