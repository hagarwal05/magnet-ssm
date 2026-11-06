# SSMs, system identification, and rollout training: literature notes

Author: ssm-lit (literature-review team). Written 2026-11-06.
Scope: the ML literature that supports or challenges our S4D design and training recipe
(`src/models/ssm.py`, `src/train.py`, `docs/RESEARCH_LOG.md` §4–5).

Tags:
- **[abstract only]**: I read only the abstract or metadata, not the full text.
- **[inferred]**: my reasoning, not a claim made by the paper.
- **[computed]**: arithmetic from our code defaults or from numbers in the source.

Full texts were read from arXiv HTML for every entry not tagged [abstract only]. Venues were
checked against the arXiv comment field, Crossref, or OpenReview/dblp; where I could not
verify a venue I give only the arXiv id.

## 0. Our design in the terms this literature uses

These facts are from our code and log. They are what the papers below are compared against.

- **Mode.** Our S4D is a linear time-invariant (LTI) diagonal SSM. We apply it in
  **convolution mode** to an 80-step window of (B, H) and recompute it from scratch at every
  prediction step. That is equivalent to a recurrent SSM whose state is **reset to zero 80
  steps back at every step** [inferred]. Memory is therefore hard-limited to 80 samples.
- **Dynamic-model class.** H enters as an input, and the prediction is fed back into the H
  window. In system-identification terms this is a NARX / **series-parallel** model during
  stage 1 (measured H as input) and a **parallel** (output-error) model over K = 32 steps
  during stage 2 (Ribeiro & Aguirre 2018, §3).
- **Timescales.** At init, Re(A) = −1/2 and Δ is log-uniform in [0.01, 0.1] (one unit = one
  16 MHz sample). The per-step decay is exp(−Δ/2), between 0.951 and 0.995, which gives
  e-folding lengths of **20–200 steps** [computed]. The longest modes are cut off by the
  80-step window anyway.
- **The task's timescales.** Excitation runs at 50–800 kHz at 16 MHz, so one period is
  **20–320 samples**. The 80-step window covers less than one period below 200 kHz. The
  stage-2 horizon K = 32 is shorter than one period in 5 of the 7 buckets (50–320 kHz). The test
  horizon is up to **900 steps** after a warm-up of 100, 500 or 900 steps (C: 1 step)
  [computed].
- **Parameterization.** S4D-Lin init (A_n = −1/2 + iπn). ZOH discretization. Re(A) is
  constrained negative via −exp(·), so every discrete |λ| < 1 for any parameter value
  [verified in code]. B is fixed to 1 (not trained). C is complex N(0, 1).
- **MC2 gives the whole upcoming B(t)** (task statement), but our model only reads B over the
  window plus B_next.

---

## 1. SSM foundations

### 1.1 Gu, Goel, Ré. "Efficiently Modeling Long Sequences with Structured State Spaces" (S4). ICLR 2022 (Outstanding Paper HM). arXiv:2111.00396.
- An SSM has three equivalent views: continuous, recurrent, and convolutional (§2.3–2.4).
  Training uses the convolutional view. Generation switches to the recurrent view, where
  "S4 requires constant memory and computation per time step". That gave ~60× faster
  generation than a Transformer on CIFAR-10 and WikiText-103 (§4.3).
- **Forecasting is done without autoregression.** "We simply concatenate the entire context
  with a sequence of masks set to the length of the forecast window" and take the last F
  outputs (App. D.3.5, Fig. 11). This "outperforms the Informer and other baselines on 40/50
  settings" and "is better on the longest setting in each task" (e.g. −37 % MSE on 30-day
  weather).
- S4 adapts to a changed sampling rate without retraining (0.5× frequency on speech) by
  rescaling Δ.

**Implication for our project.** Two levers we do not use:
1. The recurrent view, which carries state over the whole 1,000-step sequence at O(1) cost
   per step.
2. S4's own forecasting recipe. Feed the full B sequence, plus H with a "known" mask over the
   warm-up, and predict all future H in one causal convolution pass. With no feedback of
   predictions there is **no exposure bias to fix** [inferred].

### 1.2 Gu, Gupta, Goel, Ré. "On the Parameterization and Initialization of Diagonal State Space Models" (S4D). NeurIPS 2022 (Crossref DOI 10.52202/068431-2607). arXiv:2206.11893.
- Initialization of A is critical. Scaling the imaginary parts by 0.01 or 100, or randomizing
  them, "substantially" or "minorly" degrades results (§5.2).
- Constrain Re(A) < 0: "This ensures stability even in unbounded autoregressive settings"
  (§3.4, "Eigenvalue constraint"). Leaving it unconstrained "may be slightly better", but the
  difference was negligible.
- **The kernel "depends only on the elementwise product B∘C"**, which is "equivalent to
  keeping independent B and C, and simply freezing B = 1 while training C" (§3.3).
- Training B is "consistently slightly better" in the ablations (§5.1), which used
  ~100K-parameter models (4 layers, H = 128). ZOH and bilinear make no noticeable difference.

**Implication for our project.** Our parameterization matches the recommended one, with B
frozen at 1. Because we train C per mode, a trainable B adds **no expressivity** (S4D §3.3).
In recurrent mode it would be only a diagonal rescaling of the state [inferred]. It is at most
an optimization tweak, at +32 params, i.e. +16 % of 205 (2·(N/2)·H per layer, N = 8, H = 4)
[computed]. The "slightly better" result comes from models 100–500× larger than ours.
**Low value; not recommended.**

### 1.3 Gupta, Gu, Berant. "Diagonal State Spaces are as Effective as Structured State Spaces" (DSS). NeurIPS 2022 (dblp via OpenReview). arXiv:2203.14343.
- A purely diagonal state matrix, without S4's low-rank correction, matches S4 on Long Range
  Arena and Speech Commands. The updated version adds an "RNN view for autoregressive
  decoding".

**Implication for our project.** It supports the diagonal choice; nothing further to act on.

### 1.4 Smith, Warrington, Linderman. "Simplified State Space Layers for Sequence Modeling" (S5). ICLR 2023 (dblp via OpenReview). arXiv:2208.04933.
- S5 uses one MIMO diagonal SSM computed with a **parallel scan**: "a purely recurrent,
  time-domain approach" (Conclusion). It can be "used as a recurrence in the time domain for
  both online generation and offline processing" (§3.3).
- The scan supports **time-varying SSMs and irregular sampling** "by simply supplying a
  different Ā_k at each step". The convolutional S4 cannot do this, because it requires an
  LTI system and evenly spaced samples (§3.3). S5 beat CRU on irregularly sampled pendulum
  regression (§6.3).
- Learning one timescale per state (P timescales) improved performance (§4, App. E.1).

**Implication for our project.** A parallel scan is the way to train a carried-state SSM on
long chunks with no Python loop over time. This works only when the inputs are known in
advance (B, not fed-back H) [inferred]. Per-state Δ is a cheap extension.

### 1.5 Gu, Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." COLM 2024 (OpenReview). arXiv:2312.00752.
- Selection makes Δ, B and C depend on the input. The result is time-varying and computed in
  recurrent mode with a hardware-aware scan. Δ "plays a generalized role of the RNN gating
  mechanism" (Thm. 1, §3.5).
- **"No free lunch."** On uniformly sampled smooth signals selection can hurt. On long-form
  audio waveforms "it in fact significantly hampers performance", because audio "benefits
  from continuous linear time-invariant (LTI) methods". LTI layers near the raw signal were
  best (§5, App. E.4, Fig. 18).
- Complex-valued SSMs are "helpful for continuous modalities (e.g. audio, video)" (§3.6).
  Special initializations "can help in several settings such as low-data regimes".

**Implication for our project.** B(t)/H(t) at 16 MHz is a uniformly sampled continuous signal,
so Mamba's own evidence favors keeping LTI, complex S4D layers near the input [inferred].
Hysteresis switching at reversal points is input-dependent, though, so a selective or gated
layer deeper in the stack could still help. That is untested [inferred].

### 1.6 Orvieto, Smith, Gu, Fernando, Gulcehre, Pascanu, De. "Resurrecting Recurrent Neural Networks for Long Sequences" (LRU). ICML 2023 (OpenReview, oral). arXiv:2303.06349.
- Linear recurrences interleaved with nonlinear MLPs recover deep-SSM performance; removing
  the nonlinearity from the recurrence gives "a substantial boost" (§3.1).
- A **stable exponential parameterization**, λ = exp(−exp(ν_log) + iθ), plus initializing
  eigenvalues on a ring close to the unit circle, is what enables long-range memory. "It is
  the eigenvalue distribution of the recurrent layer at initialization that determines if
  the model can capture long-range reasoning" (§1, §3.3).
  - Without stability enforced, sCIFAR degraded for r_max > 0.9.
  - With stability and γ normalization, r_min = 0.9, r_max = 0.999 was best (App. C.2).
- In S4, the Δ in front of B acts as forward-pass **normalization**: states near |λ| = 1
  blow up as 1/(1 − |λ|²) otherwise (§3.4, §4).

**Implication for our project.** If we carry state across 900 steps, the eigenvalues must
reach |λ| ≈ 1 at init. Retaining e⁻¹ of a signal over one 320-sample period needs |λ| ≈
exp(−1/320) ≈ 0.9969, i.e. Δ ≈ 0.006 at Re(A) = −1/2. That is below our Δ_min = 0.01
[computed]. Keep the ZOH Δ·B scaling, or an equivalent γ, as normalization.

### 1.7 Gu, Johnson, Timalsina, Rudra, Ré. "How to Train Your HiPPO: State Space Models with Generalized Orthogonal Basis Projections." ICLR 2023 (dblp via OpenReview). arXiv:2206.12037.
- "1/Δ exactly represents the range of dependencies" an S4 channel captures (§3.4
  "Discussion", Timescales box). S4's default Δ ∈ (0.001, 0.1) covers expected dependency
  lengths of 10 to 1,000 (§3.3).
- On Path-X, lowering Δ_min to match the known dependency length made results consistent.
  Spreading Δ over a wider range was best (§4.1, Fig. 4).

**Implication for our project.** Set Δ from the physics: one period is 20–320 samples and the
horizon is 900. A range of about [0.001, 0.1] (lengths ≈ 20–2,000 at Re(A) = −1/2) fits a
carried-state model. Our [0.01, 0.1] is matched to the 80-step window only [computed].

### 1.8 Goel, Gu, Donahue, Ré. "It's Raw! Audio Generation with State-Space Models" (SaShiMi). ICML 2022, PMLR 162 (verified by reviewer on proceedings.mlr.press/v162/goel22a). arXiv:2202.09729.
- "S4's recurrent representation cannot be used for autoregressive generation due to
  numerical instability." Trained S4 matrices "generally became non-Hurwitz".
- The issue "only arises when using S4 during autoregressive generation, because S4's
  convolutional mode during training does not involve powering up Ā" (§4.1, "Stabilizing S4
  for Recurrence").
- The fix is a Hurwitz parameterization A = Λ − pp* with Re(Λ) < 0. The paper only suggests
  constraining Re(Λ), "e.g. run its entries through an exp function". In its own runs Λ was
  trained freely, which the authors found stable but slightly worse (App.).

**Implication for our project.** This is the main risk of switching from convolution to
recurrent mode, and our code is already safe from it: Re(A) = −exp(·) < 0, so |λ| < 1 for
any parameter value [verified in code]. The finite-window convolution would have hidden
instability; a recurrent mode would expose it.

### 1.9 Two theory results (supporting)
- **Orvieto, De, Gulcehre, Pascanu, Smith. "Universality of Linear Recurrences Followed by
  Non-linear Projections."** ICML 2024 (OpenReview). arXiv:2307.11888. [abstract only]
  Linear diagonal recurrences plus MLPs approximate regular causal seq2seq maps arbitrarily
  well. Complex eigenvalues near the unit disk "greatly help the RNN in storing information".
- **Ran-Milo et al. "Provable Benefits of Complex Parameterizations for Structured State
  Space Models."** NeurIPS 2024. arXiv:2410.14067. [abstract only] Real diagonal SSMs need
  much larger dimension, or exponentially large parameters, to express what complex ones can.

**Implication for our project.** These support keeping complex modes in small models; no
change needed.

---

## 2. SSMs, RNNs and neural ODEs for system identification and hysteresis

### 2.1 Bonassi, Andersson, Mattsson, Schön. "Structured state-space models are deep Wiener models." IFAC-PapersOnLine 2024 (SYSID), DOI 10.1016/j.ifacol.2024.08.536. arXiv:2312.06211.
- A deep SSM (linear SSM layers + static nonlinearities) is a **deep Wiener model**. With
  Schur-stable layers the whole model is incrementally input-to-state stable (δISS)
  (Prop. 2).
- **Silverbox benchmark** (Table 1), models trained by **free-run simulation-error
  minimization** over subsequences (App. C; 684 training subsequences, validation
  subsequences of T = 512; the training subsequence length is not stated in the text I
  read):
  - S4/S5/S5R/LRU (L = 4, n_λ = 10): RMSE 0.73–0.81 mV on the first 25,000 steps; 3.56–4.73
    mV over the full set including extrapolation.
  - LSTM (Andersson et al.): 0.31 / 4.0; TCNN: 0.75 / 4.9.
  - So the SSMs are "in line with" TCN and LSTM "while featuring significantly less learnable
    parameters".
  - The comparison cuts both ways. The LSTM is better in interpolation (0.31 vs ≥ 0.73). S5
    beats the LSTM on the full set, which includes extrapolation (3.56 vs 4.0).
- Listed open problems: minimal structures, data-driven initialization (e.g. from a best
  linear approximation), and benchmarks with long-term dependencies.

**Implication for our project.** This is the closest prior use of S4/S5/LRU for physical
system identification, and it trains them as **simulators over long free-run horizons**, not
teacher-forced next-step predictors. "SSM ≈ LSTM at fewer params" is consistent with our 205
vs 771 result [inferred]. The comparison is not like-for-like: our LSTM was trained with our
recipe, theirs with Andersson et al.'s. The split result (LSTM ahead on interpolation, S5
ahead including extrapolation) cautions against claiming SSMs are generally superior or
inferior.

### 2.2 Forgione, Mejari, Piga. "Learning neural state-space models: do we need a state estimator?" arXiv:2206.12928 (2022; venue not verified).
- Training splits data into subsequences, each with an estimation part (m_e) for the initial
  state and a fitting part (m_f).
- For the asymptotically stable Wiener–Hammerstein circuit, zero or random initial states are
  competitive if m_e is long enough. A ZERO-initialized model needs ~40 steps (WH) to ~100
  steps (pick-and-place) to converge to the true output.
- For the pick-and-place machine, which has an integrator, a learned FF or LSTM estimator is
  required (§3).
- The longest fitting lengths (m_f = 320 / 512) hurt only under short training budgets
  (300 s), because each iteration costs more. With longer training the effect was "almost
  negligible" (§3.1–3.2).

**Implication for our project.** Hysteresis has persistent memory, like an integrator rather
than a fading-memory system [inferred]. A carried-state model therefore needs an explicit
state estimator or warm-up from the given past. RHINO-MAG reaches the same conclusion: zero
init is poor. Long fitting horizons cost compute, not accuracy.

### 2.3 Beintema, Schoukens, Tóth. "Deep subspace encoders for nonlinear system identification" (SUBNET). Automatica 2023, DOI 10.1016/j.automatica.2023.111210. arXiv:2210.14816.
- The structure is: an encoder maps the past n inputs and outputs to the initial state; the
  state is then simulated forward T steps, and the loss is averaged over many overlapping
  truncated sections.
- **Guideline: "Choose T to be a few times the largest characteristic time scale for stable
  data-generating systems"** (§3.4). "Overfitting is a significant issue when T is smaller
  than the dominant timescale" (§5.4).
- Loss smoothness degrades exponentially with T for non-contractive models,
  L = O(L_f^{2T}) (Thm. 5). Truncation keeps optimization stable. The encoder lag n should
  exceed the minimal reconstruction order; larger n reduces the variance of the state
  estimate.

**Implication for our project.** SUBNET is the template for a hybrid design:
1. Use our 80-step S4D window as the **encoder** to produce an initial state.
2. Propagate a recurrent SSM forward driven by B.
3. Train on truncated sections T of a few periods.

Our K = 32 is shorter than one period for 5 of the 7 frequency buckets (50–320 kHz, i.e.
50–320 samples). It equals the 500 kHz period (32) and exceeds only the 800 kHz one (20)
[computed]. Their guideline would call that too short. It is stated for stable systems,
though, and hysteresis is not fading-memory (§2.2, §2.4), so applying it to MC2 is an
inference [inferred].

### 2.4 Ribeiro, Tiels, Umenberger, Schön, Aguirre. "On the smoothness of nonlinear system identification." Automatica 121:109158, 2020, DOI 10.1016/j.automatica.2020.109158. arXiv:1905.00820.
- Where the model is not contractive, the Lipschitz constant and β-smoothness of the
  simulation-error loss "might blow up exponentially with the simulation length" (Thm. 1).
- **Multiple shooting** makes the simulation length a design parameter and recovers a
  well-behaved objective (§4).
- A comparison with multi-step-ahead prediction-error minimization is included (§6).

**Implication for our project.** A contractive system has a unique fixed point and forgets its
initial state. Hysteresis does not (remanence), so long-horizon losses for our problem are
expected to be rough [inferred]. That argues for growing the horizon gradually (curriculum)
or for multiple shooting, rather than jumping to 900 steps.

### 2.5 Ribeiro, Aguirre. "'Parallel Training Considered Harmful?': Comparing series-parallel and parallel feedforward network training." Neurocomputing 316:222–231, 2018, DOI 10.1016/j.neucom.2018.07.071. arXiv:1706.07119.
- In series-parallel (one-step) training the measured output is fed back. In parallel
  (simulation) training the model's own output is fed back.
- Parallel training is "generally better in more realistic scenarios" (output noise, measured
  data). The computational cost is similar.
- They also note that series-parallel training "has two real advantages over parallel
  training: i) it seems less likely to be trapped in 'poor' local solutions; ii) it is more
  amenable to parallelization" (Conclusion).

**Implication for our project.** This is the sysid-literature name and evidence for our
stage 1 → stage 2 change. Their caveat supports keeping series-parallel stage 1 as an
**initializer**. Parallel training as the main stage, rather than a 12-epoch fine-tune, is
consistent with their findings [inferred].

### 2.6 Gedon, Wahlström, Schön, Ljung. "Deep State Space Models for Nonlinear System Identification." IFAC-PapersOnLine 2021, DOI 10.1016/j.ifacol.2021.08.406. arXiv:2003.14162. [abstract only]
- Six stochastic deep SSMs (VAE-RNN-type latent-variable models) evaluated on sysid
  benchmarks.

**Implication for our project.** Terminology only: "deep SSM" in sysid often means
probabilistic latent RNNs, not S4-type layers. The paper should say "structured SSM (S4D)".

### 2.7 Andersson, Ribeiro, Tiels, Wahlström, Schön. "Deep Convolutional Networks in System Identification." CDC 2019. arXiv:1909.01730. [abstract only]
- TCNs connect to Volterra series and block-oriented models; tested on Silverbox and F-16.

**Implication for our project.** A finite-window convolution is a finite-memory (Volterra/TCN-
like) model [inferred]. Our windowed S4D belongs to this class, not to the infinite-memory SSM
class that S4 was designed for.

### 2.8 Hysteresis-specific neural models
- **Zhu, Huang, Li, Li, Lei, Zhu. "A Physics-Informed Hybrid Neural Operator for Transient
  Magnetization Prediction in Power Magnetics" (PI-HNO).** arXiv:2608.02965 (2026 preprint;
  intended for IEEE TPEL). **Uses MagNetX transient data, 14 ferrites, material-specific.**
  - **This is team Sydney's MC2 entry** (`other_teams.md` §4: the same 4,777 params plus
    their talk). Its official MC2 score is 24.63 % sequence error, rank 9, Performance track
    3rd.
  - Architecture: a history GRU over the last 100 B–H samples sets the initial state of a
    future GRU driven by B and ΔB. A Preisach-inspired 1-layer Transformer reads **the
    complete 1,000-step B(t), future included**. 4,777 params.
  - Mean / p95 sequence error 13.40 / 31.39 %. A size-matched GRU (4,758) gets 18.44 / 46.17;
    the MC2-demo LSTM (2,399) gets 106.31 / 282.36 (Table III).
  - Ablations (Table IV, mean / p95 seq.):
    - no global B branch: 17.77 / 42.19;
    - A4, no *future GRU* (no B-driven state propagation, i.e. a direct B→H map):
      19.01 / 43.65;
    - no ΔB input: mean seq 17.53, but the largest energy errors;
    - T and position fed stepwise instead of into the initial state: 24.83 / 75.68.
  - Training: 200-step segments, 100 history + 100 target (§IV-C).
  - Caveat: the 13.40 / 31.39 is pooled over the paper's own test pool, not the official
    per-block metric. Against the official 24.63 % for the same model family, it is not
    leaderboard-comparable.
  - **Implication for our project.** This is independent evidence on our data family for
    (a) a carried recurrent state along B, (b) ΔB as an input, (c) conditioning the initial
    state on T, and (d) using the *future* B trajectory. MC2 supplies future B and our model
    ignores it [inferred].
- **Chandra, Daniels, Curti, Tiels, Lomonova. "Magnetic Hysteresis Modeling With Neural
  Operators."** IEEE Trans. Magn. (Crossref: 2025), DOI 10.1109/TMAG.2024.3496695.
  arXiv:2407.03261.
  - Data are **generated by a Preisach-based model** of NO27-1450H under quasi-DC excitation
    (§IV).
  - FNO is best on unseen first-order reversal curves (FORCs), relative L2 = 1.34e-3. RNN,
    LSTM, GRU and an encoder-decoder LSTM "cannot capture the hysteresis and predict for
    novel input H fields" (Table IV).
  - A rate-independent FNO (no time channel) generalizes across sampling rates
    (1/100×–100×).
  - **Implication for our project.** This is a warning that recurrent models can fit loops
    yet fail on unseen minor loops and FORCs, which bears on RQ2. The evidence is static and
    synthetic, so it does not transfer directly to dynamic MC2 data [inferred].
- **Chandra, Kapoor, Daniels, Curti, Tiels, Tartakovsky, Lomonova. "Neural oscillators for
  magnetic hysteresis modeling" (HystRNN).** arXiv:2308.12002. Published as "Generalizable
  models of magnetic hysteresis via physics-aware recurrent neural networks", Comput. Phys.
  Commun. 314, 109650 (2025), DOI 10.1016/j.cpc.2025.109650 (see Appendix A).
  - An ODE-based RNN in the spirit of coRNN, with Bouc-Wen/Duhem-style |·| terms in the
    state update. It generalizes to FORCs and minor loops where RNN, LSTM and GRU did not.
  - **Implication for our project.** A hysteresis-shaped nonlinearity inside the recurrence
    is one route to RQ2 generalization. RHINO-MAG found physics-structured GRU variants
    worse on MC2, though (see `siegen_paderborn.md` §5).
- **Guo, Nguyen, Hamzehbahmani, Sabariego.** IEEE Trans. Magn. 2025, DOI
  10.1109/TMAG.2025.3600089, arXiv:2504.04863 [abstract only]: FNO, U-FNO and DeepONet for
  dynamic hysteresis of GO steel. Cyclic-rolling and Gaussian augmentation helped; FNO best.
  **Guo, Shen, Sabariego.** arXiv:2604.04150 (2026) [abstract only]: Res-FNO on MagNet
  materials; **dB/dt "incorporated as a critical physical feature"**.
- **Purnode, Denis, Henrotte, Louppe, Geuzaine.** arXiv:2607.14321 (2026) [abstract only]: an
  RNN surrogate of a laminated-core hysteresis + eddy-current model inside 2-D FE. It is
  trained on "artificially generated magnetic field sequences" designed to be diverse.
- **Hu, Daryakenari, Shen, Kawaguchi, Karniadakis. "State-space models are accurate and
  efficient neural operators for dynamical systems."** arXiv:2409.03231 [abstract only].
  Mamba ranked among the top of 12 models on interpolation and on "strict extrapolation"
  testbeds, at the lowest computational cost. The journal version appears in *Neural
  Networks* (ScienceDirect PII S0893608025013772, from a search result I did not open).
- **Brandoit, Fyon, Ernst, Drion.** ICML 2026 (spotlight). arXiv:2605.11855 [abstract only]:
  recurrent units with bistable, hysteretic quantized states give "persistent memory" and
  match LRU/minGRU at small sizes. This is the reverse direction (hysteresis as a memory
  mechanism), but it is relevant to how a small model can hold a remanent state.

### 2.8b Link between linear circuit material models and S4D (for background-lit's Guillod §4.1)
- **Citable sources:**
  - S4D: "The diagonal structure allows it to be viewed as a collection of 1-dimensional
    SSMs" (S4D, Fig. 1 caption). The kernel is K(t) = Σ_n C_n e^{tA_n} B_n, a sum of N
    damped exponentials/oscillators (S4D §2, eq. 3).
  - S4 §2.3: each SSM is a continuous-time linear ODE x' = Ax + Bu, y = Cx + Du, discretized
    by step Δ (bilinear in S4, ZOH in S4D §3.1).
  - Bonassi et al. 2024 §4: SSM layers are "discretized continuous-time" LTI systems. This
    lets them be simulated "with a sampling time potentially different from that of the
    training data".
- **The mapping [inferred].** Guillod's linear material model, dB/dt = r·H + μ0μr·dH/dt,
  rearranges to dH/dt = (dB/dt − r·H)/(μ0μr). That is a first-order LTI ODE with input dB/dt
  and state = output H: one real pole at −r/(μ0μr). One S4D mode with a real eigenvalue,
  driven by dB/dt, is the discretized version of the same equation. S4D-Lin adds complex
  (oscillatory) poles. A deep S4D stack is a series of such linear ODE banks with static
  nonlinearities in between, i.e. a deep Wiener model (Bonassi 2024). This reading is my
  inference from the cited equations; none of these papers discusses magnetic circuit models.

### 2.9 Neural CDE and Koopman (context)
- **Kidger, Morrill, Foster, Lyons. "Neural Controlled Differential Equations for Irregular
  Time Series."** NeurIPS 2020 (spotlight). arXiv:2005.08926.
  - A CDE is driven by an input path. It is "blind to speed at which X is traversed", so time
    must be added as a channel to break this reparameterization invariance (App. on "Time
    as a channel").
  - The initial state must depend on X₀, otherwise the model is translation-invariant.
  - **Implication for our project.** Ideal (rate-independent) hysteresis has exactly this
    invariance, so a CDE driven by B is natively rate-independent. Rate-dependent effects
    (eddy currents, relaxation) are then added via time or dB/dt channels [inferred]. This is
    a principled framing for RQ2. Neural ODEs: Chen et al., arXiv:1806.07366 (metadata only).
- **Lusch, Kutz, Brunton**, Nat. Commun. 2018, DOI 10.1038/s41467-018-07210-0; **Korda,
  Mezić**, Automatica 2018, DOI 10.1016/j.automatica.2018.03.046. [abstract only] Both lift
  nonlinear dynamics into coordinates where they evolve linearly (Koopman).
  - **Implication for our project.** A deep linear SSM with nonlinear mixing is a learned
    lifting of the same spirit [inferred]. I found no Koopman hysteresis model for magnetics
    (arXiv search "Koopman AND hysteresis": no relevant hit).

### 2.10 SSMs in MC2 / magnetics: novelty check
- arXiv API searches with no relevant hits:
  - `mamba AND hysteresis`
  - `mamba AND "core loss"`
  - `"state space" AND "core loss"`
  - `S4 AND hysteresis`
  - `"linear recurrent" AND hysteresis`
  - `"MagNetX"`, whose only hit was PI-HNO, a GRU + Transformer model
- A DDG web search found no S4/S5/LRU/Mamba model for magnetic core loss or B–H prediction.
- RHINO-MAG's code has GRU, LSTM, JA, Preisach and NODE variants, but no SSM.
- Further searches (second pass, 2026-11-06), still no magnetics hit:
  - arXiv: `mamba AND actuator AND hysteresis`; `"selective state space" AND hysteresis`;
    `S5 AND hysteresis`; `"state space model" AND hysteresis AND neural`;
    `"structured state space" AND magnetic`. The last one returns only MRI imaging.
  - Crossref: "Mamba hysteresis modeling"; "state space model S4 hysteresis magnetic core".
- **One SSM-for-hysteresis paper exists, outside magnetics.** Chen, Liu, Feng, Zhan, Ji,
  Wang, Li, "Composite control of a piezoelectric fast steering mirror using a mamba-based
  dual-axis inverse model", *Smart Materials and Structures* 35, 2026 (issued 2026-09),
  DOI 10.1088/1361-665X/aea30f. [abstract only, via Crossref]
  - A Mamba inverse model of a **piezoelectric** actuator with rate-dependent hysteresis,
    used as a feedforward controller with PID feedback.
  - Mean Euclidean RMS tracking error on an asynchronous Lissajous trajectory: 8.63 µrad,
    vs 10.65 for Multi-NARMA-L2 and 36.49 for rate-dependent Prandtl–Ishlinskii.
  - Length-1024 whole-sequence inference: 0.63 ms vs 4.21 ms for an LSTM (RTX 2060). Per
    step in recurrent mode on one CPU thread, Mamba was slower: 45.84 µs vs 30.74 µs.
- A second piezo paper, Wang, Kang, Yang, Liu, "A Multi-Scale Hybrid Expert Deep Learning
  Model for Hysteresis Modeling of Piezoelectric Actuators", EEICAI 2026, DOI
  10.1109/EEICAI68535.2026.11441787, appeared in search. I have no abstract, so I don't know
  whether it uses an SSM.

- **Classical nonlinear state-space hysteresis identification also exists.** These are
  polynomial/structured NL-SS models, not S4-family layers:
  - Noël, Esfahani, Kerschen, Schoukens, "A nonlinear state-space approach to hysteresis
    identification," Mech. Syst. Signal Process. 84 (2017) 171–184, DOI
    10.1016/j.ymssp.2016.08.025, arXiv:1610.09138.
  - Esfahani, Dreesen, Tiels, Noël, Schoukens, "Parameter reduction in nonlinear state-space
    identification of hysteresis," DOI 10.1016/j.ymssp.2017.10.017, arXiv:1705.00178.
  - Krikelis, Pei, van Berkel, Schoukens, "Identification of structured nonlinear
    state–space models for hysteretic systems using neural network hysteresis operators,"
    Measurement 224 (2024) 113966, DOI 10.1016/j.measurement.2023.113966.
  - [metadata only for all three; references from reviewer-ssm, checked on arXiv/Crossref]
- **Closest non-S4 relative on MC2: K26.** Kirchgässner, "A Data-Driven, Gain-Scheduled
  Equivalent Circuit for the B-H Behavior of Power Ferrites," arXiv:2611.02222 (2026). It is
  a difference-equation circuit model at the 62.5 ns step, a play-operator bank plus
  first-order relaxations. 32 params/material (+33 for the warm-up estimator) gives official
  p95 SRE 19.43 % on A–E, against RHINO-MAG's 20.4 % (`siegen_paderborn.md` §7).
- reviewer-ssm independently re-ran 9 arXiv and 5 Crossref queries and also found no
  S4/S5/LRU/Mamba magnetics paper.

**Implication for our project.** "SSMs were barely used for MC2" holds as far as we searched,
but it is an absence-of-evidence statement and should be worded as such. Two broader claims
are **false**:
- "first SSM for hysteresis": a Mamba piezo-hysteresis model exists (Chen et al. 2026);
- "first state-space model for hysteresis": classical NL-SS identification exists (Noël et
  al. 2017 and follow-ups).

Safe wording, always with the qualifier "structured" or "S4-family" and never "state-space
model" alone: "to our knowledge, the first structured-SSM (S4-family) model for *magnetic*
B–H/H(t) prediction, and the first in the MagNet Challenge". Cite the piezo paper, the
classical NL-SS work and K26 as related work.

---

## 3. Exposure bias and autoregressive drift

### 3.1 Bengio, Vinyals, Jaitly, Shazeer. "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks." NeurIPS 2015. arXiv:1506.03099.
- Per token, the model uses the true previous token with probability ε_i, otherwise its own
  sample. ε_i decays linearly, exponentially, or by inverse sigmoid (§2.4).
- Training on "Always Sampling" from the start "yielded very poor performance".
- "Uniform Scheduled Sampling" is a regularization control: the model's sample is replaced
  by a token drawn uniformly at random, "to verify that it was important to build on the
  current model". It was "better than the baseline, but not as good as our proposed approach"
  (§4.1).
- Flipping the coin once per sequence instead of per token was "as poor as the Always
  Sampling approach". Gradients are not backpropagated through the sampling decision.
- Critiques: Huszár, arXiv:1511.05101 [abstract only], shows the objective is "improper and
  leads to an inconsistent learning algorithm". Professor Forcing (Lamb et al., NIPS 2016,
  arXiv:1610.09038) [abstract only] instead matches training and sampling dynamics
  adversarially.

**Implication for our project.** Free-running from a random init is bad, which is why
curricula exist. Our 1 → K ramp is a coarse version of this. Scheduled sampling itself has
weak theory and is not the first thing to try for a regression rollout [inferred].

### 3.2 Ross, Gordon, Bagnell. "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning" (DAgger). AISTATS 2011. arXiv:1011.0686.
- A learner with one-step error ε under the expert's distribution can incur up to **T²ε**
  over T steps under its own induced distribution; the bound is tight (§2). DAgger
  (aggregating states visited by the learner) gives linear growth.

**Implication for our project.** This is the theory behind phase 1. A 900-step horizon makes
the quadratic term dominant. That is consistent with phase 1: next-step training MSE of 1e-4
to 6e-3 (normalized) coexisted with official rollout errors of 77–988 % per material
(RESEARCH_LOG §5.2) [inferred]. A DAgger analogue is cheap here, because ground
truth H is known at every visited time step: roll out, then train on the model's own visited
histories with true targets. Stage 2 already does this implicitly.

### 3.3 Laskey, Lee, Fox, Dragan, Goldberg. "DART: Noise Injection for Robust Imitation Learning." CoRL 2017. arXiv:1703.09327.
- Noise injected at data collection lets the learner see recovery examples. The **noise level
  is optimized to approximate the error of the trained policy**: "the noise should
  approximate the error of the trained robot's policy". For Gaussian noise there is a closed
  form via tr(Σ) (§4).

**Implication for our project.** σ should be set to the measured rollout error of the
fed-back H history, not swept blindly. We can read it off the validation rollouts per
material, and possibly per warm-up block [inferred]. Our σ = 0.1 was the top of the range we
tried (RESEARCH_LOG §5.4).

### 3.4 Sanchez-Gonzalez, Godwin, Pfaff, Ying, Leskovec, Battaglia. "Learning to Simulate Complex Physics with Graph Networks" (GNS). ICML 2020. arXiv:2002.09405.
- Trained only on one-step targets. The "main determinants of long-term performance were …
  mitigating the accumulation of error by corrupting the training data with noise".
  Rollouts stayed plausible over thousands of steps.
- **Noise type matters.** Of only_last, correlated, uncorrelated and **random_walk**, "We
  found the best rollout performance for random_walk noise type". Rollout accuracy peaks at
  an intermediate noise scale, while one-step accuracy falls with noise (§4.3, App. C.2).
- An **anecdotal** remark, with no experiment reported: short rollouts during training (e.g.
  two-step) are costlier and "in our experience may not generalize to longer sequences as
  well as noise injection" (App. B).
- Checkpoints were selected on full-length rollout MSE on held-out trajectories.

**Implication for our project.** Our noise is i.i.d. per sample, which is GNS's "uncorrelated"
mode. Real rollout error in H is smooth and drifting, so **random-walk or low-pass noise** is
the better-matched corruption [inferred].

There is no conflict with our results. GNS remarks, anecdotally, that noise beats short
rollouts *instead of* noise. Our §5.4 shows that *adding* K-step rollout training on top of
noise helps. We never tested rollout training without noise. The two findings are
complementary.

### 3.5 Brandstetter, Worrall, Welling. "Message Passing Neural PDE Solvers." ICLR 2022 (spotlight). arXiv:2202.03376.
- **Pushforward trick**: unroll 2 steps but backpropagate only through the last. "We found it
  important not to backpropagate through the first unroll step". This was "far higher
  survival times" than without; full gradients were worse (Fig. 5b).
- **Gaussian input noise (σ = 0.01) "appears worse than using none"** on their E1 task: it
  improves stability but lowers accuracy (§4.2).
- **Temporal bundling**: predicting K future steps per call cuts the number of distribution
  shifts by K.

**Implication for our project.** Temporal bundling looks natural for MC2, because future B is
known: predict, say, 8–16 future H per call [inferred]. Their noise result disagrees with GNS
and with ours, so noise benefits are task-dependent [inferred].

### 3.6 Lippe, Veeling, Perdikaris, Turner, Brandstetter. "PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers." NeurIPS 2023 (spotlight). arXiv:2308.05732.
- The pushforward trick "does not improve performance in our setting, confirming previous
  results".
- **More history inputs "improves the one-step prediction but worsens rollout
  performance"**: differences of past inputs correlate with the target residual, which
  pushes the model toward unstable higher-order differencing (§4.1, App. E.4).
- MSE training neglects low-amplitude spectral components, which then grow over long
  rollouts.

**Implication for our project.** This is a caution against lengthening the fed-back **H**
window as a fix. It agrees with RHINO-MAG's ablation that a long H history adds little once
the state is anchored. Lengthen the **B** context or carry state instead [inferred].

### 3.7 List, Chen, Bali, Thuerey. "Differentiability in unrolled training of neural physics simulators on transient dynamics." Computer Methods in Applied Mechanics and Engineering 2025, DOI 10.1016/j.cma.2024.117441. arXiv:2402.12971.
- Three training modes compared: one-step (ONE), unrolled without temporal gradients (NOG,
  like pushforward), and unrolled with full gradients (WIG).
- **Correction setups** (network corrects a numerical solver): NOG reduces error 33 % on
  average versus ONE; WIG is best, with a 92 % average improvement over ONE (§4).
- **Prediction setups** (network alone steps the state, as in our model), main-text §4.2
  "Varied Learning Tasks", Fig. 14:
  - "NOG training achieves improvements of 41%, and WIG improves on this by a further 30%".
  - "the NOG setup performs better than both alternatives for smaller network sizes".
- Differences between modalities are statistically significant for smaller networks and
  shrink for heavily overparameterized ones (App.).
- "Curriculums are necessary when training with long unrolling horizons" (§2.6). The
  curricula used are m = [1, 2, 4] (§3.3, Kolmogorov) and m = 1 → 4 → 16 (§3.2, wake flow).
  Long WIG chains can have exploding gradients on chaotic targets.
- Network sizes: ~0.5k–1M+ params. The smallest KS networks (GCN 511/518, CNN 481/481, plus
  ~2k variants; Table 2) are the same order as our 205–1,001 and appear in the Fig. 14
  prediction runs.

- **Per-size KS prediction results are mixed** (Tables 10–11, L_rel, ONE / NOG / WIG):

  | config | ONE | NOG | WIG | best |
  |---|---|---|---|---|
  | GCN 0.5k* | 1.85 | 3.06 | 2.60 | ONE |
  | GCN 0.5k | 5.96 | 1.34 | 2.96 | NOG |
  | GCN 2k | 3.30 | 0.81 | 1.02 | NOG |
  | CNN 0.5k* | 3.49 | 1.54 | 2.16 | NOG |
  | CNN 0.5k | 1.52 | 1.28 | 1.22 | NOG ≈ WIG |
  | CNN 2k | 0.59 | 0.503 | 0.507 | NOG ≈ WIG |

- Systems: KS and Kolmogorov flow are chaotic; the wake flow is periodic vortex shedding
  (characteristic time from the shedding frequency, §3.2).

**Implication for our project.** Unrolled training beats one-step training in both setups on
average. The paper summarizes NOG (pushforward-style, no gradient through the fed-back
prediction) as best at small sizes. Per size it is mixed:
- NOG is clearly best in 3 of the 6 sub-2k KS configurations and ≈ WIG in 2.
- In the sixth (GCN 0.5k*), one-step training is best and NOG is ~18 % worse than WIG
  [computed].

So our stage 2, which is WIG, should be compared against a NOG variant rather than assuming
either is better [inferred]. None of their systems is hysteretic. Lengthen K with a staged
curriculum either way.

### 3.8 Hess, Monfared, Brenner, Durstewitz. "Generalized Teacher Forcing for Learning Chaotic Dynamics." ICML 2023, PMLR 202:13017–13049. arXiv:2306.04406.
- During training, the state is replaced by a **convex combination of the model state and a
  data-inferred state**, z̃ = (1 − α)z + α·ẑ. α = 0 is BPTT; α = 1 cuts all gradients. A
  suitable α gives bounded gradients.
- α is annealed from 1 toward 0 based on the Jacobian norms (App., Alg. 1).

**Implication for our project.** GTF is a continuous version of RHINO-MAG's H injection. In
our stage 2 we could feed back α·H_true + (1 − α)·Ĥ with α annealed 1 → 0. That gives a
smooth curriculum from teacher forcing to free running at any K [inferred].

### 3.9 Further gradient-horizon results (supporting)
- **Mikhaeil, Monfared, Durstewitz. "On the Difficulty of Learning Chaotic Dynamics with
  RNNs."** NeurIPS 2022, DOI 10.52202/068431-0821. arXiv:2110.07238 [abstract only].
  Gradients are bounded for RNNs with stable equilibria or cycles, and always diverge for
  chaotic ones.
  - **Implication for our project.** A periodically driven hysteretic core settles to a
    cycle, so long-horizon BPTT should be tractable, unlike chaotic systems [inferred].
- **Tallec, Ollivier. "Unbiasing Truncated Backpropagation Through Time."** arXiv:1705.08209
  [abstract only]. Truncated BPTT is biased toward short-term dependencies.
- **Wang. "LongSSM."** arXiv:2406.02080 [abstract only]. SSMs trained with zero
  hidden-state initialization extend poorly to longer sequences. Changing the
  hidden-state initialization lets models train on short contexts and generalize to long
  ones.
  - **Implication for our project.** If we move to a carried state trained on 128–512-step
    chunks and tested on 900, initialize chunk states from a warm-up or encoder, never from
    zero.

### 3.10 What horizons the literature trains on (summary)

| work | domain | training horizon | test horizon |
|---|---|---|---|
| GNS (Sanchez-Gonzalez 2020) | particle physics | 1 step + noise | thousands |
| MP-PDE (Brandstetter 2022) | PDEs | 2 (pushforward) | 250–1,000 |
| List et al. 2025 | PDEs | m = 3–4, or curriculum 1 → 4 → 16 | long |
| Forgione 2022 | sysid | m_f 40–512 | full test record |
| Bonassi 2024 | Silverbox | free-run subsequences (validation T = 512) | 40,500 |
| SUBNET (Beintema 2023) | sysid | T = a few × dominant timescale | full test record |
| Guillod, MC2 tutorial 2 (neural ODE; via `background.md` §4.1) | MagNetX N87 | curriculum n_out = 0.1 → 0.2 → 0.5 → 1.0 periods | steady-state period |
| **RHINO-MAG (MC2, 325 params)** | **MC2** | **28 warm-up + 128 (paper) / 156 (code), full BPTT, no teacher forcing** | **up to 900** |
| **ours** | **MC2** | **80 window, 1 step + noise, then K = 32 (ramp 1 → 32, 12 epochs)** | **up to 900** |

RHINO-MAG detail is from teammate notes `siegen_paderborn.md` §1–3 and the paper's Table VI.

---

## 4. Small-data transfer and foundation models (Material A, ~95 sequences)

### 4.1 Wang, Kwon, Li, Elasser, Kang, Zhou, Grigoryan, Chen. "MagNetX: Foundation Neural Network Models for Simulating Power Magnetics in Transient." APEC 2025, pp. 2438–2445, DOI 10.1109/APEC48143.2025.10977420. [abstract only; the TechRxiv PDF is behind a bot wall]
- The model "uses partial memories and the next-state flux density excitation to predict the
  response of the field intensity in the next time step". It is time-domain and
  frequency-independent; an example sequence-to-scalar LSTM is shown.

**Implication for our project.** This is the origin of the next-step formulation that the MC2
demo and our setup inherit [inferred]. "Foundation" here refers to the framework; I found no
cross-material pretraining result in the abstract.

### 4.2 Follow-ups by the same group
- **Wang, Kwon, Li, Guillod, Wouters, Sullivan, Chen. "MagNetX: Data-Driven Time-Domain
  Foundation Models for Power Magnetics in Transient."** IEEE TPEL vol. 41 no. 11, 2026, DOI
  10.1109/TPEL.2026.3685464. [search-snippet abstract only] "Key principles of a time-domain
  foundational model are defined"; an NN-based FM "is implemented to characterize different
  materials".
- **Wang, Kwon, Grigoryan, Li, Guillod, Sullivan, Chen. "Unified Time Domain Foundation Models
  for Hysteretic Passive Components."** COMPEL 2025, DOI 10.1109/COMPEL57166.2025.11121278.
  [title only]

**Implication for our project.** These are the most relevant sources for a multi-material
pretrained model, but I could not read them. Someone with IEEE access should pull the TPEL
paper before the review states what it shows.

### 4.3 Yao, Gong. "Cross-Material Support Transfer for Core-Loss Prediction Under Waveform Covariate Shift" (MIST). arXiv:2609.31659 (2026).
- **Task is MC1 (scalar core loss, single-period B, final A–E)**, not MC2 transient H.
- Setup: one 2,784-param model trained **jointly** on all five materials, with material
  identity injected via **FiLM**, the scarce material's loss reweighted (w_D = 5), and no
  fine-tuning.
- Results on material D, p95 over 5 seeds (Table III):
  - single-material baseline: 20.39 ± 2.03;
  - MIST: 12.38 ± 0.92;
  - MIST+FT: 14.01 ± 1.93 (worse on D). This fine-tunes **A/B/C/E only and leaves D
    untouched** (Table III footnote ∥).
  - sequential transfer (MagLearn: pretrain on the legacy materials, fine-tune): 15.90.
- Fine-tuning D itself is Table IV (seed 0). Head-only gives D p95 14.71 and trapezoidal
  p95 17.1 %; full gives 16.03 and 25.2 %; untouched gives 15.6 % trapezoidal at seed 0.
  The authors attribute the loss to catastrophic forgetting toward D's trapezoid-free
  training set.
- Identity ablations (Table III block d) are at 3,000 epochs. They are compared against
  MIST w/o reweighting at 3,000 epochs, 16.61 ± 1.58, not against 12.38:
  - pooling without identity (1,874 params): 145.5 ± 16.3;
  - constant "dummy" identity at matched 2,784 params (3 seeds): 125.12 ± 33.62;
  - so still an order of magnitude worse.
- The authors also report that a nominal seed is not reproducible: the same seed in a second,
  mathematically equivalent implementation moved D's p95 across 12.20–15.91.

**Implication for our project.** For Material A: train jointly over A–E (and the 10 known
materials) with a learned material embedding through FiLM, upweight A, and try *without*
fine-tuning A. In MIST, fine-tuning the scarce material on its own data hurt. Pooling without
identity is catastrophic. The seed-sensitivity finding supports
our 3-seed reporting.

### 4.4 Other transfer evidence
- **Forgione, Muni, Piga, Gallieri. "On the adaptation of recurrent neural networks for
  system identification."** Automatica 2023, DOI 10.1016/j.automatica.2023.111092.
  arXiv:2201.08660 [abstract only].
  - Adapt a nominal RNN to changed dynamics with an **additive correction learned by
    Jacobian Feature Regression** (linear in the model's parameter-Jacobian features), or a
    recurrent-NTK GP "when only few data points are available".
  - **Implication for our project.** For Material A: pretrain on related ferrites, then fit
    only a linear correction. That is a low-variance alternative to full fine-tuning on ~95
    sequences [inferred].
- **Forgione, Pura, Piga. "From System Models to Class Models: An In-Context Learning
  Paradigm."** IEEE Control Systems Letters 2023, DOI 10.1109/LCSYS.2023.3335036.
  arXiv:2308.13380 [abstract only]. A Transformer meta-model trained on synthetic systems from
  a class predicts a new system from its I/O context. Follow-ups:
  - Piga, Pura, Forgione, arXiv:2312.04083, on adapting the meta-model;
  - Rufolo et al., arXiv:2610.13411 (2026), on long sequences and using the meta-model as a
    synthetic-data generator "when real data are scarce".
  - **Implication for our project.** Synthetic hysteresis (Preisach/JA) pretraining is a
    plausible RQ2 route; Purnode 2026 does the same for an FE surrogate [inferred].
- **RHINO-MAG**: trained Material A (3C92) for 10,000 epochs vs 1,500 for others. It lists
  transfer to new materials with limited data as future work (`siegen_paderborn.md`).
- **PI-HNO**: added low-B_pp training sequences for materials that validated poorly at low
  B_pp (§IV-B). This is targeted data augmentation.
- **Generic time-series foundation models.** Chronos (Ansari et al., arXiv:2403.07815) and
  TimesFM (Das et al., arXiv:2310.10688) [abstract only; venues not verified] are pretrained
  forecasters of a series' own future. MC2 is an exogenous-input I/O problem (B drives H), so
  zero-shot use does not fit the task as posed [inferred].
- **Roschkowski, Cherifi, Gernandt. "Neural Scaling Laws for Learning-based Identification of
  Nonlinear Systems."** arXiv:2512.20447 [abstract only]. Scaling laws in data, compute and
  model size hold for sysid. This is relevant framing for RQ1 (accuracy vs size).
- Also located, abstract only: Yao et al., "Cross-Material Magnetic Core Loss Prediction via
  Operating-Condition Matching and Partial Optimal Transport", arXiv:2610.14964; Gong & Yao,
  "CAHR-Net", arXiv:2609.01991 (FiLM conditioning, MC1, 1,874 params, average p95 6.89 %).

---

## 5. Ranked changes to our S4D model and training

Ranked by expected impact relative to effort. Each item says which evidence is direct (MC2 or
MagNetX), which is from analogous domains, and which is [inferred].

1. **Give the model memory beyond 80 steps and train it the way it is evaluated.**
   Two options.
   - **(a) Keep the AR loop, switch to recurrent mode.** Use a carried S4D state warmed up
     over the full given past (100/500/900 steps), then run free from that state.
   - **(b) The S4-native option: no feedback at all.** Input [B, ΔB, T, H·mask, mask] over the
     whole 1,000-step sequence and predict H at every step. Train in convolution or scan mode
     on full sequences.
   - **Evidence for carried state.** RHINO-MAG (carried GRU state, zero init poor, wins with
     325 params). PI-HNO ablation A4 (removing the propagated state raised mean error 13.40 →
     19.01). SUBNET (encoder + propagated state).
   - **Evidence for (b).** S4's masked-forecasting recipe (App. D.3.5). Bonassi 2024
     (free-run simulation training of S4/S5/LRU).
   - **Why it matters here.** One period is 20–320 samples while our window is 80, so the
     window covers less than one period in the 50–125 kHz buckets [computed]. Option (b)
     removes exposure bias by construction and is fully parallel over time [inferred].
   - **Risks.** Must set Δ_min ≈ 1e-3 (item 4). Must anchor the initial state (LongSSM;
     Forgione 2022). Untested on MC2 for SSMs.
2. **Make long free-running training the main stage, not a short fine-tune.**
   - Change: raise K from 32 to ≥ 128–320 (at least one period at 50 kHz), with a staged
     curriculum (e.g. 1 → 8 → 32 → 128 → 320). Ablate full BPTT (current) against a
     no-gradient-through-feedback (NOG) variant.
   - **Evidence.**
     - RHINO-MAG: 128–156 free-run steps with full BPTT, no teacher forcing, generalizes to
       900.
     - Guillod's MC2 tutorial-2 neural ODE grows the horizon from 0.1 to 1.0 periods
       (`background.md` §4.1).
     - List et al.: unrolled > one-step in prediction setups on average (+41 % NOG, a
       further +30 % WIG). At sub-2k sizes the per-size results are mixed: NOG is clearly
       best in 3 of 6 KS configurations, ≈ WIG in 2, and one-step training wins in 1.
       Curriculum needed.
     - SUBNET: T should be a few × the timescale for stable systems; T below it overfits.
     - Ribeiro 2018: parallel > series-parallel in realistic settings.
     - Our §5.4: stage 2 helped in a controlled test.
   - **Cost.** About 4–10× stage-2 compute at K = 128–320 [computed from K ratio].
   - **Caution.** Losses get rougher with length for non-contractive dynamics (Ribeiro 2020),
     so grow K gradually. Optionally use GTF-style annealed mixing of true and predicted H
     (Hess 2023) as the curriculum.
3. **Match the H-history noise to the real rollout error, and make it temporally
   correlated.**
   - Change: replace i.i.d. N(0, σ²) with random-walk or low-pass noise, with σ set from the
     measured residuals of validation rollouts per material (DART). Also try σ > 0.1, since
     0.1 was the top of our sweep.
   - **Evidence.** GNS (random walk best; intermediate σ best). DART (noise ≈ learner error).
     Our §5.3–5.4 (noise is the single largest gain).
   - **Counter-evidence.** MP-PDE: Gaussian noise worse than none on a PDE task. So treat
     this as an ablation, not an assumed win.
   - Obsolete if item 1(b) is adopted, since nothing is fed back.
4. **Re-initialize Δ for the physics.**
   - Change: Δ ∈ [1e-3, 1e-1], giving dependency lengths ≈ 20–2,000 steps at Re(A) = −1/2
     [computed]. Optionally learn one Δ per state (S5).
   - **Evidence.** HiPPO-training paper (1/Δ = range; set Δ_min to the dependency length).
     LRU (eigenvalues near |λ| = 1 needed for long memory).
   - **Not recommended: a trainable B.** It is redundant with C (S4D §3.3: the kernel depends
     only on B∘C), costs +32 params (+16 % of 205), and the "slightly better" result is from
     ~100K-param models (§1.2).
   - This matters only with a carried state (item 1); inside an 80-step window it is moot
     [inferred].
5. **Add input features and use the future B we are given.**
   - Change: add ΔB and Δ²B as input channels (RHINO-MAG's feature set). For item 1(b), feed
     the whole future B trajectory. Inject T into the initial state rather than only at the
     head (PI-HNO A5).
   - **Evidence.** RHINO-MAG features (Sec. II-A2). PI-HNO ablations A1 (global B context),
     A3 (ΔB), A5 (T via initial state: p95 31.39 → 75.68 when moved). Res-FNO (dB/dt
     "critical").
   - Our head currently sees only [B_next, T] beyond the window.
6. **Material A: train one joint model with a material embedding.**
   - Change: one model over A–E (+ the 10 known materials), with FiLM material identity and
     upweighted A. Compare no-fine-tune vs fine-tune. Alternative: pretrain, then fit an
     additive JFR-style correction on A only.
   - **Evidence.** MIST: the joint FiLM model beat sequential transfer; fine-tuning the
     scarce material on its own data hurt (Table IV); identity-free pooling failed badly.
     This is MC1, not MC2, so it is analogous rather than direct evidence. Forgione 2023
     (JFR adaptation). RHINO-MAG trained A ~7× longer.
   - Also directly serves RQ2.
7. **Temporal bundling (if staying autoregressive).** Predict 8–16 future H per call from the
   known future B. This cuts the number of feedback steps by that factor.
   - **Evidence.** MP-PDE temporal bundling. Untested for hysteresis [inferred].
8. **Keep selecting on full-length validation rollouts; enlarge the validation set.** GNS
   selects on full rollouts as we do. Our §5.2 and §5.4 show selection noise with 200
   segments. This is housekeeping, not a model change.
9. **Low priority: scheduled sampling, pushforward, Mamba-style selection.**
   - Scheduled sampling: improper objective (Huszár).
   - Pushforward: no gain in PDE-Refiner, though it helped in MP-PDE.
   - Selection: hurt on uniformly sampled smooth signals in Mamba's own audio ablation.

**On RQ1, size.** "5× larger S4D bought nothing" (§5.5) is consistent with a
memory/training-regime bottleneck rather than a capacity one [inferred]. The supporting
observations:
- List et al.: training-modality differences are significant for small nets and shrink for
  overparameterized ones.
- RHINO-MAG: monotone but diminishing gains with size, *under closed-loop training*.
- Bonassi: SSMs ≈ LSTM at far fewer params.

Re-test size after items 1–2. If larger S4Ds then help, the original null result was a
training artefact.

---

## Appendix A. Foundational citations (added 2026-11-06 at writer's request)

Each entry was checked against Crossref metadata, the PMLR page, or the arXiv record (method
noted per entry).

- **LSTM.** S. Hochreiter, J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*
  9(8):1735–1780, 1997, DOI 10.1162/neco.1997.9.8.1735 (Crossref).
  - Supports: introduces the gated memory cell that lets RNNs bridge long time lags. This is
    the architecture of the MC2 demo baselines and our 771-param Dual-LSTM.
- **GRU.** K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk,
  Y. Bengio, "Learning Phrase Representations using RNN Encoder–Decoder for Statistical
  Machine Translation," *EMNLP 2014*, pp. 1724–1734, DOI 10.3115/v1/D14-1179, arXiv:1406.1078
  (Crossref + arXiv).
  - Supports: introduces the hidden unit with reset and update gates later called the GRU.
    RHINO-MAG's GRU8 and PI-HNO use it.
  - The paper describes the unit but does not use the name "GRU" in its title.
- **Vanishing/exploding gradients.**
  - Y. Bengio, P. Simard, P. Frasconi, "Learning long-term dependencies with gradient descent
    is difficult," *IEEE Trans. Neural Networks* 5(2):157–166, 1994, DOI 10.1109/72.279181
    (Crossref).
  - R. Pascanu, T. Mikolov, Y. Bengio, "On the difficulty of training recurrent neural
    networks," *ICML 2013*, PMLR 28(3):1310–1318, arXiv:1211.5063 (PMLR page metadata).
  - Supports: gradients through long recurrences vanish or explode. This motivates gated
    RNNs, the stable |λ| < 1 parameterization of SSMs/LRU (§1.6), and the horizon curricula
    in §3.
- **"Exposure bias" (term origin, as far as I can confirm).** M. Ranzato, S. Chopra, M. Auli,
  W. Zaremba, "Sequence Level Training with Recurrent Neural Networks" (MIXER), *ICLR 2016*,
  arXiv:1511.06732. Venue: the arXiv page links the dblp record conf/iclr/RanzatoCAZ15
  (iclr2016).
  - Supports: §1 gives the definition: "We refer to this discrepancy as exposure bias which
    occurs when a model is only exposed to the training data distribution, instead of its own
    predictions."
  - I found no earlier use of the term, but I did not search exhaustively. Bengio et al. 2015
    (§3.1) describe the same train/test discrepancy without naming it.
- **NARX / series-parallel.**
  - K. S. Narendra, K. Parthasarathy, "Identification and control of dynamical systems using
    neural networks," *IEEE Trans. Neural Networks* 1(1):4–27, 1990, DOI 10.1109/72.80202
    (Crossref).
    - Supports: the origin of the series-parallel vs parallel identification configurations
      for neural networks. Ribeiro & Aguirre 2018 (§2.5) call it "a very influential paper"
      recommending series-parallel training, which they then re-examine.
  - I. J. Leontaritis, S. A. Billings, "Input-output parametric models for non-linear
    systems, Part I: deterministic non-linear systems," *Int. J. Control* 41(2):303–328, 1985,
    DOI 10.1080/0020718508961129; Part II (stochastic), 41(2):329–344, DOI
    10.1080/0020718508961130 (Crossref).
    - Supports: the standard reference for the NARMAX/NARX input–output model class. I
      verified metadata only, not text.
- **HystRNN, venue confirmed.** A. Chandra, T. Kapoor, B. Daniels, M. Curti, K. Tiels,
  D. M. Tartakovsky, E. A. Lomonova, "Generalizable models of magnetic hysteresis via
  physics-aware recurrent neural networks," *Computer Physics Communications* 314, 109650,
  2025 (issued 2025-09), DOI 10.1016/j.cpc.2025.109650 (Crossref).
  - Same author list as arXiv:2308.12002 ("Neural oscillators for magnetic hysteresis
    modeling", which I read in full).
  - Crossref records no preprint relation, so treating the CPC article as the published
    version of that preprint rests on the identical authors and topic [inferred]. The
    journal text may differ from the arXiv v1 I read.
  - This supersedes "venue not verified" in §2.8.
