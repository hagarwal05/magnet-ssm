# Literature review: small state-space models for transient magnetic hysteresis (MagNet Challenge 2)

**Status: final, 2026-11-06.** All four notes files were fact-checked individually and the
whole document was reviewed by reviewer-ssm (review_log.md R-L1). Sources:
`docs/lit_notes/` (siegen_paderborn.md, other_teams.md, ssm_and_training.md, background.md,
fact-checked in review_log.md) and, for our own work, `docs/RESEARCH_LOG.md`.

Conventions used throughout:

- Every factual claim cites a numbered reference (section 10). Claims a researcher inferred rather than read
  are tagged **[inferred]**; claims the reviewer could not verify are tagged **(unverified)**.
- Other tags: **[computed]** = arithmetic on published numbers or code; **[measured]** =
  computed by our team by re-scoring the teams' released predictions with the official
  metric (the researcher who did it is named, e.g. "measured, winner-others");
  **[transcript]** = taken from the auto-generated captions of a team's challenge
  presentation (section 4.1), the only public source for several teams; **[abstract
  only]** = only the abstract or metadata of the source was read.
- Every MagNet Challenge 2 (MC2) number states its metric. Unless noted otherwise,
  "seq-p95" is the organizers' sequence error and "energy-p95" their energy error (section
  3.4), averaged over the five final-test materials A–E. These are the only numbers
  comparable to the public leaderboard (section 3.4).
- "Winner" always names a track. MC2 awarded a Performance Track (1st SiegenPaderborn, 2nd
  Hefei, 3rd Sydney), an Innovation Track (1st Georgia Tech) and a Software Engineering award
  (section 4.1); Hefei has the lowest official error but placed second in the
  Performance Track. In section 9, "the winner" means SiegenPaderborn.
- Sections 1–8 are written for adaptation into the paper's related-work section. Section 9
  is internal and is not for the paper.

---

## 1. Scope and research questions

Soft magnetic cores in power converters dissipate energy as core loss, and the classical
tools for predicting it are equation-based fits whose accuracy "is inherently limited by
their mathematical form" [REF:how-magnet]. Loss depends jointly on flux swing, frequency,
dc bias, duty cycle and temperature in a way that cannot be separated into independent
factors when high accuracy is required [REF:why-magnet]. The Princeton-led MagNet program
responded with large open measurement databases and two open modeling challenges
[REF:how-magnet] [REF:ojpel-mc1] [REF:mc2-handbook]. MagNet Challenge 1 (MC1, 2023) asked
for the scalar core loss of a periodic, steady-state excitation [REF:ojpel-mc1]. MagNet
Challenge 2 (MC2, 2025) moved to transients: given the past flux density B(t) and field
strength H(t) of a core, the upcoming B'(t) and the temperature T, predict the upcoming
H'(t) [REF:mc2-handbook]. The data are sampled at 16 MHz [REF:mc2-tutorial1], and a model
must be frequency agnostic, time-step agnostic and initial-state agnostic
[REF:mc2-handbook], the three properties the organizers propose for time-domain
"foundation" models of hysteretic components [REF:compel2025].

This review supports a study of small structured state-space models, specifically S4D
[REF:s4d], on the MC2 task. It covers the classical core-loss and hysteresis models that MC2
set out to replace (section 2); the MagNet databases, challenges and metrics (section 3);
the approaches of the MC2 teams, with emphasis on small models (section 4); recurrent and
state-space sequence models (section 5); training methods for long autoregressive rollouts
(section 6); and small-data and cross-material transfer (section 7). Section 8 states the
gap and positions our work. Field solvers, component design and measurement methods are
out of scope except where they bear on the data. The review is organized around two
research questions:

- **RQ1 (accuracy vs parameter count).** Can a small state-space model (S4D) reach
  competitive MC2 accuracy with few parameters, measured as official seq-p95 against the
  leaderboard of 32 final entries (31 competing teams and the organizers' own entry)?
- **RQ2 (generalization to unseen materials).** How well does such a model transfer to
  materials it never trained on, when trained on the ten known MC2 materials and tested on
  held-out ones?

Unless stated otherwise, MC2 numbers in this review are the organizers' official
95th-percentile metrics on the five final-test materials. Several papers report means over
test segments or use their own test splits; those numbers are not comparable with the
leaderboard, and we say so where they appear (section 3.4).

## 2. Core-loss and hysteresis modeling

### 2.1 The Steinmetz family

The Steinmetz equation (SE) [REF:steinmetz] models the volumetric core loss of a sinusoidal
excitation as P_v = k·f^a·B_ac^b, with three fitted parameters [REF:ojpel-mc1]. Its limits
are well documented: it applies only to sinusoidal excitation, it is not accurate across
wide frequency ranges, and its accuracy depends strongly on how the parameters are
extracted; even for sinusoids on N87 ferrite with least-squares-fitted parameters, errors
exceed 20 % in some regions of the B–f plane [REF:why-magnet]. A family of extensions
targets non-sinusoidal waveforms, among them the modified Steinmetz equation [REF:mse] and
the improved generalized Steinmetz equation (iGSE) [REF:igse]. The iGSE,
P_v = (1/T)∫ k_i |dB/dt|^a (ΔB)^(b−a) dt, needs only the standard Steinmetz parameters,
splits a flux trajectory into major and minor loops, and is widely used because most
alternatives need parameters that manufacturers do not publish [REF:igse] [REF:ojpel-mc1]
[REF:how-magnet]. The i²GSE adds relaxation
terms for the losses that continue during zero-voltage intervals [REF:i2gse], at the cost
of five more parameters, eight in total [REF:ojpel-mc1]. The composite waveform hypothesis
(CWH) builds the loss of rectangular-voltage waveforms from square-wave measurements
[REF:cwh], the iGCC extends it with a time-varying local equivalent frequency
[REF:igcc], and Stenglein and Dürbaum model Mn-Zn ferrite loss under dc bias with two
frequency-independent parameters, reporting about 15 % accuracy [REF:stenglein].
Bertotti's separation of loss into hysteresis, classical eddy-current and excess components
[REF:bertotti-loss] is the standard physical decomposition and is still cited by MC2-era
models [REF:pi-hno] [REF:k26].

The MagNet data quantify where the family breaks down. On N87, the ratio of loss at 80 %
and 50 % duty cycle varies by a factor from nearly 1 to 2 across frequency and flux
amplitude, which contradicts the iGSE with constant parameters; relaxation in
zero-voltage intervals is captured by some models but not by the iGSE; and loss cannot be
decoupled into multiplicative factors of B_ac, f, H_dc, duty and T when high accuracy is
needed [REF:why-magnet]. On N87 triangular waveforms with parameters fitted at 50 % duty,
the iGSE gives 14.6 % average and 61.1 % maximum relative error, between a 21-parameter
and a 109-parameter neural network [REF:how-magnet]. In MC1 the best equation-based entry,
a 60-parameter combination of iGSE, CWH and i²GSE terms, reached about 15 % average p95
loss error but 93 % on one material whose training data lacked relaxation behavior
[REF:ojpel-mc1]. The organizers summarize that despite several upgrades these
curve-fitting methods "have limited accuracy" [REF:mc2-handbook].

### 2.2 Preisach models

The Preisach model [REF:preisach] represents hysteresis as a weighted parallel
superposition of two-valued relay operators (hysterons) [REF:mc2-handbook]
[REF:mayergoyz]. Its weight distribution, or the equivalent Everett function, is most
commonly identified from first-order reversal curves [REF:moree-preisach]. The classical
model has the wiping-out and congruency properties, but measured minor loops do not show
that congruency, and applying it uncritically to data can produce drifting curves and
unclosed minor loops [REF:moree-preisach]. The classical model is rate independent, and
rate-dependent ("dynamic") generalizations exist [REF:mayergoyz-dyn]
[REF:bertotti-dynpreisach]. In Bertotti's, each elementary loop switches at a finite rate
proportional to the difference between the applied field and its threshold, so the loop
area grows as C0 + C1·√f [REF:bertotti-dynpreisach]. A differentiable Preisach formulation [REF:roussel] makes the model
trainable by gradient descent; it is the variant the MC2 performance-track winner tested (section 4.2).
Because the distribution is identified on a discretized grid, a Preisach model has no
fixed parameter count; storage grows with the grid resolution [inferred].

### 2.3 Jiles–Atherton model

The Jiles–Atherton (JA) model [REF:ja1984] [REF:ja1986] links its parameters to physical
properties of the material and computes both major and minor loops, and many different
implementations of it exist [REF:mc2-handbook]. Its parameters can be extracted from the initial anhysteretic
susceptibility, coercivity, remanence and loop-tip coordinates [REF:ja-param1989] or
numerically from standard laboratory measurements [REF:ja-param1992]; the inverse static
model used in MC2 has five parameters [REF:rhino]. In practice, fitting those few parameters
has become a literature of its own, using for example particle-swarm optimization
[REF:marion2008] or neural networks [REF:trapanese2011] [inferred from titles]. Its physical basis
is also contested: Zirka et al. argue that the JA energy balance is closer to a co-energy
balance and that the rate-dependent extension lacks a sound basis, with computed loops
departing sharply from measurements on grain-oriented steel at 50–200 Hz [REF:zirka2012].
Rate-dependent JA adds classical eddy-current and excess-loss terms to the static equation
[REF:jiles1994]. Related circuit-level descriptions include the generalized dynamic circuit
model of Zhu et al. [REF:zhu1996], permeance–capacitance models for system-level
simulation [REF:luo2018] and circuit models derived from the Landau–Lifshitz–Gilbert (LLG)
equation [REF:cui2023] [REF:dulal2024]. Taxonomies of the field distinguish Duhem-type models
such as JA, which store no turning points, from Preisach-type models, which store reversal
points [REF:moree-review].

### 2.4 Why transients are hard

The Steinmetz family maps a periodic flux waveform to a scalar, cycle-averaged loss; no
member produces the H(t) waveform that MC2 asks for [inferred from the model forms]. MC2's
energy metric also integrates H·dB over an open segment, and the organizers note that over
such a segment the "energy exchanged ≠ total core loss" because the cycle is not closed
[REF:mc2-tutorial4], so a per-cycle loss law does not define the quantity being scored
[inferred]. Transients add three difficulties that steady-state models avoid. First, the
response depends on the excitation history: the MagNetX measurements show significant
core-loss variations during frequency transitions and B–H loop areas that change during
transient phases [REF:magnetx-db]. Second, the initial magnetic state matters and is
generally unknown; the MC2 tutorial on ODE models lists "the initial state of the material
is not known" as a central problem [REF:mc2-tutorial2], and the organizers' foundation-model
definition assumes only that the initial condition affects the response for a limited time
horizon [REF:compel2025]. Third, the earlier MagNet data could not support the study at
all: the steady-state database defines B with zero mean because the excitation was not
captured from its start [REF:why-magnet], and How MagNet states that the database "does not
include sufficient data to perform a systematic study of transient behavior"
[REF:how-magnet]. Recent work that targets transient magnetization directly makes the same
point, that Steinmetz-type formulas cannot reconstruct transient B–H paths under arbitrary
excitation [REF:pi-hno]. The organizers frame the move from MC1 to MC2 as a move from
Steinmetz equations to Preisach and JA territory [REF:mc2-keynote], and the MC2 handbook
states the aim as seeking alternatives to the Preisach and JA models using a large amount of
measurement data [REF:mc2-handbook].

## 3. Data-driven magnetics and the MagNet program

### 3.1 MagNet database papers

Three companion papers in *IEEE Transactions on Power Electronics* describe the MagNet
database and its modeling framework. *Why MagNet* [REF:why-magnet] describes a database of 142,871 voltage–current
waveform pairs measured on an N87 toroid under sinusoidal and two- and three-level PWM
excitation at 25–90 °C, each averaged to a single 1,024-point cycle. It quantifies the
couplings among dc bias, temperature and duty cycle that defeat Steinmetz-style models and
argues that loss should instead be defined as a time-domain function of B(t) and T. It
views neural networks as "a form of efficient interpolation of the data" whose challenges
are integration with existing tools and generalization beyond the test distribution, and
its data are steady state only: B has zero mean by definition, because the excitation was
not captured from its start [REF:why-magnet]. *How MagNet* [REF:how-magnet] presents the
machine-learning framework on a database of more than 500,000 data points. Its feed-forward
networks on N87 triangular waveforms trace a size–accuracy curve, from 19.76 % average and
68.59 % maximum loss error at 21 parameters, through 5.33 % / 21.36 % at 454, to 1.81 % /
9.47 % at 1,594 and 1.77 % / 8.62 % at 5,515, so the gain saturates above about 1,600
parameters [inferred from the table]. It also introduces an encoder–projector–decoder with
an autoregressive LSTM decoder that outputs the H(t) waveform, reaching 3.73 % average and
18.29 % maximum sequence error on 15,327 B–H pairs, scored with the sequence error
rms(H_pred − H_meas)/rms(H_meas) [REF:how-magnet], the same formula as
the MC2 official sequence error [REF:mc2-webinar6] [inferred lineage]. It uses transfer
learning to reduce the data needed per material and states that modeling transient
characteristics "is beyond the scope of this paper" [REF:how-magnet]. *MagNet-AI*
[REF:magnet-ai] frames a trained network as a "datasheet" for a material: on N87, an LSTM
of 28,225 parameters reaches 4.52 % mean and 10.92 % p95 sequence error and a Transformer of
28,481 parameters reaches 2.99 % and 6.48 %, compressing a 3.8 GB dataset into a 204 kB
model. It also warns that core loss is highly sensitive to the phase mismatch between H(t)
and B(t), so "an approximately matched sequence does not necessarily result in a good match
for core loss" [REF:magnet-ai].

### 3.2 MagNet Challenge 1 and its size–accuracy finding

MC1 (2023) asked teams to map one steady-state cycle of B(t) (1,024 steps), the frequency
and the temperature to the volumetric core loss, ranked by the 95th-percentile relative
loss error; transient dynamics were explicitly out of scope [REF:ojpel-mc1]. The summary
paper reports final results from 24 teams, although its appendix lists 23 [REF:ojpel-mc1].
Model size was defined as "the total number of parameters that a model needs to store to
describe the characteristics of each material", excluding structure, and the challenge was
"designed to encourage models with more computation and less memory usage"
[REF:ojpel-mc1]. The organizers drew three conclusions that frame RQ1. The results
"quantitatively verified the fundamental tradeoff between model size and model accuracy";
"the winning models use about 1,000 parameters to achieve less than 10% average 95th
percentile error", with the best average p95 reaching 7 % and the smallest model 60
parameters; and "the differences in the model accuracy are usually very small among the
best performing teams, whereas the differences in model size are often very large"
[REF:ojpel-mc1]. One team's multi-objective optimization identified about 1,000 parameters
as a good balance point [REF:ojpel-mc1]. The paper also cautions that measurement error was
generally below 20 % maximum and 10 % average, and that the risk of overfitting rises when
model accuracy exceeds measurement accuracy [REF:ojpel-mc1].

Several MC1 winners reappear in MC2. First place in performance went to
Paderborn's HARDCORE, a residual dilated convolutional network with an intermediate B–H
reconstruction [REF:ojpel-mc1], whose journal version reports a Pareto optimum at 906
parameters with below 8 % average p95 error [REF:hardcore]; the OJPEL summary gives 1,755
parameters for the challenge entry, a different configuration [REF:ojpel-mc1]. Fuzhou
placed second with a sequence-to-scalar Transformer and multi-stage fine-tuning, and
Bristol third with an LSTM and feed-forward network using transfer learning; Sydney won the
software engineering award [REF:ojpel-mc1]. The innovation awards went to Sydney's magnetization-mechanism-inspired
network (MMINN) of about 1,000 parameters [REF:ojpel-mc1] [REF:mminn], TU Delft's
multi-material transfer learning with multi-objective optimization, and Mondragon's
60-parameter equation-based model [REF:ojpel-mc1]. The summary describes grey-box hybrids
as widely adopted "for its excellent capability of balancing model accuracy and model
size", and its roadmap names transient excitation with minor loops and non-zero dc bias as
a future challenge [REF:ojpel-mc1].

### 3.3 MagNetX

MagNetX extends the database to transients [REF:magnetx-db]. Its automated acquisition
records sequences with duty-cycle step changes, "a mix of high frequency transient and low
frequency steady state", at 16 MHz sampling, with sub-cycle frequencies of 50–800 kHz in
seven buckets and temperatures of 25, 50 and 70 °C [REF:magnetx-db] [REF:mc2-tutorial1].
The organizers' slides give different per-material sequence counts at different times
[REF:mc2-slides] [REF:mc2-tutorial1], and the released files differ again (our
RESEARCH_LOG §2), so counts should be taken from the data. The companion modeling paper
[REF:magnetx-models] proposes a time-domain, frequency-independent model that "uses partial
memories and the next-state flux density excitation to predict the response of the field
intensity in the next time step", instantiated as a sequence-to-scalar LSTM. In the
organizers' slides this is the input layout of MC2: B(t) and H(t) over the past 80 steps
plus the next B, evaluated either single-step or autoregressively [REF:mc2-tutorial1]. The
COMPEL 2025 paper by the same group defines three properties of a time-domain foundation
model: frequency agnostic, adaptive to universal time steps, and the hypothesis that the
initial condition affects the response only over a limited time horizon; such a model "can
be physics-driven or data-driven, or a hybrid of both" [REF:compel2025]. A 2026 journal
paper develops the time-domain foundation-model concept further [REF:magnetx-tpel]; we
cite it for the concept only, since its full text was not read.

### 3.4 The MC2 task and metrics

MC2 asks, for each material, for a function H'(t) = f(B(t), H(t), B'(t), T) that predicts
the field strength over a future interval from the excitation history, the future flux
density and the temperature, and that is frequency, time-step and initial-state agnostic
[REF:mc2-repo] [REF:mc2-handbook]. Teams trained on ten known ferrite materials and were
scored on five new ones, each posing a named challenge: A (3C92), "tiny data", with 105
training sequences; B (3C95), "frequency"; C (FEC007), "no past H", where only H(1) = 0 is
given and teams "may assume all past of H(t) before the starting point are also 0 A/m"; D
(FEC014), "flux density"; and E (T37), "special material" [REF:mc2-webinar6].
Each test segment has 1,000 steps. Except for C, a known H prefix of 100, 500 or 900 steps
is given, in equal numbers, and the model predicts the rest [REF:mc2-webinar6]
[REF:mc2-eval-script].

The official sequence error of a segment is RMS(H_pred − H_meas)/RMS(H_meas) over the
predicted region; the official energy error is the difference between the predicted and
measured ∫H dB over the predicted region, divided by the core loss of the full excitation
[REF:mc2-webinar6] [REF:mc2-eval-script]. Both are aggregated by taking the 95th percentile
within each warm-up block, averaging over blocks, and averaging over the five materials
[REF:mc2-eval-script]. The tutorial notebooks use a different sequence normalization (by
mean |H|) and a different energy aggregation, so numbers computed with the tutorial code are
not leaderboard-comparable [REF:mc2-repo] (our verification, RESEARCH_LOG §3.2). The
model-size criterion changed during the challenge, from the kilobytes of submitted code in
the handbook [REF:mc2-handbook] to "number of material-specific parameters that need to be
kept in the model" in the kickoff slides [REF:mc2-slides], and finally to a per-material
parameter count declared by each team [REF:mc2-webinar6]. The README lists five judging
criteria: accuracy, model size, explainability, novelty and software quality
[REF:mc2-repo]. Winners were selected by the committee rather than by a published formula
(section 4.2).

The organizers' reference models are two "double LSTM" sequence-to-scalar next-step
predictors: a tutorial model with two LSTM(1, 12) encoders over the 80-step B and H
windows and a ReLU head (2,399 parameters), and a pretest model with two LSTM(1, 6)
encoders and a Tanh head (771 parameters) [REF:mc2-repo]. The tutorial model is trained
on measured H histories with a next-step MSE; the pretest model's training is not
published, only its weights. Both are evaluated autoregressively, writing each prediction
back into the H window [REF:mc2-repo]. Training on measured histories never exposes the
model to its own errors, the classic exposure-bias setup (section 6) [inferred]. On the
final test, NTU2 scored 75.11 % seq-p95 and 27.43 % energy-p95 with 2,399 parameters, and
the organizers' own non-competing Princeton entry, 1,443 parameters, scored 66.85 %
seq-p95 [REF:mc2-repo]. NTU2's count matches the tutorial architecture exactly, so its
entry is most likely that model trained by the team [inferred] (`docs/reference_audit.md`
§7).

## 4. Approaches in MC2

### 4.1 Comparison table

Table 1 lists the ten best entries by the organizers' sequence-error ranking and the
smaller or otherwise instructive entries below them. Leaderboard numbers are the
organizers' official values (seq-p95 and energy-p95, mean over A–E) and average
per-material parameter counts [REF:mc2-repo]. The organizers' "seq rank" is the rank of
each team's mean per-material rank over all 32 entries, not of its A–E mean error, which is
why, for example, Bristol (25.22 %) ranks above Sydney (24.63 %) [REF:mc2-repo] [measured,
winner-others]. Awards are from
the organizers' winners announcement [REF:mc2-winners]. Method descriptions come from each
team's paper or code where public; for most teams the only public source is a five-minute
presentation, whose auto-generated captions are the basis of entries tagged
[transcript] [REF:mc2-talks].

**Table 1.** MC2 approaches. "AR" = predicted H is fed back as an input; "non-AR" = the
predicted H is never an input. Method descriptions whose only source is [REF:mc2-talks] come
from auto-generated captions of the team's presentation and are all [transcript].

| team (award) | seq rank / 32 | seq-p95 (%) | energy-p95 (%) | params | architecture | key techniques | source |
|---|---|---|---|---|---|---|---|
| Hefei (HFUT-CityUHK-XMU) (Performance 2nd; Software Eng.) | 1 | 9.17 | 1.57 | 6,564 (4,186–8,600) | 1-layer LSTM over the whole 1,000-step sequence + MLP head emitting per-step parameters of 3–4 direction-dependent asymmetric (play-like) operators; H is a convex combination of the operator outputs; non-AR | inputs B, ΔB, Δ²B, known-H channel with a mask (any warm-up length); energy term in the loss [transcript] (the A and C checkpoint configs record an RMSE loss); Pareto / Bayesian search of sizes [transcript] | [REF:hefei], [REF:hefei-code] |
| Fuzhou (honorable mention) | 2 | 12.42 | 1.91 | 63,022 | "dual-attention" LSTM sequence-to-sequence | compared 10+ regressors and 5 learning strategies, seq2seq best; smooth-L1 loss with simple physical constraints; material-specific features [all transcript] | [REF:mc2-talks] |
| Southeast1 (honorable mention) | 3 | 14.51 | 2.69 | 22,197 | analytical vector-magnetic-circuit model (6 parameters per periodic sequence, table lookup by flux amplitude) + BiLSTM residual corrector | offset calibration of the prediction to the known H [transcript] | [REF:mc2-talks] |
| Eindhoven (honorable mention) | 4 | 17.71 | 2.38 | 81,893 | MLP-only residual encoder–decoder on a 100-sample history | trend / high-frequency decomposition; variational latent [transcript] | [REF:mc2-talks] |
| Tsinghua (honorable mention) | 5 | 19.93 | 2.52 | 2,177,283 | BiLSTM + dense head + linear B→H skip; no H input | pretraining on groups of similar known materials, then fine-tuning; architecture proposed by an LLM agent [transcript] | [REF:mc2-talks] |
| **SiegenPaderborn (Performance 1st)** | 6 | 20.43 | 3.41 | **325** | 1-layer GRU, h = 8, per material; hidden unit 0 is H | true H injected into the state during warm-up, H never an input; closed-loop training (28-step warm-up + 128–156-step free rollout, full BPTT, no teacher forcing); B, ΔB, Δ²B, T inputs; \|ΔB\|-weighted, RMS(H)-normalized loss | [REF:rhino], [REF:rhino-code] |
| Bristol, MagLearn 2 (honorable mention) | 7 | 25.22 | 8.40 | 860,929 | 3-layer LSTM encoder–decoder (h = 128), decoder driven by future B; non-AR | one model for all warm-up splits; MSE with the top 5 % of errors trimmed; transfer from the best of 10 base models chosen on target validation | [REF:maglearn2] |
| XJTU1 | 8 | 22.48 | 3.64 | 1,350,913 | not public | — | [REF:mc2-repo] |
| Sydney, PI-HNO (Performance 3rd) | 9 | 24.63 | 5.44 | 4,777 | history GRU sets the state of a future GRU driven by B, ΔB; 1-layer Transformer over the full B sequence; non-AR, trained on 100-step targets | T and position injected into the initial state; smooth-L1 + open-path energy penalty | [REF:pi-hno] |
| NITC | 10 | 24.85 | 4.73 | 347,237 | mixture-of-experts LSTM encoders + decoder over 16 future steps; AR in 16-step chunks | five-term loss (MSE, derivative, peak hinge, energy, relative RMS) [transcript] | [REF:mc2-talks] |
| KU Leuven (honorable mention) | 12 | 34.54 | 5.37 | 111,985 | 1D-CNN denoising diffusion model that inpaints the unknown H; non-AR | ~1,000 denoising steps per sequence [transcript] | [REF:mc2-talks] |
| Tianjin (honorable mention) | 13 | 32.56 | 6.99 | 16,391 | dual LSTM encoders over 80-step B and H windows with cross-attention; one-step AR | sign-flip symmetry augmentation; hybrid absolute / relative loss [transcript] | [REF:mc2-talks] |
| Cambridge (Innovation 2nd) | 14 | 37.01 | 9.24 | 1,182 | H = H_static (compact DeepONet on an 11-dim hand-built memory state) + H_dyn = C₁Ḃ + C₂B̈; H never used | reversal-point and loop-closure features [transcript] | [REF:mc2-talks] |
| Silicon Austria Labs (Innovation 3rd) | 15 | 39.48 | 30.05 | 3,713 | feed-forward net on 38 hand-built features; one-step AR | last H value excluded to limit drift; 20-step rollout loss [transcript] | [REF:mc2-talks] |
| Georgia Tech (Innovation 1st) | 17 | 38.76 | 7.97 | **83** | mechanical analogy: spring (saturation), slider (pinning, with decoupled static and dynamic friction), damper (eddy current), inertia (displacement current) | initial spring stretch fitted from the warm-up; C initialized as demagnetized, which the team called a mistake [transcript] | [REF:gatech], [REF:mc2-talks] |
| Princeton (organizers, not competing) | 23 | 66.85 | 15.89 | 1,443 | not described | — | [REF:mc2-repo] |
| NTU2 | 23 | 75.11 | 27.43 | 2,399 | most likely the organizers' tutorial dual-LSTM (same parameter count) [inferred]; one-step AR | — | [REF:mc2-repo] |
| *Post-challenge:* Kirchgässner circuit (not an entry) | — | 19.43 ± 0.22 | 5.12 ± 0.19 | 32 (+33 for the warm-up estimator) | viscous Prandtl–Ishlinskii play bank + relaxations + mean-field feedback, fitted end to end | trained on full 1,000-step windows with the test's warm-ups; loss = sequence + energy error | [REF:k26] |

### 4.2 SiegenPaderborn (RHINO-MAG)

The Performance Track winner, a joint Siegen and Paderborn team [REF:mc2-keynote], trained
one single-layer gated recurrent unit (GRU) [REF:gru] per material with eight hidden units
and 325 parameters [REF:rhino]. The count includes five normalization constants, consistent
with the MC2 rule that every material-specific value kept in the model counts as a
parameter; the general count is 3h² + 16h + 5 for hidden size h [REF:rhino-code]
[computed]. The design departs from the official baseline in three ways. First, H is never
an input. Hidden unit 0 *is* the normalized H prediction; over the known segment the true H
overwrites that unit at every step while the other seven units evolve freely, and prediction
then runs closed-loop from the resulting state [REF:rhino]. Second, the model is trained the
way it is evaluated. Each training sample is a short warm-up (28 steps, or 1 step for
Material C, matching its test condition) followed by a free-running rollout with full
backpropagation through time and no teacher forcing. The rollout is 128 steps in the
paper's hyperparameter table and 156 steps in the released training script; which of the
two produced the submission is not stated [REF:rhino] [REF:rhino-code]. Third, inputs and
loss are matched to the metric. The inputs are B, its first and second central differences
and T, each max-abs normalized, with the switching frequency deliberately excluded so the
model applies to non-periodic waveforms; the loss is an RMSE weighted by |ΔB| and divided
by each source sequence's RMS(H), so that it targets the per-sequence relative error the
leaderboard scores [REF:rhino]. The submitted models were trained on all training data for
1,500–10,000 epochs per material, with no validation split and no early stopping
[REF:rhino] [REF:rhino-code]. A side effect of keeping the prediction inside a GRU state
is that it stays bounded in normalized units (by the larger of 1 and the injected value,
which on test data can exceed the training maximum), so a rollout cannot diverge
numerically [inferred].

The submission scores 20.43 % seq-p95 and 3.41 % energy-p95 [REF:mc2-repo]. The paper's
headline of 8.02 % sequence error is a per-sequence *mean*, averaged over the five
materials, and only the 20.43 % is comparable to other leaderboard entries [REF:rhino]. The
two are not computed from identical outputs: re-scoring the submitted predictions gives a
mean of 7.96 % and the official 20.43 % p95, and no percentile convention reproduces the
paper's p95 table from them, while the paper's per-material means match an independent
re-run of the published weights on four materials [measured, reviewer-ssm] [REF:k26]. The
paper's numbers therefore appear to come from re-generated predictions of the same models
[inferred]. The like-for-like pair is thus 7.96 % mean against 20.43 % p95, a factor of about
2.6 that reflects a right-skewed per-sequence error distribution: in the
organizers' per-block values for the submitted model, Material A with 10 % of H known has a
mean sequence error of 11.83 % against a 95th percentile of 21.62 %, and with 90 % known,
10.09 % against 26.42 % [REF:rhino-code]. An independent re-evaluation of the published
weights on the official test set agrees with the organizers' p95 values to within 0.3
points on four materials, but not on C (22.65 % against the organizers' 27.00 %); the cause
is not established [REF:k26]. Material A, with 105 training sequences, reaches 23.27 %
seq-p95 with no transfer or pretraining, only longer training (10,000 epochs)
[REF:rhino] [REF:mc2-repo]. The worst material is C (27.00 % seq-p95) [REF:mc2-repo].

The ablations are the most useful part of the paper for small-model design [REF:rhino].
Anchoring the state to the true H mattered most: a GRU started from a zero state without
injection "produces poor prediction results", while a single injected H value with
closed-loop training performed about as well as a full-length warm-up on three of the four
materials that have a warm-up, with a slight advantage for the full warm-up only on FEC014,
the most hysteretic material (five models per setting; individual outlier models exist in
both directions, by visual reading of the figure; on FEC007 the setups coincide by
construction). Each of the two loss modifications (ΔB weighting and RMS(H) normalization) beats
plain MSE, but combining them is "not clearly better or worse than either isolated
version." No physics-structured variant that was evaluated beat the plain GRU. An inverse Jiles–Atherton
model, a permeability-form output and a vector-field state were clearly worse, and the
last of these, like a GRU that outputs Jiles–Atherton parameters, cannot be warmed up from
the known H; Jiles–Atherton with a residual GRU was unstable, and a
Jiles–Atherton-regularized physics-informed network was dropped as unhelpful; a
magnetization-form output was close to the plain GRU and somewhat worse on some
materials (visual reading); a differentiable Preisach model was tested only in a
preliminary comparison, and in the reverse (H-to-B) direction; and a
Landau–Lifshitz–Gilbert micromagnetic model was judged computationally infeasible at the
challenge's time step and not evaluated. On a post-challenge size–accuracy
plot on the official test data, the "rather standard data-driven models are roughly within
the same performance band", with the GRU "slightly more parameter efficient" than an LSTM
with the same state injection for most sizes, and Hefei's entry better than any GRU at
larger sizes; the authors leave open whether that gap comes from architecture or
training. The paper evaluates no cross-material modeling and lists transfer learning to
data-poor materials as future work [REF:rhino]. The released repository contains code for
pooled pretraining over A–E followed by per-material fine-tuning, and for an H-as-input
GRU ablation, but no results for either are published [REF:rhino-code].

SiegenPaderborn took first place in the Performance Track, ahead of Hefei & CityU & Xiamen
and Sydney, with an award citation for "well-balanced model size, model performance, and
Pareto-front exploration" [REF:mc2-winners] [REF:mc2-keynote], while ranking 6th in
sequence error, 6th in energy error and 3rd in size [REF:mc2-repo]. The two are consistent
because the track was not judged on error alone. The kickoff slides define model performance as the 95th-percentile sequence
error and model size as the number of material-specific parameters kept in the model, and
state that the final winners are selected by an Academic Advisory Committee
[REF:mc2-slides]; the handbook adds that the judging committee evaluates accuracy, size,
generality and novelty [REF:mc2-handbook]. No published formula combines these criteria
(as far as the sources read show).

A post-challenge preprint by a RHINO-MAG co-author asks whether what the GRU learned can be
written as a governing equation [REF:k26]. It replaces the GRU with a difference-equation
equivalent circuit: a second-order correction of the measured B, a bank of six rate-dependent
(viscous) Prandtl–Ishlinskii play operators, two first-order relaxation paths, a mean-field
feedback of H in the spirit of the Jiles–Atherton effective field, and a reversible
saturation curve, with every coefficient constant or linear in temperature, for 32
parameters per material. A separate warm-up state estimator adds 33 learned parameters per
material that the 32 do not count. The circuit is trained on full 1,000-step windows with
the test's warm-up lengths and with the sum of the sequence and energy errors as the loss.
On the official A–E test it reports 19.43 ± 0.22 % seq-p95 and 5.12 ± 0.19 % energy-p95
over ten seeds, which the paper describes as on par with the GRU in sequence error
(20.43 %, a single submitted seed) and which is worse in energy error (3.41 %). A gain-scheduled variant, in which a small stateless MLP shifts the circuit
coefficients at every step, reaches 13.03 % seq-p95 with 656 parameters and 12.09 % with
5,552 [REF:k26]. In the same paper the circuit generalizes better than a GRU of the
RHINO-MAG architecture when trained at or below 320 kHz and tested at 500 and 800 kHz
(10.2 % against 54.5 % sequence error under the paper's own protocol); the GRU there is the
author's own re-trained replica, which the paper describes as weaker than the published
network [REF:k26]. The preprint is not peer
reviewed, and its acknowledgment states that an LLM assisted in every part of the work
[REF:k26]. Its parameter counts also use a different convention: it counts the RHINO-MAG GRU
as 336 parameters, while the team and the leaderboard count 325 for the same model
[REF:k26] [REF:mc2-repo]; the difference is consistent with a framework that keeps separate
input and hidden biases and excludes normalization constants [inferred]. Counts for the
same model therefore range from 320 (GRU cell only) through 325 to 336, and the circuit is
32 or 65 parameters depending on whether its warm-up estimator is counted [computed]; any
size comparison should use one convention or label each point with its source's.

### 4.3 Hefei (HFUT-CityUHK-XMU): LSTM with direction-dependent operators

Hefei has the lowest official error, 9.17 % seq-p95 and 1.57 % energy-p95, first on both
leaderboard columns, with 4,186–8,600 parameters per material; it placed second in the
Performance Track and won the Software Engineering award [REF:mc2-repo] [REF:mc2-winners].
The method is published as an LSTM-based direction-dependent asymmetric operator (LSTM-DAO)
model [REF:hefei]; the description here comes from the released inference code, since only
the paper's abstract was accessible [REF:hefei-code]. A single unidirectional LSTM reads
the whole 1,000-step sequence in one causal pass, with five channels per step: normalized B,
its first and second differences, the known H with unknown positions set to zero, and a
known-H mask. At every step an MLP head, which also reads the temperature, emits the radius,
bias and weight of three or four operators, with separate values for rising and falling B.
Each operator clips a shifted copy of B between running extrema, and H is the weighted
(convex) combination of the operator outputs [REF:hefei-code]. Predicted H is never fed
back, so the model has no exposure bias by construction, and the mask lets one network
handle any warm-up length [REF:hefei-code] [inferred]. Because the operator parameters are
re-emitted at every step, the memory effectively lives in the LSTM, and the operators act
as a bounded, direction-dependent output map rather than as classical play operators
[inferred, checked numerically by winner-others]. The LSTM holds 81–87 % of the
parameters [REF:hefei-code] [computed]. In the team's presentation, an energy term is added to the
training loss, configurations are chosen by a Pareto-front search (the captions suggest
Bayesian optimization), and accuracy is
said to fall "dramatically" below about 1,000 parameters [REF:mc2-talks] [transcript]; the
abstract confirms the Pareto-front selection [REF:hefei]. The configuration files stored
with the A and C checkpoints record an RMSE loss, so whether the energy term was used for
the submitted models is unclear [REF:hefei-code] [measured, reviewer-teams]. Hefei's lead
holds at every warm-up length: re-scoring its submission per warm-up block gives 9.86 %,
6.85 % and 7.88 % p95 (mean over A, B, D, E) for 100, 500 and 900 known steps, against
18.72 %, 19.02 % and 18.64 % for SiegenPaderborn [measured, winner-others]. Within its
900-step forecasts, Hefei's error is the lowest of the top ten at the end, rising from 6.8 %
to 11.2 % p95 between the first and last 100-step windows [measured, reviewer-teams].

### 4.4 Fuzhou: attention-based sequence-to-sequence

Fuzhou is second on the leaderboard, 12.42 % seq-p95 and 1.91 % energy-p95, with about 63,000
parameters per material, and is the best team on Material C (12.29 %) [REF:mc2-repo]. No paper
or code is public. According to the team's presentation, they compared more than ten
classical regressors (gradient-boosted trees were best but extrapolated poorly) and five
learning strategies, of which direct sequence-to-sequence learning was best, and settled on
an LSTM with a "dual attention" module, a smooth-L1 loss with simple physical constraints to
fit peaks and troughs, and material-specific input features [REF:mc2-talks] [transcript].
Unusually, Fuzhou's block error rises with the length of the known history: 9.36 %,
11.32 % and 16.67 % p95 (mean over A, B, D, E) for 100, 500 and 900 known steps, so its
model is weakest on the shortest forecasts [measured, winner-others].

### 4.5 Sydney: physics-informed hybrid neural operator (PI-HNO)

Sydney placed third in the Performance Track with 24.63 % seq-p95, 5.44 % energy-p95 and
4,777 parameters [REF:mc2-repo] [REF:mc2-winners]. The PI-HNO preprint [REF:pi-hno], from a
University of Sydney and UTS group, describes a model with exactly that parameter count and
the architecture the team presented [REF:mc2-talks]; the paper does not say it is the MC2
entry, so we treat it as the same model line rather than as the entry itself [inferred]. A history GRU reads the last 100
samples of (B, H) before the forecast boundary; its final state, plus projections of the
temperature and the boundary position, initializes a future GRU that is driven by B and ΔB
at every predicted step. A one-layer Transformer over the full 1,000-step B sequence adds a
global term, motivated as a data-driven analog of the Preisach plane, and the two branches
are added before an output MLP, motivated by the separation of H into hysteretic and
dynamic parts. The model is not autoregressive: the future GRU never sees predicted H.
Training uses 100-step history and 100-step target segments with a smooth-L1 loss plus a
penalty on the open-path energy error [REF:pi-hno]. Trained on 100-step targets, it is
evaluated on forecasts of up to 900 steps, over which its p95 error on the official test
grows from 17.1 % to 32.4 % between the first and last 100-step windows
[measured, reviewer-teams]. The paper's ablations, on its own
14-material protocol rather than the official test and from single runs, are informative
for small models: feeding the temperature and start position to the future GRU at every
step, instead of using them to initialize its state, raised the pooled p95 sequence error
from 31.39 % to 75.68 %; removing either branch raised it by 11–18 points, and removing the
ΔB input by 8.7 points, with its main effect on the energy error (p95 7.60 → 12.66 %);
and a plain GRU of matched size (4,758 parameters) reaches 46.17 % [REF:pi-hno]. The paper
reads the first result as showing that the placement of the conditioning matters
[REF:pi-hno].

### 4.6 Georgia Tech: mechanical analogy with 83 parameters

Georgia Tech won the Innovation Track with the smallest model of the challenge, 78–90
parameters per material, scoring 38.76 % seq-p95 and 7.97 % energy-p95 [REF:mc2-repo]
[REF:mc2-winners]. Their letter [REF:gatech], of which only the abstract was accessible,
models H as the sum of four "forces" in a mechanical analogy in which B is displacement: a
spring for the ideal magnetization curve, a slider for domain-wall pinning with static and
dynamic friction decoupled, a viscous damper for eddy currents, and an inertia term
proportional to d²B/dt² that the abstract attributes to displacement current and describes as
not previously reported [REF:gatech]. The initial stretch of each spring is fitted from the
known H segment. According to the presentation, the team set Material C's initial state to
zero (demagnetized), which they called a mistake, and the organizers commented that C has
"no history", not a zero state [REF:mc2-talks] [transcript]. The organizers' evaluation
slides allow teams to assume that the past H of C is 0 A/m [REF:mc2-webinar6], which
constrains H but not the magnetic state, so the two statements are compatible [inferred]. C accounts for much of their
mean: 84.68 % seq-p95, against 27.28 % averaged over A, B, D and E [REF:mc2-repo]
[computed].

### 4.7 Cambridge: compact physics-inspired DeepONet

Cambridge placed second in the Innovation Track with 1,182 parameters, 37.01 % seq-p95 and
9.24 % energy-p95 [REF:mc2-repo] [REF:mc2-winners]. No paper or code is public. In the
presentation, H is split into a static hysteresis part and a dynamic part. The static part is
a compact DeepONet whose branch input is an eleven-dimensional hand-built memory state (the
four reversal points of B nearest the current value, the relative position within the
current minor loop, the running extrema of B, temperature, the initial field and a count of
recent reversals) rather than the raw B history; the dynamic part is C₁Ḃ + C₂B̈ with
coefficients from an MLP, derived from an equivalent-circuit view of the measurement
[REF:mc2-talks] [transcript]. The model never uses the known H, which the team describes
as keeping long rollouts stable at the cost of initial-state information
[REF:mc2-talks] [transcript]. Its weakest results are on C (70.53 %) [REF:mc2-repo] and, in
the material-averaged blocks, on the longest forecasts (35.00 % p95 with 100 known steps,
against 24.36 % and 26.54 % with 500 and 900), though the pattern differs by material
[measured, winner-others].

### 4.8 Bristol: MagLearn 2

MagLearn 2 [REF:maglearn2] is Bristol's entry: its 860,929-parameter configuration matches
the leaderboard count, where Bristol scores 25.22 % seq-p95 and 8.40 % energy-p95 (seventh by
the organizers' ranking) [REF:mc2-repo]. A three-layer LSTM encoder (hidden size 128) reads
the (B, H) history and initializes a decoder LSTM of the same size, which is driven by the
future B and by an embedding of the temperature and an effective PWM frequency estimated from
B; predicted H is never fed back. One model serves all warm-ups, because each training
window is used with history-to-forecast splits of 1:9, 5:5 and 9:1, and the loss is an MSE
with the largest 5 % of pointwise errors trimmed, which the paper motivates as reducing the
influence of rare high-error points such as saturation events [REF:maglearn2]; the team's
presentation adds that the aim was a better 95th percentile [REF:mc2-talks] [transcript]. For each new material the team adapts each of the ten known-material models
briefly, picks the one that does best on target validation, and fine-tunes it fully;
material similarity did not predict the choice [REF:maglearn2]. An occlusion study finds
that the error starts to grow only when the most recent part of the H history is removed,
with the full B kept: the median threshold is about 560 ns, roughly nine samples at the
native 16 MHz rate (the paper resamples its windows, so this is not nine model steps)
[REF:maglearn2] [computed]. These thresholds mark where error growth turns rapid, not where
error stays small: keeping the mean error under 20 % needed about 2–43 µs of H history in
the same table, roughly 35–690 native samples [REF:maglearn2] [computed]. The
abstract's "2.2 %" is the mean error on the single best material (FEC014), on the paper's
own windows, not an A–E average or the official metric; Bristol's official per-material
scores range from 5.78 % on D, the best of all teams, to 56.88 % on C [REF:maglearn2]
[REF:mc2-repo]. Re-scoring the submitted predictions reproduces the paper's numbers for A,
D and E but not for B and C [measured, winner-others], so for those two the paper
apparently reports a different, presumably post-competition, model [inferred]; the team said
in its presentation that B and C overfitted in the final evaluation [REF:mc2-talks]
[transcript].

### 4.9 Other entries and cross-cutting observations

The remaining top-ten entries are documented only by their presentations (Table 1). Southeast1
(third, 14.51 %) corrects an analytical vector-magnetic-circuit model with a bidirectional
LSTM residual and calibrates the prediction's offset to the known H; its per-material scores
place it third or fourth on every material [REF:mc2-talks] [REF:mc2-repo] [transcript].
Eindhoven (fourth, 17.71 %) uses only dense residual blocks on a 100-sample history,
decomposed into trend and high-frequency parts, and its presenters cited work questioning
whether time series need attention or SSMs at all [REF:mc2-talks] [transcript]. Tsinghua
(fifth, 19.93 %) uses a bidirectional LSTM with a linear B-to-H skip, no H input at all, and
pretraining on groups of similar known materials [REF:mc2-talks] [transcript]. Among entries
that are explicitly autoregressive in H, the best is NITC (tenth, 24.85 %), which predicts
16 steps per call and reports in its own ablation that using fewer of them per call deviated
more from the ground truth [REF:mc2-talks] [transcript]. Tianjin's dual-LSTM over 80-step windows, the entry closest to
the official baseline and to our setup, reaches 32.56 % (27.2 % on A, B, D and E), and the
team reports that tuning the architecture for one-step accuracy caused "error explosion" in
rollout while tuning the loss and learning rate did not [REF:mc2-talks] [REF:mc2-repo]
[transcript]. Silicon Austria Labs (39.48 %, 3,713 parameters) trains a one-step
feed-forward model with a 20-step rollout loss and deliberately drops the last H value from
its inputs because the model over-relied on it [REF:mc2-talks] [transcript]. KU Leuven
(34.54 %) frames the task as inpainting H with a denoising diffusion model [REF:mc2-talks]
[transcript].

Four patterns recur among the strongest entries. All four are correlations across teams
with different models, data handling and effort, not controlled results: no team published
a comparison of autoregressive and non-autoregressive training on the official metric
[inferred from the sources above].

First, the top entries mostly avoid teacher-forced one-step prediction followed by rollout.
Of the top seven, four are non-autoregressive in H: Hefei and Bristol by their code and
paper [REF:hefei-code] [REF:maglearn2], Tsinghua and Southeast1 by their presentations
[REF:mc2-talks] [transcript]. SiegenPaderborn carries the prediction in a recurrent state
trained closed-loop [REF:rhino]; Fuzhou and Eindhoven could not be classified from their
presentations [REF:mc2-talks]. Below the top seven, Sydney's model is non-autoregressive
but is trained on 100-step targets and evaluated over up to 900 steps [REF:pi-hno], and NITC
feeds predictions back every 16 steps [REF:mc2-talks] [transcript]. The entries known to
predict one step at a time and roll out, with at most a short rollout loss, score 32–75 %
seq-p95 (Tianjin, SAL, NTU2) [REF:mc2-repo]. Tianjin's result also shows that this regime
is not by itself the whole story: with the layout closest to ours (one-step autoregressive
dual LSTM on 80-step windows) it reaches about 27 % on A, B, D and E, roughly half our
error [REF:mc2-repo] [computed] [transcript], and its error does not compound over the
first fed-back steps the way ours does (section 8) [measured, by our team].

Second, error grows with forecast length for every top team, least for the
non-autoregressive ones. Averaging the per-block p95 over A, B, D and E suggests otherwise:
the top seven differ by at most 9.4 points between the 100- and 900-step-warm-up
blocks, in both directions (Fuzhou is 7.3 points worse on the shortest forecasts, Tsinghua
9.4 points worse on the longest), and only SiegenPaderborn and Bristol are flat
[measured, winner-others]. But the blocks are disjoint sets of sequences, and short
forecasts are penalized by the metric itself, since over only 100 predicted steps the RMS(H)
denominator can be small [measured, winner-others]. Within the same sequences (the
900-step forecasts), the 95th-percentile error of each 100-step window grows from the first
window to the ninth for every top-ten team, by about 1.2–3.7×, and the growth persists when
each window is normalized by its own RMS(H) instead of the region's [measured,
reviewer-teams]. Growth is smallest (at most 1.65×) for the non-autoregressive teams that
also use the known H and train over the full horizon: Hefei (6.8 → 11.2 %) and Bristol by
code, Southeast1 by transcript, and possibly Fuzhou and Eindhoven, whose feedback design is
unverified. Not re-reading predicted H is not enough by itself: Tsinghua, which never reads
H, grows 2.3×, and Sydney, non-autoregressive but trained on 100-step targets, 1.9×. Growth
is also large for the models that carry the prediction forward in a closed loop or feed it
back, SiegenPaderborn (10.5 → 24.6 %), Tianjin, XJTU1 and NITC (10.7 → 39.6 %)
[measured, reviewer-teams and winner-others]. In absolute terms the non-autoregressive
teams add 2–5 points over the 900 steps (Hefei, Bristol and Southeast1 4.2–5.4; Fuzhou 4.4
and Eindhoven 2.2, both unverified designs) and the closed-loop or autoregressive ones 14–29
[measured, reviewer-teams; `results/drift/`]. A low first-window error does not
imply a good long horizon: NITC starts almost exactly where SiegenPaderborn does and ends at
the highest value in the top ten. Cambridge, which never uses H, is the one exception; it
starts high and falls (38.6 → 32.2 %), consistent with an uncorrected initial offset
[inferred]. Material averages also hide opposite
per-material trends: SiegenPaderborn's block p95 rises with more known history on A and B
but falls on D [measured, winner-others].

Third, the known H is used to set the initial state rather than as a recurrent input
(Hefei's mask channel, Sydney's history GRU, Bristol's encoder state, SiegenPaderborn's state
injection, Southeast1's offset calibration). Material C, with a single known value, is where
this matters most: five of the entries in Table 1 score 54 % or worse there (Georgia Tech 84.7 %, Leuven
76.6 %, Cambridge 70.5 %, Bristol 56.9 %, Tianjin 54.1 %), while Fuzhou, Hefei, Southeast1 and
Eindhoven reach 12–17 % [REF:mc2-repo]. Georgia Tech attributes its result to a wrongly
assumed demagnetized initial state [REF:mc2-talks] [transcript]; Tsinghua, which never reads
H, scores 22.5 % on C, so the absence of H history is not by itself the cause
[REF:mc2-repo] [inferred].

Fourth, rate features ΔB and Δ²B, or physically equivalent damper and inertia terms, appear
in most strong designs (Hefei, SiegenPaderborn, Sydney, Georgia Tech, Cambridge,
Tsinghua), and losses are shaped toward the evaluation metric or its tail (relative,
trimmed, smooth-L1 or energy terms) [REF:hefei-code] [REF:rhino] [REF:pi-hno] [REF:gatech] [REF:maglearn2]
[REF:mc2-talks].

No MC2 entry for which a method description is available used a structured state-space
model (S4, S5, LRU or Mamba) [REF:mc2-talks] [inferred]; descriptions are missing for some
teams, including XJTU1, so this is an absence-of-evidence statement (section 5.5).

## 5. Sequence models for dynamical systems

### 5.1 Recurrent networks: RNN, LSTM, GRU

Recurrent neural networks carry a hidden state from step to step and are the default
neural model for dynamical systems with memory. Gated variants, the long short-term memory
(LSTM) [REF:lstm] and the gated recurrent unit (GRU) [REF:gru], were introduced to ease the
learning of long-range dependencies, which plain recurrent networks struggle with because
gradients through long recurrences vanish or explode [REF:rnn-gradients] [REF:pascanu2013]. They dominate the MagNet literature: the How MagNet and MagNet-AI
models [REF:how-magnet] [REF:magnet-ai], the MagNetX reference model [REF:magnetx-models],
both organizer baselines [REF:mc2-repo], and most MC2 entries (Table 1). The MC2 performance-track winner's
argument for a GRU over an LSTM is practical: a GRU's single state vector is easy to
initialize from the last known H, and it has fewer parameters [REF:rhino]; in its own
size–accuracy sweep an LSTM with the same state injection is slightly less
parameter-efficient [REF:rhino]. The MC2 baseline layout, past inputs and past outputs
mapped to the next output, is the nonlinear autoregressive model with exogenous inputs
(NARX) of system identification [REF:leontaritis1985]. That field has long distinguished
training such a model with measured outputs fed back (series-parallel) from training it
with its own outputs fed back (parallel) [REF:narendra1990], and has revisited which works
better in simulation [REF:ribeiro2018] (section 6).

### 5.2 Structured state-space models: S4, S4D, S5, LRU, Mamba

Structured state-space models (SSMs) build deep sequence models from linear, continuous-time
state-space layers x′ = Ax + Bu, y = Cx + Du, discretized with a step Δ and interleaved with
static nonlinearities [REF:s4] [REF:bonassi2024]. S4 introduced structured state matrices for
modeling long sequences efficiently, and showed that one layer has three equivalent views,
continuous, recurrent and convolutional [REF:s4]. Diagonal state matrices
turned out to be enough: DSS matched S4 with a purely diagonal A [REF:dss], and S4D
simplified the construction to a collection of one-dimensional complex modes whose
convolution kernel is a sum of damped exponentials, K(t) = Σ_n C_n e^{tA_n} B_n [REF:s4d].
The S4D ablations show that the initialization of A is critical (scaling or randomizing
its imaginary parts degrades results), that constraining Re(A) < 0 "ensures stability even
in unbounded autoregressive settings" at negligible cost, and that the choice between
zero-order-hold and bilinear discretization makes no noticeable difference; training B is
"consistently slightly better" than freezing it, although the convolution kernel depends
only on the elementwise product of B and C, so with C trained a trainable B adds no
expressivity and acts only on optimization [REF:s4d]. These ablations used models of about
100,000 parameters [REF:s4d]. Gu et al. show that 1/Δ sets the range of dependencies a channel
captures, and that matching the minimum Δ to the task's dependency length stabilizes
results [REF:hippo-train].

Three later lines matter for MC2. S5 replaces many single-input SSMs with one
multi-input, multi-output diagonal SSM computed by a parallel scan, which runs in the
time domain for both training and generation and supports time-varying systems and
irregular sampling [REF:s5]. The Linear Recurrent Unit (LRU) shows that a plain linear
diagonal recurrence with a stable exponential parameterization, eigenvalues initialized
near the unit circle and a normalization factor recovers deep-SSM performance, and that the
eigenvalue distribution at initialization decides whether long-range dependencies can be
learned [REF:lru]. Mamba makes Δ, B and C functions of the input (selection), which turns the
layer into a time-varying, gated recurrence computed by a hardware-aware scan [REF:mamba].
Mamba's own ablations report a "no free lunch" result relevant here: on uniformly sampled,
continuous signals such as raw audio, selection "significantly hampers performance" and
linear time-invariant layers near the input work best, while complex-valued states help on
continuous modalities [REF:mamba]. On the theory side, linear diagonal recurrences followed
by nonlinear projections are universal approximators of regular causal sequence maps
[REF:orvieto-universality], and complex parameterizations are provably more expressive than
real ones of the same dimension [REF:ranmilo2024]. Whether these conclusions, drawn from
models of 10⁵ parameters and more, hold at a few hundred parameters is untested [inferred].

### 5.3 Recurrent versus convolutional mode

Because a time-invariant SSM layer can be computed either as a convolution or as a
recurrence, S4-family models train in the parallel convolutional view and switch to the
recurrent view for generation, where S4 needs constant memory and computation per step
[REF:s4]. Two consequences matter for an autoregressive task such as MC2. First, the
recurrent view exposes instabilities that the convolution hides: the SaShiMi authors found
that trained S4 state matrices generally became non-Hurwitz, a problem that only arises
when Ā is repeatedly applied during autoregressive generation, and proposed constraining
the real part, for example through an exponential parameterization [REF:sashimi]; S4D's
Re(A) < 0 constraint serves the same purpose [REF:s4d]. Second, the convolutional view
allows forecasting without feedback: S4's time-series experiments concatenate the whole
context with a masked forecast window and read all future outputs in one pass, which beat
Informer and other baselines on 40 of 50 settings and was better on the longest setting of
each task [REF:s4]. Mamba [REF:mamba] and S5
[REF:s5] instead use a parallel scan, which also supports input-dependent and time-varying
dynamics but, like the convolution, requires the inputs to be known in advance [inferred].
A model applied as a convolution over a fixed recent window, as in the MC2 baseline layout,
is equivalent to a recurrence whose state is reset at the start of the window, and so has
finite memory, like the temporal convolutional networks studied in system identification
[REF:andersson2019] [inferred].

### 5.4 State-space and recurrent models in system identification

System identification has adopted deep SSMs as simulators of physical systems. Bonassi et
al. show that a stack of linear SSM layers with static nonlinearities is a deep Wiener
model, incrementally input-to-state stable when its layers are Schur stable, and that S4,
S5 and LRU models trained by free-run simulation-error minimization perform "in line with"
LSTM and temporal-convolution baselines on the Silverbox benchmark with far fewer learnable
parameters; the LSTM is better in interpolation, while the best SSM (S5) is better over the
full test set including extrapolation [REF:bonassi2024]. Training a
recurrent simulator requires an initial state for each training subsequence. Forgione et
al. find that for a fading-memory benchmark a zero or random initial state is competitive if
enough steps are spent estimating it, but that for a system with an integrator a learned
state estimator was needed, which the authors conjecture is due to the integrator
[REF:forgione2022]. The deep subspace encoder (SUBNET) learns that estimator
explicitly, mapping past inputs and outputs to the initial state before simulating forward,
and, for stable data-generating systems, recommends a simulation length of "a few times the
largest characteristic time scale", noting that shorter lengths overfit [REF:subnet].
Hysteresis has persistent memory, so this guideline transfers only approximately
[inferred]. "Deep state-space model" in this
literature often means a probabilistic latent recurrent model rather than an S4-type layer
[REF:gedon2021]. The diagonal-SSM view also connects to the organizers' circuit models: the
linear material model of the MC2 analytical tutorial, dB/dt = r·H + μ·dH/dt, is a
first-order linear ODE driven by dB/dt [REF:mc2-tutorial2], which is the form of a single
real-valued SSM mode, and a deep S4D stack is a series of such linear-mode banks with static
nonlinearities between them [REF:s4d] [REF:bonassi2024] [inferred]. The same mapping gives
an SSM a continuous-time parameterization that can be resampled to a different time step
[REF:s4] [REF:bonassi2024], relevant to MC2's time-step-agnostic requirement but untested on
it [inferred].

### 5.5 Neural models of magnetic hysteresis, and SSMs in magnetics

Neural networks have modeled hysteresis for almost three decades. Early feed-forward models
were built to satisfy the memory properties of rate-independent hysteresis [REF:serpico1998]
or to model quasi-static loops [REF:makaveev2001]; later hybrids combined play operators
with a recurrent network to capture rate-dependent hysteresis in iron, reaching an NRMSE
below 0.7 % from six training loops and generalizing to unseen ramp rates [REF:grech2020].
On the MagNet data, sequence models first appeared as steady-state predictors: the How
MagNet LSTM encoder–decoder and the MagNet-AI LSTM and Transformer (section 3.1), and MC1
entries built on Transformers, LSTMs, CNNs and vision Transformers [REF:ojpel-mc1].

A second line puts physical or continuous-time structure into the network. Chandra et al.
update recurrent hidden states with the ordinary differential equations of
phenomenological hysteresis models and report generalization to first-order reversal
curves and minor loops where conventional recurrent networks struggle [REF:hystrnn] (results
as in the arXiv preprint, which is the version read); the
same group, and Guo et al., use neural operators (DeepONet, Fourier and wavelet neural
operators) for unseen reversal curves, minor loops and dynamic hysteresis, including a
rate-independent operator meant to transfer across sampling rates [REF:chandra-no]
[REF:guo-no2025] [REF:guo-fno2026]. In the Chandra et al. study, recurrent baselines (RNN,
LSTM, GRU and an encoder–decoder LSTM) could not predict unseen reversal curves, but its
data were generated by a Preisach-based model under quasi-static excitation, so the result
does not transfer directly to dynamic measured data [REF:chandra-no] [inferred]. The MC2 organizers' own tutorial on analytical methods
treats the material as a circuit, dy/dt = f(y, dB/dt) with H = g(y, dB/dt), and reports on
one N87 data file that a two-parameter linear R–L model reaches 46 % RMS field error, a
ten-parameter nonlinear circuit 18 %, and a small neural ODE [REF:neural-ode] 13 %
[REF:mc2-tutorial2]. The neural-ODE parameter count is not stated on the slides; from the
code it is a few hundred [inferred, computed from code]. Its training grows the prediction
horizon from a short fraction of a period to a full period, a curriculum the slides
justify by the slow and unstable training a randomly initialized model would otherwise
give [REF:mc2-tutorial2] [REF:magnetic-ode-code]. Neural controlled differential equations
[REF:neural-cde], in which an input path steers the hidden state, match the MC2 setting of B(t)
as a control [inferred]. Among the MC2-era papers, PI-HNO, from the University of Sydney and
UTS, combines a local recurrent branch, a Preisach-like attention branch and a soft
open-path energy penalty in 4,777 parameters [REF:pi-hno]. Under its own 14-material
protocol, which is not the official A–E test, it reports 13.40 % mean and 31.39 % p95
sequence error; a 4,758-parameter GRU baseline reaches 18.44 % / 46.17 %, and a
re-implementation of the 2,399-parameter MC2 demo LSTM 106.31 % / 282.36 % [REF:pi-hno].
The parameter count equals Sydney's MC2 leaderboard entry, which suggests the same model
line, but the paper does not name the team [inferred]. Kirchgässner fits a gain-scheduled
equivalent circuit, a nonlinear inductor with six viscous play branches and two after-effect
paths, end to end by backpropagation through time, and reports a mean sequence error of
7.33 % with 32 parameters per material against 7.77 % for the performance-track-winning GRU
across 15 MC2 materials [REF:k26]. These are means over test segments, not leaderboard
p95 values.

Taken together, this literature suggests three recurring themes [inferred]. Plain recurrent
networks fit the training distribution but generalize poorly to unseen excitation paths
[REF:hystrnn], and in PI-HNO's comparison the demo LSTM and a Transformer baseline reach
p95 errors of 282 % and 930 % on transient forecasts [REF:pi-hno]; the paper does not
attribute this to rollout, so that reading is ours [inferred]. Successful small
models add structure: physics-shaped states [REF:hystrnn] [REF:mminn] [REF:pi-hno], an
intermediate B–H representation [REF:hardcore], play operators [REF:grech2020] [REF:k26],
or careful state initialization [REF:rhino]. And waveform and loss errors can disagree, so
a model tuned for one may lose on the other [REF:magnet-ai]. In MC2 the evidence on physics
structure is split: the classical operators the performance-track winner evaluated
alongside its GRU did not help [REF:rhino], while an end-to-end-fitted circuit matched the
GRU on the official sequence error [REF:k26].

Structured SSMs are almost absent from this literature. On the machine-learning side, Hu et
al. report that Mamba ranks among the best of twelve models as a neural operator for
dynamical systems, including on extrapolation, at the lowest computational cost
[REF:hu2024] [abstract only]. The one SSM-for-hysteresis paper we found is outside
magnetics: a Mamba inverse model of a piezoelectric actuator with rate-dependent hysteresis,
used as a feedforward controller, which tracks better than a rate-dependent
Prandtl–Ishlinskii model and a NARMA network [REF:chen2026-piezo] [abstract only]. Searches of
arXiv, Crossref and the web for S4, S5, LRU or Mamba models of magnetic core loss or B–H
behavior returned no relevant results, and the MC2 performance-track winner's code base, which tests GRU,
LSTM, Jiles–Atherton, Preisach and neural-ODE variants, contains no SSM [REF:rhino-code]
(searches run 2026-11-06; queries listed in `lit_notes/ssm_and_training.md` §2.10). This is
an absence-of-evidence statement, and the paper should word it as "to our knowledge".

## 6. Training for long-horizon autoregressive rollout

### 6.1 Exposure bias

A model trained to predict the next step from true history, and then run on its own
predictions, sees at test time inputs drawn from a distribution it never trained on. In
sequence generation this mismatch is known as exposure bias, a term popularized by
Ranzato et al. [REF:exposure-bias]; scheduled sampling had described the same
train–test discrepancy months earlier [REF:scheduled-sampling]. In imitation
learning, Ross et al. show that a learner with one-step error ε under the expert's state
distribution can incur error growing as T²ε over T steps under its own distribution, a bound
that is tight, and that training on the states the learner itself visits (DAgger) reduces
the growth to linear [REF:dagger]. System identification draws the same line between
series-parallel (one-step, measured output fed back) and parallel (simulation, own output
fed back) training, and finds parallel training generally better with realistic noise and
measured data, at similar cost, while noting that series-parallel training is less likely
to be trapped in poor local solutions and parallelizes better [REF:ribeiro2018]; the latter
is one argument for one-step training as an initializer for a rollout stage [inferred]. The organizers' MC2 baselines are
series-parallel by construction: trained on measured H windows, then rolled out for up to
900 steps (section 3.4). Under PI-HNO's protocol a re-implementation of the 2,399-parameter
demo LSTM reaches 282 % p95 sequence error [REF:pi-hno] (the paper does not say how the
baseline was rolled out), and the entries known to predict
one step at a time and roll out score 32–75 % seq-p95 on the official test (section 4.9)
[REF:mc2-repo]. Those comparisons across teams are correlational, and a tuned one-step
model can do much better than the baselines (Tianjin, about 27 % on A, B, D and E), so
across teams exposure bias is a candidate explanation for weak MC2 rollouts, not an
established one [inferred]. For our own models the within-sequence curves are more direct:
at a first-step error equal to Tianjin's, our error doubles within the first ~10 fed-back
steps while Tianjin's does not (section 8) [measured, by our team].

### 6.2 Noise injection

The cheapest remedy is to corrupt the training history so that the model learns to recover
from imperfect inputs. DART injects noise into the demonstrator during data collection and
argues that the noise level should approximate the error of the trained policy, with a
closed form for Gaussian noise [REF:dart]. In learned physics simulation, Sanchez-Gonzalez
et al. train graph networks on one-step targets only and identify noise injection as one of
the main determinants of long-term rollout quality; random-walk noise worked best among the
noise types they tried, rollout accuracy peaked at an intermediate noise scale while
one-step accuracy fell with noise, and checkpoints were selected on full-length rollouts of
held-out trajectories [REF:gns]. The authors add, as an informal remark rather than an
experiment, that short rollouts during training "may not generalize to longer sequences as
well as noise injection" [REF:gns]; this is a different question from whether a rollout
loss adds to noise injection, which is what our stage 2 tests (section 8). The evidence is not uniform: on one PDE task, Brandstetter et al. found Gaussian
input noise worse than none, improving stability at the expense of accuracy [REF:mp-pde].
Noise benefits are therefore task dependent [inferred].

### 6.3 Scheduled sampling and related curricula

Scheduled sampling feeds the model its own previous prediction with a probability that
grows during training; always sampling from the start "yielded very poor performance", and
a variant that fed uniformly random tokens instead of the model's own predictions, in effect
input corruption, was better than the baseline but worse than the schedule
[REF:scheduled-sampling]. Huszár shows that the
resulting objective is improper and the learning algorithm inconsistent [REF:huszar2015],
and Professor Forcing instead matches the training and sampling dynamics adversarially
[REF:professor-forcing]. Generalized teacher forcing, developed for chaotic dynamics,
replaces the model state during training with a convex combination of the model state and a
data-inferred state, annealing the mixing weight toward free running, which keeps gradients
bounded [REF:gtf]. State injection in the MC2 performance-track winner, where the true H overwrites one
hidden unit during the warm-up [REF:rhino], is the hard, warm-up-only form of the same idea
[inferred].

### 6.4 Rollout and multi-step losses

The most direct remedy is to train on the rollout itself. In PDE surrogates, the pushforward
trick unrolls two steps and backpropagates only through the last, giving much longer stable
rollouts, while temporal bundling predicts several future steps per call and so reduces the
number of feedback events [REF:mp-pde]; a later study found that pushforward did not help in
its setting, and that giving the model more history inputs improved one-step accuracy but
worsened rollouts [REF:pde-refiner]. List et al. compare one-step training with unrolled
training without and with gradients through the unroll: in prediction setups, where the
network alone advances the state, unrolling without gradients improved accuracy by 41 % and
full gradients by a further 30 %, and curricula were necessary for long unrolls. The paper
summarizes the gradient-free variant as best for smaller networks and full gradients as
better for larger ones; per size the small-network results are mixed (on its
Kuramoto–Sivashinsky task, gradient-free unrolling is clearly best in three of six
configurations of about 2,000 parameters or fewer, about equal to full gradients in two, and one-step
training is best in one). The networks span about 500 to over a million parameters, so the
small end is the same order as MC2's small models [REF:list2025]. In system identification, the smoothness of the
simulation-error loss can degrade exponentially with the simulation length for
non-contractive models, which multiple shooting or truncated sections control
[REF:ribeiro2020] [REF:subnet]. Gradients through long recurrences stay bounded for systems
that settle to equilibria or cycles and diverge for chaotic ones [REF:mikhaeil2022], and
truncated backpropagation through time is biased toward short-term dependencies
[REF:tallec2017]. For SSMs specifically, models trained from a zero hidden state extend
poorly to longer sequences, and changing the state initialization lets models trained on
short contexts generalize to long ones [REF:longssm] [abstract only].

Within MC2, the training horizons differ widely. The performance-track winner trains closed-loop over
128–156 steps after a 28-step warm-up, with full gradients and no teacher forcing, and is
tested on horizons up to 900 [REF:rhino] [REF:rhino-code]. The post-challenge circuit model
trains over the full 1,000-step test windows with the test's warm-up lengths [REF:k26]. The
organizers' neural-ODE tutorial grows the prediction horizon from a tenth of a period to a
full period [REF:mc2-tutorial2] [REF:magnetic-ode-code]. Silicon Austria Labs used a 20-step
rollout loss [REF:mc2-talks] [transcript]. Most top entries avoid the question by not
feeding predictions back at all (section 4.9).

## 7. Small-data and cross-material transfer

Transfer across materials has been part of the MagNet program from the start. How MagNet
uses transfer learning to reduce the data needed per material [REF:how-magnet], and in MC1
Bristol's LSTM used transfer learning and TU Delft won an innovation award for multi-material
transfer learning with multi-objective optimization [REF:ojpel-mc1]. MC2 made data scarcity
explicit with Material A, which has 105 training sequences [REF:mc2-webinar6].

Teams handled A in three ways. The MC2 performance-track winner used no transfer: a 325-parameter GRU trained
only on A's data, for 10,000 epochs with closed-loop rollouts, reaches 23.27 % seq-p95,
close to its 20.43 % mean over A–E [REF:rhino] [REF:mc2-repo]. Bristol adapted each of its ten
known-material models briefly to the target, fine-tuned the best on target validation, and
augmented A by interpolating "available historical segments with 500 and 900 samples" to
full length (the paper does not say where these segments come from); Bristol placed third
on A (18.23 %) [REF:maglearn2] [REF:mc2-repo]. Tsinghua pretrained
on groups of similar known materials and fine-tuned per target, reporting that a group
matched pretraining on all materials at a fifth of the time [REF:mc2-talks] [transcript].
Hefei's design application supports fine-tuning from a base model with frozen LSTM layers,
but whether its competition model for A used it is not established [REF:hefei-code].
Hefei is best on A (12.62 %) [REF:mc2-repo]. No team reports a controlled
comparison with and without transfer on A [inferred, winner-others]. The post-challenge
circuit model offers a different view of what transfers: refitting only its five
measurement-chain parameters to recordings of the same materials on other cores recovers two
thirds of the gap to a full refit, and its dynamic parameters "carry over almost unchanged"
[REF:k26]. The circuit trails the GRU on 3C92 (Material A), which the author attributes to
too few training sequences per frequency to pin down the saturation knee at 70 °C
[REF:k26].

Outside MC2, three lines of work bear on RQ2. For the scalar MC1 task, MIST trains one
2,784-parameter model jointly on five materials with material identity injected through
FiLM conditioning and reweights the scarce material. Without any fine-tuning it reports a
lower p95 on the scarce material (12.38 ± 0.92 % over five seeds) than single-material
training (20.39 %) or sequential pretrain-then-fine-tune transfer (15.90 %), and in a
separate single-seed test fine-tuning on the scarce material made it worse; pooling the materials
without identity was an order of magnitude worse than a matched joint model [REF:mist]. The same
paper reports that a nominal seed is not reproducible across two equivalent implementations,
which argues for multi-seed reporting [REF:mist]. In system identification, Forgione et al.
adapt a nominal recurrent model to changed dynamics with an additive correction that is
linear in the model's parameter-Jacobian features, or a recurrent neural-tangent-kernel
Gaussian process when data are very few [REF:forgione-adapt] [abstract only], and in-context
meta-models trained on synthetic systems from a class predict a new system from its
input–output context [REF:forgione-icl] [abstract only]. Generic time-series foundation
models such as Chronos [REF:chronos] and TimesFM [REF:timesfm] forecast a series from its own
past and do not take an exogenous driving input, so they do not fit the B-driven MC2 task as
posed [inferred]. The MC2 performance-track winner evaluates no cross-material modeling and names transfer
learning to data-poor materials as future work [REF:rhino]; its repository contains, without
published results, code that pretrains one GRU on the pooled evaluation materials and
fine-tunes it per material, and a zero-shot variant that pairs pooled weights with the target
material's normalization constants [REF:rhino-code]. Zero-shot transfer of a transient B–H
model to an unseen material, which is RQ2, therefore has no published MC2 baseline that we
found [inferred].

## 8. Gap and positioning of our work

**The gap.** Sections 2–7 leave three things open. First, structured (S4-family)
state-space models have not, to our knowledge, been applied to magnetic B–H or H(t)
prediction. No MC2 entry with a public method description uses one (section 4.9), the
performance-track winner's code base tests recurrent, classical and neural-ODE variants but no SSM
[REF:rhino-code], and literature searches found a structured SSM (Mamba) for hysteresis only
in a piezoelectric actuator [REF:chen2026-piezo] (section 5.5). Classical nonlinear
state-space models of hysteresis do exist in system identification [REF:noel2017]
[REF:esfahani2018] [REF:krikelis2024], and the Jiles–Atherton model has been written in
state-space form for control [REF:zhao2024], so the claim must always name the structured,
S4-family class. This is a statement about what we could find, not a proof of absence. Second, the SSM literature
gives reasons to expect SSMs to suit the task: their linear modes are the discretized form
of the first-order circuit models the organizers use for magnetic materials
[REF:mc2-tutorial2] [REF:s4d] [inferred], they come with a continuous-time step that can in
principle be changed [REF:s4], and in system identification they perform in line with LSTMs with fewer
parameters [REF:bonassi2024]. But the evidence is from models of 10⁵ parameters and more, or
from benchmarks other than hysteresis (section 5.2). Third, at the small end of MC2 the
comparison between model families is thin. The small-model frontier, in official
seq-p95, is set by structured or physics-shaped models: Georgia Tech's 83-parameter
mechanical analogy (38.76 %), SiegenPaderborn's 325-parameter GRU (20.43 %), Cambridge's
1,182-parameter DeepONet (37.01 %) and, after the challenge, a 32-parameter equivalent circuit
(19.43 %) [REF:mc2-repo] [REF:k26]. Matched-size comparisons exist for GRU against LSTM, both
trained closed-loop with state injection, where the GRU is slightly more
parameter-efficient [REF:rhino], and for a plain GRU against PI-HNO at about 4,800
parameters on a non-official protocol [REF:pi-hno]. Size–accuracy sweeps on the transient
task also exist: for the GRU from 49 to 3,589 parameters, partly on the official test
materials [REF:rhino], for Hefei's model [REF:hefei] [REF:mc2-talks] [transcript], for
MagLearn 2 from 84,000 parameters up [REF:maglearn2], and for the circuit model at 32, 656
and 5,552 parameters [REF:k26]. What we did not find is any SSM, or any non-recurrent
sequence model, evaluated at a few hundred to about a thousand parameters with multiple
seeds under the official A–E metric, or a matched comparison of an SSM against a recurrent
baseline on that protocol.

**Our approach.** We study a deliberately small diagonal state-space model, S4D
[REF:s4d], built on the S4 framework [REF:s4], for the MC2 next-step formulation. Following
the official baseline, the model sees an 80-step window of past B and past H, the next B
value and the temperature, and predicts the next H; at test time it is rolled out
autoregressively, feeding each prediction back into its H window for up to 900 steps
[REF:mc2-repo]. The window is treated as a two-channel sequence (B, H), lifted pointwise
to d_model channels and passed through residual blocks of pre-LayerNorm, a diagonal SSM
convolution with a skip term, GELU and GLU channel mixing. Each channel has d_state/2
complex-conjugate modes with S4D-Lin initialization, a learned step size and zero-order-hold
discretization, following the minimal S4D reference implementation [REF:s4d]
[REF:s4-repo]. The SSM is applied as a causal depthwise convolution in real arithmetic, so
it runs on CPU and on a laptop GPU (Apple MPS). A small Tanh MLP reads the last-step
features together with the next B and the temperature. We train two sizes, 205 parameters
(d_model 4, one layer, state 8) and 1,001 parameters (d_model 8, two layers, state 16),
and compare them with the 771-parameter dual-LSTM architecture of the official baseline
[REF:mc2-repo], retrained under the same recipe because the published baseline weights
lack their normalization file.

Training has two stages. Stage 1 minimizes teacher-forced next-step mean-squared error, with
Gaussian noise added to the H history window to expose the model to imperfect fed-back
inputs, an instance of the noise-injection remedy for exposure bias [REF:gns] [REF:dart]
(section 6.2). Stage 2 fine-tunes with a K = 32 step autoregressive rollout loss,
backpropagated through the whole rollout, with the horizon ramped from 1 to K, that is,
parallel (simulation-error) training with a horizon curriculum [REF:ribeiro2018]
[REF:list2025] (section 6.4); it is kept only if it beats stage 1 on the validation rollout.
Checkpoints are selected by the 95th-percentile rollout error on held-out validation
segments, never on the test set. All scores below are the organizers' official metric on
the official A–E test set (section 3.4), computed by a scorer that reproduces four teams'
published per-material scores to four decimals.

**What our results support.** Four findings are established, with their limits.

First, next-step fitting is not the bottleneck. With next-step training alone, the
next-step training error is small (normalized MSE 1e-4 to 6e-3) while mean seq-p95 is
163–690 % across our three models, and H-history noise alone reduces the error by 2.4×
(LSTM) to 10× (1,001-parameter S4D), single seed each. Adding the multi-step stage improves
the official score further in a controlled single-seed comparison (S4D-205: 61.0 % to 55.5 %
mean seq-p95 at the same noise level and seed).

Second, within-sequence error curves on our final three-seed models show where the remaining
error arises. On the forecasts with 100 known steps (p95 over sequences, mean over A, B, D,
E), our first-step error (15.7–17.8 %) matches that of Tianjin, the competing entry with the
most similar one-step autoregressive scheme (16.7 %), but ours roughly doubles within the
first ~10 fed-back steps (31–41 %, against 16 % for Tianjin and 6 % for Hefei). Between 41 %
and 61 % of our final-window gap to Hefei is present by step 10, and the rest accrues over
the forecast: we add 22–38 points between the first and last 100-step windows, at or above
the closed-loop and autoregressive teams (14–29 points) and far above the
non-autoregressive ones (2–5 points) [measured, by our team, on the saved test predictions
and the teams' submissions [REF:mc2-repo]]. One-step accuracy is therefore not our gap. We
read the early doubling, which distinguishes us from Tianjin at equal one-step error, as
compounding of fed-back errors, an exposure-bias effect (section 6.1): **a main error
source**, though not the whole story, since about half of the gap accrues after step 10
[inferred, from a single comparison with Tianjin]. Neither H-history noise nor the
K = 32 rollout stage prevented it. The comparison with Tianjin suggests that the cause lies in
how errors re-enter our model rather than in the autoregressive scheme as such, but it is a
single comparison, and Tianjin also differs in architecture, loss and augmentation.

Third, and this is our defensible headline for RQ1, **under an identical data, training,
selection and scoring recipe, the S4D beats the matched dual-LSTM with fewer parameters**:
over three seeds, the 205-parameter S4D scores 52.4 ± 3.3 % mean seq-p95 against
64.0 ± 5.9 % for the 771-parameter LSTM, better on 14 of 15 material–seed pairs (exact paired
sign-flip test, p = 0.001). In the within-sequence curves the S4D's advantage is mainly
early: in the first 100 predicted steps the S4Ds are at 34–38 % p95 against the LSTM's 46 %;
in the last 100, S4D-205 still leads (61 % against 68 %), while S4D-1,001's mean (72 %) is
worse, driven by one seed [measured, by our team].

Fourth, S4D size does not matter in this range: the 1,001-parameter S4D scores
50.8 ± 5.1 %, not distinguishable from the 205-parameter one (p = 0.61).

**What our results do not support.** We are not competitive with the top MC2 teams. Our
best seed-mean, 50.8 % seq-p95, would rank 19th among the 32 leaderboard entries plus our
three models.
The small-model frontier is well ahead: SiegenPaderborn reaches 20.43 % with 325 parameters,
about 2.5× better, Cambridge 37.01 % with 1,182 and Georgia Tech 38.76 % with 83, while the
most accurate entry, Hefei, reaches 9.17 % with 6,564 parameters (all seq-p95, mean over
A–E) [REF:mc2-repo]; after the challenge, a 32-parameter circuit reported 19.43 %
[REF:k26]. The S4D advantage
over the LSTM is on the waveform metric only: the LSTM has the lowest mean energy-p95
(14.9 % vs 16.0 % and 19.6 %). Material A, with 105 training sequences (98 in our training split), is the weakest
material for every model (92–102 % seq-p95). Only one training recipe was tuned, the noise
sweep and the phase 1–2 experiments are single-seed, the attribution of the early doubling
to how errors re-enter the model rests on one comparison team, validation does not always rank test
outcomes, and RQ2 (cross-material generalization) has not yet been run. Our results are in
`docs/RESEARCH_LOG.md` sections 5.2–5.6.

**Positioning.** The defensible contribution is narrow. To our knowledge this is the first
structured-SSM (S4-family) model for magnetic B–H / H(t) prediction and the first in the
MagNet Challenge, and the first matched comparison of an SSM against a recurrent baseline on
the official MC2 protocol: same data split, training recipe, checkpoint selection and
scorer, three seeds, and a paired test. Related work to cite alongside this claim: the Mamba
model of piezoelectric hysteresis [REF:chen2026-piezo], classical nonlinear state-space
hysteresis identification [REF:noel2017] [REF:krikelis2024], and the closest non-S4 relative
on MC2, the 32-parameter difference-equation circuit [REF:k26]. The result, that a
205-parameter S4D beats the 771-parameter dual LSTM, is consistent with the
system-identification finding that SSMs perform in line with LSTMs with fewer parameters
[REF:bonassi2024]
[inferred], though that LSTM was trained under a different recipe from ours. A further contribution is
methodological: the scorer is verified against the organizers' published per-material
scores, and the paper reports per-seed variance and paired tests, which a single leaderboard
submission per team cannot show (a related paper reports that even nominally identical
seeds are not reproducible across implementations [REF:mist]). The within-sequence
analysis is a third contribution: on the official test it separates one-step error,
early compounding and long-horizon accumulation, for our models and for fourteen
re-scored teams [measured, by our team], which no MC2 source we found reports. It also
qualifies the RQ1 result, since the S4D's advantage over the LSTM is concentrated in the
early part of the forecast.

Three limits must be stated with it. First, the comparison is within one training regime,
one-step prediction with H-history noise and a short rollout stage, which the MC2 evidence
associates with the weaker entries (section 4.9); it does not show that an S4D would beat
a recurrent model trained closed-loop or a non-autoregressive model, and among recurrent
models the LSTM may be the weaker baseline, since the performance-track winner found a GRU slightly more
parameter-efficient [REF:rhino]. Second, the training remedies, noise injection and
multi-step rollout losses, are standard (section 6), so those findings are confirmations
in a new domain rather than new methods, and they did not stop our early compounding. The
autoregressive scheme alone does not explain our level either: Tianjin's one-step
autoregressive dual LSTM, the entry closest to our layout, reaches about 27 % on A, B, D and
E [REF:mc2-repo] [transcript] and does not compound early [measured, by our team]; what
separates us from it lies in how the model is trained or built, which we have not isolated
[inferred]. Third, the absolute accuracy is
about 2.5× behind the small-model frontier in seq-p95 and behind the LSTM in energy error,
so the work is a controlled architecture study, not a competitive entry. RQ2 is the larger
open opportunity: no MC2 paper we found evaluates zero-shot transfer to an unseen material
(section 7).

## 9. Implications for our project (internal; not for the paper)

Ranked by expected impact relative to effort, combining the four notes files and one
measurement on our own predictions. "Expected effect" is the researchers' or the writer's
judgment [inferred] unless it is plain arithmetic.

**Diagnosis (measured; RESEARCH_LOG §5.6).** reviewer-teams ran the within-sequence test of
`lit_notes/other_teams.md` §1.2 on our 45 saved seeds3 test predictions
(`~/magnet-data/runs/seeds3/seed{1,2,3}/…/Result/`, which reproduce
`results/seeds3/final_eval.csv` exactly), and main and winner-others re-ran it with identical
values; it is now committed as `src/experiments/drift_analysis.py`, with outputs in
`results/drift/` [measured]. On the
100-known-step block (the same sequences throughout, 900 predicted steps), p95 over
sequences, mean over A, B, D, E, seed means, each window's RMS error divided by RMS(H_meas)
over the whole predicted region (%):

| model | step 1 | steps 1–10 | steps 51–100 | window 1 (steps 1–100) | window 9 (steps 801–900) | added, window 1 → 9 |
|---|---|---|---|---|---|---|
| S4D-205 | 17.8 | 31.8 | 41.5 | 37.7 | 61.2 | +23.4 |
| S4D-1,001 | 15.7 | 31.1 | 37.1 | 34.0 | 72.4 | +38.4 |
| Dual-LSTM-771 | 17.8 | 40.5 | 50.9 | 46.0 | 67.6 | +21.6 |
| Tianjin (one-step AR, closest analog) | 16.7 | 16.0 | 18.2 | 17.2 | 34.5 | +17.4 |
| SiegenPaderborn (closed-loop GRU) | 4.7 | 8.4 | 12.7 | 10.5 | 24.6 | +14.0 |
| Hefei (rank 1, non-AR) | 6.8 | 5.9 | 7.8 | 6.8 | 11.2 | +4.4 |

What this shows [measured, reviewer-teams]:

- *One-step accuracy is not our gap.* Our first-step error (15.7–17.8 %) matches Tianjin's
  (16.7 %), the entry closest to our inference scheme.
- *Our error doubles within the first ~10 fed-back steps* (31–41 %), while Tianjin's does
  not (16.0 %).
- *Neither remedy we used stopped the early doubling*: not H-history noise (which alone gave
  the 2.4–10× gain, RESEARCH_LOG §5.3) and not the K = 32 rollout stage (RESEARCH_LOG §5.4,
  e.g. 61.0 → 55.5 % for S4D-205 at σ 0.05; kept in 34 of 45 seeds3 runs).
- *The S4D's advantage over the LSTM is mainly early.* In window 1 the S4Ds are at 34.0 and
  37.7 % against the LSTM's 46.0 %. By window 9 S4D-205 still leads by about 6 points
  (61.2 vs 67.6 %), but S4D-1,001's mean (72.4 %) is worse than the LSTM's, driven by seed 3
  alone (107.0 %; the other two seeds end at 59.0 and 51.3 %).

Our reading [inferred, from a single comparison with Tianjin]: fast compounding of fed-back
errors in the first ~10 steps (exposure bias) is **a main error source**, and it is what
distinguishes us from Tianjin at equal one-step error. It is not the whole story: about half
of our window-9 gap to Hefei is present by step 10 (52 % S4D-205, 41 % S4D-1,001, 61 %
LSTM), and the rest accrues over the remaining steps. Our post-step-10 growth (+29 points
for S4D-205) exceeds Tianjin's (+18). Our growth ratio (1.47–2.13×) looks mild only because
window 1 is already high. In absolute terms we add 22–38 points over the forecast (S4D-205
+23.4, S4D-1,001 +38.4, LSTM +21.6), at or above the closed-loop and autoregressive teams
(SiegenPaderborn +14.0, Tianjin +17.4, NITC +28.9) and far above the non-autoregressive ones
(Hefei +4.4, Bristol +5.4, Southeast1 +4.2) [measured, reviewer-teams]. The Tianjin
comparison suggests that the cause lies in how errors re-enter our model rather than in the
autoregressive scheme as such, but Tianjin also differs in architecture, loss and
augmentation.

**Material A** is the other priority: it alone is 43 % of our summed per-material gap to
SiegenPaderborn (68.3 of 160.0 points); each 10 points cut on A lowers our A–E mean by 2
[computed].

The ranking follows from the diagnosis. Items 1, 2, 5 and 6 act on how errors re-enter the
model, and item 3 is needed for the long-horizon variants of item 1; items 4 and 7 target
Materials A and C; items 8 and 9 are the paper's baselines and RQ2; items 10–12 act mainly
on per-step accuracy and the error tail and are demoted, because one-step accuracy is not our
gap.

1. **Reduce early error compounding, and long-horizon accumulation, by training the way we
   evaluate.** This targets both measured components of our gap and has the most cross-team
   support (section 4.9). Three routes, in the order to try them:
   - (a) *Remove the feedback path: a non-autoregressive S4D.* Run the causal S4D over the
     whole 1,000-step sequence with channels (B, ΔB, Δ²B, H·mask, mask) and a sequence loss
     over the predicted region, with no feedback of predicted H. Sources: S4's masked
     forecasting recipe [REF:s4]; Hefei's whole-sequence LSTM with a known-H mask
     [REF:hefei-code]; the non-autoregressive teams that use the known H have the smallest
     growth, +2 to +5 points over 900 steps (section 4.9). It removes exposure bias by
     construction, leaves the parameter count unchanged, and is cheap in convolution mode
     over L = 1,000 [inferred]. Needs item 3.
   - (b) *Train against the feedback: closed-loop over a longer horizon.* Warm a carried
     state up over the given history, then free-run; raise the rollout horizon from K = 32
     to 128–320 with a staged curriculum (e.g. 1 → 8 → 32 → 128 → 320), make the closed-loop
     stage the main stage rather than a 12-epoch fine-tune, and ideally end at the full test
     condition (1,000-step windows, warm-ups 100 / 500 / 900). Sources: [REF:rhino]
     (128–156 steps, full BPTT, 1,500–10,000 epochs; SiegenPaderborn adds only 14 points
     over 900 steps), [REF:k26] (full test windows), [REF:subnet] (horizon a few times the
     dominant timescale; one excitation period is 20–320 samples at 16 MHz, so K = 32 is
     shorter than one period in 5 of the 7 frequency buckets, 50–320 kHz [computed]),
     [REF:ribeiro2018], [REF:list2025] (compare full gradients against no gradient through
     the feedback, which the paper reports as better for small networks, with mixed
     per-size results), [REF:mc2-tutorial2] (horizon curriculum). Cost: stage-2 compute
     grows about 4–10× [computed from the K ratio].
   - (c) *Up-weight the first ~10 predicted steps in the rollout loss*, where our error
     doubles, so that the curriculum in (b) or our current K = 32 stage does not dilute them
     [inferred; no source tests this]. Cheapest of the three; can be combined with (b).
   - *Contrast case: study Tianjin's training.* Tianjin uses our inference scheme
     (one-step prediction fed back through an 80-step window, as we read its captions), has
     the same first-step error, and does not double early [measured, reviewer-teams]. Its
     known recipe comes only from its presentation [REF:mc2-talks] [transcript]: separate
     LSTM encoders for B and the H history with cross-attention and a 1D-CNN on the rate of
     change of B; sign-flip symmetry augmentation with frequency- and temperature-balanced
     sampling; an absolute error near H = 0 and a relative error elsewhere; Optuna tuning;
     and the remark that tuning the architecture for one-step accuracy caused "error
     explosion" in rollout while tuning the loss and learning rate did not. Which of these
     matters is unknown. Cheap diagnostics on our side [inferred]: the steps-1–10 curve for
     stage 1 alone against stage 1 + stage 2 (stage 2 was kept in 34 of 45 seeds3 runs,
     `results/seeds3/final_eval.csv`); the same curve across σ = 0, 0.05, 0.1 from the
     earlier runs; and the sensitivity of each prediction to the most recent fed-back H
     values (input gradients), since a model that leans on the last H, as Silicon Austria
     Labs found its model did [REF:mc2-talks] [transcript], would compound this way.
   - Expected effect: high. It may not close the gap alone, since Tianjin reaches about 27 %
     on A, B, D and E without it [REF:mc2-repo] [transcript]. Cost: low for (c) and the
     diagnostics, medium for (a) and (b).
2. **Change how errors re-enter: anchor H instead of feeding back an 80-step H window** (if
   staying autoregressive).
   Cheap test: keep the 80-step B window, give the model only the last 1–8 H values; the
   winner found one injected H value nearly as good as a long history on three of the four
   materials with a warm-up [REF:rhino], and
   more history inputs can improve one-step accuracy while worsening rollouts
   [REF:pde-refiner]. MagLearn 2's occlusion study cuts both ways: error growth turns rapid
   only when the last few samples are removed, but keeping error under 20 % needed roughly
   35–690 samples [REF:maglearn2], so sweep the H-window length rather than assume. Larger change:
   overwrite one recurrent output channel with the true H during warm-up (state injection
   applied to an SSM) [REF:rhino]. Expected effect: possibly high on early compounding,
   since the fed-back H window is where it enters; uncertain. Cost: low for the window
   test.
3. **Re-initialize the SSM timescales for a carried state.** Needed for 1(b) and for a
   non-AR model over 1,000 steps. Set Δ ∈ [1e-3, 1e-1] (dependency lengths of about
   20–2,000 steps at Re(A) = −1/2), since our [0.01, 0.1] matches only the 80-step window
   [computed]; keep the ZOH Δ·B scaling; optionally one Δ per state. Do not make B trainable
   for accuracy: with C trained per mode, B is redundant in the kernel (only B∘C enters), so
   it would add 32 parameters to S4D-205 (+16 %) for an optimization-only effect measured on
   ~100,000-parameter models [REF:s4d] [computed]. Sources: [REF:hippo-train], [REF:lru],
   [REF:s4d], [REF:s5]. Initialize chunk states from a warm-up or encoder, never from zero
   [REF:longssm] [REF:forgione2022] [REF:rhino]. Our Re(A) = −exp(·) parameterization is
   already safe for recurrent mode [REF:sashimi]. Expected effect: enabling (moot inside an
   80-step window). Cost: low.
4. **Material A: train longer on all its data, then try transfer.** First, A-specific long
   closed-loop or non-AR training on all of A's 105 sequences (98 in our training split), as the winner did (23.27 %
   seq-p95 on A, no transfer, 10,000 epochs) [REF:rhino]; this tests whether our A
   failure is the training regime rather than data scarcity [inferred]. Then transfer: a
   joint model over materials with FiLM material identity and an upweighted A, without and
   with fine-tuning [REF:mist]; base-model selection by short target-validation runs, then
   full fine-tuning [REF:maglearn2]; grouped-material pretraining [REF:mc2-talks]; or a
   linear Jacobian-feature correction on A only [REF:forgione-adapt]. Augment A by sign-flip
   symmetry, (B, H) → (−B, −H), as Tianjin did to double its data (whether it helps A is
   untested) [REF:mc2-talks] [transcript] [inferred], and by resampling shorter
   segments to full length [REF:maglearn2]. Expected effect: high on A; directly serves RQ2.
   Cost: low for the first step, medium for transfer.
5. **Better noise, if staying autoregressive.** Replace i.i.d. Gaussian H noise with
   random-walk or low-pass noise [REF:gns], set its scale from the measured validation
   rollout error per material [REF:dart], and try σ > 0.1, the top of our sweep. Treat it as
   an ablation: noise was worse than none on one PDE task [REF:mp-pde]. Alternatively, use an
   annealed mix of true and predicted H as the curriculum [REF:gtf]. H-history noise alone gave
   our largest measured gain (2.4–10×, RESEARCH_LOG §5.3), yet early compounding remains,
   which suggests that i.i.d. noise of σ ≤ 0.1 does not match the correlated errors our model
   actually feeds back [inferred]. Obsolete under 1(a).
6. **Temporal bundling and inference-time calibration** (if staying autoregressive).
    Predict 8–16 future H per call from the known future B, cutting the number of feedback
    events [REF:mp-pde]; in NITC's own ablation, using fewer than its 16 predicted steps per
    call deviated more from the ground truth [REF:mc2-talks] [transcript], although NITC still
    has the steepest within-sequence growth we measured (3.7×) [measured, reviewer-teams],
    though from a low start (10.7 %) and with little growth over its first 10 steps (5.3 %
    p95) [measured, reviewer-teams]. Bundling cuts the number of feedback events in exactly
    the early region where ours compound. Calibrate the predicted H offset to the last known
    H at inference [REF:mc2-talks] [transcript]. Cost: low.
7. **Material C: train under the test condition.** Train C with a single known H value,
   left-padded exactly as at test time [REF:rhino] [REF:hefei-code]; our training windows
   always carry true H history while C's test starts near H = 0 (RESEARCH_LOG §2), so the
   model never sees its test input in training [inferred]. The organizers allow assuming a
   past H of 0 A/m [REF:mc2-webinar6], but a zero H history is not a demagnetized state:
   Georgia Tech's fixed zero state failed on C [REF:mc2-talks] [transcript]. Expected effect: moderate on C (ours 47–57 %, the winner's
   27 %, the best 12–17 %). Cost: low.
8. **Fair baselines and counting for RQ1.** Re-implement the winner's GRU in our harness at
    209 and 1,029 parameters (their count 3h² + 16h + 5 at h = 6 and 16), close to our 205
    and 1,001, as a strong size-matched baseline under identical data and scoring
    [REF:rhino] [computed]. Count our normalization statistics as parameters, as the
    winner does, or state the difference; parameter counts for the same model differ across
    papers (325 vs 336) [REF:k26]. Report mean SRE alongside p95, and compare our p95 only
    with p95 values. Add the 32-, 656- and 5,552-parameter circuit points to the
    accuracy-vs-size plot, labeled with their own counting convention [REF:k26]. Re-test S4D
    size after items 1 and 3: if larger S4Ds then help, our 205 ≈ 1,001 tie was a training
    artefact [inferred]. Enlarge the 200-segment validation set, given the selection noise
    we observed (RESEARCH_LOG §5.2, §5.4). Cost: medium. Value for the paper: high.
9. **RQ2 baselines and protocol.** Cheap zero-shot baseline: the winner's per-material GRU
    weights with the target material's max-abs normalizer (their repository's
    "frankenstein" notebook; weights for all ten known materials are public)
    [REF:rhino-code]. Report the S4D under the circuit paper's frequency-extrapolation
    restriction (train ≤ 320 kHz, test 500 / 800 kHz), where its GRU8 replica, trained with
    the circuit paper's own pipeline and described there as weaker than the published
    network, failed (54.5 % vs 10.2 %) [REF:k26]. Try adapting only the input/output affine layers or normalization
    on a new material with the SSM core frozen, since in the circuit model most
    cross-specimen difference sits in a few measurement and amplitude parameters [REF:k26]
    [inferred]. Synthetic Preisach or Jiles–Atherton pretraining is a further option
    [REF:forgione-icl] [inferred]. Cost: medium.
10. **Metric-aligned loss.** Replace MSE on z-scored H, which over-weights large-amplitude
   sequences, with the per-sequence relative loss of the winner (RMSE weighted by |ΔB| and
   divided by the source sequence's RMS(H); either term alone beat MSE) [REF:rhino], or with
   the evaluation metric itself (sequence + energy error) [REF:k26]. Variants to ablate:
   quantile-trimmed MSE, intended to improve p95 but untested on that score [REF:maglearn2]
   [REF:mc2-talks] [transcript]; smooth-L1 with peak emphasis
   [REF:mc2-talks] [transcript]; an open-path energy penalty [REF:pi-hno], which should also
   help our weaker energy error (RESEARCH_LOG §5.5). Beware pure relative losses near H = 0,
   which made one team's model predict H ≈ 0 [REF:mc2-talks] [transcript]. Expected effect:
   moderate, mainly on p95 and energy; demoted because one-step accuracy is not our gap.
   Cost: low.
11. **Inputs and conditioning.** Add ΔB and Δ²B as input channels (central differences)
   [REF:rhino] [REF:pi-hno] [REF:hefei-code]; a linear skip from (B, ΔB, Δ²B) to the output
   (about three parameters) [REF:mc2-talks] [transcript]; feed the future B we are given
   (our model reads only the window plus B_next) [REF:pi-hno]; and inject T, and the warm-up
   position, into the initial state rather than only at the head; in PI-HNO's single-run
   ablation on its own protocol, moving them from the initial state to per-step inputs
   raised p95 from 31.4 % to 75.7 % [REF:pi-hno]. In the circuit model, the
   second-order correction of the measured B was the most important single element (+5.3
   points of mean error when removed) [REF:k26]. Expected effect: moderate; mostly per-step
   accuracy, so demoted, except that conditioning the initial state may also stabilize the
   first rollout steps [inferred]. Cost: low.
12. **Bound the output.** A max-abs-normalized H target with a tanh or clamp, so a drifting
   rollout cannot run away into the p95 tail [REF:rhino] [inferred]; or a structured
   direction-dependent operator head [REF:hefei-code], to be ablated, since its effect versus
   a plain head is not shown in any source we read. Cost: low.
13. **Low priority.** Scheduled sampling (improper objective [REF:huszar2015]); pushforward
    (helped in [REF:mp-pde], not in [REF:pde-refiner]); Mamba-style selection (hurt on
    uniformly sampled continuous signals in Mamba's own ablation [REF:mamba]); diffusion
    inpainting (about 1,000 network calls per sequence [REF:mc2-talks]).

## 10. References

Listed in order of first citation, by citation key. The keys become numbers [n] in the final
step. Notes in parentheses record how much of each source the team read; entries without a
note were read in full (papers) or inspected directly (code, slides, data).

- `mc2-talks`: MagNet Challenge 2 team presentations (five-minute talks), Feb. 9–12, 2026, Princeton Power Electronics YouTube channel. Cited via auto-generated captions: Hefei https://www.youtube.com/watch?v=-erftMlwTno; Fuzhou https://www.youtube.com/watch?v=pHzgWAF_6Co; Sydney https://www.youtube.com/watch?v=8Y3p8dFIU-Q; Georgia Tech https://www.youtube.com/watch?v=0-705NtutoE; Cambridge https://www.youtube.com/watch?v=2AkU2jc_aUU; Bristol https://www.youtube.com/watch?v=W7GEu5IXt68; Southeast1 https://www.youtube.com/watch?v=6REL90As66U; Eindhoven https://www.youtube.com/watch?v=6pHZcLsHtgQ; Tsinghua https://www.youtube.com/watch?v=bW8KJCIxRa8; Silicon Austria Labs https://www.youtube.com/watch?v=asrbcj6cSSk; NITC https://www.youtube.com/watch?v=nFvaGZiIaTA; Tianjin https://www.youtube.com/watch?v=Ddc0bGuUC10; KU Leuven https://www.youtube.com/watch?v=PYCW11p2ENk.
- `mc2-repo`: MagNet 2025 Organizing Team (Princeton Univ. and Dartmouth College), "MagNet Challenge 2," GitHub repository, https://github.com/minjiechen/magnetchallenge-2, commit ab4913c (Aug. 3, 2026). README: task statement, judging criteria, News #13 "Preliminary ranking of 32 teams" (leaderboard), News #16 and #17; tutorial and pretest notebooks; links to the final-evaluation data and all teams' submitted results.
- `mc2-winners`: MagNet 2025 Organizing Team, "MagNet Challenge 2 Winners (March 10th, 2026)," announcement image, README News #16, https://github.com/minjiechen/magnetchallenge-2/blob/main/img/winners.jpg.
- `how-magnet`: H. Li, D. Serrano, T. Guillod, S. Wang, E. Dogariu, A. Nadler, M. Luo, V. Bansal, N. K. Jha, Y. Chen, C. R. Sullivan, and M. Chen, "How MagNet: Machine learning framework for modeling power magnetic material characteristics," *IEEE Trans. Power Electron.*, vol. 38, no. 12, pp. 15829–15853, Dec. 2023, doi: 10.1109/TPEL.2023.3309232.
- `why-magnet`: D. Serrano, H. Li, S. Wang, T. Guillod, M. Luo, V. Bansal, N. K. Jha, Y. Chen, C. R. Sullivan, and M. Chen, "Why MagNet: Quantifying the complexity of modeling power magnetic material characteristics," *IEEE Trans. Power Electron.*, vol. 38, no. 11, pp. 14292–14316, Nov. 2023, doi: 10.1109/TPEL.2023.3291084.
- `ojpel-mc1`: M. Chen, H. Li, S. Wang, T. Guillod, D. Serrano, *et al.* (132 authors), "MagNet Challenge for data-driven power magnetics modeling," *IEEE Open J. Power Electron.*, vol. 6, pp. 883–898, 2025 (published online Sep. 30, 2024), doi: 10.1109/OJPEL.2024.3469916.
- `mc2-handbook`: IEEE PELS, Princeton Univ., and Dartmouth College, "2025 PELS MagNet Challenge 2 (MagNet 2025): 'from Steady State Losses to Transient Hysteresis!'," competition handbook, 2025 (header reads "Updated on January 30th, 2024"), https://github.com/minjiechen/magnetchallenge-2/blob/main/docs/Handbook-2025.pdf.
- `mc2-tutorial1`: S. Wang, H. Kwon, H. Li, T. Guillod, M. Chen, and C. R. Sullivan, "MagNet Challenge 2: 'From steady-state to transient models!'," Tutorial Session 1 (data-driven methods), May 16, 2025, https://github.com/minjiechen/magnetchallenge-2/blob/main/tutorials/tutorial-1/Slides-Tutorial-1.pdf.
- `compel2025`: S. Wang, H. Kwon, D. Grigoryan, H. Li, T. Guillod, C. R. Sullivan, and M. Chen, "Unified time domain foundation models for hysteretic passive components," in *Proc. IEEE Workshop on Control and Modeling for Power Electronics (COMPEL)*, Knoxville, TN, USA, 2025, pp. 1–8, doi: 10.1109/COMPEL57166.2025.11121278. (Abstract only read.)
- `s4d`: A. Gu, A. Gupta, K. Goel, and C. Ré, "On the parameterization and initialization of diagonal state space models," in *Proc. NeurIPS*, 2022, doi: 10.52202/068431-2607, arXiv:2206.11893.
- `steinmetz`: C. P. Steinmetz, "On the law of hysteresis," *Trans. AIEE*, vol. IX, no. 1, pp. 1–64, Jan. 1892, doi: 10.1109/T-AIEE.1892.5570437; reprinted in *Proc. IEEE*, vol. 72, no. 2, pp. 197–221, Feb. 1984, doi: 10.1109/PROC.1984.12842.
- `mse`: J. Reinert, A. Brockmeyer, and R. W. A. A. De Doncker, "Calculation of losses in ferro- and ferrimagnetic materials based on the modified Steinmetz equation," *IEEE Trans. Ind. Appl.*, vol. 37, no. 4, pp. 1055–1061, 2001, doi: 10.1109/28.936396. (Metadata only.)
- `igse`: K. Venkatachalam, C. R. Sullivan, T. Abdallah, and H. Tacca, "Accurate prediction of ferrite core loss with nonsinusoidal waveforms using only Steinmetz parameters," in *Proc. IEEE Workshop on Computers in Power Electronics (COMPEL)*, 2002, pp. 36–41, doi: 10.1109/CIPE.2002.1196712. (Abstract only read.)
- `i2gse`: J. Mühlethaler, J. Biela, J. W. Kolar, and A. Ecklebe, "Improved core-loss calculation for magnetic components employed in power electronic systems," *IEEE Trans. Power Electron.*, vol. 27, no. 2, pp. 964–973, Feb. 2012, doi: 10.1109/TPEL.2011.2162252. (Abstract only read.)
- `cwh`: C. R. Sullivan, J. H. Harris, and E. Herbert, "Core loss predictions for general PWM waveforms from a simplified set of measured data," in *Proc. IEEE Applied Power Electronics Conf. (APEC)*, 2010, pp. 1048–1055, doi: 10.1109/APEC.2010.5433375. (Abstract only read.)
- `igcc`: T. Guillod, J. S. Lee, H. Li, S. Wang, M. Chen, and C. R. Sullivan, "Calculation of ferrite core losses with arbitrary waveforms using the composite waveform hypothesis," in *Proc. IEEE Applied Power Electronics Conf. (APEC)*, 2023, pp. 1586–1593, doi: 10.1109/APEC43580.2023.10131348. (Abstract only read.)
- `stenglein`: E. Stenglein and T. Dürbaum, "Core loss model for arbitrary excitations with DC bias covering a wide frequency range," *IEEE Trans. Magn.*, vol. 57, no. 6, 2021, doi: 10.1109/TMAG.2021.3068188. (Abstract only read.)
- `bertotti-loss`: G. Bertotti, "General properties of power losses in soft ferromagnetic materials," *IEEE Trans. Magn.*, vol. 24, no. 1, pp. 621–630, 1988, doi: 10.1109/20.43994. (Metadata only.)
- `pi-hno`: Y. Zhu, Q. Huang, S. Li, Y. Li, G. Lei, and J. Zhu, "A physics-informed hybrid neural operator for transient magnetization prediction in power magnetics," arXiv:2608.02965v3, Aug. 2026 (preprint prepared for possible submission to *IEEE Trans. Power Electron.*).
- `k26`: W. Kirchgässner, "A data-driven, gain-scheduled equivalent circuit for the B–H behavior of power ferrites," arXiv:2611.02222 [eess.SY], Nov. 1, 2026 (preprint, not peer reviewed); code: https://github.com/wkirgsn/ferrite-bh-circuit.
- `preisach`: F. Preisach, "Über die magnetische Nachwirkung," *Z. Phys.*, vol. 94, no. 5–6, pp. 277–302, May 1935, doi: 10.1007/BF01349418. (Metadata only.)
- `mayergoyz`: I. D. Mayergoyz, "Generalized scalar Preisach models of hysteresis," in *Mathematical Models of Hysteresis*. New York, NY, USA: Springer, 1991, pp. 64–140, doi: 10.1007/978-1-4612-3028-1_2. (Metadata only.)
- `moree-preisach`: G. Mörée and M. Leijon, "Review of play and Preisach models for hysteresis in magnetic materials," *Materials*, vol. 16, no. 6, art. 2422, 2023, doi: 10.3390/ma16062422.
- `mayergoyz-dyn`: I. D. Mayergoyz, "Dynamic Preisach models of hysteresis," *IEEE Trans. Magn.*, vol. 24, no. 6, pp. 2925–2927, 1988, doi: 10.1109/20.92290. (Metadata only.)
- `bertotti-dynpreisach`: G. Bertotti, "Dynamic generalization of the scalar Preisach model of hysteresis," *IEEE Trans. Magn.*, vol. 28, no. 5, pp. 2599–2601, Sep. 1992, doi: 10.1109/20.179569. (Abstract only read.)
- `roussel`: R. Roussel *et al.*, "Differentiable Preisach modeling for characterization and optimization of particle accelerator systems with hysteresis," *Phys. Rev. Lett.*, vol. 128, no. 20, art. 204801, 2022, doi: 10.1103/PhysRevLett.128.204801. (Metadata only.)
- `ja1984`: D. C. Jiles and D. L. Atherton, "Theory of ferromagnetic hysteresis (invited)," *J. Appl. Phys.*, vol. 55, no. 6, pp. 2115–2120, 1984, doi: 10.1063/1.333582. (Metadata only.)
- `ja1986`: D. C. Jiles and D. L. Atherton, "Theory of ferromagnetic hysteresis," *J. Magn. Magn. Mater.*, vol. 61, no. 1–2, pp. 48–60, 1986, doi: 10.1016/0304-8853(86)90066-1. (Metadata only.)
- `ja-param1989`: D. C. Jiles and J. B. Thoelke, "Theory of ferromagnetic hysteresis: Determination of model parameters from experimental hysteresis loops," *IEEE Trans. Magn.*, vol. 25, no. 5, pp. 3928–3930, 1989, doi: 10.1109/20.42480. (Abstract only read.)
- `ja-param1992`: D. C. Jiles, J. B. Thoelke, and M. K. Devine, "Numerical determination of hysteresis parameters for the modeling of magnetic properties using the theory of ferromagnetic hysteresis," *IEEE Trans. Magn.*, vol. 28, no. 1, pp. 27–35, 1992, doi: 10.1109/20.119813. (Abstract only read.)
- `rhino`: H. Vater, O. Schweins, L. Hölsch, W. Kirchgässner, T. Piepenbrock, and O. Wallscheid, "RHINO-MAG: Recursive H-field inference based on observed magnetic flux density under dynamic excitation," arXiv:2603.29745 [eess.SY], v1 Mar. 31, 2026, v2 Sep. 9, 2026 (v2 read); accepted to *IEEE Trans. Power Electron.*, early access, doi: 10.1109/TPEL.2026.3733311.
- `marion2008`: R. Marion, R. Scorretti, N. Siauve, M.-A. Raulet, and L. Krähenbühl, "Identification of Jiles–Atherton model parameters using particle swarm optimization," *IEEE Trans. Magn.*, vol. 44, no. 6, pp. 894–897, Jun. 2008, doi: 10.1109/TMAG.2007.914867. (Title only.)
- `trapanese2011`: M. Trapanese, "Identification of parameters of the Jiles–Atherton model by neural networks," *J. Appl. Phys.*, vol. 109, no. 7, art. 07D355, 2011, doi: 10.1063/1.3569735. (Title only.)
- `zirka2012`: S. E. Zirka, Y. I. Moroz, R. G. Harrison, and K. Chwastek, "On physical aspects of the Jiles–Atherton hysteresis models," *J. Appl. Phys.*, vol. 112, no. 4, art. 043916, 2012, doi: 10.1063/1.4747915. (Abstract only read.)
- `jiles1994`: D. C. Jiles, "Frequency dependence of hysteresis curves in conducting magnetic materials," *J. Appl. Phys.*, vol. 76, no. 10, pp. 5849–5855, 1994, doi: 10.1063/1.358399. (Abstract only read.)
- `zhu1996`: J. G. Zhu, S. Y. R. Hui, and V. S. Ramsden, "A generalized dynamic circuit model of magnetic cores for low- and high-frequency applications—Part I," *IEEE Trans. Power Electron.*, vol. 11, no. 2, pp. 246–250, 1996, doi: 10.1109/63.486172; S. Y. R. Hui, J. G. Zhu, and V. S. Ramsden, "…—Part II," *ibid.*, pp. 251–259, doi: 10.1109/63.486173. (Metadata only.)
- `luo2018`: M. Luo, D. Dujic, and J. Allmeling, "Modeling frequency independent hysteresis effects of ferrite core materials using permeance–capacitance analogy for system-level circuit simulations," *IEEE Trans. Power Electron.*, vol. 33, no. 12, pp. 10055–10070, 2018, doi: 10.1109/TPEL.2018.2809704. (Metadata only.)
- `cui2023`: H. H. Cui, S. Dulal, S. B. Sohid, G. Gu, and L. M. Tolbert, "Unveiling the microworld inside magnetic materials via circuit models," *IEEE Power Electron. Mag.*, vol. 10, no. 3, pp. 14–22, 2023, doi: 10.1109/MPEL.2023.3301408. (Metadata only.)
- `dulal2024`: S. Dulal *et al.*, "A physics-based circuit model for nonlinear magnetic material characteristics," in *Proc. IEEE Applied Power Electronics Conf. (APEC)*, 2024, pp. 396–401, doi: 10.1109/APEC48139.2024.10509097. (Metadata only.)
- `moree-review`: G. Mörée and M. Leijon, "Review of hysteresis models for magnetic materials," *Energies*, vol. 16, no. 9, art. 3908, 2023, doi: 10.3390/en16093908. (Abstract only read.)
- `mc2-tutorial4`: M. Chen, S. Wang, H. Kwon, H. Li, T. Guillod, and C. R. Sullivan, "MagNet Challenge 2: 'From steady-state to transient models!'," Tutorial Session 4, Jun. 6, 2025, https://github.com/minjiechen/magnetchallenge-2/blob/main/tutorials/tutorial-4/Slides-Tutorial-4.pdf.
- `magnetx-db`: H. Kwon, S. Wang, H. Li, Y. Elasser, G.-G. Kang, D. Zhou, D. Grigoryan, and M. Chen, "MagNetX: Extending the MagNet database for modeling power magnetics in transient," in *Proc. IEEE Applied Power Electronics Conf. (APEC)*, 2025, pp. 566–572, doi: 10.1109/APEC48143.2025.10977252; preprint TechRxiv, doi: 10.36227/techrxiv.173396064.40994860/v1. (Abstract read; details from the organizers' slides.)
- `mc2-tutorial2`: T. Guillod, "Soft magnetic materials and ordinary differential equations: From linear circuits to neural network models," IEEE MagNet Challenge webinar (Tutorial 2), Dartmouth College, May 23, 2025, https://github.com/minjiechen/magnetchallenge-2/blob/main/tutorials/tutorial-2/Slides-Tutorial-2.pdf.
- `mc2-keynote`: M. Chen and C. R. Sullivan, "MagNet Challenge: people · magnetics · AI," keynote slides, IEEE Applied Power Electronics Conf. (APEC), Mar. 2026, slide 16 ("MagNet Challenge 2 Winners"), https://github.com/minjiechen/magnetchallenge-2/blob/main/docs/apeckeynote.pdf.
- `mc2-webinar6`: H. Kwon, S. Wang, H. Li, T. Guillod, M. Chen, and C. R. Sullivan, "MagNet Challenge 2: 'From steady-state to transient models!'," Tutorial Session 6 (final evaluation), Oct. 30, 2025, https://github.com/minjiechen/magnetchallenge-2/blob/main/tutorials/tutorial-6/Webinar-6-2025.pdf.
- `magnet-ai`: H. Li, D. Serrano, S. Wang, and M. Chen, "MagNet-AI: Neural network as datasheet for magnetics modeling and material recommendation," *IEEE Trans. Power Electron.*, vol. 38, no. 12, pp. 15854–15869, Dec. 2023, doi: 10.1109/TPEL.2023.3309233.
- `hardcore`: W. Kirchgässner, N. Förster, T. Piepenbrock, O. Schweins, and O. Wallscheid, "HARDCORE: H-field and power loss estimation for arbitrary waveforms with residual, dilated convolutional neural networks in ferrite cores," *IEEE Trans. Power Electron.*, vol. 40, no. 2, pp. 3326–3335, Feb. 2025, doi: 10.1109/TPEL.2024.3488174. (Abstract only read.)
- `mminn`: Q. Huang, Y. Li, J. Zhu, and S. Li, "Magnetization mechanism-inspired neural networks for core loss estimation," *IEEE Trans. Power Electron.*, vol. 39, no. 12, pp. 16382–16390, Dec. 2024, doi: 10.1109/TPEL.2024.3450897. (Abstract only read.)
- `mc2-slides`: S. Wang, H. Kwon, H. Li, T. Guillod, M. Chen, and C. R. Sullivan (MagNet 2025 Organizing Team), "MagNet Challenge 2: 'From steady-state to transient models!'," kickoff meeting slides, 2nd IEEE Int. Challenge in Design Methods for Power Electronics, IEEE PELS, Feb. 25, 2025, slide 17 ("Evaluation Criteria"), https://github.com/minjiechen/magnetchallenge-2/blob/main/docs/Slides-2025.pdf.
- `magnetx-models`: S. Wang, H. Kwon, H. Li, Y. Elasser, G.-G. Kang, D. Zhou, D. Grigoryan, and M. Chen, "MagNetX: Foundation neural network models for simulating power magnetics in transient," in *Proc. IEEE Applied Power Electronics Conf. (APEC)*, 2025, pp. 2438–2445, doi: 10.1109/APEC48143.2025.10977420; preprint TechRxiv, doi: 10.36227/techrxiv.173396153.34393547/v1. (Abstract read; details from the organizers' slides.)
- `magnetx-tpel`: S. Wang, H. Kwon, H. Li, T. Guillod, H. Wouters, C. R. Sullivan, and M. Chen, "MagNetX: Data-driven time-domain foundation models for power magnetics in transient," *IEEE Trans. Power Electron.*, vol. 41, no. 11, pp. 19455–19474, Nov. 2026, doi: 10.1109/TPEL.2026.3685464. (Abstract only read.)
- `mc2-eval-script`: MagNet 2025 Organizing Team, "Evaluation_Read.m" (official MC2 scoring script) and A–E ground-truth data, in "MagNet Challenge 2 final evaluation data and all teams' results," released via the data link in `mc2-repo`, accessed Nov. 3, 2026.
- `hefei`: Y. Wang, Y. Liu, L. Mo, W. Wang, S. Ren, G. Cai, S. Liu, H. Wu, M. Ma, and C. Q. Jiang, "Framework of LSTM-based direction-dependent asymmetric operator model for B-H prediction under transient excitations," *IEEE Trans. Power Electron.*, vol. 41, no. 12, pp. 21230–21243, Dec. 2026, doi: 10.1109/TPEL.2026.3715566; preprint TechRxiv, doi: 10.36227/techrxiv.177162964.49399026/v1. (Abstract only read.)
- `hefei-code`: HFUT-CityUHK-XMU team, "Mag-LSTM-DAO-Inference" (competition model, weights and normalization), https://github.com/ouihon/Mag-LSTM-DAO-Inference, and "LSTM-DAO-Inference" (design application with fine-tuning script), https://github.com/ouihon/LSTM-DAO-Inference, GitHub repositories.
- `rhino-code`: H. Vater *et al.* (Paderborn Univ., Dept. LEA), "RHINO-MAG," GitHub repository (MIT license), https://github.com/upb-lea/RHINO-MAG, commit aec7e99 (Oct. 2, 2026); includes the organizers' per-block results for the submitted models.
- `maglearn2`: J. Yang, B. Cui, Y. Huo, L. Zhang, A. Martinez, S. Liu, and J. Wang, "MagLearn 2: High-fidelity, saturation-aware, and history-efficient sequence-to-sequence modeling of transient B-H behaviour under PWM excitation," arXiv:2610.03504, Oct. 2026 (submitted to *IEEE Trans. Power Electron.*); conference version: B. Cui, J. Yang, Y. Huo, L. Zhang, A. Martinez, S. Liu, and J. Wang, "MagLearn 2: A short-window Seq2Seq framework for transient modelling of power magnetics under PWM excitations," in *Proc. IPEC-Nagasaki 2026 (ECCE Asia)*, 2026, pp. 1–7, doi: 10.23919/IPEC-Nagasaki2026-EC64663.2026.11597291; code (Materials D and E): https://github.com/JunWang-Bristol/MagLearn2.
- `gatech`: P. Chauhan, Y. Mo, D. Li, L. Chang, X. Xian, and B. Ge, "Dynamic magnetic modeling with magnetization, hysteresis, eddy, and displacement currents," *IEEE Trans. Power Electron.*, vol. 41, no. 12, pp. 20869–20873, 2026 (letter), doi: 10.1109/TPEL.2026.3712771. (Abstract only read.)
- `gru`: K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, "Learning phrase representations using RNN encoder–decoder for statistical machine translation," in *Proc. EMNLP*, 2014, pp. 1724–1734, doi: 10.3115/v1/D14-1179, arXiv:1406.1078.
- `lstm`: S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Comput.*, vol. 9, no. 8, pp. 1735–1780, 1997, doi: 10.1162/neco.1997.9.8.1735.
- `rnn-gradients`: Y. Bengio, P. Simard, and P. Frasconi, "Learning long-term dependencies with gradient descent is difficult," *IEEE Trans. Neural Netw.*, vol. 5, no. 2, pp. 157–166, 1994, doi: 10.1109/72.279181.
- `pascanu2013`: R. Pascanu, T. Mikolov, and Y. Bengio, "On the difficulty of training recurrent neural networks," in *Proc. ICML*, PMLR vol. 28, no. 3, pp. 1310–1318, 2013, arXiv:1211.5063.
- `leontaritis1985`: I. J. Leontaritis and S. A. Billings, "Input-output parametric models for non-linear systems, Part I: Deterministic non-linear systems," *Int. J. Control*, vol. 41, no. 2, pp. 303–328, 1985, doi: 10.1080/0020718508961129; "Part II: Stochastic non-linear systems," *ibid.*, pp. 329–344, doi: 10.1080/0020718508961130. (Metadata only.)
- `narendra1990`: K. S. Narendra and K. Parthasarathy, "Identification and control of dynamical systems using neural networks," *IEEE Trans. Neural Netw.*, vol. 1, no. 1, pp. 4–27, 1990, doi: 10.1109/72.80202.
- `ribeiro2018`: A. H. Ribeiro and L. A. Aguirre, "'Parallel training considered harmful?': Comparing series-parallel and parallel feedforward network training," *Neurocomputing*, vol. 316, pp. 222–231, 2018, doi: 10.1016/j.neucom.2018.07.071, arXiv:1706.07119.
- `s4`: A. Gu, K. Goel, and C. Ré, "Efficiently modeling long sequences with structured state spaces," in *Proc. ICLR*, 2022, arXiv:2111.00396.
- `bonassi2024`: F. Bonassi, C. Andersson, P. Mattsson, and T. B. Schön, "Structured state-space models are deep Wiener models," *IFAC-PapersOnLine*, vol. 58, no. 15, pp. 247–252, 2024 (SYSID 2024), doi: 10.1016/j.ifacol.2024.08.536, arXiv:2312.06211.
- `dss`: A. Gupta, A. Gu, and J. Berant, "Diagonal state spaces are as effective as structured state spaces," in *Proc. NeurIPS*, 2022, arXiv:2203.14343.
- `hippo-train`: A. Gu, I. Johnson, A. Timalsina, A. Rudra, and C. Ré, "How to train your HiPPO: State space models with generalized orthogonal basis projections," in *Proc. ICLR*, 2023, arXiv:2206.12037.
- `s5`: J. T. H. Smith, A. Warrington, and S. W. Linderman, "Simplified state space layers for sequence modeling," in *Proc. ICLR*, 2023, arXiv:2208.04933.
- `lru`: A. Orvieto, S. L. Smith, A. Gu, A. Fernando, C. Gulcehre, R. Pascanu, and S. De, "Resurrecting recurrent neural networks for long sequences," in *Proc. ICML*, 2023, arXiv:2303.06349.
- `mamba`: A. Gu and T. Dao, "Mamba: Linear-time sequence modeling with selective state spaces," in *Proc. Conf. Language Modeling (COLM)*, 2024, arXiv:2312.00752.
- `orvieto-universality`: A. Orvieto, S. De, C. Gulcehre, R. Pascanu, and S. L. Smith, "Universality of linear recurrences followed by non-linear projections: Finite-width guarantees and benefits of complex eigenvalues," in *Proc. ICML*, 2024, arXiv:2307.11888. (Abstract only read.)
- `ranmilo2024`: Y. Ran-Milo, E. Lumbroso, E. Cohen-Karlik, R. Giryes, A. Globerson, and N. Cohen, "Provable benefits of complex parameterizations for structured state space models," in *Proc. NeurIPS*, 2024, arXiv:2410.14067. (Abstract only read.)
- `sashimi`: K. Goel, A. Gu, C. Donahue, and C. Ré, "It's raw! Audio generation with state-space models," in *Proc. ICML*, PMLR vol. 162, 2022, arXiv:2202.09729.
- `andersson2019`: C. Andersson, A. H. Ribeiro, K. Tiels, N. Wahlström, and T. B. Schön, "Deep convolutional networks in system identification," in *Proc. IEEE Conf. Decision and Control (CDC)*, 2019, doi: 10.1109/CDC40024.2019.9030219, arXiv:1909.01730. (Abstract only read.)
- `forgione2022`: M. Forgione, M. Mejari, and D. Piga, "Learning neural state-space models: Do we need a state estimator?" arXiv:2206.12928, 2022.
- `subnet`: G. I. Beintema, M. Schoukens, and R. Tóth, "Deep subspace encoders for nonlinear system identification," *Automatica*, vol. 156, art. 111210, 2023, doi: 10.1016/j.automatica.2023.111210, arXiv:2210.14816.
- `gedon2021`: D. Gedon, N. Wahlström, T. B. Schön, and L. Ljung, "Deep state space models for nonlinear system identification," *IFAC-PapersOnLine*, 2021, doi: 10.1016/j.ifacol.2021.08.406, arXiv:2003.14162. (Abstract only read.)
- `serpico1998`: C. Serpico and C. Visone, "Magnetic hysteresis modeling via feed-forward neural networks," *IEEE Trans. Magn.*, vol. 34, no. 3, pp. 623–628, 1998, doi: 10.1109/20.668055. (Abstract only read.)
- `makaveev2001`: D. Makaveev, L. Dupré, M. De Wulf, and J. Melkebeek, "Modeling of quasistatic magnetic hysteresis with feed-forward neural networks," *J. Appl. Phys.*, vol. 89, no. 11, pp. 6737–6739, 2001, doi: 10.1063/1.1361268. (Metadata only.)
- `grech2020`: C. Grech, M. Buzio, M. Pentella, and N. Sammut, "Dynamic ferromagnetic hysteresis modelling using a Preisach-recurrent neural network model," *Materials*, vol. 13, no. 11, art. 2561, 2020, doi: 10.3390/ma13112561. (Abstract only read.)
- `hystrnn`: A. Chandra, T. Kapoor, B. Daniels, M. Curti, K. Tiels, D. M. Tartakovsky, and E. A. Lomonova, "Generalizable models of magnetic hysteresis via physics-aware recurrent neural networks," *Comput. Phys. Commun.*, vol. 314, art. 109650, 2025, doi: 10.1016/j.cpc.2025.109650; preprint "Neural oscillators for magnetic hysteresis modeling," arXiv:2308.12002 (preprint read).
- `chandra-no`: A. Chandra, B. Daniels, M. Curti, K. Tiels, and E. A. Lomonova, "Magnetic hysteresis modeling with neural operators," *IEEE Trans. Magn.*, vol. 61, no. 1, pp. 1–11, Jan. 2025, doi: 10.1109/TMAG.2024.3496695, arXiv:2407.03261.
- `guo-no2025`: Z. Guo, B. H. Nguyen, H. Hamzehbahmani, and R. V. Sabariego, "Dynamic hysteresis model of grain-oriented ferromagnetic material using neural operators," *IEEE Trans. Magn.*, 2025, doi: 10.1109/TMAG.2025.3600089, arXiv:2504.04863. (Abstract only read.)
- `guo-fno2026`: Z. Guo, X. Shen, and R. V. Sabariego, "A multi-scale ResNet-augmented Fourier neural operator framework for high-frequency sequence-to-sequence prediction of magnetic hysteresis," arXiv:2604.04150, 2026. (Abstract only read.)
- `neural-ode`: R. T. Q. Chen, Y. Rubanova, J. Bettencourt, and D. Duvenaud, "Neural ordinary differential equations," arXiv:1806.07366, 2018. (Abstract only read.)
- `magnetic-ode-code`: T. Guillod, "magnetic_ode_models," GitHub repository (MPL-2.0), https://github.com/otvam/magnetic_ode_models.
- `neural-cde`: P. Kidger, J. Morrill, J. Foster, and T. Lyons, "Neural controlled differential equations for irregular time series," in *Proc. NeurIPS*, 2020, arXiv:2005.08926.
- `hu2024`: Z. Hu, N. A. Daryakenari, Q. Shen, K. Kawaguchi, and G. E. Karniadakis, "State-space models are accurate and efficient neural operators for dynamical systems," arXiv:2409.03231, 2024. (Abstract only read.)
- `chen2026-piezo`: S. Chen, T. Liu, W. Feng, W. Zhan, R. Ji, L. Wang, and K. Li, "Composite control of a piezoelectric fast steering mirror using a mamba-based dual-axis inverse model," *Smart Mater. Struct.*, vol. 35, no. 9, art. 095027, 2026, doi: 10.1088/1361-665X/aea30f. (Abstract only read.)
- `exposure-bias`: M. Ranzato, S. Chopra, M. Auli, and W. Zaremba, "Sequence level training with recurrent neural networks," in *Proc. ICLR*, 2016, arXiv:1511.06732.
- `dagger`: S. Ross, G. J. Gordon, and J. A. Bagnell, "A reduction of imitation learning and structured prediction to no-regret online learning," in *Proc. AISTATS*, 2011, arXiv:1011.0686.
- `dart`: M. Laskey, J. Lee, R. Fox, A. Dragan, and K. Goldberg, "DART: Noise injection for robust imitation learning," in *Proc. Conf. Robot Learning (CoRL)*, 2017, arXiv:1703.09327.
- `gns`: A. Sanchez-Gonzalez, J. Godwin, T. Pfaff, R. Ying, J. Leskovec, and P. W. Battaglia, "Learning to simulate complex physics with graph networks," in *Proc. ICML*, 2020, arXiv:2002.09405.
- `mp-pde`: J. Brandstetter, D. Worrall, and M. Welling, "Message passing neural PDE solvers," in *Proc. ICLR*, 2022, arXiv:2202.03376.
- `scheduled-sampling`: S. Bengio, O. Vinyals, N. Jaitly, and N. Shazeer, "Scheduled sampling for sequence prediction with recurrent neural networks," in *Proc. NeurIPS*, 2015, arXiv:1506.03099.
- `huszar2015`: F. Huszár, "How (not) to train your generative model: Scheduled sampling, likelihood, adversary?" arXiv:1511.05101, 2015. (Abstract only read.)
- `professor-forcing`: A. Lamb, A. Goyal, Y. Zhang, S. Zhang, A. Courville, and Y. Bengio, "Professor forcing: A new algorithm for training recurrent networks," in *Proc. NIPS*, 2016, arXiv:1610.09038. (Abstract only read.)
- `gtf`: F. Hess, Z. Monfared, M. Brenner, and D. Durstewitz, "Generalized teacher forcing for learning chaotic dynamics," in *Proc. ICML*, PMLR vol. 202, pp. 13017–13049, 2023, arXiv:2306.04406.
- `pde-refiner`: P. Lippe, B. S. Veeling, P. Perdikaris, R. E. Turner, and J. Brandstetter, "PDE-Refiner: Achieving accurate long rollouts with neural PDE solvers," in *Proc. NeurIPS*, 2023, doi: 10.52202/075280-2946, arXiv:2308.05732.
- `list2025`: B. List, L.-W. Chen, K. Bali, and N. Thuerey, "Differentiability in unrolled training of neural physics simulators on transient dynamics," *Comput. Methods Appl. Mech. Eng.*, vol. 433, art. 117441, Jan. 2025, doi: 10.1016/j.cma.2024.117441, arXiv:2402.12971.
- `ribeiro2020`: A. H. Ribeiro, K. Tiels, J. Umenberger, T. B. Schön, and L. A. Aguirre, "On the smoothness of nonlinear system identification," *Automatica*, vol. 121, art. 109158, 2020, doi: 10.1016/j.automatica.2020.109158, arXiv:1905.00820.
- `mikhaeil2022`: J. M. Mikhaeil, Z. Monfared, and D. Durstewitz, "On the difficulty of learning chaotic dynamics with RNNs," in *Proc. NeurIPS*, 2022, doi: 10.52202/068431-0821, arXiv:2110.07238. (Abstract only read.)
- `tallec2017`: C. Tallec and Y. Ollivier, "Unbiasing truncated backpropagation through time," arXiv:1705.08209, 2017. (Abstract only read.)
- `longssm`: S. Wang, "LongSSM: On the length extension of state-space models in language modelling," arXiv:2406.02080, 2024. (Abstract only read.)
- `mist`: C. Yao and C. Gong, "Cross-material support transfer for core-loss prediction under waveform covariate shift," arXiv:2609.31659, 2026.
- `forgione-adapt`: M. Forgione, A. Muni, D. Piga, and M. Gallieri, "On the adaptation of recurrent neural networks for system identification," *Automatica*, art. 111092, 2023, doi: 10.1016/j.automatica.2023.111092, arXiv:2201.08660. (Abstract only read.)
- `forgione-icl`: M. Forgione, F. Pura, and D. Piga, "From system models to class models: An in-context learning paradigm," *IEEE Control Syst. Lett.*, 2023, doi: 10.1109/LCSYS.2023.3335036, arXiv:2308.13380. (Abstract only read.)
- `chronos`: A. F. Ansari *et al.*, "Chronos: Learning the language of time series," *Trans. Mach. Learn. Res. (TMLR)*, 2024, arXiv:2403.07815. (Abstract only read.)
- `timesfm`: A. Das, W. Kong, R. Sen, and Y. Zhou, "A decoder-only foundation model for time-series forecasting," in *Proc. ICML*, 2024, arXiv:2310.10688. (Abstract only read.)
- `noel2017`: J. P. Noël, A. F. Esfahani, G. Kerschen, and J. Schoukens, "A nonlinear state-space approach to hysteresis identification," *Mech. Syst. Signal Process.*, vol. 84, pp. 171–184, Feb. 2017, doi: 10.1016/j.ymssp.2016.08.025, arXiv:1610.09138. (Metadata only.)
- `esfahani2018`: A. F. Esfahani, P. Dreesen, K. Tiels, J.-P. Noël, and J. Schoukens, "Parameter reduction in nonlinear state-space identification of hysteresis," *Mech. Syst. Signal Process.*, vol. 104, pp. 884–895, May 2018, doi: 10.1016/j.ymssp.2017.10.017, arXiv:1705.00178. (Metadata only.)
- `krikelis2024`: K. Krikelis, J.-S. Pei, K. van Berkel, and M. Schoukens, "Identification of structured nonlinear state–space models for hysteretic systems using neural network hysteresis operators," *Measurement*, vol. 224, art. 113966, Jan. 2024, doi: 10.1016/j.measurement.2023.113966. (Metadata only.)
- `s4-repo`: A. Gu *et al.*, "state-spaces/s4" (includes the minimal S4D reference implementation), GitHub repository (Apache-2.0), https://github.com/state-spaces/s4.
