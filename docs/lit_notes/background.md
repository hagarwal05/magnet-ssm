# Literature notes: domain background (core loss, classical hysteresis, MagNet, neural ODE / physics-informed, NN hysteresis)

Author: background-lit (literature-review team). Written 2026-11-06.

Scope split: MC2 team methods are in `winner-siegen` (`docs/lit_notes/siegen_paderborn.md`) and
`winner-others`. SSM and rollout-training literature is with `ssm-lit`. This file covers the
domain background only.

Conventions:

- **[read]**: full text read. **[abstract]**: only the abstract or metadata was available, so the
  claims are limited to it. **[slides]** / **[code]**: organizer slides or code, read directly.
  **[teammate]**: taken from another teammate's notes and not re-read by me.
- **[inferred]**: my own reasoning or arithmetic. It is not a claim made by the source.
- Numbers are copied from the source named next to them.
- Access limits: TechRxiv (Cloudflare) and most IEEE Xplore pages blocked automated access.
  Why MagNet, How MagNet and MagNet-AI are CC-BY and were read in full (author versions from
  IEEE). The MagNetX APEC/TPEL papers and the COMPEL 2025 paper were **not** read in full, only
  their abstracts (from Semantic Scholar, OpenAlex and Crossref) and the organizers' slides that
  describe the same work.
- Local copies: `/tmp/mc2lit/bg/` (OJPEL, Why/How MagNet, MagNet-AI PDFs and text),
  `/tmp/mc2lit/pdf/` (organizer slides: Handbook, kickoff, Tutorials 1/2/4, Webinars 5/6, APEC
  keynote, MagNetX README), `/tmp/mc2lit/t2/` (Guillod tutorial-2 code). The brief named
  `/tmp/mc2ref/`, which does not exist on this machine; the same files are in `/tmp/mc2lit/pdf/`.

---

## 1. Core-loss modeling: Steinmetz family and why it fails for transients

### 1.1 Original Steinmetz equation (SE)

- **C. P. Steinmetz, "On the law of hysteresis," Trans. AIEE, vol. 9, no. 1, pp. 1–64, 1892,
  doi:10.1109/T-AIEE.1892.5570437** (reprint: Proc. IEEE, vol. 72, no. 2, pp. 197–221, Feb. 1984,
  doi:10.1109/PROC.1984.12842). Cited as given in the OJPEL MC1 paper's reference list (refs
  [15], [32]); DOIs checked on Crossref. I did not read the original.
- Form, as written in the OJPEL MC1 paper (eq. 1): P_v = k · f^a · B_ac^b, with three fitted
  parameters k, a, b. **[read, OJPEL]**
- Known limits, from **Why MagNet** (Serrano et al. 2023, §III-A) **[read]**: "it is only
  applicable to sinusoidal excitations; it is not accurate across wide frequencies; and, its
  accuracy is largely influenced by the data and methods used to extract the Steinmetz
  parameters." Even for pure sinusoids on N87 with least-squares-fitted parameters, the
  prediction deviates from the measurements "with errors above 20% in certain B–f regions" (Fig. 8).
- *Why it matters:* this is the 3-parameter end of the size–accuracy spectrum. It outputs a
  scalar, cycle-averaged loss for a periodic excitation.

### 1.2 Non-sinusoidal extensions (MSE, iGSE, i²GSE, CWH, iGCC, Stenglein)

- **J. Reinert, A. Brockmeyer, R. W. A. A. De Doncker, "Calculation of losses in ferro- and
  ferrimagnetic materials based on the modified Steinmetz equation," IEEE Trans. Ind. Appl.,
  vol. 37, no. 4, pp. 1055–1061, 2001, doi:10.1109/28.936396.** Metadata only (Crossref).
- **K. Venkatachalam, C. R. Sullivan, T. Abdallah, H. Tacca, "Accurate prediction of ferrite core
  loss with nonsinusoidal waveforms using only Steinmetz parameters," Proc. IEEE Workshop
  Computers in Power Electronics (COMPEL), 2002, pp. 36–41, doi:10.1109/CIPE.2002.1196712.**
  **[abstract]** The iGSE. The abstract says it separates "a flux trajectory into major and minor
  loops via a new recursive algorithm" and needs only the standard Steinmetz parameters. The
  OJPEL paper gives the iGSE as P_v = (1/T)∫ k_i |dB/dt|^a (ΔB)^(b−a) dt, where ΔB is the
  peak-to-peak flux swing. It "is widely used in practice because most other models require
  parameters that are not usually given by manufacturers." **[read, OJPEL]**
- **J. Mühlethaler, J. Biela, J. W. Kolar, A. Ecklebe, "Improved core-loss calculation for magnetic
  components employed in power electronic systems," IEEE Trans. Power Electron., vol. 27, no. 2,
  pp. 964–973, Feb. 2012, doi:10.1109/TPEL.2011.2162252.** **[abstract]** The i²GSE. Losses
  continue during zero-voltage (constant-flux) intervals "due to relaxation processes", and the
  model adds relaxation terms. According to OJPEL it adds "five more parameters to the original
  three", giving 8, and "describing the complex behaviors ... with only eight parameters is often
  insufficient". **[read, OJPEL]**
- **C. R. Sullivan, J. H. Harris, E. Herbert, "Core loss predictions for general PWM waveforms from a
  simplified set of measured data," APEC 2010, pp. 1048–1055, doi:10.1109/APEC.2010.5433375.**
  **[abstract]** The composite waveform hypothesis (CWH): loss for rectangular-voltage
  waveforms is built up from square-wave measurements. Why MagNet tests the CWH on MagNet data
  and finds it "underestimated by 2.4% on average, with a maximum discrepancy of 6.4%" in its
  validation area (triangular, 25 °C, no bias). **[read]**
- **T. Guillod, J. S. Lee, H. Li, S. Wang, M. Chen, C. R. Sullivan, "Calculation of ferrite core
  losses with arbitrary waveforms using the composite waveform hypothesis," APEC 2023,
  pp. 1586–1593, doi:10.1109/APEC43580.2023.10131348.** **[abstract]** The iGCC extends the
  iGSE by applying the CWH with "a time-varying local equivalent frequency", aimed at extreme
  duty cycles.
- **E. Stenglein, T. Dürbaum, "Core loss model for arbitrary excitations with DC bias covering a
  wide frequency range," IEEE Trans. Magn., vol. 57, no. 6, 2021, doi:10.1109/TMAG.2021.3068188.**
  **[abstract]** Mn-Zn ferrite loss is modeled as quasi-static energy loss (a function of swing
  and dc bias) times an effective frequency, using "two frequency-independent parameters". It
  reports "an adequate accuracy of 15%".
- **G. Bertotti, "General properties of power losses in soft ferromagnetic materials," IEEE Trans.
  Magn., vol. 24, no. 1, pp. 621–630, 1988, doi:10.1109/20.43994.** Metadata only. This is the
  standard loss-separation reference (hysteresis / classical eddy / excess) and is cited by
  PI-HNO and the K26 circuit paper.
- Parameter counts, from **How MagNet** Table I **[read]**: SE 3, iGSE 3, i²GSE 8, "ML Neural
  Network ≫100".

### 1.3 Evidence that the Steinmetz family breaks down

- **Why MagNet** (MagNet N87 data) **[read]**:
  - Duty cycle: the 80 %/50 % duty-cycle loss ratio varies "by a factor ranging from nearly 1 to
    2" over f and B_ac. "This contradicts the results expected from iGSE if constant Steinmetz
    parameters are used" (Fig. 17b).
  - Relaxation in zero-voltage intervals: some models capture it, "but other models such as iGSE
    do not" (§III-E).
  - Coupling: "P_V(B_ac, f, H_dc, D, T) cannot be decoupled into any set of multiplicative factors
    ... if high accuracy across a wide range of operating conditions is desired." Instead, "losses
    should be defined as P_V(t) = f(B(t), T)" (§IV). This is effectively a call for a time-domain
    model, which is what MC2 asks for. [inferred link]
- **How MagNet** **[read]**: iGSE on N87 triangular waveforms (parameters fitted at 50 % duty)
  gives an average relative error of 14.6 % and a maximum of 61.1 %. That places it "between the
  small-scale and medium-scale neural network models" (21 and 109 params, Table III). Also,
  "the accuracy of equation-based models is inherently limited by their mathematical form."
- **MC1 white-box results** (OJPEL §II-C) **[read]**: the best equation-based entry was
  Mondragon's ci2GSE (tSE + iGSE + CWH + i²GSE): 60 parameters (15 per temperature × 4), about
  15 % average p95 error, and 93 % on material 79 "due to missing relaxation data in the
  training dataset". SEU-MC's vector magnetic circuit used about 60 parameters at similar
  accuracy.
- **APEC 2026 keynote** (M. Chen, C. R. Sullivan, "MagNet Challenge: people · magnetics · AI",
  `docs/apeckeynote.pdf`) **[slides]**: "iGSE often under/over-estimate core losses for PFC
  inductors by 30%~50%", footnoted "across different switching frequencies, amplitudes,
  waveforms, temperatures, for specific materials". It is a slide claim with no data shown.
- **MC2 Handbook (2025)** **[slides]**: "Despite several upgrades to the original SE (e.g., iGSE,
  i2GSE), these curve-fitting methods have limited accuracy." It also notes that loss heats the
  core, and the heating changes the Steinmetz parameters.

### 1.4 Why the Steinmetz family cannot address the MC2 task

- The SE family maps a periodic flux waveform (or its piecewise segments) to a **scalar
  time-averaged loss**. MC2 asks for the **H(t) waveform** over an arbitrary future segment
  (Handbook: H'(t) = f(B(t), H(t), B'(t), T)). No member of the family produces H(t).
  [inferred from the model forms above]
- MC2's energy metric integrates dB/dt·H over an **open** (non-closed) segment. The organizers
  themselves point out "(open loop): Energy exchanged ≠ total core loss (cycle not closed)"
  (Tutorial 4, slide 36) **[slides]**. A per-cycle loss law does not define this quantity.
  [inferred]
- PI-HNO (Zhu et al. 2026, §I Introduction) puts it directly: Steinmetz-type formulas "cannot reconstruct
  transient" B–H paths "under arbitrary excitation conditions." **[read, intro via HTML]**
- *Framing use:* MC1 asked whether something could beat Steinmetz. MC2 moves the target to
  Preisach/JA territory. The APEC keynote slide labels the two challenges "MC1: Steady-State Loss
  – Steinmetz Equations (100 yrs old)" and "MC2: Transient Dynamics – Preisach and J-A Models
  (50 yrs old)". **[slides]**

---

## 2. Classical hysteresis models: Preisach, Jiles–Atherton, and dynamic extensions

### 2.1 Preisach

- **F. Preisach, "Über die magnetische Nachwirkung," Z. Phys., vol. 94, no. 5–6, pp. 277–302, May
  1935, doi:10.1007/BF01349418.** Metadata via Crossref. The MC2 Handbook describes it as a
  weighted parallel sum of relay hysterons R_{α,β}, each a two-valued operator. **[slides]**
- **I. D. Mayergoyz, *Mathematical Models of Hysteresis*, Springer, 1991** (chapter "Generalized
  Scalar Preisach Models of Hysteresis," doi:10.1007/978-1-4612-3028-1_2). Metadata only.
- **G. Mörée, M. Leijon, "Review of play and Preisach models for hysteresis in magnetic materials,"
  Materials, vol. 16, no. 6, 2422, 2023, doi:10.3390/ma16062422 (PMC10051722).** **[read, PMC
  full text via fetch]** Also cited in Guillod's Tutorial 2.
  - Identification: "Measurements of first-order reversal curves are the most common way to
    identify the Preisach distribution" (§11.2). The Everett-function form avoids computing a
    weight distribution explicitly (§11.1).
  - Memory: Preisach models have "the wiping-out property and the congruency property" (§10.3).
    However, "real minor loops of hysteresis do not show this kind of congruency" (§10.3). Applying
    H-congruency to data uncritically "can create drifting curves and unclosed minor loops"
    (§11.2).
  - Scope: the review covers only "rate-independent hysteresis and not dynamic models that
    consider eddy currents" (§1). Nucleation-driven loops are "not suitable for the play model or
    the Preisach model" (§12.2).
- **G. Mörée, M. Leijon, "Review of hysteresis models for magnetic materials," Energies, vol. 16,
  no. 9, 3908, 2023, doi:10.3390/en16093908.** **[abstract]** Taxonomy: Duhem-type models (no
  stored turning points; JA, Coleman–Hodgdon, Tellinen) versus Preisach-type models, which store
  reversal points (classical Preisach, Prandtl–Ishlinskii). Ideal hysteresis is rate-independent,
  and dynamic versions add rate dependence. RHINO-MAG cites this review for JA minor-loop
  non-closure and drift. **[teammate]**
- **Parameter count:** a Preisach model has no fixed count. The weight/Everett function is
  identified on a discretized (α, β) grid, so storage grows with grid resolution. [inferred from
  the identification description above; none of the sources I read gives a number]

### 2.2 Dynamic (rate-dependent) Preisach

- **G. Bertotti, "Dynamic generalization of the scalar Preisach model of hysteresis," IEEE Trans.
  Magn., vol. 28, no. 5, pp. 2599–2601, Sep. 1992, doi:10.1109/20.179569.** **[abstract]** Each
  elementary loop's flux changes at a rate "proportional to the difference between the external
  field and the loop threshold fields", not instantaneously. Loop area then follows
  "C0 + C1·sqrt(f)". It is described as suited to power-loss calculations.
- **I. D. Mayergoyz, "Dynamic Preisach models of hysteresis," IEEE Trans. Magn., vol. 24, no. 6,
  pp. 2925–2927, 1988, doi:10.1109/20.92290.** Metadata only.
- **R. Roussel et al. (7 authors), "Differentiable Preisach modeling for characterization and
  optimization of particle accelerator systems with hysteresis," Phys. Rev. Lett., vol. 128,
  no. 20, 204801, 2022, doi:10.1103/PhysRevLett.128.204801.** Metadata only.
  - RHINO-MAG tried this differentiable Preisach in a preliminary test (RHINO-MAG §III-D, Table
    II) on 3C90 at 320 kHz and 25 °C. The test predicted **B from H**, the reverse of the MC2
    direction, and the paper says so.
  - The Preisach model (328 params) was far worse than a 248-param GRU-P (MSE 0.0081 vs 0.0005).
    The authors therefore did not investigate "the inversion (prediction of H based on B)".
    **[read, RHINO-MAG arXiv PDF]**

### 2.3 Jiles–Atherton (JA)

- **D. C. Jiles, D. L. Atherton, "Theory of ferromagnetic hysteresis (invited)," J. Appl. Phys.,
  vol. 55, no. 6, pp. 2115–2120, 1984, doi:10.1063/1.333582.** Metadata only.
- **D. C. Jiles, D. L. Atherton, "Theory of ferromagnetic hysteresis," J. Magn. Magn. Mater., vol.
  61, no. 1–2, pp. 48–60, 1986, doi:10.1016/0304-8853(86)90066-1.** Metadata only (Crossref).
  The MC2 Handbook credits JA with linking to "physical parameters of the magnetic material" and
  computing minor and major loops. It also notes "many different types of implementations".
  **[slides]**
- **D. C. Jiles, J. B. Thoelke, "Theory of ferromagnetic hysteresis: determination of model
  parameters from experimental hysteresis loops," IEEE Trans. Magn., vol. 25, no. 5, pp.
  3928–3930, 1989, doi:10.1109/20.42480.** **[abstract]** Parameters are extracted from initial
  anhysteretic susceptibility, coercivity, remanence and loop-tip coordinates, to "within a few
  percent in most cases".
- **D. C. Jiles, J. B. Thoelke, M. K. Devine, "Numerical determination of hysteresis parameters for
  the modeling of magnetic properties using the theory of ferromagnetic hysteresis," IEEE Trans.
  Magn., vol. 28, no. 1, pp. 27–35, 1992, doi:10.1109/20.119813.** **[abstract]** Numerical
  parameter extraction from standard lab quantities.
- **Parameter count:** RHINO-MAG's static inverse JA has **5 parameters**. **[teammate]** The
  conventional names (M_s, a, α, k, c) are standard JA notation but were not re-verified in a
  source I read. [inferred]
- **Physical critique:** **S. E. Zirka, Y. I. Moroz, R. G. Harrison, K. Chwastek, "On physical
  aspects of the Jiles–Atherton hysteresis models," J. Appl. Phys., vol. 112, no. 4, 043916,
  2012, doi:10.1063/1.4747915.** **[abstract]** The authors argue that the JA energy balance is
  closer to a coenergy balance, so the model loses physical content. They also argue the
  rate-dependent extension lacks a sound basis: computed loops diverge sharply from tests on
  grain-oriented steel at 50/100/200 Hz.
- **Identification is a literature of its own.** OpenAlex returns many JA identification papers
  (least squares, PSO, SFLA, chaotic optimization, neural networks), for example R. Marion et
  al., IEEE TMAG 2008, doi:10.1109/TMAG.2007.914867; M. Trapanese, J. Appl. Phys. 2011,
  doi:10.1063/1.3569735. Titles only. They show that fitting JA's few parameters is not routine.
  [inferred]

### 2.4 Dynamic JA and circuit-level models

- **D. C. Jiles, "Frequency dependence of hysteresis curves in conducting magnetic materials," J.
  Appl. Phys., vol. 76, no. 10, pp. 5849–5855, 1994, doi:10.1063/1.358399.** **[abstract]** The
  static JA equation is augmented with a classical eddy-current term (depends on dM/dt,
  resistivity, geometry) and an excess term scaling as (dB/dt)^(3/2). It reduces to the static
  loop as the rate goes to zero. It assumes slow drive or a thin sample.
- **D. C. Jiles, "Frequency dependence of hysteresis curves in 'non-conducting' magnetic
  materials," IEEE Trans. Magn., vol. 29, no. 6, pp. 3490–3492, 1993, doi:10.1109/20.281206.**
  Title only. It is possibly more relevant to ferrites, which have low conductivity. [inferred;
  not read]
- **J. G. Zhu, S. Y. R. Hui, V. S. Ramsden, "A generalized dynamic circuit model of magnetic cores for
  low- and high-frequency applications. I. Theoretical calculation of the equivalent core loss
  resistance," IEEE Trans. Power Electron., vol. 11, no. 2, pp. 246–250, 1996,
  doi:10.1109/63.486172; and S. Y. R. Hui, J. G. Zhu, V. S. Ramsden, "... II. Circuit model
  formulation and implementation," same issue, pp. 251–259, doi:10.1109/63.486173.** Metadata only. PI-HNO
  cites these as a "generalized dynamic circuit" class next to Preisach and JA.
- **M. Luo, D. Dujic, J. Allmeling, "Modeling frequency independent hysteresis effects of ferrite
  core materials using permeance–capacitance analogy for system-level circuit simulations," IEEE
  Trans. Power Electron., vol. 33, no. 12, pp. 10055–10070, 2018, doi:10.1109/TPEL.2018.2809704.**
  Metadata only. Cited in Guillod's Tutorial 2.
- **Landau–Lifshitz–Gilbert (LLG) circuit models:** H. H. Cui, S. Dulal, S. B. Sohid, G. Gu, L. M.
  Tolbert, "Unveiling the microworld inside magnetic materials via circuit models," IEEE Power
  Electron. Mag., vol. 10, no. 3, pp. 14–22, 2023, doi:10.1109/MPEL.2023.3301408. S. Dulal et al.,
  "A physics-based circuit model for nonlinear magnetic material characteristics," APEC 2024,
  pp. 396–401, doi:10.1109/APEC48139.2024.10509097. Metadata only. The organizers list LLG as
  an MC2 starting point (Tutorial 1 extended reading). RHINO-MAG judged full LLG micromagnetics
  infeasible: a 62.5 ns sample needs about 6,000 solver steps. **[teammate]**
- **A. Benabou, S. Clénet, F. Piriou, "Comparison of Preisach and Jiles–Atherton models to take into
  account hysteresis phenomenon for finite element analysis," J. Magn. Magn. Mater., vol. 261,
  no. 1–2, pp. 139–160, 2003, doi:10.1016/S0304-8853(02)01463-4.** Metadata only.

### 2.5 How classical models fared in MC2 (cross-reference)

- MC2's stated aim: "The 2025 MagNet Challenge aims to seek alternatives for the Preisach model
  and the Jiles-Atherton model with the support of a massive amount of measurement data"
  (Handbook, Motivation). **[slides]**
- Guillod's Tutorial 2 lists Preisach, JA, LLG, iGSE-based, reluctance/permeance and minor-loop
  splitting as candidate physics-based models. It ends with "A last warning about such models":
  two boxes, "Physics-based material model" and "Fitted parameters have a physical meaning",
  joined by a **crossed-out implication arrow (⇏)** (slide 28, checked on the rendered slide).
  In other words, a physics-based model form does **not** guarantee that the fitted parameters
  have a physical meaning. **[slides]**
- **RHINO-MAG** (SiegenPaderborn, MC2 performance-track winner; H. Vater, O. Schweins, L. Hölsch, W.
  Kirchgässner, T. Piepenbrock, O. Wallscheid, arXiv:2603.29745 and IEEE TPEL 2026,
  doi:10.1109/TPEL.2026.3733311) **[abstract; details from teammate]**. Static inverse JA,
  GRU-predicted JA parameters, JA plus residual GRU, a JA-residual PINN regularizer,
  differentiable Preisach and LLG were all worse than a plain GRU. Abstract: "a rather black-box
  gated recurrent unit (GRU) model structure with a graceful initialization setup is found to
  offer the most attractive model size vs. model accuracy trade-off in the small-model regime".
  **[read, arXiv PDF abstract]**
- **W. Kirchgässner, "A data-driven, gain-scheduled equivalent circuit for the B-H behavior of power
  ferrites," arXiv:2611.02222, Nov. 2026 (K26).** Code: github.com/wkirgsn/ferrite-bh-circuit.
  **[read, arXiv PDF: abstract, §I, §VII, Tables III–IV]**
  - Model: a nonlinear inductor, six viscous play branches forming a Prandtl–Ishlinskii operator,
    mean-field H feedback "as in the Jiles–Atherton model", two R–L after-effect paths and a
    measurement lag. Fitted by backpropagation through time: "32 parameters per material".
    A 33-parameter warm-up estimator, shared by all circuits, is not counted (Table III caption).
  - Own evaluation, mean test SRE over 15 materials: 7.33 ± 0.03 % (10 seeds) versus 7.77 % for
    the published RHINO-MAG GRU8. Its size is 325 params by the team/leaderboard count and 336 by
    K26's; both numbers describe the same model under different counting conventions (see
    winner-siegen's `siegen_paderborn.md` for the [inferred] explanation). Cite it as
    "325 (team/leaderboard) / 336 (K26)". The circuit is better on 12 of 15 materials. Its energy error is weaker: mean NERE
    0.82 % versus 0.73 %.
  - **Official A–E metric:** average p95 SRE 19.43 ± 0.22 % and energy 5.12 ± 0.19 % (Table III).
    The text calls this "on par with that of the challenge winner RHINO-MAG (19.4 % vs 20.4 %)".
    It is also smaller than every submitted model (the smallest had 83).
  - Gain-scheduled versions: with an MLP of 16 hidden units (656 params), 13.03 % p95 SRE /
    1.91 % energy, which "would rank third by the SRE". With 64 hidden units (5,552 params),
    12.09 % / 1.61 %. "Only Hefei ... is more accurate by both metrics."
  - Details are also in winner-siegen's `siegen_paderborn.md` §7.
- *Takeaway:* in MC2, classical operators failed when bolted onto a GRU (RHINO-MAG) and matched
  the performance-track winner when the whole circuit was fitted end to end (K26: 32 params,
  19.4 % official p95 SRE). [inferred from the two sources]
- **[caveat] "Winner" needs a track name.** Awards were not given by leaderboard rank.
  SiegenPaderborn won the performance track but is 6th on official sequence error. Hefei is 1st
  on sequence error and placed 2nd in the performance track (reviewer log §0, upstream
  img/winners.jpg).

---

## 3. The MagNet program

### 3.1 Why MagNet

**D. Serrano, H. Li, S. Wang, T. Guillod, M. Luo, V. Bansal, N. K. Jha, Y. Chen, C. R. Sullivan, M.
Chen, "Why MagNet: Quantifying the complexity of modeling power magnetic material
characteristics," IEEE Trans. Power Electron., vol. 38, no. 11, pp. 14292–14316, Nov. 2023,
doi:10.1109/TPEL.2023.3291084.** CC-BY. **[read]**

- Data: N87 toroid (R34.0×20.5×12.5), 142,871 voltage/current waveform pairs. Sinusoidal, two- and
  three-level PWM excitation (GaN T-type inverter). 25/50/70/90 °C oil bath. 8-bit scope, 100,000
  samples per test, averaged to one 1,024-point cycle.
- **Steady state only:** "the initial value of B cannot be obtained ... since the voltage is not
  captured from the time that the excitation is first applied ... but only on steady-state
  conditions", so B is defined with zero mean. This is the gap MagNetX later fills. [inferred
  link]
- Findings: Steinmetz limits (§1.1), duty-cycle and relaxation effects the iGSE misses (§1.3),
  strong coupling of dc bias, temperature and duty, and no multiplicative decoupling.
  Measurement sensitivity: raising dv/dt from about 1 to 5 V/ns raises loss by 6 %. A 1.6 ns
  probe delay gives under 5 % relative error for N87.
- On ML: NNs are "a form of efficient interpolation of the data". Challenges are "compatibility
  and integration with existing tools and difficulties with generalizations to scenarios
  different from the test data set." No new model is proposed.
- *Why it matters:* this is the domain's own statement that independent-factor (Steinmetz-style)
  models cannot reach high accuracy and that a time-domain f(B(t), T) is needed.

### 3.2 How MagNet

**H. Li, D. Serrano, T. Guillod, S. Wang, E. Dogariu, A. Nadler, M. Luo, V. Bansal, N. K. Jha, Y.
Chen, C. R. Sullivan, M. Chen, "How MagNet: Machine learning framework for modeling power
magnetic material characteristics," IEEE Trans. Power Electron., vol. 38, no. 12, pp.
15829–15853, Dec. 2023, doi:10.1109/TPEL.2023.3309232.** CC-BY. **[read]**

- Database: more than 500,000 data points. Combines four earlier conference papers (COMPEL 2020,
  COMPEL 2021, APEC 2022, COMPEL 2022).
- **Size–accuracy evidence (Table III, FNN, N87 triangular):** 21 params: 19.76 % avg / 68.59 %
  max. 109: 9.81 / 47.72. 454: 5.33 / 21.36. 1,594: 1.81 / 9.47. 5,515: 1.77 / 8.62. The gain
  saturates above about 1,600 params. [inferred from the table] iGSE (3 params) gives
  14.6 % / 61.1 %.
- Sequence-to-scalar LSTM (32 states plus a 3-hidden-layer FNN, 5,569 params): 2.09 % average,
  14.35 % maximum relative core-loss error across sin/tri/trap.
- **Transient explicitly excluded:** "the MagNet database in its current state does not include
  sufficient data to perform a systematic study of transient behavior ... Modeling transient
  characteristics is beyond the scope of this paper."
- Encoder–projector–decoder: 1-layer 32-state LSTM encoder, a 3-layer 64-neuron FNN projector
  that takes T, f, H_dc, and an autoregressive LSTM decoder that outputs H(t). On 15,327 B–H pairs
  (25 °C, no bias) it reaches 3.73 % average and 18.29 % maximum sequence error.
- **Sequence metric eq. (8):** rms(H_pred − H_meas) / rms(H_meas). This is the same formula as
  the MC2 official sequence error (Webinar 6). [inferred lineage; the formulas are identical]
- Transfer learning is used to reduce the amount of data needed.

### 3.3 MagNet-AI

**H. Li, D. Serrano, S. Wang, M. Chen, "MagNet-AI: Neural network as datasheet for magnetics
modeling and material recommendation," IEEE Trans. Power Electron., vol. 38, no. 12, pp.
15854–15869, Dec. 2023, doi:10.1109/TPEL.2023.3309233.** CC-BY. **[read]**

- "Neural network as datasheet": encoder–projector–decoder, LSTM versus Transformer, on N87
  (142,871 pairs: 3,495 sin, 46,973 tri, 92,403 trap; 269,940 after phase-shift and multi-cycle
  augmentation).
- Table I: LSTM 28,225 params, 4.52 % mean and 10.92 % p95 sequence error, 0.017 ms per point.
  Transformer 28,481 params, 2.99 % and 6.48 %, 2.584 ms per point. Core-loss p95: 13.69 % (LSTM)
  and 10.02 % (Transformer). Training takes about 20 h per material on Colab Pro. A 3.8 GB dataset
  is compressed into a 204 kB model.
- **Key caveat for us:** "core loss calculation is highly sensitive to phase mismatch between H(t)
  and B(t), and an approximately matched sequence does not necessarily result in a good match for
  core loss." This is the same pattern as our log's finding that energy error does not follow
  sequence error (RESEARCH_LOG §5.5). [inferred link]
- Classical models (SE, iGSE, JA) "are established based on empirical simplifications or physical
  approximations, which limit their modeling accuracy."

### 3.4 MagNet Challenge 1 summary paper (OJPEL)

**M. Chen, H. Li, S. Wang, T. Guillod, D. Serrano, et al. (132 authors in total, per Crossref and the PDF author block), C. R. Sullivan, "MagNet
Challenge for data-driven power magnetics modeling," IEEE Open J. Power Electron., vol. 6, pp.
883–898, 2025 (published online 30 Sep. 2024), doi:10.1109/OJPEL.2024.3469916.** Open access.
**[read]**

- Task: a single-cycle B(t) (1,024 steps), f and T map to P_v. Periodic steady state only. "Other
  related topics, such as modeling transient dynamics ... were beyond the scope of the MagNet
  Challenge in 2023."
- Ranking used the 95th-percentile relative core-loss error. "We anticipate a 95th percentile
  error of less than 10% as being competitive."
- **Model size definition:** "the total number of parameters that a model needs to store to
  describe the characteristics of each material." Structure, loops and layers are not counted.
  "MagNet Challenge was designed to encourage models with more computation and less memory usage."
- **Measurement floor:** maximum measurement error "generally controlled below 20%", average
  "below 10%". Teams were encouraged to target about 10 % average error, because "the chance of
  model overfitting can significantly increase if the model accuracy is higher than the
  measurement accuracy."
- **Size–accuracy conclusions (the brief's key question):**
  - "the outcomes of the challenge quantitatively verified the fundamental tradeoff between model
    size and model accuracy" (§II).
  - "The winning models use about 1,000 parameters to achieve less than 10% average 95th
    percentile error" (§I-C).
  - Fig. 5: "The minimum average 95th percentile error reaches 7%, and the smallest model
    parameter size reaches 60." Both the Pareto front and model sizes moved down relative to the
    pre-challenge SOTA (estimated from How MagNet).
  - Fig. 6: "the differences in the model accuracy are usually very small among the best
    performing teams, whereas the differences in model size are often very large."
  - TU Delft's multi-objective optimization "shows that a total number of 1,000 parameters is a
    good balance point between model size and accuracy."
- **Winning approaches (MC1):**
  - Performance: 1. Paderborn (HARDCORE: residual dilated CNN with an intermediate B–H
    reconstruction, "1755 parameters"), 2. Fuzhou (sequence-to-scalar Transformer with multi-stage
    fine-tuning, "MagNet-GPT"), 3. Bristol (LSTM plus FNN with transfer learning; "needed a lot of
    parameters").
  - Innovation: 1. Sydney (MMINN, hysteresis plus dynamic sub-networks, about "1000 parameters"),
    2. TU Delft (multi-material transfer learning plus MOO), 3. Mondragon (ci2GSE, 60 params).
  - Software Engineering Award: University of Sydney (OJPEL appendix).
  - Grey-box hybrid was "one widely-adopted data-driven approach ... for its excellent capability
    of balancing model accuracy and model size."
  - Material D (79, waveform challenge) was hardest.
- **HARDCORE journal version:** W. Kirchgässner, N. Förster, T. Piepenbrock, O. Schweins, O.
  Wallscheid, "HARDCORE: H-field and power loss estimation for arbitrary waveforms with residual,
  dilated convolutional neural networks in ferrite cores," IEEE Trans. Power Electron., vol. 40,
  no. 2, pp. 3326–3335, Feb. 2025, doi:10.1109/TPEL.2024.3488174. **[abstract]** "A Pareto-style
  tradeoff between model size and estimation accuracy is demonstrated, which yields an optimum at
  as low as 906 parameters and down to below 8% for the average 95th percentile."
  - **[caveat]** HARDCORE's size is reported three ways: OJPEL "1755", the HARDCORE abstract "906"
    (Pareto optimum), and PI-HNO's re-implementation 1,629. They are different configurations;
    cite the one that matches the claim.
- **MMINN journal version:** Q. Huang, Y. Li, J. Zhu, S. Li, "Magnetization mechanism-inspired
  neural networks for core loss estimation," IEEE Trans. Power Electron., vol. 39, no. 12, pp.
  16382–16390, Dec. 2024, doi:10.1109/TPEL.2024.3450897. **[abstract]**
- Roadmap: future challenges should cover "transient excitations with minor-loop and non-zero dc
  bias", MHz range and mixed-frequency operation. "A majority of student teams performed time
  domain analysis."
- **[caveat]** The text says "24 teams from 17 countries" submitted final results, but the
  appendix lists 23 round-2 teams. Cite as "24 (23 listed)" or avoid the exact number.

### 3.5 MC2 (2025–26): MagNetX database and organizer models

**MagNetX database.** **H. Kwon, S. Wang, H. Li, Y. Elasser, G.-G. Kang, D. Zhou, D. Grigoryan, M.
Chen, "MagNetX: Extending the MagNet database for modeling power magnetics in transient," IEEE
APEC 2025, pp. 566–572, doi:10.1109/APEC48143.2025.10977252** (TechRxiv
doi:10.36227/techrxiv.173396064.40994860/v1, 11 Dec. 2024). **[abstract + slides]**

- Abstract: automated transient acquisition. It reveals "significant core loss variations during
  frequency transitions", and "B-H loop analysis further demonstrates loop area changes during
  transient phases". Use cases named: ac-dc converters, motor drives, power amplifiers.
- From the slides (Tutorial 1/4, kickoff, MagNetX README PDF) **[slides]**:
  - 16 MHz sampling (62.5 ns step), 50–800 kHz subcycle frequency in 7 buckets, 25/50/70 °C.
  - Duty-cycle step changes D = 0.2–0.8 with a minimum step of 0.1, 100 subcycles per sequence,
    "a mix of high frequency transient and low frequency steady state".
  - "Example symmetric magnetizing and demagnetizing sequences".
  - Ten released training materials with "103,118 sequences". Sequence lengths from 32,016 steps
    (50 kHz) to 2,016 steps (800 kHz). The kickoff slide gives 800 kHz as 1,967 steps. For
    320 kHz, the kickoff and the Tutorial 1/4 bullet lists give 4,975 steps, while the Tutorial
    1/4 tables and the MagNetX README give 5,008.
- **[caveat]** The slides disagree on sequences per material (kickoff ">5,000", Tutorials 1/4
  "13,587") and say 15 materials in total. Our downloaded counts differ from both (RESEARCH_LOG §2;
  reference_audit §8).

**MagNetX foundation models (APEC 2025).** **S. Wang, H. Kwon, H. Li, Y. Elasser, G.-G. Kang, D.
Zhou, D. Grigoryan, M. Chen, "MagNetX: Foundation neural network models for simulating power
magnetics in transient," IEEE APEC 2025, pp. 2438–2445, doi:10.1109/APEC48143.2025.10977420**
(TechRxiv doi:10.36227/techrxiv.173396153.34393547/v1). **[abstract + slides; full text not accessed]**

- Abstract: "the model uses partial memories and the next-state flux density excitation to predict
  the response of the field intensity in the next time step. The model is in time domain and is
  frequency independent. An example sequence-to-scalar LSTM neural network was designed, trained,
  and tested."
- Slides: input is B(t) and H(t) over the past 80 steps plus the next B. Training points are
  subsampled per frequency bucket from a "54 GB" set. "single-step" and "autoregressive"
  evaluation are distinguished.

**Journal version.** **S. Wang, H. Kwon, H. Li, T. Guillod, H. Wouters, C. R. Sullivan, M. Chen,
"MagNetX: Data-driven time-domain foundation models for power magnetics in transient," IEEE Trans.
Power Electron., vol. 41, no. 11, pp. 19455–19474, Nov. 2026, doi:10.1109/TPEL.2026.3685464.**
**[abstract]** The abstract is available on Semantic Scholar. I saw its opening via a search
snippet; the two later phrases below were supplied by the reviewer from the full abstract,
because Semantic Scholar rate-limited my re-fetch.
- Opening: "presents and applies the concept of time-domain foundation models (FMs) to analyzing
  nonlinear soft ferrite material under transient excitations. Key principles of a time-domain
  foundational model are defined and elaborated. A neural network (NN)-based FM is
  implemented..."
- Data: "quasi-steady-state pulsewidth modulation waveforms with minor loop excitations".
- Model: the FM "utilizes time-domain arbitrary length memory ... and predicts arbitrary future
  steps".
- Full text not read. Cite it for the concept and abstract-level claims only.

**Foundation-model principles (COMPEL 2025).** **S. Wang, H. Kwon, D. Grigoryan, H. Li, T. Guillod,
C. R. Sullivan, M. Chen, "Unified time domain foundation models for hysteretic passive
components," IEEE COMPEL 2025, Knoxville, pp. 1–8, doi:10.1109/COMPEL57166.2025.11121278.**
COMPEL'25 Best Paper Award (per the Princeton group site). **[abstract, full]**

- Three key properties: "1) frequency agnostic, 2) adaptive to universal time steps; and 3) a
  hypothesis that the system's initial condition has an impact for a limited time horizon." A
  foundation model "can be physics-driven or data-driven, or a hybrid of both". The paper presents
  "an example neural network that is simple, robust, and accurate".
- These are the same three properties as the MC2 rules ("frequency agnostic, time-step agnostic,
  and initial-state agnostic (always converging after a long time)", Handbook).

**MC2 rules and evaluation** **[slides]**: Handbook 2025, kickoff slides (25 Feb. 2025), Webinars
5 and 6, all in `minjiechen/magnetchallenge-2`.

- Task: H'(t) = f(B(t), H(t), B'(t), T). Train on 10 known materials, test on 5 new ones (A–E).
- A–E challenges (Webinar 6): A "tiny data" (105 training sequences), B "frequency", C "no past
  H" (only H(1) = 0), D "flux density", E "special material".
- Metrics: the sequence error RMS(H_pred − H_meas)/RMS(H_meas), and the energy error of the
  predicted segment normalized by the full sequence's core loss. The formulas are from Webinar 6.
  The 95th-percentile aggregation is not on Webinar 6. It comes from the README leaderboard
  headers ("Avg. 95% Seq. Error [%]", "Avg. 95% Energy Error [%]") and the organizers' scoring
  script `Evaluation_Read.m`: p95 within each warm-up block, averaged over blocks
  (RESEARCH_LOG §3.2).
- Model size: the Handbook first said "kB of the submitted MATLAB/Python codes". The kickoff
  changed this to "number of material-specific parameters that need to be kept in the model", and
  Webinar 6 to the number of parameters per material in `Parameter_Size.csv`. The criterion
  evolved; the final rule is the parameter count.
- Judging axes: accuracy, size, explainability, novelty, software quality (README).

### 3.6 The organizers' LSTM baselines (MC2)

Both are "double LSTM-based" sequence-to-scalar next-step predictors. **[code, read directly]**

| baseline | source | architecture | params (printed) | training |
|---|---|---|---|---|
| tutorial | `tutorials/tutorial-1/Demo_LSTM_Training.ipynb`, tutorial-3 eval | LSTM(1,12) on the 80-step B window and LSTM(1,12) on the 80-step H window; last hidden states concatenated with [scaled next B, T] → Linear 26→26→8→4→1, ReLU | 2,399 | 17,500 windows (7 freq. × 2,500, 3C90), random 80/10/10 train/valid/test split, MSE on the true next H (single-step, true H history), Adam lr 1e-3 × 0.6 every 50 epochs, 200 epochs, batch 2,048 |
| pretest | `Pretest/PreEvaluationExample.ipynb` | LSTM(1,6) × 2; Linear 14→14→8→1, Tanh | 771 | not published (weights only; see reference_audit §2) |

- **Inference is autoregressive:** the H window is rolled and the newest prediction is written
  into the last slot. Shorter memories are left-padded with the first value, longer ones
  truncated to the most recent 80, and other sampling rates are up- or down-sampled
  (pretest notebook header; Tutorial 4).
- **Training and inference mismatch:** the model is trained only on measured H histories and
  never sees its own errors before rollout. [inferred: this is the classic exposure-bias setup;
  the notebooks do not discuss it]
- Organizer-reported results on the 105-sequence 3C90 tutorial-3 demo set (Tutorial 4, slides
  35–36; tutorial-3 metric RMSE/mean|H|): mean NRMSE by memory length 100: 18.74 %, 300: 20.67 %,
  500: 12.25 %, 700: 15.57 %, 900: 13.40 %, and energy NRMSE 39.78 %. Our harness reproduces these
  numbers and the 33.27 % p95 (RESEARCH_LOG §3.3).
- **Pre-evaluation results file** (`Pre_Evaluation_Results.pdf`, organizing committee, dated
  09/01/2025) **[read]**. It is a blank template, not a results table: "Appendix II: MagNet
  Challenge 2 Pretest Results – Team Name – Date", with empty sequence-error and energy-error
  panels for 3C90 and N87 in the 10/90, 50/50 and 90/10 known/predicted splits. The README says
  pretest results "are self-reported, are not verified and have no influence on the final
  evaluation". **No organizer baseline numbers on the pretest set exist in this file.** Our log
  also found that the pretest segments are cut from the training sequences (RESEARCH_LOG §2).
- On A–E (README leaderboard, official metric): Princeton (organizers, not competing) 1,443
  params, 66.85 % seq p95. NTU2 2,399 params, 75.11 % seq / 27.43 % energy. NTU2 is probably the
  tutorial architecture [inferred from the identical 2,399-param count only; NTU2's report and
  weights are not available, and reference_audit §7 gives no other evidence].
- PI-HNO's re-implementation of "an LSTM model based on the MagNet Challenge 2 demonstration
  setting" ("2 LSTM(12)×1+MLP(26,8,4)", 2,399 params) scores 106.31 % mean / 282.36 % p95 sequence
  error under their own 14-material protocol. That is the worst baseline except the Transformer
  (§4.3). **[read]**

### 3.7 MC2 outcome (pointers only; team details are with winner-siegen and winner-others)

- 32 final entries (31 competing teams plus the organizers' Princeton entry, "not competing"). Parameter counts range from 83 (Gatech) to 13,185,504 (Hangzhou) (README
  leaderboard).
- Small-model points (seq p95 / params): Gatech 38.76 % / 83; SiegenPaderborn 20.43 % / 325;
  Cambridge 37.01 % / 1,182; Princeton 66.85 % / 1,443; NTU2 75.11 % / 2,399; Sydney 24.63 % /
  4,777; Hefei 9.17 % / 6,564 (best accuracy).
- APEC 2026 keynote: winners and a "Pareto-front" plot of number of parameters against average
  95th-percentile energy error (log–log). Performance: 1. Siegen & Paderborn, 2. Hefei & CityU &
  Xiamen, 3. Sydney. Innovation: 1. Georgia Tech, 2. Cambridge, 3. Silicon Austria Labs.
  **[slides]**

---

## 4. Neural ODE and physics-informed hysteresis models

### 4.1 Guillod, MC2 Tutorial 2 (analytical and neural-ODE methods)

**T. Guillod, "Soft magnetic materials and ordinary differential equations: From linear circuits to
neural network models," IEEE MagNet Challenge Webinar, 23 May 2025 (Dartmouth College).** Slides
CC BY-ND 4.0. Code: https://github.com/otvam/magnetic_ode_models (MPL 2.0; JAX + Diffrax +
Equinox + Optax). **[slides + code read]**

The local `third_party/.../tutorials/tutorial-2/` folder holds only the N87_5 data. The code was
read from `/tmp/mc2lit/t2/`.

- **Circuit view of the material:** V ∝ dB/dt and I ∝ H, so dy/dt = f(y, dB/dt), H = g(y, dB/dt).
  Listed advantages: "Causality is enforced", "Non-periodic waveforms", "Compatible with SPICE",
  "Compatible with FEM".
- **Problems flagged:** "The initial state of the material is not known!" The fix is to align
  phase and "wait for steady-state" before evaluating. The other problem is computational cost,
  since thousands of steps must be integrated per signal per optimizer iteration.
- **Data:** MagNetX "N_87_5": N87, 25 °C, 3.2 kHz repetition frequency, 48 training and 24 test
  samples.
- **Error metrics:** err_H = RMS(H_model − H_ref)/RMS(H_ref), the same form as the MC2 sequence
  metric, and err_P = (P_model − P_ref)/P_ref. The slides do not say whether results are means or
  percentiles.
- **Results:**

  | model | params | RMS err H | err loss | notes |
  |---|---|---|---|---|
  | linear R-L (dB/dt = r·H + μ0μr·dH/dt) | 2 | 46 % | 41 % | "Cannot model saturation ... nonlinear losses" |
  | nonlinear circuit, 2 decoupled ODEs, R_i(H_i), L_i(H_i) | 10 (2 × 5) | 18 % | 9 % | "No overfitting", latin-hypercube init, AdaBelief |
  | neural ODE, MLP tanh, "4 inputs and 3 outputs", "2 [hidden layers] with 16 neurons" | not stated | 13 % | 9 % | "No overfitting" |

  Neural-ODE parameter counts [inferred, computed from the code]:
  - The slide configuration (4→16→16→3) has 403 params.
  - The repository's `get_ann_single` (width 12) has 255.
  - The `get_ann_dual` model that `run_4_ann_train.py` actually trains (two MLPs 3→12→12→3) has
    486.

  The slides and the repository code disagree on width; cite the slide's results with "about
  10^2–10^3 parameters".
- **Training curriculum (code `run_4_ann_train.py`):** "start with a very short prediction
  horizon", then "slowly increase the prediction horizon towards steady-state" (n_out = 0.1 →
  0.2 → 0.5 → 1.0 → 1.0 periods). Over the same steps the loss weight shifts from field error
  toward power-loss error (0 → 0.5), and the learning rate falls from 25e-3 to 3e-3. The slides
  explain why: "The initial random networks parameters are a bad model", and "This would imply a
  very slow and/or unstable training".
  - *Relevance:* this is a horizon curriculum on a continuous-time model and the closest
    organizer-side analogue to our multi-step rollout stage. [inferred; ssm-lit covers rollout
    training generally]
- **Stated limits and outlook:** neural ODEs have "No clear physical meaning", but offer an
  "Opportunity for compact models" and are "Compatible with existing tooling". Possible
  extensions: "Neural SDE, neural CDE, latent ODE", "System with discrete states (e.g.
  Preisach-based)", "Possibility to implement a physics-informed network".
- **"What is the ideal model?" slide:** small number of parameters, small fitting dataset,
  extrapolation, unseen waveshapes, low compute, physical link, transfer between materials,
  SPICE/FEM compatibility.
- *Why it matters:* the organizers themselves present the continuous-time state-space (ODE/circuit)
  view as a principled small-model route. A 2-parameter linear ODE already reaches 46 % H error,
  and 10 nonlinear parameters reach 18 %, on one N87 file. This is the conceptual neighbour of a
  diagonal SSM. [inferred]

### 4.2 Neural differential equations (ML background cited by Guillod)

- **R. T. Q. Chen, Y. Rubanova, J. Bettencourt, D. Duvenaud, "Neural ordinary differential
  equations," arXiv:1806.07366, 2018.** **[abstract]** The hidden-state derivative is a network
  and the output comes from an ODE solver: constant memory, adaptive computation, gradients
  through any solver.
- **P. Kidger, "On neural differential equations," DPhil thesis, Univ. of Oxford,
  arXiv:2202.02435, 2022.** **[abstract]** A survey of neural ODEs, CDEs and SDEs, which treats
  RNNs and residual nets as discretizations. Guillod cites it.
- **P. Kidger, J. Morrill, J. Foster, T. Lyons, "Neural controlled differential equations for
  irregular time series," arXiv:2005.08926 (NeurIPS 2020 spotlight, per the arXiv comment).**
  **[abstract]** Adds a control path that steers the hidden state. [inferred: MC2 is exactly a
  controlled system, with B(t) as the control path]

### 4.3 Physics-aware / physics-informed hysteresis networks

- **A. Chandra, T. Kapoor, B. Daniels, M. Curti, K. Tiels, D. M. Tartakovsky, E. A. Lomonova,
  "Generalizable models of magnetic hysteresis via physics-aware recurrent neural networks,"
  Comput. Phys. Commun., vol. 314, 109650, 2025, doi:10.1016/j.cpc.2025.109650** (preprint
  "Neural oscillators for magnetic hysteresis modeling," arXiv:2308.12002). **[abstract]**
  HystRNN uses "ordinary differential equations (ODEs) governing the phenomenological hysteresis
  models to update hidden recurrent states". It generalizes to first-order reversal curves and
  minor loops where "Traditional recurrent neural architectures struggle".
- **A. Chandra, B. Daniels, M. Curti, K. Tiels, E. A. Lomonova, "Magnetic hysteresis modeling with
  neural operators," IEEE Trans. Magn., vol. 61, no. 1, pp. 1–11, 2025 (online 2024),
  doi:10.1109/TMAG.2024.3496695** (arXiv:2407.03261). **[abstract]** DeepONet, FNO and WNO
  predict unseen FORCs and minor loops. A "rate-independent FNO" is proposed for prediction at
  sampling rates different from training. They outperform recurrent baselines.
- **Z. Guo, B. H. Nguyen, H. Hamzehbahmani, R. V. Sabariego, "Dynamic hysteresis model of
  grain-oriented ferromagnetic material using neural operators," IEEE Trans. Magn., 2025,
  doi:10.1109/TMAG.2025.3600089** (arXiv:2504.04863). **[abstract]** FNO is best among FNO,
  U-FNO and DeepONet on GO steel.
- **Z. Guo, X. Shen, R. V. Sabariego, "A multi-scale ResNet-augmented Fourier neural operator
  framework for high-frequency sequence-to-sequence prediction of magnetic hysteresis,"
  arXiv:2604.04150, 2026.** **[abstract]** dB/dt is an input feature for ringing and minor loops.
  Tested on MagNet materials "79 through 3C90".
- **Y. Zhu, Q. Huang, S. Li, Y. Li, G. Lei, J. Zhu, "A physics-informed hybrid neural operator for
  transient magnetization prediction in power magnetics," arXiv:2608.02965 (v3), Aug. 2026 (arXiv
  comment: "prepared for possible submission to IEEE Transactions on Power Electronics").** **[read, introduction and results via arXiv HTML]** Authors are from the
  University of Sydney and UTS.
  - Method (PI-HNO): local recurrent branch, a Preisach-like global (attention) branch, and a soft
    open-path B–H energy penalty. Trained per material on 14 MagNetX ferrites.
  - Size and results: **4,777 params**. Sequence mean / p95 13.40 % / 31.39 %, energy 1.92 % /
    7.60 %.
  - Baselines under the same protocol (seq mean / p95, params):

    | baseline | seq mean / p95 (%) | params |
    |---|---|---|
    | MagLearn2 | 12.88 / 28.47 | 860,929 |
    | GRU | 18.44 / 46.17 | 4,758 |
    | HARDCORE | 19.28 / 50.84 | 1,629 |
    | MMINN | 26.83 / 63.97 | 1,084 |
    | MC2 demo LSTM | 106.31 / 282.36 | 2,399 |
    | Transformer | 364.92 / 930.32 | 2,105 |

  - Relation to MC2: the paper's evaluation is its own split, not the official A–E set, so it is
    not leaderboard-comparable. The 4,777-parameter count equals Sydney's MC2 leaderboard entry,
    which suggests the same model line, but the paper does not name the team. [inferred]
  - Neural ODEs are not used; a GRU-as-forward-Euler-relaxation analogy appears.
  - *Why it matters:* it is a published small-model comparison on MagNetX in which the recurrent
    and attention baselines have much larger errors than structured, physics-shaped models. That
    the gap comes from autoregressive error accumulation is [inferred]: PI-HNO reports only the
    errors and never discusses rollout.
- **Y. Xiao, C. Li, Z. Zheng, "A magnetic core loss model based on physics-informed neural network
  with cross-attention," IEEE Trans. Power Electron., vol. 41, no. 1, pp. 92–96, Jan. 2026,
  doi:10.1109/TPEL.2025.3599366.** Metadata only. A PINN for scalar loss, not H(t).
- **Empirical-model-informed NN (MC1 IISc):** N. Rajput, H. B. Sandhibigraha, N. Agrawal, V. M. Iyer,
  "An empirical model informed neural network core loss predictor for soft magnetic materials,"
  IEEE Trans. Power Electron., vol. 40, no. 8, pp. 11257–11267, 2025, doi:10.1109/TPEL.2025.3557218
  (Crossref).
  OJPEL §II-A describes the approach: a data loss plus an "empirical loss term" (SE / ISE) with
  learnable Steinmetz parameters.
- **HARDCORE** (above) calls itself "physics-informed" through its intermediate B–H
  reconstruction layer.
- **PINNs on Maxwell / micromagnetics (located, not read):** A. Kovacs et al., "Magnetostatics and
  micromagnetics with physics informed neural networks," JMMM 2022, doi:10.1016/j.jmmm.2021.168951.
  Z. Guo, B. H. Nguyen, R. V. Sabariego, "Physics-informed neural network for solving 1-D nonlinear
  time-domain magneto-quasi-static problems," IEEE TMAG 2025, doi:10.1109/TMAG.2025.3553236.
  These are field-solver PINNs and not directly comparable to MC2's material-level task.
  [inferred from the titles]
- **MC2 physics-informed attempts:** the RHINO-MAG JA-residual PINN regularizer did not help
  (§2.5, teammate). K26 is a physics-structured circuit fitted by BPTT and matched the performance-track winner.

---

## 5. Prior neural-network hysteresis and B–H modeling

### 5.1 Early feed-forward and Preisach-hybrid models (rate-independent or quasi-static)

- **C. Serpico, C. Visone, "Magnetic hysteresis modeling via feed-forward neural networks," IEEE
  Trans. Magn., vol. 34, no. 3, pp. 623–628, 1998, doi:10.1109/20.668055.** **[abstract]** An FFNN
  model built to satisfy "the general memory storage properties of systems with rate independent
  hysteresis", trained by backpropagation and validated on measurements.
- **D. Makaveev, L. Dupré, M. De Wulf, J. Melkebeek, "Modeling of quasistatic magnetic hysteresis
  with feed-forward neural networks," J. Appl. Phys., vol. 89, no. 11, pp. 6737–6739, 2001,
  doi:10.1063/1.1361268.** Metadata only.
- **F. Sixdenier, R. Scorretti, R. Marion, L. Morel, "Quasistatic hysteresis modeling with
  feed-forward neural networks: Influence of the last but one extreme values," JMMM, vol. 320,
  no. 20, pp. e992–e996, 2008, doi:10.1016/j.jmmm.2008.04.076.** Cited in OJPEL [25]; DOI from
  Crossref. Not read.
- **C. Grech, M. Buzio, M. Pentella, N. Sammut, "Dynamic ferromagnetic hysteresis modelling using a
  Preisach-recurrent neural network model," Materials, vol. 13, no. 11, 2561, 2020,
  doi:10.3390/ma13112561.** **[abstract]** Play operators combined with an RNN for rate-dependent
  hysteresis in ARMCO iron: NRMSE "better than 0.7%" from "six" training loops, and it
  generalizes to unseen ramp rates.
- **S. Quondam Antonio, F. Riganti Fulginei, A. Laudani, A. Faba, E. Cardelli, "An effective neural
  network approach to reproduce magnetic hysteresis in electrical steel under arbitrary excitation
  waveforms," JMMM, vol. 528, 167735, 2021, doi:10.1016/j.jmmm.2021.167735.** Metadata only.
- **S. Quondam Antonio et al., "Deep neural networks for the efficient simulation of macro-scale
  hysteresis processes with generic excitation waveforms," Eng. Appl. Artif. Intell., vol. 121,
  105940, 2023, doi:10.1016/j.engappai.2023.105940.** **[abstract]** A feed-forward network with
  past-H and past-M inputs, trained on Preisach-simulated trajectories. No numbers in the
  abstract.

### 5.2 Sequence models on MagNet steady-state data

- **How MagNet:** seq-to-scalar LSTM and an LSTM encoder–projector–decoder for H(t) (§3.2).
- **MagNet-AI:** LSTM versus Transformer at about 28k params (§3.3).
- **MC1 entries (OJPEL §II-B):** Fuzhou Transformer, Bristol LSTM+FNN, and assorted
  ViT/CNN/LSTM/Transformer models from ASU, XJTU, Tsinghua, NTUT, NTU and HDU that "tried to
  leverage more advanced signal processing techniques ... to reduce the load of the neural networks
  and use fewer parameters".

### 5.3 Transient / MC2-era sequence models

- **MagNetX LSTM** (organizers): §3.5–3.6.
- **RHINO-MAG GRU** (325 params, performance-track winner): the abstract reports "8.02 %" average
  sequence error and "1.07 %" average normalized-energy error on the five unseen materials. These
  are means over materials of per-sequence *means*, not p95. The official leaderboard p95 is
  20.43 % / 3.41 % (reviewer log §0). Details in `siegen_paderborn.md`.
- **J. Yang, B. Cui, Y. Huo, L. Zhang, A. Martinez, S. Liu, J. Wang, "MagLearn 2: High-fidelity,
  saturation-aware, and history-efficient sequence-to-sequence modeling of transient B-H behaviour
  under PWM excitation," arXiv:2610.03504, Oct. 2026** (Bristol group; submitted to TPEL).
  Conference version: B. Cui, J. Yang, Y. Huo, L. Zhang, A. Martinez, S. Liu, J. Wang, "MagLearn 2:
  A short-window Seq2Seq framework for transient modelling of power magnetics under PWM
  excitations," IPEC-Nagasaki 2026 (ECCE Asia), pp. 1–7,
  doi:10.23919/IPEC-Nagasaki2026-EC64663.2026.11597291. **[read, arXiv PDF §VI, Tables
  V–VII; numbers cross-checked with reviewer log §0]**
  - Model: a condition-aware LSTM seq2seq with a short lookback and a saturation-correction
    module. On a saturated 3C90 case, sequence relative RMSE falls from "67.84%" to "9.19%".
  - **It is team Bristol's model:** H128-L3 has 860,929 params (Table V), Bristol's leaderboard
    count. Bristol scores 25.22 % official seq p95, rank 7.
  - **[caveat]** The abstract's "2.2%" is the lowest *per-material mean* relative RMSE (FEC014
    2.24 %). It is not an A–E average and not the official metric.
  - **History-window finding (Table VI, occlusion sensitivity):** the error-growth threshold for
    a 500-sample history has "a median value of approximately 560 ns", which is about 9 native 16 MHz
    samples [inferred arithmetic; MagLearn 2 resamples its windows, e.g. 357 steps = 34.92 µs,
    about 98 ns per step, so these are not its own model steps], and reaches about 2.31 µs for N87. However, keeping sequence
    error at 20 % needs 2.19–43.06 µs of retained H history depending on the material. The paper
    concludes the relevant memory "is concentrated near the prediction boundary" and is "short-term
    rather than spanning a full fundamental cycle".
  - *Relevance:* the organizers' 80-step window is 5 µs [inferred: 80 × 62.5 ns]. That is long
    enough to cover the median threshold but shorter than the 20 %-error history for most
    materials in Table VI. A longer-memory model such as an SSM could test whether this
    matters. [inferred]
- **PI-HNO** (§4.3) and the **neural operators** (§4.3) also belong in this list.
- **F. Purnode, L. Denis, F. Henrotte, G. Louppe, C. Geuzaine, "Accounting for hysteresis and eddy
  currents in finite element simulations of ferromagnetic laminated cores using a recurrent neural
  network," arXiv:2607.14321, Jul. 2026.** **[abstract]** An RNN material surrogate embedded in
  2D FE at "about twice" the cost of an anhysteretic simulation. This is an example of the "tool
  compatibility" requirement Why MagNet raises.

### 5.4 Recurring themes in this literature [inferred synthesis]

1. Plain RNNs fit the training distribution well but generalize poorly to unseen excitation paths
   (Chandra 2024/2025). They also degrade under autoregressive rollout (PI-HNO's MC2-LSTM and
   Transformer baselines).
2. Successful small models add structure: physics-shaped states (HystRNN, MMINN, PI-HNO), an
   intermediate B–H layer (HARDCORE), play operators (Grech, K26), or careful state
   initialization (RHINO-MAG).
3. The sequence error and the loss/energy error can disagree (MagNet-AI). Models tuned on one may
   lose on the other.

---

## Framing points for our paper

1. **The size–accuracy trade-off is an established, organizer-endorsed axis, and the frontier has
   only been charted for steady-state loss.**
   - How MagNet's FNN sweep (21 → 5,515 params) shows accuracy saturating above about 1.6k params.
   - MC1 (OJPEL) "quantitatively verified the fundamental tradeoff". Winners sat at "about 1,000
     parameters" for under 10 % p95. Among top teams, accuracy differences are "very small" while
     size differences are "very large". Model size is counted as stored parameters per material,
     and the rules favour "more computation and less memory".
   - HARDCORE's own Pareto study puts its optimum at 906 params.
   - For MC2 the leaderboard spans 83 to 13.2 M parameters, and the APEC keynote plots a
     parameter-versus-error Pareto front.
   - **Closest prior size sweeps on the transient task (cite these):**
     - RHINO-MAG §IV-A, Fig. 6: a GRU-P hidden-unit sweep (cross-validated on training data).
     - RHINO-MAG §IV-D, Fig. 12: median error versus size per model type on the 5 final
       materials. Its §IV-C wall-clock comparison names three GRU sizes: GRU8 325, GRU32 3,589,
       GRU64 13,317 params.
     - MagLearn 2, Table V / Fig. 7: a "controlled architecture-size analysis" from H64-L1
       (84,353 params) to H192-L3 (1,930,369) on N87.
     - K26: 32 / 656 / 5,552 params (classical versus gain-scheduled circuits).
   - **Narrowed gap:** none of these is a size sweep of an SSM or other non-RNN sequence model
     at the roughly 200–3,600-parameter scale, with multiple seeds, under the official A–E
     metric. That is what our S4D study provides: 205 and 1,001 params with 3 seeds under the
     official A–E metric (RESEARCH_LOG §5.5). The 3,601-param configuration is defined
     (RESEARCH_LOG §4.2) but has no A–E result yet; it is planned. [inferred gap]
2. **The transient problem is new and explicitly carved out of prior work.**
   - Why MagNet's data are steady-state only (initial B unknown, B mean removed). How MagNet
     declares transient modeling "beyond the scope". MC1 excluded it, and the MC1 roadmap lists
     transient, minor loops and dc bias as future work.
   - MagNetX (APEC 2025) and the COMPEL 2025 foundation-model paper define the new target:
     frequency-, time-step- and initial-state-agnostic models.
   - The organizers' reference model is a teacher-forced, single-step, 80-step-window dual LSTM
     (771 / 2,399 params) evaluated autoregressively. On A–E the organizers' own entry scores
     66.85 % p95 at 1,443 params. Under PI-HNO's protocol the same 2,399-param design reaches
     282 % p95.
   - **So training for rollout, not only the architecture, is part of the problem.** Guillod's
     short-to-long horizon curriculum and our multi-step stage address the same issue.
     [inferred]
3. **Classical models fail in two different ways, and this shapes what "explainable" can mean.**
   - The Steinmetz family (3–8 params, or 60 for ci2GSE) outputs scalar per-cycle loss. It cannot
     produce H(t) or open-segment energy, and it misses duty-cycle, relaxation and bias coupling
     (Why MagNet).
   - Preisach is rate-independent with non-physical congruency and FORC-based identification.
     JA has 5 parameters, but its physics and its rate extension are contested (Zirka 2012).
     Dynamic extensions exist (Bertotti 1992; Mayergoyz 1988; Jiles 1994).
   - MC2 evidence is split: classical operators bolted onto a GRU lost (RHINO-MAG), while an
     end-to-end-fitted 32-param play/JA-style circuit was on par with the performance-track
     winner (K26: 19.4 % vs 20.4 % official p95 SRE; the winner has 325 params by the
     team/leaderboard count, 336 by K26's). The defensible claim is that structure helps when it
     is trained end to end. [inferred]
4. **Where an SSM sits: between the organizers' circuit/neural-ODE view and the RNN view.**
   - Guillod's linear R-L material model (dB/dt = r·H + μ0μr·dH/dt; 2 params, 46 % H error)
     rearranges to dH/dt = (dB/dt − r·H)/(μ0μr): one real pole, driven by dB/dt. A diagonal SSM
     layer (S4D) is a bank of such linear modes, discretized with a learned step and mixed by
     pointwise nonlinearities. [inferred mapping; none of the SSM papers discusses magnetics]
     Supporting SSM citations, from ssm-lit (`ssm_and_training.md` §2.8b):
     - S4D (Gu et al., NeurIPS 2022, arXiv:2206.11893), Fig. 1 caption: "The diagonal structure
       allows it to be viewed as a collection of 1-dimensional SSMs". Kernel
       K(t) = Σ C_n e^{tA_n} B_n (§2, eq. 3).
     - S4 (ICLR 2022, arXiv:2111.00396) §2.3: discretizes x' = Ax + Bu with step Δ.
     - Bonassi et al., IFAC-PapersOnLine 2024, doi:10.1016/j.ifacol.2024.08.536
       (arXiv:2312.06211): SSM layers are discretized continuous-time LTI systems, and a deep SSM
       is a deep Wiener model.
   - This gives the SSM a defensible "circuit-like" reading and a continuous-time
     parameterization relevant to the "universal time step" property. Our models still run at a
     fixed 16 MHz on an 80-step window; time-step transfer is untested.
   - **Novelty: claim it narrowly.**
     - I found no structured SSM (S4/S4D/S5/LRU/Mamba) in any MagNet, MC1 or MC2 paper, or in any
       magnetic-hysteresis NN paper. ssm-lit's ML-side search (arXiv, Crossref, DDG; queries in
       `ssm_and_training.md` §2.10) found none either.
     - **But an SSM-for-hysteresis paper exists outside magnetics:** Chen et al., "Composite
       control of a piezoelectric fast steering mirror using a mamba-based dual-axis inverse
       model," Smart Mater. Struct., vol. 35, no. 9, art. 095027, 2026 (online 25 Sep. 2026),
       doi:10.1088/1361-665X/aea30f. It is a Mamba
       inverse model of piezo rate-dependent hysteresis that beat Prandtl–Ishlinskii and NARMA
       baselines. [teammate, ssm-lit]
     - So "first SSM for hysteresis" is **false**. Safe wording: "to our knowledge, the first
       S4-family model for magnetic B–H / H(t) prediction and in the MagNet Challenge".
5. **Two caveats that prior work supports.**
   - (a) Energy and sequence errors can rank models differently (MagNet-AI on phase sensitivity;
     our log §5.5).
   - (b) The measurement floor: MC1 measurement error was "below 20%" max and "below 10%" average,
     with a warning that accuracy beyond measurement accuracy risks overfitting. Small
     differences between top models should be read against this floor. [inferred application to
     MC2, whose measurement error is not stated in the sources I read]
