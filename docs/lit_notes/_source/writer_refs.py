"""Build docs/LITERATURE_REVIEW.md from the keyed master LITERATURE_REVIEW.keyed.md.

Edit the keyed master (citations as [REF:key]; add new keys to R below), then run
    python docs/lit_notes/_source/writer_refs.py
It numbers citations by first appearance, merges adjacent ones, reflows paragraphs to 92
columns and writes section 10 from R.
"""
import re, sys, shutil

from pathlib import Path
HERE = Path(__file__).resolve().parent
P = str(HERE.parent.parent / "LITERATURE_REVIEW.md")
GH = "https://github.com/minjiechen/magnetchallenge-2/blob/main/"

R = {
# --- MagNet program, MC2 organizer material ---
"mc2-repo": "MagNet 2025 Organizing Team (Princeton Univ. and Dartmouth College), \"MagNet Challenge 2,\" GitHub repository, https://github.com/minjiechen/magnetchallenge-2, commit ab4913c (Aug. 3, 2026). README: task statement, judging criteria, News #13 \"Preliminary ranking of 32 teams\" (leaderboard), News #16 and #17; tutorial and pretest notebooks; links to the final-evaluation data and all teams' submitted results.",
"mc2-eval-script": "MagNet 2025 Organizing Team, \"Evaluation_Read.m\" (official MC2 scoring script) and A–E ground-truth data, in \"MagNet Challenge 2 final evaluation data and all teams' results,\" released via the data link in [mc2-repo], accessed Nov. 3, 2026.",
"mc2-winners": "MagNet 2025 Organizing Team, \"MagNet Challenge 2 Winners (March 10th, 2026),\" announcement image, README News #16, " + GH.replace("blob/main/", "blob/main/") + "img/winners.jpg.",
"mc2-talks": "MagNet Challenge 2 team presentations (five-minute talks), Feb. 9–12, 2026, Princeton Power Electronics YouTube channel. Cited via auto-generated captions: Hefei https://www.youtube.com/watch?v=-erftMlwTno; Fuzhou https://www.youtube.com/watch?v=pHzgWAF_6Co; Sydney https://www.youtube.com/watch?v=8Y3p8dFIU-Q; Georgia Tech https://www.youtube.com/watch?v=0-705NtutoE; Cambridge https://www.youtube.com/watch?v=2AkU2jc_aUU; Bristol https://www.youtube.com/watch?v=W7GEu5IXt68; Southeast1 https://www.youtube.com/watch?v=6REL90As66U; Eindhoven https://www.youtube.com/watch?v=6pHZcLsHtgQ; Tsinghua https://www.youtube.com/watch?v=bW8KJCIxRa8; Silicon Austria Labs https://www.youtube.com/watch?v=asrbcj6cSSk; NITC https://www.youtube.com/watch?v=nFvaGZiIaTA; Tianjin https://www.youtube.com/watch?v=Ddc0bGuUC10; KU Leuven https://www.youtube.com/watch?v=PYCW11p2ENk.",
"mc2-handbook": "IEEE PELS, Princeton Univ., and Dartmouth College, \"2025 PELS MagNet Challenge 2 (MagNet 2025): 'from Steady State Losses to Transient Hysteresis!',\" competition handbook, 2025 (header reads \"Updated on January 30th, 2024\"), " + GH + "docs/Handbook-2025.pdf.",
"mc2-slides": "S. Wang, H. Kwon, H. Li, T. Guillod, M. Chen, and C. R. Sullivan (MagNet 2025 Organizing Team), \"MagNet Challenge 2: 'From steady-state to transient models!',\" kickoff meeting slides, 2nd IEEE Int. Challenge in Design Methods for Power Electronics, IEEE PELS, Feb. 25, 2025, slide 17 (\"Evaluation Criteria\"), " + GH + "docs/Slides-2025.pdf.",
"mc2-tutorial1": "S. Wang, H. Kwon, H. Li, T. Guillod, M. Chen, and C. R. Sullivan, \"MagNet Challenge 2: 'From steady-state to transient models!',\" Tutorial Session 1 (data-driven methods), May 16, 2025, " + GH + "tutorials/tutorial-1/Slides-Tutorial-1.pdf.",
"mc2-tutorial2": "T. Guillod, \"Soft magnetic materials and ordinary differential equations: From linear circuits to neural network models,\" IEEE MagNet Challenge webinar (Tutorial 2), Dartmouth College, May 23, 2025, " + GH + "tutorials/tutorial-2/Slides-Tutorial-2.pdf.",
"magnetic-ode-code": "T. Guillod, \"magnetic_ode_models,\" GitHub repository (MPL-2.0), https://github.com/otvam/magnetic_ode_models.",
"mc2-tutorial4": "M. Chen, S. Wang, H. Kwon, H. Li, T. Guillod, and C. R. Sullivan, \"MagNet Challenge 2: 'From steady-state to transient models!',\" Tutorial Session 4, Jun. 6, 2025, " + GH + "tutorials/tutorial-4/Slides-Tutorial-4.pdf.",
"mc2-webinar6": "H. Kwon, S. Wang, H. Li, T. Guillod, M. Chen, and C. R. Sullivan, \"MagNet Challenge 2: 'From steady-state to transient models!',\" Tutorial Session 6 (final evaluation), Oct. 30, 2025, " + GH + "tutorials/tutorial-6/Webinar-6-2025.pdf.",
"mc2-keynote": "M. Chen and C. R. Sullivan, \"MagNet Challenge: people · magnetics · AI,\" keynote slides, IEEE Applied Power Electronics Conf. (APEC), Mar. 2026, slide 16 (\"MagNet Challenge 2 Winners\"), " + GH + "docs/apeckeynote.pdf.",
"why-magnet": "D. Serrano, H. Li, S. Wang, T. Guillod, M. Luo, V. Bansal, N. K. Jha, Y. Chen, C. R. Sullivan, and M. Chen, \"Why MagNet: Quantifying the complexity of modeling power magnetic material characteristics,\" *IEEE Trans. Power Electron.*, vol. 38, no. 11, pp. 14292–14316, Nov. 2023, doi: 10.1109/TPEL.2023.3291084.",
"how-magnet": "H. Li, D. Serrano, T. Guillod, S. Wang, E. Dogariu, A. Nadler, M. Luo, V. Bansal, N. K. Jha, Y. Chen, C. R. Sullivan, and M. Chen, \"How MagNet: Machine learning framework for modeling power magnetic material characteristics,\" *IEEE Trans. Power Electron.*, vol. 38, no. 12, pp. 15829–15853, Dec. 2023, doi: 10.1109/TPEL.2023.3309232.",
"magnet-ai": "H. Li, D. Serrano, S. Wang, and M. Chen, \"MagNet-AI: Neural network as datasheet for magnetics modeling and material recommendation,\" *IEEE Trans. Power Electron.*, vol. 38, no. 12, pp. 15854–15869, Dec. 2023, doi: 10.1109/TPEL.2023.3309233.",
"ojpel-mc1": "M. Chen, H. Li, S. Wang, T. Guillod, D. Serrano, *et al.* (132 authors), \"MagNet Challenge for data-driven power magnetics modeling,\" *IEEE Open J. Power Electron.*, vol. 6, pp. 883–898, 2025 (published online Sep. 30, 2024), doi: 10.1109/OJPEL.2024.3469916.",
"compel2025": "S. Wang, H. Kwon, D. Grigoryan, H. Li, T. Guillod, C. R. Sullivan, and M. Chen, \"Unified time domain foundation models for hysteretic passive components,\" in *Proc. IEEE Workshop on Control and Modeling for Power Electronics (COMPEL)*, Knoxville, TN, USA, 2025, pp. 1–8, doi: 10.1109/COMPEL57166.2025.11121278. (Abstract only read.)",
"magnetx-db": "H. Kwon, S. Wang, H. Li, Y. Elasser, G.-G. Kang, D. Zhou, D. Grigoryan, and M. Chen, \"MagNetX: Extending the MagNet database for modeling power magnetics in transient,\" in *Proc. IEEE Applied Power Electronics Conf. (APEC)*, 2025, pp. 566–572, doi: 10.1109/APEC48143.2025.10977252; preprint TechRxiv, doi: 10.36227/techrxiv.173396064.40994860/v1. (Abstract read; details from the organizers' slides.)",
"magnetx-models": "S. Wang, H. Kwon, H. Li, Y. Elasser, G.-G. Kang, D. Zhou, D. Grigoryan, and M. Chen, \"MagNetX: Foundation neural network models for simulating power magnetics in transient,\" in *Proc. IEEE Applied Power Electronics Conf. (APEC)*, 2025, pp. 2438–2445, doi: 10.1109/APEC48143.2025.10977420; preprint TechRxiv, doi: 10.36227/techrxiv.173396153.34393547/v1. (Abstract read; details from the organizers' slides.)",
"magnetx-tpel": "S. Wang, H. Kwon, H. Li, T. Guillod, H. Wouters, C. R. Sullivan, and M. Chen, \"MagNetX: Data-driven time-domain foundation models for power magnetics in transient,\" *IEEE Trans. Power Electron.*, vol. 41, no. 11, pp. 19455–19474, Nov. 2026, doi: 10.1109/TPEL.2026.3685464. (Metadata only; full text not read.)",
# --- classical models ---
"steinmetz": "C. P. Steinmetz, \"On the law of hysteresis,\" *Trans. AIEE*, vol. IX, no. 1, pp. 1–64, Jan. 1892, doi: 10.1109/T-AIEE.1892.5570437; reprinted in *Proc. IEEE*, vol. 72, no. 2, pp. 197–221, Feb. 1984, doi: 10.1109/PROC.1984.12842.",
"mse": "J. Reinert, A. Brockmeyer, and R. W. A. A. De Doncker, \"Calculation of losses in ferro- and ferrimagnetic materials based on the modified Steinmetz equation,\" *IEEE Trans. Ind. Appl.*, vol. 37, no. 4, pp. 1055–1061, 2001, doi: 10.1109/28.936396. (Metadata only.)",
"igse": "K. Venkatachalam, C. R. Sullivan, T. Abdallah, and H. Tacca, \"Accurate prediction of ferrite core loss with nonsinusoidal waveforms using only Steinmetz parameters,\" in *Proc. IEEE Workshop on Computers in Power Electronics (COMPEL)*, 2002, pp. 36–41, doi: 10.1109/CIPE.2002.1196712. (Abstract only read.)",
"i2gse": "J. Mühlethaler, J. Biela, J. W. Kolar, and A. Ecklebe, \"Improved core-loss calculation for magnetic components employed in power electronic systems,\" *IEEE Trans. Power Electron.*, vol. 27, no. 2, pp. 964–973, Feb. 2012, doi: 10.1109/TPEL.2011.2162252. (Abstract only read.)",
"cwh": "C. R. Sullivan, J. H. Harris, and E. Herbert, \"Core loss predictions for general PWM waveforms from a simplified set of measured data,\" in *Proc. IEEE Applied Power Electronics Conf. (APEC)*, 2010, pp. 1048–1055, doi: 10.1109/APEC.2010.5433375. (Abstract only read.)",
"igcc": "T. Guillod, J. S. Lee, H. Li, S. Wang, M. Chen, and C. R. Sullivan, \"Calculation of ferrite core losses with arbitrary waveforms using the composite waveform hypothesis,\" in *Proc. IEEE Applied Power Electronics Conf. (APEC)*, 2023, pp. 1586–1593, doi: 10.1109/APEC43580.2023.10131348. (Abstract only read.)",
"stenglein": "E. Stenglein and T. Dürbaum, \"Core loss model for arbitrary excitations with DC bias covering a wide frequency range,\" *IEEE Trans. Magn.*, vol. 57, no. 6, 2021, doi: 10.1109/TMAG.2021.3068188. (Abstract only read.)",
"bertotti-loss": "G. Bertotti, \"General properties of power losses in soft ferromagnetic materials,\" *IEEE Trans. Magn.*, vol. 24, no. 1, pp. 621–630, 1988, doi: 10.1109/20.43994. (Metadata only.)",
"preisach": "F. Preisach, \"Über die magnetische Nachwirkung,\" *Z. Phys.*, vol. 94, no. 5–6, pp. 277–302, May 1935, doi: 10.1007/BF01349418. (Metadata only.)",
"mayergoyz": "I. D. Mayergoyz, \"Generalized scalar Preisach models of hysteresis,\" in *Mathematical Models of Hysteresis*. New York, NY, USA: Springer, 1991, pp. 64–140, doi: 10.1007/978-1-4612-3028-1_2. (Metadata only.)",
"moree-preisach": "G. Mörée and M. Leijon, \"Review of play and Preisach models for hysteresis in magnetic materials,\" *Materials*, vol. 16, no. 6, art. 2422, 2023, doi: 10.3390/ma16062422.",
"moree-review": "G. Mörée and M. Leijon, \"Review of hysteresis models for magnetic materials,\" *Energies*, vol. 16, no. 9, art. 3908, 2023, doi: 10.3390/en16093908. (Abstract only read.)",
"bertotti-dynpreisach": "G. Bertotti, \"Dynamic generalization of the scalar Preisach model of hysteresis,\" *IEEE Trans. Magn.*, vol. 28, no. 5, pp. 2599–2601, Sep. 1992, doi: 10.1109/20.179569. (Abstract only read.)",
"mayergoyz-dyn": "I. D. Mayergoyz, \"Dynamic Preisach models of hysteresis,\" *IEEE Trans. Magn.*, vol. 24, no. 6, pp. 2925–2927, 1988, doi: 10.1109/20.92290. (Metadata only.)",
"roussel": "R. Roussel *et al.*, \"Differentiable Preisach modeling for characterization and optimization of particle accelerator systems with hysteresis,\" *Phys. Rev. Lett.*, vol. 128, no. 20, art. 204801, 2022, doi: 10.1103/PhysRevLett.128.204801. (Metadata only.)",
"ja1984": "D. C. Jiles and D. L. Atherton, \"Theory of ferromagnetic hysteresis (invited),\" *J. Appl. Phys.*, vol. 55, no. 6, pp. 2115–2120, 1984, doi: 10.1063/1.333582. (Metadata only.)",
"ja1986": "D. C. Jiles and D. L. Atherton, \"Theory of ferromagnetic hysteresis,\" *J. Magn. Magn. Mater.*, vol. 61, no. 1–2, pp. 48–60, 1986, doi: 10.1016/0304-8853(86)90066-1. (Metadata only.)",
"ja-param1989": "D. C. Jiles and J. B. Thoelke, \"Theory of ferromagnetic hysteresis: Determination of model parameters from experimental hysteresis loops,\" *IEEE Trans. Magn.*, vol. 25, no. 5, pp. 3928–3930, 1989, doi: 10.1109/20.42480. (Abstract only read.)",
"ja-param1992": "D. C. Jiles, J. B. Thoelke, and M. K. Devine, \"Numerical determination of hysteresis parameters for the modeling of magnetic properties using the theory of ferromagnetic hysteresis,\" *IEEE Trans. Magn.*, vol. 28, no. 1, pp. 27–35, 1992, doi: 10.1109/20.119813. (Abstract only read.)",
"marion2008": "R. Marion, R. Scorretti, N. Siauve, M.-A. Raulet, and L. Krähenbühl, \"Identification of Jiles–Atherton model parameters using particle swarm optimization,\" *IEEE Trans. Magn.*, vol. 44, no. 6, pp. 894–897, Jun. 2008, doi: 10.1109/TMAG.2007.914867. (Title only.)",
"trapanese2011": "M. Trapanese, \"Identification of parameters of the Jiles–Atherton model by neural networks,\" *J. Appl. Phys.*, vol. 109, no. 7, art. 07D355, 2011, doi: 10.1063/1.3569735. (Title only.)",
"zirka2012": "S. E. Zirka, Y. I. Moroz, R. G. Harrison, and K. Chwastek, \"On physical aspects of the Jiles–Atherton hysteresis models,\" *J. Appl. Phys.*, vol. 112, no. 4, art. 043916, 2012, doi: 10.1063/1.4747915. (Abstract only read.)",
"jiles1994": "D. C. Jiles, \"Frequency dependence of hysteresis curves in conducting magnetic materials,\" *J. Appl. Phys.*, vol. 76, no. 10, pp. 5849–5855, 1994, doi: 10.1063/1.358399. (Abstract only read.)",
"zhu1996": "J. G. Zhu, S. Y. R. Hui, and V. S. Ramsden, \"A generalized dynamic circuit model of magnetic cores for low- and high-frequency applications—Part I,\" *IEEE Trans. Power Electron.*, vol. 11, no. 2, pp. 246–250, 1996, doi: 10.1109/63.486172; S. Y. R. Hui, J. G. Zhu, and V. S. Ramsden, \"…—Part II,\" *ibid.*, pp. 251–259, doi: 10.1109/63.486173. (Metadata only.)",
"luo2018": "M. Luo, D. Dujic, and J. Allmeling, \"Modeling frequency independent hysteresis effects of ferrite core materials using permeance–capacitance analogy for system-level circuit simulations,\" *IEEE Trans. Power Electron.*, vol. 33, no. 12, pp. 10055–10070, 2018, doi: 10.1109/TPEL.2018.2809704. (Metadata only.)",
"cui2023": "H. H. Cui, S. Dulal, S. B. Sohid, G. Gu, and L. M. Tolbert, \"Unveiling the microworld inside magnetic materials via circuit models,\" *IEEE Power Electron. Mag.*, vol. 10, no. 3, pp. 14–22, 2023, doi: 10.1109/MPEL.2023.3301408. (Metadata only.)",
"dulal2024": "S. Dulal *et al.*, \"A physics-based circuit model for nonlinear magnetic material characteristics,\" in *Proc. IEEE Applied Power Electronics Conf. (APEC)*, 2024, pp. 396–401, doi: 10.1109/APEC48139.2024.10509097. (Metadata only.)",
"hardcore": "W. Kirchgässner, N. Förster, T. Piepenbrock, O. Schweins, and O. Wallscheid, \"HARDCORE: H-field and power loss estimation for arbitrary waveforms with residual, dilated convolutional neural networks in ferrite cores,\" *IEEE Trans. Power Electron.*, vol. 40, no. 2, pp. 3326–3335, Feb. 2025, doi: 10.1109/TPEL.2024.3488174. (Abstract only read.)",
"mminn": "Q. Huang, Y. Li, J. Zhu, and S. Li, \"Magnetization mechanism-inspired neural networks for core loss estimation,\" *IEEE Trans. Power Electron.*, vol. 39, no. 12, pp. 16382–16390, Dec. 2024, doi: 10.1109/TPEL.2024.3450897. (Abstract only read.)",
# --- MC2 teams and follow-ups ---
"rhino": "H. Vater, O. Schweins, L. Hölsch, W. Kirchgässner, T. Piepenbrock, and O. Wallscheid, \"RHINO-MAG: Recursive H-field inference based on observed magnetic flux density under dynamic excitation,\" arXiv:2603.29745 [eess.SY], v1 Mar. 31, 2026, v2 Sep. 9, 2026 (v2 read); accepted to *IEEE Trans. Power Electron.*, early access, doi: 10.1109/TPEL.2026.3733311.",
"rhino-code": "H. Vater *et al.* (Paderborn Univ., Dept. LEA), \"RHINO-MAG,\" GitHub repository (MIT license), https://github.com/upb-lea/RHINO-MAG, commit aec7e99 (Oct. 2, 2026); includes the organizers' per-block results for the submitted models.",
"k26": "W. Kirchgässner, \"A data-driven, gain-scheduled equivalent circuit for the B–H behavior of power ferrites,\" arXiv:2611.02222 [eess.SY], Nov. 1, 2026 (preprint, not peer reviewed); code: https://github.com/wkirgsn/ferrite-bh-circuit.",
"hefei": "Y. Wang, Y. Liu, L. Mo, W. Wang, S. Ren, G. Cai, S. Liu, H. Wu, M. Ma, and C. Q. Jiang, \"Framework of LSTM-based direction-dependent asymmetric operator model for B-H prediction under transient excitations,\" *IEEE Trans. Power Electron.*, vol. 41, no. 12, pp. 21230–21243, Dec. 2026, doi: 10.1109/TPEL.2026.3715566; preprint TechRxiv, doi: 10.36227/techrxiv.177162964.49399026. (Abstract only read.)",
"hefei-code": "HFUT-CityUHK-XMU team, \"Mag-LSTM-DAO-Inference\" (competition model, weights and normalization), https://github.com/ouihon/Mag-LSTM-DAO-Inference, and \"LSTM-DAO-Inference\" (design application with fine-tuning script), https://github.com/ouihon/LSTM-DAO-Inference, GitHub repositories.",
"maglearn2": "J. Yang, B. Cui, Y. Huo, L. Zhang, A. Martinez, S. Liu, and J. Wang, \"MagLearn 2: High-fidelity, saturation-aware, and history-efficient sequence-to-sequence modeling of transient B-H behaviour under PWM excitation,\" arXiv:2610.03504, Oct. 2026 (submitted to *IEEE Trans. Power Electron.*); conference version at IPEC-Nagasaki 2026 (ECCE Asia), IEEE Xplore document 11597291; code (Materials D and E): https://github.com/JunWang-Bristol/MagLearn2.",
"pi-hno": "Y. Zhu, Q. Huang, S. Li, Y. Li, G. Lei, and J. Zhu, \"A physics-informed hybrid neural operator for transient magnetization prediction in power magnetics,\" arXiv:2608.02965v3, Aug. 2026 (preprint prepared for possible submission to *IEEE Trans. Power Electron.*).",
"gatech": "P. Chauhan, Y. Mo, D. Li, L. Chang, X. Xian, and B. Ge, \"Dynamic magnetic modeling with magnetization, hysteresis, eddy, and displacement currents,\" *IEEE Trans. Power Electron.*, vol. 41, no. 12, pp. 20869–20873, 2026 (letter), doi: 10.1109/TPEL.2026.3712771. (Abstract only read.)",
# --- recurrent nets, sysid ---
"lstm": "S. Hochreiter and J. Schmidhuber, \"Long short-term memory,\" *Neural Comput.*, vol. 9, no. 8, pp. 1735–1780, 1997, doi: 10.1162/neco.1997.9.8.1735.",
"gru": "K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, \"Learning phrase representations using RNN encoder–decoder for statistical machine translation,\" in *Proc. EMNLP*, 2014, pp. 1724–1734, doi: 10.3115/v1/D14-1179, arXiv:1406.1078.",
"rnn-gradients": "Y. Bengio, P. Simard, and P. Frasconi, \"Learning long-term dependencies with gradient descent is difficult,\" *IEEE Trans. Neural Netw.*, vol. 5, no. 2, pp. 157–166, 1994, doi: 10.1109/72.279181.",
"pascanu2013": "R. Pascanu, T. Mikolov, and Y. Bengio, \"On the difficulty of training recurrent neural networks,\" in *Proc. ICML*, PMLR vol. 28, no. 3, pp. 1310–1318, 2013, arXiv:1211.5063.",
"leontaritis1985": "I. J. Leontaritis and S. A. Billings, \"Input-output parametric models for non-linear systems, Part I: Deterministic non-linear systems,\" *Int. J. Control*, vol. 41, no. 2, pp. 303–328, 1985, doi: 10.1080/0020718508961129; \"Part II: Stochastic non-linear systems,\" *ibid.*, pp. 329–344, doi: 10.1080/0020718508961130. (Metadata only.)",
"narendra1990": "K. S. Narendra and K. Parthasarathy, \"Identification and control of dynamical systems using neural networks,\" *IEEE Trans. Neural Netw.*, vol. 1, no. 1, pp. 4–27, 1990, doi: 10.1109/72.80202.",
"ribeiro2018": "A. H. Ribeiro and L. A. Aguirre, \"'Parallel training considered harmful?': Comparing series-parallel and parallel feedforward network training,\" *Neurocomputing*, vol. 316, pp. 222–231, 2018, doi: 10.1016/j.neucom.2018.07.071, arXiv:1706.07119.",
"andersson2019": "C. Andersson, A. H. Ribeiro, K. Tiels, N. Wahlström, and T. B. Schön, \"Deep convolutional networks in system identification,\" in *Proc. IEEE Conf. Decision and Control (CDC)*, 2019, doi: 10.1109/CDC40024.2019.9030219, arXiv:1909.01730. (Abstract only read.)",
"bonassi2024": "F. Bonassi, C. Andersson, P. Mattsson, and T. B. Schön, \"Structured state-space models are deep Wiener models,\" *IFAC-PapersOnLine*, vol. 58, no. 15, pp. 247–252, 2024 (SYSID 2024), doi: 10.1016/j.ifacol.2024.08.536, arXiv:2312.06211.",
"forgione2022": "M. Forgione, M. Mejari, and D. Piga, \"Learning neural state-space models: Do we need a state estimator?\" arXiv:2206.12928, 2022.",
"subnet": "G. I. Beintema, M. Schoukens, and R. Tóth, \"Deep subspace encoders for nonlinear system identification,\" *Automatica*, vol. 156, art. 111210, 2023, doi: 10.1016/j.automatica.2023.111210, arXiv:2210.14816.",
"gedon2021": "D. Gedon, N. Wahlström, T. B. Schön, and L. Ljung, \"Deep state space models for nonlinear system identification,\" *IFAC-PapersOnLine*, 2021, doi: 10.1016/j.ifacol.2021.08.406, arXiv:2003.14162. (Abstract only read.)",
"ribeiro2020": "A. H. Ribeiro, K. Tiels, J. Umenberger, T. B. Schön, and L. A. Aguirre, \"On the smoothness of nonlinear system identification,\" *Automatica*, vol. 121, art. 109158, 2020, doi: 10.1016/j.automatica.2020.109158, arXiv:1905.00820.",
"forgione-adapt": "M. Forgione, A. Muni, D. Piga, and M. Gallieri, \"On the adaptation of recurrent neural networks for system identification,\" *Automatica*, art. 111092, 2023, doi: 10.1016/j.automatica.2023.111092, arXiv:2201.08660. (Abstract only read.)",
"forgione-icl": "M. Forgione, F. Pura, and D. Piga, \"From system models to class models: An in-context learning paradigm,\" *IEEE Control Syst. Lett.*, 2023, doi: 10.1109/LCSYS.2023.3335036, arXiv:2308.13380. (Abstract only read.)",
# --- SSMs ---
"s4": "A. Gu, K. Goel, and C. Ré, \"Efficiently modeling long sequences with structured state spaces,\" in *Proc. ICLR*, 2022, arXiv:2111.00396.",
"s4d": "A. Gu, A. Gupta, K. Goel, and C. Ré, \"On the parameterization and initialization of diagonal state space models,\" in *Proc. NeurIPS*, 2022, doi: 10.52202/068431-2607, arXiv:2206.11893.",
"s4-repo": "A. Gu *et al.*, \"state-spaces/s4\" (includes the minimal S4D reference implementation), GitHub repository (Apache-2.0), https://github.com/state-spaces/s4.",
"dss": "A. Gupta, A. Gu, and J. Berant, \"Diagonal state spaces are as effective as structured state spaces,\" in *Proc. NeurIPS*, 2022, arXiv:2203.14343.",
"hippo-train": "A. Gu, I. Johnson, A. Timalsina, A. Rudra, and C. Ré, \"How to train your HiPPO: State space models with generalized orthogonal basis projections,\" in *Proc. ICLR*, 2023, arXiv:2206.12037.",
"s5": "J. T. H. Smith, A. Warrington, and S. W. Linderman, \"Simplified state space layers for sequence modeling,\" in *Proc. ICLR*, 2023, arXiv:2208.04933.",
"lru": "A. Orvieto, S. L. Smith, A. Gu, A. Fernando, C. Gulcehre, R. Pascanu, and S. De, \"Resurrecting recurrent neural networks for long sequences,\" in *Proc. ICML*, 2023, arXiv:2303.06349.",
"mamba": "A. Gu and T. Dao, \"Mamba: Linear-time sequence modeling with selective state spaces,\" in *Proc. Conf. Language Modeling (COLM)*, 2024, arXiv:2312.00752.",
"orvieto-universality": "A. Orvieto, S. De, C. Gulcehre, R. Pascanu, and S. L. Smith, \"Universality of linear recurrences followed by non-linear projections: Finite-width guarantees and benefits of complex eigenvalues,\" in *Proc. ICML*, 2024, arXiv:2307.11888. (Abstract only read.)",
"ranmilo2024": "Y. Ran-Milo, E. Lumbroso, E. Cohen-Karlik, R. Giryes, A. Globerson, and N. Cohen, \"Provable benefits of complex parameterizations for structured state space models,\" in *Proc. NeurIPS*, 2024, arXiv:2410.14067. (Abstract only read.)",
"sashimi": "K. Goel, A. Gu, C. Donahue, and C. Ré, \"It's raw! Audio generation with state-space models,\" arXiv:2202.09729, 2022 (venue not verified).",
"hu2024": "Z. Hu, N. A. Daryakenari, Q. Shen, K. Kawaguchi, and G. E. Karniadakis, \"State-space models are accurate and efficient neural operators for dynamical systems,\" arXiv:2409.03231, 2024. (Abstract only read.)",
"chen2026-piezo": "S. Chen, T. Liu, W. Feng, W. Zhan, R. Ji, L. Wang, and K. Li, \"Composite control of a piezoelectric fast steering mirror using a mamba-based dual-axis inverse model,\" *Smart Mater. Struct.*, vol. 35, no. 9, art. 095027, 2026, doi: 10.1088/1361-665X/aea30f. (Abstract only read.)",
"longssm": "S. Wang, \"LongSSM: On the length extension of state-space models in language modelling,\" arXiv:2406.02080, 2024. (Abstract only read.)",
# --- hysteresis NNs ---
"serpico1998": "C. Serpico and C. Visone, \"Magnetic hysteresis modeling via feed-forward neural networks,\" *IEEE Trans. Magn.*, vol. 34, no. 3, pp. 623–628, 1998, doi: 10.1109/20.668055. (Abstract only read.)",
"makaveev2001": "D. Makaveev, L. Dupré, M. De Wulf, and J. Melkebeek, \"Modeling of quasistatic magnetic hysteresis with feed-forward neural networks,\" *J. Appl. Phys.*, vol. 89, no. 11, pp. 6737–6739, 2001, doi: 10.1063/1.1361268. (Metadata only.)",
"grech2020": "C. Grech, M. Buzio, M. Pentella, and N. Sammut, \"Dynamic ferromagnetic hysteresis modelling using a Preisach-recurrent neural network model,\" *Materials*, vol. 13, no. 11, art. 2561, 2020, doi: 10.3390/ma13112561. (Abstract only read.)",
"hystrnn": "A. Chandra, T. Kapoor, B. Daniels, M. Curti, K. Tiels, D. M. Tartakovsky, and E. A. Lomonova, \"Generalizable models of magnetic hysteresis via physics-aware recurrent neural networks,\" *Comput. Phys. Commun.*, vol. 314, art. 109650, 2025, doi: 10.1016/j.cpc.2025.109650; preprint \"Neural oscillators for magnetic hysteresis modeling,\" arXiv:2308.12002 (preprint read).",
"chandra-no": "A. Chandra, B. Daniels, M. Curti, K. Tiels, and E. A. Lomonova, \"Magnetic hysteresis modeling with neural operators,\" *IEEE Trans. Magn.*, vol. 61, no. 1, pp. 1–11, Jan. 2025, doi: 10.1109/TMAG.2024.3496695, arXiv:2407.03261.",
"guo-no2025": "Z. Guo, B. H. Nguyen, H. Hamzehbahmani, and R. V. Sabariego, \"Dynamic hysteresis model of grain-oriented ferromagnetic material using neural operators,\" *IEEE Trans. Magn.*, 2025, doi: 10.1109/TMAG.2025.3600089, arXiv:2504.04863. (Abstract only read.)",
"guo-fno2026": "Z. Guo, X. Shen, and R. V. Sabariego, \"A multi-scale ResNet-augmented Fourier neural operator framework for high-frequency sequence-to-sequence prediction of magnetic hysteresis,\" arXiv:2604.04150, 2026. (Abstract only read.)",
"neural-ode": "R. T. Q. Chen, Y. Rubanova, J. Bettencourt, and D. Duvenaud, \"Neural ordinary differential equations,\" arXiv:1806.07366, 2018. (Abstract only read.)",
"neural-cde": "P. Kidger, J. Morrill, J. Foster, and T. Lyons, \"Neural controlled differential equations for irregular time series,\" in *Proc. NeurIPS*, 2020, arXiv:2005.08926.",
# --- rollout training ---
"exposure-bias": "M. Ranzato, S. Chopra, M. Auli, and W. Zaremba, \"Sequence level training with recurrent neural networks,\" in *Proc. ICLR*, 2016, arXiv:1511.06732.",
"dagger": "S. Ross, G. J. Gordon, and J. A. Bagnell, \"A reduction of imitation learning and structured prediction to no-regret online learning,\" in *Proc. AISTATS*, 2011, arXiv:1011.0686.",
"dart": "M. Laskey, J. Lee, R. Fox, A. Dragan, and K. Goldberg, \"DART: Noise injection for robust imitation learning,\" in *Proc. Conf. Robot Learning (CoRL)*, 2017, arXiv:1703.09327.",
"gns": "A. Sanchez-Gonzalez, J. Godwin, T. Pfaff, R. Ying, J. Leskovec, and P. W. Battaglia, \"Learning to simulate complex physics with graph networks,\" in *Proc. ICML*, 2020, arXiv:2002.09405.",
"mp-pde": "J. Brandstetter, D. Worrall, and M. Welling, \"Message passing neural PDE solvers,\" in *Proc. ICLR*, 2022, arXiv:2202.03376.",
"scheduled-sampling": "S. Bengio, O. Vinyals, N. Jaitly, and N. Shazeer, \"Scheduled sampling for sequence prediction with recurrent neural networks,\" in *Proc. NeurIPS*, 2015, arXiv:1506.03099.",
"huszar2015": "F. Huszár, \"How (not) to train your generative model: Scheduled sampling, likelihood, adversary?\" arXiv:1511.05101, 2015. (Abstract only read.)",
"professor-forcing": "A. Lamb, A. Goyal, Y. Zhang, S. Zhang, A. Courville, and Y. Bengio, \"Professor forcing: A new algorithm for training recurrent networks,\" in *Proc. NIPS*, 2016, arXiv:1610.09038. (Abstract only read.)",
"gtf": "F. Hess, Z. Monfared, M. Brenner, and D. Durstewitz, \"Generalized teacher forcing for learning chaotic dynamics,\" in *Proc. ICML*, PMLR vol. 202, pp. 13017–13049, 2023, arXiv:2306.04406.",
"pde-refiner": "P. Lippe, B. S. Veeling, P. Perdikaris, R. E. Turner, and J. Brandstetter, \"PDE-Refiner: Achieving accurate long rollouts with neural PDE solvers,\" in *Proc. NeurIPS*, 2023, doi: 10.52202/075280-2946, arXiv:2308.05732.",
"list2025": "B. List, L.-W. Chen, K. Bali, and N. Thuerey, \"Differentiability in unrolled training of neural physics simulators on transient dynamics,\" *Comput. Methods Appl. Mech. Eng.*, vol. 433, art. 117441, Jan. 2025, doi: 10.1016/j.cma.2024.117441, arXiv:2402.12971.",
"mikhaeil2022": "J. M. Mikhaeil, Z. Monfared, and D. Durstewitz, \"On the difficulty of learning chaotic dynamics with RNNs,\" in *Proc. NeurIPS*, 2022, doi: 10.52202/068431-0821, arXiv:2110.07238. (Abstract only read.)",
"tallec2017": "C. Tallec and Y. Ollivier, \"Unbiasing truncated backpropagation through time,\" arXiv:1705.08209, 2017. (Abstract only read.)",
# --- transfer ---
"mist": "C. Yao and C. Gong, \"Cross-material support transfer for core-loss prediction under waveform covariate shift,\" arXiv:2609.31659, 2026.",
"chronos": "A. F. Ansari *et al.*, \"Chronos: Learning the language of time series,\" arXiv:2403.07815, 2024. (Abstract only read.)",
"timesfm": "A. Das, W. Kong, R. Sen, and Y. Zhou, \"A decoder-only foundation model for time-series forecasting,\" arXiv:2310.10688, 2023. (Abstract only read.)",
}

R.update({
"noel2017": "J. P. Noël, A. F. Esfahani, G. Kerschen, and J. Schoukens, \"A nonlinear state-space approach to hysteresis identification,\" *Mech. Syst. Signal Process.*, vol. 84, pp. 171–184, Feb. 2017, doi: 10.1016/j.ymssp.2016.08.025, arXiv:1610.09138. (Metadata only.)",
"esfahani2018": "A. F. Esfahani, P. Dreesen, K. Tiels, J.-P. Noël, and J. Schoukens, \"Parameter reduction in nonlinear state-space identification of hysteresis,\" *Mech. Syst. Signal Process.*, vol. 104, pp. 884–895, May 2018, doi: 10.1016/j.ymssp.2017.10.017, arXiv:1705.00178. (Metadata only.)",
"krikelis2024": "K. Krikelis, J.-S. Pei, K. van Berkel, and M. Schoukens, \"Identification of structured nonlinear state–space models for hysteretic systems using neural network hysteresis operators,\" *Measurement*, vol. 224, art. 113966, Jan. 2024, doi: 10.1016/j.measurement.2023.113966. (Metadata only.)",
})
R["hefei"] = R["hefei"].replace("techrxiv.177162964.49399026", "techrxiv.177162964.49399026/v1")
R["maglearn2"] = R["maglearn2"].replace("conference version at IPEC-Nagasaki 2026 (ECCE Asia), IEEE Xplore document 11597291", "conference version: B. Cui, J. Yang, Y. Huo, L. Zhang, A. Martinez, S. Liu, and J. Wang, \"MagLearn 2: A short-window Seq2Seq framework for transient modelling of power magnetics under PWM excitations,\" in *Proc. IPEC-Nagasaki 2026 (ECCE Asia)*, 2026, pp. 1–7, doi: 10.23919/IPEC-Nagasaki2026-EC64663.2026.11597291")
R["sashimi"] = "K. Goel, A. Gu, C. Donahue, and C. Ré, \"It's raw! Audio generation with state-space models,\" in *Proc. ICML*, PMLR vol. 162, 2022, arXiv:2202.09729."
R["magnetx-tpel"] = R["magnetx-tpel"].replace("(Metadata only; full text not read.)", "(Abstract only read.)")
R["chronos"] = R["chronos"].replace("arXiv:2403.07815, 2024.", "*Trans. Mach. Learn. Res. (TMLR)*, 2024, arXiv:2403.07815.")
R["timesfm"] = R["timesfm"].replace("arXiv:2310.10688, 2023.", "in *Proc. ICML*, 2024, arXiv:2310.10688.")

R["zhao2024"] = "J. Zhao, J. Zhou, L. Zhang, and J. Sun, \"State space representation of Jiles–Atherton hysteresis model and application for closed-loop control,\" *Materials*, vol. 17, no. 15, art. 3695, 2024, doi: 10.3390/ma17153695. (Metadata only.)"
R["hystrnn"] = R["hystrnn"].replace("(preprint read)", "(preprint read; results are cited from it, and its identity with the journal version is inferred from the identical authors and topic)")
R["exposure-bias"] = R["exposure-bias"] + " (Term popularized here; earliest use not searched exhaustively.)"

R["s4d"] = R["s4d"].replace("in *Proc. NeurIPS*, 2022,", "in *Proc. NeurIPS*, 2022, pp. 35971–35983,")
R["andersson2019"] = R["andersson2019"].replace("(CDC)*, 2019,", "(CDC)*, 2019, pp. 3670–3676,")
R["gedon2021"] = R["gedon2021"].replace("*IFAC-PapersOnLine*, 2021,", "*IFAC-PapersOnLine*, vol. 54, no. 7, pp. 481–486, 2021,")
R["guo-no2025"] = R["guo-no2025"].replace("*IEEE Trans. Magn.*, 2025,", "*IEEE Trans. Magn.*, vol. 61, no. 10, Oct. 2025,")
R["pde-refiner"] = R["pde-refiner"].replace("in *Proc. NeurIPS*, 2023,", "in *Proc. NeurIPS*, 2023, pp. 67398–67433,")
R["forgione-adapt"] = R["forgione-adapt"].replace("*Automatica*, art. 111092, 2023,", "*Automatica*, vol. 155, art. 111092, Sep. 2023,")
R["forgione-icl"] = R["forgione-icl"].replace("*IEEE Control Syst. Lett.*, 2023,", "*IEEE Control Syst. Lett.*, vol. 7, pp. 3513–3518, 2023,")

SRC = str(HERE / "LITERATURE_REVIEW.keyed.md")  # keyed master copy; edit this, then renumber
s = open(SRC).read()
head, sep, _ = s.partition("## 10. References")
assert sep
order = []
for k in re.findall(r"\[REF:([a-z0-9-]+)\]", head):
    if k not in order:
        order.append(k)
missing = [k for k in order if k not in R]
if missing:
    sys.exit(f"missing citations: {missing}")
num = {k: i + 1 for i, k in enumerate(order)}
body = re.sub(r"\[REF:([a-z0-9-]+)\]", lambda m: f"[{num[m.group(1)]}]", head)
# merge runs of adjacent citations: "[3] [7]" -> "[3], [7]"
prev = None
while prev != body:
    prev = body
    body = re.sub(r"\[(\d+)\] \[(\d+)\]", r"[\1], [\2]", body)
    body = re.sub(r"\[(\d+)\]\n\[(\d+)\]", r"[\1],\n[\2]", body)

import textwrap
MARK = re.compile(r"^(\s*)([-*]|\d+\.)\s+")
def reflow_block(lines, width=92):
    if any(l.lstrip().startswith(("|", "#", "```", "<!--", "**Table")) for l in lines) or lines[0].strip() == "---":
        return lines
    items, cur = [], None
    for l in lines:
        m = MARK.match(l)
        if m or cur is None:
            cur = [l]; items.append(cur)
        else:
            cur.append(l)
    out = []
    for it in items:
        m = MARK.match(it[0])
        text = " ".join(x.strip() for x in it)
        if m:
            lead = it[0][:m.end()]
            text = text[len(lead.strip()):].strip()
            ind = " " * len(lead)
            out += textwrap.wrap(text, width, initial_indent=lead, subsequent_indent=ind,
                                 break_long_words=False, break_on_hyphens=False)
        else:
            ind = it[0][:len(it[0]) - len(it[0].lstrip())]
            out += textwrap.wrap(text, width, initial_indent=ind, subsequent_indent=ind,
                                 break_long_words=False, break_on_hyphens=False)
    return out
def reflow(text):
    blocks, cur = [], []
    for l in text.split("\n"):
        if l.strip() == "":
            if cur: blocks.append(cur); cur = []
            blocks.append([""])
        else:
            cur.append(l)
    if cur: blocks.append(cur)
    res = []
    for b in blocks:
        res += b if b == [""] else reflow_block(b)
    return "\n".join(res)
body = reflow(body)
body = body.replace("by citation key. The keys become numbers [n] in the final", "")
refs = ["## 10. References", "",
        "Numbered in order of first citation. Notes in parentheses record how much of each source",
        "the team read; entries without a note were read in full (papers) or inspected directly",
        "(code, slides, data). The HTML comment after each entry is its citation key in",
        "`lit_notes/` and `review_log.md`.", ""]
for k in order:
    text = R[k].replace("[mc2-repo]", f"[{num['mc2-repo']}]")
    refs.append(f"[{num[k]}] {text} <!-- {k} -->")
    refs.append("")
open(P, "w").write(body + "\n".join(refs).rstrip() + "\n")
print(len(order), "references; unused:", [k for k in R if k not in order])
