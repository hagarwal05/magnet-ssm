# Noise-level sweep with the multi-step stage (2026-11-05, MPS)

S4D small (205 params), one seed, official A–E protocol. Stage 1: 80 epochs next-step with
H-history noise σ. Stage 2: 12 epochs of K = 32 multi-step rollout loss (horizon curriculum
1 → 32 over 4 epochs), kept only if it beats stage 1 on validation rollout. Everything else
as phase 2. Launch script: `scripts/runs/sweep_noise.sh`. About 1 h per σ.

## Official sequence error (%)

| setting | A | B | C | D | E | **mean** |
|---|---|---|---|---|---|---|
| σ 0.02 + stage 2 | 100.0 | 23.1 | 76.9 | 46.9 | 56.0 | 60.6 |
| σ 0.05 + stage 2 | 85.0 | 25.7 | 79.1 | 34.6 | 53.0 | 55.5 |
| **σ 0.1 + stage 2** | 83.7 | 24.0 | 51.7 | 38.5 | 46.0 | **48.8** |
| phase 2: σ 0.05, stage 1 only | 85.0 | 51.5 | 79.1 | 36.2 | 53.0 | 61.0 |

Energy error (mean): σ 0.02 22.0, σ 0.05 17.4, **σ 0.1 12.3**, phase 2 17.0.

## Which stage was kept (validation rollout p95, %)

| σ | A | B | C | D | E |
|---|---|---|---|---|---|
| 0.02 | MS 117 < 123 | MS 35.9 < 38.1 | MS 33.8 < 37.1 | MS 37.4 < 40.2 | MS 70.7 < 78.0 |
| 0.05 | NS 86.8 < 87.8 | MS 34.8 < 38.3 | NS 36.9 < 38.5 | MS 35.1 < 40.2 | NS 66.5 < 67.2 |
| 0.1 | NS 86.0 < 89.1 | NS 34.8 < 35.0 | MS 31.4 < 34.1 | MS 33.6 < 39.1 | MS 63.6 < 68.1 |

MS = multi-step stage kept, NS = next-step (stage 1) kept.

## Reading

- **Choice: σ = 0.1 with stage 2.** It has the lowest validation error (mean selected
  rollout p95: 49.9 vs 52.0 / 59.0 for σ 0.05 / 0.02), so the choice did not depend on the
  test set. It also has the lowest test error (48.8 %) and energy error (12.3 %).
- **Stage 2 alone, controlled (σ = 0.05, same seed vs phase 2):** where it was kept it
  improved the test score (B 51.5 → 25.7, D 36.2 → 34.6); where not kept, scores were
  identical to phase 2. Mean 61.0 → 55.5 %. The keep-if-better-on-validation rule caused no
  regression.
- **More noise helped on average** (60.6 → 55.5 → 48.8), mostly through C (76.9 → 51.7) and
  E (56.0 → 46.0). σ = 0.1 is the edge of the range tried, so the optimum may lie higher.
- One seed per setting: differences of a few points are within likely seed noise. The
  3-seed run (`results/seeds3/`) uses σ = 0.1 + stage 2.
