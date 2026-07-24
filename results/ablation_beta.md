# β-ablation summary

Agent: QMIX · 200,000 steps per point · seed 42 · supply = 2000 MW · 5 episodes x 4 seeds, eps=0
(β = 0.5 point reuses the main checkpoint — identical seed/config/code path.)

| β | WUE (MWh/ep) | σ_fair (outage-rate std) | Jain index | Worst-zone outage (h) | Return |
|---|---|---|---|---|---|
| 0 | 79650 ± 20453 | 0.2416 ± 0.0158 | 0.638 ± 0.070 | 110.5 ± 7.4 | -41.0 ± 9.4 |
| 0.25 | 91126 ± 16904 | 0.0785 ± 0.0089 | 0.924 ± 0.022 | 66.7 ± 4.9 | -47.7 ± 8.8 |
| 0.5 | 85100 ± 25337 | 0.0484 ± 0.0040 | 0.955 ± 0.018 | 53.1 ± 8.0 | -50.0 ± 8.7 |
| 1 | 89895 ± 22701 | 0.0289 ± 0.0053 | 0.986 ± 0.004 | 48.2 ± 9.8 | -51.3 ± 10.7 |

## Direction check (reported as measured, not massaged)

- σ_fair monotonically non-increasing in β: **True** (0.2416 → 0.0785 → 0.0484 → 0.0289)
- WUE monotonically non-decreasing in β: **False** (79650 → 91126 → 85100 → 89895)

At least one direction is NOT monotone at this budget/seed — treat the deviation as a finding (single-seed noise is the first suspect; std columns above give the scale) and report it as such.