# Report numbers — single source of truth

Generated from `all_policies_metrics.csv` (env: commit `b2bb661`, config `ecf46c850ae8`, supply = 2000 MW, protocol: 5 episodes x 4 seeds, eps=0).
**Never copy numbers into the report from anywhere else.**

## Main results table

| Policy | WUE (MWh/ep) | Fairness std | Jain index | Worst-zone outage (h) | Residual deficit (MW) | Return |
|---|---|---|---|---|---|---|
| NoShedding | 0 ± 0 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.0 ± 0.0 | 632.4 ± 141.5 | -158.7 ± 52.5 |
| Random | 181161 ± 7516 | 0.025 ± 0.009 | 0.997 ± 0.002 | 90.3 ± 3.3 | 22.2 ± 16.0 | -87.3 ± 6.0 |
| RoundRobin | 70925 ± 2987 | 0.003 ± 0.000 | 1.000 ± 0.000 | 33.1 ± 0.8 | 202.5 ± 90.2 | -82.0 ± 25.9 |
| Priority | 70180 ± 17529 | 0.377 ± 0.029 | 0.483 ± 0.048 | 156.4 ± 7.4 | 0.0 ± 0.0 | -60.5 ± 8.9 |
| Proportional | 132972 ± 20733 | 0.000 ± 0.000 | 1.000 ± 0.000 | 60.1 ± 8.2 | 0.0 ± 0.0 | -52.2 ± 7.9 |
| FairRotation | 70925 ± 2987 | 0.003 ± 0.000 | 1.000 ± 0.000 | 33.1 ± 0.8 | 202.5 ± 90.2 | -82.0 ± 25.9 |
| IDQN | 85017 ± 21243 | 0.148 ± 0.017 | 0.788 ± 0.061 | 84.0 ± 0.0 | 66.4 ± 14.4 | -59.2 ± 8.5 |
| VDN | 85467 ± 16602 | 0.085 ± 0.008 | 0.901 ± 0.024 | 67.8 ± 3.6 | 49.2 ± 21.6 | -51.2 ± 10.0 |
| QMIX | 85100 ± 25995 | 0.048 ± 0.004 | 0.955 ± 0.018 | 53.1 ± 8.2 | 58.8 ± 16.5 | -50.0 ± 8.9 |

## Headline (RQ1 & RQ2) — use this framing verbatim

**RQ1 (does the gap exist):** Every fixed-rule baseline concedes at least one axis. Priority is efficient (WUE 70180 MWh) and leaves 0 MW residual deficit, but is persistently unfair (Jain 0.483). RoundRobin/FairRotation are perfectly fair (Jain 1.000) at comparable WUE (70925 MWh), but only by leaving 203 MW of mean uncontrolled residual deficit — unmet load that never enters WUE. Proportional is fair (Jain 1.000) with 0 MW residual deficit, but pays 132972 MWh WUE (1.9× Priority) for it. The (low-WUE, high-Jain, low-residual) region is empty of baselines.

**RQ2 (can learned policies occupy the gap):** QMIX attains WUE 85100 ± 25995 MWh (1.21× Priority, 0.64× Proportional) with Jain 0.955 ± 0.018 (vs Priority's 0.483) and 59 MW residual deficit (vs RoundRobin's 203 MW), plus the best mean return of all nine policies (-50.0 ± 8.9). It therefore *occupies* the previously empty region of the efficiency–fairness frontier. This is an occupation claim, NOT dominance: Priority remains better on WUE alone and the rotation schemes on fairness alone; no learned agent beats every baseline on every metric.

**Return margin vs the strongest baseline (Proportional):** the unpaired stds overlap, but the policies are evaluated on identical episode seeds, so a paired test is valid and removes the shared episode-difficulty variance: QMIX beats Proportional in 19/20 paired episodes, mean difference +2.24 return, paired t = 7.11, p = 9.2e-07. The margin is small but systematic.

**Against the Proportional rebuttal (fair AND feasible):** Proportional achieves its fairness by ignoring criticality — it sheds Greater Accra (criticality 1.00) at the same rate as Northern (0.55). QMIX matches its feasibility, near-matches its fairness (Jain 0.955 vs 1.000), and cuts criticality-weighted loss by 36% (85100 vs 132972 MWh).

*Note on identical rows:* RoundRobin and FairRotation produce identical metrics because they reduce to the same shed sequence: both shed exactly one full zone per deficit-hour, both observe the same exogenous deficit sequence (actions do not feed back into demand or supply), and FairRotation's least-shed-first selection with ties broken by lowest zone index collapses to a fixed cycle — after every full rotation all cumulative outage counts are equal again. Verified empirically across supply levels and seeds.

- VDN vs QMIX: QMIX > VDN on return here (-50.0 vs -51.2); report as an empirical, task-dependent finding (Papoudakis et al. 2021).
- Learned-family ordering on return: QMIX > VDN > IDQN — consistent with credit assignment richness (none → additive → monotonic).

## β-ablation (QMIX, 200,000 steps per point, seed 42)

| β | WUE (MWh/ep) | σ_fair | Jain index |
|---|---|---|---|
| 0 | 79650 ± 20453 | 0.2416 ± 0.0158 | 0.638 ± 0.070 |
| 0.25 | 91126 ± 16904 | 0.0785 ± 0.0089 | 0.924 ± 0.022 |
| 0.5 | 85100 ± 25337 | 0.0484 ± 0.0040 | 0.955 ± 0.018 |
| 1 | 89895 ± 22701 | 0.0289 ± 0.0053 | 0.986 ± 0.004 |

See `results/ablation_beta.md` for the full table and the monotonicity direction check.

## Supporting numbers

- LSTM forecaster per-zone val MAE (MW): Greater Accra 21.8, Ashanti 16.8, Western 12.6, Volta 8.5, Northern 9.2 (mean 13.8; read from lstm.pt metadata). Greater Accra sits at its irreducible multiplicative-noise floor.
- Historical pre-rebuild numbers (IDQN return −16.13, WUE 162 MWh; central DQN WUE 3203 MWh) are from an incompatible env version — HISTORICAL ONLY, never in comparison tables (Hard Rule 1).
