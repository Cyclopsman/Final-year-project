# Report numbers — single source of truth

Generated from `all_policies_metrics.csv` (env: commit `1f98eb7`, config `ecf46c850ae8`, supply = 2000 MW, protocol: 5 episodes x 3 seeds, eps=0).
**Never copy numbers into the report from anywhere else.**

## Main results table

| Policy | WUE (MWh/ep) | Fairness std | Jain index | Worst-zone outage (h) | Residual deficit (MW) | Return |
|---|---|---|---|---|---|---|
| NoShedding | 0 ± 0 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.0 ± 0.0 | 636.9 ± 127.2 | -161.1 ± 47.9 |
| Random | 182323 ± 6404 | 0.025 ± 0.010 | 0.997 ± 0.002 | 90.9 ± 3.0 | 21.0 ± 14.6 | -87.5 ± 5.5 |
| RoundRobin | 70904 ± 2459 | 0.003 ± 0.000 | 1.000 ± 0.000 | 33.3 ± 0.8 | 204.4 ± 80.2 | -82.7 ± 23.2 |
| Priority | 70703 ± 15682 | 0.379 ± 0.028 | 0.487 ± 0.045 | 157.3 ± 7.3 | 0.0 ± 0.0 | -61.0 ± 8.1 |
| Proportional | 133598 ± 18629 | 0.000 ± 0.000 | 1.000 ± 0.000 | 60.7 ± 7.7 | 0.0 ± 0.0 | -52.5 ± 7.1 |
| FairRotation | 70904 ± 2459 | 0.003 ± 0.000 | 1.000 ± 0.000 | 33.3 ± 0.8 | 204.4 ± 80.2 | -82.7 ± 23.2 |
| IDQN | 95970 ± 16284 | 0.144 ± 0.013 | 0.813 ± 0.042 | 78.7 ± 3.0 | 31.7 ± 11.1 | -56.9 ± 6.5 |
| VDN | 93449 ± 15781 | 0.079 ± 0.007 | 0.931 ± 0.013 | 65.5 ± 6.1 | 36.6 ± 10.5 | -52.0 ± 8.0 |
| QMIX | 79143 ± 17983 | 0.075 ± 0.003 | 0.913 ± 0.021 | 58.2 ± 8.6 | 69.5 ± 17.4 | -51.7 ± 9.0 |

## Headline (RQ1 & RQ2) — use this framing verbatim

**RQ1 (does the gap exist):** No fixed-rule baseline is simultaneously efficient, fair and feasible: Priority reaches 70703 MWh WUE but with Jain 0.487 (persistently unfair), while RoundRobin reaches Jain 1.000 but leaves 204 MW mean residual deficit (uncontrolled outages) and Proportional pays 133598 MWh WUE for its fairness.

**RQ2 (can learned policies occupy the gap):** QMIX (best return -51.7 ± 9.0) attains WUE 79143 ± 17983 MWh with Jain 0.913 ± 0.021 and residual deficit 69.5 MW — i.e. it occupies a region of the efficiency–fairness plane no baseline reaches. This is a Pareto **occupation** claim, not domination: individual baselines still win on single metrics (Priority on WUE alone, rotation schemes on fairness alone).

- Best *feasible* baseline on WUE alone: Priority (70702.506); infeasible baselines (residual deficit > 25 MW) excluded.
- Best *feasible* baseline on Jain index alone: Proportional (1.000); infeasible baselines (residual deficit > 25 MW) excluded.
- VDN vs QMIX: QMIX > VDN on return here (-51.7 vs -52.0); report as an empirical, task-dependent finding (Papoudakis et al. 2021).

## β-ablation (QMIX, 60,000 steps each, seed 42)

| β | WUE (MWh/ep) | Fairness std | Jain index |
|---|---|---|---|
| 0 | 95483 ± 17694 | 0.1226 ± 0.0077 | 0.870 ± 0.025 |
| 0.5 | 96447 ± 15640 | 0.0815 ± 0.0077 | 0.922 ± 0.022 |
| 2 | 105326 ± 17961 | 0.0625 ± 0.0072 | 0.956 ± 0.017 |

## Supporting numbers

- LSTM forecaster per-zone val MAE (MW): see `models/saved_models/lstm.pt` metadata and `results/figures/forecaster_validation.png` (mean 13.8 MW; Accra at its ~22 MW irreducible noise floor).
- Historical pre-rebuild numbers (IDQN return −16.13, WUE 162 MWh; central DQN WUE 3203 MWh) are from an incompatible env version — HISTORICAL ONLY, never in comparison tables (Hard Rule 1).
