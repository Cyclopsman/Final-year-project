# CLAUDE.md — DCIT 400 Final Year Project

## What this project is
Multi-agent reinforcement learning (MARL) system for optimizing electricity load-shedding
distribution across a 5-zone simulation of Ghana's power grid ("dumsor" problem).
Student: Noel Osei-Tutu (11285438), University of Ghana, supervisor Mr. Stephen Adingo.

**The core claim (do not drift from this):** no fixed-rule heuristic simultaneously achieves
low criticality-weighted unserved energy (efficiency) AND a fair outage distribution (equity).
Learned cooperative policies (IDQN → VDN → QMIX) can occupy that empty region of the Pareto
frontier. The contribution is *occupying the region*, NOT "beating every baseline on every metric".

## Grade reality (drives all prioritization)
- Report = 60% of grade. It is the single largest gap. Everything serves the report.
- Working order of priority: (1) verify/finish code, (2) train VDN + QMIX + retrain IDQN,
  (3) generate figures from real results, (4) write report, (5) supervisor engagement.
- Scope was deliberately TRIMMED in the approved proposal: NO PPO, ONE ablation only
  (fairness weight β), NO robustness analysis, NO live dashboard (static matplotlib only),
  SIX baselines. Do not re-add cut scope. Enough > everything.

## Hard rules
1. **Never mix results across environment versions.** Any change to reward calibration or
   env dynamics invalidates all prior numbers. Pre-rebuild results (IDQN return −16.13,
   WUE 162 MWh; central DQN WUE 3203 MWh) are HISTORICAL ONLY — never put them in the
   report's comparison tables.
2. **All policies compared at the same supply level.** Main results at supply_mean = 2000 MW.
   Document the calibration choice in docs/design_decisions.md.
3. **Interface contract (test t8):** env `info` dict MUST expose keys
   `weighted_unserved_energy` (float, MWh this step) and `zone_shed_fraction`
   (array of 5 floats). Training scripts read these with .get() — wrong key names
   silently read zeros and corrupt training. Never rename.
4. **Reward imbalance term is linear-PLUS-quadratic**, not pure quadratic. Pure quadratic
   made NoShedding near-optimal (documented bug). Keep documented in design_decisions.md.
5. **Every design decision** with experimental consequences gets a dated entry in
   docs/design_decisions.md — these are viva ammunition.
6. **Honest framing always.** Report VDN-vs-QMIX as an empirical finding either way
   (Papoudakis et al. 2021 says ranking is task-dependent). If a result is mixed, write it
   mixed. "Pareto zone occupation" is the defensible claim.
7. Fixed seeds everywhere; results reported as mean ± std over ≥5 eval episodes × ≥3 seeds.
8. Never commit credentials. Never paste tokens into chat or files.

## Repo layout (authoritative)
```
load-shedding-rl/
├── config/grid_config.yaml        # SINGLE source of truth for all parameters
├── src/environment/               # grid_env.py, zone.py, demand_generator.py, supply_model.py
├── src/agents/                    # baseline_agents.py, dqn_agent.py, independent_dqn.py,
│                                  # vdn_agent.py, qmix_agent.py
├── src/forecasting/lstm_forecaster.py
├── src/training/                  # train_idqn.py, train_vdn.py, train_qmix.py
├── src/evaluation/                # evaluate.py, visualize_baselines.py, make_figures.py
├── tests/test_environment.py      # 8 tests; run: python -m tests.test_environment
├── results/figures/               # all output figures (PNG, 300 dpi)
├── models/saved_models/           # checkpoints: idqn.pt, vdn.pt, qmix.pt
└── docs/design_decisions.md
```

## Environment spec (must match exactly — see SPEC.md for full math)
- 5 zones + criticality: Greater Accra 1.00, Ashanti 0.85, Western 0.80, Volta 0.60, Northern 0.55
- 1-hour steps, 168-hour episodes. Actions: per-zone shed fraction ∈ {0, .25, .5, .75, 1.0}
- Demand: two-Gaussian diurnal (peaks 07:00, 19:00), weekend dip, harmattan (Dec–Feb) effect,
  industrial_share flattens curve. Supply: Gaussian baseline (mean 2000 for experiments),
  seasonal multipliers, stochastic shocks w/ exponential-decay recovery.
- Obs per zone = 14 dims: own demand (norm), criticality, own outage rate, sin/cos hour,
  sin/cos day-of-week, system supply, system demand, system deficit (norm), other 4 zones'
  outage rates. Joint obs = 70 flat.
- Reward (shared, cooperative): α=1.0 weighted-unserved-energy, β=0.5 fairness
  (std of recent outage rates), γ=2.0 imbalance (linear+quadratic), δ=0.1 instability
  (mean |Δshed|). Normalize energy terms by peak-demand estimate.

## Agent specs
- IDQN: 5 separate Q-nets (14→128→128→5), no sharing. Double DQN, Huber, grad-clip 10, Polyak τ=0.005.
- VDN: per-agent trunks (orthogonal init) + SumMixer, JointReplayBuffer. Same DQN tricks.
- QMIX: monotonic hypernetwork mixer (abs-constrained weights), global state = concat obs (70),
  embed_dim 32, hypernet_hidden 64, StatefulReplayBuffer. Same DQN tricks.
- LSTM forecaster: 2-layer, per-zone MAE target ≈ 15 MW; supporting component only.

## Baselines (6, in BASELINE_REGISTRY with make_baseline() factory)
NoShedding, Random, RoundRobin, Priority, Proportional, FairRotation.

## Definition of DONE
- [ ] `python -m tests.test_environment` → 8/8 pass
- [ ] IDQN, VDN (150k steps), QMIX (200k steps) trained at supply=2000, checkpoints saved
- [ ] evaluate.py → results/all_policies_metrics.csv (all 9 policies, mean±std, ≥3 seeds)
- [ ] Figures: pareto_scatter.png, policy_comparison.png, zone_outage.png,
      training_curves.png, ablation_beta.png, forecaster_validation.png
- [ ] β-ablation: β ∈ {0, 0.5, 2.0} for best agent (reduced steps OK, document)
- [ ] Repo pushed to github.com/Cyclopsman/Final-year-project (own credentials, no tokens in chat)
- [ ] Report data table filled from all_policies_metrics.csv ONLY (env version stamped)

## Style
Production-quality code, type hints, docstrings that explain WHY (viva-defensible),
config-driven (no magic numbers outside YAML), deterministic seeding helpers.
