# TASKS.md — Ordered build plan to DONE
Work top-to-bottom. Each task has acceptance criteria (AC). Tick as you go.
If the local repo/bundle from earlier sessions exists, VERIFY against SPEC.md instead of rewriting.

## Phase 0 — Orientation (15 min)
- [ ] 0.1 Check for existing code: `git log --oneline` (bundle fc60fbb?) and compare tree to CLAUDE.md layout.
      If present → run tests, jump to Phase 2. If absent → Phase 1 full build.
- [ ] 0.2 `git init` (if needed), create venv, `pip install torch gymnasium numpy pandas matplotlib pyyaml`.
      AC: `python -c "import torch, gymnasium"` succeeds.

## Phase 1 — Core build (only what's missing)
- [ ] 1.1 config/grid_config.yaml exactly per SPEC §1. AC: yaml loads; keys match.
- [ ] 1.2 src/environment/ (demand_generator.py, supply_model.py, zone.py, grid_env.py) per SPEC §2.
      AC: manual 168-step rollout with random actions runs; info has t8 keys.
- [ ] 1.3 tests/test_environment.py — all 8 tests per SPEC §5.
      AC: `python -m tests.test_environment` → 8/8 PASS. COMMIT.
- [ ] 1.4 src/agents/baseline_agents.py (6 baselines + registry). AC: t7 passes; quick eval script
      shows Priority < RoundRobin on WUE, RoundRobin > Priority on fairness (sanity of Pareto gap). COMMIT.
- [ ] 1.5 src/agents/independent_dqn.py, vdn_agent.py, qmix_agent.py per SPEC §4.
      AC: 2k-step smoke train each: loss finite & falling, gradients flow (check a mixer weight moves for VDN/QMIX). COMMIT.
- [ ] 1.6 src/forecasting/lstm_forecaster.py + short train. AC: val MAE ≈ ≤20 MW/zone; lstm.pt saved. COMMIT.
- [ ] 1.7 docs/design_decisions.md — port/append entries: (a) linear+quadratic imbalance fix and why,
      (b) supply=2000 calibration rationale, (c) 5-zone abstraction, (d) trimmed scope (no PPO, one ablation).

## Phase 2 — Training (the long pole; run overnight if needed)
- [ ] 2.1 train_idqn.py 130k steps, seed 42, supply=2000 → models/saved_models/idqn.pt + training log CSV.
- [ ] 2.2 train_vdn.py 150k steps → vdn.pt + log.
- [ ] 2.3 train_qmix.py 200k steps → qmix.pt + log.
      AC each: eval return over last 10 eval episodes clearly better than Random; log saved;
      **verify t8 keys are actually nonzero in logged metrics** (silent-zero bug guard).
- [ ] 2.4 β-ablation on the best of VDN/QMIX: β ∈ {0, 2.0} at reduced steps (60k), same seed.
      AC: two extra checkpoints + logs. COMMIT (checkpoints via git-lfs or release asset if large).

## Phase 3 — Evaluation & figures
- [ ] 3.1 evaluate.py per SPEC §6 → all_policies_metrics.csv + eval_raw.json (env stamped).
      AC: 9 policies × 6 metrics, mean±std populated.
- [ ] 3.2 make_figures.py per SPEC §7 → 6 PNGs at 300 dpi.
      AC: pareto_scatter clearly shows baseline frontier + agent positions; figures legible in grayscale.
- [ ] 3.3 results/report_numbers.md — the single source for report numbers.
      AC: contains headline sentence answering RQ1 & RQ2 honestly (occupation claim, not domination).

## Phase 4 — Ship
- [ ] 4.1 README.md: setup, run tests, train, evaluate, regenerate figures (one command each).
- [ ] 4.2 Push everything to github.com/Cyclopsman/Final-year-project (own credentials; NEVER paste tokens).
      AC: fresh clone + README steps reproduce tests 8/8.
- [ ] 4.3 Fill report placeholders from report_numbers.md ONLY. Update word count on title page.
- [ ] 4.4 Email supervisor (saadingo@ug.edu.gh): 5-line progress summary + repo link + ask for meeting slot.

## Guardrails while working
- Small commits at every AC. Descriptive messages.
- If any test breaks after a change: fix before proceeding (especially t6 determinism, t8 contract).
- If training seems flat: check (a) info-key silent zeros, (b) ε schedule, (c) reward normalization —
  in that order. Do NOT retune reward weights casually; that invalidates results (Hard Rule 1).
- If QMIX ≤ VDN: that is a FINDING (cite Papoudakis et al. 2021), write it up, don't chase it.
