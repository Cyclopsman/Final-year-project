# MARL Load-Shedding for a 5-Zone Ghana Grid

DCIT 400 final-year project — Noel Osei-Tutu (11285438), University of Ghana.
Supervisor: Mr. Stephen Adingo.

Multi-agent reinforcement learning (IDQN → VDN → QMIX) for distributing
electricity load-shedding ("dumsor") across a 5-zone simulation of Ghana's
grid. **Core claim:** no fixed-rule heuristic is simultaneously efficient
(low criticality-weighted unserved energy), fair (even outage distribution)
and feasible (no uncontrolled residual deficit); learned cooperative
policies can *occupy* that empty region of the Pareto frontier.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch gymnasium numpy pandas matplotlib pyyaml
```

(CPU-only torch is sufficient: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.)

## Run the tests

```bash
python -m tests.test_environment      # expect 8/8 PASS
```

## Train

```bash
python -m src.training.train_idqn     # 130k steps  (~10 min CPU)
python -m src.training.train_vdn      # 150k steps  (~ 8 min CPU)
python -m src.training.train_qmix     # 200k steps  (~10 min CPU)
python -m src.forecasting.lstm_forecaster            # LSTM demand forecaster
python -m src.training.run_beta_ablation --agent vdn # β ∈ {0, 0.5, 2.0} @ 60k
```

Checkpoints land in `models/saved_models/`, training logs in
`results/train_log_*.csv`. All experiments run at `supply.mean_mw = 2000`
(never compare policies across supply levels).

## Evaluate

```bash
python -m src.evaluation.evaluate     # → results/all_policies_metrics.csv + eval_raw.json
```

9 policies (6 baselines + IDQN/VDN/QMIX) × 5 episodes × 3 seeds, greedy
agents, results stamped with git commit + config hash.

## Regenerate figures & report numbers

```bash
python -m src.evaluation.make_figures        # → results/figures/*.png (300 dpi)
python -m src.evaluation.make_report_numbers # → results/report_numbers.md
```

## Repository layout

```
config/grid_config.yaml        # SINGLE source of truth for all parameters
src/environment/               # grid_env.py, zone.py, demand_generator.py, supply_model.py
src/agents/                    # baselines, dqn_agent (shared), IDQN, VDN, QMIX
src/forecasting/               # lstm_forecaster.py
src/training/                  # train_* scripts + run_beta_ablation.py
src/evaluation/                # evaluate.py, make_figures.py, make_report_numbers.py
tests/test_environment.py      # 8 tests (t1–t8)
results/                       # metrics CSV, raw JSON, figures/, report_numbers.md
models/saved_models/           # idqn.pt, vdn.pt, qmix.pt, lstm.pt, ablation ckpts
docs/design_decisions.md       # dated rationale for every experimental choice
```

## Invariants (do not break)

- `info` dict keys `weighted_unserved_energy` and `zone_shed_fraction` are a
  hard interface contract (test t8) — training reads them by name.
- Imbalance penalty is linear **plus** quadratic (see design decision (a)).
- Never mix results across environment versions; regenerate everything after
  any change to `config/grid_config.yaml` env/reward sections.
