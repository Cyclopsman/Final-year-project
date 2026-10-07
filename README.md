# MARL Load-Shedding for a 5-Zone Ghana Grid

DCIT 400 final-year project, University of Ghana.

Multi-agent reinforcement learning (IDQN → VDN → QMIX) for distributing
electricity load-shedding ("dumsor") across a 5-zone simulation of Ghana's
grid. **Finding:** the learned policies occupy a tradeoff region absent from
these six tested baselines when controlled unserved energy, fairness and
residual deficit are considered together. This does not prove a universal
advantage over fixed rules or establish the true Pareto frontier. All learned
policies have nonzero residual deficit; results use one training seed.


## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch gymnasium numpy pandas matplotlib pyyaml scipy
```

(CPU-only torch is sufficient: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.)

## Run the tests

```bash
python -m tests.test_environment      # expect 8/8 PASS
python -m unittest tests.test_location_output  # location export checks
```

## Train

```bash
python -m src.training.train_idqn     # 130k steps  (~10 min CPU)
python -m src.training.train_vdn      # 150k steps  (~ 8 min CPU)
python -m src.training.train_qmix     # 200k steps  (~10 min CPU; runs a
                                      # monotonicity + IGM mixer check first)
python -m src.forecasting.lstm_forecaster             # LSTM demand forecaster
python -m src.training.run_beta_ablation --agent qmix # β ∈ {0, 0.25, 0.5, 1.0},
                                      # full budget per point (β=0.5 reuses the
                                      # main checkpoint — same seed/config)
```

Checkpoints land in `models/saved_models/`, training logs (per-episode
return, WUE, σ_fair, worst-zone outage hours) in `results/logs/*_train.csv`.
All experiments run at `supply.mean_mw = 2000` (never compare policies
across supply levels). Training seeds all RNGs *before* network init, so
identical commands reproduce identical checkpoints.

## Evaluate

```bash
python -m src.evaluation.evaluate     # → results/all_policies_metrics.csv,
                                      #   eval_raw.json, {idqn,vdn,qmix}_eval.json
```

9 policies (6 baselines + IDQN/VDN/QMIX) × 5 episodes × 4 seeds (20 episodes
each, one protocol for all), greedy agents, results stamped with git commit
+ config hash.

## Regenerate figures & report numbers

```bash
python -m src.evaluation.make_figures        # → results/figures/*.png (300 dpi)
python -m src.evaluation.make_report_numbers # → results/report_numbers.md
```

## Episode playback demo (for presentations)

```bash
python -m src.evaluation.demo --policy qmix        # PNG + hour-by-hour GIF
python -m src.evaluation.demo --policy priority --no-gif
```

Renders one 168-hour week (demand/supply/served, per-zone shed timeline,
cumulative outage ledger) to `results/figures/demo_<policy>.{png,gif}`.
Same `--seed` ⇒ same week for every policy, so runs are directly
comparable side by side. Static matplotlib only; single-episode traces are
illustrative, never report metrics.

## Where and when does shedding happen?

```bash
python -m src.evaluation.demo --policy qmix --seed 0 --map-hour 19 --no-gif
```

This also creates `results/locations/` outputs:

- `locations_qmix_seed0_schedule.csv`: all 168 hours for all five zones,
  including coordinates, episode day/hour, shed fraction and status.
- `locations_qmix_seed0_hour19.geojson`: selectable zone markers for hour 19.
  Open the file on GitHub or in a geographic information system to inspect them.
- `locations_qmix_seed0_hour19.png`: static coordinate plot for the defence.

[Open the example location map](results/locations/locations_qmix_seed0_hour19.geojson).
Change `--map-hour` from 0 to 167 to inspect another hour. Episode day 1 begins
at 00:00; these are simulated times with no real calendar date. A 25% shed
fraction means 25% of that zone's modelled load, not that every house is off
for 15 minutes. Zero means no *controlled* shedding, not guaranteed supply:
residual system imbalance can still exist and is not allocated to locations.

`config/zone_locations.yaml` holds approximate city reference points from
[GeoNames](https://www.geonames.org/advanced-search.html?continentCode=&country=GH&fclass=P&q=).
They identify aggregate zones, not region boundaries, substations or affected
premises. This is a simulation demonstration, not an ECG outage forecast.
Locations are separate display metadata: the trained environment and its
configuration hash are unchanged. Feeder level results would require actual
feeder boundaries, connectivity and operational data, plus a finer model.

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
