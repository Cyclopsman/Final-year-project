# SPEC.md — Full Technical Specification
Authoritative reference. If code and SPEC disagree, fix code (or update SPEC + design_decisions.md with rationale).

## 1. Configuration (config/grid_config.yaml)
All parameters live here. Required structure:

```yaml
env:
  n_zones: 5
  episode_hours: 168
  timestep_hours: 1
  shed_levels: [0.0, 0.25, 0.5, 0.75, 1.0]
  peak_demand_estimate_mw: 2600      # normalization constant for reward energy terms
zones:
  - {name: "Greater Accra", criticality: 1.00, base_demand_mw: 750, industrial_share: 0.35}
  - {name: "Ashanti",       criticality: 0.85, base_demand_mw: 550, industrial_share: 0.30}
  - {name: "Western",       criticality: 0.80, base_demand_mw: 420, industrial_share: 0.40}
  - {name: "Volta",         criticality: 0.60, base_demand_mw: 260, industrial_share: 0.15}
  - {name: "Northern",      criticality: 0.55, base_demand_mw: 300, industrial_share: 0.10}
demand:
  morning_peak_hour: 7
  evening_peak_hour: 19
  morning_peak_width: 2.5
  evening_peak_width: 3.0
  morning_peak_amp: 0.35             # fraction above base
  evening_peak_amp: 0.55
  weekend_dip: 0.88                  # multiplier Sat/Sun
  harmattan_multiplier: 1.10         # Dec-Feb evening bump
  noise_std_frac: 0.03
supply:
  mean_mw: 2000                      # EXPERIMENT SETTING — all agents trained/evaled here
  std_mw: 120
  seasonal_multipliers: {dry: 0.95, wet: 1.05}   # hydro-driven
  shock_prob_per_hour: 0.01
  shock_depth_frac: [0.10, 0.30]     # uniform range
  shock_recovery_rate: 0.15          # exponential decay per hour
reward:
  alpha_wue: 1.0
  beta_fairness: 0.5
  gamma_imbalance: 2.0
  delta_instability: 0.1
  fairness_window_hours: 24
training:
  gamma: 0.99
  lr: 5.0e-4
  batch_size: 64
  buffer_size: 100000
  train_every: 4
  polyak_tau: 0.005
  grad_clip: 10.0
  eps_start: 1.0
  eps_end: 0.05
  eps_decay_steps: 60000
  idqn_steps: 130000
  vdn_steps: 150000
  qmix_steps: 200000
  seed: 42
```

## 2. Environment dynamics (src/environment/)

### Demand per zone z at hour h (of week), day d:
```
diurnal(h) = 1 + A_m*exp(-((h%24 - 7)^2)/(2*w_m^2)) + A_e*exp(-((h%24 - 19)^2)/(2*w_e^2))
flatten by industrial share: diurnal_z = 1 + (diurnal(h) - 1) * (1 - industrial_share_z)
demand_z = base_demand_z * diurnal_z * weekend_mult(d) * harmattan_mult(season) * (1 + N(0, noise))
```

### Supply:
```
supply_t = max(0, N(mean, std) * seasonal_mult * (1 - shock_t))
shock: with prob p per hour start shock of depth U(range); shock decays *= (1 - recovery_rate)
```

### Step semantics (order matters):
1. Compute demand_z for all zones, total_demand, supply_t.
2. Agents' actions a_z → shed fraction s_z = shed_levels[a_z].
3. served_z = demand_z * (1 - s_z); unserved_z = demand_z * s_z.
4. residual_deficit = max(0, Σ served_z − supply_t)   # what shedding failed to cover
5. Reward components (all negative penalties):
   - WUE_t = Σ_z criticality_z * unserved_z            [MWh]
   - r_eff  = −α * WUE_t / peak_demand_estimate
   - outage_rate_z updated over rolling fairness_window; r_fair = −β * std({outage_rate_z})
   - imb = residual_deficit / max(supply_t, 1);  r_imb = −γ * (imb + imb²)   # LINEAR+QUADRATIC
   - r_inst = −δ * mean_z |s_z(t) − s_z(t−1)|
   - reward = r_eff + r_fair + r_imb + r_inst  (same scalar to all agents)
6. info dict MUST include:
   `weighted_unserved_energy` = WUE_t, `zone_shed_fraction` = np.array(s),
   plus: `served_total`, `supply`, `demand_total`, `residual_deficit`,
   `zone_outage_rates`, `reward_components` (dict).

### Observation per zone (14 dims, this exact order):
[ demand_z/base_z_peak, criticality_z, outage_rate_z,
  sin(2π h/24), cos(2π h/24), sin(2π d/7), cos(2π d/7),
  supply/peak_est, demand_total/peak_est, deficit/peak_est,
  outage_rate of other 4 zones (fixed zone order, self excluded) ]

Gymnasium API: reset(seed)→(obs_dict_or_stack, info); step(actions)→(obs, reward, terminated, truncated, info).
Truncated=True at 168 steps. Use a dict {zone_idx: obs} OR (5,14) array — pick one, tests enforce it.

## 3. Baselines (src/agents/baseline_agents.py)
Registry: `BASELINE_REGISTRY = {"no_shedding": ..., "random": ..., "round_robin": ...,
"priority": ..., "proportional": ..., "fair_rotation": ...}` + `make_baseline(name, cfg)`.
- NoShedding: always action 0.
- Random: uniform random shed level per zone.
- RoundRobin: when deficit>0, shed one zone fully in rotation (advance each hour of deficit).
- Priority: sort zones ascending criticality; shed lowest-criticality zones first,
  increasing levels until expected served ≤ supply.
- Proportional: shed all zones at the smallest common level covering the deficit fraction.
- FairRotation: like RoundRobin but rotation order = current cumulative outage hours ascending
  (always sheds the least-shed zone first).

## 4. Agents
Shared tricks for all: Double DQN target (online net argmax, target net eval), Huber loss,
grad-norm clip 10, Polyak soft update τ=0.005 every train step, ε-greedy linear decay.

- **IDQN** (independent_dqn.py): per-zone QNet(14→128→128→5), per-zone replay buffer
  (or shared transitions, independent sampling). No mixing.
- **VDN** (vdn_agent.py): classes VDNConfig, AgentQNet (14→128→128→5, orthogonal init),
  SumMixer (Q_tot = Σ Q_z(a_z)), JointReplayBuffer storing (obs[5,14], acts[5], rew, next_obs, done).
  TD target on Q_tot.
- **QMIX** (qmix_agent.py): QMixConfig, agent nets as VDN; QMixer: state=concat obs (70),
  hypernetworks produce W1 (|abs|, 5→32), b1(32), W2 (|abs|, 32→1), b2 (scalar via 2-layer);
  hypernet_hidden=64, embed_dim=32. StatefulReplayBuffer additionally stores state & next_state.
- **LSTM forecaster** (lstm_forecaster.py): input window 24h of per-zone demand (+time enc),
  2-layer LSTM hidden 64, predict next-hour demand per zone. Train on generated episodes,
  report per-zone val MAE (target ≈ 15 MW). Saved to models/saved_models/lstm.pt.

## 5. Tests (tests/test_environment.py) — 8 tests, all must pass
t1 gymnasium contract (reset/step signatures, spaces, truncation at 168)
t2 demand model shape/positivity + diurnal peaks near 07:00/19:00
t3 supply positivity + shock recovery
t4 reward components: all ≤ 0; NoShedding under deficit ⇒ large imbalance penalty;
   full shedding ⇒ zero imbalance but large WUE penalty
t5 fairness term: equal outage rates ⇒ std 0; unequal ⇒ negative term
t6 determinism: same seed ⇒ identical 168-step rollout (obs, rewards)
t7 baseline sanity: every registry baseline runs a full episode without error;
   priority never sheds Accra before Northern at equal levels
t8 INTERFACE CONTRACT: after step(), info contains keys exactly named
   `weighted_unserved_energy` (float) and `zone_shed_fraction` (len-5 array)

## 6. Evaluation protocol (src/evaluation/evaluate.py)
- Policies: 6 baselines + IDQN + VDN + QMIX (load checkpoints).
- 5 eval episodes × 4 seeds each (20 episodes/policy), supply=2000, ε=0 for agents.
  (Raised from 3 to 4 seeds 2026-07-24 — see design_decisions.md; protocol is
  identical for every policy.)
- Metrics per policy (mean ± std): weighted unserved energy (MWh/episode),
  fairness std of zone outage rates, Jain index J = (Σx)²/(n Σx²) over zone outage hours,
  worst-zone outage hours, mean residual deficit (MW), mean return.
- Output: results/all_policies_metrics.csv + results/eval_raw.json.
- Stamp env version: results include git commit hash + config hash.

## 7. Figures (src/evaluation/make_figures.py) — 300 dpi PNG into results/figures/
1. pareto_scatter.png — x=WUE, y=Jain index (or 1/std); baselines grey circles,
   agents coloured stars; annotate names; shade the "target corner".
2. policy_comparison.png — grouped bar chart of key metrics (normalized).
3. zone_outage.png — per-zone cumulative outage hours per policy (grouped bars).
4. training_curves.png — smoothed episode return vs steps for IDQN/VDN/QMIX.
5. ablation_beta.png — fairness metric & WUE vs β ∈ {0, 0.25, 0.5, 1.0} (best agent,
   full main budget per point; β grid revised 2026-07-24, see design_decisions.md).
6. forecaster_validation.png — pred vs actual 48h sample + MAE table inset.

## 8. Report data handoff
After eval, produce results/report_numbers.md summarizing every number the report needs
(one table + the headline Pareto statement), so report writing never reads raw JSON.
