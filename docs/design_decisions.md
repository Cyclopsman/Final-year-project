# Design Decisions Log

Every decision with experimental consequences, dated, with rationale.
These entries are viva ammunition — each answers a "why did you…?" question.

---

## 2026-07-17 — (a) Imbalance penalty is linear + quadratic, not pure quadratic

**Decision:** `r_imb = −γ · (imb + imb²)` where `imb = residual_deficit / max(supply, 1)`.

**Why:** An earlier iteration used a pure quadratic penalty. Because `imb` is a
fraction (typically 0.05–0.15 off-peak), squaring it made small persistent
deficits almost free: `0.1² = 0.01`, so NoShedding — which leaves a modest
uncontrolled deficit for most hours — scored near-optimally, which is both
degenerate as a learning signal and indefensible physically (any residual
deficit means uncontrolled blackouts / frequency instability). Adding the
linear term restores first-order cost for small imbalances while the quadratic
term still punishes large ones disproportionately. This is Hard Rule 4:
changing this term invalidates all trained results.

## 2026-07-17 — (b) Supply mean calibrated to 2000 MW for all experiments

**Decision:** `supply.mean_mw = 2000`, all training and evaluation at this value
(Hard Rule 2: never compare policies across supply levels).

**Why:** Total base demand is 2,280 MW and the flattened evening peak reaches
≈ 3,100 MW. A 2000 MW mean supply produces a *persistent moderate scarcity*
regime: small deficits (≈ 100–300 MW) most hours and large deficits
(≈ 1,000 MW) at evening peaks. This is the regime where load-shedding policy
actually matters: with abundant supply every policy is trivially optimal
(never shed); with extreme scarcity every policy sheds everything and the
choices are again uninteresting. 2000 MW keeps both the "which zone" and
"how much" decisions non-trivial every day of the episode, and roughly
mirrors the ~20–30% capacity shortfalls of the 2012–2016 dumsor crisis.

## 2026-07-17 — (c) Five-zone abstraction of Ghana's grid

**Decision:** Model 5 aggregate zones (Greater Accra, Ashanti, Western, Volta,
Northern) instead of feeders/substations or all 16 regions.

**Why:** (1) Load-shedding in Ghana was historically scheduled at
regional/district granularity, so zones are the natural decision unit;
(2) five agents keeps the joint action space 5⁵ = 3,125 — large enough that
coordination is non-trivial, small enough that value-decomposition methods
(VDN/QMIX) are well inside their known operating range; (3) criticality
weights (1.00 → 0.55) encode the real asymmetry (Accra: government,
hospitals, ports; Northern: lower load density) that creates the
efficiency–fairness tension the project studies. The zone parameters are
stylized estimates, not utility data — the claim is about *policy structure
under scarcity*, not about forecasting Ghana's actual grid.

## 2026-07-17 — (d) Trimmed scope (per approved proposal)

**Decision:** No PPO, ONE ablation (fairness weight β only), no robustness
analysis, no live dashboard (static matplotlib only), six baselines.

**Why:** Report = 60% of the grade; each additional experiment axis costs
training + evaluation + writing time and dilutes the core claim. The β
ablation is the one that directly probes the research question (does the
reward's fairness weight actually move the policy along the
efficiency–equity frontier?). PPO would add an on-policy/off-policy
comparison orthogonal to the claim. Scope was fixed in the approved
proposal; re-adding it now would be scope creep, not rigor.

## 2026-07-17 — (e) Episode-level season latent (sampled month)

**Decision:** Each episode samples a calendar month uniformly at reset (from
the episode seed); harmattan (demand) and hydro season (supply multiplier)
both derive from that month.

**Why:** SPEC defines harmattan and wet/dry effects but a 168-hour episode
can't span seasons. Sampling the month per episode (a) keeps the two
seasonal effects mutually consistent (harmattan months are dry-season
months), (b) exposes agents to the full seasonal distribution during
training rather than a single fixed regime, and (c) stays deterministic per
seed (t6). Uniform sampling slightly over-represents harmattan relative to
the calendar (3/12 exactly), which is acceptable — evaluation uses the same
distribution for every policy.

## 2026-07-17 — (f) Observation normalizers

**Decision:** Per-zone demand is normalized by `base_demand_z × (1 + evening_peak_amp)`
(the unflattened evening peak); system quantities (supply, total demand,
anticipated deficit) by `peak_demand_estimate_mw = 2600`.

**Why:** Keeps every observation feature roughly in [0, 1] without
per-feature statistics that would leak information across environment
versions. The anticipated deficit feature is the *pre-shedding* deficit
`max(0, demand_total − supply)` for the hour being acted on — that is the
quantity the dispatcher actually knows when deciding shedding, whereas the
post-shedding residual deficit is only known after actions are chosen.

## 2026-07-17 — (g) Baseline sanity finding: raw WUE alone cannot separate
Priority from RoundRobin — residual deficit is the hidden third axis

**Finding (10 seeds, supply = 2000):**

| policy | WUE (MWh/ep) | mean residual deficit (MW) | fairness std (outage h) | return |
|---|---|---|---|---|
| Priority | 67,915 ± 16,941 | 0.0 | ~64 | −59.1 |
| RoundRobin | 70,582 ± 3,018 | 190.6 | ~0.5 | −78.4 |
| Proportional | 129,946 ± 19,439 | 0.0 | ~0 | −51.1 |

Priority < RoundRobin on WUE and RoundRobin ≫ Priority on fairness (the
expected Pareto gap), but the WUE margin is small because WUE only counts
*controlled* shedding: RoundRobin can shed at most one zone per hour, so at
evening peaks (deficit ≈ 1,000 MW > any single zone) it leaves a large
uncontrolled residual deficit that never enters WUE — it is punished through
the imbalance penalty and shows up in return (−78 vs −59) and in the
mean-residual-deficit metric instead. Consequence for the report: the Pareto
story must be told on (WUE, fairness) *conditioned on feasibility*
(residual deficit ≈ 0); the evaluation table reports residual deficit for
exactly this reason. NoShedding is the extreme case: WUE = 0, fairness
std = 0, yet the worst return of all (−161) — pure-WUE axes without the
feasibility condition would rank it "best".

## 2026-07-17 — (h) LSTM forecaster hits the noise floor

Per-zone validation MAE (MW): Accra 21.8, Ashanti 16.8, Western 12.6,
Volta 8.5, Northern 9.2 (mean 13.8, target ≈ ≤ 20). Accra's MAE equals its
irreducible noise floor: demand noise is 3% multiplicative, and mean |N(0,σ)|
≈ 0.8σ ⇒ ≈ 0.8 × 0.03 × ~900 MW ≈ 22 MW. The model has therefore extracted
essentially all learnable structure; remaining error is aleatoric. The
forecaster is a supporting component (demand is learnable structure);
forecast-conditioned shedding is future work, per trimmed scope.

## 2026-07-17 — (i) β-ablation protocol

β ∈ {0, 0.5, 2.0} on the better of VDN/QMIX. TASKS 2.4 specifies extra runs
at 60k steps for β ∈ {0, 2.0}; to keep the three points comparable, a third
60k run at β = 0.5 is trained for the ablation figure rather than reusing
the full-length main checkpoint (different training budgets would confound
the comparison). The main-results table still uses the full-length β = 0.5
checkpoints. Ablation artifacts are tagged `_beta{value}` and never mixed
into main results (Hard Rule 1).
