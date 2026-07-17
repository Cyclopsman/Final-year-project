"""Six rule-based baseline policies (SPEC.md §3).

The baselines span the efficiency–fairness trade-off deliberately:
Priority is the efficiency extreme (protect critical zones, always dump on
the same low-criticality zones — maximally unfair), RoundRobin/FairRotation
are the equity extreme (spread outages evenly regardless of criticality),
Proportional sits between. The learned agents' claim is occupying the
Pareto region none of these reach.

Baselines act only on the (5, 14) observation array plus config constants —
the same information interface the learned agents get, so comparisons are fair.
"""
from __future__ import annotations

import numpy as np

# Observation column indices (must match GridEnv._build_obs order).
COL_DEMAND = 0
COL_SUPPLY = 7
COL_DEMAND_TOTAL = 8
COL_DEFICIT = 9


class BaselinePolicy:
    """Base class: decodes MW quantities from the normalized observation."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        env_cfg = cfg["env"]
        self.n_zones = env_cfg["n_zones"]
        self.shed_levels = np.array(env_cfg["shed_levels"])
        self.n_levels = len(self.shed_levels)
        self.full_level = self.n_levels - 1
        self.peak_est = float(env_cfg["peak_demand_estimate_mw"])
        self.criticality = np.array([z["criticality"] for z in cfg["zones"]])
        self.zone_peak = np.array(
            [
                z["base_demand_mw"] * (1.0 + cfg["demand"]["evening_peak_amp"])
                for z in cfg["zones"]
            ]
        )
        self.rng = np.random.default_rng(0)

    def reset(self, seed: int | None = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def decode(self, obs: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        demand = obs[:, COL_DEMAND] * self.zone_peak
        supply = float(obs[0, COL_SUPPLY]) * self.peak_est
        demand_total = float(obs[0, COL_DEMAND_TOTAL]) * self.peak_est
        deficit = float(obs[0, COL_DEFICIT]) * self.peak_est
        return demand, supply, demand_total, deficit

    def act(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class NoShedding(BaselinePolicy):
    """Always serve everything — the do-nothing control."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(self.n_zones, dtype=np.int64)


class RandomPolicy(BaselinePolicy):
    """Uniform random shed level per zone — the noise floor."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.rng.integers(0, self.n_levels, size=self.n_zones)


class RoundRobin(BaselinePolicy):
    """When there is a deficit, black out one zone fully, rotating each
    deficit-hour. Ghana's historical dumsor timetable in miniature."""

    def reset(self, seed: int | None = 0) -> None:
        super().reset(seed)
        self.pointer = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        _, _, _, deficit = self.decode(obs)
        actions = np.zeros(self.n_zones, dtype=np.int64)
        if deficit > 0:
            actions[self.pointer] = self.full_level
            self.pointer = (self.pointer + 1) % self.n_zones
        return actions


class Priority(BaselinePolicy):
    """Shed lowest-criticality zones first, increasing levels until the
    expected served load fits under supply. Efficiency extreme."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        demand, supply, demand_total, _ = self.decode(obs)
        actions = np.zeros(self.n_zones, dtype=np.int64)
        remaining = demand_total - supply
        if remaining <= 0:
            return actions
        for z in np.argsort(self.criticality):  # ascending criticality
            if remaining <= 0:
                break
            # Smallest level whose shed energy covers what's left of the deficit.
            needed = remaining / max(demand[z], 1e-9)
            level = int(np.searchsorted(self.shed_levels, min(needed, 1.0)))
            actions[z] = min(level, self.full_level)
            remaining -= demand[z] * self.shed_levels[actions[z]]
        return actions


class Proportional(BaselinePolicy):
    """All zones shed at the smallest common level covering the deficit
    fraction — equal relative pain, ignores criticality."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        _, supply, demand_total, deficit = self.decode(obs)
        if deficit <= 0:
            return np.zeros(self.n_zones, dtype=np.int64)
        frac_needed = deficit / max(demand_total, 1e-9)
        level = int(np.searchsorted(self.shed_levels, min(frac_needed, 1.0)))
        return np.full(self.n_zones, min(level, self.full_level), dtype=np.int64)


class FairRotation(BaselinePolicy):
    """RoundRobin variant that always sheds the least-shed zone first
    (cumulative outage hours ascending) — equity extreme."""

    def reset(self, seed: int | None = 0) -> None:
        super().reset(seed)
        self.cumulative = np.zeros(self.n_zones)

    def act(self, obs: np.ndarray) -> np.ndarray:
        _, _, _, deficit = self.decode(obs)
        actions = np.zeros(self.n_zones, dtype=np.int64)
        if deficit > 0:
            target = int(np.argmin(self.cumulative))
            actions[target] = self.full_level
            self.cumulative[target] += self.shed_levels[self.full_level]
        return actions


BASELINE_REGISTRY = {
    "no_shedding": NoShedding,
    "random": RandomPolicy,
    "round_robin": RoundRobin,
    "priority": Priority,
    "proportional": Proportional,
    "fair_rotation": FairRotation,
}


def make_baseline(name: str, cfg: dict) -> BaselinePolicy:
    if name not in BASELINE_REGISTRY:
        raise KeyError(f"unknown baseline '{name}'; options: {list(BASELINE_REGISTRY)}")
    policy = BASELINE_REGISTRY[name](cfg)
    policy.reset()
    return policy
