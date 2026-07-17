"""Stochastic supply model (SPEC.md §2).

supply_t = max(0, N(mean, std) * seasonal_mult * (1 - shock_t))

Shocks model sudden generation losses (unit trips, gas curtailment): with
probability p per hour a shock of uniform depth begins; an active shock
decays exponentially (recovery_rate per hour), matching how lost capacity
is typically restored gradually. The dry-season multiplier (<1) encodes
Akosombo's hydro sensitivity — the historical driver of dumsor.
"""
from __future__ import annotations

import numpy as np

DRY_MONTHS = (11, 12, 1, 2, 3)  # low-hydro months → "dry" multiplier


class SupplyModel:
    def __init__(self, cfg: dict):
        s = cfg["supply"]
        self.mean_mw = s["mean_mw"]
        self.std_mw = s["std_mw"]
        self.seasonal_multipliers = s["seasonal_multipliers"]
        self.shock_prob = s["shock_prob_per_hour"]
        self.shock_depth_range = s["shock_depth_frac"]
        self.recovery_rate = s["shock_recovery_rate"]
        self.shock = 0.0
        self.seasonal_mult = 1.0

    def reset(self, month: int) -> None:
        self.shock = 0.0
        season = "dry" if month in DRY_MONTHS else "wet"
        self.seasonal_mult = self.seasonal_multipliers[season]

    def sample(self, rng: np.random.Generator) -> float:
        """Advance shock state one hour and draw supply (MW)."""
        # Existing shock recovers first, then a new shock may begin/deepen.
        self.shock *= 1.0 - self.recovery_rate
        if rng.random() < self.shock_prob:
            depth = rng.uniform(*self.shock_depth_range)
            self.shock = max(self.shock, depth)
        base = rng.normal(self.mean_mw, self.std_mw)
        return float(max(0.0, base * self.seasonal_mult * (1.0 - self.shock)))
