"""Per-zone hourly demand model (SPEC.md §2).

Demand is a deterministic diurnal/weekly/seasonal shape times multiplicative
Gaussian noise. The two-Gaussian diurnal curve (peaks 07:00 and 19:00)
reflects Ghana's residential morning/evening usage; a zone's industrial
share *flattens* its curve because industrial load is roughly constant
across the day — this is why Western (40% industrial) has a flatter profile
than Volta (15%).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

WEEKEND_DAYS = (5, 6)  # day-of-week indices treated as Sat/Sun
HARMATTAN_MONTHS = (12, 1, 2)


class DemandGenerator:
    def __init__(self, cfg: dict):
        d = cfg["demand"]
        self.morning_peak_hour = d["morning_peak_hour"]
        self.evening_peak_hour = d["evening_peak_hour"]
        self.morning_peak_width = d["morning_peak_width"]
        self.evening_peak_width = d["evening_peak_width"]
        self.morning_peak_amp = d["morning_peak_amp"]
        self.evening_peak_amp = d["evening_peak_amp"]
        self.weekend_dip = d["weekend_dip"]
        self.harmattan_multiplier = d["harmattan_multiplier"]
        self.noise_std_frac = d["noise_std_frac"]
        self.base_demand = np.array([z["base_demand_mw"] for z in cfg["zones"]])
        self.industrial_share = np.array([z["industrial_share"] for z in cfg["zones"]])

    def diurnal_factor(self, hour_of_week: int) -> float:
        """System-level diurnal multiplier (before industrial flattening)."""
        h = hour_of_week % 24
        m = self.morning_peak_amp * np.exp(
            -((h - self.morning_peak_hour) ** 2) / (2 * self.morning_peak_width**2)
        )
        e = self.evening_peak_amp * np.exp(
            -((h - self.evening_peak_hour) ** 2) / (2 * self.evening_peak_width**2)
        )
        return float(1.0 + m + e)

    def mean_demand(self, hour_of_week: int, month: int) -> np.ndarray:
        """Noise-free expected demand per zone (MW). Used by tests & forecaster targets."""
        diurnal = self.diurnal_factor(hour_of_week)
        # Industrial load is flat: shrink the diurnal excursion by industrial share.
        diurnal_z = 1.0 + (diurnal - 1.0) * (1.0 - self.industrial_share)
        day = (hour_of_week // 24) % 7
        weekend = self.weekend_dip if day in WEEKEND_DAYS else 1.0
        harmattan = self.harmattan_multiplier if month in HARMATTAN_MONTHS else 1.0
        return self.base_demand * diurnal_z * weekend * harmattan

    def sample(self, hour_of_week: int, month: int, rng: np.random.Generator) -> np.ndarray:
        """Noisy demand per zone (MW), clipped to be non-negative."""
        noise = 1.0 + rng.normal(0.0, self.noise_std_frac, size=len(self.base_demand))
        return np.maximum(0.0, self.mean_demand(hour_of_week, month) * noise)
