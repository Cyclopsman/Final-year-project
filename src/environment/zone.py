"""Zone state for the 5-zone Ghana grid abstraction.

Each zone carries its static attributes (from config) plus the rolling
outage history used by the fairness reward term. The rolling window (rather
than a cumulative average) is what makes fairness *recent-history* fairness:
a zone shed heavily yesterday should still register as "owed", but events a
month old should not dominate the signal.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class Zone:
    name: str
    criticality: float
    base_demand_mw: float
    industrial_share: float
    fairness_window_hours: int = 24
    # Rolling window of applied shed fractions (one entry per hour).
    shed_history: deque = field(default_factory=deque, repr=False)
    # Cumulative shed-hours over the episode (Σ_t s_z(t)); used by
    # fair-rotation baselines and per-episode outage metrics.
    cumulative_outage_hours: float = 0.0

    def reset(self) -> None:
        self.shed_history = deque(maxlen=self.fairness_window_hours)
        self.cumulative_outage_hours = 0.0

    def record_shed(self, shed_fraction: float) -> None:
        self.shed_history.append(shed_fraction)
        self.cumulative_outage_hours += shed_fraction

    @property
    def outage_rate(self) -> float:
        """Mean shed fraction over the rolling fairness window (0 if no history)."""
        if not self.shed_history:
            return 0.0
        return float(sum(self.shed_history) / len(self.shed_history))
