"""GridEnv — 5-zone Ghana load-shedding environment (SPEC.md §2).

Cooperative MARL environment with a Gymnasium single-env API: the joint
action is a length-5 vector of shed-level indices, the observation is a
(5, 14) float32 array (row z = zone z's local observation), and one shared
scalar reward is returned (fully cooperative team game).

Interface contract (test t8, Hard Rule 3): after every step() the info dict
exposes `weighted_unserved_energy` (float, MWh this step) and
`zone_shed_fraction` (len-5 array). Training code reads these with .get();
renaming them silently corrupts training. Never rename.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from src.environment.demand_generator import DemandGenerator
from src.environment.supply_model import SupplyModel
from src.environment.zone import Zone

OBS_DIM = 14


def load_config(path: str | Path = "config/grid_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class GridEnv(gym.Env):
    """One step = one hour. Episode = 168 hours (one week), truncated at 168.

    Step semantics (order matters, SPEC §2):
      the observation returned by reset()/step() already contains the demand
      and supply for the hour the agents are about to act on, so step(actions)
      applies shedding to those pre-drawn values, computes the reward, then
      draws the next hour's demand/supply for the next observation.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: dict | str | Path = "config/grid_config.yaml"):
        if not isinstance(cfg, dict):
            cfg = load_config(cfg)
        self.cfg = cfg
        env_cfg = cfg["env"]
        self.n_zones: int = env_cfg["n_zones"]
        self.episode_hours: int = env_cfg["episode_hours"]
        self.shed_levels = np.array(env_cfg["shed_levels"], dtype=np.float64)
        self.peak_est: float = float(env_cfg["peak_demand_estimate_mw"])

        rw = cfg["reward"]
        self.alpha = rw["alpha_wue"]
        self.beta = rw["beta_fairness"]
        self.gamma_imb = rw["gamma_imbalance"]
        self.delta_inst = rw["delta_instability"]
        self.fairness_window = rw["fairness_window_hours"]

        self.zones = [
            Zone(
                name=z["name"],
                criticality=z["criticality"],
                base_demand_mw=z["base_demand_mw"],
                industrial_share=z["industrial_share"],
                fairness_window_hours=self.fairness_window,
            )
            for z in cfg["zones"]
        ]
        self.criticality = np.array([z.criticality for z in self.zones])
        # Per-zone normalizer for the local demand feature: base demand at
        # the (unflattened) evening peak — keeps the feature roughly in [0, 1].
        self.zone_peak = np.array(
            [z.base_demand_mw * (1.0 + cfg["demand"]["evening_peak_amp"]) for z in self.zones]
        )

        self.demand_gen = DemandGenerator(cfg)
        self.supply_model = SupplyModel(cfg)

        self.action_space = spaces.MultiDiscrete([len(self.shed_levels)] * self.n_zones)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_zones, OBS_DIM), dtype=np.float32
        )

        self._rng: np.random.Generator = np.random.default_rng()
        self.t = 0
        self.month = 1
        self._demand = np.zeros(self.n_zones)
        self._supply = 0.0
        self._prev_shed = np.zeros(self.n_zones)

    # ------------------------------------------------------------------ API
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self.t = 0
        # Season is an episode-level latent: sample the calendar month once
        # so harmattan (demand) and hydro season (supply) stay consistent.
        self.month = int(self._rng.integers(1, 13))
        for z in self.zones:
            z.reset()
        self.supply_model.reset(self.month)
        self._prev_shed = np.zeros(self.n_zones)
        self._draw_exogenous()
        return self._build_obs(), self._info_common()

    def step(
        self, actions
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        actions = np.asarray(actions, dtype=np.int64).reshape(self.n_zones)
        shed = self.shed_levels[actions]

        demand = self._demand
        supply = self._supply
        served = demand * (1.0 - shed)
        unserved = demand * shed
        residual_deficit = max(0.0, float(served.sum() - supply))

        # --- reward components (all penalties, all <= 0) ---
        wue = float(np.dot(self.criticality, unserved))  # MWh (1-hour steps)
        r_eff = -self.alpha * wue / self.peak_est

        for z, s in zip(self.zones, shed):
            z.record_shed(float(s))
        outage_rates = np.array([z.outage_rate for z in self.zones])
        r_fair = -self.beta * float(np.std(outage_rates))

        # Linear + quadratic (Hard Rule 4): pure quadratic vanishes for small
        # imbalances, which made NoShedding near-optimal — the linear term
        # keeps small persistent deficits costly.
        imb = residual_deficit / max(supply, 1.0)
        r_imb = -self.gamma_imb * (imb + imb**2)

        r_inst = -self.delta_inst * float(np.mean(np.abs(shed - self._prev_shed)))
        self._prev_shed = shed.copy()

        reward = r_eff + r_fair + r_imb + r_inst

        self.t += 1
        truncated = self.t >= self.episode_hours
        self._draw_exogenous()
        obs = self._build_obs()

        info = self._info_common()
        info.update(
            {
                "weighted_unserved_energy": wue,
                "zone_shed_fraction": shed.copy(),
                "served_total": float(served.sum()),
                "supply": supply,
                "demand_total": float(demand.sum()),
                "residual_deficit": residual_deficit,
                "zone_outage_rates": outage_rates,
                "reward_components": {
                    "r_eff": r_eff,
                    "r_fair": r_fair,
                    "r_imb": r_imb,
                    "r_inst": r_inst,
                },
            }
        )
        return obs, float(reward), False, truncated, info

    # ------------------------------------------------------------- internals
    def _draw_exogenous(self) -> None:
        """Draw demand and supply for the hour the agents will act on next."""
        h = self.t % self.episode_hours
        self._demand = self.demand_gen.sample(h, self.month, self._rng)
        self._supply = self.supply_model.sample(self._rng)

    def _build_obs(self) -> np.ndarray:
        h = self.t % 24
        d = (self.t // 24) % 7
        demand_total = float(self._demand.sum())
        # Anticipated (pre-shedding) deficit for the hour being acted on.
        deficit = max(0.0, demand_total - self._supply)
        outage_rates = np.array([z.outage_rate for z in self.zones])

        obs = np.zeros((self.n_zones, OBS_DIM), dtype=np.float32)
        for z in range(self.n_zones):
            others = np.delete(outage_rates, z)  # fixed zone order, self excluded
            obs[z] = [
                self._demand[z] / self.zone_peak[z],
                self.criticality[z],
                outage_rates[z],
                np.sin(2 * np.pi * h / 24),
                np.cos(2 * np.pi * h / 24),
                np.sin(2 * np.pi * d / 7),
                np.cos(2 * np.pi * d / 7),
                self._supply / self.peak_est,
                demand_total / self.peak_est,
                deficit / self.peak_est,
                *others,
            ]
        return obs

    def _info_common(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "month": self.month,
            "demand": self._demand.copy(),
            "supply_next": self._supply,
            "cumulative_outage_hours": np.array(
                [z.cumulative_outage_hours for z in self.zones]
            ),
        }

    @property
    def global_state_dim(self) -> int:
        """Flat joint-observation size used as QMIX global state (5*14 = 70)."""
        return self.n_zones * OBS_DIM
