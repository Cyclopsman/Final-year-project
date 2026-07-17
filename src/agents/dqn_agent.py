"""Shared DQN building blocks used by IDQN, VDN and QMIX (SPEC.md §4).

All three learners share the same value-learning tricks so that any
performance difference between them is attributable to the *credit
assignment* mechanism (none / additive / monotonic mixing), not to
incidental implementation choices:
  - Double DQN targets (online-net argmax, target-net evaluation)
  - Huber loss, gradient-norm clip 10
  - Polyak soft target updates (τ = 0.005) every train step
  - ε-greedy with linear decay
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class QNet(nn.Module):
    """Per-zone Q-network: obs(14) -> 128 -> 128 -> Q(5)."""

    def __init__(self, obs_dim: int = 14, n_actions: int = 5, hidden: int = 128,
                 orthogonal: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        if orthogonal:
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class AgentQNet(QNet):
    """VDN/QMIX per-agent trunk — QNet with orthogonal init (SPEC §4)."""

    def __init__(self, obs_dim: int = 14, n_actions: int = 5, hidden: int = 128):
        super().__init__(obs_dim, n_actions, hidden, orthogonal=True)


class JointReplayBuffer:
    """Stores joint transitions (obs[5,14], acts[5], rew, next_obs[5,14], done)."""

    def __init__(self, capacity: int, n_agents: int, obs_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.acts = np.zeros((capacity, n_agents), dtype=np.int64)
        self.rews = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def push(self, obs, acts, rew, next_obs, done) -> None:
        i = self.idx
        self.obs[i] = obs
        self.acts[i] = acts
        self.rews[i] = rew
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        idx = rng.integers(0, self.size, size=batch_size)
        return {
            "obs": torch.from_numpy(self.obs[idx]),
            "acts": torch.from_numpy(self.acts[idx]),
            "rews": torch.from_numpy(self.rews[idx]),
            "next_obs": torch.from_numpy(self.next_obs[idx]),
            "dones": torch.from_numpy(self.dones[idx]),
        }


class StatefulReplayBuffer(JointReplayBuffer):
    """JointReplayBuffer that additionally stores the global state (for QMIX)."""

    def __init__(self, capacity: int, n_agents: int, obs_dim: int, state_dim: int):
        super().__init__(capacity, n_agents, obs_dim)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)

    def push(self, obs, acts, rew, next_obs, done, state=None, next_state=None) -> None:
        i = self.idx
        self.states[i] = state
        self.next_states[i] = next_state
        super().push(obs, acts, rew, next_obs, done)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        idx = rng.integers(0, self.size, size=batch_size)
        return {
            "obs": torch.from_numpy(self.obs[idx]),
            "acts": torch.from_numpy(self.acts[idx]),
            "rews": torch.from_numpy(self.rews[idx]),
            "next_obs": torch.from_numpy(self.next_obs[idx]),
            "dones": torch.from_numpy(self.dones[idx]),
            "states": torch.from_numpy(self.states[idx]),
            "next_states": torch.from_numpy(self.next_states[idx]),
        }


def linear_epsilon(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    frac = min(1.0, step / decay_steps)
    return eps_start + frac * (eps_end - eps_start)


@torch.no_grad()
def polyak_update(online: nn.Module, target: nn.Module, tau: float) -> None:
    for p, tp in zip(online.parameters(), target.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * p.data)


def clip_and_step(optimizer: torch.optim.Optimizer, params, grad_clip: float) -> float:
    grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
    optimizer.step()
    return float(grad_norm)
