"""VDN — Value Decomposition Networks (Sunehag et al. 2018), SPEC.md §4.

Q_tot(o, a) = Σ_z Q_z(o_z, a_z). The additive decomposition gives each
agent a gradient through the *team* TD error, solving the credit-assignment
gap IDQN ignores, while keeping decentralised greedy execution exact
(argmax of a sum of independent terms = per-agent argmax).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.agents.dqn_agent import (
    AgentQNet,
    JointReplayBuffer,
    clip_and_step,
    linear_epsilon,
    polyak_update,
)


@dataclass
class VDNConfig:
    n_agents: int = 5
    obs_dim: int = 14
    n_actions: int = 5
    hidden: int = 128
    gamma: float = 0.99
    lr: float = 5.0e-4
    batch_size: int = 64
    buffer_size: int = 100000
    train_every: int = 4
    polyak_tau: float = 0.005
    grad_clip: float = 10.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 60000

    @classmethod
    def from_yaml(cls, cfg: dict) -> "VDNConfig":
        t = cfg["training"]
        return cls(
            gamma=t["gamma"], lr=t["lr"], batch_size=t["batch_size"],
            buffer_size=t["buffer_size"], train_every=t["train_every"],
            polyak_tau=t["polyak_tau"], grad_clip=t["grad_clip"],
            eps_start=t["eps_start"], eps_end=t["eps_end"],
            eps_decay_steps=t["eps_decay_steps"],
        )


class SumMixer(nn.Module):
    """Q_tot = Σ_z Q_z(a_z). Parameter-free; a module for API symmetry with QMIX."""

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        return agent_qs.sum(dim=-1)


class VDNAgent:
    def __init__(self, cfg: VDNConfig, seed: int = 42):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.nets = [AgentQNet(cfg.obs_dim, cfg.n_actions, cfg.hidden) for _ in range(cfg.n_agents)]
        self.targets = [AgentQNet(cfg.obs_dim, cfg.n_actions, cfg.hidden) for _ in range(cfg.n_agents)]
        for net, tgt in zip(self.nets, self.targets):
            tgt.load_state_dict(net.state_dict())
        self.mixer = SumMixer()
        params = [p for net in self.nets for p in net.parameters()]
        self.optimizer = torch.optim.Adam(params, lr=cfg.lr)
        self._params = params
        self.buffer = JointReplayBuffer(cfg.buffer_size, cfg.n_agents, cfg.obs_dim)
        self.loss_fn = nn.SmoothL1Loss()
        self.env_steps = 0

    def epsilon(self) -> float:
        return linear_epsilon(
            self.env_steps, self.cfg.eps_start, self.cfg.eps_end, self.cfg.eps_decay_steps
        )

    @torch.no_grad()
    def act(self, obs: np.ndarray, epsilon: float | None = None) -> np.ndarray:
        eps = self.epsilon() if epsilon is None else epsilon
        actions = np.zeros(self.cfg.n_agents, dtype=np.int64)
        for z in range(self.cfg.n_agents):
            if self.rng.random() < eps:
                actions[z] = self.rng.integers(0, self.cfg.n_actions)
            else:
                q = self.nets[z](torch.from_numpy(obs[z]).float())
                actions[z] = int(q.argmax())
        return actions

    def observe(self, obs, acts, rew, next_obs, done) -> None:
        self.buffer.push(obs, acts, rew, next_obs, done)
        self.env_steps += 1

    def ready(self) -> bool:
        return (
            self.buffer.size >= self.cfg.batch_size
            and self.env_steps % self.cfg.train_every == 0
        )

    def _chosen_qs(self, nets, obs: torch.Tensor, acts: torch.Tensor) -> torch.Tensor:
        """Per-agent Q(o_z, a_z) → (batch, n_agents)."""
        qs = [
            nets[z](obs[:, z]).gather(1, acts[:, z : z + 1]).squeeze(1)
            for z in range(self.cfg.n_agents)
        ]
        return torch.stack(qs, dim=1)

    def _greedy_target_qs(self, obs: torch.Tensor) -> torch.Tensor:
        """Double-DQN team target: online argmax per agent, target-net eval."""
        qs = []
        for z in range(self.cfg.n_agents):
            next_a = self.nets[z](obs[:, z]).argmax(dim=1, keepdim=True)
            qs.append(self.targets[z](obs[:, z]).gather(1, next_a).squeeze(1))
        return torch.stack(qs, dim=1)

    def train_step(self) -> float:
        batch = self.buffer.sample(self.cfg.batch_size, self.rng)
        q_tot = self.mixer(self._chosen_qs(self.nets, batch["obs"], batch["acts"]))
        with torch.no_grad():
            next_q_tot = self.mixer(self._greedy_target_qs(batch["next_obs"]))
            target = batch["rews"] + self.cfg.gamma * (1 - batch["dones"]) * next_q_tot
        loss = self.loss_fn(q_tot, target)
        self.optimizer.zero_grad()
        loss.backward()
        clip_and_step(self.optimizer, self._params, self.cfg.grad_clip)
        for net, tgt in zip(self.nets, self.targets):
            polyak_update(net, tgt, self.cfg.polyak_tau)
        return float(loss.detach())

    def save(self, path: str) -> None:
        torch.save(
            {"config": self.cfg.__dict__, "nets": [n.state_dict() for n in self.nets]},
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        for net, tgt, sd in zip(self.nets, self.targets, ckpt["nets"]):
            net.load_state_dict(sd)
            tgt.load_state_dict(sd)
