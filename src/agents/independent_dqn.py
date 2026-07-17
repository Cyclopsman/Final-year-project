"""IDQN — five fully independent DQN learners (SPEC.md §4).

Each zone trains its own Q-network on the shared team reward with no value
mixing. This is the non-cooperative lower bound of the learned family: from
any single agent's view the other four are part of a non-stationary
environment, so IDQN establishes what independent learning alone achieves
before VDN/QMIX add explicit credit assignment.

Transitions are stored jointly and each agent trains on its own slice of
the same sampled minibatch ("shared transitions, independent sampling"
variant allowed by SPEC §4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.agents.dqn_agent import (
    JointReplayBuffer,
    QNet,
    clip_and_step,
    linear_epsilon,
    polyak_update,
)


@dataclass
class IDQNConfig:
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
    def from_yaml(cls, cfg: dict) -> "IDQNConfig":
        t = cfg["training"]
        return cls(
            gamma=t["gamma"], lr=t["lr"], batch_size=t["batch_size"],
            buffer_size=t["buffer_size"], train_every=t["train_every"],
            polyak_tau=t["polyak_tau"], grad_clip=t["grad_clip"],
            eps_start=t["eps_start"], eps_end=t["eps_end"],
            eps_decay_steps=t["eps_decay_steps"],
        )


class IDQNAgent:
    def __init__(self, cfg: IDQNConfig, seed: int = 42):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.nets = [QNet(cfg.obs_dim, cfg.n_actions, cfg.hidden) for _ in range(cfg.n_agents)]
        self.targets = [QNet(cfg.obs_dim, cfg.n_actions, cfg.hidden) for _ in range(cfg.n_agents)]
        for net, tgt in zip(self.nets, self.targets):
            tgt.load_state_dict(net.state_dict())
        self.optimizers = [
            torch.optim.Adam(net.parameters(), lr=cfg.lr) for net in self.nets
        ]
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
        q = None
        if eps < 1.0:
            q = torch.stack(
                [net(torch.from_numpy(obs[z]).float()) for z, net in enumerate(self.nets)]
            )
        for z in range(self.cfg.n_agents):
            if self.rng.random() < eps:
                actions[z] = self.rng.integers(0, self.cfg.n_actions)
            else:
                actions[z] = int(q[z].argmax())
        return actions

    def observe(self, obs, acts, rew, next_obs, done) -> None:
        self.buffer.push(obs, acts, rew, next_obs, done)
        self.env_steps += 1

    def ready(self) -> bool:
        return (
            self.buffer.size >= self.cfg.batch_size
            and self.env_steps % self.cfg.train_every == 0
        )

    def train_step(self) -> float:
        batch = self.buffer.sample(self.cfg.batch_size, self.rng)
        total_loss = 0.0
        for z in range(self.cfg.n_agents):
            net, tgt, opt = self.nets[z], self.targets[z], self.optimizers[z]
            obs, next_obs = batch["obs"][:, z], batch["next_obs"][:, z]
            acts = batch["acts"][:, z]
            q = net(obs).gather(1, acts.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_a = net(next_obs).argmax(dim=1, keepdim=True)  # Double DQN
                next_q = tgt(next_obs).gather(1, next_a).squeeze(1)
                target = batch["rews"] + self.cfg.gamma * (1 - batch["dones"]) * next_q
            loss = self.loss_fn(q, target)
            opt.zero_grad()
            loss.backward()
            clip_and_step(opt, net.parameters(), self.cfg.grad_clip)
            polyak_update(net, tgt, self.cfg.polyak_tau)
            total_loss += float(loss.detach())
        return total_loss / self.cfg.n_agents

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": self.cfg.__dict__,
                "nets": [n.state_dict() for n in self.nets],
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        for net, tgt, sd in zip(self.nets, self.targets, ckpt["nets"]):
            net.load_state_dict(sd)
            tgt.load_state_dict(sd)
