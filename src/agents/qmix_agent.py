"""QMIX — monotonic value factorisation (Rashid et al. 2018), SPEC.md §4.

Q_tot = f(Q_1..Q_5; s) where f is a state-conditioned mixing network whose
weights are produced by hypernetworks and constrained non-negative (|abs|),
guaranteeing ∂Q_tot/∂Q_z ≥ 0. Monotonicity keeps decentralised argmax
consistent with the joint argmax while letting the *state* (here the 70-dim
concatenated observation) modulate how much each zone's value matters —
e.g. weighting Accra's Q more during evening-peak deficits.

Note (Papoudakis et al. 2021): QMIX ≥ VDN is task-dependent; either
ordering here is reported as a finding, not a failure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.agents.dqn_agent import (
    AgentQNet,
    StatefulReplayBuffer,
    clip_and_step,
    linear_epsilon,
    polyak_update,
)


@dataclass
class QMixConfig:
    n_agents: int = 5
    obs_dim: int = 14
    n_actions: int = 5
    hidden: int = 128
    state_dim: int = 70
    embed_dim: int = 32
    hypernet_hidden: int = 64
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
    def from_yaml(cls, cfg: dict) -> "QMixConfig":
        t = cfg["training"]
        return cls(
            gamma=t["gamma"], lr=t["lr"], batch_size=t["batch_size"],
            buffer_size=t["buffer_size"], train_every=t["train_every"],
            polyak_tau=t["polyak_tau"], grad_clip=t["grad_clip"],
            eps_start=t["eps_start"], eps_end=t["eps_end"],
            eps_decay_steps=t["eps_decay_steps"],
        )


class QMixer(nn.Module):
    """Hypernetwork mixing: W1 (|abs|, n_agents→embed), b1, W2 (|abs|, embed→1),
    b2 (scalar via 2-layer hypernet), all conditioned on the global state."""

    def __init__(self, cfg: QMixConfig):
        super().__init__()
        self.n_agents = cfg.n_agents
        self.embed_dim = cfg.embed_dim
        self.hyper_w1 = nn.Linear(cfg.state_dim, cfg.n_agents * cfg.embed_dim)
        self.hyper_b1 = nn.Linear(cfg.state_dim, cfg.embed_dim)
        self.hyper_w2 = nn.Linear(cfg.state_dim, cfg.embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.hypernet_hidden),
            nn.ReLU(),
            nn.Linear(cfg.hypernet_hidden, 1),
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # agent_qs: (batch, n_agents), state: (batch, state_dim)
        batch = agent_qs.shape[0]
        w1 = torch.abs(self.hyper_w1(state)).view(batch, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(batch, 1, self.embed_dim)
        hidden = torch.nn.functional.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        w2 = torch.abs(self.hyper_w2(state)).view(batch, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(batch, 1, 1)
        return (torch.bmm(hidden, w2) + b2).view(batch)


class QMixAgent:
    def __init__(self, cfg: QMixConfig, seed: int = 42):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.nets = [AgentQNet(cfg.obs_dim, cfg.n_actions, cfg.hidden) for _ in range(cfg.n_agents)]
        self.targets = [AgentQNet(cfg.obs_dim, cfg.n_actions, cfg.hidden) for _ in range(cfg.n_agents)]
        for net, tgt in zip(self.nets, self.targets):
            tgt.load_state_dict(net.state_dict())
        self.mixer = QMixer(cfg)
        self.target_mixer = QMixer(cfg)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        params = [p for net in self.nets for p in net.parameters()] + list(
            self.mixer.parameters()
        )
        self.optimizer = torch.optim.Adam(params, lr=cfg.lr)
        self._params = params
        self.buffer = StatefulReplayBuffer(
            cfg.buffer_size, cfg.n_agents, cfg.obs_dim, cfg.state_dim
        )
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
        # Global state = flattened joint observation (SPEC §4).
        self.buffer.push(
            obs, acts, rew, next_obs, done,
            state=np.asarray(obs, dtype=np.float32).reshape(-1),
            next_state=np.asarray(next_obs, dtype=np.float32).reshape(-1),
        )
        self.env_steps += 1

    def ready(self) -> bool:
        return (
            self.buffer.size >= self.cfg.batch_size
            and self.env_steps % self.cfg.train_every == 0
        )

    def train_step(self) -> float:
        batch = self.buffer.sample(self.cfg.batch_size, self.rng)
        chosen = torch.stack(
            [
                self.nets[z](batch["obs"][:, z])
                .gather(1, batch["acts"][:, z : z + 1])
                .squeeze(1)
                for z in range(self.cfg.n_agents)
            ],
            dim=1,
        )
        q_tot = self.mixer(chosen, batch["states"])
        with torch.no_grad():
            next_qs = []
            for z in range(self.cfg.n_agents):
                next_a = self.nets[z](batch["next_obs"][:, z]).argmax(dim=1, keepdim=True)
                next_qs.append(
                    self.targets[z](batch["next_obs"][:, z]).gather(1, next_a).squeeze(1)
                )
            next_q_tot = self.target_mixer(torch.stack(next_qs, dim=1), batch["next_states"])
            target = batch["rews"] + self.cfg.gamma * (1 - batch["dones"]) * next_q_tot
        loss = self.loss_fn(q_tot, target)
        self.optimizer.zero_grad()
        loss.backward()
        clip_and_step(self.optimizer, self._params, self.cfg.grad_clip)
        for net, tgt in zip(self.nets, self.targets):
            polyak_update(net, tgt, self.cfg.polyak_tau)
        polyak_update(self.mixer, self.target_mixer, self.cfg.polyak_tau)
        return float(loss.detach())

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": self.cfg.__dict__,
                "nets": [n.state_dict() for n in self.nets],
                "mixer": self.mixer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        for net, tgt, sd in zip(self.nets, self.targets, ckpt["nets"]):
            net.load_state_dict(sd)
            tgt.load_state_dict(sd)
        self.mixer.load_state_dict(ckpt["mixer"])
        self.target_mixer.load_state_dict(ckpt["mixer"])


def verify_mixer_consistency(
    agent: QMixAgent, env, n_states: int = 64, igm_states: int = 8, tol: float = 1e-5
) -> dict:
    """Empirical check of QMIX's two structural guarantees on REAL env states.

    (1) Monotonicity: ∂Q_tot/∂Q_z ≥ 0 for every agent, verified by autograd on
        states collected from an actual rollout (not synthetic tensors, so the
        check covers the observation ranges the mixer will really see).
    (2) IGM (Individual-Global-Max): per-agent greedy argmax equals the argmax
        of Q_tot over ALL 5^5 = 3125 joint actions — enumerable exactly here,
        which is precisely why the 5-zone abstraction keeps this check cheap.

    Monotonicity implies IGM up to ties, but (2) is checked independently so a
    mixer bug (e.g. a missing abs()) is caught by two different instruments.
    """
    # Collect real observations via a short random rollout.
    obs, _ = env.reset(seed=12345)
    states, obs_list = [], []
    rng = np.random.default_rng(0)
    for _ in range(n_states):
        obs_list.append(obs.copy())
        states.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        obs, _, _, truncated, _ = env.step(rng.integers(0, agent.cfg.n_actions, size=agent.cfg.n_agents))
        if truncated:
            obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
    state_t = torch.from_numpy(np.stack(states))

    # (1) Monotonicity via autograd: gradient of sum(Q_tot) wrt agent_qs.
    qs = torch.randn(n_states, agent.cfg.n_agents, requires_grad=True)
    q_tot = agent.mixer(qs, state_t)
    (grad,) = torch.autograd.grad(q_tot.sum(), qs)
    min_grad = float(grad.min())
    monotone = min_grad >= -tol

    # (2) IGM on a subset of states: enumerate all joint actions.
    n_a, n_z = agent.cfg.n_actions, agent.cfg.n_agents
    joint = np.stack(np.meshgrid(*[np.arange(n_a)] * n_z, indexing="ij"), -1).reshape(-1, n_z)
    igm_ok = 0
    with torch.no_grad():
        for i in range(min(igm_states, n_states)):
            o = torch.from_numpy(obs_list[i]).float()
            q_matrix = torch.stack([agent.nets[z](o[z]) for z in range(n_z)])  # (5 zones, 5 actions)
            greedy = q_matrix.argmax(dim=1).numpy()
            # per_joint_qs[j, z] = Q_z(o_z, joint[j, z]) → (3125, 5)
            per_joint_qs = q_matrix[
                torch.arange(n_z).unsqueeze(0), torch.from_numpy(joint)
            ].float()
            q_tots = agent.mixer(per_joint_qs, state_t[i].expand(len(joint), -1))
            best_joint = joint[int(q_tots.argmax())]
            # Accept value ties: greedy action must achieve the max Q_tot.
            greedy_q_tot = agent.mixer(
                q_matrix[np.arange(n_z), greedy].unsqueeze(0).float(), state_t[i : i + 1]
            )
            if np.array_equal(best_joint, greedy) or float(q_tots.max() - greedy_q_tot) <= tol:
                igm_ok += 1
    result = {
        "monotone": monotone,
        "min_dQtot_dQz": min_grad,
        "igm_pass": igm_ok,
        "igm_total": min(igm_states, n_states),
    }
    if not monotone or igm_ok < result["igm_total"]:
        raise AssertionError(f"QMIX mixer consistency check FAILED: {result}")
    return result
