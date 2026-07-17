"""Shared training loop for IDQN / VDN / QMIX (SPEC.md §4, TASKS Phase 2).

One loop for all three learners so their runs differ only in the agent
object passed in. Logs a CSV row per episode and per eval checkpoint, and
tracks mean `weighted_unserved_energy` read from the env info dict — if the
t8 keys were ever renamed this column would silently go to zero, so the
loop asserts nonzero WUE after warm-up (the silent-zero bug guard from
TASKS 2.1-2.3).
"""
from __future__ import annotations

import copy
import csv
import time
from pathlib import Path

from src.agents.dqn_agent import set_global_seeds
from src.environment.grid_env import GridEnv


def evaluate_policy(env: GridEnv, agent, n_episodes: int, seed_base: int) -> float:
    """Mean greedy (ε=0) return over n_episodes."""
    total = 0.0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        done = False
        while not done:
            obs, r, _, done, _ = env.step(agent.act(obs, epsilon=0.0))
            total += r
    return total / n_episodes


def run_training(
    agent,
    cfg: dict,
    total_steps: int,
    seed: int,
    ckpt_path: str | Path,
    log_path: str | Path,
    eval_every: int = 5000,
    eval_episodes: int = 2,
    quiet: bool = False,
) -> None:
    set_global_seeds(seed)
    env = GridEnv(copy.deepcopy(cfg))
    eval_env = GridEnv(copy.deepcopy(cfg))
    ckpt_path = Path(ckpt_path)
    log_path = Path(log_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_f = open(log_path, "w", newline="")
    writer = csv.writer(log_f)
    writer.writerow(
        ["step", "episode", "episode_return", "episode_wue", "loss", "epsilon", "eval_return"]
    )

    obs, _ = env.reset(seed=seed)
    episode = 0
    ep_return = 0.0
    ep_wue = 0.0
    last_loss = float("nan")
    best_eval = float("-inf")
    t0 = time.time()

    for step in range(1, total_steps + 1):
        actions = agent.act(obs)
        next_obs, reward, _, truncated, info = env.step(actions)
        agent.observe(obs, actions, reward, next_obs, truncated)
        obs = next_obs
        ep_return += reward
        # .get() mirrors how a mis-named key would fail: silently as zero.
        ep_wue += info.get("weighted_unserved_energy", 0.0)

        if agent.ready():
            last_loss = agent.train_step()

        if truncated:
            episode += 1
            writer.writerow(
                [step, episode, f"{ep_return:.4f}", f"{ep_wue:.2f}",
                 f"{last_loss:.6f}", f"{agent.epsilon():.4f}", ""]
            )
            # Silent-zero bug guard (t8 contract): a full episode at supply
            # 2000 MW cannot plausibly have zero unserved energy once the
            # agent sheds at all; zero here means the info keys broke.
            if episode >= 3 and ep_wue == 0.0 and agent.epsilon() > 0.5:
                raise RuntimeError(
                    "episode WUE == 0 — t8 info keys are not reaching training"
                )
            ep_return = 0.0
            ep_wue = 0.0
            obs, _ = env.reset(seed=seed + episode)

        if step % eval_every == 0 or step == total_steps:
            eval_ret = evaluate_policy(eval_env, agent, eval_episodes, seed_base=10_000)
            writer.writerow([step, episode, "", "", "", "", f"{eval_ret:.4f}"])
            log_f.flush()
            if eval_ret >= best_eval:
                best_eval = eval_ret
                agent.save(str(ckpt_path))
            if not quiet:
                rate = step / (time.time() - t0)
                eta_min = (total_steps - step) / rate / 60
                print(
                    f"step {step}/{total_steps}  eval_return={eval_ret:8.2f}  "
                    f"best={best_eval:8.2f}  eps={agent.epsilon():.3f}  "
                    f"loss={last_loss:.5f}  {rate:.0f} steps/s  ETA {eta_min:.0f} min",
                    flush=True,
                )

    # Final save if never saved (short smoke runs).
    if best_eval == float("-inf"):
        agent.save(str(ckpt_path))
    log_f.close()


def apply_beta_override(cfg: dict, beta: float | None) -> dict:
    """Return a config copy with reward.beta_fairness overridden (ablation only).

    Ablation checkpoints/logs must be tagged with the β value — never mixed
    with main results (Hard Rule 1).
    """
    cfg = copy.deepcopy(cfg)
    if beta is not None:
        cfg["reward"]["beta_fairness"] = float(beta)
    return cfg
