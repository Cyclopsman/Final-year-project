"""Evaluation protocol (SPEC.md §6).

All 9 policies (6 baselines + IDQN + VDN + QMIX), 5 episodes × 3 seeds each
at supply = 2000, greedy (ε = 0) for learned agents. Results are stamped
with the git commit and a config hash so numbers can never be silently
mixed across environment versions (Hard Rule 1).

Run: python -m src.evaluation.evaluate
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.baseline_agents import BASELINE_REGISTRY, make_baseline
from src.agents.independent_dqn import IDQNAgent, IDQNConfig
from src.agents.qmix_agent import QMixAgent, QMixConfig
from src.agents.vdn_agent import VDNAgent, VDNConfig
from src.environment.grid_env import GridEnv, load_config

# 4 seeds x 5 episodes = 20 eval episodes per policy. One protocol for ALL
# nine policies — baselines and agents must never be scored on different
# episode sets (that would be an uncontrolled comparison).
EVAL_SEEDS = (0, 1, 2, 3)
EPISODES_PER_SEED = 5

METRICS = [
    "wue_mwh",            # weighted unserved energy per episode (MWh)
    "fairness_std",       # std over zones of episode outage rates (outage hours / 168)
    "jain_index",         # Jain fairness index over zone outage hours
    "worst_zone_outage_h",
    "mean_residual_deficit_mw",
    "return",
]


class LearnedPolicy:
    """Greedy wrapper so learned agents share the baselines' act/reset API."""

    def __init__(self, agent):
        self.agent = agent

    def reset(self, seed: int | None = 0) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.agent.act(obs, epsilon=0.0)


def load_agent_policies(cfg: dict, models_dir: Path) -> dict[str, LearnedPolicy]:
    policies = {}
    specs = [
        ("idqn", IDQNAgent, IDQNConfig),
        ("vdn", VDNAgent, VDNConfig),
        ("qmix", QMixAgent, QMixConfig),
    ]
    for name, cls, cfg_cls in specs:
        ckpt = models_dir / f"{name}.pt"
        if ckpt.exists():
            agent = cls(cfg_cls.from_yaml(cfg), seed=0)
            agent.load(str(ckpt))
            policies[name] = LearnedPolicy(agent)
        else:
            print(f"WARNING: {ckpt} missing — {name} skipped")
    return policies


def jain(x: np.ndarray) -> float:
    """J = (Σx)² / (n Σx²); J = 1 for perfectly equal, 1/n for maximally unequal.
    All-zeros (nothing shed anywhere) is perfectly equal by convention."""
    if np.allclose(x, 0):
        return 1.0
    return float(x.sum() ** 2 / (len(x) * (x**2).sum()))


def run_episode(env: GridEnv, policy, seed: int) -> dict[str, float]:
    obs, _ = env.reset(seed=seed)
    policy.reset(seed)
    wue = ret = resid = 0.0
    done = False
    while not done:
        obs, r, _, done, info = env.step(policy.act(obs))
        wue += info["weighted_unserved_energy"]
        resid += info["residual_deficit"]
        ret += r
    outage_hours = np.asarray(info["cumulative_outage_hours"], dtype=float)
    ep_hours = env.episode_hours
    return {
        "wue_mwh": wue,
        "fairness_std": float(np.std(outage_hours / ep_hours)),
        "jain_index": jain(outage_hours),
        "worst_zone_outage_h": float(outage_hours.max()),
        "mean_residual_deficit_mw": resid / ep_hours,
        "return": ret,
        # raw-only extra (not in METRICS summary): per-zone breakdown for figures
        "zone_outage_hours": outage_hours.tolist(),
    }


def evaluate_policy(cfg: dict, policy, name: str) -> list[dict]:
    rows = []
    for seed in EVAL_SEEDS:
        for ep in range(EPISODES_PER_SEED):
            env = GridEnv(copy.deepcopy(cfg))
            # Unique, reproducible episode seed per (seed, episode).
            ep_seed = 1000 * seed + ep
            m = run_episode(env, policy, ep_seed)
            m.update({"policy": name, "seed": seed, "episode": ep})
            rows.append(m)
    return rows


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def config_hash(path: str = "config/grid_config.yaml") -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="models/saved_models")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    cfg = load_config()
    assert cfg["supply"]["mean_mw"] == 2000, "Hard Rule 2: evaluate at supply=2000"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policies: dict[str, object] = {
        name: make_baseline(name, copy.deepcopy(cfg)) for name in BASELINE_REGISTRY
    }
    policies.update(load_agent_policies(cfg, Path(args.models)))

    all_rows: list[dict] = []
    agent_rows: dict[str, list[dict]] = {}
    for name, policy in policies.items():
        rows = evaluate_policy(cfg, policy, name)
        all_rows.extend(rows)
        if name in ("idqn", "vdn", "qmix"):
            agent_rows[name] = rows
        mean_ret = np.mean([r["return"] for r in rows])
        print(f"{name:14s} mean return {mean_ret:8.2f} over {len(rows)} episodes", flush=True)

    df = pd.DataFrame(all_rows)
    stamp = {"git_commit": git_commit(), "config_hash": config_hash(),
             "supply_mean_mw": cfg["supply"]["mean_mw"],
             "protocol": f"{EPISODES_PER_SEED} episodes x {len(EVAL_SEEDS)} seeds, eps=0"}

    summary = df.groupby("policy")[METRICS].agg(["mean", "std"]).round(4)
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    for k, v in stamp.items():
        summary[k] = v
    summary.to_csv(out_dir / "all_policies_metrics.csv", index=False)

    with open(out_dir / "eval_raw.json", "w") as f:
        json.dump({"stamp": stamp, "episodes": all_rows}, f, indent=1, default=float)

    # Per-agent eval JSONs (idqn_eval.json / vdn_eval.json / qmix_eval.json):
    # same episodes as the raw dump, split out with per-metric mean/std so a
    # single agent's result is consumable without re-parsing the full table.
    for name, rows in agent_rows.items():
        summary_stats = {
            m: {"mean": float(np.mean([r[m] for r in rows])),
                "std": float(np.std([r[m] for r in rows]))}
            for m in METRICS
        }
        with open(out_dir / f"{name}_eval.json", "w") as f:
            json.dump({"policy": name, "stamp": stamp, "summary": summary_stats,
                       "episodes": rows}, f, indent=1, default=float)

    print(f"\nwrote {out_dir}/all_policies_metrics.csv, eval_raw.json, "
          f"{{{', '.join(sorted(agent_rows))}}}_eval.json  [{stamp}]")


if __name__ == "__main__":
    main()
