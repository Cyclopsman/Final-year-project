"""β-ablation (TASKS 2.4, design_decisions entry (i)).

Trains the chosen agent at β ∈ {0, 0.5, 2.0} for a reduced, *equal* budget
(60k steps, same seed) so the three points are comparable, then evaluates
each checkpoint (5 episodes × 3 seeds, ε = 0) and writes
results/ablation_beta.json for the ablation figure. Artifacts are tagged
_beta{value} and never mixed into main results (Hard Rule 1).

Run: python -m src.training.run_beta_ablation --agent vdn
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from src.agents.qmix_agent import QMixAgent, QMixConfig
from src.agents.vdn_agent import VDNAgent, VDNConfig
from src.environment.grid_env import GridEnv, load_config
from src.evaluation.evaluate import EPISODES_PER_SEED, EVAL_SEEDS, LearnedPolicy, run_episode
from src.training.train_utils import apply_beta_override, run_training

BETAS = (0.0, 0.5, 2.0)
STEPS = 60000

AGENTS = {"vdn": (VDNAgent, VDNConfig), "qmix": (QMixAgent, QMixConfig)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=list(AGENTS), required=True,
                   help="best of VDN/QMIX from the main runs")
    p.add_argument("--steps", type=int, default=STEPS)
    args = p.parse_args()

    base_cfg = load_config()
    seed = base_cfg["training"]["seed"]
    cls, cfg_cls = AGENTS[args.agent]
    runs = []

    for beta in BETAS:
        tag = f"{args.agent}_beta{beta:g}"
        ckpt = f"models/saved_models/{tag}.pt"
        log = f"results/train_log_{tag}.csv"
        cfg = apply_beta_override(base_cfg, beta)
        print(f"\n=== ablation {tag}: {args.steps} steps, seed {seed} ===", flush=True)
        agent = cls(cfg_cls.from_yaml(cfg), seed=seed)
        run_training(agent, cfg, args.steps, seed, ckpt, log, eval_every=10000)

        # Evaluate the checkpoint under the same protocol as the main table.
        best = cls(cfg_cls.from_yaml(cfg), seed=0)
        best.load(ckpt)
        policy = LearnedPolicy(best)
        rows = []
        for s in EVAL_SEEDS:
            for ep in range(EPISODES_PER_SEED):
                env = GridEnv(copy.deepcopy(cfg))
                rows.append(run_episode(env, policy, seed=1000 * s + ep))
        entry = {"beta": beta, "checkpoint": ckpt}
        for metric in ("fairness_std", "wue_mwh", "jain_index", "return"):
            vals = np.array([r[metric] for r in rows])
            entry[metric] = {"mean": float(vals.mean()), "std": float(vals.std())}
        print(f"{tag}: WUE {entry['wue_mwh']['mean']:.0f}  "
              f"fairness_std {entry['fairness_std']['mean']:.4f}", flush=True)
        runs.append(entry)

    out = {
        "agent": args.agent, "steps": args.steps, "seed": seed,
        "protocol": f"{EPISODES_PER_SEED} episodes x {len(EVAL_SEEDS)} seeds, eps=0",
        "runs": runs,
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/ablation_beta.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote results/ablation_beta.json")


if __name__ == "__main__":
    main()
