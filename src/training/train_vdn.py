"""Train VDN. Usage: python -m src.training.train_vdn [--steps N] [--seed S]"""
from __future__ import annotations

import argparse

from src.agents.vdn_agent import VDNAgent, VDNConfig
from src.environment.grid_env import load_config
from src.training.train_utils import apply_beta_override, run_training


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--beta", type=float, default=None, help="β-ablation override")
    p.add_argument("--out", default="models/saved_models/vdn.pt")
    p.add_argument("--log", default="results/train_log_vdn.csv")
    args = p.parse_args()

    cfg = apply_beta_override(load_config(), args.beta)
    steps = args.steps or cfg["training"]["vdn_steps"]
    seed = args.seed if args.seed is not None else cfg["training"]["seed"]
    agent = VDNAgent(VDNConfig.from_yaml(cfg), seed=seed)
    print(f"VDN: {steps} steps, seed {seed}, beta={cfg['reward']['beta_fairness']}")
    run_training(agent, cfg, steps, seed, args.out, args.log)


if __name__ == "__main__":
    main()
