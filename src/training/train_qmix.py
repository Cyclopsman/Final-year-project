"""Train QMIX. Usage: python -m src.training.train_qmix [--steps N] [--seed S]"""
from __future__ import annotations

import argparse

import copy

from src.agents.qmix_agent import QMixAgent, QMixConfig, verify_mixer_consistency
from src.environment.grid_env import GridEnv
from src.agents.dqn_agent import set_global_seeds
from src.environment.grid_env import load_config
from src.training.train_utils import apply_beta_override, run_training


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--beta", type=float, default=None, help="β-ablation override")
    p.add_argument("--out", default="models/saved_models/qmix.pt")
    p.add_argument("--log", default="results/logs/qmix_train.csv")
    args = p.parse_args()

    cfg = apply_beta_override(load_config(), args.beta)
    steps = args.steps or cfg["training"]["qmix_steps"]
    seed = args.seed if args.seed is not None else cfg["training"]["seed"]
    set_global_seeds(seed)  # seed BEFORE net init so checkpoints are reproducible
    agent = QMixAgent(QMixConfig.from_yaml(cfg), seed=seed)
    # Pre-training structural check on real env observation shapes: abort
    # before spending 200k steps if the mixer ever lost monotonicity/IGM.
    check = verify_mixer_consistency(agent, GridEnv(copy.deepcopy(cfg)))
    print(f"mixer consistency: min dQtot/dQz={check['min_dQtot_dQz']:.2e} (>=0), "
          f"IGM {check['igm_pass']}/{check['igm_total']} states exact")
    print(f"QMIX: {steps} steps, seed {seed}, beta={cfg['reward']['beta_fairness']}")
    run_training(agent, cfg, steps, seed, args.out, args.log)


if __name__ == "__main__":
    main()
