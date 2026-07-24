"""Episode playback demo — "a week in the life" of one policy.

Rolls a chosen policy through one 168-hour episode and renders:
  results/figures/demo_<policy>.png   static 3-panel figure (300 dpi)
  results/figures/demo_<policy>.gif   the same figure revealed hour by hour

Presentation aid ONLY: static matplotlib output (no live dashboard — trimmed
scope stands); numbers here are illustrative single-episode traces, never
report metrics (those come from evaluate.py's 20-episode protocol).

Panels: (1) system demand / supply / served MW — the scarcity the policy
faces; (2) per-zone shed-fraction timeline — WHERE the policy puts outages
(the decision being learned); (3) cumulative outage hours per zone — the
fairness ledger accumulating. Same --seed ⇒ same week for every policy, so
side-by-side runs are directly comparable in a talk.

Run: python -m src.evaluation.demo --policy qmix
     python -m src.evaluation.demo --policy priority --seed 0
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from src.agents.baseline_agents import BASELINE_REGISTRY, make_baseline
from src.agents.independent_dqn import IDQNAgent, IDQNConfig
from src.agents.qmix_agent import QMixAgent, QMixConfig
from src.agents.vdn_agent import VDNAgent, VDNConfig
from src.environment.grid_env import GridEnv, load_config
from src.evaluation.evaluate import LearnedPolicy

FIGDIR = Path("results/figures")
ZONE_COLORS = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#555555"]
AGENTS = {
    "idqn": (IDQNAgent, IDQNConfig),
    "vdn": (VDNAgent, VDNConfig),
    "qmix": (QMixAgent, QMixConfig),
}


def load_policy(name: str, cfg: dict):
    if name in BASELINE_REGISTRY:
        return make_baseline(name, copy.deepcopy(cfg))
    if name in AGENTS:
        cls, cfg_cls = AGENTS[name]
        agent = cls(cfg_cls.from_yaml(cfg), seed=0)
        agent.load(f"models/saved_models/{name}.pt")
        return LearnedPolicy(agent)
    raise KeyError(f"unknown policy '{name}'; options: "
                   f"{sorted(BASELINE_REGISTRY) + sorted(AGENTS)}")


def rollout(cfg: dict, policy, seed: int) -> dict:
    env = GridEnv(copy.deepcopy(cfg))
    obs, _ = env.reset(seed=seed)
    policy.reset(seed)
    T = env.episode_hours
    trace = {k: np.zeros(T) for k in ("demand", "supply", "served", "wue")}
    trace["shed"] = np.zeros((env.n_zones, T))
    done, t = False, 0
    while not done:
        obs, _, _, done, info = env.step(policy.act(obs))
        trace["demand"][t] = info["demand_total"]
        trace["supply"][t] = info["supply"]
        trace["served"][t] = info["served_total"]
        trace["wue"][t] = info["weighted_unserved_energy"]
        trace["shed"][:, t] = info["zone_shed_fraction"]
        t += 1
    trace["month"] = env.month
    return trace


def build_figure(trace: dict, zone_names: list[str], title: str):
    """Return (fig, artists) where artists are updated per animation frame."""
    T = len(trace["demand"])
    hours = np.arange(T)
    fig, axes = plt.subplots(
        3, 1, figsize=(10, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0, 1.0]},
    )
    fig.suptitle(title, fontsize=12)

    ax = axes[0]
    (l_dem,) = ax.plot([], [], color="#333333", lw=1.4, label="demand")
    (l_sup,) = ax.plot([], [], color="#0072B2", lw=1.4, label="supply")
    (l_srv,) = ax.plot([], [], color="#009E73", lw=1.2, ls="--", label="served")
    ax.set_ylim(0, trace["demand"].max() * 1.12)
    ax.set_ylabel("MW")
    ax.legend(ncols=3, fontsize=8, loc="lower right")
    # Static context: day boundaries + evening-peak markers.
    for d in range(1, 7):
        for a in axes:
            a.axvline(24 * d, color="#cccccc", lw=0.5, zorder=0)

    ax = axes[1]
    shed_masked = np.full_like(trace["shed"], np.nan)
    # extent=(left, right, bottom, top) with bottom=4.5, top=-0.5 puts array
    # row 0 (Greater Accra) at the TOP row, aligned with ytick 0's label.
    im = ax.imshow(shed_masked, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                   extent=(0, T, 4.5, -0.5), interpolation="nearest")
    ax.set_yticks(range(5), zone_names, fontsize=8)
    ax.set_ylabel("shed fraction")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 0.25, 0.5, 0.75, 1.0],
                        fraction=0.03, pad=0.01)
    cbar.ax.tick_params(labelsize=7)

    ax = axes[2]
    cum = np.cumsum(trace["shed"], axis=1)
    zone_lines = []
    for z, (name, c) in enumerate(zip(zone_names, ZONE_COLORS)):
        (ln,) = ax.plot([], [], color=c, lw=1.5, label=name)
        zone_lines.append(ln)
    ax.set_ylim(0, max(cum.max() * 1.15, 1.0))
    ax.set_ylabel("cumulative outage h")
    ax.set_xlabel("hour of week")
    ax.set_xlim(0, T)
    ax.legend(ncols=5, fontsize=7, loc="upper left")

    artists = {"l_dem": l_dem, "l_sup": l_sup, "l_srv": l_srv,
               "im": im, "zone_lines": zone_lines, "cum": cum, "hours": hours}
    return fig, artists


def draw_upto(trace: dict, artists: dict, t: int) -> None:
    h = artists["hours"][:t]
    artists["l_dem"].set_data(h, trace["demand"][:t])
    artists["l_sup"].set_data(h, trace["supply"][:t])
    artists["l_srv"].set_data(h, trace["served"][:t])
    shed_masked = np.full_like(trace["shed"], np.nan)
    shed_masked[:, :t] = trace["shed"][:, :t]
    artists["im"].set_data(shed_masked)
    for z, ln in enumerate(artists["zone_lines"]):
        ln.set_data(h, artists["cum"][z, :t])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--seed", type=int, default=0,
                   help="episode seed; keep identical across policies to compare the same week")
    p.add_argument("--gif", action="store_true", default=True)
    p.add_argument("--no-gif", dest="gif", action="store_false")
    args = p.parse_args()

    cfg = load_config()
    policy = load_policy(args.policy, cfg)
    trace = rollout(cfg, policy, args.seed)
    zone_names = [z["name"] for z in cfg["zones"]]
    wue_total = trace["wue"].sum()
    resid = np.maximum(0.0, trace["served"] - trace["supply"]).mean()
    title = (f"{args.policy.upper()} — one week at supply 2000 MW "
             f"(episode seed {args.seed}, month {trace['month']})   "
             f"WUE {wue_total:,.0f} MWh · mean residual deficit {resid:.0f} MW")

    FIGDIR.mkdir(parents=True, exist_ok=True)
    T = len(trace["demand"])

    # Static PNG: fully revealed.
    fig, artists = build_figure(trace, zone_names, title)
    draw_upto(trace, artists, T)
    png = FIGDIR / f"demo_{args.policy}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    print(f"wrote {png}")

    if args.gif:
        # Fresh figure at screen dpi; reveal 2 hours per frame (~7 s at 12 fps).
        fig2, artists2 = build_figure(trace, zone_names, title)

        def update(frame):
            draw_upto(trace, artists2, min((frame + 1) * 2, T))
            return []

        anim = animation.FuncAnimation(fig2, update, frames=T // 2, blit=False)
        gif = FIGDIR / f"demo_{args.policy}.gif"
        anim.save(gif, writer=animation.PillowWriter(fps=12), dpi=80)
        print(f"wrote {gif}")
    plt.close("all")


if __name__ == "__main__":
    main()
