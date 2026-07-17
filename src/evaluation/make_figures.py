"""Report figures (SPEC.md §7) — six 300-dpi PNGs into results/figures/.

Design rules applied throughout:
  - Colorblind-safe agent palette (Okabe–Ito derived, validated):
    IDQN vermillion, VDN blue, QMIX green; baselines are de-emphasized grey
    and identified by direct labels, never by hue.
  - Secondary encoding everywhere (distinct markers, hatches, line styles)
    so every figure survives grayscale printing (TASKS 3.2 AC).
  - One axis per plot — paired metrics get side-by-side panels, never twin axes.

Run: python -m src.evaluation.make_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.environment.grid_env import load_config

RESULTS = Path("results")
FIGDIR = RESULTS / "figures"

AGENT_COLORS = {"idqn": "#D55E00", "vdn": "#0072B2", "qmix": "#009E73"}
AGENT_MARKERS = {"idqn": "^", "vdn": "s", "qmix": "*"}
BASELINE_GREY = "#666666"
LABELS = {
    "no_shedding": "NoShedding", "random": "Random", "round_robin": "RoundRobin",
    "priority": "Priority", "proportional": "Proportional",
    "fair_rotation": "FairRotation", "idqn": "IDQN", "vdn": "VDN", "qmix": "QMIX",
}

plt.rcParams.update(
    {
        "figure.dpi": 100, "savefig.dpi": 300, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.axisbelow": True,
    }
)


def load_raw() -> tuple[pd.DataFrame, dict]:
    with open(RESULTS / "eval_raw.json") as f:
        raw = json.load(f)
    return pd.DataFrame(raw["episodes"]), raw["stamp"]


def save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGDIR / name, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGDIR / name}")


# ------------------------------------------------------------------ figures
INFEASIBLE_MW = 100  # mean residual deficit above this = uncontrolled outages


def pareto_scatter(df: pd.DataFrame) -> None:
    g = df.groupby("policy")[["wue_mwh", "jain_index", "mean_residual_deficit_mw"]].agg(
        ["mean", "std"]
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Target corner: low WUE, high Jain (top-left).
    xmax = g[("wue_mwh", "mean")].max() * 1.1
    ax.axhspan(0.9, 1.02, xmax=0.45, color="#009E73", alpha=0.08, zorder=0)
    ax.annotate("target corner:\nefficient AND fair", xy=(0.03, 0.955),
                xycoords="axes fraction", fontsize=8, color="#00543d", style="italic")

    # Custom annotations: rotation baselines share one point; infeasible
    # policies (uncontrolled residual deficit) are flagged in their label.
    custom_labels = {
        "round_robin": "RoundRobin ≡ FairRotation\n(204 MW unmet)",
        "fair_rotation": None,  # merged into round_robin's label
        "no_shedding": "NoShedding\n(637 MW unmet)",
    }
    offsets = {"no_shedding": (8, -24), "round_robin": (10, -22),
               "priority": (8, -2), "proportional": (-18, 8), "random": (-30, 8),
               "vdn": (8, 2), "qmix": (-38, -4), "idqn": (8, 2)}

    for policy, row in g.iterrows():
        x, xerr = row[("wue_mwh", "mean")], row[("wue_mwh", "std")]
        y, yerr = row[("jain_index", "mean")], row[("jain_index", "std")]
        infeasible = row[("mean_residual_deficit_mw", "mean")] > INFEASIBLE_MW
        if policy in AGENT_COLORS:
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=AGENT_MARKERS[policy],
                        color=AGENT_COLORS[policy], ms=13 if policy == "qmix" else 9,
                        capsize=3, lw=1, zorder=5,
                        markeredgecolor="white", markeredgewidth=0.8)
        else:
            # Hollow circle = infeasible (leaves uncontrolled deficit).
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", color=BASELINE_GREY,
                        ms=7, capsize=3, lw=1, alpha=0.9, zorder=4,
                        markerfacecolor="none" if infeasible else BASELINE_GREY,
                        markeredgewidth=1.4)
        label = custom_labels.get(policy, LABELS[policy])
        if policy == "round_robin":
            # Crowded corner: anchor the label in open space with a leader arrow.
            ax.annotate(label, (x, y), xytext=(x + 36000, y - 0.075),
                        fontsize=8, ha="left", va="top",
                        arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8))
        elif label is not None:
            ax.annotate(label, (x, y), xytext=offsets.get(policy, (6, 6)),
                        textcoords="offset points", fontsize=8)

    ax.annotate(
        "hollow circle = infeasible: mean uncontrolled\nresidual deficit "
        f"> {INFEASIBLE_MW} MW (unmet load not counted in WUE)",
        xy=(0.28, 0.06), xycoords="axes fraction", fontsize=7.5,
        color="#444444", style="italic",
    )
    ax.set_xlim(-xmax * 0.03, xmax)
    ax.set_ylim(0.15, 1.04)
    ax.set_xlabel("Weighted unserved energy (MWh / episode)  →  worse")
    ax.set_ylabel("Jain fairness index  →  better")
    ax.set_title("Efficiency–fairness Pareto landscape (supply = 2000 MW)")
    save(fig, "pareto_scatter.png")


def policy_comparison(df: pd.DataFrame) -> None:
    metrics = [
        ("wue_mwh", "Weighted unserved\nenergy"),
        ("fairness_std", "Fairness std\n(outage rates)"),
        ("worst_zone_outage_h", "Worst-zone\noutage hours"),
        ("mean_residual_deficit_mw", "Mean residual\ndeficit"),
    ]
    order = [p for p in LABELS if p in df["policy"].unique()]
    g = df.groupby("policy").mean(numeric_only=True).loc[order]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    width = 0.2
    x = np.arange(len(order))
    hatches = ["", "//", "..", "xx"]
    greys = ["#B8CBE0", "#8FB3D9", "#5B8FC9", "#2F6BAF"]
    for i, ((m, label), hatch, c) in enumerate(zip(metrics, hatches, greys)):
        vals = g[m] / max(g[m].max(), 1e-9)  # normalize to worst = 1
        ax.bar(x + (i - 1.5) * width, vals, width * 0.92, label=label,
               color=c, hatch=hatch, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x, [LABELS[p] for p in order], rotation=20, ha="right")
    ax.set_ylabel("Normalized (worst policy = 1.0, lower is better)")
    ax.set_title("Key metrics by policy (normalized per metric)")
    ax.legend(ncols=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    save(fig, "policy_comparison.png")


def zone_outage(df: pd.DataFrame, cfg: dict) -> None:
    zone_names = [z["name"] for z in cfg["zones"]]
    order = [p for p in LABELS if p in df["policy"].unique()]
    per_policy = {
        p: np.mean(np.stack(df[df["policy"] == p]["zone_outage_hours"]), axis=0)
        for p in order
    }
    fig, ax = plt.subplots(figsize=(7.5, 4))
    n = len(order)
    width = 0.8 / n
    x = np.arange(len(zone_names))
    for i, p in enumerate(order):
        color = AGENT_COLORS.get(p, BASELINE_GREY)
        alpha = 1.0 if p in AGENT_COLORS else 0.35 + 0.1 * i
        ax.bar(x + (i - n / 2 + 0.5) * width, per_policy[p], width * 0.9,
               label=LABELS[p], color=color, alpha=alpha,
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(x, zone_names)
    ax.set_ylabel("Cumulative outage hours / episode")
    ax.set_title("Where the outages land: per-zone outage hours by policy")
    ax.legend(ncols=3, fontsize=7.5)
    save(fig, "zone_outage.png")


def training_curves() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name in ["idqn", "vdn", "qmix"]:
        path = RESULTS / f"train_log_{name}.csv"
        if not path.exists():
            continue
        log = pd.read_csv(path)
        ep = log.dropna(subset=["episode_return"])
        smoothed = ep["episode_return"].rolling(20, min_periods=1).mean()
        ax.plot(ep["step"], smoothed, color=AGENT_COLORS[name], lw=1.8,
                label=f"{LABELS[name]} (train, smoothed)")
        ev = log.dropna(subset=["eval_return"])
        ax.plot(ev["step"], ev["eval_return"], color=AGENT_COLORS[name], lw=1,
                ls="--", alpha=0.7, marker=AGENT_MARKERS[name], ms=4,
                label=f"{LABELS[name]} (greedy eval)")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode return")
    ax.set_title("Training curves (168 h episodes, supply = 2000 MW)")
    ax.legend(fontsize=7.5, ncols=3)
    save(fig, "training_curves.png")


def ablation_beta() -> None:
    path = RESULTS / "ablation_beta.json"
    if not path.exists():
        print("ablation_beta.json missing — run the ablation first; skipping figure")
        return
    with open(path) as f:
        ab = json.load(f)
    betas = [r["beta"] for r in ab["runs"]]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    panels = [
        ("fairness_std", "Fairness std of outage rates\n(lower = fairer)"),
        ("wue_mwh", "Weighted unserved energy\n(MWh / episode)"),
    ]
    for ax, (metric, label) in zip(axes, panels):
        means = [r[metric]["mean"] for r in ab["runs"]]
        stds = [r[metric]["std"] for r in ab["runs"]]
        ax.errorbar(betas, means, yerr=stds, marker="o", ms=7, lw=1.8,
                    color="#0072B2", capsize=4)
        ax.set_xlabel("Fairness weight β")
        ax.set_ylabel(label)
        ax.set_xticks(betas)
    fig.suptitle(
        f"β-ablation ({ab['agent'].upper()}, {ab['steps']:,} steps each, seed {ab['seed']})",
        fontsize=10,
    )
    save(fig, "ablation_beta.png")


def forecaster_validation(cfg: dict) -> None:
    from src.forecasting.lstm_forecaster import (
        N_ZONES, WINDOW, LSTMForecaster, generate_series, make_windows,
    )

    ckpt = torch.load("models/saved_models/lstm.pt", map_location="cpu",
                      weights_only=False)
    model = LSTMForecaster()
    model.load_state_dict(ckpt["model"])
    model.eval()
    base = ckpt["base"]

    series = generate_series(cfg, n_weeks=2, seed=777)  # held-out seed
    norm = series.copy()
    norm[:, :N_ZONES] /= base
    x, y = make_windows(norm)
    with torch.no_grad():
        pred = model(torch.from_numpy(x)).numpy() * base
    actual = y * base

    fig, ax = plt.subplots(figsize=(7, 4))
    hours = np.arange(48)
    for z, (name, color) in enumerate(
        zip([z["name"] for z in cfg["zones"]],
            ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#555555"])
    ):
        ax.plot(hours, actual[:48, z], color=color, lw=1.6, label=name)
        ax.plot(hours, pred[:48, z], color=color, lw=1.2, ls="--", alpha=0.8)
    ax.plot([], [], color="k", lw=1.6, label="actual (solid)")
    ax.plot([], [], color="k", lw=1.2, ls="--", label="predicted (dashed)")
    ax.set_xlabel("Hour (held-out 48 h sample)")
    ax.set_ylabel("Demand (MW)")
    ax.set_title("LSTM forecaster: next-hour demand, prediction vs actual")
    ax.legend(fontsize=7, ncols=2)

    mae_rows = [
        [z["name"], f"{m:.1f}"] for z, m in zip(cfg["zones"], ckpt["val_mae_mw"])
    ]
    table = ax.table(cellText=mae_rows, colLabels=["Zone", "val MAE (MW)"],
                     loc="lower right", cellLoc="left",
                     bbox=[0.63, 0.04, 0.35, 0.42])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.set_zorder(6)
    for cell in table.get_celld().values():
        cell.set_edgecolor("#cccccc")
        cell.set_facecolor("white")  # opaque so plot lines don't bleed through
    save(fig, "forecaster_validation.png")


def main() -> None:
    cfg = load_config()
    df, stamp = load_raw()
    print(f"figures from eval stamped {stamp}")
    pareto_scatter(df)
    policy_comparison(df)
    zone_outage(df, cfg)
    training_curves()
    ablation_beta()
    forecaster_validation(cfg)


if __name__ == "__main__":
    main()
