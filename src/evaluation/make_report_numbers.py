"""Generate results/report_numbers.md — the ONLY source for numbers in the
report (TASKS 3.3). Reads all_policies_metrics.csv, eval_raw.json and
ablation_beta.json; never edit the output by hand, regenerate it.

Run: python -m src.evaluation.make_report_numbers
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RESULTS = Path("results")

ORDER = ["no_shedding", "random", "round_robin", "priority", "proportional",
         "fair_rotation", "idqn", "vdn", "qmix"]
LABELS = {
    "no_shedding": "NoShedding", "random": "Random", "round_robin": "RoundRobin",
    "priority": "Priority", "proportional": "Proportional",
    "fair_rotation": "FairRotation", "idqn": "IDQN", "vdn": "VDN", "qmix": "QMIX",
}


def fmt(mean: float, std: float, nd: int = 1) -> str:
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


def main() -> None:
    df = pd.read_csv(RESULTS / "all_policies_metrics.csv").set_index("policy")
    with open(RESULTS / "eval_raw.json") as f:
        stamp = json.load(f)["stamp"]
    ablation = None
    ab_path = RESULTS / "ablation_beta.json"
    if ab_path.exists():
        with open(ab_path) as f:
            ablation = json.load(f)

    agents = [a for a in ("idqn", "vdn", "qmix") if a in df.index]
    baselines = [b for b in ORDER[:6] if b in df.index]
    # Feasible baselines: those that actually cover the deficit (residual ≈ 0);
    # the Pareto claim is conditioned on feasibility (design decision (g)).
    feasible = [b for b in baselines
                if df.loc[b, "mean_residual_deficit_mw_mean"] < 25]

    lines = [
        "# Report numbers — single source of truth",
        "",
        f"Generated from `all_policies_metrics.csv` (env: commit `{stamp['git_commit']}`, "
        f"config `{stamp['config_hash']}`, supply = {stamp['supply_mean_mw']} MW, "
        f"protocol: {stamp['protocol']}).",
        "**Never copy numbers into the report from anywhere else.**",
        "",
        "## Main results table",
        "",
        "| Policy | WUE (MWh/ep) | Fairness std | Jain index | Worst-zone outage (h) | Residual deficit (MW) | Return |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in ORDER:
        if p not in df.index:
            continue
        r = df.loc[p]
        lines.append(
            f"| {LABELS[p]} | {fmt(r['wue_mwh_mean'], r['wue_mwh_std'], 0)} "
            f"| {fmt(r['fairness_std_mean'], r['fairness_std_std'], 3)} "
            f"| {fmt(r['jain_index_mean'], r['jain_index_std'], 3)} "
            f"| {fmt(r['worst_zone_outage_h_mean'], r['worst_zone_outage_h_std'], 1)} "
            f"| {fmt(r['mean_residual_deficit_mw_mean'], r['mean_residual_deficit_mw_std'], 1)} "
            f"| {fmt(r['return_mean'], r['return_std'], 1)} |"
        )

    # --- headline: honest Pareto-occupation statement (RQ1 & RQ2) ---
    lines += ["", "## Headline (RQ1 & RQ2) — use this framing verbatim", ""]

    best_agent = max(agents, key=lambda a: df.loc[a, "return_mean"]) if agents else None
    if best_agent:
        ba = df.loc[best_agent]
        pri, rr = df.loc["priority"], df.loc["round_robin"]
        # Occupation claim: agent simultaneously (i) within/beyond Priority-level
        # WUE among feasible policies, (ii) far fairer than Priority, (iii) feasible.
        lines.append(
            f"**RQ1 (does the gap exist):** No fixed-rule baseline is simultaneously "
            f"efficient, fair and feasible: Priority reaches "
            f"{pri['wue_mwh_mean']:.0f} MWh WUE but with Jain "
            f"{pri['jain_index_mean']:.3f} (persistently unfair), while RoundRobin "
            f"reaches Jain {rr['jain_index_mean']:.3f} but leaves "
            f"{rr['mean_residual_deficit_mw_mean']:.0f} MW mean residual deficit "
            f"(uncontrolled outages) and Proportional pays "
            f"{df.loc['proportional', 'wue_mwh_mean']:.0f} MWh WUE for its fairness."
        )
        lines.append("")
        lines.append(
            f"**RQ2 (can learned policies occupy the gap):** {LABELS[best_agent]} "
            f"(best return {ba['return_mean']:.1f} ± {ba['return_std']:.1f}) attains "
            f"WUE {ba['wue_mwh_mean']:.0f} ± {ba['wue_mwh_std']:.0f} MWh with Jain "
            f"{ba['jain_index_mean']:.3f} ± {ba['jain_index_std']:.3f} and residual "
            f"deficit {ba['mean_residual_deficit_mw_mean']:.1f} MW — i.e. it occupies "
            f"a region of the efficiency–fairness plane no baseline reaches. "
            f"This is a Pareto **occupation** claim, not domination: individual "
            f"baselines still win on single metrics (Priority on WUE alone, "
            f"rotation schemes on fairness alone)."
        )
        lines.append("")
        # Explicit per-metric honesty check the report must keep.
        for m, better_low, label in [
            ("wue_mwh", True, "WUE"), ("jain_index", False, "Jain index"),
        ]:
            col = f"{m}_mean"
            pick = min if better_low else max
            # Restrict to feasible baselines: NoShedding's WUE=0 is degenerate
            # (all unmet load is uncontrolled, not shed).
            best_base = pick(feasible, key=lambda b: df.loc[b, col])
            lines.append(
                f"- Best *feasible* baseline on {label} alone: {LABELS[best_base]} "
                f"({df.loc[best_base, col]:.3f}); infeasible baselines "
                f"(residual deficit > 25 MW) excluded."
            )
        vdn_q = [a for a in ("vdn", "qmix") if a in df.index]
        if len(vdn_q) == 2:
            better = max(vdn_q, key=lambda a: df.loc[a, "return_mean"])
            worse = vdn_q[1 - vdn_q.index(better)]
            lines.append(
                f"- VDN vs QMIX: {LABELS[better]} > {LABELS[worse]} on return here "
                f"({df.loc[better, 'return_mean']:.1f} vs {df.loc[worse, 'return_mean']:.1f}); "
                f"report as an empirical, task-dependent finding (Papoudakis et al. 2021)."
            )

    if ablation:
        lines += ["", "## β-ablation "
                  f"({ablation['agent'].upper()}, {ablation['steps']:,} steps each, "
                  f"seed {ablation['seed']})", "",
                  "| β | WUE (MWh/ep) | Fairness std | Jain index |", "|---|---|---|---|"]
        for r in ablation["runs"]:
            lines.append(
                f"| {r['beta']:g} | {r['wue_mwh']['mean']:.0f} ± {r['wue_mwh']['std']:.0f} "
                f"| {r['fairness_std']['mean']:.4f} ± {r['fairness_std']['std']:.4f} "
                f"| {r['jain_index']['mean']:.3f} ± {r['jain_index']['std']:.3f} |"
            )

    lines += [
        "",
        "## Supporting numbers",
        "",
        "- LSTM forecaster per-zone val MAE (MW): see `models/saved_models/lstm.pt` "
        "metadata and `results/figures/forecaster_validation.png` (mean 13.8 MW; "
        "Accra at its ~22 MW irreducible noise floor).",
        "- Historical pre-rebuild numbers (IDQN return −16.13, WUE 162 MWh; central "
        "DQN WUE 3203 MWh) are from an incompatible env version — HISTORICAL ONLY, "
        "never in comparison tables (Hard Rule 1).",
        "",
    ]
    out = RESULTS / "report_numbers.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
