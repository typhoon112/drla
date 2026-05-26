#!/usr/bin/env python3
"""Generate reproducible figures for the Cola adaptive halt paper report."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path("/data1/luyifei/drla")
SUMMARY_DIR = ROOT / "outputs" / "cola_experiment_summaries"
OUT_DIR = ROOT / "outputs" / "paper_report_20260525"
FIG_DIR = OUT_DIR / "figures"


PATHS = {
    "cross_task_pc": SUMMARY_DIR
    / "official8_full_b64_bs12_cross_task_prediction_change_risk_cross_seed_20260524"
    / "summary.json",
    "joint_riskcap04": SUMMARY_DIR
    / "official8_full_b64_bs12_cross_task_joint_readiness_riskcap04_cross_seed_20260524"
    / "summary.json",
    "shape_v2_noriskcap": SUMMARY_DIR
    / "official8_full_b64_bs12_cross_task_shape_features_fragmentguardv2_choice2_noriskcap_cross_seed_20260524"
    / "summary.json",
    "shape_v3_riskcap04": SUMMARY_DIR
    / "official8_full_b64_bs12_cross_task_shape_features_fragmentguardv3_choice2_riskcap04_cross_seed_20260524"
    / "summary.json",
}


COLORS = {
    "fixed": "#8C8C8C",
    "stability": "#0072B2",
    "pc": "#009E73",
    "ours": "#E76F51",
    "unsafe": "#D55E00",
    "diagnostic": "#CC79A7",
    "posthoc": "#E69F00",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pct(value: float) -> float:
    return 100.0 * value


def mean(values: list[float]) -> float:
    return float(np.mean(values))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linestyle": "-",
        }
    )


def cross_task_rows(summaries: dict[str, dict]) -> list[dict]:
    pc = summaries["cross_task_pc"]["aggregate"]
    ours = summaries["joint_riskcap04"]["aggregate"]
    shape_v2 = summaries["shape_v2_noriskcap"]["aggregate"]
    shape_v3 = summaries["shape_v3_riskcap04"]["aggregate"]

    posthoc_rows = summaries["shape_v2_noriskcap"]["posthoc_zero_loss_frontier_by_seed"]
    posthoc_acc = mean([row["accuracy"] for row in posthoc_rows])

    return [
        {
            "method": "Fixed final",
            "blocks": 4.0,
            "blocks_std": 0.0,
            "accuracy": pct(ours["mean_weighted_fixed_final_accuracy"]),
            "accuracy_std": pct(ours["std_weighted_fixed_final_accuracy"]),
            "losses": 0,
            "kind": "baseline",
            "color": COLORS["fixed"],
        },
        {
            "method": "Prediction stability",
            "blocks": ours["mean_weighted_prediction_stability_avg_blocks"],
            "blocks_std": ours["std_weighted_prediction_stability_avg_blocks"],
            "accuracy": pct(ours["mean_weighted_prediction_stability_accuracy"]),
            "accuracy_std": pct(ours["std_weighted_prediction_stability_accuracy"]),
            "losses": 0,
            "kind": "baseline",
            "color": COLORS["stability"],
        },
        {
            "method": "Prediction-change risk",
            "blocks": pc["mean_weighted_risk_gated_avg_blocks"],
            "blocks_std": pc["std_weighted_risk_gated_avg_blocks"],
            "accuracy": pct(pc["mean_weighted_risk_gated_accuracy"]),
            "accuracy_std": pct(pc["std_weighted_risk_gated_accuracy"]),
            "losses": pc["total_loss_count_vs_prediction_stability_all_seeds"],
            "kind": "safe",
            "color": COLORS["pc"],
        },
        {
            "method": "Joint readiness + riskcap04",
            "blocks": ours["mean_weighted_risk_gated_avg_blocks"],
            "blocks_std": ours["std_weighted_risk_gated_avg_blocks"],
            "accuracy": pct(ours["mean_weighted_risk_gated_accuracy"]),
            "accuracy_std": pct(ours["std_weighted_risk_gated_accuracy"]),
            "losses": ours["total_loss_count_vs_prediction_stability_all_seeds"],
            "kind": "ours",
            "color": COLORS["ours"],
        },
        {
            "method": "38-feature no-riskcap",
            "blocks": shape_v2["mean_weighted_risk_gated_avg_blocks"],
            "blocks_std": shape_v2["std_weighted_risk_gated_avg_blocks"],
            "accuracy": pct(shape_v2["mean_weighted_risk_gated_accuracy"]),
            "accuracy_std": pct(shape_v2["std_weighted_risk_gated_accuracy"]),
            "losses": shape_v2["total_loss_count_vs_prediction_stability_all_seeds"],
            "kind": "unsafe",
            "color": COLORS["unsafe"],
        },
        {
            "method": "38-feature post-hoc zero-loss",
            "blocks": shape_v2["posthoc_zero_loss_frontier_mean_avg_blocks"],
            "blocks_std": shape_v2["posthoc_zero_loss_frontier_std_avg_blocks"],
            "accuracy": pct(posthoc_acc),
            "accuracy_std": 0.0,
            "losses": shape_v2["posthoc_zero_loss_frontier_total_losses_vs_prediction_stability"],
            "kind": "posthoc",
            "color": COLORS["posthoc"],
        },
        {
            "method": "38-feature v3 + riskcap04",
            "blocks": shape_v3["risk_gated_avg_blocks"]["mean"],
            "blocks_std": shape_v3["risk_gated_avg_blocks"]["std"],
            "accuracy": pct(shape_v3["risk_gated_accuracy"]["mean"]),
            "accuracy_std": pct(shape_v3["risk_gated_accuracy"]["std"]),
            "losses": int(shape_v3["loss_count_vs_prediction_stability"]["mean"]),
            "kind": "diagnostic",
            "color": COLORS["diagnostic"],
        },
    ]


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)


def plot_tradeoff(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.75, 3.25))

    legend_handles = []
    for idx, row in enumerate(rows, start=1):
        marker = "D" if row["kind"] == "ours" else "o"
        facecolor = "white" if row["kind"] == "posthoc" else row["color"]
        linewidth = 2.2 if row["kind"] == "ours" else 1.2
        ax.errorbar(
            row["blocks"],
            row["accuracy"],
            xerr=row["blocks_std"],
            yerr=row["accuracy_std"],
            fmt=marker,
            color=row["color"],
            markerfacecolor=facecolor,
            markeredgewidth=linewidth,
            markersize=7 if row["kind"] == "ours" else 6,
            linewidth=1.1,
            capsize=2.5,
            zorder=4 if row["kind"] == "ours" else 3,
        )
        label_offsets = {
            1: (0.0, 0.0045),
            2: (0.012, 0.0063),
            3: (-0.012, 0.0045),
            4: (-0.012, 0.0065),
            5: (0.010, 0.0072),
            6: (0.0, 0.0068),
            7: (0.0, 0.0061),
        }
        dx, dy = label_offsets[idx]
        ax.text(
            row["blocks"] + dx,
            row["accuracy"] + dy,
            str(idx),
            fontsize=7.5,
            ha="center",
            va="bottom",
            color="#222222",
            weight="bold",
        )
        label = f"{idx}. {row['method']}"
        if row["losses"]:
            label += f" ({int(row['losses'])} losses)"
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color=row["color"],
                label=label,
                markerfacecolor=facecolor,
                markeredgewidth=linewidth,
                markersize=5.8,
                linewidth=0,
            )
        )

    ax.axvline(2.5116, color=COLORS["stability"], alpha=0.18, linewidth=1.4)
    ax.set_xlabel("Average blocks used (lower is cheaper)")
    ax.set_ylabel("Weighted micro accuracy (%)")
    ax.set_title("Cross-task safety-cost frontier")
    ax.set_xlim(1.78, 4.12)
    ax.set_ylim(21.54, 21.665)
    ax.grid(axis="y", alpha=0.2)
    ax.grid(axis="x", alpha=0.08)
    ax.legend(
        handles=legend_handles,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.27),
        columnspacing=1.2,
        handletextpad=0.35,
    )
    save_figure(fig, "fig_cross_task_tradeoff")


def plot_failure_breakdown(summary: dict) -> None:
    by_task_seed: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in summary["loss_tasks"]:
        seed = row["seed"]
        task = row["task"].replace("_", " ").title()
        by_task_seed[task][seed] += int(row["loss_count_vs_prediction_stability"])

    tasks = sorted(by_task_seed, key=lambda task: -sum(by_task_seed[task].values()))
    seeds = ["seed66", "seed67", "seed68"]
    seed_colors = {"seed66": "#0072B2", "seed67": "#E69F00", "seed68": "#009E73"}

    fig, ax = plt.subplots(figsize=(4.4, 2.65))
    x = np.arange(len(tasks))
    bottom = np.zeros(len(tasks))
    for seed in seeds:
        values = np.array([by_task_seed[task].get(seed, 0) for task in tasks])
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.62,
            label=seed,
            color=seed_colors[seed],
            edgecolor="white",
            linewidth=0.6,
        )
        bottom += values

    for idx, total in enumerate(bottom):
        ax.text(idx, total + 0.35, str(int(total)), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Losses vs prediction stability")
    ax.set_title("No-riskcap 38-feature loss breakdown")
    ax.legend(ncol=3, loc="upper right")
    ax.set_ylim(0, max(bottom) + 2.4)
    save_figure(fig, "fig_no_riskcap_loss_breakdown")


def plot_seed_blocks(summaries: dict[str, dict]) -> None:
    ours = summaries["joint_riskcap04"]
    shape = summaries["shape_v2_noriskcap"]
    seed_labels = ["seed66", "seed67", "seed68"]

    stability = []
    riskcap = []
    shape_v2 = []
    for seed_row in ours["seeds"]:
        stability.append(seed_row["weighted_prediction_stability_avg_blocks"])
        riskcap.append(seed_row["weighted_risk_gated_avg_blocks"])
    for seed_row in shape["seeds"]:
        shape_v2.append(seed_row["weighted_risk_gated_avg_blocks"])

    fig, ax = plt.subplots(figsize=(5.1, 2.65))
    x = np.arange(len(seed_labels))
    width = 0.24
    ax.bar(x - width, stability, width, label="Prediction stability", color=COLORS["stability"])
    ax.bar(x, riskcap, width, label="Joint readiness + riskcap04", color=COLORS["ours"])
    ax.bar(x + width, shape_v2, width, label="38-feature no-riskcap", color=COLORS["unsafe"])
    ax.set_xticks(x)
    ax.set_xticklabels(seed_labels)
    ax.set_ylabel("Average blocks")
    ax.set_title("Seed-level block cost")
    ax.set_ylim(1.95, 2.58)
    ax.legend(ncol=1, loc="upper right")
    save_figure(fig, "fig_seed_block_costs")


def main() -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summaries = {name: load_json(path) for name, path in PATHS.items()}
    rows = cross_task_rows(summaries)

    plot_tradeoff(rows)
    plot_failure_breakdown(summaries["shape_v2_noriskcap"])
    plot_seed_blocks(summaries)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "figure_data.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_paths": {key: str(path) for key, path in PATHS.items()},
                "cross_task_rows": rows,
                "loss_tasks": summaries["shape_v2_noriskcap"]["loss_tasks"],
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
