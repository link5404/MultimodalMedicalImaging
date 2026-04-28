import os
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REGION_NAMES = ["TC", "WT", "ET"]

SCENARIOS_ORDER = [
    "All modalities",
    "Drop T1n",
    "Drop T1c",
    "Drop T2w",
    "Drop T2f (FLAIR)",
]

BG = "#0F1117"
PANEL = "#181C27"
GRID = "#2A2F42"
TEXT = "#E8EAF0"


def aggregate_case_cache(case_cache: dict) -> dict:
    acc = {s: [] for s in SCENARIOS_ORDER}

    for case_results in case_cache.values():
        for scenario, vals in case_results.items():
            if scenario in acc:
                acc[scenario].append(vals)

    return {
        s: np.nanmean(np.array(vals), axis=0)
        for s, vals in acc.items()
        if vals
    }


def load_all_caches(results_dir: str) -> dict:
    models = {}

    for p in sorted(Path(results_dir).glob("*_case_cache.json")):
        stem = p.stem.replace("_case_cache", "")

        with open(p) as f:
            case_cache = json.load(f)

        if case_cache:
            models[stem] = aggregate_case_cache(case_cache)

    return models


def short_label(stem: str) -> str:
    return stem.replace("model_checkpoint_", "").replace("model_checkpoint", "baseline")


def model_sort_key(name: str):
    short = short_label(name)
    if short.startswith("baseline"):
        group = 0
    elif "FiLM_DEEP" in short:
        group = 2
    elif "FiLM" in short:
        group = 1
    else:
        group = 3
    return (group, short)


def print_summary_table(models: dict):
    scenarios = SCENARIOS_ORDER

    print("\n" + "=" * 112)
    print(f"{'Model':<35} {'Scenario':<22} {'TC':>7} {'WT':>7} {'ET':>7} {'Mean':>8}")
    print("-" * 112)

    for name in sorted(models.keys(), key=model_sort_key):
        results = models[name]
        for scenario in scenarios:
            if scenario not in results:
                continue
            vals = results[scenario]
            mean_all = np.nanmean(vals)
            print(
                f"{short_label(name):<35} "
                f"{scenario:<22} "
                f"{vals[0]:>7.4f} "
                f"{vals[1]:>7.4f} "
                f"{vals[2]:>7.4f} "
                f"{mean_all:>8.4f}"
            )
        print()

    print("=" * 112)


def build_comparison(models: dict, output_path: str):
    if not models:
        print("[!] No case-cache files found.")
        return

    names = sorted(models.keys(), key=model_sort_key)
    scenarios = [s for s in SCENARIOS_ORDER if any(s in models[n] for n in names)]

    n_models = len(names)
    cmap = plt.cm.get_cmap("tab20", n_models)
    colors = {name: cmap(i) for i, name in enumerate(names)}

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(29, 7),
        facecolor=BG,
        gridspec_kw={"width_ratios": [2, 2, 2, 1.6]},
    )
    fig.subplots_adjust(wspace=0.35)

    x_ticks = range(len(scenarios))

    for region_idx, region_name in enumerate(REGION_NAMES):
        ax = axes[region_idx]
        ax.set_facecolor(PANEL)
        ax.spines[:].set_color(GRID)

        for name in names:
            results = models[name]
            values = [
                results.get(s, np.full(3, np.nan))[region_idx]
                for s in scenarios
            ]

            label = short_label(name)
            is_baseline = label.startswith("baseline")
            lw = 2.4 if is_baseline else 1.4
            ls = "--" if is_baseline else "-"

            ax.plot(
                x_ticks,
                values,
                color=colors[name],
                linewidth=lw,
                linestyle=ls,
                marker="o",
                markersize=4.5,
                label=label,
            )

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(
            [s.replace("Drop ", "−") for s in scenarios],
            color=TEXT,
            fontsize=9,
            rotation=15,
            ha="right",
        )
        ax.set_ylim(0, 1.05)
        ax.tick_params(colors=TEXT)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
        ax.set_ylabel("Mean Dice Score", color=TEXT, fontsize=11)
        ax.set_title(
            f"{region_name} Dice — Checkpoint Comparison",
            color=TEXT,
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax.legend(
            loc="lower left",
            fontsize=6.5,
            framealpha=0.2,
            facecolor=PANEL,
            edgecolor=GRID,
            labelcolor=TEXT,
            ncol=2,
        )

    ax4 = axes[3]
    ax4.set_facecolor(PANEL)
    ax4.spines[:].set_color(GRID)

    rows = []
    row_labels = []

    for name in names:
        results = models[name]
        all_modalities = results.get("All modalities", np.full(3, np.nan))
        rows.append(all_modalities)
        row_labels.append(short_label(name))

    matrix = np.array(rows)
    im = ax4.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")

    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            v = matrix[r, c]
            ax4.text(
                c,
                r,
                f"{v:.3f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if v < 0.5 else "#111",
                fontweight="bold",
            )

    ax4.set_xticks([0, 1, 2])
    ax4.set_xticklabels(REGION_NAMES, color=TEXT, fontsize=11)
    ax4.set_yticks(range(len(row_labels)))
    ax4.set_yticklabels(row_labels, color=TEXT, fontsize=7)
    ax4.tick_params(colors=TEXT)
    ax4.set_title(
        "All-Modalities Dice",
        color=TEXT,
        fontsize=11,
        fontweight="bold",
        pad=12,
    )

    cbar = fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=TEXT)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"[✓] Comparison plot saved → {output_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--output", default="results/checkpoint_comparison.png")
    return p.parse_args()


def main():
    args = parse_args()

    print("[*] Loading checkpoint caches ...")
    models = load_all_caches(args.results_dir)
    print(f"    Found {len(models)} checkpoint caches: {list(models.keys())}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print_summary_table(models)
    build_comparison(models, args.output)


if __name__ == "__main__":
    main()
