"""
evaluate_modality_dropout.py
────────────────────────────
Loads a trained SwinUNETR checkpoint and evaluates Dice performance
under systematic single-modality dropout (zero-out one channel at a
time) as well as the full-modality baseline. Produces a summary plot.

BraTS 2023 channel order (assumed):
  0 → T1n  (T1 native)
  1 → T1c  (T1 contrast)
  2 → T2w  (T2 weighted)
  3 → T2f  (FLAIR)

Usage:
  python evaluate_modality_dropout.py \
      --checkpoint  path/to/model.pt \
      --data_dir    path/to/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
      --json_list   ./brats23_folds.json \
      --fold        1 \
      [--roi 128 128 128] \
      [--sw_batch_size 4] \
      [--infer_overlap 0.5] \
      [--device cuda]
"""

import os
import json
import argparse
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from functools import partial
from monai import transforms, data
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import AsDiscrete, Activations
from monai.utils.enums import MetricReduction


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

MODALITY_NAMES = ["T1n", "T1c", "T2w", "T2f (FLAIR)"]
REGION_NAMES   = ["TC (Tumor Core)", "WT (Whole Tumor)", "ET (Enhancing Tumor)"]

REGION_COLORS  = ["#E07B54", "#5B9BD5", "#6ABF69"]
SCENARIO_PALETTE = {
    "All modalities": "#F0F0F0",
    "Drop T1n":       "#E07B54",
    "Drop T1c":       "#C4547A",
    "Drop T2w":       "#5B9BD5",
    "Drop T2f (FLAIR)": "#6ABF69",
}


def datafold_read(datalist_path, basedir, fold=0, key="training"):
    with open(datalist_path) as f:
        json_data = json.load(f)[key]
    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str) and len(d[k]) > 0:
                d[k] = os.path.join(basedir, d[k])
    val = [d for d in json_data if "fold" in d and d["fold"] == fold]
    return val


def build_val_loader(val_files):
    val_transform = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])
    ds = data.Dataset(data=val_files, transform=val_transform)
    return data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)


def zero_out_channel(batch_image: torch.Tensor, channel: int) -> torch.Tensor:
    """Return a copy of the image with one channel zeroed out."""
    img = batch_image.clone()
    img[:, channel, ...] = 0.0
    return img


def run_evaluation(model, loader, device, model_inferer,
                   post_sigmoid, post_pred, acc_func,
                   drop_channel=None):
    """
    Evaluate the model on `loader`.
    If drop_channel is an int, zero out that modality index before inference.
    Returns np.array of shape (3,) — mean Dice for [TC, WT, ET].
    """
    model.eval()
    run_acc_list = []          # collect per-sample averages
    acc_func.reset()

    with torch.no_grad():
        for batch_data in loader:
            images = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            if drop_channel is not None:
                images = zero_out_channel(images, drop_channel)

            logits = model_inferer(images)

            val_labels_list  = decollate_batch(labels)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [
                post_pred(post_sigmoid(v)) for v in val_outputs_list
            ]

            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            run_acc_list.append(acc.cpu().numpy())   # shape (3,)

    # Mean across all samples (simple mean; not_nans weighting omitted for clarity)
    return np.nanmean(np.stack(run_acc_list, axis=0), axis=0)


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def build_plot(results: dict, save_path: str):
    """
    results: {scenario_label: np.array([dice_tc, dice_wt, dice_et])}
    """
    scenarios = list(results.keys())
    n_scenarios = len(scenarios)
    n_regions   = len(REGION_NAMES)

    x = np.arange(n_regions)
    bar_width = 0.14
    offsets = np.linspace(
        -(n_scenarios - 1) * bar_width / 2,
         (n_scenarios - 1) * bar_width / 2,
        n_scenarios,
    )

    # ── dark, clinical aesthetic ──────────────────────────────────
    BG    = "#0F1117"
    PANEL = "#181C27"
    GRID  = "#2A2F42"
    TEXT  = "#E8EAF0"
    ACCENT= "#4FC3F7"

    fig, axes = plt.subplots(
        1, 2,
        figsize=(16, 7),
        facecolor=BG,
        gridspec_kw={"width_ratios": [2, 1]},
    )
    fig.subplots_adjust(wspace=0.35)

    scenario_colors = [
        "#F0F0F0",   # All modalities — near white
        "#E07B54",   # Drop T1n
        "#C4547A",   # Drop T1c
        "#5B9BD5",   # Drop T2w
        "#6ABF69",   # Drop FLAIR
    ]

    # ── Left: grouped bar chart ──────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(PANEL)
    ax.spines[:].set_color(GRID)

    for si, (scenario, color) in enumerate(zip(scenarios, scenario_colors)):
        vals = results[scenario]
        bars = ax.bar(
            x + offsets[si], vals,
            width=bar_width * 0.88,
            color=color,
            alpha=0.88,
            label=scenario,
            zorder=3,
        )
        # value labels
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.006,
                    f"{h:.3f}",
                    ha="center", va="bottom",
                    fontsize=6.5, color=TEXT, rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(REGION_NAMES, color=TEXT, fontsize=11)
    ax.set_yticks(np.arange(0, 1.05, 0.1))
    ax.set_ylim(0, 1.12)
    ax.tick_params(colors=TEXT)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_xlabel("Segmentation Region", color=TEXT, fontsize=12, labelpad=8)
    ax.set_ylabel("Mean Dice Score", color=TEXT, fontsize=12, labelpad=8)
    ax.set_title(
        "Dice Performance — Modality Dropout Comparison",
        color=TEXT, fontsize=14, fontweight="bold", pad=14,
    )
    ax.legend(
        loc="upper right",
        framealpha=0.18,
        facecolor=PANEL,
        edgecolor=GRID,
        labelcolor=TEXT,
        fontsize=9,
    )

    # ── Right: delta heatmap vs baseline ─────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(PANEL)
    ax2.spines[:].set_color(GRID)

    baseline = results[scenarios[0]]
    drop_labels  = scenarios[1:]
    delta_matrix = np.array([
        results[s] - baseline for s in drop_labels
    ])  # shape (4, 3)

    vmax = np.abs(delta_matrix).max() or 0.01
    im = ax2.imshow(
        delta_matrix,
        cmap="RdYlGn",
        vmin=-vmax, vmax=vmax,
        aspect="auto",
    )

    for r in range(len(drop_labels)):
        for c in range(n_regions):
            v = delta_matrix[r, c]
            ax2.text(
                c, r, f"{v:+.3f}",
                ha="center", va="center",
                fontsize=9,
                color="white" if abs(v) > vmax * 0.55 else "#111",
                fontweight="bold",
            )

    ax2.set_xticks(range(n_regions))
    ax2.set_xticklabels(["TC", "WT", "ET"], color=TEXT, fontsize=10)
    ax2.set_yticks(range(len(drop_labels)))
    ax2.set_yticklabels(drop_labels, color=TEXT, fontsize=9)
    ax2.tick_params(colors=TEXT)
    ax2.set_title(
        "Δ Dice vs. Full Modality",
        color=TEXT, fontsize=13, fontweight="bold", pad=12,
    )

    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=TEXT)
    cbar.ax.yaxis.label.set_color(TEXT)

    fig.text(
        0.5, 0.01,
        "Green = better, Red = worse (relative to all-modality baseline)",
        ha="center", color=TEXT, fontsize=8.5, alpha=0.7,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\n[✓] Plot saved → {save_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Modality dropout evaluation for SwinUNETR BraTS.")
    p.add_argument("--checkpoint",    required=True,  help="Path to model .pt checkpoint")
    p.add_argument("--data_dir",      required=True,  help="BraTS 2023 training data root")
    p.add_argument("--json_list",     required=True,  help="brats23_folds.json path")
    p.add_argument("--fold",          type=int, default=1)
    p.add_argument("--roi",           type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--sw_batch_size", type=int, default=4)
    p.add_argument("--infer_overlap", type=float, default=0.5)
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_plot",   default="modality_dropout_results.png")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    roi    = tuple(args.roi)

    # ── Data ──────────────────────────────────
    print("[*] Loading validation split ...")
    val_files  = datafold_read(args.json_list, args.data_dir, fold=args.fold)
    val_loader = build_val_loader(val_files)
    print(f"    {len(val_files)} validation cases")

    # ── Model ─────────────────────────────────
    print("[*] Building model ...")
    model = SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"    Loaded checkpoint (epoch {checkpoint.get('epoch','?')}, "
          f"best_acc={checkpoint.get('best_acc', '?'):.4f})")

    model_inferer = partial(
        sliding_window_inference,
        roi_size=list(roi),
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    post_sigmoid = Activations(sigmoid=True)
    post_pred    = AsDiscrete(argmax=False, threshold=0.5)
    dice_acc     = DiceMetric(
        include_background=True,
        reduction=MetricReduction.MEAN_BATCH,
        get_not_nans=True,
    )

    # ── Evaluation scenarios ───────────────────
    scenarios = [
        ("All modalities", None),
        ("Drop T1n",        0),
        ("Drop T1c",        1),
        ("Drop T2w",        2),
        ("Drop T2f (FLAIR)",3),
    ]

    results = {}
    for label, drop_ch in scenarios:
        print(f"\n[*] Evaluating: {label} ...")
        t0 = time.time()
        avg_dice = run_evaluation(
            model, val_loader, device, model_inferer,
            post_sigmoid, post_pred, dice_acc,
            drop_channel=drop_ch,
        )
        elapsed = time.time() - t0
        results[label] = avg_dice
        print(
            f"    TC={avg_dice[0]:.4f}  WT={avg_dice[1]:.4f}  ET={avg_dice[2]:.4f}"
            f"  ({elapsed:.1f}s)"
        )

    # ── Summary table ─────────────────────────
    print("\n" + "=" * 62)
    print(f"{'Scenario':<22} {'TC':>8} {'WT':>8} {'ET':>8} {'Mean':>8}")
    print("-" * 62)
    baseline = results["All modalities"]
    for label, vals in results.items():
        mean = vals.mean()
        marker = ""
        if label != "All modalities":
            delta = mean - baseline.mean()
            marker = f"  ({delta:+.4f})"
        print(f"{label:<22} {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f} {mean:>8.4f}{marker}")
    print("=" * 62)

    # ── Plot ──────────────────────────────────
    build_plot(results, args.output_plot)


if __name__ == "__main__":
    main()