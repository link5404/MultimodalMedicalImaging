import os
import json
import argparse
import time
import importlib.util
from functools import partial
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from monai import transforms, data
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import AsDiscrete, Activations, MapTransform
from monai.utils.enums import MetricReduction


REGION_NAMES = ["TC (Tumor Core)", "WT (Whole Tumor)", "ET (Enhancing Tumor)"]
NUM_CLASSES = 3

SCENARIOS = [
    ("All modalities", None),
    ("Drop T1n", 0),
    ("Drop T1c", 1),
    ("Drop T2w", 2),
    ("Drop T2f (FLAIR)", 3),
]

SCENARIO_COLORS = ["#F0F0F0", "#E07B54", "#C4547A", "#5B9BD5", "#6ABF69"]


class ConvertBraTS2023Labelsd(MapTransform):
    def __call__(self, data_dict):
        d = dict(data_dict)
        for key in self.keys:
            label = d[key]
            result = torch.stack([
                (label == 1) | (label == 3),
                (label == 1) | (label == 2) | (label == 3),
                (label == 3),
            ], dim=0).float()
            d[key] = result
        return d


def _load_class_from_file(filepath: str, classname: str):
    spec = importlib.util.spec_from_file_location(
        f"_film_mod_{Path(filepath).stem}", filepath
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, classname)


def model_kind(checkpoint_path: str) -> str:
    name = Path(checkpoint_path).name
    if "DEEP" in name:
        return "deep_film"
    if "FiLM" in name:
        return "shallow_film"
    return "baseline"


def build_model(checkpoint_path: str, scripts_dir: str, device: torch.device):
    kind = model_kind(checkpoint_path)

    base = SwinUNETR(
        in_channels=4,
        out_channels=NUM_CLASSES,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
    ).to(device)

    if kind == "deep_film":
        model_file = os.path.join(scripts_dir, "deepFilm.py")
        ModelClass = _load_class_from_file(model_file, "SwinUNETRWithFiLM")
        model = ModelClass(base, cond_dim=128).to(device)
        label = "DeepFiLM"
    elif kind == "shallow_film":
        model_file = os.path.join(scripts_dir, "models.py")
        ModelClass = _load_class_from_file(model_file, "SwinUNETRWithFiLM")
        model = ModelClass(base, cond_dim=128).to(device)
        label = "ShallowFiLM"
    else:
        model = base
        label = "BaselineSwinUNETR"

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # Strict loading is preferred, but strict=False gives useful diagnostics instead of a hard crash.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
    best_acc = ckpt.get("best_acc", float("nan")) if isinstance(ckpt, dict) else float("nan")

    print(f"    [{label}] Loaded epoch={epoch}, best_acc={best_acc:.4f}")
    print(f"    Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"    Missing examples: {missing[:5]}")
    if unexpected:
        print(f"    Unexpected examples: {unexpected[:5]}")
    if missing or unexpected:
        print("    [WARN] State dict mismatch detected. Confirm the checkpoint type and model class.")

    model.eval()
    return model, label, kind


def make_predictor(model, kind: str):
    if kind == "shallow_film":
        def predictor(x):
            cond = torch.zeros(x.shape[0], 128, device=x.device)
            return model(x, cond)
        return predictor

    def predictor(x):
        return model(x)

    return predictor


def datafold_read(datalist_path, basedir, fold=0, key="training"):
    with open(datalist_path) as f:
        json_data = json.load(f)[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str) and len(d[k]) > 0:
                d[k] = os.path.join(basedir, d[k])

    return [d for d in json_data if "fold" in d and d["fold"] == fold]


def build_val_loader(val_files, cache_rate=1.0, num_workers=4):
    """
    Match train.py validation preprocessing exactly:
      LoadImaged -> ConvertBraTS2023Labelsd -> NormalizeIntensityd

    Note:
      cache_rate is kept in the function signature so the rest of
      evaluate_checkpoint.py does not need to change, but it is not used.
    """
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            ConvertBraTS2023Labelsd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    val_ds = data.Dataset(
        data=val_files,
        transform=val_transform,
    )

    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    return val_loader


def evaluate_case(model_inferer, image, label, post_sigmoid, post_pred, dice_acc):
    case_results = {}

    for scenario_label, drop_ch in SCENARIOS:
        img = image.clone()

        if drop_ch is not None:
            img[:, drop_ch, ...] = 0.0

        with torch.no_grad():
            logits = model_inferer(img)

        labels_list = decollate_batch(label)
        outputs_list = decollate_batch(logits)
        preds = [post_pred(post_sigmoid(v)) for v in outputs_list]

        dice_acc.reset()
        dice_acc(y_pred=preds, y=labels_list)
        acc, _ = dice_acc.aggregate()

        case_results[scenario_label] = acc.cpu().numpy().tolist()

    return case_results


def cache_path_for(checkpoint_path: str, output_dir: str) -> str:
    stem = Path(checkpoint_path).stem
    return os.path.join(output_dir, f"{stem}_case_cache.json")


def load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(path: str, cache: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=2)
    os.replace(tmp, path)


def aggregate(case_cache: dict) -> dict:
    acc = {label: [] for label, _ in SCENARIOS}

    for case_results in case_cache.values():
        for label, vals in case_results.items():
            if label in acc:
                acc[label].append(vals)

    return {
        label: np.nanmean(np.array(vals), axis=0)
        for label, vals in acc.items()
        if vals
    }


def load_baseline_for_plot(path: str):
    if not os.path.exists(path):
        return None

    with open(path) as f:
        raw = json.load(f)

    if not raw:
        return None

    first_val = next(iter(raw.values()))
    if isinstance(first_val, list) and len(first_val) == 3:
        return {k: np.array(v) for k, v in raw.items()}

    return aggregate(raw)


def build_plot(results: dict, model_label: str, checkpoint_name: str,
               baseline_cache_path: str, save_path: str):
    scenarios = list(results.keys())
    n_scenarios = len(scenarios)
    n_regions = len(REGION_NAMES)

    x = np.arange(n_regions)
    bar_width = 0.13
    offsets = np.linspace(
        -(n_scenarios - 1) * bar_width / 2,
        (n_scenarios - 1) * bar_width / 2,
        n_scenarios,
    )

    BG = "#0F1117"
    PANEL = "#181C27"
    GRID = "#2A2F42"
    TEXT = "#E8EAF0"

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(20, 7),
        facecolor=BG,
        gridspec_kw={"width_ratios": [2.2, 1]},
    )
    fig.subplots_adjust(wspace=0.35)

    ax = axes[0]
    ax.set_facecolor(PANEL)
    ax.spines[:].set_color(GRID)

    bl_vals = load_baseline_for_plot(baseline_cache_path)
    if bl_vals:
        for si, (scenario, color) in enumerate(zip(scenarios, SCENARIO_COLORS)):
            if scenario in bl_vals:
                ax.bar(
                    x + offsets[si],
                    bl_vals[scenario][:n_regions],
                    width=bar_width * 0.88,
                    color=color,
                    alpha=0.25,
                    zorder=2,
                )

    for si, (scenario, color) in enumerate(zip(scenarios, SCENARIO_COLORS)):
        vals = results[scenario][:n_regions]
        bars = ax.bar(
            x + offsets[si],
            vals,
            width=bar_width * 0.88,
            color=color,
            alpha=0.88,
            label=scenario,
            zorder=3,
        )

        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.006,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color=TEXT,
                    rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(REGION_NAMES, color=TEXT, fontsize=10)
    ax.set_yticks(np.arange(0, 1.05, 0.1))
    ax.set_ylim(0, 1.12)
    ax.tick_params(colors=TEXT)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_xlabel("Output Channel", color=TEXT, fontsize=12, labelpad=8)
    ax.set_ylabel("Mean Dice Score", color=TEXT, fontsize=12, labelpad=8)
    ax.set_title(
        f"Modality Dropout — {model_label}\n{checkpoint_name}",
        color=TEXT,
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.legend(
        loc="upper right",
        framealpha=0.18,
        facecolor=PANEL,
        edgecolor=GRID,
        labelcolor=TEXT,
        fontsize=9,
    )

    ax2 = axes[1]
    ax2.set_facecolor(PANEL)
    ax2.spines[:].set_color(GRID)

    baseline = results[scenarios[0]]
    drop_labels = scenarios[1:]
    delta_matrix = np.array([
        results[s][:n_regions] - baseline[:n_regions]
        for s in drop_labels
    ])

    vmax = np.abs(delta_matrix).max() or 0.01
    im = ax2.imshow(delta_matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    for r in range(len(drop_labels)):
        for c in range(n_regions):
            v = delta_matrix[r, c]
            ax2.text(
                c,
                r,
                f"{v:+.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(v) > vmax * 0.55 else "#111",
                fontweight="bold",
            )

    ax2.set_xticks(range(n_regions))
    ax2.set_xticklabels(["TC", "WT", "ET"], color=TEXT, fontsize=10)
    ax2.set_yticks(range(len(drop_labels)))
    ax2.set_yticklabels(drop_labels, color=TEXT, fontsize=9)
    ax2.tick_params(colors=TEXT)
    ax2.set_title(
        "Δ Dice vs. Full Modality\n(this checkpoint)",
        color=TEXT,
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=TEXT)

    fig.text(
        0.5,
        0.01,
        "Faded bars = baseline reference  |  Solid bars = current checkpoint  |  Green = better, Red = worse",
        ha="center",
        color=TEXT,
        fontsize=8,
        alpha=0.7,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\n[✓] Plot saved → {save_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate one baseline, shallow FiLM, or deep FiLM checkpoint under modality dropout."
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--json_list", required=True)
    p.add_argument("--baseline_cache", default="dropout_results_cache.json")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--roi", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--sw_batch_size", type=int, default=4)
    p.add_argument("--infer_overlap", type=float, default=0.5)
    p.add_argument("--cache_rate", type=float, default=1.0)
    p.add_argument("--output_dir", default="results")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--fast", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    roi = tuple(args.roi)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.fast:
        args.infer_overlap = 0.25
        args.sw_batch_size = 8
        print("[!] --fast mode: overlap=0.25, sw_batch_size=8 (not for final numbers)\n")

    checkpoint_abs = os.path.abspath(args.checkpoint)
    scripts_dir = os.path.normpath(
        os.path.join(os.path.dirname(checkpoint_abs), "..", "scripts")
    )

    print("[*] Reading fold list ...")
    val_files = datafold_read(args.json_list, args.data_dir, fold=args.fold)
    n_cases = len(val_files)
    print(f"    {n_cases} cases in fold {args.fold}")

    print("[*] Building validation Dataset using train.py validation transform ...")
    t0 = time.time()
    val_loader = build_val_loader(val_files, cache_rate=args.cache_rate)
    print(f"    DataLoader ready in {time.time() - t0:.1f}s\n")

    print(f"[*] Loading checkpoint: {Path(args.checkpoint).name}")
    model, label, kind = build_model(args.checkpoint, scripts_dir, device)

    predictor = make_predictor(model, kind)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=list(roi),
        sw_batch_size=args.sw_batch_size,
        predictor=predictor,
        overlap=args.infer_overlap,
    )

    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=args.threshold)
    dice_acc = DiceMetric(
        include_background=True,
        reduction=MetricReduction.MEAN_BATCH,
        get_not_nans=True,
    )

    cache_file = cache_path_for(args.checkpoint, args.output_dir)
    case_cache = load_cache(cache_file)

    if case_cache:
        print(f"[*] Resuming — {len(case_cache)}/{n_cases} cases already cached\n")

    print(f"[*] Evaluating [{label}] — {n_cases} cases × {len(SCENARIOS)} scenarios")
    print(f"    threshold={args.threshold} overlap={args.infer_overlap} sw_batch_size={args.sw_batch_size}\n")

    t_start = time.time()
    completed = 0

    for case_idx, batch_data in enumerate(val_loader):
        case_id = str(case_idx)

        if case_id in case_cache:
            print(f"    [{case_idx + 1:>4}/{n_cases}] skipped (cached)")
            completed += 1
            continue

        images = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        if images.ndim != 5 or images.shape[1] != 4:
            case_name = Path(val_files[case_idx]["image"][0]).parent.name
            raise ValueError(
                f"Bad image tensor for case {case_idx} ({case_name}): "
                f"expected [B,4,H,W,D], got {tuple(images.shape)}"
            )

        t_case = time.time()
        case_res = evaluate_case(
            model_inferer,
            images,
            labels,
            post_sigmoid,
            post_pred,
            dice_acc,
        )
        elapsed = time.time() - t_case

        case_cache[case_id] = case_res
        save_cache(cache_file, case_cache)

        completed += 1
        done_time = time.time() - t_start
        eta_sec = (done_time / completed) * max(n_cases - completed, 0)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec))

        all_dice = case_res["All modalities"]
        print(
            f"    [{case_idx + 1:>4}/{n_cases}] "
            f"TC={all_dice[0]:.4f} WT={all_dice[1]:.4f} ET={all_dice[2]:.4f} "
            f"({elapsed:.1f}s/case ETA {eta_str})"
        )

    results = aggregate(case_cache)

    col_w = 8
    header = f"{'Scenario':<22}" + "".join(f"{n:>{col_w}}" for n in ["TC", "WT", "ET"]) + f"{'Mean':>{col_w}}"
    print("\n" + "=" * len(header))
    print(f"  {label} | {Path(args.checkpoint).name}")
    print(header)
    print("-" * len(header))

    baseline_vals = results["All modalities"]
    for scenario_label, vals in results.items():
        mean = np.nanmean(vals)
        marker = ""
        if scenario_label != "All modalities":
            delta = mean - np.nanmean(baseline_vals)
            marker = f"  ({delta:+.4f})"
        print(
            f"{scenario_label:<22}"
            + "".join(f"{v:>{col_w}.4f}" for v in vals)
            + f"{mean:>{col_w}.4f}{marker}"
        )

    print("=" * len(header))
    print(f"\n[✓] Total wall time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - t_start))}")

    plot_path = os.path.join(args.output_dir, Path(args.checkpoint).stem + "_dropout.png")
    build_plot(results, label, Path(args.checkpoint).name, args.baseline_cache, plot_path)


if __name__ == "__main__":
    main()
