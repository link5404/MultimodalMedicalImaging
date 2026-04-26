import os
import json
import argparse
import torch

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    ToTensord,
    ConcatItemsd,
    ConvertToMultiChannelBasedOnBratsClassesd,
)
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR


# -------------------------
# YOUR datafold_read (unchanged)
# -------------------------
def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


# -------------------------
# Modality ablation
# -------------------------
class ZeroModalityd:
    def __init__(self, keys, modality_idx):
        self.keys = keys
        self.modality_idx = modality_idx

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]  # [4, H, W, D]
            img[self.modality_idx] = 0.0
            d[key] = img
        return d


# -------------------------
# DataLoader
# -------------------------
def get_loader(data_list, modality_idx=None):
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image"]),

        # stack 4 modalities → [4, H, W, D]
        ConcatItemsd(keys=["image"], name="image", dim=0),

        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),

        # IMPORTANT: match training script
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
    ]

    if modality_idx is not None:
        transforms.append(ZeroModalityd(keys=["image"], modality_idx=modality_idx))

    transforms.append(ToTensord(keys=["image", "label"]))

    dataset = Dataset(data=data_list, transform=Compose(transforms))
    loader = DataLoader(dataset, batch_size=1, num_workers=2)

    return loader


# -------------------------
# Model
# -------------------------
def load_model(checkpoint, device):
    model = SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=48,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # 👇 THIS LINE FIXES EVERYTHING
    state_dict = checkpoint["state_dict"]
    print(f"{checkpoint['epoch']} epochs, best val dice: {checkpoint['best_acc']:.4f}")
    model.load_state_dict(state_dict)
    model.eval()

    return model


# -------------------------
# Evaluation
# -------------------------
def evaluate(model, loader, device):
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = sliding_window_inference(
                images,
                roi_size=(128, 128, 128),
                sw_batch_size=4,
                predictor=model,
            )

            outputs = torch.softmax(outputs, dim=1)
            outputs = (outputs > 0.5).float()

            dice_metric(y_pred=outputs, y=labels)

    dice = dice_metric.aggregate().item()
    dice_metric.reset()

    return dice


# -------------------------
# Main
# -------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_files, val_files = datafold_read(
        args.json,
        args.data_root,
        fold=args.fold,
        key="training",
    )

    print(f"Validation cases: {len(val_files)}")

    model = load_model(args.checkpoint, device)

    modality_names = ["T1n", "T1c", "T2w", "T2f"]
    results = {}

    # ---- baseline ----
    print("\nBaseline (no ablation)")
    loader = get_loader(val_files, modality_idx=None)
    results["baseline"] = evaluate(model, loader, device)
    print(f"Dice: {results['baseline']:.4f}")

    # ---- ablations ----
    for i, name in enumerate(modality_names):
        print(f"\nDropping {name}")
        loader = get_loader(val_files, modality_idx=i)
        results[name] = evaluate(model, loader, device)
        print(f"Dice: {results[name]:.4f}")

    # ---- summary ----
    print("\n===== FINAL RESULTS =====")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--json", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", type=int, default=0)

    args = parser.parse_args()
    main(args)