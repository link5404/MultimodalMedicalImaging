"""
train_adain_attention.py

Trains a SwinUNETR + ModalityAwareFusion (AdaIN + cross-modality attention)
stack for peak BraTS segmentation performance.  No modality dropout — all
four channels are always present, so the fusion module focuses entirely on
learning to reweight and recalibrate modalities for better segmentation.

Training strategy:
  - Phase 1 (freeze_epochs): fusion head only — AdaIN, cross-attention,
    proj, norm.  Lets the new parameters stabilise before touching the
    pretrained backbone.
  - Phase 2 (remaining epochs): partial backbone unfreeze (swinViT,
    encoder1, encoder2) at a much lower LR alongside the fusion head.
"""

import os
import json
import time
import argparse
from functools import partial

import numpy as np
import torch
from torch.amp import autocast, GradScaler

from monai import transforms, data
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import AsDiscrete, Activations
from monai.utils.enums import MetricReduction
from tqdm.auto import tqdm

from models import RobustBraTSModel


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def datafold_read(datalist_path, basedir, fold=0, key="training"):
    with open(datalist_path) as f:
        json_data = json.load(f)[key]
    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str) and len(d[k]) > 0:
                d[k] = os.path.join(basedir, d[k])
    train = [d for d in json_data if "fold" in d and d["fold"] != fold]
    val   = [d for d in json_data if "fold" in d and d["fold"] == fold]
    return train, val


def build_train_transform(roi):
    return transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        transforms.Spacingd(keys=["image", "label"],
                            pixdim=(1.0, 1.0, 1.0),
                            mode=("bilinear", "nearest")),
        transforms.NormalizeIntensityd(keys="image", nonzero=True,
                                       channel_wise=True),
        transforms.RandSpatialCropd(keys=["image", "label"],
                                    roi_size=roi, random_size=False),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5,
                             spatial_axis=0),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5,
                             spatial_axis=1),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5,
                             spatial_axis=2),
        transforms.RandScaleIntensityd(keys=["image"], factors=0.1,
                                       prob=0.3),
        transforms.RandShiftIntensityd(keys=["image"], offsets=0.1,
                                       prob=0.3),
    ])


def build_val_transform():
    return transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        transforms.Spacingd(keys=["image", "label"],
                            pixdim=(1.0, 1.0, 1.0),
                            mode=("bilinear", "nearest")),
        transforms.NormalizeIntensityd(keys="image", nonzero=True,
                                       channel_wise=True),
    ])


def build_loader(files, train=True, roi=(128, 128, 128), batch_size=1):
    tfm = build_train_transform(roi) if train else build_val_transform()
    ds  = data.Dataset(data=files, transform=tfm)
    return data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def build_model(device, checkpoint_path=None):
    backbone = SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
    ).to(device)

    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device,
                              weights_only=True)
        except Exception:
            ckpt = torch.load(checkpoint_path, map_location=device,
                              weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        backbone.load_state_dict(state, strict=True)

    return RobustBraTSModel(backbone).to(device)


def freeze_for_warmup(model: RobustBraTSModel):
    """Phase 1: train only AdaIN + cross-attention; freeze everything else."""
    for p in model.backbone.parameters():
        p.requires_grad = False
    # Also freeze the depthwise proj and norm in fusion during warmup so the
    # backbone-touching parts remain frozen until phase 2.
    for p in model.fusion.proj.parameters():
        p.requires_grad = False
    for p in model.fusion.norm.parameters():
        p.requires_grad = False
    # Trainable fusion heads
    for p in model.fusion.adain.parameters():
        p.requires_grad = True
    for p in model.fusion.xattn.parameters():
        p.requires_grad = True


def unfreeze_for_finetune(model: RobustBraTSModel, unfreeze_all: bool = False):
    """Phase 2: unfreeze partial/full backbone + fusion proj/norm."""
    if unfreeze_all:
        for p in model.backbone.parameters():
            p.requires_grad = True
    else:
        for n, p in model.backbone.named_parameters():
            if n.startswith(("swinViT", "encoder1", "encoder2")):
                p.requires_grad = True

    # Always unfreeze the fusion proj/norm in phase 2
    for p in model.fusion.proj.parameters():
        p.requires_grad = True
    for p in model.fusion.norm.parameters():
        p.requires_grad = True


def build_optimizer(model: RobustBraTSModel,
                    lr_fusion: float = 1e-3,
                    lr_backbone: float = 1e-5,
                    weight_decay: float = 1e-5):
    fusion_params   = (list(model.fusion.adain.parameters()) +
                       list(model.fusion.xattn.parameters()) +
                       list(model.fusion.proj.parameters()) +
                       list(model.fusion.norm.parameters()))
    backbone_params = [p for p in model.backbone.parameters()
                       if p.requires_grad]
    param_groups = [{"params": fusion_params, "lr": lr_fusion}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def evaluate(model, loader, device, roi, sw_batch_size, infer_overlap):
    model.eval()
    # FIX 3: single DiceMetric instance; accumulate across all batches,
    # aggregate once at the end — not per-batch.
    dice_metric  = DiceMetric(include_background=True,
                               reduction=MetricReduction.MEAN_BATCH,
                               get_not_nans=True)
    post_sigmoid = Activations(sigmoid=True)
    post_pred    = AsDiscrete(argmax=False, threshold=0.5)
    inferer      = partial(
        sliding_window_inference,
        roi_size=list(roi),
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=infer_overlap,
    )

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = inferer(images)
            y_pred = [post_pred(post_sigmoid(v))
                      for v in decollate_batch(logits)]
            y_true = decollate_batch(labels)
            dice_metric(y_pred=y_pred, y=y_true)   # accumulate; no reset here

    acc, _ = dice_metric.aggregate()               # aggregate once over all batches
    dice_metric.reset()
    return acc.cpu().numpy()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     default=None)
    p.add_argument("--data_dir",       required=True)
    p.add_argument("--json_list",      required=True)
    p.add_argument("--fold",           type=int,   default=1)
    p.add_argument("--roi",            type=int,   nargs=3,
                   default=[128, 128, 128])
    p.add_argument("--batch_size",     type=int,   default=1)
    p.add_argument("--val_batch_size", type=int,   default=1)
    p.add_argument("--epochs",         type=int,   default=80)
    p.add_argument("--freeze_epochs",  type=int,   default=10)
    p.add_argument("--lr_fusion",      type=float, default=1e-3)
    p.add_argument("--lr_backbone",    type=float, default=1e-5)
    p.add_argument("--weight_decay",   type=float, default=1e-5)
    p.add_argument("--sw_batch_size",  type=int,   default=4)
    p.add_argument("--infer_overlap",  type=float, default=0.5)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir",   default="output_adain_attention")
    p.add_argument("--save_name", default="best_adain_attention.pt")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    roi    = tuple(args.roi)

    print("Building Data Loaders")
    train_files, val_files = datafold_read(args.json_list, args.data_dir,
                                           fold=args.fold)
    train_loader = build_loader(train_files, train=True,  roi=roi,
                                batch_size=args.batch_size)
    val_loader   = build_loader(val_files,   train=False, roi=roi,
                                batch_size=args.val_batch_size)
    print("Finished Building Data Loaders")

    print("Setting Up Model")
    model = build_model(device, args.checkpoint)
    freeze_for_warmup(model)
    optimizer = build_optimizer(model,
                                lr_fusion=args.lr_fusion,
                                lr_backbone=args.lr_backbone,
                                weight_decay=args.weight_decay)
    print("Finished Building Model")

    loss_fn = DiceLoss(sigmoid=True, softmax=False,
                       to_onehot_y=False, include_background=True)
    # FIX 2: use device.type == "cuda" (device is a torch.device, not a str)
    use_amp = device.type == "cuda"
    scaler  = GradScaler(enabled=use_amp)

    best_mean  = -1.0
    ckpt_path  = os.path.join(args.out_dir, args.save_name)

    print(f"Starting Training using device {device}")
    for epoch in range(args.epochs):

        # --- Phase transition: unfreeze backbone after warmup ---
        if epoch == args.freeze_epochs:
            print(f"Epoch {epoch+1}: unfreezing backbone for fine-tuning")
            unfreeze_for_finetune(model, unfreeze_all=False)
            optimizer = build_optimizer(model,
                                        lr_fusion=args.lr_fusion,
                                        lr_backbone=args.lr_backbone,
                                        weight_decay=args.weight_decay)

        model.train()
        t0         = time.time()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            # FIX 2: enabled flag now correctly checks device.type
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss   = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        epoch_loss /= max(1, len(train_loader))

        do_val   = (epoch + 1) % 5 == 0
        val_dice = (evaluate(model, val_loader, device, roi,
                             args.sw_batch_size, args.infer_overlap)
                    if do_val else np.array([float("nan")] * 3))
        mean_dice = float(np.nanmean(val_dice))
        elapsed   = time.time() - t0

        print(
            f"Epoch {epoch+1:03d}/{args.epochs}  "
            f"loss={epoch_loss:.4f}  "
            f"val_TC={val_dice[0]:.4f}  val_WT={val_dice[1]:.4f}  "
            f"val_ET={val_dice[2]:.4f}  mean={mean_dice:.4f}  "
            f"time={elapsed:.1f}s"
        )

        if mean_dice > best_mean:
            best_mean = mean_dice
            torch.save(
                {
                    "epoch":             epoch + 1,
                    "best_acc":          best_mean,
                    "state_dict":        model.backbone.state_dict(),
                    "fusion_state_dict": model.fusion.state_dict(),
                    "args":              vars(args),
                },
                ckpt_path,
            )
            print(f"  → Saved best checkpoint to {ckpt_path}")

    print(f"Best validation mean Dice: {best_mean:.4f}")


if __name__ == "__main__":
    main()