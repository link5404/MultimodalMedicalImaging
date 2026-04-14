# train_adain_attention_dropout.py

import os
import json
import time
import random
import argparse
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from monai import transforms, data
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import AsDiscrete, Activations
from monai.utils.enums import MetricReduction


MODALITY_NAMES = ["T1n", "T1c", "T2w", "T2f (FLAIR)"]


class ModalityAdaIN(nn.Module):
    def __init__(self, num_modalities: int = 4, hidden: int = 32):
        super().__init__()
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 2),
            )
            for _ in range(num_modalities)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        out = []
        for i in range(C):
            xi = x[:, i:i+1, ...]
            mu = xi.mean(dim=[2, 3, 4], keepdim=True)
            std = xi.std(dim=[2, 3, 4], keepdim=True) + 1e-5
            xi_norm = (xi - mu) / std
            stats = torch.cat([mu.view(B, 1), std.view(B, 1)], dim=1)
            params = self.mlps[i](stats)
            gamma = params[:, 0:1].view(B, 1, 1, 1, 1)
            beta = params[:, 1:2].view(B, 1, 1, 1, 1)
            out.append(xi_norm * (1 + gamma) + beta)
        return torch.cat(out, dim=1)


class CrossModalityAttention(nn.Module):
    def __init__(self, num_modalities: int = 4, embed_dim: int = 64):
        super().__init__()
        self.scale = embed_dim ** -0.5
        self.q_proj = nn.Linear(1, embed_dim)
        self.k_proj = nn.Linear(1, embed_dim)
        self.v_proj = nn.Linear(1, embed_dim)
        self.out_proj = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        summary = x.mean(dim=[2, 3, 4]).unsqueeze(-1)
        Q = self.q_proj(summary)
        K = self.k_proj(summary)
        V = self.v_proj(summary)
        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attended = self.out_proj(torch.bmm(attn, V))
        gate = torch.sigmoid(attended).unsqueeze(-1).unsqueeze(-1)
        return x + x * gate


class ModalityAwareFusion(nn.Module):
    def __init__(self, num_modalities: int = 4, adain_hidden: int = 32, attn_embed_dim: int = 64):
        super().__init__()
        self.adain = ModalityAdaIN(num_modalities, adain_hidden)
        self.xattn = CrossModalityAttention(num_modalities, attn_embed_dim)
        self.proj = nn.Conv3d(num_modalities, num_modalities, kernel_size=1, groups=num_modalities)
        self.norm = nn.InstanceNorm3d(num_modalities, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adain(x)
        x = self.xattn(x)
        x = self.proj(x)
        x = self.norm(x)
        return x


class RobustBraTSModel(nn.Module):
    def __init__(self, backbone: SwinUNETR):
        super().__init__()
        self.fusion = ModalityAwareFusion(num_modalities=4, adain_hidden=32, attn_embed_dim=64)
        self.backbone = backbone

    def forward(self, x):
        x = self.fusion(x)
        return self.backbone(x)


class RandomModalityDropoutd:
    def __init__(self, p=0.7, max_dropped=2):
        self.p = p
        self.max_dropped = max_dropped

    def __call__(self, data_dict):
        d = dict(data_dict)
        if random.random() > self.p:
            return d
        image = d["image"]
        if not torch.is_tensor(image):
            image = torch.as_tensor(image)
        c = image.shape[0]
        n_drop = random.randint(1, min(self.max_dropped, c))
        drop_idx = random.sample(range(c), n_drop)
        image = image.clone()
        for ch in drop_idx:
            image[ch, ...] = 0.0
        d["image"] = image
        d["drop_idx"] = drop_idx
        return d


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
    val = [d for d in json_data if "fold" in d and d["fold"] == fold]
    return train, val


def build_train_transform(roi):
    return transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        transforms.RandSpatialCropd(keys=["image", "label"], roi_size=roi, random_size=False),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        transforms.RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
        transforms.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
        RandomModalityDropoutd(p=0.7, max_dropped=2),
    ])


def build_val_transform():
    return transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])


def build_loader(files, train=True, roi=(128, 128, 128), batch_size=1):
    tfm = build_train_transform(roi) if train else build_val_transform()
    ds = data.Dataset(data=files, transform=tfm)
    return data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )


def zero_out_channel(images, channel):
    x = images.clone()
    x[:, channel, ...] = 0.0
    return x


def get_trainable_params(model):
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.fusion.proj.parameters():
        p.requires_grad = False
    for p in model.fusion.norm.parameters():
        p.requires_grad = False
    for p in model.fusion.adain.parameters():
        p.requires_grad = True
    for p in model.fusion.xattn.parameters():
        p.requires_grad = True


def unfreeze_backbone(model, unfreeze_all=False):
    if unfreeze_all:
        for p in model.backbone.parameters():
            p.requires_grad = True
    else:
        for n, p in model.backbone.named_parameters():
            if n.startswith("swinViT") or n.startswith("encoder1") or n.startswith("encoder2"):
                p.requires_grad = True


def build_optimizer(model, lr_fusion=1e-3, lr_backbone=1e-5, weight_decay=1e-5):
    fusion_params = list(model.fusion.adain.parameters()) + list(model.fusion.xattn.parameters())
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    param_groups = [{"params": fusion_params, "lr": lr_fusion}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def build_model(device, checkpoint_path=None):
    model = SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
    ).to(device)
    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(state, strict=True)
    return RobustBraTSModel(model).to(device)


def evaluate(model, loader, device, roi, sw_batch_size, infer_overlap):
    model.eval()
    dice_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    inferer = partial(
        sliding_window_inference,
        roi_size=list(roi),
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=infer_overlap,
    )
    scores = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = inferer(images)
            y_pred = [post_pred(post_sigmoid(v)) for v in decollate_batch(logits)]
            y_true = decollate_batch(labels)
            dice_metric.reset()
            dice_metric(y_pred=y_pred, y=y_true)
            acc, _ = dice_metric.aggregate()
            scores.append(acc.cpu().numpy())
    return np.nanmean(np.stack(scores, axis=0), axis=0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--json_list", required=True)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--roi", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--val_batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--freeze_epochs", type=int, default=10)
    p.add_argument("--lr_fusion", type=float, default=1e-3)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--sw_batch_size", type=int, default=4)
    p.add_argument("--infer_overlap", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default="output_adain_attention")
    p.add_argument("--save_name", default="best_adain_attention.pt")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    roi = tuple(args.roi)

    train_files, val_files = datafold_read(args.json_list, args.data_dir, fold=args.fold)
    train_loader = build_loader(train_files, train=True, roi=roi, batch_size=args.batch_size)
    val_loader = build_loader(val_files, train=False, roi=roi, batch_size=args.val_batch_size)

    model = build_model(device, args.checkpoint)
    get_trainable_params(model)

    optimizer = build_optimizer(
        model,
        lr_fusion=args.lr_fusion,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
    )

    loss_fn = DiceLoss(sigmoid=True, softmax=False, to_onehot_y=True, include_background=True)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    best_mean = -1.0
    ckpt_path = os.path.join(args.out_dir, args.save_name)

    for epoch in range(args.epochs):
        if epoch == args.freeze_epochs:
            unfreeze_backbone(model, unfreeze_all=False)
            optimizer = build_optimizer(
                model,
                lr_fusion=args.lr_fusion,
                lr_backbone=args.lr_backbone,
                weight_decay=args.weight_decay,
            )

        model.train()
        t0 = time.time()
        epoch_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                logits = model(images)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        epoch_loss /= max(1, len(train_loader))
        val_dice = evaluate(model, val_loader, device, roi, args.sw_batch_size, args.infer_overlap)
        mean_dice = float(np.nanmean(val_dice))
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1:03d}/{args.epochs} "
            f"loss={epoch_loss:.4f} "
            f"val_TC={val_dice[0]:.4f} val_WT={val_dice[1]:.4f} val_ET={val_dice[2]:.4f} "
            f"mean={mean_dice:.4f} "
            f"time={elapsed:.1f}s"
        )

        if mean_dice > best_mean:
            best_mean = mean_dice
            torch.save(
                {
                    "epoch": epoch + 1,
                    "best_acc": best_mean,
                    "state_dict": model.backbone.state_dict(),
                    "fusion_state_dict": model.fusion.state_dict(),
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"Saved best checkpoint to {ckpt_path}")

    print(f"Best validation mean Dice: {best_mean:.4f}")


if __name__ == "__main__":
    main()