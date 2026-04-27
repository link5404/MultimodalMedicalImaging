import torch
import torch.nn as nn
import os
import json
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, NormalizeIntensityd, Orientationd,
    Spacingd, EnsureTyped, EnsureChannelFirstd, ConcatItemsd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
    RandShiftIntensityd,
)
from monai.data import Dataset, DataLoader, CacheDataset
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR    = "/home/sonalis3/arc3-ws/arc3/DL_project/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
JSON_FILE   = "/home/sonalis3/arc3-ws/arc3/DL_project/brats23_folds.json"

# Pretrained SwinUNETR backbone to fine-tune FROM
PRETRAIN_PATH = "/home/sonalis3/arc3-ws/arc3/DL_project/fold4_f48_ep300_4gpu_dice0_9035/model.pt"

# Where to save the trained skip connection model
SAVE_DIR    = "/home/sonalis3/arc3-ws/arc3/DL_project/skip_connection_trained"
os.makedirs(SAVE_DIR, exist_ok=True)

# Training hyperparameters
NUM_EPOCHS   = 50        # fine-tuning — fewer epochs than training from scratch
BATCH_SIZE   = 1
LR           = 1e-4      # small LR since backbone is pretrained
WEIGHT_DECAY = 1e-5
VAL_EVERY    = 5         # run validation every N epochs
TRAIN_SPLIT  = 0.8       # 80% train, 20% validation
ROI_SIZE     = (128, 128, 128)
NUM_WORKERS  = 4

# ─────────────────────────────────────────────────────────────────────────────
# CROSS-ATTENTION SKIP MODULE  (same as inference script)
# ─────────────────────────────────────────────────────────────────────────────
class CrossAttentionSkip(nn.Module):
    def __init__(self, enc_dim, dec_dim, num_heads=4):
        super().__init__()
        self.enc_proj = nn.Linear(enc_dim, dec_dim)
        self.q_proj   = nn.Linear(dec_dim, dec_dim)
        self.attn     = nn.MultiheadAttention(embed_dim=dec_dim, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(dec_dim, dec_dim)
        self.norm     = nn.LayerNorm(dec_dim)

    def forward(self, enc_feat, dec_feat):
        B, C_dec, D, H, W = dec_feat.shape
        dec_flat = dec_feat.flatten(2).permute(0, 2, 1)
        enc_flat = enc_feat.flatten(2).permute(0, 2, 1)
        enc_flat = self.enc_proj(enc_flat)
        Q = self.q_proj(dec_flat)
        K, V = enc_flat, enc_flat
        attn_out, _ = self.attn(Q, K, V)
        out = self.norm(dec_flat + self.out_proj(attn_out))
        return out.permute(0, 2, 1).reshape(B, C_dec, D, H, W)


class SwinUNETRWithSkip(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base  = base_model
        self.skip2 = CrossAttentionSkip(enc_dim=192, dec_dim=192, num_heads=4)
        self.skip3 = CrossAttentionSkip(enc_dim=384, dec_dim=384, num_heads=4)

    def forward(self, x):
        hidden_states_out = self.base.swinViT(x, self.base.normalize)
        enc0 = self.base.encoder1(x)
        enc1 = self.base.encoder2(hidden_states_out[0])
        enc2 = self.base.encoder3(hidden_states_out[1])
        enc3 = self.base.encoder4(hidden_states_out[2])
        dec4 = self.base.encoder10(hidden_states_out[4])
        enc3_att  = self.skip2(enc_feat=enc3,                  dec_feat=enc3)
        skip3_att = self.skip3(enc_feat=hidden_states_out[3],  dec_feat=hidden_states_out[3])
        dec3 = self.base.decoder5(dec4, skip3_att)
        dec2 = self.base.decoder4(dec3, enc3_att)
        dec1 = self.base.decoder3(dec2, enc2)
        dec0 = self.base.decoder2(dec1, enc1)
        out  = self.base.decoder1(dec0, enc0) 
        return self.base.out(out)


# ─────────────────────────────────────────────
# LOAD + SPLIT DATA  (80/20)
# ─────────────────────────────────────────────
with open(JSON_FILE, "r") as f:
    data = json.load(f)

raw_files = data["training"]
for item in raw_files:
    item["image"] = [os.path.join(DATA_DIR, x) for x in item["image"]]
    item["label"] = os.path.join(DATA_DIR, item["label"])

def all_files_exist(item):
    return all(os.path.isfile(p) for p in item["image"]) and os.path.isfile(item["label"])

all_files = [item for item in raw_files if all_files_exist(item)]

# Deterministic split — same split every run
np.random.seed(42)
indices   = np.random.permutation(len(all_files))
n_train   = int(len(all_files) * TRAIN_SPLIT)
train_idx = indices[:n_train]
val_idx   = indices[n_train:]

train_files = [all_files[i] for i in train_idx]
val_files   = [all_files[i] for i in val_idx]

print(f"Total cases    : {len(all_files)}")
print(f"Training cases : {len(train_files)}  (80%)")
print(f"Validation cases: {len(val_files)}   (20%)\n")

# Save val split so inference script can load it
with open(os.path.join(SAVE_DIR, "val_files.json"), "w") as f:
    json.dump(val_files, f)

# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
# Training transforms include augmentation
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    ConcatItemsd(keys=["image"], name="image", dim=0),
    EnsureChannelFirstd(keys=["label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    # Random crop to ROI size — required for fixed-size batching
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=ROI_SIZE,
        pos=1, neg=1,
        num_samples=1,
        image_key="image",
        image_threshold=0,
    ),
    # Augmentation — important for generalization
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    RandRotate90d(keys=["image", "label"], prob=0.1, max_k=3),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    EnsureTyped(keys=["image", "label"]),
])

# Validation transforms — no augmentation, no random crop
val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    ConcatItemsd(keys=["image"], name="image", dim=0),
    EnsureChannelFirstd(keys=["label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image", "label"]),
])

train_dataset = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.1)
val_dataset   = Dataset(data=val_files,   transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_dataset,   batch_size=1,          shuffle=False, num_workers=2)

# ─────────────────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Load pretrained backbone
base_swinunetr = SwinUNETR(in_channels=4, out_channels=3, feature_size=48).to(device)
ckpt = torch.load(PRETRAIN_PATH, map_location=device, weights_only=False)
base_swinunetr.load_state_dict(ckpt["state_dict"])
print(f"Loaded pretrained backbone: {PRETRAIN_PATH}")

# Wrap with skip connections
model = SwinUNETRWithSkip(base_swinunetr).to(device)

# ─────────────────────────────────────────────────────────────────────────────
# WHAT TO TRAIN:
#   Option A (recommended for course project): Fine-tune EVERYTHING
#     — backbone + skip modules trained together end-to-end
#     — gives the best Dice improvement
#     — use a small LR (1e-4) so pretrained weights are not destroyed
#
#   Option B: Train skip modules only (faster, less improvement)
#     — uncomment the freeze block below
#
# Currently set to Option A (full fine-tune)
# ─────────────────────────────────────────────────────────────────────────────

# # Option B — freeze backbone, only train skip modules:
# for param in model.base.parameters():
#     param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable:,}\n")

# ─────────────────────────────────────────────
# LOSS, OPTIMIZER, SCHEDULER
# ─────────────────────────────────────────────
# DiceCELoss(sigmoid=True) — matches how the model was originally trained
# sigmoid=True means each output channel is treated as an independent binary prediction
# This is CORRECT for BRaTS multi-label segmentation (TC, WT, ET can overlap)
loss_fn = DiceCELoss(
    smooth_nr=0, smooth_dr=1e-5,
    squared_pred=True,
    to_onehot_y=False,
    sigmoid=True,           # must match training setup
)

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# ─────────────────────────────────────────────
# GROUND TRUTH CONVERSION
# ─────────────────────────────────────────────
def gt_to_multilabel(label_tensor):
    """
    Converts raw BRaTS labels {0,1,2,3} to 3-channel multi-label format.
    This is what the loss function compares against the model's 3 output channels.
    label_tensor : [B, 1, D, H, W]
    returns      : [B, 3, D, H, W]
    """
    tc = ((label_tensor == 1) | (label_tensor == 3)).float()
    wt = ((label_tensor == 1) | (label_tensor == 2) | (label_tensor == 3)).float()
    et = (label_tensor == 3).float()
    return torch.cat([tc, wt, et], dim=1)

# ─────────────────────────────────────────────
# VALIDATION FUNCTION
# ─────────────────────────────────────────────
dice_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)

def validate(model, loader, device):
    model.eval()
    with torch.no_grad():
        for batch in loader:
            img   = batch["image"].to(device)
            label = batch["label"]
            output = sliding_window_inference(
                inputs=img, roi_size=ROI_SIZE,
                sw_batch_size=1, predictor=model, overlap=0.5,
            )
            pred_regions = (torch.sigmoid(output.cpu()) > 0.5).float()
            gt_regions   = gt_to_multilabel(label)
            dice_metric(y_pred=pred_regions, y=gt_regions)

    scores = dice_metric.aggregate()[0]
    dice_metric.reset()
    tc = scores[0].item()
    wt = scores[1].item()
    et = scores[2].item()
    return tc, wt, et, np.mean([tc, wt, et])

# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
best_mean_dice = 0.0
history = []

print("=" * 65)
print("  STARTING TRAINING")
print("=" * 65)

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    n_batches  = 0

    for batch in train_loader:
        img   = batch["image"].to(device)
        label = batch["label"].to(device)

        # Convert GT labels to multi-label format for loss
        gt = gt_to_multilabel(label)

        optimizer.zero_grad()
        output = model(img)          # [B, 3, D, H, W] raw logits
        loss   = loss_fn(output, gt)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches  += 1

    scheduler.step()
    avg_loss = epoch_loss / n_batches
    lr_now   = scheduler.get_last_lr()[0]

    print(f"Epoch [{epoch:03d}/{NUM_EPOCHS}]  loss={avg_loss:.4f}  lr={lr_now:.2e}")

    # ── Validation ──────────────────────────────────────────────────────
    if epoch % VAL_EVERY == 0 or epoch == NUM_EPOCHS:
        tc, wt, et, mean = validate(model, val_loader, device)
        history.append({"epoch": epoch, "loss": avg_loss, "TC": tc, "WT": wt, "ET": et, "mean": mean})

        print(f"  Val Dice → TC={tc:.4f}  WT={wt:.4f}  ET={et:.4f}  Mean={mean:.4f}")

        # Save best model
        if mean > best_mean_dice:
            best_mean_dice = mean
            save_path = os.path.join(SAVE_DIR, "best_model.pt")
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "mean_dice":  mean,
                "TC": tc, "WT": wt, "ET": et,
            }, save_path)
            print(f"  *** New best model saved  mean_dice={mean:.4f} → {save_path}")

    # Save latest checkpoint every 10 epochs (safety net)
    if epoch % 10 == 0:
        torch.save({
            "epoch":      epoch,
            "state_dict": model.state_dict(),
            "optimizer":  optimizer.state_dict(),
        }, os.path.join(SAVE_DIR, f"checkpoint_epoch{epoch:03d}.pt"))

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  TRAINING COMPLETE")
print("=" * 65)
print(f"  Best mean Dice : {best_mean_dice:.4f}")
print(f"  Model saved to : {SAVE_DIR}/best_model.pt")
print(f"  Val split saved: {SAVE_DIR}/val_files.json")
print("=" * 65)

# Save training history as CSV
csv_path = os.path.join(SAVE_DIR, "training_history.csv")
with open(csv_path, "w") as f:
    f.write("epoch,loss,TC,WT,ET,mean\n")
    for h in history:
        f.write(f"{h['epoch']},{h['loss']:.4f},{h['TC']:.4f},{h['WT']:.4f},{h['ET']:.4f},{h['mean']:.4f}\n")
print(f"  History saved  : {csv_path}")