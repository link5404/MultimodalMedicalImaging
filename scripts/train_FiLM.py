import os
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai import transforms, data
from monai.transforms import AsDiscrete, Activations
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR
from monai.data import decollate_batch
from functools import partial

import torch
from torch.cuda.amp import autocast

from models import SwinUNETRWithAdaIN

# ── Paths & hyperparams ───────────────────────────────────────────────────────
root_dir  = os.path.dirname(os.path.abspath(__file__))
data_dir  = "/home/jordanatanassov/MultimodalMedicalImaging/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
json_list = "/home/jordanatanassov/MultimodalMedicalImaging/brats23_folds.json"

roi            = (128, 128, 128)
batch_size     = 2
sw_batch_size  = 4
fold           = 1
infer_overlap  = 0.5
val_every      = 10

# Two-phase epoch counts
PHASE1_EPOCHS  = 10   # AdaIN-only warmup
PHASE2_EPOCHS  = 40   # decoder + AdaIN fine-tune
max_epochs     = PHASE1_EPOCHS + PHASE2_EPOCHS

PRETRAINED_CKPT   = "model_checkpoint29_0.55.pt"   # plain SwinUNETR weights
ADAIN_RESUME_CKPT = "model_adain_checkpoint.pt"    # wrapped model resume (if exists)


# ── Utilities ─────────────────────────────────────────────────────────────────
class AverageMeter:
    def __init__(self): self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = np.where(
            self.count > 0,
            self.sum / np.where(self.count > 0, self.count, 1),
            self.sum,
        ).astype(float)


def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)[key]
    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str) and len(d[k]) > 0:
                d[k] = os.path.join(basedir, d[k])
    tr, val = [], []
    for d in json_data:
        (val if "fold" in d and d["fold"] == fold else tr).append(d)
    return tr, val


def get_loader(batch_size, data_dir, json_list, fold, roi):
    train_files, val_files = datafold_read(json_list, data_dir, fold)

    train_transform = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        transforms.CropForegroundd(
            keys=["image", "label"], source_key="image",
            k_divisible=list(roi), allow_smaller=True,
        ),
        transforms.RandSpatialCropd(keys=["image", "label"], roi_size=list(roi), random_size=False),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ])
    val_transform = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])

    train_loader = data.DataLoader(
        data.Dataset(data=train_files, transform=train_transform),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = data.DataLoader(
        data.Dataset(data=val_files, transform=val_transform),
        batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )
    return train_loader, val_loader


def save_checkpoint(model, epoch, optimizer, scheduler, filename, best_acc, dir_add=root_dir):
    path = os.path.join(dir_add, filename)
    torch.save({
        "epoch": epoch,
        "best_acc": best_acc,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, path)
    print(f"Saved checkpoint → {path}")


def load_adain_checkpoint(model, optimizer, scheduler, filename, dir_add=root_dir):
    """Load a checkpoint that was saved FROM the wrapped SwinUNETRWithAdaIN model."""
    path = os.path.join(dir_add, filename)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    print(f"Resumed AdaIN checkpoint (epoch {ckpt['epoch']}, best {ckpt['best_acc']:.4f})")
    return ckpt["epoch"] + 1, ckpt["best_acc"]


# ── Epoch functions ───────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scaler, loss_func, epoch):
    model.train()
    run_loss = AverageMeter()
    for idx, batch in enumerate(loader):
        imgs   = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        with autocast():
            loss = loss_func(model(imgs), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        run_loss.update(loss.item(), n=batch_size)
        print(f"  Epoch {epoch}  [{idx}/{len(loader)}]  loss {run_loss.avg:.4f}")
    return run_loss.avg


def val_epoch(model, loader, acc_func, model_inferer, post_sigmoid, post_pred, epoch):
    model.eval()
    run_acc = AverageMeter()
    acc_func.reset()
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            imgs, labels = batch["image"].to(device), batch["label"].to(device)
            logits = model_inferer(imgs)
            val_out = [post_pred(post_sigmoid(p)) for p in decollate_batch(logits)]
            acc_func(y_pred=val_out, y=decollate_batch(labels))
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy().astype(float))
            print(
                f"  Val {epoch} [{idx}/{len(loader)}]"
                f"  TC {run_acc.avg[0]:.4f}"
                f"  WT {run_acc.avg[1]:.4f}"
                f"  ET {run_acc.avg[2]:.4f}"
            )
    return run_acc.avg


# ── Two-phase trainer ─────────────────────────────────────────────────────────
def trainer(model, train_loader, val_loader, loss_func, acc_func,
            model_inferer, post_sigmoid, post_pred,
            start_epoch=0, val_acc_max=0.0):

    dices_tc, dices_wt, dices_et, dices_avg = [], [], [], []
    loss_epochs, trains_epoch = [], []

    for epoch in range(start_epoch, max_epochs):

        # ── Phase gate: switch optimiser & unfreeze on epoch boundary ────────
        if epoch == 0 and start_epoch == 0:
            print("=== Phase 1: AdaIN-only warmup ===")
            model.freeze_base_unfreeze_adain()
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-3, weight_decay=1e-5,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=PHASE1_EPOCHS,
            )
            scaler = torch.amp.GradScaler("cuda")

        elif epoch == PHASE1_EPOCHS and start_epoch <= PHASE1_EPOCHS:
            print("=== Phase 2: Decoder + AdaIN fine-tune ===")
            model.unfreeze_decoder_and_adain()
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=5e-5, weight_decay=1e-5,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=20, T_mult=1, eta_min=1e-6,
            )
            scaler = torch.amp.GradScaler("cuda")

        print(time.ctime(), f"Epoch {epoch}/{max_epochs-1}")
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, scaler, loss_func, epoch)
        scheduler.step()
        print(f"  Training loss {train_loss:.4f}  ({time.time()-t0:.1f}s)")

        if (epoch + 1) % val_every == 0 or epoch == 0:
            loss_epochs.append(train_loss)
            trains_epoch.append(epoch)
            t0 = time.time()

            val_acc     = val_epoch(model, val_loader, acc_func, model_inferer,
                                    post_sigmoid, post_pred, epoch)
            val_avg_acc = float(np.mean(val_acc))

            print(
                f"  Val summary  TC {val_acc[0]:.4f}"
                f"  WT {val_acc[1]:.4f}"
                f"  ET {val_acc[2]:.4f}"
                f"  Avg {val_avg_acc:.4f}  ({time.time()-t0:.1f}s)"
            )

            dices_tc.append(val_acc[0]);  dices_wt.append(val_acc[1])
            dices_et.append(val_acc[2]);  dices_avg.append(val_avg_acc)

            save_checkpoint(
                model, epoch, optimizer, scheduler,
                filename=f"model_adain_checkpoint{epoch}_{val_avg_acc:.2f}.pt",
                best_acc=val_acc_max,
            )

            if val_avg_acc > val_acc_max:
                print(f"  New best {val_acc_max:.6f} → {val_avg_acc:.6f}")
                val_acc_max = val_avg_acc
                save_checkpoint(
                    model, epoch, optimizer, scheduler,
                    filename="model_adain_best.pt",
                    best_acc=val_acc_max,
                )

    print(f"Training finished. Best avg dice: {val_acc_max:.4f}")
    return val_acc_max, dices_tc, dices_wt, dices_et, dices_avg, loss_epochs, trains_epoch


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_loader(batch_size, data_dir, json_list, fold, roi)

    # 1. Build base model and load YOUR pretrained plain-SwinUNETR weights
    base_model = SwinUNETR(
        in_channels=4, out_channels=3, feature_size=48,
        drop_rate=0.0, attn_drop_rate=0.0,
        dropout_path_rate=0.0, use_checkpoint=False,
    ).to(device)

    ckpt = torch.load(PRETRAINED_CKPT, map_location=device, weights_only=False)
    base_model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded pretrained weights from {PRETRAINED_CKPT}")

    # 2. Wrap with AdaIN
    model = SwinUNETRWithAdaIN(base_model, feature_size=48).to(device)

    # 3. Loss / metrics
    dice_loss    = DiceCELoss(to_onehot_y=False, sigmoid=True, lambda_dice=1.0, lambda_ce=1.0)
    post_sigmoid = Activations(sigmoid=True)
    post_pred    = AsDiscrete(argmax=False, threshold=0.5)
    dice_acc     = DiceMetric(include_background=True,
                              reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)

    model_inferer = partial(
        sliding_window_inference,
        roi_size=list(roi), sw_batch_size=sw_batch_size,
        predictor=model, overlap=infer_overlap,
    )

    # 4. Optionally resume a previous AdaIN fine-tune run
    #    (skip this block entirely if starting fresh)
    start_epoch    = 0
    val_acc_max    = 0.0
    adain_ckpt_path = os.path.join(root_dir, ADAIN_RESUME_CKPT)
    if os.path.exists(adain_ckpt_path):
        # Need a temporary optimizer/scheduler to load states into
        _opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        _sch = torch.optim.lr_scheduler.CosineAnnealingLR(_opt, T_max=PHASE1_EPOCHS)
        start_epoch, val_acc_max = load_adain_checkpoint(model, _opt, _sch, ADAIN_RESUME_CKPT)
        print(f"Resuming AdaIN fine-tune from epoch {start_epoch}")
    else:
        print("No AdaIN checkpoint found — starting Phase 1 from scratch.")

    # 5. Train
    (val_acc_max, dices_tc, dices_wt,
     dices_et, dices_avg, loss_epochs, trains_epoch) = trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_func=dice_loss,
        acc_func=dice_acc,
        model_inferer=model_inferer,
        post_sigmoid=post_sigmoid,
        post_pred=post_pred,
        start_epoch=start_epoch,
        val_acc_max=val_acc_max,
    )

    torch.save({"state_dict": model.state_dict(), "best_acc": val_acc_max},
               os.path.join(root_dir, "model_adain_final.pt"))

    # 7. Plots
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1); plt.title("Epoch Average Loss"); plt.xlabel("epoch")
    plt.plot(trains_epoch, loss_epochs, color="red")
    plt.subplot(1, 2, 2); plt.title("Val Mean Dice"); plt.xlabel("epoch")
    plt.plot(trains_epoch, dices_avg, color="green")
    plt.savefig(os.path.join(root_dir, "loss_and_dice.png"))

    plt.figure("dice", (18, 6))
    plt.subplot(1, 3, 1); plt.title("Val Mean Dice TC"); plt.xlabel("epoch")
    plt.plot(trains_epoch, dices_tc, color="blue")
    plt.subplot(1, 3, 2); plt.title("Val Mean Dice WT"); plt.xlabel("epoch")
    plt.plot(trains_epoch, dices_wt, color="brown")
    plt.subplot(1, 3, 3); plt.title("Val Mean Dice ET"); plt.xlabel("epoch")
    plt.plot(trains_epoch, dices_et, color="purple")
    plt.savefig(os.path.join(root_dir, "dice_per_region.png"))