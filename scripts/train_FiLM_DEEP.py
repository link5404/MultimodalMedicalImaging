import os
import json
import shutil
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from monai.losses.dice import DiceLoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import (
    AsDiscrete,
    Activations,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR
from monai import data
from monai.data import decollate_batch
from monai.transforms import MapTransform

from functools import partial

import torch
from torch.amp import autocast, GradScaler

from tqdm.auto import tqdm
from deepFilm import SwinUNETRWithFiLM

class ConvertBraTS2023Labelsd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key]
            result = torch.stack([
                (label == 1) | (label == 3),               # TC: NCR + ET
                (label == 1) | (label == 2) | (label == 3), # WT: all tumor
                (label == 3),                               # ET: enhancing only
            ], dim=0).float()
            d[key] = result
        return d


root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = "/home/jordanatanassov/MultimodalMedicalImaging/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(
            self.count > 0,
            self.sum / np.where(self.count > 0, self.count, 1),  # avoid div-by-zero
            self.sum
        ).astype(float)  # ensures a clean ndarray, not an object array

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


def save_checkpoint(model, epoch, optimizer, scheduler, filename="model.pt", best_acc=0, dir_add=root_dir):
    save_dict = {
        "epoch": epoch,
        "best_acc": best_acc,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    filename = os.path.join(dir_add, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)

def get_loader(batch_size, data_dir, json_list, fold, roi):
    data_dir = data_dir
    datalist_json = json_list
    train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, fold=fold)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            ConvertBraTS2023Labelsd(keys="label"),
            transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                k_divisible=[roi[0], roi[1], roi[2]],
                allow_smaller=True,
            ),
            transforms.RandSpatialCropd(
                keys=["image", "label"],
                roi_size=[roi[0], roi[1], roi[2]],
                random_size=False,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            ConvertBraTS2023Labelsd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    train_ds = data.Dataset(data=train_files, transform=train_transform)
    """
    train_ds = data.CacheDataset(
        data=train_files,
        transform=train_transform,
        cache_rate=1.0,
        num_workers=4,
    )   
    """
    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_ds = data.Dataset(data=validation_files, transform=val_transform)
    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,  
    )
    # Debug one sample
    print("=== INSPECTING ONE SAMPLE ===")
    for i, batch_data in enumerate(val_loader):
        if i >= 1:  # just one batch
            break
        
        raw_label = batch_data["label"]  # before any transforms
        print("Raw label unique values:", torch.unique(raw_label).cpu().numpy())
        print("Raw label shape:", raw_label.shape)
        
        # Load the actual NIfTI file directly
        sample_info = val_loader.dataset.data[0]  # first validation sample
        seg_path = sample_info["label"]
        print(f"Loading raw seg from: {seg_path}")
        
        raw_seg = nib.load(seg_path).get_fdata()
        print("Raw NIfTI unique values:", np.unique(raw_seg))
        print("Raw NIfTI value counts:")
        for val in np.unique(raw_seg):
            count = np.sum(raw_seg == val)
            print(f"  Label {int(val)}: {count} voxels")
        
        # Check if label 4 (ET) exists
        et_count = np.sum(raw_seg == 4)
        print(f"Raw ET (label 4) voxels: {et_count}")
        break

    return train_loader, val_loader
    
json_list = "/home/jordanatanassov/MultimodalMedicalImaging/brats23_folds.json"
roi = (128, 128, 128)
batch_size = 2
sw_batch_size = 4
fold = 1
infer_overlap = 0.5
max_epochs = 100
val_every = 1


if __name__ == "__main__":
    train_loader, val_loader = get_loader(batch_size, data_dir, json_list, fold, roi)

    case_id = "BraTS-GLI-00000-000"
    img_add = os.path.normpath(os.path.join(data_dir, case_id, f"{case_id}-t2w.nii.gz"))
    label_add = os.path.normpath(os.path.join(data_dir, case_id, f"{case_id}-seg.nii.gz"))
    img = nib.load(img_add).get_fdata()
    label = nib.load(label_add).get_fdata()
    print(f"image shape: {img.shape}, label shape: {label.shape}")
    """out
    plt.figure("image", (18, 6))
    plt.subplot(1, 2, 1)
    plt.title("image")
    plt.imshow(img[:, :, 78], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("label")
    plt.imshow(label[:, :, 78])
    plt.show()
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=48,
        use_checkpoint=False,
    ).to(device)

    print("Loading Checkpoint")
    ckpt = torch.load("/home/jordanatanassov/MultimodalMedicalImaging/scripts/model_checkpoint9_0.87.pt", map_location=device, weights_only=False)
    state = base.state_dict()
    for k, v in ckpt["state_dict"].items():
        if k in state and state[k].shape == v.shape:
            state[k] = v
    base.load_state_dict(state, strict=False)
    def probe_decoder_channels(base, device, roi=(128,128,128)):
        hooks = {}
        handles = []
        for name in ["decoder1","decoder2","decoder3","decoder4"]:
            block = getattr(base, name)
            def make_hook(n):
                def h(module, input, output):
                    hooks[n] = output.shape[1]
                return h
            handles.append(block.register_forward_hook(make_hook(name)))
        dummy = torch.zeros(1, 4, *roi).to(device)
        with torch.no_grad():
            base(dummy)
        for h in handles:
            h.remove()
        print("Actual decoder channels:", hooks)

    probe_decoder_channels(base, device)

    model = SwinUNETRWithFiLM(base, cond_dim=128).to(device)
    model.freeze_base_unfreeze_film()
    print("Done Loading And Attatching FiLM")
    print(f"Trainable FiLM+encoder params: {model.film_parameter_count():,}")

    # 3. Dynamo/lru_cache warning — define model_inferer BEFORE torch.compile:
    def infer_fn(x):
        return model(x)

    model_inferer = partial(
        sliding_window_inference,
        roi_size=[roi[0], roi[1], roi[2]],
        sw_batch_size=sw_batch_size,
        predictor=infer_fn,
        overlap=infer_overlap,
    )
    from monai.losses import DiceCELoss

    dice_loss = DiceCELoss(
        to_onehot_y=False, 
        sigmoid=True, 
        lambda_dice=1.0, 
        lambda_ce=1.0,
        weight=torch.tensor([0.25, 0.25, 1.0]).to(device)  # TC, WT, ET weights
    )
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.25)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    scaler = GradScaler('cuda')
    
    
    def train_epoch(model, loader, optimizer, epoch, loss_func):
        model.train()
        
        start_time = time.time()
        run_loss = AverageMeter()
        for idx, batch_data in enumerate(loader):
            print(f"processing batch {idx}/{len(loader)}")
            data, target = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            
            with autocast("cuda"):
                logits = model(data)
                loss = loss_func(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss.update(loss.item(), n=batch_size)
            print(
                "Epoch {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()
        return run_loss.avg



    def val_epoch(
        model,
        loader,
        epoch,
        acc_func,
        model_inferer=None,
        post_sigmoid=None,
        post_pred=None,
    ):
        model.eval()
        start_time = time.time()
        run_acc = AverageMeter()

        with torch.no_grad():
            acc_func.reset()
            for idx, batch_data in enumerate(loader):
                data, target = batch_data["image"].to(device), batch_data["label"].to(device)
                print("ET voxels in target:", target[:, 2].sum().item())

                logits = model_inferer(data)
                val_labels_list = decollate_batch(target)
                val_outputs_list = decollate_batch(logits)
                val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
                #acc_func.reset()
                acc_func(y_pred=val_output_convert, y=val_labels_list)
                acc, not_nans = acc_func.aggregate()
                #run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy().astype(float))

                print("GT ET sum:", target[:, 2].sum().item())
                print("Pred ET sum:", val_output_convert[0][2].sum().item())
                dice_tc = run_acc.avg[0]
                dice_wt = run_acc.avg[1]
                dice_et = run_acc.avg[2]
                print(
                    "Val {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
                    ", dice_tc:",
                    dice_tc,
                    ", dice_wt:",
                    dice_wt,
                    ", dice_et:",
                    dice_et,
                    ", time {:.2f}s".format(time.time() - start_time),
                )
                start_time = time.time()

        return run_acc.avg


    def trainer(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_func,
        val_acc_max,
        acc_func,
        scheduler,
        model_inferer=None,
        start_epoch=0,
        post_sigmoid=None,
        post_pred=None,
    ):
        val_acc_max = val_acc_max
        dices_tc = []
        dices_wt = []
        dices_et = []
        dices_avg = []
        loss_epochs = []
        trains_epoch = []
                    # Replace CosineAnnealingLR with:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=50, T_mult=1, eta_min=1e-6
            )
        for epoch in range(start_epoch, max_epochs):
            print(time.ctime(), "Epoch:", epoch)
            epoch_time = time.time()
            if epoch == 10:
                model.unfreeze_base()
                for g in optimizer.param_groups:
                    g['lr'] = 1e-5
                print("Unfroze base model for full fine-tuning.")

            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                epoch=epoch,
                loss_func=loss_func,
            )
            scheduler.step() 
            print(
                "Final training  {}/{}".format(epoch, max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )

            if (epoch + 1) % val_every == 0 or epoch == 0:
                loss_epochs.append(train_loss)
                trains_epoch.append(int(epoch))
                epoch_time = time.time()
                val_acc = val_epoch(
                    model,
                    val_loader,
                    epoch=epoch,
                    acc_func=acc_func,
                    model_inferer=model_inferer,
                    post_sigmoid=post_sigmoid,
                    post_pred=post_pred,
                )
                dice_tc = val_acc[0]
                dice_wt = val_acc[1]
                dice_et = val_acc[2]
                val_avg_acc = np.mean(val_acc)
                print(
                    "Final validation stats {}/{}".format(epoch, max_epochs - 1),
                    ", dice_tc:",
                    dice_tc,
                    ", dice_wt:",
                    dice_wt,
                    ", dice_et:",
                    dice_et,
                    ", Dice_Avg:",
                    val_avg_acc,
                    ", time {:.2f}s".format(time.time() - epoch_time),
                )
                dices_tc.append(dice_tc)
                dices_wt.append(dice_wt)
                dices_et.append(dice_et)
                dices_avg.append(val_avg_acc)

                # checkpoint the model at every validation step to checkpoint because arc
                save_checkpoint(
                    model,
                    epoch=epoch,
                    optimizer=optimizer,      # add
                    scheduler=scheduler,      # add
                    filename=f"model_checkpoint_FiLM_DEEP{epoch}_{val_avg_acc:.2f}.pt",
                    best_acc=val_acc_max,
                    dir_add=root_dir,
                )

                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    save_checkpoint(
                        model,
                        epoch,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        filename="model_best.pt",
                        best_acc=val_acc_max,
                        dir_add=root_dir,
                    ) 
        print("Training Finished !, Best Accuracy: ", val_acc_max)
        return (
            val_acc_max,
            dices_tc,
            dices_wt,
            dices_et,
            dices_avg,
            loss_epochs,
            trains_epoch,
        )


    start_epoch = 0
    val_acc_max_loaded = 0.0
        
        
    (
        val_acc_max,
        dices_tc,
        dices_wt,
        dices_et,
        dices_avg,
        loss_epochs,
        trains_epoch,
    ) = trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        val_acc_max=val_acc_max_loaded,
        loss_func=dice_loss,
        acc_func=dice_acc,
        scheduler=scheduler,
        model_inferer=model_inferer,
        start_epoch=start_epoch,
        post_sigmoid=post_sigmoid,
        post_pred=post_pred,
    )
    
    
    save_checkpoint(
        model,
        epoch=max_epochs - 1,
        optimizer=optimizer,
        scheduler=scheduler,
        filename="model_final.pt",
        best_acc=val_acc_max,
        dir_add=root_dir,
    )
    
    
    print(f"train completed, best average dice: {val_acc_max:.4f} ")


    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    plt.xlabel("epoch")
    plt.plot(trains_epoch, loss_epochs, color="red")
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    plt.xlabel("epoch")
    plt.plot(trains_epoch, dices_avg, color="green")
    plt.savefig(os.path.join(root_dir, "loss_and_dice.png"))
    plt.figure("train", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title("Val Mean Dice TC")
    plt.xlabel("epoch")
    plt.plot(trains_epoch, dices_tc, color="blue")
    plt.subplot(1, 3, 2)
    plt.title("Val Mean Dice WT")
    plt.xlabel("epoch")
    plt.plot(trains_epoch, dices_wt, color="brown")
    plt.subplot(1, 3, 3)
    plt.title("Val Mean Dice ET")
    plt.xlabel("epoch")
    plt.plot(trains_epoch, dices_et, color="purple")
    plt.savefig(os.path.join(root_dir, "dice_per_region.png"))