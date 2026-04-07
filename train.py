import os, torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.metrics.meandice import DiceMetric
from monai.inferers.utils import sliding_window_inference
from monai.data.utils import decollate_batch
from monai.transforms.post.array import AsDiscrete, Activations
from monai.transforms.compose import Compose
from typing import cast, List
from monai.networks.utils import one_hot
import tqdm

import config
from dataset import load_datalists, get_dataloaders
from model import get_model
from utils.transforms import get_train_transforms, get_val_transforms
from utils.losses import get_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"started training with {device_str}")
    print("started data management")
    # ── Data ────────────────────────────────────────────────────────────────
    train_files, val_files = load_datalists(config.DATA_DIR, config.VAL_FRACTION)
    train_tx = get_train_transforms(config.ROI_SIZE, config.MASK_PROB, config.MAX_MASKED)
    val_tx   = get_val_transforms(config.ROI_SIZE)
    train_loader, val_loader = get_dataloaders(
        train_files, val_files, train_tx, val_tx, config.BATCH_SIZE
    )
    print("end data management")
    # ── Model, loss, optimizer ───────────────────────────────────────────────
    print("loading model, scheduler")
    model     = get_model(config.NUM_CLASSES, None, device=device_str)
    loss_fn   = get_loss(config.LOSS_TYPE)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.MAX_EPOCHS)
    print("done loading model and scheduler")
    # ── Fine-tuning option: freeze encoder, train decoder first ──────────────
    # Uncomment to freeze the Swin backbone for the first N epochs:
    # for name, param in model.named_parameters():
    #     if "swinViT" in name:
    #         param.requires_grad = False

    # ── Metrics ─────────────────────────────────────────────────────────────
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    post_pred   = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label  = AsDiscrete(to_onehot=config.NUM_CLASSES)

    best_dice = 0.0
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    print("start training")
    for epoch in range(1, config.MAX_EPOCHS + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            preds = model(images)
            labels_onehot = one_hot(labels, num_classes=config.NUM_CLASSES)  # [B, 3, 128, 128, 128]
            loss = loss_fn(preds, labels_onehot)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch}/{config.MAX_EPOCHS}  loss: {epoch_loss/len(train_loader):.4f}")

        # ── Validate ─────────────────────────────────────────────────────────
        if epoch % config.VAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)
                    preds  = sliding_window_inference(
                        images, config.ROI_SIZE, 4, model, overlap=0.5
                    )
                    preds_list  = cast(List[torch.Tensor],[post_pred(p) for p in cast(List[torch.Tensor], decollate_batch(preds))])
                    labels_list = cast(List[torch.Tensor], [post_label(l) for l in cast(List[torch.Tensor], decollate_batch(labels))])
                    dice_metric(preds_list, labels_list)
            mean_dice = dice_metric.aggregate().mean().item() # type:ignore
            dice_metric.reset()
            print(f"  Val Dice: {mean_dice:.4f}")

            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(model.state_dict(),
                           os.path.join(config.CHECKPOINT_DIR, "best_model.pt"))
                print(f"  Best Model Saved (dice={best_dice:.4f})")

if __name__ == "__main__":
    main()