import os, torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import AsDiscrete, Activations, Compose

import config
from dataset import load_datalists, get_dataloaders
from model import get_model
from utils.transforms import get_train_transforms, get_val_transforms
from utils.losses import get_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Data ────────────────────────────────────────────────────────────────
    train_files, val_files = load_datalists(config.TRAIN_JSON, config.VAL_JSON)
    train_tx = get_train_transforms(config.ROI_SIZE, config.MASK_PROB, config.MAX_MASKED)
    val_tx   = get_val_transforms(config.ROI_SIZE)
    train_loader, val_loader = get_dataloaders(
        train_files, val_files, train_tx, val_tx, config.BATCH_SIZE
    )

    # ── Model, loss, optimizer ───────────────────────────────────────────────
    model     = get_model(config.NUM_CLASSES, config.PRETRAINED_WEIGHTS, device)
    loss_fn   = get_loss(config.LOSS_TYPE)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.MAX_EPOCHS)

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

    for epoch in range(1, config.MAX_EPOCHS + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss  = loss_fn(preds, labels)
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
                    preds_list  = [post_pred(p) for p in decollate_batch(preds)]
                    labels_list = [post_label(l) for l in decollate_batch(labels)]
                    dice_metric(preds_list, labels_list)

            mean_dice = dice_metric.aggregate().mean().item()
            dice_metric.reset()
            print(f"  Val Dice: {mean_dice:.4f}")

            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(model.state_dict(),
                           os.path.join(config.CHECKPOINT_DIR, "best_model.pt"))
                print(f"  ✓ Saved best model (dice={best_dice:.4f})")

if __name__ == "__main__":
    main()