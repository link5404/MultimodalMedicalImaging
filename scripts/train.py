def main():
    import yaml
    import torch
    from torch.utils.data import DataLoader
    from monai.losses.dice import DiceCELoss
    from monai.metrics.meandice import DiceMetric
    from monai.transforms.post.array import Activations, AsDiscrete
    from monai.data.utils import decollate_batch
    from monai.inferers.utils import sliding_window_inference

    
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))


    from src.dataset import build_datasets, MODALITIES
    from src.model import BraTSSwinUNETR

    with open("configs/train.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds = build_datasets(
        cfg["data_root"],
        cfg["image_size"],
        cfg["val_split"],
        cfg["seed"],
        cfg["dropout_prob"],
        cfg["max_drop_modalities"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=True
    )


    model = BraTSSwinUNETR(cfg["in_channels"], cfg["out_channels"], cfg["feature_size"]).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    post_pred = AsDiscrete(argmax=True, to_onehot=cfg["out_channels"])
    post_label = AsDiscrete(to_onehot=cfg["out_channels"])
    metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)

    for epoch in range(cfg["max_epochs"]):
        model.train()
        for batch in train_loader:
            x = torch.cat([batch[m].to(device) for m in MODALITIES], dim=1)
            y = batch["label"].to(device)

            optimizer.zero_grad()
            loss = loss_fn(model.forward(x), y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        metric.reset()
        with torch.no_grad():
            for batch in val_loader:
                
                
                x = torch.cat([batch[m].to(device) for m in MODALITIES], dim=1)
                y = batch["label"].to(device)


                logits = sliding_window_inference(x, cfg["image_size"], 1, model)
                print(f"DEBUG: logits.shape={logits.shape}" )


                val_out = post_pred(logits)  # [B, C, H, W, D] -> [B, num_classes, H, W, D]
                val_y = post_label(y)        # [B, 1, H, W, D] -> [B, num_classes, H, W, D]
                
                metric(y_pred=val_out, y=val_y)
                print(val_out[0].shape, val_y[0].shape)

        dice_result, not_nans = metric.aggregate()
        print(f"epoch={epoch+1} dice={dice_result.item():.4f} (n={not_nans})")
        #print(f"epoch={epoch+1} dice={metric.aggregate().item():.4f}")
        
        print('epoch done')

if __name__ == "__main__":
    main()
