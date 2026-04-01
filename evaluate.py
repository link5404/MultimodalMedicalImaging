# evaluate.py
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.data import DataLoader, Dataset, decollate_batch
from monai.transforms import AsDiscrete, Activations, Compose

import config
from model import get_model
from utils.transforms import get_val_transforms

TEST_JSON       = "/path/to/test_split.json"
BEST_CHECKPOINT = "/path/to/checkpoints/best_model.pt"

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = get_model(config.NUM_CLASSES, BEST_CHECKPOINT, device)
    model.eval()

    import json
    with open(TEST_JSON) as f:
        test_files = json.load(f)["training"]

    test_ds     = Dataset(test_files, transform=get_val_transforms(config.ROI_SIZE))
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4)

    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    post_pred   = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label  = AsDiscrete(to_onehot=config.NUM_CLASSES)

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            preds  = sliding_window_inference(
                images, config.ROI_SIZE, 4, model, overlap=0.5
            )
            preds_list  = [post_pred(p) for p in decollate_batch(preds)]
            labels_list = [post_label(l) for l in decollate_batch(labels)]
            dice_metric(preds_list, labels_list)

    per_class_dice = dice_metric.aggregate()
    print("Per-class Dice:", per_class_dice)
    print("Mean Dice:", per_class_dice.mean().item())

if __name__ == "__main__":
    evaluate()