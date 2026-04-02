import torch
from monai.networks.nets import swin_unetr

def get_model(num_classes, pretrained_weights=None, device="cuda"):
    model = swin_unetr.SwinUNETR(
        in_channels=4,              # T1, T1c, T2, FLAIR
        out_channels=num_classes,
        feature_size=48,
        use_checkpoint=True,        # gradient checkpointing → lower VRAM
    ).to(device)

    if pretrained_weights:
        state = torch.load(pretrained_weights, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded pretrained weights from {pretrained_weights}")

    return model