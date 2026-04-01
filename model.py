import torch
from monai.networks.nets import SwinUNETR

def get_model(num_classes, pretrained_weights=None, device="cuda"):
    model = SwinUNETR(
        img_size=(128, 128, 128),
        in_channels=4,              # T1, T1c, T2, FLAIR
        out_channels=num_classes,
        feature_size=48,
        use_checkpoint=True,        # gradient checkpointing → lower VRAM
    ).to(device)

    if pretrained_weights:
        state = torch.load(pretrained_weights, map_location=device)
        # strip 'swinViT.' prefix if loading SSL pretrained backbone only
        model.load_state_dict(state, strict=False)
        print(f"Loaded pretrained weights from {pretrained_weights}")

    return model