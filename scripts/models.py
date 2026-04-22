import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class AdaptiveInstanceNorm3d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.in_norm = nn.InstanceNorm3d(num_features, affine=False, eps=eps)
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * self.in_norm(x) + self.beta


class SwinUNETRWithAdaIN(nn.Module):
    def __init__(self, base_model: SwinUNETR, feature_size: int = 48):
        super().__init__()
        self.base = base_model

        decoder_channels = [
            feature_size * 16,
            feature_size * 8,
            feature_size * 4,
            feature_size * 2,
            feature_size,
        ]

        self.adain_layers = nn.ModuleList([
            AdaptiveInstanceNorm3d(c) for c in decoder_channels
        ])

        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        decoder_blocks = [
            self.base.decoder5,
            self.base.decoder4,
            self.base.decoder3,
            self.base.decoder2,
            self.base.decoder1,
        ]
        for i, block in enumerate(decoder_blocks):
            hook = block.register_forward_hook(
                lambda module, inp, out, i=i: self.adain_layers[i](out)
            )
            self._hooks.append(hook)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)

    def freeze_base_unfreeze_adain(self):
        for p in self.base.parameters():
            p.requires_grad = False
        for p in self.adain_layers.parameters():
            p.requires_grad = True

    def unfreeze_decoder_and_adain(self):
        decoder_parts = [
            self.base.decoder1, self.base.decoder2,
            self.base.decoder3, self.base.decoder4,
            self.base.decoder5,
        ]
        for part in decoder_parts:
            for p in part.parameters():
                p.requires_grad = True
        for p in self.adain_layers.parameters():
            p.requires_grad = True