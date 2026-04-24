import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class FiLM3d(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gb = self.to_gamma_beta(cond)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class SwinUNETRWithFiLM(nn.Module):
    def __init__(self, base_model: SwinUNETR, cond_dim: int = 128):
        super().__init__()
        self.base = base_model
        self.film = FiLM3d(channels=3, cond_dim=cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        out = self.film(out, cond)
        return out

    def freeze_base_unfreeze_film(self):
        for p in self.base.parameters():
            p.requires_grad = False
        for p in self.film.parameters():
            p.requires_grad = True

    def unfreeze_base(self):
        for p in self.base.parameters():
            p.requires_grad = True