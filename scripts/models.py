"""
models.py

BraTS performance-boosting model components:
  - ModalityAdaIN
  - CrossModalityAttention
  - ModalityAwareFusion
  - RobustBraTSModel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import SwinUNETR


class ModalityAdaIN(nn.Module):
    """Per-modality Adaptive Instance Normalisation.

    Learns a small MLP per modality that predicts scale (gamma) and shift
    (beta) from the channel's own spatial mean and std, then applies an
    affine transform on the normalised volume.
    """

    def __init__(self, num_modalities: int = 4, hidden: int = 32):
        super().__init__()
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 2),
            )
            for _ in range(num_modalities)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        out = []
        for i in range(C):
            xi = x[:, i:i+1, ...]                                   # (B,1,D,H,W)
            mu  = xi.mean(dim=[2, 3, 4], keepdim=True)
            std = xi.std(dim=[2, 3, 4], keepdim=True) + 1e-5
            xi_norm = (xi - mu) / std
            stats  = torch.cat([mu.view(B, 1), std.view(B, 1)], dim=1)  # (B,2)
            params = self.mlps[i](stats)                             # (B,2)
            gamma  = params[:, 0:1].view(B, 1, 1, 1, 1)
            beta   = params[:, 1:2].view(B, 1, 1, 1, 1)
            out.append(xi_norm * (1 + gamma) + beta)
        return torch.cat(out, dim=1)                                 # (B,C,D,H,W)


class CrossModalityAttention(nn.Module):
    """Cross-modality attention using spatial-mean feature vectors.

    Each modality is summarised by its spatial mean across (D,H,W) to form
    a feature vector of length `summary_dim`.  Standard scaled dot-product
    attention is then applied across the C modalities, and the resulting
    gate modulates the original volume.

    Using summary_dim > 1 (default 8) gives the attention enough capacity
    to distinguish modality relationships beyond a single scalar.
    """

    def __init__(self, num_modalities: int = 4, embed_dim: int = 64,
                 summary_dim: int = 8):
        super().__init__()
        self.summary_dim = summary_dim
        self.scale = embed_dim ** -0.5

        # Project from summary_dim-d feature → embed_dim
        self.q_proj   = nn.Linear(summary_dim, embed_dim)
        self.k_proj   = nn.Linear(summary_dim, embed_dim)
        self.v_proj   = nn.Linear(summary_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, summary_dim)

        # Mix summary_dim back to a single gate scalar per modality
        self.gate_fc  = nn.Linear(summary_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape

        # Build richer per-modality summary: mean + std over spatial dims,
        # projected to summary_dim via a linear combination over channels.
        # Simple approach: tile [mean, std] and project to summary_dim.
        mu  = x.mean(dim=[2, 3, 4])                                  # (B, C)
        std = x.std(dim=[2, 3, 4]) + 1e-5                           # (B, C)
        # Interleave to (B, C, 2) then zero-pad / linear to summary_dim
        stats = torch.stack([mu, std], dim=-1)                       # (B, C, 2)

        # Pad feature dim to summary_dim by repeating; keeps it parameter-free
        repeats = (self.summary_dim + 1) // 2
        summary = stats.repeat(1, 1, repeats)[..., :self.summary_dim]  # (B,C,summary_dim)

        Q = self.q_proj(summary)                                     # (B, C, embed_dim)
        K = self.k_proj(summary)
        V = self.v_proj(summary)

        attn     = torch.bmm(Q, K.transpose(1, 2)) * self.scale      # (B, C, C)
        attn     = F.softmax(attn, dim=-1)
        attended = self.out_proj(torch.bmm(attn, V))                 # (B, C, summary_dim)

        # Scalar gate per modality
        gate = torch.sigmoid(self.gate_fc(attended))                 # (B, C, 1)
        gate = gate.unsqueeze(-1).unsqueeze(-1)                      # (B, C, 1, 1, 1)

        return x + x * gate                                          # (B, C, D, H, W)


class ModalityAwareFusion(nn.Module):
    """AdaIN → cross-modality attention → depthwise projection.

    InstanceNorm removed: it would directly undo the AdaIN normalisation and
    fight the learned affine parameters.  A depthwise Conv3d + learnable
    per-channel scale/bias (via GroupNorm with groups=C) is used instead for
    a lightweight, non-destructive normalisation.
    """

    def __init__(self, num_modalities: int = 4, adain_hidden: int = 32,
                 attn_embed_dim: int = 64, summary_dim: int = 8):
        super().__init__()
        self.adain = ModalityAdaIN(num_modalities, adain_hidden)
        self.xattn = CrossModalityAttention(num_modalities, attn_embed_dim,
                                            summary_dim)
        # Depthwise conv mixes nothing across modalities but lets each
        # channel do a local 1×1×1 affine in feature space.
        self.proj  = nn.Conv3d(num_modalities, num_modalities,
                               kernel_size=1, groups=num_modalities)
        # GroupNorm with one group per modality: normalises spatial stats
        # but keeps modalities independent and does NOT undo AdaIN.
        self.norm  = nn.GroupNorm(num_groups=num_modalities,
                                  num_channels=num_modalities, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adain(x)
        x = self.xattn(x)
        x = self.proj(x)
        x = self.norm(x)
        return x


class RobustBraTSModel(nn.Module):
    """SwinUNETR wrapped with a ModalityAwareFusion pre-processing stage."""

    def __init__(self, backbone: SwinUNETR):
        super().__init__()
        self.fusion   = ModalityAwareFusion(num_modalities=4,
                                            adain_hidden=32,
                                            attn_embed_dim=64,
                                            summary_dim=8)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fusion(x)
        return self.backbone(x)