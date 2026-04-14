# modality_modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityAdaIN(nn.Module):
    """
    Per-modality Adaptive Instance Normalization.
    
    For each of the 4 input channels, a small MLP maps that channel's
    spatial statistics (mean, std) to affine params (gamma, beta).
    This replaces (or augments) the static NormalizeIntensityd transform.
    
    Input:  (B, 4, D, H, W)
    Output: (B, 4, D, H, W)  — normalised, then re-scaled per modality
    """
    def __init__(self, num_modalities: int = 4, hidden: int = 32):
        super().__init__()
        # One MLP per modality: maps [mean, std] → [gamma, beta]
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 2),   # outputs [gamma, beta]
            )
            for _ in range(num_modalities)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        out = []
        for i in range(C):
            xi = x[:, i:i+1, ...]           # (B, 1, D, H, W)
            # Compute spatial stats per sample
            mu  = xi.mean(dim=[2,3,4], keepdim=True)      # (B,1,1,1,1)
            std = xi.std(dim=[2,3,4], keepdim=True) + 1e-5
            # Normalise
            xi_norm = (xi - mu) / std
            # MLP on [mu, std] → affine params
            stats = torch.cat([mu.view(B,1), std.view(B,1)], dim=1)  # (B, 2)
            params = self.mlps[i](stats)                              # (B, 2)
            gamma = params[:, 0:1].view(B, 1, 1, 1, 1)
            beta  = params[:, 1:2].view(B, 1, 1, 1, 1)
            out.append(xi_norm * (1 + gamma) + beta)
        return torch.cat(out, dim=1)  # (B, C, D, H, W)


class CrossModalityAttention(nn.Module):
    """
    Lightweight cross-modality attention gate.
    
    Treats the 4 modalities as a sequence of length 4. For each modality i,
    it attends over all modalities j to produce a weighted feature mixture.
    Spatial dims are pooled to produce channel-level queries/keys/values,
    then the resulting weights are broadcast back spatially.
    
    This gives the model a soft "which modality matters here" signal without
    requiring it to be fully blind to dropped channels.
    
    Input:  (B, 4, D, H, W)  — after AdaIN
    Output: (B, 4, D, H, W)  — attended, residual-connected
    """
    def __init__(self, num_modalities: int = 4, embed_dim: int = 64):
        super().__init__()
        self.scale = embed_dim ** -0.5
        # Project each modality's spatial summary to Q/K/V
        self.q_proj = nn.Linear(1, embed_dim)
        self.k_proj = nn.Linear(1, embed_dim)
        self.v_proj = nn.Linear(1, embed_dim)
        self.out_proj = nn.Linear(embed_dim, 1)
        self.norm = nn.LayerNorm(num_modalities)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        # Global average pool over spatial dims → (B, C, 1)
        summary = x.mean(dim=[2,3,4]).unsqueeze(-1)   # (B, C, 1)

        Q = self.q_proj(summary)   # (B, C, E)
        K = self.k_proj(summary)
        V = self.v_proj(summary)

        # Attention: (B, C, C)
        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Weighted value mixture: (B, C, E) → (B, C, 1)
        attended = self.out_proj(torch.bmm(attn, V))  # (B, C, 1)
        # Scale factor to gate each modality channel
        gate = torch.sigmoid(attended)                 # (B, C, 1)
        gate = gate.unsqueeze(-1).unsqueeze(-1)        # (B, C, 1, 1, 1)

        # Residual: original + gated modulation
        return x + x * gate


class ModalityAwareFusion(nn.Module):
    """
    Full fusion wrapper: AdaIN → cross-modal attention → 1×1 proj → concat.
    
    Drop this in before SwinUNETR.  The output is still shape (B, 4, D, H, W)
    so the SwinUNETR in_channels=4 contract is preserved.
    """
    def __init__(self, num_modalities: int = 4,
                 adain_hidden: int = 32,
                 attn_embed_dim: int = 64):
        super().__init__()
        self.adain   = ModalityAdaIN(num_modalities, adain_hidden)
        self.xattn   = CrossModalityAttention(num_modalities, attn_embed_dim)
        # Per-modality 1×1 spatial conv to mix attended features back
        self.proj    = nn.Conv3d(num_modalities, num_modalities,
                                 kernel_size=1, groups=num_modalities)
        self.norm    = nn.InstanceNorm3d(num_modalities, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adain(x)
        x = self.xattn(x)
        x = self.proj(x)
        x = self.norm(x)
        return x