import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


# ---------------------------------------------------------------------------
# FiLM primitive
# ---------------------------------------------------------------------------

class FiLM3d(nn.Module):
    """
    Feature-wise Linear Modulation for 3-D feature maps.
    Applies a per-channel affine transform conditioned on `cond`:
        out = x * (1 + gamma) + beta
    where gamma, beta are predicted from `cond`.
    """
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, channels * 2)
        # zero-init so FiLM starts as identity (safe when unfreezing base)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gb = self.to_gamma_beta(cond)                       # [B, 2*C]
        gamma, beta = gb.chunk(2, dim=-1)                   # [B, C] each
        # broadcast over spatial dims
        for _ in range(x.dim() - 2):
            gamma = gamma.unsqueeze(-1)
            beta  = beta.unsqueeze(-1)
        return x * (1.0 + gamma) + beta


# ---------------------------------------------------------------------------
# Intensity-stats conditioning encoder
# ---------------------------------------------------------------------------

class CondEncoder(nn.Module):
    """
    Encodes per-modality intensity statistics into a conditioning vector.

    Input  : image batch  [B, 4, H, W, D]  (T1, T1ce, T2, FLAIR)
    Output : cond vector  [B, cond_dim]

    Stats computed per channel (8 total):
        mean and std over nonzero voxels  →  [B, 8]
    Then projected to cond_dim with a 2-layer MLP + SiLU.
    """
    def __init__(self, cond_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(8, 64),
            nn.SiLU(),
            nn.Linear(64, cond_dim),
            nn.SiLU(),
        )

    @staticmethod
    def _intensity_stats(image_batch: torch.Tensor) -> torch.Tensor:
        """Returns [B, 8] tensor of (mean, std) for each of the 4 modalities."""
        B = image_batch.shape[0]
        stats = []
        for c in range(4):
            ch = image_batch[:, c]                              # [B, H, W, D]
            mask = (ch != 0).float()
            n    = mask.sum(dim=[1, 2, 3]).clamp(min=1)        # [B]
            mean = (ch * mask).sum(dim=[1, 2, 3]) / n          # [B]
            var  = ((ch - mean.view(B, 1, 1, 1)) ** 2 * mask).sum(dim=[1, 2, 3]) / n
            std  = var.sqrt()
            stats.extend([mean, std])
        return torch.stack(stats, dim=1)                        # [B, 8]

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        stats = self._intensity_stats(image_batch)              # [B, 8]
        return self.mlp(stats)                                  # [B, cond_dim]


# ---------------------------------------------------------------------------
# SwinUNETR + deep FiLM injection via forward hooks
# ---------------------------------------------------------------------------

class SwinUNETRWithFiLM(nn.Module):
    """
    Wraps a pretrained SwinUNETR and injects FiLM conditioning into each of
    the four decoder stages (decoder1–decoder4) using PyTorch forward hooks.

    The conditioning vector is derived *automatically* from the input image's
    per-modality intensity statistics via `CondEncoder`, so no external `cond`
    argument is needed during inference — just pass the image.

    Architecture
    ____________
    SwinUNETR (feature_size=48) decoder channel widths:
        decoder1 → 48   (finest / shallowest)
        decoder2 → 96
        decoder3 → 192
        decoder4 → 384  (coarsest / deepest)

    FiLM is injected at the *output* of each decoder block, before the next
    up-sampling step.

    Usage
    _____
        model = SwinUNETRWithFiLM(base_model, cond_dim=128)
        model.freeze_base_unfreeze_film()   # stage 1: train only FiLM + encoder
        logits = model(image)               # [B, 3, H, W, D]

        # after ~10 epochs:
        model.unfreeze_base()               # stage 2: full fine-tune
    """

    # Decoder names → output channel counts for feature_size=48
    _DECODER_CHANNELS = {
        "decoder1": 48,
        "decoder2": 48,
        "decoder3": 96,
        "decoder4": 192,
    }

    def __init__(self, base_model: SwinUNETR, cond_dim: int = 128):
        super().__init__()
        self.base     = base_model
        self.cond_dim = cond_dim

        # Conditioning pathway
        self.cond_encoder = CondEncoder(cond_dim)
        self.cond_proj    = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 2),
            nn.SiLU(),
            nn.Linear(cond_dim * 2, cond_dim),
        )

        # One FiLM layer per decoder stage
        self.film_layers = nn.ModuleDict({
            name: FiLM3d(channels=ch, cond_dim=cond_dim)
            for name, ch in self._DECODER_CHANNELS.items()
        })

        # Slot used by hooks to receive the current cond embedding
        self._current_cond: torch.Tensor | None = None
        self._register_film_hooks()

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_film_hooks(self):
        """Attach a forward hook to each decoder block that applies FiLM."""
        self._hook_handles = []
        for name, film in self.film_layers.items():
            block = getattr(self.base, name)

            def _make_hook(f: FiLM3d):
                def hook(module, input, output):
                    if self._current_cond is not None:
                        return f(output, self._current_cond)
                    return output
                return hook

            handle = block.register_forward_hook(_make_hook(film))
            self._hook_handles.append(handle)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: image batch  [B, 4, H, W, D]

        Returns:
            logits          [B, 3, H, W, D]
        """
        cond = self.cond_encoder(x)         # [B, cond_dim]
        self._current_cond = self.cond_proj(cond)
        out = self.base(x)
        self._current_cond = None
        return out

    # ------------------------------------------------------------------
    # Training-phase helpers
    # ------------------------------------------------------------------

    def freeze_base_unfreeze_film(self):
        """
        Stage 1 — freeze the SwinUNETR backbone, train only the FiLM layers
        and the conditioning encoder.  Call this at the start of training.
        """
        for p in self.base.parameters():
            p.requires_grad = False
        for module in (self.cond_encoder, self.cond_proj, self.film_layers):
            for p in module.parameters():
                p.requires_grad = True

    def unfreeze_base(self):
        """
        Stage 2 — unfreeze the backbone for full fine-tuning.
        Pair with a much lower LR (e.g. 1e-5) for the base parameters.
        """
        for p in self.base.parameters():
            p.requires_grad = True

    def film_parameter_count(self) -> int:
        """Convenience: count trainable FiLM + encoder parameters."""
        params = list(self.cond_encoder.parameters()) + \
                 list(self.cond_proj.parameters())    + \
                 list(self.film_layers.parameters())
        return sum(p.numel() for p in params)