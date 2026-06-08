import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding, standard transformer-style."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) -> emb: (B, dim)."""
        device = t.device
        half_dim = self.dim // 2
        if half_dim <= 1:
            # Degenerate case: return all-zeros embedding
            return torch.zeros(t.shape[0], self.dim, device=device)
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation. Injects conditioning into feature maps."""

    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.scale_proj = nn.Linear(cond_dim, feature_dim)
        self.shift_proj = nn.Linear(cond_dim, feature_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale = self.scale_proj(cond)[:, :, None]  # (B, C, 1)
        shift = self.shift_proj(cond)[:, :, None]  # (B, C, 1)
        return x * (1.0 + scale) + shift


class DownBlock1D(nn.Module):
    """Conv1d -> GroupNorm -> ReLU -> Conv1d(stride=2) -> GroupNorm -> ReLU."""

    def __init__(self, in_dim: int, out_dim: int, cond_dim: int, kernel_size: int = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, out_dim, kernel_size, stride=2, padding=kernel_size // 2)
        self.norm1 = nn.GroupNorm(min(8, out_dim), out_dim)
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, stride=1, padding=kernel_size // 2)
        self.norm2 = nn.GroupNorm(min(8, out_dim), out_dim)
        self.film = FiLMBlock(cond_dim, out_dim)
        self.residual = nn.Conv1d(in_dim, out_dim, 1, stride=2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.film(x, cond)
        x = F.relu(x + residual)
        return x


class UpBlock1D(nn.Module):
    """Upsample -> Conv1d -> GroupNorm -> ReLU -> Conv1d -> GroupNorm -> ReLU
    with skip connection from the corresponding DownBlock.
    """

    def __init__(self, in_dim: int, out_dim: int, cond_dim: int, kernel_size: int = 5):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv1d(in_dim, out_dim, kernel_size, stride=1, padding=kernel_size // 2)
        self.norm1 = nn.GroupNorm(min(8, out_dim), out_dim)
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, stride=1, padding=kernel_size // 2)
        self.norm2 = nn.GroupNorm(min(8, out_dim), out_dim)
        self.film = FiLMBlock(cond_dim, out_dim)
        self.residual = nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Match spatial dimension with skip (upsample may not be exact)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
        # Concatenate skip connection along channel dim
        x = torch.cat([x, skip], dim=1)
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.film(x, cond)
        x = F.relu(x + residual)
        return x


class DiffusionUNet1D(nn.Module):
    """1D Convolutional U-Net for action noise prediction.

    Input: noisy actions (B, action_dim, horizon)
    Output: predicted noise (B, action_dim, horizon)

    Conditioning via FiLM at each block, driven by
    concatenated timestep embedding + condition vector.
    """

    def __init__(
        self,
        action_dim: int = 2,
        horizon: int = 10,
        cond_dim: int = 256,
        dims: List[int] | None = None,
        time_emb_dim: int = 128,
    ):
        super().__init__()
        if dims is None:
            dims = [64, 128, 256]

        self.action_dim = action_dim
        self.horizon = horizon
        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        # Combined conditioning: timestep + condition
        total_cond_dim = 256 + cond_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(total_cond_dim, 256),
            nn.ReLU(),
        )

        # Input projection
        self.input_proj = nn.Conv1d(action_dim, dims[0], kernel_size=5, padding=2)

        # Encoder
        self.down_blocks = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.down_blocks.append(
                DownBlock1D(dims[i], dims[i + 1], 256)
            )

        # Bottleneck
        bottleneck_dim = dims[-1]
        self.bottleneck_conv1 = nn.Conv1d(bottleneck_dim, bottleneck_dim, 3, padding=1)
        self.bottleneck_norm1 = nn.GroupNorm(min(8, bottleneck_dim), bottleneck_dim)
        self.bottleneck_conv2 = nn.Conv1d(bottleneck_dim, bottleneck_dim, 3, padding=1)
        self.bottleneck_norm2 = nn.GroupNorm(min(8, bottleneck_dim), bottleneck_dim)
        self.bottleneck_film = FiLMBlock(256, bottleneck_dim)

        # Decoder: match upblocks to skips (excluding deepest which = bottleneck)
        # skip channels from shallow to deep: dims[0], dims[1], ..., dims[-1]
        # upblock i: prev_out_channels + skip_channels -> new_out_channels
        self.up_blocks = nn.ModuleList()
        prev_dim = dims[-1]  # bottleneck output channels
        reversed_dims = list(reversed(dims))  # e.g. [256, 128, 64]
        for i in range(len(reversed_dims) - 1):
            skip_dim = reversed_dims[i + 1]  # matching skip channel count
            self.up_blocks.append(
                UpBlock1D(prev_dim + skip_dim, skip_dim, 256)
            )
            prev_dim = skip_dim

        # Output projection
        self.output_conv = nn.Conv1d(dims[0], action_dim, kernel_size=1)

    def _compute_cond(self, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Combine timestep embedding and condition into FiLM conditioning."""
        t_emb = self.time_mlp(self.time_emb(t))  # (B, 256)
        combined = torch.cat([t_emb, cond], dim=-1)  # (B, 256 + cond_dim)
        return self.cond_proj(combined)  # (B, 256)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Predict noise given noisy actions, timestep, and condition.

        Args:
            x: (B, action_dim, horizon) noisy action sequence.
            t: (B,) timestep indices.
            cond: (B, cond_dim) condition vector.

        Returns:
            (B, action_dim, horizon) predicted noise.
        """
        original_horizon = x.shape[-1]
        film_cond = self._compute_cond(t, cond)

        x = self.input_proj(x)
        skips = [x]

        for block in self.down_blocks:
            x = block(x, film_cond)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck_conv1(x)
        x = self.bottleneck_norm1(x)
        x = F.relu(x)
        x = self.bottleneck_conv2(x)
        x = self.bottleneck_norm2(x)
        x = self.bottleneck_film(x, film_cond)
        x = F.relu(x)

        # Decoder with skip connections
        for i, block in enumerate(self.up_blocks):
            skip = skips[-(i + 2)]
            x = block(x, skip, film_cond)

        x = self.output_conv(x)

        # Ensure output matches original horizon (down/up sampling may not be exact)
        if x.shape[-1] != original_horizon:
            x = F.interpolate(x, size=original_horizon, mode="linear", align_corners=False)
        return x
