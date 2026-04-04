from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from src.models.time_embedding import SinusoidalTimeEmbedding


def _num_groups(channels: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )

        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        if in_ch == out_ch:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.shortcut(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class EpsilonUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        channel_multipliers: List[int],
        num_res_blocks: int,
        time_embed_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_res_blocks != 1:
            raise ValueError("This baseline expects num_res_blocks=1 for PA1 simplicity.")

        chs = [base_channels * m for m in channel_multipliers]

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, chs[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        current_ch = chs[0]
        for i, target_ch in enumerate(chs):
            self.down_blocks.append(ResidualBlock(current_ch, target_ch, time_embed_dim, dropout))
            current_ch = target_ch
            if i < len(chs) - 1:
                self.downsamples.append(Downsample(current_ch))

        self.mid1 = ResidualBlock(current_ch, current_ch, time_embed_dim, dropout)
        self.mid2 = ResidualBlock(current_ch, current_ch, time_embed_dim, dropout)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i in reversed(range(len(chs))):
            skip_ch = chs[i]
            self.up_blocks.append(ResidualBlock(current_ch + skip_ch, skip_ch, time_embed_dim, dropout))
            current_ch = skip_ch
            if i > 0:
                self.upsamples.append(Upsample(current_ch))

        self.out_norm = nn.GroupNorm(_num_groups(current_ch), current_ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(current_ch, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(timesteps)

        h = self.in_conv(x)
        skips = []

        for i, block in enumerate(self.down_blocks):
            h = block(h, t_emb)
            skips.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        for i, block in enumerate(self.up_blocks):
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = torch.nn.functional.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)
            if i < len(self.upsamples):
                h = self.upsamples[i](h)

        return self.out_conv(self.out_act(self.out_norm(h)))
