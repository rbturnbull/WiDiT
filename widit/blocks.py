
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp

from .window import (
    _prod,
    _to_sizes,
    window_partition_nd,
    window_unpartition_nd,
    WindowAttention,
)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # x: (N, T, C), shift/scale: (N, C)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _pad_channels_last(x: torch.Tensor, pads: tuple[int, ...]) -> torch.Tensor:
    """
    Pad an (N, S1, S2, ..., Sk, C) tensor by pads=(p1,...,pk) on the positive side of each spatial dim.
    F.pad expects channels-first ordering, so we permute to N,C,*, pad, then permute back.
    """
    if not any(pads):
        return x
    k = len(pads)
    # to N,C,*
    x_cf = x.permute(0, k + 1, *range(1, k + 1))  # (N, C, S1, ..., Sk)
    # Build pad tuple for F.pad: (..., S1) pairs in reverse order
    pad_pairs = []
    for p in reversed(pads):
        pad_pairs.extend([0, p])  # (left=0, right=p)
    x_cf = F.pad(x_cf, tuple(pad_pairs))
    # back to channels-last
    x = x_cf.permute(0, *range(2, k + 2), 1)
    return x


def _roll_channels_last(x: torch.Tensor, shifts: Sequence[int], invert: bool = False) -> torch.Tensor:
    """
    Roll along spatial dims (N, S1, ..., Sk, C). If invert=True, rolls in the opposite direction.
    """
    k = len(shifts)
    if all(s == 0 for s in shifts):
        return x
    s = tuple((-si if not invert else si) for si in shifts)
    dims = tuple(range(1, k + 1))
    return torch.roll(x, shifts=s, dims=dims)


class WiditBlock(nn.Module):
    """
    N-D windowed attention + MLP with adaLN-Zero conditioning; optional ND shift.

    Args:
        dim: channel dimension per token
        num_heads: attention heads
        window_size: int or sequence per axis (ws1,...,wsk)
        shift_size: int or sequence per axis (defaults to 0 or ws_i//2 typically)
        mlp_ratio: hidden multiplier for MLP
        spatial_dim: number of spatial axes (2 for 2D, 3 for 3D)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int | Sequence[int],
        shift_size: int | Sequence[int] = 0,
        mlp_ratio: float = 4.0,
        spatial_dim: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.k = spatial_dim
        self.ws: tuple[int, ...] = _to_sizes(window_size, self.k)
        self.shift: tuple[int, ...] = _to_sizes(shift_size, self.k)

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention(dim, self.ws, num_heads, spatial_dim=self.k, qkv_bias=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        # shift/scale/gate x2 (msa + mlp)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def forward(self, x_seq: torch.Tensor, c: torch.Tensor, *grid_sizes: int) -> torch.Tensor:
        """
        x_seq: (N, T, C), where T == prod(grid_sizes)
        c:     (N, C) conditioning vector (timestep embedding projected to dim)
        grid_sizes: per-axis token grid (e.g., (Hp, Wp) or (Dp, Hp, Wp))
        """
        N, T, C = x_seq.shape
        assert len(grid_sizes) == self.k, f"Expected {self.k} grid sizes, got {len(grid_sizes)}"
        assert _prod(grid_sizes) == T, f"Token/grid mismatch: prod{grid_sizes} != {T}"

        ws = self.ws
        shifts = self.shift

        # Unpack adaLN-Zero params
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada(c).chunk(6, dim=1)

        # (N, S1, ..., Sk, C)
        x = x_seq.view(N, *grid_sizes, C)

        # Pad to multiples of window sizes
        pads = tuple((w - Si % w) % w for Si, w in zip(grid_sizes, ws))
        if any(pads):
            x = _pad_channels_last(x, pads)
        padded_sizes = tuple(Si + pi for Si, pi in zip(grid_sizes, pads))

        # Optional ND shift (Swin-style)
        if any(shifts):
            x = _roll_channels_last(x, shifts, invert=False)

        # Window partition -> (N*nW, *ws, C) -> tokens per window
        x_win = window_partition_nd(x, ws)
        Tw = self.attn.tokens_per_window()
        x_win = x_win.view(-1, Tw, C)

        # Repeat conditioning per window
        nW = _prod(Si // w for Si, w in zip(padded_sizes, ws))
        shift_msa_w = shift_msa.repeat_interleave(nW, dim=0)
        scale_msa_w = scale_msa.repeat_interleave(nW, dim=0)
        gate_msa_w  = gate_msa.repeat_interleave(nW, dim=0)

        # Attention (+ adaLN-Zero)
        h = self.norm1(x_win)
        h = modulate(h, shift_msa_w, scale_msa_w)
        h = self.attn(h)
        x_win = x_win + gate_msa_w.unsqueeze(1) * h

        # Merge windows back
        x = window_unpartition_nd(x_win.view(-1, *ws, C), ws, *padded_sizes)

        # Undo shift and crop padding
        if any(shifts):
            x = _roll_channels_last(x, shifts, invert=True)
        if any(pads):
            # crop each spatial axis back to original size
            slicer = [slice(None)] + [slice(0, Si) for Si in grid_sizes] + [slice(None)]
            x = x[tuple(slicer)]

        # MLP (+ adaLN-Zero)
        x = x.contiguous().view(N, T, C)
        h = self.norm2(x)
        h = modulate(h, shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class WiditFinalLayer(nn.Module):
    """
    ND final projection with adaLN-Zero:
      x: (N, T, C) -> linear to (N, T, (p^k) * out_channels)
    """
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, spatial_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        self.p = patch_size
        self.k = spatial_dim
        self.out_channels = out_channels
        self.linear = nn.Linear(hidden_size, (patch_size ** spatial_dim) * out_channels, bias=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))
