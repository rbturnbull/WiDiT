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
    """
    Apply adaLN-style modulation to a sequence of tokens.

    Args:
      x:     (batch, tokens, channels)
      shift: (batch, channels)
      scale: (batch, channels)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _pad_channels_last(x: torch.Tensor, pads_per_axis: tuple[int, ...]) -> torch.Tensor:
    """
    Pad an (N, S1, S2, ..., Sk, C) tensor on the positive side of each spatial dim.
    Internally swaps to channels-first for F.pad, then swaps back.

    Args:
      x:            (N, S1, S2, ..., Sk, C)
      pads_per_axis: tuple of length k with right-side padding for each axis
    """
    if not any(pads_per_axis):
        return x
    k = len(pads_per_axis)

    # To N,C,*
    x_nchw = x.permute(0, k + 1, *range(1, k + 1))  # (N, C, S1, ..., Sk)

    # Build pad tuple for F.pad in reverse spatial order: (..., S1) pairs
    pad_pairs: list[int] = []
    for p in reversed(pads_per_axis):
        pad_pairs.extend([0, p])  # (left=0, right=p)
    x_nchw = F.pad(x_nchw, tuple(pad_pairs))

    # Back to channels-last
    x = x_nchw.permute(0, *range(2, k + 2), 1)
    return x


def _roll_channels_last(x: torch.Tensor, shift_sizes: Sequence[int], invert: bool = False) -> torch.Tensor:
    """
    Roll along the spatial dims of a channels-last tensor (N, S1, ..., Sk, C).
    If invert=True, rolls in the opposite direction.
    """
    k = len(shift_sizes)
    if all(s == 0 for s in shift_sizes):
        return x
    shifts = tuple((-s if not invert else s) for s in shift_sizes)
    dims = tuple(range(1, k + 1))
    return torch.roll(x, shifts=shifts, dims=dims)


class WiDiTBlock(nn.Module):
    """
    N-D windowed attention + MLP with adaLN-Zero conditioning; optional N-D Swin shift.

    Args:
        dim:          token channel dimension
        num_heads:    attention heads
        window_size:  int or per-axis sequence (w1, ..., wk)
        shift_size:   int or per-axis sequence (defaults to 0 or w_i//2 typically)
        mlp_ratio:    hidden multiplier for MLP
        spatial_dim:  number of spatial axes (2 for 2D, 3 for 3D)
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
        self.channels = dim
        self.spatial_dims = spatial_dim
        self.window_sizes: tuple[int, ...] = _to_sizes(window_size, self.spatial_dims)
        self.shift_sizes: tuple[int, ...] = _to_sizes(shift_size, self.spatial_dims)

        self.pre_attn_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention(dim, self.window_sizes, num_heads, spatial_dim=self.spatial_dims, qkv_bias=True)
        self.pre_mlp_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )

        # AdaLN-Zero conditioner -> produces shift/scale/gate for MSA and MLP (6 * C)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def forward(
        self,
        tokens: torch.Tensor,
        timestep_embedding: torch.Tensor | None = None,
        *token_grid_sizes: int,
    ) -> torch.Tensor:
        """
        Args:
          tokens:            (N, T, C) token sequence where T == prod(token_grid_sizes)
          timestep_embedding:        (N, C) conditioning vector (e.g., timestep embedding projected to C). Optional.
          token_grid_sizes:  per-axis token grid sizes, e.g. (Hp, Wp) or (Dp, Hp, Wp)

        Returns:
          (N, T, C)
        """
        batch_size, num_tokens, channels = tokens.shape
        assert channels == self.channels, f"Channel mismatch: got {channels}, expected {self.channels}"
        assert len(token_grid_sizes) == self.spatial_dims, \
            f"Expected {self.spatial_dims} grid sizes, got {len(token_grid_sizes)}"
        assert _prod(token_grid_sizes) == num_tokens, \
            f"Token/grid mismatch: prod{token_grid_sizes} != {num_tokens}"

        window_sizes = self.window_sizes
        shift_sizes = self.shift_sizes

        # ---- adaLN-Zero params ----
        if timestep_embedding is None:
            # No conditioning: behave like standard LN + Attn/MLP residuals
            device = tokens.device
            dtype = tokens.dtype
            zeros = torch.zeros(batch_size, channels, device=device, dtype=dtype)
            ones  = torch.ones(batch_size, channels, device=device, dtype=dtype)
            shift_attn, scale_attn, gate_attn = zeros, zeros, ones
            shift_mlp,  scale_mlp,  gate_mlp  = zeros, zeros, ones
        else:
            shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = \
                self.adaln(timestep_embedding).chunk(6, dim=1)

        # (N, S1, ..., Sk, C)
        x_spatial = tokens.view(batch_size, *token_grid_sizes, channels)

        # Pad to multiples of window sizes
        pads_per_axis = tuple((w - Si % w) % w for Si, w in zip(token_grid_sizes, window_sizes))
        if any(pads_per_axis):
            x_spatial = _pad_channels_last(x_spatial, pads_per_axis)
        padded_grid = tuple(Si + pi for Si, pi in zip(token_grid_sizes, pads_per_axis))

        # Optional N-D Swin shift
        if any(shift_sizes):
            x_spatial = _roll_channels_last(x_spatial, shift_sizes, invert=False)

        # Window partition -> (N*nW, *ws, C) -> window tokens
        windows = window_partition_nd(x_spatial, window_sizes)
        tokens_per_window = _prod(window_sizes)
        windows = windows.view(-1, tokens_per_window, channels)

        # Repeat conditioning per window
        num_windows = _prod(Si // w for Si, w in zip(padded_grid, window_sizes))
        shift_attn_w = shift_attn.repeat_interleave(num_windows, dim=0)
        scale_attn_w = scale_attn.repeat_interleave(num_windows, dim=0)
        gate_attn_w  = gate_attn.repeat_interleave(num_windows, dim=0)

        # --- MSA + adaLN-Zero ---
        h = self.pre_attn_norm(windows)
        h = modulate(h, shift_attn_w, scale_attn_w)
        h = self.attn(h)
        windows = windows + gate_attn_w.unsqueeze(1) * h

        # Merge windows back
        x_spatial = window_unpartition_nd(windows.view(-1, *window_sizes, channels),
                                          window_sizes, *padded_grid)

        # Undo shift and crop padding
        if any(shift_sizes):
            x_spatial = _roll_channels_last(x_spatial, shift_sizes, invert=True)
        if any(pads_per_axis):
            slicer = [slice(None)] + [slice(0, Si) for Si in token_grid_sizes] + [slice(None)]
            x_spatial = x_spatial[tuple(slicer)]

        # --- MLP + adaLN-Zero ---
        tokens = x_spatial.contiguous().view(batch_size, num_tokens, channels)
        h = self.pre_mlp_norm(tokens)
        h = modulate(h, shift_mlp, scale_mlp)
        tokens = tokens + gate_mlp.unsqueeze(1) * self.mlp(h)
        return tokens

    def init_weights(self) -> None:
        """
        adaLN-Zero: zero the *last* Linear in the adaLN MLP so the residual gates
        start closed and the network behaves like a vanilla transformer at init.
        """
        # Find the last Linear in self.adaln safely (sequence: SiLU -> Linear)
        last_linear = None
        for module in reversed(self.adaln):
            if isinstance(module, nn.Linear):
                last_linear = module
                break
        if last_linear is not None:
            nn.init.constant_(last_linear.weight, 0)
            if last_linear.bias is not None:
                nn.init.constant_(last_linear.bias, 0)


class WiDiTFinalLayer(nn.Module):
    """
    N-D final projection with adaLN-Zero:
      tokens: (N, T, C) -> linear -> (N, T, (p^k) * out_channels)
    """
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, spatial_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Adaptive Layer Norm
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        self.patch_size = patch_size
        self.spatial_dims = spatial_dim
        self.out_channels = out_channels
        self.linear = nn.Linear(hidden_size, (patch_size ** spatial_dim) * out_channels, bias=True)

    def forward(self, tokens: torch.Tensor, timestep_embedding: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
          tokens:     (N, T, C)
          timestep_embedding: (N, C) optional conditioning vector (e.g., timestep embedding projected to C)

        Returns:
          (N, T, (patch_size^spatial_dims) * out_channels)
        """
        if timestep_embedding is None:
            batch_size, _, channels = tokens.shape
            device, dtype = tokens.device, tokens.dtype
            shift = torch.zeros(batch_size, channels, device=device, dtype=dtype)
            scale = torch.zeros(batch_size, channels, device=device, dtype=dtype)
        else:
            shift, scale = self.adaln(timestep_embedding).chunk(2, dim=1)
        return self.linear(modulate(self.norm(tokens), shift, scale))

    def init_weights(self) -> None:
        """
        adaLN-Zero for the head + zero output projection.
        Matches the behavior you previously had in models.py.
        """
        # Zero the last Linear in ada (SiLU -> Linear)
        last_linear = None
        for module in reversed(self.adaln):
            if isinstance(module, nn.Linear):
                last_linear = module
                break
        if last_linear is not None:
            nn.init.constant_(last_linear.weight, 0)
            if last_linear.bias is not None:
                nn.init.constant_(last_linear.bias, 0)

        # Zero the output projection so the head starts as identity via residuals
        nn.init.constant_(self.linear.weight, 0)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)
