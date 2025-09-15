# windows.py
from __future__ import annotations

from functools import reduce
from operator import mul
from typing import Iterable, Sequence

import torch
import torch.nn as nn


def _to_sizes(ws: int | Sequence[int], spatial_dim: int) -> tuple[int, ...]:
    if isinstance(ws, int):
        return (ws,) * spatial_dim
    ws = tuple(ws)
    assert len(ws) == spatial_dim, f"window_size must have length {spatial_dim}, got {len(ws)}"
    return ws


def _prod(xs: Iterable[int]) -> int:
    return reduce(mul, xs, 1)


def _build_rel_pos_index(ws: tuple[int, ...]) -> torch.Tensor:
    """
    Build a flattened relative-position index for arbitrary spatial_dim.
    ws: per-axis window sizes, e.g. (8,8) or (4,4,4)
    Returns: (T, T) long tensor of indices in [0, prod(2*ws_i-1)-1]
    """
    ranges = [torch.arange(w) for w in ws]  # each (w_i,)
    coords = torch.stack(torch.meshgrid(*ranges, indexing="ij"))  # (D, w1, w2, ...)

    coords_flat = torch.flatten(coords, start_dim=1)  # (D, T)
    rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (D, T, T)

    # shift to non-negative and mixed-radix flatten
    rel = rel.permute(1, 2, 0).contiguous()  # (T, T, D)
    bases = [2 * w - 1 for w in ws]
    for i, w in enumerate(ws):
        rel[:, :, i] += (w - 1)

    # multipliers for mixed-radix flattening
    multipliers = []
    running = 1
    for b in reversed(bases[1:]):
        running *= b
        multipliers.append(running)
    multipliers = list(reversed(multipliers)) + [1]  # length == D

    idx = torch.zeros(rel.shape[:2], dtype=torch.long)
    for i, m in enumerate(multipliers):
        idx += rel[:, :, i] * m
    return idx


def window_partition_nd(x: torch.Tensor, ws: int | Sequence[int]) -> torch.Tensor:
    """
    x: (N, S1, S2, ..., Sk, C)
      -> (N * nW, ws1, ws2, ..., wsk, C)
    where nW = prod_i (S_i // ws_i)
    """
    assert x.ndim >= 3, f"expected (N, ..., C), got {x.shape}"
    N, *spatial, C = x.shape
    spatial_dim = len(spatial)
    ws = _to_sizes(ws, spatial_dim)

    shape_blocks: list[int] = []
    for Si, wi in zip(spatial, ws):
        assert Si % wi == 0, f"size {Si} not divisible by window {wi}"
        shape_blocks += [Si // wi, wi]

    # (N, S1//w1, w1, S2//w2, w2, ..., C)
    x = x.view(N, *shape_blocks, C)

    # permute to group all block counts first, then all window sizes, then C
    # indices: [N, (S1//w1, w1), (S2//w2, w2), ..., C]
    # permutation: [0, 1, 3, 5, ..., 2, 4, 6, ..., last]
    block_idxs = [2 * i + 1 for i in range(spatial_dim)]  # positions of S//w
    win_idxs = [2 * i + 2 for i in range(spatial_dim)]    # positions of w
    perm = [0] + block_idxs + win_idxs + [x.ndim - 1]
    x = x.permute(*perm).contiguous()

    nW = _prod(Si // wi for Si, wi in zip(spatial, ws))
    out_shape = (N * nW, *ws, C)
    return x.reshape(*out_shape)


def window_unpartition_nd(windows: torch.Tensor, ws: int | Sequence[int], *spatial: int) -> torch.Tensor:
    """
    windows: (N * nW, ws1, ..., wsk, C)
      -> (N, S1, ..., Sk, C)
    """
    *batch_and_wins, C = windows.shape
    spatial_dim = len(batch_and_wins) - 1
    ws = _to_sizes(ws, spatial_dim)

    assert len(spatial) == len(ws), f"Provide S dims: expected {len(ws)}, got {len(spatial)}"
    assert tuple(batch_and_wins[1:]) == tuple(ws), (
        f"window tensor doesn't match ws: {batch_and_wins[1:]} vs {list(ws)}"
    )

    block_counts = [Si // wi for Si, wi in zip(spatial, ws)]
    nW = _prod(block_counts)
    N = windows.shape[0] // nW

    # reshape to (N, S1//w1, S2//w2, ..., w1, w2, ..., C)
    x = windows.view(N, *block_counts, *ws, C)

    # interleave blocks and windows per axis: (N, b1, w1, b2, w2, ..., C)
    # current order: (N, b1, b2, ..., w1, w2, ..., C)
    K = spatial_dim
    perm = [0]
    for i in range(K):
        perm.extend([1 + i, 1 + K + i])
    perm.append(x.ndim - 1)  # C at end
    x = x.permute(*perm).contiguous()

    # finally merge each (bi, wi) -> Si
    out_shape = [N]
    for bi, wi in zip(block_counts, ws):
        out_shape.append(bi * wi)
    out_shape.append(C)
    return x.view(*out_shape)


class WindowAttention(nn.Module):
    """
    N-D Windowed Multi-Head Attention with relative position bias.
    Works for 2D or 3D by setting `spatial_dim` and `window_size`.
    Expect input x of shape (B_, T, C) where T == prod(window_size).
    """
    def __init__(
        self,
        dim: int,
        window_size: int | Sequence[int],
        num_heads: int,
        spatial_dim: int,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.spatial_dim = spatial_dim
        self.ws: tuple[int, ...] = _to_sizes(window_size, spatial_dim)
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # relative position bias
        with torch.no_grad():
            idx = _build_rel_pos_index(self.ws)
        self.register_buffer("rel_pos_index", idx, persistent=False)
        self.rel_pos_bias = nn.Parameter(torch.zeros(_prod(2 * w - 1 for w in self.ws), num_heads))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

        self.scale = (dim // num_heads) ** -0.5

    def tokens_per_window(self) -> int:
        return _prod(self.ws)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B_, T, C) where T == prod(ws)
        """
        B_, T, C = x.shape
        T_w = self.tokens_per_window()
        assert T == T_w, f"T={T} must equal tokens-per-window={T_w} (window_size={self.ws})"

        qkv = self.qkv(x).reshape(B_, T, 3, self.num_heads, C // self.num_heads)
        q = qkv[:, :, 0].transpose(1, 2)  # (B_, H, T, Dh)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B_, H, T, T)

        # ensure index lives on the same device as the bias before advanced indexing
        idx_flat = self.rel_pos_index.view(-1).to(self.rel_pos_bias.device)
        bias = (
            self.rel_pos_bias[idx_flat]
            .view(T, T, self.num_heads)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(attn.dtype)
        )
        attn = (attn + bias).softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B_, T, C)
        return self.proj(out)
