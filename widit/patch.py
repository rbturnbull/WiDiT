from typing import Sequence

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed as PatchEmbed2D


def _to_tuple(patch_size: int | Sequence[int], expected_dims: int) -> tuple[int, ...]:
    if isinstance(patch_size, int):
        return (patch_size,) * expected_dims
    patch_size = tuple(patch_size)
    assert len(patch_size) == expected_dims, f"expected {expected_dims} elements, got {len(patch_size)}"
    return patch_size


class PatchEmbed(nn.Module):
    """
    ND Patch Embedding that:
      • delegates to timm PatchEmbed for 2D inputs: (N, C, H, W)
      • uses Conv3d for 3D inputs: (N, C, D, H, W)

    Args:
      input_size: API parity with timm; not strictly required for forward
      patch_size: int or tuple; if int, broadcast to all spatial dims
      in_chans:   input channels (C)
      embed_dim:  output channels per token
      bias:       bias for conv/linear proj
      spatial_dim: 2 or 3. If None, inferred at forward from input rank.

    Returns (forward):
      (N, T, embed_dim) where T is number of patches
      (product over spatial dims / product over patch_size)
    """
    def __init__(
        self,
        input_size: int | Sequence[int] | None,
        patch_size: int | Sequence[int],
        in_chans: int,
        embed_dim: int,
        bias: bool = True,
        spatial_dim: int | None = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.bias = bias
        self.spatial_dim = spatial_dim

        # Lazy init: build only the required path (2D or 3D) when needed.
        self.patch_embedding_2d: PatchEmbed2D | None = None
        self.patch_embedding_3d: nn.Conv3d | None = None

        # If spatial_dim is fixed, eagerly build the corresponding path.
        if spatial_dim == 2:
            self._build_2d_patch_embedding()
        elif spatial_dim == 3:
            self._build_3d_patch_embedding()
        elif spatial_dim is not None:
            raise ValueError(f"spatial_dim must be 2, 3, or None; got {spatial_dim}")

    # --- builders ---

    def _build_2d_patch_embedding(self) -> None:
        if self.patch_embedding_2d is not None:
            return
        patch_size_2d = _to_tuple(self.patch_size, 2)
        img_size = self.input_size if isinstance(self.input_size, (int, tuple, list)) else None
        self.patch_embedding_2d = PatchEmbed2D(
            img_size=img_size,
            patch_size=patch_size_2d,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            bias=self.bias,
            flatten=True,  # (N, T, embed_dim)
        )

    def _build_3d_patch_embedding(self) -> None:
        if self.patch_embedding_3d is not None:
            return
        patch_size_3d = _to_tuple(self.patch_size, 3)
        self.patch_embedding_3d = nn.Conv3d(
            self.in_chans, self.embed_dim, kernel_size=patch_size_3d, stride=patch_size_3d, bias=self.bias
        )

    # --- forwards ---

    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        assert self.patch_embedding_2d is not None, "2D patch embedding not initialized"
        return self.patch_embedding_2d(x)  # (N, T, embed_dim)

    def _forward_3d(self, x: torch.Tensor) -> torch.Tensor:
        assert self.patch_embedding_3d is not None, "3D patch embedding not initialized"
        x = self.patch_embedding_3d(x)             # (N, embed_dim, D/pd, H/ph, W/pw)
        N, embed_dim, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)           # (N, T, embed_dim)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Determine spatial dimensionality for this input
        if self.spatial_dim is None:
            if x.ndim == 4:   # (N, C, H, W)
                spatial_dims = 2
            elif x.ndim == 5: # (N, C, D, H, W)
                spatial_dims = 3
            else:
                raise ValueError(f"Unsupported input rank {x.ndim}; expected 4 (2D) or 5 (3D).")
        else:
            spatial_dims = self.spatial_dim

        if spatial_dims == 2:
            assert x.ndim == 4, f"Expected (N,C,H,W) for 2D, got shape {tuple(x.shape)}"
            if self.patch_embedding_2d is None:
                self._build_2d_patch_embedding()
            return self._forward_2d(x)

        if spatial_dims == 3:
            assert x.ndim == 5, f"Expected (N,C,D,H,W) for 3D, got shape {tuple(x.shape)}"
            if self.patch_embedding_3d is None:
                self._build_3d_patch_embedding()
            return self._forward_3d(x)

        raise ValueError(f"spatial_dim must be 2 or 3, got {spatial_dims}")
