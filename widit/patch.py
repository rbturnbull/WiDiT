from typing import Sequence

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed as PatchEmbed2D


def _to_tuple(p: int | Sequence[int], k: int) -> tuple[int, ...]:
    if isinstance(p, int):
        return (p,) * k
    p = tuple(p)
    assert len(p) == k, f"expected {k} elements, got {len(p)}"
    return p


class PatchEmbed(nn.Module):
    """
    ND Patch Embedding that:
      • delegates to timm PatchEmbed for 2D inputs: (N, C, H, W)
      • uses Conv3d for 3D inputs: (N, C, D, H, W)

    Args:
      input_size: kept for API parity with timm; not strictly required for forward
      patch_size: int or tuple; if int, broadcast to all spatial dims
      in_chans:   input channels (C)
      embed_dim:  output channels per token
      bias:       bias for conv/linear proj
      spatial_dim: 2 or 3. If None, inferred at forward from input rank.

    Returns (forward):
      (N, T, embed_dim) where T is number of patches (prod over spatial dims / prod(patch_size))
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

        # Lazy init: build the concrete path (2D or 3D) only when needed.
        self._pe2d: PatchEmbed2D | None = None
        self._pe3d: nn.Conv3d | None = None

        # If spatial_dim is fixed, we can eagerly build just that path.
        if spatial_dim == 2:
            self._build_2d()
        elif spatial_dim == 3:
            self._build_3d()
        elif spatial_dim is not None:
            raise ValueError(f"spatial_dim must be 2, 3, or None; got {spatial_dim}")

    # --- builders ---

    def _build_2d(self) -> None:
        if self._pe2d is not None:
            return
        p2 = _to_tuple(self.patch_size, 2)
        img_size = self.input_size if isinstance(self.input_size, (int, tuple, list)) else None
        self._pe2d = PatchEmbed2D(
            img_size=img_size,
            patch_size=p2,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            bias=self.bias,
            flatten=True,  # (N, T, D)
        )

    def _build_3d(self) -> None:
        if self._pe3d is not None:
            return
        p3 = _to_tuple(self.patch_size, 3)
        self._pe3d = nn.Conv3d(
            self.in_chans, self.embed_dim, kernel_size=p3, stride=p3, bias=self.bias
        )

    # --- forwards ---

    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        assert self._pe2d is not None, "2D path not initialized"
        return self._pe2d(x)  # (N, T, D)

    def _forward_3d(self, x: torch.Tensor) -> torch.Tensor:
        assert self._pe3d is not None, "3D path not initialized"
        x = self._pe3d(x)                    # (N, Dm, D/pd, H/ph, W/pw)
        N, Dm, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)     # (N, T, Dm)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Determine spatial dimensionality
        if self.spatial_dim is None:
            if x.ndim == 4:   # N,C,H,W
                sd = 2
            elif x.ndim == 5: # N,C,D,H,W
                sd = 3
            else:
                raise ValueError(f"Unsupported input rank {x.ndim}; expected 4 (2D) or 5 (3D).")
        else:
            sd = self.spatial_dim

        if sd == 2:
            assert x.ndim == 4, f"Expected (N,C,H,W) for 2D, got shape {tuple(x.shape)}"
            if self._pe2d is None:
                self._build_2d()
            return self._forward_2d(x)
        elif sd == 3:
            assert x.ndim == 5, f"Expected (N,C,D,H,W) for 3D, got shape {tuple(x.shape)}"
            if self._pe3d is None:
                self._build_3d()
            return self._forward_3d(x)
        else:
            raise ValueError(f"spatial_dim must be 2 or 3, got {sd}")
