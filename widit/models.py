from typing import Sequence

import torch
import torch.nn as nn

from .patch import PatchEmbed
from .blocks import WiditBlock, WiditFinalLayer
from .timesteps import TimestepEmbedder
from .window import _to_sizes, _prod

class Widit(nn.Module):
    """
    SwinIR-style DiT with N-D windowed attention (2D or 3D), no downsampling.

    Pipeline:
      two PatchEmbeds (x, conditioned) -> concat tokens -> depth×(WiditBlock) -> WiditFinalLayer -> unpatchify

    Args:
      spatial_dim: 2 for 2D, 3 for 3D
      input_size: kept for API parity; not required by forward
      patch_size: int or per-axis tuple
      in_channels: input channels
      hidden_size: token embedding dim (sum of x/y embed dims)
      depth, num_heads, window_size, mlp_ratio: transformer hyperparams
      learn_sigma: if True, predict mean+sigma (out_channels = 2*in_channels)
    """
    def __init__(
        self,
        *,
        spatial_dim: int,
        input_size: int | Sequence[int] | None = None,
        patch_size: int | Sequence[int] = 2,
        in_channels: int = 1,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        window_size: int | Sequence[int] = 8,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = True,
    ):
        super().__init__()
        assert spatial_dim in (2, 3), f"spatial_dim must be 2 or 3, got {spatial_dim}"
        self.k = spatial_dim
        self.in_channels = in_channels
        self.learn_sigma = learn_sigma
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = _to_sizes(patch_size, self.k)
        self.window_size = _to_sizes(window_size, self.k)

        # Split hidden_size evenly across x/y patch embeds
        half = hidden_size // 2
        assert half * 2 == hidden_size, "hidden_size must be even (x/y embeds split evenly)"

        assert (hidden_size % num_heads) == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"

        # Patch embedders (channels-first inputs)
        self.x_embed = PatchEmbed(
            input_size=input_size,
            patch_size=self.patch_size,
            in_chans=in_channels,
            embed_dim=half,
            bias=True,
            spatial_dim=self.k,
        )
        self.y_embed = PatchEmbed(
            input_size=input_size,
            patch_size=self.patch_size,
            in_chans=in_channels,
            embed_dim=half,
            bias=True,
            spatial_dim=self.k,
        )

        # Timestep embedder -> same dim as token dim (hidden_size)
        self.t_embed = TimestepEmbedder(hidden_size)

        # Blocks (Swin shift pattern: 0, ws//2, 0, ws//2, ...)
        blocks = []
        shift_even = (0,) * self.k
        shift_odd = tuple(w // 2 for w in self.window_size)
        for i in range(depth):
            shift = shift_even if (i % 2 == 0) else shift_odd
            blocks.append(
                WiditBlock(
                    dim=hidden_size,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                    spatial_dim=self.k,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # Head
        self.head = WiditFinalLayer(
            hidden_size=hidden_size,
            patch_size=self.patch_size[0] if self.k == 2 else self.patch_size[0],  # scalar p (same per-axis)
            out_channels=self.out_channels,
            spatial_dim=self.k,
        )

        self._init_weights()

    # ------------------- init -------------------

    def _init_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic)

        # Timestep MLP small init (matches common DiT setups)
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)

        # Zero-init ada last linear in each block (adaLN-Zero)
        for b in self.blocks:
            nn.init.constant_(b.ada[-1].weight, 0)
            nn.init.constant_(b.ada[-1].bias,  0)

        # Head: zero-init ada last and output linear
        nn.init.constant_(self.head.ada[-1].weight, 0)
        nn.init.constant_(self.head.ada[-1].bias,  0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias,   0)

        # PatchEmbed init: xavier on conv/timm proj weights, zero bias
        def _init_patch_embed(pe: PatchEmbed):
            # Build already happened in __init__ (since spatial_dim is fixed),
            # so exactly one of the paths must be present.
            if pe.patch_embedding_2d is not None:
                proj = pe.patch_embedding_2d.proj  # timm conv2d
                w = proj.weight.data
                nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
                if proj.bias is not None:
                    nn.init.constant_(proj.bias, 0)
            elif pe.patch_embedding_3d is not None:
                proj = pe.patch_embedding_3d       # conv3d
                w = proj.weight.data
                nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
                if proj.bias is not None:
                    nn.init.constant_(proj.bias, 0)
            else:
                # If user constructed with spatial_dim=None, the path might not be built yet.
                # In this model we always pass spatial_dim (2 or 3), so this shouldn't happen.
                pass

        _init_patch_embed(self.x_embed)
        _init_patch_embed(self.y_embed)

    def _unpatchify(self, x: torch.Tensor, spatial: tuple[int, ...]) -> torch.Tensor:
        """
        x:       (N, T, (p^k) * out_channels)
        spatial: (S1,...,Sk) original spatial sizes
        returns: (N, out_channels, *spatial)
        """
        N, T, PPc = x.shape
        p = self.patch_size
        c = self.out_channels
        k = self.k
        assert all(p[i] == p[0] for i in range(k)), "unpatchify assumes equal patch along each axis"
        p_scalar = p[0]

        # grid sizes in tokens
        grid = tuple(Si // p_scalar for Si in spatial)
        assert _prod(grid) == T, f"Token count mismatch in unpatchify: prod{grid} != {T}"
        assert PPc == (p_scalar ** k) * c, f"Last dim should be p^k * c, got {PPc} vs {(p_scalar ** k) * c}"

        # reshape to (N, g1,...,gk, p,...,p, c)
        x = x.view(N, *grid, *([p_scalar] * k), c)

        # permute to (N, c, g1, p, g2, p, ..., gk, p)
        # then merge (gi, p) -> Si for each axis
        # Build permutation programmatically:
        # current dims: [N] + [g1..gk] + [p1..pk] + [c]
        # target order: N, c, g1, p1, g2, p2, ..., gk, pk
        perm = [0, 1 + 2 * k]  # N, c
        for i in range(k):
            perm.extend([1 + i, 1 + k + i])
        x = x.permute(*perm).contiguous()

        # merge pairs
        out_shape = [N, c]
        for i in range(k):
            out_shape.append(grid[i] * p_scalar)
        return x.view(*out_shape)

    def forward(self, x: torch.Tensor, t: torch.Tensor, conditioned: torch.Tensor) -> torch.Tensor:
        """
        x, conditioned: (N, C, *spatial)  where len(spatial) == k
        t:              (N,)
        returns:        (N, out_channels, *spatial)
        """
        assert x.ndim == 2 + self.k, f"x has incorrect shape {tuple(x.shape)}"
        assert conditioned.shape == x.shape, "`conditioned` must match x shape"

        N = x.shape[0]
        spatial = tuple(x.shape[2 + i] for i in range(self.k))

        # Patch embed both inputs and concat along channel dim (token dim)
        ex = self.x_embed(x)               # (N, T, hidden/2)
        ey = self.y_embed(conditioned)     # (N, T, hidden/2)
        z = torch.cat([ex, ey], dim=-1)    # (N, T, hidden)

        # Token grid sizes (#tokens along each axis)
        p = self.patch_size
        token_grid = tuple(Si // p[0] for Si in spatial)  # equal p enforced in unpatchify
        assert _prod(token_grid) == z.shape[1], f"Token count mismatch: prod{token_grid} vs {z.shape[1]}"

        # Timestep conditioning
        c = self.t_embed(t)                # (N, hidden)

        # Transformer blocks
        for blk in self.blocks:
            z = blk(z, c, *token_grid)

        # Head and unpatchify
        out_tokens = self.head(z, c)       # (N, T, p^k * out_channels)
        return self._unpatchify(out_tokens, spatial)


# ---- WiDiT presets ----
def WiDiT2D_B_2(**kw):   return Widit(spatial_dim=2, depth=12, hidden_size=768,   patch_size=2, num_heads=12, **kw)
def WiDiT2D_M_2(**kw):   return Widit(spatial_dim=2, depth=12, hidden_size=1024,  patch_size=2, num_heads=16, **kw)
def WiDiT2D_L_2(**kw):   return Widit(spatial_dim=2, depth=24, hidden_size=1024,  patch_size=2, num_heads=16, **kw)
def WiDiT2D_XL_2(**kw):  return Widit(spatial_dim=2, depth=28, hidden_size=1152,  patch_size=2, num_heads=16, **kw)

def WiDiT3D_B_2(**kw):   return Widit(spatial_dim=3, depth=12, hidden_size=768,   patch_size=2, num_heads=12, **kw)
def WiDiT3D_M_2(**kw):   return Widit(spatial_dim=3, depth=12, hidden_size=1024,  patch_size=2, num_heads=16, **kw)
def WiDiT3D_L_2(**kw):   return Widit(spatial_dim=3, depth=24, hidden_size=1024,  patch_size=2, num_heads=16, **kw)
def WiDiT3D_XL_2(**kw):  return Widit(spatial_dim=3, depth=28, hidden_size=1152,  patch_size=2, num_heads=16, **kw)


PRESETS = {
    # 2D
    "WiDiT-B/2":  WiDiT2D_B_2,
    "WiDiT-M/2":  WiDiT2D_M_2,
    "WiDiT-L/2":  WiDiT2D_L_2,
    "WiDiT-XL/2": WiDiT2D_XL_2,
    # 3D
    "WiDiT3D-B/2":  WiDiT3D_B_2,
    "WiDiT3D-M/2":  WiDiT3D_M_2,
    "WiDiT3D-L/2":  WiDiT3D_L_2,
    "WiDiT3D-XL/2": WiDiT3D_XL_2,
}
