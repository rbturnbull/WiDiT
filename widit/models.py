from typing import Sequence

import torch
import torch.nn as nn

from .patch import PatchEmbed
from .blocks import WiDiTBlock, WiDiTFinalLayer
from .timesteps import TimestepEmbedder
from .window import _to_sizes, _prod


class WiDiT(nn.Module):
    """
    SwinIR-style DiT with N-D windowed attention (2D or 3D), no downsampling.

    Pipeline:
      two PatchEmbeds (input, conditioned) -> concat tokens
      -> depth × (WiDiTBlock)
      -> WiDiTFinalLayer
      -> unpatchify back to image/volume

    Args:
      spatial_dim:   2 for 2D, 3 for 3D
      input_size:    kept for API parity; not required by forward
      patch_size:    int or per-axis tuple
      in_channels:   input channels
      hidden_size:   token embedding dim (sum of input/conditioned embed dims)
      depth:         number of transformer blocks
      num_heads:     attention heads
      window_size:   int or per-axis tuple for window attention
      mlp_ratio:     MLP hidden multiplier
      learn_sigma:   if True, predict mean+sigma (out_channels = 2*in_channels)
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

        # Core hyperparameters
        self.spatial_dims = spatial_dim
        self.in_channels = in_channels
        self.learn_sigma = learn_sigma
        self.out_channels = in_channels * 2 if learn_sigma else in_channels

        # Normalize per-axis hyperparameters
        self.patch_size_per_axis = _to_sizes(patch_size, self.spatial_dims)
        self.window_size_per_axis = _to_sizes(window_size, self.spatial_dims)

        # Embedding split sanity checks
        half_hidden = hidden_size // 2
        assert half_hidden * 2 == hidden_size, "hidden_size must be even (split evenly across the two patch embeds)."
        assert (hidden_size % num_heads) == 0, (
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        )

        # Patch embedders (channels-first inputs)
        self.input_patch_embed = PatchEmbed(
            input_size=input_size,
            patch_size=self.patch_size_per_axis,
            in_chans=in_channels,
            embed_dim=half_hidden,
            bias=True,
        )
        self.conditioned_patch_embed = PatchEmbed(
            input_size=input_size,
            patch_size=self.patch_size_per_axis,
            in_chans=in_channels,
            embed_dim=half_hidden,
            bias=True,
        )

        # Optional conditioning via timestep embedding → match token dim (hidden_size)
        self.timestep_embedder = TimestepEmbedder(hidden_size)

        # Transformer blocks (Swin shift pattern: 0, ws//2, 0, ws//2, ...)
        shift_none = (0,) * self.spatial_dims
        shift_half = tuple(w // 2 for w in self.window_size_per_axis)

        self.blocks = nn.ModuleList([
            WiDiTBlock(
                dim=hidden_size,
                num_heads=num_heads,
                window_size=self.window_size_per_axis,
                shift_size=(shift_none if (i % 2 == 0) else shift_half),
                mlp_ratio=mlp_ratio,
                spatial_dim=self.spatial_dims,
            )
            for i in range(depth)
        ])

        # Final projection head
        # (We currently enforce equal patch size along each axis when unpatchifying.)
        patch_scalar = self.patch_size_per_axis[0]
        self.final = WiDiTFinalLayer(
            hidden_size=hidden_size,
            patch_size=patch_scalar,
            out_channels=self.out_channels,
            spatial_dim=self.spatial_dims,
        )

        self.init_weights()

    def init_weights(self) -> None:
        def _xavier_linear(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_xavier_linear)

        # Timestep MLP small init
        self.timestep_embedder.init_weights()

        self.input_patch_embed.init_weights()
        self.conditioned_patch_embed.init_weights()
        
        for block in self.blocks:
            block.init_weights()
        
        self.final.init_weights()


    def _unpatchify(self, token_tensor: torch.Tensor, spatial_sizes: tuple[int, ...]) -> torch.Tensor:
        """
        Args:
          token_tensor:  (N, T, (p^k) * out_channels)
          spatial_sizes: (S1, ..., Sk) original spatial sizes

        Returns:
          (N, out_channels, *spatial_sizes)
        """
        batch_size, num_tokens, last_dim = token_tensor.shape
        patch_sizes = self.patch_size_per_axis
        out_channels = self.out_channels
        k = self.spatial_dims

        # enforce equal patch along each axis for now
        assert all(p == patch_sizes[0] for p in patch_sizes), \
            "unpatchify assumes equal patch along each axis"
        patch_scalar = patch_sizes[0]

        # token grid sizes along each axis
        token_grid_sizes = tuple(Si // patch_scalar for Si in spatial_sizes)
        assert _prod(token_grid_sizes) == num_tokens, \
            f"Token count mismatch in unpatchify: prod{token_grid_sizes} != {num_tokens}"
        assert last_dim == (patch_scalar ** k) * out_channels, \
            f"Last dim should be p^k * out_channels, got {last_dim} vs {(patch_scalar ** k) * out_channels}"

        # (N, g1,...,gk, p,...,p, C_out)
        x = token_tensor.view(batch_size, *token_grid_sizes, *([patch_scalar] * k), out_channels)

        # Permute to (N, C_out, g1, p, g2, p, ..., gk, p)
        perm = [0, 1 + 2 * k]  # N, C_out
        for i in range(k):
            perm.extend([1 + i, 1 + k + i])
        x = x.permute(*perm).contiguous()

        # Merge (gi, p) -> Si
        output_shape = [batch_size, out_channels]
        for i in range(k):
            output_shape.append(token_grid_sizes[i] * patch_scalar)
        return x.view(*output_shape)

    # ------------------- forward -------------------

    def forward(
        self,
        input_tensor: torch.Tensor,
        conditioned_tensor: torch.Tensor,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
          input_tensor:       (N, C, *spatial)
          conditioned_tensor: (N, C, *spatial), must match input_tensor shape
          timestep:           (N,) or None — optional conditioning (e.g., diffusion timestep)

        Returns:
          (N, out_channels, *spatial)
        """
        # Basic shape checks
        assert input_tensor.ndim == 2 + self.spatial_dims, \
            f"input_tensor has incorrect shape {tuple(input_tensor.shape)}"
        assert conditioned_tensor.shape == input_tensor.shape, \
            "`conditioned_tensor` must match `input_tensor` shape"

        batch_size = input_tensor.shape[0]
        spatial_sizes = tuple(input_tensor.shape[2 + i] for i in range(self.spatial_dims))

        # Patch-embed inputs and concatenate along token channel dim
        tokens_input = self.input_patch_embed(input_tensor)              # (N, T, hidden/2)
        tokens_conditioned = self.conditioned_patch_embed(conditioned_tensor)  # (N, T, hidden/2)
        tokens = torch.cat([tokens_input, tokens_conditioned], dim=-1)  # (N, T, hidden)

        # Token grid sizes along each axis (in tokens, not pixels/voxels)
        patch_scalar = self.patch_size_per_axis[0]  # equal p enforced by unpatchify
        token_grid_sizes = tuple(Si // patch_scalar for Si in spatial_sizes)
        assert _prod(token_grid_sizes) == tokens.shape[1], \
            f"Token count mismatch: prod{token_grid_sizes} vs {tokens.shape[1]}"

        # Optional timestep conditioning → per-sample vector matching token dim
        timestep_embedding = None
        if timestep is not None:
            timestep_embedding = self.timestep_embedder(timestep)  # (N, hidden)
            assert timestep_embedding.shape == (batch_size, tokens.shape[-1]), \
                f"timestep embedding shape {tuple(timestep_embedding.shape)} must be (N, hidden={tokens.shape[-1]})"

        # Transformer backbone
        for block in self.blocks:
            tokens = block(tokens, timestep_embedding, *token_grid_sizes)

        # Final projection and unpatchify
        out_tokens = self.final(tokens, timestep_embedding)  # (N, T, p^k * out_channels)
        return self._unpatchify(out_tokens, spatial_sizes)


# ---- WiDiT presets ----
def WiDiT2D_B_2(**kw):   return WiDiT(spatial_dim=2, depth=12, hidden_size=768,   patch_size=2, num_heads=12, **kw)
def WiDiT2D_M_2(**kw):   return WiDiT(spatial_dim=2, depth=12, hidden_size=1024,  patch_size=2, num_heads=16, **kw)
def WiDiT2D_L_2(**kw):   return WiDiT(spatial_dim=2, depth=24, hidden_size=1024,  patch_size=2, num_heads=16, **kw)
def WiDiT2D_XL_2(**kw):  return WiDiT(spatial_dim=2, depth=28, hidden_size=1152,  patch_size=2, num_heads=16, **kw)

def WiDiT3D_B_2(**kw):   return WiDiT(spatial_dim=3, depth=12, hidden_size=768,   patch_size=2, num_heads=12, **kw)
def WiDiT3D_M_2(**kw):   return WiDiT(spatial_dim=3, depth=12, hidden_size=1024,  patch_size=2, num_heads=16, **kw)
def WiDiT3D_L_2(**kw):   return WiDiT(spatial_dim=3, depth=24, hidden_size=1024,  patch_size=2, num_heads=16, **kw)
def WiDiT3D_XL_2(**kw):  return WiDiT(spatial_dim=3, depth=28, hidden_size=1152,  patch_size=2, num_heads=16, **kw)

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
