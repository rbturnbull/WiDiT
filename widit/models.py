from typing import Sequence
from pathlib import Path

import torch
import torch.nn as nn

from .patch import PatchEmbed
from .blocks import WiDiTBlock, WiDiTFinalLayer
from .timesteps import TimestepEmbedder
from .window import _to_sizes, _prod


class WiDiT(nn.Module):
    """
    SwinIR-style DiT with N-D windowed attention (2D or 3D), no downsampling.

    If `use_conditioning=True`, the model expects an additional conditioning image/volume
    and uses two PatchEmbed streams concatenated along the token channel.
    If `use_conditioning=False`, only the main input stream is used.
    """

    def __init__(
        self,
        *,
        spatial_dim: int,
        input_size: int | Sequence[int] | None = None,
        patch_size: int | Sequence[int] = 2,
        in_channels: int = 1,
        out_channels: int|None = None,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        window_size: int | Sequence[int] = 4,
        mlp_ratio: float = 4.0,
        use_conditioning: bool = True,
        use_flash_attention: bool = False,
        **kwargs,
    ):
        super().__init__()

        # Store config to be able to recreate model later
        self.config = dict(
            spatial_dim=spatial_dim,
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            out_channels=out_channels,
            use_conditioning=use_conditioning,
            use_flash_attention=use_flash_attention,
        )

        assert spatial_dim in (2, 3), f"spatial_dim must be 2 or 3, got {spatial_dim}"

        # Core hyperparameters
        self.spatial_dims = spatial_dim
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conditioning = use_conditioning
        self.use_flash_attention = use_flash_attention

        # Normalize per-axis hyperparameters
        self.patch_size_per_axis = _to_sizes(patch_size, self.spatial_dims)
        self.window_size_per_axis = _to_sizes(window_size, self.spatial_dims)

        # Attention sanity check
        assert (hidden_size % num_heads) == 0, (
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        )

        # Patch embedders (channels-first inputs)
        if self.use_conditioning:
            # split token dim evenly across the two streams
            assert (hidden_size % 2) == 0, "hidden_size must be even when use_conditioning=True"
            half_hidden = hidden_size // 2
            self.input_patch_embed = PatchEmbed(
                input_size=input_size,
                patch_size=self.patch_size_per_axis,
                in_chans=in_channels,
                embed_dim=half_hidden,
                bias=True,
                spatial_dim=self.spatial_dims,
            )
            self.conditioned_patch_embed = PatchEmbed(
                input_size=input_size,
                patch_size=self.patch_size_per_axis,
                in_chans=in_channels,
                embed_dim=half_hidden,
                bias=True,
                spatial_dim=self.spatial_dims,
            )
        else:
            # single stream uses full hidden_size
            self.input_patch_embed = PatchEmbed(
                input_size=input_size,
                patch_size=self.patch_size_per_axis,
                in_chans=in_channels,
                embed_dim=hidden_size,
                bias=True,
                spatial_dim=self.spatial_dims,
            )
            self.conditioned_patch_embed = None  # explicitly absent

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
                use_flash_attention=self.use_flash_attention,
            )
            for i in range(depth)
        ])

        # Final projection head (equal patch per axis enforced in unpatchify)
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

        self.timestep_embedder.init_weights()
        self.input_patch_embed.init_weights()
        if self.conditioned_patch_embed is not None:
            self.conditioned_patch_embed.init_weights()
        for block in self.blocks:
            block.init_weights()
        self.final.init_weights()

    def _unpatchify(self, token_tensor: torch.Tensor, spatial_sizes: tuple[int, ...]) -> torch.Tensor:
        batch_size, num_tokens, last_dim = token_tensor.shape
        patch_sizes = self.patch_size_per_axis
        out_channels = self.out_channels
        k = self.spatial_dims

        assert all(p == patch_sizes[0] for p in patch_sizes), "unpatchify assumes equal patch along each axis"
        patch_scalar = patch_sizes[0]

        token_grid_sizes = tuple(Si // patch_scalar for Si in spatial_sizes)
        assert _prod(token_grid_sizes) == num_tokens, \
            f"Token count mismatch in unpatchify: prod{token_grid_sizes} != {num_tokens}"
        assert last_dim == (patch_scalar ** k) * out_channels, \
            f"Last dim should be p^k * out_channels, got {last_dim} vs {(patch_scalar ** k) * out_channels}"

        x = token_tensor.view(batch_size, *token_grid_sizes, *([patch_scalar] * k), out_channels)
        perm = [0, 1 + 2 * k]  # N, C_out
        for i in range(k):
            perm.extend([1 + i, 1 + k + i])
        x = x.permute(*perm).contiguous()

        output_shape = [batch_size, out_channels]
        for i in range(k):
            output_shape.append(token_grid_sizes[i] * patch_scalar)
        return x.view(*output_shape)

    def forward(
        self,
        input_tensor: torch.Tensor,
        timestep: torch.Tensor | None = None,
        *,
        conditioned: torch.Tensor | None = None,
        **_: dict,
    ) -> torch.Tensor:
        """
        Args:
          input_tensor: (N, C, *spatial)
          timestep:     (N,) or None — optional diffusion timestep (or other scalar schedule)
          conditioned:  optional (N, C, *spatial) — REQUIRED if `use_conditioning=True`,
                        must be omitted if `use_conditioning=False`.

        Returns:
          (N, out_channels, *spatial)
        """
        assert input_tensor.ndim == 2 + self.spatial_dims, \
            f"input_tensor has incorrect shape {tuple(input_tensor.shape)}"

        if self.use_conditioning:
            assert conditioned is not None, \
                "This model was constructed with use_conditioning=True, but `conditioned` was not provided."
            assert conditioned.shape == input_tensor.shape, \
                "`conditioned` must match `input_tensor` shape"
        else:
            assert conditioned is None, \
                "This model was constructed with use_conditioning=False; do not pass `conditioned`."

        batch_size = input_tensor.shape[0]
        spatial_sizes = tuple(input_tensor.shape[2 + i] for i in range(self.spatial_dims))

        # Patch-embed and (optionally) concatenate
        tokens_input = self.input_patch_embed(input_tensor)  # (N, T, H or H/2)
        if self.use_conditioning:
            tokens_cond = self.conditioned_patch_embed(conditioned)        # (N, T, H/2)
            tokens = torch.cat([tokens_input, tokens_cond], dim=-1)        # (N, T, H)
        else:
            tokens = tokens_input                                          # (N, T, H)

        # Token grid (per axis, in tokens)
        patch_scalar = self.patch_size_per_axis[0]
        token_grid_sizes = tuple(Si // patch_scalar for Si in spatial_sizes)
        assert _prod(token_grid_sizes) == tokens.shape[1], \
            f"Token count mismatch: prod{token_grid_sizes} vs {tokens.shape[1]}"

        # Optional timestep conditioning
        timestep_embedding = None
        if timestep is not None:
            timestep_embedding = self.timestep_embedder(timestep)          # (N, hidden)
            assert timestep_embedding.shape == (batch_size, tokens.shape[-1]), \
                f"timestep embedding shape {tuple(timestep_embedding.shape)} must be (N, hidden={tokens.shape[-1]})"

        # Transformer backbone
        for block in self.blocks:
            tokens = block(tokens, timestep_embedding, *token_grid_sizes)

        # Head & unpatchify
        out_tokens = self.final(tokens, timestep_embedding)                 # (N, T, p^k * out_channels)
        return self._unpatchify(out_tokens, spatial_sizes)

    def save(self, path: str|Path):
        """Save model weights and configuration to a single file."""
        obj = {
            "model_state": self.state_dict(),
            "config": self.config,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, path)

    @classmethod
    def load(cls, path: str, map_location="cpu"):
        """Load WiDiT model and weights from file."""
        checkpoint = torch.load(path, map_location=map_location)
        config = checkpoint["config"]
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state"])
        return model


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
    "WiDiT2D-B/2":  WiDiT2D_B_2,
    "WiDiT2D-M/2":  WiDiT2D_M_2,
    "WiDiT2D-L/2":  WiDiT2D_L_2,
    "WiDiT2D-XL/2": WiDiT2D_XL_2,
    # 3D
    "WiDiT3D-B/2":  WiDiT3D_B_2,
    "WiDiT3D-M/2":  WiDiT3D_M_2,
    "WiDiT3D-L/2":  WiDiT3D_L_2,
    "WiDiT3D-XL/2": WiDiT3D_XL_2,
}
