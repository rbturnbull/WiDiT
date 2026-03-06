from typing import Sequence
from abc import ABC
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .patch import PatchEmbed
from .blocks import WiDiTBlock, WiDiTFinalLayer
from .timesteps import TimestepEmbedder
from .window import _to_sizes, _prod


class ModelBase(nn.Module, ABC):
    def save(self, path: str | Path) -> None:
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
        """Load model and weights from file."""
        checkpoint = torch.load(path, map_location=map_location)
        config = checkpoint["config"]
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state"])
        return model


class WiDiT(ModelBase):
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
        use_flash_attention: bool = True,
        timestep_embed_dim: int | None = None,
        **kwargs,
    ):
        super().__init__()

        # Store config to be able to recreate model later
        resolved_timestep_embed_dim = timestep_embed_dim or hidden_size

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
            timestep_embed_dim=resolved_timestep_embed_dim,
        )

        assert spatial_dim in (2, 3), f"spatial_dim must be 2 or 3, got {spatial_dim}"

        # Core hyperparameters
        self.spatial_dim = spatial_dim
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conditioning = use_conditioning
        self.use_flash_attention = use_flash_attention

        # Normalize per-axis hyperparameters
        self.patch_size_per_axis = _to_sizes(patch_size, self.spatial_dim)
        self.window_size_per_axis = _to_sizes(window_size, self.spatial_dim)

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
                spatial_dim=self.spatial_dim,
            )
            self.conditioned_patch_embed = PatchEmbed(
                input_size=input_size,
                patch_size=self.patch_size_per_axis,
                in_chans=in_channels,
                embed_dim=half_hidden,
                bias=True,
                spatial_dim=self.spatial_dim,
            )
        else:
            # single stream uses full hidden_size
            self.input_patch_embed = PatchEmbed(
                input_size=input_size,
                patch_size=self.patch_size_per_axis,
                in_chans=in_channels,
                embed_dim=hidden_size,
                bias=True,
                spatial_dim=self.spatial_dim,
            )
            self.conditioned_patch_embed = None  # explicitly absent

        # Optional conditioning via timestep embedding → match token dim (hidden_size)
        self.timestep_embed_dim = resolved_timestep_embed_dim
        self.timestep_embedder = TimestepEmbedder(self.timestep_embed_dim)
        self.timestep_in_proj = None
        if self.timestep_embed_dim != hidden_size:
            self.timestep_in_proj = nn.Linear(self.timestep_embed_dim, hidden_size)

        # Transformer blocks (Swin shift pattern: 0, ws//2, 0, ws//2, ...)
        shift_none = (0,) * self.spatial_dim
        shift_half = tuple(w // 2 for w in self.window_size_per_axis)
        self.blocks = nn.ModuleList([
            WiDiTBlock(
                dim=hidden_size,
                num_heads=num_heads,
                window_size=self.window_size_per_axis,
                shift_size=(shift_none if (i % 2 == 0) else shift_half),
                mlp_ratio=mlp_ratio,
                spatial_dim=self.spatial_dim,
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
            spatial_dim=self.spatial_dim,
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
        k = self.spatial_dim

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
        assert input_tensor.ndim == 2 + self.spatial_dim, \
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
        spatial_sizes = tuple(input_tensor.shape[2 + i] for i in range(self.spatial_dim))

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
            timestep_embedding = self.timestep_embedder(timestep)          # (N, embed_dim)
            assert timestep_embedding.shape == (batch_size, self.timestep_embed_dim), \
                f"timestep embedding shape {tuple(timestep_embedding.shape)} must be (N, hidden={self.timestep_embed_dim})"
            if self.timestep_in_proj is not None:
                timestep_embedding = self.timestep_in_proj(timestep_embedding)
            assert timestep_embedding.shape == (batch_size, tokens.shape[-1]), \
                f"timestep embedding shape {tuple(timestep_embedding.shape)} must be (N, hidden={tokens.shape[-1]})"

        # Transformer backbone
        for block in self.blocks:
            tokens = block(tokens, timestep_embedding, *token_grid_sizes)

        # Head & unpatchify
        out_tokens = self.final(tokens, timestep_embedding)                 # (N, T, p^k * out_channels)
        return self._unpatchify(out_tokens, spatial_sizes)

    pass


def get_conv(spatial_dim: int):
    return nn.Conv3d if spatial_dim == 3 else nn.Conv2d


def get_maxpool(spatial_dim: int):
    return nn.MaxPool3d if spatial_dim == 3 else nn.MaxPool2d


def get_upsample_mode(spatial_dim: int):
    return "trilinear" if spatial_dim == 3 else "bilinear"


def conv(
    spatial_dim: int,
    in_channels: int,
    filters: int,
    kernel_size: int,
    padding: int,
    activation: nn.Module | None,
):
    Conv = get_conv(spatial_dim)

    layers = [
        Conv(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="reflect",
        )
    ]

    if activation is not None:
        layers.append(activation)

    return nn.Sequential(*layers)


def conv_block(
    spatial_dim: int,
    in_channels: int,
    filters: int,
    kernel_size: int,
    padding: int,
    activation: nn.Module,
):
    return nn.Sequential(
        conv(spatial_dim, in_channels, filters, kernel_size, padding, activation),
        conv(spatial_dim, filters, filters, kernel_size, padding, activation),
    )


def down_block(
    spatial_dim: int,
    in_channels: int,
    filters: int,
    kernel_size: int,
    padding: int,
    activation: nn.Module,
):
    MaxPool = get_maxpool(spatial_dim)

    return nn.Sequential(
        MaxPool(kernel_size=2, stride=2),
        conv_block(spatial_dim, in_channels, filters, kernel_size, padding, activation),
    )


def up_block(
    spatial_dim: int,
    in_channels: int,
    filters: int,
    kernel_size: int,
    padding: int,
    activation: nn.Module,
):
    mode = get_upsample_mode(spatial_dim)

    return nn.Sequential(
        conv_block(spatial_dim, in_channels, filters, kernel_size, padding, activation),
        nn.Upsample(scale_factor=2, mode=mode, align_corners=False),
    )


class Unet(ModelBase):
    def __init__(
        self,
        *,
        in_channels: int,
        filters: int,
        kernel_size: int | Sequence[int],
        layers: int,
        spatial_dim: int = 3,
        out_channels: int|None = None,
        use_conditioning: bool = True,
        timestep_embed_dim: int | None = None,
    ):
        super().__init__()

        # Store config to be able to recreate model later
        self.config = dict(
            in_channels=in_channels,
            filters=filters,
            kernel_size=kernel_size,
            layers=layers,
            spatial_dim=spatial_dim,
            out_channels=out_channels,
            use_conditioning=use_conditioning,
        )

        assert spatial_dim in (2, 3), "spatial_dim must be 2 or 3"
        assert layers > 0, "Layers must be positive"

        out_channels = out_channels or in_channels
        self.out_channels = out_channels
        self.in_channels = in_channels

        if use_conditioning:
            in_channels *= 2  # Concatenate conditioning channel
        self.use_conditioning = use_conditioning

        self.layers = layers
        self.spatial_dim = spatial_dim
        self.kernel_size_per_axis = _to_sizes(kernel_size, self.spatial_dim)
        padding = tuple(k // 2 for k in self.kernel_size_per_axis)
        act = nn.ReLU()

        # Initial conv
        self.inconv = conv_block(
            spatial_dim,
            in_channels,
            filters,
            self.kernel_size_per_axis,
            padding,
            act,
        )

        # Timestep embedding (shared size projected per block)
        self.timestep_embed_dim = timestep_embed_dim or (filters * 4)
        self.config["timestep_embed_dim"] = self.timestep_embed_dim
        self.timestep_embedder = TimestepEmbedder(self.timestep_embed_dim)

        self.in_time_proj = nn.Linear(self.timestep_embed_dim, filters)

        # Down path
        self.down_blocks = nn.ModuleList()
        self.down_time_projs = nn.ModuleList()
        for i in range(layers):
            in_f = filters * (2**i)
            out_f = in_f * 2
            self.down_blocks.append(
                down_block(
                    spatial_dim,
                    in_f,
                    out_f,
                    self.kernel_size_per_axis,
                    padding,
                    act,
                )
            )
            self.down_time_projs.append(nn.Linear(self.timestep_embed_dim, out_f))

        # Bottleneck upsample
        self.bottleneck = nn.Upsample(
            scale_factor=2,
            mode=get_upsample_mode(spatial_dim),
            align_corners=False,
        )
        self.bottleneck_time_proj = nn.Linear(
            self.timestep_embed_dim, filters * (2**layers)
        )

        # Up path
        self.up_blocks = nn.ModuleList()
        self.up_time_projs = nn.ModuleList()
        for i in range(layers, 1, -1):
            out_f = filters * (2 ** (i - 1))
            in_f = filters * (2**i) + out_f
            self.up_blocks.append(
                up_block(
                    spatial_dim,
                    in_f,
                    out_f,
                    self.kernel_size_per_axis,
                    padding,
                    act,
                )
            )
            self.up_time_projs.append(nn.Linear(self.timestep_embed_dim, out_f))

        # Output
        self.outconv = nn.Sequential(
            conv(
                spatial_dim,
                filters * 3,
                filters,
                self.kernel_size_per_axis,
                padding,
                act,
            ),
            conv(
                spatial_dim,
                filters,
                out_channels,
                kernel_size=1,
                padding=0,
                activation=None,
            ),
        )
        self.out_time_proj = nn.Linear(self.timestep_embed_dim, filters * 3)

        self.init_weights()

    def init_weights(self) -> None:
        def _xavier_linear(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_xavier_linear)
        self.timestep_embedder.init_weights()

    def _add_timestep(
        self,
        x: torch.Tensor,
        timestep_embedding: torch.Tensor | None,
        projection: nn.Linear | None,
    ) -> torch.Tensor:
        if timestep_embedding is None or projection is None:
            return x

        temb = projection(timestep_embedding)
        temb = temb.view(temb.shape[0], temb.shape[1], *([1] * (x.ndim - 2)))
        return x + temb

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
        assert input_tensor.ndim == 2 + self.spatial_dim, \
            f"input_tensor has incorrect shape {tuple(input_tensor.shape)}"

        if self.use_conditioning:
            assert conditioned is not None, \
                "This model was constructed with use_conditioning=True, but `conditioned` was not provided."
            assert conditioned.shape == input_tensor.shape, \
                "`conditioned` must match `input_tensor` shape"
        else:
            assert conditioned is None, \
                "This model was constructed with use_conditioning=False; do not pass `conditioned`."

        skip_conn = []
        timestep_embedding = (
            self.timestep_embedder(timestep) if timestep is not None else None
        )
        if timestep_embedding is not None:
            assert timestep_embedding.shape == (input_tensor.shape[0], self.timestep_embed_dim), \
                f"timestep embedding shape {tuple(timestep_embedding.shape)} must be (N, hidden={self.timestep_embed_dim})"

        # Concatenate conditioning if provided
        x = input_tensor
        if conditioned is not None:
            x = torch.cat((x, conditioned), dim=1)

        x = self.inconv(x)
        x = self._add_timestep(x, timestep_embedding, self.in_time_proj)
        skip_conn.append(x)

        for i in range(self.layers - 1):
            x = self.down_blocks[i](x)
            x = self._add_timestep(x, timestep_embedding, self.down_time_projs[i])
            skip_conn.append(x)

        x = self.down_blocks[-1](x)
        x = self._add_timestep(x, timestep_embedding, self.down_time_projs[-1])

        x = self.bottleneck(x)
        x = self._add_timestep(x, timestep_embedding, self.bottleneck_time_proj)

        for i in range(self.layers - 1):
            skip = skip_conn.pop()
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[2:],
                    mode=get_upsample_mode(self.spatial_dim),
                    align_corners=False,
                )
            x = self.up_blocks[i](torch.cat((skip, x), dim=1))
            x = self._add_timestep(x, timestep_embedding, self.up_time_projs[i])

        skip = skip_conn.pop()
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode=get_upsample_mode(self.spatial_dim),
                align_corners=False,
            )
        x = torch.cat((skip, x), dim=1)
        x = self._add_timestep(x, timestep_embedding, self.out_time_proj)
        x = self.outconv(x)

        return x

    pass


# ---- WiDiT presets ----
def WiDiT2D_B(**kw):   return WiDiT(spatial_dim=2, depth=12, hidden_size=768,   patch_size=2, num_heads=12, **kw)
def WiDiT2D_M(**kw):   return WiDiT(spatial_dim=2, depth=12, hidden_size=1024,  patch_size=2, num_heads=16, **kw)
def WiDiT2D_L(**kw):   return WiDiT(spatial_dim=2, depth=24, hidden_size=1024,  patch_size=2, num_heads=16, **kw)
def WiDiT2D_XL(**kw):  return WiDiT(spatial_dim=2, depth=28, hidden_size=1152,  patch_size=2, num_heads=16, **kw)

def WiDiT3D_B(**kw):   return WiDiT(spatial_dim=3, depth=12, hidden_size=768,   patch_size=2, num_heads=12, **kw)
def WiDiT3D_M(**kw):   return WiDiT(spatial_dim=3, depth=12, hidden_size=1024,  patch_size=2, num_heads=16, **kw)
def WiDiT3D_L(**kw):   return WiDiT(spatial_dim=3, depth=24, hidden_size=1024,  patch_size=2, num_heads=16, **kw)
def WiDiT3D_XL(**kw):  return WiDiT(spatial_dim=3, depth=28, hidden_size=1152,  patch_size=2, num_heads=16, **kw)

# ---- Unet presets ----
def Unet2D_B(**kw):   return Unet(spatial_dim=2, layers=3, filters=64,  kernel_size=3, **kw)
def Unet2D_M(**kw):   return Unet(spatial_dim=2, layers=3, filters=128, kernel_size=3, **kw)
def Unet2D_L(**kw):   return Unet(spatial_dim=2, layers=4, filters=128, kernel_size=3, **kw)
def Unet2D_XL(**kw):  return Unet(spatial_dim=2, layers=4, filters=256, kernel_size=3, **kw)

def Unet3D_B(**kw):   return Unet(spatial_dim=3, layers=3, filters=64,  kernel_size=3, **kw)
def Unet3D_M(**kw):   return Unet(spatial_dim=3, layers=3, filters=128, kernel_size=3, **kw)
def Unet3D_L(**kw):   return Unet(spatial_dim=3, layers=4, filters=128, kernel_size=3, **kw)
def Unet3D_XL(**kw):  return Unet(spatial_dim=3, layers=4, filters=256, kernel_size=3, **kw)

PRESETS = {
    # 2D
    "WiDiT2D-B":  WiDiT2D_B,
    "WiDiT2D-M":  WiDiT2D_M,
    "WiDiT2D-L":  WiDiT2D_L,
    "WiDiT2D-XL": WiDiT2D_XL,
    "Unet2D-B":  Unet2D_B,
    "Unet2D-M":  Unet2D_M,
    "Unet2D-L":  Unet2D_L,
    "Unet2D-XL": Unet2D_XL,
    # 3D
    "WiDiT-B":   WiDiT3D_B,
    "WiDiT-M":   WiDiT3D_M,
    "WiDiT-L":   WiDiT3D_L,
    "WiDiT-XL":  WiDiT3D_XL,
    "WiDiT3D-B":  WiDiT3D_B,
    "WiDiT3D-M":  WiDiT3D_M,
    "WiDiT3D-L":  WiDiT3D_L,
    "WiDiT3D-XL": WiDiT3D_XL,
    "Unet-B":    Unet3D_B,
    "Unet-M":    Unet3D_M,
    "Unet-L":    Unet3D_L,
    "Unet-XL":   Unet3D_XL,
    "Unet3D-B":  Unet3D_B,
    "Unet3D-M":  Unet3D_M,
    "Unet3D-L":  Unet3D_L,
    "Unet3D-XL": Unet3D_XL,
}
