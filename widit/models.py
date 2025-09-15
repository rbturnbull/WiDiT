# swinir_dit.py
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Mlp


class SwinIRDiT(nn.Module):
    """
    2D windowed attention backbone without downsampling (best SR fidelity).
    Two parallel patch embedders (x and conditioned) -> concat -> SwinIR blocks -> head.
    """
    def __init__(
        self,
        input_size=500,
        patch_size=2,
        in_channels=1,
        hidden_size=768,
        depth=12,
        num_heads=12,
        window_size=8,
        mlp_ratio=4.0,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size

        # two embedders whose dims sum to hidden_size
        self.x_embed = PatchEmbed(input_size, patch_size, in_channels, hidden_size // 2, bias=True)
        self.y_embed = PatchEmbed(input_size, patch_size, in_channels, hidden_size // 2, bias=True)
        self.t_embed = TimestepEmbedder(hidden_size)

        blocks = []
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else window_size // 2
            blocks.append(SwinIRBlock(hidden_size, num_heads, window_size, shift, mlp_ratio))
        self.blocks = nn.ModuleList(blocks)

        self.head = FinalLayer(hidden_size, patch_size, self.out_channels)
        self._init_weights()

    def _init_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_basic)

        for pe in [self.x_embed, self.y_embed]:
            w = pe.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(pe.proj.bias, 0)

        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)

        for b in self.blocks:
            nn.init.constant_(b.ada[-1].weight, 0)
            nn.init.constant_(b.ada[-1].bias, 0)

        nn.init.constant_(self.head.ada[-1].weight, 0)
        nn.init.constant_(self.head.ada[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)

    def unpatchify(self, x, H, W):
        # x: (N, T, p*p*c) -> (N, c, H, W)
        N, T, PPc = x.shape
        p = self.patch_size
        c = self.out_channels
        h = H // p
        w = W // p
        assert h * w == T, "Token count mismatch in unpatchify."
        x = x.view(N, h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(N, c, H, W)

    def forward(self, x, t, conditioned):
        # x, conditioned: (N, C, H, W); t: (N,)
        N, C, H, W = x.shape
        ex = self.x_embed(x)              # (N, T, D/2)
        ey = self.y_embed(conditioned)    # (N, T, D/2)
        z = torch.cat([ex, ey], dim=-1)   # (N, T, D)
        Hp, Wp = H // self.patch_size, W // self.patch_size
        assert Hp * Wp == z.shape[1]
        c = self.t_embed(t)               # (N, D)

        for blk in self.blocks:
            z = blk(z, c, Hp, Wp)

        out = self.head(z, c)             # (N, T, p*p*outC)
        return self.unpatchify(out, H, W)

# Config helpers (2D)
def SwinIRDiT_B_2(**kw):  return SwinIRDiT(depth=12, hidden_size=768,  patch_size=2, num_heads=12, **kw)
def SwinIRDiT_L_2(**kw):  return SwinIRDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kw)
def SwinIRDiT_XL_2(**kw): return SwinIRDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kw)

SwinIRDiT_models = {
    "SwinIRDiT-B/2": SwinIRDiT_B_2,
    "SwinIRDiT-L/2": SwinIRDiT_L_2,
    "SwinIRDiT-XL/2": SwinIRDiT_XL_2,
}

# =========================================================
# ========================= 3D ============================
# =========================================================

# ----- 3D PatchEmbed -----
class PatchEmbed3D(nn.Module):
    """Conv3d(patch, patch, patch) -> tokens (N, T, D)."""
    def __init__(self, input_size, patch_size, in_chans, embed_dim, bias=True):
        super().__init__()
        p = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=p, stride=p, bias=bias)

    def forward(self, x):  # x: (N, C, D, H, W)
        x = self.proj(x)   # (N, Dm, D/p, H/p, W/p)
        N, Dm, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (N, T, Dm)
        return x




class SwinIRDiT3D(nn.Module):
    """
    3D windowed attention backbone without downsampling (best SR fidelity).
    Two parallel patch embedders (x and conditioned) -> concat -> SwinIR 3D blocks -> head.
    """
    def __init__(
        self,
        input_size=128,
        patch_size=2,
        in_channels=1,
        hidden_size=768,
        depth=12,
        num_heads=12,
        window_size=4,
        mlp_ratio=4.0,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size

        # two 3D embedders whose dims sum to hidden_size
        self.x_embed = PatchEmbed3D(input_size, patch_size, in_channels, hidden_size // 2, bias=True)
        self.y_embed = PatchEmbed3D(input_size, patch_size, in_channels, hidden_size // 2, bias=True)
        self.t_embed = TimestepEmbedder(hidden_size)

        blocks = []
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else window_size // 2  # shifted cubic windows
            blocks.append(SwinIRBlock3D(hidden_size, num_heads, window_size, shift, mlp_ratio))
        self.blocks = nn.ModuleList(blocks)

        self.head = FinalLayer3D(hidden_size, patch_size, self.out_channels)
        self._init_weights()

    def _init_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_basic)

        for pe in [self.x_embed, self.y_embed]:
            w = pe.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            if pe.proj.bias is not None:
                nn.init.constant_(pe.proj.bias, 0)

        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)

        for b in self.blocks:
            nn.init.constant_(b.ada[-1].weight, 0)
            nn.init.constant_(b.ada[-1].bias, 0)

        nn.init.constant_(self.head.ada[-1].weight, 0)
        nn.init.constant_(self.head.ada[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)

    def unpatchify(self, x, D, H, W):
        # x: (N, T, p^3 * c) -> (N, c, D, H, W)
        N, T, PPc = x.shape
        p = self.patch_size
        c = self.out_channels
        d, h, w = D // p, H // p, W // p
        assert d * h * w == T, f"Token count mismatch in unpatchify(3D): {d*h*w} != {T}"
        assert PPc == (p ** 3) * c, f"Last dim should be p^3*c, got {PPc} vs {(p**3)*c}"

        x = x.view(N, d, h, w, p, p, p, c)                # (N, d, h, w, pd, ph, pw, c)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous() # (N, c, d, pd, h, ph, w, pw)
        x = x.view(N, c, D, H, W)
        return x

    def forward(self, x, t, conditioned):
        # x, conditioned: (N, C, D, H, W); t: (N,)
        assert len(x.shape) == 5, f"x has incorrect shape {x.shape}"
        assert len(conditioned.shape) == 5, f"`conditioned` has incorrect shape {conditioned.shape}"
        N, C, D, H, W = x.shape
        ex = self.x_embed(x)            # (N, T, D/2)
        ey = self.y_embed(conditioned)  # (N, T, D/2)
        z = torch.cat([ex, ey], dim=-1) # (N, T, Dtot)
        Dp, Hp, Wp = D // self.patch_size, H // self.patch_size, W // self.patch_size
        assert Dp * Hp * Wp == z.shape[1], f"Token count mismatch: {Dp*Hp*Wp} vs {z.shape[1]}"
        c = self.t_embed(t)             # (N, Dtot)

        for blk in self.blocks:
            z = blk(z, c, Dp, Hp, Wp)

        out = self.head(z, c)           # (N, T, p^3*outC)
        return self.unpatchify(out, D, H, W)

# Config helpers (3D)
def SwinIRDiT3D_B_2(**kw):  return SwinIRDiT3D(depth=12, hidden_size=768,  patch_size=2, num_heads=12, **kw)
def SwinIRDiT3D_M_2(**kw):  return SwinIRDiT3D(depth=12, hidden_size=1024, patch_size=2, num_heads=12, **kw)
def SwinIRDiT3D_L_2(**kw):  return SwinIRDiT3D(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kw)
def SwinIRDiT3D_XL_2(**kw): return SwinIRDiT3D(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kw)

SwinIRDiT3D_models = {
    "SwinIRDiT3D-B/2": SwinIRDiT3D_B_2,
    "SwinIRDiT3D-M/2": SwinIRDiT3D_M_2,
    "SwinIRDiT3D-L/2": SwinIRDiT3D_L_2,
    "SwinIRDiT3D-XL/2": SwinIRDiT3D_XL_2,
}

__all__ = [
    # shared
    "TimestepEmbedder",
    # 2D
    "SwinIRDiT", "SwinIRBlock", "WindowAttention", "FinalLayer",
    "SwinIRDiT_B_2", "SwinIRDiT_L_2", "SwinIRDiT_XL_2", "SwinIRDiT_models",
    # 3D
    "SwinIRDiT3D", "SwinIRBlock3D", "WindowAttention3D", "FinalLayer3D",
    "SwinIRDiT3D_B_2", "SwinIRDiT3D_M_2", "SwinIRDiT3D_L_2", "SwinIRDiT3D_XL_2", "SwinIRDiT3D_models",
]
