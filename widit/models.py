# swinir_dit.py
# Unified SwinIR-style DiT for 2D & 3D Super-Resolution (no checkpointing, no merging)

import math
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Mlp

# ---------------- Shared utils ----------------
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))

# =========================================================
# ========================= 2D ============================
# =========================================================

# ----- 2D window helpers -----
def window_partition(x, ws):
    # x: (N,H,W,C) -> (num_win*N, ws, ws, C)
    N, H, W, C = x.shape
    x = x.view(N, H // ws, ws, W // ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, ws, ws, C)
    return x

def window_unpartition(windows, ws, H, W):
    # windows: (num_win*N, ws, ws, C) -> (N,H,W,C)
    Nw, _, _, C = windows.shape
    N = Nw // ((H // ws) * (W // ws))
    x = windows.view(N, H // ws, W // ws, ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(N, H, W, C)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.ws = window_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        ws = window_size
        coords = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing='ij'))
        coords_flat = torch.flatten(coords, 1)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1
        idx = rel.sum(-1)
        self.register_buffer("rel_pos_index", idx, persistent=False)
        self.rel_pos_bias = nn.Parameter(torch.zeros((2 * ws - 1) * (2 * ws - 1), num_heads))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)
        self.scale = (dim // num_heads) ** -0.5

    def forward(self, x):
        # x: (B_, T, C) with T = ws*ws
        B_, T, C = x.shape
        qkv = self.qkv(x).reshape(B_, T, 3, self.num_heads, C // self.num_heads)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        bias = (
            self.rel_pos_bias[self.rel_pos_index.view(-1)]
            .view(T, T, self.num_heads)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(attn.dtype)
        )
        attn = (attn + bias).softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B_, T, C)
        return self.proj(out)

class SwinIRBlock(nn.Module):
    """2D windowed attention + MLP with adaLN-Zero conditioning; optional shift."""
    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.ws = window_size
        self.shift = shift_size
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))  # shift/scale/gate x2

    def forward(self, x_seq, c, H, W):
        # x_seq: (N, T, C), T=H*W
        N, T, C = x_seq.shape
        assert T == H * W
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada(c).chunk(6, dim=1)

        x = x_seq.view(N, H, W, C)
        ws = self.ws
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[1], x.shape[2]

        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))

        x_win = window_partition(x, ws).view(-1, ws * ws, C)

        nW = (Hp // ws) * (Wp // ws)
        rep = nW
        shift_msa_w = shift_msa.repeat_interleave(rep, dim=0)
        scale_msa_w = scale_msa.repeat_interleave(rep, dim=0)
        gate_msa_w = gate_msa.repeat_interleave(rep, dim=0)

        h = self.norm1(x_win)
        h = modulate(h, shift_msa_w, scale_msa_w)
        h = self.attn(h)
        x_win = x_win + gate_msa_w.unsqueeze(1) * h

        x = window_unpartition(x_win.view(-1, ws, ws, C), ws, Hp, Wp)

        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))
        if pad_h or pad_w:
            x = x[:, :H, :W, :]

        x = x.contiguous().reshape(N, T, C)
        h = self.norm2(x)
        h = modulate(h, shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.ada(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))

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

# ----- 3D window helpers -----
def window_partition_3d(x, ws):
    # x: (N, D, H, W, C) -> (num_win*N, ws, ws, ws, C)
    N, D, H, W, C = x.shape
    x = x.view(N, D // ws, ws, H // ws, ws, W // ws, ws, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(-1, ws, ws, ws, C)
    return x

def window_unpartition_3d(windows, ws, D, H, W):
    # windows: (num_win*N, ws, ws, ws, C) -> (N, D, H, W, C)
    Nw, _, _, _, C = windows.shape
    N = Nw // ((D // ws) * (H // ws) * (W // ws))
    x = windows.view(N, D // ws, H // ws, W // ws, ws, ws, ws, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(N, D, H, W, C)
    return x

class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.ws = window_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        ws = window_size
        coords = torch.stack(
            torch.meshgrid(torch.arange(ws), torch.arange(ws), torch.arange(ws), indexing='ij')
        )  # (3, ws, ws, ws)
        coords_flat = torch.flatten(coords, 1)  # (3, T)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (3, T, T)
        rel = rel.permute(1, 2, 0).contiguous()  # (T, T, 3)

        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 2] += ws - 1
        rel[:, :, 0] *= (2 * ws - 1) * (2 * ws - 1)
        rel[:, :, 1] *= (2 * ws - 1)
        idx = rel.sum(-1)
        self.register_buffer("rel_pos_index", idx, persistent=False)
        self.rel_pos_bias = nn.Parameter(torch.zeros((2 * ws - 1) ** 3, num_heads))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

        self.scale = (dim // num_heads) ** -0.5

    def forward(self, x):  # x: (B_, T, C), T=ws^3
        B_, T, C = x.shape
        qkv = self.qkv(x).reshape(B_, T, 3, self.num_heads, C // self.num_heads)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        bias = (
            self.rel_pos_bias[self.rel_pos_index.view(-1)]
            .view(T, T, self.num_heads)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(attn.dtype)
        )
        attn = (attn + bias).softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B_, T, C)
        return self.proj(out)

class SwinIRBlock3D(nn.Module):
    """3D windowed attention + MLP with adaLN-Zero conditioning; optional cubic shift."""
    def __init__(self, dim, num_heads, window_size=4, shift_size=0, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.ws = window_size
        self.shift = shift_size

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention3D(dim, window_size, num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))  # shift/scale/gate x2

    def forward(self, x_seq, c, Dp, Hp, Wp):
        # x_seq: (N, T, C), T=Dp*Hp*Wp
        N, T, C = x_seq.shape
        assert T == Dp * Hp * Wp
        ws = self.ws

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada(c).chunk(6, dim=1)

        # (N, D, H, W, C)
        x = x_seq.view(N, Dp, Hp, Wp, C)

        # pad to multiples of window size
        pad_d = (ws - Dp % ws) % ws
        pad_h = (ws - Hp % ws) % ws
        pad_w = (ws - Wp % ws) % ws
        if pad_d or pad_h or pad_w:
            x_ncdhw = x.permute(0, 4, 1, 2, 3)  # (N, C, D, H, W)
            x_ncdhw = nn.functional.pad(x_ncdhw, (0, pad_w, 0, pad_h, 0, pad_d))
            x = x_ncdhw.permute(0, 2, 3, 4, 1)  # (N, D', H', W', C)
        Dpp, Hpp, Wpp = x.shape[1], x.shape[2], x.shape[3]

        # optional cubic shift
        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift, -self.shift), dims=(1, 2, 3))

        # window partition
        x_win = window_partition_3d(x, ws).view(-1, ws * ws * ws, C)  # (N*nW, T_w, C)

        # repeat conditioning per window
        nW = (Dpp // ws) * (Hpp // ws) * (Wpp // ws)
        rep = nW
        shift_msa_w = shift_msa.repeat_interleave(rep, dim=0)
        scale_msa_w = scale_msa.repeat_interleave(rep, dim=0)
        gate_msa_w = gate_msa.repeat_interleave(rep, dim=0)

        # attention with modulation
        h = self.norm1(x_win)
        h = modulate(h, shift_msa_w, scale_msa_w)
        h = self.attn(h)
        x_win = x_win + gate_msa_w.unsqueeze(1) * h

        # merge windows
        x = window_unpartition_3d(x_win.view(-1, ws, ws, ws, C), ws, Dpp, Hpp, Wpp)

        # undo shift and crop padding
        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift, self.shift), dims=(1, 2, 3))
        if pad_d or pad_h or pad_w:
            x = x[:, :Dp, :Hp, :Wp, :]

        # MLP
        x = x.contiguous().view(N, T, C)
        h = self.norm2(x)
        h = modulate(h, shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x

class FinalLayer3D(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, (patch_size ** 3) * out_channels, bias=True)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.ada(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))

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
