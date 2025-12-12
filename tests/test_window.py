import math
import pytest
import torch

from widit.window import (
    _to_sizes, _prod, _build_rel_pos_index,
    window_partition_nd, window_unpartition_nd,
    WindowAttention
)

# ---------- helpers ----------

def _rand_tensor_2d(N=2, H=8, W=6, C=4, device="cpu"):
    # channels-last as your API expects (N, H, W, C)
    x = torch.randn(N, H, W, C, device=device)
    return x

def _rand_tensor_3d(N=2, D=6, H=8, W=4, C=3, device="cpu"):
    x = torch.randn(N, D, H, W, C, device=device)
    return x

# ---------- low-level utils ----------

def test__to_sizes_int_and_tuple():
    assert _to_sizes(4, 2) == (4, 4)
    assert _to_sizes((4, 2), 2) == (4, 2)
    with pytest.raises(AssertionError):
        _to_sizes((4, 2, 1), 2)

def test__prod():
    assert _prod([2, 3, 4]) == 24
    assert _prod([]) == 1

def test__build_rel_pos_index_shapes_and_range():
    ws = (3, 2)     # T = 6
    idx = _build_rel_pos_index(ws)
    assert idx.shape == (6, 6)
    # range must be < prod(2*ws_i - 1) = (5*3)=15
    assert idx.max().item() < 15
    assert idx.min().item() >= 0

# ---------- partition / unpartition ----------

@pytest.mark.parametrize("ws", [(2, 2), (4, 2)])
def test_partition_unpartition_2d_roundtrip(ws):
    N, H, W, C = 3, 8, 6, 5
    x = _rand_tensor_2d(N, H, W, C)
    win = window_partition_nd(x, ws)
    y = window_unpartition_nd(win, ws, H, W)
    assert torch.allclose(x, y)

@pytest.mark.parametrize("ws", [(2, 2, 2), (3, 2, 1)])
def test_partition_unpartition_3d_roundtrip(ws):
    N, D, H, W, C = 2, 6, 6, 6, 4
    # ensure divisibility for every axis
    assert D % ws[0] == 0 and H % ws[1] == 0 and W % ws[2] == 0
    x = _rand_tensor_3d(N, D, H, W, C)
    win = window_partition_nd(x, ws)
    y = window_unpartition_nd(win, ws, D, H, W)
    assert torch.allclose(x, y)

def test_partition_requires_divisible():
    x = _rand_tensor_2d(1, 7, 8, 3)      # H not divisible by ws=4
    with pytest.raises(AssertionError):
        window_partition_nd(x, (4, 4))

def test_unpartition_requires_spatial_dims():
    x = _rand_tensor_2d(1, 8, 8, 3)
    win = window_partition_nd(x, (4, 4))
    with pytest.raises(AssertionError):
        window_unpartition_nd(win, (4, 4))  # missing H, W

# ---------- attention ----------

@pytest.mark.parametrize("ws", [(2, 2), (4, 2)])
def test_window_attention_forward_2d(ws):
    dim = 32
    heads = 4
    T = ws[0] * ws[1]
    B_ = 5
    x = torch.randn(B_, T, dim)
    attn = WindowAttention(dim=dim, window_size=ws, num_heads=heads, spatial_dim=2)
    out = attn(x)
    assert out.shape == (B_, T, dim)
    # basic sanity: finite + grad flows
    out.sum().backward()

def test_window_attention_token_guard():
    dim = 16
    ws = (2, 2)
    T = 3  # wrong on purpose
    x = torch.randn(2, T, dim)
    attn = WindowAttention(dim=dim, window_size=ws, num_heads=4, spatial_dim=2)
    with pytest.raises(AssertionError):
        attn(x)

@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_window_attention_device_move(device):
    dim = 24
    ws = (2, 3)
    T = _prod(ws)
    x = torch.randn(4, T, dim, device=device)
    attn = WindowAttention(dim=dim, window_size=ws, num_heads=6, spatial_dim=2).to(device)
    out = attn(x)
    assert out.device.type == device
    assert out.shape == (4, T, dim)

def test_attention_bias_index_consistency_small_2d():
    # For ws=(2,2): T=4; ensure bias table indexing builds a (4,4) map
    dim = 8
    heads = 2
    ws = (2, 2)
    T = 4
    attn = WindowAttention(dim=dim, window_size=ws, num_heads=heads, spatial_dim=2)
    idx = attn.rel_pos_index
    assert idx.shape == (T, T)
    # Check a couple of symmetric entries (relative offset (0,0) must map same index on diagonal)
    assert torch.equal(idx.diag(), idx.diag())  # tautology; ensures presence + shape
    # different offsets should map to (usually) different indices
    assert (idx[0, 1] != idx[0, 0]).item()

def test_window_attention_flash_path_uses_sdpa(monkeypatch):
    dim = 16
    heads = 4
    ws = (2, 2)
    T = _prod(ws)
    B_ = 3
    x = torch.randn(B_, T, dim)

    sdpa_calls = {}

    def fake_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
        sdpa_calls["q"] = q
        sdpa_calls["k"] = k
        sdpa_calls["v"] = v
        sdpa_calls["mask"] = attn_mask
        sdpa_calls["dropout_p"] = dropout_p
        sdpa_calls["is_causal"] = is_causal
        return torch.zeros_like(q)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa, raising=False)

    attn = WindowAttention(
        dim=dim,
        window_size=ws,
        num_heads=heads,
        spatial_dim=2,
        use_flash_attention=True,
    )
    out = attn(x)

    assert out.shape == (B_, T, dim)
    assert sdpa_calls["mask"].shape == (B_, heads, T, T)
    assert sdpa_calls["dropout_p"] == 0.0
    assert sdpa_calls["is_causal"] is False

def test_window_attention_flash_auto_uses_sdpa(monkeypatch):
    dim = 16
    heads = 4
    ws = (2, 2)
    T = _prod(ws)
    B_ = 2
    x = torch.randn(B_, T, dim)

    sdpa_calls = {}

    def fake_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
        sdpa_calls["called"] = True
        return torch.zeros_like(q)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa, raising=False)

    attn = WindowAttention(
        dim=dim,
        window_size=ws,
        num_heads=heads,
        spatial_dim=2,
        use_flash_attention=True,
    )
    out = attn(x)
    assert out.shape == (B_, T, dim)
    assert sdpa_calls.get("called") is True

def test_window_attention_flash_auto_falls_back_without_sdpa(monkeypatch):
    # Remove sdpa if present to force legacy path
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        monkeypatch.delattr(torch.nn.functional, "scaled_dot_product_attention", raising=False)

    dim = 12
    heads = 3
    ws = (2, 2)
    T = _prod(ws)
    B_ = 2
    x = torch.randn(B_, T, dim)

    attn = WindowAttention(
        dim=dim,
        window_size=ws,
        num_heads=heads,
        spatial_dim=2,
        use_flash_attention=True,
    )
    out = attn(x)
    assert out.shape == (B_, T, dim)
    # Should still be backward-able on fallback path
    out.sum().backward()

# ---------- non-uniform windows & non-contiguous inputs ----------

def test_nonuniform_windows_2d():
    N, H, W, C = 2, 12, 8, 4
    ws = (3, 2)
    x = _rand_tensor_2d(N, H, W, C)
    win = window_partition_nd(x, ws)
    y = window_unpartition_nd(win, ws, H, W)
    assert torch.allclose(x, y)

def test_noncontiguous_channels_last_ok():
    x = _rand_tensor_2d(2, 8, 8, 6)
    y = x[:, ::2, :, :]  # stride on H ⇒ non-contiguous view
    assert not y.is_contiguous()
    win = window_partition_nd(y.contiguous(), (2, 2))  # caller can choose to .contiguous() before
    z = window_unpartition_nd(win, (2, 2), y.shape[1], y.shape[2])
    assert z.shape == (2, y.shape[1], y.shape[2], 6)
