import pytest
import torch
import torch.nn.functional as F

from widit.blocks import WiDiTBlock, WiDiTFinalLayer, _pad_channels_last, _roll_channels_last
from widit.window import _prod


def _rand_tokens(N, sizes, C):
    """Create (N, T, C) tokens from sizes=(S1,...,Sk)."""
    T = _prod(sizes)
    x = torch.randn(N, T, C)
    return x, T


# -------------------- WiDiTBlock (2D) --------------------

@pytest.mark.parametrize("sizes,ws,shift", [
    ((8, 8), (4, 4), (0, 0)),      # divisible, no shift
    ((8, 8), (4, 4), (2, 2)),      # divisible, with shift
    ((7, 6), (4, 3), (2, 1)),      # padding case, non-uniform windows
])
def test_widit_block_2d_forward_backward(sizes, ws, shift):
    torch.manual_seed(0)
    N, C = 2, 32
    x, T = _rand_tokens(N, sizes, C)
    c = torch.randn(N, C)

    blk = WiDiTBlock(dim=C, num_heads=4, window_size=ws, shift_size=shift, mlp_ratio=4.0, spatial_dim=2)
    out = blk(x, c, *sizes)

    assert out.shape == (N, T, C)
    # gradient sanity
    (out.sum()).backward()
    grads = [p.grad for p in blk.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_widit_block_2d_token_grid_mismatch_raises():
    torch.manual_seed(0)
    N, C = 1, 16
    sizes = (8, 8)
    x, T = _rand_tokens(N, sizes, C)
    c = torch.randn(N, C)

    blk = WiDiTBlock(dim=C, num_heads=2, window_size=(4, 4), spatial_dim=2)
    # pass wrong grid sizes on purpose
    with pytest.raises(AssertionError):
        blk(x, c, 7, 9)


def test_widit_block_2d_nonuniform_windows():
    torch.manual_seed(0)
    N, C = 2, 24
    sizes = (12, 8)        # both divisible by (3,2)
    ws = (3, 2)
    shift = (1, 0)

    x, T = _rand_tokens(N, sizes, C)
    c = torch.randn(N, C)

    blk = WiDiTBlock(dim=C, num_heads=6, window_size=ws, shift_size=shift, spatial_dim=2)
    out = blk(x, c, *sizes)
    assert out.shape == (N, T, C)
    (out.mean()).backward()


# -------------------- WiDiTBlock (3D) --------------------

@pytest.mark.parametrize("sizes,ws,shift", [
    ((6, 6, 6), (3, 3, 3), (0, 0, 0)),  # divisible, no shift
    ((6, 6, 6), (3, 2, 2), (1, 1, 1)),  # divisible, with shift, non-uniform ws
    ((5, 7, 6), (2, 3, 2), (1, 1, 1)),  # padding case
])
def test_widit_block_3d_forward_backward(sizes, ws, shift):
    torch.manual_seed(0)
    N, C = 2, 48
    x, T = _rand_tokens(N, sizes, C)
    c = torch.randn(N, C)

    blk = WiDiTBlock(dim=C, num_heads=6, window_size=ws, shift_size=shift, mlp_ratio=3.0, spatial_dim=3)
    out = blk(x, c, *sizes)

    assert out.shape == (N, T, C)
    (out.sum()).backward()
    grads = [p.grad for p in blk.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_widit_block_device_cuda_if_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    N, C = 2, 32
    sizes = (8, 8)
    ws = (4, 4)
    x, T = _rand_tokens(N, sizes, C)
    c = torch.randn(N, C)

    blk = WiDiTBlock(dim=C, num_heads=4, window_size=ws, spatial_dim=2).cuda()
    out = blk(x.cuda(), c.cuda(), *sizes)
    assert out.is_cuda
    assert out.shape == (N, T, C)


# -------------------- WiDiTFinalLayer --------------------

@pytest.mark.parametrize("k,sizes,patch,outc", [
    (2, (8, 8), 2, 3),
    (3, (6, 4, 5), 2, 1),
])
def test_widit_final_layer_shapes_and_backward(k, sizes, patch, outc):
    torch.manual_seed(0)
    N, C = 2, 32
    x, T = _rand_tokens(N, sizes, C)
    c = torch.randn(N, C)

    head = WiDiTFinalLayer(hidden_size=C, patch_size=patch, out_channels=outc, spatial_dim=k)
    out = head(x, c)
    expected_last = (patch ** k) * outc
    assert out.shape == (N, T, expected_last)
    (out.sum()).backward()
    grads = [p.grad for p in head.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_widit_final_layer_cuda_if_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    N, C = 2, 16
    sizes = (7, 9)  # T == 63
    x, T = _rand_tokens(N, sizes, C)
    c = torch.randn(N, C)

    head = WiDiTFinalLayer(hidden_size=C, patch_size=2, out_channels=1, spatial_dim=2).cuda()
    out = head(x.cuda(), c.cuda())
    assert out.is_cuda
    assert out.shape == (N, T, (2 ** 2) * 1)


def test_pad_channels_last_noop_2d_identity_and_grad():
    # (N, H, W, C), all pads = 0 → must return the SAME object
    x = torch.randn(2, 5, 7, 3, requires_grad=True)
    y = _pad_channels_last(x, pads_per_axis=(0, 0))

    # Identity check: the function should just return `x` (no copy, no permute)
    assert y is x, "Expected early-return identity when all pads are zero."
    assert y.shape == x.shape

    # Grad still flows (sanity)
    y.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape


def test_pad_channels_last_noop_3d_identity_and_grad():
    # (N, D, H, W, C), all pads = 0 → must return the SAME object
    x = torch.randn(1, 4, 5, 6, 2, requires_grad=True)
    y = _pad_channels_last(x, pads_per_axis=(0, 0, 0))

    assert y is x, "Expected early-return identity when all pads are zero (3D)."
    assert y.shape == x.shape

    y.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape


def test_pad_channels_last_applies_right_side_padding_2d_and_matches_manual():
    # Non-zero pad path: verify shape & values match manual NCHW padding
    x = torch.arange(2*2*3*1, dtype=torch.float32).view(2, 2, 3, 1)  # (N,H,W,C)
    pads = (1, 2)  # pad H by +1 (bottom), W by +2 (right)

    # Function under test
    y = _pad_channels_last(x, pads)

    # Manual reference via NCHW → pad → NHWC
    x_nchw = x.permute(0, 3, 1, 2)           # (N,C,H,W)
    # F.pad format for 2D: (left, right, top, bottom)
    ref_nchw = F.pad(x_nchw, (0, pads[1], 0, pads[0]))
    ref = ref_nchw.permute(0, 2, 3, 1)

    assert y.shape == ref.shape == (2, 2 + pads[0], 3 + pads[1], 1)
    assert torch.allclose(y, ref)


def test_pad_channels_last_applies_right_side_padding_3d_and_matches_manual():
    # 3D non-zero pad: verify shape & values match manual ND pad path
    x = torch.randn(1, 2, 3, 4, 2)  # (N,D,H,W,C)
    pads = (1, 0, 2)  # pad D by +1 (back), H by +0, W by +2 (right)

    # Function under test
    y = _pad_channels_last(x, pads)

    # Manual reference via NCDHW → F.pad → NDHWC
    x_ncdhw = x.permute(0, 4, 1, 2, 3)  # (N,C,D,H,W)
    # F.pad format for 3D: (W_left, W_right, H_top, H_bottom, D_front, D_back)
    ref_ncdhw = F.pad(x_ncdhw, (0, pads[2], 0, pads[1], 0, pads[0]))
    ref = ref_ncdhw.permute(0, 2, 3, 4, 1)

    assert y.shape == ref.shape == (1, 2 + pads[0], 3 + pads[1], 4 + pads[2], 2)
    assert torch.allclose(y, ref)


import torch
import pytest

from widit.blocks import _roll_channels_last


# ---------- No-op branch coverage (all shifts == 0) ----------

def test_roll_channels_last_noop_2d_identity_and_grad():
    # (N, H, W, C)
    x = torch.randn(2, 5, 7, 3, requires_grad=True)
    y = _roll_channels_last(x, shift_sizes=(0, 0), invert=False)

    # Must be exact same tensor object (early return)
    assert y is x
    assert y.shape == x.shape
    # Values unchanged & grad flows
    y.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape


def test_roll_channels_last_noop_3d_identity_and_grad():
    # (N, D, H, W, C)
    x = torch.randn(1, 4, 5, 6, 2, requires_grad=True)
    y = _roll_channels_last(x, shift_sizes=(0, 0, 0), invert=True)

    assert y is x
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape


# ---------- Functional correctness for non-zero shifts ----------

def test_roll_channels_last_matches_torch_roll_2d():
    # Small deterministic tensor to check positions
    x = torch.arange(2*3*4*1, dtype=torch.float32).view(2, 3, 4, 1)  # (N,H,W,C)
    shifts_hw = (1, 2)  # our helper uses negative for forward (invert=False)

    y = _roll_channels_last(x, shift_sizes=shifts_hw, invert=False)

    # Reference using torch.roll over dims (H=1, W=2)
    ref = torch.roll(x, shifts=(-shifts_hw[0], -shifts_hw[1]), dims=(1, 2))
    assert torch.allclose(y, ref)


def test_roll_channels_last_invert_roundtrip_3d():
    x = torch.randn(2, 4, 5, 3, 1, requires_grad=True)  # (N,D,H,W,C)
    shifts_dhw = (1, 0, 2)

    # Apply forward (invert=False), then inverse (invert=True) -> should recover x
    y = _roll_channels_last(x, shift_sizes=shifts_dhw, invert=False)
    z = _roll_channels_last(y, shift_sizes=shifts_dhw, invert=True)

    assert torch.allclose(z, x)

    # Non-identity path should produce a different object than x
    assert y is not x

    # Grad check through both rolls
    z.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
