import pytest
import torch

from widit.blocks import WiditBlock, WiditFinalLayer
from widit.window import _prod


def _rand_tokens(N, sizes, C):
    """Create (N, T, C) tokens from sizes=(S1,...,Sk)."""
    T = _prod(sizes)
    x = torch.randn(N, T, C)
    return x, T


# -------------------- WiditBlock (2D) --------------------

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

    blk = WiditBlock(dim=C, num_heads=4, window_size=ws, shift_size=shift, mlp_ratio=4.0, spatial_dim=2)
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

    blk = WiditBlock(dim=C, num_heads=2, window_size=(4, 4), spatial_dim=2)
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

    blk = WiditBlock(dim=C, num_heads=6, window_size=ws, shift_size=shift, spatial_dim=2)
    out = blk(x, c, *sizes)
    assert out.shape == (N, T, C)
    (out.mean()).backward()


# -------------------- WiditBlock (3D) --------------------

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

    blk = WiditBlock(dim=C, num_heads=6, window_size=ws, shift_size=shift, mlp_ratio=3.0, spatial_dim=3)
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

    blk = WiditBlock(dim=C, num_heads=4, window_size=ws, spatial_dim=2).cuda()
    out = blk(x.cuda(), c.cuda(), *sizes)
    assert out.is_cuda
    assert out.shape == (N, T, C)


# -------------------- WiditFinalLayer --------------------

@pytest.mark.parametrize("k,sizes,patch,outc", [
    (2, (8, 8), 2, 3),
    (3, (6, 4, 5), 2, 1),
])
def test_widit_final_layer_shapes_and_backward(k, sizes, patch, outc):
    torch.manual_seed(0)
    N, C = 2, 32
    x, T = _rand_tokens(N, sizes, C)
    c = torch.randn(N, C)

    head = WiditFinalLayer(hidden_size=C, patch_size=patch, out_channels=outc, spatial_dim=k)
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

    head = WiditFinalLayer(hidden_size=C, patch_size=2, out_channels=1, spatial_dim=2).cuda()
    out = head(x.cuda(), c.cuda())
    assert out.is_cuda
    assert out.shape == (N, T, (2 ** 2) * 1)
