import pytest
import torch

from widit.patch import PatchEmbed


def test_2d_basic_shape():
    torch.manual_seed(0)
    N, C, H, W = 2, 3, 16, 12
    p = (4, 3)               # non-square
    D = 32

    m = PatchEmbed(input_size=(H, W), patch_size=p, in_chans=C, embed_dim=D, bias=True, spatial_dim=2)
    x = torch.randn(N, C, H, W)
    y = m(x)
    T = (H // p[0]) * (W // p[1])
    assert y.shape == (N, T, D)
    (y.mean()).backward()


def test_3d_basic_shape():
    torch.manual_seed(0)
    N, C, Dd, H, W = 2, 1, 12, 16, 10
    p = (3, 4, 5)
    E = 24

    m = PatchEmbed(input_size=(Dd, H, W), patch_size=p, in_chans=C, embed_dim=E, bias=False, spatial_dim=3)
    x = torch.randn(N, C, Dd, H, W)
    y = m(x)
    T = (Dd // p[0]) * (H // p[1]) * (W // p[2])
    assert y.shape == (N, T, E)
    (y.sum()).backward()


def test_2d_infer_spatial_dim_from_input_rank():
    torch.manual_seed(0)
    N, C, H, W = 1, 3, 8, 8
    m = PatchEmbed(input_size=None, patch_size=4, in_chans=C, embed_dim=16, bias=True, spatial_dim=None)
    y = m(torch.randn(N, C, H, W))
    assert y.shape == (N, (H//4)*(W//4), 16)


def test_3d_infer_spatial_dim_from_input_rank():
    torch.manual_seed(0)
    N, C, Dd, H, W = 1, 2, 8, 8, 8
    m = PatchEmbed(input_size=None, patch_size=2, in_chans=C, embed_dim=20, bias=True, spatial_dim=None)
    y = m(torch.randn(N, C, Dd, H, W))
    assert y.shape == (N, (Dd//2)*(H//2)*(W//2), 20)


def test_bad_rank_raises():
    m = PatchEmbed(input_size=None, patch_size=2, in_chans=3, embed_dim=8, spatial_dim=None)
    with pytest.raises(ValueError):
        m(torch.randn(2, 3))  # rank 2


def test_spatial_dim_mismatch_raises():
    m2 = PatchEmbed(input_size=None, patch_size=2, in_chans=3, embed_dim=8, spatial_dim=2)
    with pytest.raises(AssertionError):
        m2(torch.randn(2, 3, 4, 5, 6))  # 3D input passed to 2D

    m3 = PatchEmbed(input_size=None, patch_size=(2, 2, 2), in_chans=3, embed_dim=8, spatial_dim=3)
    with pytest.raises(AssertionError):
        m3(torch.randn(2, 3, 8, 8))     # 2D input passed to 3D


def test_cuda_if_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    N, C, H, W = 2, 3, 12, 12
    m = PatchEmbed(input_size=(H, W), patch_size=4, in_chans=C, embed_dim=32, spatial_dim=2).cuda()
    x = torch.randn(N, C, H, W, device="cuda")
    y = m(x)
    assert y.is_cuda
    assert y.shape == (N, (H//4)*(W//4), 32)
