import pytest
import torch

from widit.patch import PatchEmbed


def test_2d_basic_shape():
    torch.manual_seed(0)
    N, C, H, W = 2, 3, 16, 12
    p = (4, 3)               # non-square
    D = 32

    m = PatchEmbed(input_size=(H, W), patch_size=p, in_chans=C, embed_dim=D, bias=True )
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

    m = PatchEmbed(input_size=(Dd, H, W), patch_size=p, in_chans=C, embed_dim=E, bias=False )
    x = torch.randn(N, C, Dd, H, W)
    y = m(x)
    T = (Dd // p[0]) * (H // p[1]) * (W // p[2])
    assert y.shape == (N, T, E)
    (y.sum()).backward()


def test_2d_infer_spatial_dim_from_input_rank():
    torch.manual_seed(0)
    N, C, H, W = 1, 3, 8, 8
    m = PatchEmbed(input_size=None, patch_size=4, in_chans=C, embed_dim=16, bias=True )
    y = m(torch.randn(N, C, H, W))
    assert y.shape == (N, (H//4)*(W//4), 16)


def test_3d_infer_spatial_dim_from_input_rank():
    torch.manual_seed(0)
    N, C, Dd, H, W = 1, 2, 8, 8, 8
    m = PatchEmbed(input_size=None, patch_size=2, in_chans=C, embed_dim=20, bias=True )
    y = m(torch.randn(N, C, Dd, H, W))
    assert y.shape == (N, (Dd//2)*(H//2)*(W//2), 20)


def test_bad_rank_raises():
    m = PatchEmbed(input_size=None, patch_size=2, in_chans=3, embed_dim=8 )
    with pytest.raises(ValueError):
        m(torch.randn(2, 3))  # rank 2


def test_cuda_if_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    N, C, H, W = 2, 3, 12, 12
    m = PatchEmbed(input_size=(H, W), patch_size=4, in_chans=C, embed_dim=32 ).cuda()
    x = torch.randn(N, C, H, W, device="cuda")
    y = m(x)
    assert y.is_cuda
    assert y.shape == (N, (H//4)*(W//4), 32)


def _fill_sentinel(t: torch.Tensor, value: float = 0.1234) -> None:
    with torch.no_grad():
        t.fill_(value)


def _assert_not_allclose(a: torch.Tensor, value: float, atol: float = 1e-6) -> None:
    # True if at least one element differs from the sentinel value.
    assert not torch.allclose(a, torch.full_like(a, value), atol=atol), "Tensor unchanged; init likely not applied."


@pytest.mark.parametrize("bias", [True, False])
def test_patch_embed_2d_init_weights_changes_weights_and_zeroes_bias(bias: bool):
    torch.manual_seed(0)
    N, C, H, W = 1, 3, 16, 12
    embed_dim = 32
    patch = (4, 3)

    pe = PatchEmbed(
        input_size=(H, W),
        patch_size=patch,
        in_chans=C,
        embed_dim=embed_dim,
        bias=bias,
    )

    N, C, H, W = 2, 3, 16, 12
    pe(torch.randn(N, C, H, W))

    assert pe.patch_embedding_2d is not None
    proj = pe.patch_embedding_2d.proj  # Conv2d inside timm PatchEmbed

    # Set sentinel values
    _fill_sentinel(proj.weight, 0.1234)
    if proj.bias is not None:
        _fill_sentinel(proj.bias, 0.5678)

    # Re-init
    pe.init_weights()

    # Weights should no longer be the sentinel value (Xavier was applied)
    _assert_not_allclose(proj.weight, 0.1234)

    # Bias should be zero if it exists and we had set a nonzero sentinel before.
    if proj.bias is not None:
        assert torch.allclose(proj.bias, torch.zeros_like(proj.bias)), "Bias not zeroed by init."

    # Forward still works after re-init
    x = torch.randn(N, C, H, W)
    y = pe(x)
    # expected token count
    T = (H // patch[0]) * (W // patch[1])
    assert y.shape == (N, T, embed_dim)


@pytest.mark.parametrize("bias", [True, False])
def test_patch_embed_3d_init_weights_changes_weights_and_zeroes_bias(bias: bool):
    torch.manual_seed(0)
    N, C, D, H, W = 1, 2, 8, 8, 6
    embed_dim = 24
    patch = (2, 2, 3)  # mixed patch sizes to ensure shape handling

    pe = PatchEmbed(
        input_size=(D, H, W),
        patch_size=patch,
        in_chans=C,
        embed_dim=embed_dim,
        bias=bias,
    )

    pe(torch.randn(N, C, D, H, W))

    assert pe.patch_embedding_3d is not None
    proj3d = pe.patch_embedding_3d  # Conv3d

    # Set sentinel values
    _fill_sentinel(proj3d.weight, 0.1234)
    if proj3d.bias is not None:
        _fill_sentinel(proj3d.bias, 0.5678)

    # Re-init
    pe.init_weights()

    # Weights should no longer be the sentinel value (Xavier was applied)
    _assert_not_allclose(proj3d.weight, 0.1234)

    # Bias should be zero if it exists and we had set a nonzero sentinel before.
    if proj3d.bias is not None:
        assert torch.allclose(proj3d.bias, torch.zeros_like(proj3d.bias)), "Bias not zeroed by init."

    # Forward still works after re-init
    x = torch.randn(N, C, D, H, W)
    y = pe(x)
    T = (D // patch[0]) * (H // patch[1]) * (W // patch[2])
    assert y.shape == (N, T, embed_dim)


def test_patch_embed_init_weights_is_idempotent_like():
    """
    Calling init_weights twice should not error, and should still produce
    a valid projection (we don't assert exact equality because Xavier is random).
    """
    torch.manual_seed(0)
    pe = PatchEmbed(
        input_size=(16, 12),
        patch_size=(4, 3),
        in_chans=3,
        embed_dim=16,
        bias=True,
    )
    # Call weights - the weights should not be instantiated yet
    pe.init_weights()
    N, C, H, W = 2, 3, 16, 12
    pe(torch.randn(N, C, H, W))    
    # Now Initialize
    pe.init_weights()
    pe.init_weights()  # should not raise

    x = torch.randn(1, 3, 16, 12)
    y = pe(x)
    assert y.shape[0] == 1 and y.shape[-1] == 16
