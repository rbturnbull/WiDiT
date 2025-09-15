# tests/test_models.py
import pytest
import torch

# Adjust these to match your package layout
from widit.models import Widit, PRESETS


def _rand_2d(n=2, c=1, h=16, w=12):
    """Sizes chosen divisible by patch_size=2 used in tests/presets."""
    x = torch.randn(n, c, h, w)
    cond = torch.randn_like(x)
    t = torch.randint(0, 1000, (n,), dtype=torch.long)
    return x, t, cond


def _rand_3d(n=2, c=1, d=8, h=8, w=6):
    """All dimensions divisible by 2 (default patch_size)."""
    x = torch.randn(n, c, d, h, w)
    cond = torch.randn_like(x)
    t = torch.randint(0, 1000, (n,), dtype=torch.long)
    return x, t, cond


# -------------------- Core WiDiT (2D) --------------------

@pytest.mark.parametrize("learn_sigma", [True, False])
def test_widit_2d_forward_shapes_and_grad(learn_sigma):
    torch.manual_seed(0)
    x, t, cond = _rand_2d(n=2, c=3, h=16, w=12)  # grid = (8,6)

    model = Widit(
        spatial_dim=2,
        input_size=(16, 12),
        patch_size=2,
        in_channels=3,
        hidden_size=128,
        depth=4,
        num_heads=4,
        window_size=8,     # grid 8x6 gets padded along width inside the block(s)
        mlp_ratio=2.0,
        learn_sigma=learn_sigma,
    )
    y = model(x, t, cond)

    outc = 3 * (2 if learn_sigma else 1)
    assert y.shape == (2, outc, 16, 12)

    # Use a non-zero target to avoid zero-grad at init (adaLN-Zero + zero head)
    target = torch.randn_like(y)
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_widit_2d_tuple_window_and_padding():
    torch.manual_seed(0)
    # Choose H,W s.t. token grid (H/p, W/p) is NOT a multiple of window -> triggers padding.
    x, t, cond = _rand_2d(n=1, c=2, h=18, w=14)  # p=2 -> grid 9x7
    model = Widit(
        spatial_dim=2,
        input_size=(18, 14),
        patch_size=2,
        in_channels=2,
        hidden_size=96,
        depth=3,
        num_heads=3,
        window_size=(7, 4),  # non-uniform ws; grid 9x7 pads to 14x8 internally
        mlp_ratio=2.0,
        learn_sigma=True,
    )
    y = model(x, t, cond)
    assert y.shape == (1, 4, 18, 14)  # 2*in_channels with learn_sigma=True


# -------------------- Core WiDiT (3D) --------------------

@pytest.mark.parametrize("learn_sigma", [True, False])
def test_widit_3d_forward_shapes_and_grad(learn_sigma):
    torch.manual_seed(0)
    x, t, cond = _rand_3d(n=2, c=1, d=8, h=8, w=6)  # p=2 -> grid 4x4x3

    model = Widit(
        spatial_dim=3,
        input_size=(8, 8, 6),
        patch_size=2,
        in_channels=1,
        hidden_size=144,
        depth=3,
        num_heads=6,
        window_size=(4, 4, 4),  # pads W-grid 3 -> 4 internally
        mlp_ratio=2.0,
        learn_sigma=learn_sigma,
    )
    y = model(x, t, cond)
    outc = 1 * (2 if learn_sigma else 1)
    assert y.shape == (2, outc, 8, 8, 6)

    target = torch.randn_like(y)
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_widit_3d_nonuniform_window_padding():
    torch.manual_seed(0)
    # Make dims divisible by patch_size=2 to avoid conv3d flooring.
    x, t, cond = _rand_3d(n=1, c=2, d=10, h=10, w=8)  # grid 5x5x4
    model = Widit(
        spatial_dim=3,
        input_size=(10, 10, 8),
        patch_size=2,
        in_channels=2,
        hidden_size=160,
        depth=2,
        num_heads=5,
        window_size=(5, 3, 4),  # will pad each token-grid axis appropriately
        mlp_ratio=2.0,
        learn_sigma=True,
    )
    y = model(x, t, cond)
    assert y.shape == (1, 4, 10, 10, 8)  # 2*in_channels, original spatial sizes


# -------------------- CUDA (optional) --------------------

def test_widit_cuda_if_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    x, t, cond = _rand_2d(n=2, c=1, h=16, w=16)
    model = Widit(
        spatial_dim=2,
        input_size=(16, 16),
        patch_size=2,
        in_channels=1,
        hidden_size=64,
        depth=2,
        num_heads=4,
        window_size=8,
        mlp_ratio=2.0,
        learn_sigma=True,
    ).cuda()
    y = model(x.cuda(), t.cuda(), cond.cuda())
    assert y.is_cuda
    assert y.shape == (2, 2, 16, 16)


# -------------------- Error handling --------------------

def test_widit_bad_spatial_dim_asserts():
    with pytest.raises(AssertionError):
        _ = Widit(spatial_dim=4)  # only 2 or 3 supported


def test_widit_mismatched_conditioned_shape_raises():
    x, t, cond = _rand_2d(n=1, c=1, h=8, w=8)
    cond = cond[:, :, :7, :8]  # wrong H
    model = Widit(
        spatial_dim=2,
        input_size=(8, 8),
        patch_size=2,
        in_channels=1,
        hidden_size=64,
        depth=1,
        num_heads=2,
    )
    with pytest.raises(AssertionError):
        _ = model(x, t, cond)


# -------------------- PRESETS --------------------

def _small_input_for_preset(name: str):
    """Return (x, t, cond) sized to be compatible with default patch/window and produce grads."""
    if "3D" in name:
        # patch_size=2 by default → pick multiples of 2
        return _rand_3d(n=1, c=1, d=8, h=8, w=6)
    else:
        return _rand_2d(n=1, c=1, h=16, w=12)


def test_presets_dict_has_expected_entries():
    expected = {
        "WiDiT-B/2", "WiDiT-M/2", "WiDiT-L/2", "WiDiT-XL/2",
        "WiDiT3D-B/2", "WiDiT3D-M/2", "WiDiT3D-L/2", "WiDiT3D-XL/2",
    }
    assert expected.issubset(set(PRESETS.keys())), f"Missing presets: {expected - set(PRESETS.keys())}"


@pytest.mark.parametrize("name", [
    "WiDiT-B/2", "WiDiT-M/2", "WiDiT-L/2", "WiDiT-XL/2",
    "WiDiT3D-B/2", "WiDiT3D-M/2", "WiDiT3D-L/2", "WiDiT3D-XL/2",
])
def test_every_preset_builds_and_runs(name):
    torch.manual_seed(0)
    ctor = PRESETS[name]
    model = ctor(in_channels=1, learn_sigma=True)  # keep defaults for each preset
    x, t, cond = _small_input_for_preset(name)
    y = model(x, t, cond)

    # Validate output shape
    if "3D" in name:
        n, c, d, h, w = x.shape
        assert y.shape == (n, 2 * c, d, h, w)
    else:
        n, c, h, w = x.shape
        assert y.shape == (n, 2 * c, h, w)

    # Use non-zero target so grads are non-zero at init
    target = torch.randn_like(y)
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
