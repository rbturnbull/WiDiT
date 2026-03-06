# tests/test_models.py
import pytest
import torch

from widit import WiDiT, Unet, PRESETS


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

@pytest.mark.parametrize("out_channels", [1, 2])
def test_widit_2d_forward_shapes_and_grad(out_channels):
    torch.manual_seed(0)
    x, t, cond = _rand_2d(n=2, c=3, h=16, w=12)  # grid = (8,6)

    model = WiDiT(
        spatial_dim=2,
        input_size=(16, 12),
        patch_size=2,
        in_channels=3,
        hidden_size=128,
        depth=4,
        num_heads=4,
        window_size=8,     # grid 8x6 gets padded along width inside the block(s)
        mlp_ratio=2.0,
        out_channels=out_channels,
    )
    # NEW ORDER: (x, timestep, *, conditioned=...)
    y = model(x, t, conditioned=cond)

    assert y.shape == (2, out_channels, 16, 12)

    target = torch.randn_like(y)
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_widit_2d_tuple_window_and_padding():
    torch.manual_seed(0)
    # Choose H,W s.t. token grid (H/p, W/p) is NOT a multiple of window -> triggers padding.
    x, t, cond = _rand_2d(n=1, c=2, h=18, w=14)  # p=2 -> grid 9x7
    model = WiDiT(
        spatial_dim=2,
        input_size=(18, 14),
        patch_size=2,
        in_channels=2,
        hidden_size=96,
        depth=3,
        num_heads=3,
        window_size=(7, 4),  # non-uniform ws; grid 9x7 pads to 14x8 internally
        mlp_ratio=2.0,
        out_channels=4,
    )
    y = model(x, t, conditioned=cond)
    assert y.shape == (1, 4, 18, 14)  # out_channels = 4


def test_widit_2d_timestep_none():
    torch.manual_seed(0)
    x, t, cond = _rand_2d(n=1, c=2, h=16, w=12)
    model = WiDiT(
        spatial_dim=2,
        input_size=(16, 12),
        patch_size=2,
        in_channels=2,
        hidden_size=96,
        depth=2,
        num_heads=4,
        window_size=8,
        mlp_ratio=2.0,
    )
    y = model(x, timestep=None, conditioned=cond)
    assert y.shape == (1, 2, 16, 12)
    (y.sum()).backward()  # ensure differentiable without timestep


# -------------------- Core WiDiT (3D) --------------------

@pytest.mark.parametrize("out_channels", [1, 2])
def test_widit_3d_forward_shapes_and_grad(out_channels):
    torch.manual_seed(0)
    x, t, cond = _rand_3d(n=2, c=1, d=8, h=8, w=6)  # p=2 -> grid 4x4x3

    model = WiDiT(
        spatial_dim=3,
        input_size=(8, 8, 6),
        patch_size=2,
        in_channels=1,
        hidden_size=144,
        depth=3,
        num_heads=6,
        window_size=(4, 4, 4),  # pads W-grid 3 -> 4 internally
        mlp_ratio=2.0,
        out_channels=out_channels,
    )
    y = model(x, t, conditioned=cond)
    assert y.shape == (2, out_channels, 8, 8, 6)

    target = torch.randn_like(y)
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_widit_3d_nonuniform_window_padding():
    torch.manual_seed(0)
    # Make dims divisible by patch_size=2 to avoid conv3d flooring.
    x, t, cond = _rand_3d(n=1, c=2, d=10, h=10, w=8)  # grid 5x5x4
    model = WiDiT(
        spatial_dim=3,
        input_size=(10, 10, 8),
        patch_size=2,
        in_channels=2,
        hidden_size=160,
        depth=2,
        num_heads=5,
        window_size=(5, 3, 4),  # will pad each token-grid axis appropriately
        mlp_ratio=2.0,
        out_channels=4,
    )
    y = model(x, t, conditioned=cond)
    assert y.shape == (1, 4, 10, 10, 8)  # 2*in_channels, original spatial sizes


# -------------------- CUDA (optional) --------------------

def test_widit_cuda_if_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    x, t, cond = _rand_2d(n=2, c=1, h=16, w=16)
    model = WiDiT(
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
    y = model(x.cuda(), t.cuda(), conditioned=cond.cuda())
    assert y.is_cuda
    assert y.shape == (2, 2, 16, 16)


# -------------------- Error handling --------------------

def test_widit_bad_spatial_dim_asserts():
    with pytest.raises(AssertionError):
        _ = WiDiT(spatial_dim=4)  # only 2 or 3 supported


def test_widit_mismatched_conditioned_shape_raises():
    x, t, cond = _rand_2d(n=1, c=1, h=8, w=8)
    cond = cond[:, :, :7, :8]  # wrong H
    model = WiDiT(
        spatial_dim=2,
        input_size=(8, 8),
        patch_size=2,
        in_channels=1,
        hidden_size=64,
        depth=1,
        num_heads=2,
    )
    with pytest.raises(AssertionError):
        _ = model(x, t, conditioned=cond)  # NEW ORDER


# -------------------- Persistence --------------------

def test_widit_save_and_load_roundtrip(tmp_path):
    torch.manual_seed(123)
    model = WiDiT(
        spatial_dim=2,
        input_size=(8, 8),
        patch_size=2,
        in_channels=1,
        hidden_size=32,
        depth=2,
        num_heads=4,
        window_size=4,
        mlp_ratio=3.0,
        learn_sigma=False,
        use_conditioning=False,
    )
    for param in model.parameters():
        param.data.copy_(torch.randn_like(param))  # give the weights non-trivial values

    save_path = tmp_path / "checkpoints" / "widit.pt"
    model.save(save_path)

    assert save_path.exists()
    loaded = WiDiT.load(save_path, map_location="cpu")

    assert loaded.config == model.config
    original_state = model.state_dict()
    loaded_state = loaded.state_dict()
    for key in original_state:
        torch.testing.assert_close(loaded_state[key], original_state[key])
    assert all(not p.is_cuda for p in loaded.parameters())


# -------------------- PRESETS --------------------

def _small_input_for_preset(name: str):
    """Return (x, t, cond) sized to be compatible with default patch/window and produce grads."""
    is_3d = ("3D" in name) or (
        (name.startswith("WiDiT-") or name.startswith("Unet-"))
        and "2D" not in name
    )
    is_unet = "Unet" in name
    needs_large = ("-L" in name) or ("-XL" in name)

    # Unet has `layers` maxpool downsamples; ensure spatial dims >= 2^(layers+1)
    # for reflect padding to be valid at the deepest layers.
    if is_3d:
        if is_unet and needs_large:
            return _rand_3d(n=1, c=1, d=32, h=32, w=32)
        if is_unet:
            return _rand_3d(n=1, c=1, d=16, h=16, w=16)
        return _rand_3d(n=1, c=1, d=8, h=8, w=8)
    else:
        if is_unet and needs_large:
            return _rand_2d(n=1, c=1, h=32, w=32)
        if is_unet:
            return _rand_2d(n=1, c=1, h=16, w=16)
        return _rand_2d(n=1, c=1, h=16, w=12)


def test_presets_dict_has_expected_entries():
    expected = {
        "WiDiT-B", "WiDiT-M", "WiDiT-L", "WiDiT-XL",
        "WiDiT2D-B", "WiDiT2D-M", "WiDiT2D-L", "WiDiT2D-XL",
        "WiDiT3D-B", "WiDiT3D-M", "WiDiT3D-L", "WiDiT3D-XL",
        "Unet-B", "Unet-M", "Unet-L", "Unet-XL",
        "Unet2D-B", "Unet2D-M", "Unet2D-L", "Unet2D-XL",
        "Unet3D-B", "Unet3D-M", "Unet3D-L", "Unet3D-XL",
    }
    assert expected.issubset(set(PRESETS.keys())), f"Missing presets: {expected - set(PRESETS.keys())}"


@pytest.mark.parametrize("name", [
    "WiDiT-B", "WiDiT-M", "WiDiT-L", "WiDiT-XL",
    "WiDiT2D-B", "WiDiT2D-M", "WiDiT2D-L", "WiDiT2D-XL",
    "WiDiT3D-B", "WiDiT3D-M", "WiDiT3D-L", "WiDiT3D-XL",
    "Unet-B", "Unet-M", "Unet-L", "Unet-XL",
    "Unet2D-B", "Unet2D-M", "Unet2D-L", "Unet2D-XL",
    "Unet3D-B", "Unet3D-M", "Unet3D-L", "Unet3D-XL",
])
def test_every_preset_builds_and_runs(name):
    torch.manual_seed(0)
    ctor = PRESETS[name]
    model = ctor(in_channels=1, out_channels=1)  # keep defaults for each preset
    x, t, cond = _small_input_for_preset(name)
    y = model(x, t, conditioned=cond)  # NEW ORDER

    # Validate output shape
    if ("3D" in name) or (
        (name.startswith("WiDiT-") or name.startswith("Unet-"))
        and "2D" not in name
    ):
        n, c, d, h, w = x.shape
        assert y.shape == (n, c, d, h, w)
    else:
        n, c, h, w = x.shape
        assert y.shape == (n, c, h, w)

    target = torch.randn_like(y)
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


# -------------------- Unet interpolation path --------------------

def test_unet_2d_interpolates_for_odd_sizes(monkeypatch):
    import widit.models as models

    calls = []
    orig = models.F.interpolate

    def _wrapped(*args, **kwargs):
        calls.append((args, kwargs))
        return orig(*args, **kwargs)

    monkeypatch.setattr(models.F, "interpolate", _wrapped)

    model = Unet(spatial_dim=2, in_channels=1, filters=8, kernel_size=3, layers=2, use_conditioning=False)
    x = torch.randn(1, 1, 9, 11)
    t = torch.randint(0, 1000, (1,), dtype=torch.long)

    y = model(x, t)
    assert y.shape == (1, 1, 9, 11)
    assert len(calls) == 2


def test_unet_3d_interpolates_for_odd_sizes(monkeypatch):
    import widit.models as models

    calls = []
    orig = models.F.interpolate

    def _wrapped(*args, **kwargs):
        calls.append((args, kwargs))
        return orig(*args, **kwargs)

    monkeypatch.setattr(models.F, "interpolate", _wrapped)

    model = Unet(spatial_dim=3, in_channels=1, filters=4, kernel_size=3, layers=2, use_conditioning=False)
    x = torch.randn(1, 1, 9, 9, 11)
    t = torch.randint(0, 1000, (1,), dtype=torch.long)

    y = model(x, t)
    assert y.shape == (1, 1, 9, 9, 11)
    assert len(calls) == 2


# -------- Optional conditioning OFF path --------

def test_widit_unconditioned_path_runs_and_shapes():
    x = torch.randn(2, 3, 16, 12)
    model = WiDiT(spatial_dim=2, in_channels=3, hidden_size=128, depth=2,
                  num_heads=4, patch_size=2, window_size=8, use_conditioning=False, learn_sigma=True)
    y = model(x, timestep=torch.randint(0, 1000, (2,)))
    assert y.shape == (2, 3, 16, 12)  # 2 * in_channels


def test_widit_unconditioned_rejects_conditioned_arg():
    x = torch.randn(1, 1, 8, 8)
    cond = torch.randn_like(x)
    model = WiDiT(spatial_dim=2, in_channels=1, hidden_size=64, depth=1,
                  num_heads=2, patch_size=2, window_size=4, use_conditioning=False)
    with pytest.raises(AssertionError):
        _ = model(x, timestep=None, conditioned=cond)


def test_widit_conditioned_requires_conditioned_arg():
    x = torch.randn(1, 1, 8, 8)
    model = WiDiT(spatial_dim=2, in_channels=1, hidden_size=64, depth=1,
                  num_heads=2, patch_size=2, window_size=4, use_conditioning=True)
    with pytest.raises(AssertionError):
        _ = model(x, timestep=None)  # no conditioned passed
