import math
import pytest
import torch

from widit.timesteps import TimestepEmbedder


def test_timestep_embedding_even_dim_shapes_and_values():
    N = 5
    dim = 6  # even
    t = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    emb = TimestepEmbedder.timestep_embedding(t, dim, max_period=1000)

    assert emb.shape == (N, dim)
    assert emb.dtype == torch.float32
    # t=0 should give cos=1, sin=0 for all frequencies
    assert torch.allclose(emb[0, : dim // 2], torch.ones(dim // 2), atol=1e-6)
    assert torch.allclose(emb[0, dim // 2 :], torch.zeros(dim // 2), atol=1e-6)

def test_timestep_embedding_odd_dim_zero_tail():
    N = 3
    dim = 5  # odd → last column should be zeros
    t = torch.arange(N)
    emb = TimestepEmbedder.timestep_embedding(t, dim)
    assert emb.shape == (N, dim)
    tail = emb[:, -1]
    assert torch.allclose(tail, torch.zeros_like(tail), atol=0)

def test_timestep_embedding_is_deterministic():
    t = torch.tensor([0.5, 1.5, 3.0])
    dim = 8
    e1 = TimestepEmbedder.timestep_embedding(t, dim)
    e2 = TimestepEmbedder.timestep_embedding(t.clone(), dim)
    assert torch.allclose(e1, e2, atol=0, rtol=0)

def test_accepts_noncontiguous_t():
    base = torch.arange(10)
    t = base.view(5, 2)[:, 0]  # non-contiguous (stride > 1)
    emb = TimestepEmbedder.timestep_embedding(t, 8)
    assert emb.shape == (5, 8)

def test_raises_on_wrong_shape():
    with pytest.raises(ValueError):
        TimestepEmbedder.timestep_embedding(torch.zeros(2, 2), 8)

@pytest.mark.parametrize("freq_dim,hidden", [(16, 32), (7, 7), (9, 13)])
def test_forward_shapes(freq_dim, hidden):
    N = 4
    t = torch.arange(N)
    m = TimestepEmbedder(hidden_size=hidden, frequency_embedding_size=freq_dim)
    out = m(t)
    assert out.shape == (N, hidden)
    assert torch.isfinite(out).all()

def test_forward_autograd():
    torch.manual_seed(0)
    N = 3
    m = TimestepEmbedder(hidden_size=12, frequency_embedding_size=8)
    t = torch.tensor([0.0, 1.0, 2.0])
    out = m(t)                        # (N, 12)
    loss = (out ** 2).mean()
    loss.backward()
    # at least one parameter should have a non-zero grad
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)

def test_device_cpu_vs_cuda_consistency():
    N = 4
    dim = 10
    t_cpu = torch.linspace(0, 3, N)

    e_cpu = TimestepEmbedder.timestep_embedding(t_cpu, dim)

    if torch.cuda.is_available():
        t_cuda = t_cpu.to("cuda")
        e_cuda = TimestepEmbedder.timestep_embedding(t_cuda, dim).cpu()
        # cosine/sine are deterministic; tiny fp diffs acceptable
        assert torch.allclose(e_cpu, e_cuda, atol=1e-6, rtol=1e-6)
    else:
        pytest.skip("CUDA not available")

def test_module_on_cuda_if_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    N = 3
    m = TimestepEmbedder(hidden_size=16, frequency_embedding_size=8).cuda()
    t = torch.arange(N, device="cuda", dtype=torch.float32)
    out = m(t)
    assert out.is_cuda
    assert out.shape == (N, 16)
