import torch
from torch import nn
import math


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
        if t.ndim != 1:
            raise ValueError(f"`t` must be 1-D of shape (N,), got {tuple(t.shape)}")

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

    def init_weights(self, std: float = 0.02) -> None:
        """
        Initialize weights like your previous models.py:
        - Normal(0, std) on the two Linear *weights* in the MLP.
        - Leave biases unchanged.
        """
        nn.init.normal_(self.mlp[0].weight, std=std)
        nn.init.normal_(self.mlp[2].weight, std=std)
