from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbed(nn.Module):
    """Sinusoidal time embedding (as in DDPM / transformer positional encoding)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [N] -> [N, embed_dim]"""
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / (half - 1)
        )
        args = t.unsqueeze(1) * freqs.unsqueeze(0)  # [N, half]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [N, embed_dim]


class MLPScore(nn.Module):
    """MLP score network: (t, x) -> s_theta(t, x) ≈ ∇_x log q_t(x).

    Predicts eps by default; call .score(t, x, schedule) to get the true score.

    Architecture: sinusoidal time embed -> concat with x -> MLP -> output dim
    """

    def __init__(
        self,
        dim: int,
        hidden_dims: tuple[int, ...] = (256, 256, 256),
        time_embed_dim: int = 64,
        activation: str = "silu",
        predict_score: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.predict_score = predict_score

        self.time_embed = SinusoidalTimeEmbed(time_embed_dim)

        activation_fn = {
            "silu": nn.SiLU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
        }[activation]

        in_dim = dim + time_embed_dim
        layers = []
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), activation_fn()])
            in_dim = h
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """t: [N], x: [N, d] -> predicted eps (or score), shape [N, d]"""
        t_emb = self.time_embed(t)
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)

    def score(self, t: torch.Tensor, x: torch.Tensor, schedule) -> torch.Tensor:
        """Return ∇_x log q_t(x), converting from eps-prediction if needed."""
        out = self.forward(t, x)
        if self.predict_score:
            return out
        return schedule.eps_to_score(out, t)
