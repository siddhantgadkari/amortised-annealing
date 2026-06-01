from __future__ import annotations

import torch


def _normalise(log_weights: torch.Tensor) -> torch.Tensor:
    lw = log_weights - log_weights.max()
    w = torch.exp(lw)
    return w / w.sum()


def systematic_resample(log_weights: torch.Tensor) -> torch.Tensor:
    """Systematic resampling — lower variance than multinomial."""
    w = _normalise(log_weights)
    N = w.shape[0]
    cdf = torch.cumsum(w, dim=0)
    u0 = torch.rand(1, device=w.device, dtype=w.dtype) / N
    u = u0 + torch.arange(N, device=w.device, dtype=w.dtype) / N
    return torch.searchsorted(cdf, u).clamp(0, N - 1)


def stratified_resample(log_weights: torch.Tensor) -> torch.Tensor:
    """Stratified resampling — independent uniform draw in each stratum."""
    w = _normalise(log_weights)
    N = w.shape[0]
    cdf = torch.cumsum(w, dim=0)
    u = (torch.rand(N, device=w.device, dtype=w.dtype) + torch.arange(N, device=w.device, dtype=w.dtype)) / N
    return torch.searchsorted(cdf, u).clamp(0, N - 1)


def multinomial_resample(log_weights: torch.Tensor) -> torch.Tensor:
    w = _normalise(log_weights)
    N = w.shape[0]
    return torch.multinomial(w, N, replacement=True)


def resample(log_weights: torch.Tensor, method: str = "systematic") -> torch.Tensor:
    """Dispatch resampling by name. Returns [N] integer indices."""
    if method == "systematic":
        return systematic_resample(log_weights)
    elif method == "stratified":
        return stratified_resample(log_weights)
    elif method == "multinomial":
        return multinomial_resample(log_weights)
    else:
        raise ValueError(f"Unknown resampling method: {method!r}")
