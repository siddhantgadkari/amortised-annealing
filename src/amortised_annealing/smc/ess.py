from __future__ import annotations

import math

import torch


def ess(log_weights: torch.Tensor) -> float:
    """Effective Sample Size from unnormalised log-weights.

    ESS = (Σ w_i)² / Σ w_i²  in [1, N].
    """
    lw = log_weights - log_weights.max()
    w = torch.exp(lw)
    w = w / w.sum()
    return (1.0 / (w ** 2).sum()).item()


def log_ess_ratio(log_weights: torch.Tensor) -> float:
    """log(ESS / N) — convenient for thresholding."""
    N = log_weights.shape[0]
    return math.log(ess(log_weights)) - math.log(N)
