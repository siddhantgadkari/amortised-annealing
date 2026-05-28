from __future__ import annotations

from typing import Any

import torch


def dsm_loss(
    model: Any,
    x0: torch.Tensor,
    schedule,
    t_eps: float = 1e-4,
    log_uniform_t: bool = False,
    loss_type: str = "eps",
) -> torch.Tensor:
    """Denoising Score Matching (DSM) loss.

    Two parameterisations are supported:

    loss_type="eps"  (default):
        L = E[ || model(t, x_t) - eps ||^2 ]
        model must have predict_score=False.

    loss_type="score":
        L = E[ sigma(t)^2 * || s_theta(t, x_t) - (-eps/sigma(t)) ||^2 ]
        Uses model.score() so it is correct for both predict_score settings.
        For predict_score=False this is mathematically identical to "eps".
        For predict_score=True this correctly supervises a score-predicting network.

    Args:
        model:         MLPScore instance.
        x0:            [N, d] clean samples from the training distribution.
        schedule:      DiffusionSchedule providing marginal_sample / eps_to_score.
        t_eps:         Minimum time value to avoid t=0 singularity.
        log_uniform_t: Sample t log-uniformly (equal focus per decade of t).
        loss_type:     "eps" or "score".
    """
    N = x0.shape[0]
    if log_uniform_t:
        u = torch.rand(N, device=x0.device, dtype=x0.dtype)
        t = t_eps * (1.0 / t_eps) ** u
    else:
        t = torch.rand(N, device=x0.device, dtype=x0.dtype) * (1.0 - t_eps) + t_eps
    x_t, eps = schedule.marginal_sample(x0, t)

    if loss_type == "eps":
        eps_pred = model(t, x_t)
        return ((eps_pred - eps) ** 2).sum(-1).mean()
    elif loss_type == "score":
        sigma = schedule.sigma(t).view(-1, 1)
        target_score = -eps / sigma
        score_pred = model.score(t, x_t, schedule)
        return (sigma ** 2 * (score_pred - target_score) ** 2).sum(-1).mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}. Choose 'eps' or 'score'.")


@torch.no_grad()
def loss_by_t_bin(
    model: Any,
    x_data: torch.Tensor,
    schedule,
    device: torch.device,
    batch_size: int = 8192,
    bins: list[tuple[float, float]] | None = None,
    loss_type: str = "eps",
) -> dict[str, float]:
    """Measure DSM loss separately in each time bin.

    Reveals whether the model fails specifically at small t (low-noise regime),
    which is where accurate scores are needed to land in narrow data modes.

    Args:
        model:      Trained score model.
        x_data:     [N, d] training samples (CPU tensor).
        schedule:   DiffusionSchedule.
        device:     Torch device.
        batch_size: Samples per bin evaluation.
        bins:       List of (t_lo, t_hi) pairs. Defaults to log-spaced bins.
        loss_type:  "eps" or "score" — must match how the model was trained.
    """
    if bins is None:
        bins = [(1e-4, 1e-2), (1e-2, 5e-2), (5e-2, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0)]

    model.eval()
    n = len(x_data)
    out = {}
    for lo, hi in bins:
        idx      = torch.randint(n, (batch_size,))
        x0       = x_data[idx].to(device)
        t        = torch.rand(batch_size, device=device, dtype=x0.dtype) * (hi - lo) + lo
        x_t, eps = schedule.marginal_sample(x0, t)
        if loss_type == "eps":
            pred = model(t, x_t)
            loss = ((pred - eps) ** 2).sum(-1).mean().item()
        else:
            sigma        = schedule.sigma(t).view(-1, 1)
            target_score = -eps / sigma
            score_pred   = model.score(t, x_t, schedule)
            loss = (sigma ** 2 * (score_pred - target_score) ** 2).sum(-1).mean().item()
        out[f"{lo:g}:{hi:g}"] = round(loss, 4)
    return out


