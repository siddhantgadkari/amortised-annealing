from __future__ import annotations

import torch


def dsm_loss(
    model: torch.nn.Module,
    x0: torch.Tensor,
    schedule,
    t_eps: float = 1e-4,
    log_uniform_t: bool = False,
) -> torch.Tensor:
    """Denoising Score Matching (DSM) loss.

    For an eps-prediction model:
        L = E_{t, x_0, eps}[ || model(t, x_t) - eps ||^2 ]

    Args:
        model:         MLPScore with predict_score=False (eps-prediction).
        x0:            [N, d] clean samples from the training distribution.
        schedule:      DiffusionSchedule providing marginal_sample.
        t_eps:         Minimum time value to avoid t=0 singularity.
        log_uniform_t: If True, sample t log-uniformly so each decade of t
                       receives equal training focus. Helps when the model
                       underperforms at small t (sharp score near data modes).

    Returns:
        Scalar loss.
    """
    N = x0.shape[0]
    if log_uniform_t:
        # t = t_eps * (1/t_eps)^u, u ~ Uniform[0,1]  =>  log t ~ Uniform[log t_eps, 0]
        u = torch.rand(N, device=x0.device, dtype=x0.dtype)
        t = t_eps * (1.0 / t_eps) ** u
    else:
        t = torch.rand(N, device=x0.device, dtype=x0.dtype) * (1.0 - t_eps) + t_eps
    x_t, eps = schedule.marginal_sample(x0, t)

    eps_pred = model(t, x_t)
    return ((eps_pred - eps) ** 2).sum(-1).mean()


@torch.no_grad()
def loss_by_t_bin(
    model: torch.nn.Module,
    x_data: torch.Tensor,
    schedule,
    device: torch.device,
    batch_size: int = 8192,
    bins: list[tuple[float, float]] | None = None,
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
        bins:       List of (t_lo, t_hi) pairs. Defaults to log-spaced bins
                    covering [1e-4, 1.0].

    Returns:
        Dict mapping "t_lo:t_hi" -> mean loss for that bin.
    """
    if bins is None:
        bins = [(1e-4, 1e-2), (1e-2, 5e-2), (5e-2, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0)]

    model.eval()
    n = len(x_data)
    out = {}
    for lo, hi in bins:
        idx  = torch.randint(n, (batch_size,))
        x0   = x_data[idx].to(device)
        t    = torch.rand(batch_size, device=device, dtype=x0.dtype) * (hi - lo) + lo
        x_t, eps = schedule.marginal_sample(x0, t)
        eps_pred = model(t, x_t)
        loss = ((eps_pred - eps) ** 2).sum(-1).mean().item()
        out[f"{lo:g}:{hi:g}"] = round(loss, 4)
    return out


