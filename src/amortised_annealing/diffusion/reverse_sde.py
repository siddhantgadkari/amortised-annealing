from __future__ import annotations

from typing import Optional

import torch
from tqdm import tqdm


class ReverseSDE:
    """Reverse-time SDE for a trained score network.

    For VP schedule the reverse SDE (Ito, running t: 1 -> 0) is:
        dx = [0.5*beta(t)*x + beta(t)*s_theta(t,x)] dt + sqrt(beta(t)) dW

    where dt > 0 and we advance from t_curr to t_curr - dt.

    For VE schedule the reverse SDE is:
        dx = -d(sigma^2)/dt * s_theta(t,x) dt + sqrt(d(sigma^2)/dt) dW

    The schedule exposes `reverse_drift` and `reverse_diffusion` so this class
    is schedule-agnostic.

    The score network is called as: model.score(t, x, schedule)
    which handles the eps-to-score conversion internally.
    """

    def __init__(self, model, schedule):
        self.model = model
        self.schedule = schedule

    @torch.no_grad()
    def step(
        self,
        x: torch.Tensor,
        t_cur: torch.Tensor,
        dt: float,
        temperature_scale: float = 1.0,
    ) -> torch.Tensor:
        """One Euler-Maruyama step: x_{t-dt} = x_t + drift*dt + diffusion*sqrt(dt)*eps.

        Args:
            x:                 [N, d] current particles
            t_cur:             [N] or scalar current time
            dt:                Step size (positive; we subtract internally)
            temperature_scale: Multiply the score by this factor.
                               Set to beta_target/beta_train for heuristic annealing.
                               This is a proposal heuristic, NOT an exact correction.

        Returns:
            x_next: [N, d]
        """
        if t_cur.dim() == 0:
            t_cur = t_cur.expand(x.shape[0])

        score = self.model.score(t_cur, x, self.schedule) * temperature_scale
        # Guard against score blow-up from poorly-trained or out-of-distribution inputs
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        score = score.clamp(-1e3, 1e3)
        drift = self.schedule.reverse_drift(t_cur, x, score)
        diffusion = self.schedule.reverse_diffusion(t_cur, x)
        eps = torch.randn_like(x)
        x_next = x + drift * dt + diffusion * (dt**0.5) * eps
        return torch.nan_to_num(x_next, nan=0.0)


@torch.no_grad()
def euler_maruyama_sample(
    reverse_sde: ReverseSDE,
    n_samples: int,
    n_steps: int,
    device: torch.device,
    t_start: float = 1.0,
    t_end: float = 1e-3,
    temperature_scale: float = 1.0,
    x_init: Optional[torch.Tensor] = None,
    show_progress: bool = False,
) -> torch.Tensor:
    """Generate samples via Euler-Maruyama integration of the reverse SDE.

    Starts from x_T ~ N(0, I) (or x_init if provided) and integrates
    the reverse SDE from t_start down to t_end.

    Args:
        reverse_sde:       ReverseSDE wrapping model + schedule.
        n_samples:         Number of particles to generate.
        n_steps:           Number of discretisation steps.
        device:            Torch device.
        t_start:           Starting time (≈1 for VP, pure noise).
        t_end:             Ending time (>0 to avoid score singularity at t=0).
        temperature_scale: Score multiplier (use 1.0 for standard sampling).
        x_init:            Optional [n_samples, d] starting particles.
        show_progress:     Show tqdm bar.

    Returns:
        x: [n_samples, d] approximate samples from the target.
    """
    dim = reverse_sde.model.dim

    x = x_init.clone().to(device) if x_init is not None else torch.randn(n_samples, dim, device=device)

    times = torch.linspace(t_start, t_end, n_steps + 1, device=device)
    dt = (t_start - t_end) / n_steps

    iterator = range(n_steps)
    if show_progress:
        iterator = tqdm(iterator, desc="reverse SDE", dynamic_ncols=True)

    for i in iterator:
        t = times[i].expand(n_samples)
        x = reverse_sde.step(x, t, dt, temperature_scale=temperature_scale)

    return x

