from __future__ import annotations

import math
from typing import Callable

import torch
from tqdm import tqdm

from .ess import ess
from .particles import ParticleCloud
from .resampling import resample
from .sampler import SMCDiagnostics


class FKCAnnealedSampler:
    """Feynman-Kac Corrector annealed sampler (Skreta et al. ICML 2025, Prop D.1).

    Runs the reverse SDE from t=1 (noise) to t=0 with the score scaled by
    gamma = beta_target / beta_train.  At each EM step the path-space importance
    weight is accumulated:

        d log w = [(gamma-1)*div(f_t)  +  beta(t)/2 * gamma*(gamma-1)*||s||^2] dt

    For VP linear drift, div(f_t) = -d/2 * beta(t) is constant and cancels in
    SNIS normalisation, so include_divergence defaults to False.

    Resampling is applied whenever ESS/N drops below ess_threshold.
    """

    def __init__(
        self,
        reverse_sde,
        energy_fn: Callable[[torch.Tensor], torch.Tensor],
        beta_train: float,
        beta_target: float,
        n_steps: int = 500,
        ess_threshold: float = 0.5,
        resampling_method: str = "systematic",
        include_divergence: bool = False,
        t_start: float = 1.0,
        t_end: float = 1e-3,
    ):
        self._reverse_sde = reverse_sde
        self._energy_fn = energy_fn
        self.beta_train = beta_train
        self.beta_target = beta_target
        self.n_steps = n_steps
        self.ess_threshold = ess_threshold
        self.resampling_method = resampling_method
        self.include_divergence = include_divergence
        self.t_start = t_start
        self.t_end = t_end

    @torch.no_grad()
    def run(
        self,
        N: int,
        d: int,
        device: torch.device,
        show_progress: bool = True,
    ) -> tuple[ParticleCloud, SMCDiagnostics]:
        gamma = self.beta_target / self.beta_train
        schedule = self._reverse_sde.schedule

        x = torch.randn(N, d, device=device)
        log_weights = torch.zeros(N, device=device)

        times = torch.linspace(self.t_start, self.t_end, self.n_steps + 1, device=device)
        dt = (self.t_start - self.t_end) / self.n_steps

        diag = SMCDiagnostics()
        cloud = ParticleCloud(x, log_weights)
        diag.record(self.t_start, cloud, self._energy_fn)

        steps = range(self.n_steps)
        pbar = tqdm(steps, desc="FKC", dynamic_ncols=True) if show_progress else None

        for i in (pbar if pbar is not None else steps):
            t = times[i].expand(N)

            score = self._reverse_sde.model.score(t, x, schedule)
            score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
            score = score.clamp(-1e3, 1e3)

            beta_t = schedule.beta(t).view(N)
            score_norm_sq = (score ** 2).sum(dim=-1)
            correction = 0.5 * beta_t * gamma * (gamma - 1) * score_norm_sq

            if self.include_divergence:
                correction = correction + (gamma - 1) * (-d / 2) * beta_t

            log_weights = log_weights + correction * dt

            drift = schedule.reverse_drift(t, x, gamma * score)
            diffusion = schedule.reverse_diffusion(t, x)
            x = x + drift * dt + diffusion * math.sqrt(dt) * torch.randn_like(x)
            x = torch.nan_to_num(x, nan=0.0)

            log_weights = torch.nan_to_num(log_weights, nan=float("-inf"))
            finite = torch.isfinite(log_weights)
            if finite.any():
                log_weights = log_weights - log_weights[finite].max()

            ess_ratio = ess(log_weights) / N
            if ess_ratio < self.ess_threshold:
                idx = resample(log_weights, method=self.resampling_method)
                x = x[idx]
                log_weights = torch.zeros(N, device=device)
                diag.n_resamples += 1

            cloud = ParticleCloud(x, log_weights)
            diag.record(times[i + 1].item(), cloud, self._energy_fn)

            if pbar is not None:
                pbar.set_postfix(
                    t=f"{times[i+1].item():.3f}",
                    ess=f"{ess_ratio:.2f}",
                    E_min=f"{diag.best_energies[-1]:.3f}",
                )

        return ParticleCloud(x, log_weights), diag
