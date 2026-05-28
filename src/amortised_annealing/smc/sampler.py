from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import torch
from tqdm import tqdm

from .ess import ess
from .particles import ParticleCloud
from .resampling import resample


@dataclass
class SMCDiagnostics:
    betas:             List[float] = field(default_factory=list)
    ess_ratios:        List[float] = field(default_factory=list)
    log_weight_means:  List[float] = field(default_factory=list)
    log_weight_stds:   List[float] = field(default_factory=list)
    best_energies:     List[float] = field(default_factory=list)
    mean_energies:     List[float] = field(default_factory=list)
    n_resamples:       int = 0

    def record(
        self,
        beta: float,
        cloud: ParticleCloud,
        energy_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self.betas.append(beta)
        self.ess_ratios.append(cloud.ess_ratio())
        lw = cloud.log_weights
        self.log_weight_means.append(lw.mean().item())
        self.log_weight_stds.append(lw.std().item())
        with torch.no_grad():
            e = energy_fn(cloud.x)
        self.best_energies.append(e.min().item())
        self.mean_energies.append(e.mean().item())


class SMCSampler:
    """Generic Sequential Monte Carlo sampler.

    For each step k along the beta ladder:
      1. Mutate:  x <- mutation_kernel(x, beta_k)
      2. Weight:  log_w += weight_update(x, beta_{k-1}, beta_k)
      3. Resample if ESS/N < ess_threshold

    mutation_kernel: (x: Tensor, beta: float) -> Tensor
    weight_update:   (x: Tensor, beta_prev: float, beta_curr: float) -> Tensor [N]
    """

    def __init__(
        self,
        mutation_kernel: Callable,
        weight_update:   Callable,
        energy_fn:       Callable[[torch.Tensor], torch.Tensor],
        ess_threshold:   float = 0.5,
        resampling_method: str = "systematic",
    ):
        self.mutation_kernel   = mutation_kernel
        self.weight_update     = weight_update
        self.energy_fn         = energy_fn
        self.ess_threshold     = ess_threshold
        self.resampling_method = resampling_method

    def run(
        self,
        initial_cloud: ParticleCloud,
        beta_ladder:   torch.Tensor,
        show_progress: bool = True,
    ) -> tuple[ParticleCloud, SMCDiagnostics]:
        """Run SMC over the full beta ladder.

        Args:
            initial_cloud: Particles approximating pi_{beta_ladder[0]}.
            beta_ladder:   [K+1] tensor; beta_ladder[0] is the starting temperature.
            show_progress: tqdm progress bar.

        Returns:
            (final_cloud, diagnostics)
        """
        cloud = initial_cloud.clone()
        diag  = SMCDiagnostics()
        N     = cloud.n_particles
        ladder = beta_ladder.tolist()

        diag.record(ladder[0], cloud, self.energy_fn)

        steps = range(1, len(ladder))
        pbar = tqdm(steps, desc="SMC", dynamic_ncols=True) if show_progress else None

        for k in (pbar if pbar is not None else steps):
            beta_prev = ladder[k - 1]
            beta_curr = ladder[k]

            cloud.x = self.mutation_kernel(cloud.x, beta_curr)

            delta_lw = self.weight_update(cloud.x, beta_prev, beta_curr)
            delta_lw = torch.nan_to_num(delta_lw, nan=float("-inf"))
            cloud.log_weights = cloud.log_weights + delta_lw

            finite = torch.isfinite(cloud.log_weights)
            if finite.any():
                cloud.log_weights = cloud.log_weights - cloud.log_weights[finite].max()
            cloud.log_weights = torch.nan_to_num(cloud.log_weights, nan=float("-inf"))

            ess_ratio = ess(cloud.log_weights) / N
            if ess_ratio < self.ess_threshold:
                idx = resample(cloud.log_weights, method=self.resampling_method)
                cloud.x = cloud.x[idx]
                cloud.log_weights = torch.zeros(N, device=cloud.log_weights.device)
                diag.n_resamples += 1

            diag.record(beta_curr, cloud, self.energy_fn)

            if pbar is not None:
                pbar.set_postfix(
                    beta=f"{beta_curr:.2f}",
                    ess=f"{ess_ratio:.2f}",
                    E_min=f"{diag.best_energies[-1]:.3f}",
                )

        return cloud, diag
