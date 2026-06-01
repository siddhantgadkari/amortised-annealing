from __future__ import annotations

from typing import Callable

import torch

from ..mcmc.langevin import ULA


class ULAProposal:
    """Classical annealed SMC proposal: AIS weight update + ULA mutation kernel.

    At each beta step the AIS weight -(beta_curr - beta_prev) * E(x) is applied
    at the current (old) positions to reweight pi_{beta_prev} -> pi_{beta_curr},
    then particles are mutated by running `n_steps` of ULA targeting pi_{beta_curr}
    to improve mixing.
    """

    def __init__(
        self,
        energy: Callable[[torch.Tensor], torch.Tensor],
        step_size: float = 1e-3,
        n_steps: int = 10,
    ):
        self._ula = ULA(energy, step_size=step_size)
        self._energy = energy
        self.n_steps = n_steps

    def mutation_kernel(self, x: torch.Tensor, beta: float) -> torch.Tensor:
        x_new, _ = self._ula.run(x, beta, self.n_steps)
        return x_new

    def weight_update(self, x: torch.Tensor, beta_prev: float, beta_curr: float) -> torch.Tensor:
        with torch.no_grad():
            e = self._energy(x)
        return -(beta_curr - beta_prev) * e


class DiffusionAnnealingProposal:
    """Diffusion-informed annealed SMC proposal.

    Mutation: add noise at t_start (noise-reinject trick) then run
    `n_diffusion_steps` of Euler-Maruyama of the reverse SDE.
    The score can be scaled by (beta_curr / beta_train) as a heuristic push
    toward the colder target; this is a proposal heuristic, not an exact
    correction — the AIS weight corrects the mismatch.

    This is a heuristic proposal; the standard AIS weight corrects only the beta change,
    not the proposal bias. FKC/path weights are needed for a principled correction.

    Weight: standard AIS incremental weight -(Δβ) * E(x).
    """

    def __init__(
        self,
        reverse_sde,
        energy: Callable[[torch.Tensor], torch.Tensor],
        beta_train: float,
        n_diffusion_steps: int = 10,
        t_start: float = 0.3,
        t_end: float = 1e-3,
        score_scaling: bool = True,
    ):
        self._reverse_sde = reverse_sde
        self._energy = energy
        self.beta_train = beta_train
        self.n_diffusion_steps = n_diffusion_steps
        self.t_start = t_start
        self.t_end = t_end
        self.score_scaling = score_scaling

    def mutation_kernel(self, x: torch.Tensor, beta_curr: float) -> torch.Tensor:
        scale = (beta_curr / self.beta_train) if self.score_scaling else 1.0
        schedule = self._reverse_sde.schedule
        N = x.shape[0]
        device = x.device

        t_inj = torch.full((N,), self.t_start, device=device)
        x, _ = schedule.marginal_sample(x, t_inj)

        times = torch.linspace(self.t_start, self.t_end, self.n_diffusion_steps + 1, device=device)
        dt = (self.t_start - self.t_end) / self.n_diffusion_steps

        with torch.no_grad():
            for i in range(self.n_diffusion_steps):
                t = times[i].expand(N)
                x = self._reverse_sde.step(x, t, dt, temperature_scale=scale)

        return x

    def weight_update(self, x: torch.Tensor, beta_prev: float, beta_curr: float) -> torch.Tensor:
        with torch.no_grad():
            e = self._energy(x)
        return -(beta_curr - beta_prev) * e
