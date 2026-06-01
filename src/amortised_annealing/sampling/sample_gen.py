from __future__ import annotations
from ..mcmc.langevin import ULA, MALA
from ..energies.base import Energy, BoltzmannDistribution

import torch
from tqdm import tqdm
from typing import Tuple, Dict, List

class SampleGenerator:
    def __init__(self, energy: Energy,
                 betaM: float,
                 mcmc_method: str = "ULA",
                 step_size: float = 1e-3,
                 device: torch.device = torch.device("cpu")
    ):
        if mcmc_method == "ULA":
            self.method = ULA(energy, step_size)
        elif mcmc_method == "MALA":
            self.method = MALA(energy, step_size)
        else:
            raise ValueError(f"Unknown MCMC method: {mcmc_method}")

        self.energy = energy
        self.betaM = betaM
        self.boltzmann_dist = BoltzmannDistribution(energy, betaM)
        self.device = device

    def sample(
        self,
        n_samples: int,
        n_steps: int,
        burn_in: int = 0,
        trace_every: int = 100,
        progress: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, List], List[float]]:
        """Generate samples from pi_betaM using MCMC.

        Returns:
            final_x:    [n_samples, dim] — final chain state
            traces:     dict of scalar lists recorded every trace_every steps:
                          steps, mean_energy, median_energy, min_energy,
                          q05_energy, q95_energy, mean_x_norm, mean_displacement
            step_rates: per-step mean acceptance rates (length n_steps); always 1.0 for ULA
        """
        x = torch.randn(n_samples, self.energy.dim, device=self.device)

        burn_iter = tqdm(range(burn_in), desc="burn-in", leave=False) if progress else range(burn_in)
        for _ in burn_iter:
            x, _ = self.method.step(x, self.betaM)

        traces: Dict[str, List] = {
            "steps":          [],
            "mean_energy":    [],
            "median_energy":  [],
            "min_energy":     [],
            "q05_energy":     [],
            "q95_energy":     [],
            "mean_x_norm":    [],
            "mean_displacement": [],
        }
        step_rates: List[float] = []

        x_prev_save = x.clone()

        sample_iter = tqdm(range(n_steps), desc="sampling") if progress else range(n_steps)
        for i in sample_iter:
            x, acc = self.method.step(x, self.betaM)
            step_rates.append(acc)

            if (i + 1) % trace_every == 0:
                with torch.no_grad():
                    e = self.energy.energy(x).float()
                qs = torch.quantile(e, torch.tensor([0.05, 0.95], device=e.device))
                disp = (x - x_prev_save).norm(dim=1).mean().item()
                traces["steps"].append(i + 1)
                traces["mean_energy"].append(round(e.mean().item(), 4))
                traces["median_energy"].append(round(e.median().item(), 4))
                traces["min_energy"].append(round(e.min().item(), 4))
                traces["q05_energy"].append(round(qs[0].item(), 4))
                traces["q95_energy"].append(round(qs[1].item(), 4))
                traces["mean_x_norm"].append(round(x.norm(dim=1).mean().item(), 4))
                traces["mean_displacement"].append(round(disp, 4))
                x_prev_save = x.clone()

        return x, traces, step_rates
