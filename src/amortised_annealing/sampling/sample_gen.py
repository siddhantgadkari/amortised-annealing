from __future__ import annotations
from ..mcmc.langevin import ULA, MALA
from ..energies.base import Energy, BoltzmannDistribution

import torch
from typing import Tuple 

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

    def sample(self, n_samples: int, n_steps: int, burn_in: int = 0, save_every: int = 1) -> Tuple[torch.Tensor, float]:
        """Generate samples from pi_betaM using MCMC.

        Returns:
            samples: [n_samples, num_snapshots, dim] where num_snapshots = n_steps // save_every
            acc_rate: mean acceptance rate over the sampling phase
        """
        x = torch.randn(n_samples, self.energy.dim, device=self.device)

        for _ in range(burn_in):
            x, _ = self.method.step(x, self.betaM)

        snapshots = []
        total_accept = 0.0
        for i in range(n_steps):
            x, acc = self.method.step(x, self.betaM)
            total_accept += acc
            if (i + 1) % save_every == 0:
                snapshots.append(x.clone())

        samples = torch.stack(snapshots, dim=1)  # [n_samples, num_snapshots, dim]
        return samples, total_accept / n_steps