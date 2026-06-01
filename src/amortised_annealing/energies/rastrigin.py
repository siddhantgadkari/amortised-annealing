from __future__ import annotations

import math

import torch

from .base import Energy


class Rastrigin(Energy):
    """Rastrigin function: highly multimodal benchmark.

    E(x) = A*d + sum_i [x_i^2 - A*cos(2*pi*x_i)]

    Global minimum at x=0 with E=0. Many local minima on integer grid.
    """

    def __init__(self, dim: int = 2, A: float = 10.0):
        self._dim = dim
        self.A = A

    @property
    def dim(self) -> int:
        return self._dim

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        x, single = self._check_x(x)
        e = self._energy(x)
        return e.squeeze(0) if single else e

    def _energy(self, x: torch.Tensor) -> torch.Tensor:
        A = self.A
        d = self._dim
        return A * d + (x**2 - A * torch.cos(2 * math.pi * x)).sum(-1)
    
    @property
    def global_minima(self) -> torch.Tensor | None:
        return torch.zeros(1, self.dim) 

    @property
    def global_minimum_energy(self) -> float | None:
        return 0.0 
