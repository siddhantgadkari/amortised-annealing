from __future__ import annotations

import math

import torch

from .base import Energy


class Ackley(Energy):
    """Ackley function: multimodal benchmark with nearly flat outer region.

    E(x) = -a*exp(-b*sqrt(1/d * sum x_i^2)) - exp(1/d * sum cos(c*x_i)) + a + e

    Global minimum at x=0 with E=0.
    Default domain: x_i in [-32.768, 32.768].
    """

    def __init__(self, dim: int = 2, a: float = 20.0, b: float = 0.2, c: float = 2 * math.pi):
        self._dim = dim
        self.a = a
        self.b = b
        self.c = c

    @property
    def dim(self) -> int:
        return self._dim

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        x, single = self._check_x(x)
        e = self._energy(x)
        return e.squeeze(0) if single else e

    def _energy(self, x: torch.Tensor) -> torch.Tensor:
        d = self._dim
        term1 = -self.a * torch.exp(-self.b * torch.sqrt((x**2).mean(-1)))
        term2 = -torch.exp(torch.cos(self.c * x).mean(-1))
        return term1 + term2 + self.a + math.e
    
    @property
    def global_minima(self) -> torch.Tensor | None:
        return torch.zeros(1, self.dim) 

    @property
    def global_minimum_energy(self) -> float | None:
        return 0.0 

