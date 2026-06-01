from __future__ import annotations

import torch

from .base import Energy


class DoubleWell(Energy):
    """Double-well potential: E(x) = a*(x[0]^4 - b*x[0]^2) + c*sum(x[1:]^2).

    Minima near x[0] ≈ ±sqrt(b/2) for the first coordinate;
    remaining coordinates are harmonic.
    """

    def __init__(self, dim: int = 2, a: float = 1.0, b: float = 4.0, c: float = 1.0):
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
        x0 = x[:, 0]
        well = self.a * (x0**4 - self.b * x0**2)
        if self._dim > 1:
            harmonic = self.c * (x[:, 1:] ** 2).sum(-1)
        else:
            harmonic = torch.zeros_like(well)
        return well + harmonic

    @property
    def global_minima(self) -> torch.Tensor | None:
        x = torch.zeros(2, self.dim)
        x[:, 0] = torch.tensor([
            -(self.b / 2) ** 0.5,
             (self.b / 2) ** 0.5,
        ])
        return x

    @property
    def global_minimum_energy(self) -> float | None:
        return -self.a * self.b**2 / 4


class ManyWell(Energy):
    """Many-well potential: product of double-wells in pairs of dimensions.

    E(x) = sum_{k} a*(x[2k]^4 - b*x[2k]^2) + c*x[2k+1]^2

    For odd-dimensional inputs the last coordinate gets a harmonic term.
    """

    def __init__(self, dim: int = 4, a: float = 1.0, b: float = 4.0, c: float = 1.0):
        if dim < 2:
            raise ValueError("ManyWell requires dim >= 2")
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
        total = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        k = 0
        while 2 * k + 1 < self._dim:
            x0 = x[:, 2 * k]
            x1 = x[:, 2 * k + 1]
            total += self.a * (x0**4 - self.b * x0**2) + self.c * x1**2
            k += 1
        if self._dim % 2 == 1:
            total += self.c * x[:, -1] ** 2
        return total

    @property
    def global_minima(self) -> torch.Tensor | None:
        n_wells = self.dim // 2
        val = (self.b / 2) ** 0.5

        signs = torch.cartesian_prod(*[
            torch.tensor([-1.0, 1.0]) for _ in range(n_wells)
        ])
        if signs.dim() == 1:
            signs = signs.unsqueeze(1)

        minima = torch.zeros(signs.shape[0], self.dim)

        for k in range(n_wells):
            minima[:, 2 * k] = signs[:, k] * val

        return minima

    @property
    def global_minimum_energy(self) -> float | None:
        n_wells = self.dim // 2
        return -n_wells * self.a * self.b**2 / 4