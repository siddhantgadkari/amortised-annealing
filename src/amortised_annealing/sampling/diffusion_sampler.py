from __future__ import annotations

import torch

from ..diffusion.reverse_sde import euler_maruyama_sample


class DiffusionModelSampler:
    """Generates samples from a trained score model via reverse SDE."""

    def __init__(
        self,
        reverse_sde,
        n_steps: int = 500,
        t_start: float = 1.0,
        t_end: float = 1e-3,
    ):
        self._rsde = reverse_sde
        self.n_steps = n_steps
        self.t_start = t_start
        self.t_end = t_end

    def sample(
        self,
        n_samples: int,
        device: torch.device,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """Return [n_samples, d] samples via Euler-Maruyama."""
        return euler_maruyama_sample(
            self._rsde,
            n_samples=n_samples,
            n_steps=self.n_steps,
            device=device,
            t_start=self.t_start,
            t_end=self.t_end,
            show_progress=show_progress,
        )
