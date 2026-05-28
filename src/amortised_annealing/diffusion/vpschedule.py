from __future__ import annotations

import torch


class VPSchedule():
    """Variance-Preserving (VP) diffusion schedule.

    Forward SDE:  dx = -0.5 * beta(t) * x dt + sqrt(beta(t)) dW
    Linear beta schedule:  beta(t) = beta_min + (beta_max - beta_min) * t

    Marginal:
        alpha(t) = exp(-0.5 * ∫{0->t} beta(s) ds)
        sigma(t) = sqrt(1 - alpha(t)^2)

    At t=0: alpha=1, sigma=0 (clean).
    At t=1: alpha -> 0, sigma -> 1 (noise-dominated).
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def _int_beta(self, t: torch.Tensor) -> torch.Tensor:
        """∫{0->t} beta(s) ds = beta_min*t + 0.5*(beta_max-beta_min)*t^2"""
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Instantaneous noise level beta(t)."""
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * self._int_beta(t))

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.clamp(1.0 - self.alpha(t)**2, min=1e-8))

    def reverse_drift(self, t: torch.Tensor, x: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        """Reverse SDE drift for Euler-Maruyama: f_rev(t, x) = 0.5*beta(t)*x + beta(t)*s(t,x).

        When integrating from t=1 to t=0, we compute:
            x_{t-dt} = x_t + dt * reverse_drift(t, x_t, score) + sqrt(beta(t)*dt) * eps
        """
        beta_t = self.beta(t).view(-1, *([1] * (x.ndim - 1)))
        return 0.5 * beta_t * x + beta_t * score

    def reverse_diffusion(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Diffusion coefficient for reverse SDE: sqrt(beta(t))."""
        beta_t = self.beta(t).view(-1, *([1] * (x.ndim - 1)))
        return torch.sqrt(beta_t)

    def marginal_sample(self, x0: torch.Tensor, t: torch.Tensor):
        """Sample x_t ~ q(x_t | x_0) = N(alpha(t)*x0, sigma(t)^2 I).

        Returns (x_t, eps) where eps is the noise that was added.
        """
        a = self.alpha(t).view(-1, 1)
        s = self.sigma(t).view(-1, 1)
        eps = torch.randn_like(x0)
        return a * x0 + s * eps, eps

    def eps_to_score(self, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Convert eps-prediction to score: score = -eps / sigma(t)."""
        return -eps / self.sigma(t).view(-1, 1)
