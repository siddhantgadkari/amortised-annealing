from __future__ import annotations
from typing import Callable, Tuple 
import torch 
import math

def _grad_energy(x: torch.Tensor, 
                 energy: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    with torch.enable_grad():
        x = x.detach().requires_grad_(True) 
        e = energy(x)
        grads = torch.autograd.grad(e.sum(), x)[0]

    return grads.detach()

class ULA: 
    """Unadjusted Langevin Algorithm targeting pi_beta ∝ exp(-beta * E(x)).

    Euler-Maruyama discretisation update with step size h:
        x_{k+1} = x_k - beta * h * grad E(x_k) + sqrt(2h) * eps
    """

    def __init__(
            self, 
            energy: Callable[[torch.Tensor], torch.Tensor],
            step_size: float=1e-3,
    ): 
        self.energy = energy 
        self.step_size = step_size

    @torch.no_grad()
    def step(self, x: torch.Tensor, beta: float) -> torch.Tensor: 
        h = self.step_size
        grad_E = _grad_energy(x, self.energy)
        noise = torch.randn_like(x) # need same shape and device as x
        x_next = x - beta * h * grad_E + math.sqrt(2 * h) * noise
        return x_next
    
    @torch.no_grad()
    def run(
        self, 
        x0: torch.Tensor,
        beta: float, 
        n_steps: int,
    ) -> torch.Tensor: 
        """Run ULA for n_steps starting from x0"""
        for _ in range(n_steps):
            x0 = self.step(x0, beta)
        return x0

class MALA: 
    """Metropolis-Adjusted Langevin Algorithm targeting pi_beta ∝ exp(-beta * E(x)).

    Proposes ULA step and accepts/rejects with MH criterion.
    """

    def __init__(
            self, 
            energy: Callable[[torch.Tensor], torch.Tensor],
            step_size: float=1e-3,
    ): 
        self.energy = energy 
        self.step_size = step_size

    @torch.no_grad()
    def step(self, x: torch.Tensor, beta: float) -> Tuple[torch.Tensor, float]: 
        h = self.step_size
        
        grad_curr = _grad_energy(x, self.energy)
        mean_fwd = x - beta * h * grad_curr
        x_prop = mean_fwd + math.sqrt(2 * h) * torch.randn_like(x)

        # Log proposal densities (Gaussian)
        grad_prop = _grad_energy(x_prop, self.energy)
        mean_bwd = x_prop - beta * h * grad_prop
        log_q_fwd = -((x_prop - mean_fwd)**2).sum(-1) / (4 * h)
        log_q_bwd = -((x - mean_bwd)**2).sum(-1) / (4 * h)

        e_curr = self.energy(x)
        e_prop = self.energy(x_prop)
        log_accept_ratio = -beta * (e_prop - e_curr) + log_q_bwd - log_q_fwd
        accept = torch.log(torch.rand_like(log_accept_ratio)) < log_accept_ratio
        x_next = torch.where(accept.unsqueeze(-1), x_prop, x)

        return x_next, accept.float().mean().item() # return acceptance rate for this step
    
    def run(
        self, 
        x0: torch.Tensor,
        beta: float, 
        n_steps: int,
    ) -> Tuple[torch.Tensor, float]: 
        """Run MALA for n_steps starting from x0"""
        total_accept = 0.0
        for _ in range(n_steps):
            x0, accept = self.step(x0, beta)
            total_accept += accept

        accept_rate = total_accept / n_steps
        return x0, accept_rate