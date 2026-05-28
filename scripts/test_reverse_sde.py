#!/usr/bin/env python
"""Sanity-check the reverse SDE on standard Gaussian data.

For VP diffusion, if x0 ~ N(0, I) then q_t(x) = N(0, I) for all t
(since alpha(t)^2 + sigma(t)^2 = 1 by construction).  The analytic
score is therefore s(t, x) = -x, which we can plug in directly without
any trained network.

A correct reverse SDE should map N(0, I) at t=1 back to N(0, I) at t~0.
If this test fails, the bug is in the SDE integration, not the score model.
"""
from __future__ import annotations
import argparse

import torch

from amortised_annealing.diffusion import VPSchedule, ReverseSDE, euler_maruyama_sample


class AnalyticGaussianScore:
    """Oracle score for x0 ~ N(0, I) under VP diffusion: s(t, x) = -x."""

    def __init__(self, dim: int):
        self.dim = dim

    def score(self, t: torch.Tensor, x: torch.Tensor, schedule) -> torch.Tensor:
        return -x


def _print_stats(label: str, x: torch.Tensor) -> None:
    mean = x.mean(0)          # [dim]
    std  = x.std(0)           # [dim]
    print(f"\n{label}")
    print(f"  mean: min={mean.min():.4f}  max={mean.max():.4f}  (expect ~0)")
    print(f"  std:  min={std.min():.4f}  max={std.max():.4f}  (expect ~1)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim",       type=int,   default=10)
    parser.add_argument("--n-samples", type=int,   default=4096)
    parser.add_argument("--n-steps",   type=int,   default=500)
    parser.add_argument("--beta-min",  type=float, default=0.1)
    parser.add_argument("--beta-max",  type=float, default=20.0)
    parser.add_argument("--progress",  action="store_true")
    args = parser.parse_args()

    device   = torch.device("cpu")
    schedule = VPSchedule(beta_min=args.beta_min, beta_max=args.beta_max)
    model    = AnalyticGaussianScore(dim=args.dim)
    rsde     = ReverseSDE(model, schedule)

    # Reference: pure noise at t=1
    x_noise = torch.randn(args.n_samples, args.dim)
    _print_stats("Input (t=1, should be N(0,I))", x_noise)

    # Run reverse SDE with analytic score
    x_out = euler_maruyama_sample(
        rsde,
        n_samples     = args.n_samples,
        n_steps       = args.n_steps,
        device        = device,
        show_progress = args.progress,
    )
    _print_stats(f"Output (t~0, {args.n_steps} steps, should be N(0,I))", x_out)

    # Quick check
    mean_ok = x_out.mean(0).abs().max().item() < 0.1
    std_ok  = (x_out.std(0) - 1.0).abs().max().item() < 0.1
    status  = "PASS" if (mean_ok and std_ok) else "FAIL"
    print(f"\n[{status}]  mean_ok={mean_ok}  std_ok={std_ok}\n")


if __name__ == "__main__":
    main()
