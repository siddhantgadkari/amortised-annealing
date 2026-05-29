#!/usr/bin/env python
"""Evaluate a trained score model by generating samples and printing energy stats.

Prints a side-by-side comparison against the MCMC samples the model was trained on.

Usage:
    python scripts/eval_model.py <model_run_name>
    python scripts/eval_model.py <model_run_name> --n-samples 4096 --n-steps 500
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import torch
import yaml

from amortised_annealing.diffusion import VPSchedule, MLPScore, ReverseSDE
from amortised_annealing.energies import DoubleWell, ManyWell, Ackley, Rastrigin
from amortised_annealing.sampling import DiffusionModelSampler

ROOT      = Path(__file__).parent.parent
MODEL_DIR = ROOT / "data" / "models"

ENERGY_MAP = {
    "double_well": DoubleWell,
    "many_well":   ManyWell,
    "ackley":      Ackley,
    "rastrigin":   Rastrigin,
}


def _resolve_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def _energy_stats(samples: torch.Tensor, energy_fn) -> dict:
    with torch.no_grad():
        energies = energy_fn(samples).float().cpu()
    qs = torch.quantile(energies, torch.tensor([0.01, 0.05, 0.25, 0.75, 0.95]))
    return {
        "mean":   round(energies.mean().item(), 4),
        "median": round(energies.median().item(), 4),
        "min":    round(energies.min().item(), 4),
        "std":    round(energies.std().item(), 4),
        "q01":    round(qs[0].item(), 4),
        "q05":    round(qs[1].item(), 4),
        "q25":    round(qs[2].item(), 4),
        "q75":    round(qs[3].item(), 4),
        "q95":    round(qs[4].item(), 4),
    }


def _print_comparison(label_a: str, stats_a: dict, label_b: str, stats_b: dict) -> None:
    keys = ["mean", "median", "min", "std", "q01", "q05", "q25", "q75", "q95"]
    col = max(len(label_a), len(label_b), 20)
    print(f"\n{'':20s}  {label_a:>{col}}  {label_b:>{col}}")
    print("-" * (22 + 2 * col))
    for k in keys:
        print(f"  {k:<18s}  {stats_a[k]:>{col}.4f}  {stats_b[k]:>{col}.4f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", help="Model run name (subfolder of data/models/)")
    parser.add_argument("--n-samples", type=int,   default=4096)
    parser.add_argument("--n-steps",   type=int,   default=500)
    parser.add_argument("--t-start",   type=float, default=1.0)
    parser.add_argument("--t-end",     type=float, default=1e-3)
    parser.add_argument("--progress",  action="store_true")
    args = parser.parse_args()

    run_dir = MODEL_DIR / args.run_name
    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = _resolve_device(cfg["job"]["device"])
    torch.set_default_dtype(getattr(torch, cfg["job"]["dtype"]))

    # --- build energy ---
    sample_cfg = cfg["samples"]
    sample_summary_path = ROOT / "data" / "samples" / sample_cfg["run_name"] / "summary.json"
    with open(sample_summary_path) as f:
        sample_summary = json.load(f)

    energy_name = sample_summary["energy"]
    dim         = sample_summary["dim"]
    energy      = ENERGY_MAP[energy_name](dim=dim)

    # --- load model ---
    model_cfg = cfg["model"]
    model = MLPScore(
        dim            = dim,
        hidden_dims    = tuple(model_cfg["hidden_dims"]),
        time_embed_dim = model_cfg.get("time_embed_dim", 64),
        activation     = model_cfg.get("activation", "silu"),
        predict_score  = model_cfg.get("predict_score", False),
    )
    state = torch.load(run_dir / "ema_model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    # --- build schedule + reverse SDE ---
    sched_cfg = cfg["schedule"]
    schedule  = VPSchedule(
        beta_min=sched_cfg.get("beta_min", 0.1),
        beta_max=sched_cfg.get("beta_max", 20.0),
    )
    rsde = ReverseSDE(model, schedule)

    # --- generate samples ---
    print(f"Generating {args.n_samples} samples with {args.n_steps} EM steps...")
    diff_sampler = DiffusionModelSampler(rsde, n_steps=args.n_steps, t_start=args.t_start, t_end=args.t_end)
    x = diff_sampler.sample(args.n_samples, device, show_progress=args.progress)

    # --- compute stats ---
    gen_stats  = _energy_stats(x.cpu(), energy)
    mcmc_stats = {
        "mean":   sample_summary["mean_energy"],
        "median": sample_summary["median_energy"],
        "min":    sample_summary["min_energy"],
        "std":    sample_summary["std_energy"],
        "q01":    sample_summary["energy_quantiles"].get("q01", float("nan")),
        "q05":    sample_summary["energy_quantiles"].get("q05", float("nan")),
        "q25":    sample_summary["energy_quantiles"].get("q25", float("nan")),
        "q75":    sample_summary["energy_quantiles"].get("q75", float("nan")),
        "q95":    sample_summary["energy_quantiles"].get("q95", float("nan")),
    }

    # --- print ---
    print(f"\nEnergy: {energy_name}  dim={dim}  beta_m={sample_summary['beta_m']}")
    print(f"MCMC samples:  {sample_cfg['run_name']}")
    print(f"Score model:   {args.run_name}")
    _print_comparison("diffusion", gen_stats, "mcmc (train)", mcmc_stats)


if __name__ == "__main__":
    main()
