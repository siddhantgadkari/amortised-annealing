#!/usr/bin/env python
from __future__ import annotations
import argparse, json, shutil, time
from pathlib import Path

import torch
import yaml

from amortised_annealing.energies import DoubleWell, ManyWell, Ackley, Rastrigin
from amortised_annealing.sampling.sample_gen import SampleGenerator

ROOT        = Path(__file__).parent.parent
CONFIGS_DIR = ROOT / "configs" / "sampling"
DATA_DIR    = ROOT / "data" / "samples"

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
    final = samples[:, -1, :]  # [n_particles, dim] — use final snapshot
    with torch.no_grad():
        energies = energy_fn(final).float().cpu()
    qs = torch.quantile(energies, torch.tensor([0.01, 0.05, 0.25, 0.75, 0.95]))
    return {
        "mean_energy":    round(energies.mean().item(), 4),
        "median_energy":  round(energies.median().item(), 4),
        "min_energy":     round(energies.min().item(), 4),
        "std_energy":     round(energies.std().item(), 4),
        "energy_quantiles": {
            "q01": round(qs[0].item(), 4),
            "q05": round(qs[1].item(), 4),
            "q25": round(qs[2].item(), 4),
            "q75": round(qs[3].item(), 4),
            "q95": round(qs[4].item(), 4),
        },
    }


def _health_stats(samples: torch.Tensor) -> dict:
    final = samples[:, -1, :]
    return {
        "nan_count":      int(torch.isnan(final).any(-1).sum().item()),
        "diverged_count": int(torch.isinf(final).any(-1).sum().item()),
    }


def _n_evals(n_particles: int, n_steps: int, burn_in: int, method: str) -> dict:
    # Count per-sample evaluations across all particles and steps.
    total = n_particles * (n_steps + burn_in)
    if method == "ULA":
        # One grad eval per step (energy computed as part of the backward pass).
        return {"n_energy_evals": total, "n_grad_evals": total}
    else:  # MALA
        # Two grad evals (current + proposal) + two explicit energy evals for MH.
        return {"n_energy_evals": 4 * total, "n_grad_evals": 2 * total}


def run_config(cfg_path: Path) -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    job     = cfg["job"]
    target  = cfg["target"]
    sampler = cfg["sampler"]
    output  = cfg["output"]

    run_name = output["run_name"]
    out_dir  = DATA_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(job["seed"])
    torch.set_default_dtype(getattr(torch, job["dtype"]))

    device = _resolve_device(job["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        gpu_name = torch.cuda.get_device_name(device)
    else:
        gpu_name = None

    energy = ENERGY_MAP[target["energy"]](dim=target["dim"])

    gen = SampleGenerator(
        energy      = energy,
        betaM       = target["beta_m"],
        mcmc_method = sampler["method"],
        step_size   = sampler["step_size"],
        device      = device,
    )

    print(f"[{run_name}]")
    t0 = time.perf_counter()
    samples, step_rates = gen.sample(
        n_samples  = sampler["n_particles"],
        n_steps    = sampler["n_steps"],
        burn_in    = sampler["burn_in"],
        save_every = sampler["save_every"],
        progress   = True,
    )
    wall = time.perf_counter() - t0

    peak_gpu_mb = (
        round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 1)
        if device.type == "cuda" else None
    )

    acc = torch.tensor(step_rates)
    summary = {
        "status":             "completed",
        "seed":               job["seed"],
        "energy":             target["energy"],
        "dim":                target["dim"],
        "beta_m":             target["beta_m"],
        "sampler":            sampler["method"],
        "n_particles":        sampler["n_particles"],
        "n_steps":            sampler["n_steps"],
        "burn_in":            sampler["burn_in"],
        "save_every":         sampler["save_every"],
        "step_size":          sampler["step_size"],
        "wall_clock_seconds": round(wall, 2),
        "device":             str(device),
        "gpu_name":           gpu_name,
        "peak_gpu_memory_mb": peak_gpu_mb,
        **_n_evals(sampler["n_particles"], sampler["n_steps"], sampler["burn_in"], sampler["method"]),
        **_energy_stats(samples, energy),
        **_health_stats(samples),
        "acceptance_rate": {
            "mean": round(acc.mean().item(), 4),
            "min":  round(acc.min().item(), 4),
            "max":  round(acc.max().item(), 4),
        },
    }

    torch.save(samples.cpu(), out_dir / "particles.pt")
    shutil.copy(cfg_path, out_dir / "config.yaml")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  done in {wall:.1f}s  acc={summary['acceptance_rate']['mean']:.3f}  -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MCMC sampling from a config file.")
    parser.add_argument(
        "configs", nargs="*", type=Path,
        help="Config YAML path(s). Defaults to all YAMLs in configs/sampling/.",
    )
    args = parser.parse_args()
    paths = args.configs or sorted(CONFIGS_DIR.glob("*.yaml"))
    for p in paths:
        run_config(p)


if __name__ == "__main__":
    main()
