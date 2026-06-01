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


def _energy_stats(x: torch.Tensor, energy_fn) -> dict:
    """Compute energy stats for final samples [N, dim]."""
    with torch.no_grad():
        energies = energy_fn(x).float().cpu()
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


def _coord_stats(x: torch.Tensor) -> dict:
    """Summary stats of particle positions [N, dim]."""
    norms = x.norm(dim=1)
    return {
        "mean_x_norm": round(norms.mean().item(), 4),
        "max_x_norm":  round(norms.max().item(), 4),
        "coord_mean":  round(x.mean().item(), 4),
        "coord_std":   round(x.std().item(), 4),
    }


def _health_stats(x: torch.Tensor) -> dict:
    return {
        "nan_count":      int(torch.isnan(x).any(-1).sum().item()),
        "diverged_count": int(torch.isinf(x).any(-1).sum().item()),
    }


def _threshold_props(x: torch.Tensor, energy_fn, global_min_energy: float | None) -> dict:
    if global_min_energy is None:
        return {}
    with torch.no_grad():
        e = energy_fn(x).float().cpu() - global_min_energy
    n = x.shape[0]
    return {
        "prop_excess_E_lt_0.01": round((e < 0.01).float().mean().item(), 4),
        "prop_excess_E_lt_0.1":  round((e < 0.1).float().mean().item(), 4),
        "prop_excess_E_lt_1.0":  round((e < 1.0).float().mean().item(), 4),
    }


def _stationarity(traces: dict) -> dict:
    energies = traces.get("mean_energy", [])
    if len(energies) < 5:
        return {"energy_tail_slope": None}
    tail = energies[int(len(energies) * 0.8):]
    n = len(tail)
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(tail) / n
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(xs, tail))
    den = sum((xi - x_mean) ** 2 for xi in xs)
    slope = num / den if den > 0 else 0.0
    return {"energy_tail_slope": round(slope, 6)}


def _n_evals(n_particles: int, n_steps: int, burn_in: int, method: str) -> dict:
    total = n_particles * (n_steps + burn_in)
    if method == "ULA":
        return {"n_energy_evals": total, "n_grad_evals": total}
    else:  # MALA
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

    trace_every = sampler.get("trace_every", sampler.get("save_every", 100))

    gen = SampleGenerator(
        energy      = energy,
        betaM       = target["beta_m"],
        mcmc_method = sampler["method"],
        step_size   = sampler["step_size"],
        device      = device,
    )

    print(f"[{run_name}]")
    t0 = time.perf_counter()
    final_x, traces, step_rates = gen.sample(
        n_samples   = sampler["n_particles"],
        n_steps     = sampler["n_steps"],
        burn_in     = sampler["burn_in"],
        trace_every = trace_every,
        progress    = True,
    )
    wall = time.perf_counter() - t0

    final_x_cpu = final_x.cpu()
    peak_gpu_mb = (
        round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 1)
        if device.type == "cuda" else None
    )

    acc = torch.tensor(step_rates)
    global_min = getattr(energy, "global_minimum_energy", None)

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
        "trace_every":        trace_every,
        "step_size":          sampler["step_size"],
        "wall_clock_seconds": round(wall, 2),
        "device":             str(device),
        "gpu_name":           gpu_name,
        "peak_gpu_memory_mb": peak_gpu_mb,
        **_n_evals(sampler["n_particles"], sampler["n_steps"], sampler["burn_in"], sampler["method"]),
        **_energy_stats(final_x_cpu, energy),
        **_coord_stats(final_x_cpu),
        **_health_stats(final_x_cpu),
        **_threshold_props(final_x_cpu, energy, global_min),
        **_stationarity(traces),
        "acceptance_rate": {
            "mean": round(acc.mean().item(), 4),
            "min":  round(acc.min().item(), 4),
            "max":  round(acc.max().item(), 4),
        },
        "traces": traces,
    }

    torch.save(final_x_cpu, out_dir / "particles.pt")
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
