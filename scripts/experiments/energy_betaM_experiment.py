#!/usr/bin/env python
"""Sweep experiment: energy × dim × βM × seed.

Full pipeline per (dim, βM, seed):
  MCMC sampling → score model training → model sample generation
  → ULA SMC + Diffusion SMC + FKC annealing → aggregate across seeds → plots.

Edit the USER CONFIGURATION block, then run:
    uv run python scripts/experiments/energy_betaM_experiment.py

Re-plot only:
    uv run python scripts/experiments/energy_betaM_experiment.py --plot-only
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

_SCRIPTS = Path(__file__).parent.parent
sys.path.insert(0, str(_SCRIPTS))
from run_sampling import run_config as _run_sample_config  # noqa: E402
from run_training import run_config as _run_train_config   # noqa: E402

from amortised_annealing.diffusion import VPSchedule, MLPScore, ReverseSDE
from amortised_annealing.energies import DoubleWell, ManyWell, Ackley, Rastrigin
from amortised_annealing.sampling import DiffusionModelSampler
from amortised_annealing.smc import (
    FKCAnnealedSampler,
    ParticleCloud,
    SMCDiagnostics,
    SMCSampler,
    ULAProposal,
    DiffusionAnnealingProposal,
)

ROOT                 = Path(__file__).parent.parent.parent
SAMPLE_DIR           = ROOT / "data" / "samples"
MODEL_DIR            = ROOT / "data" / "models"
MODEL_SAMPLE_DIR     = ROOT / "data" / "model_samples"
EXPERIMENT_DIR       = ROOT / "data" / "experiments" / "energy_betaM_experiments"
SAMPLING_CONFIGS_DIR = ROOT / "configs" / "sampling"
TRAINING_CONFIGS_DIR = ROOT / "configs" / "models"

ENERGY_MAP = {
    "double_well": DoubleWell,
    "many_well":   ManyWell,
    "ackley":      Ackley,
    "rastrigin":   Rastrigin,
}

# ===========================================================================
# USER CONFIGURATION *
# ===========================================================================
ENERGY  = "rastrigin"
DIMS    = [2, 5, 10, 20]
BETA_MS = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
BETA_H  = 10.0
SEEDS   = [0, 1, 2]

# MCMC sampling
MCMC_N_PARTICLES = 8192
MCMC_N_STEPS     = 10_000
MCMC_BURN_IN     = 2_000
MCMC_SAVE_EVERY  = 100
MCMC_STEP_SIZE   = 1e-3

# Score model architecture
MODEL_HIDDEN_DIMS    = [256, 256, 256]
MODEL_TIME_EMBED_DIM = 64
LOSS_TYPE            = "eps"

# Score model training
TRAIN_N_STEPS = 20_000
BATCH_SIZE    = 512
LR            = 2e-4
T_EPS         = 1e-4
GRAD_CLIP     = 1.0
EMA_DECAY     = 0.999

# Model sample generation (N = MCMC_N_PARTICLES)
MODEL_SAMPLE_N_STEPS = 500
MODEL_SAMPLE_T_START = 1.0
MODEL_SAMPLE_T_END   = 1e-3

# Annealing — shared
N_PARTICLES   = 2048
N_SMC_STEPS   = 50
ESS_THRESHOLD = 0.5

# Annealing — ULA SMC
N_ULA_STEPS   = 10
ULA_STEP_SIZE = 1e-3

# Annealing — diffusion SMC
N_DIFFUSION_STEPS  = 10
DIFF_T_START       = 0.05
DIFF_T_END         = 1e-3
DIFF_SCORE_SCALING = True

# Annealing — FKC
FKC_N_STEPS            = 500
FKC_T_START            = 1.0
FKC_T_END              = 1e-3
FKC_INCLUDE_DIVERGENCE = False

# Reuse flags — set False to force re-run
REUSE_SAMPLES       = True
REUSE_MODELS        = True
REUSE_MODEL_SAMPLES = True
REUSE_ANNEALING     = True  # skip (dim, betaM) if already in results_seed{s}.json

DESC = ""

# Contour plots
CONTOUR_KDE = False  # True → scipy gaussian_kde contour lines; False → scatter
# ===========================================================================


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------

def _beta_str(beta: float) -> str:
    return f"{beta:g}".replace(".", "p")


def _preset_tag() -> str:
    sizes = MODEL_HIDDEN_DIMS
    if len(set(sizes)) == 1:
        return f"mlp{sizes[0]}x{len(sizes)}"
    return "mlp_" + "_".join(str(s) for s in sizes)


def _sample_run_name(dim: int, beta_m: float, seed: int) -> str:
    return f"{ENERGY}_d{dim}_beta{_beta_str(beta_m)}_ula_seed{seed}"


def _model_run_name(sample_run: str, seed: int) -> str:
    return f"{sample_run}_{_preset_tag()}_{LOSS_TYPE}_seed{seed}"


def _experiment_dir() -> Path:
    tag = ENERGY if not DESC else f"{ENERGY}_{DESC}"
    return EXPERIMENT_DIR / tag


# ---------------------------------------------------------------------------
# Config factories
# ---------------------------------------------------------------------------

def _make_sampling_config(dim: int, beta_m: float, seed: int) -> dict:
    run_name = _sample_run_name(dim, beta_m, seed)
    return {
        "job":    {"seed": seed, "device": "auto", "dtype": "float32"},
        "target": {"energy": ENERGY, "dim": dim, "beta_m": beta_m},
        "sampler": {
            "method":      "ULA",
            "step_size":   MCMC_STEP_SIZE,
            "n_particles": MCMC_N_PARTICLES,
            "n_steps":     MCMC_N_STEPS,
            "burn_in":     MCMC_BURN_IN,
            "save_every":  MCMC_SAVE_EVERY,
            "init_scale":  1.0,
        },
        "output": {"root": "runs/sampling", "run_name": run_name},
    }


def _make_training_config(sample_run: str, seed: int) -> dict:
    model_run = _model_run_name(sample_run, seed)
    return {
        "job":      {"seed": seed, "device": "auto", "dtype": "float32"},
        "samples":  {"run_name": sample_run, "snapshot": -1},
        "schedule": {"type": "vp", "beta_min": 0.1, "beta_max": 20.0},
        "model": {
            "hidden_dims":    MODEL_HIDDEN_DIMS,
            "time_embed_dim": MODEL_TIME_EMBED_DIM,
            "activation":     "silu",
            "predict_score":  LOSS_TYPE == "score",
        },
        "training": {
            "n_steps":       TRAIN_N_STEPS,
            "batch_size":    BATCH_SIZE,
            "lr":            LR,
            "t_eps":         T_EPS,
            "grad_clip":     GRAD_CLIP,
            "ema_decay":     EMA_DECAY,
            "log_every":     500,
            "log_uniform_t": False,
            "loss_type":     LOSS_TYPE,
        },
        "output": {"root": "data/models", "run_name": model_run},
    }


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def setup_and_sample(dim: int, seed: int) -> dict[float, str]:
    """Ensure MCMC samples exist for all BETA_MS. Returns {beta_m: sample_run_name}."""
    SAMPLING_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    runs: dict[float, str] = {}
    for beta_m in BETA_MS:
        run_name = _sample_run_name(dim, beta_m, seed)
        out_dir  = SAMPLE_DIR / run_name
        if REUSE_SAMPLES and (out_dir / "particles.pt").exists():
            print(f"    [sample] REUSE  {run_name}")
            runs[beta_m] = run_name
            continue
        print(f"    [sample] RUN    {run_name}")
        cfg      = _make_sampling_config(dim, beta_m, seed)
        cfg_path = SAMPLING_CONFIGS_DIR / f"{run_name}.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        _run_sample_config(cfg_path)
        runs[beta_m] = run_name
    return runs


def setup_and_train(sample_runs: dict[float, str], seed: int) -> dict[float, str]:
    """Ensure trained models exist. Returns {beta_m: model_run_name}."""
    TRAINING_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    runs: dict[float, str] = {}
    for beta_m in sorted(sample_runs):
        sample_run = sample_runs[beta_m]
        model_run  = _model_run_name(sample_run, seed)
        out_dir    = MODEL_DIR / model_run
        if REUSE_MODELS and (out_dir / "ema_model.pt").exists():
            print(f"    [train]  REUSE  {model_run}")
            runs[beta_m] = model_run
            continue
        print(f"    [train]  RUN    {model_run}")
        cfg      = _make_training_config(sample_run, seed)
        cfg_path = TRAINING_CONFIGS_DIR / f"{model_run}.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        _run_train_config(cfg_path)
        runs[beta_m] = model_run
    return runs


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(model_run: str, device: torch.device):
    """Returns (reverse_sde, energy, dim, beta_m, sample_run)."""
    run_dir = MODEL_DIR / model_run
    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    sample_run     = cfg["samples"]["run_name"]
    sample_summary = json.loads((SAMPLE_DIR / sample_run / "summary.json").read_text())

    dim    = sample_summary["dim"]
    beta_m = sample_summary["beta_m"]
    energy = ENERGY_MAP[sample_summary["energy"]](dim=dim)

    model_cfg = cfg["model"]
    model = MLPScore(
        dim            = dim,
        hidden_dims    = tuple(model_cfg["hidden_dims"]),
        time_embed_dim = model_cfg.get("time_embed_dim", 64),
        activation     = model_cfg.get("activation", "silu"),
        predict_score  = model_cfg.get("predict_score", False),
    )
    state = torch.load(run_dir / "ema_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    sched_cfg = cfg["schedule"]
    schedule  = VPSchedule(
        beta_min=sched_cfg.get("beta_min", 0.1),
        beta_max=sched_cfg.get("beta_max", 20.0),
    )
    return ReverseSDE(model, schedule), energy, dim, beta_m, sample_run


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _energy_stats(x: torch.Tensor, energy_fn) -> dict:
    with torch.no_grad():
        e = energy_fn(x.cpu()).float()
    return {
        "mean_energy":   round(e.mean().item(),   4),
        "min_energy":    round(e.min().item(),    4),
        "std_energy":    round(e.std().item(),    4),
        "median_energy": round(e.median().item(), 4),
    }


def _full_energy_stats(x: torch.Tensor, energy_fn) -> dict:
    with torch.no_grad():
        e = energy_fn(x.cpu()).float()
    qs = torch.quantile(e, torch.tensor([0.01, 0.05, 0.25, 0.75, 0.95]))
    return {
        "mean_energy":   round(e.mean().item(),   4),
        "median_energy": round(e.median().item(), 4),
        "min_energy":    round(e.min().item(),    4),
        "std_energy":    round(e.std().item(),    4),
        "q01": round(qs[0].item(), 4),
        "q05": round(qs[1].item(), 4),
        "q25": round(qs[2].item(), 4),
        "q75": round(qs[3].item(), 4),
        "q95": round(qs[4].item(), 4),
    }


def _mcmc_stats_from_summary(sample_run: str) -> dict:
    """Load MCMC energy stats, recomputing quantiles from particles if needed."""
    summary = json.loads((SAMPLE_DIR / sample_run / "summary.json").read_text())
    base = {
        "mean_energy":   summary["mean_energy"],
        "median_energy": summary["median_energy"],
        "min_energy":    summary["min_energy"],
        "std_energy":    summary["std_energy"],
    }
    eq = summary.get("energy_quantiles", {})
    if "q01" in eq:
        for k in ["q01", "q05", "q25", "q75", "q95"]:
            base[k] = eq.get(k, float("nan"))
    else:
        pts = torch.load(
            SAMPLE_DIR / sample_run / "particles.pt",
            map_location="cpu", weights_only=True,
        )
        final = pts[:, -1, :]
        energy_fn = ENERGY_MAP[ENERGY](dim=final.shape[-1])
        with torch.no_grad():
            e = energy_fn(final).float()
        qs = torch.quantile(e, torch.tensor([0.01, 0.05, 0.25, 0.75, 0.95]))
        for k, v in zip(["q01", "q05", "q25", "q75", "q95"], qs.tolist()):
            base[k] = round(v, 4)
    return base


def _delta_stats(model_s: dict, mcmc_s: dict) -> dict:
    eps = 1e-8
    result = {}
    for key in ["mean_energy", "median_energy", "min_energy", "std_energy"]:
        short = key.replace("_energy", "")
        result[f"delta_{short}"] = round(
            abs(model_s[key] - mcmc_s[key]) / (abs(mcmc_s[key]) + eps), 4
        )
    return result


def _compute_diag_stats(diag: SMCDiagnostics) -> dict:
    ess  = diag.ess_ratios
    stds = diag.log_weight_stds
    return {
        "min_ess":      round(min(ess),  4) if ess  else None,
        "mean_ess":     round(sum(ess) / len(ess), 4) if ess else None,
        "max_logw_var": round(max(s**2 for s in stds), 4) if stds else None,
    }


# ---------------------------------------------------------------------------
# Model sample generation
# ---------------------------------------------------------------------------

def generate_model_samples(model_run: str, seed: int, device: torch.device) -> dict:
    out_dir = MODEL_SAMPLE_DIR / model_run

    if REUSE_MODEL_SAMPLES and (out_dir / "samples.pt").exists():
        print(f"    [model samples] REUSE  {model_run}")
        summary = json.loads((out_dir / "summary.json").read_text())
        return {k: summary[k] for k in ("model_stats", "mcmc_stats", "delta_stats", "wall_clock_seconds")}

    rsde, energy, dim, beta_m, sample_run = _load_model(model_run, device)

    print(f"    [model samples] RUN    {model_run}  N={MCMC_N_PARTICLES}")
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    x = DiffusionModelSampler(
        rsde, n_steps=MODEL_SAMPLE_N_STEPS,
        t_start=MODEL_SAMPLE_T_START, t_end=MODEL_SAMPLE_T_END,
    ).sample(MCMC_N_PARTICLES, device, show_progress=True)
    wall = round(time.perf_counter() - t0, 2)

    model_stats = _full_energy_stats(x.cpu(), energy)
    mcmc_stats  = _mcmc_stats_from_summary(sample_run)
    delta_stats = _delta_stats(model_stats, mcmc_stats)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(x.cpu(), out_dir / "samples.pt")

    summary = {
        "model_run":          model_run,
        "n_samples":          MCMC_N_PARTICLES,
        "n_steps":            MODEL_SAMPLE_N_STEPS,
        "t_start":            MODEL_SAMPLE_T_START,
        "t_end":              MODEL_SAMPLE_T_END,
        "seed":               seed,
        "wall_clock_seconds": wall,
        "model_stats":        model_stats,
        "mcmc_stats":         mcmc_stats,
        "delta_stats":        delta_stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(
            {"model_run": model_run, "n_samples": MCMC_N_PARTICLES,
             "n_steps": MODEL_SAMPLE_N_STEPS, "t_start": MODEL_SAMPLE_T_START,
             "t_end": MODEL_SAMPLE_T_END, "seed": seed},
            f, default_flow_style=False, sort_keys=False,
        )

    return {"model_stats": model_stats, "mcmc_stats": mcmc_stats,
            "delta_stats": delta_stats, "wall_clock_seconds": wall}


# ---------------------------------------------------------------------------
# Annealing
# ---------------------------------------------------------------------------

def _run_all_annealing(
    model_run: str, sample_run: str, seed: int, device: torch.device
) -> dict:
    rsde, energy, dim, beta_m, _ = _load_model(model_run, device)

    pts = torch.load(
        SAMPLE_DIR / sample_run / "particles.pt", map_location=device, weights_only=True,
    )
    x0 = pts[:, -1, :]
    if x0.shape[0] > N_PARTICLES:
        x0 = x0[torch.randperm(x0.shape[0], device=device)[:N_PARTICLES]]
    initial_cloud = ParticleCloud(x0, torch.zeros(N_PARTICLES, device=device))
    beta_ladder   = torch.linspace(beta_m, BETA_H, N_SMC_STEPS + 1, device=device)

    results = {}

    # ULA SMC
    print("      ULA SMC...")
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    ula_p = ULAProposal(energy, step_size=ULA_STEP_SIZE, n_steps=N_ULA_STEPS)
    ula_cloud, ula_diag = SMCSampler(
        ula_p.mutation_kernel, ula_p.weight_update, energy, ess_threshold=ESS_THRESHOLD,
    ).run(initial_cloud, beta_ladder, show_progress=True)
    results["ula"] = {
        **_energy_stats(ula_cloud.x, energy),
        "final_ess":      round(ula_cloud.ess_ratio(), 4),
        **_compute_diag_stats(ula_diag),
        "n_resamples":    ula_diag.n_resamples,
        "wall_seconds":   round(time.perf_counter() - t0, 2),
        "ess_trajectory": [round(e, 4) for e in ula_diag.ess_ratios],
    }

    # Diffusion SMC
    print("      Diffusion SMC...")
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    diff_p = DiffusionAnnealingProposal(
        rsde, energy, beta_train=beta_m,
        n_diffusion_steps=N_DIFFUSION_STEPS,
        t_start=DIFF_T_START, t_end=DIFF_T_END, score_scaling=DIFF_SCORE_SCALING,
    )
    diff_cloud, diff_diag = SMCSampler(
        diff_p.mutation_kernel, diff_p.weight_update, energy, ess_threshold=ESS_THRESHOLD,
    ).run(initial_cloud, beta_ladder, show_progress=True)
    results["diffusion"] = {
        **_energy_stats(diff_cloud.x, energy),
        "final_ess":      round(diff_cloud.ess_ratio(), 4),
        **_compute_diag_stats(diff_diag),
        "n_resamples":    diff_diag.n_resamples,
        "wall_seconds":   round(time.perf_counter() - t0, 2),
        "ess_trajectory": [round(e, 4) for e in diff_diag.ess_ratios],
    }

    # FKC
    print("      FKC...")
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    fkc_cloud, fkc_diag = FKCAnnealedSampler(
        rsde, energy, beta_train=beta_m, beta_target=BETA_H,
        n_steps=FKC_N_STEPS, t_start=FKC_T_START, t_end=FKC_T_END,
        include_divergence=FKC_INCLUDE_DIVERGENCE, ess_threshold=ESS_THRESHOLD,
    ).run(N_PARTICLES, dim, device, show_progress=True)
    results["fkc"] = {
        **_energy_stats(fkc_cloud.x, energy),
        "final_ess":      round(fkc_cloud.ess_ratio(), 4),
        **_compute_diag_stats(fkc_diag),
        "n_resamples":    fkc_diag.n_resamples,
        "wall_seconds":   round(time.perf_counter() - t0, 2),
        "ess_trajectory": [round(e, 4) for e in fkc_diag.ess_ratios],
    }

    return results


# ---------------------------------------------------------------------------
# Per-seed run (with incremental save)
# ---------------------------------------------------------------------------

def run_seed(seed: int, device: torch.device) -> list[dict]:
    out_path = _experiment_dir() / f"results_seed{seed}.json"

    # Load existing results for resumption
    existing: dict[tuple, dict] = {}
    if REUSE_ANNEALING and out_path.exists():
        for r in json.loads(out_path.read_text()).get("runs", []):
            existing[(r["dim"], r["beta_m"])] = r

    results = list(existing.values())

    for dim in DIMS:
        print(f"\n  dim={dim}  seed={seed}")
        sample_runs = setup_and_sample(dim, seed)
        model_runs  = setup_and_train(sample_runs, seed)

        for beta_m in BETA_MS:
            key = (dim, beta_m)
            if key in existing:
                continue

            model_run  = model_runs[beta_m]
            sample_run = sample_runs[beta_m]
            print(f"\n    beta_M={beta_m}  [{model_run}]")

            ms    = generate_model_samples(model_run, seed, device)
            ann   = _run_all_annealing(model_run, sample_run, seed, device)

            sample_summary = json.loads((SAMPLE_DIR / sample_run / "summary.json").read_text())
            model_summary  = json.loads((MODEL_DIR  / model_run  / "summary.json").read_text())

            entry = {
                "dim":    dim,
                "beta_m": beta_m,
                "model_run": model_run,
                "sampling_wall_seconds":      sample_summary.get("wall_clock_seconds"),
                "training_wall_seconds":      model_summary.get("wall_clock_seconds"),
                "model_sampling_wall_seconds": ms["wall_clock_seconds"],
                "mcmc_stats":  ms["mcmc_stats"],
                "model_stats": ms["model_stats"],
                "delta_stats": ms["delta_stats"],
                **ann,
            }
            results.append(entry)
            existing[key] = entry

            # Incremental save after each (dim, betaM)
            _experiment_dir().mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({"seed": seed, "runs": results}, indent=2))

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_dict(dicts: list[dict], skip_keys: set | None = None) -> dict:
    """Mean ± std for every scalar field across a list of same-shaped dicts."""
    skip_keys = skip_keys or set()
    result = {}
    for key in dicts[0]:
        if key in skip_keys:
            continue
        vals = [d[key] for d in dicts if key in d and isinstance(d[key], (int, float))]
        if vals:
            result[key] = {
                "mean": round(sum(vals) / len(vals), 6),
                "std":  round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 6),
            }
    return result


def aggregate_results() -> None:
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)

    for seed in SEEDS:
        path = _experiment_dir() / f"results_seed{seed}.json"
        if not path.exists():
            print(f"  WARNING: missing results_seed{seed}.json — skipping")
            continue
        for run in json.loads(path.read_text()).get("runs", []):
            groups[(run["dim"], run["beta_m"])].append(run)

    SKIP = {"ess_trajectory"}
    aggregate_runs = []
    for (dim, beta_m), seed_runs in sorted(groups.items()):
        agg: dict = {"dim": dim, "beta_m": beta_m}
        agg["delta_stats"] = _aggregate_dict([r["delta_stats"] for r in seed_runs])
        for method in ("ula", "diffusion", "fkc"):
            method_dicts = [r[method] for r in seed_runs if method in r]
            if method_dicts:
                agg[method] = _aggregate_dict(method_dicts, skip_keys=SKIP)
        aggregate_runs.append(agg)

    path = _experiment_dir() / "aggregate.json"
    path.write_text(json.dumps({"runs": aggregate_runs}, indent=2))
    print(f"\nAggregate saved to {path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results() -> None:
    import matplotlib.pyplot as plt

    aggregate = json.loads((_experiment_dir() / "aggregate.json").read_text())
    config    = json.loads((_experiment_dir() / "config.json").read_text())

    plots_dir = _experiment_dir() / "plots"
    plots_dir.mkdir(exist_ok=True)

    methods = [
        ("ula",       "ULA SMC",       "steelblue"),
        ("diffusion", "Diffusion SMC", "darkorange"),
        ("fkc",       "FKC",           "mediumseagreen"),
    ]

    for dim in DIMS:
        runs = sorted(
            [r for r in aggregate["runs"] if r["dim"] == dim],
            key=lambda r: r["beta_m"],
        )
        if not runs:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

        for ax, metric, ylabel in [
            (ax1, "mean_energy", "Mean energy"),
            (ax2, "min_energy",  "Min energy"),
        ]:
            for method_key, label, color in methods:
                betas = [r["beta_m"] for r in runs if method_key in r]
                vals  = [r[method_key][metric]["mean"] for r in runs if method_key in r]
                stds  = [r[method_key][metric]["std"]  for r in runs if method_key in r]
                if not vals:
                    continue
                ax.plot(betas, vals, "o-", label=label, color=color)
                ax.fill_between(
                    betas,
                    [v - s for v, s in zip(vals, stds)],
                    [v + s for v, s in zip(vals, stds)],
                    color=color, alpha=0.15,
                )
            ax.set_xlabel(r"$\beta_M$ (training temperature)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} at $\\beta_H = {config['beta_h']}$")
            ax.legend()
            ax.grid(True, alpha=0.3)

        desc_str = f"  [{config['desc']}]" if config.get("desc") else ""
        fig.suptitle(
            f"{config['energy']}  d={dim}  {len(config['seeds'])} seeds  ±1σ{desc_str}",
            fontsize=9,
        )
        fig.tight_layout()
        out = plots_dir / f"d{dim}_energy.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to {out}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Contour plots
# ---------------------------------------------------------------------------

_scipy_kde = None
try:
    from scipy.stats import gaussian_kde as _scipy_kde  # type: ignore[assignment]
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _plot_kde_overlay(ax, pts2d: "np.ndarray", color: str, label: str) -> None:
    kde = _scipy_kde(pts2d.T)
    lo = pts2d.min(axis=0) - 0.5
    hi = pts2d.max(axis=0) + 0.5
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 80), np.linspace(lo[1], hi[1], 80))
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax.contour(xx, yy, zz, levels=5, colors=[color], linewidths=1.5, alpha=0.9)
    ax.plot([], [], color=color, linewidth=1.5, label=label)


def plot_sample_contours(kde: bool = CONTOUR_KDE) -> None:
    import matplotlib.pyplot as plt

    if kde and not _HAS_SCIPY:
        print("  WARNING: scipy not installed — falling back to scatter.")
        kde = False

    plots_dir = _experiment_dir() / "plots"
    plots_dir.mkdir(exist_ok=True)
    energy_cls = ENERGY_MAP[ENERGY]

    for dim in DIMS:
        energy_fn = energy_cls(dim=dim)

        # ------------------------------------------------------------------
        # Pass 1: load all samples for this dim across every (beta_m, seed).
        # PCA is fitted once on the union so the projection is consistent
        # across beta_ms, and the plot range is fixed for the whole dim.
        # ------------------------------------------------------------------
        sample_data: dict[float, tuple] = {}  # beta_m -> (diff_np, mcmc_np)
        all_raw: list[np.ndarray] = []

        for beta_m in BETA_MS:
            diff_parts, mcmc_parts = [], []
            for seed in SEEDS:
                sample_run = _sample_run_name(dim, beta_m, seed)
                model_run  = _model_run_name(sample_run, seed)
                d_path = MODEL_SAMPLE_DIR / model_run / "samples.pt"
                m_path = SAMPLE_DIR / sample_run / "particles.pt"
                if d_path.exists():
                    diff_parts.append(
                        torch.load(d_path, map_location="cpu", weights_only=True).numpy()
                    )
                if m_path.exists():
                    pts = torch.load(m_path, map_location="cpu", weights_only=True)
                    mcmc_parts.append(pts[:, -1, :].numpy())
            if diff_parts and mcmc_parts:
                d = np.concatenate(diff_parts, axis=0)
                m = np.concatenate(mcmc_parts, axis=0)
                sample_data[beta_m] = (d, m)
                all_raw.extend([d, m])

        if not sample_data:
            continue

        all_np = np.concatenate(all_raw, axis=0)

        # ------------------------------------------------------------------
        # Dimensionality reduction (fitted once for this dim)
        # ------------------------------------------------------------------
        if dim == 2:
            mean_vec      = None
            xlabel, ylabel = "x₀", "x₁"
            pca_info       = ""

            def _proj(x):     return x
            def _energy_grid(xx, yy):
                pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
                with torch.no_grad():
                    e = energy_fn(torch.tensor(pts, dtype=torch.float32)).numpy()
                return e.reshape(xx.shape)

        else:
            mean_vec  = all_np.mean(axis=0)
            centered  = all_np - mean_vec
            _, s_vals, Vt = np.linalg.svd(centered, full_matrices=False)
            pca_components = Vt[:2]
            exp_var   = s_vals[:2] ** 2 / (s_vals ** 2).sum()
            xlabel    = f"PC1 ({exp_var[0]:.0%})"
            ylabel    = f"PC2 ({exp_var[1]:.0%})"
            pca_info  = f"  PCA {exp_var[0]+exp_var[1]:.0%} var."

            def _proj(x):
                return (x - mean_vec) @ pca_components.T

            def _energy_grid(xx, yy):
                pts_2d   = np.stack([xx.ravel(), yy.ravel()], axis=1)
                pts_full = mean_vec + pts_2d @ pca_components
                with torch.no_grad():
                    e = energy_fn(torch.tensor(pts_full, dtype=torch.float32)).numpy()
                return e.reshape(xx.shape)

        # ------------------------------------------------------------------
        # Fixed plot range across all beta_ms for this dim
        # ------------------------------------------------------------------
        all_2d = _proj(all_np)
        pad = 0.12
        x0, x1 = all_2d[:, 0].min(), all_2d[:, 0].max()
        y0, y1 = all_2d[:, 1].min(), all_2d[:, 1].max()
        dx, dy  = max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)
        x0 -= pad * dx;  x1 += pad * dx
        y0 -= pad * dy;  y1 += pad * dy

        # Energy grid — computed once per dim (landscape doesn't change with beta_m)
        xx, yy = np.meshgrid(np.linspace(x0, x1, 120), np.linspace(y0, y1, 120))
        zz = _energy_grid(xx, yy)

        # Global optima projected
        gm      = energy_fn.global_minima
        gm_2d   = _proj(gm.numpy()) if gm is not None else None

        # ------------------------------------------------------------------
        # Pass 2: one figure per beta_m (fixed range, fixed energy bg)
        # ------------------------------------------------------------------
        for beta_m, (diff_np, mcmc_np) in sample_data.items():
            diff_2d = _proj(diff_np)
            mcmc_2d = _proj(mcmc_np)

            fig, ax = plt.subplots(figsize=(6, 5))
            cf = ax.contourf(xx, yy, zz, levels=25, cmap="YlOrRd_r", alpha=0.75)
            ax.contour(xx, yy, zz, levels=25, colors="k", linewidths=0.3, alpha=0.3)
            plt.colorbar(cf, ax=ax, label="E(x)")

            _DIFF_COLOR = "lime"
            _MCMC_COLOR = "#1A3A8A"

            if kde:
                _plot_kde_overlay(ax, diff_2d, color=_DIFF_COLOR, label="Diffusion model")
                _plot_kde_overlay(ax, mcmc_2d, color=_MCMC_COLOR, label="MCMC (ULA)")
            else:
                n = min(2000, len(diff_2d))
                ax.scatter(diff_2d[:n, 0], diff_2d[:n, 1],
                           c=_DIFF_COLOR, s=5, alpha=0.6, linewidths=0,
                           label="Diffusion model", zorder=3)
                n = min(2000, len(mcmc_2d))
                ax.scatter(mcmc_2d[:n, 0], mcmc_2d[:n, 1],
                           c=_MCMC_COLOR, s=5, alpha=0.6, linewidths=0,
                           label="MCMC (ULA)", zorder=3)

            if gm_2d is not None:
                ax.scatter(gm_2d[:, 0], gm_2d[:, 1],
                           marker="*", s=120, c="gold", edgecolors="k",
                           linewidths=0.5, zorder=5, label="Global min")

            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
            ax.set_xlim(x0, x1);   ax.set_ylim(y0, y1)
            ax.set_title(f"{ENERGY}  d={dim}  β_M={beta_m}{pca_info}", fontsize=10)
            ax.legend(loc="upper right", markerscale=1, fontsize=8)

            fig.tight_layout()
            out = plots_dir / f"contour_d{dim}_beta{_beta_str(beta_m)}.svg"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            print(f"  [contour] d={dim} β_M={beta_m} → {out.name}")


def plot_contour_grids(kde: bool = CONTOUR_KDE) -> None:
    """One grid figure per dim combining all beta_m contour plots in sorted order."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D

    if kde and not _HAS_SCIPY:
        print("  WARNING: scipy not installed — falling back to scatter.")
        kde = False

    plots_dir = _experiment_dir() / "plots"
    plots_dir.mkdir(exist_ok=True)
    energy_cls = ENERGY_MAP[ENERGY]

    _DIFF_COLOR = "lime"
    _MCMC_COLOR = "#1A3A8A"
    beta_ms_sorted = sorted(BETA_MS)

    for dim in DIMS:
        energy_fn = energy_cls(dim=dim)

        # Load samples — identical to plot_sample_contours Pass 1
        sample_data: dict[float, tuple] = {}
        all_raw: list[np.ndarray] = []

        for beta_m in beta_ms_sorted:
            diff_parts, mcmc_parts = [], []
            for seed in SEEDS:
                sample_run = _sample_run_name(dim, beta_m, seed)
                model_run  = _model_run_name(sample_run, seed)
                d_path = MODEL_SAMPLE_DIR / model_run / "samples.pt"
                m_path = SAMPLE_DIR / sample_run / "particles.pt"
                if d_path.exists():
                    diff_parts.append(
                        torch.load(d_path, map_location="cpu", weights_only=True).numpy()
                    )
                if m_path.exists():
                    pts = torch.load(m_path, map_location="cpu", weights_only=True)
                    mcmc_parts.append(pts[:, -1, :].numpy())
            if diff_parts and mcmc_parts:
                d = np.concatenate(diff_parts, axis=0)
                m = np.concatenate(mcmc_parts, axis=0)
                sample_data[beta_m] = (d, m)
                all_raw.extend([d, m])

        if not sample_data:
            continue

        all_np = np.concatenate(all_raw, axis=0)

        if dim == 2:
            xlabel, ylabel = "x₀", "x₁"
            pca_info = ""

            def _proj(x):
                return x

            def _energy_grid(xx, yy):
                pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
                with torch.no_grad():
                    e = energy_fn(torch.tensor(pts, dtype=torch.float32)).numpy()
                return e.reshape(xx.shape)

        else:
            mean_vec = all_np.mean(axis=0)
            centered = all_np - mean_vec
            _, s_vals, Vt = np.linalg.svd(centered, full_matrices=False)
            pca_components = Vt[:2]
            exp_var  = s_vals[:2] ** 2 / (s_vals ** 2).sum()
            xlabel   = f"PC1 ({exp_var[0]:.0%})"
            ylabel   = f"PC2 ({exp_var[1]:.0%})"
            pca_info = f"  PCA {exp_var[0]+exp_var[1]:.0%} var."

            def _proj(x):
                return (x - mean_vec) @ pca_components.T

            def _energy_grid(xx, yy):
                pts_2d   = np.stack([xx.ravel(), yy.ravel()], axis=1)
                pts_full = mean_vec + pts_2d @ pca_components
                with torch.no_grad():
                    e = energy_fn(torch.tensor(pts_full, dtype=torch.float32)).numpy()
                return e.reshape(xx.shape)

        all_2d = _proj(all_np)
        pad = 0.12
        x0, x1 = all_2d[:, 0].min(), all_2d[:, 0].max()
        y0, y1 = all_2d[:, 1].min(), all_2d[:, 1].max()
        dx, dy  = max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)
        x0 -= pad * dx;  x1 += pad * dx
        y0 -= pad * dy;  y1 += pad * dy

        xx, yy = np.meshgrid(np.linspace(x0, x1, 100), np.linspace(y0, y1, 100))
        zz     = _energy_grid(xx, yy)
        vmin, vmax = float(zz.min()), float(zz.max())

        gm    = energy_fn.global_minima
        gm_2d = _proj(gm.numpy()) if gm is not None else None

        # 3 rows × 4 cols for up to 12 beta_ms
        ncols = 4
        nrows = (len(beta_ms_sorted) + ncols - 1) // ncols

        fig = plt.figure(figsize=(ncols * 4.2 + 0.7, nrows * 3.8 + 0.6))
        gs  = gridspec.GridSpec(
            nrows, ncols + 1, figure=fig,
            width_ratios=[1.0] * ncols + [0.04],
            hspace=0.40, wspace=0.18,
        )
        cax = fig.add_subplot(gs[:, -1])

        last_cf = None
        for idx, beta_m in enumerate(beta_ms_sorted):
            row, col = divmod(idx, ncols)
            ax = fig.add_subplot(gs[row, col])

            cf = ax.contourf(xx, yy, zz, levels=20, cmap="YlOrRd_r",
                             alpha=0.75, vmin=vmin, vmax=vmax)
            ax.contour(xx, yy, zz, levels=20, colors="k",
                       linewidths=0.2, alpha=0.25, vmin=vmin, vmax=vmax)
            last_cf = cf

            if beta_m in sample_data:
                diff_2d = _proj(sample_data[beta_m][0])
                mcmc_2d = _proj(sample_data[beta_m][1])
                if kde:
                    _plot_kde_overlay(ax, diff_2d, color=_DIFF_COLOR, label="Diffusion")
                    _plot_kde_overlay(ax, mcmc_2d, color=_MCMC_COLOR, label="MCMC")
                else:
                    n = min(1000, len(diff_2d))
                    ax.scatter(diff_2d[:n, 0], diff_2d[:n, 1],
                               c=_DIFF_COLOR, s=3, alpha=0.5, linewidths=0, zorder=3)
                    n = min(1000, len(mcmc_2d))
                    ax.scatter(mcmc_2d[:n, 0], mcmc_2d[:n, 1],
                               c=_MCMC_COLOR, s=3, alpha=0.5, linewidths=0, zorder=3)

            if gm_2d is not None:
                ax.scatter(gm_2d[:, 0], gm_2d[:, 1],
                           marker="*", s=70, c="gold", edgecolors="k",
                           linewidths=0.4, zorder=5)

            ax.set_title(f"β_M={beta_m}", fontsize=8, pad=3)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=7)
            if row == nrows - 1:
                ax.set_xlabel(xlabel, fontsize=7)

        if last_cf is not None:
            plt.colorbar(last_cf, cax=cax, label="E(x)")

        # Shared legend below the grid
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=_DIFF_COLOR,
                   markersize=6, label="Diffusion model"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=_MCMC_COLOR,
                   markersize=6, label="MCMC (ULA)"),
        ]
        if gm_2d is not None:
            handles.append(Line2D(
                [0], [0], marker="*", color="w", markerfacecolor="gold",
                markeredgecolor="k", markeredgewidth=0.4, markersize=9,
                label="Global min",
            ))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                   fontsize=8, bbox_to_anchor=(0.46, -0.02))

        fig.suptitle(f"{ENERGY}  d={dim}{pca_info}", fontsize=11)
        out = plots_dir / f"contour_grid_d{dim}.svg"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  [contour grid] d={dim} → {out.name}")


# ---------------------------------------------------------------------------
# Config snapshot
# ---------------------------------------------------------------------------

def _config_snapshot() -> dict:
    return {
        "energy":  ENERGY,
        "dims":    DIMS,
        "beta_ms": BETA_MS,
        "beta_h":  BETA_H,
        "seeds":   SEEDS,
        "desc":    DESC,
        "mcmc": {
            "n_particles": MCMC_N_PARTICLES,
            "n_steps":     MCMC_N_STEPS,
            "burn_in":     MCMC_BURN_IN,
            "save_every":  MCMC_SAVE_EVERY,
            "step_size":   MCMC_STEP_SIZE,
        },
        "model": {
            "hidden_dims":    MODEL_HIDDEN_DIMS,
            "time_embed_dim": MODEL_TIME_EMBED_DIM,
            "loss_type":      LOSS_TYPE,
            "train_n_steps":  TRAIN_N_STEPS,
            "batch_size":     BATCH_SIZE,
            "lr":             LR,
        },
        "model_sampling": {
            "n_steps": MODEL_SAMPLE_N_STEPS,
            "t_start": MODEL_SAMPLE_T_START,
            "t_end":   MODEL_SAMPLE_T_END,
        },
        "annealing": {
            "n_particles":   N_PARTICLES,
            "n_smc_steps":   N_SMC_STEPS,
            "ess_threshold": ESS_THRESHOLD,
            "ula": {"n_steps": N_ULA_STEPS, "step_size": ULA_STEP_SIZE},
            "diffusion": {
                "n_steps": N_DIFFUSION_STEPS, "t_start": DIFF_T_START,
                "t_end": DIFF_T_END, "score_scaling": DIFF_SCORE_SCALING,
            },
            "fkc": {
                "n_steps": FKC_N_STEPS, "t_start": FKC_T_START,
                "t_end": FKC_T_END, "include_divergence": FKC_INCLUDE_DIVERGENCE,
            },
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global ENERGY

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--energy",
        choices=list(ENERGY_MAP),
        default=None,
        help="Override ENERGY from the USER CONFIG block",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run a single seed only (no aggregation); run --aggregate when all seeds done",
    )
    parser.add_argument("--plot-only",  action="store_true",
                        help="Re-plot from existing aggregate.json (no pipeline)")
    parser.add_argument("--aggregate",  action="store_true",
                        help="Aggregate all seed results and plot (use after parallel seeds finish)")
    parser.add_argument("--plot-contours", action="store_true",
                        help="Plot energy contour + sample overlays per (dim, betaM)")
    parser.add_argument("--plot-contour-grids", action="store_true",
                        help="Plot one grid figure per dim combining all betaM contour panels")
    args = parser.parse_args()

    if args.energy is not None:
        ENERGY = args.energy

    if args.plot_only:
        plot_results()
        return

    if args.plot_contours:
        plot_sample_contours()
        return

    if args.plot_contour_grids:
        plot_contour_grids()
        return

    if args.aggregate:
        print("Aggregating across seeds...")
        aggregate_results()
        print("Plotting...")
        plot_results()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)

    seeds_to_run = [args.seed] if args.seed is not None else SEEDS

    exp_dir = _experiment_dir()
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.joinpath("config.json").write_text(json.dumps(_config_snapshot(), indent=2))

    print(f"=== {ENERGY}  dims={DIMS}  beta_H={BETA_H}  seeds={seeds_to_run} ===")
    print(f"    beta_Ms={BETA_MS}  device={device}\n")

    for seed in seeds_to_run:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")
        run_seed(seed, device)

    if args.seed is None:
        print("\nAggregating across seeds...")
        aggregate_results()
        print("\nPlotting...")
        plot_results()
    else:
        print(f"\nSeed {args.seed} done. Run --aggregate once all seeds complete.")


if __name__ == "__main__":
    main()
