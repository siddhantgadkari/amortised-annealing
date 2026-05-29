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
# USER CONFIGURATION
# ===========================================================================
ENERGY  = "ackley"
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
            runs[beta_m] = run_name
            continue
        print(f"    [sample] {run_name}")
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
            runs[beta_m] = model_run
            continue
        print(f"    [train]  {model_run}")
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
        summary = json.loads((out_dir / "summary.json").read_text())
        return {k: summary[k] for k in ("model_stats", "mcmc_stats", "delta_stats", "wall_clock_seconds")}

    rsde, energy, dim, beta_m, sample_run = _load_model(model_run, device)

    print(f"    [model samples] {model_run}  N={MCMC_N_PARTICLES}")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip pipeline and re-plot from saved aggregate.json")
    args = parser.parse_args()

    if args.plot_only:
        plot_results()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)

    exp_dir = _experiment_dir()
    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = exp_dir / "config.json"
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps(_config_snapshot(), indent=2))

    print(f"=== {ENERGY}  dims={DIMS}  beta_H={BETA_H}  seeds={SEEDS} ===")
    print(f"    beta_Ms={BETA_MS}  device={device}\n")

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")
        run_seed(seed, device)

    print("\nAggregating across seeds...")
    aggregate_results()

    print("\nPlotting...")
    plot_results()


if __name__ == "__main__":
    main()
