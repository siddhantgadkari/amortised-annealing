#!/usr/bin/env python
"""Temperature-transfer surface: Experiments A and B.

Experiment A — triangular (β_M, β_H) heatmap:
  For each valid cell with β_H ≥ β_M, evaluate direct model samples and
  ULA SMC + Diffusion SMC annealing from β_M → β_H.

Experiment B — delta diagnostic:
  Quantile energy discrepancy Δ_E between model and MCMC samples,
  computed once per (dim, β_M) before any annealing.

Reuses MCMC samples, trained models, and model samples from
energy_betaM_experiment.py where they exist; runs the full pipeline
(sample → train → model_sample) for any missing (dim, β_M) combinations.

Usage:
    # Pilot: one seed, small grid — inspect before committing to full run
    uv run python scripts/experiments/temp_transfer.py --pilot

    # Full experiment (all seeds, auto-aggregates + plots when done)
    uv run python scripts/experiments/temp_transfer.py

    # Per-seed (e.g. on separate machines), then aggregate manually
    uv run python scripts/experiments/temp_transfer.py --seed 0
    uv run python scripts/experiments/temp_transfer.py --aggregate

    # Re-plot from existing aggregate
    uv run python scripts/experiments/temp_transfer.py --plot-only
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from amortised_annealing.diffusion import (
    VPSchedule, MLPScore, ReverseSDE, TrainingConfig, train_score_model,
)
from amortised_annealing.energies import DoubleWell, ManyWell, Ackley, Rastrigin
from amortised_annealing.sampling import DiffusionModelSampler
from amortised_annealing.sampling.sample_gen import SampleGenerator
from amortised_annealing.smc import (
    ParticleCloud, SMCDiagnostics, SMCSampler, ULAProposal, DiffusionAnnealingProposal,
)

ROOT             = Path(__file__).parent.parent.parent
SAMPLE_DIR       = ROOT / "data" / "samples"
MODEL_DIR        = ROOT / "data" / "models"
MODEL_SAMPLE_DIR = ROOT / "data" / "model_samples"
EXP_BASE         = ROOT / "data" / "experiments" / "temp_transfer"

ENERGY_MAP = {
    "double_well": DoubleWell,
    "many_well":   ManyWell,
    "ackley":      Ackley,
    "rastrigin":   Rastrigin,
}

# ===========================================================================
# USER CONFIGURATION
# ===========================================================================
ENERGY = "ackley"
DIMS   = [5, 10, 20]

# Training temperatures — models trained at each β_M
BETA_MS = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

# Target temperatures — only β_H ≥ β_M cells evaluated
BETA_HS = [5.0, 10.0, 20.0, 50.0, 100.0]

SEEDS = [0, 1, 2]

# Pilot subset (--pilot flag)
PILOT_DIMS    = [20]
PILOT_BETA_MS = [5.0, 10.0, 20.0]
PILOT_BETA_HS = [20.0, 50.0]
PILOT_SEEDS   = [0]

# Model architecture — must match energy_betaM_experiment.py
MODEL_HIDDEN_DIMS    = [256, 256, 256]
MODEL_TIME_EMBED_DIM = 64
LOSS_TYPE            = "eps"

# MCMC sampling
MCMC_N_PARTICLES = 8192
MCMC_N_STEPS     = 10_000
MCMC_BURN_IN     = 2_000
MCMC_TRACE_EVERY = 100
MCMC_STEP_SIZE   = 1e-3

# Score model training
TRAIN_N_STEPS = 20_000
BATCH_SIZE    = 512
LR            = 2e-4
T_EPS         = 1e-4
GRAD_CLIP     = 1.0
EMA_DECAY     = 0.999
LOG_EVERY     = 500

# Model sampling
MODEL_SAMPLE_N_STEPS = 500
MODEL_SAMPLE_T_START = 1.0
MODEL_SAMPLE_T_END   = 1e-3

# Annealing
N_PARTICLES        = 4096
N_SMC_STEPS        = 32
ESS_THRESHOLD      = 0.5
N_ULA_STEPS        = 10
ULA_STEP_SIZE      = 1e-3
N_DIFFUSION_STEPS  = 10
DIFF_T_START       = 0.5
DIFF_T_END         = 1e-3
DIFF_SCORE_SCALING = True

# Reuse flags
REUSE_SAMPLES       = True
REUSE_MODELS        = True
REUSE_MODEL_SAMPLES = True
REUSE_ANNEALING     = True   # skip cell if already in results_seed{s}.json

DESC = ""
# ===========================================================================


# ---------------------------------------------------------------------------
# Paths + naming (must produce same names as energy_betaM_experiment.py)
# ---------------------------------------------------------------------------

def _exp_dir() -> Path:
    tag = ENERGY if not DESC else f"{ENERGY}_{DESC}"
    return EXP_BASE / tag


def _beta_str(b: float) -> str:
    return f"{b:g}".replace(".", "p")


def _preset_tag() -> str:
    s = MODEL_HIDDEN_DIMS
    return f"mlp{s[0]}x{len(s)}" if len(set(s)) == 1 else "mlp_" + "_".join(str(x) for x in s)


def _sample_run_name(dim: int, beta_m: float, seed: int) -> str:
    return f"{ENERGY}_d{dim}_beta{_beta_str(beta_m)}_ula_seed{seed}"


def _model_run_name(sample_run: str, seed: int) -> str:
    return f"{sample_run}_{_preset_tag()}_{LOSS_TYPE}_seed{seed}"


# ---------------------------------------------------------------------------
# Pipeline helpers (sample → train → model_sample)
# ---------------------------------------------------------------------------

def _ensure_samples(dim: int, beta_m: float, seed: int, device: torch.device) -> str:
    run_name = _sample_run_name(dim, beta_m, seed)
    out_dir  = SAMPLE_DIR / run_name

    if REUSE_SAMPLES and (out_dir / "particles.pt").exists():
        print(f"    [sample] REUSE  {run_name}")
        return run_name

    print(f"    [sample] RUN    {run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    energy = ENERGY_MAP[ENERGY](dim=dim)
    gen    = SampleGenerator(energy=energy, betaM=beta_m, mcmc_method="ULA",
                             step_size=MCMC_STEP_SIZE, device=device)
    t0 = time.perf_counter()
    final_x, traces, step_rates = gen.sample(
        MCMC_N_PARTICLES, MCMC_N_STEPS, burn_in=MCMC_BURN_IN,
        trace_every=MCMC_TRACE_EVERY, progress=True,
    )
    wall = time.perf_counter() - t0

    final_x_cpu = final_x.cpu()
    with torch.no_grad():
        e = energy.energy(final_x_cpu).float()
    qs = torch.quantile(e, torch.tensor([0.01, 0.05, 0.25, 0.75, 0.95]))
    acc = torch.tensor(step_rates)

    summary = {
        "status": "completed", "seed": seed, "energy": ENERGY,
        "dim": dim, "beta_m": beta_m, "sampler": "ULA",
        "n_particles": MCMC_N_PARTICLES, "n_steps": MCMC_N_STEPS,
        "burn_in": MCMC_BURN_IN, "trace_every": MCMC_TRACE_EVERY,
        "step_size": MCMC_STEP_SIZE, "wall_clock_seconds": round(wall, 2),
        "mean_energy":   round(e.mean().item(), 4),
        "median_energy": round(e.median().item(), 4),
        "min_energy":    round(e.min().item(), 4),
        "std_energy":    round(e.std().item(), 4),
        "energy_quantiles": {
            "q01": round(qs[0].item(), 4), "q05": round(qs[1].item(), 4),
            "q25": round(qs[2].item(), 4), "q75": round(qs[3].item(), 4),
            "q95": round(qs[4].item(), 4),
        },
        "acceptance_rate": {"mean": round(acc.mean().item(), 4)},
        "traces": traces,
    }
    torch.save(final_x_cpu, out_dir / "particles.pt")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    cfg = {
        "job":     {"seed": seed, "device": "auto", "dtype": "float32"},
        "target":  {"energy": ENERGY, "dim": dim, "beta_m": beta_m},
        "sampler": {"method": "ULA", "step_size": MCMC_STEP_SIZE,
                    "n_particles": MCMC_N_PARTICLES, "n_steps": MCMC_N_STEPS,
                    "burn_in": MCMC_BURN_IN, "trace_every": MCMC_TRACE_EVERY},
        "output":  {"run_name": run_name},
    }
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, sort_keys=False)
    print(f"           done in {wall:.1f}s")
    return run_name


def _ensure_model(sample_run: str, seed: int, device: torch.device) -> str:
    model_run = _model_run_name(sample_run, seed)
    out_dir   = MODEL_DIR / model_run

    if REUSE_MODELS and (out_dir / "ema_model.pt").exists():
        print(f"    [train]  REUSE  {model_run}")
        return model_run

    print(f"    [train]  RUN    {model_run}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    x_data   = torch.load(SAMPLE_DIR / sample_run / "particles.pt",
                          map_location="cpu", weights_only=True)
    schedule = VPSchedule(beta_min=0.1, beta_max=20.0)
    model    = MLPScore(dim=x_data.shape[-1], hidden_dims=tuple(MODEL_HIDDEN_DIMS),
                        time_embed_dim=MODEL_TIME_EMBED_DIM, activation="silu",
                        predict_score=False)
    cfg_tr = TrainingConfig(
        n_steps=TRAIN_N_STEPS, batch_size=BATCH_SIZE, lr=LR, t_eps=T_EPS,
        grad_clip=GRAD_CLIP, ema_decay=EMA_DECAY, log_every=LOG_EVERY,
        loss_type=LOSS_TYPE, seed=seed,
    )
    t0 = time.perf_counter()
    ema_model, loss_history = train_score_model(model, schedule, x_data, cfg_tr, device)
    wall = time.perf_counter() - t0

    torch.save(ema_model.state_dict(), out_dir / "ema_model.pt")
    cfg_yaml = {
        "job":      {"seed": seed, "device": "auto", "dtype": "float32"},
        "samples":  {"run_name": sample_run},
        "schedule": {"type": "vp", "beta_min": 0.1, "beta_max": 20.0},
        "model":    {"hidden_dims": MODEL_HIDDEN_DIMS, "time_embed_dim": MODEL_TIME_EMBED_DIM,
                     "activation": "silu", "predict_score": False},
        "training": {"n_steps": TRAIN_N_STEPS, "batch_size": BATCH_SIZE, "lr": LR,
                     "t_eps": T_EPS, "grad_clip": GRAD_CLIP, "ema_decay": EMA_DECAY,
                     "log_every": LOG_EVERY, "loss_type": LOSS_TYPE},
        "output":   {"run_name": model_run},
    }
    summary = {
        "status": "completed", "run_name": model_run, "sample_run_name": sample_run,
        "n_training_samples": x_data.shape[0], "dim": x_data.shape[-1],
        "n_params": sum(p.numel() for p in ema_model.parameters()),
        "wall_clock_seconds": round(wall, 2),
        "final_loss": round(loss_history[-1], 6) if loss_history else None,
        "loss_history": [round(l, 6) for l in loss_history],
    }
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg_yaml, f, sort_keys=False)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"           done in {wall:.1f}s  final_loss={summary['final_loss']}")
    return model_run


def _ensure_model_samples(model_run: str, seed: int, device: torch.device) -> None:
    out_dir = MODEL_SAMPLE_DIR / model_run

    if REUSE_MODEL_SAMPLES and (out_dir / "samples.pt").exists():
        print(f"    [model samples] REUSE  {model_run}")
        return

    rsde, energy, dim, beta_m, sample_run = _load_model(model_run, device)
    print(f"    [model samples] RUN    {model_run}")
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    x = DiffusionModelSampler(
        rsde, n_steps=MODEL_SAMPLE_N_STEPS,
        t_start=MODEL_SAMPLE_T_START, t_end=MODEL_SAMPLE_T_END,
    ).sample(MCMC_N_PARTICLES, device, show_progress=True)
    wall = round(time.perf_counter() - t0, 2)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(x.cpu(), out_dir / "samples.pt")
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"model_run": model_run, "n_samples": MCMC_N_PARTICLES,
                   "n_steps": MODEL_SAMPLE_N_STEPS, "seed": seed,
                   "wall_clock_seconds": wall}, f, indent=2)
    print(f"           done in {wall:.1f}s")


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

    mc = cfg["model"]
    model = MLPScore(dim=dim, hidden_dims=tuple(mc["hidden_dims"]),
                     time_embed_dim=mc.get("time_embed_dim", 64),
                     activation=mc.get("activation", "silu"),
                     predict_score=mc.get("predict_score", False))
    state = torch.load(run_dir / "ema_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    sc = cfg["schedule"]
    schedule = VPSchedule(beta_min=sc.get("beta_min", 0.1), beta_max=sc.get("beta_max", 20.0))
    return ReverseSDE(model, schedule), energy, dim, beta_m, sample_run


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _energy_stats(x: torch.Tensor, energy_fn) -> dict:
    with torch.no_grad():
        e = energy_fn(x.cpu()).float()
    qs = torch.quantile(e, torch.tensor([0.01, 0.05, 0.10, 0.50, 0.90, 0.95]))
    return {
        "mean_energy":   round(e.mean().item(),   4),
        "min_energy":    round(e.min().item(),     4),
        "median_energy": round(e.median().item(), 4),
        "std_energy":    round(e.std().item(),     4),
        "q01": round(qs[0].item(), 4),
        "q05": round(qs[1].item(), 4),
        "q10": round(qs[2].item(), 4),
        "q90": round(qs[4].item(), 4),
        "q95": round(qs[5].item(), 4),
    }


def _diag_stats(diag: SMCDiagnostics) -> dict:
    ess  = diag.ess_ratios
    stds = diag.log_weight_stds
    return {
        "final_ess":     round(ess[-1],  4) if ess else None,
        "min_ess":       round(min(ess), 4) if ess else None,
        "mean_ess":      round(sum(ess) / len(ess), 4) if ess else None,
        "max_logw_var":  round(max(s ** 2 for s in stds), 4) if stds else None,
        "n_resamples":   diag.n_resamples,
        "ess_trajectory": [round(e, 4) for e in ess],
    }


def _load_particles(path: Path, device=None) -> torch.Tensor:
    """Load particles.pt handling both new [N,D] and legacy [N,S,D] formats,
    and both old/new PyTorch pickle conventions."""
    kwargs: dict = {"map_location": device or "cpu"}
    try:
        x = torch.load(path, weights_only=True, **kwargs)
    except Exception:
        x = torch.load(path, weights_only=False, **kwargs)
    if x.dim() == 3:
        x = x[:, -1, :]  # legacy format: take final snapshot
    return x


def _subsample(x: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
    x = x.to(device)
    return x[torch.randperm(x.shape[0], device=device)[:n]] if x.shape[0] > n else x


# ---------------------------------------------------------------------------
# Experiment B: delta diagnostic
# ---------------------------------------------------------------------------

def _delta_diagnostic(dim: int, beta_m: float, seed: int) -> dict:
    """Quantile energy discrepancy Δ_E between model and MCMC samples.

    Δ_E = (|q10_model - q10_mcmc| + |q50_model - q50_mcmc| + |q90_model - q90_mcmc|)
          / (1 + |q50_mcmc|)

    A single scalar: small = model learned the distribution; large = training failed.
    """
    sample_run = _sample_run_name(dim, beta_m, seed)
    model_run  = _model_run_name(sample_run, seed)
    mcmc_path  = SAMPLE_DIR / sample_run / "particles.pt"
    model_path = MODEL_SAMPLE_DIR / model_run / "samples.pt"

    if not mcmc_path.exists() or not model_path.exists():
        return {}

    energy_fn = ENERGY_MAP[ENERGY](dim=dim).energy
    x_mcmc  = _load_particles(mcmc_path)
    x_model = _load_particles(model_path)

    with torch.no_grad():
        e_mcmc  = energy_fn(x_mcmc).float()
        e_model = energy_fn(x_model).float()

    qs_mcmc  = torch.quantile(e_mcmc,  torch.tensor([0.10, 0.50, 0.90]))
    qs_model = torch.quantile(e_model, torch.tensor([0.10, 0.50, 0.90]))
    q10_m, q50_m, q90_m = qs_mcmc.tolist()
    q10_d, q50_d, q90_d = qs_model.tolist()

    delta_e = (abs(q10_d - q10_m) + abs(q50_d - q50_m) + abs(q90_d - q90_m)) / (1 + abs(q50_m))

    return {
        "delta_e_quantile": round(delta_e, 4),
        "delta_median":     round(abs(q50_d - q50_m), 4),
        "q10_mcmc":  round(q10_m, 4), "q10_model": round(q10_d, 4),
        "q50_mcmc":  round(q50_m, 4), "q50_model": round(q50_d, 4),
        "q90_mcmc":  round(q90_m, 4), "q90_model": round(q90_d, 4),
    }


# ---------------------------------------------------------------------------
# Experiment A: one (β_M, β_H) cell
# ---------------------------------------------------------------------------

def _run_cell(
    model_run: str, sample_run: str,
    beta_m: float, beta_h: float,
    seed: int, device: torch.device,
) -> dict:
    rsde, energy, dim, _, _ = _load_model(model_run, device)
    energy_fn = energy.energy

    mcmc_pts  = _load_particles(SAMPLE_DIR / sample_run / "particles.pt", device)
    model_pts = _load_particles(MODEL_SAMPLE_DIR / model_run / "samples.pt", device)
    x0_mcmc  = _subsample(mcmc_pts,  N_PARTICLES, device)
    x0_model = _subsample(model_pts, N_PARTICLES, device)

    beta_ladder = torch.linspace(beta_m, beta_h, N_SMC_STEPS + 1, device=device)

    # Direct model (no annealing) — same samples regardless of β_H
    direct = _energy_stats(x0_model, energy_fn)

    # ULA SMC
    torch.manual_seed(seed)
    t0   = time.perf_counter()
    ula_p = ULAProposal(energy, step_size=ULA_STEP_SIZE, n_steps=N_ULA_STEPS)
    ula_cloud, ula_diag = SMCSampler(
        ula_p.mutation_kernel, ula_p.weight_update, energy,
        ess_threshold=ESS_THRESHOLD,
    ).run(ParticleCloud(x0_mcmc, torch.zeros(N_PARTICLES, device=device)),
          beta_ladder, show_progress=False)
    ula_wall = time.perf_counter() - t0
    ula = {
        **_energy_stats(ula_cloud.x, energy_fn),
        **_diag_stats(ula_diag),
        "wall_seconds":   round(ula_wall, 2),
        "n_energy_evals": N_PARTICLES * N_SMC_STEPS * N_ULA_STEPS,
        "n_grad_evals":   N_PARTICLES * N_SMC_STEPS * N_ULA_STEPS,
    }

    # Diffusion SMC
    torch.manual_seed(seed)
    t0   = time.perf_counter()
    diff_p = DiffusionAnnealingProposal(
        rsde, energy, beta_train=beta_m,
        n_diffusion_steps=N_DIFFUSION_STEPS,
        t_start=DIFF_T_START, t_end=DIFF_T_END,
        score_scaling=DIFF_SCORE_SCALING,
    )
    diff_cloud, diff_diag = SMCSampler(
        diff_p.mutation_kernel, diff_p.weight_update, energy,
        ess_threshold=ESS_THRESHOLD,
    ).run(ParticleCloud(x0_model, torch.zeros(N_PARTICLES, device=device)),
          beta_ladder, show_progress=False)
    diff_wall = time.perf_counter() - t0
    diff = {
        **_energy_stats(diff_cloud.x, energy_fn),
        **_diag_stats(diff_diag),
        "wall_seconds": round(diff_wall, 2),
        "n_nn_evals":   N_PARTICLES * N_SMC_STEPS * N_DIFFUSION_STEPS,
    }

    return {"direct_model": direct, "ula_smc": ula, "diff_smc": diff}


# ---------------------------------------------------------------------------
# Per-seed run
# ---------------------------------------------------------------------------

def run_seed(
    seed: int, device: torch.device,
    dims: list[int], beta_ms: list[float], beta_hs: list[float],
) -> None:
    out_path = _exp_dir() / f"results_seed{seed}.json"

    # Load existing results for reuse
    done: dict[tuple, dict] = {}   # (dim, beta_m, beta_h) -> cell result
    delta_done: dict[tuple, dict] = {}  # (dim, beta_m) -> delta
    if REUSE_ANNEALING and out_path.exists():
        existing = json.loads(out_path.read_text())
        for r in existing.get("runs", []):
            key = (r["dim"], r["beta_m"])
            delta_done[key] = r.get("delta", {})
            for cell in r.get("cells", []):
                done[(r["dim"], r["beta_m"], cell["beta_h"])] = cell

    runs = []

    for dim in dims:
        print(f"\n  dim={dim}  seed={seed}")
        for beta_m in beta_ms:
            print(f"  β_M={beta_m}")

            # Ensure pipeline artefacts exist
            sample_run = _ensure_samples(dim, beta_m, seed, device)
            model_run  = _ensure_model(sample_run, seed, device)
            _ensure_model_samples(model_run, seed, device)

            # Experiment B: delta diagnostic (once per β_M)
            dk = (dim, beta_m)
            if dk in delta_done and delta_done[dk]:
                delta = delta_done[dk]
            else:
                delta = _delta_diagnostic(dim, beta_m, seed)

            cells = []
            for beta_h in sorted(beta_hs):
                if beta_h < beta_m:
                    continue
                ck = (dim, beta_m, beta_h)
                if ck in done:
                    print(f"    β_H={beta_h}  REUSE")
                    cells.append(done[ck])
                    continue

                print(f"    β_H={beta_h}  running...")
                cell_result = _run_cell(model_run, sample_run, beta_m, beta_h, seed, device)
                cell_result["beta_h"] = beta_h
                cells.append(cell_result)

            runs.append({"dim": dim, "beta_m": beta_m, "delta": delta, "cells": cells})

            # Save after each β_M so progress is not lost
            _exp_dir().mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({"seed": seed, "runs": runs}, indent=2))

    print(f"\n  Seed {seed} complete → {out_path}")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _agg_scalar(vals: list[float]) -> dict:
    vals = [v for v in vals if v is not None and v == v]  # drop None/nan
    if not vals:
        return {"mean": None, "std": None}
    return {
        "mean": round(sum(vals) / len(vals), 4),
        "std":  round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
    }


def _agg_method(dicts: list[dict], keys: list[str]) -> dict:
    agg = {}
    for k in keys:
        vals = [d[k] for d in dicts if k in d and d[k] is not None]
        agg[k] = _agg_scalar(vals)
    return agg


def aggregate_results() -> None:
    seed_files = sorted(_exp_dir().glob("results_seed*.json"))
    if not seed_files:
        print("  No seed files found.")
        return

    # Collect all cells: (dim, beta_m, beta_h) → list of per-seed cell dicts
    cells: dict[tuple, list[dict]] = defaultdict(list)
    deltas: dict[tuple, list[dict]] = defaultdict(list)

    for sf in seed_files:
        data = json.loads(sf.read_text())
        for r in data.get("runs", []):
            dk = (r["dim"], r["beta_m"])
            if r.get("delta"):
                deltas[dk].append(r["delta"])
            for cell in r.get("cells", []):
                ck = (r["dim"], r["beta_m"], cell["beta_h"])
                cells[ck].append(cell)

    STAT_KEYS = ["mean_energy", "min_energy", "median_energy", "std_energy",
                 "q01", "q05", "q10", "q90", "q95"]
    DIAG_KEYS = ["final_ess", "min_ess", "mean_ess", "n_resamples", "max_logw_var"]
    DELTA_KEYS = ["delta_e_quantile", "delta_median",
                  "q10_mcmc", "q10_model", "q50_mcmc", "q50_model",
                  "q90_mcmc", "q90_model"]

    aggregated = []
    for (dim, beta_m, beta_h), cell_list in sorted(cells.items()):
        dk = (dim, beta_m)
        agg_delta = _agg_method(deltas.get(dk, [{}]), DELTA_KEYS)
        agg_entry = {
            "dim": dim, "beta_m": beta_m, "beta_h": beta_h,
            "n_seeds": len(cell_list),
            "delta": agg_delta,
            "direct_model": _agg_method([c["direct_model"] for c in cell_list], STAT_KEYS),
            "ula_smc":  _agg_method([c["ula_smc"]  for c in cell_list], STAT_KEYS + DIAG_KEYS),
            "diff_smc": _agg_method([c["diff_smc"] for c in cell_list], STAT_KEYS + DIAG_KEYS),
        }
        aggregated.append(agg_entry)

    out = _exp_dir() / "aggregate.json"
    out.write_text(json.dumps({"n_seeds": len(seed_files), "cells": aggregated}, indent=2))
    print(f"  Aggregate saved → {out}  ({len(aggregated)} cells, {len(seed_files)} seeds)")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_results() -> None:
    agg_path = _exp_dir() / "aggregate.json"
    if not agg_path.exists():
        print("  No aggregate.json — run --aggregate first.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data  = json.loads(agg_path.read_text())
    cells = data["cells"]

    plots_dir = _exp_dir() / "plots"
    plots_dir.mkdir(exist_ok=True)

    energy_obj  = ENERGY_MAP[ENERGY](dim=2)
    e_star_map  = {}  # populated per dim below

    dims    = sorted(set(c["dim"]    for c in cells))
    beta_ms = sorted(set(c["beta_m"] for c in cells))
    beta_hs = sorted(set(c["beta_h"] for c in cells))

    def _get(cell, method, key):
        return cell.get(method, {}).get(key, {}).get("mean")

    # ── Triangular heatmaps ────────────────────────────────────────────────
    metrics = [
        ("min_energy",   "Min energy"),
        ("q01",          "q01 energy"),
        ("mean_energy",  "Mean energy"),
        ("min_ess",      "Min ESS"),
    ]

    for dim in dims:
        e_star = getattr(ENERGY_MAP[ENERGY](dim=dim), "global_minimum_energy", None)
        e_star_map[dim] = e_star

        for method_key, method_label in [("ula_smc", "ULA SMC"), ("diff_smc", "Diffusion SMC"),
                                          ("direct_model", "Direct model")]:
            n_bm = len(beta_ms)
            n_bh = len(beta_hs)
            fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.5))

            for ax, (stat_key, stat_label) in zip(axes, metrics):
                if method_key == "direct_model" and stat_key in ("min_ess", "max_logw_var"):
                    ax.axis("off")
                    continue

                grid = np.full((n_bh, n_bm), np.nan)
                for cell in cells:
                    if cell["dim"] != dim:
                        continue
                    bm_i = beta_ms.index(cell["beta_m"])
                    bh_i = beta_hs.index(cell["beta_h"])
                    v = _get(cell, method_key, stat_key)
                    if v is not None:
                        grid[bh_i, bm_i] = v

                im = ax.imshow(grid, aspect="auto", origin="lower",
                               cmap="viridis_r" if "energy" in stat_key else "viridis")
                ax.set_xticks(range(n_bm))
                ax.set_xticklabels([_beta_str(b) for b in beta_ms], fontsize=7)
                ax.set_yticks(range(n_bh))
                ax.set_yticklabels([_beta_str(b) for b in beta_hs], fontsize=7)
                ax.set_xlabel(r"$\beta_M$", fontsize=8)
                ax.set_ylabel(r"$\beta_H$", fontsize=8)
                ax.set_title(stat_label, fontsize=8)

                # Annotate cells
                for bh_i, bm_i in np.ndindex(grid.shape):
                    v = grid[bh_i, bm_i]
                    if not np.isnan(v):
                        ax.text(bm_i, bh_i, f"{v:.2f}", ha="center", va="center",
                                fontsize=5, color="white")

                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            e_str = f" (E*={e_star:.3g})" if e_star is not None else ""
            fig.suptitle(f"{ENERGY}  d={dim}  {method_label}{e_str}", fontsize=9)
            fig.tight_layout()
            slug = method_key.replace("_", "")
            out  = plots_dir / f"heatmap_d{dim}_{slug}.svg"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {out.name}")

    # ── Delta diagnostic: Δ_E vs β_M per dim ──────────────────────────────
    fig, axes = plt.subplots(1, len(dims), figsize=(4 * len(dims), 3.5), squeeze=False)
    for ax, dim in zip(axes[0], dims):
        bms, vals, errs = [], [], []
        for beta_m in beta_ms:
            matches = [c for c in cells if c["dim"] == dim and c["beta_m"] == beta_m]
            if not matches:
                continue
            v = matches[0].get("delta", {}).get("delta_e_quantile", {})
            if isinstance(v, dict) and v.get("mean") is not None:
                bms.append(beta_m)
                vals.append(v["mean"])
                errs.append(v.get("std", 0.0))
        if bms:
            ax.errorbar(bms, vals, yerr=errs, fmt="o-", color="steelblue", capsize=3)
        ax.axhline(0.15, color="red", lw=0.8, linestyle="--", alpha=0.5, label="τ=0.15")
        ax.set_xlabel(r"$\beta_M$")
        ax.set_ylabel(r"$\Delta_E$ (quantile discrepancy)")
        ax.set_title(f"d={dim}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{ENERGY}  Experiment B: delta diagnostic", fontsize=10)
    fig.tight_layout()
    out = plots_dir / "delta_diagnostic.svg"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")

    # ── G_gap improvement vs β_M, sliced by β_H ───────────────────────────
    eps = 1e-8
    for beta_h in beta_hs:
        fig, axes = plt.subplots(1, len(dims), figsize=(4 * len(dims), 3.5), squeeze=False)
        for ax, dim in zip(axes[0], dims):
            e_star = e_star_map.get(dim)
            for method_key, label, color in [
                ("ula_smc",  "ULA SMC",       "steelblue"),
                ("diff_smc", "Diffusion SMC", "darkorange"),
            ]:
                bms, vals = [], []
                for beta_m in beta_ms:
                    if beta_h < beta_m:
                        continue
                    match = next((c for c in cells
                                  if c["dim"] == dim and c["beta_m"] == beta_m
                                  and c["beta_h"] == beta_h), None)
                    if match is None:
                        continue
                    baseline_key = "ula_smc" if method_key == "ula_smc" else "direct_model"
                    # use MCMC stats as ULA baseline; model stats for diffusion
                    # here both come from cells so we use direct_model as proxy for model
                    e_init  = _get(match, baseline_key, "mean_energy")
                    e_final = _get(match, method_key,   "mean_energy")
                    if e_init is None or e_final is None:
                        continue
                    denom = (e_init - e_star + eps) if e_star is not None else (abs(e_init) + eps)
                    bms.append(beta_m)
                    vals.append((e_init - e_final) / denom)
                if bms:
                    ax.plot(bms, vals, "o-", color=color, label=label, lw=1.5)
            ax.axhline(0, color="k", lw=0.8, linestyle="--", alpha=0.4)
            ax.set_xlabel(r"$\beta_M$")
            ax.set_ylabel(r"$G_\mathrm{gap}$")
            ax.set_title(f"d={dim}", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"{ENERGY}  G_gap  β_H={beta_h}", fontsize=10)
        fig.tight_layout()
        out = plots_dir / f"ggap_betaH{_beta_str(beta_h)}.svg"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out.name}")

    print(f"\nAll plots saved to {plots_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global ENERGY

    parser = argparse.ArgumentParser(description="Temperature-transfer surface (Experiments A+B).")
    parser.add_argument("--energy",    choices=list(ENERGY_MAP), default=None)
    parser.add_argument("--seed",      type=int, default=None,
                        help="Run a single seed (use --aggregate when all done)")
    parser.add_argument("--pilot",     action="store_true",
                        help=f"Small grid: dims={PILOT_DIMS} β_M={PILOT_BETA_MS} "
                             f"β_H={PILOT_BETA_HS} seeds={PILOT_SEEDS}")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.energy:
        ENERGY = args.energy

    if args.plot_only:
        plot_results()
        return

    if args.aggregate:
        print("Aggregating...")
        aggregate_results()
        print("Plotting...")
        plot_results()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)

    if args.pilot:
        dims, beta_ms, beta_hs = PILOT_DIMS, PILOT_BETA_MS, PILOT_BETA_HS
        seeds = PILOT_SEEDS
        print(f"=== PILOT  {ENERGY}  dims={dims}  β_M={beta_ms}  β_H={beta_hs} ===")
    else:
        dims, beta_ms, beta_hs = DIMS, BETA_MS, BETA_HS
        seeds = [args.seed] if args.seed is not None else SEEDS
        print(f"=== {ENERGY}  dims={dims}  seeds={seeds} ===")

    _exp_dir().mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        print(f"\n{'='*60}\n  SEED {seed}\n{'='*60}")
        run_seed(seed, device, dims, beta_ms, beta_hs)

    if args.seed is None:
        print("\nAggregating...")
        aggregate_results()
        print("Plotting...")
        plot_results()
    else:
        print(f"\nSeed {args.seed} done. Run --aggregate when all seeds complete.")


if __name__ == "__main__":
    main()
