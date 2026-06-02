#!/usr/bin/env python
"""Cost-quality experiment: amortised diffusion vs direct ULA annealing.

Core question: For a desired quality level E*, is amortised diffusion ever cheaper
than running direct annealing from scratch?

Methods compared:
  1. Direct fixed-temp ULA  — start from N(0,I), run ULA at fixed β_H
  2. Direct annealed ULA SMC — start from N(0,I), anneal 0→β_H via SMC (true thesis baseline)
  3. Amortised diffusion + optional local ULA polish

Oracle cost = energy gradient evaluations (∇E evals):
  - MCMC setup:           N_train × MCMC_N_STEPS  (counterfactual for fair N_train comparison)
  - Diffusion inference:  0 oracle evals (NN evals only)
  - Local ULA polish:     N_samples × local_steps
  - Direct ULA:           N_particles × n_steps

Usage:
    # Per-seed on server (run 3 in parallel via tmux)
    uv run python scripts/experiments/cost_quality.py --seed 0 --no-plot
    uv run python scripts/experiments/cost_quality.py --seed 1 --no-plot
    uv run python scripts/experiments/cost_quality.py --seed 2 --no-plot

    # Aggregate + plot locally
    uv run python scripts/experiments/cost_quality.py --aggregate --no-plot
    uv run python scripts/experiments/cost_quality.py --plot-only
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
    ParticleCloud, SMCDiagnostics, SMCSampler, ULAProposal,
)

ROOT             = Path(__file__).parent.parent.parent
SAMPLE_DIR       = ROOT / "data" / "samples"
MODEL_DIR        = ROOT / "data" / "models"
MODEL_SAMPLE_DIR = ROOT / "data" / "model_samples"
EXP_BASE         = ROOT / "data" / "experiments" / "cost_quality"

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
DIMS    = [5, 10, 20]
BETA_MS = [1.0, 5.0, 20.0]
BETA_HS = [20.0, 50.0]
SEEDS   = [0, 1, 2]

# Training data sizes — subsample from existing 8192-particle MCMC runs
N_TRAIN_SIZES = [2048, 8192]

# Inference sample sizes — subsample from a single large batch
N_SAMPLE_SIZES = [512, 2048, 8192]
MAX_N_SAMPLES  = 8192   # generate once, subsample for the N_samples sweep

# Local ULA polish steps applied after diffusion sampling
LOCAL_STEPS_LIST = [0, 10]

# Direct fixed-temp ULA baseline
N_DIRECT_PARTICLES   = 4096
DIRECT_ULA_STEP_SIZE = 1e-3
# Eval budget checkpoints (grad evals = N_particles × steps)
DIRECT_FIXED_BUDGETS = [int(1e5), int(3e5), int(1e6), int(3e6), int(1e7)]

# Direct annealed ULA SMC baseline — multiple budget configs
# oracle cost = N_DIRECT_PARTICLES × (n_smc × n_ula  +  n_smc)  [ULA grads + weight updates]
DIRECT_SMC_CONFIGS = [
    {"n_smc":  16, "n_ula":  5},   # ~344K  oracle evals
    {"n_smc":  32, "n_ula": 10},   # ~1.44M oracle evals
    {"n_smc":  64, "n_ula": 20},   # ~5.50M oracle evals
    {"n_smc": 128, "n_ula": 40},   # ~21.0M oracle evals  ← matches N_train=2048 setup cost
    {"n_smc": 256, "n_ula": 80},   # ~83.9M oracle evals  ← matches N_train=8192 setup cost
]

# Model architecture (must match energy_betaM_experiment.py / temp_transfer.py)
MODEL_HIDDEN_DIMS    = [256, 256, 256]
MODEL_TIME_EMBED_DIM = 64
LOSS_TYPE            = "eps"

# MCMC generation (existing runs; used only for oracle cost accounting)
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

# Model sampling (diffusion inference)
MODEL_SAMPLE_N_STEPS = 500
MODEL_SAMPLE_T_START = 1.0
MODEL_SAMPLE_T_END   = 1e-3

# Local ULA polish
LOCAL_ULA_STEP_SIZE = 1e-3

# Reuse flags
REUSE_SAMPLES       = True
REUSE_MODELS_NTRAIN = True
REUSE_LARGE_SAMPLES = True
REUSE_DIRECT_ULA    = True
REUSE_AMORTISED     = True
# ===========================================================================


def _exp_dir() -> Path:
    return EXP_BASE / ENERGY


def _beta_str(b: float) -> str:
    return f"{b:g}".replace(".", "p")


def _preset_tag() -> str:
    s = MODEL_HIDDEN_DIMS
    return f"mlp{s[0]}x{len(s)}" if len(set(s)) == 1 else "mlp_" + "_".join(str(x) for x in s)


def _sample_run_name(dim: int, beta_m: float, seed: int) -> str:
    return f"{ENERGY}_d{dim}_beta{_beta_str(beta_m)}_ula_seed{seed}"


def _model_run_name_ntrain(sample_run: str, n_train: int, seed: int) -> str:
    if n_train == MCMC_N_PARTICLES:
        # Reuse standard naming matching temp_transfer / energy_betaM_experiment
        return f"{sample_run}_{_preset_tag()}_{LOSS_TYPE}_seed{seed}"
    return f"{sample_run}_ntrain{n_train}_{_preset_tag()}_{LOSS_TYPE}_seed{seed}"


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _ensure_samples(dim: int, beta_m: float, seed: int, device: torch.device) -> str:
    """Ensure MCMC particles exist. Reuses any existing run from prior experiments."""
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
    acc = torch.tensor(step_rates)

    summary = {
        "status": "completed", "seed": seed, "energy": ENERGY,
        "dim": dim, "beta_m": beta_m, "sampler": "ULA",
        "n_particles": MCMC_N_PARTICLES, "n_steps": MCMC_N_STEPS,
        "burn_in": MCMC_BURN_IN, "trace_every": MCMC_TRACE_EVERY,
        "step_size": MCMC_STEP_SIZE, "wall_clock_seconds": round(wall, 2),
        "mean_energy": round(e.mean().item(), 4),
        "min_energy":  round(e.min().item(), 4),
        "acceptance_rate": {"mean": round(acc.mean().item(), 4)},
        "traces": traces,
    }
    torch.save(final_x_cpu, out_dir / "particles.pt")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"           done in {wall:.1f}s")
    return run_name


def _ensure_model_ntrain(
    sample_run: str, n_train: int, seed: int, device: torch.device
) -> tuple[str, float]:
    """Train model on n_train particles. Returns (model_run, train_wall_seconds)."""
    model_run = _model_run_name_ntrain(sample_run, n_train, seed)
    out_dir   = MODEL_DIR / model_run

    if REUSE_MODELS_NTRAIN and (out_dir / "ema_model.pt").exists():
        print(f"    [train]  REUSE  {model_run}")
        train_wall = 0.0
        if (out_dir / "summary.json").exists():
            train_wall = json.loads((out_dir / "summary.json").read_text()).get(
                "wall_clock_seconds", 0.0)
        return model_run, train_wall

    print(f"    [train]  RUN    {model_run}  (n_train={n_train})")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    x_all = torch.load(SAMPLE_DIR / sample_run / "particles.pt",
                       map_location="cpu", weights_only=True)
    dim   = x_all.shape[-1]

    if n_train < x_all.shape[0]:
        g   = torch.Generator().manual_seed(seed)
        idx = torch.randperm(x_all.shape[0], generator=g)[:n_train]
        x_data = x_all[idx]
    else:
        x_data = x_all

    schedule = VPSchedule(beta_min=0.1, beta_max=20.0)
    model    = MLPScore(dim=dim, hidden_dims=tuple(MODEL_HIDDEN_DIMS),
                        time_embed_dim=MODEL_TIME_EMBED_DIM, activation="silu",
                        predict_score=False)
    cfg_tr = TrainingConfig(
        n_steps=TRAIN_N_STEPS, batch_size=BATCH_SIZE, lr=LR, t_eps=T_EPS,
        grad_clip=GRAD_CLIP, ema_decay=EMA_DECAY, log_every=LOG_EVERY,
        loss_type=LOSS_TYPE, seed=seed,
    )
    t0 = time.perf_counter()
    ema_model, loss_history = train_score_model(model, schedule, x_data, cfg_tr, device)
    wall = round(time.perf_counter() - t0, 2)

    torch.save(ema_model.state_dict(), out_dir / "ema_model.pt")
    summary = {
        "status": "completed", "run_name": model_run, "sample_run_name": sample_run,
        "n_training_samples": int(x_data.shape[0]), "n_train": n_train, "dim": int(dim),
        "n_params": sum(p.numel() for p in ema_model.parameters()),
        "wall_clock_seconds": wall,
        "final_loss": round(loss_history[-1], 6) if loss_history else None,
    }
    cfg_yaml = {
        "job":      {"seed": seed, "device": "auto", "dtype": "float32"},
        "samples":  {"run_name": sample_run},
        "n_train":  n_train,
        "schedule": {"type": "vp", "beta_min": 0.1, "beta_max": 20.0},
        "model":    {"hidden_dims": MODEL_HIDDEN_DIMS, "time_embed_dim": MODEL_TIME_EMBED_DIM,
                     "activation": "silu", "predict_score": False},
        "training": {"n_steps": TRAIN_N_STEPS, "batch_size": BATCH_SIZE, "lr": LR,
                     "t_eps": T_EPS, "grad_clip": GRAD_CLIP, "ema_decay": EMA_DECAY,
                     "log_every": LOG_EVERY, "loss_type": LOSS_TYPE},
        "output":   {"run_name": model_run},
    }
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg_yaml, f, sort_keys=False)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"           done in {wall:.1f}s  final_loss={summary['final_loss']}")
    return model_run, wall


def _load_model_run(model_run: str, device: torch.device):
    """Returns (reverse_sde, energy_obj, dim, beta_m)."""
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
    return ReverseSDE(model, schedule), energy, dim, beta_m


def _ensure_large_sample_batch(model_run: str, seed: int, device: torch.device) -> float:
    """Generate MAX_N_SAMPLES from model; save as 'samples_large.pt'. Returns wall_seconds."""
    out_dir   = MODEL_SAMPLE_DIR / model_run
    save_path = out_dir / "samples_large.pt"

    if REUSE_LARGE_SAMPLES and save_path.exists():
        print(f"    [inf]    REUSE  {model_run}")
        meta_path = out_dir / "summary_large.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text()).get("wall_clock_seconds", 0.0)
        return 0.0

    rsde, energy, dim, _ = _load_model_run(model_run, device)
    print(f"    [inf]    RUN    {model_run}  (n={MAX_N_SAMPLES})")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    x = DiffusionModelSampler(
        rsde, n_steps=MODEL_SAMPLE_N_STEPS,
        t_start=MODEL_SAMPLE_T_START, t_end=MODEL_SAMPLE_T_END,
    ).sample(MAX_N_SAMPLES, device, show_progress=True)
    wall = round(time.perf_counter() - t0, 2)

    torch.save(x.cpu(), save_path)
    with open(out_dir / "summary_large.json", "w") as f:
        json.dump({"model_run": model_run, "n_samples": MAX_N_SAMPLES,
                   "n_steps": MODEL_SAMPLE_N_STEPS, "seed": seed,
                   "wall_clock_seconds": wall}, f, indent=2)
    print(f"           done in {wall:.1f}s")
    return wall


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _load_tensor(path: Path, device=None) -> torch.Tensor:
    x = torch.load(path, weights_only=True, map_location=device or "cpu")
    if x.dim() == 3:
        x = x[:, -1, :]
    return x


def _subsample(x: torch.Tensor, n: int, seed: int, device: torch.device) -> torch.Tensor:
    x = x.to(device)
    if x.shape[0] <= n:
        return x
    g = torch.Generator(device=device).manual_seed(seed)
    return x[torch.randperm(x.shape[0], device=device, generator=g)[:n]]


def _energy_stats(x: torch.Tensor, energy_fn) -> dict:
    with torch.no_grad():
        e = energy_fn(x.cpu()).float()
    qs = torch.quantile(e, torch.tensor([0.01, 0.05, 0.10, 0.50, 0.90, 0.95]))
    return {
        "best_energy":   round(e.min().item(),    4),
        "mean_energy":   round(e.mean().item(),   4),
        "median_energy": round(e.median().item(), 4),
        "std_energy":    round(e.std().item(),    4),
        "q01": round(qs[0].item(), 4),
        "q05": round(qs[1].item(), 4),
        "q10": round(qs[2].item(), 4),
        "q90": round(qs[4].item(), 4),
        "q95": round(qs[5].item(), 4),
    }


def _apply_local_ula(
    x: torch.Tensor, energy_fn, beta_h: float, n_steps: int, seed: int
) -> tuple[torch.Tensor, int]:
    """ULA polish: n_steps of Langevin targeting π_{β_H}. Returns (x_refined, n_grad_evals)."""
    if n_steps == 0:
        return x, 0
    torch.manual_seed(seed)
    x = x.detach().clone()
    for _ in range(n_steps):
        x = x.requires_grad_(True)
        with torch.enable_grad():
            grad = torch.autograd.grad(energy_fn(x).sum(), x)[0]
        x = (x - LOCAL_ULA_STEP_SIZE * beta_h * grad.detach()
             + (2 * LOCAL_ULA_STEP_SIZE) ** 0.5 * torch.randn_like(x)).detach()
    return x, x.shape[0] * n_steps


# ---------------------------------------------------------------------------
# Direct baselines
# ---------------------------------------------------------------------------

def _run_direct_fixed_ula(
    dim: int, beta_h: float, seed: int, device: torch.device,
) -> list[dict]:
    """Fixed-temperature ULA from N(0,I). Checkpoints at DIRECT_FIXED_BUDGETS."""
    energy    = ENERGY_MAP[ENERGY](dim=dim)
    energy_fn = energy.energy

    # Sort budgets; run until all passed
    budgets_sorted = sorted(DIRECT_FIXED_BUDGETS)
    max_evals      = budgets_sorted[-1]
    max_steps      = max_evals // N_DIRECT_PARTICLES + 1

    torch.manual_seed(seed)
    x     = torch.randn(N_DIRECT_PARTICLES, dim, device=device)
    t0    = time.perf_counter()
    results    = []
    n_evals    = 0
    budget_idx = 0

    for _ in range(max_steps):
        x = x.requires_grad_(True)
        with torch.enable_grad():
            grad = torch.autograd.grad(energy_fn(x).sum(), x)[0]
        x = (x - DIRECT_ULA_STEP_SIZE * beta_h * grad.detach()
             + (2 * DIRECT_ULA_STEP_SIZE) ** 0.5 * torch.randn_like(x)).detach()
        n_evals += N_DIRECT_PARTICLES

        while budget_idx < len(budgets_sorted) and n_evals >= budgets_sorted[budget_idx]:
            results.append({
                "n_grad_evals": budgets_sorted[budget_idx],
                "wall_seconds": round(time.perf_counter() - t0, 3),
                **_energy_stats(x, energy_fn),
            })
            budget_idx += 1

        if budget_idx >= len(budgets_sorted):
            break

    return results


def _run_direct_annealed_smc(
    dim: int, beta_h: float, seed: int, device: torch.device,
) -> list[dict]:
    """Annealed ULA SMC from N(0,I) → β_H. Multiple budget configs."""
    energy    = ENERGY_MAP[ENERGY](dim=dim)
    energy_fn = energy.energy
    results   = []

    for cfg in DIRECT_SMC_CONFIGS:
        n_smc = cfg["n_smc"]
        n_ula = cfg["n_ula"]
        n_grad_evals   = N_DIRECT_PARTICLES * n_smc * n_ula   # ULA mutation grad evals
        n_energy_evals = N_DIRECT_PARTICLES * n_smc           # weight-update energy evals
        oracle_total   = n_grad_evals + n_energy_evals

        torch.manual_seed(seed)
        x0 = torch.randn(N_DIRECT_PARTICLES, dim, device=device)
        # Start ladder from small positive beta to avoid degenerate weights at 0
        beta_ladder = torch.linspace(0.01, beta_h, n_smc + 1, device=device)

        proposal = ULAProposal(energy, step_size=DIRECT_ULA_STEP_SIZE, n_steps=n_ula)
        t0 = time.perf_counter()
        cloud, diag = SMCSampler(
            proposal.mutation_kernel, proposal.weight_update, energy,
            ess_threshold=0.5,
        ).run(
            ParticleCloud(x0, torch.zeros(N_DIRECT_PARTICLES, device=device)),
            beta_ladder, show_progress=False,
        )
        wall = round(time.perf_counter() - t0, 3)

        ess = diag.ess_ratios
        results.append({
            "n_smc":            n_smc,
            "n_ula":            n_ula,
            "n_grad_evals":     n_grad_evals,
            "n_energy_evals":   n_energy_evals,
            "oracle_cost_total": oracle_total,
            "wall_seconds":     wall,
            "final_ess":        round(ess[-1], 4) if ess else None,
            **_energy_stats(cloud.x, energy_fn),
        })

    return results


# ---------------------------------------------------------------------------
# Amortised sweep
# ---------------------------------------------------------------------------

def _run_amortised_cell(
    model_run: str,
    n_train: int,
    beta_h: float,
    seed: int,
    device: torch.device,
    mcmc_wall: float,
    train_wall: float,
    inference_wall: float,
) -> list[dict]:
    """All (n_samples, local_steps) combinations for one (model, beta_h)."""
    _, energy, dim, beta_m = _load_model_run(model_run, device)
    energy_fn = energy.energy

    x_large = _load_tensor(MODEL_SAMPLE_DIR / model_run / "samples_large.pt", device)

    # Counterfactual oracle cost: charge only N_train chains × N_steps
    oracle_setup = n_train * MCMC_N_STEPS

    results = []
    for n_samples in N_SAMPLE_SIZES:
        x_sub = _subsample(x_large, n_samples, seed, device)

        for local_steps in LOCAL_STEPS_LIST:
            t0 = time.perf_counter()
            x_refined, oracle_polish = _apply_local_ula(
                x_sub, energy_fn, beta_h, local_steps, seed
            )
            inf_wall_cell = round(time.perf_counter() - t0, 3)

            oracle_total = oracle_setup + oracle_polish
            total_wall   = round(mcmc_wall + train_wall + inference_wall + inf_wall_cell, 2)

            results.append({
                "n_train":     n_train,
                "n_samples":   n_samples,
                "local_steps": local_steps,
                **_energy_stats(x_refined, energy_fn),
                "oracle_cost_setup_mcmc":    oracle_setup,
                "oracle_cost_polish_ula":    oracle_polish,
                "oracle_cost_total":         oracle_total,
                "wall_seconds_setup_mcmc":   round(mcmc_wall,     2),
                "wall_seconds_setup_train":  round(train_wall,    2),
                "wall_seconds_inference":    round(inference_wall, 2),
                "wall_seconds_polish":       inf_wall_cell,
                "wall_seconds_total":        total_wall,
            })

    return results


# ---------------------------------------------------------------------------
# Per-seed run
# ---------------------------------------------------------------------------

def run_seed(seed: int, device: torch.device) -> None:
    out_path = _exp_dir() / f"results_seed{seed}.json"

    existing: dict = {}
    if REUSE_AMORTISED and out_path.exists():
        existing = json.loads(out_path.read_text())

    amortised_runs: list[dict] = list(existing.get("amortised", []))
    direct_fixed   : dict       = {
        f"{e['dim']}_{_beta_str(e['beta_h'])}": e
        for e in existing.get("direct_ula_fixed", [])
    }
    direct_annealed: dict       = {
        f"{e['dim']}_{_beta_str(e['beta_h'])}": e
        for e in existing.get("direct_ula_annealed", [])
    }

    def _save() -> None:
        _exp_dir().mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "seed": seed,
            "direct_ula_fixed":    list(direct_fixed.values()),
            "direct_ula_annealed": list(direct_annealed.values()),
            "amortised":           amortised_runs,
        }, indent=2))

    # ── Direct baselines (one run per (dim, beta_h), independent of beta_m) ──
    for dim in DIMS:
        for beta_h in BETA_HS:
            key = f"{dim}_{_beta_str(beta_h)}"

            if REUSE_DIRECT_ULA and key in direct_fixed:
                print(f"  [direct fixed]   REUSE  d={dim} β_H={beta_h}")
            else:
                print(f"  [direct fixed]   RUN    d={dim} β_H={beta_h}")
                direct_fixed[key] = {
                    "dim": dim, "beta_h": beta_h,
                    "points": _run_direct_fixed_ula(dim, beta_h, seed, device),
                }
                _save()

            if REUSE_DIRECT_ULA and key in direct_annealed:
                print(f"  [direct anneal]  REUSE  d={dim} β_H={beta_h}")
            else:
                print(f"  [direct anneal]  RUN    d={dim} β_H={beta_h}")
                direct_annealed[key] = {
                    "dim": dim, "beta_h": beta_h,
                    "points": _run_direct_annealed_smc(dim, beta_h, seed, device),
                }
                _save()

    # ── Amortised runs ────────────────────────────────────────────────────────
    done_keys = {(r["dim"], r["beta_m"]) for r in amortised_runs}

    for dim in DIMS:
        for beta_m in BETA_MS:
            if (dim, beta_m) in done_keys:
                print(f"  REUSE  d={dim} β_M={beta_m}")
                continue

            print(f"\n  dim={dim}  β_M={beta_m}")

            sample_run = _ensure_samples(dim, beta_m, seed, device)
            mcmc_summary = json.loads((SAMPLE_DIR / sample_run / "summary.json").read_text())
            mcmc_wall    = mcmc_summary.get("wall_clock_seconds", 0.0)

            bh_cells = []
            for beta_h in BETA_HS:
                cells_this_bh: list[dict] = []
                for n_train in N_TRAIN_SIZES:
                    model_run, train_wall = _ensure_model_ntrain(
                        sample_run, n_train, seed, device
                    )
                    inf_wall = _ensure_large_sample_batch(model_run, seed, device)

                    cells = _run_amortised_cell(
                        model_run, n_train, beta_h, seed, device,
                        mcmc_wall, train_wall, inf_wall,
                    )
                    cells_this_bh.extend(cells)

                bh_cells.append({"beta_h": beta_h, "cells": cells_this_bh})

            amortised_runs.append({"dim": dim, "beta_m": beta_m, "beta_h_cells": bh_cells})
            _save()
            print(f"    → saved (d={dim} β_M={beta_m})")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _agg(vals: list) -> dict:
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"mean": None, "std": None}
    return {
        "mean": round(statistics.mean(vals), 5),
        "std":  round(statistics.stdev(vals), 5) if len(vals) > 1 else 0.0,
    }


def aggregate_results() -> None:
    seed_files = sorted(_exp_dir().glob("results_seed*.json"))
    if not seed_files:
        print("  No seed files found.")
        return

    STAT_KEYS = ["best_energy", "q01", "q05", "mean_energy", "median_energy"]
    COST_KEYS = ["oracle_cost_total", "oracle_cost_setup_mcmc", "oracle_cost_polish_ula",
                 "wall_seconds_total", "wall_seconds_setup_train", "wall_seconds_inference"]

    # Amortised
    am_groups: dict[tuple, list] = defaultdict(list)
    for sf in seed_files:
        data = json.loads(sf.read_text())
        for block in data.get("amortised", []):
            for bh_block in block.get("beta_h_cells", []):
                for cell in bh_block["cells"]:
                    key = (block["dim"], block["beta_m"], bh_block["beta_h"],
                           cell["n_train"], cell["n_samples"], cell["local_steps"])
                    am_groups[key].append(cell)

    agg_am = []
    for (dim, beta_m, beta_h, n_train, n_samples, local_steps), cells in sorted(am_groups.items()):
        entry = {
            "dim": dim, "beta_m": beta_m, "beta_h": beta_h,
            "n_train": n_train, "n_samples": n_samples, "local_steps": local_steps,
            "n_seeds": len(cells),
        }
        for k in STAT_KEYS + COST_KEYS:
            entry[k] = _agg([c.get(k) for c in cells])
        agg_am.append(entry)

    # Direct fixed ULA
    fix_groups: dict[tuple, list] = defaultdict(list)
    for sf in seed_files:
        data = json.loads(sf.read_text())
        for entry in data.get("direct_ula_fixed", []):
            for pt in entry.get("points", []):
                key = (entry["dim"], entry["beta_h"], pt["n_grad_evals"])
                fix_groups[key].append(pt)

    agg_fixed = []
    for (dim, beta_h, n_evals), pts in sorted(fix_groups.items()):
        entry = {"dim": dim, "beta_h": beta_h, "n_grad_evals": n_evals, "n_seeds": len(pts)}
        for k in STAT_KEYS:
            entry[k] = _agg([p.get(k) for p in pts])
        agg_fixed.append(entry)

    # Direct annealed SMC — key by oracle_cost_total (grad evals + weight-update energy evals)
    ann_groups: dict[tuple, list] = defaultdict(list)
    for sf in seed_files:
        data = json.loads(sf.read_text())
        for entry in data.get("direct_ula_annealed", []):
            for pt in entry.get("points", []):
                # Fall back to n_grad_evals for backwards compat with old result files
                cost_key = pt.get("oracle_cost_total", pt["n_grad_evals"])
                key = (entry["dim"], entry["beta_h"], cost_key)
                ann_groups[key].append(pt)

    agg_annealed = []
    for (dim, beta_h, oracle_cost), pts in sorted(ann_groups.items()):
        entry = {
            "dim": dim, "beta_h": beta_h,
            "oracle_cost_total": oracle_cost,
            "n_grad_evals":      pts[0].get("n_grad_evals", oracle_cost),
            "n_energy_evals":    pts[0].get("n_energy_evals", 0),
            "n_seeds": len(pts),
        }
        for k in STAT_KEYS:
            entry[k] = _agg([p.get(k) for p in pts])
        agg_annealed.append(entry)

    out = _exp_dir() / "aggregate.json"
    out.write_text(json.dumps({
        "n_seeds": len(seed_files),
        "amortised":          agg_am,
        "direct_ula_fixed":   agg_fixed,
        "direct_ula_annealed": agg_annealed,
    }, indent=2))
    print(f"  Aggregate saved → {out}  ({len(agg_am)} amortised cells)")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results() -> None:
    agg_path = _exp_dir() / "aggregate.json"
    if not agg_path.exists():
        print("  No aggregate.json — run --aggregate first.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data      = json.loads(agg_path.read_text())
    plots_dir = _exp_dir() / "plots"
    plots_dir.mkdir(exist_ok=True)

    dims    = sorted(set(c["dim"]    for c in data["amortised"]))
    beta_hs = sorted(set(c["beta_h"] for c in data["amortised"]))
    beta_ms = sorted(set(c["beta_m"] for c in data["amortised"]))

    bm_colors  = {1.0: "steelblue", 5.0: "darkorange", 20.0: "mediumseagreen"}
    nt_markers = {2048: "s", 8192: "o"}

    def _m(d):
        return d["mean"] if isinstance(d, dict) else (d if d is not None else float("nan"))

    def _s(d):
        return d["std"]  if isinstance(d, dict) else 0.0

    # ── Plot 1 & 2: cost-quality curves ──────────────────────────────────────
    for stat_key, stat_label, fname in [
        ("best_energy", "Best energy (min)", "best"),
        ("q01",         "q01 energy",        "q01"),
        ("mean_energy", "Mean energy",       "mean"),
    ]:
        for beta_h in beta_hs:
            fig, axes = plt.subplots(1, len(dims), figsize=(5.5 * len(dims), 4.5), squeeze=False)

            for ax, dim in zip(axes[0], dims):
                # Direct fixed ULA — curve
                fix_pts = sorted(
                    [p for p in data["direct_ula_fixed"]
                     if p["dim"] == dim and p["beta_h"] == beta_h],
                    key=lambda p: p["n_grad_evals"],
                )
                if fix_pts:
                    xs = [p["n_grad_evals"] for p in fix_pts]
                    ys = [_m(p.get(stat_key, {})) for p in fix_pts]
                    es = [_s(p.get(stat_key, {})) for p in fix_pts]
                    ax.errorbar(xs, ys, yerr=es, fmt="s--", color="grey",
                                label="Direct ULA (fixed β)", capsize=3, lw=1.2, ms=5, alpha=0.7)

                # Direct annealed SMC — points (x = oracle_cost_total)
                ann_pts = sorted(
                    [p for p in data["direct_ula_annealed"]
                     if p["dim"] == dim and p["beta_h"] == beta_h],
                    key=lambda p: p.get("oracle_cost_total", p["n_grad_evals"]),
                )
                if ann_pts:
                    xs = [p.get("oracle_cost_total", p["n_grad_evals"]) for p in ann_pts]
                    ys = [_m(p.get(stat_key, {})) for p in ann_pts]
                    es = [_s(p.get(stat_key, {})) for p in ann_pts]
                    ax.errorbar(xs, ys, yerr=es, fmt="^-", color="black",
                                label="Direct ULA SMC (annealed)", capsize=3, lw=1.5, ms=6)

                # Amortised — one point per (beta_m, n_train) at n_samples=MAX, local_steps=0
                # Additional point with local_steps=10 shown as smaller marker
                for beta_m in beta_ms:
                    color = bm_colors.get(beta_m, "purple")
                    for n_train in sorted(N_TRAIN_SIZES, reverse=True):
                        marker = nt_markers.get(n_train, "D")
                        for local_steps in sorted(LOCAL_STEPS_LIST):
                            cells = [c for c in data["amortised"]
                                     if c["dim"] == dim and c["beta_m"] == beta_m
                                     and c["beta_h"] == beta_h and c["n_train"] == n_train
                                     and c["n_samples"] == MAX_N_SAMPLES
                                     and c["local_steps"] == local_steps]
                            if not cells:
                                continue
                            c = cells[0]
                            x_val = _m(c.get("oracle_cost_total", {}))
                            y_val = _m(c.get(stat_key, {}))
                            y_err = _s(c.get(stat_key, {}))
                            lbl   = (f"Amort β_M={beta_m:g} N_tr={n_train}"
                                     if local_steps == 0 else None)
                            size  = 60 if local_steps == 0 else 30
                            alpha = 0.9 if local_steps == 0 else 0.5
                            ax.errorbar([x_val], [y_val], yerr=[y_err],
                                        fmt=marker, color=color, ms=size ** 0.5,
                                        capsize=2, alpha=alpha, label=lbl, zorder=5)

                ax.set_xscale("log")
                ax.set_xlabel("Oracle cost (∇E evals)", fontsize=8)
                ax.set_ylabel(stat_label, fontsize=8)
                ax.set_title(f"d={dim}", fontsize=9)
                ax.legend(fontsize=6, loc="upper right")
                ax.grid(True, alpha=0.3)

            fig.suptitle(f"{ENERGY}  β_H={beta_h}  Cost vs {stat_label}", fontsize=10)
            fig.tight_layout()
            out = plots_dir / f"cost_quality_{fname}_betaH{_beta_str(beta_h)}.svg"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {out.name}")

    # ── Plot 3: N_train effect ────────────────────────────────────────────────
    for beta_h in beta_hs:
        fig, axes = plt.subplots(1, len(dims), figsize=(4.5 * len(dims), 4), squeeze=False)
        for ax, dim in zip(axes[0], dims):
            for n_train in sorted(N_TRAIN_SIZES):
                xs, ys, es = [], [], []
                for beta_m in beta_ms:
                    pts = [c for c in data["amortised"]
                           if c["dim"] == dim and c["beta_m"] == beta_m
                           and c["beta_h"] == beta_h and c["n_train"] == n_train
                           and c["n_samples"] == 2048 and c["local_steps"] == 0]
                    if not pts:
                        continue
                    xs.append(beta_m)
                    ys.append(_m(pts[0].get("q01", {})))
                    es.append(_s(pts[0].get("q01", {})))
                color = "steelblue" if n_train == min(N_TRAIN_SIZES) else "darkorange"
                if xs:
                    ax.errorbar(xs, ys, yerr=es, fmt="o-", color=color,
                                label=f"N_train={n_train}", capsize=3, lw=1.5)
            ax.set_xlabel(r"$\beta_M$", fontsize=8)
            ax.set_ylabel("q01 energy", fontsize=8)
            ax.set_title(f"d={dim}", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"{ENERGY}  β_H={beta_h}  N_train effect  (N_samples=2048, local_steps=0)",
                     fontsize=10)
        fig.tight_layout()
        out = plots_dir / f"ntrain_effect_betaH{_beta_str(beta_h)}.svg"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out.name}")

    # ── Plot 4: N_samples effect ──────────────────────────────────────────────
    for beta_h in beta_hs:
        for dim in dims:
            fig, axes = plt.subplots(1, len(beta_ms), figsize=(4 * len(beta_ms), 4), squeeze=False)
            for ax, beta_m in zip(axes[0], beta_ms):
                for n_train in sorted(N_TRAIN_SIZES):
                    xs, ys, es = [], [], []
                    for n_samples in sorted(N_SAMPLE_SIZES):
                        pts = [c for c in data["amortised"]
                               if c["dim"] == dim and c["beta_m"] == beta_m
                               and c["beta_h"] == beta_h and c["n_train"] == n_train
                               and c["n_samples"] == n_samples and c["local_steps"] == 0]
                        if not pts:
                            continue
                        xs.append(n_samples)
                        ys.append(_m(pts[0].get("best_energy", {})))
                        es.append(_s(pts[0].get("best_energy", {})))
                    ls = "-" if n_train == max(N_TRAIN_SIZES) else "--"
                    if xs:
                        ax.errorbar(xs, ys, yerr=es, fmt=f"o{ls}", capsize=3, lw=1.5,
                                    label=f"N_tr={n_train}")
                ax.set_xscale("log")
                ax.set_xlabel("N_samples (inference)", fontsize=8)
                ax.set_ylabel("Best energy", fontsize=8)
                ax.set_title(f"β_M={beta_m:g}", fontsize=9)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
            fig.suptitle(f"{ENERGY}  d={dim}  β_H={beta_h}  Inference budget effect",
                         fontsize=10)
            fig.tight_layout()
            out = plots_dir / f"nsamples_effect_d{dim}_betaH{_beta_str(beta_h)}.svg"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {out.name}")

    print(f"\nAll plots saved to {plots_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global ENERGY

    parser = argparse.ArgumentParser(description="Cost-quality: amortised diffusion vs direct ULA.")
    parser.add_argument("--energy",    choices=list(ENERGY_MAP), default=None)
    parser.add_argument("--seed",      type=int, default=None,
                        help="Run a single seed (use --aggregate when all done)")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--no-plot",   action="store_true",
                        help="Skip plotting after run/aggregate (use on server)")
    args = parser.parse_args()

    if args.energy:
        ENERGY = args.energy

    if args.plot_only:
        plot_results()
        return

    if args.aggregate:
        print("Aggregating...")
        aggregate_results()
        if not args.no_plot:
            plot_results()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)

    seeds = [args.seed] if args.seed is not None else SEEDS
    _exp_dir().mkdir(parents=True, exist_ok=True)

    print(f"=== COST-QUALITY  {ENERGY}  dims={DIMS}  β_M={BETA_MS}  β_H={BETA_HS}  seeds={seeds} ===\n")

    for seed in seeds:
        print(f"\n{'='*60}\n  SEED {seed}\n{'='*60}")
        run_seed(seed, device)

    if args.seed is None:
        print("\nAggregating...")
        aggregate_results()
        if not args.no_plot:
            plot_results()
    else:
        print(f"\nSeed {args.seed} done. Run --aggregate when all seeds complete.")


if __name__ == "__main__":
    main()
