#!/usr/bin/env python
"""Budget frontier: given the same total oracle budget B, amortised diffusion vs direct SMC restarts.

For each (dim, beta_h):
  - Direct SMC: spend budget on repeated annealed SMC restarts
  - Diffusion:  pay C_setup once, then get cheap reuse batches at C_use each

This is the decisive experiment: is it ever better to spend budget on reuse vs restarts?

Usage:
    # On GPU server — run 3 seeds in parallel for --extend-diffusion:
    uv run python scripts/experiments/budget_frontier.py --extend-diffusion --seed 0 --no-plot
    uv run python scripts/experiments/budget_frontier.py --extend-diffusion --seed 1 --no-plot
    uv run python scripts/experiments/budget_frontier.py --extend-diffusion --seed 2 --no-plot

    uv run python scripts/experiments/budget_frontier.py --run-direct --no-plot
    uv run python scripts/experiments/budget_frontier.py --aggregate --no-plot
    uv run python scripts/experiments/budget_frontier.py --plot-only
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from amortised_annealing.diffusion import VPSchedule, MLPScore, ReverseSDE
from amortised_annealing.energies import Ackley, DoubleWell, ManyWell, Rastrigin
from amortised_annealing.sampling import DiffusionModelSampler
from amortised_annealing.smc import ParticleCloud, SMCSampler, ULAProposal

ROOT             = Path(__file__).parent.parent.parent
SAMPLE_DIR       = ROOT / "data" / "samples"
MODEL_DIR        = ROOT / "data" / "models"
EXP_BASE         = ROOT / "data" / "experiments" / "budget_frontier"
REUSE_SWEEP_BASE = ROOT / "data" / "experiments" / "reuse_sweep"

ENERGY_MAP = {
    "ackley":      Ackley,
    "double_well": DoubleWell,
    "many_well":   ManyWell,
    "rastrigin":   Rastrigin,
}
ENERGY = "ackley"

# ===========================================================================
# EXPERIMENT CONFIGURATION
# ===========================================================================
DIMS    = [10, 20]
BETA_HS = [20.0, 50.0]
SEEDS   = [0, 1, 2]
B_MAX   = 100_000_000

N_DIRECT_PARTICLES   = 4096
DIRECT_ULA_STEP_SIZE = 1e-3

# oracle cost per direct run = N_DIRECT_PARTICLES × (n_smc × n_ula + n_smc)
DIRECT_SMC_CONFIGS = [
    {"n_smc":  16, "n_ula":  5},   # 393,216   → 254 restarts in 100M
    {"n_smc":  32, "n_ula": 10},   # 1,441,792 →  69 restarts
    {"n_smc":  64, "n_ula": 20},   # 5,505,024 →  18 restarts
    {"n_smc": 128, "n_ula": 40},   # 21,495,808 →  4 restarts
    {"n_smc": 256, "n_ula": 80},   # 84,934,656 →  1 restart
]

DIFFUSION_BETA_MS   = [5.0, 20.0]
N_TRAIN_DIFF        = 2048
R_MAX_EXTENDED      = 1000
N_SAMPLES_DIFF      = 8192
LOCAL_STEPS_DIFF    = 10
C_USE               = N_SAMPLES_DIFF * LOCAL_STEPS_DIFF   # 81_920

MODEL_HIDDEN_DIMS    = [256, 256, 256]
MODEL_TIME_EMBED_DIM = 64
MCMC_N_PARTICLES     = 8192
MCMC_N_STEPS         = 10_000
LOCAL_ULA_STEP_SIZE  = 1e-3
MODEL_SAMPLE_N_STEPS = 500
MODEL_SAMPLE_T_START = 1.0
MODEL_SAMPLE_T_END   = 1e-3

B_GRID_N = 300

# Per-(dim, beta_h) thresholds tuned to the frontier — loose thresholds
# produce uninformative ratio=0.019 rows where the cheapest direct config
# already succeeds on its first restart.
THRESHOLDS: dict[tuple[int, float], list[float]] = {
    (10, 20.0): [0.10, 0.08, 0.06, 0.05, 0.04, 0.035, 0.03],
    (10, 50.0): [0.06, 0.05, 0.04, 0.035, 0.03, 0.027, 0.025],
    (20, 20.0): [0.30, 0.25, 0.20, 0.175, 0.15, 0.125, 0.10],
    (20, 50.0): [0.15, 0.125, 0.10, 0.09, 0.08, 0.07],
}

# Warm-start ablation: ULA from N(0,I) → π_{β_M}, then SMC from β_M → β_H
WARM_N_PARTICLES = 2048
WARM_T_ULA       = 10_000   # = MCMC_N_STEPS; C_ULA = 2048×10000 = C_setup (same as diffusion)
WARM_SMC_CONFIGS = [
    {"n_smc": 16, "n_ula":  5},
    {"n_smc": 32, "n_ula": 10},
    {"n_smc": 64, "n_ula": 20},
]
WARM_BETA_MS = [5.0, 20.0]
# ===========================================================================


# ---------------------------------------------------------------------------
# Naming helpers (must match reuse_sweep.py and cost_quality.py)
# ---------------------------------------------------------------------------

def _beta_str(b: float) -> str:
    return f"{b:g}".replace(".", "p")


def _preset_tag() -> str:
    s = MODEL_HIDDEN_DIMS
    return f"mlp{s[0]}x{len(s)}" if len(set(s)) == 1 else "mlp_" + "_".join(str(x) for x in s)


def _sample_run_name(dim: int, beta_m: float, seed: int) -> str:
    return f"{ENERGY}_d{dim}_beta{_beta_str(beta_m)}_ula_seed{seed}"


def _model_run_name_ntrain(sample_run: str, n_train: int, seed: int) -> str:
    if n_train == MCMC_N_PARTICLES:
        return f"{sample_run}_{_preset_tag()}_eps_seed{seed}"
    return f"{sample_run}_ntrain{n_train}_{_preset_tag()}_eps_seed{seed}"


def _run_name(dim: int, beta_m: float, beta_h: float, n_train: int) -> str:
    return f"d{dim}_bm{_beta_str(beta_m)}_bh{_beta_str(beta_h)}_ntrain{n_train}"


def _oracle_cost_direct(n_smc: int, n_ula: int) -> int:
    return N_DIRECT_PARTICLES * (n_smc * n_ula + n_smc)


def _oracle_cost_warm(n_smc: int, n_ula: int) -> int:
    return WARM_N_PARTICLES * (WARM_T_ULA + n_smc * (n_ula + 1))


# ---------------------------------------------------------------------------
# Model loading / utilities
# ---------------------------------------------------------------------------

def _load_model(model_run: str, device: torch.device):
    """Returns (reverse_sde, energy_obj, dim, beta_m)."""
    run_dir = MODEL_DIR / model_run
    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    sample_run     = cfg["samples"]["run_name"]
    sample_summary = json.loads((SAMPLE_DIR / sample_run / "summary.json").read_text())
    dim    = sample_summary["dim"]
    beta_m = sample_summary["beta_m"]
    energy = ENERGY_MAP[sample_summary["energy"]](dim=dim)

    mc    = cfg["model"]
    model = MLPScore(
        dim=dim, hidden_dims=tuple(mc["hidden_dims"]),
        time_embed_dim=mc.get("time_embed_dim", 64),
        activation=mc.get("activation", "silu"),
        predict_score=mc.get("predict_score", False),
    )
    state = torch.load(run_dir / "ema_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    sc       = cfg["schedule"]
    schedule = VPSchedule(beta_min=sc.get("beta_min", 0.1), beta_max=sc.get("beta_max", 20.0))
    return ReverseSDE(model, schedule), energy, dim, beta_m


def _apply_local_ula(
    x: torch.Tensor, energy_fn, beta_h: float, n_steps: int, seed: int
) -> torch.Tensor:
    if n_steps == 0:
        return x
    torch.manual_seed(seed)
    x = x.detach().clone()
    for _ in range(n_steps):
        x = x.requires_grad_(True)
        with torch.enable_grad():
            grad = torch.autograd.grad(energy_fn(x).sum(), x)[0]
        x = (x - LOCAL_ULA_STEP_SIZE * beta_h * grad.detach()
             + (2 * LOCAL_ULA_STEP_SIZE) ** 0.5 * torch.randn_like(x)).detach()
    return x


def _energy_stats(e: torch.Tensor) -> dict:
    qs = torch.quantile(e, torch.tensor([0.01, 0.05, 0.10, 0.50, 0.90]))
    return {
        "best_energy": round(float(e.min()),  4),
        "mean_energy": round(float(e.mean()), 4),
        "q01": round(float(qs[0]), 4),
        "q05": round(float(qs[1]), 4),
        "q10": round(float(qs[2]), 4),
        "q50": round(float(qs[3]), 4),
        "q90": round(float(qs[4]), 4),
    }


def _load_reuse_sweep_energies(run_name: str, seed: int, r_max: int) -> list[torch.Tensor]:
    """Load pre-computed energy tensors from reuse_sweep raw .pt files."""
    raw_dir = REUSE_SWEEP_BASE / ENERGY / "raw"
    energies = []
    for r in range(r_max):
        pt = raw_dir / f"{run_name}_seed{seed}_r{r:03d}.pt"
        if not pt.exists():
            return []
        energies.append(torch.load(pt, weights_only=True))
    return energies


# ---------------------------------------------------------------------------
# Step 1: Extend diffusion to R=1000
# ---------------------------------------------------------------------------

def extend_diffusion(seed: int, overwrite: bool, device: torch.device) -> None:
    out_dir = EXP_BASE / ENERGY
    out_dir.mkdir(parents=True, exist_ok=True)

    for dim in DIMS:
        for beta_m in DIFFUSION_BETA_MS:
            for beta_h in BETA_HS:
                run_name  = _run_name(dim, beta_m, beta_h, N_TRAIN_DIFF)
                json_path = out_dir / f"diffusion_{run_name}_seed{seed}.json"

                if json_path.exists() and not overwrite:
                    existing = json.loads(json_path.read_text())
                    if existing.get("r_max", 0) >= R_MAX_EXTENDED:
                        print(f"  SKIP  {json_path.name}")
                        continue

                sample_run = _sample_run_name(dim, beta_m, seed)
                model_run  = _model_run_name_ntrain(sample_run, N_TRAIN_DIFF, seed)
                if not (MODEL_DIR / model_run / "ema_model.pt").exists():
                    print(f"  SKIP  model not found: {model_run}")
                    continue

                print(f"  RUN   diffusion_{run_name}_seed{seed}")
                rsde, energy_obj, _dim, _bm = _load_model(model_run, device)
                energy_fn = energy_obj.energy
                c_setup   = N_TRAIN_DIFF * MCMC_N_STEPS  # 20_480_000

                # Try reusing reuse_sweep energies for first 100 batches
                all_energies: list[torch.Tensor] = _load_reuse_sweep_energies(
                    run_name, seed, 100
                )
                r_start = len(all_energies)
                if r_start > 0:
                    print(f"    Preloaded {r_start} batches from reuse_sweep")

                t0 = time.perf_counter()
                for r in range(r_start, R_MAX_EXTENDED):
                    batch_seed = seed * 10_000 + r
                    torch.manual_seed(batch_seed)
                    with torch.no_grad():
                        x = DiffusionModelSampler(
                            rsde,
                            n_steps=MODEL_SAMPLE_N_STEPS,
                            t_start=MODEL_SAMPLE_T_START,
                            t_end=MODEL_SAMPLE_T_END,
                        ).sample(N_SAMPLES_DIFF, device, show_progress=False)
                    x = _apply_local_ula(x, energy_fn, beta_h, LOCAL_STEPS_DIFF, batch_seed)
                    with torch.no_grad():
                        e = energy_fn(x.cpu()).float()
                    all_energies.append(e)

                    if (r + 1) % 200 == 0:
                        elapsed = time.perf_counter() - t0
                        print(f"    r={r+1:4d}/{R_MAX_EXTENDED}  elapsed={elapsed:.1f}s")

                wall = round(time.perf_counter() - t0, 2)

                # Post-hoc cumulative stats
                batch_best = [float(e.min()) for e in all_energies]
                batch_q01  = [float(torch.quantile(e, 0.01)) for e in all_energies]

                # Cumulative best: running min
                cumulative_best: list[float] = []
                running_min = float("inf")
                for b in batch_best:
                    running_min = min(running_min, b)
                    cumulative_best.append(running_min)

                # Cumulative q01: np.partition (O(n) per step, fast in practice)
                all_e_np = np.stack([e.numpy() for e in all_energies])  # (R, N)
                cumulative_q01: list[float] = []
                for r in range(len(all_energies)):
                    n_total = (r + 1) * N_SAMPLES_DIFF
                    k = max(0, int(0.01 * n_total) - 1)
                    flat = all_e_np[: r + 1].ravel()
                    cumulative_q01.append(float(np.partition(flat, k)[k]))

                total_cost = [c_setup + (r + 1) * C_USE for r in range(len(all_energies))]

                summary = {
                    "dim": dim, "beta_m": beta_m, "beta_h": beta_h,
                    "n_train": N_TRAIN_DIFF, "seed": seed,
                    "n_samples": N_SAMPLES_DIFF, "local_steps": LOCAL_STEPS_DIFF,
                    "r_max": len(all_energies),
                    "oracle_cost_setup": c_setup,
                    "oracle_cost_per_use": C_USE,
                    "wall_seconds_new_batches": wall,
                    "cumulative_best": cumulative_best,
                    "cumulative_q01":  cumulative_q01,
                    "batch_best": batch_best,
                    "batch_q01":  batch_q01,
                    "total_cost": total_cost,
                }
                with open(json_path, "w") as f:
                    json.dump(summary, f, indent=2)
                print(f"  Saved {json_path.name}  "
                      f"(r_max={len(all_energies)}  "
                      f"final_best={cumulative_best[-1]:.4f}  wall={wall:.1f}s)")


# ---------------------------------------------------------------------------
# Step 2: Direct SMC restarts
# ---------------------------------------------------------------------------

def run_direct(overwrite: bool, device: torch.device) -> None:
    out_dir = EXP_BASE / ENERGY
    out_dir.mkdir(parents=True, exist_ok=True)

    for dim in DIMS:
        energy_obj = ENERGY_MAP[ENERGY](dim=dim)
        energy_fn  = energy_obj.energy

        for beta_h in BETA_HS:
            for cfg in DIRECT_SMC_CONFIGS:
                n_smc = cfg["n_smc"]
                n_ula = cfg["n_ula"]
                c_k   = _oracle_cost_direct(n_smc, n_ula)
                n_restarts = B_MAX // c_k

                fname    = f"direct_d{dim}_bH{beta_h:g}_smc{n_smc}_ula{n_ula}.json"
                out_path = out_dir / fname

                if out_path.exists() and not overwrite:
                    print(f"  SKIP  {fname}")
                    continue

                print(f"  RUN   d={dim}  bH={beta_h}  smc={n_smc}  ula={n_ula}  "
                      f"c_k={c_k:,}  n_restarts={n_restarts}")
                beta_ladder = torch.linspace(0.01, beta_h, n_smc + 1, device=device)
                proposal    = ULAProposal(energy_obj, step_size=DIRECT_ULA_STEP_SIZE, n_steps=n_ula)

                restarts: list[dict] = []
                t0 = time.perf_counter()
                for restart_id in range(n_restarts):
                    torch.manual_seed(restart_id)
                    x0   = torch.randn(N_DIRECT_PARTICLES, dim, device=device)
                    cloud, _ = SMCSampler(
                        proposal.mutation_kernel, proposal.weight_update, energy_obj,
                        ess_threshold=0.5,
                    ).run(
                        ParticleCloud(x0, torch.zeros(N_DIRECT_PARTICLES, device=device)),
                        beta_ladder, show_progress=False,
                    )
                    with torch.no_grad():
                        e = energy_fn(cloud.x.cpu()).float()
                    restarts.append({"restart_id": restart_id, **_energy_stats(e)})

                    if (restart_id + 1) % 20 == 0 or restart_id + 1 == n_restarts:
                        elapsed = time.perf_counter() - t0
                        print(f"    restart {restart_id+1}/{n_restarts}  "
                              f"cum_best={min(r['best_energy'] for r in restarts):.4f}  "
                              f"elapsed={elapsed:.1f}s")

                wall = round(time.perf_counter() - t0, 2)
                result = {
                    "dim": dim, "beta_h": beta_h,
                    "n_smc": n_smc, "n_ula": n_ula,
                    "oracle_cost_per_run": int(c_k),
                    "n_restarts": n_restarts,
                    "wall_seconds": wall,
                    "restarts": restarts,
                }
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)
                cum_best = min(r["best_energy"] for r in restarts)
                print(f"  Saved {fname}  (cum_best={cum_best:.4f}  wall={wall:.1f}s)")


# ---------------------------------------------------------------------------
# Step 2b: Warm-start ULA-SMC restarts
# ---------------------------------------------------------------------------

def run_warm_start(overwrite: bool, device: torch.device) -> None:
    out_dir = EXP_BASE / ENERGY
    out_dir.mkdir(parents=True, exist_ok=True)

    for dim in DIMS:
        energy_obj = ENERGY_MAP[ENERGY](dim=dim)
        energy_fn  = energy_obj.energy

        for beta_m in WARM_BETA_MS:
            for beta_h in BETA_HS:
                for cfg in WARM_SMC_CONFIGS:
                    n_smc = cfg["n_smc"]
                    n_ula = cfg["n_ula"]
                    c_k   = _oracle_cost_warm(n_smc, n_ula)
                    c_ula = WARM_N_PARTICLES * WARM_T_ULA
                    c_smc = WARM_N_PARTICLES * (n_smc * n_ula + n_smc)
                    n_restarts = B_MAX // c_k

                    fname    = (f"warm_start_d{dim}_bM{_beta_str(beta_m)}"
                                f"_bH{beta_h:g}_smc{n_smc}_ula{n_ula}.json")
                    out_path = out_dir / fname

                    if out_path.exists() and not overwrite:
                        print(f"  SKIP  {fname}")
                        continue

                    print(f"  RUN   d={dim}  bM={beta_m}  bH={beta_h}  "
                          f"smc={n_smc}  ula={n_ula}  c_k={c_k:,}  n_restarts={n_restarts}")
                    beta_ladder = torch.linspace(beta_m, beta_h, n_smc + 1, device=device)
                    proposal    = ULAProposal(energy_obj, step_size=DIRECT_ULA_STEP_SIZE,
                                             n_steps=n_ula)

                    restarts: list[dict] = []
                    t0 = time.perf_counter()
                    for restart_id in range(n_restarts):
                        torch.manual_seed(restart_id)
                        x0 = torch.randn(WARM_N_PARTICLES, dim, device=device)
                        # ULA phase: relax from N(0,I) → π_{β_M}
                        x0 = _apply_local_ula(x0, energy_fn, beta_m, WARM_T_ULA, restart_id)
                        # SMC phase: anneal β_M → β_H from warm particles
                        cloud, _ = SMCSampler(
                            proposal.mutation_kernel, proposal.weight_update, energy_obj,
                            ess_threshold=0.5,
                        ).run(
                            ParticleCloud(x0, torch.zeros(WARM_N_PARTICLES, device=device)),
                            beta_ladder, show_progress=False,
                        )
                        with torch.no_grad():
                            e = energy_fn(cloud.x.cpu()).float()
                        restarts.append({"restart_id": restart_id, **_energy_stats(e)})

                        elapsed = time.perf_counter() - t0
                        print(f"    restart {restart_id+1}/{n_restarts}  "
                              f"cum_best={min(r['best_energy'] for r in restarts):.4f}  "
                              f"elapsed={elapsed:.1f}s")

                    wall = round(time.perf_counter() - t0, 2)
                    result = {
                        "dim": dim, "beta_m": beta_m, "beta_h": beta_h,
                        "n_smc": n_smc, "n_ula": n_ula,
                        "n_warm_particles": WARM_N_PARTICLES, "t_ula": WARM_T_ULA,
                        "oracle_cost_ula_phase": int(c_ula),
                        "oracle_cost_smc_phase": int(c_smc),
                        "oracle_cost_per_restart": int(c_k),
                        "n_restarts": n_restarts,
                        "wall_seconds": wall,
                        "restarts": restarts,
                    }
                    with open(out_path, "w") as f:
                        json.dump(result, f, indent=2)
                    cum_best = min(r["best_energy"] for r in restarts)
                    print(f"  Saved {fname}  (cum_best={cum_best:.4f}  wall={wall:.1f}s)")


# ---------------------------------------------------------------------------
# Step 3: Aggregate — build frontier JSONs + threshold table
# ---------------------------------------------------------------------------

def _agg_list_mean_std(arrays: list[list[float]]) -> tuple[list[float], list[float]]:
    n = min(len(a) for a in arrays)
    means, stds = [], []
    for i in range(n):
        vals = np.array([a[i] for a in arrays])
        means.append(float(vals.mean()))
        stds.append(float(vals.std()))
    return means, stds


def build_aggregate(no_plot: bool) -> None:
    data_dir = EXP_BASE / ENERGY

    for dim in DIMS:
        for beta_h in BETA_HS:
            print(f"\n  Building frontier d={dim}  β_H={beta_h}")

            # ── Load diffusion data ──────────────────────────────────────────
            diff_data: dict[float, list[dict]] = {}
            for beta_m in DIFFUSION_BETA_MS:
                rn = _run_name(dim, beta_m, beta_h, N_TRAIN_DIFF)
                seeds_data = []
                for seed in SEEDS:
                    p = data_dir / f"diffusion_{rn}_seed{seed}.json"
                    if p.exists():
                        seeds_data.append(json.loads(p.read_text()))
                if seeds_data:
                    diff_data[beta_m] = seeds_data

            if not diff_data:
                print("    No diffusion data — skipping")
                continue

            # ── Load direct SMC data ─────────────────────────────────────────
            direct_data: dict[str, dict] = {}
            for cfg in DIRECT_SMC_CONFIGS:
                n_smc, n_ula = cfg["n_smc"], cfg["n_ula"]
                p = data_dir / f"direct_d{dim}_bH{beta_h:g}_smc{n_smc}_ula{n_ula}.json"
                if p.exists():
                    key = f"smc{n_smc}_ula{n_ula}"
                    direct_data[key] = json.loads(p.read_text())

            if not direct_data:
                print("    No direct SMC data — skipping")
                continue

            # ── Budget grid ──────────────────────────────────────────────────
            min_c = min(d["oracle_cost_per_run"] for d in direct_data.values())
            b_grid = np.logspace(np.log10(min_c), np.log10(B_MAX), B_GRID_N).tolist()

            # ── Direct frontier ──────────────────────────────────────────────
            direct_per_config: dict[str, list] = {}
            for key, d in direct_data.items():
                c_k          = d["oracle_cost_per_run"]
                best_elist   = [r["best_energy"] for r in d["restarts"]]
                curve: list[float] = []
                for B in b_grid:
                    m = int(B // c_k)
                    if m < 1:
                        curve.append(float("nan"))
                    else:
                        m = min(m, len(best_elist))
                        curve.append(float(min(best_elist[:m])))
                direct_per_config[key] = curve

            direct_frontier_best = [
                float(min(
                    (curve[i] for curve in direct_per_config.values()
                     if not math.isnan(curve[i])),
                    default=float("nan"),
                ))
                for i in range(len(b_grid))
            ]

            # ── Diffusion frontier ───────────────────────────────────────────
            diff_per_config: dict[str, dict] = {}
            for beta_m, seeds_data in diff_data.items():
                c_setup = seeds_data[0]["oracle_cost_setup"]
                seed_curves: list[list[float]] = []
                for sd in seeds_data:
                    cb = sd["cumulative_best"]
                    curve: list[float] = []
                    for B in b_grid:
                        if B < c_setup + C_USE:
                            curve.append(float("nan"))
                        else:
                            r_batch = min(int((B - c_setup) // C_USE), len(cb))
                            if r_batch < 1:
                                curve.append(float("nan"))
                            else:
                                curve.append(cb[r_batch - 1])
                    seed_curves.append(curve)

                mean_c: list[float] = []
                std_c:  list[float] = []
                for i in range(len(b_grid)):
                    vals = [c[i] for c in seed_curves if not math.isnan(c[i])]
                    if not vals:
                        mean_c.append(float("nan"))
                        std_c.append(float("nan"))
                    elif len(vals) == 1:
                        mean_c.append(vals[0])
                        std_c.append(0.0)
                    else:
                        arr = np.array(vals)
                        mean_c.append(float(arr.mean()))
                        std_c.append(float(arr.std()))

                bm_key = f"bm{_beta_str(beta_m)}_n{N_TRAIN_DIFF}"
                diff_per_config[bm_key] = {"mean": mean_c, "std": std_c}

            # Diffusion lower envelope over beta_m
            diff_best_mean: list[float] = []
            diff_best_std:  list[float] = []
            for i in range(len(b_grid)):
                valid = {
                    k: d["mean"][i]
                    for k, d in diff_per_config.items()
                    if not math.isnan(d["mean"][i])
                }
                if not valid:
                    diff_best_mean.append(float("nan"))
                    diff_best_std.append(float("nan"))
                else:
                    best_k = min(valid, key=valid.__getitem__)
                    diff_best_mean.append(valid[best_k])
                    diff_best_std.append(diff_per_config[best_k]["std"][i])

            # ── Warm-start frontier ──────────────────────────────────────────
            warm_per_config: dict[str, list] = {}
            for beta_m in WARM_BETA_MS:
                for cfg in WARM_SMC_CONFIGS:
                    n_smc, n_ula = cfg["n_smc"], cfg["n_ula"]
                    fname = (f"warm_start_d{dim}_bM{_beta_str(beta_m)}"
                             f"_bH{beta_h:g}_smc{n_smc}_ula{n_ula}.json")
                    p = data_dir / fname
                    if not p.exists():
                        continue
                    d   = json.loads(p.read_text())
                    c_k = d["oracle_cost_per_restart"]
                    best_elist = [r["best_energy"] for r in d["restarts"]]
                    curve: list[float] = []
                    for B in b_grid:
                        m = int(B // c_k)
                        if m < 1:
                            curve.append(float("nan"))
                        else:
                            m = min(m, len(best_elist))
                            curve.append(float(min(best_elist[:m])))
                    key = f"bm{_beta_str(beta_m)}_smc{n_smc}_ula{n_ula}"
                    warm_per_config[key] = curve

            warm_frontier_best = [
                float(min(
                    (curve[i] for curve in warm_per_config.values()
                     if not math.isnan(curve[i])),
                    default=float("nan"),
                ))
                for i in range(len(b_grid))
            ]

            frontier = {
                "budget_grid":                b_grid,
                "direct_frontier_best":       direct_frontier_best,
                "direct_per_config":          direct_per_config,
                "diffusion_frontier_best_mean": diff_best_mean,
                "diffusion_frontier_best_std":  diff_best_std,
                "diffusion_per_config":        diff_per_config,
                "warm_start_frontier_best":   warm_frontier_best,
                "warm_start_per_config":      warm_per_config,
            }
            fpath = data_dir / f"frontier_d{dim}_bH{beta_h:g}.json"
            with open(fpath, "w") as f:
                json.dump(frontier, f, indent=2)
            print(f"    Saved {fpath.name}")

    # ── Threshold table ──────────────────────────────────────────────────────
    _build_threshold_table(data_dir)

    if not no_plot:
        plot_frontiers()


def _build_threshold_table(data_dir: Path) -> None:
    rows: list[dict] = []

    for dim in DIMS:
        for beta_h in BETA_HS:
            thresholds = THRESHOLDS.get((dim, beta_h))
            if not thresholds:
                continue

            # Load direct data
            direct_data: dict[str, dict] = {}
            for cfg in DIRECT_SMC_CONFIGS:
                n_smc, n_ula = cfg["n_smc"], cfg["n_ula"]
                p = data_dir / f"direct_d{dim}_bH{beta_h:g}_smc{n_smc}_ula{n_ula}.json"
                if p.exists():
                    key = f"smc{n_smc}_ula{n_ula}"
                    direct_data[key] = json.loads(p.read_text())

            # Load diffusion data
            diff_data: dict[float, list[dict]] = {}
            for beta_m in DIFFUSION_BETA_MS:
                rn = _run_name(dim, beta_m, beta_h, N_TRAIN_DIFF)
                sds = []
                for seed in SEEDS:
                    p = data_dir / f"diffusion_{rn}_seed{seed}.json"
                    if p.exists():
                        sds.append(json.loads(p.read_text()))
                if sds:
                    diff_data[beta_m] = sds

            # Load warm-start data for this (dim, beta_h)
            warm_data: dict[str, dict] = {}
            for beta_m in WARM_BETA_MS:
                for cfg in WARM_SMC_CONFIGS:
                    n_smc, n_ula = cfg["n_smc"], cfg["n_ula"]
                    fname = (f"warm_start_d{dim}_bM{_beta_str(beta_m)}"
                             f"_bH{beta_h:g}_smc{n_smc}_ula{n_ula}.json")
                    p = data_dir / fname
                    if p.exists():
                        key = f"bm{_beta_str(beta_m)}_smc{n_smc}_ula{n_ula}"
                        warm_data[key] = json.loads(p.read_text())

            if not direct_data or not diff_data:
                continue

            for thresh in thresholds:
                # Direct: cheapest (config, prefix-m) achieving cumulative best <= thresh
                direct_budget: int | None = None
                direct_cfg_key: str | None = None
                for key, d in direct_data.items():
                    c_k        = d["oracle_cost_per_run"]
                    best_elist = [r["best_energy"] for r in d["restarts"]]
                    running    = float("inf")
                    for m, be in enumerate(best_elist, 1):
                        running = min(running, be)
                        if running <= thresh:
                            cost = m * c_k
                            if direct_budget is None or cost < direct_budget:
                                direct_budget = cost
                                direct_cfg_key = key
                            break

                # Diffusion: cheapest (beta_m) achieving mean cumulative_best <= thresh
                diff_budget: int | None = None
                diff_bm_key: float | None = None
                for beta_m, seeds_data in diff_data.items():
                    c_setup = seeds_data[0]["oracle_cost_setup"]
                    arrays  = [sd["cumulative_best"] for sd in seeds_data]
                    n       = min(len(a) for a in arrays)
                    mean_cb = [sum(a[r] for a in arrays) / len(arrays) for r in range(n)]
                    for r, val in enumerate(mean_cb):
                        if val <= thresh:
                            cost = c_setup + (r + 1) * C_USE
                            if diff_budget is None or cost < diff_budget:
                                diff_budget = cost
                                diff_bm_key = beta_m
                            break

                # Warm-start: cheapest (beta_m, smc_cfg) achieving cumulative best <= thresh
                warm_budget: int | None = None
                warm_cfg_key: str | None = None
                for key, d in warm_data.items():
                    c_k        = d["oracle_cost_per_restart"]
                    best_elist = [r["best_energy"] for r in d["restarts"]]
                    running    = float("inf")
                    for m, be in enumerate(best_elist, 1):
                        running = min(running, be)
                        if running <= thresh:
                            cost = m * c_k
                            if warm_budget is None or cost < warm_budget:
                                warm_budget = cost
                                warm_cfg_key = key
                            break

                # Three-way winner
                budgets = {k: v for k, v in [
                    ("direct",    direct_budget),
                    ("diffusion", diff_budget),
                    ("warm_start", warm_budget),
                ] if v is not None}
                if not budgets:
                    winner = "not_reached"
                else:
                    winner = min(budgets, key=budgets.__getitem__)

                # budget_ratio = direct / diff (existing interpretation)
                if direct_budget is None and diff_budget is None:
                    ratio = "N/A"
                elif direct_budget is None:
                    ratio = "inf"
                elif diff_budget is None:
                    ratio = "0.0"
                else:
                    ratio = round(direct_budget / diff_budget, 3)

                rows.append({
                    "dim":            dim,
                    "beta_h":         beta_h,
                    "threshold":      thresh,
                    "direct_budget":  direct_budget if direct_budget is not None else "N/A",
                    "direct_cfg":     direct_cfg_key or "N/A",
                    "diff_budget":    diff_budget if diff_budget is not None else "N/A",
                    "diff_bm":        diff_bm_key if diff_bm_key is not None else "N/A",
                    "warm_budget":    warm_budget if warm_budget is not None else "N/A",
                    "warm_cfg":       warm_cfg_key or "N/A",
                    "budget_ratio":   ratio,
                    "winner":         winner,
                })

    if rows:
        csv_path = EXP_BASE / ENERGY / "threshold_table.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Threshold table → {csv_path}")

        # Pretty-print
        hdr = list(rows[0].keys())
        w   = [4, 6, 9, 14, 14, 14, 6, 14, 18, 12, 12]
        fmt = "  ".join(f"{{:<{n}}}" for n in w)
        print(f"\n{'='*120}")
        print("THRESHOLD TABLE  (budget_ratio=direct/diff;  >1 → diff cheaper;  winner = cheapest of all three)")
        print(f"{'='*120}")
        print(fmt.format(*hdr))
        print("-" * 120)
        for r in rows:
            print(fmt.format(*[str(r[k]) for k in hdr]))


# ---------------------------------------------------------------------------
# Step 4: Plots
# ---------------------------------------------------------------------------

def plot_frontiers() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_dir  = EXP_BASE / ENERGY
    plots_dir = data_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    bm_colors = {
        f"bm{_beta_str(5.0)}_n{N_TRAIN_DIFF}":  "mediumseagreen",
        f"bm{_beta_str(20.0)}_n{N_TRAIN_DIFF}": "darkorange",
    }
    cfg_greys = {
        "smc16_ula5":  "#aaaaaa",
        "smc32_ula10": "#888888",
        "smc64_ula20": "#666666",
        "smc128_ula40": "#444444",
        "smc256_ula80": "#222222",
    }
    C_SETUP = N_TRAIN_DIFF * MCMC_N_STEPS

    for dim in DIMS:
        for beta_h in BETA_HS:
            fpath = data_dir / f"frontier_d{dim}_bH{beta_h:g}.json"
            if not fpath.exists():
                print(f"  Missing {fpath.name} — run --aggregate first")
                continue

            fr = json.loads(fpath.read_text())
            b_grid = fr["budget_grid"]

            fig, ax = plt.subplots(figsize=(9, 5.5))

            # Individual direct config curves (grey dashed thin)
            for key, curve in fr["direct_per_config"].items():
                ys  = [v if not math.isnan(v) else None for v in curve]
                xs  = [b_grid[i] for i, y in enumerate(ys) if y is not None]
                yss = [y for y in ys if y is not None]
                if xs:
                    ax.plot(xs, yss, color=cfg_greys.get(key, "#999999"),
                            ls="--", lw=0.8, alpha=0.6, label=f"Direct {key}")

            # Direct lower envelope (black thick)
            dfe = fr["direct_frontier_best"]
            xs  = [b_grid[i] for i, v in enumerate(dfe) if not math.isnan(v)]
            ys  = [v for v in dfe if not math.isnan(v)]
            if xs:
                ax.plot(xs, ys, color="black", lw=2.5, ls="-", label="Direct SMC (envelope)", zorder=5)

            # Individual diffusion per-beta_m mean (colored dashed thin)
            for bm_key, d in fr["diffusion_per_config"].items():
                color = bm_colors.get(bm_key, "steelblue")
                mean_c = d["mean"]
                xs = [b_grid[i] for i, v in enumerate(mean_c) if not math.isnan(v)]
                ys = [v for v in mean_c if not math.isnan(v)]
                if xs:
                    ax.plot(xs, ys, color=color, ls="--", lw=1.0, alpha=0.7,
                            label=f"Diffusion {bm_key} (mean)")

            # Diffusion lower envelope (colored thick + shaded std)
            dfm = fr["diffusion_frontier_best_mean"]
            dfs = fr["diffusion_frontier_best_std"]
            xs  = [b_grid[i] for i, v in enumerate(dfm) if not math.isnan(v)]
            ys  = [v for v in dfm if not math.isnan(v)]
            es  = [dfs[i] for i, v in enumerate(dfm) if not math.isnan(v)]
            if xs:
                ax.plot(xs, ys, color="steelblue", lw=2.5, ls="-",
                        label="Diffusion (best β_M, mean±std)", zorder=4)
                lo = [max(0.0, y - e) for y, e in zip(ys, es)]
                hi = [y + e for y, e in zip(ys, es)]
                ax.fill_between(xs, lo, hi, color="steelblue", alpha=0.15)

            # Individual warm-start per-config curves (salmon dashed thin)
            for key, curve in fr.get("warm_start_per_config", {}).items():
                xs  = [b_grid[i] for i, v in enumerate(curve) if not math.isnan(v)]
                yss = [v for v in curve if not math.isnan(v)]
                if xs:
                    ax.plot(xs, yss, color="#e8897a", ls="--", lw=0.8, alpha=0.6,
                            label=f"Warm-start {key}")

            # Warm-start lower envelope (crimson thick)
            wfe = fr.get("warm_start_frontier_best", [])
            xs  = [b_grid[i] for i, v in enumerate(wfe) if not math.isnan(v)]
            ys  = [v for v in wfe if not math.isnan(v)]
            if xs:
                ax.plot(xs, ys, color="crimson", lw=2.5, ls="-",
                        label="Warm-start (envelope)", zorder=4)

            # Vertical dashed line at C_setup
            ax.axvline(C_SETUP, color="steelblue", ls=":", lw=1.2, alpha=0.7,
                       label=f"C_setup = {C_SETUP:,}")

            ax.set_xscale("log")
            ax.set_xlabel("Oracle cost", fontsize=9)
            ax.set_ylabel("Cumulative best energy", fontsize=9)
            ax.set_title(f"{ENERGY}  d={dim}  β_H={beta_h}  Budget frontier", fontsize=10)
            ax.legend(fontsize=6, loc="upper right", ncol=2)
            ax.grid(True, alpha=0.3)

            out = plots_dir / f"frontier_d{dim}_bH{beta_h:g}.svg"
            fig.tight_layout()
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Budget frontier: amortised diffusion vs direct SMC restarts."
    )
    parser.add_argument("--extend-diffusion", action="store_true",
                        help="Run R=1000 diffusion batches from existing models")
    parser.add_argument("--run-direct",       action="store_true",
                        help="Run direct SMC restarts up to B_MAX")
    parser.add_argument("--run-warm-start",   action="store_true",
                        help="Run warm-start ULA-SMC restarts (ULA→β_M then SMC→β_H)")
    parser.add_argument("--aggregate",        action="store_true",
                        help="Build frontier JSONs + threshold table (no GPU needed)")
    parser.add_argument("--threshold-only",   action="store_true",
                        help="Rebuild only threshold_table.csv from existing data files")
    parser.add_argument("--plot-only",        action="store_true",
                        help="Plot from existing frontier JSONs")
    parser.add_argument("--no-plot",          action="store_true",
                        help="Skip plots (use on server)")
    parser.add_argument("--seed",             type=int, default=None,
                        help="Single seed for --extend-diffusion (run 3 in parallel)")
    parser.add_argument("--overwrite",        action="store_true",
                        help="Force redo existing output files")
    args = parser.parse_args()

    if args.plot_only:
        plot_frontiers()
        return

    if args.threshold_only:
        print("Rebuilding threshold table from existing data files...")
        _build_threshold_table(EXP_BASE / ENERGY)
        return

    if args.aggregate:
        print("Building aggregate frontier...")
        build_aggregate(no_plot=args.no_plot)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)
    print(f"Device: {device}")

    if args.extend_diffusion:
        seeds = [args.seed] if args.seed is not None else SEEDS
        print(f"\n=== EXTEND DIFFUSION  {ENERGY}  dims={DIMS}  β_M={DIFFUSION_BETA_MS}  "
              f"β_H={BETA_HS}  seeds={seeds}  R={R_MAX_EXTENDED} ===\n")
        for seed in seeds:
            print(f"\n{'='*60}\n  SEED {seed}\n{'='*60}")
            extend_diffusion(seed, args.overwrite, device)
        print(f"\nDone. Run --aggregate when all seeds complete.")
        return

    if args.run_direct:
        print(f"\n=== RUN DIRECT SMC  {ENERGY}  dims={DIMS}  β_H={BETA_HS}  "
              f"B_MAX={B_MAX:,} ===\n")
        run_direct(args.overwrite, device)
        print(f"\nDone. Run --aggregate when diffusion seeds also complete.")
        return

    if args.run_warm_start:
        print(f"\n=== RUN WARM-START  {ENERGY}  dims={DIMS}  β_M={WARM_BETA_MS}  "
              f"β_H={BETA_HS}  N={WARM_N_PARTICLES}  T_ULA={WARM_T_ULA} ===\n")
        run_warm_start(args.overwrite, device)
        print(f"\nDone. Run --aggregate to rebuild frontiers.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
