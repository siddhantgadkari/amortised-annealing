#!/usr/bin/env python
"""Sweep DIFF_T_START for diffusion SMC across (energy, dim, beta_m, seed).

Reuses existing models and MCMC samples from energy_betaM_experiment.py —
no sampling or training is run here. Only the annealing step is repeated
for each t_start value, plus a ULA baseline once per (dim, beta_m).

Results are stored separately from the main experiment in:
    data/experiments/tstart_sweep/{energy}[_{desc}]/

Usage:
    uv run python scripts/experiments/tstart_sweep.py
    uv run python scripts/experiments/tstart_sweep.py --seed 0
    uv run python scripts/experiments/tstart_sweep.py --energy rastrigin --seed 1
    uv run python scripts/experiments/tstart_sweep.py --aggregate
    uv run python scripts/experiments/tstart_sweep.py --plot-only
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from amortised_annealing.diffusion import VPSchedule, MLPScore, ReverseSDE
from amortised_annealing.energies import DoubleWell, ManyWell, Ackley, Rastrigin
from amortised_annealing.smc import (
    ParticleCloud,
    SMCDiagnostics,
    SMCSampler,
    ULAProposal,
    DiffusionAnnealingProposal,
)

ROOT       = Path(__file__).parent.parent.parent
SAMPLE_DIR = ROOT / "data" / "samples"
MODEL_DIR  = ROOT / "data" / "models"
SWEEP_BASE = ROOT / "data" / "experiments" / "tstart_sweep"

ENERGY_MAP = {
    "double_well": DoubleWell,
    "many_well":   ManyWell,
    "ackley":      Ackley,
    "rastrigin":   Rastrigin,
}

# ===========================================================================
# USER CONFIGURATION *
# Must match the settings used in energy_betaM_experiment.py so that model
# and sample run names resolve correctly.
# ===========================================================================
ENERGY  = "rastrigin"
DIMS    = [2, 5, 10, 20]
BETA_MS = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
BETA_H  = 10.0
SEEDS   = [0, 1, 2]

# Values of t_start to sweep
T_START_VALUES = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

# Model architecture — must match the trained models
MODEL_HIDDEN_DIMS    = [256, 256, 256]
MODEL_TIME_EMBED_DIM = 64
LOSS_TYPE            = "eps"

# Annealing — shared
N_PARTICLES   = 2048
N_SMC_STEPS   = 50
ESS_THRESHOLD = 0.5

# Annealing — ULA baseline
N_ULA_STEPS   = 10
ULA_STEP_SIZE = 1e-3

# Annealing — diffusion SMC (t_start is swept; these are fixed)
N_DIFFUSION_STEPS  = 10
DIFF_T_END         = 1e-3
DIFF_SCORE_SCALING = True

# Reuse flags
REUSE_ULA  = True  # skip (dim, beta_m, seed) if ULA result already saved
REUSE_DIFF = True  # skip (dim, beta_m, t_start, seed) if result already saved

DESC = ""  # optional tag appended to sweep dir name
# ===========================================================================


# ---------------------------------------------------------------------------
# Naming helpers (must produce same names as energy_betaM_experiment.py)
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


def _sweep_dir() -> Path:
    tag = ENERGY if not DESC else f"{ENERGY}_{DESC}"
    return SWEEP_BASE / tag


# ---------------------------------------------------------------------------
# Model loading (mirrors energy_betaM_experiment._load_model)
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
    model = MLPScore(
        dim            = dim,
        hidden_dims    = tuple(mc["hidden_dims"]),
        time_embed_dim = mc.get("time_embed_dim", 64),
        activation     = mc.get("activation", "silu"),
        predict_score  = mc.get("predict_score", False),
    )
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
    return {
        "mean_energy":   round(e.mean().item(),   4),
        "min_energy":    round(e.min().item(),     4),
        "std_energy":    round(e.std().item(),     4),
        "median_energy": round(e.median().item(),  4),
    }


def _compute_diag_stats(diag: SMCDiagnostics) -> dict:
    ess  = diag.ess_ratios
    stds = diag.log_weight_stds
    return {
        "min_ess":      round(min(ess),  4) if ess  else None,
        "mean_ess":     round(sum(ess) / len(ess), 4) if ess else None,
        "max_logw_var": round(max(s**2 for s in stds), 4) if stds else None,
    }


def _aggregate_scalar(vals: list[float]) -> dict:
    return {
        "mean": round(sum(vals) / len(vals), 6),
        "std":  round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 6),
    }


# ---------------------------------------------------------------------------
# Per-seed run
# ---------------------------------------------------------------------------

def run_seed(seed: int, device: torch.device) -> None:
    out_path = _sweep_dir() / f"results_seed{seed}.json"

    ula_done:  dict[tuple, dict] = {}  # (dim, beta_m) -> result
    diff_done: dict[tuple, dict] = {}  # (dim, beta_m, t_start) -> result

    if out_path.exists():
        existing = json.loads(out_path.read_text())
        if REUSE_ULA:
            for r in existing.get("ula_runs", []):
                ula_done[(r["dim"], r["beta_m"])] = r
        if REUSE_DIFF:
            for r in existing.get("diffusion_runs", []):
                diff_done[(r["dim"], r["beta_m"], r["t_start"])] = r

    ula_runs  = list(ula_done.values())
    diff_runs = list(diff_done.values())

    def _save() -> None:
        _sweep_dir().mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(
            {"seed": seed, "ula_runs": ula_runs, "diffusion_runs": diff_runs},
            indent=2,
        ))

    for dim in DIMS:
        print(f"\n  dim={dim}  seed={seed}")
        for beta_m in BETA_MS:
            sample_run = _sample_run_name(dim, beta_m, seed)
            model_run  = _model_run_name(sample_run, seed)

            model_path  = MODEL_DIR  / model_run  / "ema_model.pt"
            sample_path = SAMPLE_DIR / sample_run / "particles.pt"

            if not model_path.exists():
                print(f"    SKIP  {model_run}  (model not found)")
                continue
            if not sample_path.exists():
                print(f"    SKIP  {sample_run}  (samples not found)")
                continue

            rsde, energy_fn, _, _, _ = _load_model(model_run, device)

            # ULA initial cloud: MCMC particles at β_M
            pts = torch.load(sample_path, map_location=device, weights_only=True)
            x0_mcmc = pts  # [N, dim]
            if x0_mcmc.shape[0] > N_PARTICLES:
                x0_mcmc = x0_mcmc[torch.randperm(x0_mcmc.shape[0], device=device)[:N_PARTICLES]]
            ula_initial_cloud = ParticleCloud(x0_mcmc, torch.zeros(N_PARTICLES, device=device))

            # Diffusion SMC initial cloud: model samples at β_M
            model_samples_path = MODEL_SAMPLE_DIR / model_run / "samples.pt"
            if not model_samples_path.exists():
                print(f"    SKIP  {model_run}  (model samples not found)")
                continue
            x0_model = torch.load(model_samples_path, map_location=device, weights_only=True)
            if x0_model.shape[0] > N_PARTICLES:
                x0_model = x0_model[torch.randperm(x0_model.shape[0], device=device)[:N_PARTICLES]]
            diff_initial_cloud = ParticleCloud(x0_model, torch.zeros(N_PARTICLES, device=device))

            beta_ladder = torch.linspace(beta_m, BETA_H, N_SMC_STEPS + 1, device=device)

            # ------------------------------------------------------------------
            # ULA baseline — run once per (dim, beta_m)
            # ------------------------------------------------------------------
            ula_key = (dim, beta_m)
            if ula_key not in ula_done:
                print(f"    [ula]  RUN    dim={dim} beta_m={beta_m}")
                torch.manual_seed(seed)
                t0 = time.perf_counter()
                ula_p = ULAProposal(energy_fn, step_size=ULA_STEP_SIZE, n_steps=N_ULA_STEPS)
                ula_cloud, ula_diag = SMCSampler(
                    ula_p.mutation_kernel, ula_p.weight_update, energy_fn,
                    ess_threshold=ESS_THRESHOLD,
                ).run(ula_initial_cloud, beta_ladder, show_progress=True)
                ula_result = {
                    "dim": dim, "beta_m": beta_m,
                    **_energy_stats(ula_cloud.x, energy_fn),
                    "final_ess":    round(ula_cloud.ess_ratio(), 4),
                    **_compute_diag_stats(ula_diag),
                    "n_resamples":  ula_diag.n_resamples,
                    "wall_seconds": round(time.perf_counter() - t0, 2),
                }
                ula_done[ula_key] = ula_result
                ula_runs.append(ula_result)
                _save()
            else:
                print(f"    [ula]  REUSE  dim={dim} beta_m={beta_m}")

            # ------------------------------------------------------------------
            # Diffusion SMC — once per t_start
            # ------------------------------------------------------------------
            for t_start in T_START_VALUES:
                diff_key = (dim, beta_m, t_start)
                if diff_key in diff_done:
                    print(f"    [diff] REUSE  dim={dim} beta_m={beta_m} t_start={t_start}")
                    continue
                print(f"    [diff] RUN    dim={dim} beta_m={beta_m} t_start={t_start}")
                torch.manual_seed(seed)
                t0 = time.perf_counter()
                diff_p = DiffusionAnnealingProposal(
                    rsde, energy_fn, beta_train=beta_m,
                    n_diffusion_steps=N_DIFFUSION_STEPS,
                    t_start=t_start, t_end=DIFF_T_END,
                    score_scaling=DIFF_SCORE_SCALING,
                )
                diff_cloud, diff_diag = SMCSampler(
                    diff_p.mutation_kernel, diff_p.weight_update, energy_fn,
                    ess_threshold=ESS_THRESHOLD,
                ).run(diff_initial_cloud, beta_ladder, show_progress=True)
                diff_result = {
                    "dim": dim, "beta_m": beta_m, "t_start": t_start,
                    **_energy_stats(diff_cloud.x, energy_fn),
                    "final_ess":    round(diff_cloud.ess_ratio(), 4),
                    **_compute_diag_stats(diff_diag),
                    "n_resamples":  diff_diag.n_resamples,
                    "wall_seconds": round(time.perf_counter() - t0, 2),
                }
                diff_done[diff_key] = diff_result
                diff_runs.append(diff_result)
                _save()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results() -> None:
    from collections import defaultdict

    ula_groups:  dict[tuple, list[dict]] = defaultdict(list)
    diff_groups: dict[tuple, list[dict]] = defaultdict(list)

    for seed in SEEDS:
        path = _sweep_dir() / f"results_seed{seed}.json"
        if not path.exists():
            print(f"  WARNING: missing results_seed{seed}.json — skipping")
            continue
        data = json.loads(path.read_text())
        for r in data.get("ula_runs", []):
            ula_groups[(r["dim"], r["beta_m"])].append(r)
        for r in data.get("diffusion_runs", []):
            diff_groups[(r["dim"], r["beta_m"], r["t_start"])].append(r)

    SCALAR_KEYS = [
        "mean_energy", "min_energy", "std_energy", "median_energy",
        "final_ess", "min_ess", "mean_ess", "max_logw_var",
        "n_resamples", "wall_seconds",
    ]

    def _agg(groups: dict, id_keys: list[str]) -> list[dict]:
        result = []
        for key, runs in sorted(groups.items()):
            entry = dict(zip(id_keys, key))
            for sk in SCALAR_KEYS:
                vals = [r[sk] for r in runs if sk in r and isinstance(r[sk], (int, float))]
                if vals:
                    entry[sk] = _aggregate_scalar(vals)
            result.append(entry)
        return result

    aggregate = {
        "ula_runs":       _agg(ula_groups,  ["dim", "beta_m"]),
        "diffusion_runs": _agg(diff_groups, ["dim", "beta_m", "t_start"]),
    }

    _sweep_dir().mkdir(parents=True, exist_ok=True)
    (_sweep_dir() / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
    print(f"Aggregate saved to {_sweep_dir() / 'aggregate.json'}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results() -> None:
    import matplotlib.pyplot as plt

    agg_path = _sweep_dir() / "aggregate.json"
    if not agg_path.exists():
        print("No aggregate.json found. Run --aggregate first.")
        return

    aggregate = json.loads(agg_path.read_text())

    ula_index: dict[tuple, dict] = {}
    for r in aggregate["ula_runs"]:
        ula_index[(r["dim"], r["beta_m"])] = r

    diff_index: dict[tuple, dict] = {}
    for r in aggregate["diffusion_runs"]:
        diff_index[(r["dim"], r["beta_m"], r["t_start"])] = r

    plots_dir = _sweep_dir() / "plots"
    plots_dir.mkdir(exist_ok=True)

    for dim in DIMS:
        beta_ms_in_data = sorted({
            r["beta_m"] for r in aggregate["diffusion_runs"] if r["dim"] == dim
        })
        if not beta_ms_in_data:
            continue

        ncols = 4
        nrows = (len(beta_ms_in_data) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.5 * nrows), squeeze=False)

        for idx, beta_m in enumerate(beta_ms_in_data):
            ax = axes[idx // ncols][idx % ncols]

            t_starts = sorted({
                r["t_start"] for r in aggregate["diffusion_runs"]
                if r["dim"] == dim and r["beta_m"] == beta_m
            })
            if not t_starts:
                ax.set_visible(False)
                continue

            def _get(metric: str, t: float) -> tuple[float, float]:
                v = diff_index.get((dim, beta_m, t), {}).get(metric, {})
                return v.get("mean", np.nan), v.get("std", 0.0)

            means, stds = zip(*[_get("mean_energy", t) for t in t_starts])
            ax.plot(t_starts, means, "o-", color="darkorange", lw=1.5, label="Diffusion SMC")
            ax.fill_between(
                t_starts,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                color="darkorange", alpha=0.2,
            )

            ula = ula_index.get((dim, beta_m))
            if ula and "mean_energy" in ula:
                ula_mean = ula["mean_energy"]["mean"]
                ula_std  = ula["mean_energy"]["std"]
                ax.axhline(ula_mean, color="steelblue", linestyle="--", lw=1.5, label="ULA SMC")
                ax.axhspan(ula_mean - ula_std, ula_mean + ula_std, alpha=0.1, color="steelblue")

            ax.set_xscale("log")
            ax.set_xlabel("t_start")
            ax.set_ylabel("Mean energy")
            ax.set_title(f"β_M={beta_m}", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        for idx in range(len(beta_ms_in_data), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        n_seeds = len(SEEDS)
        fig.suptitle(
            f"{ENERGY}  d={dim}  {n_seeds} seeds  mean energy vs t_start  ±1σ",
            fontsize=10,
        )
        fig.tight_layout()
        out = plots_dir / f"d{dim}_tstart_sweep.svg"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved to {out}")

    # Summary plot: one panel per dim, best t_start performance vs ULA
    _plot_summary(aggregate, plots_dir)


def _plot_summary(aggregate: dict, plots_dir: Path) -> None:
    """One figure: for each dim, mean_energy of best t_start vs ULA across beta_ms."""
    import matplotlib.pyplot as plt

    diff_index: dict[tuple, dict] = {}
    for r in aggregate["diffusion_runs"]:
        diff_index[(r["dim"], r["beta_m"], r["t_start"])] = r

    ula_index: dict[tuple, dict] = {}
    for r in aggregate["ula_runs"]:
        ula_index[(r["dim"], r["beta_m"])] = r

    fig, axes = plt.subplots(1, len(DIMS), figsize=(4 * len(DIMS), 4), squeeze=False)
    axes = axes[0]

    for ax, dim in zip(axes, DIMS):
        beta_ms_in_data = sorted({
            r["beta_m"] for r in aggregate["diffusion_runs"] if r["dim"] == dim
        })
        if not beta_ms_in_data:
            ax.set_visible(False)
            continue

        # Best-t_start mean_energy for each beta_m (min across t_starts)
        best_means, best_stds = [], []
        ula_means,  ula_stds  = [], []

        for beta_m in beta_ms_in_data:
            t_starts_here = sorted({
                r["t_start"] for r in aggregate["diffusion_runs"]
                if r["dim"] == dim and r["beta_m"] == beta_m
            })
            candidates = [
                diff_index.get((dim, beta_m, t), {}).get("mean_energy", {})
                for t in t_starts_here
            ]
            valid = [(c.get("mean", np.nan), c.get("std", 0.0)) for c in candidates if "mean" in c]
            if valid:
                best_m, best_s = min(valid, key=lambda x: x[0])
            else:
                best_m, best_s = np.nan, 0.0
            best_means.append(best_m)
            best_stds.append(best_s)

            ula = ula_index.get((dim, beta_m), {})
            ula_me = ula.get("mean_energy", {})
            ula_means.append(ula_me.get("mean", np.nan))
            ula_stds.append(ula_me.get("std", 0.0))

        ax.plot(beta_ms_in_data, best_means, "o-", color="darkorange", lw=1.5, label="Diffusion (best t)")
        ax.fill_between(beta_ms_in_data,
                        [m - s for m, s in zip(best_means, best_stds)],
                        [m + s for m, s in zip(best_means, best_stds)],
                        color="darkorange", alpha=0.2)
        ax.plot(beta_ms_in_data, ula_means, "s--", color="steelblue", lw=1.5, label="ULA SMC")
        ax.fill_between(beta_ms_in_data,
                        [m - s for m, s in zip(ula_means, ula_stds)],
                        [m + s for m, s in zip(ula_means, ula_stds)],
                        color="steelblue", alpha=0.15)

        ax.set_xlabel(r"$\beta_M$")
        ax.set_ylabel("Mean energy")
        ax.set_title(f"d={dim}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{ENERGY}  best t_start vs ULA  ±1σ", fontsize=10)
    fig.tight_layout()
    out = plots_dir / "summary_best_tstart.svg"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Summary plot saved to {out}")


# ---------------------------------------------------------------------------
# Config snapshot
# ---------------------------------------------------------------------------

def _config_snapshot() -> dict:
    return {
        "energy":         ENERGY,
        "dims":           DIMS,
        "beta_ms":        BETA_MS,
        "beta_h":         BETA_H,
        "seeds":          SEEDS,
        "t_start_values": T_START_VALUES,
        "desc":           DESC,
        "model": {
            "hidden_dims":    MODEL_HIDDEN_DIMS,
            "time_embed_dim": MODEL_TIME_EMBED_DIM,
            "loss_type":      LOSS_TYPE,
        },
        "annealing": {
            "n_particles":   N_PARTICLES,
            "n_smc_steps":   N_SMC_STEPS,
            "ess_threshold": ESS_THRESHOLD,
            "ula": {"n_steps": N_ULA_STEPS, "step_size": ULA_STEP_SIZE},
            "diffusion": {
                "n_steps":       N_DIFFUSION_STEPS,
                "t_end":         DIFF_T_END,
                "score_scaling": DIFF_SCORE_SCALING,
            },
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global ENERGY

    parser = argparse.ArgumentParser(
        description="Sweep DIFF_T_START for diffusion SMC, reusing existing models/samples."
    )
    parser.add_argument(
        "--energy",
        choices=list(ENERGY_MAP),
        default=None,
        help="Override ENERGY",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run a single seed only; run --aggregate when all seeds done",
    )
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate all seed results and plot")
    parser.add_argument("--plot-only", action="store_true",
                        help="Re-plot from existing aggregate.json")
    args = parser.parse_args()

    if args.energy is not None:
        ENERGY = args.energy

    if args.plot_only:
        plot_results()
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

    sweep_dir = _sweep_dir()
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "config.json").write_text(json.dumps(_config_snapshot(), indent=2))

    print(f"=== t_start sweep  {ENERGY}  dims={DIMS}  seeds={seeds_to_run} ===")
    print(f"    t_starts={T_START_VALUES}")
    print(f"    beta_Ms={BETA_MS}  beta_H={BETA_H}  device={device}\n")

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
