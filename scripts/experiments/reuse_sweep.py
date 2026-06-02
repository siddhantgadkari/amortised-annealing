#!/usr/bin/env python
"""Reuse sweep: run Algorithm 2 (diffusion sample + ULA polish) R_max times.

For each selected config, generates R_MAX independent inference batches from
the same trained model, accumulating:
  cumulative_best(R) = min energy seen across all R batches
  cumulative_q01(R)  = q01 over all R*N_s pooled energy values (exact)

Answers: "If I keep reusing the same trained model, how does best-found sample improve?"

Raw energy tensors are saved as .pt (not in JSON). The summary JSON stores per-batch
scalars and cumulative stats only.

Usage:
    # Per-seed on server (run in parallel via tmux):
    uv run python scripts/experiments/reuse_sweep.py --seed 0 --no-plot
    uv run python scripts/experiments/reuse_sweep.py --seed 1 --no-plot
    uv run python scripts/experiments/reuse_sweep.py --seed 2 --no-plot

    # Aggregate + plot:
    uv run python scripts/experiments/reuse_sweep.py --aggregate --no-plot
    uv run python scripts/experiments/reuse_sweep.py --plot-only
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from amortised_annealing.diffusion import VPSchedule, MLPScore, ReverseSDE
from amortised_annealing.energies import Ackley, DoubleWell, ManyWell, Rastrigin
from amortised_annealing.sampling import DiffusionModelSampler

ROOT             = Path(__file__).parent.parent.parent
SAMPLE_DIR       = ROOT / "data" / "samples"
MODEL_DIR        = ROOT / "data" / "models"
EXP_BASE         = ROOT / "data" / "experiments" / "reuse_sweep"
COST_QUALITY_AGG = ROOT / "data" / "experiments" / "cost_quality" / "ackley" / "aggregate.json"

ENERGY_MAP = {"ackley": Ackley, "double_well": DoubleWell, "many_well": ManyWell,
              "rastrigin": Rastrigin}
ENERGY = "ackley"

# ===========================================================================
# EXPERIMENT CONFIGURATION
# ===========================================================================
SWEEP_CONFIGS = [
    # (dim, beta_m, beta_h, n_train)
    (10,  5.0, 20.0, 2048),
    (10, 20.0, 20.0, 2048),
    (10,  5.0, 50.0, 2048),
    (10, 20.0, 50.0, 2048),
    (20,  5.0, 20.0, 2048),
    (20, 20.0, 20.0, 2048),
    (20,  5.0, 50.0, 2048),
    (20, 20.0, 50.0, 2048),
]
N_SAMPLES   = 8192
LOCAL_STEPS = 10
R_MAX       = 100
SEEDS       = [0, 1, 2]

# Must match cost_quality.py model training settings
MODEL_HIDDEN_DIMS    = [256, 256, 256]
MODEL_TIME_EMBED_DIM = 64
MCMC_N_PARTICLES     = 8192
MCMC_N_STEPS         = 10_000
LOCAL_ULA_STEP_SIZE  = 1e-3
MODEL_SAMPLE_N_STEPS = 500
MODEL_SAMPLE_T_START = 1.0
MODEL_SAMPLE_T_END   = 1e-3
# ===========================================================================


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


def _load_model_run(model_run: str, device: torch.device):
    """Returns (reverse_sde, energy_obj, dim, beta_m). Mirrors cost_quality.py."""
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
        dim=dim,
        hidden_dims=tuple(mc["hidden_dims"]),
        time_embed_dim=mc.get("time_embed_dim", 64),
        activation=mc.get("activation", "silu"),
        predict_score=mc.get("predict_score", False),
    )
    state = torch.load(run_dir / "ema_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    sc = cfg["schedule"]
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


def _run_name(dim: int, beta_m: float, beta_h: float, n_train: int) -> str:
    return f"d{dim}_bm{_beta_str(beta_m)}_bh{_beta_str(beta_h)}_ntrain{n_train}"


def run_reuse_sweep(
    dim: int, beta_m: float, beta_h: float, n_train: int, seed: int, device: torch.device
) -> None:
    """Run R_MAX independent Algorithm 2 calls; save raw .pt files and summary JSON."""
    run_name  = _run_name(dim, beta_m, beta_h, n_train)
    out_dir   = EXP_BASE / ENERGY
    raw_dir   = out_dir / "raw"
    json_path = out_dir / f"{run_name}_seed{seed}.json"

    if json_path.exists():
        print(f"  REUSE  {run_name} seed={seed}")
        return

    sample_run = _sample_run_name(dim, beta_m, seed)
    model_run  = _model_run_name_ntrain(sample_run, n_train, seed)
    model_path = MODEL_DIR / model_run / "ema_model.pt"

    if not model_path.exists():
        print(f"  SKIP   {model_run} — ema_model.pt not found")
        return

    print(f"  RUN    {run_name} seed={seed}")
    rsde, energy, _dim, _beta_m = _load_model_run(model_run, device)
    energy_fn = energy.energy

    c_setup = n_train * MCMC_N_STEPS          # oracle cost paid once (Algorithm 1)
    c_use   = N_SAMPLES * LOCAL_STEPS          # oracle cost per call (Algorithm 2)

    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_best: list[float] = []
    batch_q01:  list[float] = []
    cumulative_best: list[float] = []
    cumulative_q01:  list[float] = []
    all_energies: list[torch.Tensor] = []      # kept in CPU memory for exact cumulative stats

    t0 = time.perf_counter()
    for r in range(R_MAX):
        batch_seed = seed * 1000 + r
        torch.manual_seed(batch_seed)

        # Diffusion sample (Algorithm 2, step 1)
        with torch.no_grad():
            x = DiffusionModelSampler(
                rsde,
                n_steps=MODEL_SAMPLE_N_STEPS,
                t_start=MODEL_SAMPLE_T_START,
                t_end=MODEL_SAMPLE_T_END,
            ).sample(N_SAMPLES, device, show_progress=False)

        # Local ULA polish (Algorithm 2, step 2)
        x = _apply_local_ula(x, energy_fn, beta_h, LOCAL_STEPS, batch_seed)

        # Compute energies on CPU
        with torch.no_grad():
            e = energy_fn(x.cpu()).float()

        # Save raw energies for later recomputation if needed
        torch.save(e, raw_dir / f"{run_name}_seed{seed}_r{r:03d}.pt")

        # Per-batch scalars
        batch_best.append(float(e.min()))
        batch_q01.append(float(torch.quantile(e, 0.01)))

        # Exact cumulative stats from pooled samples
        all_energies.append(e)
        pooled = torch.cat(all_energies)
        cumulative_best.append(float(pooled.min()))
        cumulative_q01.append(float(torch.quantile(pooled, 0.01)))

        if (r + 1) % 10 == 0:
            print(f"    r={r+1:3d}/{R_MAX}  best={cumulative_best[-1]:.4f}  "
                  f"q01={cumulative_q01[-1]:.4f}  elapsed={time.perf_counter()-t0:.1f}s")

    wall = round(time.perf_counter() - t0, 2)

    total_cost = [c_setup + (r + 1) * c_use for r in range(R_MAX)]
    avg_cost   = [c_setup / (r + 1) + c_use  for r in range(R_MAX)]

    summary = {
        "dim": dim, "beta_m": beta_m, "beta_h": beta_h,
        "n_train": n_train, "seed": seed,
        "n_samples": N_SAMPLES, "local_steps": LOCAL_STEPS, "r_max": R_MAX,
        "oracle_cost_setup": c_setup,
        "oracle_cost_per_use": c_use,
        "wall_seconds": wall,
        "batch_best":       batch_best,
        "batch_q01":        batch_q01,
        "cumulative_best":  cumulative_best,
        "cumulative_q01":   cumulative_q01,
        "total_cost":       total_cost,
        "avg_cost":         avg_cost,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {json_path.name}  (wall={wall:.1f}s  "
          f"final_best={cumulative_best[-1]:.4f})")


def aggregate_reuse() -> None:
    """Aggregate per-seed JSON files into per-config mean/std arrays."""
    in_dir   = EXP_BASE / ENERGY
    agg_path = in_dir / "aggregate.json"
    out: list[dict] = []

    for dim, beta_m, beta_h, n_train in SWEEP_CONFIGS:
        run_name = _run_name(dim, beta_m, beta_h, n_train)
        seed_data = []
        for seed in SEEDS:
            p = in_dir / f"{run_name}_seed{seed}.json"
            if p.exists():
                seed_data.append(json.loads(p.read_text()))

        if not seed_data:
            print(f"  No data for {run_name}")
            continue

        def _agg_list(key: str) -> tuple[list[float], list[float]]:
            arrays = [d[key] for d in seed_data if key in d]
            n = min(len(a) for a in arrays)
            means = [sum(a[r] for a in arrays) / len(arrays) for r in range(n)]
            stds: list[float] = []
            for r in range(n):
                vals = [a[r] for a in arrays]
                mu  = sum(vals) / len(vals)
                var = sum((v - mu) ** 2 for v in vals) / max(len(vals) - 1, 1)
                stds.append(var ** 0.5)
            return means, stds

        cb_mean, cb_std = _agg_list("cumulative_best")
        cq_mean, cq_std = _agg_list("cumulative_q01")
        ac_mean, ac_std = _agg_list("avg_cost")
        tc_mean, _      = _agg_list("total_cost")

        out.append({
            "dim": dim, "beta_m": beta_m, "beta_h": beta_h, "n_train": n_train,
            "n_seeds": len(seed_data),
            "oracle_cost_setup":    seed_data[0]["oracle_cost_setup"],
            "oracle_cost_per_use":  seed_data[0]["oracle_cost_per_use"],
            "cumulative_best_mean": cb_mean, "cumulative_best_std": cb_std,
            "cumulative_q01_mean":  cq_mean, "cumulative_q01_std":  cq_std,
            "avg_cost_mean":        ac_mean, "avg_cost_std":        ac_std,
            "total_cost":           tc_mean,
        })
        print(f"  Aggregated {run_name}: {len(seed_data)} seeds")

    with open(agg_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {agg_path}  ({len(out)} configs)")


def plot_reuse() -> None:
    agg_path = EXP_BASE / ENERGY / "aggregate.json"
    if not agg_path.exists():
        print(f"No aggregate.json at {agg_path} — run --aggregate first.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = EXP_BASE / ENERGY / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    agg      = json.loads(agg_path.read_text())
    cq_data  = json.loads(COST_QUALITY_AGG.read_text()) if COST_QUALITY_AGG.exists() else None

    dims    = sorted(set(c["dim"]    for c in agg))
    beta_hs = sorted(set(c["beta_h"] for c in agg))
    bm_colors = {5.0: "darkorange", 20.0: "mediumseagreen"}

    def _cq_m(d):
        return d["mean"] if isinstance(d, dict) else (d if d is not None else float("nan"))

    for dim in dims:
        for beta_h in beta_hs:
            configs = [c for c in agg if c["dim"] == dim and c["beta_h"] == beta_h]
            if not configs:
                continue

            R_range = list(range(1, R_MAX + 1))

            # --- Plot A: total_cost vs cumulative_best ---
            fig, ax = plt.subplots(figsize=(7, 5))
            for cfg in configs:
                cb_mean = cfg.get("cumulative_best_mean", [])
                cb_std  = cfg.get("cumulative_best_std", [])
                tc      = cfg.get("total_cost", [])
                if not cb_mean:
                    continue
                color = bm_colors.get(cfg["beta_m"], "steelblue")
                label = f"β_M={cfg['beta_m']:g} N_tr={cfg['n_train']}"
                ax.plot(tc, cb_mean, color=color, lw=1.5, label=label)
                lo = [m - s for m, s in zip(cb_mean, cb_std)]
                hi = [m + s for m, s in zip(cb_mean, cb_std)]
                ax.fill_between(tc, lo, hi, color=color, alpha=0.15)

            if cq_data:
                direct_pts = [
                    p for p in cq_data["direct_ula_annealed"]
                    if p["dim"] == dim and p["beta_h"] == beta_h
                ]
                for pt in direct_pts:
                    x_val = pt.get("oracle_cost_total", pt.get("n_grad_evals", 0))
                    y_val = _cq_m(pt.get("best_energy", {}))
                    ax.scatter([x_val], [y_val], color="black", marker="^", zorder=5, s=50)
                if direct_pts:
                    ax.scatter([], [], color="black", marker="^", s=50,
                               label="Direct SMC (annealed)")

            ax.set_xscale("log")
            ax.set_xlabel("Total oracle cost  (C_setup + R × C_use)", fontsize=9)
            ax.set_ylabel("Cumulative best energy", fontsize=9)
            ax.set_title(f"Reuse quality: d={dim}  β_H={beta_h}", fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            out_path = plots_dir / f"reuse_cost_quality_d{dim}_bH{beta_h:g}.svg"
            fig.tight_layout()
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {out_path.name}")

            # --- Plot B: avg_cost_per_use vs R ---
            fig, ax = plt.subplots(figsize=(7, 5))
            for cfg in configs:
                ac_mean = cfg.get("avg_cost_mean", [])
                if not ac_mean:
                    continue
                color = bm_colors.get(cfg["beta_m"], "steelblue")
                ls    = "-" if cfg["n_train"] == 2048 else "--"
                label = f"β_M={cfg['beta_m']:g} N_tr={cfg['n_train']}"
                ax.plot(R_range[:len(ac_mean)], ac_mean, color=color, ls=ls, lw=1.5,
                        label=label)

            if cq_data:
                direct_pts = sorted(
                    [p for p in cq_data["direct_ula_annealed"]
                     if p["dim"] == dim and p["beta_h"] == beta_h],
                    key=lambda p: p.get("oracle_cost_total", 0),
                )
                for pt in direct_pts:
                    c = pt.get("oracle_cost_total", pt.get("n_grad_evals", 0))
                    ax.axhline(c, color="black", ls=":", lw=0.8, alpha=0.6,
                               label=f"Direct SMC  cost={c:.2e}")

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Number of reuses R", fontsize=9)
            ax.set_ylabel("Avg oracle cost per call  (C_setup/R + C_use)", fontsize=9)
            ax.set_title(f"Break-even view: d={dim}  β_H={beta_h}", fontsize=10)
            ax.legend(fontsize=6, ncol=2)
            ax.grid(True, alpha=0.3)
            out_path = plots_dir / f"reuse_avg_cost_d{dim}_bH{beta_h:g}.svg"
            fig.tight_layout()
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reuse sweep: run Algorithm 2 R_max times.")
    parser.add_argument("--seed",      type=int, default=None, help="Run single seed")
    parser.add_argument("--aggregate", action="store_true",    help="Aggregate seed files")
    parser.add_argument("--plot-only", action="store_true",    help="Plot from existing aggregate")
    parser.add_argument("--no-plot",   action="store_true",    help="Skip plotting")
    args = parser.parse_args()

    if args.plot_only:
        plot_reuse()
        return

    if args.aggregate:
        print("Aggregating...")
        aggregate_reuse()
        if not args.no_plot:
            plot_reuse()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)

    seeds = [args.seed] if args.seed is not None else SEEDS
    print(f"=== REUSE SWEEP  {ENERGY}  seeds={seeds}  R_max={R_MAX} ===\n")

    for seed in seeds:
        print(f"\n{'='*60}\n  SEED {seed}\n{'='*60}")
        for dim, beta_m, beta_h, n_train in SWEEP_CONFIGS:
            run_reuse_sweep(dim, beta_m, beta_h, n_train, seed, device)

    if args.seed is None:
        print("\nAggregating...")
        aggregate_reuse()
        if not args.no_plot:
            plot_reuse()
    else:
        print(f"\nSeed {args.seed} done. Run --aggregate when all seeds complete.")


if __name__ == "__main__":
    main()
