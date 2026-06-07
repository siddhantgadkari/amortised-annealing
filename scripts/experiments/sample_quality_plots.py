#!/usr/bin/env python
"""Plot: MCMC sample energy vs diffusion model sample energy across beta_M.

For each energy, produces a figure with:
  - Row 1: min energy of raw samples (q01 as proxy for best-of-cloud)
  - Row 2: mean energy of raw samples
  One column per dimension. ULA MCMC samples vs diffusion model samples.

This shows how well the diffusion model approximates the training distribution,
and whether coverage (not just concentration) is sufficient for downstream use.

Usage:
    uv run python scripts/experiments/sample_quality_plots.py
    uv run python scripts/experiments/sample_quality_plots.py --energy ackley
    uv run python scripts/experiments/sample_quality_plots.py --metric min mean q01
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT     = Path(__file__).parent.parent.parent
EXP_BASE = ROOT / "data" / "experiments" / "energy_betaM_experiments"
ENERGIES = ["ackley", "rastrigin", "double_well", "many_well"]
SEEDS    = [0, 1, 2]


def load_sample_stats(energy: str) -> list[dict]:
    """Aggregate mcmc_stats and model_stats across seeds per (dim, beta_m)."""
    groups: dict[tuple, dict] = defaultdict(lambda: {"mcmc": defaultdict(list),
                                                       "model": defaultdict(list)})
    for seed in SEEDS:
        path = EXP_BASE / energy / f"results_seed{seed}.json"
        if not path.exists():
            continue
        for run in json.loads(path.read_text()).get("runs", []):
            key = (run["dim"], run["beta_m"])
            for metric, val in run["mcmc_stats"].items():
                if isinstance(val, (int, float)):
                    groups[key]["mcmc"][metric].append(val)
            for metric, val in run["model_stats"].items():
                if isinstance(val, (int, float)):
                    groups[key]["model"][metric].append(val)

    rows = []
    for (dim, beta_m), g in sorted(groups.items()):
        row = {"energy": energy, "dim": dim, "beta_m": beta_m}
        for source in ("mcmc", "model"):
            for metric, vals in g[source].items():
                if vals:
                    row[f"{source}_{metric}_mean"] = statistics.mean(vals)
                    row[f"{source}_{metric}_std"]  = (statistics.stdev(vals)
                                                       if len(vals) > 1 else 0.0)
        rows.append(row)
    return rows


def plot_energy(energy: str, rows: list[dict], metrics: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows   = [r for r in rows if r["energy"] == energy]
    dims   = sorted(set(r["dim"] for r in rows))
    ndims  = len(dims)
    nrows  = len(metrics)

    MCMC_COLOR  = "steelblue"
    MODEL_COLOR = "darkorange"

    METRIC_LABELS = {
        "min":    "Min energy",
        "mean":   "Mean energy",
        "q01":    "q01 energy  (best 1%)",
        "median": "Median energy",
    }

    fig, axes = plt.subplots(nrows, ndims, figsize=(4.2 * ndims, 3.8 * nrows), squeeze=False)

    for row_idx, metric in enumerate(metrics):
        mcmc_key_m  = f"mcmc_{metric}_energy_mean"  if metric != "q01" else "mcmc_q01_mean"
        mcmc_key_s  = f"mcmc_{metric}_energy_std"   if metric != "q01" else "mcmc_q01_std"
        model_key_m = f"model_{metric}_energy_mean" if metric != "q01" else "model_q01_mean"
        model_key_s = f"model_{metric}_energy_std"  if metric != "q01" else "model_q01_std"

        for col_idx, dim in enumerate(dims):
            ax = axes[row_idx][col_idx]
            subset = sorted([r for r in rows if r["dim"] == dim], key=lambda r: r["beta_m"])
            if not subset:
                ax.set_visible(False)
                continue

            betas      = [r["beta_m"] for r in subset]
            mcmc_vals  = [r.get(mcmc_key_m,  float("nan")) for r in subset]
            mcmc_stds  = [r.get(mcmc_key_s,  0.0)          for r in subset]
            model_vals = [r.get(model_key_m, float("nan")) for r in subset]
            model_stds = [r.get(model_key_s, 0.0)          for r in subset]

            ax.plot(betas, mcmc_vals,  "o-",  color=MCMC_COLOR,  lw=1.5, ms=4,
                    label="ULA MCMC samples  (training data)")
            ax.fill_between(
                betas,
                [v - s for v, s in zip(mcmc_vals,  mcmc_stds)],
                [v + s for v, s in zip(mcmc_vals,  mcmc_stds)],
                color=MCMC_COLOR, alpha=0.15,
            )
            ax.plot(betas, model_vals, "s--", color=MODEL_COLOR, lw=1.5, ms=4,
                    label="Diffusion model samples")
            ax.fill_between(
                betas,
                [v - s for v, s in zip(model_vals, model_stds)],
                [v + s for v, s in zip(model_vals, model_stds)],
                color=MODEL_COLOR, alpha=0.15,
            )

            ax.set_xlabel(r"$\beta_M$", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9)
            if row_idx == 0:
                ax.set_title(f"d = {dim}", fontsize=10)
            ax.grid(True, alpha=0.3)
            if row_idx == 0 and col_idx == ndims - 1:
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        f"{energy.replace('_', ' ').title()}  —  MCMC samples vs diffusion model samples  (3 seeds ±1σ)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    plots_dir = EXP_BASE / energy / "plots"
    plots_dir.mkdir(exist_ok=True)
    out = plots_dir / "sample_quality.svg"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy", choices=ENERGIES, default=None)
    parser.add_argument(
        "--metric", nargs="+",
        choices=["min", "mean", "q01", "median"],
        default=["min", "mean"],
        help="Which energy metrics to plot (default: min mean)",
    )
    args = parser.parse_args()

    energies = [args.energy] if args.energy else ENERGIES

    for energy in energies:
        print(f"\n{energy}:")
        rows = load_sample_stats(energy)
        if not rows:
            print("  No data found.")
            continue
        plot_energy(energy, rows, args.metric)


if __name__ == "__main__":
    main()
