#!/usr/bin/env python
"""Table + plots: ULA SMC vs Diffusion SMC — mean and min energy across beta_M.

Reads energy_betaM_experiments aggregate JSONs and produces:
  - A printed table per (energy, dim) showing ula vs diffusion mean/min energy
  - A CSV saved to data/experiments/energy_betaM_experiments/ula_vs_diffusion_table.csv
  - (--plot) One figure per energy: 2 rows × 4 cols (min row, mean row; one col per dim)

The gap ratio columns show diff/ula for min and mean energy:
  ratio < 1 → diffusion better; ratio > 1 → ula better

Usage:
    uv run python scripts/experiments/ula_vs_diffusion_table.py
    uv run python scripts/experiments/ula_vs_diffusion_table.py --energy ackley
    uv run python scripts/experiments/ula_vs_diffusion_table.py --plot
    uv run python scripts/experiments/ula_vs_diffusion_table.py --plot --energy ackley
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT     = Path(__file__).parent.parent.parent
EXP_BASE = ROOT / "data" / "experiments" / "energy_betaM_experiments"

ENERGIES = ["ackley", "rastrigin", "double_well", "many_well"]


def _get(run: dict, method: str, metric: str) -> float | None:
    v = run.get(method, {}).get(metric)
    if isinstance(v, dict):
        return v.get("mean")
    return v


def build_rows(energy: str, dim_filter: int | None) -> list[dict]:
    agg_path = EXP_BASE / energy / "aggregate.json"
    if not agg_path.exists():
        return []
    agg = json.loads(agg_path.read_text())

    rows = []
    for run in sorted(agg["runs"], key=lambda r: (r["dim"], r["beta_m"])):
        dim    = run["dim"]
        beta_m = run["beta_m"]

        if dim_filter is not None and dim != dim_filter:
            continue

        ula_min  = _get(run, "ula",       "min_energy")
        ula_mean = _get(run, "ula",       "mean_energy")
        ula_std  = _get(run, "ula",       "std_energy")
        diff_min  = _get(run, "diffusion", "min_energy")
        diff_mean = _get(run, "diffusion", "mean_energy")
        diff_std  = _get(run, "diffusion", "std_energy")

        ula_min_std  = run.get("ula",       {}).get("min_energy",  {}).get("std")
        diff_min_std = run.get("diffusion", {}).get("min_energy",  {}).get("std")
        ula_mean_std  = run.get("ula",       {}).get("mean_energy", {}).get("std")
        diff_mean_std = run.get("diffusion", {}).get("mean_energy", {}).get("std")

        if ula_min is None or diff_min is None:
            continue

        rows.append({
            "energy":       energy,
            "dim":          dim,
            "beta_m":       beta_m,
            "ula_min":      ula_min,
            "ula_min_std":  ula_min_std or 0.0,
            "diff_min":     diff_min,
            "diff_min_std": diff_min_std or 0.0,
            "min_ratio":    round(diff_min / ula_min, 3) if ula_min else None,
            "ula_mean":     ula_mean,
            "ula_mean_std": ula_mean_std or 0.0,
            "diff_mean":    diff_mean,
            "diff_mean_std":diff_mean_std or 0.0,
            "mean_ratio":   round(diff_mean / ula_mean, 3) if ula_mean else None,
        })
    return rows


def print_table(rows: list[dict], energy: str, dim: int) -> None:
    subset = [r for r in rows if r["energy"] == energy and r["dim"] == dim]
    if not subset:
        return

    print(f"\n{'='*85}")
    print(f"  {energy.upper()}   d={dim}   (ULA = MCMC start + ULA kernel;  Diff = model start + diff kernel)")
    print(f"{'='*85}")
    hdr = f"  {'β_M':>5} │ {'ULA min':>9} {'±':>2} {'σ':>7} │ {'Diff min':>9} {'±':>2} {'σ':>7} │ {'min ratio':>9} ║ {'ULA mean':>9} {'±':>2} {'σ':>7} │ {'Diff mean':>9} {'±':>2} {'σ':>7} │ {'mean ratio':>10}"
    print(hdr)
    print(f"  {'-'*5}-+-{'-'*9}---{'-'*7}-+-{'-'*9}---{'-'*7}-+-{'-'*9}-╫-{'-'*9}---{'-'*7}-+-{'-'*9}---{'-'*7}-+-{'-'*10}")
    for r in subset:
        ratio_min  = f"{r['min_ratio']:>9.3f}" if r["min_ratio"] else "       N/A"
        ratio_mean = f"{r['mean_ratio']:>10.3f}" if r["mean_ratio"] else "        N/A"
        print(
            f"  {r['beta_m']:>5g} │"
            f" {r['ula_min']:>9.4f}  {r['ula_min_std']:>7.4f} │"
            f" {r['diff_min']:>9.4f}  {r['diff_min_std']:>7.4f} │"
            f" {ratio_min} ║"
            f" {r['ula_mean']:>9.4f}  {r['ula_mean_std']:>7.4f} │"
            f" {r['diff_mean']:>9.4f}  {r['diff_mean_std']:>7.4f} │"
            f" {ratio_mean}"
        )
    print()
    # Summary: high-beta regime where things converge
    hi = [r for r in subset if r["beta_m"] >= 7.0]
    if hi:
        avg_min_ratio  = sum(r["min_ratio"]  for r in hi if r["min_ratio"])  / len(hi)
        avg_mean_ratio = sum(r["mean_ratio"] for r in hi if r["mean_ratio"]) / len(hi)
        print(f"  β_M ≥ 7 avg ratio: min={avg_min_ratio:.3f}  mean={avg_mean_ratio:.3f}  (1.0 = identical)")


def plot_energy(energy: str, all_rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows  = [r for r in all_rows if r["energy"] == energy]
    dims  = sorted(set(r["dim"] for r in rows))
    ndims = len(dims)

    ULA_COLOR  = "steelblue"
    DIFF_COLOR = "darkorange"

    metrics = [
        ("min",  "ula_min",  "ula_min_std",  "diff_min",  "diff_min_std",  "Min energy (post-SMC)"),
        ("mean", "ula_mean", "ula_mean_std", "diff_mean", "diff_mean_std", "Mean energy (post-SMC)"),
    ]

    fig, axes = plt.subplots(2, ndims, figsize=(4.2 * ndims, 7), squeeze=False)

    for row_idx, (metric_tag, ula_key, ula_std_key, diff_key, diff_std_key, ylabel) in enumerate(metrics):
        for col_idx, dim in enumerate(dims):
            ax = axes[row_idx][col_idx]

            subset = sorted(
                [r for r in rows if r["dim"] == dim],
                key=lambda r: r["beta_m"],
            )
            if not subset:
                ax.set_visible(False)
                continue

            betas     = [r["beta_m"]     for r in subset]
            ula_vals  = [r[ula_key]       for r in subset]
            ula_stds  = [r[ula_std_key]   for r in subset]
            diff_vals = [r[diff_key]      for r in subset]
            diff_stds = [r[diff_std_key]  for r in subset]

            ax.plot(betas, ula_vals,  "o-", color=ULA_COLOR,  lw=1.5, ms=4, label="ULA SMC  (MCMC start)")
            ax.fill_between(
                betas,
                [v - s for v, s in zip(ula_vals,  ula_stds)],
                [v + s for v, s in zip(ula_vals,  ula_stds)],
                color=ULA_COLOR, alpha=0.15,
            )
            ax.plot(betas, diff_vals, "s--", color=DIFF_COLOR, lw=1.5, ms=4, label="Diff SMC  (model start)")
            ax.fill_between(
                betas,
                [v - s for v, s in zip(diff_vals, diff_stds)],
                [v + s for v, s in zip(diff_vals, diff_stds)],
                color=DIFF_COLOR, alpha=0.15,
            )

            ax.set_xlabel(r"$\beta_M$", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(f"d = {dim}", fontsize=10)
            ax.grid(True, alpha=0.3)
            if row_idx == 0 and col_idx == ndims - 1:
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        f"{energy.replace('_', ' ').title()}  —  ULA SMC vs Diffusion SMC  (β_H = 10, 3 seeds ±1σ)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    plots_dir = EXP_BASE / energy / "plots"
    plots_dir.mkdir(exist_ok=True)
    out = plots_dir / "ula_vs_diffusion.svg"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy", choices=ENERGIES, default=None,
                        help="Restrict to one energy (default: all)")
    parser.add_argument("--dim", type=int, default=None,
                        help="Restrict to one dimension")
    parser.add_argument("--plot", action="store_true",
                        help="Save SVG plots (one per energy)")
    args = parser.parse_args()

    energies = [args.energy] if args.energy else ENERGIES

    all_rows: list[dict] = []
    for energy in energies:
        rows = build_rows(energy, args.dim)
        all_rows.extend(rows)

        dims = sorted(set(r["dim"] for r in rows))
        for dim in ([args.dim] if args.dim else dims):
            print_table(rows, energy, dim)

    # CSV
    if all_rows:
        out = EXP_BASE / "ula_vs_diffusion_table.csv"
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nCSV saved → {out}")

    # Plots
    if args.plot and all_rows:
        for energy in energies:
            plot_energy(energy, all_rows)


if __name__ == "__main__":
    main()
