#!/usr/bin/env python
"""Break-even reuse analysis: how many Algorithm 2 calls before amortised beats direct SMC?

For a desired quality threshold E*, computes:
  R*(E*) = ceil(C_setup / (C_direct(E*) - C_use))

where:
  C_direct(E*) = cheapest direct annealed SMC oracle cost achieving quality E*
  C_setup      = oracle_cost_setup_mcmc (amortised MCMC + training, paid once)
  C_use        = oracle_cost_polish_ula (per-call ULA polish cost)

Edge case: if C_direct <= C_use → no_break_even.
  avg_cost = C_setup/R + C_use → C_use from above as R→∞.
  If C_use >= C_direct, the amortised method can never beat direct on avg cost per call.

Two questions answered:
  Break-even table (economic): "How many independent tasks before avg cost/task < direct SMC?"
  Plot B (visual): avg_cost_per_call(R) curves vs direct SMC horizontal lines.

Usage:
    uv run python scripts/experiments/break_even.py           # table + plots
    uv run python scripts/experiments/break_even.py --no-plot # table only
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT     = Path(__file__).parent.parent.parent
EXP_BASE = ROOT / "data" / "experiments" / "cost_quality"
ENERGY   = "ackley"

METRICS = ["best_energy", "q01", "mean_energy"]

# Fixed threshold candidates (energy values to test if achievable)
FIXED_THRESHOLDS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

# Inference config to use for break-even (largest batch, both polish options)
N_SAMPLES_BREAKEVEN = 8192


def _m(d) -> float | None:
    if isinstance(d, dict):
        return d.get("mean")
    return d if d is not None else None


def _s(d) -> float:
    if isinstance(d, dict):
        return d.get("std") or 0.0
    return 0.0


def compute_break_even(data: dict, metric: str, dim: int, beta_h: float) -> list[dict]:
    """Compute break-even rows for one (metric, dim, beta_h)."""
    direct_pts = [
        p for p in data["direct_ula_annealed"]
        if p["dim"] == dim and p["beta_h"] == beta_h
    ]
    if not direct_pts:
        return []

    amort_cells = [
        c for c in data["amortised"]
        if c["dim"] == dim and c["beta_h"] == beta_h and c["n_samples"] == N_SAMPLES_BREAKEVEN
    ]
    if not amort_cells:
        return []

    # Build threshold grid: direct SMC quality values + fixed candidates
    direct_qualities = [_m(p.get(metric)) for p in direct_pts if _m(p.get(metric)) is not None]
    thresh_set: set[float] = set(FIXED_THRESHOLDS)
    for q in direct_qualities:
        thresh_set.add(round(q, 5))
    thresholds = sorted(thresh_set)

    rows: list[dict] = []
    for thresh in thresholds:
        # C_direct: cheapest direct SMC config achieving quality <= thresh
        qualifying_direct = [
            p for p in direct_pts
            if _m(p.get(metric)) is not None and _m(p.get(metric)) <= thresh
        ]
        if not qualifying_direct:
            continue
        best_direct = min(qualifying_direct,
                          key=lambda p: p.get("oracle_cost_total", p.get("n_grad_evals", 0)))
        c_direct = best_direct.get("oracle_cost_total", best_direct.get("n_grad_evals", 0))

        # For each amortised config achieving threshold, compute R*
        best_row: dict | None = None
        best_r_star = float("inf")

        for cell in amort_cells:
            metric_val = _m(cell.get(metric))
            if metric_val is None or metric_val > thresh:
                continue

            c_setup = _m(cell.get("oracle_cost_setup_mcmc")) or 0.0
            c_use   = _m(cell.get("oracle_cost_polish_ula")) or 0.0

            if c_direct <= c_use:
                r_star_val = float("inf")
                r_star_str = "no_break_even"
            else:
                r_star_val = math.ceil(c_setup / (c_direct - c_use))
                r_star_str = str(r_star_val)

            if r_star_val < best_r_star:
                best_r_star = r_star_val
                best_row = {
                    "dim":         dim,
                    "beta_h":      beta_h,
                    "metric":      metric,
                    "threshold":   thresh,
                    "c_direct":    int(c_direct),
                    "c_setup":     int(round(c_setup)),
                    "c_use":       int(round(c_use)),
                    "r_star":      r_star_str,
                    "beta_m":      cell["beta_m"],
                    "n_train":     cell["n_train"],
                    "n_samples":   cell["n_samples"],
                    "local_steps": cell["local_steps"],
                    "metric_val":  round(metric_val, 4),
                }

        if best_row is not None:
            rows.append(best_row)

    return rows


def run_break_even(no_plot: bool = False) -> None:
    agg_path = EXP_BASE / ENERGY / "aggregate.json"
    if not agg_path.exists():
        print(f"No aggregate.json at {agg_path}. Run cost_quality --aggregate first.")
        return

    data = json.loads(agg_path.read_text())
    dims    = sorted(set(c["dim"]    for c in data["amortised"]))
    beta_hs = sorted(set(c["beta_h"] for c in data["amortised"]))

    all_rows: list[dict] = []
    for metric in METRICS:
        for dim in dims:
            for beta_h in beta_hs:
                rows = compute_break_even(data, metric, dim, beta_h)
                all_rows.extend(rows)

    if all_rows:
        hdr = ["dim", "beta_h", "metric", "threshold", "c_direct", "c_setup",
               "c_use", "r_star", "beta_m", "n_train", "local_steps", "metric_val"]
        w = [4, 6, 12, 10, 14, 14, 10, 14, 7, 8, 12, 10]
        fmt = "  ".join(f"{{:<{n}}}" for n in w)
        print(f"\n{'='*110}")
        print("BREAK-EVEN TABLE (best amortised config per threshold)")
        print(f"{'='*110}")
        print(fmt.format(*hdr))
        print("-" * 110)
        for r in all_rows:
            print(fmt.format(
                r["dim"], r["beta_h"], r["metric"], r["threshold"],
                r["c_direct"], r["c_setup"], r["c_use"], r["r_star"],
                r["beta_m"], r["n_train"], r["local_steps"], r["metric_val"],
            ))

    out_dir  = EXP_BASE / ENERGY
    csv_path = out_dir / "break_even.csv"
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nCSV saved → {csv_path}")

    if not no_plot:
        plot_break_even(data, dims, beta_hs)


def plot_break_even(data: dict, dims: list, beta_hs: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plots_dir = EXP_BASE / ENERGY / "plots"
    plots_dir.mkdir(exist_ok=True)

    R_range = np.arange(1, 201)
    bm_colors = {1.0: "steelblue", 5.0: "darkorange", 20.0: "mediumseagreen"}
    nt_styles = {2048: "-", 8192: "--"}

    for dim in dims:
        for beta_h in beta_hs:
            fig, ax = plt.subplots(figsize=(8, 5))

            amort_cells = [
                c for c in data["amortised"]
                if c["dim"] == dim and c["beta_h"] == beta_h
                and c["n_samples"] == N_SAMPLES_BREAKEVEN
            ]
            seen: set[str] = set()
            for cell in amort_cells:
                c_setup = _m(cell.get("oracle_cost_setup_mcmc")) or 0.0
                c_use   = _m(cell.get("oracle_cost_polish_ula")) or 0.0
                avg_costs = c_setup / R_range + c_use

                color = bm_colors.get(cell["beta_m"], "purple")
                ls    = nt_styles.get(cell["n_train"], ":")
                alpha = 1.0 if cell["local_steps"] == 10 else 0.5
                label = (f"β_M={cell['beta_m']:g} N_tr={cell['n_train']} "
                         f"steps={cell['local_steps']}")
                if label not in seen:
                    ax.plot(R_range, avg_costs, color=color, ls=ls, alpha=alpha,
                            lw=1.2, label=label)
                    seen.add(label)

            direct_pts = sorted(
                [p for p in data["direct_ula_annealed"]
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
            ax.set_title(f"Break-even curves: d={dim}  β_H={beta_h}", fontsize=10)
            ax.legend(fontsize=6, loc="upper right", ncol=2)
            ax.grid(True, alpha=0.3)

            out = plots_dir / f"break_even_d{dim}_betaH{beta_h:g}.svg"
            fig.tight_layout()
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {out.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Break-even reuse analysis.")
    parser.add_argument("--no-plot", action="store_true", help="Skip plots")
    args = parser.parse_args()
    run_break_even(no_plot=args.no_plot)


if __name__ == "__main__":
    main()
