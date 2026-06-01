#!/usr/bin/env python
"""Backfill existing particles.pt files to the new [N, dim] format.

For each run in data/samples/:
  - If particles.pt is already 2D [N, dim]: skip (already migrated).
  - If particles.pt is 3D [N, S, dim]: compute chain traces from all S snapshots,
    update summary.json with traces + coord/threshold/stationarity stats,
    then overwrite particles.pt with just the final frame [N, dim].

Run once after pulling the new sample_gen.py changes to migrate existing data.

Usage:
    uv run python scripts/backfill_samples.py
    uv run python scripts/backfill_samples.py --dry-run
    uv run python scripts/backfill_samples.py --runs run_name1 run_name2
"""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

import torch

ROOT       = Path(__file__).parent.parent
SAMPLE_DIR = ROOT / "data" / "samples"

ENERGY_MAP: dict = {}
try:
    from amortised_annealing.energies import DoubleWell, ManyWell, Ackley, Rastrigin
    ENERGY_MAP = {
        "double_well": DoubleWell,
        "many_well":   ManyWell,
        "ackley":      Ackley,
        "rastrigin":   Rastrigin,
    }
except ImportError:
    pass  # energy stats will be skipped if package not importable


def _coord_stats(x: torch.Tensor) -> dict:
    norms = x.norm(dim=1)
    return {
        "mean_x_norm": round(norms.mean().item(), 4),
        "max_x_norm":  round(norms.max().item(), 4),
        "coord_mean":  round(x.mean().item(), 4),
        "coord_std":   round(x.std().item(), 4),
    }


def _threshold_props(x: torch.Tensor, energy_fn, global_min_energy: float | None) -> dict:
    if global_min_energy is None or energy_fn is None:
        return {}
    with torch.no_grad():
        e = energy_fn(x).float().cpu() - global_min_energy
    return {
        "prop_excess_E_lt_0.01": round((e < 0.01).float().mean().item(), 4),
        "prop_excess_E_lt_0.1":  round((e < 0.1).float().mean().item(), 4),
        "prop_excess_E_lt_1.0":  round((e < 1.0).float().mean().item(), 4),
    }


def _stationarity(mean_energies: list[float]) -> dict:
    if len(mean_energies) < 5:
        return {"energy_tail_slope": None}
    tail = mean_energies[int(len(mean_energies) * 0.8):]
    n = len(tail)
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(tail) / n
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(xs, tail))
    den = sum((xi - x_mean) ** 2 for xi in xs)
    slope = num / den if den > 0 else 0.0
    return {"energy_tail_slope": round(slope, 6)}


def _compute_traces(particles: torch.Tensor, energy_fn, save_every: int) -> dict:
    """Compute scalar traces from a [N, S, dim] trajectory tensor."""
    n_snaps = particles.shape[1]
    traces: dict = {
        "steps":             [],
        "mean_energy":       [],
        "median_energy":     [],
        "min_energy":        [],
        "q05_energy":        [],
        "q95_energy":        [],
        "mean_x_norm":       [],
        "mean_displacement": [],
    }
    x_prev: torch.Tensor | None = None
    for s in range(n_snaps):
        x = particles[:, s, :]
        traces["steps"].append((s + 1) * save_every)
        traces["mean_x_norm"].append(round(x.norm(dim=1).mean().item(), 4))
        if x_prev is not None:
            disp = (x - x_prev).norm(dim=1).mean().item()
        else:
            disp = float("nan")
        traces["mean_displacement"].append(round(disp, 4) if not (disp != disp) else None)
        x_prev = x
        if energy_fn is not None:
            with torch.no_grad():
                e = energy_fn(x.float()).float()
            qs = torch.quantile(e, torch.tensor([0.05, 0.95]))
            traces["mean_energy"].append(round(e.mean().item(), 4))
            traces["median_energy"].append(round(e.median().item(), 4))
            traces["min_energy"].append(round(e.min().item(), 4))
            traces["q05_energy"].append(round(qs[0].item(), 4))
            traces["q95_energy"].append(round(qs[1].item(), 4))
        else:
            for k in ["mean_energy", "median_energy", "min_energy", "q05_energy", "q95_energy"]:
                traces[k].append(None)
    return traces


def backfill_run(run_dir: Path, dry_run: bool = False) -> str:
    particles_path = run_dir / "particles.pt"
    summary_path   = run_dir / "summary.json"

    if not particles_path.exists():
        return "skip (no particles.pt)"

    pts = torch.load(particles_path, map_location="cpu", weights_only=True)

    # Determine migration case:
    #   3D [N,S,D]:          full migration — compute traces + trim
    #   2D non-contiguous:   botched earlier run — just re-save contiguous (traces already in summary)
    #   2D contiguous:       already done
    if pts.dim() == 2 and pts.is_contiguous():
        return "skip (already done)"
    if pts.dim() == 2 and not pts.is_contiguous():
        if dry_run:
            return f"would repack: {tuple(pts.shape)} (non-contiguous storage)"
        torch.save(pts.contiguous(), particles_path)
        return f"repacked: {tuple(pts.shape)} (freed oversized storage)"
    if pts.dim() != 3:
        return f"skip (unexpected shape {tuple(pts.shape)})"

    n, s, dim = pts.shape

    # Load summary to get energy info
    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    energy_name = summary.get("energy")
    energy_dim  = summary.get("dim", dim)
    save_every  = summary.get("save_every", summary.get("trace_every", 1))

    energy_fn  = None
    global_min = None
    if energy_name and energy_name in ENERGY_MAP:
        energy_obj = ENERGY_MAP[energy_name](dim=energy_dim)
        energy_fn  = energy_obj.energy
        global_min = getattr(energy_obj, "global_minimum_energy", None)

    final_x = pts[:, -1, :]

    if dry_run:
        return f"would backfill: {n}×{s}×{dim} → {n}×{dim}"

    # Compute traces
    traces = _compute_traces(pts, energy_fn, save_every)

    # Update summary
    summary["trace_every"] = save_every
    summary.pop("save_every", None)
    summary.update(_coord_stats(final_x))
    summary.update(_threshold_props(final_x, energy_fn, global_min))
    mean_es = [v for v in traces["mean_energy"] if v is not None]
    summary.update(_stationarity(mean_es))
    summary["traces"] = traces

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Overwrite particles.pt with final frame only.
    # .contiguous() forces a new allocation so torch.save doesn't carry the full [N,S,D] storage.
    torch.save(final_x.contiguous(), particles_path)

    return f"backfilled: {n}×{s}×{dim} → {n}×{dim}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate particles.pt from [N,S,D] to [N,D].")
    parser.add_argument("--dry-run", action="store_true", help="Report what would happen without writing.")
    parser.add_argument("--runs", nargs="+", metavar="RUN_NAME", help="Specific run names to process.")
    args = parser.parse_args()

    if args.runs:
        dirs = [SAMPLE_DIR / r for r in args.runs]
    else:
        dirs = sorted(d for d in SAMPLE_DIR.iterdir() if d.is_dir())

    n_done = n_skip = n_err = 0
    for run_dir in dirs:
        try:
            result = backfill_run(run_dir, dry_run=args.dry_run)
        except Exception as exc:
            result = f"ERROR: {exc}"
            n_err += 1
        else:
            if result.startswith("skip"):
                n_skip += 1
            else:
                n_done += 1
        prefix = "[dry]" if args.dry_run else "    "
        print(f"{prefix} {run_dir.name}: {result}")

    print(f"\nDone: {n_done} migrated, {n_skip} skipped, {n_err} errors")


if __name__ == "__main__":
    main()
