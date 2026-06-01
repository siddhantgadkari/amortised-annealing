#!/usr/bin/env python
"""Diagnose a trained score model by measuring DSM loss per time bin.

A global loss that looks fine can hide catastrophic failure at small t,
which is exactly where the reverse SDE needs accurate scores to land
particles in narrow data modes (e.g. Rastrigin).

Usage:
    python scripts/diagnose_score.py <model_run_name>
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import torch
import yaml

from amortised_annealing.diffusion import VPSchedule, MLPScore, loss_by_t_bin
from amortised_annealing.energies import DoubleWell, ManyWell, Ackley, Rastrigin

ROOT      = Path(__file__).parent.parent
MODEL_DIR = ROOT / "data" / "models"
DATA_DIR  = ROOT / "data" / "samples"

ENERGY_MAP = {
    "double_well": DoubleWell,
    "many_well":   ManyWell,
    "ackley":      Ackley,
    "rastrigin":   Rastrigin,
}

BINS = [(1e-4, 1e-2), (1e-2, 5e-2), (5e-2, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0)]


def _resolve_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", help="Model run name (subfolder of data/models/)")
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    run_dir = MODEL_DIR / args.run_name
    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = _resolve_device(cfg["job"]["device"])
    torch.set_default_dtype(getattr(torch, cfg["job"]["dtype"]))

    # Load training data
    sample_cfg  = cfg["samples"]
    x_data      = torch.load(DATA_DIR / sample_cfg["run_name"] / "particles.pt", map_location="cpu", weights_only=True)

    # Load model
    model_cfg = cfg["model"]
    dim       = x_data.shape[-1]
    model     = MLPScore(
        dim            = dim,
        hidden_dims    = tuple(model_cfg["hidden_dims"]),
        time_embed_dim = model_cfg.get("time_embed_dim", 64),
        activation     = model_cfg.get("activation", "silu"),
        predict_score  = model_cfg.get("predict_score", False),
    )
    state = torch.load(run_dir / "ema_model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device)

    sched_cfg = cfg["schedule"]
    schedule  = VPSchedule(
        beta_min=sched_cfg.get("beta_min", 0.1),
        beta_max=sched_cfg.get("beta_max", 20.0),
    )

    loss_type = cfg.get("training", {}).get("loss_type", "eps")

    # Per-bin loss
    losses = loss_by_t_bin(model, x_data, schedule, device,
                           batch_size=args.batch_size, bins=BINS, loss_type=loss_type)

    print(f"\nDSM loss by time bin — {args.run_name}")
    print(f"  {'bin':<14}  loss      interpretation")
    print("  " + "-" * 50)
    for (lo, hi), (bin_str, val) in zip(BINS, losses.items()):
        if lo < 0.05:
            region = "<-- sharp score (modes matter here)"
        elif lo < 0.3:
            region = "<-- transitional"
        else:
            region = "    high noise (easy)"
        print(f"  t=[{lo:6g},{hi:4g}]  {val:.4f}    {region}")

    # Compare against training summary final loss for context
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"\n  global final loss (training): {summary.get('final_loss', 'n/a')}")

    print()
    print("If small-t bins are much worse than large-t bins, retrain with log_uniform_t: true")
    print()


if __name__ == "__main__":
    main()
