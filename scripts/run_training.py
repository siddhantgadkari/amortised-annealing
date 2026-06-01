#!/usr/bin/env python
from __future__ import annotations
import argparse, json, shutil, time
from pathlib import Path

import torch
import yaml

from amortised_annealing.diffusion import (
    VPSchedule,
    MLPScore,
    TrainingConfig,
    train_score_model,
)

ROOT        = Path(__file__).parent.parent
CONFIGS_DIR = ROOT / "configs" / "models"
DATA_DIR    = ROOT / "data"


def _resolve_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def _load_samples(samples_cfg: dict) -> torch.Tensor:
    """Load particles.pt and select the configured snapshot(s).

    Returns [N, dim] tensor on CPU.
    """
    particles_path = DATA_DIR / "samples" / samples_cfg["run_name"] / "particles.pt"
    particles = torch.load(particles_path, map_location="cpu")  # [N, num_snapshots, dim]

    snapshot = samples_cfg.get("snapshot", -1)
    if snapshot is None:
        # Flatten all snapshots into one pool
        n, s, d = particles.shape
        return particles.reshape(n * s, d)
    else:
        return particles[:, snapshot, :]  # [N, dim]


def _build_schedule(schedule_cfg: dict):
    if schedule_cfg["type"] == "vp":
        return VPSchedule(
            beta_min=schedule_cfg.get("beta_min", 0.1),
            beta_max=schedule_cfg.get("beta_max", 20.0),
        )
    raise ValueError(f"Unknown schedule type: {schedule_cfg['type']}")


def run_config(cfg_path: Path) -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    job      = cfg["job"]
    model_c  = cfg["model"]
    train_c  = cfg["training"]
    output   = cfg["output"]

    run_name = output["run_name"]
    out_dir  = DATA_DIR / "models" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(job["seed"])
    torch.set_default_dtype(getattr(torch, job["dtype"]))

    device = _resolve_device(job["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        gpu_name = torch.cuda.get_device_name(device)
    else:
        gpu_name = None

    x_data   = _load_samples(cfg["samples"])
    schedule = _build_schedule(cfg["schedule"])

    model = MLPScore(
        dim            = x_data.shape[-1],
        hidden_dims    = tuple(model_c["hidden_dims"]),
        time_embed_dim = model_c.get("time_embed_dim", 64),
        activation     = model_c.get("activation", "silu"),
        predict_score  = model_c.get("predict_score", False),
    )
    n_params = sum(p.numel() for p in model.parameters())

    train_cfg = TrainingConfig(
        n_steps       = train_c.get("n_steps",       20_000),
        batch_size    = train_c.get("batch_size",    512),
        lr            = train_c.get("lr",            2e-4),
        t_eps         = train_c.get("t_eps",         1e-4),
        grad_clip     = train_c.get("grad_clip",     1.0),
        ema_decay     = train_c.get("ema_decay",     0.999),
        log_every     = train_c.get("log_every",     500),
        log_uniform_t = train_c.get("log_uniform_t", False),
        loss_type     = train_c.get("loss_type",     "eps"),
        seed          = job["seed"],
    )

    print(f"[{run_name}]  n_params={n_params:,}  n_data={x_data.shape[0]:,}")
    t0 = time.perf_counter()
    ema_model, loss_history = train_score_model(model, schedule, x_data, train_cfg, device)
    wall = time.perf_counter() - t0

    peak_gpu_mb = (
        round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 1)
        if device.type == "cuda" else None
    )

    summary = {
        "status":             "completed",
        "run_name":           run_name,
        "sample_run_name":    cfg["samples"]["run_name"],
        "snapshot":           cfg["samples"].get("snapshot", -1),
        "n_training_samples": x_data.shape[0],
        "dim":                x_data.shape[-1],
        "n_params":           n_params,
        "wall_clock_seconds": round(wall, 2),
        "device":             str(device),
        "gpu_name":           gpu_name,
        "peak_gpu_memory_mb": peak_gpu_mb,
        "final_loss":         round(loss_history[-1], 6) if loss_history else None,
        "loss_history":       [round(l, 6) for l in loss_history],
    }

    torch.save(ema_model.state_dict(), out_dir / "ema_model.pt")
    shutil.copy(cfg_path, out_dir / "config.yaml")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  done in {wall:.1f}s  final_loss={summary['final_loss']}  -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a score model from a config file.")
    parser.add_argument(
        "configs", nargs="*", type=Path,
        help="Config YAML path(s). Defaults to all YAMLs in configs/models/.",
    )
    args = parser.parse_args()
    paths = args.configs or sorted(CONFIGS_DIR.glob("*.yaml"))
    for p in paths:
        run_config(p)


if __name__ == "__main__":
    main()
