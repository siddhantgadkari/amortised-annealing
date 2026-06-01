#!/usr/bin/env python
"""End-to-end sweep for a configurable energy: sample → train → eval.

For each (dim, beta) combination:
  1. Run MCMC sampling (skipped if data already exists)
  2. Train score model (always overwrites)
  3. Generate diffusion samples and compare energy stats to MCMC

Prints a summary table at the end.

Set ENERGY at the top to switch targets (rastrigin, double_well, ackley, many_well).
"""
from __future__ import annotations
import json, time
from pathlib import Path

import torch
import yaml

from amortised_annealing.energies import Rastrigin, DoubleWell, Ackley, ManyWell
from amortised_annealing.sampling.sample_gen import SampleGenerator
from amortised_annealing.diffusion import (
    VPSchedule, MLPScore, ReverseSDE, euler_maruyama_sample,
    TrainingConfig, train_score_model,
)

ROOT             = Path(__file__).parent.parent
SAMPLES_DIR      = ROOT / "data" / "samples"
MODELS_DIR       = ROOT / "data" / "models"
CFG_SAMPLING_DIR = ROOT / "configs" / "sampling"
CFG_MODELS_DIR   = ROOT / "configs" / "models"

# ── Target ─────────────────────────────────────────────────────────────────
# Options: "rastrigin", "double_well", "ackley", "many_well"
ENERGY = "ackley"

_ENERGY_MAP = {
    "rastrigin":   Rastrigin,
    "double_well": DoubleWell,
    "ackley":      Ackley,
    "many_well":   ManyWell,
}

# ── Sweep ──────────────────────────────────────────────────────────────────
DIMS   = [2, 5, 10]
BETAS  = [0.5, 1.0, 2.0, 5.0]
SEED   = 0
DEVICE_STR = "auto"

# ── Sampling ───────────────────────────────────────────────────────────────
MCMC_METHOD = "ULA"
STEP_SIZE   = 0.001
N_PARTICLES = 8192
N_STEPS     = 10000
BURN_IN     = 2000
SAVE_EVERY  = 100

# ── Training ───────────────────────────────────────────────────────────────
PRESET_NAME    = "mlp256x3"
HIDDEN_DIMS    = [256, 256, 256]
TIME_EMBED_DIM = 64
ACTIVATION     = "silu"
LOSS_TYPE      = "eps"
LOG_UNIFORM_T  = False
N_TRAIN_STEPS  = 20000
BATCH_SIZE     = 512
LR             = 2e-4
T_EPS          = 1e-4
GRAD_CLIP      = 1.0
EMA_DECAY      = 0.999
LOG_EVERY      = 500

# ── Schedule ───────────────────────────────────────────────────────────────
BETA_MIN = 0.1
BETA_MAX = 20.0

# ── Eval ───────────────────────────────────────────────────────────────────
N_EVAL_SAMPLES = 4096
N_EM_STEPS     = 500


def _resolve_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def _beta_str(b: float) -> str:
    return f"{b:g}".replace(".", "p")


def _make_energy(dim: int):
    cls = _ENERGY_MAP[ENERGY]
    return cls(dim=dim)


def _energy_stats(samples: torch.Tensor, energy_fn) -> dict:
    with torch.no_grad():
        energies = energy_fn(samples).float().cpu()
    qs = torch.quantile(energies, torch.tensor([0.25, 0.75]))
    return {
        "mean":   round(energies.mean().item(), 3),
        "median": round(energies.median().item(), 3),
        "min":    round(energies.min().item(), 3),
        "std":    round(energies.std().item(), 3),
        "q25":    round(qs[0].item(), 3),
        "q75":    round(qs[1].item(), 3),
    }


# ── Step 1: Sampling ────────────────────────────────────────────────────────

def run_sampling(dim: int, beta: float, device: torch.device) -> str:
    run_name = f"{ENERGY}_d{dim}_beta{_beta_str(beta)}_{MCMC_METHOD.lower()}_seed{SEED}"
    out_dir  = SAMPLES_DIR / run_name

    if (out_dir / "particles.pt").exists():
        print(f"  [sample] {run_name} — exists, skipping")
        return run_name

    print(f"  [sample] {run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    energy = _make_energy(dim)
    gen    = SampleGenerator(energy=energy, betaM=beta, mcmc_method=MCMC_METHOD,
                             step_size=STEP_SIZE, device=device)
    t0 = time.perf_counter()
    final_x, _traces, step_rates = gen.sample(N_PARTICLES, N_STEPS, burn_in=BURN_IN,
                                              trace_every=SAVE_EVERY, progress=True)
    wall = time.perf_counter() - t0

    stats = _energy_stats(final_x, energy)
    cfg = {
        "job":     {"seed": SEED, "device": DEVICE_STR, "dtype": "float32"},
        "target":  {"energy": ENERGY, "dim": dim, "beta_m": beta},
        "sampler": {"method": MCMC_METHOD, "step_size": STEP_SIZE,
                    "n_particles": N_PARTICLES, "n_steps": N_STEPS,
                    "burn_in": BURN_IN, "trace_every": SAVE_EVERY},
        "output":  {"run_name": run_name},
    }
    summary = {
        "status": "completed", "energy": ENERGY, "dim": dim, "beta_m": beta,
        "sampler": MCMC_METHOD, "n_particles": N_PARTICLES,
        "wall_clock_seconds": round(wall, 2),
        "mean_energy": stats["mean"], "median_energy": stats["median"],
        "min_energy":  stats["min"],  "std_energy":    stats["std"],
        "energy_quantiles": {"q25": stats["q25"], "q75": stats["q75"]},
        "acceptance_rate": {"mean": round(sum(step_rates) / len(step_rates), 4)},
    }
    torch.save(final_x.cpu(), out_dir / "particles.pt")
    with open(out_dir / "config.yaml",  "w") as f: yaml.dump(cfg,     f, sort_keys=False)
    with open(out_dir / "summary.json", "w") as f: json.dump(summary, f, indent=2)

    CFG_SAMPLING_DIR.mkdir(parents=True, exist_ok=True)
    with open(CFG_SAMPLING_DIR / f"{run_name}.yaml", "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    print(f"         done in {wall:.1f}s")
    return run_name


# ── Step 2: Training ────────────────────────────────────────────────────────

def run_training(sample_run_name: str, dim: int, device: torch.device) -> str:
    model_run_name = f"{sample_run_name}_{PRESET_NAME}_{LOSS_TYPE}_seed{SEED}"
    out_dir        = MODELS_DIR / model_run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [train]  {model_run_name}")

    torch.manual_seed(SEED)
    x_data = torch.load(SAMPLES_DIR / sample_run_name / "particles.pt", map_location="cpu", weights_only=True)

    model    = MLPScore(dim=dim, hidden_dims=tuple(HIDDEN_DIMS),
                        time_embed_dim=TIME_EMBED_DIM, activation=ACTIVATION,
                        predict_score=(LOSS_TYPE == "score"))
    schedule = VPSchedule(beta_min=BETA_MIN, beta_max=BETA_MAX)
    cfg_tr   = TrainingConfig(
        n_steps=N_TRAIN_STEPS, batch_size=BATCH_SIZE, lr=LR, t_eps=T_EPS,
        grad_clip=GRAD_CLIP, ema_decay=EMA_DECAY, log_every=LOG_EVERY,
        log_uniform_t=LOG_UNIFORM_T, loss_type=LOSS_TYPE, seed=SEED,
    )

    t0 = time.perf_counter()
    ema_model, loss_history = train_score_model(model, schedule, x_data, cfg_tr, device)
    wall = time.perf_counter() - t0

    summary = {
        "status": "completed", "run_name": model_run_name,
        "sample_run_name": sample_run_name, "dim": dim,
        "n_params": sum(p.numel() for p in ema_model.parameters()),
        "wall_clock_seconds": round(wall, 2),
        "final_loss": round(loss_history[-1], 6) if loss_history else None,
        "loss_history": [round(l, 6) for l in loss_history],
    }
    cfg_yaml = {
        "job":      {"seed": SEED, "device": DEVICE_STR, "dtype": "float32"},
        "samples":  {"run_name": sample_run_name},
        "schedule": {"type": "vp", "beta_min": BETA_MIN, "beta_max": BETA_MAX},
        "model":    {"hidden_dims": HIDDEN_DIMS, "time_embed_dim": TIME_EMBED_DIM,
                     "activation": ACTIVATION, "predict_score": LOSS_TYPE == "score"},
        "training": {"n_steps": N_TRAIN_STEPS, "batch_size": BATCH_SIZE, "lr": LR,
                     "t_eps": T_EPS, "grad_clip": GRAD_CLIP, "ema_decay": EMA_DECAY,
                     "log_every": LOG_EVERY, "log_uniform_t": LOG_UNIFORM_T,
                     "loss_type": LOSS_TYPE},
        "output":   {"root": "data/models", "run_name": model_run_name},
    }
    torch.save(ema_model.state_dict(), out_dir / "ema_model.pt")
    with open(out_dir / "summary.json", "w") as f: json.dump(summary,  f, indent=2)
    with open(out_dir / "config.yaml",  "w") as f: yaml.dump(cfg_yaml, f, sort_keys=False)
    CFG_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CFG_MODELS_DIR / f"{model_run_name}.yaml", "w") as f:
        yaml.dump(cfg_yaml, f, sort_keys=False)

    print(f"         done in {wall:.1f}s  final_loss={summary['final_loss']}")
    return model_run_name


# ── Step 3: Eval ────────────────────────────────────────────────────────────

def run_eval(model_run_name: str, sample_run_name: str, dim: int, device: torch.device) -> dict:
    model = MLPScore(dim=dim, hidden_dims=tuple(HIDDEN_DIMS),
                     time_embed_dim=TIME_EMBED_DIM, activation=ACTIVATION,
                     predict_score=(LOSS_TYPE == "score"))
    state = torch.load(MODELS_DIR / model_run_name / "ema_model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    schedule = VPSchedule(beta_min=BETA_MIN, beta_max=BETA_MAX)
    x = euler_maruyama_sample(ReverseSDE(model, schedule),
                              n_samples=N_EVAL_SAMPLES, n_steps=N_EM_STEPS, device=device)

    energy     = _make_energy(dim)
    diff_stats = _energy_stats(x.cpu(), energy)

    with open(SAMPLES_DIR / sample_run_name / "summary.json") as f:
        s = json.load(f)
    mcmc_stats = {"mean":   s["mean_energy"],   "median": s["median_energy"],
                  "min":    s["min_energy"],     "std":    s["std_energy"],
                  "q25":    s["energy_quantiles"]["q25"],
                  "q75":    s["energy_quantiles"]["q75"]}

    return {"diff": diff_stats, "mcmc": mcmc_stats}


# ── Table ───────────────────────────────────────────────────────────────────

def _print_table(rows: list[dict]) -> None:
    cols  = ["mean", "median", "min", "std", "q25", "q75"]
    width = 8

    header = (f"{'dim':>4}  {'beta':>5}  " +
              "  ".join(f"{'mcmc_'+c:>{width}}" for c in cols) + "  " +
              "  ".join(f"{'diff_'+c:>{width}}" for c in cols))
    sep    = "─" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in rows:
        line = (f"{r['dim']:>4}  {r['beta']:>5}  " +
                "  ".join(f"{r['mcmc'][c]:>{width}.3f}" for c in cols) + "  " +
                "  ".join(f"{r['diff'][c]:>{width}.3f}" for c in cols))
        print(line)
    print(sep + "\n")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    if ENERGY not in _ENERGY_MAP:
        raise ValueError(f"Unknown ENERGY={ENERGY!r}. Choose from {list(_ENERGY_MAP)}")

    device = _resolve_device(DEVICE_STR)
    torch.set_default_dtype(torch.float32)

    results = []
    for dim in DIMS:
        for beta in BETAS:
            print(f"\n{'─'*60}")
            print(f"  {ENERGY}  dim={dim}  beta={beta}")
            print(f"{'─'*60}")
            sample_run = run_sampling(dim, beta, device)
            model_run  = run_training(sample_run, dim, device)
            stats      = run_eval(model_run, sample_run, dim, device)
            results.append({"dim": dim, "beta": beta, **stats})

    _print_table(results)

    out_path = ROOT / "data" / f"{ENERGY}_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
