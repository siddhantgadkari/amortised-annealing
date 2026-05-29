#!/usr/bin/env python
"""FKC annealing experiment: sample → train → FKC + ULA baseline → plot.

Runs FKCAnnealedSampler for each trained model (one per β_M) targeting π_{β_H},
with a ULA SMC baseline for comparison.

Edit the USER CONFIGURATION block at the top, then run:
    uv run python scripts/experiments/run_fkc_experiment.py

To re-plot from already-saved results:
    uv run python scripts/experiments/run_fkc_experiment.py --plot-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

# -- import parent scripts for sampling/training pipeline --
_SCRIPTS = Path(__file__).parent.parent
sys.path.insert(0, str(_SCRIPTS))
from run_sampling import run_config as _run_sample_config  # noqa: E402
from run_training import run_config as _run_train_config   # noqa: E402

from amortised_annealing.diffusion import VPSchedule, MLPScore, ReverseSDE
from amortised_annealing.energies import DoubleWell, ManyWell, Ackley, Rastrigin
from amortised_annealing.smc import (
    FKCAnnealedSampler,
    ParticleCloud,
    SMCSampler,
    ULAProposal,
)

ROOT                 = Path(__file__).parent.parent.parent
SAMPLE_DIR           = ROOT / "data" / "samples"
MODEL_DIR            = ROOT / "data" / "models"
EXPERIMENT_DIR       = ROOT / "data" / "experiments" / "fkc"
SAMPLING_CONFIGS_DIR = ROOT / "configs" / "sampling"
TRAINING_CONFIGS_DIR = ROOT / "configs" / "models"

ENERGY_MAP = {
    "double_well": DoubleWell,
    "many_well":   ManyWell,
    "ackley":      Ackley,
    "rastrigin":   Rastrigin,
}

# ===========================================================================
# USER CONFIGURATION — edit this block to configure the experiment
# ===========================================================================
ENERGY  = "ackley"
DIM     = 2
BETA_MS = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
BETA_H  = 10.0

# MCMC sampling
MCMC_N_PARTICLES = 8192
MCMC_N_STEPS     = 10_000
MCMC_BURN_IN     = 2_000
MCMC_SAVE_EVERY  = 100
MCMC_STEP_SIZE   = 1e-3

# Score model architecture
MODEL_HIDDEN_DIMS    = [256, 256, 256]
MODEL_TIME_EMBED_DIM = 64
LOSS_TYPE            = "eps"

# Score model training
TRAIN_N_STEPS = 20_000
BATCH_SIZE    = 512
LR            = 2e-4
T_EPS         = 1e-4
GRAD_CLIP     = 1.0
EMA_DECAY     = 0.999

# FKC
FKC_N_PARTICLES        = 2048
FKC_N_STEPS            = 500
FKC_T_START            = 1.0
FKC_T_END              = 1e-3
FKC_INCLUDE_DIVERGENCE = False
FKC_ESS_THRESHOLD      = 0.5

# ULA baseline (initialised from MCMC last snapshot, same N as FKC)
N_ULA_STEPS       = 10
ULA_STEP_SIZE     = 1e-3
N_SMC_STEPS       = 50
ULA_ESS_THRESHOLD = 0.5

# Reuse flags — set to False to force re-run even if outputs exist
REUSE_SAMPLES = True
REUSE_MODELS  = True

SEED = 0

# Short label appended to the experiment directory name. Leave as "" to omit.
DESC = ""
# ===========================================================================


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------

def _beta_str(beta: float) -> str:
    return f"{beta:g}".replace(".", "p")


def _preset_tag() -> str:
    sizes = MODEL_HIDDEN_DIMS
    if len(set(sizes)) == 1:
        return f"mlp{sizes[0]}x{len(sizes)}"
    return "mlp_" + "_".join(str(s) for s in sizes)


def _sample_run_name(beta_m: float) -> str:
    return f"{ENERGY}_d{DIM}_beta{_beta_str(beta_m)}_ula_seed{SEED}"


def _model_run_name(sample_run: str) -> str:
    return f"{sample_run}_{_preset_tag()}_{LOSS_TYPE}_seed{SEED}"


def _experiment_dir() -> Path:
    tag = f"{ENERGY}_d{DIM}_betaH{_beta_str(BETA_H)}"
    if DESC:
        tag += f"_{DESC}"
    return EXPERIMENT_DIR / tag


# ---------------------------------------------------------------------------
# Config factories
# ---------------------------------------------------------------------------

def _make_sampling_config(beta_m: float) -> dict:
    run_name = _sample_run_name(beta_m)
    return {
        "job":    {"seed": SEED, "device": "auto", "dtype": "float32"},
        "target": {"energy": ENERGY, "dim": DIM, "beta_m": beta_m},
        "sampler": {
            "method":      "ULA",
            "step_size":   MCMC_STEP_SIZE,
            "n_particles": MCMC_N_PARTICLES,
            "n_steps":     MCMC_N_STEPS,
            "burn_in":     MCMC_BURN_IN,
            "save_every":  MCMC_SAVE_EVERY,
            "init_scale":  1.0,
        },
        "output": {"root": "runs/sampling", "run_name": run_name},
    }


def _make_training_config(sample_run: str) -> dict:
    model_run = _model_run_name(sample_run)
    return {
        "job":      {"seed": SEED, "device": "auto", "dtype": "float32"},
        "samples":  {"run_name": sample_run, "snapshot": -1},
        "schedule": {"type": "vp", "beta_min": 0.1, "beta_max": 20.0},
        "model": {
            "hidden_dims":    MODEL_HIDDEN_DIMS,
            "time_embed_dim": MODEL_TIME_EMBED_DIM,
            "activation":     "silu",
            "predict_score":  LOSS_TYPE == "score",
        },
        "training": {
            "n_steps":       TRAIN_N_STEPS,
            "batch_size":    BATCH_SIZE,
            "lr":            LR,
            "t_eps":         T_EPS,
            "grad_clip":     GRAD_CLIP,
            "ema_decay":     EMA_DECAY,
            "log_every":     500,
            "log_uniform_t": False,
            "loss_type":     LOSS_TYPE,
        },
        "output": {"root": "data/models", "run_name": model_run},
    }


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def setup_and_sample() -> dict[float, str]:
    """Ensure MCMC samples exist for all BETA_MS. Returns {beta_m: sample_run_name}."""
    SAMPLING_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    runs: dict[float, str] = {}

    for beta_m in BETA_MS:
        run_name = _sample_run_name(beta_m)
        out_dir  = SAMPLE_DIR / run_name

        if REUSE_SAMPLES and (out_dir / "particles.pt").exists():
            print(f"  [reuse]  {run_name}")
            runs[beta_m] = run_name
            continue

        print(f"  [sample] {run_name}")
        cfg      = _make_sampling_config(beta_m)
        cfg_path = SAMPLING_CONFIGS_DIR / f"{run_name}.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        _run_sample_config(cfg_path)
        runs[beta_m] = run_name

    return runs


def setup_and_train(sample_runs: dict[float, str]) -> dict[float, str]:
    """Ensure trained models exist for all sample runs. Returns {beta_m: model_run_name}."""
    TRAINING_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    runs: dict[float, str] = {}

    for beta_m in sorted(sample_runs):
        sample_run = sample_runs[beta_m]
        model_run  = _model_run_name(sample_run)
        out_dir    = MODEL_DIR / model_run

        if REUSE_MODELS and (out_dir / "ema_model.pt").exists():
            print(f"  [reuse]  {model_run}")
            runs[beta_m] = model_run
            continue

        print(f"  [train]  {model_run}")
        cfg      = _make_training_config(sample_run)
        cfg_path = TRAINING_CONFIGS_DIR / f"{model_run}.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        _run_train_config(cfg_path)
        runs[beta_m] = model_run

    return runs


def run_all(model_runs: dict[float, str], device: torch.device) -> list[dict]:
    """Run FKC and ULA for every model. Returns list of result dicts sorted by beta_m."""
    results = []
    for beta_m in sorted(model_runs):
        print(f"\n  beta_M={beta_m}  [{model_runs[beta_m]}]")
        result = _run_fkc_and_ula(model_runs[beta_m], device)
        results.append(result)
        _print_result(result)
    return results


# ---------------------------------------------------------------------------
# FKC + ULA
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

    model_cfg = cfg["model"]
    model = MLPScore(
        dim            = dim,
        hidden_dims    = tuple(model_cfg["hidden_dims"]),
        time_embed_dim = model_cfg.get("time_embed_dim", 64),
        activation     = model_cfg.get("activation", "silu"),
        predict_score  = model_cfg.get("predict_score", False),
    )
    state = torch.load(run_dir / "ema_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    sched_cfg = cfg["schedule"]
    schedule  = VPSchedule(
        beta_min=sched_cfg.get("beta_min", 0.1),
        beta_max=sched_cfg.get("beta_max", 20.0),
    )
    return ReverseSDE(model, schedule), energy, dim, beta_m, sample_run


def _mcmc_particles(sample_run: str, device: torch.device) -> torch.Tensor:
    pts = torch.load(
        SAMPLE_DIR / sample_run / "particles.pt",
        map_location=device, weights_only=True,
    )
    x = pts[:, -1, :]  # last snapshot: [N, dim]
    if x.shape[0] > FKC_N_PARTICLES:
        x = x[torch.randperm(x.shape[0], device=device)[:FKC_N_PARTICLES]]
    return x


def _energy_stats(x: torch.Tensor, energy_fn) -> dict:
    with torch.no_grad():
        e = energy_fn(x.cpu()).float()
    return {
        "mean_energy":   round(e.mean().item(),   4),
        "min_energy":    round(e.min().item(),    4),
        "std_energy":    round(e.std().item(),    4),
        "median_energy": round(e.median().item(), 4),
    }


def _run_fkc_and_ula(model_run: str, device: torch.device) -> dict:
    rsde, energy, dim, beta_m, sample_run = _load_model(model_run, device)

    # FKC — starts from Gaussian noise
    print("    FKC...")
    fkc_sampler = FKCAnnealedSampler(
        rsde, energy, beta_train=beta_m, beta_target=BETA_H,
        n_steps=FKC_N_STEPS,
        t_start=FKC_T_START,
        t_end=FKC_T_END,
        include_divergence=FKC_INCLUDE_DIVERGENCE,
        ess_threshold=FKC_ESS_THRESHOLD,
    )
    fkc_cloud, fkc_diag = fkc_sampler.run(FKC_N_PARTICLES, dim, device, show_progress=True)

    # ULA baseline — starts from MCMC last snapshot
    print("    ULA SMC...")
    x0          = _mcmc_particles(sample_run, device)
    init_cloud  = ParticleCloud(x0, torch.zeros(FKC_N_PARTICLES, device=device))
    beta_ladder = torch.linspace(beta_m, BETA_H, N_SMC_STEPS + 1, device=device)

    ula_proposal = ULAProposal(energy, step_size=ULA_STEP_SIZE, n_steps=N_ULA_STEPS)
    ula_sampler  = SMCSampler(
        ula_proposal.mutation_kernel, ula_proposal.weight_update, energy,
        ess_threshold=ULA_ESS_THRESHOLD,
    )
    ula_cloud, ula_diag = ula_sampler.run(init_cloud, beta_ladder, show_progress=True)

    return {
        "beta_m":    beta_m,
        "model_run": model_run,
        "fkc": {
            **_energy_stats(fkc_cloud.x, energy),
            "final_ess":   round(fkc_cloud.ess_ratio(), 4),
            "n_resamples": fkc_diag.n_resamples,
        },
        "ula": {
            **_energy_stats(ula_cloud.x, energy),
            "final_ess":   round(ula_cloud.ess_ratio(), 4),
            "n_resamples": ula_diag.n_resamples,
        },
    }


def _print_result(r: dict) -> None:
    f, u = r["fkc"], r["ula"]
    print(f"    FKC: mean={f['mean_energy']:.4f}  min={f['min_energy']:.4f}  ESS={f['final_ess']:.3f}")
    print(f"    ULA: mean={u['mean_energy']:.4f}  min={u['min_energy']:.4f}  ESS={u['final_ess']:.3f}")


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def _config_snapshot() -> dict:
    return {
        "energy":  ENERGY,
        "dim":     DIM,
        "beta_ms": BETA_MS,
        "beta_h":  BETA_H,
        "desc":    DESC,
        "seed":    SEED,
        "mcmc": {
            "n_particles": MCMC_N_PARTICLES,
            "n_steps":     MCMC_N_STEPS,
            "burn_in":     MCMC_BURN_IN,
            "save_every":  MCMC_SAVE_EVERY,
            "step_size":   MCMC_STEP_SIZE,
        },
        "model": {
            "hidden_dims":    MODEL_HIDDEN_DIMS,
            "time_embed_dim": MODEL_TIME_EMBED_DIM,
            "loss_type":      LOSS_TYPE,
            "train_n_steps":  TRAIN_N_STEPS,
            "batch_size":     BATCH_SIZE,
            "lr":             LR,
        },
        "fkc": {
            "n_particles":        FKC_N_PARTICLES,
            "n_steps":            FKC_N_STEPS,
            "t_start":            FKC_T_START,
            "t_end":              FKC_T_END,
            "include_divergence": FKC_INCLUDE_DIVERGENCE,
            "ess_threshold":      FKC_ESS_THRESHOLD,
        },
        "ula_baseline": {
            "n_particles":  FKC_N_PARTICLES,
            "n_smc_steps":  N_SMC_STEPS,
            "n_ula_steps":  N_ULA_STEPS,
            "step_size":    ULA_STEP_SIZE,
            "ess_threshold": ULA_ESS_THRESHOLD,
        },
    }


def save_results(runs: list[dict]) -> Path:
    out_dir = _experiment_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {"config": _config_snapshot(), "runs": runs}
    path = out_dir / "results.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {path}")
    return path


def load_results(path: Path | None = None) -> dict:
    p = path or (_experiment_dir() / "results.json")
    if not p.exists():
        raise FileNotFoundError(f"No results found at {p}. Run without --plot-only first.")
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(path: Path | None = None) -> None:
    import matplotlib.pyplot as plt

    data = load_results(path)
    cfg  = data["config"]
    runs = sorted(data["runs"], key=lambda r: r["beta_m"])

    beta_ms  = [r["beta_m"]              for r in runs]
    fkc_mean = [r["fkc"]["mean_energy"]  for r in runs]
    ula_mean = [r["ula"]["mean_energy"]  for r in runs]
    fkc_min  = [r["fkc"]["min_energy"]   for r in runs]
    ula_min  = [r["ula"]["min_energy"]   for r in runs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for ax, fkc_vals, ula_vals, ylabel in [
        (ax1, fkc_mean, ula_mean, "Mean energy"),
        (ax2, fkc_min,  ula_min,  "Min energy"),
    ]:
        ax.plot(beta_ms, fkc_vals, "o-", label="FKC",      color="mediumseagreen")
        ax.plot(beta_ms, ula_vals, "s-", label="ULA SMC",  color="steelblue")
        ax.set_xlabel("$\\beta_M$ (training temperature)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} at $\\beta_H = {cfg['beta_h']}$")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fkc_cfg  = cfg["fkc"]
    desc_str = f"  [{cfg['desc']}]" if cfg.get("desc") else ""
    subtitle = (
        f"{cfg['energy']}  d={cfg['dim']}{desc_str}\n"
        f"FKC: n_steps={fkc_cfg['n_steps']}  t_start={fkc_cfg['t_start']}  "
        f"incl_div={fkc_cfg['include_divergence']}  ess_thr={fkc_cfg['ess_threshold']}"
    )
    fig.suptitle(subtitle, fontsize=9)
    fig.tight_layout()

    out = _experiment_dir() / "plot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _detect_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip the pipeline and re-plot from the saved results.json",
    )
    args = parser.parse_args()

    if args.plot_only:
        plot_results()
        return

    device = _detect_device()
    torch.set_default_dtype(torch.float32)

    print(f"=== FKC experiment: {ENERGY}  d={DIM}  beta_H={BETA_H} ===")
    print(f"    beta_Ms={BETA_MS}  device={device}\n")

    print("[1/3] Sampling")
    sample_runs = setup_and_sample()

    print("\n[2/3] Training")
    model_runs = setup_and_train(sample_runs)

    print("\n[3/3] FKC + ULA baseline")
    results = run_all(model_runs, device)

    save_results(results)
    plot_results()


if __name__ == "__main__":
    main()
