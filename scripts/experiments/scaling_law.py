#!/usr/bin/env python
"""Empirical training-data scaling diagnostic.

Investigates how model fidelity (delta_E vs holdout chain) and downstream
optimisation quality (Algorithm 2 at beta_H=50) vary with training sample count
N_train, under fixed architecture and training budget.

Cross-seed holdout: train on seed s, fidelity evaluated against seed (s+1)%3.
Existing N_train=2048 and N_train=8192 models from cost_quality.py are reused.
Zero new MCMC data is generated.

Usage:
    # Per-seed on server (run in parallel):
    uv run python scripts/experiments/scaling_law.py --seed 0 --no-plot
    uv run python scripts/experiments/scaling_law.py --seed 1 --no-plot
    uv run python scripts/experiments/scaling_law.py --seed 2 --no-plot

    # Aggregate + plot locally:
    uv run python scripts/experiments/scaling_law.py --aggregate --no-plot
    uv run python scripts/experiments/scaling_law.py --plot-only
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from amortised_annealing.diffusion import (
    VPSchedule, MLPScore, ReverseSDE, TrainingConfig, train_score_model,
)
from amortised_annealing.energies import DoubleWell, ManyWell, Ackley, Rastrigin
from amortised_annealing.sampling import DiffusionModelSampler

ROOT       = Path(__file__).parent.parent.parent
SAMPLE_DIR = ROOT / "data" / "samples"
MODEL_DIR  = ROOT / "data" / "models"
EXP_BASE   = ROOT / "data" / "experiments" / "scaling_law"

ENERGY_MAP = {
    "double_well": DoubleWell, "many_well": ManyWell,
    "ackley": Ackley, "rastrigin": Rastrigin,
}

# ===========================================================================
# USER CONFIGURATION
# ===========================================================================
ENERGY   = "ackley"
DIMS     = [5, 10, 20]
BETA_MS  = [1.0, 5.0, 20.0]
N_TRAINS = [256, 512, 1024, 2048, 8192]
SEEDS    = [0, 1, 2]

BETA_H_EVAL   = 50.0
N_EVAL        = 8192    # direct diffusion samples for fidelity (no ULA)
N_DOWNSTREAM  = 8192    # Algorithm 2 batch size
LOCAL_STEPS   = 10
FIDELITY_TAUS = [0.05, 0.10, 0.20]   # delta_E thresholds for N* computation

# Architecture / training — must match cost_quality.py exactly
MODEL_HIDDEN_DIMS    = [256, 256, 256]
MODEL_TIME_EMBED_DIM = 64
LOSS_TYPE            = "eps"
MCMC_N_PARTICLES     = 8192
MCMC_N_STEPS         = 10_000
TRAIN_N_STEPS        = 20_000
BATCH_SIZE           = 512
LR                   = 2e-4
T_EPS                = 1e-4
GRAD_CLIP            = 1.0
EMA_DECAY            = 0.999
LOG_EVERY            = 500
MODEL_SAMPLE_N_STEPS = 500
MODEL_SAMPLE_T_START = 1.0
MODEL_SAMPLE_T_END   = 1e-3
LOCAL_ULA_STEP_SIZE  = 1e-3

REUSE_MODELS = True
REUSE_EVALS  = True
# ===========================================================================


def _exp_dir() -> Path:
    return EXP_BASE / ENERGY


def _beta_str(b: float) -> str:
    return f"{b:g}".replace(".", "p")


def _preset_tag() -> str:
    s = MODEL_HIDDEN_DIMS
    return f"mlp{s[0]}x{len(s)}" if len(set(s)) == 1 else "mlp_" + "_".join(str(x) for x in s)


def _sample_run_name(dim: int, beta_m: float, seed: int) -> str:
    return f"{ENERGY}_d{dim}_beta{_beta_str(beta_m)}_ula_seed{seed}"


def _model_run_name_ntrain(sample_run: str, n_train: int, seed: int) -> str:
    if n_train == MCMC_N_PARTICLES:
        return f"{sample_run}_{_preset_tag()}_{LOSS_TYPE}_seed{seed}"
    return f"{sample_run}_ntrain{n_train}_{_preset_tag()}_{LOSS_TYPE}_seed{seed}"


def _result_path(dim: int, beta_m: float, n_train: int, seed: int) -> Path:
    return _exp_dir() / f"results_d{dim}_bm{_beta_str(beta_m)}_ntrain{n_train}_seed{seed}.json"


# ---------------------------------------------------------------------------
# Model training (mirrors cost_quality._ensure_model_ntrain, adds best_loss)
# ---------------------------------------------------------------------------

def _ensure_model_ntrain(
    sample_run: str, n_train: int, seed: int, device: torch.device,
) -> tuple[str, float | None]:
    """Train or reuse model. Returns (model_run, best_train_loss or None if reused)."""
    model_run = _model_run_name_ntrain(sample_run, n_train, seed)
    out_dir   = MODEL_DIR / model_run

    if REUSE_MODELS and (out_dir / "ema_model.pt").exists():
        print(f"    [train]  REUSE  {model_run}")
        return model_run, None

    print(f"    [train]  RUN    {model_run}  (n_train={n_train})")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    x_all = torch.load(SAMPLE_DIR / sample_run / "particles.pt",
                       map_location="cpu", weights_only=True)
    dim   = x_all.shape[-1]

    if n_train < x_all.shape[0]:
        g   = torch.Generator().manual_seed(seed)
        idx = torch.randperm(x_all.shape[0], generator=g)[:n_train]
        x_data = x_all[idx]
    else:
        x_data = x_all

    schedule = VPSchedule(beta_min=0.1, beta_max=20.0)
    model    = MLPScore(
        dim=dim, hidden_dims=tuple(MODEL_HIDDEN_DIMS),
        time_embed_dim=MODEL_TIME_EMBED_DIM, activation="silu", predict_score=False,
    )
    cfg_tr = TrainingConfig(
        n_steps=TRAIN_N_STEPS, batch_size=BATCH_SIZE, lr=LR, t_eps=T_EPS,
        grad_clip=GRAD_CLIP, ema_decay=EMA_DECAY, log_every=LOG_EVERY,
        loss_type=LOSS_TYPE, seed=seed,
    )
    t0 = time.perf_counter()
    ema_model, loss_history = train_score_model(model, schedule, x_data, cfg_tr, device)
    wall = round(time.perf_counter() - t0, 2)

    final_loss = round(loss_history[-1], 6) if loss_history else None
    best_loss  = round(min(loss_history),  6) if loss_history else None

    torch.save(ema_model.state_dict(), out_dir / "ema_model.pt")
    cfg_yaml = {
        "job":      {"seed": seed, "device": "auto", "dtype": "float32"},
        "samples":  {"run_name": sample_run},
        "n_train":  n_train,
        "schedule": {"type": "vp", "beta_min": 0.1, "beta_max": 20.0},
        "model":    {"hidden_dims": MODEL_HIDDEN_DIMS, "time_embed_dim": MODEL_TIME_EMBED_DIM,
                     "activation": "silu", "predict_score": False},
        "training": {"n_steps": TRAIN_N_STEPS, "batch_size": BATCH_SIZE, "lr": LR,
                     "t_eps": T_EPS, "grad_clip": GRAD_CLIP, "ema_decay": EMA_DECAY,
                     "log_every": LOG_EVERY, "loss_type": LOSS_TYPE},
        "output":   {"run_name": model_run},
    }
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg_yaml, f, sort_keys=False)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "status": "completed", "run_name": model_run, "sample_run_name": sample_run,
            "n_training_samples": int(x_data.shape[0]), "n_train": n_train, "dim": int(dim),
            "n_params": sum(p.numel() for p in ema_model.parameters()),
            "wall_clock_seconds": wall, "final_loss": final_loss, "best_loss": best_loss,
        }, f, indent=2)
    print(f"           done in {wall:.1f}s  final={final_loss}  best={best_loss}")
    return model_run, best_loss


def _load_model_run(model_run: str, device: torch.device):
    """Returns (reverse_sde, energy_obj, dim, beta_m). Identical to cost_quality.py."""
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
        dim=dim, hidden_dims=tuple(mc["hidden_dims"]),
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
    x: torch.Tensor, energy_fn, beta_h: float, n_steps: int, seed: int,
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


def _quantile_stats(e: torch.Tensor) -> dict:
    qs = torch.quantile(e, torch.tensor([0.01, 0.05, 0.10, 0.50, 0.90]))
    return {
        "q01":  round(float(qs[0]), 5), "q05": round(float(qs[1]), 5),
        "q10":  round(float(qs[2]), 5), "q50": round(float(qs[3]), 5),
        "q90":  round(float(qs[4]), 5),
        "mean": round(float(e.mean()), 5),
        "min":  round(float(e.min()),  5),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate_model(
    dim: int, beta_m: float, n_train: int, seed: int,
    model_run: str, device: torch.device,
    best_train_loss: float | None = None,
) -> None:
    """Compute fidelity + downstream for one model and save result JSON."""
    rpath = _result_path(dim, beta_m, n_train, seed)
    if REUSE_EVALS and rpath.exists():
        print(f"    [eval]   REUSE  {model_run}")
        return

    rsde, energy, _dim, _bm = _load_model_run(model_run, device)
    energy_fn = energy.energy

    # Holdout: independent MCMC chain from adjacent seed
    holdout_seed = (seed + 1) % 3
    holdout_run  = _sample_run_name(dim, beta_m, holdout_seed)
    holdout_x = torch.load(SAMPLE_DIR / holdout_run / "particles.pt",
                            map_location="cpu", weights_only=True)
    with torch.no_grad():
        e_hold = energy_fn(holdout_x.cpu()).float()
    q_hold = _quantile_stats(e_hold)

    sampler = DiffusionModelSampler(
        rsde, n_steps=MODEL_SAMPLE_N_STEPS,
        t_start=MODEL_SAMPLE_T_START, t_end=MODEL_SAMPLE_T_END,
    )

    # --- Fidelity: direct diffusion samples, no ULA ---
    torch.manual_seed(seed * 100 + 7)
    with torch.no_grad():
        x_eval = sampler.sample(N_EVAL, device, show_progress=False)
    with torch.no_grad():
        e_eval = energy_fn(x_eval.cpu()).float()
    q_eval = _quantile_stats(e_eval)

    delta_E = (abs(q_eval["q10"] - q_hold["q10"])
               + abs(q_eval["q50"] - q_hold["q50"])
               + abs(q_eval["q90"] - q_hold["q90"])) / (1 + abs(q_hold["q50"]))
    delta_tail = (abs(q_eval["q01"] - q_hold["q01"])
                  + abs(q_eval["q05"] - q_hold["q05"])) / (1 + abs(q_hold["q05"]))

    # --- Downstream: Algorithm 2 at beta_H_eval ---
    torch.manual_seed(seed * 100 + 13)
    with torch.no_grad():
        x_down = sampler.sample(N_DOWNSTREAM, device, show_progress=False)
    x_down = _apply_local_ula(x_down, energy_fn, BETA_H_EVAL, LOCAL_STEPS, seed * 100 + 17)
    with torch.no_grad():
        e_down = energy_fn(x_down.cpu()).float()
    q_down = _quantile_stats(e_down)

    # Retrieve training loss from model summary if not passed from fresh training
    final_train_loss = None
    model_summary_path = MODEL_DIR / model_run / "summary.json"
    if model_summary_path.exists():
        msummary = json.loads(model_summary_path.read_text())
        final_train_loss = msummary.get("final_loss")
        if best_train_loss is None:
            best_train_loss = msummary.get("best_loss")

    result = {
        "dim": dim, "beta_m": beta_m, "n_train": n_train, "seed": seed,
        "holdout_seed": holdout_seed,
        "counterfactual_setup_cost": n_train * MCMC_N_STEPS,
        "final_train_loss": final_train_loss,
        "best_train_loss":  best_train_loss,
        "delta_E":    round(delta_E,    6),
        "delta_tail": round(delta_tail, 6),
        "holdout":    q_hold,
        "model_eval": q_eval,
        "downstream": {
            **{f"downstream_{k}": v for k, v in q_down.items()},
            "downstream_oracle_cost_per_use": N_DOWNSTREAM * LOCAL_STEPS,
        },
    }
    _exp_dir().mkdir(parents=True, exist_ok=True)
    rpath.write_text(json.dumps(result, indent=2))
    print(f"    [eval]   DONE   {model_run}  "
          f"δE={delta_E:.4f}  δtail={delta_tail:.4f}  "
          f"down_best={q_down['min']:.4f}")


# ---------------------------------------------------------------------------
# Per-seed run
# ---------------------------------------------------------------------------

def run_seed(seed: int, device: torch.device) -> None:
    for dim in DIMS:
        for beta_m in BETA_MS:
            sample_run = _sample_run_name(dim, beta_m, seed)
            sample_path = SAMPLE_DIR / sample_run / "particles.pt"
            if not sample_path.exists():
                print(f"  SKIP  {sample_run} — particles.pt not found")
                continue

            print(f"\n  dim={dim}  β_M={beta_m}")
            for n_train in N_TRAINS:
                model_run, best_loss = _ensure_model_ntrain(sample_run, n_train, seed, device)
                _evaluate_model(dim, beta_m, n_train, seed, model_run, device, best_loss)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _agg(vals: list) -> dict:
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"mean": None, "std": None}
    return {
        "mean": round(statistics.mean(vals), 6),
        "std":  round(statistics.stdev(vals), 6) if len(vals) > 1 else 0.0,
    }


def aggregate_results() -> None:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for path in sorted(_exp_dir().glob("results_d*_bm*_ntrain*_seed*.json")):
        d = json.loads(path.read_text())
        key = (d["dim"], d["beta_m"], d["n_train"])
        groups[key].append(d)

    if not groups:
        print("  No result files found.")
        return

    agg_rows: list[dict] = []
    for (dim, beta_m, n_train), entries in sorted(groups.items()):
        row: dict = {
            "dim": dim, "beta_m": beta_m, "n_train": n_train,
            "n_seeds": len(entries),
            "counterfactual_setup_cost": n_train * MCMC_N_STEPS,
        }
        for key in ("delta_E", "delta_tail", "final_train_loss", "best_train_loss"):
            row[key] = _agg([e.get(key) for e in entries])

        for key in ("min", "q01", "q05", "q10", "q50", "q90", "mean"):
            row[f"model_{key}"] = _agg([e["model_eval"].get(key) for e in entries])
            row[f"holdout_{key}"] = _agg([e["holdout"].get(key) for e in entries])
            row[f"downstream_{key}"] = _agg([
                e["downstream"].get(f"downstream_{key}") for e in entries
            ])

        agg_rows.append(row)

    # Compute N* for each (dim, beta_m) and each tau
    nstar: list[dict] = []
    for dim in DIMS:
        for beta_m in BETA_MS:
            curve = sorted(
                [r for r in agg_rows if r["dim"] == dim and r["beta_m"] == beta_m],
                key=lambda r: r["n_train"],
            )
            for tau in FIDELITY_TAUS:
                found = next(
                    (r["n_train"] for r in curve
                     if (r["delta_E"].get("mean") or float("inf")) <= tau),
                    None,
                )
                nstar.append({
                    "dim": dim, "beta_m": beta_m, "tau": tau,
                    "n_star": found if found is not None else "not_reached",
                })

    out = {
        "n_seeds": len(SEEDS),
        "results": agg_rows,
        "n_star": nstar,
    }
    out_path = _exp_dir() / "aggregate.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  Saved → {out_path}  ({len(agg_rows)} rows)")

    # Print N* table
    print(f"\n{'N* TABLE':=<60}")
    print(f"{'dim':<5} {'beta_m':<8} {'tau':<6} {'N*'}")
    print("-" * 40)
    for r in nstar:
        print(f"{r['dim']:<5} {r['beta_m']:<8} {r['tau']:<6} {r['n_star']}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results() -> None:
    agg_path = _exp_dir() / "aggregate.json"
    if not agg_path.exists():
        print(f"No aggregate.json at {agg_path} — run --aggregate first.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data     = json.loads(agg_path.read_text())
    results  = data["results"]
    nstar    = data["n_star"]
    plots_dir = _exp_dir() / "plots"
    plots_dir.mkdir(exist_ok=True)

    dims    = sorted(set(r["dim"]    for r in results))
    beta_ms = sorted(set(r["beta_m"] for r in results))
    bm_colors = {1.0: "steelblue", 5.0: "darkorange", 20.0: "mediumseagreen"}

    def _m(d):
        return d["mean"] if isinstance(d, dict) and d["mean"] is not None else float("nan")

    def _s(d):
        return d["std"]  if isinstance(d, dict) and d["std"]  is not None else 0.0

    # --- Plot 1: delta_E vs N_train per dim ---
    for metric_key, metric_label, fname in [
        ("delta_E",    "ΔE (distributional fidelity)", "delta_E"),
        ("delta_tail", "Δtail (low-energy fidelity)",  "delta_tail"),
    ]:
        for dim in dims:
            fig, ax = plt.subplots(figsize=(6, 4))
            for beta_m in beta_ms:
                pts = sorted(
                    [r for r in results if r["dim"] == dim and r["beta_m"] == beta_m],
                    key=lambda r: r["n_train"],
                )
                if not pts:
                    continue
                xs = [r["n_train"] for r in pts]
                ys = [_m(r[metric_key]) for r in pts]
                es = [_s(r[metric_key]) for r in pts]
                color = bm_colors.get(beta_m, "purple")
                ax.errorbar(xs, ys, yerr=es, fmt="o-", color=color, capsize=3, lw=1.5,
                            label=f"β_M={beta_m:g}")

            # Add tau threshold lines
            for tau in FIDELITY_TAUS:
                ax.axhline(tau, color="grey", ls=":", lw=0.8, alpha=0.6)
                ax.text(max(N_TRAINS) * 1.02, tau, f"τ={tau}", fontsize=6, va="center")

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("N_train", fontsize=9)
            ax.set_ylabel(metric_label, fontsize=9)
            ax.set_title(f"{ENERGY}  d={dim}  {metric_label}", fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            out = plots_dir / f"{fname}_d{dim}.svg"
            fig.tight_layout()
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {out.name}")

    # --- Plot 2: downstream_best vs N_train per dim ---
    for dim in dims:
        fig, ax = plt.subplots(figsize=(6, 4))
        for beta_m in beta_ms:
            pts = sorted(
                [r for r in results if r["dim"] == dim and r["beta_m"] == beta_m],
                key=lambda r: r["n_train"],
            )
            if not pts:
                continue
            xs = [r["n_train"] for r in pts]
            ys = [_m(r["downstream_min"]) for r in pts]
            es = [_s(r["downstream_min"]) for r in pts]
            color = bm_colors.get(beta_m, "purple")
            ax.errorbar(xs, ys, yerr=es, fmt="o-", color=color, capsize=3, lw=1.5,
                        label=f"β_M={beta_m:g}")

        ax.set_xscale("log")
        ax.set_xlabel("N_train", fontsize=9)
        ax.set_ylabel(f"Downstream best energy (β_H={BETA_H_EVAL:g})", fontsize=9)
        ax.set_title(f"{ENERGY}  d={dim}  Downstream optimisation quality", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        out = plots_dir / f"downstream_best_d{dim}.svg"
        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out.name}")

    # --- Plot 3: N* heatmap for each tau ---
    for tau in FIDELITY_TAUS:
        tau_rows = [r for r in nstar if r["tau"] == tau]
        if not tau_rows:
            continue

        mat = np.full((len(dims), len(beta_ms)), float("nan"))
        for r in tau_rows:
            ri = dims.index(r["dim"])
            ci = beta_ms.index(r["beta_m"])
            if r["n_star"] != "not_reached":
                mat[ri, ci] = r["n_star"]

        fig, ax = plt.subplots(figsize=(5, 3.5))
        im = ax.imshow(mat, aspect="auto", cmap="viridis_r")
        plt.colorbar(im, ax=ax, label="N* (training samples needed)")
        ax.set_xticks(range(len(beta_ms)))
        ax.set_yticks(range(len(dims)))
        ax.set_xticklabels([f"β_M={b:g}" for b in beta_ms])
        ax.set_yticklabels([f"d={d}" for d in dims])
        for ri in range(len(dims)):
            for ci in range(len(beta_ms)):
                val = tau_rows[ri * len(beta_ms) + ci]["n_star"]
                ax.text(ci, ri, str(val), ha="center", va="center", fontsize=8,
                        color="white" if not np.isnan(mat[ri, ci]) else "red")
        ax.set_title(f"N* (δE ≤ {tau}) — {ENERGY}", fontsize=10)
        out = plots_dir / f"nstar_heatmap_tau{str(tau).replace('.', 'p')}.svg"
        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out.name}")

    print(f"\nAll plots saved to {plots_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Empirical training-data scaling diagnostic."
    )
    parser.add_argument("--seed",      type=int, default=None, help="Run single seed")
    parser.add_argument("--aggregate", action="store_true",    help="Aggregate result JSONs")
    parser.add_argument("--plot-only", action="store_true",    help="Plot from aggregate.json")
    parser.add_argument("--no-plot",   action="store_true",    help="Skip plotting")
    args = parser.parse_args()

    if args.plot_only:
        plot_results()
        return

    if args.aggregate:
        print("Aggregating...")
        aggregate_results()
        if not args.no_plot:
            plot_results()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)

    seeds = [args.seed] if args.seed is not None else SEEDS
    _exp_dir().mkdir(parents=True, exist_ok=True)
    print(f"=== SCALING LAW  {ENERGY}  dims={DIMS}  β_M={BETA_MS}  "
          f"N_trains={N_TRAINS}  seeds={seeds} ===\n")

    for seed in seeds:
        print(f"\n{'='*60}\n  SEED {seed}\n{'='*60}")
        run_seed(seed, device)

    if args.seed is None:
        print("\nAggregating...")
        aggregate_results()
        if not args.no_plot:
            plot_results()
    else:
        print(f"\nSeed {args.seed} done. Run --aggregate when all seeds complete.")


if __name__ == "__main__":
    main()
