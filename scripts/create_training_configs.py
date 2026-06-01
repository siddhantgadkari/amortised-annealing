from __future__ import annotations
import argparse
from itertools import product
from pathlib import Path
import yaml

# --- which sample sets to train on ---
# Each entry is a run_name from configs/sampling/ (i.e. a data/samples/{run_name}/ dir).
# Set to None to use all available sample dirs automatically.
SAMPLE_RUN_NAMES = None  # or e.g. ["rastrigin_d10_beta5_ula_seed0", ...]

# --- model architecture presets ---
MODEL_PRESETS = {
    "mlp128x3": {"hidden_dims": [128, 128, 128], "time_embed_dim": 64},
    "mlp256x3": {"hidden_dims": [256, 256, 256], "time_embed_dim": 64},
    "mlp512x4": {"hidden_dims": [512, 512, 512, 512], "time_embed_dim": 128},
}

# --- fixed model defaults ---
ACTIVATION = "silu"

# --- fixed training defaults ---
N_STEPS       = 20_000
BATCH_SIZE    = 512
LR            = 2e-4
T_EPS         = 1e-4
GRAD_CLIP     = 1.0
EMA_DECAY     = 0.999
LOG_EVERY     = 500
LOG_UNIFORM_T = False
LOSS_TYPES    = ["eps"]   # sweep over e.g. ["eps", "score"]
SEEDS         = [0]

# --- schedule ---
SCHEDULE = {"type": "vp", "beta_min": 0.1, "beta_max": 20.0}

# --- snapshot to use from the [n_particles, num_snapshots, dim] tensor ---
# -1: final snapshot (recommended); null: flatten all; integer: specific index
SNAPSHOT = -1

ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data" / "samples"
OUT_DIR     = ROOT / "configs" / "models"


def _get_sample_run_names() -> list[str]:
    if SAMPLE_RUN_NAMES is not None:
        return SAMPLE_RUN_NAMES
    return sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())


def _model_tag(preset_name: str) -> str:
    return preset_name


def _make_config(run_name: str, preset_name: str, preset: dict, loss_type: str, seed: int) -> dict:
    model_run_name = f"{run_name}_{_model_tag(preset_name)}_{loss_type}_seed{seed}"
    return {
        "job": {
            "seed":   seed,
            "device": "auto",
            "dtype":  "float32",
        },
        "samples": {
            "run_name": run_name,
            "snapshot": SNAPSHOT,
        },
        "schedule": SCHEDULE,
        "model": {
            "hidden_dims":    preset["hidden_dims"],
            "time_embed_dim": preset["time_embed_dim"],
            "activation":     ACTIVATION,
            "predict_score":  loss_type == "score",
        },
        "training": {
            "n_steps":       N_STEPS,
            "batch_size":    BATCH_SIZE,
            "lr":            LR,
            "t_eps":         T_EPS,
            "grad_clip":     GRAD_CLIP,
            "ema_decay":     EMA_DECAY,
            "log_every":     LOG_EVERY,
            "log_uniform_t": LOG_UNIFORM_T,
            "loss_type":     loss_type,
        },
        "output": {
            "root":     "data/models",
            "run_name": model_run_name,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training configs by sweeping sample sets × model presets × seeds.")
    parser.add_argument("--samples", nargs="+", metavar="RUN_NAME",
                        help="Exact sample run names to use. Defaults to all dirs in data/samples/.")
    parser.add_argument("--filter", metavar="SUBSTR",
                        help="Substring filter applied to auto-discovered sample run names (e.g. 'rastrigin').")
    parser.add_argument("--presets", nargs="+", choices=list(MODEL_PRESETS), metavar="PRESET",
                        help=f"Model presets to use. Choices: {list(MODEL_PRESETS)}. Defaults to all.")
    parser.add_argument("--seeds", nargs="+", type=int, metavar="SEED",
                        help="Seeds to sweep. Defaults to [0].")
    parser.add_argument("--loss-types", nargs="+", choices=["eps", "score"], metavar="LOSS",
                        help="Loss types to sweep. Choices: eps, score. Defaults to ['eps'].")
    args = parser.parse_args()

    # Resolve sample run names
    if args.samples:
        sample_runs = args.samples
    else:
        all_runs = _get_sample_run_names()
        sample_runs = [r for r in all_runs if args.filter in r] if args.filter else all_runs

    if not sample_runs:
        print(f"No sample dirs found in {DATA_DIR}. Run run_sampling.py first.")
        return

    presets     = {k: MODEL_PRESETS[k] for k in (args.presets or MODEL_PRESETS)}
    seeds       = args.seeds or SEEDS
    loss_types  = args.loss_types or LOSS_TYPES

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for run_name, (preset_name, preset), loss_type, seed in product(
        sample_runs, presets.items(), loss_types, seeds
    ):
        cfg = _make_config(run_name, preset_name, preset, loss_type, seed)
        out_path = OUT_DIR / f"{cfg['output']['run_name']}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        count += 1

    print(f"Generated {count} configs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
