from __future__ import annotations
from itertools import product
from pathlib import Path
import yaml

# --- sweep axes ---
DIMS     = [2, 5, 10, 20]
BETA_MS  = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
ENERGIES = ["double_well", "many_well", "ackley", "rastrigin"]
METHODS  = ["ULA", "MALA"]

# --- fixed sampler defaults ---
SEED        = 0
DEVICE      = "auto"
DTYPE       = "float32"
STEP_SIZE   = 0.001
N_PARTICLES = 8192
N_STEPS     = 10000
BURN_IN     = 2000
SAVE_EVERY  = 100
INIT_SCALE  = 1.0
OUTPUT_ROOT = "runs/sampling"

OUT_DIR = Path(__file__).parent.parent / "configs" / "sampling"


def _beta_str(beta: float) -> str:
    return f"{beta:g}".replace(".", "p")


def _make_config(energy: str, dim: int, beta_m: float, method: str) -> dict:
    run_name = f"{energy}_d{dim}_beta{_beta_str(beta_m)}_{method.lower()}_seed{SEED}"
    return {
        "job":     {"seed": SEED, "device": DEVICE, "dtype": DTYPE},
        "target":  {"energy": energy, "dim": dim, "beta_m": beta_m},
        "sampler": {
            "method":       method,
            "step_size":    STEP_SIZE,
            "n_particles":  N_PARTICLES,
            "n_steps":      N_STEPS,
            "burn_in":      BURN_IN,
            "save_every":   SAVE_EVERY,
            "init_scale":   INIT_SCALE,
        },
        "output":  {"root": OUTPUT_ROOT, "run_name": run_name},
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    for energy, dim, beta_m, method in product(ENERGIES, DIMS, BETA_MS, METHODS):
        cfg = _make_config(energy, dim, beta_m, method)
        out_path = OUT_DIR / f"{cfg['output']['run_name']}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        count += 1

    print(f"Generated {count} configs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
