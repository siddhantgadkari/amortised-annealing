# Amortised Annealing

Score-based diffusion models trained on MCMC samples for amortised annealing.

---

## Workflow overview

```
1. create sampling configs   →   2. run sampling   →   3. create training configs
                                                                    ↓
                         6. diagnose   ←   5. eval model   ←   4. train model
```

---

## 1. Generate sampling configs

Sweeps over energies × dims × beta values × MCMC methods and writes one YAML per combination to `configs/sampling/`.

```bash
# Generate all 128 configs (4 energies × 4 dims × 4 betas × 2 methods)
python scripts/create_sampler_configs.py
```

Sweep axes are set at the top of the script:
```python
DIMS     = [2, 5, 10, 20]
BETA_MS  = [0.1, 1.0, 5.0, 10.0]
ENERGIES = ["double_well", "many_well", "ackley", "rastrigin"]
METHODS  = ["ULA", "MALA"]
```

Output config location: `configs/sampling/{energy}_d{dim}_beta{beta}_{method}_seed{seed}.yaml`

Example config (`configs/sampling/rastrigin_d10_beta5_ula_seed0.yaml`):
```yaml
job:
  seed: 0
  device: auto      # auto = cuda if available, else cpu
  dtype: float32

target:
  energy: rastrigin
  dim: 10
  beta_m: 5.0

sampler:
  method: ULA
  step_size: 0.001
  n_particles: 8192
  n_steps: 10000
  burn_in: 2000
  save_every: 100   # save a snapshot every 100 steps → 100 snapshots total

output:
  root: runs/sampling
  run_name: rastrigin_d10_beta5_ula_seed0
```

---

## 2. Run sampling

Runs MCMC from a config and saves results to `data/samples/{run_name}/`.

```bash
# Run one config
python scripts/run_sampling.py configs/sampling/rastrigin_d10_beta5_ula_seed0.yaml

# Run all configs in configs/sampling/
python scripts/run_sampling.py

# Run multiple specific configs
python scripts/run_sampling.py configs/sampling/rastrigin_d10_beta5_ula_seed0.yaml \
                               configs/sampling/rastrigin_d2_beta5_ula_seed0.yaml
```

Output per run (`data/samples/{run_name}/`):
| File | Description |
|------|-------------|
| `particles.pt` | `[n_particles, num_snapshots, dim]` tensor — snapshots of all chains |
| `config.yaml` | Copy of the input config |
| `summary.json` | Energy stats, acceptance rate, wall-clock time, hardware info |

`num_snapshots = n_steps / save_every` (e.g. 10000 / 100 = 100 snapshots).

---

## 3. Generate training configs

Sweeps over sample sets × model presets × seeds and writes configs to `configs/models/`.

```bash
# All available sample dirs × all 3 model presets × seed 0
python scripts/create_training_configs.py

# Only rastrigin samples, medium model only
python scripts/create_training_configs.py --filter rastrigin --presets mlp256x3

# Specific sample run, two model sizes, multiple seeds
python scripts/create_training_configs.py \
  --samples rastrigin_d10_beta5_ula_seed0 \
  --presets mlp256x3 mlp512x4 \
  --seeds 0 1 2

# All samples matching a substring, small model, two seeds
python scripts/create_training_configs.py --filter d10 --presets mlp128x3 --seeds 0 1
```

Available model presets:
| Preset | Architecture |
|--------|-------------|
| `mlp128x3` | 3 × 128 hidden, time_embed=64 |
| `mlp256x3` | 3 × 256 hidden, time_embed=64 |
| `mlp512x4` | 4 × 512 hidden, time_embed=128 |

Output config location: `configs/models/{sample_run_name}_{preset}_seed{seed}.yaml`

Example config (`configs/models/rastrigin_d10_beta5_ula_seed0_mlp256x3_seed0.yaml`):
```yaml
job:
  seed: 0
  device: auto
  dtype: float32

samples:
  run_name: rastrigin_d10_beta5_ula_seed0
  snapshot: -1    # -1 = final snapshot only (8192 samples)
                  # null = all snapshots flattened (8192 × 100 = 819200 samples)
                  # int  = specific snapshot index

schedule:
  type: vp
  beta_min: 0.1
  beta_max: 20.0

model:
  hidden_dims: [256, 256, 256]
  time_embed_dim: 64
  activation: silu
  predict_score: false

training:
  n_steps: 20000
  batch_size: 512
  lr: 2e-4
  t_eps: 1e-4
  grad_clip: 1.0
  ema_decay: 0.999
  log_every: 500
  log_uniform_t: false   # set true if small-t bins have high loss (see diagnose)

output:
  root: data/models
  run_name: rastrigin_d10_beta5_ula_seed0_mlp256x3_seed0
```

**Tips for multimodal targets (Rastrigin, Ackley):**
- Use `snapshot: null` + `batch_size: 4096` to train on all snapshots
- Enable `log_uniform_t: true` to focus training on sharp low-noise scores
- Run `diagnose_score.py` first to check if this is needed (see step 6)

---

## 4. Train model

Trains a score model from a config and saves results to `data/models/{run_name}/`.

```bash
# Train from a specific config
python scripts/run_training.py configs/models/rastrigin_d10_beta5_ula_seed0_mlp256x3_seed0.yaml

# Train all configs in configs/models/
python scripts/run_training.py
```

Output per run (`data/models/{run_name}/`):
| File | Description |
|------|-------------|
| `ema_model.pt` | EMA-smoothed model weights (use this for sampling) |
| `config.yaml` | Copy of the training config |
| `summary.json` | Final loss, loss history, n_params, wall-clock time |

---

## 5. Evaluate model

Generates samples from a trained model and prints energy statistics side-by-side against the MCMC training samples.

```bash
python scripts/eval_model.py rastrigin_d10_beta5_ula_seed0_mlp256x3_seed0

# More samples, more EM steps
python scripts/eval_model.py rastrigin_d10_beta5_ula_seed0_mlp256x3_seed0 \
  --n-samples 8192 --n-steps 1000 --progress
```

Example output:
```
Generating 4096 samples with 500 EM steps...

Energy: rastrigin  dim=10  beta_m=5.0
MCMC samples:  rastrigin_d10_beta5_ula_seed0
Score model:   rastrigin_d10_beta5_ula_seed0_mlp256x3_seed0

                       diffusion  mcmc (train)
--------------------------------------------------
  mean                   14.2301       12.7000
  median                 12.1000       10.9000
  min                     3.8000        3.1000
  ...
```

---

## 6. Diagnose score quality

Measures DSM loss per time bin to identify if the model fails at small `t` (the low-noise regime where it needs to resolve data modes).

```bash
python scripts/diagnose_score.py rastrigin_d10_beta5_ula_seed0_mlp256x3_seed0
```

Example output:
```
DSM loss by time bin
  bin               loss      interpretation
  t=[0.0001,0.01]   9.97    <-- sharp score (modes matter here)
  t=[0.01,  0.05]   9.95    <-- sharp score (modes matter here)
  t=[0.05,  0.1 ]   9.46    <-- transitional
  t=[0.1,   0.3 ]   6.65    <-- transitional
  t=[0.3,   0.6 ]   1.63        high noise (easy)
  t=[0.6,   1.0 ]   0.05        high noise (easy)
```

If small-`t` bins are much worse than large-`t` bins: set `log_uniform_t: true` in the training config and retrain.

---

## 7. Sanity-check the reverse SDE

Tests the SDE integrator in isolation using an analytic Gaussian score (no trained model needed). Should always PASS.

```bash
python scripts/test_reverse_sde.py
python scripts/test_reverse_sde.py --dim 20 --n-steps 1000
```

---

## Directory layout

```
configs/
  sampling/          MCMC sampling configs (generated by create_sampler_configs.py)
  models/            Training configs (generated by create_training_configs.py)

data/
  samples/{run_name}/
    particles.pt     [n_particles, num_snapshots, dim]
    config.yaml
    summary.json
  models/{run_name}/
    ema_model.pt
    config.yaml
    summary.json

scripts/
  create_sampler_configs.py
  run_sampling.py
  create_training_configs.py
  run_training.py
  eval_model.py
  diagnose_score.py
  test_reverse_sde.py

src/amortised_annealing/
  energies/          DoubleWell, ManyWell, Ackley, Rastrigin
  mcmc/              ULA, MALA
  sampling/          SampleGenerator
  diffusion/         VPSchedule, MLPScore, ReverseSDE, euler_maruyama_sample,
                     dsm_loss, loss_by_t_bin, train_score_model
```
