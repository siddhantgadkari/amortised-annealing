# Amortised Annealing

Research code for a final year thesis. The project investigates whether score-based diffusion models trained on MCMC samples collected at an intermediate inverse temperature &beta;<sub>M</sub> can be reused to generate high-quality samples at a higher target temperature &beta;<sub>H</sub>, enabling amortised annealing without re-running MCMC from scratch. Experiments study the trade-off between training cost, sample quality, and the degree to which a single trained model can be transferred across temperatures.

> **Note:** This is research code, not a polished library. Some scripts correspond to exploratory experiments, interfaces may be inconsistent, and full reproducibility requires substantial compute.

---

## What this repository contains

**Energy functions** (`src/amortised_annealing/energies/`)
- Double Well and Many Well (both implemented in `double_well.py`)
- Ackley and Rastrigin — high-dimensional multimodal benchmarks

**MCMC samplers** (`src/amortised_annealing/mcmc/`)
- Unadjusted Langevin Algorithm (ULA) and Metropolis-Adjusted Langevin Algorithm (MALA), both in `langevin.py`

**Diffusion model** (`src/amortised_annealing/diffusion/`)
- VP (variance-preserving) noise schedule
- MLP score network with sinusoidal time embedding
- Denoising score matching (DSM) loss and per-time-bin diagnostic
- Euler–Maruyama reverse SDE sampler with EMA-smoothed weights

**Sequential Monte Carlo** (`src/amortised_annealing/smc/`)
- Feynman-Kac corrector (FKC), effective sample size (ESS) monitoring, resampling, and a full SMC sampler — used in annealing and cost-quality experiments

**Experiment scripts** (`scripts/` and `scripts/experiments/`)
- Full pipeline: config generation, MCMC sampling, score model training, evaluation, diagnostics
- Experimental analyses: temperature-transfer sweeps, cost-quality frontiers, model-reuse studies, budget-frontier comparisons, scaling laws, break-even analysis, ULA vs. diffusion comparisons

**Config-driven execution** — all runs are specified by YAML files under `configs/`

---

## Repository structure

```
configs/
  sampling/            MCMC sampling configs (one YAML per run)
  models/              Score model training configs

data/
  samples/             MCMC particle snapshots and summaries
  model_samples/       Diffusion model samples and summaries
  models/              Trained model weights and training summaries
  experiments/         Aggregated experiment results (beta_M sweeps, cost-quality, etc.)
  annealing/           Annealing run outputs

scripts/
  create_sampler_configs.py   Generate sampling config sweeps
  run_sampling.py             Run MCMC from a config
  create_training_configs.py  Generate training config sweeps
  run_training.py             Train a score model from a config
  eval_model.py               Evaluate a trained model against MCMC samples
  diagnose_score.py           Per-time-bin DSM loss diagnostics
  run_annealing.py            Run annealing experiments
  run_experiment.py           General experiment runner
  test_reverse_sde.py         Sanity-check the reverse SDE on a Gaussian
  experiments/                Analysis and plotting scripts for thesis experiments

src/amortised_annealing/
  energies/            Energy function implementations
  mcmc/                ULA and MALA samplers
  diffusion/           Score model, VP schedule, DSM training, reverse SDE
  sampling/            Sample generation utilities
  smc/                 Sequential Monte Carlo / Feynman-Kac corrector
```

---

## Setup

The project requires Python ≥ 3.13. Dependencies are managed with [uv](https://github.com/astral-sh/uv) and pinned in `uv.lock`. Core dependencies are PyTorch 2.8.0, NumPy, SciPy, Matplotlib, and tqdm.

```bash
# Install uv if not already available
pip install uv

# Create environment and install dependencies
uv sync
```

Alternatively, install into an existing environment:

```bash
pip install torch==2.8.0 numpy scipy matplotlib tqdm
pip install -e .
```

---

## Typical workflow

The thesis experiments follow this pipeline:

**1. Generate sampling configs**
```bash
python scripts/create_sampler_configs.py
```
This sweeps over energies, dimensions, &beta;<sub>M</sub> values, and MCMC methods, writing one YAML per combination to `configs/sampling/`.

**2. Run MCMC sampling**
```bash
python scripts/run_sampling.py configs/sampling/rastrigin_d10_beta5_ula_seed0.yaml
```
Saves particle snapshots, a config copy, and a summary to `data/samples/{run_name}/`.

**3. Generate training configs and train a score model**
```bash
python scripts/create_training_configs.py --filter rastrigin --presets mlp256x3 --seeds 0 1
python scripts/run_training.py configs/models/rastrigin_d10_beta5_ula_seed0_mlp256x3_seed0.yaml
```
Trained EMA weights, config, and training summary are saved to `data/models/{run_name}/`.

**4. Evaluate generated samples**
```bash
python scripts/eval_model.py rastrigin_d10_beta5_ula_seed0_mlp256x3_seed0
```
Generates samples via the reverse SDE and prints energy statistics against the MCMC training set.

**5. Run diagnostics or annealing experiments**
```bash
python scripts/diagnose_score.py rastrigin_d10_beta5_ula_seed0_mlp256x3_seed0
python scripts/run_annealing.py <config>
```

Full thesis sweeps cover a grid of energies × dimensions × &beta;<sub>M</sub> values × seeds and require substantial GPU compute to reproduce.

---

## Outputs

| Artifact | Location | Contents |
|----------|----------|----------|
| MCMC run | `data/samples/{run_name}/` | `particles.pt` — `[n_particles, n_snapshots, dim]` tensor; `config.yaml`; `summary.json` (energy stats, acceptance rate, wall time, hardware) |
| Model run | `data/models/{run_name}/` | `ema_model.pt`; `config.yaml`; `summary.json` (loss history, parameter count, wall time) |
| Experiment | `data/experiments/{experiment}/` | Aggregated JSON results per experimental condition |
| Annealing | `data/annealing/{run_name}/` | Per-run annealing outputs and summaries |

---

## Experiments

The thesis organises experiments around four main themes:

**Intermediate-temperature training (&beta;<sub>M</sub> sweep)**
Trains score models at varying &beta;<sub>M</sub> values and measures how sample quality degrades as &beta;<sub>M</sub> moves further from the target &beta;<sub>H</sub>. Covers Rastrigin, Ackley, Double Well, and Many Well across multiple dimensions.

**Direct diffusion sample quality**
Assesses whether a score model trained purely on MCMC samples can match MCMC-quality statistics at the training temperature, and how model capacity and training data volume affect this.

**Temperature transfer and annealing**
Tests how well diffusion samples initialise downstream annealing (MCMC continuation or SMC/FKC correction) when the model was trained at &beta;<sub>M</sub> < &beta;<sub>H</sub>. Includes temperature-transfer sweeps and break-even analysis.

**Cost-quality and budget-frontier comparisons**
Measures sample quality as a function of total compute budget, comparing strategies that vary the split between MCMC sampling cost, diffusion training cost, and annealing continuation cost. Includes reuse-sweep and budget-frontier scripts.

---

## Notes and limitations

- **Research code.** Scripts were written to support thesis experiments and may not be polished or general-purpose.
- **Data artifacts.** Generated samples, trained models, and experiment results are large and are not fully tracked in git. Paths in configs may need adjusting to match local directory layout.
- **Compute.** Full sweeps were run on GPU. CPU execution is supported but will be slow for training and large MCMC runs.
- **Reproducibility.** Seeds are set where possible, but exact numerical reproducibility across hardware or PyTorch versions is not guaranteed.
