# MAPPO for SUMO Traffic Signal Control

Multi-Agent Proximal Policy Optimization (MAPPO) implementation for cooperative traffic signal control using the SUMO simulation environment.

## Overview

This project implements MAPPO — a state-of-the-art multi-agent reinforcement learning algorithm — for optimizing traffic signal control in urban networks. Each traffic intersection is modeled as an independent agent that learns to coordinate with other intersections to minimize network-wide vehicle waiting time and queue length.

Two variants are provided for comparison:

| Variant | Module | Description |
|---------|--------|-------------|
| **Baseline** | `MAPPO/` | Standard MLP centralized critic |
| **Critic-BN** | `MAPPO_BN/` | Centralized critic with `BatchNorm1d` before each hidden activation (BN experiment) |

Both variants share the same actor architecture, environment wrapper, and training loop. Only the critic network differs.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Training](#training)
- [Evaluation](#evaluation)
- [Plotting Metrics](#plotting-metrics)
- [Experiment Sweeps](#experiment-sweeps)
- [Configuration Reference](#configuration-reference)

---

## Prerequisites

### 1. Python

Python 3.8–3.10 is recommended.

### 2. SUMO

Install the SUMO traffic simulator:

```bash
# macOS (Homebrew)
brew install sumo

# Ubuntu / Debian
sudo apt-get install sumo sumo-tools sumo-doc

# Windows
# Download installer from https://eclipse.dev/sumo/
```

> **Note:** `SUMO_HOME` and `PROJ_DATA` are auto-detected from the installed `sumo` Python package at runtime. You do **not** need to set these environment variables manually.

---

## Installation

### Option A — Automated setup (recommended)

A setup script creates a dedicated virtual environment and installs all dependencies:

```bash
# macOS / Linux
bash setup_MAPPO_venv.sh

# Windows
setup_MAPPO_venv.bat
```

Activate the environment before running any scripts:

```bash
# macOS / Linux
source MAPPO_venv/bin/activate

# Windows
MAPPO_venv\Scripts\activate
```

### Option B — Manual install into an existing environment

```bash
pip install -r requirements.txt
```

### Verify the installation

```bash
python -c "import sumo_rl, torch, tianshou; print('All dependencies OK')"
```

---

## Project Structure

```
multi-agent-sumo/
├── MAPPO/                      # Baseline variant (standard MLP critic)
│   ├── agents/
│   │   ├── mappo_policy.py     # MAPPO policy (actor + critic update)
│   │   └── multi_agent_manager.py
│   ├── config/
│   │   ├── config.py           # Config dataclasses
│   │   └── default_configs.py  # Preset configs (default, debug, fast-test)
│   ├── envs/
│   │   ├── sumo_env_wrapper.py # SUMO ↔ Tianshou adapter
│   │   └── env_utils.py
│   ├── networks/
│   │   ├── actor.py            # Decentralized actor (per-agent)
│   │   ├── critic.py           # Centralized critic
│   │   └── utils.py
│   ├── training/
│   │   ├── trainer.py          # MAPPOTrainer
│   │   └── evaluator.py
│   └── utils/
│       ├── checkpoint.py       # Save / load checkpoints
│       ├── logger.py           # CSV metrics logger
│       ├── metrics.py          # Traffic metrics helpers
│       └── reward_normalizer.py
├── MAPPO_BN/                   # Critic-BN variant (BatchNorm1d on critic)
│   └── ...                     # Same structure as MAPPO/
├── scripts/
│   ├── train_mappo.py          # Main training entry-point
│   ├── evaluate.py             # Evaluate a saved checkpoint
│   ├── plot_metrics.py         # Plot training curves from CSV logs
│   └── run_experiment_sweep.py # Parallel multi-seed sweep launcher
├── configs/
│   ├── base_config.yaml        # Default full-run config (baseline)
│   ├── critic_bn_config.yaml   # Critic-BN variant config
│   └── debug_config.yaml       # Quick debug config (5 epochs, CPU)
├── RESCO/
│   └── cologne3/               # Cologne 3-intersection SUMO scenario
│       ├── cologne3.net.xml
│       └── cologne3.rou.xml
├── plots/                      # Output plots directory
├── requirements.txt
├── setup_MAPPO_venv.sh
├── setup_MAPPO_venv.bat
└── main.py                     # Minimal standalone SUMO-RL example
```

---

## Training

All training is launched through [`scripts/train_mappo.py`](scripts/train_mappo.py).

### Quick start

```bash
# Baseline — default config (100 epochs, cologne3)
python scripts/train_mappo.py

# Critic-BN variant
python scripts/train_mappo.py --use-critic-bn

# Load a YAML config file
python scripts/train_mappo.py --config configs/base_config.yaml

# Debug mode (5 epochs, small networks, CPU)
python scripts/train_mappo.py --debug

# Fast smoke-test (short episodes, no GUI)
python scripts/train_mappo.py --fast-test

# Enable SUMO GUI (requires display / XQuartz on macOS)
python scripts/train_mappo.py --use-gui
```

### Full argument reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config FILE` | str | — | Path to a YAML config file |
| `--debug` | flag | off | Use built-in debug preset (5 epochs, CPU, small nets) |
| `--fast-test` | flag | off | Use built-in fast-test preset (short episodes, headless) |
| `--use-critic-bn` | flag | off | Enable BatchNorm1d on the centralized critic (BN experiment) |
| `--use-gui` | flag | off | Launch SUMO with GUI |
| `--num-seconds N` | int | from config | Override simulation duration (seconds) |
| `--net-file FILE` | str | from config | Override SUMO network file |
| `--route-file FILE` | str | from config | Override SUMO route file |
| `--max-epoch N` | int | from config | Override number of training epochs |
| `--seed N` | int | from config | Random seed |
| `--device {cpu,cuda,mps}` | str | from config | Compute device |
| `--experiment-name NAME` | str | auto | Name shown in log directory |
| `--log-dir DIR` | str | `logs` | Root directory for run logs |

### Training output

Each run creates a timestamped directory under `logs/`:

```
logs/
└── mappo_seed42_20260503_142301/
    ├── config.json        # Saved config for reproducibility
    ├── metrics.csv        # Per-epoch training and evaluation metrics
    └── checkpoints/
        ├── checkpoint_epoch_10.pt
        ├── checkpoint_epoch_20.pt
        └── best_checkpoint.pt
```

`metrics.csv` columns include: `epoch`, `wallclock_time_s`, `episode_reward`, `episode_length`, `mean_waiting_time`, `std_waiting_time`, `mean_queue_length`, `std_queue_length`, `loss`, `actor_loss`, `critic_loss`, `entropy`, `clip_frac`.

### Typical training time

| Config | Approximate time |
|--------|-----------------|
| `--debug` (5 epochs) | ~5–10 minutes |
| `--fast-test` | ~30 minutes |
| Full run (100 epochs) | ~2–4 hours (GPU) / ~6–12 hours (CPU) |

---

## Evaluation

Evaluate a saved checkpoint with [`scripts/evaluate.py`](scripts/evaluate.py):

```bash
# Basic evaluation (10 episodes)
python scripts/evaluate.py \
    --checkpoint logs/mappo_seed42_YYYYMMDD_HHMMSS/checkpoints/best_checkpoint.pt

# With SUMO GUI
python scripts/evaluate.py \
    --checkpoint logs/.../best_checkpoint.pt \
    --use-gui

# More episodes
python scripts/evaluate.py \
    --checkpoint logs/.../best_checkpoint.pt \
    --n-episode 20

# Also run a random-action baseline for comparison
python scripts/evaluate.py \
    --checkpoint logs/.../best_checkpoint.pt \
    --compare-baseline
```

### Evaluation arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint FILE` | required | Path to `.pt` checkpoint file |
| `--n-episode N` | `10` | Number of evaluation episodes |
| `--use-gui` | off | Enable SUMO GUI |
| `--device {cpu,cuda,mps}` | `cpu` | Compute device |
| `--compare-baseline` | off | Also evaluate a random-action baseline |

---

## Plotting Metrics

[`scripts/plot_metrics.py`](scripts/plot_metrics.py) reads the `metrics.csv` files produced during training and generates a `metrics_overview.png` figure with four panels:

- Mean waiting time vs epoch
- Mean waiting time vs wall-clock time
- Mean queue length vs epoch
- Episode reward vs epoch (smoothed)

```bash
# Single run
python scripts/plot_metrics.py \
    --csv logs/mappo_seed42_YYYYMMDD/metrics.csv \
    --out-dir plots/

# Multiple seeds — plots mean ± std across seeds
python scripts/plot_metrics.py \
    --csv logs/mappo_seed0_*/metrics.csv \
         logs/mappo_seed1_*/metrics.csv \
         logs/mappo_seed2_*/metrics.csv \
    --labels "seed=0" "seed=1" "seed=2" \
    --out-dir plots/multi_seed/
```

Add `--show` to display the figure interactively instead of saving.

---

## Experiment Sweeps

[`scripts/run_experiment_sweep.py`](scripts/run_experiment_sweep.py) automates running multiple (variant × seed) combinations in parallel. Each subprocess writes its stdout/stderr to `logs/<variant>_seed<N>/run.log`. A summary table is printed when all runs complete.

```bash
# Default: both variants × seeds {0, 1, 2}, up to 4 parallel processes
python scripts/run_experiment_sweep.py

# Custom seeds and parallelism
python scripts/run_experiment_sweep.py --seeds 0 1 2 3 4 --max-parallel 3

# Only the critic-BN variant
python scripts/run_experiment_sweep.py --variants critic-bn --seeds 0 1 2

# Quick smoke-test (1 epoch per run)
python scripts/run_experiment_sweep.py --max-epoch 1 --seeds 0
```

### Sweep arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--variants {baseline,critic-bn}` | both | Which variants to run |
| `--seeds N [N ...]` | `0 1 2` | Random seeds |
| `--max-parallel N` | `4` | Maximum concurrent subprocesses |
| `--max-epoch N` | from config | Override epochs for every run |

---

## Configuration Reference

YAML config files live in [`configs/`](configs/). Pass one with `--config`. Any field can be overridden by a CLI argument.

### `configs/base_config.yaml` — default full run

```yaml
sumo:
  net_file: "RESCO/cologne3/cologne3.net.xml"
  route_file: "RESCO/cologne3/cologne3.rou.xml"
  num_seconds: 26000
  use_gui: false
  delta_time: 5
  yellow_time: 2
  min_green: 5
  max_green: 50

network:
  actor_hidden: [128, 128]
  critic_hidden: [256, 256]
  activation: "relu"
  use_orthogonal_init: true

mappo:
  lr_actor: 0.0003
  lr_critic: 0.001
  gamma: 0.99
  gae_lambda: 0.95
  eps_clip: 0.2
  value_clip: true
  advantage_normalization: true
  vf_coef: 0.5
  ent_coef: 0.01
  max_grad_norm: 0.5
  reward_normalization: false

training:
  max_epoch: 100
  step_per_epoch: 10000
  episode_per_collect: 10
  batch_size: 256
  repeat_per_collect: 4
  n_train_envs: 4
  n_test_envs: 2
  test_interval: 5
  save_interval: 10

seed: 42
device: "cuda"   # or "cpu" / "mps"
```

### `configs/critic_bn_config.yaml` — Critic-BN variant

Identical to the baseline except:

```yaml
network:
  use_critic_bn: true   # inserts BatchNorm1d before each hidden activation

mappo:
  lr_actor: 1.0e-4
  max_grad_norm: 0.3
  reward_normalization: true
```

### `configs/debug_config.yaml` — rapid debugging

```yaml
network:
  actor_hidden: [64, 64]
  critic_hidden: [128, 128]

training:
  max_epoch: 5
  step_per_epoch: 1000
  n_train_envs: 1
  n_test_envs: 1

device: "cpu"
```

---

## SUMO Scenario

The bundled scenario is **Cologne 3** from the [RESCO benchmark](https://github.com/Pi-Star-Lab/RESCO), a real-world sub-network of Cologne, Germany with 3 signalized intersections.

| File | Description |
|------|-------------|
| [`RESCO/cologne3/cologne3.net.xml`](RESCO/cologne3/cologne3.net.xml) | Road network definition |
| [`RESCO/cologne3/cologne3.rou.xml`](RESCO/cologne3/cologne3.rou.xml) | Vehicle demand / routes |

Vehicle departures are concentrated around simulation second ~23 500, so `num_seconds` should be set to at least `26 000` for a full traffic scenario (the default).
