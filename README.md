# MAPPO and IPPO for SUMO Traffic Signal Control

Multi-Agent Proximal Policy Optimization (MAPPO) and Independent Proximal Policy Optimization (IPPO) implementations for cooperative traffic signal control using the SUMO simulation environment.

## 🚦 Overview

This project implements MAPPO and an IPPO baseline for optimizing traffic signal control in urban networks. Each traffic intersection is modeled as an agent. MAPPO uses centralized training with decentralized execution, while IPPO trains each agent with its own local actor-critic learner for comparison.

### Key Features

- ✅ **MAPPO Algorithm**: Centralized training with decentralized execution
- ✅ **IPPO Baseline**: Independent PPO learners with local critics
- ✅ **SUMO Integration**: Compatible with SUMO-RL traffic environments
- ✅ **Tianshou Framework**: Built on the efficient Tianshou RL library
- ✅ **WandB Logging**: Comprehensive experiment tracking
- ✅ **Flexible Configuration**: YAML-based configuration system
- ✅ **GPU Support**: CUDA-accelerated training
- ✅ **GUI & Headless**: Support for both visualization and fast training modes

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## 🔧 Installation

### Prerequisites

1. **SUMO**: Install SUMO simulation environment
   ```bash
   # Ubuntu/Debian
   sudo apt-get install sumo sumo-tools sumo-doc
   
   # macOS (using Homebrew)
   brew install sumo
   
   # Windows: Download from https://eclipse.dev/sumo/
   ```

2. **Set SUMO_HOME environment variable** *(optional — auto-detected if not set)*:

   The training and evaluation scripts automatically detect `SUMO_HOME` and `PROJ_DATA`
   from the installed `sumo` Python package, so **no manual configuration is required**
   as long as you install dependencies via `pip install -r requirements.txt`.

   If you want to set them permanently anyway (e.g. for tools outside this project):
   ```bash
   # Auto-detect the correct path from the installed package
   export SUMO_HOME=$(python -c "import sumo, os; print(os.path.dirname(sumo.__file__))")
   export PROJ_DATA=$SUMO_HOME/data/proj

   # Add the above lines to ~/.bashrc or ~/.zshrc to make them permanent
   ```

### Install Dependencies

```bash
# Clone the repository (if applicable)
# cd multi-agent-sumo

# Install Python dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Test SUMO installation
sumo --version

# Test Python environment
python -c "import sumo_rl, torch, tianshou; print('All dependencies OK!')"
```

## 🚀 Quick Start

### 1. Run with Default Configuration

```bash
python scripts/train_mappo.py
```

This will train MAPPO on the cologne3 scenario with default hyperparameters.

To train the IPPO baseline with matching defaults:

```bash
python scripts/train_ippo.py
```

### 2. Debug Mode (with GUI)

```bash
python scripts/train_mappo.py --debug
```

Uses faster training settings for MAPPO. For IPPO:

```bash
python scripts/train_ippo.py --debug
```

### 3. Custom Configuration

```bash
python scripts/train_mappo.py --config configs/base_config.yaml
```

The IPPO loader accepts YAML files with an `ippo:` section. It can also read existing `mappo:` hyperparameters as a compatibility fallback:

```bash
python scripts/train_ippo.py --config configs/base_config.yaml
```

### 4. Evaluate Trained Model

```bash
python scripts/evaluate.py --checkpoint logs/experiment_name/checkpoints/checkpoint_epoch_100.pt --use-gui
```

Evaluate an IPPO checkpoint with:

```bash
python scripts/evaluate_ippo.py --checkpoint logs/experiment_name/checkpoints/checkpoint_epoch_100.pt --use-gui
```

## 📁 Project Structure

```
multi-agent-sumo/
├── MAPPO/              # MAPPO package
│   ├── agents/                 # MAPPO policy and manager
│   │   ├── mappo_policy.py     # MAPPO policy implementation
│   │   └── multi_agent_manager.py  # Multi-agent coordination
│   ├── config/                 # Configuration system
│   │   ├── config.py           # Config dataclasses
│   │   └── default_configs.py  # Preset configurations
│   ├── envs/                   # Environment wrappers
│   │   ├── sumo_env_wrapper.py # SUMO-Tianshou adapter
│   │   └── env_utils.py        # Environment utilities
│   ├── networks/               # Neural networks
│   │   ├── actor.py            # Actor network (discrete actions)
│   │   ├── critic.py           # Centralized critic
│   │   └── utils.py            # Network utilities
│   ├── training/               # Training infrastructure
│   │   ├── trainer.py          # Main trainer class
│   │   └── evaluator.py        # Policy evaluation
│   └── utils/                  # Utilities
│       ├── checkpoint.py       # Save/load checkpoints
│       ├── logger.py           # WandB logger
│       └── metrics.py          # Traffic metrics
├── IPPO/                       # IPPO baseline package
│   ├── agents/                 # IPPO policy and manager
│   │   ├── ippo_policy.py      # Independent PPO policy implementation
│   │   └── multi_agent_manager.py  # Multi-agent dispatch
│   ├── config/                 # IPPO configuration system
│   ├── envs/                   # Environment wrappers
│   ├── networks/               # Actor and local critic networks
│   ├── training/               # IPPO trainer and evaluator
│   └── utils/                  # Checkpointing, logging, and metrics
├── scripts/                    # Executable scripts
│   ├── train_mappo.py          # MAPPO training script
│   ├── train_ippo.py           # IPPO training script
│   ├── evaluate.py             # MAPPO evaluation script
│   └── evaluate_ippo.py        # IPPO evaluation script
├── configs/                    # Configuration files
│   ├── base_config.yaml        # Default configuration
│   └── debug_config.yaml       # Debug configuration
├── RESCO/                      # SUMO scenarios
│   └── cologne3/               # Cologne network
│       ├── cologne3.net.xml    # Network definition
│       └── cologne3.rou.xml    # Traffic routes
├── IMPLEMENTATION_PLAN.md      # Detailed implementation plan
├── PLAN_SUMMARY.md             # Quick reference guide
├── requirements.txt            # Python dependencies
├── main.py                     # Original simple example
└── README.md                   # This file
```

## 💻 Usage

### Training Options

```bash
# Basic MAPPO training
python scripts/train_mappo.py

# Basic IPPO training
python scripts/train_ippo.py

# With GUI visualization
python scripts/train_mappo.py --use-gui
python scripts/train_ippo.py --use-gui

# Custom configuration
python scripts/train_mappo.py --config configs/base_config.yaml
python scripts/train_ippo.py --config configs/base_config.yaml

# Override specific parameters
python scripts/train_mappo.py --max-epoch 200 --seed 123
python scripts/train_ippo.py --max-epoch 200 --seed 123

# Fast test run
python scripts/train_mappo.py --fast-test
python scripts/train_ippo.py --fast-test

# Disable WandB logging
python scripts/train_mappo.py --no-wandb
python scripts/train_ippo.py --no-wandb
```

### Full Training Arguments

```bash
python scripts/train_mappo.py --help
# or
python scripts/train_ippo.py --help

Arguments:
  --config CONFIG           Path to YAML config file
  --debug                   Use debug config (GUI, short training)
  --fast-test               Fast test config (no GUI, short episodes)
  --use-gui                 Enable SUMO GUI
  --num-seconds SECONDS     Simulation duration
  --net-file FILE           SUMO network file
  --route-file FILE         SUMO route file
  --max-epoch EPOCHS        Maximum training epochs
  --seed SEED               Random seed
  --device {cpu,cuda}       Device to use
  --no-wandb                Disable W&B logging
  --experiment-name NAME    Experiment name
  --log-dir DIR             Logging directory
```

### Evaluation Options

```bash
# Evaluate trained MAPPO model
python scripts/evaluate.py --checkpoint path/to/checkpoint.pt

# Evaluate trained IPPO model
python scripts/evaluate_ippo.py --checkpoint path/to/checkpoint.pt

# With GUI visualization
python scripts/evaluate.py --checkpoint path/to/checkpoint.pt --use-gui
python scripts/evaluate_ippo.py --checkpoint path/to/checkpoint.pt --use-gui

# Multiple episodes
python scripts/evaluate.py --checkpoint path/to/checkpoint.pt --n-episode 20
python scripts/evaluate_ippo.py --checkpoint path/to/checkpoint.pt --n-episode 20

# Compare with baseline
python scripts/evaluate.py --checkpoint path/to/checkpoint.pt --compare-baseline
python scripts/evaluate_ippo.py --checkpoint path/to/checkpoint.pt --compare-baseline
```

## ⚙️ Configuration

Configuration is managed through YAML files and dataclasses. See [`configs/base_config.yaml`](configs/base_config.yaml) for the complete configuration template.

### Key Configuration Sections

#### 1. SUMO Environment
```yaml
sumo:
  net_file: "RESCO/cologne3/cologne3.net.xml"
  route_file: "RESCO/cologne3/cologne3.rou.xml"
  num_seconds: 3600
  use_gui: false
  delta_time: 5
```

#### 2. Neural Network Architecture
```yaml
network:
  actor_hidden: [128, 128]
  critic_hidden: [256, 256]
  activation: "relu"
  use_orthogonal_init: true
```

#### 3. MAPPO/IPPO Hyperparameters
```yaml
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
```

For IPPO-specific config files, use the same fields under `ippo:`:

```yaml
ippo:
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
```

`scripts/train_ippo.py` can also load an existing `mappo:` section as a compatibility fallback, which makes it easy to compare MAPPO and IPPO with the same hyperparameters.

#### 4. Training Settings
```yaml
training:
  max_epoch: 100
  step_per_epoch: 10000
  episode_per_collect: 10
  batch_size: 256
  repeat_per_collect: 4
  n_train_envs: 4
  n_test_envs: 2
```

## 🎯 Training

### Training Process

1. **Initialization**: 
   - Creates SUMO environments
   - Initializes actor and critic networks
   - MAPPO uses a shared centralized critic
   - IPPO uses one local critic per agent
   - Sets up optimizers and replay buffers

2. **Data Collection**:
   - Agents interact with environment using current policy
   - Experiences stored in replay buffer

3. **Policy Update**:
   - Sample mini-batches from buffer
   - Compute advantages using GAE
   - Update actors with PPO objective
   - Update the centralized MAPPO critic or each IPPO agent's local critic

4. **Evaluation**:
   - Periodically evaluate on test environment
   - Log metrics to WandB
   - Save checkpoints

### Monitoring Training

Training metrics are logged to:
- **Console**: Real-time progress
- **WandB Dashboard**: Comprehensive visualization (if enabled)
- **TensorBoard**: Local logging (in log directory)

Key metrics to monitor:
- `train/episode_reward`: Episode return
- `train/loss/actor`: Policy loss
- `train/loss/critic`: Value function loss
- `eval/mean_waiting_time`: Average vehicle waiting time (lower is better)
- `eval/mean_queue_length`: Average queue length

### Typical Training Time

- **Debug config**: ~5-10 minutes
- **Fast test**: ~30 minutes
- **Full training (100 epochs)**: ~2-4 hours (GPU) / ~6-12 hours (CPU)

## 📊 Evaluation

### Evaluate Trained Policy

```bash
python scripts/evaluate.py \
    --checkpoint logs/experiment/checkpoints/checkpoint_epoch_100.pt \
    --n-episode 10 \
    --use-gui
```

For IPPO checkpoints:

```bash
python scripts/evaluate_ippo.py \
    --checkpoint logs/experiment/checkpoints/checkpoint_epoch_100.pt \
    --n-episode 10 \
    --use-gui
```

### Evaluation Metrics

- **Mean Reward**: Episode return
- **Mean Waiting Time**: Average vehicle waiting time (seconds)
- **Mean Queue Length**: Average number of stopped vehicles
- **Episode Length**: Number of simulation steps

### Baseline Comparison

Compare your trained policy against random baseline:

```bash
python scripts/evaluate.py \
    --checkpoint path/to/checkpoint.pt \
    --compare-baseline
```

Use `scripts/evaluate_ippo.py` for IPPO checkpoints.

## 📈 Results

### Expected Performance

After 100 epochs of training on cologne3:

| Metric | Random Policy | IPPO Baseline | MAPPO (Expected) |
|--------|--------------|---------------|------------------|
| Avg Waiting Time | ~200-300s | Scenario-dependent | ~100-150s |
| Avg Queue Length | ~15-20 vehicles | Scenario-dependent | ~8-12 vehicles |
| Episode Reward | Variable | Improving baseline | Improving |

IPPO is intended as the decentralized PPO baseline. Because each agent only learns from local observations and a local value function, it is useful for measuring the benefit of MAPPO's centralized critic.

### Visualizing Results

1. **WandB Dashboard**: View real-time training curves
2. **SUMO GUI**: Watch learned traffic signal behavior
3. **TensorBoard**: Local metric visualization

```bash
tensorboard --logdir logs/
```

## 🐛 Troubleshooting

### Common Issues

#### 1. SUMO Not Found
```
Error: SUMO_HOME environment variable not set
```
**Solution**: Set SUMO_HOME in your shell configuration
```bash
export SUMO_HOME="/usr/share/sumo"  # Adjust path
```

#### 2. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size or use CPU
```bash
python scripts/train_mappo.py --device cpu
# or
python scripts/train_ippo.py --device cpu
# or
python scripts/train_mappo.py --config configs/base_config.yaml  # Edit batch_size
```

#### 3. Import Errors
```
ModuleNotFoundError: No module named 'MAPPO'
```
or
```
ModuleNotFoundError: No module named 'IPPO'
```
**Solution**: Install dependencies and add project to PYTHONPATH
```bash
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 4. Training Too Slow
**Solutions**:
- Use `--fast-test` for quicker validation
- Reduce `num_seconds` in config
- Use GPU if available
- Reduce `n_train_envs`

### Debug Mode

For troubleshooting, use debug mode:
```bash
python scripts/train_mappo.py --debug
# or
python scripts/train_ippo.py --debug
```

This enables:
- GUI visualization
- Shorter episodes
- Smaller networks
- More frequent logging
- No WandB (less overhead)

## 📚 Architecture Details

### MAPPO Overview

**Centralized Training, Decentralized Execution (CTDE)**:
- **Training**: Centralized critic sees all agent observations
- **Execution**: Each agent acts based on local observations only

### IPPO Overview

**Independent PPO Baseline**:
- **Training**: Each agent has its own actor, local critic, and optimizer
- **Execution**: Each agent acts from its local observation only
- **Purpose**: Provides a decentralized baseline to compare against MAPPO's centralized critic

### Key Components

1. **Actor Network** ([`actor.py`](MAPPO/networks/actor.py))
   - Input: Local traffic state (queue, speed, etc.)
   - Output: Categorical distribution over traffic phases
   - One per agent (decentralized)

2. **Centralized Critic** ([`critic.py`](MAPPO/networks/critic.py))
   - Input: Concatenated observations from all agents
   - Output: Value estimate for global state
   - Shared across all agents

   **IPPO Local Critic** ([`critic.py`](IPPO/networks/critic.py))
   - Input: One agent's local observation
   - Output: Value estimate for that agent's local state
   - One critic per agent

3. **MAPPO Policy** ([`mappo_policy.py`](MAPPO/agents/mappo_policy.py))
   - PPO update with clipped objective
   - GAE for advantage estimation
   - Entropy bonus for exploration

   **IPPO Policy** ([`ippo_policy.py`](IPPO/agents/ippo_policy.py))
   - Same PPO clipped objective
   - GAE with local critic values
   - Independent actor-critic updates per agent

4. **Multi-Agent Manager** ([`multi_agent_manager.py`](MAPPO/agents/multi_agent_manager.py))
   - Coordinates multiple agents
   - Manages state aggregation for critic
   - Dispatches observations to policies

   **IPPO Multi-Agent Manager** ([`multi_agent_manager.py`](IPPO/agents/multi_agent_manager.py))
   - Dispatches observations to independent policies
   - Does not build a global critic input

### Reward Structure

Default reward from SUMO-RL:
```python
reward = previous_delay - current_delay
```

This encourages reducing cumulative vehicle delay at intersections.

## 🔬 Extending the Implementation

### Adding New Scenarios

1. Add SUMO network and route files to `RESCO/`
2. Create new config file in `configs/`
3. Train:
```bash
python scripts/train_mappo.py --net-file RESCO/new_scenario/network.net.xml --route-file RESCO/new_scenario/routes.rou.xml
# or
python scripts/train_ippo.py --net-file RESCO/new_scenario/network.net.xml --route-file RESCO/new_scenario/routes.rou.xml
```

### Modifying Reward Function

Edit [`MAPPO/envs/sumo_env_wrapper.py`](MAPPO/envs/sumo_env_wrapper.py) or [`IPPO/envs/sumo_env_wrapper.py`](IPPO/envs/sumo_env_wrapper.py) to implement custom rewards.

### Hyperparameter Tuning

Create WandB sweep configuration and run:
```bash
wandb sweep sweep_config.yaml
wandb agent <sweep_id>
```

## 📖 Citation

If you use this implementation in your research, please cite:

```bibtex
@software{mappo_sumo_traffic,
  title = {MAPPO and IPPO for SUMO Traffic Signal Control},
  year = {2026},
  note = {Implementation of MAPPO and IPPO for traffic management}
}
```

### References

1. **MAPPO**: Yu et al. (2021) - "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
2. **IPPO/PPO**: Schulman et al. (2017) - "Proximal Policy Optimization Algorithms"
3. **SUMO-RL**: Alegre (2019) - "SUMO-RL"
4. **Traffic Control**: Ault & Sharon (2021) - "Reinforcement Learning Benchmarks for Traffic Signal Control"

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is provided for educational and research purposes.

## 🙏 Acknowledgments

- **Tianshou**: High-quality RL library
- **SUMO**: Open-source traffic simulation
- **SUMO-RL**: SUMO-Python integration
- **RESCO**: Traffic control benchmarks

---

**Questions or issues?** Please open an issue on GitHub or refer to the [implementation plan](IMPLEMENTATION_PLAN.md) for technical details.
