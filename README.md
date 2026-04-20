# multi-agent-sumo

Multi-agent traffic signal control for the CS5756 final project.

## Table of Contents

- [Setup](#setup)
- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Problem Formulation](#problem-formulation)
- [Environment](#environment)
- [Method](#method)
- [Evaluation](#evaluation)
- [Expected Outcome](#expected-outcome)
- [References](#references)

## Setup

1. Install [SUMO](https://eclipse.dev/sumo/) (Simulation of Urban MObility)

2. Install project dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

3. In `main.py`, set the `SUMO_HOME` environment variable to your SUMO installation path:

```python
# Set the SUMO_HOME environment variable to the path where SUMO is installed
# change it to your SUMO installation path
os.environ["SUMO_HOME"] = r"D:\Program Files (x86)\Sumo"
```

If you are using conda, activate your environment first, then run the same command.

## Project Overview

This project studies cooperative multi-agent reinforcement learning (MARL) for
traffic signal control using SUMO-based benchmarks. Each traffic intersection is
modeled as an agent, and agents coordinate to improve global traffic efficiency.

We compare two policy-gradient baselines:

- MAPPO (Multi-Agent PPO, centralized critic)
- IPPO (Independent PPO, decentralized/independent critics)

## Motivation

Signalized intersections contribute significantly to traffic delay. Better signal
coordination can reduce waiting times and improve throughput in urban networks.
This project focuses on whether centralized multi-agent training (MAPPO) yields
better coordination than independent training (IPPO).

## Problem Formulation

Each intersection is treated as an MDP agent.

- State space (per intersection):
  - currently enabled phases
  - stopped vehicle queue length
  - approaching vehicles count
  - total waiting time for stopped vehicles
  - sum of speeds of approaching vehicles
  - maximum waiting time over stopped vehicles
  - arrivals during last step
  - departures during last step
- Action space:
  - a set of non-conflicting phase combinations
  - default action duration is 10s, including yellow transition when needed
- Reward:
  - change in cumulative vehicle delay between consecutive time steps

In the multi-agent setting, all intersections act simultaneously, and policy
quality is evaluated primarily by network-level average waiting time.

## Environment

- Simulator: [SUMO](https://eclipse.dev/sumo/) (Simulation of Urban MObility)
- Benchmark: RESCO traffic signal control benchmark
- Scenario: `cologne3` (multi-intersection urban subnetwork of Cologne)

## Method

The training pipeline will integrate the RESCO/SUMO environment with a Python
MARL implementation:

- shared PPO-style training/evaluation utilities
- an IPPO implementation (independent learning per agent)
- a MAPPO implementation (centralized critic over joint information)

## Evaluation

Primary metric:

- average waiting time on the map

Main comparison:

- final performance of MAPPO vs IPPO on `cologne3`

## Expected Outcome

Based on prior MARL behavior in cooperative domains, MAPPO is expected to match
or outperform IPPO, with lower final average waiting time due to improved
coordination through centralized critic training.

## References

1. Ault, James, and Guni Sharon. Reinforcement Learning Benchmarks for Traffic
   Signal Control. NeurIPS 2021 Datasets and Benchmarks.
2. Levinson, D. M. Speed and delay on signalized arterials. Journal of
   Transportation Engineering, 124(3), 1998.
3. Tirachini, A. Estimation of travel time and the benefits of upgrading fare
   payment technology in urban bus services. Transportation Research Part C,
   30, 2013.
