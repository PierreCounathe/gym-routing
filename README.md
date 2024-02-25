# Gym Routing: Reinforcement Learning Environments for Routing Problems

Gym-Routing offers a suite of Gymnasium environments tailored for training Reinforcement Learning (RL) agents to tackle various routing problems.

## Table of contents
- Installation
- Key features and improvements
- Tutorials and RL Agents Implementations
- Repo setup


# Installation
1. Clone the repo
```shell
git clone https://github.com/PierreCounathe/gym-routing
```

2. Install `gym_routing`
```shell
pip install -e gym-routing
```

3. Create an environment
```python
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

import gym_routing

env = gym.make("gym_routing/TSP-v0")
env = FlattenObservation(env)
```

# Key features and improvements

## Environments
- [x] (TSP) Vanilla Traveling Salesman Problem
    - [x] Action Masked TSP
- [ ] (VRP) Vehicle Routing Problem
- [ ] (CVRP) Capacitated Vehicle Routing Problem

# Tutorials and RL Agents Implementations

This section is WIP.

# Repo setup

To ensure clean and consistent code, this repo uses pre-commit hooks (black, flake8, mypy, isort), that can be installed the following way:

```shell
pip install pre-commit
pre-commit install
```
