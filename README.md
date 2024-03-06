# Gym Routing: Reinforcement Learning Environments for Routing Problems

Gym-Routing aims at offering a suite of Gymnasium environments tailored for training Reinforcement Learning (RL) agents to tackle various routing problems.

## Table of contents
- [Installation](https://github.com/PierreCounathe/gym-routing?tab=readme-ov-file#installation)
- [Key features and improvements](https://github.com/PierreCounathe/gym-routing?tab=readme-ov-file#key-features-and-improvements)
- [Tutorials and RL Agents Implementations](https://github.com/PierreCounathe/gym-routing?tab=readme-ov-file#tutorials-and-rl-agents-implementations)
- [Repo setup](https://github.com/PierreCounathe/gym-routing?tab=readme-ov-file#repo-setup)


# Installation
1. Pip install the package `gym_routing` defined in the repo
```shell
pip install git+https://github.com/PierreCounathe/gym-routing
```

2. Import `gym_routing` and make the gym environment
```python
import gym_routing

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

env = gym.make("gym_routing/TSP-v0")  # use "gym_routing/ActionMaskedTSP-v0" to use the action masked environment
env = FlattenObservation(env)
```

# Key features and improvements

## Environments
- [x] (TSP) Vanilla Traveling Salesman Problem
    - [x] Action Masked TSP
- [ ] (VRP) Vehicle Routing Problem
- [ ] (CVRP) Capacitated Vehicle Routing Problem

# Tutorials and RL Agents Implementations

This section is WIP: I am working on simple implementations of algorithms using stable-baselines3, that achieve high performance.

# Repo setup

To ensure clean and consistent code, this repo uses pre-commit hooks (black, flake8, mypy, isort), that can be installed the following way:

```shell
pip install pre-commit
pre-commit install
```
