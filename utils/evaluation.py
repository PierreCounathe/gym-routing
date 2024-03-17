from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation
from sb3_contrib.common.maskable.utils import get_action_masks

from utils.data_generation import load_problem_instance


class NoActionMaskingSupport(Exception):
    pass


def collect_action_masked(env: gym.Env, model: Any, observation: Any):
    """
    Collect an action from the model, given an observation, using action masks.
    This should only be used with models that supporf action masking.
    """
    action_masks = get_action_masks(env)
    try:
        action, _states = model.predict(
            observation, deterministic=True, action_masks=action_masks
        )
    except TypeError as e:
        raise NoActionMaskingSupport(f"The model does not support action masking: {e}")
    return action


def collect_action_no_mask(env: gym.Env, model: Any, observation: Any):
    """
    Collect an action from the model, given an observation, using no action masks.
    """
    action, _states = model.predict(observation, deterministic=True)
    return action


def collect_action(enable_masking: bool) -> Callable[[gym.Env, Any, Any], Any]:
    """
    Returns the right action collection function, depending on whether action masking
    is enabled or not.
    """
    if enable_masking:
        return collect_action_masked
    else:
        return collect_action_no_mask


def evaluate_model_on_instance(
    model: Any,
    problem: str,
    problem_size: int,
    instance_index_and_seed: int,
    enable_masking: bool,
    flatten_obs: bool,
    render_mode: str = None,
) -> float:
    """
    Load a problem instance that was generated using the data generation script,
    and evaluate the model's reward on it.
    enable_masking and flatten_obs values should correspond the model used.
    Return the episode reward.
    """
    env = load_problem_instance(problem, problem_size, instance_index_and_seed)
    env.render_mode = render_mode
    if flatten_obs:
        env = FlattenObservation(env)
    observation, info = env.reset(seed=instance_index_and_seed)
    terminated = False
    truncated = False
    episode_reward = 0
    while not terminated and not truncated:
        action = collect_action(enable_masking)(env, model, observation)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
    env.close()
    return episode_reward


def evaluate_model_on_all_instances(
    model: Any,
    problem: str,
    problem_size: int,
    n_samples: int,
    enable_masking: bool,
    flatten_obs: bool,
) -> tuple[float, float]:
    """
    Run model evaluation on n_samples problem instances previously generated.
    Return the episode rewards mean and standard deviation.
    """
    episode_rewards = []
    for i in range(n_samples):
        episode_rewards.append(
            evaluate_model_on_instance(
                model,
                problem,
                problem_size,
                i,
                enable_masking,
                flatten_obs,
                render_mode=None,
            )
        )
    return np.mean(episode_rewards), np.std(episode_rewards)
