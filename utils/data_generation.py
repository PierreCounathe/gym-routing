import os
import pickle

import gymnasium as gym

from constants import DATA_FOLDER, PROBLEM_TO_ENVIRONMENT, RoutingProblem


def generate_problem_instances(problem: str, n_samples: int, size: int) -> None:
    """
    Generate routing problem instances.
    """
    RoutingEnvironment = PROBLEM_TO_ENVIRONMENT[RoutingProblem(problem)]

    instances_folder = os.path.join(DATA_FOLDER, f"{problem}_{size}")
    if not os.path.exists(instances_folder):
        os.makedirs(instances_folder)
    # Generate reset problem instances, to get the data
    # and to be able to have an Agent solve them
    for sample_index in range(n_samples):
        instance_path = os.path.join(instances_folder, f"instance_{sample_index}.pkl")
        instance = RoutingEnvironment(size=size)
        instance.reset(seed=sample_index)
        with open(instance_path, "wb") as f:
            pickle.dump(instance, f)


def load_problem_instance(problem: str, size: int, instance_index: int) -> gym.Env:
    """
    Load a problem instance.
    """
    instance_path = os.path.join(
        DATA_FOLDER, f"{problem}_{size}", f"instance_{instance_index}.pkl"
    )
    with open(instance_path, "rb") as f:
        instance = pickle.load(f)
    return instance
