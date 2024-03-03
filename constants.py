from enum import Enum

from gym_routing.envs import ActionMaskedTSPEnv


class RoutingProblem(Enum):
    TSP = "tsp"


PROBLEM_TO_ENVIRONMENT = {
    # Use the Action Masked environment by default: the mask can be ignored.
    RoutingProblem.TSP: ActionMaskedTSPEnv
}

DATA_FOLDER = "data/"
