from gymnasium.envs.registration import register

register(
    id="gym_routing/TSP-v0",
    entry_point="gym_routing.envs:TSPEnv",
)

register(
    id="gym_routing/ActionMaskedTSP-v0",
    entry_point="gym_routing.envs:ActionMaskedTSPEnv",
)
