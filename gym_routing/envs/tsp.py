import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class TSPEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }
    STARTING_NODE = 0

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The number of stops to visit
        self.window_size = 512  # The size of the PyGame window
        self.max_duration = 100 * size  # The maximum number of steps in an episode

        # The observations are constituted by the positions of the nodes,
        # the index of the current node, and a binary vector indicating
        # which nodes have been visited.
        self.observation_space = spaces.Dict(
            {
                "nodes": spaces.Box(
                    low=0, high=1, shape=(self.size, 2), dtype=np.float64
                ),
                "current_node": spaces.Discrete(self.size),
                "visited_nodes": spaces.MultiBinary(self.size),
            }
        )

        # The action value corresponds to the index of the next node to visit.
        self.action_space = spaces.Discrete(self.size)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        """
        Generates a new, random instance of the problem.
        Returns observation, info.
        """
        # The following line is recommended by the Gym documentation
        # it allows to seed self.np_random
        super().reset(seed=seed)

        # The nodes' positions are sampled uniformly at random
        self._nodes = self.np_random.uniform(size=(self.size, 2))
        self._current_node = self.STARTING_NODE
        self._visited_nodes = np.zeros(self.size, dtype=np.int8)
        self._visited_nodes[self._current_node] = 1

        # Duration and distance of the episode
        self.duration = 0
        self.episode_distance = 0
        self.visit_order = [self._current_node]

        # Compute the distance matrix between the nodes
        self._distance_matrix = self._compute_distance()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """
        Take a step in the environment.
        Returns observation, reward, terminated, truncated, info.
        """
        new_node = action
        traveled_distance = self._distance_matrix[self._current_node, new_node]
        self._visited_nodes[new_node] = 1
        self._current_node = new_node
        self.visit_order.append(int(new_node))
        self.duration += 1
        self.episode_distance += traveled_distance

        if self._current_node == self.STARTING_NODE:
            terminated = True
            if all(self._visited_nodes):
                reward = self.size - traveled_distance
            else:
                reward = -self._visited_nodes.sum() - traveled_distance
        else:
            terminated = False
            reward = -traveled_distance

        observation = self._get_obs()
        info = self._get_info()

        truncated = False
        if self.duration >= self.max_duration:
            truncated = True

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        """
        Return the current observation.
        """
        return {
            "nodes": self._nodes,
            "current_node": self._current_node,
            "visited_nodes": self._visited_nodes,
        }

    def _get_info(self):
        """
        Return the current info.
        """
        return {
            "n_visited_nodes": self._visited_nodes.sum(),
            "visit_order": self.visit_order,
        }

    def _compute_distance(self) -> np.ndarray:
        """
        Compute the distance matrix between the nodes.
        """
        return np.linalg.norm(self._nodes[:, None] - self._nodes[None, :], axis=-1)

    def render(self):
        """
        Render the environment.
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """
        Render a frame of the environment.
        """
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Draw lines between visited nodes, following the visit order
        for i in range(len(self.visit_order) - 1):
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                self._nodes[self.visit_order[i]] * self.window_size,
                self._nodes[self.visit_order[i + 1]] * self.window_size,
                2,
            )

        # Draw a red square for the starting node
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(self._nodes[self.STARTING_NODE] * self.window_size, (8, 8)),
        )

        # Draw circles representing nodes
        for i in range(1, self.size):
            if self._visited_nodes[i]:
                pygame.draw.circle(
                    canvas,
                    (0, 255, 0),
                    (
                        int(self._nodes[i][0] * self.window_size),
                        int(self._nodes[i][1] * self.window_size),
                    ),
                    8,
                )
            else:
                pygame.draw.circle(
                    canvas,
                    (0, 0, 255),
                    (
                        int(self._nodes[i][0] * self.window_size),
                        int(self._nodes[i][1] * self.window_size),
                    ),
                    8,
                )

        if self.render_mode == "human":
            if pygame.display.get_init():  # Check if display is initialized
                self.window.blit(canvas, (0, 0))  # Blit canvas onto window
                pygame.event.pump()
                pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        """
        Close the pygame window.
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
