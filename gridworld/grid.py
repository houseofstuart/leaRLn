import numpy as np
import gymnasium as gym
from typing import Optional


class GridWorldEnv(gym.Env):
    def __init__(self, size=5, render_mode=None):
        super().__init__()

        self.size = 5
        # Intialise positons, set each episode in reset()
        self._agent_location = np.array([-1, -1], dtype=np.int32)

        # Agent observations
        self.observation_space = gym.spaces.MultiDiscrete([5,5])

        # Define available agent actions
        self.action_space = gym.spaces.Discrete(4)

        # Map action numbers to grid movements
        self._action_to_direction = {
            0: np.array([0, 1]),  # Right (column + 1)
            1: np.array([-1, 0]), # Up    (row - 1)
            2: np.array([0, -1]), # Left  (row + 1)
            3: np.array([1, 0])   # Down  (column + 1)
        }

        # Define world cells
        self.start_cell = np.array([0, 0], dtype=np.int32)
        self.target_cell = np.array([4, 4], dtype=np.int32)
        self.wall_cells = np.array(
            [[1,1], [1,3], [2,3], [3,1]], dtype=np.int32
        )
        self.hazard_cells = np.array(
            [[0,3], [3,3]], dtype=np.int32
        )

    def _get_obs(self):
        return self._agent_location

    def _get_info(self):
        return None

    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed)

        # Place agent onto starting position
        self._agent_location = self.start_cell.copy()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Map the discrete action to movement direction
        direction = self._action_to_direction[action]

        forward_step_location = self._agent_location + direction
        blocked = np.any(np.all(self.wall_cells == forward_step_location, axis=1))

        # Update agent position, keeping within grid bounds
        if not blocked:    
            self._agent_location = np.clip(
                self._agent_location + direction, 0, self.size - 1
            )

        # Check if agent reached the target or hazard
        harzarded = np.any(
            np.all(self.hazard_cells == self._agent_location, axis=1)
        )
        terminated = np.array_equal(self._agent_location, self.target_cell) or (harzarded)

        # Check if agent reached the step limit
        truncated = False

        # Give the agent a reward
        reward = 10 if terminated else -5 if harzarded else -0.1

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info