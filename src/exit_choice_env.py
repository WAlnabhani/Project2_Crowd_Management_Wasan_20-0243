import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.config import (
    WORLD_WIDTH,
    WORLD_HEIGHT,
    EXITS,
    N_EXITS,
    DISTANCE_BINS,
    DENSITY_BINS,
)


def discretize_distance(distance):
    """
    Converts continuous distance into:
    0 = near
    1 = medium
    2 = far
    """
    return int(np.digitize(distance, DISTANCE_BINS))


def discretize_density(density):
    """
    Converts exit load into:
    0 = low
    1 = medium
    2 = high
    """
    return int(np.digitize(density, DENSITY_BINS))


def build_state_from_position(position, exit_loads):
    """
    Builds the discretized Q-learning state.

    State:
    distance category to each exit + density category at each exit

    Example with 3 exits:
    (
        distance_exit_1,
        distance_exit_2,
        distance_exit_3,
        density_exit_1,
        density_exit_2,
        density_exit_3
    )
    """
    distances = np.linalg.norm(EXITS - position, axis=1)

    distance_state = [
        discretize_distance(distance)
        for distance in distances
    ]

    density_state = [
        discretize_density(load)
        for load in exit_loads
    ]

    return tuple(distance_state + density_state)


class ExitChoiceEnv(gym.Env):
    """
    Single-agent environment for learning exit selection.

    The agent observes:
    - distance category to each exit
    - congestion/density category at each exit

    The agent chooses:
    - exit 1, exit 2, or exit 3
    """

    metadata = {"render_modes": []}

    def __init__(self, seed=None):
        super().__init__()

        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(N_EXITS)

        # 3 distance categories for each exit + 3 density categories for each exit
        self.observation_space = spaces.MultiDiscrete([3] * (2 * N_EXITS))

        self.position = None
        self.exit_loads = None
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Random pedestrian position
        x = self.rng.uniform(5, WORLD_WIDTH - 5)
        y = self.rng.uniform(5, WORLD_HEIGHT - 5)
        self.position = np.array([x, y], dtype=float)

        # Stochastic congestion/load at each exit
        self.exit_loads = self.rng.integers(low=0, high=25, size=N_EXITS)

        self.state = build_state_from_position(
            position=self.position,
            exit_loads=self.exit_loads,
        )

        return np.array(self.state, dtype=np.int64), {}

    def step(self, action):
        action = int(action)

        distances = np.linalg.norm(EXITS - self.position, axis=1)

        # Stochastic noise represents changing congestion during evacuation
        congestion_noise = self.rng.normal(loc=0.0, scale=1.0, size=N_EXITS)
        effective_loads = np.maximum(0, self.exit_loads + congestion_noise)

        # Estimated cost of each exit:
        # distance cost + congestion cost
        exit_costs = (distances / 6.0) + (effective_loads * 0.6)

        chosen_cost = exit_costs[action]
        best_action = int(np.argmin(exit_costs))

        # Reward encourages choosing close and less congested exits
        reward = -float(chosen_cost)

        if action == best_action:
            reward += 10.0
        else:
            reward -= 2.0

        terminated = True
        truncated = False

        info = {
            "position": self.position.tolist(),
            "exit_loads": self.exit_loads.tolist(),
            "exit_costs": exit_costs.tolist(),
            "chosen_exit": action,
            "best_exit": best_action,
        }

        return np.array(self.state, dtype=np.int64), reward, terminated, truncated, info