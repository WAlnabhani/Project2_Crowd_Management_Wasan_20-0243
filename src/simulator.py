import numpy as np

from src.config import (
    WORLD_WIDTH,
    WORLD_HEIGHT,
    N_PEDESTRIANS,
    MAX_STEPS,
    MAX_SPEED,
    EXIT_RADIUS,
    SAFE_DISTANCE,
    REPULSION_STRENGTH,
    STEERING_STRENGTH,
    EXITS,
    MAX_EXIT_CAPACITY_PER_STEP,
)
from src.pedestrian import Pedestrian


class CrowdSimulator:
    """
    Continuous 2D crowd evacuation simulator.

    Pedestrians move using:
    1. Steering force toward the selected exit.
    2. Repulsion force from nearby pedestrians.
    3. Boundary clipping to keep them inside the room.
    """

    def __init__(
        self,
        n_pedestrians=N_PEDESTRIANS,
        scenario="uniform",
        seed=1,
        max_steps=MAX_STEPS,
    ):
        self.n_pedestrians = n_pedestrians
        self.scenario = scenario
        self.seed = seed
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.time_step = 0
        self.pedestrians = self._create_pedestrians()
        self.history = []

    def _create_pedestrians(self):
        pedestrians = []

        for _ in range(self.n_pedestrians):
            if self.scenario == "corner":
                # Most pedestrians start near the bottom-left corner
                x = self.rng.normal(20, 8)
                y = self.rng.normal(20, 8)
                x = np.clip(x, 5, WORLD_WIDTH - 5)
                y = np.clip(y, 5, WORLD_HEIGHT - 5)
            else:
                # Uniform spawn over the room
                x = self.rng.uniform(10, WORLD_WIDTH - 10)
                y = self.rng.uniform(10, WORLD_HEIGHT - 10)

            position = np.array([x, y], dtype=float)
            velocity = np.zeros(2, dtype=float)

            # Initial choice: nearest exit
            chosen_exit = self.get_nearest_exit(position)

            pedestrians.append(
                Pedestrian(
                    position=position,
                    velocity=velocity,
                    chosen_exit=chosen_exit,
                )
            )

        return pedestrians

    def get_nearest_exit(self, position):
        distances = np.linalg.norm(EXITS - position, axis=1)
        return int(np.argmin(distances))

    def get_random_exit(self):
        return int(self.rng.integers(0, len(EXITS)))

    def get_exit_loads(self):
        """
        Counts how many active pedestrians are currently heading to each exit.
        """
        loads = np.zeros(len(EXITS), dtype=int)

        for pedestrian in self.pedestrians:
            if not pedestrian.evacuated:
                loads[pedestrian.chosen_exit] += 1

        return loads

    def set_exit_choices(self, method="nearest", q_policy=None):
        """
        Assigns exit choices to all active pedestrians.

        method:
        - nearest: choose closest exit
        - random: choose random exit
        - q_learning: use trained Q-learning policy

        For q_learning, exit loads are updated dynamically while assigning
        choices. This allows the policy to react to predicted congestion.
        """

        if method == "q_learning":
            if q_policy is None:
                raise ValueError("q_policy is required when method='q_learning'")

            # Start with zero planned loads, then update as pedestrians choose exits
            planned_exit_loads = np.zeros(len(EXITS), dtype=int)

            active_pedestrians = [
                pedestrian for pedestrian in self.pedestrians
                if not pedestrian.evacuated
            ]

            # Sort pedestrians by distance to nearest exit
            # This makes assignment more stable and reproducible
            active_pedestrians.sort(
                key=lambda p: np.min(np.linalg.norm(EXITS - p.position, axis=1))
            )

            for pedestrian in active_pedestrians:
                chosen_exit = int(
                    q_policy.choose_exit(
                        position=pedestrian.position,
                        exit_loads=planned_exit_loads,
                    )
                )

                pedestrian.chosen_exit = chosen_exit
                planned_exit_loads[chosen_exit] += 1

            return

        for pedestrian in self.pedestrians:
            if pedestrian.evacuated:
                continue

            if method == "nearest":
                pedestrian.chosen_exit = self.get_nearest_exit(pedestrian.position)

            elif method == "random":
                pedestrian.chosen_exit = self.get_random_exit()

            else:
                raise ValueError(f"Unknown method: {method}")

    def _steering_force(self, pedestrian):
        """
        Force that moves the pedestrian toward the selected exit.
        """
        target_exit = EXITS[pedestrian.chosen_exit]
        direction = target_exit - pedestrian.position
        distance = np.linalg.norm(direction)

        if distance == 0:
            return np.zeros(2)

        direction = direction / distance
        return STEERING_STRENGTH * direction

    def _repulsion_force(self, pedestrian):
        """
        Force that pushes pedestrians away from each other if they are too close.
        """
        force = np.zeros(2, dtype=float)

        for other in self.pedestrians:
            if other is pedestrian or other.evacuated:
                continue

            difference = pedestrian.position - other.position
            distance = np.linalg.norm(difference)

            if 0 < distance < SAFE_DISTANCE:
                direction = difference / distance
                strength = REPULSION_STRENGTH * (SAFE_DISTANCE - distance) / SAFE_DISTANCE
                force += strength * direction

        return force

    def _keep_inside_room(self, position):
        """
        Keeps pedestrians inside the 2D area.
        """
        position[0] = np.clip(position[0], 0, WORLD_WIDTH)
        position[1] = np.clip(position[1], 0, WORLD_HEIGHT)
        return position

    def step(self):
        """
        Runs one simulation step.

        Important:
        Exit capacity is limited. This creates realistic bottlenecks,
        so choosing the nearest exit is not always optimal when that exit
        is crowded.
        """
        self.time_step += 1

        exit_candidates = {exit_id: [] for exit_id in range(len(EXITS))}

        for pedestrian in self.pedestrians:
            if pedestrian.evacuated:
                continue

            steering = self._steering_force(pedestrian)
            repulsion = self._repulsion_force(pedestrian)

            total_force = steering + repulsion

            speed = np.linalg.norm(total_force)
            if speed > MAX_SPEED:
                total_force = total_force / speed * MAX_SPEED

            pedestrian.velocity = total_force
            pedestrian.position = pedestrian.position + pedestrian.velocity
            pedestrian.position = self._keep_inside_room(pedestrian.position)

            target_exit = EXITS[pedestrian.chosen_exit]
            distance_to_exit = np.linalg.norm(pedestrian.position - target_exit)

            if distance_to_exit <= EXIT_RADIUS:
                exit_candidates[pedestrian.chosen_exit].append(pedestrian)

        # Only a limited number of pedestrians can evacuate through each exit per step
        for exit_id, candidates in exit_candidates.items():
            if len(candidates) == 0:
                continue

            candidates.sort(
                key=lambda p: np.linalg.norm(p.position - EXITS[exit_id])
            )

            allowed_to_leave = candidates[:MAX_EXIT_CAPACITY_PER_STEP]

            for pedestrian in allowed_to_leave:
                pedestrian.evacuated = True
                pedestrian.evacuation_time = self.time_step

        self._save_step_history()

    def _save_step_history(self):
        active_count = sum(not p.evacuated for p in self.pedestrians)
        evacuated_count = self.n_pedestrians - active_count
        exit_loads = self.get_exit_loads()

        self.history.append(
            {
                "step": self.time_step,
                "active": active_count,
                "evacuated": evacuated_count,
                "exit_loads": exit_loads.tolist(),
            }
        )

    def all_evacuated(self):
        return all(p.evacuated for p in self.pedestrians)

    def run(self, method="nearest", q_policy=None, reselect_every=30):
        """
        Runs the full evacuation simulation.
        """
        self.set_exit_choices(method=method, q_policy=q_policy)

        while self.time_step < self.max_steps and not self.all_evacuated():

            # Re-select exits every 30 steps to respond to congestion
            if self.time_step % reselect_every == 0:
                self.set_exit_choices(method=method, q_policy=q_policy)

            self.step()

        return self.get_metrics()

    def get_metrics(self):
        evacuation_times = [
            p.evacuation_time
            for p in self.pedestrians
            if p.evacuation_time is not None
        ]

        evacuated_count = len(evacuation_times)
        completion_rate = evacuated_count / self.n_pedestrians

        if len(evacuation_times) > 0:
            avg_evacuation_time = float(np.mean(evacuation_times))
            max_evacuation_time = float(np.max(evacuation_times))
        else:
            avg_evacuation_time = None
            max_evacuation_time = None

        return {
            "scenario": self.scenario,
            "seed": self.seed,
            "n_pedestrians": self.n_pedestrians,
            "total_steps": self.time_step,
            "evacuated_count": evacuated_count,
            "completion_rate": completion_rate,
            "avg_evacuation_time": avg_evacuation_time,
            "max_evacuation_time": max_evacuation_time,
            "final_exit_loads": self.get_exit_loads().tolist(),
        }