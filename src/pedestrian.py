from dataclasses import dataclass
import numpy as np


@dataclass
class Pedestrian:
    """
    Represents one pedestrian in the continuous 2D evacuation simulator.
    """
    position: np.ndarray
    velocity: np.ndarray
    chosen_exit: int
    evacuated: bool = False
    evacuation_time: int | None = None

    def distance_to(self, point: np.ndarray) -> float:
        return float(np.linalg.norm(self.position - point))