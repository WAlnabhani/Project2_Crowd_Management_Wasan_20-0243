import numpy as np

from src.config import EXITS, WORLD_WIDTH, WORLD_HEIGHT, HEATMAP_BINS
from src.density import compute_density_heatmap, get_density_at_position


class AgentTools:
    """
    Tool layer used by the LLM narrator.

    Tools:
    1. get_density_at
    2. get_exit_load
    3. find_peak
    """

    def __init__(self, simulator):
        self.simulator = simulator

    def get_density_at(self, x, y, radius=8.0):
        """
        Returns the number of active pedestrians near a point.
        """
        density = get_density_at_position(
            pedestrians=self.simulator.pedestrians,
            x=x,
            y=y,
            radius=radius,
        )

        return {
            "tool": "get_density_at",
            "x": float(x),
            "y": float(y),
            "radius": float(radius),
            "density": int(density),
        }

    def get_exit_load(self, exit_id=None):
        """
        Returns current load at one exit or all exits.

        exit_id uses human numbering:
        exit_id = 1 means Exit 1.
        """
        loads = self.simulator.get_exit_loads()

        if exit_id is None:
            return {
                "tool": "get_exit_load",
                "exit_loads": {
                    f"Exit {i + 1}": int(load)
                    for i, load in enumerate(loads)
                },
            }

        index = int(exit_id) - 1

        if index < 0 or index >= len(loads):
            return {
                "tool": "get_exit_load",
                "error": "Invalid exit_id",
            }

        return {
            "tool": "get_exit_load",
            "exit_id": int(exit_id),
            "load": int(loads[index]),
        }

    def find_peak(self):
        """
        Finds the highest-density area in the room.
        """
        heatmap = compute_density_heatmap(
            pedestrians=self.simulator.pedestrians,
            bins=HEATMAP_BINS,
        )

        peak_value = int(np.max(heatmap))

        if peak_value == 0:
            return {
                "tool": "find_peak",
                "peak_density": 0,
                "location": None,
            }

        peak_index = np.unravel_index(np.argmax(heatmap), heatmap.shape)

        row, col = peak_index

        cell_width = WORLD_WIDTH / HEATMAP_BINS
        cell_height = WORLD_HEIGHT / HEATMAP_BINS

        x_center = (col + 0.5) * cell_width
        y_center = (row + 0.5) * cell_height

        return {
            "tool": "find_peak",
            "peak_density": peak_value,
            "location": {
                "x": float(x_center),
                "y": float(y_center),
            },
        }

    def summarize_tools(self):
        """
        Returns all main tool outputs together.
        """
        peak = self.find_peak()
        loads = self.get_exit_load()

        if peak["location"] is None:
            density_at_peak = {
                "density": 0
            }
        else:
            density_at_peak = self.get_density_at(
                peak["location"]["x"],
                peak["location"]["y"],
            )

        return {
            "step": self.simulator.time_step,
            "exit_loads": loads,
            "peak_area": peak,
            "density_at_peak": density_at_peak,
        }