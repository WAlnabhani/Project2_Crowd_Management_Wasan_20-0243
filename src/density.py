import numpy as np
import matplotlib.pyplot as plt

from src.config import WORLD_WIDTH, WORLD_HEIGHT, HEATMAP_BINS, HEATMAP_PATH


def compute_density_heatmap(pedestrians, bins=HEATMAP_BINS):
    """
    Computes a 2D density heatmap from active pedestrian positions.
    """
    active_positions = [
        p.position for p in pedestrians
        if not p.evacuated
    ]

    if len(active_positions) == 0:
        return np.zeros((bins, bins))

    active_positions = np.array(active_positions)

    heatmap, _, _ = np.histogram2d(
        active_positions[:, 1],
        active_positions[:, 0],
        bins=bins,
        range=[[0, WORLD_HEIGHT], [0, WORLD_WIDTH]]
    )

    return heatmap


def save_density_heatmap(pedestrians, path=HEATMAP_PATH):
    """
    Saves density heatmap as an image.
    """
    heatmap = compute_density_heatmap(pedestrians)

    plt.figure(figsize=(6, 5))
    plt.imshow(
        heatmap,
        origin="lower",
        extent=[0, WORLD_WIDTH, 0, WORLD_HEIGHT],
        aspect="auto"
    )
    plt.colorbar(label="Pedestrian Density")
    plt.title("Crowd Density Heatmap")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return heatmap


def get_density_at_position(pedestrians, x, y, radius=8.0):
    """
    Counts pedestrians near a specific point.
    """
    point = np.array([x, y], dtype=float)
    count = 0

    for p in pedestrians:
        if not p.evacuated:
            distance = np.linalg.norm(p.position - point)
            if distance <= radius:
                count += 1

    return count