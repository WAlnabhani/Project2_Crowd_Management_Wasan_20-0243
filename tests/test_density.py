import numpy as np

from src.simulator import CrowdSimulator
from src.density import compute_density_heatmap, get_density_at_position
from src.config import HEATMAP_BINS


def test_density_heatmap_shape():
    sim = CrowdSimulator(n_pedestrians=20, scenario="uniform", seed=1)

    heatmap = compute_density_heatmap(sim.pedestrians)

    assert heatmap.shape == (HEATMAP_BINS, HEATMAP_BINS)


def test_density_heatmap_counts_active_pedestrians():
    sim = CrowdSimulator(n_pedestrians=20, scenario="uniform", seed=1)

    heatmap = compute_density_heatmap(sim.pedestrians)
    active_count = sum(not p.evacuated for p in sim.pedestrians)

    assert int(np.sum(heatmap)) == active_count


def test_get_density_at_position_returns_number():
    sim = CrowdSimulator(n_pedestrians=20, scenario="uniform", seed=1)

    density = get_density_at_position(
        pedestrians=sim.pedestrians,
        x=50,
        y=50,
        radius=20,
    )

    assert isinstance(density, int)
    assert density >= 0