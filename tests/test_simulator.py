import numpy as np

from src.simulator import CrowdSimulator


def test_simulator_creates_pedestrians():
    sim = CrowdSimulator(n_pedestrians=10, scenario="uniform", seed=1)

    assert len(sim.pedestrians) == 10
    assert sim.time_step == 0


def test_pedestrians_move_after_step():
    sim = CrowdSimulator(n_pedestrians=10, scenario="uniform", seed=1)

    initial_positions = [
        p.position.copy()
        for p in sim.pedestrians
    ]

    sim.step()

    moved = [
        not np.allclose(p.position, initial_positions[i])
        for i, p in enumerate(sim.pedestrians)
    ]

    assert any(moved)


def test_simulation_returns_metrics():
    sim = CrowdSimulator(n_pedestrians=10, scenario="uniform", seed=1)
    metrics = sim.run(method="nearest")

    assert "completion_rate" in metrics
    assert "avg_evacuation_time" in metrics
    assert metrics["n_pedestrians"] == 10