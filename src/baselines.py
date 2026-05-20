from src.simulator import CrowdSimulator


def run_nearest_exit_baseline(
    n_pedestrians=40,
    scenario="uniform",
    seed=1,
):
    """
    Baseline 1:
    Every pedestrian always chooses the nearest exit.
    """
    simulator = CrowdSimulator(
        n_pedestrians=n_pedestrians,
        scenario=scenario,
        seed=seed,
    )

    metrics = simulator.run(method="nearest")
    metrics["method"] = "nearest_exit"

    return metrics


def run_random_exit_baseline(
    n_pedestrians=40,
    scenario="uniform",
    seed=1,
):
    """
    Baseline 2:
    Every pedestrian randomly chooses an exit.
    """
    simulator = CrowdSimulator(
        n_pedestrians=n_pedestrians,
        scenario=scenario,
        seed=seed,
    )

    metrics = simulator.run(method="random")
    metrics["method"] = "random_choice"

    return metrics


def run_q_learning_policy(
    q_policy,
    n_pedestrians=40,
    scenario="uniform",
    seed=1,
):
    """
    RL method:
    Every pedestrian uses the trained Q-learning policy to choose an exit.
    """
    simulator = CrowdSimulator(
        n_pedestrians=n_pedestrians,
        scenario=scenario,
        seed=seed,
    )

    metrics = simulator.run(method="q_learning", q_policy=q_policy)
    metrics["method"] = "q_learning"

    return metrics