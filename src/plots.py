import matplotlib.pyplot as plt
import numpy as np

from src.config import WORLD_WIDTH, WORLD_HEIGHT, EXITS


def plot_learning_curve(rewards, path="outputs/learning_curve.png"):
    """
    Saves the Q-learning reward curve with a moving average.

    The raw reward curve can look noisy because each episode has random
    pedestrian positions and stochastic exit congestion. The moving average
    makes the learning trend easier to interpret.
    """
    plt.figure(figsize=(9, 5))

    episodes = np.arange(len(rewards))

    # Raw episode rewards
    plt.plot(
        episodes,
        rewards,
        alpha=0.35,
        label="Raw episode reward",
    )

    # Moving average for clearer training trend
    window = 100

    if len(rewards) >= window:
        moving_average = np.convolve(
            rewards,
            np.ones(window) / window,
            mode="valid",
        )

        moving_episodes = episodes[window - 1:]

        plt.plot(
            moving_episodes,
            moving_average,
            linewidth=2.5,
            label=f"{window}-episode moving average",
        )

    plt.title("Q-learning Training Curve")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_simulation_snapshot(pedestrians, path="outputs/simulation_snapshot.png"):
    """
    Saves one snapshot of pedestrian positions and exits.
    """
    plt.figure(figsize=(7, 7))

    active_x = []
    active_y = []
    evacuated_x = []
    evacuated_y = []

    for pedestrian in pedestrians:
        if pedestrian.evacuated:
            evacuated_x.append(pedestrian.position[0])
            evacuated_y.append(pedestrian.position[1])
        else:
            active_x.append(pedestrian.position[0])
            active_y.append(pedestrian.position[1])

    if active_x:
        plt.scatter(
            active_x,
            active_y,
            label="Active pedestrians",
            s=30,
            alpha=0.8,
        )

    if evacuated_x:
        plt.scatter(
            evacuated_x,
            evacuated_y,
            label="Evacuated pedestrians",
            s=30,
            alpha=0.8,
        )

    plt.scatter(
        EXITS[:, 0],
        EXITS[:, 1],
        marker="s",
        s=160,
        label="Exits",
    )

    # Better label placement for exits
    label_offsets = [
        (2, 0),       # Exit 1
        (-13, 0),     # Exit 2
        (2, -4),      # Exit 3
    ]

    for i, exit_pos in enumerate(EXITS):
        dx, dy = label_offsets[i]

        plt.text(
            exit_pos[0] + dx,
            exit_pos[1] + dy,
            f"Exit {i + 1}",
            fontsize=9,
            weight="bold",
        )

    plt.xlim(-5, WORLD_WIDTH + 5)
    plt.ylim(-5, WORLD_HEIGHT + 8)

    plt.title("Crowd Simulation Snapshot", pad=18)
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_method_comparison(results_df, path="outputs/method_comparison.png"):
    """
    Plots average evacuation time for each method.
    """
    grouped = results_df.groupby("method")["avg_evacuation_time"].mean()

    plt.figure(figsize=(7, 5))
    grouped.plot(kind="bar")

    plt.title("Average Evacuation Time by Method")
    plt.xlabel("Method")
    plt.ylabel("Average Evacuation Time")
    plt.xticks(rotation=0)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_exit_loads(exit_loads, path="outputs/exit_loads.png"):
    """
    Saves a bar chart showing the load at each exit.
    """
    exit_names = [
        f"Exit {i + 1}"
        for i in range(len(exit_loads))
    ]

    plt.figure(figsize=(6, 4))
    plt.bar(exit_names, exit_loads)

    plt.title("Exit Loads")
    plt.xlabel("Exit")
    plt.ylabel("Number of Pedestrians")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()