import os
import pandas as pd

from src.config import EVAL_RESULTS_PATH, Q_TABLE_PATH
from src.q_learning import train_q_learning, load_q_table
from src.baselines import (
    run_nearest_exit_baseline,
    run_random_exit_baseline,
    run_q_learning_policy,
)
from src.plots import plot_method_comparison


def get_or_train_policy():
    """
    Loads the Q-table if it exists.
    If not, trains a new Q-learning policy.
    """
    if os.path.exists(Q_TABLE_PATH):
        print("Loading existing Q-table...")
        policy = load_q_table(Q_TABLE_PATH)
    else:
        print("Q-table not found. Training Q-learning policy...")
        policy, rewards = train_q_learning()

    return policy


def run_evaluation():
    """
    Runs final evaluation:

    Scenarios:
    1. uniform
    2. corner

    Seeds:
    1, 2, 3

    Methods:
    1. random_choice
    2. nearest_exit
    3. q_learning
    """

    os.makedirs("outputs", exist_ok=True)

    scenarios = ["uniform", "corner"]
    seeds = [1, 2, 3]
    n_pedestrians = 50

    policy = get_or_train_policy()

    all_results = []

    for scenario in scenarios:
        for seed in seeds:
            print(f"Running scenario={scenario}, seed={seed}")

            # Random baseline
            random_metrics = run_random_exit_baseline(
                n_pedestrians=n_pedestrians,
                scenario=scenario,
                seed=seed,
            )
            all_results.append(random_metrics)

            # Nearest-exit baseline
            nearest_metrics = run_nearest_exit_baseline(
                n_pedestrians=n_pedestrians,
                scenario=scenario,
                seed=seed,
            )
            all_results.append(nearest_metrics)

            # Q-learning method
            q_metrics = run_q_learning_policy(
                q_policy=policy,
                n_pedestrians=n_pedestrians,
                scenario=scenario,
                seed=seed,
            )
            all_results.append(q_metrics)

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(EVAL_RESULTS_PATH, index=False)
    plot_method_comparison(results_df)

    print("\nEvaluation finished.")
    print(f"Results saved to: {EVAL_RESULTS_PATH}")
    print("Comparison plot saved to: outputs/method_comparison.png")

    print("\nAverage results by method:")
    print(
        results_df.groupby("method")[
            ["avg_evacuation_time", "completion_rate", "total_steps"]
        ].mean()
    )

    return results_df


if __name__ == "__main__":
    run_evaluation()