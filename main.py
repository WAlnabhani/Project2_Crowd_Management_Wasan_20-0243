import argparse
import os

from src.simulator import CrowdSimulator
from src.q_learning import train_q_learning, load_q_table
from src.config import Q_TABLE_PATH, STORY_PATH
from src.agent_tools import AgentTools
from src.narrator import EvacuationNarrator
from src.density import save_density_heatmap
from src.plots import plot_simulation_snapshot, plot_exit_loads
from src.evaluation import run_evaluation


def get_or_train_policy():
    """
    Loads Q-table if available. Otherwise trains a new Q-learning policy.
    """
    if os.path.exists(Q_TABLE_PATH):
        print("Loading existing Q-table...")
        return load_q_table(Q_TABLE_PATH)

    print("Q-table not found. Training Q-learning policy...")
    policy, _ = train_q_learning()
    return policy


def run_demo_simulation(method, scenario, n_pedestrians, seed):
    """
    Runs the integrated simulator + RL policy + LLM narrator.
    """
    os.makedirs("outputs", exist_ok=True)

    q_policy = None
    if method == "q_learning":
        q_policy = get_or_train_policy()

    simulator = CrowdSimulator(
        n_pedestrians=n_pedestrians,
        scenario=scenario,
        seed=seed,
    )

    narrator = EvacuationNarrator()
    tools = AgentTools(simulator)

    simulator.set_exit_choices(method=method, q_policy=q_policy)

    saved_mid_outputs = False

    print("\nStarting integrated Project 2 simulation...")
    print(f"Method: {method}")
    print(f"Scenario: {scenario}")
    print(f"Pedestrians: {n_pedestrians}")
    print(f"Seed: {seed}\n")

    while simulator.time_step < simulator.max_steps and not simulator.all_evacuated():

        # Re-select exits every 30 steps to react to congestion
        if simulator.time_step % 30 == 0:
            simulator.set_exit_choices(method=method, q_policy=q_policy)

        simulator.step()

        # LLM narrator speaks every 30 steps
        if simulator.time_step % 30 == 0:
            summary = tools.summarize_tools()
            sentence = narrator.narrate_step(simulator.time_step, summary)
            print(sentence)

            # Save useful mid-evacuation visuals before everyone exits
            if not saved_mid_outputs:
                save_density_heatmap(
                    simulator.pedestrians,
                    path="outputs/density_heatmap.png",
                )

                plot_simulation_snapshot(
                    simulator.pedestrians,
                    path="outputs/simulation_snapshot.png",
                )

                plot_exit_loads(
                    simulator.get_exit_loads(),
                    path="outputs/exit_loads.png",
                )

                print("Saved mid-evacuation exit loads:", simulator.get_exit_loads())

                saved_mid_outputs = True

    metrics = simulator.get_metrics()
    metrics["method"] = method

    print("\nFinal metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # If the simulation ended before step 30, save visuals anyway
    if not saved_mid_outputs:
        save_density_heatmap(
            simulator.pedestrians,
            path="outputs/density_heatmap.png",
        )

        plot_simulation_snapshot(
            simulator.pedestrians,
            path="outputs/simulation_snapshot.png",
        )

        plot_exit_loads(
            simulator.get_exit_loads(),
            path="outputs/exit_loads.png",
        )

    # Save final story
    story = narrator.final_story(metrics)
    narrator.save_story(story, path=STORY_PATH)

    print("\nSaved outputs:")
    print("outputs/density_heatmap.png")
    print("outputs/simulation_snapshot.png")
    print("outputs/exit_loads.png")
    print(STORY_PATH)

    print("\nFinal evacuation story:\n")
    print(story)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Project 2: Crowd Management Exit Selection"
    )

    parser.add_argument(
        "--mode",
        choices=["train", "simulate", "evaluate", "full"],
        default="full",
        help=(
            "train: train Q-learning, "
            "simulate: run one demo, "
            "evaluate: run comparisons, "
            "full: train/load + simulate"
        ),
    )

    parser.add_argument(
        "--method",
        choices=["q_learning", "nearest", "random"],
        default="q_learning",
        help="Exit selection method used in the simulator.",
    )

    parser.add_argument(
        "--scenario",
        choices=["uniform", "corner"],
        default="uniform",
        help="Spawn scenario.",
    )

    parser.add_argument(
        "--n_pedestrians",
        type=int,
        default=50,
        help="Number of pedestrians.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed.",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_q_learning()

    elif args.mode == "simulate":
        run_demo_simulation(
            method=args.method,
            scenario=args.scenario,
            n_pedestrians=args.n_pedestrians,
            seed=args.seed,
        )

    elif args.mode == "evaluate":
        run_evaluation()

    elif args.mode == "full":
        run_demo_simulation(
            method=args.method,
            scenario=args.scenario,
            n_pedestrians=args.n_pedestrians,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()