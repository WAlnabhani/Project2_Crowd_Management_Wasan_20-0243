import os
import pickle
from collections import defaultdict

import numpy as np

from src.config import (
    Q_EPISODES,
    ALPHA,
    GAMMA,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY,
    Q_TABLE_PATH,
    LEARNING_CURVE_PATH,
    N_EXITS,
)
from src.exit_choice_env import ExitChoiceEnv, build_state_from_position
from src.plots import plot_learning_curve


class QLearningPolicy:
    """
    Tabular Q-learning policy for choosing the best exit.
    """

    def __init__(self, q_table=None):
        if q_table is None:
            self.q_table = defaultdict(lambda: np.zeros(N_EXITS))
        else:
            self.q_table = q_table

    def get_q_values(self, state):
        state = tuple(state)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(N_EXITS)

        return self.q_table[state]

    def choose_action(self, state, epsilon=0.0):
        """
        Epsilon-greedy action selection.
        """
        state = tuple(state)

        if np.random.random() < epsilon:
            return np.random.randint(N_EXITS)

        q_values = self.get_q_values(state)
        return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state, done):
        """
        Standard Q-learning update rule.
        """
        state = tuple(state)
        next_state = tuple(next_state)

        current_q = self.get_q_values(state)[action]

        if done:
            target = reward
        else:
            next_max = np.max(self.get_q_values(next_state))
            target = reward + GAMMA * next_max

        new_q = current_q + ALPHA * (target - current_q)
        self.q_table[state][action] = new_q

    def choose_exit(self, position, exit_loads):
        """
        Used during deployment inside the crowd simulator.
        """
        state = build_state_from_position(position, exit_loads)
        return self.choose_action(state, epsilon=0.0)


def train_q_learning(
    episodes=Q_EPISODES,
    seed=1,
    save_path=Q_TABLE_PATH,
):
    """
    Trains tabular Q-learning on the single-agent ExitChoiceEnv.
    """
    env = ExitChoiceEnv(seed=seed)
    policy = QLearningPolicy()

    epsilon = EPSILON_START
    rewards_history = []

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)

        total_reward = 0.0
        done = False

        while not done:
            action = policy.choose_action(state, epsilon=epsilon)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            policy.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )

            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    os.makedirs("outputs", exist_ok=True)

    save_q_table(policy, save_path)
    plot_learning_curve(rewards_history, path=LEARNING_CURVE_PATH)

    return policy, rewards_history


def save_q_table(policy, path=Q_TABLE_PATH):
    """
    Saves Q-table to a pickle file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    normal_dict = {
        key: value
        for key, value in policy.q_table.items()
    }

    with open(path, "wb") as file:
        pickle.dump(normal_dict, file)


def load_q_table(path=Q_TABLE_PATH):
    """
    Loads Q-table and returns a QLearningPolicy.
    """
    with open(path, "rb") as file:
        loaded_table = pickle.load(file)

    q_table = defaultdict(lambda: np.zeros(N_EXITS))
    q_table.update(loaded_table)

    return QLearningPolicy(q_table=q_table)


if __name__ == "__main__":
    policy, rewards = train_q_learning()
    print("Training finished.")
    print(f"Q-table saved to: {Q_TABLE_PATH}")
    print(f"Learning curve saved to: {LEARNING_CURVE_PATH}")