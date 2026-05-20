import numpy as np

from src.q_learning import QLearningPolicy
from src.exit_choice_env import ExitChoiceEnv
from src.config import N_EXITS


def test_qlearning_action_is_valid():
    env = ExitChoiceEnv(seed=1)
    state, _ = env.reset()

    policy = QLearningPolicy()
    action = policy.choose_action(state, epsilon=0.0)

    assert 0 <= action < N_EXITS


def test_qlearning_update_changes_q_value():
    policy = QLearningPolicy()

    state = (0, 1, 2, 0, 1, 2)
    next_state = (0, 1, 2, 0, 1, 2)
    action = 1
    reward = 10.0

    old_value = policy.get_q_values(state)[action]

    policy.update(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        done=True,
    )

    new_value = policy.get_q_values(state)[action]

    assert new_value != old_value


def test_exit_choice_env_step_runs():
    env = ExitChoiceEnv(seed=1)
    state, _ = env.reset()

    next_state, reward, terminated, truncated, info = env.step(0)

    assert len(next_state) == 6
    assert isinstance(reward, float)
    assert terminated is True
    assert "best_exit" in info