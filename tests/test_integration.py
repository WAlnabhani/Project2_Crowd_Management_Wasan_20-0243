from src.simulator import CrowdSimulator
from src.q_learning import QLearningPolicy
from src.agent_tools import AgentTools
from src.narrator import EvacuationNarrator


def test_three_layer_integration_runs():
    """
    Tests the three required layers together:

    1. Core simulator
    2. Q-learning policy interface
    3. Agent tools and narrator
    """
    sim = CrowdSimulator(
        n_pedestrians=10,
        scenario="uniform",
        seed=1,
    )

    policy = QLearningPolicy()
    sim.set_exit_choices(method="q_learning", q_policy=policy)

    sim.step()

    tools = AgentTools(sim)
    summary = tools.summarize_tools()

    narrator = EvacuationNarrator()
    sentence = narrator.narrate_step(
        step=sim.time_step,
        tool_summary=summary,
    )

    assert sim.time_step == 1
    assert "exit_loads" in summary
    assert isinstance(sentence, str)
    assert len(sentence) > 0