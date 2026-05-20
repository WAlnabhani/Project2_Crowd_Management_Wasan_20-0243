from src.simulator import CrowdSimulator
from src.agent_tools import AgentTools


def test_get_exit_load_returns_all_exits():
    sim = CrowdSimulator(n_pedestrians=20, scenario="uniform", seed=1)
    tools = AgentTools(sim)

    result = tools.get_exit_load()

    assert result["tool"] == "get_exit_load"
    assert "Exit 1" in result["exit_loads"]
    assert "Exit 2" in result["exit_loads"]
    assert "Exit 3" in result["exit_loads"]


def test_find_peak_returns_peak_density():
    sim = CrowdSimulator(n_pedestrians=20, scenario="uniform", seed=1)
    tools = AgentTools(sim)

    result = tools.find_peak()

    assert result["tool"] == "find_peak"
    assert "peak_density" in result
    assert result["peak_density"] >= 0


def test_summarize_tools_contains_required_outputs():
    sim = CrowdSimulator(n_pedestrians=20, scenario="uniform", seed=1)
    sim.step()

    tools = AgentTools(sim)
    summary = tools.summarize_tools()

    assert "exit_loads" in summary
    assert "peak_area" in summary
    assert "density_at_peak" in summary