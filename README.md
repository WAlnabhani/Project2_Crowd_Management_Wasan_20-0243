# Project 2: Crowd Management Simulation (Exit Selection)

## Overview

This project implements a continuous 2D crowd evacuation simulator where pedestrians select exits using a trained tabular Q-learning policy. The system compares the learned policy against two baselines: nearest-exit selection and random-exit selection.

The project includes three integrated layers:

1. Core technical layer: continuous boids-style crowd simulator.
2. Reinforcement learning layer: tabular Q-learning exit-selection policy.
3. Agentic AI layer: single-loop LLM narrator using grounded simulation tools.

## Project Features

- Continuous 2D evacuation simulator.
- 30–50 pedestrians.
- Three exits.
- Boids-style steering and repulsion.
- Exit bottleneck capacity.
- Density heatmap generation.
- Tabular Q-learning for exit selection.
- Baseline comparison:
  - Always nearest exit.
  - Random exit choice.
- Evaluation on two scenarios:
  - Uniform spawn.
  - Corner-heavy spawn.
- Three seeds per scenario.
- LLM narrator that produces:
  - One sentence every 30 steps.
  - Final 3-paragraph evacuation story.
- Pytest integration tests.

## Folder Structure

```text
project2_exit_selection/
├── main.py
├── requirements.txt
├── README.md
├── conftest.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── pedestrian.py
│   ├── simulator.py
│   ├── density.py
│   ├── exit_choice_env.py
│   ├── q_learning.py
│   ├── baselines.py
│   ├── evaluation.py
│   ├── agent_tools.py
│   ├── narrator.py
│   └── plots.py
├── outputs/
├── docs/
│   ├── contributions.md
│   ├── ai_assistance_log.md
│   ├── report_outline.md
│   └── prompt_iterations/
└── tests/
```

## Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

Optional: install and run Ollama with llama3.2:3b:

```bash
ollama pull llama3.2:3b
```

The project still runs if Ollama is unavailable because a fallback narrator is included.

## Run the Full Project

```bash
python main.py --mode full --method q_learning --scenario uniform --n_pedestrians 50 --seed 1
```

## Train Q-learning Only

```bash
python main.py --mode train
```

## Run Evaluation

```bash
python main.py --mode evaluate
```

This compares:

- Q-learning
- Nearest-exit baseline
- Random-choice baseline

## Run Tests

```bash
python -m pytest -o cache_dir=outputs/.pytest_cache
```

Expected result:

```text
13 passed
```

## Main Outputs

After running the project, the `outputs` folder contains:

- `q_table.pkl`
- `learning_curve.png`
- `density_heatmap.png`
- `simulation_snapshot.png`
- `exit_loads.png`
- `method_comparison.png`
- `evaluation_results.csv`
- `evacuation_story.md`

## Example Evaluation Result

Average results by method:

```text
method          avg_evacuation_time    completion_rate    total_steps
nearest_exit    37.97                  1.00               63.33
q_learning      51.10                  1.00               134.50
random_choice   148.63                 0.97               500.00
```

The Q-learning method performed better than the random-choice baseline and achieved a full completion rate. The nearest-exit baseline remained strong because the simplified room layout makes distance highly important.

## Final Demonstration Result

For 50 pedestrians in the uniform scenario using Q-learning:

```text
scenario: uniform
seed: 1
n_pedestrians: 50
total_steps: 151
evacuated_count: 50
completion_rate: 1.0
avg_evacuation_time: 49.92
max_evacuation_time: 151.0
final_exit_loads: [0, 0, 0]
method: q_learning
```

## Agentic AI Layer

The project includes a single-loop LLM narrator. The narrator uses three grounded tools:

1. `get_density_at`
2. `get_exit_load`
3. `find_peak`

During the simulation, the narrator produces one sentence every 30 steps using real simulation metrics. At the end, it produces a final 3-paragraph chronological evacuation story.

Example narration:

```text
At step 30, the evacuation continued with exit loads {'Exit 1': 8, 'Exit 2': 17, 'Exit 3': 7} and a peak local density of 4.
```

## MDP Formulation

### State

The Q-learning state includes:

- Discretized distance to each exit.
- Discretized density/load at each exit.

For three exits, the state is represented as:

```text
(distance_exit_1, distance_exit_2, distance_exit_3,
 density_exit_1, density_exit_2, density_exit_3)
```

### Actions

The action space is:

```text
0 = choose Exit 1
1 = choose Exit 2
2 = choose Exit 3
```

### Reward

The reward encourages the agent to select exits that are both close and less congested. The reward penalizes high estimated cost due to long distance or heavy congestion and gives a bonus when the selected exit has the lowest estimated cost.

## Limitations

The simulator is simplified and does not model panic, injuries, individual human psychology, complex obstacles, or detailed building geometry. The Q-learning state uses discretized distance and congestion values, so some continuous information is simplified. The LLM narrator is grounded in tool outputs but does not directly control pedestrian movement.

## Reproducibility

The project uses fixed seeds during evaluation. It includes a test suite covering the simulator, density heatmaps, Q-learning, agent tools, and three-layer integration.

The test suite passed successfully:

```text
13 passed
```