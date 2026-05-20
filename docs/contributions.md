# Contributions

## Individual Work

This project was completed individually by Wasan.

## Module Ownership

### Core Simulator

- Designed and implemented the continuous 2D crowd evacuation simulator.
- Implemented pedestrian movement using steering toward exits.
- Implemented repulsion between pedestrians.
- Added exit bottleneck capacity.
- Implemented uniform and corner-heavy spawn scenarios.

### Reinforcement Learning

- Implemented the single-agent Gymnasium exit-choice environment.
- Formulated the tabular Q-learning MDP.
- Implemented Q-table training and saving.
- Deployed the trained Q-learning policy in the multi-pedestrian simulator.

### Baselines and Evaluation

- Implemented the nearest-exit baseline.
- Implemented the random-choice baseline.
- Compared Q-learning against both baselines.
- Evaluated the system on two scenarios and three seeds.
- Generated evaluation results and plots.

### Agentic AI Layer

- Implemented the LLM narrator.
- Implemented the required tools:
  - `get_density_at`
  - `get_exit_load`
  - `find_peak`
- Integrated the narrator to produce one sentence every 30 steps.
- Generated the final 3-paragraph evacuation story.

### Testing and Reproducibility

- Implemented tests for simulator behavior.
- Implemented tests for density heatmaps.
- Implemented tests for Q-learning.
- Implemented tests for agent tools.
- Implemented integration testing for the three-layer system.
- Verified that all tests passed successfully.

## AI Assistance

AI assistance was used for project planning, code organization, debugging, documentation drafting, and prompt improvement. All code was reviewed, executed, tested, and modified by the student before submission.

## Verification

The project was run locally. The integrated pipeline generated simulation metrics, heatmaps, evaluation results, and the final evacuation story. The test suite passed successfully with 13 tests.