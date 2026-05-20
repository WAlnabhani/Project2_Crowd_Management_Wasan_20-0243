import numpy as np

# General simulation config

WORLD_WIDTH = 100.0
WORLD_HEIGHT = 100.0

N_PEDESTRIANS = 40
MAX_STEPS = 500

# Pedestrian movement
MAX_SPEED = 1.2
EXIT_RADIUS = 3.0
SAFE_DISTANCE = 3.0
REPULSION_STRENGTH = 0.8
STEERING_STRENGTH = 1.0
# Exit bottleneck
MAX_EXIT_CAPACITY_PER_STEP = 1

# Three exits: left, right, top
EXITS = np.array([
    [0.0, 50.0],      # Exit 1
    [100.0, 50.0],    # Exit 2
    [50.0, 100.0],    # Exit 3
], dtype=float)

N_EXITS = len(EXITS)

# Heatmap
HEATMAP_BINS = 10

# Q-learning
DISTANCE_BINS = [25.0, 60.0]   # near, medium, far
DENSITY_BINS = [8, 18]         # low, medium, high

Q_EPISODES = 3000
ALPHA = 0.1
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995

# Output paths
Q_TABLE_PATH = "outputs/q_table.pkl"
LEARNING_CURVE_PATH = "outputs/learning_curve.png"
HEATMAP_PATH = "outputs/density_heatmap.png"
EVAL_RESULTS_PATH = "outputs/evaluation_results.csv"
STORY_PATH = "outputs/evacuation_story.md"