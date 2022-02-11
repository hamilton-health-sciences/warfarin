"""Defining constants and modifiable hyperparameters managed by `dvc`."""

import yaml

from ray import tune


# Load dvc-managed params.
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# COMBINE-AF preprocessing parameters.

## Patients with weekly mg doses above this will be removed
DOSE_OUTLIER_THRESHOLD = params["preprocess"]["dose_outlier_threshold"]

## If an event occurs more than this many days away from the last entry,
## ignore it.
EVENT_RANGE = params["preprocess"]["event_range"]

## These are the patient features that are extracted from the baseline data and
## merged with the rest of the data. Patients without these columns will be
## excluded.
STATIC_STATE_COLS = params["preprocess"]["static_state_columns"]

## Clinical visits that are more than MAX_TIME_ELAPSED are put into separate
## trajectories
MAX_TIME_ELAPSED = params["preprocess"]["max_time_elapsed"]

## Remove trajectories that are not useful for RL modeling because they contain
## < 1 valid transition.
MIN_INR_COUNTS = 2

# Data-related parameters shared throughout the code.

## These are the adverse events that are extracted from the events data and
## merged with the rest of the data. Note that STROKE indicates ischemic stroke.
EVENTS_TO_KEEP = list(params["data"]["events_to_keep"].keys())
EVENTS_TO_KEEP_NAMES = [params["data"]["events_to_keep"][event_code]
                        for event_code in EVENTS_TO_KEEP]

## Events to split trajectories on.
EVENTS_TO_SPLIT = params["data"]["events_to_split"]

## The adverse events we want to consider when defining rewards based on events
## and upsampling trajectories with events.
ADV_EVENTS = params["data"]["events_to_evaluate"]

## Processed static columns of the state space.
STATE_COLS = params["data"]["state_columns"]

# Replay buffer parameters.

## Trajectories with fewer than MIN_TRAINING_TRAJECTORY_LENGTH transitions will
## be removed
MIN_TRAIN_TRAJECTORY_LENGTH = (
    params["data"]["min_train_trajectory_length"]
)

## General parameters for feature engineering in the replay buffers.
REPLAY_BUFFER_PARAMS = params["replay_buffer"]["init"]

# Behavior cloner parameters.

## Hyperparameter search options for the BC algo
BC_TUNE_SEED = params["behavior_cloner"]["tune_seed"]
BC_MIN_TRAINING_EPOCHS = params["behavior_cloner"]["min_training_epochs"]
BC_MAX_TRAINING_EPOCHS = params["behavior_cloner"]["max_training_epochs"]
BC_TARGET_METRIC = params["behavior_cloner"]["target_metric"]
BC_TARGET_MODE = params["behavior_cloner"]["target_mode"]
BC_GRID_SEARCH = {
    name: tune.grid_search(values)
    for name, values in params["behavior_cloner"]["hyperparams"].items()
}

# If set to anything other than `None`, will write dataframes from each
# individual step of the preprocessing pipeline to this path, named by the
# preprocessing function.
AUDIT_PATH = "./data/auditing"
AUDIT_PLOT_PATH = "./data/auditing"

# Path to cache things.
CACHE_PATH = "./cache"

# Raw warfarin dose bins
WARFARIN_DOSE_BOUNDS = [-0.001, 5, 12.5, 17.5, 22.5, 27.5, 30, 32.5, 35, 45,
                        1000]
WARFARIN_DOSE_BIN_LABELS = ["<=5", "(5, 12.5]", "(12.5, 17.5]", "(17.5, 22.5]",
                            "(22.5, 27.5]", "(27.5, 30]", "(30, 32.5]",
                            "(32.5, 35]", "(35, 45]", ">45"]

# Age bins
AGE_BOUNDS = [-0.001, 50, 60, 65, 70, 75, 80, 91]
AGE_BIN_LABELS = ["<=50", "(50, 60]", "(60, 65]", "(65, 70]", "(70, 75]",
                  "(75, 80]", ">80"]

# Weight bins
WEIGHT_BOUNDS = [-0.001, 55, 70, 80, 90, 100, 250]
WEIGHT_BIN_LABELS = ["<=55", "(55, 70]", "(70, 80]", "(80, 90]", "(90, 100]",
                     ">100"]

# Hyperparameter search options for the BCQ algo

# Number of hyperparameter combinations to try.
NUM_HYPERPARAMETER_SAMPLES = 50

# Minimum number of training epochs before early termination of the model.
MIN_TRAINING_EPOCHS = 0

# Maximum number of training epochs for each combination of hyperparameters.
MAX_TRAINING_EPOCHS = 2_500

# How often to plot in epochs.
PLOT_EVERY = 100

# Evaluation constants

# Number of bootstrapping samples used for sampling distributions.
NUM_BOOTSTRAP_SAMPLES = 1_000

# The upper thresholds for mean absolute agreement to consider a trajectory
# agreed-upon.
AGREEMENT_THRESHOLDS = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15]

# For INR binning in evaluations. In practice, the closedness of the endpoints
# are not prescribed here, so will need to be modified in the evaluations code
# if changed.
INR_BIN_BOUNDARIES = [-float("inf"), 1.5, 2., 3., 3.5, float("inf")]
INR_BIN_LABELS = ["< 1.5", "1.5 - 2", "2 - 3", "3 - 3.5", "> 3.5"]

# Dose change labels
ACTION_LABELS = ["↓ > 20%",
                 "↓ 10-20%",
                 "↓ ≤ 10%",
                 "Maintain",
                 "↑ ≤ 10%",
                 "↑ 10-20%",
                 "↑ > 20%"]
