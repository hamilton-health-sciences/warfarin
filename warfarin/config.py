"""Defining constants."""

# If set to anything other than `None`, will write dataframes from each
# individual step of the preprocessing pipeline to this path, named by the
# preprocessing function.
AUDIT_PATH = "./data/auditing"
AUDIT_PLOT_PATH = "./data/auditing"

# Path to cache things.
CACHE_PATH = "./cache"

# Patients with weekly mg doses above this will be removed
DOSE_OUTLIER_THRESHOLD = 140

# Clinical visits that are more than MAX_TIME_ELAPSED are put into separate
# trajectories
MAX_TIME_ELAPSED = 90

# Remove trajectories that are not useful for RL modeling because they contain
# < 1 valid transition.
MIN_INR_COUNTS = 2

# Trajectories with fewer than MIN_TRAINING_TRAJECTORY_LENGTH transitions will
# be removed
MIN_TRAIN_TRAJECTORY_LENGTH = 5

# The reward associated with INRs that are in therapeutic range
INR_REWARD = 1

# The reward associated with adverse events (ADV_EVENTS)
EVENT_REWARD = 0

# These are the adverse events that are extracted from the events data and
# merged with the rest of the data. Note that STROKE indicates ischemic stroke.
EVENTS_TO_KEEP = ["DEATH", "STROKE", "MAJOR_BLEED", "MINOR_BLEED", "HEM_STROKE",
                  "HOSP", "SYS_EMB"]
EVENTS_TO_KEEP_NAMES = ["Death", "Ischemic Stroke", "Major Bleeding",
                        "Minor Bleeding", "Hemorrhagic Stroke",
                        "Hospitalization", "Systemic Embolism"]

# Events to split on. We don't split on minor bleeds as this would create way
# too many short trajectories, particularly in RE-LY where the per-patient
# rate of minor bleeding is >90%.
EVENTS_TO_SPLIT = ["DEATH", "STROKE", "MAJOR_BLEED", "HEM_STROKE", "HOSP"]

# If an event occurs more than this many days away from the last entry,
# ignore it.
EVENT_RANGE = 90

# These are the patient features that are extracted from the baseline data and
# merged with the rest of the data. Patients without these columns will be
# excluded.
STATIC_STATE_COLS = ["SEX", "RACE2", "SMOKE", "BMED_ASPIRIN",
                     "BMED_AMIOD", "DIABETES", "HX_CHF", "HYPERTENSION",
                     "HX_MI", "BMED_THIENO", "AGE_DEIDENTIFIED", "WEIGHT"]

# The adverse events we want to consider when defining rewards based on events
# and upsampling trajectories with events.
ADV_EVENTS = ["STROKE", "HEM_STROKE", "MAJOR_BLEED"]

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

# Raw columns of the state space.
STATE_COLS = ["AGE_DEIDENTIFIED", "SEX", "WEIGHT", "RACE2", "SMOKE",
              "BMED_ASPIRIN", "BMED_AMIOD", "BMED_THIENO", "DIABETES", "HX_CHF",
              "HYPERTENSION", "HX_MI", "INR_VALUE", "WARFARIN_DOSE",
              "STROKE_FLAG", "MAJOR_BLEED_FLAG", "MINOR_BLEED_FLAG",
              "HEM_STROKE_FLAG", "HOSP_FLAG", "WARFARIN_DOSE_BIN", "AGE_BIN",
              "WEIGHT_BIN"]

# Hyperparameter search options for the BCQ algo

# Number of hyperparameter combinations to try.
NUM_HYPERPARAMETER_SAMPLES = 50

# Minimum number of training epochs before early termination of the model.
MIN_TRAINING_EPOCHS = 0

# Maximum number of training epochs for each combination of hyperparameters.
MAX_TRAINING_EPOCHS = 2_500

# How often to plot in epochs.
PLOT_EVERY = 100

# Hyperparameter search options for the BC algo

NUM_BC_HYPERPARAMETER_SAMPLES = 25

MIN_BC_TRAINING_EPOCHS = 50

MAX_BC_TRAINING_EPOCHS = 500

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
