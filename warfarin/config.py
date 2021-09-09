"""Defining constants."""

# If set to anything other than `None`, will write dataframes from each
# individual step of the preprocessing pipeline to this path, named by the
# preprocessing function.
AUDIT_PATH = "./data/auditing"
AUDIT_PLOT_PATH = "./data/auditing"

# Patients with weekly mg doses above this will be removed
DOSE_OUTLIER_THRESHOLD = 140

# Clinical visits that are more than MAX_TIME_ELAPSED are put into separate
# trajectories
MAX_TIME_ELAPSED = 90

# Trajectories with fewer than MIN_INR_COUNTS entries will be removed
MIN_INR_COUNTS = 10

# The reward associated with INRs that are in therapeutic range
INR_REWARD = 1

# The reward associated with adverse events (ADV_EVENTS)
EVENT_REWARD = 0

# These are the adverse events that are extracted from the events data and
# merged with the rest of the data. Note that STROKE indicates ischemic stroke.
EVENTS_TO_KEEP = ["DEATH", "STROKE", "MAJOR_BLEED", "MINOR_BLEED", "HEM_STROKE",
                  "HOSP"]

# Events to split on. We don't split on minor bleeds as this would create way
# too many short trajectories, particularly in RE-LY where the per-patient
# rate of minor bleeding is 90%.
EVENTS_TO_SPLIT = ["DEATH", "STROKE", "MAJOR_BLEED", "HEM_STROKE", "HOSP"]

# If an event occurs more than this many days away from the last entry,
# ignore it.
EVENT_RANGE = 90

# These are the patient features that are extracted from the baseline data and
# merged with the rest of the data
STATIC_STATE_COLS = ["SEX", "CONTINENT", "SMOKE", "BMED_ASPIRIN", "BMED_AMIOD",
                     "DIABETES", "HX_CHF", "HYPERTENSION", "HX_MI",
                     "BMED_THIENO", "AGE_DEIDENTIFIED", "WEIGHT"]

# The adverse events we want to consider when defining rewards based on events
# and upsampling trajectories with events.
ADV_EVENTS = ["STROKE", "HEM_STROKE", "MAJOR_BLEED"]

# Raw warfarin dose bins
WARFARIN_DOSE_BOUNDS = [-0.001, 5, 12.5, 17.5, 22.5, 27.5, 30, 32.5, 35, 45, 1000]
WARFARIN_DOSE_BIN_LABELS = ["<=5", "(5, 12.5]", "(12.5, 17.5]", "(17.5, 22.5]",
                            "(22.5, 27.5]", "(27.5, 30]", "(30, 32.5]",
                            "(32.5, 35]", "(35, 45]", ">45"]

# Age bins
AGE_BOUNDS = [-0.001, 50, 60, 65, 70, 75, 80, 91]
AGE_BIN_LABELS = ["<=50", "(50, 60]", "(60, 65]", "(65, 70]", "(70, 75]",
                  "(75, 80]", ">80"]

# Weight bins
WEIGHT_BOUNDS = [-0.001, 55, 70, 80, 90, 100, 200]
WEIGHT_BIN_LABELS = ["<=55", "(55, 70]", "(70, 80]", "(80, 90]", "(90, 100]",
                     ">100"]

# Raw columns of the state space.
STATE_COLS = ["AGE_DEIDENTIFIED", "SEX", "WEIGHT", "CONTINENT", "SMOKE",
              "BMED_ASPIRIN", "BMED_AMIOD", "BMED_THIENO", "DIABETES", "HX_CHF",
              "HYPERTENSION", "HX_MI", "INR_VALUE", "WARFARIN_DOSE",
              "STROKE_FLAG", "MAJOR_BLEED_FLAG", "MINOR_BLEED_FLAG",
              "HEM_STROKE_FLAG", "HOSP_FLAG", "WARFARIN_DOSE_BIN", "AGE_BIN",
              "WEIGHT_BIN"]

# These are the columns of the state space
STATE_COLS_TO_FILL = ["INR_VALUE","SEX", "BMED_ASPIRIN", "BMED_AMIOD",
                      "DIABETES", "HX_CHF", "HYPERTENSION", "HX_MI",
                      "BMED_THIENO", "AGE_DEIDENTIFIED", "MINOR_BLEED_FLAG",
                      "MAJOR_BLEED_FLAG", "HOSP_FLAG", "WEIGHT",
                      "WARFARIN_DOSE", "CONTINENT_EAST ASIA",
                      "CONTINENT_EASTERN EUROPE", "CONTINENT_LATIN AMERICA",
                      "CONTINENT_NORTH AMERICA","CONTINENT_SOUTH ASIA",
                      "CONTINENT_WESTERN EUROPE", "SMOKE_CURRENT SMOKER",
                      "SMOKE_FORMER SMOKER", "SMOKE_NEVER SMOKED",
                      "INR_VALUE_BIN_<=1.5", "INR_VALUE_BIN_(1.5, 2)",
                      "INR_VALUE_BIN_[2, 3]", "INR_VALUE_BIN_(3, 3.5)",
                      "INR_VALUE_BIN_>=3.5","WARFARIN_DOSE_BIN_<=5",
                      "WARFARIN_DOSE_BIN_(5, 12.5]",
                      "WARFARIN_DOSE_BIN_(12.5, 17.5]",
                      "WARFARIN_DOSE_BIN_(17.5, 22.5]",
                      "WARFARIN_DOSE_BIN_(22.5, 27.5]",
                      "WARFARIN_DOSE_BIN_(27.5, 30]",
                      "WARFARIN_DOSE_BIN_(30, 32.5]",
                      "WARFARIN_DOSE_BIN_(32.5, 35]",
                      "WARFARIN_DOSE_BIN_(35, 45]",
                      "WARFARIN_DOSE_BIN_>45", "AGE_BIN_<=50",
                      "AGE_BIN_(50, 60]", "AGE_BIN_(60, 65]",
                      "AGE_BIN_(65, 70]", "AGE_BIN_(70, 75]",
                      "AGE_BIN_(75, 80]", "AGE_BIN_>80", "WEIGHT_BIN_<=55",
                      "WEIGHT_BIN_(55, 70]", "WEIGHT_BIN_(70, 80]",
                      "WEIGHT_BIN_(80, 90]", "WEIGHT_BIN_(90, 100]",
                      "WEIGHT_BIN_>100"]

# These are columns that are dropped from the preprocessing data before being
# stored. These are intermediate values that are not used anywhere later in the
# pipeline .
DROP_COLS = ["CUMU_MEASUR",
             "END_TRAJ",
             "END_TRAJ_CUMU",
             "FIRST_DAY",
             "FLAG",
             "INR_BIN",
             "INR_BIN_CODES",
             "INR_MEASURED",
             "INR_NEXT_LOWQ",
             "INR_NEXT_UPQ",
             "INR_VALUE_CHANGE",
             "INR_VALUE_CHANGE_SIGN",
             "IS_NULL",
             "IS_NULL_CUMU",
             "LAST_DAY",
             "MISSING_ID",
             "PREV_DOSE",
             "REMOVE",
             "REMOVE_AFTER",
             "REMOVE_PRIOR",
             "START_TRAJ",
             "START_TRAJ_CUMU",
             "SUBJID_NEW",
             "SUBJID_NEW_2",
             "SUBJID_NEW_NEW",
             "USUBJID_O_NEW_2",
             "USUBJID_O_NEW_NEW",
             "WARFARIN_DOSE_CHANGE_BIN",
             "WARFARIN_DOSE_CHANGE_SIGN"]

# Hyperparameter search options

# Number of hyperparameter combinations to try.
NUM_HYPERPARAMETER_SAMPLES = 100

# Minimum number of training epochs before early termination of the model.
MIN_TRAINING_EPOCHS = 250

# Maximum number of training epochs for each combination of hyperparameters.
MAX_TRAINING_EPOCHS = 1_000

# How often to plot in epochs.
PLOT_EVERY = 25

# Evaluation constants

# The upper thresholds for mean absolute agreement to consider a trajectory
# agreed-upon.
AGREEMENT_THRESHOLDS = [0.01, 0.025, 0.05, 0.10]

# For INR binning in evaluations. In practice, the closedness of the endpoints
# are not prescribed here, so will need to be modified in the evaluations code
# if changed.
INR_BIN_BOUNDARIES = [-float("inf"), 1., 2., 3., 4., float("inf")]
INR_BIN_LABELS = ["< 1", "1 - 2", "2 - 3", "3 - 4", "> 4"]

# Dose change labels
ACTION_LABELS = ["Decrease > 20%",
                 "Decrease 10-20%",
                 "Decrease < 10%",
                 "Maintain",
                 "Increase < 10%",
                 "Increase 10-20%",
                 "Increase > 20%"]
