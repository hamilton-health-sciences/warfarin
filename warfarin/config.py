
# coding: utf-8

"""
Defining constants.
"""

# Patients with weekly mg doses above this will be removed
DOSE_OUTLIER_THRESHOLD = 140

# Clinical visits that are more than MAX_TIME_ELAPSED are put into separate trajectories
MAX_TIME_ELAPSED = 90

# Trajectories with fewer than MIN_INR_COUNTS entries will be removed
MIN_INR_COUNTS = 10

# The reward associated with INRs that are in therapeutic range 
INR_REWARD = 1

# These are the adverse events that are extracted from the events data and merged with the rest of the data
# Note that STROKE indicates ischemic stroke
EVENTS_TO_KEEP = ['DEATH', 'STROKE', 'MAJOR_BLEED', 'MINOR_BLEED', 'HEM_STROKE', 'HOSP']

# These are the patient features that are extracted from the baseline data and merged with the rest of the data
STATIC_STATE_COLS = ["SEX", "CONTINENT", "SMOKE", "BMED_ASPIRIN", "BMED_AMIOD", "DIABETES", "HX_CHF",
                         "HYPERTENSION", 'HX_MI', "BMED_THIENO", "AGE_DEIDENTIFIED", "WEIGHT"]

# The adverse events we want to consider in Warfarin dosing
ADV_EVENTS = ["STROKE", "HEM_STROKE", "MAJOR_BLEED"]

# These are the columns of the state space 
STATE_COLS_TO_FILL = ['INR_VALUE','SEX', 'BMED_ASPIRIN', 'BMED_AMIOD', 'DIABETES', 'HX_CHF','HYPERTENSION','HX_MI','BMED_THIENO',            
                     'AGE_DEIDENTIFIED', 'MINOR_BLEED_FLAG','MAJOR_BLEED_FLAG','HOSP_FLAG','WEIGHT','WARFARIN_DOSE','CONTINENT_EAST ASIA',  
                     'CONTINENT_EASTERN EUROPE', 'CONTINENT_LATIN AMERICA', 'CONTINENT_NORTH AMERICA','CONTINENT_SOUTH ASIA',
                     'CONTINENT_WESTERN EUROPE', 'SMOKE_CURRENT SMOKER','SMOKE_FORMER SMOKER', 'SMOKE_NEVER SMOKED','INR_VALUE_BIN_<=1.5',
                     'INR_VALUE_BIN_(1.5, 2)', 'INR_VALUE_BIN_[2, 3]', 'INR_VALUE_BIN_(3, 3.5)', 'INR_VALUE_BIN_>=3.5','WARFARIN_DOSE_BIN_<=5',
                     'WARFARIN_DOSE_BIN_(5, 12.5]','WARFARIN_DOSE_BIN_(12.5, 17.5]','WARFARIN_DOSE_BIN_(17.5, 22.5]','WARFARIN_DOSE_BIN_(22.5, 27.5]',
                     'WARFARIN_DOSE_BIN_(27.5, 30]', 'WARFARIN_DOSE_BIN_(30, 32.5]','WARFARIN_DOSE_BIN_(32.5, 35]','WARFARIN_DOSE_BIN_(35, 45]', 'WARFARIN_DOSE_BIN_>45', 'AGE_BIN_<=50', 'AGE_BIN_(50, 60]', 'AGE_BIN_(60, 65]', 'AGE_BIN_(65, 70]',
                     'AGE_BIN_(70, 75]','AGE_BIN_(75, 80]','AGE_BIN_>80','WEIGHT_BIN_<=55', 'WEIGHT_BIN_(55, 70]', 
'WEIGHT_BIN_(70, 80]', 'WEIGHT_BIN_(80, 90]', 'WEIGHT_BIN_(90, 100]', 'WEIGHT_BIN_>100']

# These are columns that are dropped from the preprocessing data before being stored
# These are intermediate values that are not used anywhere later in the pipeline 
DROP_COLS = ['CUMU_MEASUR',
             'END_TRAJ',
             'END_TRAJ_CUMU',
             'FIRST_DAY',
             'FLAG',
             'INR_BIN',
             'INR_BIN_CODES',
             'INR_MEASURED',
             'INR_NEXT_LOWQ',
             'INR_NEXT_UPQ',
             'INR_VALUE_CHANGE',
             'INR_VALUE_CHANGE_SIGN',
             'IS_NULL',
             'IS_NULL_CUMU',
             'LAST_DAY',
             'MISSING_ID',
             'PREV_DOSE',
             'REMOVE',
             'REMOVE_AFTER',
             'REMOVE_PRIOR',
             'START_TRAJ',
             'START_TRAJ_CUMU',
             'SUBJID_NEW',
             'SUBJID_NEW_2',
             'SUBJID_NEW_NEW',
             'USUBJID_O_NEW_2',
             'USUBJID_O_NEW_NEW',
             'WARFARIN_DOSE_CHANGE_BIN',
             'WARFARIN_DOSE_CHANGE_SIGN']

# Hyperparameter search options
NUM_HYPERPARAMETER_SAMPLES = 10

# 1 epoch = 100 steps
# TODO is this meaningful?
STEPS_PER_EPOCH = 100

MIN_TRAINING_EPOCHS = 250

MAX_TRAINING_EPOCHS = 2_500
