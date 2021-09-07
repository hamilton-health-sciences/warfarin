import pandas as pd

import numpy as np

from warfarin import config


def engineer_features(df):
    import pdb; pdb.set_trace()
    # TODO can we do this more gracefully?
    # Clamp INR in [0.5, 4.5]
    df["INR_VALUE"] = np.maximum(np.minimum(df["INR_VALUE"], 4.5), 0.5)

    # Compute the dose change undertaken by the clinician
    df["WARFARIN_DOSE_MULT"] = df.groupby(
        ["TRIAL", "SUBJID", "TRAJID"]
    )["WARFARIN_DOSE"].shift(-1) / df["WARFARIN_DOSE"]

    # Adverse event flags. Ensure they're carried over even between trajectories
    # within the same patient.
    for adv_event_name in config.EVENTS_TO_KEEP:
        colname = f"{adv_event_name}_FLAG"
        df[colname] = np.minimum(
            df.groupby("SUBJID")[adv_event_name].cumsum(), 1
        )

    # Discretize continuous features
    df["WARFARIN_DOSE_BIN"] = pd.cut(df["WARFARIN_DOSE"],
                                     bins=config.WARFARIN_DOSE_BOUNDS,
                                     labels=config.WARFARIN_DOSE_BIN_LABELS)
    df["AGE_BIN"] = pd.cut(df["AGE_DEIDENTIFIED"],
                           bins=config.AGE_BOUNDS,
                           labels=config.AGE_BIN_LABELS)
    df["WEIGHT_BIN"] = pd.cut(df["WEIGHT"],
                              bins=config.WEIGHT_BOUNDS,
                              labels=config.WEIGHT_BIN_LABELS)

    # Subset to state columns
    import pdb; pdb.set_trace()

    return df


def extract_observed_decision(df):
    pass


def compute_reward(df):
    pass


def compute_k(df):
    pass


def compute_done(df):
    pass


def compute_sample_probability(df, relative_event_sample_probability):
    pass
