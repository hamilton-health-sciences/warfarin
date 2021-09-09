import pandas as pd

import numpy as np

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from warfarin import config


def engineer_state_features(df, params=None):
    if params is None:
        params = {}

    # TODO can we do this more gracefully?
    # Clamp INR in [0.5, 4.5]
    df["INR_VALUE"] = np.maximum(np.minimum(df["INR_VALUE"], 4.5), 0.5)

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

    # Subset to columns used in the state space
    df = df[config.STATE_COLS]

    # One-hot encode categorical features
    df_cat = df.select_dtypes("category")
    encoder = OneHotEncoder(drop="first")
    if "categorical" in params:
        encoder.set_params(params["categorical"])
    else:
        encoder.fit(df_cat)
        params["categorical"] = encoder.get_params()
    colnames = [
        "_".join((df_cat.columns[i], str(level)))
        for i in range(len(df_cat.columns))
        for level in encoder.categories_[i][1:]
    ]
    df_cat_ohe = pd.DataFrame(
        encoder.transform(df_cat).todense(),
        index=df_cat.index,
        columns=colnames
    )
    # Ensure missing values propagate
    for colname in df_cat.columns:
        colnull = df_cat[colname].isnull()
        new_colnames = [c for c in df_cat_ohe if colname in c]
        df_cat_ohe.loc[colnull, new_colnames] = np.nan
    df_cat_ohe = df_cat_ohe[[c for c in df_cat_ohe if "nan" not in c]].copy()

    # Scale continuous features to [0, 1]
    df_cts = df.select_dtypes("number")
    encoder = MinMaxScaler()
    if "continuous" in params:
        encoder.set_params(params["continuous"])
    else:
        encoder.fit(df_cts)
        params["continuous"] = encoder.get_params()
    df_cts_scaled = pd.DataFrame(
        encoder.transform(df_cts),
        index=df_cts.index,
        columns=df_cts.columns
    )

    # Join as state cols
    df = df_cat_ohe.join(df_cts_scaled)

    # Provide previous INRs in state
    df["INR_VALUE_PREV_1"] = df.groupby(
        ["SUBJID", "TRAJID"]
    )["INR_VALUE"].shift(1).fillna(df["INR_VALUE"])
    for i in range(2, 5):
        df[f"INR_VALUE_PREV_{i}"] = df.groupby(
            ["SUBJID", "TRAJID"]
        )["INR_VALUE"].shift(i).fillna(df[f"INR_VALUE_PREV_{i - 1}"])

    return df, params


def extract_observed_decision(df):
    # Compute the dose change undertaken by the clinician
    df["WARFARIN_DOSE_MULT"] = df.groupby(
        ["TRIAL", "SUBJID", "TRAJID"]
    )["WARFARIN_DOSE"].shift(-1) / df["WARFARIN_DOSE"]

    # Actions
    cond = [
        df["WARFARIN_DOSE_MULT"] < 0.8,
        df["WARFARIN_DOSE_MULT"] <= 0.9,
        df["WARFARIN_DOSE_MULT"] < 1.,
        df["WARFARIN_DOSE_MULT"] == 1.,
        df["WARFARIN_DOSE_MULT"] < 1.1,
        df["WARFARIN_DOSE_MULT"] <= 1.2,
        df["WARFARIN_DOSE_MULT"] > 1.2
    ]
    action = pd.Categorical(
        np.select(cond, config.ACTION_LABELS),
        categories=config.ACTION_LABELS,
        ordered=True
    )
    action_code = action.codes.astype(float)
    action_code[action_code < 0] = np.nan

    return action_code


def compute_k(df):
    k = df.reset_index().groupby(
        ["TRIAL", "SUBJID", "TRAJID"]
    )["STUDY_DAY"].diff().shift(-1)
    k.name = "k"

    return k


def compute_reward(df):
    # Explode an empty dataframe to be indexed by every study day, including
    # those where we don't have observations.
    df = df[["INR_VALUE"] + config.ADV_EVENTS].copy()
    days = df.reset_index().groupby(
        ["TRIAL", "SUBJID", "TRAJID"]
    )["STUDY_DAY"].min().to_frame().rename(
        columns={"STUDY_DAY": "FIRST_DAY"}
    ).join(
        df.reset_index().groupby(
            ["TRIAL", "SUBJID", "TRAJID"]
        )["STUDY_DAY"].max()
    ).rename(columns={"STUDY_DAY": "LAST_DAY"})
    days["STUDY_DAY"] = [
        np.arange(start, end + 1).astype(int)
        for start, end in zip(days["FIRST_DAY"], days["LAST_DAY"])
    ]
    days = days.explode(column="STUDY_DAY")
    days = days.drop(columns=["FIRST_DAY", "LAST_DAY"])

    # Put the observed values into the empty exploded frame.
    df_interp = days.reset_index().set_index(
        ["TRIAL", "SUBJID", "TRAJID", "STUDY_DAY"]
    ).join(df)

    # Linearly interpolate INR. We don't need a groupby here because INRs are
    # always observed at the start and end of a trajectory.
    # TODO do we need a groupby?
    df_interp["INR_VALUE"] = df_interp["INR_VALUE"].interpolate()

    # Impute adverse events as having not occurred on days without an
    # observation
    df_interp[config.ADV_EVENTS] = df_interp[config.ADV_EVENTS].fillna(0)

    # Compute whether interpolated INR is in range.
    df_interp["IN_RANGE"] = ((df_interp["INR_VALUE"] >= 2) &
                             (df_interp["INR_VALUE"] <= 3))

    # Compute daily rewards
    is_event = (df_interp[config.ADV_EVENTS].sum(axis=1) > 0).fillna(0)
    df_interp["REWARD"] = (df_interp["IN_RANGE"] * config.INR_REWARD +
                           is_event * config.EVENT_REWARD)

    # Compute intermediate time index between visits
    df_interp.loc[df.index, "IS_OBSERVED"] = 1
    df_interp["IS_OBSERVED"] = df_interp["IS_OBSERVED"].fillna(0).astype(int)
    df_interp["NUM_INR_MEASURED"] = df_interp.groupby(
        ["TRIAL", "SUBJID", "TRAJID"]
    )["IS_OBSERVED"].cumsum()
    df_interp = df_interp.reset_index().set_index(
        ["TRIAL", "SUBJID", "TRAJID", "NUM_INR_MEASURED"]
    )
    df_interp["LAST_VISIT_DAY"] = df_interp.groupby(
        ["TRIAL", "SUBJID", "TRAJID", "NUM_INR_MEASURED"]
    )["STUDY_DAY"].min().rename("LAST_VISIT_DAY")
    df_interp = df_interp.reset_index().set_index(["TRIAL", "SUBJID", "TRAJID"])
    df_interp["t"] = df_interp["STUDY_DAY"] - df_interp["LAST_VISIT_DAY"] - 1

    # 
    import pdb; pdb.set_trace()


def compute_done(df):
    last_entries = df.groupby(["TRIAL", "SUBJID", "TRAJID"]).apply(
        pd.DataFrame.last_valid_index
    )
    df["LAST"] = 0
    df.loc[last_entries, "LAST"] = 1
    df["DONE"] = df.groupby(["TRIAL", "SUBJID", "TRAJID"])["LAST"].shift(-1)
    
    return df["DONE"]


def compute_sample_probability(df, relative_event_sample_probability):
    df["IS_EVENT"] = df[config.ADV_EVENTS].sum(axis=1)
    exp_event = df.groupby(["TRIAL", "SUBJID", "TRAJID"])["IS_EVENT"] > 0
    df["SAMPLE_PROB"] = 1.
    df.loc[exp_event == 1, "SAMPLE_PROB"] = relative_event_sample_probability

    return df["SAMPLE_PROB"]
