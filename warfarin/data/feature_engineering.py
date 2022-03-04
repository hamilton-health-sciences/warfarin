"""Transforms processed data into a format suitable for RL modeling."""

import pandas as pd

import numpy as np

from sklearn.preprocessing import (OneHotEncoder, MinMaxScaler,
                                   QuantileTransformer)

from warfarin import config
from warfarin.utils import interpolate_inr, code_quantitative_decision


def engineer_state_features(df,
                            time_varying_cross,
                            include_duration_time_varying,
                            include_dose_time_varying,
                            transforms=None):
    """
    Construct the state vectors.

    Use sklearn transforms to one-hot encode certain categorical variables, and
    transform continuous variables into the range [0, 1].

    Args:
        df: Merged longitudinal data, indexed by "TRIAL", "SUBJID", "TRAJID",
            and "STUDY_DAY".
        time_varying_cross: Whether to cross trajectory boundaries with time-
                            varying state features (like INR). If True, an
                            indicator is added for each previous INR which is 1
                            if an interruption has occurred since that INR.
        include_duration_time_varying: Whether to include the duration on dose
                                       in the time varying state features.
        include_dose_time_varying: Whether to include the warfarin dose in the
                                   time varying state features.
        transforms: Transform objects. If left to None, the parameters of the
                    transforms will be learned from the data (meant to be
                    applied to training data).

    Returns:
        df: The engineered state features.
        transforms: The transform objects.
    """
    if transforms is None:
        transforms = {}

    # Transform INR to the uniform distribution after rounding to nearest tenth.
    # The rounding takes are of any hanging decimals. As the output is bounded
    # in [0, 1], the later MinMax transform is a noop for this column.
    df["INR_VALUE"] = df["INR_VALUE"].round(1)
    if "inr_qn" in transforms:
        inr_qn = transforms["inr_qn"]
    else:
        inr_qn = QuantileTransformer(output_distribution="uniform",
                                     n_quantiles=df["INR_VALUE"].nunique())
        inr_qn.fit(df[["INR_VALUE"]])
        transforms["inr_qn"] = inr_qn
    df["INR_VALUE"] = inr_qn.transform(df[["INR_VALUE"]])

    # Dose durations. See comment above, as strategy is identical to INR except
    # we do not need to round.
    df["DURATION"] = df.index.get_level_values("STUDY_DAY")
    df["DURATION"] = df.groupby(["TRIAL", "SUBJID"])["DURATION"].diff().fillna(
        df["DURATION"]
    )
    if "duration_qn" in transforms:
        duration_qn = transforms["duration_qn"]
    else:
        duration_qn = QuantileTransformer(output_distribution="uniform",
                                          n_quantiles=df["DURATION"].nunique())
        duration_qn.fit(df[["DURATION"]])
        transforms["duration_qn"] = duration_qn
    df["DURATION"] = duration_qn.transform(df[["DURATION"]])

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
    if "categorical" in transforms:
        encoder = transforms["categorical"]
    else:
        encoder = OneHotEncoder(drop="first")
        encoder.fit(df_cat)
        transforms["categorical"] = encoder
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
    if "continuous" in transforms:
        encoder = transforms["continuous"]
    else:
        encoder = MinMaxScaler()
        encoder.fit(df_cts)
        transforms["continuous"] = encoder
    df_cts_scaled = pd.DataFrame(
        encoder.transform(df_cts),
        index=df_cts.index,
        columns=df_cts.columns
    )

    # Join as state cols
    df = df_cat_ohe.join(df_cts_scaled)

    # Provide previous INRs (and optionally doses, durations) in state
    time_varying = ["INR_VALUE"]
    if include_dose_time_varying:
        time_varying += ["WARFARIN_DOSE"]
    if include_duration_time_varying:
        time_varying += ["DURATION"]
    if time_varying_cross:
        group_vars = ["SUBJID"]
    else:
        group_vars = ["SUBJID", "TRAJID"]

    for varname in time_varying:
        df[f"{varname}_PREV_1"] = df.groupby(
            group_vars
        )[varname].shift(1).fillna(df[varname])
        for i in range(2, 5):
            df[f"{varname}_PREV_{i}"] = df.groupby(
                group_vars
            )[varname].shift(i).fillna(df[f"{varname}_PREV_{i - 1}"])

    # Add time-varying "cross-trajectory" indicator to each previous INR
    if time_varying_cross:
        for i in range(1, 5):
            df["TRAJNUM"] = df.index.get_level_values("TRAJID")
            df[f"PREV_{i}_CROSSED"] = (
                df.groupby(
                    "SUBJID"
                )["TRAJNUM"].shift(i).fillna(0) != df["TRAJNUM"]
            ).astype(int).fillna(0)
            df = df.drop("TRAJNUM", axis=1)

    return df, transforms


def extract_observed_decision(df):
    """
    Extract the observed clinician option.

    Args:
        df: The merged longitudinal data.

    Returns:
        option: The observed dose adjustment options in the data as a pd.Series.
    """
    # Compute the dose change undertaken by the clinician
    dose = df.groupby(["TRIAL", "SUBJID", "TRAJID"])["WARFARIN_DOSE"].shift(-1)
    prev_dose = df["WARFARIN_DOSE"]
    df["WARFARIN_DOSE_MULT"] = dose / prev_dose

    # Ensure that when both the current and previous dose is zero, we record it
    # as no change.
    sel_consecutive_zero = (dose == 0.) & (prev_dose == 0.)
    df.loc[sel_consecutive_zero, "WARFARIN_DOSE_MULT"] = 1.

    # Actions
    action_code = code_quantitative_decision(df["WARFARIN_DOSE_MULT"])
    df["OPTION"] = action_code
    option = df["OPTION"]

    return option


def compute_k(df):
    """
    Compute the number of days elapsed between the current and next visit.

    Args:
        df: The merged longitudinal data.

    Returns:
        k: The number of days elapsed as a pd.Series.
    """
    df = df.reset_index()
    df["k"] = df.groupby(["TRIAL", "SUBJID", "TRAJID"])["STUDY_DAY"].diff()
    df["k"] = df.groupby(["TRIAL", "SUBJID", "TRAJID"])["k"].shift(-1)
    df = df.set_index(["TRIAL", "SUBJID", "TRAJID", "STUDY_DAY"])
    k = df["k"]

    return k


def compute_reward(df,
                   discount_factor,
                   inr_reward,
                   event_reward):
    """
    Compute the reward associated with each SMDP transition.

    Args:
        df: The merged longitudinal data.
        discount_factor: The discount factor.
        inr_reward: The reward for in-range INRs.
        event_reward: The reward for adverse events.

    Returns:
        reward: The cumulative discounted reward.
    """
    # Explode an empty dataframe to be indexed by every study day, including
    # those where we don't have observations.
    df = df[["INR_VALUE"] + config.ADV_EVENTS].copy()
    df_interp = interpolate_inr(df)

    # Impute adverse events as having not occurred on days without an
    # observation
    df_interp[config.ADV_EVENTS] = df_interp[config.ADV_EVENTS].fillna(0)

    # Compute whether interpolated INR is in range.
    df_interp["IN_RANGE"] = ((df_interp["INR_VALUE"] >= 2) &
                             (df_interp["INR_VALUE"] <= 3))

    # Compute daily rewards
    is_event = (df_interp[config.ADV_EVENTS].sum(axis=1) > 0).fillna(0)
    df_interp["REWARD"] = (df_interp["IN_RANGE"] * inr_reward +
                           is_event * event_reward)

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
    df_interp["LAST_VISIT_DAY"] = df_interp.groupby(
        ["TRIAL", "SUBJID", "TRAJID"]
    )["LAST_VISIT_DAY"].shift(1)
    df_interp = df_interp.reset_index().set_index(["TRIAL", "SUBJID", "TRAJID"])
    df_interp["t"] = df_interp["STUDY_DAY"] - df_interp["LAST_VISIT_DAY"] - 1
    df_interp["LAST_VISIT_DAY"] = df_interp[
        "LAST_VISIT_DAY"
    ].fillna(-1).astype(int)

    # Compute reward
    df_interp["DISC_REWARD"] = (
        df_interp["REWARD"] * discount_factor**df_interp["t"]
    )
    cum_disc_reward = df_interp.groupby(
        ["TRIAL", "SUBJID", "TRAJID", "LAST_VISIT_DAY"]
    )["DISC_REWARD"].sum().reset_index().rename(
        columns={"LAST_VISIT_DAY": "STUDY_DAY",
                 "DISC_REWARD": "CUM_DISC_REWARD"}
    ).set_index(["TRIAL", "SUBJID", "TRAJID", "STUDY_DAY"])

    df = df.join(cum_disc_reward)
    reward = df["CUM_DISC_REWARD"]

    return reward


def compute_done(df):
    """
    Compute whether the transition is the last in a given trajectory.

    Args:
        df: The merged longitudinal data.

    Returns:
        done: Whether or not this is the last transition in the trajectory.
    """
    df["HAS_INR_AND_DOSE"] = (~pd.isnull(df["INR_VALUE"]) &
                              ~pd.isnull(df["WARFARIN_DOSE"]))
    last_entries = df[df["HAS_INR_AND_DOSE"]].groupby(
        ["TRIAL", "SUBJID", "TRAJID"]
    ).apply(pd.DataFrame.last_valid_index)
    df["LAST"] = 0
    df.loc[last_entries, "LAST"] = 1
    df["DONE"] = df.groupby(["TRIAL", "SUBJID", "TRAJID"])["LAST"].shift(-1)

    done = df["DONE"]

    return done


def compute_sample_probability(df, option, relative_event_sample_probability,
                               weight_option_frequency):
    """
    Compute the probability of sampling a given transition.

    If `weight_option_frequency` is set to `True`, transition sampling
    probabilities are set to be inversely proportional to the frequency of
    occurrence of the coresponding option in `option`.

    Otherwise, we assign a sample probability of 1 to all transitions. If the
    transition is a part of a trajectory which experiences an adverse event as
    defined in `config.ADV_EVENTS`, we assign a relative sampling probability of
    `relative_event_sample_probability`.

    Args:
        df: The merged longitudinal data.
        option: The observed option taken at the timestep.
        relative_event_sample_probability: The probability of sampling a
                                           transition in an event trajectory
                                           (relative to 1).
        weight_option_frequency: If True, relative_event_sample_probability will
                                 be ignored and sample probabilities will be
                                 inversely proportional to the frequency of
                                 occurence in `df`.

    Returns:
        sample_prob: The probability of sampling a given transition.
    """
    if weight_option_frequency:
        sample_rel_prob = option.value_counts().max() / option.value_counts()
        sample_prob = sample_rel_prob.to_dict()
        df["SAMPLE_PROB"] = option.map(sample_prob)
    else:
        df["IS_EVENT"] = df[config.ADV_EVENTS].sum(axis=1)
        exp_event = df.groupby(["TRIAL", "SUBJID", "TRAJID"])["IS_EVENT"].sum() > 0
        df["SAMPLE_PROB"] = 1.
        df.loc[exp_event, "SAMPLE_PROB"] = relative_event_sample_probability

    return df["SAMPLE_PROB"]
