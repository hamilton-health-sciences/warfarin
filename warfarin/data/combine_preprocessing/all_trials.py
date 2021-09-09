"""Shared data processing steps that are used for all RCTs in COMBINE-AF."""

import numpy as np

import pandas as pd

from warfarin import config
from warfarin.data.auditing import auditable
from warfarin.data.utils import split_data_ids


@auditable("inr", "events", "baseline")
def preprocess_all(inr, events, baseline):
    """
    Preliminary preprocessing steps on INR, events, and baseline dataframes.

    Performs the following in order:
        1. Converts relevant indices to ints.
        2. Subsets to patients with at least one INR measurement, i.e. warfarin
           patients.
        3. One-hot encodes event occurences.
        4. Removes INRs taken before warfarin initiation as they are available
           only in one trial.
        5. Removes INR/dose entries from before the first INR measurement and
           after the last INR measurement.

    Args:
        inr: The raw INR dataframe.
        events: The raw events dataframe.
        baseline: The raw baseline dataframe.

    Returns:
        inr: The INR dataframe.
        events: The events dataframe.
        baseline: The baseline dataframe.
    """
    # Fix dtypes of useful indexing columns
    baseline["SUBJID"] = baseline["SUBJID"].astype(int)
    inr["SUBJID"] = inr["SUBJID"].astype(int)
    inr["STUDY_DAY"] = inr["STUDY_DAY"].astype(int)
    events["SUBJID"] = events["SUBJID"].astype(int)

    # Set indices to have columns appear first
    baseline = baseline.set_index(["TRIAL", "SUBJID"]).reset_index()
    inr = inr.set_index(["TRIAL", "SUBJID", "STUDY_DAY"]).reset_index()
    events = events.set_index(["TRIAL", "SUBJID"]).reset_index()

    # Only keep patient IDs that have INR values, removing non-warfarin patients
    inr_ids = inr["SUBJID"].unique().tolist()
    baseline_sel = baseline["SUBJID"].isin(inr_ids)
    baseline = baseline[baseline_sel].copy()
    events_sel = events["SUBJID"].isin(inr_ids)
    events = events[events_sel].copy()

    baseline = baseline.rename(columns={"REGION": "CONTINENT"})

    # Make age in baseline data an int
    baseline.loc[baseline["AGE_DEIDENTIFIED"] == ">89", "AGE_DEIDENTIFIED"] = 90
    baseline["AGE_DEIDENTIFIED"] = baseline["AGE_DEIDENTIFIED"].astype(int)

    # One-hot encode adverse events
    events.loc[:, "DEATH"] = (
        events["EVENT_NAME"] == "All Cause Death"
    ).astype(int)
    events.loc[:, "STROKE"] = (
        events["EVENT_NAME"] == "Ischemic Stroke"
    ).astype(int)
    events.loc[:, "HEM_STROKE"] = (
        events["EVENT_NAME"] == "Hemorrhagic Stroke"
    ).astype(int)
    events.loc[:, "MAJOR_BLEED"] = (
        events["EVENT_NAME"] == "Major Bleeding"
    ).astype(int)
    events.loc[:, "MINOR_BLEED"] = (
        events["EVENT_NAME"] == "Minor Bleeding"
    ).astype(int)
    events.loc[:, "HOSP"] = (
        events["EVENT_NAME"] == "Hospitalization"
    ).astype(int)
    events = events.drop("EVENT_NAME", axis=1)

    # Drop negative days, which appear in most ENGAGE patients. It seems that
    # these patients had INR measurements prior to beginning the study drug.
    # As this does not occur in the other trials, we remove them.
    inr = inr[inr["STUDY_DAY"] >= 0].copy()

    # Remove entries before first INR measurement and after last INR measurement
    first_days = inr[~inr["INR_VALUE"].isnull()].groupby(
        "SUBJID"
    )["STUDY_DAY"].min().to_frame().rename(columns={"STUDY_DAY": "FIRST_DAY"})
    last_days = inr[~inr["INR_VALUE"].isnull()].groupby(
        "SUBJID"
    )["STUDY_DAY"].max().to_frame().rename(columns={"STUDY_DAY": "LAST_DAY"})
    inr_df = inr.merge(
        first_days,
        on="SUBJID",
        how="left"
    ).merge(last_days, on="SUBJID", how="right")
    inr = inr_df[(inr_df["STUDY_DAY"] >= inr_df["FIRST_DAY"]) &
                 (inr_df["STUDY_DAY"] <= inr_df["LAST_DAY"])].copy()

    # Remove temporary columns
    inr = inr.drop(["FIRST_DAY", "LAST_DAY"], axis=1)

    return inr, events, baseline


@auditable()
def remove_outlying_doses(inr):
    """
    Remove patients who have outliers.

    The assumption is that these patients have data entry issues.
    """
    drop_ids = inr[
        inr["WARFARIN_DOSE"] >= config.DOSE_OUTLIER_THRESHOLD
    ]["SUBJID"].unique()
    inr = inr[~inr["SUBJID"].isin(drop_ids)].copy()

    return inr


@auditable()
def merge_inr_events(inr, events):
    """
    Merge INR data with events data.

    Args:
        inr: The dataframe of processed INR measurements.
        events: The dataframe of events.

    Returns:
        inr_merged: The dataframe of INR and events, merged.
    """
    # Subset to non-imputed INRs, i.e. those actually observed
    inr = inr[inr["INR_TYPE"] == "Y"].copy()

    # TODO Rankin score does not seem to be carrying over
    # Set Rankin score for ischemic strokes
    events["RANKIN_SCORE"] = np.where(
        events["RANKIN_SCORE"].isnull(),
        np.where(events["STROKE"] >= 1, 3, np.nan),
        events["RANKIN_SCORE"]
    )

    # Index events by study day
    events.loc[:, "STUDY_DAY"] = events["EVENT_T2"].astype(int)

    # Aggregate events by day
    events = events.groupby(["TRIAL", "SUBJID", "STUDY_DAY"]).agg(
        {"DEATH": "sum",
         "STROKE": "sum",
         "HEM_STROKE": "sum",
         "MAJOR_BLEED": "sum",
         "MINOR_BLEED": "sum",
         "HOSP": "sum",
         "RANKIN_SCORE": "max"}
    ).reset_index()
    events["ADV_EVENTS_SUM"] = events[config.EVENTS_TO_KEEP].sum(axis=1)
    events = events[events["ADV_EVENTS_SUM"] >= 1]

    # Subset events to patients with at least one INR measurement after
    # previous processing steps
    events = events[events["SUBJID"].isin(inr["SUBJID"])].copy()

    # Merge INR and events data
    inr_merged = pd.concat(
        [inr, events[["TRIAL", "SUBJID", "STUDY_DAY", "RANKIN_SCORE"] +
                     config.EVENTS_TO_KEEP]]
    )

    # Impute days with missing events data as not having an event
    for ev_name in config.EVENTS_TO_KEEP:
        inr_merged.loc[:, ev_name] = inr_merged[ev_name].fillna(0).astype(int)

    # Forward-fill trajectory ID to capture events in the correct trajectory.
    # The remaining NaN TRAJIDs are from events that occur before the first INR
    # measurement, so we drop these and convert the index back to int.
    inr_merged = inr_merged.sort_values(by=["TRIAL", "SUBJID", "STUDY_DAY"])
    inr_merged["TRAJID"] = inr_merged.groupby(
        ["TRIAL", "SUBJID"]
    )["TRAJID"].fillna(method="ffill")
    inr_merged = inr_merged[~inr_merged["TRAJID"].isnull()].copy()
    inr_merged["TRAJID"] = inr_merged["TRAJID"].astype(int)
    inr_merged = inr_merged.groupby(
        ["TRIAL", "SUBJID", "TRAJID", "STUDY_DAY"]
    ).sum(min_count=1).reset_index()

    # Remove events that occur over `config.EVENT_RANGE` days since the last
    # INR entry
    noninr_sel = pd.isnull(inr_merged["INR_VALUE"])
    event_sel = inr_merged[config.EVENTS_TO_KEEP].sum(axis=1) > 0
    out_of_range_sel = inr_merged["STUDY_DAY"].diff() > config.EVENT_RANGE
    inr_merged = inr_merged[~(noninr_sel & event_sel & out_of_range_sel)].copy()

    # If there are multiple events that happen between INR measurements, only
    # take the first one
    temp = inr_merged.copy()
    temp.loc[:, "INR_MEASURED"] = (~pd.isnull(temp["INR_VALUE"])).astype(int)
    temp.loc[:, "CUMU_MEASUR"] = temp.iloc[::-1].groupby(
        ["SUBJID", "TRAJID"]
    )["INR_MEASURED"].cumsum()[::-1]
    prev_inr_measured = temp.groupby("SUBJID")["INR_MEASURED"].shift().fillna(0)
    mask = np.logical_or(prev_inr_measured, temp["INR_MEASURED"])
    inr_merged = inr_merged[mask].copy()

    return inr_merged


@auditable()
def split_trajectories_at_events(inr_merged):
    """
    Split trajectories at adverse events.

    Specifically, the last entry of each trajectory is set to the entry
    containing the event indicator.

    Args:
        inr_merged: Dataframe containing INR and adverse events data.

    Returns:
        inr_merged: Dataframe with updated TRAJID column.
    """
    # Check whether entry is an event
    inr_merged = inr_merged.set_index(["TRIAL", "SUBJID", "TRAJID"])
    inr_merged["IS_EVENT"] = (
        inr_merged[config.EVENTS_TO_SPLIT].sum(axis=1) > 0
    )
    inr_merged["EVENT_TRAJ_IDX"] = inr_merged.groupby(
        ["TRIAL", "SUBJID"]
    )["IS_EVENT"].cumsum()

    # Construct a new ID by cumsumming whether the previous entry is an event
    # and adding it to the existing trajectory ID
    inr_merged = inr_merged.reset_index()
    inr_merged["TRAJID"] = (
        inr_merged.groupby(
            ["TRIAL", "SUBJID"]
        )["EVENT_TRAJ_IDX"].shift(1).fillna(0).astype(int) +
        inr_merged["TRAJID"]
    )

    # Re-index the new trajectory ID from 0 within each subject
    inr_merged["TRAJID"] = (
        inr_merged.groupby(
            ["TRIAL", "SUBJID"]
        )["TRAJID"].diff().fillna(0) > 0
    )
    inr_merged["TRAJID"] = inr_merged.groupby(
        ["TRIAL", "SUBJID"]
    )["TRAJID"].cumsum()

    # Remove extraneous columns
    inr_merged = inr_merged.drop(["IS_EVENT", "EVENT_TRAJ_IDX"], axis=1)

    return inr_merged


@auditable()
def impute_inr_and_dose(inr_merged):
    """
    Imputes values for INR and Warfarin dose on study days that are missing
    entries.

    There are some entries that do not have a Warfarin dose, in which case we
    backfill the dose. In particular, adverse events do not necessarily coincide
    with clinical visits and may not have a Warfarin dose or INR value
    associated with the event. Aside from adverse events, the INR should not be
    missing.

    Args:
        inr_merged: Dataframe containing INR and events data.

    Returns:
        inr_merged: INR and events data with imputed INRs and doses.
    """
    # Backfill the warfarin dose within a subject. Note that we do not group
    # by trajectory here, in case the warfarin dose is reported after an adverse
    # event that terminates a trajectory. We may want to stick some time gap
    # limit on this in the future, possibly by splitting trajectories based on
    # time elapsed (TODO).
    inr_merged["WARFARIN_DOSE"] = inr_merged.groupby(
        "SUBJID"
    )["WARFARIN_DOSE"].fillna(method="bfill")

    # There are remaining null INR values, but these are trajectories that only
    # contain an adverse event, with no corresponding INR measurement. This
    # action forward fills INR and doses when a previous INR or dose is
    # available in a given trajectory. We then drop remaining null entries.
    inr_merged = inr_merged.groupby(
        ["SUBJID", "TRAJID"]
    ).apply(lambda df: df.fillna(method="ffill"))
    inr_merged = inr_merged[~inr_merged["INR_VALUE"].isnull() &
                            ~inr_merged["WARFARIN_DOSE"].isnull()].copy()

    return inr_merged


@auditable()
def split_trajectories_at_gaps(measured_inrs):
    """
    Split patient trajectory into multiple trajectories if a certain amount of
    time elapses between visits.

    Args:
        measured_inrs: Dataframe containing merged INR and events data.

    Returns:
        measured_inrs: Dataframe containing the same data, but with some
                       trajectories split with new index TRAJID.
    """
    # Compute the time difference between adjacent visits in a trajectory
    measured_inrs.loc[:, "TIME_DIFF"] = measured_inrs.groupby(
        ["SUBJID", "TRAJID"]
    )["STUDY_DAY"].diff().fillna(0)

    # Label each entry (visit) with whether it should be the first entry in a
    # trajectory, or not
    first_measures = measured_inrs.groupby(
        ["SUBJID", "TRAJID"]
    ).first().reset_index()
    first_measures["IS_FIRST"] = 1
    measured_inrs = measured_inrs.merge(
        first_measures[["SUBJID", "TRAJID", "STUDY_DAY", "IS_FIRST"]],
        how="left",
        on=["SUBJID", "TRAJID", "STUDY_DAY"]
    )
    measured_inrs.loc[measured_inrs["TIME_DIFF"] > config.MAX_TIME_ELAPSED,
                      "IS_FIRST"] = 1
    measured_inrs.loc[:, "IS_FIRST"] = measured_inrs["IS_FIRST"].fillna(0)
    measured_inrs["IS_FIRST"] = measured_inrs["IS_FIRST"].astype(int)

    # Construct new trajectory ID
    measured_inrs["TRAJID"] = (
        measured_inrs["TRAJID"] +
        measured_inrs.groupby("SUBJID")["IS_FIRST"].cumsum() - 1
    )

    # Remove extraneous columns
    measured_inrs = measured_inrs.drop(columns=["IS_FIRST", "TIME_DIFF"])

    return measured_inrs


@auditable()
def merge_inr_baseline(inr_merged, baseline):
    """
    Merge INR data with baseline features.

    Args:
        inr_merged: The merged INR and events data.
        baseline: Dataframe of baseline information.

    Returns:
        merged_data: The INR, events, and baseline data merged.
    """
    baseline = baseline[config.STATIC_STATE_COLS + ["SUBJID"]].copy()

    # Convert categorical columns to categorical
    for colname in baseline.columns:
        if baseline[colname].dtype == object:
            if "Y" in baseline[colname].unique():
                baseline[colname] = baseline[colname].map({"Y": 1, "N": 0})
            else:
                baseline[colname] = pd.Categorical(baseline[colname])

    merged_data = inr_merged.merge(baseline, on="SUBJID", how="left")

    return merged_data


@auditable("train", "val", "test")
def split_data(inr_merged, test_ids):
    """
    Split data into train, validation, and test data.

    Args:
        inr_merged: The dataframe containing the preprocessed, merged data.

    Returns:
        train: The data used for model training.
        val: The data used for validation (hyperparameter tuning).
        test: The data used for final model testing.
    """
    aristotle_ids = inr_merged["SUBJID"][inr_merged["TRIAL"] == "ARISTOTLE"]
    other_ids = np.setdiff1d(aristotle_ids, test_ids)
    test_data = inr_merged[inr_merged["SUBJID"].isin(test_ids)].copy()

    other_patient_ids = inr_merged[
        inr_merged["TRIAL"] != "ARISTOTLE"
    ]["SUBJID"].unique()
    other_patient_ids = np.append(other_patient_ids, other_ids)

    arist_rely_data = inr_merged[
        inr_merged["TRIAL"].isin(["ARISTOTLE", "RELY"]) &
        inr_merged["SUBJID"].isin(other_patient_ids)
    ]
    val_ids, other_ids = split_data_ids(arist_rely_data, split_perc=0.2)
    val_data = inr_merged[inr_merged["SUBJID"].isin(val_ids)].copy()

    train_sel = ~inr_merged["SUBJID"].isin(np.append(val_ids, test_ids))
    train_data = inr_merged[train_sel].copy()

    return train_data, val_data, test_data


@auditable()
def remove_short_trajectories(measured_inrs, min_length=config.MIN_INR_COUNTS):
    """
    Remove trajectories with fewer than `min_length` entries.

    Args:
        measured_inrs: Dataframe containing merged INR, events, and baseline
                       data.

    Returns:
        measured_inrs: Dataframe with short trajectories removed.
    """
    # Remove trajectories with fewer than `min_length` entries
    measured_inrs = measured_inrs.set_index(["TRIAL", "SUBJID", "TRAJID"])
    measured_inrs["COUNT"] = measured_inrs.groupby(
        ["TRIAL", "SUBJID", "TRAJID"]
    ).size()
    measured_inrs = measured_inrs[
        measured_inrs["COUNT"] >= min_length
    ]
    measured_inrs = measured_inrs.drop(columns=["COUNT"])
    measured_inrs = measured_inrs.reset_index()

    # Re-index the new trajectory ID from 0 within each subject
    measured_inrs["TRAJID"] = (
        measured_inrs.groupby(
            ["TRIAL", "SUBJID"]
        )["TRAJID"].diff().fillna(0) > 0
    )
    measured_inrs["TRAJID"] = measured_inrs.groupby(
        ["TRIAL", "SUBJID"]
    )["TRAJID"].cumsum()

    return measured_inrs
