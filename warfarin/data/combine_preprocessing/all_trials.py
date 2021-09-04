import numpy as np

import pandas as pd

from warfarin import config
from warfarin.data.auditing import auditable


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
    # Fix dtypes of columns
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

    # TODO one-hot encode somewhere else?
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
def merge_inr_events(inr, events):
    """
    Merge INR data with events data.

    Args:
        inr: The dataframe of processed INR measurements.
        events: The dataframe of events.

    Returns:
        inr_merged: The dataframe of INR and events, merged.
    """
    # TODO move to separate function
    # Remove patients who have outliers. The assumption is that these patients
    # have data entry issues.
    drop_ids = inr[
        inr["WARFARIN_DOSE"] >= config.DOSE_OUTLIER_THRESHOLD
    ]["SUBJID"].unique()
    print(
        f"\tDropping {len(drop_ids)} patients with outlier doses exceeding "
        f"{config.DOSE_OUTLIER_THRESHOLD}mg weekly..."
    )
    inr = inr[~inr["SUBJID"].isin(drop_ids)]  # ["WARFARIN_DOSE"].max()

    # Subset to non-imputed INRs, i.e. those actually observed
    inr = inr[inr["INR_TYPE"] == "Y"]

    # TODO we need a more complex imputation strategy here if we're using this
    # variable.
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

    # TODO what is this
    # inr = inr.groupby(["SUBJID", "TRAJID", "STUDY_DAY"]).last().reset_index()

    # Merge INR and events data
    inr_merged = pd.concat(
        [inr, events[["TRIAL", "SUBJID", "STUDY_DAY", "RANKIN_SCORE"] +
                     config.EVENTS_TO_KEEP]]
    )
    # inr_merged = inr_merged.groupby(
    #     ["TRIAL", "SUBJID", "STUDY_DAY"]
    # ).sum(min_count=1).reset_index()

    # Impute days with missing events data as not having an event
    for ev_name in config.EVENTS_TO_KEEP:
        inr_merged.loc[:, ev_name] = inr_merged[ev_name].fillna(0).astype(int)

    # Forward-fill trajectory ID to capture events in the correct trajectory.
    # The remaining NaN TRAJIDs are from events that occur before the first INR
    # measurement, so we drop these and convert the index back to int.
    # TODO validate this assumption in audit
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

    # TODO move to auditing
    # print(
    #     "\tNum stroke occurrences: \n"
    #     f"{inr_merged.groupby('TRIAL')['STROKE'].value_counts()}"
    # )
    # print(
    #     "\tNum hem strokes: \n"
    #     f"{inr_merged.groupby('TRIAL')['HEM_STROKE'].value_counts()}"
    # )

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

    # TODO move to auditing
    # print(
    #     "\tMasking events that do not occur after an INR measurement. This "
    #     f"removes: {temp.shape[0] - inr_merged.shape[0]} entries."
    # )

    # TODO not sure what this is, but I think it's strictly related to re-
    # indexing so can be skipped
    # inr_merged.groupby(
    #     ["SUBJID", "CUMU_MEASUR"]
    # ).size().reset_index().sort_values(by=0)
    # inr_merged["SUBJID_NEW"] = inr_merged.groupby(
    #     "SUBJID"
    # )["SUBJID_NEW"].fillna(method="ffill")
    # inr_merged = inr_merged[~inr_merged["SUBJID_NEW"].isnull()]

    # t1 = time.time()
    # print(f"\tDone merging. Took {t1 - t0} seconds")

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
    # TODO should minor bleeding be considered an event? See e.g. patient
    # with ID 63553
    # Check whether entry is an event
    inr_merged = inr_merged.set_index(["TRIAL", "SUBJID", "TRAJID"])
    # inr_merged = inr_merged.sort_values(by="STUDY_DAY")
    inr_merged["IS_EVENT"] = (inr_merged[config.EVENTS_TO_KEEP].sum(axis=1) > 0)
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

    # TODO remove trajectories now too short (see e.g. ID 32214 where event
    # occurs in first entry...)

    # TODO move to auditing
    # print(
    #     "\tDone splitting along adverse events. Went from "
    #     f"{inr_merged['SUBJID_NEW'].nunique()} trajectories to "
    #     f"{inr_merged['SUBJID_NEW_2'].nunique()} trajectories."
    # )
    # print("\tEvents in merged df:")
    # print(inr_merged[config.EVENTS_TO_KEEP].sum())

    return inr_merged
