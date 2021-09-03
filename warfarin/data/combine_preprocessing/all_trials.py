import numpy as np

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
