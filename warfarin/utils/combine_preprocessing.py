# coding: utf-8

"""Functions for pre-processing COMBINE-AF data."""

from copy import deepcopy

import time

import pandas as pd

import numpy as np

from warfarin import config
from warfarin.data.auditing import auditable


def decode(df):
    """
    Decode dataframes from bytes.

    :param df: any dataframe
    """
    str_df = df.select_dtypes([np.object])
    str_df = str_df.stack().str.decode("utf-8").unstack()
    for col in str_df:
        df[col] = [x.strip() if isinstance(x, str) else x for x in str_df[col]]
    return df


def bin_inr(df, num_bins=3, colname="INR_VALUE"):
    """
    Map INR values to an INR range (categorized into num_bins bins).

    :param df: random dataframe containing INR data
    :param num_bins: the number of INR range bins we want to create
    :param colname: the name of the column in df which contains the INR values
    :return: a pd Series of the mapped INR range bin labels
    """
    if num_bins == 3:
        cut_labels = ["<2", "[2,3]", ">2"]
        cut_bins = [0, 1.999, 3.00, 10]
    elif num_bins == 5:
        cut_labels = ["<=1.5", "(1.5, 2)", "[2, 3]", "(3, 3.5)", ">=3.5"]
        cut_bins = [0, 1.5, 1.999, 3, 3.499, 10]
    else:
        raise ValueError(f"Did not understand number of bins: {num_bins}")

    return pd.cut(df[colname], bins=cut_bins, labels=cut_labels)

def load_raw_data_sas(base_path):
    """
    Load raw SAS files from specified path.

    :param base_path: the file path to SAS files
    """
    t0 = time.time()

    print("Loading raw data...")

    # Baseline
    print("Reading baseline SAS file...")
    baseline = pd.read_sas(base_path + "baseline_raw.sas7bdat",
                           format="sas7bdat")
    print("\tDecoding to readable format and writing to feather file...")
    baseline = decode(baseline)
    baseline.to_feather(base_path + "baseline_raw.feather")

    print(f"\tLoaded baseline with shape: {baseline.shape}")

    # Events

    print("Reading events SAS file...")
    events = pd.read_sas(base_path + "events_raw.sas7bdat", format="sas7bdat")

    try:
        events = events.drop(["PROC_TYPE", "HOSP_REAS"], axis=1)
    except KeyError:
        pass

    print("\tDecoding to readable format and writing to feather file...")
    str_df = events.select_dtypes([np.object])
    str_df = str_df.stack().str.decode("utf-8").unstack()
    for col in str_df:
        events[col] = [x.strip()
                       if isinstance(x, str) else x for x in str_df[col]]

    events.to_feather(base_path + "events_raw.feather")

    print(f"\tLoaded events with shape: {events.shape}")

    print("Reading INR SAS file...")
    inr = pd.read_sas(base_path + "inr_raw.sas7bdat", format="sas7bdat")
    print("\tDecoding to readable format and writing to feather file...")
    inr = decode(inr)
    inr.to_feather(base_path + "inr_raw.feather")

    print(f"\tLoaded INR with shape: {inr.shape}")

    t1 = time.time()
    print(f"Done retrieving raw data. Took {(t1 - t0):,.2f} seconds \n")

    return inr, events, baseline


def load_raw_data(base_path):
    """
    Load raw SAS files from specified path.

    :param base_path: the file path to SAS files
    """
    t0 = time.time()

    print("Loading raw data...")

    # Baseline
    baseline = pd.read_feather(base_path + "baseline_raw.feather")
    print(f"\tLoaded baseline with shape: {baseline.shape}")

    # Events
    events = pd.read_feather(base_path + "events_raw.feather")
    print(f"\tLoaded events with shape: {events.shape}")

    # INR
    inr = pd.read_feather(base_path + "inr_raw.feather")
    print(f"\tLoaded INR with shape: {inr.shape}")

    t1 = time.time()
    print(f"DONE retrieving raw data. Took {(t1 - t0):,.2f} seconds \n")

    return inr, events, baseline


@auditable()
def merge_inr_events(inr, events):
    """
    Merge INR data with adverse events.

    :param inr: dataframe of INR data
    :param events: dataframe of adverse event data
    :return: merged dataframe containing both INR and adverse event data
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
    inr_merged = inr_merged.groupby(
        ["TRIAL", "SUBJID", "STUDY_DAY"]
    ).sum(min_count=1).reset_index()

    # Impute days with missing events data as not having an event
    for ev_name in config.EVENTS_TO_KEEP:
        inr_merged.loc[:, ev_name] = inr_merged[ev_name].fillna(0).astype(int)

    # Forward-fill trajectory ID to capture events in the correct trajectory.
    # The remaining NaN TRAJIDs are from events that occur before the first INR
    # measurement, so we drop these and convert the index back to int.
    # TODO validate this assumption in audit
    inr_merged["TRAJID"] = inr_merged.groupby(
        ["TRIAL", "SUBJID"]
    )["TRAJID"].fillna(method="ffill")
    inr_merged = inr_merged[~inr_merged["TRAJID"].isnull()].copy()
    inr_merged["TRAJID"] = inr_merged["TRAJID"].astype(int)

    import pdb; pdb.set_trace()

    print(
        "\tNum stroke occurrences: \n"
        f"{inr_merged.groupby('TRIAL')['STROKE'].value_counts()}"
    )
    print(
        "\tNum hem strokes: \n"
        f"{inr_merged.groupby('TRIAL')['HEM_STROKE'].value_counts()}"
    )

    # If there are multiple events that happen between INR measurements, only
    # take the first one
    temp = inr_merged
    temp.loc[:, "INR_MEASURED"] = np.where(temp["INR_TYPE"] == "Y", 1, 0)
    temp.loc[:, "CUMU_MEASUR"] = temp.iloc[::-1].groupby(
        "SUBJID"
    )["INR_MEASURED"].cumsum()[::-1]

    prev_inr_measured = temp.groupby("SUBJID")["INR_MEASURED"].shift().fillna(0)
    mask = np.logical_or(prev_inr_measured, temp["INR_MEASURED"])
    inr_merged = temp[mask].copy()
    print(
        "\tMasking events that do not occur after an INR measurement. This "
        f"removes: {temp.shape[0] - inr_merged.shape[0]} entries."
    )

    inr_merged.groupby(
        ["SUBJID", "CUMU_MEASUR"]
    ).size().reset_index().sort_values(by=0)
    inr_merged["SUBJID_NEW"] = inr_merged.groupby(
        "SUBJID"
    )["SUBJID_NEW"].fillna(method="ffill")
    inr_merged = inr_merged[~inr_merged["SUBJID_NEW"].isnull()]

    t1 = time.time()
    print(f"\tDone merging. Took {t1 - t0} seconds")

    return inr_merged


@auditable()
def split_traj_along_events(inr_merged):
    """
    Split trajectory along adverse events.

    Since the dynamics around adverse events are unexpected, these transitions
    are removed by ending the trajectory at the adverse event.

    :param inr_merged: dataframe containing INR and adverse events data
    :return: dataframe with new trajectory ID column "SUBJID_NEW_2"
    """
    prev_inr_measured = inr_merged.groupby(
        "SUBJID_NEW"
    )["INR_MEASURED"].shift().fillna(0)
    inr_merged["INTERRUPT"] = np.logical_and(
        prev_inr_measured,
        1 - inr_merged["INR_MEASURED"]
    )
    inr_merged = split_traj(inr_merged, id_col="SUBJID_NEW")
    inr_merged = inr_merged.rename(columns={"SUBJID_NEW_NEW": "SUBJID_NEW_2"})
    print(
        "\tDone splitting along adverse events. Went from "
        f"{inr_merged['SUBJID_NEW'].nunique()} trajectories to "
        f"{inr_merged['SUBJID_NEW_2'].nunique()} trajectories."
    )
    print("\tEvents in merged df:")
    print(inr_merged[config.EVENTS_TO_KEEP].sum())

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

    Demographic data is constant over time, so demographic features are
    forward-filled.

    :param inr_merged: dataframe containing clinical visits with INR data and
                       adverse events
    :return: dataframe with imputed values around missing doses and INR values
    """
    inr_merged["WARFARIN_DOSE"] = inr_merged.groupby(
        "SUBJID"
    )["WARFARIN_DOSE"].fillna(method="bfill")
    inr_merged["STUDY_DAY"] = inr_merged["TIMESTEP"]

    measured_inrs = inr_merged.groupby("SUBJID").fillna(method="ffill")
    measured_inrs["SUBJID"] = inr_merged["SUBJID"]

    null_entries = measured_inrs[measured_inrs["WARFARIN_DOSE"].isnull()]
    print(
        f"After imputing, there are still {null_entries.shape[0]} null entries,"
        f" from {null_entries['SUBJID'].nunique()} patients"
    )

    n0 = measured_inrs.shape[0]
    measured_inrs = measured_inrs[~measured_inrs["WARFARIN_DOSE"].isnull()]
    num_removed = n0 - measured_inrs.shape[0]
    print(f"Removed {num_removed:,.0f} entries with NaN Warfarin doses")

    print(
        f"There are {measured_inrs['SUBJID'].nunique()} patients, "
        f"{measured_inrs['SUBJID_NEW_2'].nunique()} trajectories"
    )
    print("\n")
    print(measured_inrs["TRIAL"].value_counts())
    print("\n")

    return measured_inrs


@auditable()
def split_traj_by_time_elapsed(measured_inrs):
    """
    Split patient trajectory into multiple trajectories if a certain amount of
    time elapses between visits.

    :param measured_inrs: dataframe containing patient INR data over time
    :return: dataframe containing new trajectory ID
    """
    id_col = "SUBJID_NEW_2"

    measured_inrs["STUDY_WEEK"] = measured_inrs["STUDY_DAY"] // 7
    measured_inrs.loc[:, "TIME_DIFF"] = measured_inrs.groupby(
        id_col
    )["STUDY_DAY"].diff().fillna(0)
    first_measures = measured_inrs.groupby(id_col).first().reset_index()
    first_measures["IS_FIRST"] = 1
    measured_inrs = measured_inrs.merge(
        first_measures[[id_col, "STUDY_DAY", "IS_FIRST"]],
        how="left",
        on=[id_col, "STUDY_DAY"]
    )
    measured_inrs.loc[measured_inrs["TIME_DIFF"] > config.MAX_TIME_ELAPSED,
                      "IS_FIRST"] = 1
    measured_inrs.loc[:, "IS_FIRST"] = measured_inrs["IS_FIRST"].fillna(0)

    new_id = "USUBJID_O_NEW"
    measured_inrs.loc[:, new_id] = measured_inrs.groupby(
        id_col
    )["IS_FIRST"].cumsum()
    measured_inrs.loc[:, new_id] = (measured_inrs[id_col].astype(str) +
                                    measured_inrs[new_id].astype(str))
    measured_inrs = measured_inrs.drop(columns=["IS_FIRST", "TIME_DIFF"])

    print(
        f"After splitting... \n\t{measured_inrs[new_id].nunique():,.0f} "
        f"trajectories, {measured_inrs.shape[0]:,.0f} samples"
    )

    return measured_inrs


@auditable()
def remove_short_traj(measured_inrs, id_col="USUBJID_O_NEW"):
    """
    Remove trajectories with fewer than MIN_INR_COUNTS of INR visits. 
    
    :param measured_inrs: dataframe containing INR data over time
    :param id_col: the column name of the trajectory ID. Defaults to
                   "USUBJID_O_NEW"
    :return: dataframe with short trajectories removed 
    """
    # Remove patients with fewer than min_inr_counts
    counts = measured_inrs.groupby(id_col).size()
    patient_ids = counts[counts >= config.MIN_INR_COUNTS].index.tolist()    
    num_removed = measured_inrs[id_col].nunique() - len(patient_ids)
    measured_inrs = measured_inrs[measured_inrs[id_col].isin(patient_ids)]

    num_traj = len(patient_ids)
    num_samples = measured_inrs.shape[0]
    num_patients = measured_inrs['SUBJID'].nunique()

    print(
        f"Removing {num_removed:,.0f} trajectories with fewer than "
        f"{config.MIN_INR_COUNTS} INR measurements.."
    )
    print(
        f"\tRemaining {num_patients:,.0f} patients, {num_traj:,.0f} "
        f"trajectories, {num_samples:,.0f} samples remaining..."
    )

    return measured_inrs


@auditable()
def remove_clinically_unintuitive(df):
    """
    Remove entries that are unlikely to occur in clinical practice.

    Unlikely clinical behaviour includes:
    1. Increasing the dosage drastically when the INR is high
    2. Decreasing the dosage drastically when the INR is low

    To remove these entries, these entries are masked, and the trajectory is
    split along these points, creating new trajectories.

    :param df: dataframe containing INR data
    :return: dataframe with new trajectory ID
    """
    print("Removing clinically unintuitive cases...")

    df_analyze = deepcopy(df)

    df_analyze.loc[df_analyze["WARFARIN_DOSE"] == 0,
                   "WARFARIN_DOSE_CHANGE"] = 100

    df_analyze["INR_BIN"] = bin_inr(df_analyze, 5)
    df_analyze["INR_VALUE_PREV"] = df_analyze.groupby(
        "USUBJID_O_NEW"
    )["INR_VALUE"].shift(1)
    df_analyze["INR_VALUE_NEXT"] = df_analyze.groupby(
        "USUBJID_O_NEW"
    )["INR_VALUE"].shift(-1)
    df_analyze["INR_VALUE_CHANGE"] = (
        df_analyze["INR_VALUE"] - df_analyze["INR_VALUE_PREV"]
    ) / df_analyze["INR_VALUE_PREV"]
    df_analyze["WARFARIN_DOSE_CHANGE_SIGN"] = np.where(
        ((df_analyze["WARFARIN_DOSE_CHANGE"] == 0) |
         (df_analyze["WARFARIN_DOSE_CHANGE"].isnull())),
        "0",
        np.where(df_analyze["WARFARIN_DOSE_CHANGE"] > 0, ">0", "<0")
    )

    df_analyze.loc[((df_analyze["INR_BIN"] == "<=1.5") &
                    (df_analyze["WARFARIN_DOSE_CHANGE_SIGN"] == "<0")),
                   "INR_VALUE"] = np.nan
    df_analyze.loc[((df_analyze["INR_BIN"] == ">=3.5") &
                    (df_analyze["WARFARIN_DOSE_CHANGE_SIGN"] == ">0")),
                   "INR_VALUE"] = np.nan

    # Split trajectories along INR_TYPE=NaN entries
    df_analyze["IS_NULL"] = df_analyze["INR_VALUE"].isnull()
    df_analyze["IS_NULL_CUMU"] = df_analyze.groupby(
        "USUBJID_O_NEW"
    )["IS_NULL"].cumsum()

    df_analyze["INTERRUPT"] = np.minimum(
        1, df_analyze["IS_NULL_CUMU"].diff() > 0
    )
    df_analyze = split_traj(df_analyze, id_col="USUBJID_O_NEW")

    print(
        "\tPrevious number of trajectories: "
        f"{df_analyze['USUBJID_O_NEW'].nunique()} | Mean traj length: "
        f"{df_analyze.groupby('USUBJID_O_NEW').size().mean():,.1f}, "
        "Median traj length: "
        f"{df_analyze.groupby('USUBJID_O_NEW').size().median()}"
    )

    print(
        "\tNew number of trajectories: "
        f"{df_analyze['USUBJID_O_NEW_NEW'].nunique()} | "
        "Mean traj length: "
        f"{df_analyze.groupby('USUBJID_O_NEW_NEW').size().mean():,.1f}, "
        "Median traj length: "
        f"{df_analyze.groupby('USUBJID_O_NEW_NEW').size().median()}"
    )

    df_analyze["USUBJID_O_NEW"] = df_analyze["USUBJID_O_NEW_NEW"]

    return df_analyze


@auditable()
def remove_phys_implausible(df, inr_buffer_range=0.25):
    """
    Remove entries that are unlikely to be seen, given our understanding of
    Warfarin behaviour.

    Unlikely Warfarin behaviour includes:
    1. Increase in INR after a decrease in Warfarin dose
    2. Decrease in INR after an increase in Warfarin dose

    Since there are other factors that influence INR levels, and since there is
    a lot of noise around INR data, these values are not necessarily invalid.
    However, these dynamics are unlikely to happen given our understanding of
    Warfarin behaviour. As such, these entries suggest that there may be other
    factors influencing the patient at that time (which are unobserved in the
    given dataset).

    To allow some room for randomness, there is a buffer range around the INR
    change to account for random fluctuations in the INR levels. For example,
    if the Warfarin dose increases, but the INR decreases by less than the
    inr_buffer_range, this entry is still valid.

    To remove these entries, these entries are masked, and the trajectory is
    split along these points, creating new trajectories.

    :param df: dataframe containing INR data over time
    :param inr_buffer_range: the buffer range around the INR change for the
                             entry to be considered valid
    :return: dataframe with new patient ID
    """
    print("Removing physologically implausible cases...")

    df_analyze = deepcopy(df)

    # df_analyze["INR_VALUE"] = df_analyze["INR_VALUE"].round(1)
    # df_analyze["WARFARIN_DOSE_CHANGE"] = (
    #     df_analyze.groupby("SUBJID")["WARFARIN_DOSE"].shift(-1) /
    #     (df_analyze["WARFARIN_DOSE"])
    # ) - 1
    df_analyze.loc[df_analyze["WARFARIN_DOSE"] == 0,
                   "WARFARIN_DOSE_CHANGE"] = 100

    df_analyze["INR_BIN"] = bin_inr(df_analyze, 5)
    df_analyze["INR_VALUE_PREV"] = df_analyze.groupby(
        "SUBJID"
    )["INR_VALUE"].shift(1)
    df_analyze["INR_VALUE_NEXT"] = df_analyze.groupby(
        "SUBJID"
    )["INR_VALUE"].shift(-1)
    df_analyze["INR_VALUE_CHANGE"] = (
        df_analyze["INR_VALUE"] - df_analyze["INR_VALUE_PREV"]
    ) / df_analyze["INR_VALUE_PREV"]
    df_analyze["WARFARIN_DOSE_CHANGE_SIGN"] = np.where(
        ((df_analyze["WARFARIN_DOSE_CHANGE"] == 0) |
         (df_analyze["WARFARIN_DOSE_CHANGE"].isnull())),
        "0",
        np.where(df_analyze["WARFARIN_DOSE_CHANGE"] > 0, ">0", "<0")
    )
    df_analyze["INR_VALUE_CHANGE_SIGN"] = np.where(
        ((np.abs(df_analyze["INR_VALUE_CHANGE"]) <= inr_buffer_range) |
         (df_analyze["INR_VALUE_CHANGE"].isnull())),
        "0",
        np.where(df_analyze["INR_VALUE_CHANGE"] > 0, ">0", "<0")
    )
    df_analyze["FLAG"] = np.where(
        df_analyze["WARFARIN_DOSE_CHANGE_SIGN"] != "0",
        np.where(
            df_analyze["INR_VALUE_CHANGE_SIGN"] != "0",
            (df_analyze["WARFARIN_DOSE_CHANGE_SIGN"] !=
             df_analyze["INR_VALUE_CHANGE_SIGN"]),
            0
        ),
        0
    )

    print(
        f"\tFlagged {df_analyze['FLAG'].sum():,.0f} "
        f"({df_analyze['FLAG'].sum() / len(df_analyze):,.2%}) transitions out "
        f"of {len(df_analyze):,.0f} original entries"
    )

    df_analyze.loc[df_analyze["FLAG"] == 1, "INR_VALUE"] = np.nan
    df_analyze["IS_NULL"] = df_analyze["INR_VALUE"].isnull()
    df_analyze["IS_NULL_CUMU"] = df_analyze.groupby(
        "SUBJID"
    )["IS_NULL"].cumsum()

    df_analyze["INTERRUPT"] = np.minimum(
        1, df_analyze["IS_NULL_CUMU"].diff() > 0
    )
    df_analyze = split_traj(df_analyze, id_col="USUBJID_O_NEW")

    print(
        f"\tPreviously had: {df_analyze['USUBJID_O_NEW'].nunique():,.0f} "
        "trajectories, now we have: "
        f"{df_analyze['USUBJID_O_NEW_NEW'].nunique():,.0f} trajectories"
    )
    print(
        "\tMean traj length: -------- "
        f"{df_analyze.groupby('USUBJID_O_NEW_NEW').size().mean():,.0f} "
        "entries"
    )
    print(
        "\tMedian traj length: ------ "
        f"{df_analyze.groupby('USUBJID_O_NEW_NEW').size().median():,.0f} "
        "entries"
    )

    df_analyze["USUBJID_O_NEW"] = df_analyze["USUBJID_O_NEW_NEW"]
    return df_analyze


def agg_weekly(inr_merged):
    """
    One of the previous versions of the setup aggregated results by week to
    remove the noise of more frequent clinical visits. However, this was
    deprecated because this removed some of the signal and led to lower
    performance.

    :param inr_merged: dataframe containing INR and events data
    :return: dataframe with aggregated INR and events data by week
    """
    inr_merged["NEXT_TIMESTEP"] = inr_merged.groupby(
        ["SUBJID", "STUDY_WEEK"]
    )["TIMESTEP"].shift(-1)
    inr_merged["LAST_DAY_OF_WEEK"] = (inr_merged["STUDY_WEEK"] + 1) * 7
    inr_merged["NEXT_TIMESTEP"] = inr_merged["NEXT_TIMESTEP"].fillna(
        inr_merged["LAST_DAY_OF_WEEK"]
    )
    inr_merged["NUM_TIMESTEPS_ON_DOSE"] = (inr_merged["NEXT_TIMESTEP"] -
                                           inr_merged["TIMESTEP"])

    # Define a lambda function to compute the weighted mean:
    # wm = lambda x: np.average(
    #     x, weights=inr_merged.loc[x.index, "NUM_TIMESTEPS_ON_DOSE"]
    # )

    inr_weekly = inr_merged.groupby(["USUBJID_O_NEW", "STUDY_WEEK"]).agg(
        {"SUBJID": "first",
         "TRIAL": "first",
         "WARFARIN_DOSE": "mean",
         "INR_VALUE": "mean",
         "INR_TYPE": "last",
         "HOSP": "sum",
         "DEATH": "sum",
         "MAJOR_BLEED": "sum",
         "MINOR_BLEED": "sum",
         "STROKE": "sum",
         "HEM_STROKE": "sum",
         "RANKIN_SCORE": "mean"}
    ).reset_index()

    print(f"Aggregated by week... \n\t{inr_weekly.shape[0]} entries")
    return inr_weekly


@auditable()
def merge_inr_base(inr_merged, baseline):
    """
    Merge INR data with baseline features.

    :param inr_merged: dataframe with INR and events data
    :param baseline: dataframe with patient baseline features
    :return:
    """
    merged_data = inr_merged.merge(baseline[config.STATIC_STATE_COLS + ["SUBJID"]],
                                   on="SUBJID", how="left")
    print(
        "Merged with baseline data... \n\t"
        f"{merged_data['SUBJID'].nunique():,.0f} patients, "
        f"{merged_data.shape[0]:,.0f} weekly entries"
    )
    print(f"\n{merged_data['TRIAL'].value_counts()} \n")
    print(f"\n{merged_data.groupby('TRIAL')['SUBJID'].nunique()} \n")

    return merged_data


@auditable()
def prepare_features(merged_data):
    """
    Create a few features and discretize continuous variables.

    These continuous variables were discretized based on quantiles, and slight
    adjustments were made based on what was seen in the literature.

    :param merged_data: dataframe containing INR, events, and baseline data
                        merged (preprocessed)
    :return: dataframe with discretized versions of the continuous variables.
    """
    print("\nDiscretizing some features...")

    # INR value forced between range
    merged_data.loc[:, "INR_VALUE"] = np.maximum(
        np.minimum(merged_data["INR_VALUE"], 4.5), 0.5
    )

    num_zero = merged_data[merged_data["WARFARIN_DOSE"] == 0].shape[0]
    print(f"\t{num_zero} ({(num_zero / merged_data.shape[0]):,.2%}) entries "
          "have Warfarin weekly dosage of 0mg")

    perc_in_range = merged_data[
        (merged_data["INR_VALUE"] >= 2) & (merged_data["INR_VALUE"] <= 3)
    ].shape[0] / merged_data.shape[0]
    print(f"\t{perc_in_range:,.2%} of entries are within range")

    merged_data.loc[:, "WARFARIN_DOSE_PREV"] = merged_data.groupby(
        "USUBJID_O_NEW"
    )["WARFARIN_DOSE"].shift(1)
    merged_data.loc[:, "WARFARIN_DOSE_MULT"] = merged_data.groupby(
        "USUBJID_O_NEW"
    )["WARFARIN_DOSE"].shift(-1) / merged_data["WARFARIN_DOSE"]

    # Use original patient ID for these flags even if it was split into
    # separate trajectories
    merged_data["MINOR_BLEED_FLAG"] = np.minimum(
        1.0, merged_data.groupby("SUBJID")["MINOR_BLEED"].cumsum()
    )
    merged_data["MAJOR_BLEED_FLAG"] = np.minimum(
        1.0, merged_data.groupby("SUBJID")["MAJOR_BLEED"].cumsum()
    )
    merged_data["HOSP_FLAG"] = np.minimum(
        1.0, merged_data.groupby("SUBJID")["HOSP"].cumsum()
    )
    merged_data["AGE_DEIDENTIFIED"] = merged_data["AGE_DEIDENTIFIED"].apply(
        lambda x: int(90) if x == ">89" else int(x))

    # Discretize continuous features
    merged_data["INR_VALUE_BIN"] = bin_inr(merged_data, num_bins=5)

    cut_bins = [-0.001, 5, 12.5, 17.5, 22.5, 27.5, 30, 32.5, 35, 45, 1000]
    cut_labels = ["<=5", "(5, 12.5]", "(12.5, 17.5]", "(17.5, 22.5]",
                  "(22.5, 27.5]", "(27.5, 30]", "(30, 32.5]", "(32.5, 35]",
                  "(35, 45]", ">45"]
    merged_data["WARFARIN_DOSE_BIN"] = pd.cut(merged_data["WARFARIN_DOSE"],
                                              bins=cut_bins,
                                              labels=cut_labels)

    cut_bins = [-0.001, 50, 60, 65, 70, 75, 80, 91]
    cut_labels = ["<=50", "(50, 60]", "(60, 65]", "(65, 70]", "(70, 75]",
                  "(75, 80]", ">80"]
    merged_data["AGE_BIN"] = pd.cut(merged_data["AGE_DEIDENTIFIED"],
                                    bins=cut_bins,
                                    labels=cut_labels)

    # weights = pd.DataFrame(
    #     new_data.baseline.groupby("SUBJID")["WEIGHT"].last()
    # ).reset_index()
    # if "WEIGHT" not in merged_data.columns:
    #     merged_data = merged_data.merge(weights, how="left", on="SUBJID")

    cut_bins = [-0.001, 55, 70, 80, 90, 100, 200]
    cut_labels = ["<=55", "(55, 70]", "(70, 80]", "(80, 90]", "(90, 100]",
                  ">100"]
    merged_data["WEIGHT_BIN"] = pd.cut(merged_data["WEIGHT"],
                                       bins=cut_bins,
                                       labels=cut_labels)

    return merged_data


def load_data(base_path, suffix):
    """
    Load preprocesesed data.

    :param base_path: path to where the dataframes are stored
    :param suffix: suffix of the dataset
    :return: INR, baseline, events, and merged data dataframes
    """
    inr = pd.read_feather(base_path + f"inr{suffix}.feather")
    baseline = pd.read_feather(base_path + f"baseline{suffix}.feather")
    events = pd.read_feather(base_path + f"events{suffix}.feather")
    merged_data = pd.read_feather(base_path + f"merged_data{suffix}.feather")

    return inr, baseline, events, merged_data


def split_data(inr_merged):
    """
    Split data into train, validation, and test data.

    Half of the ARISTOTLE data was held-out for the test data. The remainder
    was mixed between validation and train. Validation contained only ARISTOTLE
    and RElY data. Due to the way the data was stored across trials, there was
    more confidence in ARISTOTLE and RELY data than ENGAGE and ROCKET_AF for our
    problem.

    :param inr_merged: dataframe containing the preprocessed, merged data
    :return: train, validation, and test sets
    """
    print("----------------------------------------------")
    print("Creating test dataset from ARISTOTLE data...")
    aristotle_data = inr_merged[inr_merged["TRIAL"] == "ARISTOTLE"]
    test_ids, other_ids = split_data_ids(aristotle_data, split_perc=0.65)
    test_data = inr_merged[inr_merged["SUBJID"].isin(test_ids)].copy()

    other_patient_ids = inr_merged[
        inr_merged["TRIAL"] != "ARISTOTLE"
    ]["SUBJID"].unique()
    other_patient_ids = np.append(other_patient_ids, other_ids)

    print("----------------------------------------------")
    print("Creating valid dataset from ARISTOTLE and RELY data...")
    arist_rely_data = inr_merged[
        inr_merged["TRIAL"].isin(["ARISTOTLE", "RELY"]) &
        inr_merged["SUBJID"].isin(other_patient_ids)
    ]
    val_ids, other_ids = split_data_ids(arist_rely_data, split_perc=0.2)
    val_data = inr_merged[inr_merged["SUBJID"].isin(val_ids)].copy()

    train_sel = ~inr_merged["SUBJID"].isin(np.append(val_ids, test_ids))
    train_data = inr_merged[train_sel].copy()
    # train_data, val_data = ReplayBuffer.split_data(
    #     inr_merged[inr_merged["SUBJID"].isin(other_patient_ids)],
    #     split=[0.94, 0.06]
    # )

    num_test_patients = len(test_ids)
    num_train_patients = train_data["SUBJID"].nunique()
    num_val_patients = val_data["SUBJID"].nunique()
    num_total_patients = (num_test_patients + num_train_patients +
                          num_val_patients)

    num_test_samples = len(test_data)
    num_train_samples = len(train_data)
    num_val_samples = len(val_data)
    num_total_samples = num_test_samples + num_train_samples + num_val_samples

    print("----------------------------------------------")
    print(
        f"Total patients: {num_total_patients:,.0f}, "
        f"total samples: {num_total_samples:,.0f}"
    )
    print(
        f"\t Train patients: {num_train_patients} "
        f"({num_train_patients / num_total_patients:,.2%}), "
        f"total samples: {num_train_samples:,.0f} "
        f"({num_train_samples / num_total_samples:,.2%})"
    )
    print(
        f"\t Validation patients: {num_val_patients} "
        f"({num_val_patients / num_total_patients:,.2%}), "
        f"total samples: {num_val_samples:,.0f} "
        f"({num_val_samples / num_total_samples:,.2%})"
    )
    print(
        f"\t Test patients: {num_test_patients} "
        f"({num_test_patients / num_total_patients:,.2%}), "
        f"total samples: {num_test_samples:,.0f} "
        f"({num_test_samples / num_total_samples:,.2%})"
    )

    return train_data, val_data, test_data


def split_data_ids(data, split_perc, random_seed=42, id_col="SUBJID"):
    """
    Create two subgroups of IDs.

    This is used to split the IDs into two groups, based on the percentage
    split given. The percentage split is for the first group. For example,
    split_perc=0.3 suggests that the first group should contain 30% of the IDs.

    :param data: dataframe containing ID column
    :param split_perc: percentage of first group
    :param random_seed: seed for the np random shuffle to ensure reproducibility
    :param id_col: name of the ID column we want to subset
    :return: two groups of IDs
    """
    np.random.seed(random_seed)
    patient_ids = data[id_col].unique()
    np.random.shuffle(patient_ids)

    indx = int(split_perc * len(patient_ids))
    left_ids = patient_ids[:indx]
    right_ids = patient_ids[indx:]

    num_total = len(left_ids) + len(right_ids)
    num_left = len(left_ids)
    num_right = len(right_ids)

    print(
        f"\tFirst group: {num_left} patients ({num_left / num_total:,.2%}), "
        f"Second group: {num_right} patients ({num_right / num_total:,.2%})"
    )

    return left_ids, right_ids
