"""Functions for pre-processing COMBINE-AF data."""

from copy import deepcopy

import numpy as np

from warfarin.data.auditing import auditable
from warfarin.data.utils import split_traj


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
