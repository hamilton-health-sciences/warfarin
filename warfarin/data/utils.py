"""Miscellaneous utilities for data processing."""

import numpy as np


def decode(df):
    """
    Decode dataframe string columns from bytes.

    Args:
        df: Any dataframe.

    Returns:
        df: The dataframe with bytestrings decoded to UTF-8 strings.
    """
    str_df = df.select_dtypes([np.object])
    for col in str_df:
        try:
            df[col] = str_df[col].str.decode("utf-8").str.strip()
        except:
            df[col] = str_df[col].str.decode("ISO-8859-1").str.strip()
    return df


def split_traj(df, reason):
    """
    Split trajectories using "INTERRUPT" flag in the input dataframe.

    Args:
        df: Any dataframe containing an "INTERRUPT" flag.
        reason: The reason for terminating this group of trajectories, which
                will be recorded in the "TERMINATION_CONDITION" column.

    Returns:
        df: Dataframe with trajectories split on "INTERRUPT", trajectories
            indexed by new column "TRAJID".
    """
    if "TRAJID" in df.columns:
        group_cols = ["SUBJID", "TRAJID"]
    else:
        group_cols = ["SUBJID"]

    # Identify trajectory endpoints
    df["REMOVE_PRIOR"] = df.groupby(group_cols)["INTERRUPT"].shift(1)
    df["REMOVE_AFTER"] = df.groupby(group_cols)["INTERRUPT"].shift(-1)
    new_traj_start_sel = np.logical_and(
        np.logical_or(df["REMOVE_PRIOR"].isnull(), df["REMOVE_PRIOR"] == 1),
        ~df["INTERRUPT"]
    )
    new_traj_end_sel = np.logical_and(
        np.logical_or(df["REMOVE_AFTER"].isnull(), df["REMOVE_AFTER"] == 1),
        ~df["INTERRUPT"]
    )
    if "TRAJID" in df.columns:
        # Ensure we respect the existing trajectory endpoints
        df["START_TRAJ"] = np.logical_or(
            new_traj_start_sel,
            df["TRAJID"] != df.groupby("SUBJID")["TRAJID"].shift(1)
        )
        df["END_TRAJ"] = np.logical_or(
            new_traj_end_sel,
            df["TRAJID"] != df.groupby("SUBJID")["TRAJID"].shift(-1)
        )
    else:
        df["START_TRAJ"] = new_traj_start_sel
        df["END_TRAJ"] = new_traj_end_sel

    # Always group by SUBJID even when TRAJID is set in order to respect
    # the existing endpoints.
    df["START_TRAJ_CUMU"] = df.groupby("SUBJID")["START_TRAJ"].cumsum()
    df["END_TRAJ_CUMU"] = df.groupby("SUBJID")["END_TRAJ"].cumsum()

    # Exclude transitions between endpoints
    df = df[(df["START_TRAJ_CUMU"] >= df["END_TRAJ_CUMU"]) &
            (~np.logical_and(df["START_TRAJ"], df["END_TRAJ"])) &
            (~df["INTERRUPT"])].copy()

    # Index the formed trajectories, starting from 0
    df["TRAJID"] = df["START_TRAJ_CUMU"]
    df["TRAJID"] = (df.groupby("SUBJID")["TRAJID"].diff().fillna(0) > 0)
    df["TRAJID"] = df.groupby("SUBJID")["TRAJID"].cumsum()

    # Code reasons. We assume that all trajectories have already been given
    # termination conditions prior to this splitting operation.
    if "INR_TYPE" in df.columns:
        df_ = df[(~df["INR_TYPE"].isnull()) & (df["INR_TYPE"] == "Y")]
    else:
        df_ = df.copy()
    last_day = df_.groupby(
        ["TRIAL", "SUBJID", "TRAJID"]
    )["STUDY_DAY"].max().to_frame().rename(
        {"STUDY_DAY": "LAST_DAY"}, axis=1
    )
    df = df.set_index(["TRIAL", "SUBJID", "TRAJID"]).join(
        last_day
    ).reset_index()
    traj_end = (df["STUDY_DAY"] == df["LAST_DAY"])
    # Some rows consist of all NAs. We could prune these earlier perhaps?
    # TODO - currently this is necessary to prevent >1 labelled termination
    # transition per trajectory.
    if "INR_TYPE" in df.columns:
        traj_end = traj_end & (~df["INR_TYPE"].isnull())
    no_traj_term_cond_set = df["TERMINATION_CONDITION"].isnull()
    df.loc[traj_end & no_traj_term_cond_set,
           "TERMINATION_CONDITION"] = reason

    # Remove intermediate columns
    df = df.drop(["REMOVE_PRIOR", "REMOVE_AFTER", "START_TRAJ", "END_TRAJ",
                  "START_TRAJ_CUMU", "END_TRAJ_CUMU", "LAST_DAY"],
                 axis=1)

    return df
