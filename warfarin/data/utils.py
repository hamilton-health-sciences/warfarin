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


def split_traj(df):
    """
    Split trajectories using "INTERRUPT" flag in the input dataframe.

    Args:
        df: Any dataframe containing an "INTERRUPT" flag.

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

    # Remove intermediate columns
    df = df.drop(["REMOVE_PRIOR", "REMOVE_AFTER", "START_TRAJ", "END_TRAJ",
                  "START_TRAJ_CUMU", "END_TRAJ_CUMU"],
                 axis=1)

    return df


def split_data_ids(data, split_perc, random_seed=42):
    """
    Create two subgroups of IDs.

    This is used to split the IDs into two groups, based on the percentage
    split given. The percentage split is for the first group. For example,
    split_perc=0.3 suggests that the first group should contain 30% of the IDs.

    Args:
        data: Dataframe containing column "SUBJID".
        split_perc: Percentage of first group.
        random_seed: Seed for the random shuffle to ensure reproducibility.

    Returns:
        left_ids: IDs in the first group.
        right_ids: IDs in the second group.
    """
    np.random.seed(random_seed)
    patient_ids = data["SUBJID"].unique()
    np.random.shuffle(patient_ids)

    indx = int(split_perc * len(patient_ids))
    left_ids = patient_ids[:indx]
    right_ids = patient_ids[indx:]

    # num_total = len(left_ids) + len(right_ids)
    # num_left = len(left_ids)
    # num_right = len(right_ids)

    # print(
    #     f"\tFirst group: {num_left} patients ({num_left / num_total:,.2%}), "
    #     f"Second group: {num_right} patients ({num_right / num_total:,.2%})"
    # )

    return left_ids, right_ids
