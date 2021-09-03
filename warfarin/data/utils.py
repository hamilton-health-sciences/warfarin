import numpy as np


def split_traj(df):
    """
    Split trajectories using "INTERRUPT" flag in the input dataframe.

    Args:
        df: Any dataframe containing an "INTERRUPT" flag.

    Returns:
        df: Dataframe with trajectories split on "INTERRUPT", trajectories
            indexed by new column "TRAJID".
    """
    # Identify trajectory endpoints
    df["REMOVE_PRIOR"] = df.groupby("SUBJID")["INTERRUPT"].shift(1)
    df["REMOVE_AFTER"] = df.groupby("SUBJID")["INTERRUPT"].shift(-1)
    df["START_TRAJ"] = np.logical_and(
        np.logical_or(df["REMOVE_PRIOR"].isnull(), df["REMOVE_PRIOR"] == 1),
        ~df["INTERRUPT"]
    )
    df["END_TRAJ"] = np.logical_and(
        np.logical_or(df["REMOVE_AFTER"].isnull(), df["REMOVE_AFTER"] == 1),
        ~df["INTERRUPT"]
    )

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

