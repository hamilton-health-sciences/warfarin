import numpy as np


def interpolate_inr(df):
    """
    Linearly interpolate INR values on unobserved study days.

    Args:
        df: A DataFrame indexed by TRIAL, SUBJID, TRAJID, and STUDY_DAY, and
            containing a column INR_VALUE.

    Returns:
        df_interp: A DataFrame exploded to be indexed TRIAL, SUBJID, TRAJID, and
                   STUDY_DAY where STUDY_DAY now can assume all possible
                   intermediate values as well. INR_VALUE will be linearly
                   interpolated on those intermediate study days.
    """
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

    # Linearly interpolate INR
    df_interp["INR_VALUE"] = df_interp.groupby(
        ["TRIAL", "SUBJID", "TRAJID"]
    )["INR_VALUE"].interpolate()

    return df_interp
