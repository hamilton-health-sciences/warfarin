import numpy as np


def interpolate_inr(df):
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

    # Linearly interpolate INR. We don't need a groupby here because INRs are
    # always observed at the start and end of a trajectory.
    # TODO do we need a groupby?
    df_interp["INR_VALUE"] = df_interp["INR_VALUE"].interpolate()

    return df_interp



