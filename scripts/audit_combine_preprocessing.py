import os

import pandas as pd

import numpy as np

from warfarin import config


def message(msg, indent=1):
    indent_str = "\t" * indent
    msg_indented = indent_str + str(msg).replace("\n", f"\n{indent_str}")
    print(msg_indented)


def audit_preprocess_all():
    baseline_fn = os.path.join(config.AUDIT_PATH,
                               "preprocess_all_baseline.feather")
    inr_fn = os.path.join(config.AUDIT_PATH, "preprocess_all_inr.feather")
    events_fn = os.path.join(config.AUDIT_PATH, "preprocess_all_events.feather")
    baseline = pd.read_feather(baseline_fn)
    inr = pd.read_feather(inr_fn)
    events = pd.read_feather(events_fn)

    message("Auditing results of `preprocess_all`...", 0)

    # Get number of patients in each frame
    num_patients_baseline = baseline["SUBJID"].nunique()
    num_patients_inr = inr["SUBJID"].nunique()
    num_patients_events = events["SUBJID"].nunique()
    message(f"Number of patients (baseline):\t {num_patients_baseline}")
    message(f"Number of patients (INR):\t {num_patients_inr}")
    message(f"Number of patients (events):\t {num_patients_events}")

    # Get number of entries in each frame
    num_entries_baseline = len(baseline)
    num_entries_inr = len(inr)
    num_entries_events = len(events)
    message(f"Number of entries (baseline):\t {num_entries_baseline}")
    message(f"Number of entries (INR):\t {num_entries_inr}")
    message(f"Number of entries (events):\t {num_entries_events}")

    # Trajectory length stats (in entries)
    traj_length = inr[inr["INR_TYPE"] == "Y"].groupby(
        ["TRIAL", "SUBJID"], as_index=False
    )["INR_VALUE"].count().groupby("TRIAL")["INR_VALUE"].describe()
    message("Trajectory length (# entries):")
    message(traj_length[["min", "25%", "50%", "mean", "75%", "max"]], 2)

    # Trajectory length stats (in days)
    last = inr[inr["INR_TYPE"] == "Y"].groupby(
        ["TRIAL", "SUBJID"]
    )["STUDY_DAY"].max()
    first = inr[inr["INR_TYPE"] == "Y"].groupby(
        ["TRIAL", "SUBJID"]
    )["STUDY_DAY"].min()
    traj_length_days = (last - first).reset_index().groupby(
        "TRIAL"
    )["STUDY_DAY"].describe()
    message("Trajectory length (days):")
    message(traj_length_days[["min", "25%", "50%", "mean", "75%", "max"]], 2)

    # Event statistics
    message("Event counts:")
    message(events[config.EVENTS_TO_KEEP].sum(), 2)
    message("Event rate (per patient):")
    message(events[config.EVENTS_TO_KEEP].sum() / events["SUBJID"].nunique(), 2)

    # Event statistics by trial
    message("Event counts by trial:")
    message(events.groupby("TRIAL")[config.EVENTS_TO_KEEP].sum(), 2)

    message("Event rate (per patient) by trial:")
    message(
        events.groupby("TRIAL")[config.EVENTS_TO_KEEP].sum() /
        np.asarray(events.groupby("TRIAL")["SUBJID"].nunique()).reshape(-1, 1),
        2
    )

    # Negative INR values
    message("Counts of negative doses (likely typos):")
    message(
        inr[(inr["WARFARIN_DOSE"] < 0)].groupby(
            ["TRIAL", "SUBJID"]
        )["WARFARIN_DOSE"].count(),
        2
    )

    # TODO figure out what to do when 1 or 2 -- missing data or assume 0
    # dose on that day? look at examples to figure this out?
    # In ENGAGE and ROCKET data, ensure we have three daily entries for each
    # following INR_TYPE = Y
    subset = inr[inr["TRIAL"].isin(["ENGAGE", "ROCKET_AF"])].copy()
    subset["WARFARIN_DOSE_NONNULL"] = ~subset["WARFARIN_DOSE"].isnull()
    subset["NUM_DAILY_DOSES"] = subset.groupby(
        "SUBJID", as_index=False
    )["WARFARIN_DOSE_NONNULL"].rolling(
        3, min_periods=1
    ).sum()["WARFARIN_DOSE_NONNULL"]
    subset["NUM_PREVIOUS_DAILY_DOSES"] = subset.groupby(
        "SUBJID", as_index=False
    )["NUM_DAILY_DOSES"].shift(1)["NUM_DAILY_DOSES"].fillna(0)
    subset = subset.drop(columns=["WARFARIN_DOSE_NONNULL", "NUM_DAILY_DOSES"])
    message("Number of daily entries before each `INR_TYPE = Y` in ENGAGE "
            "and ROCKET-AF:")
    message(
        subset[subset["INR_TYPE"] == "Y"].groupby(
            "TRIAL"
        )["NUM_PREVIOUS_DAILY_DOSES"].value_counts(),
        2
    )


def audit_preprocess_trial_specific(trial_names):
    df_path = os.path.join(config.AUDIT_PATH,
                           f"preprocess_{trial_names}.feather")
    inr = pd.read_feather(df_path)

    message(f"Auditing `preprocess_{trial_names}`...", 0)

    # Number of patients
    message("Number of patients after splitting on interruptions:")
    message(inr.groupby("TRIAL")["SUBJID"].nunique(), 2)

    # Number of trajectories
    message("Number of trajectories after splitting on interruptions:")
    message(
        inr.groupby(
            ["TRIAL", "SUBJID", "TRAJID"], as_index=False
        ).size().groupby("TRIAL")["TRAJID"].count(),
        2
    )

    # Trajectory length stats (in entries)
    traj_length = inr[inr["INR_TYPE"] == "Y"].groupby(
        ["TRIAL", "SUBJID"], as_index=False
    )["INR_VALUE"].count().groupby("TRIAL")["INR_VALUE"].describe()
    message("Trajectory length after splitting on interruptions (# entries):")
    message(traj_length[["min", "25%", "50%", "mean", "75%", "max"]], 2)

    # Trajectory length stats (in days)
    last = inr[inr["INR_TYPE"] == "Y"].groupby(
        ["TRIAL", "SUBJID"]
    )["STUDY_DAY"].max()
    first = inr[inr["INR_TYPE"] == "Y"].groupby(
        ["TRIAL", "SUBJID"]
    )["STUDY_DAY"].min()
    traj_length_days = (last - first).reset_index().groupby(
        "TRIAL"
    )["STUDY_DAY"].describe()
    message("Trajectory length after splitting on interruptions (days):")
    message(traj_length_days[["min", "25%", "50%", "mean", "75%", "max"]], 2)

    # Number of non-null INRs
    message("Number of observed INRs:")
    message(
        inr[inr["INR_TYPE"] == "Y"].groupby(
            ["TRIAL", "SUBJID"], as_index=False
        )["INR_VALUE"].count().groupby("TRIAL")["INR_VALUE"].mean(),
        2
    )

    # Number of non-null doses
    message("Number of observed warfarin doses:")
    message(
        inr[inr["INR_TYPE"] == "Y"].groupby(
            ["TRIAL", "SUBJID"], as_index=False
        )["WARFARIN_DOSE"].count().groupby("TRIAL")["WARFARIN_DOSE"].mean(),
        2
    )

    # Number of zero INR values
    message("Number of zero INR values (impossible):")
    message(
        inr[(inr["INR_VALUE"] == 0) & (inr["INR_TYPE"] == "Y")].groupby(
            "TRIAL"
        )["INR_VALUE"].count(),
        2
    )

    # Number of zero warfarin doses
    message("Number of zero warfarin doses:")
    message(
        inr[(inr["WARFARIN_DOSE"] == 0) & (inr["INR_TYPE"] == "Y")].groupby(
            "TRIAL"
        )["WARFARIN_DOSE"].count(),
        2
    )


def audit_remove_outlying_doses():
    df_path = os.path.join(config.AUDIT_PATH, "remove_outlying_doses.feather")
    df = pd.read_feather(df_path)

    message("Warfarin dose summary stats:")
    message(df["WARFARIN_DOSE"].describe(), 2)


def audit_merge_inr_events():
    df_path = os.path.join(config.AUDIT_PATH, "merge_inr_events.feather")
    df = pd.read_feather(df_path)

    message("Maximum number of times a study day is recorded for a patient "
            "(should be 1):")
    message(
        df.groupby(["TRIAL", "SUBJID", "STUDY_DAY"])["TRAJID"].count(), 2
    )

    message("Number of event occurrences by trial:")
    message(df.groupby("TRIAL")[config.EVENTS_TO_KEEP].sum(), 2)

    message("Rate of events by trial (per patient):")
    message(
        df.groupby("TRIAL")[config.EVENTS_TO_KEEP].sum() /
        np.asarray(df.groupby("TRIAL")["SUBJID"].nunique()).reshape(-1, 1),
        2
    )


def audit_split_trajectories_at_events():
    df_path = os.path.join(config.AUDIT_PATH,
                           "split_trajectories_at_events.feather")
    df = pd.read_feather(df_path)

    import pdb; pdb.set_trace()


def main():
    audit_preprocess_all()
    for trial_names in ["engage_rocket", "rely", "aristotle"]:
        audit_preprocess_trial_specific(trial_names)
    audit_remove_outlying_doses()
    audit_merge_inr_events()


if __name__ == "__main__":
    main()
