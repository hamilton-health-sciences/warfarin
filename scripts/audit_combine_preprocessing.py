import os

import pandas as pd

import numpy as np

from plotnine import *

from warfarin import config


def main():
    """
    Audit the results of the preprocessing pipeline.

    Preprocessing steps decorated with `@auditable` output their intermediate
    dataframes to `data/auditing`. Auditing is then accomplished by presenting
    a number of useful summary statistics before and after each step of the
    pipeline.
    
    We also plot the (a) INR-dose change and (b) dose change-INR change
    transition heatmaps after applying all preprocessing steps. These provide a
    picture of how much (a) clinically unintuitive and (b) physiologically
    unintuitive behaviour is present in the data.
    """
    audit_preprocess_all()
    for trial_names in ["engage_rocket", "rely", "aristotle"]:
        audit_preprocess_trial_specific(trial_names)
    audit_remove_outlying_doses()
    audit_merge_inr_events()
    audit_split_trajectories_at_events()
    audit_impute_inr_and_dose()
    audit_split_trajectories_at_gaps()
    audit_merge_inr_baseline()
    audit_remove_short_trajectories()
    audit_split_data()
    plot_transitions()


def message(msg, indent=1):
    """Print output in a structured way."""
    indent_str = "\t" * indent
    msg_indented = indent_str + str(msg).replace("\n", f"\n{indent_str}")
    print(msg_indented)


def trajectory_length_stats(df, traj_id_cols):
    """
    Output statistics on trajectory length.

    Args:
        df: The dataframe containing longitudinal INR data.
        traj_id_cols: The columns used to identify a trajectory.
    """
    # Number of patients
    message("Number of patients:")
    message(df.groupby("TRIAL")["SUBJID"].nunique(), 2)

    # Number of trajectories
    if "TRAJID" in traj_id_cols:
        message("Number of trajectories:")
        message(
            df.groupby(
                traj_id_cols, as_index=False
            ).size().groupby("TRIAL")["TRAJID"].count(),
            2
        )

    # Trajectory length stats (in entries)
    traj_length = df.groupby(
        traj_id_cols, as_index=False
    )["INR_VALUE"].count().groupby("TRIAL")["INR_VALUE"].describe()
    message("Trajectory length (# entries):")
    message(traj_length[["min", "25%", "50%", "mean", "75%", "max"]], 2)

    # Trajectory length stats (in days)
    last = df.groupby(traj_id_cols)["STUDY_DAY"].max()
    first = df.groupby(traj_id_cols)["STUDY_DAY"].min()
    traj_length_days = (last - first).reset_index().groupby(
        "TRIAL"
    )["STUDY_DAY"].describe()
    message("Trajectory length (days):")
    message(traj_length_days[["min", "25%", "50%", "mean", "75%", "max"]], 2)


def missing_doses(df):
    """
    Output statistics on missing doses.

    Args:
        df: The dataframe containing longitudinal INR data.
    """
    df["DOSE_NULL"] = df["WARFARIN_DOSE"].isnull()

    message("Missing doses by trial:")
    message(df.groupby("TRIAL")["DOSE_NULL"].sum(), 2)

    _df = df.groupby(["TRIAL", "SUBJID"], as_index=False)["DOSE_NULL"].sum()
    _df["AT_LEAST_ONE_DOSE_NULL"] = _df["DOSE_NULL"] > 0
    message("Number of patients with missing doses by trial:")
    message(_df.groupby("TRIAL")["AT_LEAST_ONE_DOSE_NULL"].sum(), 2)


def audit_preprocess_all():
    """
    Audit the results of the `preprocess_all` step of the pipeline.
    """
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

    trajectory_length_stats(inr[inr["INR_TYPE"] == "Y"],
                            ["TRIAL", "SUBJID", "TRAJID"])

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
        np.asarray(inr.groupby("TRIAL")["SUBJID"].nunique()).reshape(-1, 1),
        2
    )

    # Null INR values
    message("Null INR values:")
    inr["INR_NULL"] = inr["INR_VALUE"].isnull()
    message(
        inr[inr["INR_TYPE"] == "Y"].groupby(
            "TRIAL"
        )["INR_NULL"].sum(),
        2
    )

    # Negative doses
    message("Counts of negative doses (likely typos):")
    message(
        inr[(inr["WARFARIN_DOSE"] < 0)].groupby(
            ["TRIAL", "SUBJID"]
        )["WARFARIN_DOSE"].count(),
        2
    )

    # In ENGAGE and ROCKET data, how often do we actually have three daily
    # entries for each following observed INR
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
    """
    Audit the results of trial-specific preprocessing steps.

    Args:
        trial_names: The names of the trials to audit.
    """
    df_path = os.path.join(config.AUDIT_PATH,
                           f"preprocess_{trial_names}.feather")
    inr = pd.read_feather(df_path)

    message(f"Auditing results of `preprocess_{trial_names}`...", 0)

    trajectory_length_stats(inr[inr["INR_TYPE"] == "Y"],
                            ["TRIAL", "SUBJID", "TRAJID"])

    # Number of non-null INRs
    message("Number of observed INRs per patient:")
    message(
        inr[inr["INR_TYPE"] == "Y"].groupby(
            ["TRIAL", "SUBJID"], as_index=False
        )["INR_VALUE"].count().groupby("TRIAL")["INR_VALUE"].mean(),
        2
    )

    # Number of non-null doses
    message("Number of observed warfarin doses per patient:")
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
    """
    Audit the results of the `merge_trials_and_remove_outlying_doses` step.
    """
    df_path = os.path.join(config.AUDIT_PATH,
                           "merge_trials_and_remove_outlying_doses.feather")
    df = pd.read_feather(df_path)

    df = df[df["INR_TYPE"] == "Y"]

    message("Auditing results of `merge_trials_and_remove_outlying_doses`...",
            0)

    message("Warfarin dose summary stats:")
    message(df["WARFARIN_DOSE"].describe(), 2)

    trajectory_length_stats(df, ["TRIAL", "SUBJID", "TRAJID"])


def audit_merge_inr_events():
    """
    Audit the results of INR and events dataframe merging.
    """
    df_path = os.path.join(config.AUDIT_PATH, "merge_inr_events.feather")
    df = pd.read_feather(df_path)

    message("Auditing results of `merge_inr_events`...", 0)

    message("Maximum number of times a study day is recorded for a patient "
            "(should be 1):")
    message(
        df.groupby(["TRIAL", "SUBJID", "STUDY_DAY"])["TRAJID"].count().max(), 2
    )

    message("Number of event occurrences by trial:")
    message(df.groupby("TRIAL")[config.EVENTS_TO_KEEP].sum(), 2)

    message("Rate of events by trial (per patient):")
    message(
        df.groupby("TRIAL")[config.EVENTS_TO_KEEP].sum() /
        np.asarray(df.groupby("TRIAL")["SUBJID"].nunique()).reshape(-1, 1),
        2
    )

    trajectory_length_stats(df, ["TRIAL", "SUBJID", "TRAJID"])


def audit_split_trajectories_at_events():
    """
    Audit the results of splitting the trajectories on events.
    """
    df_path = os.path.join(config.AUDIT_PATH,
                           "split_trajectories_at_events.feather")
    df = pd.read_feather(df_path)

    message("Auditing results of `split_trajectories_at_events`...", 0)

    message("Number of event occurrences by trial:")
    message(df.groupby("TRIAL")[config.EVENTS_TO_KEEP].sum(), 2)

    message("Rate of events by trial (per patient):")
    message(
        df.groupby("TRIAL")[config.EVENTS_TO_KEEP].sum() /
        np.asarray(df.groupby("TRIAL")["SUBJID"].nunique()).reshape(-1, 1),
        2
    )

    trajectory_length_stats(df, ["TRIAL", "SUBJID", "TRAJID"])

    message("Maximum number of splittable events in a transition "
            "(should be 1):")
    df["SPLITTABLE_EVENT"] = df[config.EVENTS_TO_SPLIT].sum(axis=1) > 0
    message(
        df.groupby(["SUBJID", "TRAJID"])["SPLITTABLE_EVENT"].sum().max(), 2
    )

    missing_doses(df)


def audit_impute_inr_and_dose():
    """
    Audit the results of imputing INRs and doses.
    """
    pre_df_path = os.path.join(config.AUDIT_PATH,
                               "split_trajectories_at_events.feather")
    df_path = os.path.join(config.AUDIT_PATH, "impute_inr_and_dose.feather")
    pre_df = pd.read_feather(pre_df_path)
    df = pd.read_feather(df_path)

    message("Auditing results of `impute_inr_and_dose`...", 0)

    missing_doses(df)

    # Extract the time gaps between imputed warfarin doses and their source
    pre_df["WARFARIN_DOSE_RECORD_DATE"] = np.where(
        pre_df["WARFARIN_DOSE"].isnull(),
        np.nan,
        pre_df["STUDY_DAY"]
    )
    source_day = pre_df.groupby(
        ["TRIAL", "SUBJID"]
    )["WARFARIN_DOSE_RECORD_DATE"].bfill()
    pre_df["DOSE_IMPUTATION_GAP"] = (
        source_day - pre_df["STUDY_DAY"]
    )
    desc = pre_df[pre_df["DOSE_IMPUTATION_GAP"] > 0].groupby(
        "TRIAL"
    )["DOSE_IMPUTATION_GAP"].describe()[["min", "25%", "50%", "75%", "max"]]

    message("Number of days between backfilled and source doses by trial:")
    message(desc, 2)


def audit_split_trajectories_at_gaps():
    """
    Audit the results of splitting trajectories at long stretches in time where
    no INR or dose was recorded.
    """
    df_path = os.path.join(config.AUDIT_PATH,
                           "split_trajectories_at_gaps.feather")
    df = pd.read_feather(df_path)

    message("Auditing results of `split_trajectories_at_gaps`...", 0)

    trajectory_length_stats(df, ["TRIAL", "SUBJID", "TRAJID"])


def audit_merge_inr_baseline():
    """
    Audit the results of merging the INR and baseline data.
    """
    df_path = os.path.join(config.AUDIT_PATH, "merge_inr_baseline.feather")
    df = pd.read_feather(df_path)

    message("Auditing results of `merge_inr_baseline`...", 0)

    trajectory_length_stats(df, ["TRIAL", "SUBJID", "TRAJID"])

    message("Missing values in baseline columns:")
    message(df[config.STATIC_STATE_COLS].isnull().sum(), 2)


def audit_remove_short_trajectories():
    """
    Audit the results of removing short trajectories.
    """
    df_path = os.path.join(config.AUDIT_PATH,
                           "remove_short_trajectories.feather")
    df = pd.read_feather(df_path)

    message("Auditing results of `remove_short_trajectories`...", 0)

    trajectory_length_stats(df, ["TRIAL", "SUBJID", "TRAJID"])


def audit_split_data():
    """
    Audit the results of splitting the data into train/validation/test.
    """
    train_df_path = os.path.join(config.AUDIT_PATH, "split_data_train.feather")
    train_df = pd.read_feather(train_df_path)

    val_df_path = os.path.join(config.AUDIT_PATH, "split_data_val.feather")
    val_df = pd.read_feather(val_df_path)

    test_df_path = os.path.join(config.AUDIT_PATH, "split_data_test.feather")
    test_df = pd.read_feather(test_df_path)

    msg = "Data split not correct"
    assert len(np.intersect1d(train_df["SUBJID"], val_df["SUBJID"])) == 0
    assert np.sum(val_df["TRIAL"] != "ARISTOTLE") == 0, msg
    assert np.sum(train_df["TRIAL"] == "RELY") == 0, msg
    assert np.sum(val_df["TRIAL"] == "RELY") == 0, msg
    assert np.sum(test_df["TRIAL"] != "RELY") == 0, msg


def plot_transitions():
    """
    Plot the final transition heatmaps after removing short trajectories, which
    is the final preprocessing step prior to splitting the data into train/val/
    test.
    """
    df_path = os.path.join(config.AUDIT_PATH,
                           "remove_short_trajectories.feather")
    df = pd.read_feather(df_path).set_index(["TRIAL", "SUBJID", "TRAJID"])

    df["DAYS_ELAPSED"] = df.groupby(
        ["TRIAL", "SUBJID", "TRAJID"]
    )["STUDY_DAY"].diff().shift(-1)

    df["REL_DOSE_CHANGE"] = df.groupby(
        ["TRIAL", "SUBJID", "TRAJID"]
    )["WARFARIN_DOSE"].shift(-1) / df["WARFARIN_DOSE"] - 1

    df["INR_CHANGE"] = df.groupby(
        ["TRIAL", "SUBJID", "TRAJID"]
    )["INR_VALUE"].diff().shift(-1)

    _df = df.loc[:, ["INR_VALUE", "REL_DOSE_CHANGE", "INR_CHANGE"]].dropna()
    _df["INR_BIN"] = pd.cut(
        _df["INR_VALUE"],
        [0., 1., 2., 3., 4., np.inf],
        labels=["< 1", "[1, 2)", "[2, 3]", "(3, 4]", "> 4"]
    )
    _df.loc[_df["INR_VALUE"] == 2., "INR_BIN"] == "[2, 3]"
    action_ints = np.array(
        pd.cut(_df["REL_DOSE_CHANGE"],
               [-np.inf, -0.2, -0.1, 0., 0.1, 0.2, np.inf]).cat.codes
    )
    action_ints[action_ints > 2] += 1
    action_ints[_df["REL_DOSE_CHANGE"] == 0.] = 3
    _df["REL_DOSE_CHANGE_BIN"] = pd.Categorical(
        pd.Series(action_ints).map(
            dict(zip(np.arange(7).astype(int), config.ACTION_LABELS))
        ),
        ordered=True,
        categories=config.ACTION_LABELS
    )
    _df["INR_CHANGE_BIN"] = pd.cut(
        _df["INR_CHANGE"],
        [-np.inf, -1., -0.25, 0.25, 1., np.inf],
        labels=["Decrease > 1.0", "Decrease < 1.0", "Approx. Same",
                "Increase < 1.0", "Increase > 1.0"]
    )

    _df = _df.reset_index()
    plt_df = _df[["TRIAL", "INR_BIN", "REL_DOSE_CHANGE_BIN"]].value_counts()
    plt_df = plt_df.reset_index().rename(columns={0: "count"}).set_index(
        ["TRIAL", "INR_BIN", "REL_DOSE_CHANGE_BIN"]
    )
    plt_df["freq"] = (plt_df["count"] /
                      plt_df.groupby(["TRIAL", "INR_BIN"])["count"].sum())
    plt_df["freq_txt"] = plt_df["freq"].apply(lambda f: f"{f*100:.2f}%")
    plt = (
        ggplot(plt_df.reset_index(),
               aes(x="INR_BIN", y="REL_DOSE_CHANGE_BIN")) +
        geom_tile(aes(fill="freq")) +
        geom_text(aes(label="freq_txt"), size=6) +
        scale_fill_gradient(low="#FFFFFF", high="#4682B4", guide=False) +
        facet_wrap("TRIAL") +
        ylab("INR") +
        ylab("Resulting Change in Dose")# +
        # theme(axis_text_x=element_text(angle=90, margin={"t": 2.}))
    )
    plt_path = os.path.join(config.AUDIT_PLOT_PATH,
                            "inr_dose_change_tile.png")
    plt.save(plt_path)

    plt_df = _df[
        ["TRIAL", "REL_DOSE_CHANGE_BIN", "INR_CHANGE_BIN"]
    ].value_counts()
    plt_df = plt_df.reset_index().rename(columns={0: "count"}).set_index(
        ["TRIAL", "REL_DOSE_CHANGE_BIN", "INR_CHANGE_BIN"]
    )
    plt_df["freq"] = (plt_df["count"] /
                      plt_df.groupby(["TRIAL", "REL_DOSE_CHANGE_BIN"])["count"].sum())
    plt_df["freq_txt"] = plt_df["freq"].apply(lambda f: f"{f*100:.2f}%")
    plt = (
        ggplot(plt_df.reset_index(),
               aes(x="REL_DOSE_CHANGE_BIN", y="INR_CHANGE_BIN")) +
        geom_tile(aes(fill="freq")) +
        geom_text(aes(label="freq_txt"), size=6) +
        scale_fill_gradient(low="#FFFFFF", high="#4682B4", guide=False) +
        facet_wrap("TRIAL") +
        xlab("Change in Dose") +
        ylab("Resulting change in INR") +
        theme(axis_text_x=element_text(angle=90, margin={"t": 2.}))
    )
    plt_path = os.path.join(config.AUDIT_PLOT_PATH,
                            "dose_change_inr_change_tile.png")
    plt.save(plt_path)


if __name__ == "__main__":
    main()
