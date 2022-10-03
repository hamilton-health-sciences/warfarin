"""Compute summary statistics of the processed data."""

import pandas as pd

import numpy as np

from warfarin import config
from warfarin.utils import interpolate_inr


def compute_mean_std_fmt(df, var_name):
    result = (
        df.groupby(
            "TRIAL"
        )[var_name].mean().map("{:.1f}".format) +
        " (" +
        df.groupby("TRIAL")[var_name].std().map(
            "{:.1f}".format
        ) + ")"
    )
    
    return result


def compute_pct_fmt(df, var_name):
    result = df.groupby("TRIAL")[var_name].mean() * 100

    if isinstance(var_name, str):
        result = result.map("{:.1f}".format)
    else:
        for col in result.columns:
            result[col] = result[col].map("{:.1f}".format)

    return result


def main(args):
    # Load the data
    df_all = pd.read_feather(args.merged_data_path)
    raw_baseline = pd.read_feather(args.raw_baseline_data_path)
    linker = pd.read_sas(args.rely_subjids_path)
    linker["RELY_SUBJID"] = linker["USUBJID_O"].str.decode("utf-8").astype(int)
    linker = linker.drop(["USUBJID_O"], axis=1)
    linker["SUBJID"] = linker["SUBJID"].astype(int)
    linker = linker.set_index("SUBJID")
    raw_baseline["SUBJID"] = raw_baseline["SUBJID"].astype(int)
    df_all = df_all.set_index("SUBJID").join(
        raw_baseline.set_index("SUBJID"), rsuffix="raw"
    ).join(linker, how="left").reset_index()

    # Subset test set to correct IDs
    test_ids = np.loadtxt(args.combine_test_ids_path).astype(int)
    df_all = df_all[(df_all["TRIAL"] != "RELY") | df_all["SUBJID"].isin(test_ids)].copy()
    num_rely_centres = (df_all.loc[df_all["RELY_SUBJID"].notnull(), "RELY_SUBJID"].astype(int) // 1000).nunique()

    # Median follow time across all trials
    median_follow_time_months = (
        df_all.groupby(["TRIAL", "SUBJID"])["STUDY_DAY"].max() -
        df_all.groupby(["TRIAL", "SUBJID"])["STUDY_DAY"].min()
    ).median() / 30.4375

    # Uniquely identify trajectories
    df_all["USUBJID"] = (
        df_all["SUBJID"].astype(str) + "." + df_all["TRAJID"].astype(str)
    )

    # Compute summary statistics
    trials = df_all["TRIAL"].unique()
    summary_df = pd.DataFrame(index=trials)

    # Patient/trajectory stats
    summary_df.loc["RELY", "Number of centres"] = num_rely_centres
    summary_df["Number of dose-response pairs"] = df_all.groupby("TRIAL")["STUDY_DAY"].count()
    summary_df["Median follow time (all trials, in months)"] = median_follow_time_months
    traj_lengths = (
        df_all.groupby(["TRIAL", "USUBJID"])["STUDY_DAY"].max() -
        df_all.groupby(["TRIAL", "USUBJID"])["STUDY_DAY"].min()
    )
    df_all["INR_NONNULL"] = ~df_all["INR_VALUE"].isnull()
    summary_df["Number of patients"] = (
        df_all.groupby("TRIAL")["SUBJID"].nunique()
    )
    summary_df["Number of trajectories"] = (
        df_all.groupby("TRIAL")["USUBJID"].nunique()
    )
    summary_df["Trajectory length"] = compute_mean_std_fmt(
        traj_lengths.to_frame(),
        "STUDY_DAY"
    )
    summary_df["Number of INRs per trajectory"] = compute_mean_std_fmt(
        df_all.groupby(["TRIAL", "SUBJID"])["INR_NONNULL"].sum().to_frame(),
        "INR_NONNULL"
    )

    # Events
    total_patient_years = traj_lengths.groupby("TRIAL").sum() / 365.25
    events = df_all.groupby("TRIAL")[config.EVENTS_TO_KEEP].sum().T
    events = events / np.array(total_patient_years) * 100
    events.index = config.EVENTS_TO_KEEP_NAMES
    for col in events.columns:
        events[col] = events[col].apply("{:.2f}".format)
    summary_df = pd.concat([summary_df, events.T], axis=1)

    # TTR
    inr_interp = interpolate_inr(
        df_all.reset_index().set_index(
            ["TRIAL", "SUBJID", "TRAJID", "STUDY_DAY"]
        )[["INR_VALUE"]]
    )
    inr_interp["INR_IN_RANGE"] = ((inr_interp["INR_VALUE"] >= 2.) &
                                  (inr_interp["INR_VALUE"] <= 3.))
    ttr = inr_interp.groupby(
        ["TRIAL", "SUBJID", "TRAJID"]
    )["INR_IN_RANGE"].mean().to_frame()[["INR_IN_RANGE"]]
    summary_df["TTR (%)"] = compute_pct_fmt(ttr, "INR_IN_RANGE")

    # Subset to the first occurrence in each subject for computing baseline
    # stats
    df_all = df_all.groupby(["TRIAL", "SUBJID"]).first().reset_index()

    # Age
    summary_df["Age"] = compute_mean_std_fmt(df_all, "AGE_DEIDENTIFIED")

    # Sex
    summary_df["Female (%)"] = compute_pct_fmt(df_all, "SEX")
    
    # Continent
    continent = pd.get_dummies(df_all["CONTINENT"])
    continent["TRIAL"] = df_all["TRIAL"]
    summary_df = summary_df.join(
        compute_pct_fmt(continent, continent.columns[:-1])
    )

    # Race
    race = pd.get_dummies(df_all["RACE2"])
    race["TRIAL"] = df_all["TRIAL"]
    summary_df = summary_df.join(
        compute_pct_fmt(race, race.columns[:-1])
    )

    # Hispanic
    df_all["HISPANIC"] = (df_all["HISPANIC"] == "Y")
    summary_df["Hispanic (%)"] = compute_pct_fmt(df_all, "HISPANIC")

    # Weight
    summary_df["Weight (kg)"] = compute_mean_std_fmt(df_all, "WEIGHT")

    # Systolic BP
    summary_df["Systolic BP (mmHg)"] = compute_mean_std_fmt(df_all, "SYSBP")

    # BMI
    summary_df["BMI"] = compute_mean_std_fmt(df_all, "BMI")

    # Diabetes status
    summary_df["Diabetes (%)"] = compute_pct_fmt(df_all, "DIABETES")

    # Hypertension
    summary_df["Hypertension (%)"] = compute_pct_fmt(df_all, "HYPERTENSION")

    # CAD
    df_all["CAD"] = (df_all["HX_CAD"] == "Y")
    summary_df["CAD (%)"] = compute_pct_fmt(df_all, "CAD")

    # MI
    summary_df["MI (%)"] = compute_pct_fmt(df_all, "HX_MI")

    # CABG
    df_all["CABG"] = (df_all["HX_CABG"] == "Y")
    summary_df["CABG (%)"] = compute_pct_fmt(df_all, "CABG")
    summary_df.loc["RELY", "CABG (%)"] = "N/A"

    # PCI
    df_all["PCI"] = (df_all["HX_PCI"] == "Y")
    summary_df["PCI (%)"] = compute_pct_fmt(df_all, "PCI")
    summary_df.loc["RELY", "PCI (%)"] = "N/A"

    # CHF
    summary_df["HF (%)"] = compute_pct_fmt(df_all, "HX_CHF")

    # Stroke or TIA
    df_all["STROKE_TIA"] = ((df_all["HX_TIA"] == "Y") |
                            (df_all["HX_STROKE"] == "Y"))
    summary_df["Stroke or TIA (%)"] = compute_pct_fmt(df_all, "STROKE_TIA")

    # AFIB Paroxysmal
    df_all["PAROXYSMAL_AFIB"] = (df_all["AFIB_TYPE"] == 1.)
    summary_df["Paroxysmal AF"] = compute_pct_fmt(df_all, "PAROXYSMAL_AFIB")

    # Ever smoked
    df_all["EVER_SMOKER"] = df_all["SMOKE"].isin(
        ["CURRENT SMOKER", "FORMER SMOKER"]
    )
    summary_df["Ever smoked (%)"] = compute_pct_fmt(df_all, "EVER_SMOKER")

    # CHADS2 score
    df_all["CHADS2 = 1"] = (df_all["CHADS2"] == 1)
    df_all["CHADS2 = 2"] = (df_all["CHADS2"] == 2)
    df_all["CHADS2 >= 3"] = (df_all["CHADS2"] >= 3)
    summary_df["CHADS2 score"] = compute_mean_std_fmt(df_all, "CHADS2")
    summary_df["CHADS2 = 1 (%)"] = compute_pct_fmt(df_all, "CHADS2 = 1")
    summary_df["CHADS2 = 2 (%)"] = compute_pct_fmt(df_all, "CHADS2 = 2")
    summary_df["CHADS2 >= 3 (%)"] = compute_pct_fmt(df_all, "CHADS2 >= 3")

    # Baseline meds
    # TODO NSAIDs?
    df_all["VKA use"] = (df_all["VKA_USE"] == "Y")
    df_all["BETABLK"] = (df_all["BMED_BETABLK"] == "Y")
    df_all["CCBLK"] = (df_all["BMED_CCBLK"] == "Y")
    df_all["DIGOXIN"] = (df_all["BMED_DIGOX"] == "Y")
    df_all["PPI"] = (df_all["BMED_PPI"] == "Y")
    summary_df["Prior VKA use (%)"] = compute_pct_fmt(df_all, "VKA use")
    summary_df["Aspirin (%)"] = compute_pct_fmt(df_all, "BMED_ASPIRIN")
    summary_df["Thienopyridines (%)"] = compute_pct_fmt(df_all, "BMED_THIENO")
    summary_df["Beta blockers (%)"] = compute_pct_fmt(df_all, "BETABLK")
    summary_df["Calcium channel blockers (%)"] = compute_pct_fmt(df_all, "CCBLK")
    summary_df["Digoxin (%)"] = compute_pct_fmt(df_all, "DIGOXIN")
    summary_df["Proton pump inhibitors (%)"] = compute_pct_fmt(df_all, "PPI")

    # Lab values
    summary_df["Creatinine"] = compute_mean_std_fmt(df_all, "BL_CREAT")

    summary_df.T.to_csv(args.output_path)


if __name__  == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_data_path", type=str, required=True)
    parser.add_argument("--raw_baseline_data_path", type=str, required=True)
    parser.add_argument("--combine_test_ids_path", type=str, required=True)
    parser.add_argument("--rely_subjids_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parsed_args = parser.parse_args()

    main(parsed_args)
