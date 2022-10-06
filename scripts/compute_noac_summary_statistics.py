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
    event_times = pd.read_csv(args.cox_model_inputs_path)
    events = pd.read_feather(args.raw_events_data_path)
    events["SUBJID"] = events["SUBJID"].astype(int)
    raw_baseline = pd.read_feather(args.raw_baseline_data_path)
    linker = pd.read_sas(args.rely_subjids_path)
    linker["RELY_SUBJID"] = linker["USUBJID_O"].str.decode("utf-8").astype(int)
    linker = linker.drop(["USUBJID_O"], axis=1)
    linker["SUBJID"] = linker["SUBJID"].astype(int)
    linker = linker.set_index("SUBJID")
    raw_baseline["SUBJID"] = raw_baseline["SUBJID"].astype(int)
    df_all = raw_baseline.set_index("SUBJID").join(
        events.set_index("SUBJID"),
        how="right",
        rsuffix="_events"
    ).join(
        linker, how="left"
    ).reset_index()

    # Subset test set to correct IDs
    test_ids = event_times.iloc[:, 0].unique()
    df_all = df_all[df_all["RELY_SUBJID"].isin(test_ids)].copy()
    num_rely_centres = (df_all.loc[df_all["RELY_SUBJID"].notnull(), "RELY_SUBJID"].astype(int) // 1000).nunique()

    # Compute summary statistics
    trials = df_all["TRIAL"].unique()
    summary_df = pd.DataFrame(index=trials)

    # Patient/trajectory stats
    summary_df.loc["RELY", "Number of centres"] = num_rely_centres
    summary_df["Number of patients"] = (
        df_all.groupby("TRIAL")["SUBJID"].nunique()
    )

    # Events
    total_patient_years = event_times["time_to_event"].sum() / 365.25
    events_subset = df_all[df_all["EVENT_NAME"].isin(["All Cause Death",
                                                      "Major Bleeding",
                                                      "Minor Bleeding",
                                                      "Ischemic Stroke",
                                                      "Hemorrhagic Stroke",
                                                      "Hospitalization",
                                                      "Systemic Embolism"])]
    events_subset = events_subset.groupby("SUBJID").first().reset_index()
    events_total = events_subset.groupby("EVENT_NAME")["SUBJID"].count()
    for event_name in events_total.index:
        summary_df.loc["RELY", event_name] = events_total.loc[event_name] / total_patient_years * 100

    # Subset to the first occurrence in each subject for computing baseline
    # stats
    df_all = df_all.groupby(["TRIAL", "SUBJID"]).first().reset_index()

    # Age
    df_all["AGE_DEIDENTIFIED"] = df_all["AGE_DEIDENTIFIED"].replace(">89", "90").astype(int)
    summary_df["Age"] = compute_mean_std_fmt(df_all, "AGE_DEIDENTIFIED")

    # Sex
    summary_df["Female (%)"] = compute_pct_fmt(df_all, "SEX")
    
    # TODO Continent
    continent = pd.get_dummies(df_all["REGION"])
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
    df_all["DIABETES"] = (df_all["DIABETES"] == "Y")
    summary_df["Diabetes (%)"] = compute_pct_fmt(df_all, "DIABETES")

    # Hypertension
    df_all["HYPERTENSION"] = (df_all["HYPERTENSION"] == "Y")
    summary_df["Hypertension (%)"] = compute_pct_fmt(df_all, "HYPERTENSION")

    # CAD
    df_all["CAD"] = (df_all["HX_CAD"] == "Y")
    summary_df["CAD (%)"] = compute_pct_fmt(df_all, "CAD")

    # MI
    df_all["HX_MI"] = (df_all["HX_MI"] == "Y")
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
    df_all["HX_CHF"] = (df_all["HX_CHF"] == "Y")
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
    df_all["BMED_ASPIRIN"] = (df_all["BMED_ASPIRIN"] == "Y")
    df_all["BMED_THIENO"] = (df_all["BMED_THIENO"] == "Y")
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
    parser.add_argument("--raw_events_data_path", type=str, required=True)
    parser.add_argument("--cox_model_inputs_path", type=str, required=True)
    parser.add_argument("--rely_subjids_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parsed_args = parser.parse_args()

    main(parsed_args)
