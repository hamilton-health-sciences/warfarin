import os

import pandas as pd

import numpy as np


def main(args):
    # Centre- and country-level variables
    consistency = pd.read_csv(args.consistency_path)
    centre_country = consistency[
        ["CENTID", "centre_algorithm_consistency", "acclinic", "sechosp",
         "CTRYID", "dale", "hsp", "highincome"]
    ].drop_duplicates()
    centre_country["CENTID"] = centre_country["CENTID"].astype(int)
    centre_country = centre_country.set_index("CENTID")

    # Baseline variables
    baseline =  pd.read_csv(args.baseline_filename, low_memory=False)
    baseline = baseline.rename(columns = {"PID": "pid"})
    folup_path = os.path.join(args.saslibs_path,
                              "rely_pdbl/42/SASlibFinal/folup.sas7bdat")
    folup = pd.read_sas(folup_path)
    basco_path = os.path.join(args.saslibs_path,
                              "rely_pdbl/42/SASlibFinal/basco.sas7bdat")
    basco = pd.read_sas(basco_path)
    basco = basco.rename(columns = {"PTNO": "pid"})
    # keep pid boacsdc age wt sex: basco[["pid", "BOACSDC", "AGE", "WT", "SEX"]]
    bascocm_path = os.path.join(args.saslibs_path,
                                "rely_pdbl/42/SASlibFinal/bascocm.sas7bdat")
    bascocm = pd.read_sas(bascocm_path)
    bascocm = bascocm.rename(columns = {"PTNO": "pid"})

    bases = baseline.merge(
        basco, on="pid"
    ).merge(
        bascocm, on="pid"
    ).merge(
        folup, on="pid"
    ).copy()

    bases["male"] = np.where(bases["SEX"]==1, 1, 0)
    bases["white"] = np.where(bases["BETHNIC"].isin([6, 9]), 1, 0)
    bases["warfuse"] = np.where(bases["BOACSDC"]==b"Experienced", 1, 0)
    bases["smoker"] = np.where(bases["BTOBAHIS"]==2, 1, 0)
    bases["alcohol"] = np.where(bases["BALCOHOL"]==2, 1, 0)
    bases["hf"] = np.where(bases["BHRTFAIL"]==2, 1, 0)
    bases["stroke"] = np.where(bases["BSTROKE"]==2, 1, 0)
    bases["diab"] = np.where(bases["BDIAB"]==2, 1, 0)
    bases["hypt"] = np.where(bases["BHYPT"]==2, 1, 0)
    bases["amiod"] = np.where(bases["AMIO_B"]==1, 1, 0)
    bases["insulin"] = np.where(bases["INSUL_B"]==1, 1, 0)

    # Medication information
    data_medication = pd.read_csv(args.drugs_path).rename(
        columns={"subjid": "RELY_SUBJID"}
    )
    summary_medication = data_medication.groupby("RELY_SUBJID").sum()
    summary_medication = summary_medication.replace({2:1, 3:1})

    # Events
    events = pd.read_feather(args.events_path).rename(
        columns={"EVENT_T2": "STUDY_DAY"}
    )
    events["SUBJID"] = events["SUBJID"].astype(int)
    other = pd.read_sas(args.other_path)
    other["SUBJID"] = other["SUBJID"].astype(int)
    other = other.set_index("SUBJID")

    # ID mapping
    linker = pd.read_sas(args.rely_subjid_path)
    linker.columns = ["RELY_SUBJID", "SUBJID"]
    linker["RELY_SUBJID"] = linker["RELY_SUBJID"].astype(int)
    linker["SUBJID"] = linker["SUBJID"].astype(int)

    rely_events = events.set_index("SUBJID").join(
        linker.set_index("SUBJID")
    )
    rely_events = rely_events[~rely_events["RELY_SUBJID"].isnull()].copy()
    rely_events["RELY_SUBJID"] = rely_events["RELY_SUBJID"].astype(int)
    rely_events = rely_events.sort_values(
        by=["RELY_SUBJID", "STUDY_DAY"]
    ).set_index("RELY_SUBJID")
    rely_events["COMPOSITE"] = (
        (rely_events["EVENT_NAME"] == "Major Bleeding") |
        (rely_events["EVENT_NAME"] == "Ischemic Stroke") |
        (rely_events["EVENT_NAME"] == "Systemic Embolism")
    )
    rely_events = rely_events[rely_events["COMPOSITE"]]
    other = other.join(linker.set_index("SUBJID"))
    censor_var = "T2_SAFCENS"
    censor_time = other.set_index("RELY_SUBJID")[[censor_var]]

    # Merge data
    baseline_sub = bases[
        ["pid", "ROAC", "AGE", "WT", "male", "white", "warfuse", "smoker",
         "alcohol", "hf", "stroke", "diab", "hypt", "amiod", "insulin"]
    ].set_index("pid")
    # Subset to NOAC patients
    baseline_sub = baseline_sub[baseline_sub["ROAC"] == 2]
    baseline_sub = baseline_sub.drop(["ROAC"], axis=1)
    merged = baseline_sub.join(summary_medication)
    # Merged with centre- and country- level variables
    merged["CENTID"] = merged.index // 1000
    merged = merged.join(centre_country, on="CENTID")
    # Merge with those who experience events
    merged = merged.join(
        rely_events[["STUDY_DAY"]].rename({"STUDY_DAY": "time_to_event"},
                                          axis=1)
    )
    # Merge with censor time
    merged = merged.join(censor_time)
    merged["composite_outcome"] = ~merged["time_to_event"].isnull()
    merged["time_to_event"] = merged["time_to_event"].fillna(
        merged[censor_var])
    merged = merged.drop([censor_var], axis=1)
    merged["composite_outcome"] = merged["composite_outcome"].astype(int)
    merged = merged.dropna()

    merged.to_csv(args.output_filename)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--consistency_path",
        type=str,
        required=True,
        help="path to file with centre-level algorithm consistency columns"
    )
    parser.add_argument(
        "--baseline_filename",
        type=str,
        required=True,
        help="Path to RELY baseline CSV file"
    )
    parser.add_argument(
        "--saslibs_path",
        type=str,
        required=True,
        help="Path to original RELY SAS libs"
    )
    parser.add_argument(
        "--events_path",
        type=str,
        required=True,
        help="Path to adverse events file"
    )
    parser.add_argument(
        "--rely_subjid_path",
        type=str,
        required=True,
        help="Path to linker"
    )
    parser.add_argument(
        "--drugs_path",
        type=str,
        required=True,
        help="Path to baseline medications file"
    )
    parser.add_argument(
        "--other_path",
        type=str,
        required=True,
        help="Path to the 'other' COMBINE-AF file, giving censoring times"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        required=True,
        help="Path to the output file"
    )
    args = parser.parse_args()

    main(args)
