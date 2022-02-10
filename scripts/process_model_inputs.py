"""Generate associations between algorithm consistency, TTR, and events."""

import os

import pandas as pd

import numpy as np


def main(args):
    # Load patient-level data
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

    # Load centre-level data
    centre_path = os.path.join(args.saslibs_path,
                               "rely_anticoag/SASlib1/plate1.sas7bdat")
    centre = pd.read_sas(centre_path)
    centre["centre"] = centre["id"]
    centre["survey"] = 1
    centre.drop(columns = "id", inplace = True)

    randlist_path = os.path.join(args.saslibs_path,
                                 "rely_pdbl/42/SASlibFinal/randlist.sas7bdat")
    warfarin = pd.read_sas(randlist_path)
    warfarin.loc[warfarin["allocate"]==3, "centre"] = warfarin[warfarin["allocate"]==3]["pid"]//1000
    allcentre = warfarin

    # TTR
    inrvis_path = os.path.join(args.saslibs_path,
                               "rely_pdbl/42/SASlibFinal/inrvis.sas7bdat")
    inrvis = pd.read_sas(inrvis_path)
    inrvis = inrvis.rename(columns = {"PTNO": "pid"})
    inrvis = inrvis[inrvis["pid"].notna()]
    inrvis[(inrvis["RANGE"]==b"RANGE1") & (inrvis["ACTEVENT"]==99)] = inrvis[(inrvis["RANGE"]==b"RANGE1") & (inrvis["ACTEVENT"]==99)].drop_duplicates()

    both = inrvis.merge(allcentre, left_on = "pid", right_on = "pid")

    # INR nomogram
    nomo_path = os.path.join(args.saslibs_path,
                             "rely_pdbl/42/iTTR/nomogrampriorevent2.sas7bdat")
    nomogrampriorevent2 = pd.read_sas(nomo_path)
    nomogrampriorevent2 = nomogrampriorevent2.rename(columns = {"PID": "pid"})
    nomogram = nomogrampriorevent2.merge(allcentre[["pid", "centre"]], left_on = "pid", right_on = "pid")

    nomogram["nomogram1"] = nomogram["Alternative1"]/10
    nomogram["nomogram2"] = nomogram["Alternative2"]/10
    nomogram["nomogram3"] = nomogram["Alternative3"]/10
    nomogram["nomogram4"] = nomogram["Alternative4"]/10
    nomogram["nomogram5"] = nomogram["Alternative5"]/10

    nomogramcent = nomogram[["nomogram1", "nomogram2", "nomogram3", "nomogram4", "nomogram5", "centre"]].groupby("centre").agg("mean")
    nomogramcent = nomogramcent.rename(columns = {
            "nomogram1": "centnomogram1",
            "nomogram2": "centnomogram2",
            "nomogram3": "centnomogram3",
            "nomogram4": "centnomogram4",
            "nomogram5": "centnomogram5"
        })

    # Country characteristic
    gdp_path = os.path.join(args.saslibs_path, "rely_pdbl/42/iTTR/gdp.sas7bdat")
    gdp = pd.read_sas(gdp_path)
    gdp = gdp.rename(columns = {"V1": "ctryname"})
    gdp["highincome"] = np.where(gdp["GDP"]>np.percentile(gdp["GDP"], 55), 1, 0)

    health_path = os.path.join(args.saslibs_path,
                               "rely_pdbl/42/iTTR/health.sas7bdat")
    health = pd.read_sas(health_path)
    health = health.rename(columns = {"Country": "ctryname"})

    ctry = gdp.merge(health, on = "ctryname")

    # Centre characteristics
    centre["sechosp"] = np.where(centre["alocat"]==2, 1, 0)
    centre["acclinic"] = np.where(centre["aantc"]==1, 1, 0)

    # Patient characteristics
    bases = baseline.merge(basco, on = "pid").merge(bascocm, on = "pid").merge(nomogram, on = "pid").merge(folup, on = "pid").merge(both, on = "pid")

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

    bases["centre"] = bases["pid"]//1000
    bases = bases.rename(columns = {"COUNTRYD": "ctryname"})
    bases["ctryname"] = bases["ctryname"].replace(b"China, Peoples Republic of", b"China")

    # merge pt, centre and ctry level characteristics
    ctrycentpt = bases[["pid", "centre", "ctryname", "AGE", "WT", "male", "white", "warfuse", "smoker", "alcohol", "hf", "stroke", "diab", "hypt", "amiod", "insulin"]].merge(
        centre[["centre", "sechosp", "acclinic"]], on = "centre").merge(
        ctry[["ctryname", "highincome", "dale", "hsp"]], on = "ctryname")

    ctrycentpt = ctrycentpt.drop_duplicates()

    ctrycentpt_path = os.path.join(args.model_results_path,
                                   "ctrycentpt_characteristic.csv")
    ctrycentpt.to_csv(ctrycentpt_path, index=False)

    # medication imformation
    data_medication = pd.read_csv(args.drugs_path).rename(
        columns={"subjid": "RELY_SUBJID"}
    )
    summary_medication = data_medication.groupby("RELY_SUBJID").sum()
    summary_medication = summary_medication.replace({2:1, 3:1})

    # patient-, centre- and country-level characteristics
    data_characteristics = pd.read_csv(ctrycentpt_path).rename(
        columns={"pid": "RELY_SUBJID"}
    ).set_index("RELY_SUBJID")

    # TTR data (used to compute algorithm consistency)
    ttr_path = os.path.join(args.model_results_path,
                            "hierarchical_ttr_linked.csv")
    data_RELY = pd.read_csv(ttr_path)

    # trajectory data (used to extract adverse event and time to adverse event)
    data = pd.read_feather(args.events_path).rename(columns = {"EVENT_T2": "STUDY_DAY"})

    id_map = data_RELY[["RELY_SUBJID", "SUBJID"]].drop_duplicates()
    data = data.merge(id_map, on = "SUBJID")

    # ## 1. Extract composite outcome and time to event

    ## reference: SAS code
    # var1=0;
    # surv1=surv;
    # if max(stroke,noncns,carddeath,majb)=1 then var1=1;
    # if var1=1 then surv1=min(stkday,ncnsday,cardday,majbday);
    # if surv1=. then surv1=1;

    # extract patients that developed the composite outcome
    data["composite_outcome"] = data[["STROKE", "MAJOR_BLEED", "SYS_EMB"]].sum(axis = 1)
    summary_event = data.loc[data["composite_outcome"] >= 1][["RELY_SUBJID", "STUDY_DAY", "composite_outcome"]].sort_values(by = ["RELY_SUBJID", "STUDY_DAY"])
    summary_event = summary_event.drop_duplicates(subset = "RELY_SUBJID", keep = "first")

    # extract patients who are censored
    summary_censor = pd.DataFrame(data[~data["RELY_SUBJID"].isin(summary_event["RELY_SUBJID"])].groupby("RELY_SUBJID")["STUDY_DAY"].max()).reset_index()
    summary_censor["composite_outcome"] = 0

    # combine summary_event and summary_censor
    summary = pd.concat([summary_event, summary_censor], axis = 0)
    summary = summary.rename(columns = {"STUDY_DAY": "time_to_event"})
    summary["composite_outcome"] = summary["composite_outcome"].replace({2:1})

    # for patients that are in the TTR file but not in the trajectory file
    # treat those as censored, and get the time to event info (i.e. last study day) from the TTR file.

    data_RELY = data_RELY.sort_values(by = ["RELY_SUBJID", "STUDY_DAY"])
    summary = data_RELY[["RELY_SUBJID", "STUDY_DAY"]].drop_duplicates(subset = "RELY_SUBJID", keep = "last").merge(summary, left_on = ["RELY_SUBJID"], right_on = ["RELY_SUBJID"], how = "left")
    summary["composite_outcome"].fillna(0, inplace = True)
    summary.loc[summary["composite_outcome"]==0, "time_to_event"] = np.nan
    summary["time_to_event"].fillna(summary["STUDY_DAY"], inplace = True)
    summary = summary.drop(columns = "STUDY_DAY")

    # ## 2. Compute centre-level algorithm consistency prior to adverse event or censoring

    data_RELY = data_RELY.sort_values(by = ["RELY_SUBJID", "STUDY_DAY"])
    data_RELY = data_RELY.merge(summary, left_on = ["RELY_SUBJID", "STUDY_DAY"], right_on = ["RELY_SUBJID", "time_to_event"], how = "outer")
    data_RELY["STUDY_DAY"] = data_RELY["STUDY_DAY"].fillna(data_RELY["time_to_event"])

    for id in summary[summary["composite_outcome"]==1]["RELY_SUBJID"].unique():
        termination = int(summary[summary["RELY_SUBJID"]==id]["time_to_event"])
        data_RELY.drop(data_RELY[(data_RELY["RELY_SUBJID"]==id) & (data_RELY["STUDY_DAY"]>termination)].index, inplace = True)

    # prepare data for the threshold method
    coxph_data (data_RELY, data_characteristics, summary_medication, summary, method = "threshold")

    # prepare data for the RL model
    coxph_data (data_RELY, data_characteristics, summary_medication, summary, method = "RL")

    # prepare MLM and WLS data for the threshold method
    mlm_data (data_RELY, data_characteristics, method = "threshold")

    # prepare MLM and WLS data for the RL model
    mlm_data (data_RELY, data_characteristics, method = "RL")


# ## 3. Prepare data for the Cox PH model

# In[31]:


# In addition to the patient-, center-, and country level variables used in the linear regression model
# 4 additional patient-level variables were added to the Cox proportional hazards model on the basis of clinical relevance to the outcome:
## baseline use of aspirin (yes versus no)
## beta-blockers (yes versus no)
## ace-inhibitors (yes versus no)
## statins (yes versus no)

# For patients who experienced stroke, systemic embolism, or major hemorrhage
# % algorithm-consistent dosing was calculated using warfarin prescriptions before the outcome
# for other patients, it was calculated using all warfarin prescriptions during the study

# Mean % algorithm consistency was then calculated for each center and analyzed as a center-level variable.


# In[32]:


def coxph_data (data, data_characteristics, summary_medication, summary, method = "threshold"):
    
    """
    data: a clean copy of the TTR file, with all the rows after the adverse event removed
    data_characteristics: a dataframe containing all the patient-, centre- and country-level characteristics
    summary_medication: a dataframe containing the baseline medication information
    summary: a data frame containing the adverse event (i.e., composite outcome) and time to event information
    method: "threshold" for the threshold method, "RL" for the RL model.
    """
    if method == "threshold":
        quant_diff = "THRESHOLD_ACTION_DIFF"
    elif method == "RL":
        quant_diff = "POLICY_ACTION_DIFF"
    
    # remove rows with missing/invalid values
    data[[quant_diff]] = data[[quant_diff]].replace([np.inf, -np.inf], np.nan)
    data = data[data[quant_diff].notna()]
    
    # Compute an subject-level algorithm consistency index
    ## This analysis assessed the warfarin dose modification documented in response to each INR result to determine
    ## whether it was algorithm-consistent, defined as within 5% of the dose recommended by the algorithm.
    ## Algorithm consistency was expressed as the percentage (%) of total dose instructions consistent with the algorithm in each patient.
    data["algorithm_consistency"] = (abs(data[quant_diff]) <= 0.05) * 1
    algorithm_consistency = data[["RELY_SUBJID", "algorithm_consistency"]].groupby("RELY_SUBJID").agg(algorithm_consistency = ("algorithm_consistency", "mean"))
    
    # merge DFs together
    df = algorithm_consistency.join(
        data_characteristics, how = "left", on = "RELY_SUBJID").join(
        summary_medication, how = "left", on = "RELY_SUBJID").join(
        summary.set_index("RELY_SUBJID"), how = "left", on = "RELY_SUBJID")
    
    df = df.rename(columns = {"centre": "CENTID", "ctryname": "CTRYID"})

    # Compute an center-level algorithm consistency index
    ## Algorithm-consistency was analyzed as a center-level variable because
    ## although warfarin dosing was tracked in individual patients, dosing was performed by healthcare professionals at centers.
    df = df.merge(df[["CENTID", "algorithm_consistency"]].groupby("CENTID").agg(centre_algorithm_consistency = ("algorithm_consistency", "mean")), on = "CENTID")
    
    # save data
    output_path = os.path.join(args.model_results_path, f"coxMLM_{method}.csv")
    df.to_csv(output_path, index = False)


def mlm_data(data, data_characteristics, method = "threshold"):
    """
    Generate the inputs for the multi-level model.

    Args:
        data: The original copy of the TTR file.
        data_characteristics: A dataframe containing all the patient-, centre-
                              and country-level characteristics.

    Returns:
        df: The processed dataframe.
    """
    if method == "threshold":
        quant_diff = "THRESHOLD_ACTION_DIFF"
    elif method == "RL":
        quant_diff = "POLICY_ACTION_DIFF"
    
    # remove rows with missing/invalid values
    data[[quant_diff]] = data[[quant_diff]].replace([np.inf, -np.inf], np.nan)
    data = data[data[quant_diff].notna()]
    
    # Extract the TTR information for each trajectory and each subject
    data_TTR = data[["RELY_SUBJID", "TRAJID", "TRAJECTORY_LENGTH", "APPROXIMATE_TTR"]].groupby(["RELY_SUBJID", "TRAJID"]).agg("mean").reset_index()
    
    # Compute a TTR for each subject as a weighted average of trajectory-level TTR, with weights being trajectory length
    weighted_mean = lambda x: np.average(x, weights = data_TTR.loc[x.index, "TRAJECTORY_LENGTH"])
    TTR = data_TTR[["RELY_SUBJID", "TRAJECTORY_LENGTH", "APPROXIMATE_TTR"]].groupby("RELY_SUBJID").agg(TTR = ("APPROXIMATE_TTR", weighted_mean))
    
    # Compute an subject-level algorithm consistency index
    ## This analysis assessed the warfarin dose modification documented in response to each INR result to determine
    ## whether it was algorithm-consistent, defined as within 5% of the dose recommended by the algorithm.
    ## Algorithm consistency was expressed as the percentage (%) of total dose instructions consistent with the algorithm in each patient.
    data["algorithm_consistency"] = (abs(data[quant_diff]) <= 0.05) * 1
    algorithm_consistency = data[["RELY_SUBJID", "algorithm_consistency"]].groupby("RELY_SUBJID").agg(algorithm_consistency = ("algorithm_consistency", "mean"))
    
    # merge all DFs together
    df = TTR.join(
        algorithm_consistency, how = "left", on = "RELY_SUBJID").join(
        data_characteristics, how = "left", on = "RELY_SUBJID")
    df = df.rename(columns = {"centre": "CENTID", "ctryname": "CTRYID"})

    # Compute an center-level algorithm consistency index
    ## Algorithm-consistency was analyzed as a center-level variable because
    ## although warfarin dosing was tracked in individual patients, dosing was performed by healthcare professionals at centers.
    df = df.merge(df[["CENTID", "algorithm_consistency"]].groupby("CENTID").agg(centre_algorithm_consistency = ("algorithm_consistency", "mean")), on = "CENTID")
    
    # save the MLM data
    output_path = os.path.join(args.model_results_path, f"MLM_{method}.csv")
    df.to_csv(output_path, index = False)

    # prepare data for the weighted linear regression (WLS)
    # compuate centre-level algorithm consistency and TTR average
    output_path = os.path.join(args.model_results_path, f"WLS_cent_{method}.csv")
    df_wls_cent = df[["CENTID", "algorithm_consistency", "TTR"]].groupby("CENTID").agg(["mean", "count"])
    df_wls_cent.columns = [" ".join(col).strip() for col in df_wls_cent.columns.values]
    df_wls_cent.rename(columns = {"centre_algorithm_consistency count": "sample_size"}, inplace=True)
    df_wls_cent.to_csv(output_path, index = False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_filename", type=str, required=True,
                        help="Path to RELY baseline CSV file")
    parser.add_argument("--saslibs_path", type=str, required=True,
                        help="Path to RELY SAS libraries")
    parser.add_argument("--model_results_path", type=str, required=True,
                        help="Path to directory containing modeling results")
    parser.add_argument("--drugs_path", type=str, required=True,
                        help="Path to CSV file containing baseline drugs info")
    parser.add_argument("--events_path", type=str, required=True,
                        help="Path to the cleaned events data frame")
    args = parser.parse_args()

    main(args)
