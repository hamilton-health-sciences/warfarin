
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from textwrap import wrap



class ThresholdModel():

    def __init__(self):
        self.threshold_action_map = {
                        0.0: None,
                        0.9 : 1,
                       1 : 2,
                       1.1 : 3,
                       1.15 : 4}
        self.threshold_action_map_7 = {
                        0.0 : None,
                       0.9 : 2,
                       1 : 3,
                       1.1 : 4,
                       1.15 : 5}
        self.name = "Threshold Model"

        
    def grp_range(self, a):
        idx = a.cumsum()
        id_arr = np.ones(idx[-1], dtype=int)
        id_arr[0] = 0
        id_arr[idx[:-1]] = -a[:-1] + 1
        return id_arr.cumsum()

    def prepare_data(self, df):

        if "WARFARIN_DOSE_PREV" not in df.columns:
            df["WARFARIN_DOSE_PREV"] = df.groupby("USUBJID_O_NEW")["WARFARIN_DOSE"].shift(1)
            df["WARFARIN_DOSE_MULT"] = 1 + (df.groupby("USUBJID_O_NEW")["WARFARIN_DOSE"].diff()) / (
                        0.01 + df["WARFARIN_DOSE_PREV"])
#         df = df.dropna(subset=["DELTA_INR"])
        return df

    def predict_dosage_thresholds(self, df, colname="INR_VALUE", return_dose=True):

        # NOTE: Ignores INR < 1 and INR > 4 (consistent with the paper)
        if return_dose:
            df = self.prepare_data(df)
        
        # Create list of conditions for thresholds
        conditions = [
            (df[colname] >= 1) & (df[colname] <= 1.5),
            (df[colname] > 1.5) & (df[colname] < 2),
            (df[colname] >= 2) & (df[colname] <= 3),
            (df[colname] > 3) & (df[colname] <= 4)
        ]

        buckets = [1, 2, 3, 4]
        categories = np.select(conditions, buckets)
        counts = np.unique(categories, return_counts=1)[1]
        cumu_counts = self.grp_range(counts)[np.argsort(categories)] + 1

        conditions = [
            (df[colname] >= 1) & (df[colname] <= 1.5),
            (df[colname] > 1.5) & (df[colname] < 2),
            (df[colname] >= 2) & (df[colname] <= 3),
            (df[colname] > 3) & (df[colname] <= 4) & (cumu_counts % 2),
            (df[colname] > 3) & (df[colname] <= 4) & ((cumu_counts + 1) % 2)
        ]

        # Create list of associated recommended dosage multipliers (weekly)
        values = [1.15, 1.1, 1, 1, 0.9]

        # Create a new column and use np.select to assign values to it using our lists as arguments
        df.loc[:, "REC_DOSAGE_MULT"] = np.select(conditions, values)
        if return_dose:
            df.loc[:, "REC_DOSAGE"] = df["REC_DOSAGE_MULT"] * df["WARFARIN_DOSE_PREV"]

        return df

    def get_model_results(self, df, threshold=0.05, plot=True, verbose=True):

        diff_dosages = df["REC_DOSAGE"] - df["WARFARIN_DOSE"]
        diff_dosages_perc = (df["REC_DOSAGE"] - df["WARFARIN_DOSE"]) / (df["WARFARIN_DOSE"] + 0.0001)

        diff_dosages_filter = diff_dosages[abs(diff_dosages) <= 10]
        diff_dosages_perc_filter = diff_dosages_perc[abs(diff_dosages_perc) <= 0.4]

        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(15, 4))
            diff_dosages.hist(ax=ax[0])
            ax[0].set_title("\n".join(wrap("Difference in rec. and actual weekly Warfarin dose (mg)", 30)));
            diff_dosages_perc.hist(ax=ax[1])
            ax[1].set_title("\n".join(wrap("% Difference in rec. and actual weekly Warfarin dose", 30)));
            ax[1].xaxis.set_major_formatter(mtick.PercentFormatter(1))

            fig, ax = plt.subplots(1, 2, figsize=(15, 4))
            diff_dosages_filter.hist(ax=ax[0])
            ax[0].set_title(
                "\n".join(wrap("Difference in rec. and actual weekly Warfarin dose (mg) (removed outliers)", 30)));
            diff_dosages_perc_filter.hist(ax=ax[1])
            ax[1].set_title(
                "\n".join(wrap("% Difference in rec. and actual weekly Warfarin dose (removed outliers)", 30)));
            ax[1].xaxis.set_major_formatter(mtick.PercentFormatter(1))

            print(f"Ignoring extreme values... \n\t" +
                  f"Difference in dosages: ------- {diff_dosages_filter.shape[0]:,.0f} points, {diff_dosages.shape[0] - diff_dosages_filter.shape[0]:,.0f} points were ignored ({(diff_dosages.shape[0] - diff_dosages_filter.shape[0]) / diff_dosages.shape[0]:.2%}) \n\t" +
                  f"% Difference in dosages: ----- {diff_dosages_perc_filter.shape[0]:,.0f} points, {diff_dosages_perc.shape[0] - diff_dosages_perc_filter.shape[0]:,.0f} points were igored ({(diff_dosages_perc.shape[0] - diff_dosages_perc_filter.shape[0]) / diff_dosages_perc.shape[0]:.2%})")

        perc_adherence = sum(abs(diff_dosages_perc) <= threshold) / len(diff_dosages_perc)
        if verbose:
            print(
                f"\n{perc_adherence:.2%} ({(perc_adherence * len(diff_dosages_perc)):,.0f}) of the points are within {threshold:.1%} of the actual Warfarin dose")
        return perc_adherence