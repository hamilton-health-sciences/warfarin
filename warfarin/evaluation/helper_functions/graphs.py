import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as grd
from matplotlib.ticker import PercentFormatter
from matplotlib import ticker
import random
from copy import deepcopy
from scipy import stats
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
import math

import sys; sys.path.append("/dhi_work/wum/warfarin")
from utils.smdp_buffer import SMDPReplayBuffer
from utils.config import MIN_INR_COUNTS
# from .replay_buffer_ded import ReplayBufferDed
from .constants import Constants
from .threshold_model import ThresholdModel
from .model import Model


class Graphs():

    def __init__(self):
        pass

    @staticmethod
    def get_ci(df, sig_digs=1):
        #     assert (sig_digs == 3) or (sig_digs == 2), "Please provide one of: [2,3] for sig digs"
        subset = df[df["DIFF_ACTIONS_BIN"] == 0]["ADV_EVENTS"]
        mean = subset.mean()
        stderr = subset.std() / math.sqrt(len(subset))
        ci = stderr * 1.96
        if sig_digs == 3:
            print(
                f"Mean: {mean:,.3%}, Standard error: {stderr:,.3%}, 95% CI: {ci:,.3%}, [{mean - ci:,.3%}, {mean + ci:,.3%}]")
        elif sig_digs == 2:
            print(
                f"Mean: {mean:,.2%}, Standard error: {stderr:,.2%}, 95% CI: {ci:,.2%}, [{mean - ci:,.2%}, {mean + ci:,.2%}]")
        elif sig_digs == 1:
            print(
                f"Mean: {mean:,.1%}, Standard error: {stderr:,.1%}, 95% CI: {ci:,.1%}, [{mean - ci:,.1%}, {mean + ci:,.1%}]")
        return ci

    @staticmethod
    def get_sens_wrapper(buffer, model, flipped=False, only_direc=False, incl_flags=True):

        buffer_data = deepcopy(buffer.data)
#         state_method = Constants.state_method_map[model.state_dim]
        policy = \
        model.get_model_actions(np.array(SMDPReplayBuffer.get_state(buffer_data, incl_flags=incl_flags))).transpose()[0]

        buffer_data['EVENT_NEXT_STEP'] = np.minimum(1, buffer_data.groupby('USUBJID_O_NEW')[
            Constants.neg_reward_events].shift(-1).sum(axis=1))

        actions_df = pd.DataFrame({"ID": buffer_data["USUBJID_O_NEW"], "INR_VALUE_NORM": buffer_data["INR_VALUE"],
                                   "CONTINENT_EAST ASIA": buffer_data["CONTINENT_EAST ASIA"],
                                   "CONTINENT_EASTERN EUROPE": buffer_data["CONTINENT_EASTERN EUROPE"],
                                   "CONTINENT_WESTERN EUROPE": buffer_data["CONTINENT_WESTERN EUROPE"],
                                   "SEX": buffer_data["SEX"],
                                   'EVENT_NEXT_STEP': buffer_data['EVENT_NEXT_STEP'],
                                   "CLINICIAN_ACTION_CTS": buffer_data["WARFARIN_DOSE_MULT"],
                                   "SURVIVOR_FLAG": buffer_data["SURVIVOR_FLAG"],
                                   "CLINICIAN_ACTION": buffer_data["ACTION"],
                                   "POLICY_ACTION": policy})

        actions_df["NEXT_INR"] = actions_df.groupby('ID')['INR_VALUE_NORM'].shift(-1)
        actions_df["NEXT_INR_INRANGE"] = np.where(
            np.logical_and(actions_df['NEXT_INR'] >= 0.375, actions_df['NEXT_INR'] <= 0.625), 1, 0)
        actions_df = actions_df.dropna()

        if not flipped:
            return Graphs.get_sens(actions_df)
        else:
            return Graphs.get_sens_flipped(actions_df, only_direc)

    @staticmethod
    def get_sens_flipped(actions_df, only_direc=False):

        # Positive: clinician right 
        num_clinician_right = actions_df[actions_df["NEXT_INR_INRANGE"] == 1].shape[0]  # TP + FN
        num_clinician_wrong = actions_df[actions_df['NEXT_INR_INRANGE'] != 1].shape[0]  # TN + FP

        if not only_direc:
            num_clinician_right_and_same = actions_df[(actions_df["NEXT_INR_INRANGE"] == 1) & (
                        actions_df["CLINICIAN_ACTION"] == actions_df["POLICY_ACTION"])].shape[0]
            num_same = actions_df[(actions_df["CLINICIAN_ACTION"] == actions_df["POLICY_ACTION"])].shape[0]
            num_wrong_and_diff = actions_df[(actions_df["NEXT_INR_INRANGE"] != 1) & (
                        actions_df["CLINICIAN_ACTION"] != actions_df["POLICY_ACTION"])].shape[0]
            num_diff = actions_df[(actions_df["CLINICIAN_ACTION"] != actions_df["POLICY_ACTION"])].shape[0]

        else:
            same_inc = (actions_df["CLINICIAN_ACTION"] > 3) & (actions_df['POLICY_ACTION'] > 3)
            same_dec = (actions_df["CLINICIAN_ACTION"] < 3) & (actions_df['POLICY_ACTION'] < 3)
            same = actions_df["CLINICIAN_ACTION"] == actions_df["POLICY_ACTION"]
            same_mask = np.logical_or(np.logical_or(same_inc, same_dec), same)
            num_same = actions_df[same_mask].shape[0]
            num_clinician_right_and_same = actions_df[(actions_df["NEXT_INR_INRANGE"] == 1) & same_mask].shape[0]
            num_wrong_and_diff = actions_df[(actions_df['NEXT_INR_INRANGE'] != 1) & ~same_mask].shape[0]
            num_diff = actions_df[~same_mask].shape[0]

        try:
            sens = num_clinician_right_and_same / num_clinician_right
        except ZeroDivisionError:
            sens = 0

        try:
            ppv = num_clinician_right_and_same / num_same
        except ZeroDivisionError:
            ppv = 0

        try:
            npv = num_wrong_and_diff / num_diff
        except ZeroDivisionError:
            npv = 0

        try:
            spec = num_wrong_and_diff / num_clinician_wrong
        except ZeroDivisionError:
            spec = 0

        print(
            f"Num same: {num_same}, Sensitivity: {sens:,.2%}, Specificity: {spec:,.2%}, PPV: {ppv:,.2%}, NPV: {npv:,.2%}")

        return sens, spec

    #         # Positive: clinician right
    #         num_clinician_right = actions_df[actions_df["NEXT_INR_INRANGE"] == 1].shape[0]
    #         if not only_direc:
    #             num_clinician_right_and_same = actions_df[(actions_df["NEXT_INR_INRANGE"] == 1) & (actions_df["CLINICIAN_ACTION"] == actions_df["POLICY_ACTION"])].shape[0]
    #             num_same = actions_df[(actions_df["CLINICIAN_ACTION"] == actions_df["POLICY_ACTION"])].shape[0]
    #         else:
    #             same_inc = (actions_df["CLINICIAN_ACTION"] > 3) & (actions_df['POLICY_ACTION'] > 3)
    #             same_dec = (actions_df["CLINICIAN_ACTION"] < 3) & (actions_df['POLICY_ACTION'] < 3)
    #             same = actions_df["CLINICIAN_ACTION"] == actions_df["POLICY_ACTION"]
    #             same_mask = np.logical_or(np.logical_or(same_inc, same_dec), same)
    #             num_same = actions_df[same_mask].shape[0]
    #             num_clinician_right_and_same = actions_df[(actions_df["NEXT_INR_INRANGE"] == 1) & same_mask].shape[0]

    #         try:
    #             sens = num_clinician_right_and_same / num_clinician_right
    #         except ZeroDivisionError:
    #             sens = 0

    #         try:
    #             spec = num_clinician_right_and_same / num_same
    #         except ZeroDivisionError:
    #             spec = 0

    #         print(f"Sensitivity: {sens:,.2%}, Specificity: {spec:,.2%}")

    #         return sens, spec

    @staticmethod
    def get_sens(actions_df):

        # Positive: clinician wrong
        # TP: clinician wrong AND diff action
        # FP: clinician right AND diff action
        # TN: clinician right AND same action
        # FN: clinician wrong AND same action

        # Numerator: clinician wrong and diff actions
        # Denominator: num clinician wrong --> sensitivity ()
        # Denominator: num diff actions --> specificity (should be sensitivity)

        num_clinician_wrong = actions_df[actions_df["NEXT_INR_INRANGE"] == 0].shape[0]
        num_clinician_wrong_and_diff = actions_df[(actions_df["NEXT_INR_INRANGE"] == 0) & (
                    actions_df["CLINICIAN_ACTION"] != actions_df["POLICY_ACTION"])].shape[0]
        num_diff = actions_df[(actions_df["CLINICIAN_ACTION"] != actions_df["POLICY_ACTION"])].shape[0]

        try:
            sens = num_clinician_wrong_and_diff / num_clinician_wrong
        except ZeroDivisionError:
            sens = 0

        try:
            spec = num_clinician_wrong_and_diff / num_diff
        except ZeroDivisionError:
            spec = 0

        print(f"Sensitivity: {sens:,.2%}, Specificity: {spec:,.2%}")

        return sens, spec

    @staticmethod
    def create_hist(data, x, y, hue, title=None, col_pal=None):

        if col_pal is None:
            sns.set_palette(sns.color_palette("Paired"))
        else:
            sns.set_palette(sns.color_palette(col_pal))
        ax = (data[x]
              .groupby(data[hue])
              .value_counts(normalize=True)
              .rename(y)
              .reset_index()
              .pipe((sns.barplot, "data"), x=x, y=y, hue=hue))

        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel(x.title())
        ax.legend(title=hue.title())

        if title is not None:
            ax.set_title(title)

        return ax

    @staticmethod
    def plot_policy(df, policy):

        plt.figure(figsize=(15, 6))
        x = np.arange(len(policy))
        plt.plot(df["INR_VALUE"], label="INR value")
        plt.fill_between(x, 2, 3,
                         facecolor="orange",
                         color='grey',
                         alpha=0.2)
        plt.legend()
        plt.xlabel("Timestep")
        plt.ylabel("Measured INR")

        plt.figure(figsize=(15, 2))
        x = np.arange(len(policy))
        plt.plot(x, df["ACTION"][:-1], label="Clinician")
        plt.plot(x, sample_policy, label="Policy")
        plt.legend()
        plt.xlabel("Timestep")
        plt.ylabel("Action")

    @staticmethod
    def plot_reward_space(all_data, stat=None):

        df_to_plot = all_data.sort_values(by="ACTION")
        if df_to_plot["ACTION"].nunique() == 3:
            df_to_plot["ACTION_NAME"] = df_to_plot["ACTION"].apply(lambda x: Constants.action_map[x])
        elif df_to_plot["ACTION"].nunique() == 5:
            df_to_plot["ACTION_NAME"] = df_to_plot["ACTION"].apply(lambda x: Constants.action_map_5[x])
        else:
            return "ERROR: Could not find action space"
        df_to_plot["REWARD_BIN"] = np.where(df_to_plot["REWARD"] > 0, ">0",
                                            np.where(df_to_plot["REWARD"] == 0, "0", "<0"))

        g = sns.FacetGrid(df_to_plot, col="INR_BIN", row="REWARD_BIN", subplot_kws=dict(alpha=0.5))
        if stat is not None:
            g.map(sns.histplot, "ACTION_NAME", stat=stat)
        else:
            g.map(sns.histplot, "ACTION_NAME")
        g.add_legend()

    @staticmethod
    def plot_actions(data):

        sns.set_palette(sns.color_palette("tab10"))

        piv = data[["ACTION_NAME", "INR_BIN"]].pivot_table(index='ACTION_NAME', columns='INR_BIN',
                                                           aggfunc=len, fill_value=0)

        piv = piv.reindex(index=['Decrease', 'No Change', 'Increase'])

        fig, axs = plt.subplots(ncols=len(piv.columns), sharey=True, sharex=True, figsize=(15, 3))
        for col, ax in zip(piv.columns, axs):
            piv[col].plot.bar(ax=ax, rot=25)
            ax.set_xlabel("Action")
            ax.set_ylabel("Count")
            ax.set_title(f"INR range: {col}")

    @staticmethod
    def plot_heatmap(data, norm_values=True, axs=None, x_axis="INR_BIN", num_actions=None, action_col="ACTION",
                     return_norm2=True, norm_across_dataset=False):

        if num_actions is None:
            num_actions = data["ACTION_NAME"].nunique()
        if num_actions == 3:
            data["ACTION_NAME"] = data[action_col].apply(lambda x: Constants.action_map[x])
            df2 = data.groupby(["ACTION_NAME", x_axis], as_index=False)[action_col].count().rename(
                columns={action_col: "Count"})
            df_p = pd.pivot_table(df2, 'Count', 'ACTION_NAME', x_axis)
            df_p = df_p.reindex(Constants.action_map.values())
        elif num_actions == 5:
            data["ACTION_NAME"] = data[action_col].apply(lambda x: Constants.action_map_5[x])
            df2 = data.groupby(["ACTION_NAME", x_axis], as_index=False)[action_col].count().rename(
                columns={action_col: "Count"})
            df_p = pd.pivot_table(df2, 'Count', 'ACTION_NAME', x_axis)
            indices = list(Constants.action_map_5.values())
            indices.reverse()
            df_p = df_p.reindex(indices)
        elif num_actions == 7:
            data["ACTION_NAME"] = data[action_col].apply(lambda x: Constants.action_map_7[x])
            df2 = data.groupby(["ACTION_NAME", x_axis], as_index=False)[action_col].count().rename(
                columns={action_col: "Count"})
            df_p = pd.pivot_table(df2, 'Count', 'ACTION_NAME', x_axis)
            indices = list(Constants.action_map_7.values())
            indices.reverse()
            df_p = df_p.reindex(indices)

        if norm_values:
            df_norm = df_p.div(df_p.sum(axis=1), axis=0)
            df_norm2 = df_p.div(df_p.sum(axis=0), axis=1)

            if return_norm2:
                if axs is None:
                    _, axs = plt.subplots(ncols=2, nrows=1, figsize=(15, 5), sharex=True, sharey=True)

                sns.heatmap(df_norm2, cmap="Blues", annot=True, fmt=".1%", ax=axs[0], vmin=0, vmax=1)
                sns.heatmap(df_norm, cmap="Blues", annot=True, fmt=".1%", ax=axs[1], vmin=0, vmax=1)

                axs[0].set_title("Actions Taken Based on INR (Normalized by INR Bin)")
                axs[1].set_title("Actions Taken Based on INR (Normalized by Action)")

                for ax in axs:
                    ax.set_xlabel("INR Measurement")
                    ax.set_ylabel("Action")

            else:
                if axs is None:
                    _, axs = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))
                sns.heatmap(df_norm2, cmap="Blues", annot=True, fmt=".1%", ax=axs, vmin=0, vmax=1)
                axs.set_title("Actions Taken Based on INR (Normalized by INR Bin)")

                axs.set_xlabel("INR Measurement")
                axs.set_ylabel("Action")

        else:
            axs = sns.heatmap(df_p, cmap="Blues", vmin=0, vmax=1)
            axs.set_title("Actions Taken Based on INR")

        return axs

    @staticmethod
    def get_action_bins(data, num_actions=5):
        if num_actions == 3:
            cut_bins = [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]
            cut_labels = [-2, -1, -0.5, 0, 0.5, 1, 2]
        elif num_actions == 5:
            cut_bins = [-4.5, -3.5, -2.5, -1.5, -0.8, -0.3, 0.3, 0.8, 1.5, 2, 3, 4]
            cut_labels = [-4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4]
        else:
            print(f"ERROR: could not find action num: {data['CLINICIAN_ACTION'].nunique()}")
        return cut_bins, cut_labels

    @staticmethod
    def plot_ucurve(data, model, model2=None, groupby_col=None, adverse_events=["MAJOR_BLEED", "STROKE"], axs=None,
                    outcome_agg_method='mean', incl_hist=False, ylabel=None):

        sns.set_palette(sns.color_palette("tab10"))
        if isinstance(model, ThresholdModel):
            if data["INR_VALUE"].max() <= 1:
                data["INR_VALUE_ADJ"] = data["INR_VALUE"] * 4 + 0.5
            pred_df = model.predict_dosage_thresholds(data, colname="INR_VALUE_ADJ")
            rec_dosages = pred_df["REC_DOSAGE_MULT"]
            rec_dosages = rec_dosages[rec_dosages > 0]
            policy = rec_dosages.apply(lambda x: model.threshold_action_map[x])
        else:
            state_method = Constants.state_method_map[model.state_dim]
            policy = model.get_model_actions(np.array(SMDPReplayBuffer.get_state(data, method=state_method))).transpose()[0]

        ylabel = ylabel if ylabel is not None else "Adverse Events"

        both_actions = pd.DataFrame(
            {"ID": data["USUBJID_O_NEW"], "INR_VALUE_NORM": data["INR_VALUE"], "CLINICIAN_ACTION": data["ACTION"],
             "POLICY_ACTION": policy})
        if adverse_events == Constants.neg_reward_events:
            next_state_events = data.groupby("USUBJID_O_NEW")[adverse_events].shift(-1)
            both_actions.loc[:, "ADV_EVENTS"] = next_state_events.sum(axis=1)
        else:
            #             if len(adverse_events) > 1:
            print(f"Not negative reward event")
            both_actions.loc[:, "ADV_EVENTS"] = data[adverse_events[0]]

        both_actions["DIFF_ACTIONS"] = both_actions["POLICY_ACTION"] - both_actions["CLINICIAN_ACTION"]
        both_actions = both_actions.dropna()

        if groupby_col is not None:
            cut_bins, cut_labels = Graphs.get_action_bins(both_actions)
            both_actions = both_actions.groupby(groupby_col).agg(
                {'DIFF_ACTIONS': 'mean', 'ADV_EVENTS': outcome_agg_method})  # [["DIFF_ACTIONS", "ADV_EVENTS"]].mean()
            both_actions.loc[:, "DIFF_ACTIONS_BIN"] = pd.cut(both_actions['DIFF_ACTIONS'], bins=cut_bins,
                                                             labels=cut_labels)
        else:
            both_actions.loc[:, "DIFF_ACTIONS_BIN"] = both_actions["DIFF_ACTIONS"]

        if model2 is not None:
            if isinstance(model2, ThresholdModel):
                if data["INR_VALUE"].max() <= 1:
                    data["INR_VALUE_ADJ"] = data["INR_VALUE"] * 4 + 0.5
                pred_df = model2.predict_dosage_thresholds(data, colname="INR_VALUE_ADJ")
                rec_dosages = pred_df["REC_DOSAGE_MULT"]
                rec_dosages = rec_dosages[rec_dosages > 0]
                policy2 = rec_dosages.apply(lambda x: model2.threshold_action_map[x])
            else:
                state_method = Constants.state_method_map[model2.state_dim]
                policy2 = \
                model2.get_model_actions(np.array(SMDPReplayBuffer.get_state(data, method=state_method))).transpose()[
                    0]

            both_actions2 = pd.DataFrame(
                {"ID": data["USUBJID_O_NEW"], "INR_VALUE_NORM": data["INR_VALUE"], "CLINICIAN_ACTION": data["ACTION"],
                 "POLICY_ACTION": policy2})
            if adverse_events == Constants.neg_reward_events:
                print("neg reward events")
                next_state_events = data.groupby("USUBJID_O_NEW")[adverse_events].shift(-1)
                both_actions2.loc[:, "ADV_EVENTS"] = next_state_events.sum(axis=1)
            else:
                #             if len(adverse_events) > 1:
                both_actions2.loc[:, "ADV_EVENTS"] = data[adverse_events[0]]

            both_actions2["DIFF_ACTIONS"] = both_actions2["POLICY_ACTION"] - both_actions2["CLINICIAN_ACTION"]
            both_actions2 = both_actions2.dropna()

            if groupby_col is not None:
                both_actions2 = both_actions2.groupby(groupby_col).agg({'DIFF_ACTIONS': 'mean',
                                                                        'ADV_EVENTS': outcome_agg_method})
                both_actions2.loc[:, "DIFF_ACTIONS_BIN"] = pd.cut(both_actions2['DIFF_ACTIONS'], bins=cut_bins,
                                                                  labels=cut_labels)

            else:
                both_actions2.loc[:, "DIFF_ACTIONS_BIN"] = both_actions2["DIFF_ACTIONS"]

            both_actions["MODEL"] = model.name
            both_actions2["MODEL"] = model2.name
            concat_actions = pd.concat([both_actions, both_actions2])

            if incl_hist:

                fig, ax = plt.subplots(2, 2, figsize=(13, 11), sharex=True, sharey='row')
                plt.subplots_adjust(wspace=0.1, hspace=0.1)

                sns.lineplot(data=both_actions, x="DIFF_ACTIONS_BIN", y="ADV_EVENTS", ax=ax[0][0])
                sns.lineplot(data=both_actions2, x="DIFF_ACTIONS_BIN", y="ADV_EVENTS", ax=ax[0][1])

                ax[0][0].set_ylabel(ylabel)
                ax[0][0].set_title("Model: " + model.name)
                ax[0][1].set_title("Model: " + model2.name)

                both_actions["DIFF_ACTIONS_BIN"].hist(ax=ax[1][0], bins=5, histtype="stepfilled", alpha=0.7)
                both_actions2["DIFF_ACTIONS_BIN"].hist(ax=ax[1][1], bins=5, histtype="stepfilled", alpha=0.7)

                ax[1][0].set_ylabel("Count")

                min_xvalue = min(both_actions['DIFF_ACTIONS_BIN'].min(), both_actions2['DIFF_ACTIONS_BIN'].min())
                max_xvalue = max(both_actions['DIFF_ACTIONS_BIN'].max(), both_actions2['DIFF_ACTIONS_BIN'].max())

                for axarr in ax.reshape(-1):
                    axarr.set_xlim(min_xvalue,
                                   max_xvalue)
                    axarr.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
                    axarr.set_xlabel("Difference in Policy and Clinician Actions")

            else:
                ax = sns.relplot(
                    data=concat_actions, x="DIFF_ACTIONS_BIN", y="ADV_EVENTS",
                    col="MODEL",
                    kind="line"
                )

                ax.set(xlabel="Difference in Policy and Clinician Actions", ylabel=ylabel)

            ax[0][0].yaxis.set_major_formatter(PercentFormatter(1)) if (outcome_agg_method == "mean") or (
                    outcome_agg_method == "last") else None

            return both_actions, both_actions2

        else:
            axes = sns.lineplot(data=both_actions, x="DIFF_ACTIONS_BIN", y="ADV_EVENTS", markers=True)
            axes.set(xlabel="Difference in Policy and Clinician Actions", ylabel=ylabel)
            axes.yaxis.set_major_formatter(PercentFormatter(1))

            return both_actions

    @staticmethod
    def plot_ucurve_new(data, model="naive", groupby_col=None, adverse_events=["MAJOR_BLEED", "STROKE"], axes=None,
                        outcome_agg_method='mean', incl_hist=False, ylabel=None, seed=42, show_ci=True, num_actions=5, incl_flags=True,
                        pct_agree=False, use_abs=False, use_qcut=False, cut_bins=None, cut_bins_0=None):

        sns.set_palette(sns.color_palette("tab10"))

        if model == "naive":
            print(f"Using naive baseline...")
            middle_action = math.floor(num_actions / 2)
            policy = np.ones(data.shape[0]) * middle_action
        elif model == "clinician":
            print(f"Using clinician actions...")
            policy = data['ACTION']
        elif model == "random":
            np.random.seed(seed)
            policy = np.random.choice(num_actions, len(data))
            print(f"Using random baseline...")
        elif model == "threshold":
            model = ThresholdModel()
            if data["INR_VALUE"].max() <= 1:
                data["INR_VALUE_ADJ"] = data["INR_VALUE"] * 4 + 0.5
            pred_df = model.predict_dosage_thresholds(data, colname="INR_VALUE_ADJ", return_dose=False)
            rec_dosages = pred_df["REC_DOSAGE_MULT"]
            rec_dosages = rec_dosages[rec_dosages > 0]
            if num_actions == 5:
                policy = rec_dosages.apply(lambda x: model.threshold_action_map[x])
            elif num_actions == 7:
                policy = rec_dosages.apply(lambda x: model.threshold_action_map_7[x])
        else:
            state_method = Constants.state_method_map[model.state_dim]
            policy = model.get_model_actions(np.array(SMDPReplayBuffer.get_state(data, incl_flags=incl_flags))).transpose()[0]

        ylabel = ylabel if ylabel is not None else "Adverse Events"

        both_actions = pd.DataFrame(
            {"ID": data["USUBJID_O_NEW"], "INR_VALUE_NORM": data["INR_VALUE"], "CLINICIAN_ACTION": data["ACTION"],
             "POLICY_ACTION": policy})
        if adverse_events == Constants.neg_reward_events:
            print("neg reward events")
            next_state_events = data.groupby("USUBJID_O_NEW")[adverse_events].shift(-1)
            both_actions.loc[:, "ADV_EVENTS"] = next_state_events.sum(axis=1)
        else:
            both_actions.loc[:, "ADV_EVENTS"] = data[adverse_events[0]]

        #         both_actions["POLICY_ACTION"] = both_actions["POLICY_ACTION"].fillna(method="ffill")
        both_actions["DIFF_ACTIONS"] = both_actions["POLICY_ACTION"] - both_actions["CLINICIAN_ACTION"]
        #         return None, both_actions, None
        #         both_actions = both_actions.dropna()

        if use_abs:
            both_actions['DIFF_ACTIONS'] = both_actions['DIFF_ACTIONS'].apply(lambda x: abs(x))

        if groupby_col is not None:
            # Diff actions actually becomes agreement 
            both_actions['AGREE'] = np.where(both_actions['DIFF_ACTIONS'] == 0, 1, 0)
            both_actions['TRAJ_LENGTH'] = 1
            both_actions = both_actions.groupby(groupby_col).agg(
                {'DIFF_ACTIONS': 'mean', 'AGREE': 'mean', 'ADV_EVENTS': outcome_agg_method,
                 'TRAJ_LENGTH': 'sum'})  # [["DIFF_ACTIONS", "ADV_EVENTS"]].mean()
            both_actions = both_actions[both_actions["TRAJ_LENGTH"] >= MIN_INR_COUNTS] #TODO: remove this later on

            if cut_bins is None:
                if use_qcut:
                    _, cut_bins = pd.qcut(both_actions['DIFF_ACTIONS'] * 10, 6, retbins=True)
                    cut_labels = [(cut_bins[i + 1] + cut_bins[i]) / 2 for i in range(len(cut_bins) - 1)]
                    if cut_bins_0 is None:
                        cut_bins[0] = cut_bins[0] - 0.001
                    else:
                        cut_bins[0], cut_bins[1] = cut_bins_0[0]
                        cut_labels[0] = cut_bins_0[1]
                    if use_abs:
                        cut_labels[0] = 0
                    print(cut_bins)
                    print(cut_labels)
                    both_actions.loc[:, "DIFF_ACTIONS_BIN"] = pd.cut(both_actions['DIFF_ACTIONS'] * 10, bins=cut_bins,
                                                                     labels=cut_labels).astype(int)
                else:
                    cut_bins, cut_labels = Graphs.get_action_bins(both_actions)
                    both_actions.loc[:, "DIFF_ACTIONS_BIN"] = pd.cut(both_actions['DIFF_ACTIONS'], bins=cut_bins,
                                                                     labels=cut_labels).astype(float) * 10
            else:
                cut_labels = [(cut_bins[i + 1] + cut_bins[i]) / 2 for i in range(len(cut_bins) - 1)]
                cut_labels[0] = 0
                both_actions.loc[:, "DIFF_ACTIONS_BIN"] = pd.cut(both_actions['DIFF_ACTIONS'] * 10, bins=cut_bins,
                                                                 labels=cut_labels)
                both_actions.loc[:, "DIFF_ACTIONS_BIN"] = both_actions.loc[:, "DIFF_ACTIONS_BIN"].fillna(
                    cut_labels[-1]).astype(int)

        else:
            both_actions['AGREE'] = np.where(both_actions['DIFF_ACTIONS'] == 0, 1, 0)
            both_actions.loc[:, "DIFF_ACTIONS_BIN"] = both_actions["DIFF_ACTIONS"]

        if pct_agree:
            x_axis = "AGREE"
        else:
            x_axis = "DIFF_ACTIONS_BIN"

        if axes is None:
            if show_ci:
                axes = sns.lineplot(data=both_actions, x=x_axis, y="ADV_EVENTS", markers=True)
            else:
                axes = sns.lineplot(data=both_actions, x=x_axis, y="ADV_EVENTS", markers=True, ci=None)
        else:
            if show_ci:
                sns.lineplot(data=both_actions, x=x_axis, y="ADV_EVENTS", markers=True, ax=axes)
            else:
                sns.lineplot(data=both_actions, x=x_axis, y="ADV_EVENTS", markers=True, ax=axes, ci=None)
        axes.set(xlabel="% Difference between Model and Clinician Warfarin Dosages (% Dose Change)", ylabel=ylabel)
        axes.yaxis.set_major_formatter(PercentFormatter(1))

        return axes, both_actions, cut_bins

    @staticmethod
    def plot_value_bins(survivors_array, nonsurvivors_array, ax, title):

        P = np.arange(0, 1.1, 0.1)
        survivors_array = np.histogram(survivors_array, bins=P)[0]
        survivors_array = survivors_array / sum(survivors_array)
        nonsurvivors_array = np.histogram(nonsurvivors_array, bins=P)[0]
        nonsurvivors_array = nonsurvivors_array / sum(nonsurvivors_array)

        x = survivors_array
        y = nonsurvivors_array
        x_w = np.empty(x.shape)
        x_w.fill(1 / x.shape[0])
        y_w = np.empty(y.shape)
        y_w.fill(1 / y.shape[0])

        # Width of a bar 
        width = 0.3
        ind = np.arange(1, 11)

        # Plotting
        if len(x):
            ax.bar(ind, x, width, label='Survivors', color="g")
        if len(y):
            ax.bar(ind + width, y, width, label='Nonsurvivors', color="b")

        ax.set_xlabel('Value Bin')
        ax.set_ylabel('% of Dataset')
        ax.set_title(f"{title}")

        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(np.arange(1, 11))

        # Finding the best position for legends and putting it
        ax.legend(loc='best')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_ylim([0, 1])

    @staticmethod
    def get_action_heatmap_by_feature(train_buffer, val_buffer, colname, num_actions, state_method, model):

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        state_cols = Constants.state_cols_mapping[state_method]

        sample_state1 = deepcopy(train_buffer.state)
        idx = state_cols.index(colname)
        sample_state1 = sample_state1[sample_state1[:, idx] == 1]
        model.get_model_results(sample_state1, state_method=state_method, num_actions=num_actions)
        Graphs.plot_heatmap(model.df, axs=axes[0], num_actions=num_actions, return_norm2=False);
        orig_title = axes[0].title.get_text()
        axes[0].set_title(f"Training Data")

        sample_state2 = deepcopy(val_buffer.state)
        idx = state_cols.index(colname)
        sample_state2 = sample_state2[sample_state2[:, idx] == 1]
        model.get_model_results(sample_state2, state_method=state_method, num_actions=num_actions)
        Graphs.plot_heatmap(model.df, axs=axes[1], num_actions=num_actions, return_norm2=False);
        axes[1].set_title(f"Validation Data")

        fig.suptitle("RL Algo: " + orig_title, fontsize=16)
        plt.tight_layout()

        ###############################################
        # Clinician
        ###############################################
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        buffer_data = deepcopy(train_buffer.data)
        buffer_data = buffer_data[buffer_data[colname] == 1]
        buffer_data = buffer_data[~buffer_data['WARFARIN_DOSE_MULT'].isnull()]
        buffer_data['ACTION'] = SMDPReplayBuffer.get_action(
            buffer_data[['STUDY_WEEK', 'WARFARIN_DOSE', 'INR_VALUE', 'WARFARIN_DOSE_MULT']], num_actions=num_actions)
        Graphs.plot_heatmap(buffer_data, axs=axes[0], num_actions=num_actions, return_norm2=False);
        orig_title = axes[0].title.get_text()
        axes[0].set_title(f"Training Data")

        buffer_data = deepcopy(val_buffer.data)
        buffer_data = buffer_data[buffer_data[colname] == 1]
        buffer_data = buffer_data[~buffer_data['WARFARIN_DOSE_MULT'].isnull()]
        buffer_data['ACTION'] = SMDPReplayBuffer.get_action(
            buffer_data[['STUDY_WEEK', 'WARFARIN_DOSE', 'INR_VALUE', 'WARFARIN_DOSE_MULT']], num_actions=num_actions)
        Graphs.plot_heatmap(buffer_data, axs=axes[1], num_actions=num_actions, return_norm2=False);
        axes[1].set_title(f"Validation Data")

        fig.suptitle("Clinician: " + orig_title, fontsize=16)
        plt.tight_layout()

    @staticmethod
    def get_ucurve_by_demographic_feature(buffer_data, model, events_to_plot, colname1, colname2, num_actions=7):

        fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)

        threshold_model = ThresholdModel()
        subset = buffer_data[buffer_data[colname1] == 1]
        ax[0], both_actions1, cut_bins = Graphs.plot_ucurve_new(subset, model='threshold', groupby_col="ID",
                                                                adverse_events=events_to_plot, incl_hist=True,
                                                                outcome_agg_method="last", axes=ax[0],
                                                                num_actions=num_actions, use_abs=True, use_qcut=True)
        ax[0], both_actions2, _ = Graphs.plot_ucurve_new(subset, model, groupby_col="ID", adverse_events=events_to_plot,
                                                         incl_hist=True, outcome_agg_method="last", axes=ax[0],
                                                         use_abs=True, cut_bins=cut_bins)

        ax[0].legend(['RE-LY Algorithm', 'RL Algorithm'])
        ax[0].set_ylabel('Rate of Adverse Events (per Year)')
        ax[0].set_xlabel('Absolute Difference between Model and Clinician (% Dose Change)')
        ax[0].set_title(f"{colname1}: \n{events_to_plot}")

        Graphs.get_ci(both_actions1, 2);
        Graphs.get_ci(both_actions2, 2);
        print(
            f"percent of patients: {both_actions1[both_actions1['DIFF_ACTIONS_BIN'] == 0].shape[0] / both_actions1.shape[0]:,.2%}, num: {sum(both_actions1['DIFF_ACTIONS_BIN'] == 0)}")
        print(
            f"percent of patients: {both_actions2[both_actions2['DIFF_ACTIONS_BIN'] == 0].shape[0] / both_actions2.shape[0]:,.2%}, num: {sum(both_actions2['DIFF_ACTIONS_BIN'] == 0)}")

        both_actions11, both_actions12 = both_actions1, both_actions2

        print("\n")
        subset = buffer_data[(buffer_data[colname2] == 1)]

        ax[1], both_actions1, cut_bins = Graphs.plot_ucurve_new(subset, model='threshold', groupby_col="ID",
                                                                adverse_events=events_to_plot, incl_hist=True,
                                                                outcome_agg_method="last", axes=ax[1],
                                                                num_actions=num_actions, use_abs=True, use_qcut=True)
        ax[1], both_actions2, _ = Graphs.plot_ucurve_new(subset, model, groupby_col="ID", adverse_events=events_to_plot,
                                                         incl_hist=True, outcome_agg_method="last", axes=ax[1],
                                                         use_abs=True, cut_bins=cut_bins)
        ax[1].legend(['RE-LY Algorithm', 'RL Algorithm'])
        ax[1].set_xlabel('Absolute Difference between Model and Clinician (% Dose Change)')
        ax[1].set_ylabel('Rate of Adverse Events (per Year)')
        ax[1].set_title(f"{colname2}: \n{events_to_plot}")

        Graphs.get_ci(both_actions1, 2);
        Graphs.get_ci(both_actions2, 2);
        print(
            f"percent of patients: {both_actions1[both_actions1['DIFF_ACTIONS_BIN'] == 0].shape[0] / both_actions1.shape[0]:,.2%}, num: {sum(both_actions1['DIFF_ACTIONS_BIN'] == 0)}")
        print(
            f"percent of patients: {both_actions2[both_actions2['DIFF_ACTIONS_BIN'] == 0].shape[0] / both_actions2.shape[0]:,.2%}, num: {sum(both_actions2['DIFF_ACTIONS_BIN'] == 0)}")

        ax[0].set_ylim([0, 1])

        plt.show()
        fig, ax = plt.subplots(1, 2, figsize=(15, 2), sharex=True)
        both_actions11['DIFF_ACTIONS_BIN'].hist(alpha=0.5, ax=ax[0])
        both_actions12['DIFF_ACTIONS_BIN'].hist(alpha=0.5, ax=ax[0])
        both_actions1['DIFF_ACTIONS_BIN'].hist(alpha=0.5, ax=ax[1])
        both_actions2['DIFF_ACTIONS_BIN'].hist(alpha=0.5, ax=ax[1])

    @staticmethod
    def plot_unintuitive_cases(df_analyze, id_col='SUBJID', norm_across_dataset=False):
        df_analyze.loc[df_analyze["WARFARIN_DOSE"] == 0, "WARFARIN_DOSE_CHANGE"] = 100

        df_analyze['INR_BIN'] = Model.bin_inr(df_analyze, 5)
        df_analyze['INR_VALUE_PREV'] = df_analyze.groupby('SUBJID')['INR_VALUE'].shift(1)
        df_analyze['INR_VALUE_NEXT'] = df_analyze.groupby('SUBJID')['INR_VALUE'].shift(-1)
        df_analyze['INR_VALUE_CHANGE'] = (df_analyze['INR_VALUE_NEXT'] - df_analyze['INR_VALUE']) / (
        df_analyze['INR_VALUE'])

        subset = df_analyze.dropna(subset=['WARFARIN_DOSE_CHANGE', 'INR_VALUE_CHANGE'])
        # subset = subset[subset['TRIAL'] == 'ARISTOTLE']

        print(subset.shape)
        subset['WARFARIN_DOSE_CHANGE_SIGN'] = np.where(subset['WARFARIN_DOSE_CHANGE'] == 0, "0",
                                                       np.where(subset['WARFARIN_DOSE_CHANGE'] > 0, ">0", "<0"))
        subset['INR_VALUE_CHANGE_SIGN'] = np.where(np.abs(subset['INR_VALUE_CHANGE']) <= 0.01, "0",
                                                   np.where(subset['INR_VALUE_CHANGE'] > 0, ">0", "<0"))

        subset_df = subset[['WARFARIN_DOSE_CHANGE_SIGN', 'INR_VALUE_CHANGE_SIGN', 'SUBJID']]
        ax = Graphs.create_heatmap_helper_mod(subset_df, y_axis="INR_VALUE_CHANGE_SIGN",
                                              x_axis="WARFARIN_DOSE_CHANGE_SIGN", annot=True,
                                              norm_across_dataset=norm_across_dataset);
        ax.set_xlabel("Change in Warfarin dose")
        ax.set_ylabel("Change in INR value")

        df_analyze['INR_VALUE_CHANGE'] = (df_analyze['INR_VALUE'] - df_analyze['INR_VALUE_PREV']) / (
        df_analyze['INR_VALUE_PREV'])

        subset = df_analyze.dropna(subset=['WARFARIN_DOSE_CHANGE', 'INR_VALUE_CHANGE'])

        print(subset.shape)
        subset['WARFARIN_DOSE_CHANGE_SIGN'] = np.where(subset['WARFARIN_DOSE_CHANGE'] == 0, "0",
                                                       np.where(subset['WARFARIN_DOSE_CHANGE'] > 0, ">0", "<0"))

        subset_df = subset[['WARFARIN_DOSE_CHANGE_SIGN', 'INR_BIN', 'SUBJID']]
        ax = Graphs.create_heatmap_helper_mod(subset_df, x_axis="INR_BIN", y_axis="WARFARIN_DOSE_CHANGE_SIGN",
                                              annot=True, norm_across_dataset=norm_across_dataset);
        ax.set_ylabel("Change in Warfarin dose")
        ax.set_xlabel("INR value")

    ################################################
    # DEAD-ENDS RELATED GRAPHS
    ################################################

    @staticmethod
    def plot_flags(buffer_data, state_method, model_d, model_r, delta_r, delta_d, delta_ry, delta_dy, ax=None,
                   start_time=None, end_time=None, end_time_range=None, start_time_range=None, return_perc_only=False):

        buffer_data = Model.get_network_values(buffer_data, state_method, model_d, model_r, delta_r, delta_d, delta_ry,
                                               delta_dy)

        red_flags = buffer_data[buffer_data["Red_Flag"] == 1]
        yellow_flags = buffer_data[buffer_data["Yellow_Flag"] == 1]

        # TODO: asssert that start time and end time cannot both be None

        # Last step
        if start_time is not None:
            last_steps = buffer_data[buffer_data["TIME_TO_START"] == start_time]
            print(f"Red flags distribution in measurement week: {start_time}. {len(last_steps)} samples")
        elif start_time_range is not None:
            last_steps = buffer_data[(buffer_data["TIME_TO_START"] >= start_time_range[0]) & (
                        buffer_data["TIME_TO_START"] <= start_time_range[1])]
            print(
                f"Red flags distribution for weeks {start_time_range[0]} to {start_time_range[1]} from the last measurement. {len(last_steps)} samples")
        elif end_time_range is not None:
            last_steps = buffer_data[
                (buffer_data["TIME_TO_END"] >= end_time_range[0]) & (buffer_data["TIME_TO_END"] <= end_time_range[1])]
            print(
                f"Red flags distribution for weeks {end_time_range[0]} to {end_time_range[1]} from the last measurement. {len(last_steps)} samples")
        elif end_time is not None:
            last_steps = buffer_data[buffer_data["TIME_TO_END"] == end_time]
            print(f"Red flags distribution for {end_time} weeks from the last measurement. {len(last_steps)} samples")
        else:
            last_steps = buffer_data[buffer_data["TIME_TO_END"] == 0]
            print(f"Red flags distribution in last measurement week. {len(last_steps)} samples")

        red_flags = last_steps[last_steps["Red_Flag"] == 1]
        agg_red_flags = red_flags["SURVIVOR_FLAG"].value_counts() / last_steps["SURVIVOR_FLAG"].value_counts()
        y_true = np.array(1 - last_steps["SURVIVOR_FLAG"])
        y_pred = np.array(last_steps["Red_Flag"])
        total_counts = len(last_steps)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity_r = tp / (tp + fn)
        specificity_r = tn / (tn + fp)

        yellow_flags = last_steps[last_steps["Yellow_Flag"] == 1]
        agg_yellow_flags = yellow_flags["SURVIVOR_FLAG"].value_counts() / last_steps["SURVIVOR_FLAG"].value_counts()
        y_true = np.array(1 - last_steps["SURVIVOR_FLAG"])
        y_pred = np.array(last_steps["Yellow_Flag"])
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity_y = tp / (tp + fn)
        specificity_y = tn / (tn + fp)

        print(f"Red Flags    | Sensitivity: {sensitivity_r:,.1%}, Specificity: {specificity_r:,.1%}")
        print(f"Yellow Flags | Sensitivity: {sensitivity_y:,.1%}, Specificity: {specificity_y:,.1%}")

        x = np.array([agg_red_flags[1], agg_yellow_flags[1]])
        y = np.array([agg_red_flags[0], agg_yellow_flags[0]])

        if return_perc_only:
            return x, y

        if ax is None:
            fig = plt.figure(figsize=(5, 4))
            ax = fig.gca()

        x_w = np.empty(x.shape)
        x_w.fill(1 / x.shape[0])
        y_w = np.empty(y.shape)
        y_w.fill(1 / y.shape[0])

        # Width of a bar 
        width = 0.3
        ind = np.array([0, 1])

        # Plotting
        if len(x):
            ax.bar(ind, x, width, label='Survivors', color="g")
        if len(y):
            ax.bar(ind + width, y, width, label='Nonsurvivors', color="b")

        ax.set_xlabel(' ')
        ax.set_ylabel('% of Group')
        # ax.set_title(f"")

        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(["Red Flag", "Yellow Flag"])

        # Finding the best position for legends and putting it
        ax.legend(loc='best')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_ylim([0, 1])

    #     @staticmethod
    #     def plot_sample_patient(model_d, model_r, state_dim, state_method, data, pat_id="52257.01.0", delta_r=0.75,
    #                             delta_d=-0.25, delta_ry=0.85, delta_dy=-0.15):

    #         pat_data = data[data["USUBJID_O_NEW"] == pat_id]

    #         inr_series = np.array(pat_data["INR_VALUE"] * 4 + 0.5)
    #         x = np.arange(len(inr_series))
    #         fig = plt.figure(figsize=(7, 4))
    #         ax = fig.gca()
    #         ax.plot(x, inr_series)
    #         ax.axhspan(2, 3, color='grey', alpha=0.2)
    #         ax.set_ylim([0.25, 4.75])
    #         plt.show()

    #         temp_state = ReplayBufferDed.get_state(pat_data, method=state_method)

    #         q, imt, i = model_r.model(torch.FloatTensor(temp_state))
    #         imt = imt.exp()
    #         imt = (imt / imt.max(1, keepdim=True)[0] > model_r.bcq_threshold).float()
    #         values = np.array([[x] for x in torch.max(imt * q + (1. - imt) * -1e8, 1)[0].to("cpu").detach().numpy()])
    #         medians = np.array([[x] for x in torch.median(q, 1)[0].to("cpu").detach().numpy()])
    #         medians_r = deepcopy(medians)

    #         scaled_median_r = Model.clip_values(medians, Constants.clip_r)
    #         scaled_value_r = Model.clip_values(values, Constants.clip_r)

    #         q, imt, i = model_d.model(torch.FloatTensor(temp_state))
    #         imt = imt.exp()
    #         imt = (imt / imt.max(1, keepdim=True)[0] > model_r.bcq_threshold).float()
    #         values = np.array([[x] for x in torch.max(imt * q + (1. - imt) * -1e8, 1)[0].to("cpu").detach().numpy()])
    #         medians = np.array([[x] for x in torch.median(q, 1)[0].to("cpu").detach().numpy()])
    #         medians_d = deepcopy(medians)

    #         flag = np.where(np.logical_or(medians_r <= delta_r, medians_d <= delta_d), 1, 0)
    #         yellow_flag = np.where(np.logical_or(medians_r <= delta_ry, medians_d <= delta_dy), 1, 0)

    #         scaled_median_d = Model.clip_values(medians, Constants.clip_d)
    #         scaled_value_d = Model.clip_values(values, Constants.clip_d)

    #         adj_median_d = np.floor(10 * (1 + scaled_median_d[:, 0]))
    #         adj_value_d = np.floor(10 * (1 + scaled_value_d[:, 0]))

    #         adj_median_r = np.floor(10 * (scaled_median_r[:, 0]))
    #         adj_value_r = np.floor(10 * (scaled_value_r[:, 0]))

    #         fig = plt.figure(figsize=(7, 4))
    #         plt.plot(adj_median_d, label="Median D-Network")
    #         plt.plot(adj_median_r, label="Median R-Network")
    #         plt.legend()
    #         plt.ylim([0, 10.5])
    #         plt.show()

    #         fig = plt.figure(figsize=(7, 4))
    #         plt.plot(adj_value_d, label="Max D-Network")
    #         plt.plot(adj_value_r, label="Max R-Network")
    #         plt.legend()
    #         plt.ylim([0, 10.5])

    #         fig = plt.figure(figsize=(7, 4))
    #         x_step = np.arange(len(yellow_flag))
    #         plt.step(x_step, yellow_flag, label="Yellow Flag", color="y", where = 'post')
    #         plt.step(x_step, flag, label="Red Flag", color="red", where = 'post')
    #         plt.legend()
    #         plt.ylim([-0.1, 1.1])

    #     @staticmethod
    #     def plot_value_histograms_full(train_buffer, val_data, model_d, model_r, min_time, max_time, state_method=2,
    #                                    only_final=False, scale_values=True):
    #         #######################
    #         # Survivors vs nonsurvivors
    #         #######################
    #         val_data["TIME_TO_END"] = val_data.groupby('USUBJID_O_NEW').STUDY_WEEK.transform('max') - val_data.STUDY_WEEK

    #         nonsurvivor_ids = val_data[(val_data[Constants.neg_reward_events].sum(axis=1) > 0)]["USUBJID_O_NEW"].unique()
    #         subset_nonsurvivors = val_data[
    #             (val_data["TIME_TO_END"] >= min_time) & (val_data["TIME_TO_END"] <= max_time) & (
    #                 val_data["USUBJID_O_NEW"].isin(nonsurvivor_ids))]
    #         temp_state = ReplayBufferDed.get_state(subset_nonsurvivors, method=state_method)
    #         subset_survivors = val_data[
    #             (val_data["TIME_TO_END"] >= min_time) & (val_data["TIME_TO_END"] <= max_time) & ~(
    #                 val_data["USUBJID_O_NEW"].isin(nonsurvivor_ids))]
    #         temp_state_survivors = ReplayBufferDed.get_state(subset_survivors, method=state_method)

    #         num_nonsurvivors = subset_nonsurvivors['USUBJID_O_NEW'].nunique()
    #         nonsurvivors_samples = subset_nonsurvivors.shape[0]
    #         num_survivors = subset_survivors['USUBJID_O_NEW'].nunique()
    #         survivors_samples = subset_survivors.shape[0]

    #         print(
    #             f"{num_nonsurvivors} nonsurvivors ({nonsurvivors_samples} samples) and {num_survivors} survivors ({survivors_samples} samples) for: {Constants.neg_reward_events[0]}")
    #         print(f"For weeks: {min_time} to {max_time}")

    #         #######################
    #         # D-network
    #         #######################
    #         if len(temp_state):
    #             q, imt, i = model_d.model(torch.FloatTensor(temp_state))
    #             imt = imt.exp()
    #             imt = (imt / imt.max(1, keepdim=True)[0] > model_d.bcq_threshold).float()
    #             values = np.array([[x] for x in torch.max(imt * q + (1. - imt) * -1e8, 1)[0].to("cpu").detach().numpy()])
    #             medians = np.array([[x] for x in torch.median(q, 1)[0].to("cpu").detach().numpy()])
    #         else:
    #             values = np.array([])
    #             medians = np.array([])

    #         if len(temp_state_survivors):
    #             q, imt, i = model_d.model(torch.FloatTensor(temp_state_survivors))
    #             imt = imt.exp()
    #             imt = (imt / imt.max(1, keepdim=True)[0] > model_d.bcq_threshold).float()
    #             values_survivors = np.array(
    #                 [[x] for x in torch.max(imt * q + (1. - imt) * -1e8, 1)[0].to("cpu").detach().numpy()])
    #             medians_survivors = np.array([[x] for x in torch.median(q, 1)[0].to("cpu").detach().numpy()])
    #         else:
    #             values_survivors = np.array([])
    #             medians_survivors = np.array([])

    #         if scale_values:
    #             scaled_median_survivors_d = Model.clip_values(medians_survivors, Constants.clip_d)
    #             scaled_value_survivors_d = Model.clip_values(values_survivors, Constants.clip_d)

    #             scaled_median_d = Model.clip_values(medians, Constants.clip_d)
    #             scaled_value_d = Model.clip_values(values, Constants.clip_d)
    #         else:
    #             scaled_median_survivors_d = medians_survivors
    #             scaled_value_survivors_d = values_survivors

    #             scaled_median_d = medians
    #             scaled_value_d = values

    #         if not only_final:
    #             fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    #             Graphs.plot_value_bins(1 + scaled_median_survivors_d, 1 + scaled_median_d, ax[0],
    #                                    "D-Network Median Q-values (V_D)")
    #             Graphs.plot_value_bins(1 + scaled_value_survivors_d, 1 + scaled_value_d, ax[1], "D-Network Max Q-values (Q_D)")

    #         #######################
    #         # R-network
    #         #######################
    #         if len(temp_state):
    #             q, imt, i = model_r.model(torch.FloatTensor(temp_state))
    #             imt = imt.exp()
    #             imt = (imt / imt.max(1, keepdim=True)[0] > model_r.bcq_threshold).float()
    #             values = np.array([[x] for x in torch.max(imt * q + (1. - imt) * -1e8, 1)[0].to("cpu").detach().numpy()])
    #             medians = np.array([[x] for x in torch.median(q, 1)[0].to("cpu").detach().numpy()])
    #         else:
    #             values_survivors = np.array([[]])
    #             medians_survivors = np.array([[]])

    #         if len(temp_state_survivors):
    #             q, imt, i = model_r.model(torch.FloatTensor(temp_state_survivors))
    #             imt = imt.exp()
    #             imt = (imt / imt.max(1, keepdim=True)[0] > model_r.bcq_threshold).float()
    #             values_survivors = np.array(
    #                 [[x] for x in torch.max(imt * q + (1. - imt) * -1e8, 1)[0].to("cpu").detach().numpy()])
    #             medians_survivors = np.array([[x] for x in torch.median(q, 1)[0].to("cpu").detach().numpy()])
    #         else:
    #             values_survivors = np.array([[]])
    #             medians_survivors = np.array([[]])

    #         if scale_values:
    #             scaled_median_survivors_r = Model.clip_values(medians_survivors, Constants.clip_r)
    #             scaled_value_survivors_r = Model.clip_values(values_survivors, Constants.clip_r)

    #             scaled_median_r = Model.clip_values(medians, Constants.clip_r)
    #             scaled_value_r = Model.clip_values(values, Constants.clip_r)
    #         else:
    #             scaled_median_survivors_r = medians_survivors
    #             scaled_value_survivors_r = values_survivors

    #             scaled_median_r = medians
    #             scaled_value_r = values

    #         if not only_final:
    #             fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    #             Graphs.plot_value_bins(scaled_median_survivors_r, scaled_median_r, ax[0], "R-Network Median Q-values (V_R)")
    #             Graphs.plot_value_bins(scaled_value_survivors_r, scaled_value_r, ax[1], "R-Network Max Q-values (Q_R)")

    #         #######################
    #         # KS-statistic between survivors and nonsurvivors
    #         #######################
    #         ks_df = pd.DataFrame(columns=['network', 'ks_stat', 'pval'])

    #         if len(temp_state) > 0 and len(temp_state_survivors) > 0:
    #             # D-network
    #             dist1 = 1 + scaled_median_survivors_d[:, 0]
    #             dist2 = 1 + scaled_median_d[:, 0]
    #             ks_stat, pval = stats.ks_2samp(dist1, dist2)
    #             ks_df = ks_df.append({"network": "D-network Median", "ks_stat": ks_stat, "pval": pval}, ignore_index=True)

    #             # R-network
    #             dist1 = scaled_median_survivors_r[:, 0]
    #             dist2 = scaled_median_r[:, 0]
    #             ks_stat, pval = stats.ks_2samp(dist1, dist2)
    #             ks_df = ks_df.append({"network": "R-network Median", "ks_stat": ks_stat, "pval": pval}, ignore_index=True)

    #             # Combined
    #             dist1 = (1 + scaled_median_survivors_d[:, 0]) + scaled_median_survivors_r[:, 0]
    #             dist2 = 1 + scaled_median_d[:, 0] + scaled_median_r[:, 0]
    #             ks_stat, pval = stats.ks_2samp(dist1, dist2)
    #             ks_df = ks_df.append({"network": "Combined Median", "ks_stat": ks_stat, "pval": pval}, ignore_index=True)

    #             # D-network
    #             dist1 = 1 + scaled_value_survivors_d[:, 0]
    #             dist2 = 1 + scaled_value_d[:, 0]
    #             ks_stat, pval = stats.ks_2samp(dist1, dist2)
    #             ks_df = ks_df.append({"network": "D-network Max", "ks_stat": ks_stat, "pval": pval}, ignore_index=True)

    #             # R-network
    #             dist1 = scaled_value_survivors_r[:, 0]
    #             dist2 = scaled_value_r[:, 0]
    #             ks_stat, pval = stats.ks_2samp(dist1, dist2)
    #             ks_df = ks_df.append({"network": "R-network Max", "ks_stat": ks_stat, "pval": pval}, ignore_index=True)

    #             # Combined
    #             dist1 = (1 + scaled_value_survivors_d[:, 0]) + scaled_value_survivors_r[:, 0]
    #             dist2 = 1 + scaled_value_d[:, 0] + scaled_value_r[:, 0]
    #             ks_stat, pval = stats.ks_2samp(dist1, dist2)
    #             ks_df = ks_df.append({"network": "Combined Max", "ks_stat": ks_stat, "pval": pval}, ignore_index=True)

    #             print(ks_df)

    #         #######################
    #         # On same histogram - Median
    #         #######################
    #         fig, ax = plt.subplots(1, 2, figsize=(15, 4))

    #         if len(temp_state_survivors):
    #             survivors_array_median = np.concatenate(
    #                 [1 + scaled_median_survivors_d[:, 0], scaled_median_survivors_r[:, 0]])
    #             survivors_array_value = np.concatenate([1 + scaled_value_survivors_d[:, 0], scaled_value_survivors_r[:, 0]])
    #         else:
    #             survivors_array_median = np.array([])
    #             survivors_array_value = np.array([])
    #         if len(temp_state):
    #             nonsurvivors_array_median = np.concatenate([1 + scaled_median_d[:, 0], scaled_median_r[:, 0]])
    #             nonsurvivors_array_value = np.concatenate([1 + scaled_value_d[:, 0], scaled_value_r[:, 0]])
    #         else:
    #             nonsurvivors_array_median = np.array([])
    #             nonsurvivors_array_value = np.array([])

    #         Graphs.plot_value_bins(survivors_array_median, nonsurvivors_array_median, ax[0],
    #                                f"Median Q-Values for {Constants.neg_reward_events[0]}")
    #         Graphs.plot_value_bins(survivors_array_value, nonsurvivors_array_value, ax[1],
    #                                f"Max Q-Values for {Constants.neg_reward_events[0]}")

    #     @staticmethod
    #     def plot_policy(df, policy):

    #         fig = plt.figure(figsize=(15, 10))

    #         # create a 2 X 2 grid
    #         gs = grd.GridSpec(3, 1, height_ratios=[3, 1, 1], wspace=0.1)

    #         # line plot
    #         ax2 = plt.subplot(gs[0])
    #         ax2.spines['right'].set_visible(False)
    #         ax2.spines['top'].set_visible(False)
    #         ax2.xaxis.set_ticks_position('bottom')
    #         ax2.yaxis.set_ticks_position('left')
    #         x = np.arange(len(policy))
    #         ax2.plot(x, df["INR_VALUE"], 'k', lw=1, label="Measured INR Value")
    #         ax2.fill_between(x, 2, 3,
    #                          facecolor="orange",
    #                          color='grey',
    #                          alpha=0.2)
    #         ax2.legend()
    #         ax2.set_ylabel("INR Value")

    #         # line plot
    #         ax2 = plt.subplot(gs[1])
    #         ax2.spines['right'].set_visible(False)
    #         ax2.spines['top'].set_visible(False)
    #         ax2.xaxis.set_ticks_position('bottom')
    #         ax2.yaxis.set_ticks_position('left')
    #         x = np.arange(len(policy))
    #         ax2.plot(x, df["ACTION"], 'k--', lw=1, label="Clinician")
    #         ax2.plot(x, policy, lw=1, label="RL Agent")
    #         ax2.legend()
    #         ax2.set_ylabel("Action")
    #         ax2.set_xlabel("Timestep")

    #         plt.show()

    #     @staticmethod
    #     def plot_policy_two_models(df, policy1, policy2, name1, name2):

    #         fig = plt.figure(figsize=(15, 12))

    #         # create a 2 X 2 grid
    #         gs = grd.GridSpec(3, 1, height_ratios=[3, 1, 1], wspace=0.15)

    #         # line plot
    #         ax2 = plt.subplot(gs[0])
    #         ax2.spines['right'].set_visible(False)
    #         ax2.spines['top'].set_visible(False)
    #         ax2.xaxis.set_ticks_position('bottom')
    #         ax2.yaxis.set_ticks_position('left')
    #         x = np.arange(len(policy1))
    #         ax2.plot(x, df["INR_VALUE"], 'k', lw=1, label="Measured INR Value")
    #         ax2.fill_between(x, 2, 3,
    #                          facecolor="orange",
    #                          color='grey',
    #                          alpha=0.2)
    #         ax2.legend()
    #         ax2.set_ylabel("INR Value")

    #         # line plot
    #         ax2 = plt.subplot(gs[1])
    #         ax2.spines['right'].set_visible(False)
    #         ax2.spines['top'].set_visible(False)
    #         ax2.xaxis.set_ticks_position('bottom')
    #         ax2.yaxis.set_ticks_position('left')
    #         x = np.arange(len(policy1))
    #         ax2.plot(x, df["ACTION"], 'k--', lw=1, label="Clinician")
    #         ax2.plot(x, policy1, lw=1, label=f"{name1}")
    #         ax2.legend(loc="lower right")
    #         ax2.set_ylabel("Action")

    #         # line plot
    #         ax2 = plt.subplot(gs[2])
    #         ax2.spines['right'].set_visible(False)
    #         ax2.spines['top'].set_visible(False)
    #         ax2.xaxis.set_ticks_position('bottom')
    #         ax2.yaxis.set_ticks_position('left')
    #         x = np.arange(len(policy2))
    #         ax2.plot(x, df["ACTION"], 'k--', lw=1, label="Clinician")
    #         ax2.plot(x, policy2, lw=1, label=f"{name2}")
    #         ax2.legend(loc="lower right")
    #         ax2.set_ylabel("Action")
    #         ax2.set_xlabel("Timestep")

    #         plt.show()

    @staticmethod
    def plot_policy_scatter(df):

        fig = plt.figure(figsize=(15, 7))

        colname = "POLICY_ACTION"
        df[colname] = model.get_model_actions(
            np.array(SMDPReplayBuffer.get_state(buffer.data[buffer.data.USUBJID_O_NEW == patient_id])))
        plt.scatter(df[df[colname] == 0]["STUDY_WEEK"], df[df[colname] == 0]["INR_VALUE"], c="r",
                    label="Decrease Dosage")
        plt.scatter(df[df[colname] == 1]["STUDY_WEEK"], df[df[colname] == 1]["INR_VALUE"], c="b",
                    label="Maintain Dosage")
        plt.scatter(df[df[colname] == 2]["STUDY_WEEK"], df[df[colname] == 2]["INR_VALUE"], c="g",
                    label="Increase Dosage")

        # line plot
        x = df["STUDY_WEEK"]
        plt.plot(x, df["INR_VALUE"], 'k', lw=0.5, label="Measured INR Value")
        plt.fill_between(x, 2, 3,
                         facecolor="orange",
                         color='grey',
                         alpha=0.15)
        plt.legend()
        plt.xlabel("Timestep")
        plt.ylabel("INR Value")

    @staticmethod
    def plot_sample_policy(model, patient_id=None, model2=None, name1=None, name2=None):

        if patient_id is None:
            ids_list = data.merged_data.USUBJID_O_NEW
            patient_id = random.choice(ids_list)
            print(f"Selected patient ID: {patient_id}")

        state_method = Constants.state_method_map[model.state_dim]
        df = data.merged_data[data.merged_data.USUBJID_O_NEW == patient_id]
        policy = model.get_model_actions(
            np.array(SMDPReplayBuffer.get_state(buffer.data[buffer.data.USUBJID_O_NEW == patient_id], method=state_method)))

        if model2 is None:
            plot_policy(df, policy)

        else:
            state_method = Constants.state_method_map[model.state_dim]
            policy2 = model2.get_model_actions(np.array(
                SMDPReplayBuffer.get_state(buffer.data[buffer.data.USUBJID_O_NEW == patient_id], method=state_method)))
            plot_policy_two_models(df, policy, policy2, name1, name2)

    @staticmethod
    def plot_smooth_ucurves(buffer_data, model, show_ci=False):

        state_method = Constants.state_method_map[model.state_dim]
        policy = \
        model.get_model_actions(np.array(SMDPReplayBuffer.get_state(buffer_data, method=state_method))).transpose()[0]

        buffer_data['EVENT_NEXT_STEP'] = np.minimum(1, buffer_data.groupby('USUBJID_O_NEW')[
            Constants.neg_reward_events].shift(-1).sum(axis=1))
        print(buffer_data['EVENT_NEXT_STEP'].value_counts())

        actions_df = pd.DataFrame({"ID": buffer_data["USUBJID_O_NEW"], "INR_VALUE_NORM": buffer_data["INR_VALUE"],
                                   "CONTINENT_EAST ASIA": buffer_data["CONTINENT_EAST ASIA"],
                                   "CONTINENT_EASTERN EUROPE": buffer_data["CONTINENT_EASTERN EUROPE"],
                                   "CONTINENT_WESTERN EUROPE": buffer_data["CONTINENT_WESTERN EUROPE"],
                                   "SEX": buffer_data["SEX"],
                                   'EVENT_NEXT_STEP': buffer_data['EVENT_NEXT_STEP'],
                                   "CLINICIAN_ACTION_CTS": buffer_data["WARFARIN_DOSE_MULT"],
                                   "SURVIVOR_FLAG": buffer_data["SURVIVOR_FLAG"],
                                   "CLINICIAN_ACTION": buffer_data["ACTION"],
                                   "POLICY_ACTION": policy})

        actions_df["NEXT_INR"] = actions_df.groupby('ID')['INR_VALUE_NORM'].shift(-1)
        actions_df["NEXT_INR_INRANGE"] = np.where(
            np.logical_and(actions_df['NEXT_INR'] >= 0.375, actions_df['NEXT_INR'] <= 0.625), 1, 0)
        actions_df = actions_df.dropna()

        df = pd.DataFrame({'subject_id': actions_df['ID'],
                           'clinician_action': actions_df['CLINICIAN_ACTION_CTS'],
                           'model_action': actions_df['POLICY_ACTION'],
                           'event_before_next_timestep': actions_df['EVENT_NEXT_STEP'].astype(bool),
                           })

        num_before = df.shape[0]
        # Filter out large clinician actions (usually because the dosage shot up from 0)
        df = df[df['clinician_action'] < 10]
        print(f"Dropping {num_before - df.shape[0]} large doses. {df.shape[0]:,.0f} entries left")

        df["model_action"] = df["model_action"].map({
            0: "Decrease >10%",
            1: "Decrease <10%",
            2: "Keep Same",
            3: "Increase <10%",
            4: "Increase >10%"
        })

        """
        Computation for the plot.
        """
        # Window size - number of timesteps to consider per non-overlapping window.
        window_size = 5

        # Kernel bandwidth for exponential smoothing - higher = more smooth, lower = less smooth
        bandwidth = 1e-3

        # Number of bootstrap samples for computing 95% CIs
        num_bootstrap_samples = 15

        # Compute the empirical (under the clinician policy) means for our 5 action bins and map the
        # discrete actions to the means of the bins
        means = {
            "Decrease >10%": np.mean(df["clinician_action"][df["clinician_action"] < 0.9]),
            "Decrease <10%": np.mean(df["clinician_action"][(df["clinician_action"] >= 0.9) &
                                                            (df["clinician_action"] < 1.)]),
            "Keep Same": np.mean(df["clinician_action"][df["clinician_action"] == 1.]),
            "Increase <10%": np.mean(df["clinician_action"][(df["clinician_action"] > 1.) &
                                                            (df["clinician_action"] <= 1.1)]),
            "Increase >10%": np.mean(df["clinician_action"][df["clinician_action"] > 1.1])
        }
        df["model_action_cts"] = df["model_action"].map(means)

        # Compute the differences
        df["difference"] = df["model_action_cts"] - df["clinician_action"]

        # Prune rows for samples after events have occurred
        idx = df.groupby("subject_id")["event_before_next_timestep"].idxmax()
        idx = idx[df.loc[idx].set_index("subject_id")["event_before_next_timestep"] == True]
        sel_remove = df.reset_index().apply(
            lambda x: x["index"] > idx[x["subject_id"]] if x["subject_id"] in idx else False,
            axis=1
        )

        # Add this to align the indices 
        sel_remove.index = df.index

        df = df.loc[~sel_remove]

        # Compute windows, and exclude rows so they're non-overlapping
        df_window = df.groupby("subject_id").rolling(window_size)[
            ["event_before_next_timestep", "difference"]
        ].mean().dropna()
        df_window = df_window.reset_index()
        sel = (df_window.set_index("subject_id")["level_1"] -
               df_window.groupby("subject_id")["level_1"].max()) % window_size == 0
        df_window = df_window.set_index("subject_id").loc[sel].drop("level_1", axis=1)

        # Event occurred if mean > 0
        df_window["event_before_next_timestep"] = (df_window["event_before_next_timestep"] > 0.).astype(int)

        # Remove subject ID and sort by difference
        df_window = df_window.reset_index().drop("subject_id", axis=1).sort_values(by="difference")

        # Compute kernel for exp smoothing
        diffs = np.asarray(df_window["difference"])
        #     kernel = np.exp(-(np.subtract.outer(diffs, diffs)**2) / bandwidth)
        #     smoothed_occurrences = np.matmul(kernel, np.asarray(df_window["event_before_next_timestep"]))
        #     df_window["smoothed_event_rate"] = smoothed_occurrences / kernel.sum(axis=1) * 100.

        kernel = np.asfortranarray(np.exp(-(np.subtract.outer(diffs, diffs) ** 2) / bandwidth))
        occurrences = np.asarray(df_window["event_before_next_timestep"])
        df_window["smoothed_event_rate"] = np.matmul(kernel, occurrences) / kernel.sum(axis=1) * 100.

        df_window["difference"] = df_window["difference"] * 100.

        """
        Plotting.
        """
        if show_ci:
            # 95% CI via bootstrap samples
            samples = np.zeros((kernel.shape[0], num_bootstrap_samples))
            for i in tqdm(range(num_bootstrap_samples)):
                idx = np.random.choice(len(df_window), replace=True, size=len(df_window))
                k = kernel[:, idx]
                samples[:, i] = np.matmul(k, occurrences[idx]) / k.sum(axis=1) * 100.
            ci_lower, ci_upper = np.quantile(samples, [0.025, 0.975], axis=1)
            df_window["smoothed_event_rate_lower"] = ci_lower
            df_window["smoothed_event_rate_upper"] = ci_upper

            # Plot
            g = ggplot(df_window) + \
                geom_line(aes(x="difference", y="smoothed_event_rate")) + \
                xlim([-50, 50]) + ylim([0., 7.]) + \
                xlab("Model - Clinician (% Dose Change)") + \
                ylab("Windowed Event Rate (%)") + \
                geom_ribbon(aes(x="difference", ymin="smoothed_event_rate_lower", ymax="smoothed_event_rate_upper"),
                            alpha=0.2) + \
                ggtitle(f"UCurve for: {Constants.neg_reward_events}")

        else:
            # Plot
            g = ggplot(df_window) + \
                geom_line(aes(x="difference", y="smoothed_event_rate")) + \
                xlim([-50, 50]) + ylim([0., 5.]) + \
                xlab("Model Proposed Dose Change (%) - Clinician Proposed Dose Change (%)") + \
                ylab("Event Rate (%)") + \
                ggtitle(f"UCurve for: {Constants.neg_reward_events}")

        fig = g.draw()
        return g, df_window

    @staticmethod
    def create_heatmap_helper_mod(df, x_axis, y_axis, xlabel=None, ylabel=None, annot=False, norm_across_dataset=False):
        # Counts the number of rows (transitions) for each x, y pair 
        df2 = df.groupby([y_axis, x_axis], as_index=False).size().rename(
            columns={'size': "Count"})
        df2.loc[df2['Count'] == 0, 'Count'] = np.nan
        df_p = pd.pivot_table(df2, 'Count', y_axis, x_axis)

        #     df_p = df_p.iloc[::-1]
        try:
            df_p = df_p[['<0', '0', '>0']]
        except Exception as e:
            pass
        df_p = df_p.reindex([">0", "0", "<0"])

        df_norm = df_p.div(df_p.sum(axis=1), axis=0)
        if norm_across_dataset:
            print("Norm across dataset...")
            df_norm2 = df_p / df_p.sum().sum()
        else:
            df_norm2 = df_p.div(df_p.sum(axis=0), axis=1)

        _, axs = plt.subplots(ncols=1, nrows=1, figsize=(7, 5))
        sns.heatmap(df_norm2, cmap="Blues", annot=annot, fmt=".0%", ax=axs, vmin=0, vmax=1)

        if xlabel is not None:
            axs.set_xlabel(xlabel)
        if ylabel is not None:
            axs.set_ylabel(ylabel)

        return axs

