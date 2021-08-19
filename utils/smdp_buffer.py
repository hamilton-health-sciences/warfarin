import pandas as pd
import numpy as np
import time
import feather
import os
from warnings import warn 

from .config import *
from .utils import interpolate_daily


class SMDPReplayBuffer():

    def __init__(self, save_folder="warfarin_rl/utils/buffers/", filename="buffer_data", id_col="USUBJID_O_NEW", root_dir="../", storage_dir="./data/buffer_data/"):

        t0 = time.time()
#         self.save_folder = root_dir + save_folder
#         self.data_path = root_dir + "data/buffer_data/" + f"{filename}.feather"
        self.root_dir = root_dir
        self.save_folder = storage_dir
        self.data_path = storage_dir + f"{filename}.feather"

        self.id_col = id_col
        self.features_to_norm = ["WARFARIN_DOSE", "INR_VALUE", "AGE_DEIDENTIFIED", "WEIGHT"]
        self.features_ranges = {}
        self.data = None

    def get_data(self):
        try:
            t0 = time.time()
            self.data = feather.read_dataframe(self.data_path)
            t1 = time.time()
            print(f"Retrieved saved data. Took {(t1 - t0):,.2f} seconds")
            print(f"Buffer data: {self.data.shape}, {self.data[self.id_col].nunique()}")
        except Exception as e:
            print(f"Failed with Exception: {e}")

    def prepare_data(self, data, num_actions=3, incl_adverse_events=False, features_ranges={}, state_method=18):
        
        print(f"Removing un-used columns...")
        for col in ['SUBJID_NEW', 'FIRST_DAY', 'LAST_DAY', 'START_TRAJ_CUMU', 'END_TRAJ_CUMU', 'INTERRUPT', 'REMOVE_PRIOR', 'REMOVE_AFTER', 'INR_VALUE_PREV', 'WARFARIN_DOSE_CHANGE_SIGN', 'IS_NULL', 'USUBJID_O_NEW_NEW', 'INR_VALUE_CHANGE_SIGN', 'FLAG',
               'IS_NULL_CUMU', 'PREV_DOSE', 'REMOVE', 'INR_MEASURED', 'CUMU_MEASUR', 'MISSING_ID', 'START_TRAJ', 'END_TRAJ', 'SUBJID_NEW_2']:
            if col in data.columns:
                data = data.drop(columns=[col])
                print(f"\tDropping col: {col}")
            
        is_training = True if not features_ranges else False
        self.features_ranges = features_ranges
        self.data = data

        print(
            f"Preparing data for replay buffer... \n\t{self.data[self.id_col].nunique():,.0f} patients and {self.data.shape[0]:,.0f} weekly entries")
        t0 = time.time()

        for col in ["DELTA_INR", "WARFARIN_DOSE_PREV", "DELTA_INR_ADJ"]:
            try:
                self.data = self.data.drop(columns=[col])
            except Exception as e:
                pass

        # Reward
        print(f"Determining rewards at each option decision...")
        self.data.loc[:, "REWARD"] = SMDPReplayBuffer.get_reward(self.data, adverse_events=incl_adverse_events, state_method=state_method)

        # k (time elapsed) for SMDP
        print(f"Determining k for each option decision...")
        self.data.loc[:, "k"] = SMDPReplayBuffer.get_k(self.data, id_col=self.id_col)
        
        # Action
        print(f"Determining (clinician) action at option decision...")
        self.data.loc[:, 'WARFARIN_DOSE_MULT'] = self.data.groupby(self.id_col)['WARFARIN_DOSE'].shift(-1) / np.maximum(
            self.data[
                'WARFARIN_DOSE'], 0.0001)
        self.data.loc[:, "ACTION"] = SMDPReplayBuffer.get_action(self.data, num_actions=num_actions)

        # State
        print(f"Preparing state features at each option decision...")

        # Rankin_score is not used in the state space
        if "RANKIN_SCORE" in self.data.columns:
            self.data = self.data.drop(columns="RANKIN_SCORE")
        
        self.data = SMDPReplayBuffer.get_encodings(self.data.drop(columns="SUBJID"))

        if is_training:
            self.features_ranges, self.data = SMDPReplayBuffer.normalize_features(self.data, self.features_to_norm,
                                                                              features_ranges={})
            self.data = self.data.reset_index()
        else:
            self.data = SMDPReplayBuffer.normalize_features(self.data, self.features_to_norm,
                                                        self.features_ranges).reset_index()

        print(
            f"There are {self.data.dropna().shape[0]:,.0f} entries with NaN values (out of {self.data.shape[0]} entries)... \n\t {self.data.dropna().shape[0]:,.0f} samples")

        self.data = SMDPReplayBuffer.get_ttr(self.data)

        t1 = time.time()
        print(f"DONE preparing buffer data! Took {t1-t0:,.2f} seconds.")
        
    @staticmethod
    def get_k(df, id_col="SUBJID"):
        k = df.groupby(id_col)['STUDY_DAY'].diff(-1).abs()
        return k.values
    

    @staticmethod
    def load_buffers(buffer_name, suffix, is_ais=False, storage_dir="./data/buffer_data"):

        train_buffer = SMDPReplayBuffer(filename=buffer_name + "_train", storage_dir=storage_dir)
        val_buffer = SMDPReplayBuffer(filename=buffer_name + "_valid", storage_dir=storage_dir)
        test_buffer = SMDPReplayBuffer(filename=buffer_name + "_test", storage_dir=storage_dir)
        events_buffer = SMDPReplayBuffer(filename=buffer_name + "_events", storage_dir=storage_dir)

        ###########################################
        # Load buffers
        ###########################################
        if is_ais:
            print(f"Retrieving AIS buffers...")
        train_buffer.load_buffer(buffer_name=suffix, dataset="train", ais=is_ais)
        val_buffer.load_buffer(buffer_name=suffix, dataset="valid", ais=is_ais)
        if not is_ais:
            test_buffer.load_buffer(buffer_name=suffix, dataset="test")
        events_buffer.load_buffer(buffer_name=suffix, dataset="events", ais=is_ais)

        try:
            train_buffer.data = feather.read_dataframe(train_buffer.data_path)
            val_buffer.data = feather.read_dataframe(val_buffer.data_path)
            test_buffer.data = feather.read_dataframe(test_buffer.data_path)
            events_buffer.data = feather.read_dataframe(events_buffer.data_path)
        except Exception as e:
            warn("Unable to read buffer data with the following exception")
            print(e)

        return train_buffer, val_buffer, test_buffer, events_buffer
    

    @staticmethod
    def get_reward(df, discount_factor=0.99, state_method=18, events_reward=-10):
        """
        In the SMDP framework, we need the cumulative return of each underlying time step, discounted to each option decision.
        
        :param df:
        :param adverse_events:
        :param discount_factor:
        :param state_method:
        :param event_reward:
        :return:
        """
        
        print(f"\tInterpolating daily values...")
        df_exploded_merged = SMDPReplayBuffer.interpolate_daily(df, state_method)

        df_exploded_merged['INR_MEASURED'] = ~df_exploded_merged['INR_VALUE_BIN'].isnull()
        df_exploded_merged['CUMU_INR_MEASURED'] = df_exploded_merged.groupby('USUBJID_O_NEW')['INR_MEASURED'].cumsum()

        df_exploded_merged['IN_RANGE'] = np.logical_and(df_exploded_merged['INR_VALUE'] >= 2, df_exploded_merged['INR_VALUE'] <= 3)

        print(f"\tCalculating daily reward signals...")
        df_exploded_merged['REWARD'] = df_exploded_merged['IN_RANGE'] * INR_REWARD

        if adverse_events:
            df_exploded_merged['REWARD'] = np.where(df_exploded_merged.loc[:, ADV_EVENTS].sum(axis=1) > 0, event_reward, df_exploded_merged['REWARD'])

        print(f"\tGetting time elapsed between clinical visits...")
        first_days = df_exploded_merged.groupby(['USUBJID_O_NEW', 'CUMU_INR_MEASURED'])['STUDY_DAY'].first().reset_index().rename(columns={'STUDY_DAY': 'FIRST_STUDY_DAY'})
        df_exploded_merged = df_exploded_merged.merge(first_days, how='left', on=['USUBJID_O_NEW', 'CUMU_INR_MEASURED'])
        df_exploded_merged['t'] = df_exploded_merged['STUDY_DAY'] - df_exploded_merged['FIRST_STUDY_DAY'] - 1

        print(f"\tDiscounting rewards to clinical visits using disc factor and time elapsed...")
        df_exploded_merged['DISC_REWARD'] = (~df_exploded_merged['INR_MEASURED']) * (discount_factor ** df_exploded_merged['t'])

        disc_rewards = df_exploded_merged.groupby(['USUBJID_O_NEW', 'CUMU_INR_MEASURED'])['DISC_REWARD'].sum().reset_index()

        df['INR_MEASURED'] = ~df['INR_VALUE_BIN'].isnull()
        df['CUMU_INR_MEASURED'] = df.groupby('USUBJID_O_NEW')['INR_MEASURED'].cumsum()

        df = df.merge(disc_rewards, how='left', on=['USUBJID_O_NEW', 'CUMU_INR_MEASURED'])

        return df['DISC_REWARD'].values
    

    @staticmethod
    def get_ttr(df, colname='INR_VALUE'):

        df["BELOW_RANGE"] = np.where((df[colname] < 2), 1, 0)
        df["IN_RANGE"] = np.where((df[colname] >= 2) & (df[colname] <= 3), 1, 0)
        df["ABOVE_RANGE"] = np.where((df[colname] > 3), 1, 0)
        #         df["DELTA_INR"] = df[colname].shift(-1) - df[colname]

        df["TTR"] = df.groupby("USUBJID_O_NEW")["IN_RANGE"].cumsum() / (
                df.groupby("USUBJID_O_NEW")["IN_RANGE"].cumcount() + 1)
        df["CHANGE_IN_TTR"] = df.groupby("USUBJID_O_NEW")["TTR"].diff()
        df["CHANGE_IN_TTR_ADJ"] = - df["CHANGE_IN_TTR"]

        return df

    @staticmethod
    def get_action(df, colname="WARFARIN_DOSE_MULT", num_actions=3, action_space="percent"):
        """Returns the actions of each transition. 

        Determines and returns the actions for each of the df entries (each entry is a transition), based on the specified action space.

        Args:
            df: Dataframe of transitions.
            colname: The column which is used to calculate the action. 
                When action_space == 'percent', colname is the Warfarin dose as a multiple of the previous dose.
                When action_space == 'absolute', colname is the absolute Warfarin dose prescribed at that time.
            num_actions: When action_space == 'percent', 
                num_actions specifies the number of actions in the action space.
            action_space: One of ['percent', 'absolute'].

        Returns:
            A np.array of actions corresponding to each transition in the dataframe df, of dimension [num transitions, 1]. 
            The actions are integers.

        """

        if action_space == "percent":
            
            if num_actions not in [3,5,7]:
                warn(f"Could not understand num_actions given: {num_actions}, type: {type(num_actions)}. Will use 7 as default.")
                num_actions = 7
                
            if num_actions == 3:
                conditions = [
                    (df[colname] < 1),
                    (df[colname] == 1),
                    (df[colname] > 1)
                ]

            elif num_actions == 5:
                conditions = [
                    (df[colname] < 0.9),
                    (df[colname] >= 0.9) & (df[colname] < 1),
                    (df[colname] == 1),
                    (df[colname] > 1) & (df[colname] <= 1.1),
                    (df[colname] > 1.1)
                ]

            elif num_actions == 7:
                conditions = [
                    (df[colname] < 0.8),
                    (df[colname] >= 0.8) & (df[colname] < 0.9),
                    (df[colname] >= 0.9) & (df[colname] < 1),
                    (df[colname] == 1),
                    (df[colname] > 1) & (df[colname] <= 1.1),
                    (df[colname] > 1.1) & (df[colname] <= 1.2),
                    (df[colname] > 1.2)
                ]


            values = np.arange(0, len(conditions))
            return np.select(conditions, values)

        elif action_space == "absolute":
            pass

    def load_buffer(self, buffer_name="unknown", dataset=None, ais=False):
        print(f"\nLoading buffer: {buffer_name}...")
        t0 = time.time()

        file_path = f"{self.save_folder}{buffer_name}/{dataset}"

        self.k = np.load(f"{file_path}_k.npy")
        self.state = np.load(f"{file_path}_state.npy")
        self.action = np.load(f"{file_path}_action.npy")
        self.next_state = np.load(f"{file_path}_next_state.npy")
        self.reward = np.load(f"{file_path}_reward.npy")
        self.not_done = np.load(f"{file_path}_not_done.npy")

        if ais:
            self.obs_state = np.load(f"{file_path}_obs_state.npy")
            self.next_obs_state = np.load(f"{file_path}_next_obs_state.npy")
            
        t1 = time.time()
        print(f"Done loading buffer! Took {t1 - t0:,.2f} seconds")

    @staticmethod
    def get_encodings(df):
        for col in df.columns:
            if "Y" in df[col].unique():
                df[col] = np.where(df[col] == "Y", 1, 0)
        return pd.get_dummies(df.set_index("USUBJID_O_NEW"))

    @staticmethod
    def normalize_features(df, cols=None, features_ranges={}):
        cols_to_norm = cols if cols is not None else df.columns
        is_training = True if not features_ranges else False

        if is_training:
            for col in cols_to_norm:
                if (df[col].dtype.kind in 'biufc') and (col != "STUDY_WEEK") and (df[col].max() != df[col].min()):
                    features_ranges[col] = {'min': df[col].min(), 'max': df[col].max()}
                    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            return features_ranges, df

        else:
            print(f"Normalizing using training feature ranges...")
            for col in cols_to_norm:
                if col in features_ranges:
                    min_val = features_ranges[col]['min']
                    max_val = features_ranges[col]['max']
                    df[col] = np.minimum(np.maximum((df[col] - min_val) / (max_val - min_val), 0), 1)
            return df

    @staticmethod
    def get_state_features(sample, method):
        
        if (method == 0) or (method == 4) or (method == 5):
            state_cols = ["INR_VALUE", "USUBJID_O_NEW"]

        if (method == 8) or (method == 10):
            state_cols = ["INR_VALUE", "SEX", "BMED_ASPIRIN", "BMED_AMIOD", "DIABETES", "HX_CHF", "HYPERTENSION",
                          'HX_MI', "BMED_THIENO", "AGE_DEIDENTIFIED", "MINOR_BLEED_FLAG", "MAJOR_BLEED_FLAG"] + [x for x
                                                                                                                 in
                                                                                                                 sample.columns
                                                                                                                 if
                                                                                                                 "CONTINENT" in x] + [
                             x for x in sample.columns if "SMOKE" in x] + ["USUBJID_O_NEW"]
        if method == 9:
            state_cols = ["INR_VALUE", "SEX", "BMED_ASPIRIN", "BMED_AMIOD", "DIABETES", "HX_CHF", "HYPERTENSION",
                          'HX_MI', "BMED_THIENO", "AGE_DEIDENTIFIED", "MINOR_BLEED_FLAG", "MAJOR_BLEED_FLAG",
                          "HOSP_FLAG"] + [x for x in sample.columns if
                                          "CONTINENT" in x] + [x for x in sample.columns
                                                               if "SMOKE" in x] + ["USUBJID_O_NEW"]

        if method == 12:
            state_cols = ["INR_VALUE", "SEX", "BMED_ASPIRIN", "BMED_AMIOD", "DIABETES", "HX_CHF", "HYPERTENSION",
                          'HX_MI', "BMED_THIENO", "AGE_DEIDENTIFIED", "MINOR_BLEED_FLAG", "MAJOR_BLEED_FLAG",
                          "HOSP_FLAG"] + [x for x in sample.columns if
                                          "CONTINENT" in x] + [x for x in sample.columns
                                                               if "SMOKE" in x] + [x for x in sample.columns if
                                                                                   "WARFARIN_DOSE_BIN" in x] + [
                             "USUBJID_O_NEW"]

        if method == 13:
            state_cols = ["INR_VALUE", "SEX", "BMED_ASPIRIN", "BMED_AMIOD", "DIABETES", "HX_CHF", "HYPERTENSION",
                          'HX_MI', "BMED_THIENO", "AGE_DEIDENTIFIED", "MINOR_BLEED_FLAG", "MAJOR_BLEED_FLAG",
                          "HOSP_FLAG"] + [x for x in sample.columns if
                                          "CONTINENT" in x] + [x for x in sample.columns
                                                               if "SMOKE" in x] + [x for x in sample.columns if
                                                                                   "WARFARIN_DOSE_BIN" in x] + [x for x
                                                                                                                in
                                                                                                                sample.columns
                                                                                                                if
                                                                                                                "INR_VALUE_BIN" in x] + [
                             "USUBJID_O_NEW"]

        if (method == 14) or (method == 17):
            state_cols = ["INR_VALUE", "SEX", "BMED_ASPIRIN", "BMED_AMIOD", "DIABETES", "HX_CHF", "HYPERTENSION",
                          'HX_MI', "BMED_THIENO", "AGE_DEIDENTIFIED", "MINOR_BLEED_FLAG", "MAJOR_BLEED_FLAG",
                          "HOSP_FLAG", "WEIGHT"] + [x for x in sample.columns if
                                                    ("CONTINENT" in x) or ("SMOKE" in x) or ("WARFARIN_DOSE_BIN" in x)
                                                    or ("AGE_BIN" in x) or ("INR_VALUE_BIN" in x) or (
                                                                "WEIGHT_BIN" in x)] + ["USUBJID_O_NEW"]

        if (method == 15) or (method == 16):  # This one is pretty useless
            state_cols = ["INR_VALUE", "SEX", "BMED_ASPIRIN", "BMED_AMIOD", "DIABETES", "HX_CHF", "HYPERTENSION",
                          'HX_MI', "BMED_THIENO", "AGE_DEIDENTIFIED", "MINOR_BLEED_FLAG", "MAJOR_BLEED_FLAG",
                          "HOSP_FLAG", "WEIGHT", "WARFARIN_DOSE"] + [x for x in sample.columns if
                                                                     ("CONTINENT" in x) or ("SMOKE" in x) or (
                                                                                 "WARFARIN_DOSE_BIN" in x)
                                                                     or ("AGE_BIN" in x) or ("INR_VALUE_BIN" in x) or (
                                                                                 "WEIGHT_BIN" in x) or (
                                                                                 "AVG_TIME_ELAPSED_BIN" in x)] + [
                             "USUBJID_O_NEW"]

        if (method == 18) or (method == 19):
            state_cols = ["INR_VALUE", "SEX", "BMED_ASPIRIN", "BMED_AMIOD", "DIABETES", "HX_CHF", "HYPERTENSION",
                          'HX_MI', "BMED_THIENO", "AGE_DEIDENTIFIED", "MINOR_BLEED_FLAG", "MAJOR_BLEED_FLAG",
                          "HOSP_FLAG", "WEIGHT", "WARFARIN_DOSE"] + [x for x in sample.columns if
                                                                     ("CONTINENT" in x) or ("SMOKE" in x) or (
                                                                                 "WARFARIN_DOSE_BIN" in x)
                                                                     or ("AGE_BIN" in x) or ("INR_VALUE_BIN" in x) or (
                                                                                 "WEIGHT_BIN" in x)] + ["USUBJID_O_NEW"]

        if (method == 20) or (method == 21):
            state_cols = ["INR_VALUE", "SEX", "BMED_ASPIRIN", "BMED_AMIOD", "DIABETES", "HX_CHF", "HYPERTENSION",
                          'HX_MI', "BMED_THIENO", "AGE_DEIDENTIFIED", "WEIGHT", "WARFARIN_DOSE"] + [x for x in
                                                                                                    sample.columns if
                                                                                                    (
                                                                                                                "CONTINENT" in x) or (
                                                                                                                "SMOKE" in x) or (
                                                                                                                "WARFARIN_DOSE_BIN" in x)
                                                                                                    or (
                                                                                                                "AGE_BIN" in x) or (
                                                                                                                "INR_VALUE_BIN" in x) or (
                                                                                                                "WEIGHT_BIN" in x)] + [
                             "USUBJID_O_NEW"]

        return state_cols

    @staticmethod
    def get_state(sample, method=0, return_next=False, verbose=True):

        state_cols = SMDPReplayBuffer.get_state_features(sample, method)

        sample_state = sample[state_cols]
        sample_state["USUBJID_O_NEW"] = sample_state["USUBJID_O_NEW"].astype(object)

        #         if (method != 0) and (method != 5) and (method != 6) and (method != 7):
        if 1 <= method <= 4:
            sample_state.loc[:, "AVG_INR"] = sample_state.groupby('USUBJID_O_NEW')['INR_VALUE'].transform(
                lambda x: x.rolling(5, 1).mean())
            sample_state.loc[:, "DEV_INR"] = np.minimum((sample_state.groupby('USUBJID_O_NEW')['INR_VALUE'].transform(
                lambda x: x.rolling(5, 1).max() - x.rolling(5, 1).min())) / sample_state["AVG_INR"].fillna(0), 1)
            sample_state.loc[:, "CUM_AVG_INR"] = sample_state.groupby('USUBJID_O_NEW')['INR_VALUE'].apply(
                lambda x: x.shift().expanding().mean())
            sample_state.loc[:, "CUM_AVG_INR"] = np.where(sample_state["CUM_AVG_INR"].isnull(), sample_state['AVG_INR'],
                                                          sample_state['CUM_AVG_INR'])
            state_cols = state_cols + ["AVG_INR", "DEV_INR", "CUM_AVG_INR"]

        elif (method != 10) and (method != 11) and (method != 15) and (method != 17) and (method != 19) and (
                method != 21):
            sample_state.loc[:, "INR_1"] = sample_state.groupby('USUBJID_O_NEW')['INR_VALUE'].shift(1).fillna(
                sample_state["INR_VALUE"])
            sample_state.loc[:, "INR_2"] = sample_state.groupby('USUBJID_O_NEW')['INR_VALUE'].shift(2).fillna(
                sample_state["INR_1"])
            sample_state.loc[:, "INR_3"] = sample_state.groupby('USUBJID_O_NEW')['INR_VALUE'].shift(3).fillna(
                sample_state["INR_2"])
            sample_state.loc[:, "INR_4"] = sample_state.groupby('USUBJID_O_NEW')['INR_VALUE'].shift(4).fillna(
                sample_state["INR_3"])
            state_cols = state_cols + ["INR_1", "INR_2", "INR_3", "INR_4"]

        if return_next:
            next_state = sample_state.copy(deep=True)
            keep_cols = [x for x in state_cols if x != "USUBJID_O_NEW"]
            next_state[keep_cols] = sample_state.groupby("USUBJID_O_NEW")[state_cols].shift(-1)

            sample_state = sample_state.drop(columns="USUBJID_O_NEW")
            next_state = next_state.drop(columns="USUBJID_O_NEW")

        else:
            sample_state = sample_state.drop(columns="USUBJID_O_NEW")

        if verbose:
            print(f"State space dimension: {sample_state.shape[1]}")

        if return_next:
            return sample_state.values, next_state.values

        return sample_state.values

#     def create_buffer(self, sample=None, state_method=0, shuffle=True, return_flag=False):

#         if sample is None:
#             sample = self.data

#         event_reward = min(sample["REWARD"].values)
#         sample_k = sample['k'].values
#         sample_state, sample_next_state = (SMDPReplayBuffer.get_state(sample, method=state_method, return_next=True))
#         sample_reward = sample["REWARD"].values
#         sample_action = sample["ACTION"].values
#         sample_not_done = np.where(sample.groupby("USUBJID_O_NEW")["INR_VALUE"].shift(-1).isnull(), 0, 1)
#         sample_not_done = np.append(sample_not_done[1:], 0)
#         sample_event_flag = np.where(sample_reward == event_reward, 1, 0)

#         keep_entries1 = sample.groupby("USUBJID_O_NEW").cumcount(ascending=False) > 0
#         keep_entries2 = ~np.isnan(sample_state).any(axis=1)        
#         keep_entries3 = ~np.isnan(sample_next_state).any(axis=1)
#         keep_entries = np.logical_and(keep_entries1, keep_entries2, keep_entries3)

#         if sum(keep_entries2) > 0:
#             print(f"WARNING: There are NaNs in the state space - please investigate. ")
                   
#         sample_k = sample_k[keep_entries]
#         sample_state = sample_state[keep_entries]
#         sample_next_state = sample_next_state[keep_entries]
#         sample_reward = sample_reward[keep_entries]
#         sample_action = sample_action[keep_entries]
#         sample_not_done = sample_not_done[keep_entries]
#         sample_event_flag = sample_event_flag[keep_entries]

#         max_length = len(sample_next_state)
#         indices = np.arange(max_length)
#         if shuffle:
#             np.random.seed(42)
#             np.random.shuffle(indices)

#         sample_k = sample_k[:max_length][indices]
#         sample_not_done = sample_not_done[:max_length][indices]
#         sample_state = sample_state[:max_length][indices]
#         sample_reward = sample_reward[:max_length][indices]
#         sample_action = sample_action[:max_length][indices]
#         sample_next_state = sample_next_state[:max_length][indices]
#         sample_event_flag = sample_event_flag[:max_length][indices]

#         print(f"Created buffer. {len(sample_state):,.0f} samples")

#         if return_flag:
#             return sample_k, sample_state, sample_reward, sample_action, sample_next_state, sample_not_done, sample_event_flag
#         else:
#             return sample_k, sample_state, sample_reward, sample_action, sample_next_state, sample_not_done

#     def save_buffer(self, sample=None, buffer_name="unknown", dataset=None, state_method=0, return_flag=True):

#         print(f"\nSaving buffer: {buffer_name}...")
#         t0 = time.time()
#         if sample is None:
#             sample = self.data
#         if not os.path.exists(self.save_folder + buffer_name):
#             os.makedirs(self.save_folder + buffer_name)
#             print(f"Created new directory for buffer: {self.save_folder + buffer_name}")

#         if return_flag:
#             sample_k, sample_state, sample_reward, sample_action, sample_next_state, sample_not_done, sample_event_flag = self.create_buffer(
#                 sample,
#                 state_method, return_flag=return_flag)
#         else:
#             sample_k, sample_state, sample_reward, sample_action, sample_next_state, sample_not_done = self.create_buffer(
#                 sample,
#                 state_method)
#         max_length = len(sample_not_done)

#         if dataset is not None:
#             file_path = f"{self.save_folder}{buffer_name}/{dataset}_{buffer_name}"
#         else:
#             file_path = f"{self.save_folder}{buffer_name}/{buffer_name}"

#         np.save(f"{file_path}_k.npy", sample_k[:max_length].reshape((max_length, 1)))
#         np.save(f"{file_path}_state.npy", sample_state[:max_length])
#         np.save(f"{file_path}_action.npy", sample_action[:max_length].reshape((max_length, 1)))
#         np.save(f"{file_path}_next_state.npy", sample_next_state[:max_length])
#         np.save(f"{file_path}_reward.npy", sample_reward[:max_length].reshape((max_length, 1)))
#         np.save(f"{file_path}_not_done.npy", sample_not_done[:max_length].reshape((max_length, 1)))
#         if return_flag:
#             np.save(f"{file_path}_event_flag.npy", sample_event_flag[:max_length].reshape((max_length, 1)))
#         t1 = time.time()
#         print(f"Done saving buffer! Took {t1 - t0:,.2f} seconds")
