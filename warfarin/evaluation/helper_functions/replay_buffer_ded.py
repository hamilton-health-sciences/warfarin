import pandas as pd
import numpy as np
import time
import feather
import os

from .constants import Constants
from .replay_buffer import ReplayBuffer


class ReplayBufferDed():

    def __init__(self, save_folder="warfarin_rl/utils/buffers/", filename="buffer_data", id_col="USUBJID_O_NEW"):

        t0 = time.time()
        self.save_folder = save_folder
        self.data_path = "data/clean_data/" + f"{filename}.feather"

        self.id_col = id_col
        self.features_to_norm = ["WARFARIN_DOSE", "INR_VALUE", "DELTA_INR", "DELTA_INR_ADJ", "AGE_DEIDENTIFIED",
                                 "WARFARIN_DOSE_PREV"]
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

    def prepare_data(self, data, num_actions=3, neg_rewards=True, features_ranges={}):

        is_training = True if not features_ranges else False
        self.features_ranges = features_ranges
        self.data = data

        print(
            f"Preparing data for replay buffer... \n\t{self.data[self.id_col].nunique():,.0f} patients and {self.data.shape[0]:,.0f} weekly entries")
        t0 = time.time()

        # End trajectory if an event is experienced
        prev_size = self.data.shape[0]
        self.data = ReplayBufferDed.get_trajectories(self.data)
        print(f"Removed {prev_size - self.data.shape[0]} entries which occurred after adverse events")

        # Reward
        print(f"Determining rewards at each time step...")
        self.data.loc[:, "REWARD"] = ReplayBufferDed.get_reward(self.data, neg_rewards)

        # Action
        print(f"Determining (clinician) action at each time step...")
        self.data.loc[:, "ACTION"] = ReplayBufferDed.get_action(self.data, num_actions=num_actions)

        # State
        print(f"Determining state at each time step...")

        # Rankin_score is not used in the state space
        if "RANKIN_SCORE" in self.data.columns:
            self.data = self.data.drop(columns="RANKIN_SCORE")
        try:
            self.data = ReplayBufferDed.get_encodings(self.data.drop(columns="USUBJID_O"))
        except KeyError:
            self.data = ReplayBufferDed.get_encodings(self.data.drop(columns="SUBJID"))

        if is_training:
            try:
                self.features_ranges, self.data = ReplayBufferDed.normalize_features(self.data, self.features_to_norm,
                                                                                     features_ranges={})
            except:
                print(self.data)
                print(self.features_to_norm)
                print(ReplayBufferDed.normalize_features(self.data, self.features_to_norm))
            self.data = self.data.reset_index()
        else:
            self.data = ReplayBufferDed.normalize_features(self.data, self.features_to_norm,
                                                           self.features_ranges).reset_index()
        print(
            f"Dropping {self.data.shape[0] - self.data.dropna().shape[0]:,.0f} NaN entries... \n\t {self.data.dropna().shape[0]:,.0f} samples")
        # TODO: CHANGE THIS
        #         self.data = self.data.drop(columns=['RANKIN_SCORE'])
        #         self.data = self.data.dropna()
        self.data = ReplayBufferDed.get_ttr(self.data)

        # Save
        t1 = time.time()
        print(f"Saving data...")
        feather.write_dataframe(self.data, self.data_path)

        t2 = time.time()
        print(f"Done cleaning the data! Took {t2 - t0:,.2f} seconds, saving took {t2 - t1:,.2f} seconds")

    @staticmethod
    def get_trajectories(df, id_col='USUBJID_O_NEW'):
        subset = df[Constants.neg_reward_events + [id_col]]
        subset["SUM_EVENTS"] = subset[Constants.neg_reward_events].sum(axis=1)
        mask = subset.groupby(id_col)["SUM_EVENTS"].apply(lambda x: x.shift().eq(1).cumsum().eq(0))
        return df[mask]

    @staticmethod
    def get_reward(df, neg_rewards=True):

        colname = "INR_VALUE"
        df["BELOW_RANGE"] = np.where((df[colname] < 2), 1, 0)
        df["IN_RANGE"] = np.where((df[colname] >= 2) & (df[colname] <= 3), 1, 0)
        df["ABOVE_RANGE"] = np.where((df[colname] > 3), 1, 0)

        df_next_state = df.groupby("USUBJID_O_NEW").shift(-1)
        if neg_rewards:
            rewards = np.where(df_next_state[Constants.neg_reward_events].sum(axis=1) > 0, -1, 0)
        else:
            rewards = 1 + np.where(df_next_state[Constants.neg_reward_events].sum(axis=1) > 0, -1, 0)
        print(np.unique(rewards, return_counts=True))

        return rewards

    @staticmethod
    def get_ttr(df):

        df["TTR"] = df.groupby("USUBJID_O_NEW")["IN_RANGE"].cumsum() / (
                df.groupby("USUBJID_O_NEW")["IN_RANGE"].cumcount() + 1)
        df["CHANGE_IN_TTR"] = df.groupby("USUBJID_O_NEW")["TTR"].diff()
        df["CHANGE_IN_TTR_ADJ"] = - df["CHANGE_IN_TTR"]

        return df

    @staticmethod
    def get_action(df, colname="WARFARIN_DOSE_MULT", num_actions=3):

        if num_actions == 3:
            conditions = [
                (df[colname] < 1),
                (df[colname] == 1),
                (df[colname] > 1)
            ]

            values = np.arange(0, len(conditions))

        elif num_actions == 5:
            conditions = [
                (df[colname] < 0.9),
                (df[colname] >= 0.9) & (df[colname] < 1),
                (df[colname] == 1),
                (df[colname] > 1) & (df[colname] <= 1.1),
                (df[colname] > 1.1)
            ]
            values = np.arange(0, len(conditions))

        else:
            print(f"ERROR: Could not understand num_actions given: {num_actions}")
        # Use np.select to assign values to it using our lists as arguments
        return np.select(conditions, values)

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
        print(f"is training: {is_training}")

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
        if method == 0:
            state_cols = ["INR_VALUE", "USUBJID_O_NEW"]

        if method == 1:
            state_cols = ["INR_VALUE", "SEX", "BMED_ASPIRIN"] + [x for x in sample.columns if "CONTINENT" in x] + [x for
                                                                                                                   x in
                                                                                                                   sample.columns
                                                                                                                   if
                                                                                                                   "SMOKE" in x]

        if method == 2:
            state_cols = ["INR_VALUE", "SEX", "BMED_ASPIRIN", "BMED_AMIOD", "DIABETES", "HX_CHF", "HYPERTENSION",
                          'HX_MI'] + [x for x in sample.columns if "CONTINENT" in x] + [x for x in sample.columns if
                                                                                        "SMOKE" in x] + [
                             "USUBJID_O_NEW"]

        if method == 3:
            state_cols = ["INR_VALUE", "SEX", "BMED_ASPIRIN", "BMED_AMIOD", "DIABETES", "HX_CHF", "HYPERTENSION",
                          'HX_MI', "BMED_THIENO", "AGE_DEIDENTIFIED"] + [x for x in sample.columns if
                                                                         "CONTINENT" in x] + [x for x in sample.columns
                                                                                              if "SMOKE" in x] + [
                             "USUBJID_O_NEW"]
        if method == 4:
            state_cols = ["INR_VALUE", "USUBJID_O_NEW"]

        if method == 5:
            state_cols = ["INR_VALUE", "USUBJID_O_NEW"]

        if method == 6:
            state_cols = ["INR_VALUE", "SEX", "BMED_ASPIRIN", "BMED_AMIOD", "DIABETES", "HX_CHF", "HYPERTENSION",
                          'HX_MI', "BMED_THIENO", "AGE_DEIDENTIFIED"] + [x for x in sample.columns if
                                                                         "CONTINENT" in x] + [x for x in sample.columns
                                                                                              if "SMOKE" in x] + [
                             "USUBJID_O_NEW"]
        if method == 7:
            state_cols = ["INR_VALUE", "SEX", "BMED_ASPIRIN", "BMED_AMIOD", "DIABETES", "HX_CHF", "HYPERTENSION",
                          'HX_MI', "BMED_THIENO", "AGE_DEIDENTIFIED", "MINOR_BLEED_FLAG"] + [x for x in sample.columns
                                                                                             if
                                                                                             "CONTINENT" in x] + [x for
                                                                                                                  x in
                                                                                                                  sample.columns
                                                                                                                  if
                                                                                                                  "SMOKE" in x] + [
                             "USUBJID_O_NEW"]
        if method == 8:
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
        if method == 1000: 
            state_cols = [x for x in sample.columns if "HIDDEN" in x.split("_")]  + ["USUBJID_O_NEW"]

        return state_cols

    @staticmethod
    def get_state(sample, method=0, return_next=False):

        state_cols = ReplayBufferDed.get_state_features(sample, method)

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

        elif method < 1000:
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
            next_state[keep_cols] = sample_state[state_cols].groupby("USUBJID_O_NEW").shift(-1)

            sample_state = sample_state.drop(columns="USUBJID_O_NEW")
            next_state = next_state.drop(columns="USUBJID_O_NEW")

        else:
            sample_state = sample_state.drop(columns="USUBJID_O_NEW")

        print(f"State space dimension: {sample_state.shape[1]}")

        if return_next:
            return sample_state.values, next_state.values

        return sample_state.values

    def create_buffer(self, sample=None, state_method=0, shuffle=True, return_flag=False):

        if sample is None:
            sample = self.data

        #         nonsurvivor_ids = sample[(sample[Constants.neg_reward_events].sum(axis=1) > 0)]["USUBJID_O_NEW"].unique()
        event_reward = min(sample["REWARD"].values)
        sample_state, sample_next_state = (ReplayBufferDed.get_state(sample, method=state_method, return_next=True))
        sample_reward = sample["REWARD"].values
        sample_action = sample["ACTION"].values
        sample_not_done = np.where(sample.groupby("USUBJID_O_NEW")["INR_VALUE"].shift(-1).isnull(), 0, 1)
        sample_not_done = np.append(sample_not_done[1:], 0)
        sample_event_flag = np.where(sample_reward == event_reward, 1, 0)
        print(f"sanity check:")
        print(np.unique(sample_reward, return_counts=True))
        print(np.unique(sample_event_flag, return_counts=True))

        # Drop last entry 
        keep_entries = sample.groupby("USUBJID_O_NEW").cumcount(ascending=False) > 0
        sample_state = sample_state[keep_entries]
        sample_next_state = sample_next_state[keep_entries]
        sample_reward = sample_reward[keep_entries]
        sample_action = sample_action[keep_entries]
        sample_not_done = sample_not_done[keep_entries]
        sample_event_flag = sample_event_flag[keep_entries]

        # Drop entries where state has nan entries
        keep_entries = ~np.isnan(sample_state).any(axis=1)
        sample_state = sample_state[keep_entries]
        sample_next_state = sample_next_state[keep_entries]
        sample_reward = sample_reward[keep_entries]
        sample_action = sample_action[keep_entries]
        sample_not_done = sample_not_done[keep_entries]
        sample_event_flag = sample_event_flag[keep_entries]

        # Drop entries where next state has nan entries
        keep_entries = ~np.isnan(sample_next_state).any(axis=1)
        sample_state = sample_state[keep_entries]
        sample_next_state = sample_next_state[keep_entries]
        sample_reward = sample_reward[keep_entries]
        sample_action = sample_action[keep_entries]
        sample_not_done = sample_not_done[keep_entries]
        sample_event_flag = sample_event_flag[keep_entries]

        max_length = len(sample_next_state)
        indices = np.arange(max_length)
        if shuffle:
            np.random.seed(42)
            np.random.shuffle(indices)

        sample_not_done = sample_not_done[:max_length][indices]
        sample_state = sample_state[:max_length][indices]
        sample_reward = sample_reward[:max_length][indices]
        sample_action = sample_action[:max_length][indices]
        sample_next_state = sample_next_state[:max_length][indices]
        sample_event_flag = sample_event_flag[:max_length][indices]

        print(f"Created buffer. {len(sample_state):,.0f} samples")

        if np.isnan(np.sum(sample_state)):
            print("WARNING: null values found in state space - please resolve")

        if return_flag:
            return sample_state, sample_reward, sample_action, sample_next_state, sample_not_done, sample_event_flag
        else:
            return sample_state, sample_reward, sample_action, sample_next_state, sample_not_done

    def load_buffer(self, buffer_name="unknown", dataset=None):
        print(f"\nLoading buffer: {buffer_name}...")
        t0 = time.time()

        if dataset is not None:
            file_path = f"{self.save_folder}{buffer_name}/{dataset}_{buffer_name}"
        else:
            file_path = f"{self.save_folder}{buffer_name}/{buffer_name}"

        self.state = np.load(f"{file_path}_state.npy")
        self.action = np.load(f"{file_path}_action.npy")
        self.next_state = np.load(f"{file_path}_next_state.npy")
        self.reward = np.load(f"{file_path}_reward.npy")
        self.not_done = np.load(f"{file_path}_not_done.npy")

        t1 = time.time()
        print(f"Done loading buffer! Took {t1 - t0:,.2f} seconds")

    def save_buffer(self, sample=None, buffer_name="unknown", dataset=None, state_method=0, return_flag=True):

        print(f"\nSaving buffer: {buffer_name}...")
        t0 = time.time()
        if sample is None:
            sample = self.data
        if not os.path.exists(self.save_folder + buffer_name):
            os.makedirs(self.save_folder + buffer_name)
            print(f"Created new directory for buffer: {self.save_folder + buffer_name}")

        if return_flag:
            sample_state, sample_reward, sample_action, sample_next_state, sample_not_done, sample_event_flag = self.create_buffer(
                sample,
                state_method, return_flag=return_flag)
        else:
            sample_state, sample_reward, sample_action, sample_next_state, sample_not_done = self.create_buffer(sample,
                                                                                                                state_method)
        max_length = len(sample_not_done)

        if dataset is not None:
            file_path = f"{self.save_folder}{buffer_name}/{dataset}_{buffer_name}"
        else:
            file_path = f"{self.save_folder}{buffer_name}/{buffer_name}"

        np.save(f"{file_path}_state.npy", sample_state[:max_length])
        np.save(f"{file_path}_action.npy", sample_action[:max_length].reshape((max_length, 1)))
        np.save(f"{file_path}_next_state.npy", sample_next_state[:max_length])
        np.save(f"{file_path}_reward.npy", sample_reward[:max_length].reshape((max_length, 1)))
        np.save(f"{file_path}_not_done.npy", sample_not_done[:max_length].reshape((max_length, 1)))
        if return_flag:
            np.save(f"{file_path}_event_flag.npy", sample_event_flag[:max_length].reshape((max_length, 1)))
        t1 = time.time()
        print(f"Done saving buffer! Took {t1 - t0:,.2f} seconds")

    @staticmethod
    def split_data_ids(data, split, random_seed=42, id_col="SUBJID"):

        assert (len(split) == 2, "Too many values for split!")
        assert (split[0] + split[1] == 1, "Split percentages do not add up to 1")

        np.random.seed(random_seed)
        patient_ids = data[id_col].unique()
        np.random.shuffle(patient_ids)

        indx = int(split[0] * len(patient_ids))
        left_ids = patient_ids[:indx]
        right_ids = patient_ids[indx:]

        num_total = len(left_ids) + len(right_ids)
        num_left = len(left_ids)
        num_right = len(right_ids)

        print(
            f"First group: {num_left} patients ({num_left / num_total:,.2%}), Second group: {num_right} patients ({num_right / num_total:,.2%})")

        return left_ids, right_ids

    @staticmethod
    def split_data(data, split, random_seed=42, id_col="SUBJID"):

        assert (len(split) == 2, "Too many values for split!")
        assert (split[0] + split[1] == 1, "Split percentages do not add up to 1")

        left_ids, right_ids = ReplayBufferDed.split_data_ids(data, split, random_seed, id_col)
        left_data, right_data = data[data[id_col].isin(left_ids)], data[data[id_col].isin(right_ids)]

        num_left = left_data.shape[0]
        num_right = right_data.shape[0]
        num_total = num_left + num_right

        print(
            f"First group: {num_left} samples ({num_left / num_total:,.2%}), Second group: {num_right} samples ({num_right / num_total:,.2%})")

        return left_data, right_data

    def save_train_valid_split(self, save_buffer=False, random_seed=42, state_method=0, suffix=None,
                               split=[0.8, 0.2, 0]):

        ''' 
        Deprecated. This was previously used in an older iteration with a reduced dataset. To properly support train/valid/test split, the pipeline was rewritten. 
        '''

        if self.data is None:
            print(f"ERROR: Did not calculate merged data yet! :(")
            return None

        np.random.seed(random_seed)
        patient_ids = np.array(self.data["USUBJID_O_NEW"].unique())
        np.random.shuffle(patient_ids)
        indx = int(0.8 * len(self.data.USUBJID_O_NEW.unique()))
        train_ids = patient_ids[:indx]
        val_ids = patient_ids[indx:]
        print(f"{len(train_ids):,.0f} patients in training, {len(val_ids):,.0f} in validation")

        train_df = self.data[self.data.USUBJID_O_NEW.isin(train_ids)]
        valid_df = self.data[self.data.USUBJID_O_NEW.isin(val_ids)]

        if save_buffer:
            self.save_buffer(train_df, buffer_name=suffix, dataset="train", state_method=state_method)
            self.save_buffer(valid_df, buffer_name=suffix, dataset="valid", state_method=state_method)

        return train_df, valid_df