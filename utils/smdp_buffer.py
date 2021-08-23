"""Data storage in the format used for the RL model."""

import time

from warnings import warn

import numpy as np

import torch

import feather

import pandas as pd

from .config import INR_REWARD, ADV_EVENTS
from .utils import interpolate_daily


class SMDPReplayBuffer(object):
    """
    SMDP Replay Buffer

    This class has all the code related to creating, storing, loading, and
    sampling this replay buffer.
    """

    def __init__(self,
                 data_path=None,
                 id_col="USUBJID_O_NEW",
                 batch_size=None,
                 buffer_size=1e6,
                 device=None):
        """
        The batch size, buffer size, etc are only used when sampling from the
        replay buffer. They are not needed when creating the buffers.
        """

        self.batch_size = batch_size
        self.max_size = int(buffer_size)
        self.device = device
        self.data_path = data_path
        self.id_col = id_col

        self.features_to_norm = ["WARFARIN_DOSE", "INR_VALUE",
                                 "AGE_DEIDENTIFIED", "WEIGHT"]
        self.features_ranges = {}
        self.data = None

        self.ptr = 0
        self.crt_size = 0

#         self.k = np.zeros((self.max_size, 1))
#         self.state = np.zeros((self.max_size, state_dim))
#         self.action = np.zeros((self.max_size, 1))
#         self.next_state = np.array(self.state)
#         self.reward = np.zeros((self.max_size, 1))
#         self.not_done = np.zeros((self.max_size, 1))
#         self.event_flag = np.zeros((self.max_size, 1))


    def sample(self, ind=None, return_flag=False):
        ind = np.random.randint(
            0, self.crt_size, size=self.batch_size
        ) if ind is None else ind
        if return_flag:
            return (
                torch.FloatTensor(self.k[ind]).to(self.device),
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.LongTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device),
                torch.FloatTensor(self.event_flag[ind]).to(self.device)
            )
        else:
            return (
                torch.FloatTensor(self.k[ind]).to(self.device),
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.LongTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device)
            )

    def save(self):
        print(f"\nSaving buffer: {self.data_path}...")
        t0 = time.time()
        feather.write_dataframe(self.data, self.data_path)
        t1 = time.time()
        print(
            f"Done saving buffer! Took {t1 - t0:,.2f} seconds. Data stored "
            f"at: {self.data_path}"
        )


    def load(self, size=-1, incl_hist=True, seed=42, is_ais=False):

        if is_ais:
            pass
        else:
            # Load the dataframe from the feather format
            t0 = time.time()
            self.data = feather.read_dataframe(self.data_path)
            t1 = time.time()
            print(f"Retrieved saved data. Took {(t1 - t0):,.2f} seconds")
            print(
                f"Buffer data: {self.data.shape}, "
                f"{self.data[self.id_col].nunique()}"
            )

            # Convert from dataframe to replay buffer features
            (self.k, self.state, self.reward, self.action, self.next_state,
             self.not_done) = self.create_buffer(
                incl_hist=incl_hist, seed=seed
            )

            # Adjust crt_size if we"re using a custom size
            size = min(int(size), self.max_size) if size > 0 else self.max_size
            self.crt_size = min(self.reward.shape[0], size)

        print(f"Replay Buffer loaded with {self.crt_size} elements.")

    @staticmethod
    def load_buffers(data_dir, is_ais=False):
        train_buffer = SMDPReplayBuffer(data_path=data_dir + "train_data")
        val_buffer = SMDPReplayBuffer(data_path=data_dir + "train_data")
        test_buffer = SMDPReplayBuffer(data_path=data_dir + "train_data")
        events_buffer = SMDPReplayBuffer(data_path=data_dir + "train_data")

        ###########################################
        # Load buffers
        ###########################################
        train_buffer.load(is_ais=is_ais)
        val_buffer.load(is_ais=is_ais)
        test_buffer.load(is_ais=is_ais)
        events_buffer.load(is_ais=is_ais)

        return train_buffer, val_buffer, test_buffer, events_buffer

    def prepare_data(self,
                     data,
                     num_actions=7,
                     events_reward=None,
                     discount_factor=0.99):
        is_training = not self.features_ranges
        # self.features_ranges = features_ranges
        self.data = data

        print("Preparing data for replay buffer...")
        print(
            f"\t{self.data[self.id_col].nunique():,.0f} patients and "
            f"{self.data.shape[0]:,.0f} entries"
        )
        t0 = time.time()

        # Remove NaN entries
        state_cols = SMDPReplayBuffer.get_state_features(self.data,
                                                         incl_flags=True)
        dem_state_cols = [
            x for x in state_cols
            if ("INR_VALUE" not in x) and ("WARFARIN_DOSE" not in x)
        ]
        state_entries = self.data.loc[:, dem_state_cols]
        mask = state_entries.isnull().any(axis=1)
        self.data = self.data[~mask]
        print(
            f"\tMasking {mask.sum()} entries that have NaN demographic features"
        )

        if self.data[["INR_VALUE", "WARFARIN_DOSE"]].isnull().values.any():
            warn(
                "There are NaN values in the state space in INR or Warfarin "
                "dose - please investigate!"
            )

        # Reward
        print("Determining rewards at each option decision...")
        self.data.loc[:, "REWARD"] = SMDPReplayBuffer.get_reward(
            self.data,
            discount_factor=discount_factor,
            events_reward=events_reward
        )

        # k (time elapsed)
        print("Determining k for each option decision...")
        self.data.loc[:, "k"] = SMDPReplayBuffer.get_k(self.data,
                                                       id_col=self.id_col)

        # Action
        print("Determining (clinician) action at option decision...")
        self.data.loc[:, "WARFARIN_DOSE_MULT"] = self.data.groupby(
            self.id_col
        )["WARFARIN_DOSE"].shift(-1) / np.maximum(self.data["WARFARIN_DOSE"],
                                                  0.0001)
        self.data.loc[:, "ACTION"] = SMDPReplayBuffer.get_action(
            self.data,
            num_actions=num_actions
        )

        # State
        print("Preparing state features at each option decision...")

        # Rankin_score is not used in the state space
        if "RANKIN_SCORE" in self.data.columns:
            self.data = self.data.drop(columns="RANKIN_SCORE")

        self.data = SMDPReplayBuffer.get_encodings(
            self.data.drop(columns="SUBJID")
        )

        if is_training:
            features_ranges, self.data = SMDPReplayBuffer.normalize_features(
                self.data,
                self.features_to_norm,
                features_ranges={}
            )
            self.features_ranges = features_ranges
            self.data = self.data.reset_index()
        else:
            self.data = SMDPReplayBuffer.normalize_features(
                self.data,
                self.features_to_norm,
                self.features_ranges
            ).reset_index()

        # Done flag
        last_entries = self.data.groupby("USUBJID_O_NEW").apply(
            pd.DataFrame.last_valid_index
        ).values
        self.data["IS_LAST"] = 0
        self.data.loc[last_entries, "IS_LAST"] = 1
        self.data["DONE_FLAG"] = self.data["IS_LAST"].shift(-1)
        self.data = self.data.drop(columns="IS_LAST")

        # Event flag
        self.data["EVENT_OCCUR"] = self.data[ADV_EVENTS].sum(axis=1)
        self.data["EVENT_FLAG"] = self.data.groupby(
            "USUBJID_O_NEW"
        )["EVENT_OCCUR"].shift(-1).fillna(0)
        self.data = self.data.drop(columns="EVENT_OCCUR")

        print(
            f"There are {self.data.dropna().shape[0]:,.0f} entries with NaN "
            f"values (out of {self.data.shape[0]} entries)... \n\t "
            f"{self.data.dropna().shape[0]:,.0f} samples"
        )

        self.data = SMDPReplayBuffer.get_ttr(self.data)

        t1 = time.time()
        print(f"DONE preparing buffer data! Took {t1 - t0:,.2f} seconds.")

    @staticmethod
    def get_k(df, id_col="SUBJID"):
        k = df.groupby(id_col)["STUDY_DAY"].diff(-1).abs()

        return k.values

    @staticmethod
    def get_reward(df, discount_factor, events_reward=None):
        """
        In the SMDP framework, we need the cumulative return of each underlying
        time step, discounted to each option decision.

        :param df:
        :param adverse_events:
        :param discount_factor:
        :param event_reward:
        :return:
        """
        print("\tInterpolating daily values...")
        df_exploded_merged = interpolate_daily(df)

        df_exploded_merged["INR_MEASURED"] = ~df_exploded_merged[
            "INR_VALUE_BIN"
        ].isnull()
        df_exploded_merged["CUMU_INR_MEASURED"] = df_exploded_merged.groupby(
            "USUBJID_O_NEW"
        )["INR_MEASURED"].cumsum()

        df_exploded_merged["IN_RANGE"] = np.logical_and(
            df_exploded_merged["INR_VALUE"] >= 2,
            df_exploded_merged["INR_VALUE"] <= 3
        )

        print("\tCalculating daily reward signals...")
        df_exploded_merged["REWARD"] = (
            df_exploded_merged["IN_RANGE"] * INR_REWARD
        )

        if events_reward is not None:
            df_exploded_merged["REWARD"] = np.where(
                df_exploded_merged.loc[:, ADV_EVENTS].sum(axis=1) > 0,
                events_reward,
                df_exploded_merged["REWARD"]
            )

        print("\tGetting time elapsed between clinical visits...")
        first_days = df_exploded_merged.groupby(
            ["USUBJID_O_NEW", "CUMU_INR_MEASURED"]
        )["STUDY_DAY"].first().reset_index().rename(
            columns={"STUDY_DAY": "FIRST_STUDY_DAY"}
        )
        df_exploded_merged = df_exploded_merged.merge(
            first_days, how="left", on=["USUBJID_O_NEW", "CUMU_INR_MEASURED"]
        )
        df_exploded_merged["t"] = (df_exploded_merged["STUDY_DAY"] -
                                   df_exploded_merged["FIRST_STUDY_DAY"] - 1)

        print(
            "\tDiscounting rewards to clinical visits using disc factor and "
            "time elapsed..."
        )
        df_exploded_merged["DISC_REWARD"] = (
            ~df_exploded_merged["INR_MEASURED"]
        ) * (discount_factor ** df_exploded_merged["t"])

        disc_rewards = df_exploded_merged.groupby(
            ["USUBJID_O_NEW", "CUMU_INR_MEASURED"]
        )["DISC_REWARD"].sum().reset_index()

        df["INR_MEASURED"] = ~df["INR_VALUE_BIN"].isnull()
        df["CUMU_INR_MEASURED"] = df.groupby(
            "USUBJID_O_NEW"
        )["INR_MEASURED"].cumsum()

        df = df.merge(disc_rewards,
                      how="left",
                      on=["USUBJID_O_NEW", "CUMU_INR_MEASURED"])

        return df["DISC_REWARD"].values

    @staticmethod
    def get_ttr(df, colname="INR_VALUE"):
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
    def get_action(df, colname="WARFARIN_DOSE_MULT", num_actions=3):
        """
        Returns the actions of each transition.

        Determines and returns the actions for each of the df entries (each
        entry is a transition), based on the specified action space.

        Args:
            df: Dataframe of transitions.
            colname: The column which is used to calculate the action.
            num_actions: specifies the number of actions in the action space.

        Returns:
            An np.array of actions corresponding to each transition in the
            dataframe df, of dimension [num transitions, 1]. The actions are
            integers.
        """
        if num_actions not in [3, 5, 7]:
            warn(
                f"Could not understand num_actions given: {num_actions}, type: "
                f"{type(num_actions)}. Will use 7 as default."
            )
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

    @staticmethod
    def get_encodings(df):
        for col in df.columns:
            if ("Y" in df[col].unique()) or ("N" in df[col].unique()):
                df[col] = np.where(df[col] == "Y", 1, 0)
        return pd.get_dummies(df.set_index("USUBJID_O_NEW"))

    @staticmethod
    def normalize_features(df, cols=None, features_ranges={}):
        cols_to_norm = cols if cols is not None else df.columns
        is_training = not features_ranges

        if is_training:
            for col in cols_to_norm:
                if ((df[col].dtype.kind in "biufc") and
                    (col != "STUDY_WEEK") and
                    (df[col].max() != df[col].min())):
                    features_ranges[col] = {"min": df[col].min(),
                                            "max": df[col].max()}
                    df[col] = (
                        df[col] - df[col].min()
                    ) / (df[col].max() - df[col].min())
            return features_ranges, df

        else:
            print("Normalizing using training feature ranges...")
            for col in cols_to_norm:
                if col in features_ranges:
                    min_val = features_ranges[col]["min"]
                    max_val = features_ranges[col]["max"]
                    df[col] = np.minimum(
                        np.maximum(
                            (df[col] - min_val) / (max_val - min_val), 0
                        ),
                        1
                    )
            return df

    @staticmethod
    def get_state_features(sample, incl_flags):
        if incl_flags:
            state_cols = ["INR_VALUE", "SEX", "BMED_ASPIRIN", "BMED_AMIOD",
                          "DIABETES", "HX_CHF", "HYPERTENSION", "HX_MI",
                          "BMED_THIENO", "AGE_DEIDENTIFIED", "MINOR_BLEED_FLAG",
                          "MAJOR_BLEED_FLAG", "HOSP_FLAG", "WEIGHT",
                          "WARFARIN_DOSE"]
            state_cols += [x for x in sample.columns if
                           ("CONTINENT" in x) or
                           ("SMOKE" in x) or
                           ("WARFARIN_DOSE_BIN" in x) or
                           ("AGE_BIN" in x) or
                           ("INR_VALUE_BIN" in x) or
                           ("WEIGHT_BIN" in x)]
            state_cols += ["USUBJID_O_NEW"]

        else:
            state_cols = ["INR_VALUE", "SEX", "BMED_ASPIRIN", "BMED_AMIOD",
                          "DIABETES", "HX_CHF", "HYPERTENSION", "HX_MI",
                          "BMED_THIENO", "AGE_DEIDENTIFIED", "WEIGHT",
                          "WARFARIN_DOSE"]
            state_cols += [x for x in sample.columns if
                           ("CONTINENT" in x) or
                           ("SMOKE" in x) or
                           ("WARFARIN_DOSE_BIN" in x) or
                           ("AGE_BIN" in x) or
                           ("INR_VALUE_BIN" in x) or
                           ("WEIGHT_BIN" in x)]
            state_cols += ["USUBJID_O_NEW"]

        return state_cols

    @staticmethod
    def get_state(sample,
                  incl_flags=True,
                  incl_hist=True,
                  return_next=False,
                  verbose=True):
        state_cols = SMDPReplayBuffer.get_state_features(sample, incl_flags)

        sample_state = sample[state_cols]
        sample_state["USUBJID_O_NEW"] = sample_state["USUBJID_O_NEW"].astype(
            object
        )

        # Assumption: the INR levels are the same prior to the clinical visit
        # TODO: Make this a flag instead?
        # TODO: Do we need actions in the state space?
        if incl_hist:
            sample_state.loc[:, "INR_1"] = sample_state.groupby(
                "USUBJID_O_NEW"
            )["INR_VALUE"].shift(1).fillna(
                sample_state["INR_VALUE"]
            )
            sample_state.loc[:, "INR_2"] = sample_state.groupby(
                "USUBJID_O_NEW"
            )["INR_VALUE"].shift(2).fillna(
                sample_state["INR_1"]
            )
            sample_state.loc[:, "INR_3"] = sample_state.groupby(
                "USUBJID_O_NEW"
            )["INR_VALUE"].shift(3).fillna(
                sample_state["INR_2"]
            )
            sample_state.loc[:, "INR_4"] = sample_state.groupby(
                "USUBJID_O_NEW"
            )["INR_VALUE"].shift(4).fillna(
                sample_state["INR_3"]
            )
            state_cols = state_cols + ["INR_1", "INR_2", "INR_3", "INR_4"]

        if return_next:
            next_state = sample_state.copy(deep=True)
            keep_cols = [x for x in state_cols if x != "USUBJID_O_NEW"]
            next_state[keep_cols] = sample_state.groupby(
                "USUBJID_O_NEW"
            )[state_cols].shift(-1)

            sample_state = sample_state.drop(columns="USUBJID_O_NEW")
            next_state = next_state.drop(columns="USUBJID_O_NEW")

        else:
            sample_state = sample_state.drop(columns="USUBJID_O_NEW")

        if verbose:
            print(f"\tState space dimension: {sample_state.shape[1]}")

        if return_next:
            return sample_state.values, next_state.values

        return sample_state.values

    def create_buffer(self,
                      sample=None,
                      incl_hist=True,
                      shuffle=True,
                      return_flag=False,
                      seed=42):
        if sample is None:
            sample = self.data

        sample_k = sample["k"].values
        sample_state, sample_next_state = SMDPReplayBuffer.get_state(
            sample,
            incl_hist=incl_hist,
            return_next=True
        )
        sample_reward = sample["REWARD"].values
        sample_action = sample["ACTION"].values
        sample_not_done = 1 - sample["DONE_FLAG"].values
        sample_event_flag = sample["EVENT_FLAG"].values

        keep_entries1 = sample.groupby(
            "USUBJID_O_NEW"
        ).cumcount(ascending=False) > 0
        keep_entries2 = ~np.isnan(sample_state).any(axis=1)
        keep_entries3 = ~np.isnan(sample_next_state).any(axis=1)
        keep_entries = np.logical_and(keep_entries1,
                                      keep_entries2,
                                      keep_entries3)

        if sum(keep_entries2) > 0:
            warn("\tThere are NaNs in the state space - please investigate.")

        sample_k = sample_k[keep_entries]
        sample_state = sample_state[keep_entries]
        sample_next_state = sample_next_state[keep_entries]
        sample_reward = np.expand_dims(sample_reward[keep_entries], axis=1)
        sample_action = np.expand_dims(sample_action[keep_entries], axis=1)
        sample_not_done = sample_not_done[keep_entries]
        sample_event_flag = np.expand_dims(sample_event_flag[keep_entries],
                                           axis=1)

        max_length = len(sample_next_state)
        indices = np.arange(max_length)

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)

        # TODO: This would not work if k, not_done, reward, action, or
        # event_flag has more than 1 dimension or if state/next state only has 1
        # dimension
        sample_k = np.expand_dims(sample_k[:max_length][indices], axis=1)
        sample_not_done = np.expand_dims(sample_not_done[:max_length][indices],
                                         axis=1)
        sample_state = sample_state[:max_length][indices]
        sample_reward = np.expand_dims(sample_reward[:max_length][indices],
                                       axis=1)
        sample_action = np.expand_dims(sample_action[:max_length][indices],
                                       axis=1)
        sample_next_state = sample_next_state[:max_length][indices]
        sample_event_flag = np.expand_dims(
            sample_event_flag[:max_length][indices], axis=1
        )

        print(f"\tCreated buffer. {len(sample_state):,.0f} samples")

        if return_flag:
            return (sample_k, sample_state, sample_reward, sample_action,
                    sample_next_state, sample_not_done, sample_event_flag)
        else:
            return (sample_k, sample_state, sample_reward, sample_action,
                    sample_next_state, sample_not_done)

