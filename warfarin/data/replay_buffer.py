"""Replay buffers represent the data in the format for the RL model."""

from __future__ import annotations

from typing import Optional

from warnings import warn

import pandas as pd

import numpy as np

import torch
from torch.utils.data import TensorDataset

from warfarin import config
from warfarin.utils import code_quantitative_decision
from warfarin.data.feature_engineering import (engineer_state_features,
                                               extract_observed_decision,
                                               compute_k,
                                               compute_reward,
                                               compute_done,
                                               compute_sample_probability)


class WarfarinReplayBuffer(TensorDataset):
    """
    Represent the data in a format suitable for RL modeling.

    Specifically, we take in the processed and merged dataset (specifically,
    either the train, val or test split) and build the state, option, reward,
    and transition flags needed for training.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 discount_factor: float,
                 option_means: dict = None,
                 min_trajectory_length: int = 1,
                 state_transforms = None,
                 rel_event_sample_prob: int = 1,
                 weight_option_frequency: bool = False,
                 time_varying: str = "across",
                 include_dose_time_varying: bool = False,
                 device: str = "cpu",
                 seed: int = 42) -> None:
        """
        Args:
            df: The processed data frame, containing the information required to
                compute the transitions.
            discount_factor: The discount factor used to compute option rewards.
            option_means: The mean dose change of each option, if not to be
                          computed from the data.
            state_transforms: If not None, will use these transforms to engineer
                              state features.
            min_trajectory_length: The minimum length of trajectories in the
                                   buffer.
            rel_event_sample_prob: The factor by which to increase the
                                   probability of sampling transitions from
                                   trajectories with adverse events. When set to
                                   1 (default), this corresponds to no
                                   difference between events transitions and
                                   non-events transitions.
            weight_option_frequency: If True, will ignore rel_event_sample_prob
                                     and the probability of sampling each
                                     transition will be inversely proportional
                                     to the frequency of occurrence of the
                                     option in the replay buffer.
            time_varying: "across" if across trajectories, "within" or None if
                          within trajectories.
            include_dose_time_varying: Whether to include the dose in time-
                                       varying features. Potentially sub-optimal
                                       as it represents a violation of standard
                                       Markov assumptions.
            device: The device to store the data on.
            seed: The seed for randomly sampling batches.
        """
        self.df = df.set_index(["TRIAL", "SUBJID", "TRAJID", "STUDY_DAY"])
        self.discount_factor = discount_factor
        self.rel_event_sample_prob = rel_event_sample_prob
        self.weight_option_frequency = weight_option_frequency

        self.time_varying = "within" if time_varying is None else time_varying
        self.include_dose_time_varying = include_dose_time_varying

        self._option_means = None

        # The devices batches will reside on
        self.device = device

        # Seeded RNG for reproducibility
        self.rng = np.random.default_rng(seed)

        # Feature engineering transforms
        self.state_transforms = state_transforms

        # Threshold for eliminating short trajectories
        self.min_trajectory_length = min_trajectory_length

        # Data that makes up each transition
        k, s, o, ns, r, nd, p = self._extract_transitions()
        self.k = k
        self.state = s
        self.option = o
        self.next_state = ns
        self.reward = r
        self.not_done = nd
        self.sample_prob = p

        # Number of transitions in the buffer
        self.size = len(self.state)

        # Cache for tensor representations of the buffer for fast training/eval
        self._tensors = {
            "k": torch.from_numpy(
                np.array(self.k).astype(np.float32)[:, np.newaxis]
            ).to(self.device),
            "state": torch.from_numpy(
                np.array(self.state).astype(np.float32)
            ).to(self.device),
            "option": torch.from_numpy(
                np.array(self.option).astype(int)[:, np.newaxis]
            ).to(self.device),
            "next_state": torch.from_numpy(
                np.array(self.next_state).astype(np.float32)
            ).to(self.device),
            "reward": torch.from_numpy(
                np.array(self.reward).astype(np.float32)[:, np.newaxis]
            ).to(self.device),
            "not_done": torch.from_numpy(
                np.array(self.not_done).astype(np.float32)[:, np.newaxis]
            ).to(self.device)
        }

        # Initialize as torch dataset to support on-device dataloading
        super().__init__(self._tensors["k"],
                         self._tensors["state"],
                         self._tensors["option"],
                         self._tensors["next_state"],
                         self._tensors["reward"],
                         self._tensors["not_done"])

    def _extract_transitions(self):
        """
        Extract the observed state, next state, options, reward, time elapsed,
        trajectory done flags, and transition sample probabilities.
        """
        self._raw_df = self.df.copy()

        # Subset to trajectories with min length
        sel = self.df.groupby(
            ["TRIAL", "SUBJID", "TRAJID"]
        )["INR_VALUE"].count() > self.min_trajectory_length
        self.df = self.df.loc[sel].copy()

        # When an adverse event that we don't split on occurs in the middle of
        # a trajectory, it may have a missing INR and/or dose. To ensure the
        # dynamics around this point are correct, we remove that row of the
        # data frame, as if we're not splitting on it, the assumption is that
        # the surrounding dose-response is correct, and no clinical decision
        # was made at the time.
        nonsplittable_events = np.setdiff1d(config.EVENTS_TO_KEEP,
                                            config.EVENTS_TO_SPLIT)
        if len(nonsplittable_events) > 0:
            last = self.df.reset_index().groupby(
                ["TRIAL", "SUBJID", "TRAJID"]
            )["STUDY_DAY"].max().reset_index()
            last["LAST"] = True
            last = last.set_index(["TRIAL", "SUBJID", "TRAJID", "STUDY_DAY"])
            last = self.df.join(last)["LAST"].fillna(False)
            is_nonsplittable_event = (
                self.df[nonsplittable_events].sum(axis=1) > 0
            )
            inr_null = self.df["INR_VALUE"].isnull()
            dose_null = self.df["WARFARIN_DOSE"].isnull()
            remove_sel = (~last &
                          is_nonsplittable_event &
                          (inr_null | dose_null))
            self.df = self.df.loc[~remove_sel].copy()

        # Generate features
        state, self.state_transforms = engineer_state_features(
            self.df.copy(),
            time_varying_cross=(self.time_varying == "across"),
            include_dose_time_varying=self.include_dose_time_varying,
            transforms=self.state_transforms
        )
        next_state = state.groupby(
            ["TRIAL", "SUBJID", "TRAJID"]
        ).shift(-1).copy()
        option = extract_observed_decision(self.df.copy())
        k = compute_k(self.df.copy())
        reward = compute_reward(self.df.copy(), self.discount_factor)
        done = compute_done(self.df.copy())
        sample_prob = compute_sample_probability(self.df.copy(),
                                                 option.copy(),
                                                 self.rel_event_sample_prob,
                                                 self.weight_option_frequency)

        # Maintain all state and observed actions, even missing ones, for
        # evaluation
        self.observed_state = state
        self.observed_option = option

        # Count up states with missing values that are not adverse events
        nonmissing_state = (state.isnull().sum(axis=1) == 0)
        num_missing_states = (
            (~nonmissing_state) &
            (self.df[config.EVENTS_TO_KEEP].sum(axis=1) == 0)
        ).sum()
        if num_missing_states > 0:
            warn("Looks like there are missing state values in "
                 f"{num_missing_states} states. Probably a bad thing")

        # Subset to transitions with no missing data
        sel = (nonmissing_state &
               ~(option.isnull()) &
               (next_state.isnull().sum(axis=1) == 0) &
               ~(k.isnull()) &
               ~(reward.isnull()) &
               ~(done.isnull()) &
               ~(sample_prob.isnull()))
        k = k.loc[sel]
        state = state.loc[sel]
        option = option.loc[sel]
        next_state = next_state.loc[sel]
        reward = reward.loc[sel]
        not_done = 1. - done.loc[sel]
        sample_prob = (np.array(sample_prob.loc[sel]) /
                       np.sum(sample_prob.loc[sel]))

        return k, state, option, next_state, reward, not_done, sample_prob

    @property
    def state_dim(self):
        return self.state.shape[1]

    @property
    def option_means(self):
        if self._option_means:
            return self._option_means

        prev_dose = self.df["WARFARIN_DOSE"]
        dose = self.df.groupby(
            ["TRIAL", "SUBJID", "TRAJID"]
        )["WARFARIN_DOSE"].shift(-1)
        obs_action_quant = dose / prev_dose
        obs_action_quant[(prev_dose == 0.) & (dose == 0.)] = 1.
        df = obs_action_quant.to_frame()
        option = code_quantitative_decision(obs_action_quant)
        df["OPTION"] = option
        option_means = df[np.isfinite(df["WARFARIN_DOSE"])].groupby(
            "OPTION"
        )["WARFARIN_DOSE"].mean().to_dict()

        return option_means
