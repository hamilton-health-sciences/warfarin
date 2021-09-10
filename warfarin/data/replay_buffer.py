"""Replay buffers represent the data in the format for the RL model."""

from __future__ import annotations

from typing import Optional

from warnings import warn

import pandas as pd

import numpy as np

import torch

from warfarin.data.feature_engineering import (engineer_state_features,
                                               extract_observed_decision,
                                               compute_k,
                                               compute_reward,
                                               compute_done,
                                               compute_sample_probability)


class WarfarinReplayBuffer:
    """
    Represent the data in a format suitable for RL modeling.

    Specifically, we take in the processed and merged dataset (specifically,
    either the train, val or test split) and build the state, option, reward,
    and transition flags needed for training.
    """

    # TODO paramterize event reward magnitude
    def __init__(self,
                 df: pd.DataFrame,
                 discount_factor: float,
                 rel_event_sample_prob: int = 1,
                 batch_size: Optional[int] = None,
                 device: str = "cpu",
                 seed: int = 42) -> None:
        """
        Args:
            df: The processed data frame, containing the information required to
                compute the transitions.
            discount_factor: The discount factor used to compute option rewards.
            rel_event_sample_prob: The factor by which to increase the
                                   probability of sampling transitions from
                                   trajectories with adverse events. When set to
                                   1 (default), this corresponds to no
                                   difference between events transitions and
                                   non-events transitions.
            batch_size: The size of the batch used during the training process.
            device: The device to store the data on.
            seed: The seed for randomly sampling batches.
        """
        self.df = df.set_index(["TRIAL", "SUBJID", "TRAJID", "STUDY_DAY"])
        self.discount_factor = discount_factor
        self.rel_event_sample_prob = rel_event_sample_prob

        # The size of batches to sample
        self.batch_size = batch_size

        # The devices batches will reside on
        self.device = device

        # Seeded RNG for reproducibility
        self.rng = np.random.default_rng(seed)

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
        self._cache = {}

    def _extract_transitions(self):
        """
        Extract the observed state, next state, options, reward, time elapsed,
        trajectory done flags, and transition sample probabilities.
        """
        # TODO actually use these state transform params!
        state, state_transform_params = engineer_state_features(self.df.copy())
        next_state = state.groupby(
            ["TRIAL", "SUBJID", "TRAJID"]
        ).shift(-1).copy()
        option = extract_observed_decision(self.df.copy())
        k = compute_k(self.df.copy())
        reward = compute_reward(self.df.copy(), self.discount_factor)
        done = compute_done(self.df.copy())
        sample_prob = compute_sample_probability(self.df.copy(),
                                                 self.rel_event_sample_prob)

        # Maintain all state and observed actions, even missing ones, for
        # evaluation
        self.observed_state = state
        self.observed_option = option

        nonmissing_state = (state.isnull().sum(axis=1) == 0)
        num_missing_state_trans = len(nonmissing_state) - nonmissing_state.sum()
        if num_missing_state_trans > 0:
            warn("Looks like there are missing state values in "
                 f"{num_missing_state_trans} transitions. Probably a bad thing")

        # TODO why are things null here

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
        sample_prob = sample_prob.loc[sel]

        return k, state, option, next_state, reward, not_done, sample_prob

    def sample(self, idx=None):
        """
        Sample a batch of transitions.

        Args:
            idx: If given, will sample the prescribed indices. Otherwise, will
                 sample a random batch of size `self.batch_size`.
        """
        if idx is None:
            norm_sample_p = self.sample_prob / self.sample_prob.sum()
            idx = self.rng.choice(np.arange(self.size).astype(int),
                                  size=self.batch_size,
                                  p=norm_sample_p)

        if not self._cache:
            k = np.array(self.k).astype(np.float32)[:, np.newaxis]
            state = np.array(self.state).astype(np.float32)
            option = np.array(self.option).astype(int)[:, np.newaxis]
            next_state = np.array(self.next_state).astype(np.float32)
            reward = np.array(self.reward).astype(np.float32)[:, np.newaxis]
            not_done = np.array(self.not_done).astype(np.float32)[:, np.newaxis]

            self._cache["k"] = torch.from_numpy(k).to(self.device)
            self._cache["state"] = torch.from_numpy(state).to(self.device)
            self._cache["option"] = torch.from_numpy(option).to(self.device)
            self._cache["next_state"] = torch.from_numpy(
                next_state
            ).to(self.device)
            self._cache["reward"] = torch.from_numpy(reward).to(self.device)
            self._cache["not_done"] = torch.from_numpy(not_done).to(self.device)

        return (self._cache["k"][idx],
                self._cache["state"][idx],
                self._cache["option"][idx],
                self._cache["next_state"][idx],
                self._cache["reward"][idx],
                self._cache["not_done"][idx])

    @property
    def state_dim(self):
        return self.state.shape[1]
