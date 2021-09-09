from __future__ import annotations

from typing import Optional

from warnings import warn

import pandas as pd

import numpy as np

from warfarin.data.feature_engineering import (engineer_state_features,
                                               extract_observed_decision,
                                               compute_k,
                                               compute_reward,
                                               compute_done,
                                               compute_sample_probability)


class WarfarinReplayBuffer:
    def __init__(self,
                 df: pd.DataFrame,
                 discount_factor: float,
                 rel_event_sample_prob: int = 1,
                 batch_size: int = None,
                 device: str = "cpu",
                 seed: int = 42) -> None:
        # Discount factor
        self.discount_factor = discount_factor

        # The size of batches to sample
        self.batch_size = batch_size

        # The devices batches will reside on
        self.device = device

        # Seeded RNG for reproducibility
        self.rng = np.random.default_rng(seed)

        # Data that makes up each transition
        k, s, o, ns, r, d, p = self._extract_transitions(df)
        self.k = k
        self.state = s
        self.option = o
        self.next_state = ns
        self.reward = r
        self.done = d
        self.sample_prob = p

        # Number of transitions in the buffer
        self.size = len(self.state)

    def _extract_transitions(self, df):
        id_cols = ["TRIAL", "SUBJID", "TRAJID", "STUDY_DAY"]
        df = df.set_index(id_cols)

        state, state_transform_params = engineer_state_features(df.copy())
        next_state = state.groupby(
            ["TRIAL", "SUBJID", "TRAJID"]
        ).shift(-1).copy()
        option = extract_observed_decision(df.copy())
        k = compute_k(df.copy())
        reward = compute_reward(df.copy(), self.discount_factor)
        done = compute_done(df.copy())
        sample_prob = compute_sample_probability(df.copy(),
                                                 rel_event_sample_prob)

        nonmissing_state = (state.isnull().sum(axis=1) == 0)
        num_missing_state_trans = len(nonmissing_state) - nonmissing_state.sum()
        if num_missing_state_trans > 0:
            warn("Looks like there are missing state values in "
                 f"{num_missing_state_trans} transitions. Probably a bad thing")

        # TODO we still need to maintain the un-subset data so that we can do
        # e.g. proper TTR calc of the full trajectory
        # Subset to transitions with no missing data
        sel = (nonmissing_state &
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
        done = done.loc[sel]
        sample_prob = sample_prob.loc[sel]

        return k, state, option, next_state, reward, done, sample_prob

    def sample(self, idx=None):
        if idx is None:
            norm_sample_p = self.sample_prob / self.sample_prob.sum()
            idx = self.rng.choice(np.arange(self.size).astype(int),
                                  size=self.batch_size,
                                  p=norm_sample_p)

        if not self._cache:
            k = np.array(self.k)
            state = np.array(self.state)
            option = np.array(self.option)
            next_state = np.array(self.next_state)
            reward = np.array(self.reward)
            done = np.array(self.done)

            self._cache["k"] = torch.from_numpy(k).to(self.device)
            self._cache["state"] = torch.from_numpy(state).to(self.device)
            self._cache["option"] = torch.from_numpy(option).to(self.device)
            self._cache["next_state"] = torch.from_numpy(
                next_state
            ).to(self.device)
            self._cache["reward"] = torch.from_numpy(reward).to(self.device)
            self._cache["done"] = torch.from_numpy(done).to(self.device)

        return (self._cache["k"][idx],
                self._cache["state"][idx],
                self._cache["option"][idx],
                self._cache["next_state"][idx],
                self._cache["reward"][idx],
                self._cache["done"][idx])
