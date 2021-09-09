from typing import Optional

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
                 k: pd.Series,
                 state: pd.DataFrame,
                 option: pd.Series,
                 next_state: pd.DataFrame,
                 reward: pd.Series,
                 done: pd.Series,
                 sample_prob: Optional[pd.Series] = None,
                 device: str = "cpu",
                 batch_size: int = None,
                 seed: int = 42) -> None:
        # Data that makes up each transition
        self.k = k
        self.state = state
        self.option = option
        self.next_state = next_state
        self.reward = reward
        self.done = done

        # The number of transitions in the buffer
        self.size = len(self.state)

        # The relative probability of sampling a given transition (does not have
        # to sum to 1)
        if sample_prob is None:
            sample_prob = np.ones(self.size)
        self.sample_prob = sample_prob

        # The size of batches to sample
        self.batch_size = batch_size

        # The devices batches will reside on
        self.device = device

        # Seeded RNG for reproducibility
        self.rng = np.random.default_rng(seed)

    def sample(self, idx=None):
        if idx is None:
            norm_sample_p = self.sample_prob / self.sample_prob.sum()
            idx = self.rng.choice(np.arange(self.size).astype(int),
                                  size=self.batch_size,
                                  p=norm_sample_p)

        k = np.array(self.k.iloc[idx])
        state = np.array(self.state.iloc[idx])
        option = np.array(self.option.iloc[idx])
        next_state = np.array(self.next_state.iloc[idx])
        reward = np.array(self.next_state.iloc[idx])
        done = np.array(self.done.iloc[idx])

        return (torch.from_numpy(k).to(self.device),
                torch.from_numpy(state).to(self.device),
                torch.from_numpy(option).to(self.device),
                torch.from_numpy(next_state).to(self.device),
                torch.from_numpy(reward).to(self.device),
                torch.from_numpy(done).to(self.device))

    @staticmethod
    def from_data(df, rel_event_sample_prob: int = 1):
        id_cols = ["TRIAL", "SUBJID", "TRAJID", "STUDY_DAY"]
        df = df.set_index(id_cols)

        state, state_transform_params = engineer_state_features(df.copy())
        next_state = state.groupby(
            ["TRIAL", "SUBJID", "TRAJID"]
        ).shift(-1).copy()
        option = extract_observed_decision(df.copy())
        k = compute_k(df.copy())
        reward = compute_reward(df.copy())
        done = compute_done(df.copy())
        sample_prob = compute_sample_probability(df.copy(),
                                                 rel_event_sample_prob)

        import pdb; pdb.set_trace()
        # TODO what are remaining dose change nulls??

        return WarfarinReplayBuffer(k,
                                    state,
                                    option,
                                    next_state,
                                    reward,
                                    done,
                                    sample_prob,
                                    state_transform_params)
