"""Baseline model which always chooses a random action."""

import numpy as np


class RandomModel:
    """
    Implements a model that provides a random action.
    """

    def __init__(self, observed: np.ndarray, seed: int = 42):
        """
        Args:
            observed: The observations, so that the frequencies of each action
                      can be computed.
            seed: The seed to the RNG.
        """
        self._actions, _counts = np.unique(observed, return_counts=True)
        self._prob = _counts / len(observed)

        self.rng = np.random.default_rng(seed)

    def select_action(self, num):
        """
        Return the chosen actions.

        Args:
            num: The number of actions to return.

        Returns:
            x: The randomly selected actions.
        """
        x = self.rng.choice(self._actions, p=self._prob, size=num)

        return x
