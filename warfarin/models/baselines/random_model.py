import numpy as np


class RandomModel:
    """
    Implements a model that provides a random action.
    """

    def __init__(self, seed: int = 42):
        """
        Args:
            seed: The seed to the RNG.
        """
        self.rng = np.random.default_rng(seed)
    
    def select_action(self, num):
        """
        Return the chosen actions.

        Args:
            num: The number of actions to return.

        Returns:
            x: The randomly selected actions.
        """
        x = self.rng.choice(np.arange(7).astype(int), size=num)

        return x
