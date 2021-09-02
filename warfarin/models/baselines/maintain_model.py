import numpy as np


class MaintainModel:
    """
    Dummy model that always returns the "maintain" option.
    """

    def select_action(self, num):
        return np.array([3] * num)
