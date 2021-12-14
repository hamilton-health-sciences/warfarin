import scipy as sp
import scipy.stats

import pandas as pd

import numpy as np

import torch


def compute_cutpoints(x):
    """
    Compute the cutpoints based on frequency for an ordinal regression model.

    Args:
        x: The pd.Series representing the ordinal output variable, from which
           frequencies will be calculated. Order should be based on sorting.

    Returns:
        cutpoints_tensor: The tensor containing the cutpoints.
    """
    x = pd.Categorical(x, ordered=True)
    counts = x.value_counts().cumsum()[:-1]
    freqs = np.array(counts / len(x))
    cutpoints = sp.stats.logistic.ppf(freqs).astype(np.float32)
    cutpoints_tensor = torch.from_numpy(cutpoints)

    return cutpoints_tensor
