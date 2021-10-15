import pandas as pd

import numpy as np

from warfarin import config


def code_quantitative_decision(x):
    cond = [
        x < 0.8,
        x <= 0.9,
        x < 1.,
        x == 1.,
        x < 1.1,
        x <= 1.2,
        x > 1.2
    ]
    action = pd.Categorical(
        np.select(cond, config.ACTION_LABELS),
        categories=config.ACTION_LABELS,
        ordered=True
    )
    action_code = action.codes.astype(float)
    action_code[action_code < 0] = np.nan

    return action_code
