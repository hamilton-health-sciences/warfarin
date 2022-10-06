import pandas as pd

import numpy as np

from warfarin import config


def code_quantitative_decision(x):
    cond = [
        x < 0.8,   # [-inf, -20%)
        x < 0.9,   # [-20%, -10%)
        x < 1.,    # [-10%, 0.)
        x == 1.,   # 0.
        x <= 1.1,  # (0., +10%]
        x <= 1.2,  # (+10%, +20%]
        x > 1.2    # (+20%, inf]
    ]
    action = pd.Categorical(
        np.select(cond, config.ACTION_LABELS),
        categories=config.ACTION_LABELS,
        ordered=True
    )
    action_code = action.codes.astype(float)
    action_code[action_code < 0] = np.nan

    return action_code
