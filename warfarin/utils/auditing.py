from warnings import warn

import os

import pandas as pd

from warfarin import config


class auditable:
    """
    Enabling auditing of individual preprocessing steps.
    """

    def __init__(self, *names):
        os.makedirs(config.AUDIT_PATH, exist_ok=True)
        self.names = names

    def __call__(self, f):
        """
        Args:
            f: A function that returns a dataframe.

        Returns:
            f_audit: A function that has the same signature as `f`, but writes
                     the dataframe(s) returned from `f` to `config.AUDIT_PATH`
                     if the config option is set.
        """
        if config.AUDIT_PATH is None:
            return f

        fxn_name = f.__name__
        def f_audit(*args, **kwargs):
            dfs = df = f(*args, **kwargs)
            if isinstance(df, pd.DataFrame):
                if self.names and len(self.names) == 1:
                    fn = f"{fxn_name}_{self.names[0]}"
                else:
                    fn = fxn_name
                output_fn = os.path.join(config.AUDIT_PATH, f"{fn}.feather")
                try:
                    df.to_feather(output_fn)
                except ValueError:
                    df.reset_index().to_feather(output_fn)
            elif isinstance(dfs, tuple):
                if self.names and len(self.names) == len(dfs):
                    fns = [f"{fxn_name}_{name}" for name in self.names]
                else:
                    fns = [f"{fxn_name}_{i}" for i in range(len(dfs))]
                for i, df in enumerate(dfs):
                    fn = fns[i]
                    output_fn = os.path.join(config.AUDIT_PATH,
                                             f"{fn}.feather")
                    try:
                        df.to_feather(output_fn)
                    except ValueError:
                        df.reset_index().to_feather(output_fn)
            else:
                warn("`auditable` was called on a function that doesn't return "
                     "a data frame. This probably means something is wrong.")

            return dfs

        return f_audit
