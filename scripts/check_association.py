import pandas as pd

import numpy as np

import statsmodels.api as sm


def main(args):
    df = pd.read_csv(args.input_fn)

    df["POLICY_ACTION_AGREE"] = (df["POLICY_ACTION_DIFF"] < 0.05).astype(float)
    df.loc[df["POLICY_ACTION_DIFF"].isnull(), "POLICY_ACTION_AGREE"] = np.nan
    df["THRESHOLD_ACTION_AGREE"] = (df["THRESHOLD_ACTION_DIFF"] < 0.05).astype(float)
    df.loc[df["THRESHOLD_ACTION_DIFF"].isnull(), "THRESHOLD_ACTION_AGREE"] = np.nan

    model_df = df.groupby(["SUBJID", "TRAJID"])[
        ["POLICY_ACTION_AGREE", "THRESHOLD_ACTION_AGREE", "APPROXIMATE_TTR",
         "TRAJECTORY_LENGTH"]
    ].mean()

    model_df = model_df.dropna()

    print(
        sm.WLS(model_df["APPROXIMATE_TTR"],
               sm.add_constant(model_df["POLICY_ACTION_AGREE"]),
               weights=model_df["TRAJECTORY_LENGTH"]).fit().summary()
    )
    print(
        sm.WLS(model_df["APPROXIMATE_TTR"],
               sm.add_constant(model_df["THRESHOLD_ACTION_AGREE"]),
               weights=model_df["TRAJECTORY_LENGTH"]).fit().summary()
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fn", type=str, required=True)
    args = parser.parse_args()

    main(args)
