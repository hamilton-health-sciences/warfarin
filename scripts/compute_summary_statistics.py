import pandas as pd


def main(args):
    # Load the data
    df_all = pd.read_feather(args.input_data)

    # Compute some stuff for later use
    df_all["USUBJID"] = (
        df_all["SUBJID"].astype(str) + "." + df_all["TRAJID"].astype(str)
    )

    # Compute summary statistics
    trials = df_all["TRIAL"].unique()
    summary_df = pd.DataFrame(index=trials)
    summary_df["Number of patients"] = (
        df_all.groupby("TRIAL")["SUBJID"].nunique()
    )
    summary_df["Number of trajectories"] = (
        df_all.groupby("TRIAL")["USUBJID"].nunique()
    )
    summary_df["Age"] = (
        df_all.groupby(
            "TRIAL"
        )["AGE_DEIDENTIFIED"].mean().map("{:.2f}".format) +
        " (" +
        df_all.groupby("TRIAL")["AGE_DEIDENTIFIED"].std().map(
            "{:.2f}".format
        ) + ")"
    )
    import pdb; pdb.set_trace()


if __name__  == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parsed_args = parser.parse_args()

    main(parsed_args)
