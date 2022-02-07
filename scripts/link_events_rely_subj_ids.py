"""Link the COMBINE events to RELY IDs."""

import pandas as pd


def main(args):
    # Load the events file
    df = pd.read_feather(args.events_path)

    # Filter down to RE-LY patients
    df = df[df["TRIAL"] == "RELY"]

    # Load the linking file provided to us.
    linker = pd.read_sas(args.rely_subjid_path)
    linker.columns = ["RELY_SUBJID", "SUBJID"]
    linker["RELY_SUBJID"] = linker["RELY_SUBJID"].astype(int)

    # Join the events to the linking file.
    df_joined = df.set_index("SUBJID").join(
        linker.set_index("SUBJID")
    ).reset_index()
    df_joined["CENTRE"] = df_joined["RELY_SUBJID"].astype(str).str[:-3].astype(int)
    df_joined = df_joined.set_index(
        ["CENTRE", "RELY_SUBJID"]
    )
    df_joined = df_joined.drop(["TRIAL"], axis=1)

    # Write the output.
    df_joined.reset_index().to_feather(args.output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--events_path", type=str, required=True)
    parser.add_argument("--rely_subjid_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    main(args)
