import pandas as pd


def main(args):
    df = pd.read_csv(args.hierarchical_ttr_path)
    linker = pd.read_sas(args.rely_subjid_path)
    linker.columns = ["RELY_SUBJID", "SUBJID"]
    linker["RELY_SUBJID"] = linker["RELY_SUBJID"].astype(int)

    df_joined = df.set_index("SUBJID").join(
        linker.set_index("SUBJID")
    ).reset_index()
    df_joined["CENTRE"] = df_joined["RELY_SUBJID"].astype(str).str[:-3].astype(int)
    df_joined = df_joined.set_index(
        ["CENTRE", "RELY_SUBJID", "TRAJID", "STUDY_DAY"]
    )
    df_joined = df_joined.drop(["TRIAL"], axis=1)

    df_joined.to_csv(args.output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hierarchical_ttr_path", type=str, required=True)
    parser.add_argument("--rely_subjid_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    main(args)
