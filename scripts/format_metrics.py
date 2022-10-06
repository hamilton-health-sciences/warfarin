import os

import json

import pandas as pd


def main(args):
    metrics_raw = open(args.metrics_filename, "r").read()
    metrics = json.loads(metrics_raw)
    df = pd.DataFrame([metrics]).T.astype(float)

    entries = []
    for algo in ["POLICY", "THRESHOLD", "MAINTAIN", "RANDOM"]:
        for stat in ["sensitivity", "specificity"]:
            pfx = f"{algo}/classification/{stat}"
            val = df.loc[pfx][0] * 100
            val_lower = df.loc[f"{pfx}_lower"][0] * 100
            val_upper = df.loc[f"{pfx}_upper"][0] * 100
            entries.append(
                {"algorithm": algo,
                 "statistic": stat,
                 "value": (
                     f"{val:.1f}% ({val_lower:.1f}%, {val_upper:.1f}%)"
                 )}
            )
    classification_df = pd.DataFrame(entries)
    classification_df["algorithm"] = classification_df["algorithm"].map(
        {"POLICY": "BCQ-SMDP",
         "THRESHOLD": "Benchmark Algorithm",
         "MAINTAIN": "No Change",
         "RANDOM": "Random"}
    )
    classification_df = classification_df.pivot(columns="statistic",
                                                index="algorithm")
    classification_df.columns = classification_df.columns.get_level_values(1)

    classification_fn = os.path.join(args.output_prefix, "classification.csv")
    classification_df.to_csv(classification_fn)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_filename", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, required=True)
    args = parser.parse_args()

    main(args)
