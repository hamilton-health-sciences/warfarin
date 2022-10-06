import os

import pandas as pd

from warfarin.evaluation.plotting import plot_policy_heatmap


def main(args):
    # Load file
    df = pd.read_csv(args.hierarchical_ttr_path)

    # Maintain
    maintain_df = df[["MAINTAIN_ACTION", "INR_VALUE"]].copy()
    maintain_df.columns = ["ACTION", "INR_VALUE"]
    maintain_plot = plot_policy_heatmap(maintain_df)

    # Random
    random_df = df[["RANDOM_ACTION", "INR_VALUE"]].copy()
    random_df.columns = ["ACTION", "INR_VALUE"]
    random_plot = plot_policy_heatmap(random_df)

    # Output
    maintain_plot_path = os.path.join(args.output_dir, "heatmap_maintain.jpg")
    random_plot_path = os.path.join(args.output_dir, "heatmap_random.jpg")
    maintain_plot.save(maintain_plot_path, dpi=600)
    random_plot.save(random_plot_path, dpi=600)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hierarchical_ttr_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)
