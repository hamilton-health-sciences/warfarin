"""Preprocess the COMBINE-AF files into data buffers for model development."""

from data.combine_preprocessing import load_data # ...


def main(args):
    # Load the data from SAS files
    baseline_df, inr_df, events_df = load_data(args.baseline_fn,
                                               args.inr_fn,
                                               args.events_fn)
    # ...


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_fn", type=str, required=True,
                        help="Path to baseline data SAS file")
    parser.add_argument("--inr_fn", type=str, required=True,
                        help="Path to INR data SAS file")
    parser.add_argument("--events_fn", type=str, required=True,
                        help="Path to events data SAS file")
    # ... options/flags for steps 6/7?
    args = parser.parse_args()

    main(args)
