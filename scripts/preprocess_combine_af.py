"""Preprocess the COMBINE-AF files into data buffers for model development."""

import os

import numpy as np

import pandas as pd

from warfarin.data.combine_preprocessing import (preprocess_all,
                                                 preprocess_engage_rocket,
                                                 preprocess_rely,
                                                 preprocess_aristotle,
                                                 remove_outlying_doses,
                                                 merge_inr_events,
                                                 split_trajectories_at_events,
                                                 impute_inr_and_dose,
                                                 split_trajectories_at_gaps,
                                                 merge_inr_baseline,
                                                 split_data,
                                                 remove_short_trajectories)

# from warfarin.utils.combine_preprocessing import (load_raw_data,
#                                                   remove_clinically_unintuitive,
#                                                   remove_phys_implausible,
#                                                   prepare_features)
from warfarin.utils import timer
from warfarin import config


def preprocess(args):
    """
    Run the preprocessing pipeline end-to-end.

    Args:
        args: The command-line args.
    """
    # Load the data from feather files
    baseline = pd.read_feather(args.baseline_path)
    inr = pd.read_feather(args.inr_path)
    events = pd.read_feather(args.events_path)

    # Perform non-trial-specific initial preprocessing
    inr, events, baseline = preprocess_all(inr, events, baseline)

    # Trial-specific preprocessing
    engage_rocket_data = preprocess_engage_rocket(inr, baseline)
    rely_data = preprocess_rely(inr, baseline)
    aristotle_data = preprocess_aristotle(inr, baseline)
    inr = pd.concat([engage_rocket_data, rely_data, aristotle_data])

    # Perform non-trial-specific preprocessing
    inr = remove_outlying_doses(inr)
    inr_events_merged = merge_inr_events(inr, events)
    inr_events_merged = split_trajectories_at_events(inr_events_merged)
    inr_events_merged = impute_inr_and_dose(inr_events_merged)
    inr_events_merged = split_trajectories_at_gaps(inr_events_merged)
    merged_all = merge_inr_baseline(inr_events_merged, baseline)

    # Prune trajectories with only one entry, as these are not useful for
    # training or evaluation (and represent cases where only an adverse
    # event occurred)
    merged_all = remove_short_trajectories(merged_all, min_length=2)

    # Save the output prior to splitting
    baseline_path = os.path.join(args.output_directory, "baseline.feather")
    inr_path = os.path.join(args.output_directory, "inr.feather")
    events_path = os.path.join(args.output_directory, "events.feather")
    merged_path = os.path.join(args.output_directory, "merged.feather")
    baseline.reset_index(drop=True).to_feather(baseline_path)
    inr.reset_index(drop=True).to_feather(inr_path)
    events.reset_index(drop=True).to_feather(events_path)
    merged_all.reset_index(drop=True).to_feather(merged_path)

    # Split the data
    test_ids = np.loadtxt(args.test_ids_path).astype(int)
    train_data, val_data, test_data = split_data(merged_all, test_ids)

    # Save the split output
    train_path = os.path.join(args.output_directory, "train_data.feather")
    val_path = os.path.join(args.output_directory, "val_data.feather")
    test_path = os.path.join(args.output_directory, "test_data.feather")
    train_data.reset_index(drop=True).to_feather(train_path)
    val_data.reset_index(drop=True).to_feather(val_path)
    test_data.reset_index(drop=True).to_feather(test_path)


def main(args):
    # Write out preprocessing parameters
    if args.output_preprocess_args is not None:
        preprocess_args_fn = os.path.join(args.data_folder,
                                          args.output_preprocess_args)
        json.dumps(vars(args), open(preprocess_args_fn, "w"))

    # Make directory for the cleaned data
    os.makedirs(args.output_directory, exist_ok=True)

    # Run pipeline
    with timer("preprocessing data"):
        preprocess(args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inr_path",
        required=True,
        type=str,
        help="Path to INR data feather file"
    )
    parser.add_argument(
        "--baseline_path",
        required=True,
        type=str,
        help="Path to baseline data feather file"
    )
    parser.add_argument(
        "--events_path",
        required=True,
        type=str,
        help="Path to events data feather file"
    )
    parser.add_argument(
        "--output_directory",
        required=True,
        type=str,
        help="Path to the directory to output the cleaned data"
    )
    parser.add_argument(
        "--test_ids_path",
        type=str,
        required=True,
        help=("Path to the IDs of subjects to include in the test set. Needed "
              "to ensure that the test set doesn't change during model "
              "development")
    )
    parser.add_argument(
        "--output_preprocess_args",
        default=None,
        type=str,
        help=("Will output the arguments passed to this script to this JSON "
              "file if given")
    )
    parser.add_argument("--seed", default=42, type=int)
    parsed_args = parser.parse_args()

    main(parsed_args)
