"""Preprocess the COMBINE-AF files into data buffers for model development."""

import os

import numpy as np

import pandas as pd

from warfarin.data.combine_preprocessing import (
    preprocess_all,
    preprocess_engage_rocket,
    preprocess_rely,
    preprocess_aristotle,
    merge_trials_and_remove_outlying_doses,
    merge_inr_events,
    split_trajectories_at_events,
    impute_inr_and_dose,
    split_trajectories_at_gaps,
    merge_inr_baseline,
    split_data,
    remove_short_trajectories
)
from warfarin.data.utils import remove_unintuitive_decisions
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

    # Perform non-trial-specific preprocessing
    inr = merge_trials_and_remove_outlying_doses(engage_rocket_data, rely_data,
                                                 aristotle_data)
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
    train_data, val_data, test_data = split_data(merged_all, seed=args.seed)

    # Remove unintuitive clinical decisions that may confound learning
    if args.remove_unintuitive_decisions:
        train_data = remove_unintuitive_decisions(train_data)
        # Make sure we don't overwrite the auditing dataframe or the main call
        # to `remove_short_trajectories`
        train_data = remove_short_trajectories(train_data,
                                               audit_name="train",
                                               min_length=2)

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
        "--remove_unintuitive_decisions",
        action="store_true",
        default=False,
        help="Whether to remove unintuitive clinical decisions from the "
             "training set"
    )
    parser.add_argument(
        "--output_directory",
        required=True,
        type=str,
        help="Path to the directory to output the cleaned data"
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
