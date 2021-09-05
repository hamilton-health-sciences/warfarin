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

from warfarin.utils.combine_preprocessing import (load_raw_data,
                                                  remove_clinically_unintuitive,
                                                  remove_phys_implausible,
                                                  prepare_features)
from warfarin.utils import timer
from warfarin import config


def preprocess(args):
    """
    Run the preprocessing pipeline end-to-end.
    """
    # Load the data from feather files
    inr, events, baseline = load_raw_data(
        args.data_folder + args.raw_data_folder
    )

    inr, events, baseline = preprocess_all(inr, events, baseline)

    engage_rocket_data = preprocess_engage_rocket(inr, baseline)
    rely_data = preprocess_rely(inr, baseline)
    aristotle_data = preprocess_aristotle(inr, baseline)

    inr = pd.concat([engage_rocket_data, rely_data, aristotle_data])

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
    baseline_path = os.path.join(args.clean_data_path,
                                 f"baseline{args.suffix}.feather")
    inr_path = os.path.join(args.clean_data_path, f"inr{args.suffix}.feather")
    events_path = os.path.join(args.clean_data_path,
                               f"events{args.suffix}.feather")
    merged_path = os.path.join(args.clean_data_path,
                               f"merged{args.suffix}.feather")
    baseline.reset_index(drop=True).to_feather(baseline_path)
    inr.reset_index(drop=True).to_feather(inr_path)
    events.reset_index(drop=True).to_feather(events_path)
    merged_all.reset_index(drop=True).to_feather(merged_path)

    # Split the data
    test_ids = np.loadtxt(args.test_ids_path).astype(int)
    train_data, val_data, test_data = split_data(merged_all, test_ids)

    # if args.remove_clin:
    #     train_data = remove_clinically_unintuitive(train_data)
    #     train_data = remove_short_traj(train_data)
    # if args.remove_phys:
    #     train_data = remove_phys_implausible(train_data)
    #     train_data = remove_short_traj(train_data)

    # train_data = prepare_features(train_data)
    # val_data = prepare_features(val_data)
    # test_data = prepare_features(test_data)

    train_path = os.path.join(args.split_data_path,
                              f"train_data{args.suffix}.feather")
    val_path = os.path.join(args.split_data_path,
                            f"val_data{args.suffix}.feather")
    test_path = os.path.join(args.split_data_path,
                             f"test_data{args.suffix}.feather")
    train_data.reset_index(drop=True).to_feather(train_path)
    val_data.reset_index(drop=True).to_feather(val_path)
    test_data.reset_index(drop=True).to_feather(test_path)
    # print(
    #     f"Stored the train, val, test data in directory: {args.split_data_path}"
    # )


def main(args):
    # Ensure data directory exists
    raw_data_dir = os.path.join(args.data_folder, args.raw_data_folder)
    if not os.path.exists(raw_data_dir):
        raise Exception(
            f"The specified data folder does not exist: {raw_data_dir}"
        )

    # Construct suffix
    if args.remove_clin:
        args.suffix = args.suffix + "_wo_clin"
    if args.remove_phys:
        args.suffix = args.suffix + "_wo_phys"

    # Write out preprocessing parameters
    if args.output_preprocess_args is not None:
        preprocess_args_fn = os.path.join(args.data_folder,
                                          args.output_preprocess_args)
        json.dumps(vars(args), open(preprocess_args_fn, "w"))

    # Make directory for the cleaned data
    args.clean_data_path = os.path.join(args.data_folder,
                                        args.clean_data_folder)
    os.makedirs(args.clean_data_path, exist_ok=True)

    # Make directory for the split data
    args.split_data_path = os.path.join(args.data_folder, "split_data")
    os.makedirs(args.split_data_path, exist_ok=True)

    # Run pipeline
    with timer("preprocessing data"):
        preprocess(args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder",
        default="./data/",
        type=str,
        required=True,
        help=("Path to folder containing data. The raw data should be within "
              "this folder. Cleaned data will be stored here as well.")
    )
    parser.add_argument(
        "--raw_data_folder",
        default="raw_data/",
        type=str,
        help="Subfolder within the data folder containing the raw data."
    )
    parser.add_argument(
        "--clean_data_folder",
        default="clean_data/",
        type=str,
        help=("Subfolder within the data folder for storing the preprocessed/"
              "clean data.")
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
        "--remove_clin",
        default=False,
        action="store_const",
        const=True,
        help=("Flag to indicate whether or not we want to remove clinically "
              "unintuitive cases. Note that when this parameter is marked as "
              "True, '_wo_clin' will be appended to the data suffix.")
    )
    parser.add_argument(
        "--remove_phys",
        default=False,
        action="store_const",
        const=True,
        help=("Flag to indicate whether or not we want to remove "
              "physiologically implausible cases. Note that when this "
              "parameter is marked as True, '_wo_phys' will be appended to the "
              "data suffix.")
    )
    parser.add_argument(
        "--suffix",
        default="",
        type=str,
        help="Suffix to identify the preprocessed data."
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
