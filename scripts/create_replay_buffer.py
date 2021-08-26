"""Preprocess the COMBINE-AF files into data buffers for model development."""

import argparse

import os

from warnings import warn

import pickle

import pandas as pd

from warfarin.utils.smdp_buffer import SMDPReplayBuffer
from warfarin.utils.utils import load_split_data, get_events_data


def main(args):
    print("Creating replay buffer using the following parameters:")
    print(args)

    # Make sure input data exists
    if not os.path.exists(args.input_fn):
        raise Exception("The specified data does not exist: {args.input_fn}")

    # Make sure output directory exists
    if not os.path.exists(os.path.dirname(args.output_fn)):
        os.makedirs(os.path.dirname(args.output_fn))

    # Load feature normalizations if given
    if args.normalization is None:
        features_ranges = None
    else:
        features_ranges = pickle.load(open(args.normalization, "rb"))

    # Load the data
    data = pd.read_feather(args.input_fn)

    # Prepare replay buffer
    buf = SMDPReplayBuffer.from_data(
        data,
        num_actions=args.num_actions,
        events_reward=args.events_reward,
        discount_factor=args.discount_factor,
        features_ranges=features_ranges,
        training=(features_ranges is None)
    )

    # Store normalizations if given
    if args.output_normalization:
        print("Storing train buffer results in: {args.output_normalization}")
        pickle.dump(
            buf.features_ranges, open(args.output_normalization, "wb")
        )

    # Generate events buffer and save, if given output path
    if args.output_events_fn:
        events_data = get_events_data(data)
        remove_ids = events_data["USUBJID_O_NEW"].value_counts()[
            events_data["USUBJID_O_NEW"].value_counts() < 2
        ].index.tolist()
        events_data = events_data[~events_data["USUBJID_O_NEW"].isin(remove_ids)]

        events_buf = SMDPReplayBuffer.from_data(
            events_data,
            num_actions=args.num_actions,
            events_reward=args.events_reward,
            discount_factor=args.discount_factor,
            features_ranges=buf.features_ranges
        )

        # Save
        events_buf.save(args.output_events_fn)

    # Save buffer
    print(f"Saving buffer to: {args.output_fn}")
    buf.save(args.output_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_fn",
        type=str,
        required=True,
        help="Path to input data"
    )
    parser.add_argument(
        "--incl_hist",
        default=False,
        action="store_const",
        const=True,
        help="Flag to indicate whether we want to concatenate history to the "
             "state space."
    )
    parser.add_argument(
        "--num_actions",
        default="",
        type=int,
        help="Number of actions to use in BCQ."
    )
    parser.add_argument(
        "--events_reward",
        default=None,
        type=int,
        help="An integer reward value associated with adverse events. If not "
             "specified, the reward signal will only come from the INR value."
    )
    parser.add_argument(
        "--discount_factor",
        default=0.99,
        type=int,
        help="Discount factor to use when calculating the discounted reward."
    )
    parser.add_argument(
        "--output_fn",
        type=str,
        help="Path to output replay buffer"
    )
    parser.add_argument(
        "--output_events_fn",
        type=str,
        help="If given, will output trajectories with events to buffer at "
             "this path."
    )
    parser.add_argument(
        "--output_normalization",
        default=None,
        type=str,
        help="If specified, will store a pickle of feature normalizations here."
    )
    parser.add_argument(
        "--normalization",
        default=None,
        type=str,
        help="If specified, will normalize features according to "
             "normalizations stored here."
    )
    parser.add_argument("--seed", default=42, type=int)
    parsed_args = parser.parse_args()

    main(parsed_args)
