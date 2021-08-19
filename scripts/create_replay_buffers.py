"""Preprocess the COMBINE-AF files into data buffers for model development."""

from utils.utils import SMDPReplayBuffer #load_data # ...
from utils.combine_preprocessing import load_split_data, get_events_data

import argparse
import os
import numpy as np


def main(args):
    
    if not os.path.exists(f"{args.data_folder}split_data"):
        raise Exception(f"The specified data folder does not exist: {args.data_folder}split_data")

    args.buffer_dir = args.data_folder + "replay_buffers/"
    if not os.path.exists(f"{args.buffer_dir}"):
        print(f"Making directory for storing replay buffers at {args.buffer_dir}... ")
        os.makedirs(f"{args.buffer_dir}")

    if not os.path.exists(f"{args.buffer_dir}{args.buffer_suffix}"):
        print(f"Making directory for this replay buffer at {args.buffer_dir}{args.buffer_suffix}... ")
        os.makedirs(f"{args.buffer_dir}{args.buffer_suffix}")

    train_data, val_data, test_data = load_split_data(args.data_folder + "split_data/", suffix=args.data_suffix)
    
    buffer_name = f"{args.buffer_suffix}/buffer_data"

    train_buffer = SMDPReplayBuffer(filename=buffer_name + "_train")
    val_buffer = SMDPReplayBuffer(filename=buffer_name + "_valid")
    test_buffer = SMDPReplayBuffer(filename=buffer_name + "_test")

    events_data = get_events_data(train_data)
    remove_ids = events_data["USUBJID_O_NEW"].value_counts()[events_data["USUBJID_O_NEW"].value_counts() < 2].index.tolist()
    events_data = events_data[~events_data["USUBJID_O_NEW"].isin(remove_ids)]
    events_buffer = SMDPReplayBuffer(filename=buffer_name + "_events")
        
    ###########################################
    # Prepare replay buffers
    ###########################################
    print(f"\n----------------------------------------------")
    print(f"Training data")
    train_buffer.prepare_data(train_data, incl_adverse_events=args.incl_events_reward, num_actions=args.num_actions)
    features_ranges = train_buffer.features_ranges
    
    print(f"\n----------------------------------------------")
    print(f"Validation data")
    val_buffer.prepare_data(val_data, num_actions=args.num_actions, incl_adverse_events=args.incl_events_reward, features_ranges=features_ranges)
    
    print(f"\n----------------------------------------------")
    print(f"Test data")
    test_buffer.prepare_data(test_data, num_actions=args.num_actions, incl_adverse_events=args.incl_events_reward, features_ranges=features_ranges)
    
    print(f"\n----------------------------------------------")
    events_buffer.prepare_data(events_data, num_actions=args.num_actions, incl_adverse_events=args.incl_events_reward, features_ranges=features_ranges)

    ###########################################
    # Save buffers
    ###########################################
#     state_dim = SMDPReplayBuffer.get_state(test_buffer.data, args.state_method).shape[1]

#     print(f"\n----------------------------------------------")
#     print(f"Saving buffers!")
#     print(f"----------------------------------------------")
#     suffix = f"actions_{args.num_actions}_state_{state_dim}_{args.buffer_suffix}"
#     train_buffer.save_buffer(buffer_name=suffix, dataset="train", state_method=args.state_method)
#     val_buffer.save_buffer(buffer_name=suffix, dataset="valid", state_method=args.state_method)
#     test_buffer.save_buffer(buffer_name=suffix, dataset="test", state_method=args.state_method)
#     events_buffer.save_buffer(buffer_name=suffix, dataset="events", state_method=args.state_method)

    print(f"DONE storing replay buffers!")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default="./data/", type=str, required=True, help="Path to folder containing data. There should be folder called 'split_data' containing the preprocessed train, val, test data. Generated replay buffers will be stored here.")
    parser.add_argument("--state_method", default=18, type=int, help="Method number of state space.")
    parser.add_argument("--num_actions", default="", type=int, help="Number of actions to use in BCQ.")
    parser.add_argument("--events_reward", default=-1, type=bool, help="Flag to indicate whether we want to .")
    parser.add_argument("--data_suffix", default="", type=str, help="Suffix to identify the preprocessed data.")
    parser.add_argument("--buffer_suffix", default="smdp", type=str, help="Suffix to identify the generated replay buffer.")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    main(args)
