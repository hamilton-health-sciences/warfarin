"""Preprocess the COMBINE-AF files into data buffers for model development."""

from utils.smdp_buffer_mod import SMDPReplayBuffer #load_data # ...
from utils.utils import load_split_data, get_events_data

import argparse
import os
import numpy as np
from warnings import warn
import pickle

def main(args):
    
    print("\n----------------------------------------------")
    print("Creating replay buffers using the following parameters:")
    print(args)
    print("\n----------------------------------------------")
    
    if not os.path.exists(f"{args.data_folder}split_data"):
        raise Exception(f"The specified data folder does not exist: {args.data_folder}split_data")

    args.buffer_dir = args.data_folder + "replay_buffers/"
    if not os.path.exists(f"{args.buffer_dir}"):
        print(f"Making directory for storing replay buffers at {args.buffer_dir}... ")
        os.makedirs(f"{args.buffer_dir}")
        
    train_buffer = SMDPReplayBuffer()
    val_buffer = SMDPReplayBuffer()
    test_buffer = SMDPReplayBuffer()
    
    store_train_features = False
    if args.stored_feature_scales_path is not None:
        if not os.path.exists(f"{args.stored_feature_scales_path}"):
            warn(f"Could not find feature normalizations at: {args.stored_feature_scales_path}. Will normalize features using training data.")
            store_train_features = True
        else: 
            train_buffer.features_ranges = pickle.load(args.stored_feature_scales_path)
            
        
    train_data, val_data, test_data = load_split_data(args.data_folder + "split_data/", suffix=args.data_suffix)

    events_data = get_events_data(train_data)
    remove_ids = events_data["USUBJID_O_NEW"].value_counts()[events_data["USUBJID_O_NEW"].value_counts() < 2].index.tolist()
    events_data = events_data[~events_data["USUBJID_O_NEW"].isin(remove_ids)]
    events_buffer = SMDPReplayBuffer()
        
    ###########################################
    # Prepare replay buffers
    ###########################################
    print(f"\n----------------------------------------------")
    print(f"Training data")
    train_buffer.prepare_data(train_data, num_actions=args.num_actions, events_reward=args.events_reward, discount_factor=args.discount_factor)
    if store_train_features:
        print(f"Storing train buffer results in: {args.stored_feature_scales_path}")
        pickle.dump(train_buffer.features_ranges, args.stored_feature_scales_path)
    
    print(f"\n----------------------------------------------")
    print(f"Validation data")
    val_buffer.features_ranges = train_buffer.features_ranges
    val_buffer.prepare_data(val_data, num_actions=args.num_actions, events_reward=args.events_reward, discount_factor=args.discount_factor)
    
    print(f"\n----------------------------------------------")
    print(f"Test data")
    test_buffer.features_ranges = train_buffer.features_ranges
    test_buffer.prepare_data(test_data, num_actions=args.num_actions, events_reward=args.events_reward, discount_factor=args.discount_factor)
    
    print(f"\n----------------------------------------------")
    events_buffer.features_ranges = train_buffer.features_ranges
    events_buffer.prepare_data(events_data, num_actions=args.num_actions, events_reward=args.events_reward, discount_factor=args.discount_factor)
    
    ###########################################
    # Save buffers
    ###########################################
    state_dim = SMDPReplayBuffer.get_state(test_buffer.data, incl_hist=args.incl_hist).shape[1]
    suffix = f"actions_{args.num_actions}_state_{state_dim}_{args.buffer_suffix}"
    
    buffer_folder = args.buffer_dir + suffix
    if not os.path.exists(f"{buffer_folder}"):
        print(f"Making directory for this replay buffer at {buffer_folder}... ")
        os.makedirs(f"{buffer_folder}")

    print(f"\n----------------------------------------------")
    print(f"Saving all buffers to: {buffer_folder}")
    print(f"\n----------------------------------------------")
    train_buffer.data_path = buffer_folder
    val_buffer.data_path = buffer_folder
    test_buffer.data_path = buffer_folder
    events_buffer.data_path = buffer_folder
    
    train_buffer.save()
    val_buffer.save()
    test_buffer.save()
    events_buffer.save()

    print(f"DONE storing replay buffers!")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default="./data/", type=str, required=True, help="Path to folder containing data. There should be folder called 'split_data' containing the preprocessed train, val, test data. Generated replay buffers will be stored here.")
    parser.add_argument("--incl_hist", default=False, action="store_const", const=True, help="Flag to indicate whether we want to concatenate history to the state space.")
    parser.add_argument("--num_actions", default="", type=int, help="Number of actions to use in BCQ.")
    parser.add_argument("--events_reward", default=None, type=int, help="An integer reward value associated with adverse events. If not specified, the reward signal will only come from the INR value.")
    parser.add_argument("--discount_factor", default=0.99, type=int, help="Discount factor to use when calculating the discounted reward.")
    parser.add_argument("--data_suffix", default="", type=str, help="Suffix to identify the preprocessed data.")
    parser.add_argument("--buffer_suffix", default="smdp", type=str, help="Suffix to identify the generated replay buffer.")
    parser.add_argument(
        "--stored_feature_scales_path",
        default=None,
        type=str,
        help="Path to the stored feature scalings/normalizations. If not specified, "
             "the script will generate the normalizations and store the result "
             "in the clean_data_folder as feature_normalizations."
    )
    
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    main(args)
