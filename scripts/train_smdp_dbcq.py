#TODO: Update this description

"""
Based in part on code from Taylor Killian and Haoran Zhang at University of
Toronto and the Vector Institute, originally provided under the MIT licence.
"""

import time

import argparse

import os

import sys

import wandb

import numpy as np

import pandas as pd

import torch
import torch.nn.functional as F

from utils.smdp_buffer import SMDPReplayBuffer
from models.smdp_dBCQ import discrete_BCQ
from models.policy_eval import eval_policy, eval_good_actions, eval_wis


def train_bcq(train_replay_buffer,
              valid_replay_buffer,
              events_replay_buffer,
              num_actions,
              state_dim,
              device,
              args,
              parameters,
              folder_paths,
              resume):
    # Initialize and load policy
    policy = discrete_BCQ(
        num_actions,
        state_dim,
        device,
        args.BCQ_threshold,
        parameters["discount"],
        parameters["optimizer"],
        parameters["optimizer_parameters"],
        parameters["polyak_target_update"],
        parameters["target_update_freq"],
        parameters["tau"],
        parameters["initial_eps"],
        parameters["end_eps"],
        parameters["eps_decay_period"],
        parameters["eval_eps"],
        args.hidden_states,
        args.num_layers
    )

    if resume:
        checkpoint_file = folder_paths["models"] + "/checkpoint"
#         if os.path.exists(checkpoint_file):
        try:
            state = torch.load(folder_paths["models"] + "/checkpoint")
            policy.Q.load_state_dict(state["Q_state_dict"])
            training_iters = state["training_iters"]
            args.max_timesteps += training_iters
            print(
                f"Resuming training at iteration: {training_iters}, to "
                f"iteration: {args.max_timesteps}..."
            )
        except Exception as e:
            print("Failed to resume training. Starting at epoch 0.")
    else:
        training_iters = 0

    # Load replay buffer
    use_rep = args.suffix.split("_")[0] == "ais"

    train_replay_buffer.load()
    valid_replay_buffer.load()
    events_replay_buffer.load()

    evaluations, policy_actions, loss = [], [], []
    episode_num = 0
    done = True

    min_events_rate = float("inf")

    while training_iters < args.max_timesteps:
        for _ in range(int(parameters["eval_freq"])):
            qloss = policy.train(train_replay_buffer, events_replay_buffer)

        loss.append(qloss.item())

        np.save(folder_paths["results"] + "/BCQ_Warfarin_Qloss", loss)

        avg_reward_train, events_rate, reward, actions = eval_policy(
            policy,
            valid_replay_buffer,
            1,
            train_replay_buffer
        )

        if use_rep:
            # TODO Run weighted importance sampling with learned policy and
            # behavior policy
            # # Initialize a dataloader for policy evaluation (will need
            # # representations, observations, demographics, rewards and actions
            # # from the test dataset)
            # test_representations = torch.load(
            #     args.test_representations_file
            # )  # Load the test representations
            # obs, dem, actions, rewards = torch.load(args.test_data_file)
            # pol_eval_dataset = TensorDataset(
            #     test_representations, obs, dem, actions, rewards
            # )
            # pol_eval_dataloader = DataLoader(
            #     pol_eval_dataset,
            #     batch_size=parameters["batch_size"],
            #     shuffle=False
            # )

            # # Load the pretrained policy for whether or not the demographic
            # # context was used to train the representations
            # behav_input = 32  # args.state_dim
            # num_nodes = 64
            # behav_pol = FC_BC(
            #     state_dim=behav_input,
            #     num_actions=args.num_actions,
            #     num_nodes=num_nodes
            # ).to(device)
            # behav_pol.load_state_dict(torch.load(args.behav_policy_file))
            # behav_pol.eval()

            # wis = eval_wis(
            #     policy,
            #     behav_pol,
            #     pol_eval_dataloader,
            #     parameters["discount"],
            #     is_demog=1,
            #     device=device,
            #     use_rep=use_rep
            # )

            wis = None
        else:
            wis = None

        evaluations.append(reward)
        policy_actions.append(actions)
        np.save(folder_paths["results"] + "/BCQ_Warfarin_Reward", evaluations)

        training_iters += int(parameters["eval_freq"])
        print(f"Training iterations: {training_iters}")

        checkpoint_state = {"Q_state_dict": policy.Q.state_dict(),
                            "training_iters": training_iters}
        torch.save(checkpoint_state, folder_paths["models"] + "/checkpoint")

        if events_rate <= min_events_rate:
            print("Saving policy....")
            torch.save(
                checkpoint_state,
                folder_paths["models"] + f"/checkpoint_{training_iters}"
            )
            fig = None
            min_events_rate = events_rate
        if (training_iters % 100000) == 0:
            print("Saving policy....")
            torch.save(
                checkpoint_state,
                folder_paths["models"] + f"/checkpoint_{training_iters}"
            )
            fig = None

        wandb.log({
            "train_loss": qloss.item(),
            "train_percent_reasonable_actions": avg_reward_train,
            "val_percent_reasonable_actions": reward,
            "val_events_rate": events_rate,
            "min_val_events_rate": min_events_rate,
            "wis": wis,
            "heatmap": fig
        })

        print("Saving policy....")
    checkpoint_state = {"Q_state_dict": policy.Q.state_dict(),
                       "training_iters": training_iters}
    torch.save(checkpoint_state, folder_paths["models"] + "/checkpoint")


if __name__ == "__main__":

    # Warfarin data
    warfarin_parameters = {
        "train_buffer_size": 1e7,
        "valid_buffer_size": 1e6,
        "events_buffer_size": 1e6
    }

    parameters = {
        # Exploration
        "start_timesteps": 1e3,
        "initial_eps": 0.1,
        "end_eps": 0.1,
        "eps_decay_period": 1,
        # Evaluation
        "eval_freq": 1e3,
        "eval_eps": 0,
        # Learning
        "discount": 0.99,
        "buffer_size": 869240,
        "optimizer": "Adam",
        "train_freq": 1,
        "polyak_target_update": True,
        "target_update_freq": 100,
        "tau": 0.005
    }

    # Load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--data_folder", default="./data")
    parser.add_argument("--save_folder", default="./output/bcq")
    parser.add_argument("--buffer_folder", default="./data/replay_buffers")
    parser.add_argument("--num_actions", default=5, type=int)
    parser.add_argument("--state_dim", default=18, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--suffix", type=str)
    parser.add_argument("--resume", default=False)
    parser.add_argument("--max_timesteps", default=1e5, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)  # Learning rate
    parser.add_argument(
        "--BCQ_threshold",
        default=0.3,
        type=float
    )  # Threshold hyper-parameter for BCQ
    parser.add_argument(
        "--hidden_states",
        default=25,
        type=int
    )
    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument("--events_batch_size", default=2, type=int)

    # TODO: These should NOT be the defaults
    parser.add_argument(
        "--wandb_key",
        default="251aff8dbc2297dad3c63077733a310a5c166e0d",
        type=str
    )
    parser.add_argument("--project", default="warfarin_semi_dbcq", type=str)

    args = parser.parse_args()

    parameters["batch_size"] = args.batch_size
    parameters["optimizer_parameters"] = {"lr": args.lr}

    args.buffer_name = (f"actions_{args.num_actions}_state_{args.state_dim}"
                        f"_{args.suffix}")
    #     args.buffer_df_file = f"buffer_data_{args.suffix}"
    args.savename = (f"lr{args.lr}_bcq{args.BCQ_threshold}_hstates"
                     f"{args.hidden_states}_evBatchsize{args.events_batch_size}"
                     f"_seed{args.seed}")

    args.test_representations_file = (f"{args.data_folder}/"
                                      "test_representations.pt")
    args.test_data_file = f"{args.data_folder}/test_tuples"
    args.behav_policy_file = f"{args.data_folder}/BC_model.pt"

    folder_paths = {
        "results": f"{args.save_folder}/results/{args.buffer_name}"
                   f"/{args.savename}",
        "models": f"{args.save_folder}/models/{args.buffer_name}"
                  f"/{args.savename}",
        "buffers": f"{args.buffer_folder}/{args.buffer_name}"
    }

#     if not os.path.exists(args.save_folder):
#         print(f"Creating folder to save results...")
#         os.makedirs(args.save_folder)

    if not os.path.exists(folder_paths["results"]):
        print("Creating results folder...")
        os.makedirs(folder_paths["results"])

    if not os.path.exists(folder_paths["models"]):
        print("Creating models folder...")
        os.makedirs(folder_paths["models"])

    if not os.path.exists(folder_paths["buffers"]):
        sys.exit(
            f"ERROR: Buffer does not exist for buffer name {args.buffer_name}"
        )

    if args.BCQ_threshold == 1.0:
        args.project = "warfarin_behav_cloning"

    # Set up W&B
    _ = os.system("wandb login {}".format(args.wandb_key))
    os.environ["WANDB_API_KEY"] = args.wandb_key
    args.unique_run_id = wandb.util.generate_id()
    wandb.init(id=args.unique_run_id,
               resume="allow",
               project=args.project,
               group=args.buffer_name,
               name=args.savename)
    wandb.config.update(args)

    print("---------------------------------------")
    print(
        f"Setting: Training Seed: {args.seed}, Max timesteps: "
        f"{args.max_timesteps}, Learning rate: {args.lr}, Hidden states: "
        f"{args.hidden_states}, Number of hidden layers: {args.num_layers - 1}"
    )
    print("---------------------------------------")

    # Make env and determine properties
    state_dim, num_actions = args.state_dim, args.num_actions

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        train_replay_buffer = SMDPReplayBuffer.from_filename(
            data_path=folder_paths["buffers"] + "/train_data",
            batch_size=parameters["batch_size"] - args.events_batch_size,
            buffer_size=warfarin_parameters["train_buffer_size"],
            device=device
        )
        valid_replay_buffer = SMDPReplayBuffer.from_filename(
            data_path=folder_paths["buffers"] + "/val_data",
            batch_size=parameters["batch_size"],
            buffer_size=warfarin_parameters["valid_buffer_size"],
            device=device
        )
        events_replay_buffer = SMDPReplayBuffer.from_filename(
            data_path=folder_paths["buffers"] + "/events_data",
            batch_size=args.events_batch_size,
            buffer_size=warfarin_parameters["events_buffer_size"],
            device=device
        )

        train_bcq(train_replay_buffer,
                  valid_replay_buffer,
                  events_replay_buffer,
                  num_actions,
                  state_dim,
                  device,
                  args,
                  parameters,
                  folder_paths,
                  args.resume)
    else:
        # TODO: Add support for CPU
        print("Could not find GPU. Please try again...")

    t1 = time.time()
    print(f"Done! Took {t1 - t0} seconds")
