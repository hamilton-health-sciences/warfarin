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

from models.smdp_dBCQ import discrete_BCQ
from utils.smdp_buffer import SMDPReplayBuffer


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


# Runs policy for X episodes and returns average rewardnroon
# A fixed seed is used for the eval environment
def eval_policy(policy,
                valid_replay_buffer,
                eval_episodes=10,
                train_replay_buffer=None):
    device = "cuda:0"

    if train_replay_buffer is not None:
        train_correct, train_counts = 0, 0
        state = torch.FloatTensor(train_replay_buffer.state).to(device)
        train_correct += eval_good_actions(
            policy,
            state,
            num_actions=policy.num_actions
        )
        train_counts += len(state)
        avg_reward_train = train_correct / train_counts

    # Calculate number of "reasonable" actions
    correct, counts = 0, 0
    buffer_size = valid_replay_buffer.crt_size
    ind = np.arange(buffer_size)
    k, state, action, next_state, reward, done = valid_replay_buffer.sample(
        ind,
        return_flag=False
    )
    pred_action = policy.select_action(np.array(state.to("cpu")), eval=True)
    correct += eval_good_actions(policy, state, num_actions=policy.num_actions)
    counts += len(action)
    avg_reward = correct / counts

    # Calculate rate of events
    reward_np = np.array(reward.to("cpu")).transpose()[0]
    event_flag = np.where(reward_np <= -10, 1, 0)
    both_actions = pd.DataFrame(
        {"CLINICIAN_ACTION": np.array(action.to("cpu")).transpose()[0],
         "POLICY_ACTION": pred_action.transpose()[0],
         "ADV_EVENTS": event_flag}
    )
    both_actions.loc[:, "DIFF_ACTIONS"] = (both_actions["POLICY_ACTION"] -
                                           both_actions["CLINICIAN_ACTION"])
    events_rate = both_actions[
        both_actions["DIFF_ACTIONS"] == 0
    ]["ADV_EVENTS"].mean()

    print("---------------------------------------")
    print(
        f"Evaluation over {eval_episodes} episodes: Validation % Reasonable "
        f"Actions: {avg_reward:.3%}" +
        (f" | Training % Reasonable Actions: {avg_reward_train:.3%}"
         if train_replay_buffer is not None else "")
    )
    print("---------------------------------------")

    if train_replay_buffer is not None:
        return avg_reward_train, events_rate, avg_reward, pred_action

    else:
        return events_rate, avg_reward, pred_action


def eval_good_actions(policy, state, num_actions):
    pred_action = policy.select_action(
        np.array(state.to("cpu")),
        eval=True
    ).transpose()[0]
    inr_state = state[:, 0].cpu().detach().numpy()

    num_good_actions = sum([
        sum(
            np.where(np.logical_and(inr_state > 0.625, pred_action <= 2), 1, 0)
        ),
        sum(
            np.where(
                np.logical_and(
                    np.logical_and(inr_state >= 0.375, inr_state <= 0.625),
                    pred_action == 3
                ),
                1,
                0
            )
        ),
        sum(
            np.where(
                np.logical_and(inr_state < 0.375, pred_action >= 4), 1, 0)
        ),
    ])

    return num_good_actions


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_wis(policy,
             behav_policy,
             pol_dataloader,
             discount,
             is_demog,
             device,
             use_rep=True):
    """
    The following is the original dBCQ's evaluation script that we'll need to
    replace with weighted importance sampling between the learned `policy` and
    the observed policy.
    """
    wis_returns = 0
    wis_weighting = 0

    # Loop through the dataloader (representations, observations, actions,
    # demographics, rewards)
    for representations, obs_state, demog, actions, rewards in pol_dataloader:
        representations = representations.to(device)
        obs_state = obs_state.to(device)
        actions = actions.to(device)
        demog = demog.to(device)

        cur_obs, cur_actions = (obs_state[:, :-2, :][:, 1:, :],
                                actions[:, :-2, :][:, 1:, :].argmax(dim=-1))
        cur_demog, cur_rewards = demog[:, :-2, :][:, 1:, :], rewards[:, :-2]

        cur_rep = representations[:, :-2, :]

        # Mask out the data corresponding to the padded observations
        mask = (cur_obs == 0).all(dim=2)

        # Compute the discounted rewards for each trajectory in the minibatch
        discount_array = torch.Tensor(
            discount ** np.arange(cur_rewards.shape[1])
        )[None, :]
        discounted_rewards = (discount_array * cur_rewards).sum(
            dim=-1
        ).squeeze().sum(dim=-1)

        print(
            f"rewards: {rewards[:, :-2].shape}, behav policy output: "
            f"{behav_policy(cur_rep.flatten(end_dim=1)).shape}, actions: "
            f"{cur_actions.flatten()[:, None].shape}"
        )

        # Evaluate the probabilities of the observed action according to the
        # trained policy and the behavior policy
        with torch.no_grad():
            print(f"is_demog: {is_demog}")
            # Gather the probability from the observed behavior policy
            if is_demog:
                p_obs = F.softmax(
                    behav_policy(cur_rep.flatten(end_dim=1)), dim=-1
                ).gather(
                    1, cur_actions.flatten()[:, None]
                ).reshape(
                    cur_rep.shape[:2]
                )
            else:
                p_obs = F.softmax(
                    behav_policy(cur_obs.flatten(end_dim=1)), dim=-1
                ).gather(1, cur_actions.flatten()[:, None]).reshape(
                    cur_obs.shape[:2]
                )
            if use_rep:
                # Compute the Q values of the dBCQ policy
                q_val, _, _ = policy.Q(representations)
            else:
                q_val, _, _ = policy.Q(
                    torch.cat((cur_obs.flatten(end_dim=1),
                               cur_demog.flatten(end_dim=1)),
                              dim=-1)
                )
            p_new = F.softmax(q_val, dim=-1).gather(
                2, cur_actions[:, :, None]
            ).squeeze()  # Gather the probabilities from the trained policy

        # Check for whether there are any zero probabilities in p_obs and
        # replace with small probability since behav_pol may mispredict actual
        # actions...
        if not (p_obs > 0).all():
            p_obs[p_obs == 0] = 0.1

        # Eliminate spurious probabilities due to padded observations after
        # trajectories have concluded. We do this by forcing the probabilities
        # for these observations to be 1 so they don't affect the product
        p_obs[mask] = 1.
        p_new[mask] = 1.

        cum_ir = torch.clamp((p_new / p_obs).prod(axis=1), 1e-30, 1e4)

        wis_rewards = cum_ir.cpu() * discounted_rewards
        wis_returns += wis_rewards.sum().item()
        wis_weighting += cum_ir.cpu().sum().item()

    wis_eval = (wis_returns / wis_weighting)
    print("---------------------------------------")
    print(f"Evaluation over the test set: {wis_eval:.3f}")
    print("---------------------------------------")
    return wis_eval


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
        train_replay_buffer = SMDPReplayBuffer(
            data_path=folder_paths["buffers"] + "/train_data",
            batch_size=parameters["batch_size"] - args.events_batch_size,
            buffer_size=warfarin_parameters["train_buffer_size"],
            device=device
        )
        valid_replay_buffer = SMDPReplayBuffer(
            data_path=folder_paths["buffers"] + "/val_data",
            batch_size=parameters["batch_size"],
            buffer_size=warfarin_parameters["valid_buffer_size"],
            device=device
        )
        events_replay_buffer = SMDPReplayBuffer(
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
