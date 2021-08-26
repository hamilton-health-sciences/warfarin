import numpy as np

import torch

import pandas as pd


def eval_policy(policy,
                valid_replay_buffer,
                eval_episodes=10,
                train_replay_buffer=None):
    """
    Runs policy for X episodes and returns average rewardnroon
    A fixed seed is used for the eval environment
    """
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

    # print("---------------------------------------")
    # print(
    #     f"Evaluation over {eval_episodes} episodes: Validation % Reasonable "
    #     f"Actions: {avg_reward:.3%}" +
    #     (f" | Training % Reasonable Actions: {avg_reward_train:.3%}"
    #      if train_replay_buffer is not None else "")
    # )
    # print("---------------------------------------")

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


