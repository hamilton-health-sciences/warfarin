import numpy as np

import torch

import pandas as pd

from sklearn.metrics import classification_report


# TODO get the device from the policy object?
device = "cuda"


def eval_policy(policy, replay_buffer):
    """
    Evaluate a policy.

    Computes the following metrics:

        (1) Proportion of actions that are "reasonable" (defined as choosing to
            lower the dose when the INR is high, and raising it when the INR is
            low).
        (2) Sensitivity and specificity of the detection of "immediately good"
            actions. Specifically, the "positive class" is defined as the next
            INR being in range. An action is predicted to be "good" when it is
            selected by the model. We also compute Youden's J statistic from
            these two statistics.
        (3) TTR (%) estimated in trajectories at agree with the model, at the
            thresholds prescribed in `warfarin.config.AGREEMENT_THRESHOLDS`.
            This is to assess sensitivity to the threshold.
    """
    stats = {}

    # Predicted action (for computing convergence)
    state = np.array(replay_buffer.state)
    pred_action = policy.select_action(state, eval=True)[:, 0]

    # Reasonable-ness
    prop_reasonable = eval_reasonable_actions(policy, replay_buffer)
    stats["proportion_reasonable_actions"] = prop_reasonable

    # Classification metrics
    sens, spec, jstat = eval_classification(policy, replay_buffer)
    stats["sensitivity_good_actions"] = sens
    stats["specificity_good_actions"] = spec
    stats["jindex_good_actions"] = jstat

    # # TTR at agreement
    # for thresh in config.AGREEMENT_THRESHOLDS:
    #     num_traj_agree, num_trans_agree, ttr = eval_ttr_at_agreement(
    #         policy, replay_buffer, thresh
    #     )
    #     stats[f"number_trajectories_agree_{thresh}"] = num_traj_agree
    #     stats[f"number_transitions_agree_{thresh}"] = num_trans_agree
    #     stats[f"ttr_agreement_{thresh}"] = ttr

    return stats, pred_action


def eval_reasonable_actions(policy, replay_buffer):
    state = torch.FloatTensor(replay_buffer.state).to(device)
    reasonable = eval_good_actions(
        policy,
        state,
        num_actions=policy.num_actions
    )
    total = len(state)

    return reasonable / total


def eval_classification(policy, replay_buffer):
    buffer_size = replay_buffer.crt_size
    _, state, obs_action, next_state, reward, _ = replay_buffer.sample(
        np.arange(buffer_size),
        return_flag=False
    )
    policy_action = policy.select_action(state.cpu().numpy(), eval=True)
    # TODO parameterize clamped range in config so we dont need magic #s
    next_inr = next_state[:, 0].cpu().numpy() * 4 + 0.5
    actions_reward = pd.DataFrame({
        "OBSERVED_ACTION": obs_action.cpu().numpy()[:, 0],
        "POLICY_ACTION": policy_action[:, 0],
        "OBSERVED_ACTION_GOOD": (next_inr >= 2.) & (next_inr <= 3.)
    })

    # Compute stats
    stats = classification_report(
        actions_reward["OBSERVED_ACTION_GOOD"].astype(int),
        actions_reward["OBSERVED_ACTION"] == actions_reward["POLICY_ACTION"],
        output_dict=True
    )
    sens, spec = stats["1"]["recall"], stats["0"]["recall"]
    jindex = sens + spec - 1.

    return sens, spec, jindex


def eval_ttr_at_agreement(policy, replay_buffer, threshold):
    """
    Estimate the TTR in the trajectories which agree with the policy decisions.

    Args:
        policy: The policy object.
        replay_buffer: The replay buffer to evaluate on.
        threshold: The agreement threshold. Trajectories with mean absolute
                   disagreement less than this threshold will be selected.

    Returns:
        num_traj: The number of trajectories the policy agrees with at the given
                  threshold.
        num_trans: The number of transitions the policy agrees with at the given
                   threshold.
        ttr: The estimated time in therapeutic range of the trajectories.
    """
    pass


#     # Calculate rate of events
#     reward_np = np.array(reward.to("cpu")).transpose()[0]
#     event_flag = np.where(reward_np <= -10, 1, 0)
#     both_actions = pd.DataFrame(
#         {"CLINICIAN_ACTION": np.array(action.to("cpu")).transpose()[0],
#          "POLICY_ACTION": pred_action.transpose()[0],
#          "ADV_EVENTS": event_flag}
#     )
#     both_actions.loc[:, "DIFF_ACTIONS"] = (both_actions["POLICY_ACTION"] -
#                                            both_actions["CLINICIAN_ACTION"])
#     events_rate = both_actions[
#         both_actions["DIFF_ACTIONS"] == 0
#     ]["ADV_EVENTS"].mean()
# 
#     # print("---------------------------------------")
#     # print(
#     #     f"Evaluation over {eval_episodes} episodes: Validation % Reasonable "
#     #     f"Actions: {avg_reward:.3%}" +
#     #     (f" | Training % Reasonable Actions: {avg_reward_train:.3%}"
#     #      if train_replay_buffer is not None else "")
#     # )
#     # print("---------------------------------------")
# 
#     if train_replay_buffer is not None:
#         return avg_reward_train, events_rate, avg_reward, pred_action
# 
#     else:
#         return events_rate, avg_reward, pred_action


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
