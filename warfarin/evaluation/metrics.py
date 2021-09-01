import numpy as np

import torch

import pandas as pd

from sklearn.metrics import classification_report

from warfarin import config


# TODO get the device from the policy object?
device = "cuda"


def eval_reasonable_actions(df):
    """
    Evaluate the proportion of actions that are "reasonable", defined as
    choosing to maintain dose when in range, raise dose when below range, and
    increase dose when above range.

    Args:
        df: A dataframe containing columns "INR" and "POLICY_ACTION".

    Returns:
        prop: The proportion of policy actions which are "reasonable".
    """
    reasonable = np.sum(
        # Current INR is in range and the model chooses to maintain dose
        (((df["INR"] >= 2.) & (df["INR"] <= 3.)) & (df["POLICY_ACTION"] == 3)) |
        # Current INR is below range and model chooses to raise dose
        ((df["INR"] < 2.) & (df["POLICY_ACTION"] > 3)) |
        # Current INR is above range and model chooses to lower dose
        ((df["INR"] > 3.) & (df["POLICY_ACTION"] < 3))
    )
    total = len(df)
    prop = reasonable / total

    return prop


def eval_classification(df):
    """
    Evaluate each case of agreement between the policy and observed clinicians
    by identifying whether agreement predicts the next INR being in range.

    Note: We may present this as disagreement predicting the next INR being out
          of range. In that case, sensitivity and specificity are simply
          switched.

    Args:
        df: Dataframe containing columns "OBSERVED_ACTION", "POLICY_ACTION", and
            "NEXT_INR_IN_RANGE".

    Returns:
        sens, spec, jindex: The sensitivity, specificity, and Youden's J-index
                            for exact agreement (i.e. agreement is defined as
                            chosen actions are exactly equal).
        sens_dir, spec_dir, jindex_dir: The sensitivity, specificity, and
                                        Youden's J-index for directionally
                                        consistent (lower, maintain, raise)
                                        agreement.
    """
    # Compute stats
    stats = classification_report(
        df["NEXT_INR_IN_RANGE"],
        df["OBSERVED_ACTION"] == df["POLICY_ACTION"],
        output_dict=True
    )
    sens, spec = stats["1"]["recall"], stats["0"]["recall"]
    jindex = sens + spec - 1.

    # Compute directional stats
    same_direction = (
        # Both lower
        ((df["OBSERVED_ACTION"] < 3) & (df["POLICY_ACTION"] < 3)) |
        # Both maintain
        ((df["OBSERVED_ACTION"] == 3) & (df["POLICY_ACTION"] == 3)) |
        # Both raise
        ((df["OBSERVED_ACTION"] > 3) & (df["POLICY_ACTION"] > 3))
    )
    stats_dir = classification_report(
        df["NEXT_INR_IN_RANGE"],
        same_direction,
        output_dict=True
    )
    sens_dir, spec_dir = stats_dir["1"]["recall"], stats_dir["0"]["recall"]
    jindex_dir = sens_dir + spec_dir - 1.

    return sens, spec, jindex, sens_dir, spec_dir, jindex_dir


def eval_ttr_at_agreement(df):
    """
    Estimate the TTR in the trajectories which agree with the policy decisions.

    Args:
        threshold: The agreement threshold. Trajectories with mean absolute
                   disagreement less than this threshold will be selected.

    Returns:
        ttr_stats: The dictionary of metrics.
    """
    action_diff_cols = [c for c in df.columns if "ACTION_DIFF" in c]

    traj_length = df.groupby("USUBJID_O_NEW")["NEXT_INR_IN_RANGE"].count()
    traj_length.name = "TRAJECTORY_LENGTH"
    disagreement_ttr = np.abs(df[action_diff_cols + ["NEXT_INR_IN_RANGE"]])
    disagreement_ttr = disagreement_ttr.groupby("USUBJID_O_NEW").mean()
    disagreement_ttr = disagreement_ttr.join(traj_length)

    ttr_stats = {}
    for algo_diff_col in action_diff_cols:
        algo = algo_diff_col.split("_")[0]
        for threshold in config.AGREEMENT_THRESHOLDS:
            sel = (disagreement_ttr[algo_diff_col] < threshold)
            weighted_ttr = (
                disagreement_ttr[sel]["NEXT_INR_IN_RANGE"] *
                disagreement_ttr[sel]["TRAJECTORY_LENGTH"]
            ).sum() / disagreement_ttr[sel]["TRAJECTORY_LENGTH"].sum()
            ttr_stats[f"{threshold}_{algo}_ttr"] = weighted_ttr
            ttr_stats[f"{threshold}_{algo}_num_traj"] = sel.sum()
            ttr_stats[f"{threshold}_{algo}_num_trans"] = (
                disagreement_ttr["TRAJECTORY_LENGTH"][sel]
            ).sum()
            ttr_stats[f"{threshold}_{algo}_prop_traj"] = sel.sum() / len(sel)
    import pdb; pdb.set_trace()


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
