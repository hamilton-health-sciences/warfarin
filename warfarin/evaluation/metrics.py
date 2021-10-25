"""Quantitative evaluation metrics."""

import numpy as np

import pandas as pd

import statsmodels.api as sm

import scipy as sp
import scipy.stats

import torch
from torch.nn import functional as F

from warfarin import config


def _compute_exact_binomial_ci(r, n, alpha):
    f = sp.stats.f.ppf(1 - alpha / 2., 2 * (n - r + 1), 2 * r)

    ll = r / (r + (n - r + 1) * f)
    uu = (r + 1) * f / ((n - r) + (r + 1)*f)

    return ll, uu


def compute_sensitivity(y, yhat, alpha=0.05):
    y = y.astype(bool)
    yhat = yhat.astype(bool)
    r = (y & yhat).sum()
    n = y.sum()

    sens = r / n
    sens_l, sens_u = _compute_exact_binomial_ci(r, n, alpha)
    
    return sens, sens_l, sens_u


def compute_specificity(y, yhat, alpha=0.05):
    y = y.astype(bool)
    yhat = yhat.astype(bool)
    r = (~y & ~yhat).sum()
    n = (~y).sum()

    spec = r / n
    spec_l, spec_u = _compute_exact_binomial_ci(r, n, alpha)

    return spec, spec_l, spec_u


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
        (((df["INR_VALUE"] >= 2.) & (df["INR_VALUE"] <= 3.))
         & (df["POLICY_ACTION"] == 3)) |
        # Current INR is below range and model chooses to raise dose
        ((df["INR_VALUE"] < 2.) & (df["POLICY_ACTION"] > 3)) |
        # Current INR is above range and model chooses to lower dose
        ((df["INR_VALUE"] > 3.) & (df["POLICY_ACTION"] < 3))
    )
    total = len(df)
    prop = reasonable / total

    return prop


def eval_classification(df, policy_col, prefix):
    """
    Evaluate each case of agreement between the policy and observed clinicians
    by identifying whether agreement predicts the next INR being in range.

    Note: We may present this as disagreement predicting the next INR being out
          of range. In that case, sensitivity and specificity are simply
          switched.

    Args:
        df: Dataframe containing columns "OBSERVED_ACTION", `policy_col`, and
            "NEXT_INR_IN_RANGE".
        policy_col: The column to pull the policy decisions to come from.
        prefix: The prefix for the output metrics.

    Returns:
        stats: The sensitivity (+ CI), specificity (+ CI) and Youden's J-index
               for exact and directional agreement.
    """
    # Compute stats
    sens, sens_lower, sens_upper = compute_sensitivity(
        df["NEXT_INR_IN_RANGE"],
        df["OBSERVED_ACTION"] == df[policy_col]
    )
    spec, spec_lower, spec_upper = compute_specificity(
        df["NEXT_INR_IN_RANGE"],
        df["OBSERVED_ACTION"] == df[policy_col]
    )
    jindex = sens + spec - 1.

    # Compute directional stats
    same_direction = (
        # Both lower
        ((df["OBSERVED_ACTION"] < 3) & (df[policy_col] < 3)) |
        # Both maintain
        ((df["OBSERVED_ACTION"] == 3) & (df[policy_col] == 3)) |
        # Both raise
        ((df["OBSERVED_ACTION"] > 3) & (df[policy_col] > 3))
    )
    sens_dir, sens_dir_lower, sens_dir_upper = compute_sensitivity(
        df["NEXT_INR_IN_RANGE"],
        same_direction
    )
    spec_dir, spec_dir_lower, spec_dir_upper = compute_specificity(
        df["NEXT_INR_IN_RANGE"],
        same_direction
    )
    jindex_dir = sens_dir + spec_dir - 1.

    stats = {
        f"{prefix}/classification/sensitivity": sens,
        f"{prefix}/classification/sensitivity_lower": sens_lower,
        f"{prefix}/classification/sensitivity_upper": sens_upper,
        f"{prefix}/classification/specificity": spec,
        f"{prefix}/classification/specificity_lower": spec_lower,
        f"{prefix}/classification/specificity_upper": spec_upper,
        f"{prefix}/classification/jindex": jindex,
        f"{prefix}/classification_dir/sensitivity": sens_dir,
        f"{prefix}/classification_dir/sensitivity_lower": sens_dir_lower,
        f"{prefix}/classification_dir/sensitivity_upper": sens_dir_upper,
        f"{prefix}/classification_dir/specificity": spec_dir,
        f"{prefix}/classification_dir/specificity_lower": spec_dir_lower,
        f"{prefix}/classification_dir/specificity_upper": spec_dir_upper,
        f"{prefix}/classification_dir/jindex": jindex_dir
    }

    return stats


def eval_at_agreement(disagreement_ttr):
    """
    Estimate the TTR and event rate n the trajectories which agree with the
    policy decisions.

    Args:
        disagreement_ttr: Dataframe indexed by trajectory containing columns
                          containing "ACTION_DIFF", "APPROXIMATE_TTR" and
                          adverse event occurrences.

    Returns:
        stats: The dictionary of metrics.
    """
    action_diff_cols = [
        c for c in disagreement_ttr.columns if "ACTION_DIFF" in c
    ]

    stats = {}
    for algo_diff_col in action_diff_cols:
        algo = algo_diff_col.split("_")[0]
        for threshold in config.AGREEMENT_THRESHOLDS:
            sel = (disagreement_ttr[algo_diff_col] < threshold)
            weighted_ttr = (
                disagreement_ttr[sel]["APPROXIMATE_TTR"] *
                disagreement_ttr[sel]["TRAJECTORY_LENGTH"]
            ).sum() / disagreement_ttr[sel]["TRAJECTORY_LENGTH"].sum()
            stats[f"{threshold}_{algo}/ttr"] = weighted_ttr
            for event_name in config.ADV_EVENTS + ["ANY_EVENT"]:
                stats[f"{threshold}_{algo}/{event_name}_per_yr"] = (
                    disagreement_ttr[sel][event_name].sum() /
                    disagreement_ttr[sel]["TRAJECTORY_LENGTH"].sum() *
                    365.25
                )
            stats[f"{threshold}_{algo}/num_traj"] = sel.sum()
            stats[f"{threshold}_{algo}/num_trans"] = (
                disagreement_ttr["TRAJECTORY_LENGTH"][sel]
            ).sum()
            stats[f"{threshold}_{algo}/prop_traj"] = sel.sum() / len(sel)

    return stats


def compute_performance_tests(disagreement_ttr):
    performance_tests = {}

    event_names = config.ADV_EVENTS + ["ANY_EVENT"]

    sel = ((~np.isfinite(disagreement_ttr)).sum(axis=1) == 0)
    df = disagreement_ttr[sel]
    for thresh in config.AGREEMENT_THRESHOLDS:
        policy_agree = (df["POLICY_ACTION_DIFF"] < thresh).astype(float)
        threshold_agree = (df["THRESHOLD_ACTION_DIFF"] < thresh).astype(float)
        both_agree = policy_agree * threshold_agree
        lm_df = pd.DataFrame({
            "policy_agree": policy_agree,
            "threshold_agree": threshold_agree,
            "both_agree": both_agree,
            "ttr": df["APPROXIMATE_TTR"]
        })
        lm_df[event_names] = df[event_names]
        # Test `H0: beta_{policy_agree} - beta_{threshold_agree} = 0`
        # First for ttr
        _X = sm.add_constant(
            lm_df[["policy_agree", "threshold_agree"]]#, "both_agree"]]
        )
        _y = lm_df["ttr"]
        _w = df["TRAJECTORY_LENGTH"]
        results = sm.WLS(_y, _X, _w).fit()
        ttest = results.t_test(np.array([0., 1., -1.])).summary_frame()
        tval = float(ttest.iloc[:, 2])
        pval = float(ttest.iloc[:, 3])
        performance_tests[f"{thresh}_ttr_performance_t"] = tval
        performance_tests[f"{thresh}_ttr_performance_p"] = pval

        # TODO remove
        # Temporary: output power analysis on test set.
        N = 11_573
        beta_diff = ttest["coef"]
        t_crit = sp.stats.t.ppf(1. - (0.005 / 2), df=(N-4))
        true_crit = beta_diff / (
            np.sqrt(
                np.dot(
                    # results.cov_params() gives `inv(X^T X)`, use as estimate
                    # of true covariance
                    np.matmul(len(_X) * results.cov_params(),
                              np.array([0.,1.,-1.])),
                    np.array([0.,1.,-1.])
                )
            ) / np.sqrt(N)
        )
        power = 1. - sp.stats.t.cdf(t_crit - true_crit, df=(N-4))
        print(f"At threshold {thresh}, we see an approximate improvement "
              f"in TTR of {beta_diff}. We have a power of {power} to detect "
              f"that level of improvement in RE-LY alone.")

        # Then for each event
        for event in event_names:
            try:
                results = sm.Logit(
                    lm_df[event],
                    sm.add_constant(
                        lm_df[["policy_agree", "threshold_agree", "both_agree"]]
                    )
                ).fit()
                ttest = results.t_test(
                    np.array([0., 1., -1., 0.])
                ).summary_frame()
                tval = float(ttest.iloc[:, 2])
                pval = float(ttest.iloc[:, 3])
                performance_tests[f"{thresh}_{event}_performance_t"] = tval
                performance_tests[f"{thresh}_{event}_performance_p"] = pval
            except np.linalg.LinAlgError:
                print("Failed to compute improvement in {event} rate due to "
                      "singularity issues with the model. Event rate too low.")

    return performance_tests


def _compute_importance(df, policy_prob_col, behavior_prob_col, id_vars):
    importance = df[policy_prob_col] / df[behavior_prob_col]
    return importance.groupby(id_vars).prod()


def wis_returns(df, replay_buffer, learned_policy, behavior_policy):
    """
    Compute the naive importance sampling estimator of the mean return of the
    learned policy.

    Specifically, for the $n$th trajectory, define

        $\rho_n = \prod_t \frac{\pi_\ell(a_t | s_t)}{\pi_b(a_t | s_t)}$

    where $\pi_\ell$ is the learned policy, $\pi_b$ is the behavior policy, and
    $s_t, a_t$ is the state-option pair observed at time $t$. The product is
    taken over all timesteps in the trajectory. Then the weighted importance
    sampling estimate is:

        $WIS = \frac{\sum_n \rho_n R_n}{\sum_n \rho_n}$

    where $R_n = \sum_t \gamma^t r_t$ is the summed discounted rewards.

    Note: We obtain a distribution over options from the dBCQ model by
          `softmax`ing the Q-values. This improves the effective sample size
          by reducing the number of trajectories with zero importance.

    Args:
        df: The decisions of the models fo interest.
        replay_buffer: The replay buffer to evaluate on.
        learned_policy: The learned policy to evaluate.
        behavior_policy: The behavioral cloning model of the observed policy.

    Returns:
        stats: A dictionary of WIS-related statistics.
    """
    # Extract the state and observed option
    state = replay_buffer.tensors[1]
    option = replay_buffer.tensors[2][:, 0]

    # Extract the behavioral policy probabiltiies
    policy_q = learned_policy.masked_q(state)
    policy_probs = F.softmax(policy_q, dim=1)
    behavior_probs = behavior_policy(state)

    # Extract the benchmark policy "probabilities"
    threshold_option = df["THRESHOLD_ACTION"].loc[replay_buffer.option.index]
    threshold_option_eq = np.array(threshold_option == replay_buffer.option)
    threshold_option_prob = threshold_option_eq.astype(float)

    # Extract RL policy "probabilities" - soft and hard
    idx = torch.arange(policy_probs.shape[0]).long()
    policy_option_prob = policy_probs[idx, option].detach().cpu().numpy()
    policy_option = df["POLICY_ACTION"].loc[replay_buffer.option.index]
    policy_option_eq = np.array(policy_option == replay_buffer.option)
    policy_option_hard_prob = policy_option_eq.astype(float)

    # Behavior probabilities
    behavior_option_prob = behavior_probs[idx, option].detach().cpu().numpy()

    # Discount rewards from t = 0
    days_since_start = (
        replay_buffer.reward.reset_index()["STUDY_DAY"] -
        replay_buffer.reward.reset_index().groupby(
            ["TRIAL", "SUBJID", "TRAJID"]
        )["STUDY_DAY"].shift(1)
    ).fillna(0)
    days_since_start.index = replay_buffer.reward.index
    start_discount = replay_buffer.discount_factor**(days_since_start)

    # Construct stats needed for WIS
    wis_df = pd.DataFrame(
        {"threshold_importance": threshold_option_prob / behavior_option_prob,
         "policy_importance": policy_option_prob / behavior_option_prob,
         "policy_hard_importance": (
             policy_option_hard_prob / behavior_option_prob
         ),
         "clinician_importance": np.ones(behavior_option_prob.shape),
         "reward": start_discount * replay_buffer.reward},
        index=replay_buffer.reward.index
    )
    wis_df = wis_df.groupby(["TRIAL", "SUBJID" , "TRAJID"]).agg(
        {"threshold_importance": "prod",
         "policy_importance": "prod",
         "policy_hard_importance": "prod",
         "clinician_importance": "prod",
         "reward": "sum"}
    )

    def _weighted_mean(w, x):
        return ((w * x) / w.sum()).sum()

    # Compute WIS value estimates and 95% CIs
    idxs = [
        np.random.choice(len(wis_df), size=len(wis_df), replace=True)
        for _ in range(config.NUM_BOOTSTRAP_SAMPLES)
    ]
    stats, sample_distns = {}, {}
    for colname in wis_df.columns:
        if "_importance" in colname:
            algo_name = "_".join(colname.split("_")[:-1])
            algo_value_colname = f"{algo_name}_value"
            if colname not in sample_distns:
                sample_distns[algo_value_colname] = []
            algo_name = "_".join(colname.split("_")[:-1])
            stats[algo_value_colname] = _weighted_mean(
                wis_df[colname], wis_df["reward"]
            )
            for idx in idxs:
                sample_distns[algo_value_colname].append(
                    _weighted_mean(
                        wis_df[colname].iloc[idx], wis_df["reward"].iloc[idx]
                    )
                )
            ci_lower, ci_upper = np.quantile(sample_distns[algo_value_colname],
                                             [0.025, 0.975])
            stats[f"{algo_name}_value_ci_lower"] = ci_lower
            stats[f"{algo_name}_value_ci_upper"] = ci_upper

    stats = {f"wis/{k}": v for k, v in stats.items()}
    bootstrap_df = pd.DataFrame(sample_distns)

    return stats, bootstrap_df
