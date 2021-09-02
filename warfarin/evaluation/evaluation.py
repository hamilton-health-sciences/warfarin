"""Entry point for evaluation metric and plots for a given learned policy."""

from warnings import warn

import numpy as np

import pandas as pd

from plotnine import ggtitle, facet_wrap

from warfarin.models.baselines import ThresholdModel, RandomModel, MaintainModel
from warfarin.evaluation.metrics import (eval_reasonable_actions,
                                         eval_classification,
                                         eval_ttr_at_agreement)
from warfarin.evaluation.plotting import (plot_policy_heatmap,
                                          plot_agreement_ttr_curve)


def evaluate_and_plot_policy(policy, replay_buffer, eval_state=None):
    """
    Evaluate and plot a policy.

    Begins by building the dataframe with information needed for all evaluation
    metrics and plots, then passes this dataframe to the relevant metric
    computation and plotting functions.

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

    And also generates the following plots which may compare to the four
    baseline models (threshold model, random model, maintain dose model):

        (1) Heatmap of the current INR vs. policy option space.
        (2) Agreement curve and agreement histogram.
        (3) The above plots broken out by continent.

    Args:
        policy: The learned policy.
        replay_buffer: The replay buffer of data to evaluate on.
        eval_state: A pass-through state var that should be modified and
                    returned.

    Returns:
        metrics: The dictionary of quantitative metrics, mapping name to value.
        plots: The dictionary of plots, mapping title to plot object.
        eval_state: The pass-through state var, modified to be received in the
                    next call to this function.
    """
    # Extract policy decisions, observed decisions, and INR into dataframe
    buffer_size = replay_buffer.crt_size
    state = np.array(replay_buffer.get_state(replay_buffer.data))
    obs_action = np.array(replay_buffer.data["ACTION"])
    policy_action = policy.select_action(state, eval=True)[:, 0]
    inr = state[:, 0] * 4 + 0.5
    df = pd.DataFrame(
        {"OBSERVED_ACTION": obs_action,
         "POLICY_ACTION": policy_action,
         "INR": inr},
        index=replay_buffer.data["USUBJID_O_NEW"]
    )

    # Next INR and whether it's in range
    df["NEXT_INR"] = df["INR"].shift(-1)
    df["NEXT_INR_IN_RANGE"] = (
        (df["NEXT_INR"] >= 2.) & (df["NEXT_INR"] <= 3.)
    ).astype(int)

    # Extract continent
    continent_cols = [c for c in replay_buffer.data.columns if "CONTINENT" in c]
    df["CONTINENT"] = pd.Categorical(
        replay_buffer.data[continent_cols].idxmax(axis=1).apply(
            lambda s: s.split("_")[1]
        )
    )

    # Extract baseline policy decisions
    tm = ThresholdModel()
    rm = RandomModel()
    mm = MaintainModel()
    df["PREVIOUS_INR"] = df.groupby("USUBJID_O_NEW")["INR"].shift(1)
    df["THRESHOLD_ACTION"] = tm.select_action(
        np.array(df["PREVIOUS_INR"]),
        np.array(df["INR"])
    )
    df["RANDOM_ACTION"] = rm.select_action(len(df))
    df["MAINTAIN_ACTION"] = mm.select_action(len(df))

    # Map actions to the means of their bins
    # TODO use empirical means?
    code_to_quant = {
        0: -0.25,
        1: -0.15,
        2: -0.05,
        3: 0.,
        4: 0.05,
        6: 0.15,
        7: 0.25
    }
    df["OBSERVED_ACTION_QUANT"] = df["OBSERVED_ACTION"].map(code_to_quant)
    df["POLICY_ACTION_QUANT"] = df["POLICY_ACTION"].map(code_to_quant)
    df["THRESHOLD_ACTION_QUANT"] = df["THRESHOLD_ACTION"].map(code_to_quant)
    df["RANDOM_ACTION_QUANT"] = df["RANDOM_ACTION"].map(code_to_quant)
    df["MAINTAIN_ACTION_QUANT"] = df["MAINTAIN_ACTION"].map(code_to_quant)

    # Compute differences
    df["POLICY_ACTION_DIFF"] = (df["POLICY_ACTION_QUANT"] -
                                df["OBSERVED_ACTION_QUANT"])
    df["THRESHOLD_ACTION_DIFF"] = (df["THRESHOLD_ACTION_QUANT"] -
                                   df["OBSERVED_ACTION_QUANT"])
    df["RANDOM_ACTION_DIFF"] = (df["THRESHOLD_ACTION_QUANT"] -
                                df["OBSERVED_ACTION_QUANT"])
    df["MAINTAIN_ACTION_DIFF"] = (df["MAINTAIN_ACTION_QUANT"] -
                                  df["OBSERVED_ACTION_QUANT"])

    # Compute algorithm-observed differences
    action_diff_cols = [c for c in df.columns if "ACTION_DIFF" in c]

    traj_length = df.groupby("USUBJID_O_NEW")["NEXT_INR_IN_RANGE"].count()
    traj_length.name = "TRAJECTORY_LENGTH"
    disagreement_ttr = np.abs(df[action_diff_cols + ["NEXT_INR_IN_RANGE"]])
    disagreement_ttr = disagreement_ttr.groupby("USUBJID_O_NEW").mean()
    disagreement_ttr = disagreement_ttr.join(traj_length)

    # Compute results
    metrics = compute_metrics(df, disagreement_ttr)
    plots = compute_plots(df, disagreement_ttr)
    eval_state = {"prev_selected_actions": policy_action}

    return metrics, plots, eval_state


def compute_metrics(df, disagreement_ttr):
    stats = {}

    # Reasonable-ness
    prop_reasonable = eval_reasonable_actions(df)
    stats["proportion_reasonable_actions"] = prop_reasonable

    # Classification metrics
    sens, spec, jstat, sens_dir, spec_dir, jstat_dir = eval_classification(
        df
    )
    stats["sensitivity_good_actions"] = sens
    stats["specificity_good_actions"] = spec
    stats["jindex_good_actions"] = jstat
    stats["sensitivity_good_actions_dir"] = sens_dir
    stats["specificity_good_actions_dir"] = spec_dir
    stats["jindex_good_actions_dir"] = jstat_dir

    # TTR at agreement
    agreement_ttr_stats = eval_ttr_at_agreement(disagreement_ttr)
    stats = {**stats, **agreement_ttr_stats}

    # Ensure integer types are correct
    for k, v in stats.items():
        if pd.api.types.is_int64_dtype(v):
            stats[k] = np.array(v).astype(int).item()

    return stats


def compute_plots(df, disagreement_ttr):
    """
    Plot a policy using all available plots.

    Args:
        df: Dataframe of model decisions, baseline model decisions, and relevant
            statistics for evaluation and plotting.
        disagreement_ttr: Dataframe of trajectory-level disagreements and TTR.

    Returns:
        plots: Dictionary mapping the name of the plot to the plot object.
    """
    plots = {}

    # Observed policy heatmap
    obs_df = df[["OBSERVED_ACTION", "INR", "CONTINENT"]].copy()
    obs_df.columns = ["ACTION", "INR", "CONTINENT"]
    plots["observed_policy_heatmap"] = (
        plot_policy_heatmap(obs_df) +
        ggtitle("Observed Policy")
    )

    # RL policy heatmap
    rl_df = df[["POLICY_ACTION", "INR", "CONTINENT"]].copy()
    rl_df.columns = ["ACTION", "INR", "CONTINENT"]
    plots["learned_policy_heatmap"] = (
        plot_policy_heatmap(rl_df) +
        ggtitle("RL Policy")
    )

    # Agreement curves and histograms
    agreement_curve, agreement_histogram = plot_agreement_ttr_curve(
        df.copy()
    )
    plots["absolute_agreement_curve"] = agreement_curve
    plots["absolute_agreement_histogram"] = agreement_histogram

    # Break out all plots by continent
    plots_all = {}
    for plot_name, plot in plots.items():
        plots_all[plot_name] = plot
        for subvar in ["CONTINENT"]:
            plots_all[f"{plot_name}_{subvar}"] = plot + facet_wrap(subvar)
 
    return plots_all
