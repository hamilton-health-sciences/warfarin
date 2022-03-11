"""Entry point for evaluation metric and plots for a given learned policy."""

import numpy as np

import torch

import pandas as pd

from plotnine import ggtitle, facet_wrap

from warfarin import config
from warfarin.utils import interpolate_inr, code_quantitative_decision
from warfarin.models.baselines import ThresholdModel, RandomModel, MaintainModel
from warfarin.evaluation.metrics import (eval_reasonable_actions,
                                         eval_classification,
                                         eval_at_agreement,
                                         eval_agreement_associations,
                                         compute_performance_tests,
                                         wis_returns)
from warfarin.evaluation.plotting import (plot_policy_heatmap,
                                          plot_agreement_ttr_curve,
                                          plot_wis_boxplot)


def evaluate_and_plot_policy(policy, replay_buffer, behavior_policy=None,
                             eval_state=None, plot=True,
                             compute_all_metrics=False, include_tests=False):
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
        (3) The above plots broken out by continent or race, as available.
        (4) The scatterplot of TTR at agreement by algorithm (per threshold).

    Args:
        policy: The learned policy.
        replay_buffer: The replay buffer of data to evaluate on.
        behavior_policy: If given, the BehaviorCloner policy for WIS estimates
                         of returns.
        eval_state: A pass-through state var that should be modified and
                    returned.
        plot: Whether or not to generate plots, because plotting is slow.
        compute_all_metrics: Whether or not to compute all metrics, or just WIS-
        include_tests: Whether to include statistical tests of performance.
                             estimated policy value.

    Returns:
        metrics: The dictionary of quantitative metrics, mapping name to value.
        plots: The dictionary of plots, mapping title to plot object. If the
               `plot` argument is `False`, this will be an empty dictionary.
        eval_state: The pass-through state var, modified to be received in the
                    next call to this function.
    """
    if eval_state is None:
        eval_state = {}

    # Extract policy decisions, observed decisions, and INR into dataframe
    state = torch.from_numpy(
        np.array(replay_buffer.observed_state).astype(np.float32)
    ).to(policy.device)
    obs_action = np.array(replay_buffer.observed_option)
    prev_dose = replay_buffer.df["WARFARIN_DOSE"]
    dose = replay_buffer.df.groupby(
        ["TRIAL", "SUBJID", "TRAJID"]
    )["WARFARIN_DOSE"].shift(-1)
    obs_action_quant = dose / prev_dose
    obs_action_quant[(dose == 0) & (prev_dose == 0)] = 1.
    policy_action = policy.select_action(state)[:, 0]

    if "CONTINENT" in replay_buffer.df.columns:
        continent_or_race = replay_buffer.df["CONTINENT"]
    else:
        continent_or_race = replay_buffer.df["RACE2"]

    df = pd.DataFrame(
        {"OBSERVED_ACTION": obs_action,
         "OBSERVED_ACTION_QUANT": obs_action_quant,
         "POLICY_ACTION": policy_action,
         "INR_VALUE": replay_buffer.df["INR_VALUE"],
         "CONTINENT": continent_or_race},
        index=replay_buffer.df.index
    )

    # TODO subset to `.state` index (not e.g. `.observed_state`)?
    # Although this may drop terminal trnasitions needed for TTR computation.

    # Next INR and whether it's in range
    df["NEXT_INR"] = df.groupby(
        ["TRIAL", "SUBJID", "TRAJID"]
    )["INR_VALUE"].shift(-1)
    df["NEXT_INR_IN_RANGE"] = (
        (df["NEXT_INR"] >= 2.) & (df["NEXT_INR"] <= 3.)
    ).astype(int)

    # Extract baseline policy decisions
    tm = ThresholdModel()
    rm = RandomModel(
        np.array(df["OBSERVED_ACTION"][~df["OBSERVED_ACTION"].isnull()])
    )
    mm = MaintainModel()
    df["PREVIOUS_INR"] = df.groupby(
        ["TRIAL", "SUBJID", "TRAJID"]
    )["INR_VALUE"].shift(1)
    # The threshold model prescribes exact changes, so these come out as percent
    # change while the other models use the discrete action space.
    df["THRESHOLD_ACTION_QUANT"] = tm.select_action_quant(
        np.array(df["PREVIOUS_INR"]),
        np.array(df["INR_VALUE"])
    )
    df["THRESHOLD_ACTION"] = code_quantitative_decision(
        df["THRESHOLD_ACTION_QUANT"]
    )
    df["RANDOM_ACTION"] = rm.select_action(len(df))
    df["MAINTAIN_ACTION"] = mm.select_action(len(df))

    # Map actions to the means of their bins
    code_to_quant = replay_buffer.option_means

    df["POLICY_ACTION_QUANT"] = df["POLICY_ACTION"].map(code_to_quant)
    df["RANDOM_ACTION_QUANT"] = df["RANDOM_ACTION"].map(code_to_quant)
    df["MAINTAIN_ACTION_QUANT"] = df["MAINTAIN_ACTION"].map(code_to_quant)

    # Compute differences
    df["POLICY_ACTION_DIFF"] = (df["POLICY_ACTION_QUANT"] -
                                df["OBSERVED_ACTION_QUANT"])
    df["THRESHOLD_ACTION_DIFF"] = (df["THRESHOLD_ACTION_QUANT"] -
                                   df["OBSERVED_ACTION_QUANT"])
    df["RANDOM_ACTION_DIFF"] = (df["RANDOM_ACTION_QUANT"] -
                                df["OBSERVED_ACTION_QUANT"])
    df["MAINTAIN_ACTION_DIFF"] = (df["MAINTAIN_ACTION_QUANT"] -
                                  df["OBSERVED_ACTION_QUANT"])
    action_diff_cols = [c for c in df.columns if "ACTION_DIFF" in c]

    # Compute equality of action
    df["POLICY_ACTION_EQ"] = np.abs(df["POLICY_ACTION_DIFF"]) <= 0.05
    df["THRESHOLD_ACTION_EQ"] = np.abs(df["THRESHOLD_ACTION_DIFF"]) <= 0.05
    df["RANDOM_ACTION_EQ"] = np.abs(df["RANDOM_ACTION_DIFF"]) <= 0.05
    df["MAINTAIN_ACTION_EQ"] = np.abs(df["MAINTAIN_ACTION_DIFF"]) <= 0.05
    action_eq_cols = [c for c in df.columns if "ACTION_EQ" in c]

    # Use linearly interpolated INR to compute TTR
    if "inr_interp" in eval_state:
        inr_interp = eval_state["inr_interp"]
    else:
        inr_interp = interpolate_inr(df[["INR_VALUE"]])
        inr_interp["INR_IN_RANGE"] = ((inr_interp["INR_VALUE"] >= 2.) &
                                      (inr_interp["INR_VALUE"] <= 3.))
        eval_state["inr_interp"] = inr_interp
    ttr = inr_interp.groupby(
        ["TRIAL", "SUBJID", "TRAJID"]
    )["INR_IN_RANGE"].mean().to_frame()[["INR_IN_RANGE"]]
    traj_length = (
        df.reset_index().groupby(
            ["TRIAL", "SUBJID", "TRAJID"]
        )["STUDY_DAY"].max() -
        df.reset_index().groupby(
            ["TRIAL", "SUBJID", "TRAJID"]
        )["STUDY_DAY"].min()
    )
    traj_length.name = "TRAJECTORY_LENGTH"
    quant_disagreement = np.abs(df[action_diff_cols])
    disagreement = df[action_eq_cols]
    disagreement_ttr = disagreement.join(quant_disagreement).join(ttr)
    disagreement_ttr = disagreement_ttr.groupby(
        ["TRIAL", "SUBJID", "TRAJID"]
    ).mean()
    disagreement_ttr = disagreement_ttr.join(traj_length)
    disagreement_ttr = disagreement_ttr.rename(
        columns={"INR_IN_RANGE": "APPROXIMATE_TTR"}
    )

    # Extract event info
    raw_df = replay_buffer._raw_df.copy()
    events = (
        raw_df[config.ADV_EVENTS].groupby(
            ["TRIAL", "SUBJID", "TRAJID"]
        ).sum() > 0
    ).astype(int)
    disagreement_ttr_events = disagreement_ttr.join(events)
    disagreement_ttr_events["ANY_EVENT"] = (
        disagreement_ttr_events[config.ADV_EVENTS].sum(axis=1) > 0
    ).astype(int)

    # Extract dataframe for hierarchical TTR model
    hierarchical_ttr = df.join(
        ttr.rename(
            columns={"INR_IN_RANGE": "APPROXIMATE_TTR"}
        )
    ).join(traj_length)

    # Compute results
    metrics, wis_bootstrap_df = compute_metrics(
        df, disagreement_ttr_events, eval_state, compute_all_metrics, policy,
        behavior_policy, replay_buffer, include_tests
    )
    if plot:
        plots = compute_plots(df, disagreement_ttr_events, metrics,
                              wis_bootstrap_df)
    else:
        plots = {}
    if "prev_selected_actions" in eval_state:
        metrics["action_change"] = np.abs(
            policy_action - eval_state["prev_selected_actions"]
        ).mean()
    else:
        metrics["action_change"] = np.max(policy_action)
    eval_state["prev_selected_actions"] = policy_action

    metric_names = list(metrics.keys())
    for k in metric_names:
        if isinstance(metrics[k], np.float64):
            metrics[k] = float(metrics[k])
        elif isinstance(metrics[k], np.int64):
            metrics[k] = int(metrics[k])

    return metrics, plots, hierarchical_ttr, eval_state


def compute_metrics(df, disagreement_ttr, eval_state, compute_all_metrics,
                    learned_policy, behavior_policy, replay_buffer,
                    include_tests):
    stats = {}

    # WIS estimates of returns
    stats, wis_bootstrap_df = wis_returns(df, replay_buffer, learned_policy,
                                          behavior_policy, compute_all_metrics)

    # Reasonable-ness
    prop_reasonable = eval_reasonable_actions(df)
    stats["proportion_reasonable_actions"] = prop_reasonable

    # Classification metrics
    if compute_all_metrics:
        policy_stats = eval_classification(df, "POLICY_ACTION", "POLICY")
        threshold_stats = eval_classification(df, "THRESHOLD_ACTION", "THRESHOLD")
        maintain_stats = eval_classification(df, "MAINTAIN_ACTION", "MAINTAIN")
        random_stats = eval_classification(df, "RANDOM_ACTION", "RANDOM")
        stats = {**policy_stats, **threshold_stats, **maintain_stats, **random_stats,
                 **stats}

    # TTR at agreement
    if compute_all_metrics:
        agreement_stats = eval_at_agreement(disagreement_ttr)
        stats = {**stats, **agreement_stats}

    # Statistical tests
    if include_tests:
        performance_tests = compute_performance_tests(disagreement_ttr)
        stats = {**stats, **performance_tests}

    # Agreement/TTR + events associations
    if compute_all_metrics:
        association_stats = eval_agreement_associations(disagreement_ttr)
        stats = {**stats, **association_stats}

    # Ensure integer types are correct
    for k, v in stats.items():
        if pd.api.types.is_int64_dtype(v):
            stats[k] = np.array(v).astype(int).item()

    return stats, wis_bootstrap_df


def compute_plots(df, disagreement_ttr, metrics, wis_bootstrap_df):
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
    obs_df = df[["OBSERVED_ACTION", "INR_VALUE", "CONTINENT"]].copy()
    obs_df.columns = ["ACTION", "INR_VALUE", "CONTINENT"]
    plots["heatmap/observed"] = (
        plot_policy_heatmap(obs_df.copy()) +
        ggtitle("Observed Policy")
    )
    plots["heatmap/observed/continent"] = (
        plot_policy_heatmap(obs_df.copy(), group_vars=["CONTINENT"]) +
        ggtitle("Observed Policy by Continent")
    )

    # RL policy heatmap
    rl_df = df[["POLICY_ACTION", "INR_VALUE", "CONTINENT"]].copy()
    rl_df.columns = ["ACTION", "INR_VALUE", "CONTINENT"]
    plots["heatmap/learned"] = (
        plot_policy_heatmap(rl_df.copy()) +
        ggtitle("RL Policy")
    )
    plots["heatmap/learned/continent"] = (
        plot_policy_heatmap(rl_df.copy(), group_vars=["CONTINENT"]) +
        ggtitle("RL Policy by Continent")
    )

    # Threshold policy heatmap
    threshold_df = df[["THRESHOLD_ACTION", "INR_VALUE", "CONTINENT"]].copy()
    threshold_df.columns = ["ACTION", "INR_VALUE", "CONTINENT"]
    plots["heatmap/threshold"] = (
        plot_policy_heatmap(threshold_df.copy()) +
        ggtitle("Benchmark Policy")
    )
    plots["heatmap/threshold/continent"] = (
        plot_policy_heatmap(threshold_df.copy(), group_vars=["CONTINENT"]) +
        ggtitle("Benchmark Policy by Continent")
    )

    # WIS plot
    if wis_bootstrap_df is not None:
        plots["wis/comparison_boxplot"] = plot_wis_boxplot(wis_bootstrap_df)

    # Agreement curves and histograms
    agreement_plots = plot_agreement_ttr_curve(
        df, disagreement_ttr
    )
    plots = {**plots, **agreement_plots}

    # Break out all plots by continent
    plots_all = {}
    for plot_name, plot in plots.items():
        plots_all[plot_name] = plot
        for subvar in ["CONTINENT"]:
            plots_all[f"{plot_name}/{subvar}"] = plot + facet_wrap(subvar)

    return plots_all
