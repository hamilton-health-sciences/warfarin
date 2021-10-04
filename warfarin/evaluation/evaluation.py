"""Entry point for evaluation metric and plots for a given learned policy."""

import numpy as np

import torch

import pandas as pd

from plotnine import ggtitle, facet_wrap

from warfarin import config
from warfarin.utils import interpolate_inr
from warfarin.models.baselines import ThresholdModel, RandomModel, MaintainModel
from warfarin.evaluation.metrics import (eval_reasonable_actions,
                                         eval_classification,
                                         eval_at_agreement,
                                         compute_performance_tests)
from warfarin.evaluation.plotting import (plot_policy_heatmap,
                                          plot_agreement_ttr_curve)


def evaluate_and_plot_policy(policy, replay_buffer, eval_state=None, plot=True,
                             include_tests=False):
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
        (4) The scatterplot of TTR at agreement by algorithm (per threshold).

    Args:
        policy: The learned policy.
        replay_buffer: The replay buffer of data to evaluate on.
        eval_state: A pass-through state var that should be modified and
                    returned.
        plot: Whether or not to generate plots, because plotting is slow.
        include_tests: Whether or not to generate statistical tests.

    Returns:
        metrics: The dictionary of quantitative metrics, mapping name to value.
        plots: The dictionary of plots, mapping title to plot object. If the
               `plot` argument is `False`, this will be an empty dictionary.
        eval_state: The pass-through state var, modified to be received in the
                    next call to this function.
    """
    # Extract policy decisions, observed decisions, and INR into dataframe
    state = torch.from_numpy(
        np.array(replay_buffer.observed_state).astype(np.float32)
    ).to(policy.device)
    obs_action = np.array(replay_buffer.observed_option)
    obs_action_quant = replay_buffer.df.groupby(
        ["TRIAL", "SUBJID", "TRAJID"]
    )["WARFARIN_DOSE"].shift(-1) / replay_buffer.df["WARFARIN_DOSE"] - 1.
    policy_action = policy.select_action(state)[:, 0]

    df = pd.DataFrame(
        {"OBSERVED_ACTION": obs_action,
         "OBSERVED_ACTION_QUANT": obs_action_quant,
         "POLICY_ACTION": policy_action,
         "INR_VALUE": replay_buffer.df["INR_VALUE"],
         "CONTINENT": replay_buffer.df["CONTINENT"]},
        index=replay_buffer.df.index
    )

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
    df["RANDOM_ACTION"] = rm.select_action(len(df))
    df["MAINTAIN_ACTION"] = mm.select_action(len(df))

    # Map actions to the means of their bins
    sel = np.isfinite(df["OBSERVED_ACTION_QUANT"])
    code_to_quant = df[sel].groupby(
        "OBSERVED_ACTION"
    )["OBSERVED_ACTION_QUANT"].mean().to_dict()

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

    # Compute algorithm-observed differences
    action_diff_cols = [c for c in df.columns if "ACTION_DIFF" in c]

    # Use linearly interpolated INR to compute TTR
    inr_interp = interpolate_inr(df[["INR_VALUE"]])
    inr_interp["INR_IN_RANGE"] = ((inr_interp["INR_VALUE"] >= 2.) &
                                  (inr_interp["INR_VALUE"] <= 3.))
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
    disagreement = np.abs(df[action_diff_cols])
    disagreement_ttr = disagreement.join(ttr)
    disagreement_ttr = disagreement_ttr.groupby(
        ["TRIAL", "SUBJID", "TRAJID"]
    ).mean()
    disagreement_ttr = disagreement_ttr.join(traj_length)
    disagreement_ttr = disagreement_ttr.rename(
        columns={"INR_IN_RANGE": "APPROXIMATE_TTR"}
    )

    # Extract event info
    events = (
        replay_buffer._raw_df[config.ADV_EVENTS].groupby(
            ["TRIAL", "SUBJID", "TRAJID"]
        ).sum() > 0
    ).astype(int)
    disagreement_ttr_events = disagreement_ttr.join(events)
    disagreement_ttr_events["ANY_EVENT"] = (
        disagreement_ttr_events[config.ADV_EVENTS].sum(axis=1) > 0
    ).astype(int)

    # Compute results
    metrics = compute_metrics(df, disagreement_ttr_events, eval_state,
                              include_tests)
    if plot:
        plots = compute_plots(df, disagreement_ttr_events)
    else:
        plots = {}
    eval_state = {"prev_selected_actions": policy_action}

    return metrics, plots, eval_state


def compute_metrics(df, disagreement_ttr, eval_state, include_tests):
    stats = {}

    # Reasonable-ness
    prop_reasonable = eval_reasonable_actions(df)
    stats["proportion_reasonable_actions"] = prop_reasonable

    # Classification metrics
    sens, spec, jstat, sens_dir, spec_dir, jstat_dir = eval_classification(
        df
    )
    stats["good_action_classification/sensitivity"] = sens
    stats["good_action_classification/specificity"] = spec
    stats["good_action_classification/jindex"] = jstat
    stats["good_action_dir_classification/sensitivity"] = sens_dir
    stats["good_action_dir_classification/specificity"] = spec_dir
    stats["good_action_dir_classification/jindex"] = jstat_dir

    # TTR at agreement
    agreement_stats = eval_at_agreement(disagreement_ttr)
    stats = {**stats, **agreement_stats}

    # Statistical tests
    if include_tests:
        performance_tests = compute_performance_tests(disagreement_ttr)
        stats = {**stats, **performance_tests}

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
    obs_df = df[["OBSERVED_ACTION", "INR_VALUE", "CONTINENT"]].copy()
    obs_df.columns = ["ACTION", "INR_VALUE", "CONTINENT"]
    plots["heatmap/observed"] = (
        plot_policy_heatmap(obs_df) +
        ggtitle("Observed Policy")
    )

    # RL policy heatmap
    rl_df = df[["POLICY_ACTION", "INR_VALUE", "CONTINENT"]].copy()
    rl_df.columns = ["ACTION", "INR_VALUE", "CONTINENT"]
    plots["heatmap/learned"] = (
        plot_policy_heatmap(rl_df) +
        ggtitle("RL Policy")
    )

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
