import numpy as np

import pandas as pd

from plotnine import *

from warfarin import config


def plot_policy_heatmap(df):
    """
    Plot the heatmap of current INR-decision frequency.

    Args:
        df: A dataframe containing columns "ACTION" and "INR_BIN".

    Returns:
        plot: The heatmap object.
    """
    # Map the action to its name
    df["ACTION"] = pd.Categorical(
        df["ACTION"].map(
            dict(zip(np.arange(7).astype(int), config.ACTION_LABELS))
        ),
        ordered=True,
        categories=config.ACTION_LABELS
    )

    # Bin INR
    df["INR_BIN"] = pd.cut(df["INR"],
                           config.INR_BIN_BOUNDARIES,
                           right=False,
                           labels=config.INR_BIN_LABELS)
    if "2 - 3" in config.INR_BIN_LABELS:
        df.loc[df["INR"] == 3., "INR_BIN"] = "2 - 3"
    if "3 - 4" in config.INR_BIN_LABELS:
        df.loc[df["INR"] == 4., "INR_BIN"] = "3 - 4"

    # Plot
    plot_df = df[["INR_BIN", "ACTION"]].value_counts()
    plot_df.name = "COUNT"
    plot_df = plot_df / plot_df.reset_index().groupby("INR_BIN")["COUNT"].sum() 
    plot_df.name = "PROPORTION"
    plot_df = plot_df.reset_index()
    plot_df["PERCENTAGE"] = plot_df["PROPORTION"].map(
        lambda frac: f"{(frac*100):.2f}%"
    )
    plot = (
        ggplot(plot_df, aes(x="INR_BIN", y="ACTION")) +
        geom_tile(aes(fill="PROPORTION")) +
        geom_text(aes(label="PERCENTAGE"), color="#43464B") +
        xlab("INR") +
        ylab("Decision") +
        scale_fill_gradient(low="#FFFFFF", high="#4682B4", guide=False) +
        theme(axis_line_x=element_blank(),
              axis_line_y=element_blank(),
              panel_grid=element_blank())
    )

    return plot


def plot_agreement_ttr_curve(df):
    """
    Plot absolute agreement vs. TTR.

    Args:
        df: Dataframe with columns "OBSERVED_ACTION", "POLICY_ACTION", and
            "NEXT_INR".

    Returns:
        agreement_curve: The agreement curve plot object.
        histogram_curve: The histogram curve plot object.
    """
    action_map = {
        0: -0.25,
        1: -0.15,
        2: -0.05,
        3: 0.,
        4: 0.05,
        5: 0.15,
        6: 0.25
    }
    df["OBSERVED_ACTION"] = df["OBSERVED_ACTION"].map(action_map)
    df["POLICY_ACTION"] = df["POLICY_ACTION"].map(action_map)
    df["THRESHOLD_ACTION"] = df["THRESHOLD_ACTION"].map(action_map)
    df["RL_AGREEMENT"] = np.abs(
        df["POLICY_ACTION"] - df["OBSERVED_ACTION"]
    )
    df["THRESHOLD_AGREEMENT"] = np.abs(
        df["THRESHOLD_ACTION"] - df["OBSERVED_ACTION"]
    )
    df["NEXT_INR"] = df["INR"].shift(-1)
    df["NEXT_IN_RANGE"] = (df["NEXT_INR"] >= 2.) & (df["NEXT_INR"] <= 3.)

    # Drop transitions where threshold model does not provide prediction
    # df = df.loc[~pd.isnull(df["THRESHOLD_ACTION"])]

    # Drop trajectories where one of the models does not provide a prediction
    # df = df.loc[
    #     pd.isnull(df["THRESHOLD_ACTION"]).groupby("USUBJID_O_NEW").sum() == 0
    # ]
    # df = df.dropna()

    # Plot absolute agreement vs. TTR
    plot_df = df.groupby("USUBJID_O_NEW")[
        ["RL_AGREEMENT", "THRESHOLD_AGREEMENT", "NEXT_IN_RANGE"]
    ].mean()
    plot_df.columns = ["MEAN_RL_AGREEMENT",
                       "MEAN_THRESHOLD_AGREEMENT",
                       "APPROXIMATE_TTR"]
    plot_df["TRAJECTORY_LENGTH"] = df.groupby("USUBJID_O_NEW")[
        "NEXT_IN_RANGE"
    ].count()

    plot_df = plot_df.join(df[["CONTINENT"]])

    plot_df = plot_df.melt(id_vars=["APPROXIMATE_TTR",
                                    "TRAJECTORY_LENGTH",
                                    "CONTINENT"])
    plot_df.columns = ["APPROXIMATE_TTR",
                       "TRAJECTORY_LENGTH",
                       "CONTINENT",
                       "MODEL",
                       "MEAN_ABSOLUTE_AGREEMENT"]
    plot_df["MODEL"] = plot_df["MODEL"].map({
        "MEAN_RL_AGREEMENT": "RL Algorithm",
        "MEAN_THRESHOLD_AGREEMENT": "Benchmark Algorithm"
    })
    plot_df["APPROXIMATE_TTR"] *= 100.
    plot_df["MEAN_ABSOLUTE_AGREEMENT"] *= 100.
    # TODO figure out why loess segfaults or add CIs manually
    mean_abs_diff_label = ("Mean Absolute Difference Between Algorithm & "
                           "Observed Dose Change (%)")
    agreement_ttr = (
        ggplot(plot_df,
               aes(x="MEAN_ABSOLUTE_AGREEMENT",
                   y="APPROXIMATE_TTR",
                   weight="TRAJECTORY_LENGTH",
                   group="MODEL",
                   color="MODEL")) +
        geom_smooth(method="lowess") +
        # geom_point() +
        xlim([0., 50.]) +
        ylim([0., 100.]) +
        xlab(mean_abs_diff_label) +
        ylab("TTR (%)") +
        scale_color_discrete(name="Algorithm")
    )

    # Plot histogram of agreement
    agreement_histogram = (
        ggplot(plot_df,
               aes(x="MEAN_ABSOLUTE_AGREEMENT",
                   group="MODEL",
                   # color="MODEL",
                   fill="MODEL")) +
        geom_histogram(binwidth=1.) +
        xlim([0., 50.]) +
        xlab(mean_abs_diff_label) +
        ylab("Count") +
        scale_fill_discrete(name="Algorithm")
    )

    return agreement_ttr, agreement_histogram


def plot_policy(policy, replay_buffer):
    """
    Plot a policy using all available plots.

    Args:
        policy: The BCQ policy.
        replay_buffer: The replay buffer of data to evaluate on.

    Returns:
        plots: Dictionary mapping the name of the plot to the plot object.
    """
    # Extract policy decisions, observed decisions, and INR into dataframe
    buffer_size = replay_buffer.crt_size
    state = np.array(replay_buffer.get_state(replay_buffer.data))
    obs_action = np.array(replay_buffer.data["ACTION"])
    policy_action = policy.select_action(state, eval=True)
    inr = state[:, 0] * 4 + 0.5
    df = pd.DataFrame(
        {"OBSERVED_ACTION": obs_action,
         "POLICY_ACTION": policy_action[:, 0],
         "INR": inr},
        index=replay_buffer.data["USUBJID_O_NEW"]
    )

    # Extract continent
    continent_cols = [c for c in replay_buffer.data.columns if "CONTINENT" in c]
    df["CONTINENT"] = pd.Categorical(
        replay_buffer.data[continent_cols].idxmax(axis=1).apply(
            lambda s: s.split("_")[1]
        )
    )

    # Extract threshold policy decisions
    tm = ThresholdModel()
    df["PREVIOUS_INR"] = df.groupby("USUBJID_O_NEW")["INR"].shift(1)
    df["THRESHOLD_ACTION"] = tm.select_action(
        np.array(df["PREVIOUS_INR"]),
        np.array(df["INR"])
    )

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
        # for subvar in ["CONTINENT"]:
        #     plots_all[f"{plot_name}_{subvar}"] = plot + facet_wrap(subvar)
 
    return plots_all
