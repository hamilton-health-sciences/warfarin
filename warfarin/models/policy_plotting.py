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
    df["ABSOLUTE_AGREEMENT"] = np.abs(
        df["POLICY_ACTION"] - df["OBSERVED_ACTION"]
    )
    df["IN_RANGE"] = ((df["NEXT_INR"] >= 2.) &
                      (df["NEXT_INR"] <= 3.)).astype(int)
    import pdb; pdb.set_trace()


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
    _, state, obs_action, next_state, reward, _ = replay_buffer.sample(
        np.arange(buffer_size),
        return_flag=False
    )
    import pdb; pdb.set_trace()
    policy_action = policy.select_action(state.cpu().numpy(), eval=True)
    inr = state[:, 0].cpu().numpy() * 4 + 0.5
    next_inr = next_state[:, 0].cpu().numpy() * 4 + 0.5
    df = pd.DataFrame(
        {"OBSERVED_ACTION": obs_action.cpu().numpy()[:, 0],
         "POLICY_ACTION": policy_action[:, 0],
         "INR": inr,
         "NEXT_INR": next_inr},
        index=replay_buffer.data["USUBJID_O_NEW"]
    )

    # Observed policy heatmap
    obs_df = df[["OBSERVED_ACTION", "INR"]].copy()
    obs_df.columns = ["ACTION", "INR"]
    obs_heatmap = (
        plot_policy_heatmap(obs_df) +
        ggtitle("Observed Policy")
    )

    # RL policy heatmap
    rl_df = df[["POLICY_ACTION", "INR"]].copy()
    rl_df.columns = ["ACTION", "INR"]
    rl_heatmap = (
        plot_policy_heatmap(rl_df) +
        ggtitle("RL Policy")
    )

    # Agreement histogram
    agreement_curve, agreement_histogram = plot_agreement_ttr_curve(
        df.copy()
    )

    plots = {
        "observed_policy_heatmap": obs_heatmap,
        "learned_policy_heatmap": rl_heatmap,
        "absolute_agreement_curve": agreement_curve,
        "absolute_agreement_histogram": agreement_histogram
    }

    return plots
