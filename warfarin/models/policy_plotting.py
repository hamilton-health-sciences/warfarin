import numpy as np

import pandas as pd

from plotnine import *

from warfarin import config


def plot_policy(policy, replay_buffer):
    # Extract policy decisions, observed decisions, and INR into dataframe
    buffer_size = replay_buffer.crt_size
    _, state, obs_action, next_state, reward, _ = replay_buffer.sample(
        np.arange(buffer_size),
        return_flag=False
    )
    policy_action = policy.select_action(state.cpu().numpy(), eval=True)
    inr = state[:, 0].cpu().numpy() * 4 + 0.5
    df = pd.DataFrame({
        "OBSERVED_ACTION": obs_action.cpu().numpy()[:, 0],
        "POLICY_ACTION": policy_action[:, 0],
        "INR": inr
    })

    # Map actions to labels
    df["OBSERVED_ACTION"] = pd.Categorical(
        df["OBSERVED_ACTION"].map(
            dict(zip(np.arange(7).astype(int), config.ACTION_LABELS))
        ),
        ordered=True,
        categories=config.ACTION_LABELS
    )
    df["POLICY_ACTION"] = pd.Categorical(
        df["POLICY_ACTION"].map(
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
    plot_df = df[["INR_BIN", "POLICY_ACTION"]].value_counts()
    plot_df.name = "COUNT"
    plot_df = plot_df / plot_df.reset_index().groupby("INR_BIN")["COUNT"].sum() 
    plot_df.name = "PROPORTION"
    plot_df = plot_df.reset_index()
    plot_df["PERCENTAGE"] = plot_df["PROPORTION"].map(
        lambda frac: f"{(frac*100):.2f}%"
    )
    plot = (
        ggplot(plot_df, aes(x="INR_BIN", y="POLICY_ACTION")) +
        geom_tile(aes(fill="PROPORTION")) +
        geom_text(aes(label="PERCENTAGE"), color="#43464B") +
        xlab("INR") +
        ylab("Decision") +
        scale_fill_gradient(low="#FFFFFF", high="#4682B4", guide=False) +
        theme(axis_line_x=element_blank(),
              axis_line_y=element_blank(),
              panel_grid=element_blank())
              # panel_background=element_blank(),
              # panel_grid_major=element_blank(),
              # panel_grid_minor=element_blank())
    )

    return {"policy_action_heatmap": plot}
