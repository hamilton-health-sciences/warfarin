"""Plotting the results of the modeling."""

import itertools

from collections import OrderedDict

import numpy as np

import pandas as pd

from plotnine import *  # pylint: disable=wildcard-import,unused-wildcard-import

from skmisc.loess import loess

from warfarin import config


def _loess_predictive_interval(data, xseq, **params):
    """
    Custom method for plotting prediction interval from LOESS in plotnine.
    """
    x, y = data["x"], data["y"]

    l = loess(x, y)
    l.fit()
    pred = l.predict(xseq, stderror=True)
    conf = pred.confidence()

    lowess = pred.values

    data = pd.DataFrame({
        "x": xseq,
        "y": lowess
    })

    if params["se"]:
        ll = conf.lower
        ul = conf.upper
        data["se"] = (ul - ll) / (1.96 * 2)
        data["ymin"] = ll
        data["ymax"] = ul

    return data


def plot_policy_heatmap(df, group_vars=None):
    """
    Plot the heatmap of current INR-decision frequency.

    Args:
        df: A dataframe containing columns "ACTION" and "INR_BIN".

    Returns:
        plot: The heatmap object.
    """
    if group_vars is None:
        group_vars = []

    # Map the action to its name
    df["ACTION"] = pd.Categorical(
        df["ACTION"].map(
            dict(zip(np.arange(7).astype(int), config.ACTION_LABELS))
        ),
        ordered=True,
        categories=config.ACTION_LABELS
    )

    # Bin INR
    df["INR_BIN"] = pd.cut(df["INR_VALUE"],
                           config.INR_BIN_BOUNDARIES,
                           right=False,
                           labels=config.INR_BIN_LABELS)
    if "2 - 3" in config.INR_BIN_LABELS:
        df.loc[df["INR_VALUE"] == 3., "INR_BIN"] = "2 - 3"
    if "3 - 3.5" in config.INR_BIN_LABELS:
        df.loc[df["INR_VALUE"] == 3.5, "INR_BIN"] = "3 - 3.5"

    # Plot
    plot_df = df[group_vars + ["INR_BIN", "ACTION"]].value_counts()
    plot_df = plot_df.reindex(
        list(
            itertools.product(
                *[df[c].cat.categories
                  for c in group_vars + ["INR_BIN", "ACTION"]]
            )
        )
    ).fillna(0)
    plot_df = plot_df.reset_index()
    plot_df.columns = group_vars + ["INR_BIN", "ACTION", "COUNT"]
    for c in group_vars + ["INR_BIN", "ACTION"]:
        plot_df[c] = pd.Categorical(plot_df[c],
                                    ordered=df[c].cat.ordered,
                                    categories=df[c].cat.categories)
    plot_df = plot_df.set_index(group_vars + ["INR_BIN", "ACTION"])
    plot_df = plot_df / plot_df.reset_index().groupby(
        group_vars + ["INR_BIN"]
    )[["COUNT"]].sum()
    plot_df.columns = ["PROPORTION"]
    plot_df = plot_df.reset_index()
    plot_df["PERCENTAGE"] = plot_df["PROPORTION"].map(
        lambda frac: f"{(frac*100):.2f}%"
    )
    plot_df.loc[plot_df["PROPORTION"] == 0., "PERCENTAGE"] = ""
    plot_df["SHOW"] = (plot_df["PROPORTION"] != 0.).astype(float)
    plot_df["INR_BIN"] = pd.Categorical(plot_df["INR_BIN"],
                                        ordered=True,
                                        categories=df["INR_BIN"].cat.categories)
    if len(group_vars) > 0:
        size = 6
    else:
        size = 10

    plot = (
        ggplot(plot_df,
               aes(x="INR_BIN", y="ACTION", fill="PROPORTION", alpha="SHOW")) +
        geom_tile() +
        geom_text(aes(label="PERCENTAGE"), color="#43464B", size=size) +
        xlab("INR") +
        ylab("Decision") +
        scale_fill_gradient(low="#FFFFFF", high="#4682B4", guide=False) +
        scale_alpha(guide=False) +
        theme(axis_line_x=element_blank(),
              axis_line_y=element_blank(),
              panel_grid=element_blank())
    )
    if group_vars:
        plot += facet_wrap(group_vars[0])

    return plot


def plot_agreement_ttr_curve(df, disagreement_ttr):
    """
    Plot absolute agreement vs. TTR.

    Both returned plots are weighted by trajectory length to avoid spurious
    differences due to short trajectories.

    Args:
        df: Dataframe with raw decision information.
        disagreement_ttr: Dataframe with information on TTR and agreeement levels.

    Returns:
        agreement_curve: The agreement curve plot object.
        histogram_curve: The histogram plot object.
    """
    plots = {}

    diff_algo_name_map = {
        "POLICY_ACTION_DIFF": "RL Algorithm",
        "THRESHOLD_ACTION_DIFF": "Rule-Based",
        "MAINTAIN_ACTION_DIFF": "Always Maintain",
        "RANDOM_ACTION_DIFF": "Random"
    }
    df = df.reset_index().drop_duplicates(
        subset=["TRIAL", "SUBJID", "TRAJID"]
    ).set_index(["TRIAL", "SUBJID", "TRAJID"])

    disagreement_ttr = disagreement_ttr.drop(
        [c for c in disagreement_ttr.columns if "_EQ" in c],
        axis=1
    )
    plot_df = disagreement_ttr.join(df[["CONTINENT"]])

    plot_df = plot_df.melt(id_vars=[*config.ADV_EVENTS,
                                    "ANY_EVENT",
                                    "APPROXIMATE_TTR",
                                    "TRAJECTORY_LENGTH",
                                    "CONTINENT"])
    plot_df.columns = [*config.ADV_EVENTS,
                       "ANY_EVENT",
                       "APPROXIMATE_TTR",
                       "TRAJECTORY_LENGTH",
                       "CONTINENT",
                       "MODEL",
                       "MEAN_ABSOLUTE_AGREEMENT"]
    plot_df["MODEL"] = plot_df["MODEL"].map(diff_algo_name_map)
    plot_df["APPROXIMATE_TTR"] *= 100.
    plot_df["MEAN_ABSOLUTE_AGREEMENT"] *= 100.
    mean_abs_diff_label = ("Mean Absolute Difference Between Algorithm & "
                           "Observed Dose Change (%)")
    plots["absolute_agreement/ttr/curve"] = (
        ggplot(plot_df,
               aes(x="MEAN_ABSOLUTE_AGREEMENT",
                   y="APPROXIMATE_TTR",
                   weight="TRAJECTORY_LENGTH",
                   group="MODEL",
                   color="MODEL")) +
        geom_smooth(method=_loess_predictive_interval) +
        xlim([0., 50.]) +
        ylim([0., 100.]) +
        xlab(mean_abs_diff_label) +
        ylab("TTR (%)") +
        scale_color_discrete(name="Algorithm")
    )

    # Plot event curves. We don't weight these by trajectory length in days,
    # unlike TTR, so as to be consistent with the event rate per unit time.
    for event_name in config.ADV_EVENTS + ["ANY_EVENT"]:
        plots[f"absolute_agreement/events/{event_name}/curve"] = (
            ggplot(plot_df,
                   aes(x="MEAN_ABSOLUTE_AGREEMENT",
                       y=event_name,
                       group="MODEL",
                       color="MODEL")) +
            geom_smooth(method=_loess_predictive_interval) +
            coord_cartesian(xlim=[0., 50.], ylim=[0., 0.1]) +
            xlab(mean_abs_diff_label) +
            ylab(f"Rate of {event_name}") +
            scale_color_discrete(name="Algorithm")
        )

    # Plot histogram of agreement
    plots["absolute_agreement/histogram"] = (
        ggplot(plot_df,
               aes(x="MEAN_ABSOLUTE_AGREEMENT",
                   group="MODEL",
                   fill="MODEL",
                   weight="TRAJECTORY_LENGTH")) +
        geom_histogram(binwidth=1.) +
        xlim([0., 50.]) +
        xlab(mean_abs_diff_label) +
        ylab("Count") +
        scale_fill_discrete(name="Algorithm")
    )

    # Plot density of agreement
    plots["absolute_agreement/density"] = (
        ggplot(plot_df,
               aes(x="MEAN_ABSOLUTE_AGREEMENT",
                   group="MODEL",
                   fill="MODEL",
                   weight="TRAJECTORY_LENGTH")) +
        geom_density(alpha=0.5) +
        xlim([0., 50.]) +
        xlab(mean_abs_diff_label) +
        ylab("Relative Frequency") +
        scale_fill_discrete(name="Algorithm")
    )

    # Plot TTR @ agreement of each algorithm as boxplot at each threshold
    for threshold in config.AGREEMENT_THRESHOLDS:
        cols = ["APPROXIMATE_TTR", "TRAJECTORY_LENGTH",
                *list(diff_algo_name_map.keys())]
        plot_df = disagreement_ttr[cols].melt(
            id_vars=["APPROXIMATE_TTR", "TRAJECTORY_LENGTH"]
        ).dropna()
        plot_df["Algorithm"] = plot_df["variable"].map(diff_algo_name_map)
        plot_df = plot_df[plot_df["value"] < threshold]
        plot_df["TTR (%)"] = plot_df["APPROXIMATE_TTR"] * 100
        plots[f"absolute_agreement/ttr/{threshold}_scatter"] = (
            ggplot(plot_df,
                   aes(x="Algorithm",
                       y="TTR (%)",
                       group="Algorithm",
                       fill="Algorithm",
                       size="TRAJECTORY_LENGTH")) +
            geom_jitter() +
            scale_fill_discrete(guide=False) +
            scale_size(guide=False)
        )

    return plots


def plot_wis_boxplot(df):
    plot_df = df.melt()
    algo_names = OrderedDict([
        ["clinician_value", "Observed Clinicians"],
        ["threshold_value", "Benchmark (Deterministic)"],
        ["policy_hard_value", "RL Policy (Deterministic)"],
        ["policy_value", "RL Policy (Probabilistic)"]
    ])
    plot_df["algo"] = plot_df["variable"].map(dict(algo_names))
    plot_df["algo"] = pd.Categorical(
        plot_df["algo"],
        algo_names.values(),
        ordered=True
    )
    # TODO pull in actual means?
    plt = (ggplot(plot_df) +
           geom_boxplot(aes(x="algo", y="value", fill="algo")) +
           scale_fill_discrete(guide=False) +
           xlab("") +
           ylab("Estimated Value (Discounted TTR)") +
           coord_flip())

    return plt
