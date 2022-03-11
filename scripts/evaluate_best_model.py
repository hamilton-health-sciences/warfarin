"""Evaluate the reinforcement learning model."""

import os

from warnings import warn

import json

import torch

from ray.tune import Analysis

import plotnine
from plotnine.exceptions import PlotnineError

from warfarin import config
from warfarin.models import SMDBCQ, BehaviorCloner
from warfarin.utils.modeling import get_dataloader
from warfarin.evaluation import evaluate_and_plot_policy


def main(args):
    # Set size of plots
    plotnine.options.figure_size = (8, 6)
    plotnine.options.dpi = 300

    # Load up the metrics for models trained as part of the grid search
    analysis = Analysis(args.logs_path)
    dfs = analysis.trial_dataframes

    # Select the best model by name and iteration number (useful for sanity
    # checks)
    if args.trial_name_filter and args.step_idx:
        best_trial_name = [
            k for k in dfs if args.trial_name_filter in k
        ][0]
        best_trial_iter = args.step_idx
    # Or more likely, select the best model by optimizing a given metric across
    # the hyperparameter sweep
    else:
        # Get best model according to the chosen metric
        if args.mode == "max":
            max_metric = lambda k: dfs[k].loc[
                dfs[k]["training_iteration"] >= (config.BCQ_EVAL_MIN_TRAINING_EPOCHS - 1),
                args.target_metric
            ].max()
            best_trial_name = max(dfs, key=max_metric)
        elif args.mode == "min":
            min_metric = lambda k: dfs[k].loc[
                dfs[k]["training_iteration"] >= (config.BCQ_EVAL_MIN_TRAINING_EPOCHS - 1),
                args.target_metric
            ].min()
            best_trial_name = min(dfs, key=max_metric)
        df = dfs[best_trial_name]
        if args.mode == "max":
            idx = df.loc[
                df["training_iteration"] >= (config.BCQ_EVAL_MIN_TRAINING_EPOCHS - 1),
                args.target_metric
            ].idxmax()
        elif args.mode == "min":
            idx = df.loc[
                df["training_iteration"] >= (config.BCQ_EVAL_MIN_TRAINING_EPOCHS - 1),
                args.target_metric
            ].idxmin()
        best_trial_iter = df.iloc[idx]["training_iteration"]

    # Load the checkpointed model for evaluation
    trial_config = analysis.get_all_configs()[best_trial_name]

    # Load the data
    train_data, _ = get_dataloader(
        data_path=args.train_data_path,
        cache_name="train_buffer.pkl",
        batch_size=trial_config["batch_size"],
        min_trajectory_length=config.MIN_TRAIN_TRAJECTORY_LENGTH,
        **config.REPLAY_BUFFER_PARAMS
    )
    data, _ = get_dataloader(
        data_path=args.data_path,
        cache_name="eval_buffer.pkl",
        batch_size=trial_config["batch_size"],
        state_transforms=train_data.state_transforms,
        option_means=train_data.option_means,
        **config.REPLAY_BUFFER_PARAMS
    )

    # TODO probably refactor this as it's duplicated from the tuning script rn
    # Construct the model and load it
    num_actions = len(config.ACTION_LABELS)
    policy = SMDBCQ(
        num_actions=num_actions,
        state_dim=data.state_dim,
        device="cuda",
        BCQ_threshold=trial_config["bcq_threshold"],
        discount=data.discount_factor,
        optimizer_parameters={"lr": trial_config["learning_rate"]},
        polyak_target_update=True,
        tau=trial_config["tau"],
        hidden_states=trial_config["hidden_dim"],
        num_layers=trial_config["num_layers"]
    )
    state_dict_path = os.path.join(best_trial_name,
                                   f"checkpoint_{best_trial_iter:06d}",
                                   "model.pt")
    policy.load(state_dict_path)

    # If using, load up the behavior policy for WIS returns estimates
    if args.behavior_policy_path:
        behavior_policy = BehaviorCloner.load(args.behavior_policy_path)
    else:
        behavior_policy = None

    # Compute evaluation metrics on the buffer
    metrics, plots, hierarchical_ttr, _ = evaluate_and_plot_policy(
        policy, data, compute_all_metrics=True, include_tests=True,
        behavior_policy=behavior_policy
    )

    # Create output directories
    plots_dir = os.path.join(args.output_prefix, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Write config options (best hyperparameters)
    config_output = os.path.join(args.output_prefix, "config.json")
    json.dump(trial_config, open(config_output, "w"))

    # Write quantitative metrics
    metrics_output = os.path.join(args.output_prefix, "metrics.json")
    json.dump(metrics, open(metrics_output, "w"))

    # Write dataframe for hierarchical ttr
    hierarchical_ttr_output = os.path.join(args.output_prefix,
                                           "hierarchical_ttr.csv")
    hierarchical_ttr.to_csv(hierarchical_ttr_output)

    # Output plots
    for plot_name, plot in plots.items():
        plot_name = plot_name.replace("/", "_")
        print(f"Saving {plot_name}...")
        plot_fn = os.path.join(plots_dir, f"{plot_name}.jpg")
        try:
            plot.save(plot_fn)
        except PlotnineError:
            warn(f"Failed to save plot {plot_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs_path",
        type=str,
        default="./ray_logs/dbcq",
        help="The path to the output Ray Tune logs"
    )
    parser.add_argument(
        "--trial_name_filter",
        type=str,
        required=False,
        help=("If given, will select this trial rather than based on an "
              "evaluation metric")
    )
    parser.add_argument(
        "--step_idx",
        type=int,
        required=False,
        help=("If given, will select this epoch rather than based on an "
              "evaluation metric")
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help="Path to data used to train the model"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the data to evaluate on"
    )
    parser.add_argument(
        "--target_metric",
        type=str,
        default=config.BCQ_EVAL_TARGET_METRIC,
        help="The metric on the basis of which to select the model"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=config.BCQ_EVAL_TARGET_MODE,
        choices=["min", "max"],
        help="Whether to maximize or minimize the target metric"
    )
    parser.add_argument(
        "--behavior_policy_path",
        type=str,
        required=False,
        help="Path to behavior policy savefile"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        required=True,
        help="The output directory for metrics and plots, need not exist"
    )
    parsed_args = parser.parse_args()

    main(parsed_args)
