import os

import json

import torch

from ray.tune import Analysis

from warfarin.utils.smdp_buffer import SMDPReplayBuffer
from warfarin.models.smdp_dBCQ import discrete_BCQ
from warfarin.models.policy_eval import eval_policy
from warfarin.models.policy_plotting import plot_policy


def main(args):
    # Load up the metrics for models trained as part of the grid search
    analysis = Analysis("./ray_logs/dbcq")
    dfs = analysis.trial_dataframes
    # TODO remove this line:
    dfs = {k: df for k, df in dfs.items() if args.target_metric in df}

    # Get best model according to the chosen metric
    if args.mode == "max":
        max_metric = lambda k: dfs[k][args.target_metric].max()
        best_trial_name = max(dfs, key=max_metric)
    elif args.mode == "min":
        min_metric = lambda k: dfs[k][args.target_metric].min()
        best_trial_name = min(dfs, key=max_metric)
    df = dfs[best_trial_name]
    if args.mode == "max":
        idx = df[args.target_metric].idxmax()
    elif args.mode == "min":
        idx = df[args.target_metric].idxmin()
    best_trial_iter = df.iloc[idx]["training_iteration"]

    # Load the checkpointed model for evaluation
    config = analysis.get_all_configs()[best_trial_name]

    # TODO probably refactor this as it's duplicated from the tuning script rn
    # Data dimensionality from buffers
    num_actions = 7  # TODO
    state_dim = 56  # ...

    # Build the policy object
    policy = discrete_BCQ(
        num_actions,
        state_dim,
        "cuda",
        config["bcq_threshold"],
        config["discount"],
        config["optimizer"],
        {"lr": config["learning_rate"]},
        config["polyak_target_update"],
        config["target_update_freq"],
        config["tau"],
        0.1,
        0.1,
        1,
        0.,
        config["hidden_dim"],
        config["num_layers"]
    )
    state_dict_path = os.path.join(best_trial_name,
                                   f"checkpoint_{best_trial_iter:06d}",
                                   "model.pt")
    state = torch.load(state_dict_path)
    policy.Q.load_state_dict(state)

    # TODO don't hardcode device
    # Load the buffer we're evaluating on
    if "test" in args.buffer_path:
        raise ValueError("We're not testing on the test set yet.")
    buf = SMDPReplayBuffer.from_filename(
        data_path=args.buffer_path,
        batch_size=config["batch_size"],
        buffer_size=1e7,
        device="cuda"
    )
    buf.max_size = len(buf.data)

    # Compute evaluation metrics on the buffer
    metrics = eval_policy(policy, buf)
    plots = plot_policy(policy, buf)

    # Create output directories
    plots_dir = os.path.join(args.output_prefix, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Write config options (best hyperparameters)
    config_output = os.path.join(args.output_prefix, "config.json")
    json.dump(config, open(config_output, "w"))

    # Write quantitative metrics
    metrics_output = os.path.join(args.output_prefix, "metrics.json")
    json.dump(metrics, open(metrics_output, "w"))

    # Output plots
    for plot_name, plot in plots.items():
        plot_fn = os.path.join(plots_dir, f"{plot_name}.jpg")
        plot.save(plot_fn)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--buffer_path",
        type=str,
        required=True,
        help="Path to the buffer to evaluate on"
    )
    parser.add_argument(
        "--target_metric",
        type=str,
        required=True,
        help="The metric on the basis of which to select the model"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["min", "max"],
        help="Whether to maximize or minimize the target metric"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        required=True,
        help="The output directory for metrics and plots, need not exist"
    )
    parsed_args = parser.parse_args()

    main(parsed_args)
