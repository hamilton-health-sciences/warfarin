import os

import torch

from ray.tune import Analysis

from warfarin.models.smdp_dBCQ import discrete_BCQ


def main(args):
    # Load up the metrics for models trained as part of the grid search
    analysis = Analysis("./ray_logs/dbcq")
    dfs = analysis.trial_dataframes
    # TODO remove this line:
    dfs = {k: df for k, df in dfs.items() if "val_jindex_good_actions" in df}

    # Get best model according to the chosen metric
    if args.mode == "max":
        max_metric = lambda k: dfs[k][args.target_metric].max()
        best_trial_name = max(dfs, key=max_metric)
    elif args.mode == "min":
        min_metric = lambda k: dfs[k][args.target_metric].min()
        best_trial_name = min(dfs, key=max_metric)
    df = dfs[best_trial_name]
    idx = df["val_jindex_good_actions"].idxmax()
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

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
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
    parsed_args = parser.parse_args()

    main(parsed_args)
