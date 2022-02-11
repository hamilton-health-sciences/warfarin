"""Train the behavioural cloning network for WIS-based evaluation."""

from functools import partial

import os

import pickle

import argparse

from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler

import tensorflow as tf

from warfarin import config as global_config
from warfarin.models import BehaviorCloner
from warfarin.utils.modeling import get_dataloaders, checkpoint_and_log
from warfarin.utils.ordinal import compute_cutpoints
from warfarin.evaluation import evaluate_behavioral_cloning


def train_run(config, train_data_path, val_data_path, smoke_test=False):
    # Generate data buffers
    train_data, train_loader, val_data, val_loader = get_dataloaders(
        train_data_path,
        val_data_path,
        config["batch_size"],
        min_train_trajectory_length=global_config.MIN_TRAIN_TRAJECTORY_LENGTH,
        weight_option_frequency_train=config["weight_option_frequency"],
        use_random_train=True,
        **global_config.REPLAY_BUFFER_PARAMS
    )

    trial_dir = tune.get_trial_dir()
    if not trial_dir:
        trial_dir = "./smoke_test"

    # Store transforms
    state_transforms_path = os.path.join(trial_dir, "transforms.pkl")
    pickle.dump(train_data.state_transforms,
                open(state_transforms_path, "wb"))

    # Compute frequency-based cutpoints
    cutpoints = compute_cutpoints(train_data.option)

    # Build the model trainer
    num_actions = len(global_config.ACTION_LABELS)
    model = BehaviorCloner(
        state_dim=train_data.state_dim,
        num_actions=num_actions,
        num_layers=config["num_layers"],
        hidden_dim=config["hidden_dim"],
        lr=config["learning_rate"],
        likelihood=config["likelihood"],
        cutpoints=cutpoints,
        device="cuda"
    )

    # Train the model
    writer = tf.summary.create_file_writer(trial_dir)
    for epoch in range(global_config.BC_MAX_TRAINING_EPOCHS):
        epoch_loss = 0.
        for batch_idx, batch in enumerate(train_loader):
            loss = model.train(batch)
            epoch_loss += loss
            if smoke_test:
                break
        epoch_loss /= (batch_idx + 1)

        # Get metrics of interest
        train_metrics, train_plots = evaluate_behavioral_cloning(model,
                                                                 train_data)
        val_metrics, val_plots = evaluate_behavioral_cloning(model, val_data)
        metrics = {**{f"train/{k}": v for k, v in train_metrics.items()},
                   **{f"val/{k}": v for k, v in val_metrics.items()}}
        plots = {**{f"train/{k}": v for k, v in train_plots.items()},
                 **{f"val/{k}": v for k, v in val_plots.items()}}

        # Checkpoint and log metrics using Ray Tune
        checkpoint_and_log(epoch, model, writer, metrics, plots)


def trial_namer(trial):
    trial_id = trial.trial_id
    lr = trial.config["learning_rate"]
    bs = trial.config["batch_size"]
    num_layers = trial.config["num_layers"]
    hidden_dim = trial.config["hidden_dim"]
    name = (f"{trial_id}_bs={bs}_lr={lr:.2e}_num_layers={num_layers}_"
            f"hidden_dim={hidden_dim}")

    return name


def tune_run(tune_seed: int,
             output_dir: str,
             train_data_path: str,
             val_data_path: str,
             target_metric: str,
             mode: str,
             smoke_test: bool,
             tune_smoke_test: bool,
             experiment_name: str = "bc"):
    tune_config = global_config.BC_GRID_SEARCH
    if smoke_test or tune_smoke_test:
        global_config.BC_MIN_TRAINING_EPOCHS = 1
        global_config.BC_MAX_TRAINING_EPOCHS = 1

    if smoke_test:
        train_conf = tune_config
        for k, v in train_conf.items():
            if hasattr(v, "sample"):
                train_conf[k] = v.sample()
            elif isinstance(v, dict):
                if "grid_search" in v:
                    train_conf[k] = v["grid_search"][0]
        train_run(train_conf,
                  train_data_path=train_data_path,
                  val_data_path=val_data_path,
                  init_seed=init_seed,
                  smoke_test=True)
        exit()

    # How progress will be reported to the CLI
    par_cols = ["batch_size", "learning_rate", "num_layers", "hidden_dim"]
    reporter = CLIReporter(
        parameter_columns=par_cols,
        metric_columns=[target_metric]
    )

    # Run hyperparameter tuning
    tune.run(
        partial(
            train_run,
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            smoke_test=tune_smoke_test
        ),
        resources_per_trial={"cpu": 8, "gpu": 1},
        config=tune_config,
        progress_reporter=reporter,
        local_dir=output_dir,
        name=experiment_name,
        # TODO
        trial_name_creator=trial_namer
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data",
        type=str,
        default=os.path.join(os.getcwd(), "data/clean_data/train_data.feather")
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default=os.path.join(os.getcwd(), "data/clean_data/val_data.feather")
    )
    parser.add_argument("--output_dir", type=str, default="./ray_logs")
    parser.add_argument("--experiment_name", type=str, default="bc")
    parser.add_argument("--smoke_test", action="store_true", default=False)
    parser.add_argument("--tune_smoke_test", action="store_true", default=False)
    args = parser.parse_args()

    tune_run(
        # Hyperparameter optimizer parameters
        tune_seed=global_config.BC_TUNE_SEED,
        output_dir=args.output_dir,
        target_metric=global_config.BC_TARGET_METRIC,
        mode=global_config.BC_TARGET_MODE,
        # Model/data parameters
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        # Smoke tests for faster iteration on tuning procedure
        smoke_test=args.smoke_test,
        tune_smoke_test=args.tune_smoke_test,
        # Custom naming for clarity between multiple runs
        experiment_name=args.experiment_name
    )


if __name__ == "__main__":
    main()
