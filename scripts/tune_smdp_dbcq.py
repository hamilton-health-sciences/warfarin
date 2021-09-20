"""Hyperparameter tuning and model training."""

from functools import partial

from warnings import warn

import os

import io

import pickle

from plotnine.exceptions import PlotnineError

from uuid import uuid4

import argparse

import pandas as pd

import numpy as np

import torch

from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler

import tensorflow as tf

from warfarin import config as global_config
from warfarin.models import SMDBCQ
from warfarin.utils.modeling import store_plot_tensorboard, get_dataloader
from warfarin.evaluation import evaluate_and_plot_policy


def train_run(config: dict,
              checkpoint_dir: str,
              train_data_path: str,
              val_data_path: str,
              init_seed: int,
              smoke_test: bool = False):
    """
    Train a model with a given set of hyperparameters.

    Args:
        config: The hyperparameters of interest.
        checkpoint_dir: The checkpointing directory path created by Ray Tune.
        train_data_path: The path to the training buffer.
        val_data_path: The path to the validation buffer.
        init_seed: Random seed for reproducibility within a set of
                   hyperparameters.
    """
    # Random seed from trial name to initialize weights differently within each
    # trial but reproducibly. If smoke testing, leave the `init_seed` as-is
    # (fixed to a constant).
    if not smoke_test:
        trial_name_noid = tune.get_trial_name()[8:]
        init_seed = init_seed + int.from_bytes(trial_name_noid.encode(), "little")
        torch.manual_seed(init_seed % (2**32 - 1))

    # Load the train data
    train_data, train_loader = get_dataloader(
        data_path=train_data_path,
        cache_name="train_buffer.pkl",
        batch_size=config["batch_size"],
        discount_factor=config["discount"],
        min_trajectory_length=global_config.MIN_TRAIN_TRAJECTORY_LENGTH
    )
    val_data, val_loader = get_dataloader(
        data_path=val_data_path,
        cache_name="val_buffer.pkl",
        batch_size=config["batch_size"],
        discount_factor=config["discount"],
        state_transforms=train_data.state_transforms
    )

    # Get the trial directory
    trial_dir = tune.get_trial_dir()
    if not trial_dir:
        trial_dir = "./smoke_test"

    # Store transforms
    state_transforms_path = os.path.join(trial_dir, "transforms.pkl")
    pickle.dump(train_data.state_transforms,
                open(state_transforms_path, "wb"))

    # Build the model trainer
    num_actions = len(global_config.ACTION_LABELS)
    policy = SMDBCQ(
        num_actions=num_actions,
        state_dim=train_data.state_dim,
        device="cuda",
        BCQ_threshold=config["bcq_threshold"],
        discount=config["discount"],
        optimizer=config["optimizer"],
        optimizer_parameters={"lr": config["learning_rate"]},
        polyak_target_update=config["polyak_target_update"],
        target_update_frequency=config["target_update_freq"],
        tau=config["tau"],
        hidden_states=config["hidden_dim"],
        num_layers=config["num_layers"]
    )

    start = 0

    # Load state from checkpoint if given
    if checkpoint_dir:
        # Tune state
        with open(os.path.join(checkpoint_dir, "checkpoint")) as f:
            state = json.loads(f.read())
            start = state["step"] + 1

        # Model
        state_dict = torch.load(os.path.join(checkpoint_dir, "model.pt"))
        policy.Q.load_state_dict(state_dict)

    # Train the model. Epochs refer to approximate batch coverage.
    running_state = None
    writer = tf.summary.create_file_writer(trial_dir)
    for epoch in range(global_config.MAX_TRAINING_EPOCHS):
        batch_qloss = 0.
        for batch_idx, batch in enumerate(train_loader):
            qloss = policy.train(batch)
            batch_qloss += qloss.item()
            # Only look at one batch if smoke testing.
            if smoke_test:
                break
            # Otherwise, compute whether the replay buffer has been covered
            # approximately once and treat that as an epoch.
            if ((batch_idx + 1) * config["batch_size"]) > len(train_data):
                break
        batch_qloss = batch_qloss / (batch_idx + 1)

        # Evaluate the policy
        plot_epoch = (epoch % global_config.PLOT_EVERY == 0)
        train_metrics, train_plots, running_state = evaluate_and_plot_policy(
            policy,
            train_data,
            running_state,
            plot=plot_epoch
        )
        val_metrics, val_plots, running_state = evaluate_and_plot_policy(
            policy,
            val_data,
            running_state,
            plot=plot_epoch
        )
        train_metrics["qloss"] = batch_qloss
        metrics = {**{f"train/{k}": v for k, v in train_metrics.items()},
                   **{f"val/{k}": v for k, v in val_metrics.items()}}
        plots = {**{f"train/{k}": v for k, v in train_plots.items()},
                 **{f"val/{k}": v for k, v in val_plots.items()}}

        with tune.checkpoint_dir(step=epoch) as ckpt_dir_write:
            if ckpt_dir_write:
                # Checkpoint the model
                ckpt_fn = os.path.join(ckpt_dir_write, "model.pt")
                policy.save(ckpt_fn)

                # Store plots for Tensorboard
                for plot_name, plot in plots.items():
                    try:
                        store_plot_tensorboard(plot_name, plot, epoch, writer)
                    except (ValueError, PlotnineError) as exc:
                        warn(str(exc))

        tune.report(**metrics)


def trial_namer(trial):
    """
    Name a trial.

    Args:
        trial: The Ray Tune trial to name.

    Returns:
        name: The name of the trial, containing its id and relevant
              hyperparameters.
    """
    trial_id = trial.trial_id
    discount = trial.config["discount"]
    bs = trial.config["batch_size"]
    lr = trial.config["learning_rate"]
    tau = trial.config["tau"]
    num_layers = trial.config["num_layers"]
    hidden_dim = trial.config["hidden_dim"]
    bcq_t = trial.config["bcq_threshold"]
    name = (f"{trial_id}_discount={discount}_bs={bs}_lr={lr:.2e}_tau={tau:.2e}_"
            f"num_layers={num_layers}_hidden_dim={hidden_dim}_"
            f"threshold={bcq_t:.1f}")

    return name


def tune_run(num_samples: int,
             tune_seed: int,
             init_seed: int,
             output_dir: str,
             resume_errored: bool,
             train_data_path: str,
             val_data_path: str,
             target_metric: str,
             mode: str,
             smoke_test: bool,
             tune_smoke_test: bool):
    """
    Run the hyperparameter tuning procedure.

    Args:
        num_samples: The number of hyperparameter combinations to try.
        tune_seed: The seed for the hyperparameter selection process, used when
                   any parameter employs random search.
        init_seed: The seed to use within a given hyperparameter setting.
        output_dir: The output directory for the `ray` logs.
        resume_errored: Whether to resume errored trials or start from scratch.
        train_data_path: The path to the training data.
        val_data_path: The path to the validation data.
        target_metric: The metric to optimize over the space of possible
                       hyperparameters.
        mode: Whether to "max" or "min" the target metric.
        smoke_test: Whether to conduct a "smoke test" where hyperparameters are
                    not tuned but the training loop is run once to ensure code
                    validity.
        tune_smoke_test: Whether to conduct a "smoke test" where hyperparameters
                         are tuned with one hyperparameter sample and the
                         training loop run once to ensure code validity.
    """
    tune_config = {
        # Fixed hyperparams
        "optimizer": "Adam",
        "discount": 0.99,
        # Fixed no-ops
        "polyak_target_update": True,
        "target_update_freq": 100,
        # Searchable hyperparams
        "batch_size": tune.choice([32, 64, 128, 256]),
        "learning_rate": tune.loguniform(1e-7, 1e-2),
        "tau": tune.loguniform(5e-4, 5e-2),
        "num_layers": tune.choice([2, 3]),
        "hidden_dim": tune.choice([16, 32, 64, 128, 256]),
        "bcq_threshold": tune.choice([0.2, 0.3])
    }

    if smoke_test or tune_smoke_test:
        global_config.MIN_TRAINING_EPOCHS = 1
        global_config.MAX_TRAINING_EPOCHS = 1

    if smoke_test:
        train_conf = tune_config
        for k, v in train_conf.items():
            if hasattr(v, "sample"):
                train_conf[k] = v.sample()
        train_run(train_conf,
                  checkpoint_dir=None,
                  train_data_path=train_data_path,
                  val_data_path=val_data_path,
                  init_seed=init_seed,
                  smoke_test=True)
        exit()

    # Specify algorithm for selecting the next set of hyperparameters to try
    # intelligently (TPE algo)
    searcher = HyperOptSearch(
        metric=target_metric,
        mode=mode,
        random_state_seed=tune_seed
    )

    # Aggressively terminate underperforming models after a minimum number of
    # iterations
    scheduler = AsyncHyperBandScheduler(
        metric=target_metric,
        mode=mode,
        max_t=global_config.MAX_TRAINING_EPOCHS,
        grace_period=global_config.MIN_TRAINING_EPOCHS
    )

    if resume_errored:
        resume = "ERRORED_ONLY"
    else:
        resume = None

    # How progress will be reported to the CLI
    par_cols = ["discount", "batch_size", "learning_rate", "tau", "num_layers",
                "hidden_dim"]
    reporter = CLIReporter(
        parameter_columns=par_cols,
        metric_columns=[target_metric]
    )

    # Run tuning
    tune.run(
        partial(
            train_run,
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            init_seed=init_seed,
            smoke_test=tune_smoke_test
        ),
        resources_per_trial={
            "cpu": 8,
            "gpu": 1
        },
        config=tune_config,
        num_samples=num_samples,
        search_alg=searcher,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=output_dir,
        name="dbcq",
        trial_name_creator=trial_namer,
        resume=resume
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to training data or buffer"
    )
    parser.add_argument(
        "--val_data",
        type=str,
        required=True,
        help="Path to validation data or buffer"
    )
    parser.add_argument(
        "--target_metric",
        type=str,
        required=True,
        help="Which metric to perform model selection on"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["min", "max"],
        help="Whether to minimize or maximize the target metric"
    )
    parser.add_argument(
        "--resume_errored",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="The number of hyperparameter combinations to sample"
    )
    parser.add_argument(
        "--tune_seed",
        type=int,
        default=0,
        help="The seed for the hyperparameter optimizer"
    )
    parser.add_argument(
        "--init_seed",
        type=int,
        default=1,
        help=("The seed for reproducibility within a given set of "
              "hyperparameters")
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ray_logs",
        help="The output directory for the hyperparameter optimizer logs"
    )
    parser.add_argument(
        "--smoke_test",
        default=False,
        action="store_true",
        help="Perform a smoke test of the `train` call and exit"
    )
    parser.add_argument(
        "--tune_smoke_test",
        default=False,
        action="store_true",
        help="Perform a smoke test of the tuning procedure and exit"
    )
    args = parser.parse_args()

    if args.tune_smoke_test:
        num_samples = 1
    else:
        num_samples = global_config.NUM_HYPERPARAMETER_SAMPLES

    tune_run(
        # Hyperparameter optimizer parameters
        num_samples=num_samples,
        tune_seed=args.tune_seed,
        init_seed=args.init_seed,
        output_dir=args.output_dir,
        resume_errored=args.resume_errored,
        target_metric=args.target_metric,
        mode=args.mode,
        # Model/data parameters
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        # Smoke tests for faster iteration on tuning procedure
        smoke_test=args.smoke_test,
        tune_smoke_test=args.tune_smoke_test
    )


if __name__ == "__main__":
    main()
