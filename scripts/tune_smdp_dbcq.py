"""Hyperparameter tuning and model training."""

from functools import partial

import os

import io

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
from warfarin.data import WarfarinReplayBuffer
from warfarin.models.smdp_dBCQ import discrete_BCQ
from warfarin.evaluation import evaluate_and_plot_policy


def store_plot_tensorboard(plot_name, plot, step, writer):
    """
    Store a plot so that it's visible in Tensorboard.

    Args:
        plot_name: The name of the plot.
        plot: The plotnine plot object.
        step: The corresponding step (epoch).
        writer: The TF writer to log the image to.

    Raises:
        PlotnineError if the plot could not be drawn.
    """
    # Put the figure in a TF image object
    buf = io.BytesIO()
    fig = plot.draw()
    fig.savefig(buf, format="png")
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    # Write the image
    with writer.as_default():
        tf.summary.image(plot_name, image, step=step)
        writer.flush()


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
    # Load the data
    train_df = pd.read_feather(train_data_path)
    train_data = WarfarinReplayBuffer(
        df=train_df,
        discount_factor=config["discount"],
        batch_size=config["batch_size"],
        device="cuda"
    )
    val_df = pd.read_feather(val_data_path)
    val_data = WarfarinReplayBuffer(
        df=val_df,
        discount_factor=config["discount"],
        batch_size=config["batch_size"],
        device="cuda"
    )

    # Data dimensionality from buffers
    num_actions = 7  # TODO

    # Build the model trainer
    policy = discrete_BCQ(
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

    # Train the model
    running_state = None
    if smoke_test:
        trial_dir = "./smoke_test"
    else:
        trial_dir = tune.get_trial_dir()
    writer = tf.summary.create_file_writer(trial_dir)
    for epoch in range(start, global_config.MAX_TRAINING_EPOCHS):
        # Number of batches for approximate coverage of the full buffer
        if smoke_test:
            num_batches = 1
        else:
            num_batches = int(
                np.ceil(train_data.size / config["batch_size"])
            )

        # Train on the full buffer approximately once
        for _ in range(num_batches):
            qloss = policy.train(train_data)

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
        metrics = {**{f"train_{k}": v for k, v in train_metrics.items()},
                   **{f"val_{k}": v for k, v in val_metrics.items()}}
        plots = {**{f"train_{k}": v for k, v in train_plots.items()},
                 **{f"val_{k}": v for k, v in val_plots.items()}}

        with tune.checkpoint_dir(step=epoch) as ckpt_dir_write:
            # Checkpoint the model
            # TODO do we also need to store the target Q network of the policy?
            ckpt_fn = os.path.join(ckpt_dir_write, "model.pt")
            torch.save(policy.Q.state_dict(), ckpt_fn)

            # Store plots for Tensorboard
            if plot_epoch:
                for plot_name, plot in plots.items():
                    try:
                        store_plot_tensorboard(plot_name,
                                               plot,
                                               epoch,
                                               writer)
                    except PlotnineError:
                        pass

        # TODO: implement WIS ?
        tune.report(**metrics)


def trial_namer(trial):
    # TODO implement
    return str(uuid4())


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
    # TODO: make number of layers searchable over a wider space by generalizing
    # the model class
    tune_config = {
        # Fixed hyperparams
        "optimizer": "Adam",
        "polyak_target_update": True,
        "target_update_freq": 100,
        # Searchable hyperparams
        "discount": tune.loguniform(0.95, 0.999),
        "batch_size": tune.choice([2, 4, 8, 32, 64, 128, 256]),
        "learning_rate": tune.loguniform(1e-7, 1e-4),
        "tau": tune.loguniform(1e-5, 1e-2),
        "num_layers": tune.choice([2, 3]),
        "hidden_dim": tune.choice([16, 32, 64, 128, 256]),
        "bcq_threshold": tune.choice([0.2, 0.3])
    }

    if smoke_test:
        global_config.MAX_TRAINING_EPOCHS = 1
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
    par_cols = ["discount", "batch_size", "learning_rate", "num_layers",
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
            init_seed=init_seed
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
        help="Path to training buffer"
    )
    parser.add_argument(
        "--val_data",
        type=str,
        required=True,
        help="Path to validation buffer"
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

    if args.num_samples is None:
        num_samples = global_config.NUM_HYPERPARAMETER_SAMPLES
    else:
        num_samples = args.num_samples

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
