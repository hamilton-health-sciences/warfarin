"""Hyperparameter tuning and model training."""

from typing import Optional

from functools import partial

from warnings import warn

import os

import io

import pickle

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
from warfarin.models import SMDBCQ, BehaviorCloner
from warfarin.utils.modeling import (get_dataloaders,
                                     evaluate_policy,
                                     checkpoint_and_log)


def train_run(config: dict,
              checkpoint_dir: str,
              train_data_path: str,
              val_data_path: str,
              feasibility_behavior_policy_path: str,
              no_freeze_feasibility_init: str,
              wis_behavior_policy_path: str,
              init_seed: int,
              smoke_test: bool = False):
    """
    Train a model with a given set of hyperparameters.

    Args:
        config: The hyperparameters of interest.
        checkpoint_dir: The checkpointing directory path created by Ray Tune.
        train_data_path: The path to the training buffer.
        val_data_path: The path to the validation buffer.
        feasibility_behavior_policy_path: The path to the behavior policy (for
                                          generative network init) savefile.
        no_freeze_feasibility_init: If False, and
                                    `feasibility_behavior_policy_path` is given,
                                    the generative network will be frozen.
        wis_behavior_policy_path: The path to the behavior policy (for WIS)
                                  savefile.
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

    # Load the data
    train_data, train_loader, val_data, _ = get_dataloaders(
        train_data_path,
        val_data_path,
        config["batch_size"],
        config["discount"],
        min_train_trajectory_length=global_config.MIN_TRAIN_TRAJECTORY_LENGTH
    )

    # Get the trial directory
    trial_dir = tune.get_trial_dir()
    if not trial_dir:
        trial_dir = "./smoke_test"

    # Store transforms
    state_transforms_path = os.path.join(trial_dir, "transforms.pkl")
    pickle.dump(train_data.state_transforms,
                open(state_transforms_path, "wb"))

    # Load initializer of generative network if given
    if feasibility_behavior_policy_path is None:
        gen_network_init = None
    else:
        gen_network_init = BehaviorCloner.load(feasibility_behavior_policy_path)

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
        num_layers=config["num_layers"],
        generative_network_init=gen_network_init,
        freeze_generative_network=(not no_freeze_feasibility_init)
    )

    # Load the behavior policy for WIS
    wis_behavior_policy = BehaviorCloner.load(wis_behavior_policy_path)

    # Train the model. Epochs refer to approximate batch coverage.
    running_state = {"train": {}, "val": {}}
    writer = tf.summary.create_file_writer(trial_dir)
    for epoch in range(global_config.MAX_TRAINING_EPOCHS):
        # Train over approximately the full buffer (approximately because we
        # randomly sample from the buffer to create each batch).
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
        running_state["train"]["batch_qloss"] = batch_qloss / (batch_idx + 1)

        # Evaluate the policy, checkpoint the model, log the metrics and plots.
        metrics, plots = evaluate_policy(epoch, policy, train_data, val_data,
                                         wis_behavior_policy, running_state)
        checkpoint_and_log(epoch, policy, writer, metrics, plots)


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
             feasibility_behavior_policy_path: Optional[str],
             no_freeze_feasibility_init: bool,
             wis_behavior_policy_path: str,
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
        feasibility_behavior_policy_path: The path to the behavior policy, for
                                          initializing the generative network of
                                          the BCQ model.
        no_freeze_feasibility_init: If False, and the
                                    feasibility_behavior_policy_path is given,
                                    then the generative network will be frozen
                                    after initialization. Otherwise ignored.
        wis_behavior_policy_path: The path to the behavior policy, for WIS estimates
                                  of performance.
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
        "batch_size": tune.grid_search([32, 128]),
        "learning_rate": tune.grid_search([1e-7, 1e-6, 1e-5]),
        "num_layers": tune.grid_search([2, 3]),
        "hidden_dim": 64,
        "bcq_threshold": 0.3,
        "tau": 5e-3
    }

    if smoke_test or tune_smoke_test:
        global_config.MIN_TRAINING_EPOCHS = 1
        global_config.MAX_TRAINING_EPOCHS = 1

    if smoke_test:
        train_conf = tune_config
        for k, v in train_conf.items():
            if hasattr(v, "sample"):
                train_conf[k] = v.sample()
            elif isinstance(v, dict):
                if "grid_search" in v:
                    train_conf[k] = v["grid_search"][0]
        train_run(
            train_conf,
            checkpoint_dir=None,
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            feasibility_behavior_policy_path=feasibility_behavior_policy_path,
            no_freeze_feasibility_init=no_freeze_feasibility_init,
            wis_behavior_policy_path=wis_behavior_policy_path,
            init_seed=init_seed,
            smoke_test=True
        )
        exit()

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
            feasibility_behavior_policy_path=feasibility_behavior_policy_path,
            no_freeze_feasibility_init=no_freeze_feasibility_init,
            behavior_policy_path=behavior_policy_path,
            init_seed=init_seed,
            smoke_test=tune_smoke_test
        ),
        resources_per_trial={
            "cpu": 8,
            "gpu": 1
        },
        config=tune_config,
        num_samples=num_samples,
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
        "--feasibility_behavior_policy",
        type=str,
        required=False,
        default=None,
        help="Path to the behavior policy used to initialize the generative "
             "network of the BCQ model"
    )
    parser.add_argument(
        "--no_freeze_initialized_feasibility",
        default=False,
        action="store_true",
        help="If set, does not freeze the initialized generative network when "
             "given. Ignored if no initial generative network given."
    )
    parser.add_argument(
        "--wis_behavior_policy",
        type=str,
        required=True,
        help="Path to the behavior policy savefile for WIS estimates"
    )
    parser.add_argument(
        "--resume_errored",
        action="store_true",
        default=False
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
        # Initialization of generative network
        feasibility_behavior_policy_path=args.feasibility_behavior_policy,
        no_freeze_feasibility_init=args.no_freeze_initialized_feasibility,
        # WIS behavior cloner
        wis_behavior_policy_path=args.wis_behavior_policy,
        # Model/data parameters
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        # Smoke tests for faster iteration on tuning procedure
        smoke_test=args.smoke_test,
        tune_smoke_test=args.tune_smoke_test
    )


if __name__ == "__main__":
    main()
