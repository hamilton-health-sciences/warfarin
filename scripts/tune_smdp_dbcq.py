from functools import partial

import os

from uuid import uuid4

import argparse

import numpy as np

import torch

from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler

from warfarin import config as global_config
from warfarin.utils.smdp_buffer import SMDPReplayBuffer
from warfarin.models.smdp_dBCQ import discrete_BCQ
from warfarin.models.policy_eval import eval_policy


def train_run(config: dict,
              checkpoint_dir: str,
              train_buffer_path: str,
              events_buffer_path: str,
              val_buffer_path: str,
              init_seed: int):
    """
    Train a model given a set of hyperparameters.

    Args:
        config: The hyperparameters of interest.
        checkpoint_dir: The checkpointing directory path created by Ray Tune.
        train_buffer_path: The path to the training buffer.
        events_buffer_path: The path to the events buffer.
        val_buffer_path: The path to the validation buffer.
        init_seed: Random seed for reproducibility within a set of
                   hyperparameters.
    """
    # Load the data
    train_buffer = SMDPReplayBuffer.from_filename(
        data_path=train_buffer_path,
        batch_size=config["batch_size"],
        buffer_size=1e7,
        device="cuda"
    )
    train_buffer.max_size = len(train_buffer.data)
    events_buffer = SMDPReplayBuffer.from_filename(
        data_path=events_buffer_path,
        batch_size=config["batch_size"],
        buffer_size=1e6,
        device="cuda"
    )
    events_buffer.max_size = len(events_buffer.data)
    val_buffer = SMDPReplayBuffer.from_filename(
        data_path=val_buffer_path,
        batch_size=config["batch_size"],
        buffer_size=1e6,
        device="cuda"
    )
    val_buffer.max_size = len(val_buffer.data)

    # Data dimensionality from buffers
    num_actions = 7  # TODO
    state_dim = 56  # ...

    # Build the model trainer
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

    # Train
    for epoch in range(start, global_config.MAX_TRAINING_EPOCHS):
        # Number of batches for approximate coverage of the full buffer
        num_batches = int(
            np.ceil(len(train_buffer.data) / config["batch_size"])
        )

        # Train on the full buffer approximately once
        for _ in range(num_batches):
            qloss = policy.train(train_buffer, events_buffer)

        # Checkpoint the model
        with tune.checkpoint_dir(step=epoch) as ckpt_dir_write:
            ckpt_fn = os.path.join(ckpt_dir_write, "model.pt")
            # TODO do we also need to store the target Q network of the policy?
            torch.save(policy.Q.state_dict(), ckpt_fn)

        # Evaluate the policy
        train_results = eval_policy(policy, train_buffer)
        val_results = eval_policy(policy, val_buffer)
        # TODO: implement WIS ?
        tune.report(
            **{f"train_{k}": v for k, v in train_results.items()},
            **{f"val_{k}": v for k, v in val_results.items()}
        )


def trial_namer(trial):
    # TODO implement
    return str(uuid4())


def tune_run(num_samples: int,
             tune_seed: int,
             init_seed: int,
             output_dir: str,
             resume_errored: bool,
             train_buffer_path: str,
             events_buffer_path: str,
             val_buffer_path: str,
             target_metric: str,
             mode: str):
    # TODO: make number of layers searchable over a wider space by generalizing
    # the model class
    tune_config = {
        # Data
        "train_buffer_path": train_buffer_path,
        "val_buffer_path": val_buffer_path,
        "events_buffer_path": events_buffer_path,
        # Fixed hyperparams
        "optimizer": "Adam",
        "polyak_target_update": True,
        "target_update_freq": 100,
        "tau": 5e-3,
        # Searchable hyperparams
        "discount": tune.loguniform(0.95, 0.999),
        "batch_size": tune.choice([2, 4, 8, 32, 64, 128, 256]),
        "learning_rate": tune.loguniform(1e-7, 1e-2),
        "num_layers": tune.choice([2, 3]),
        "hidden_dim": tune.choice([4, 8, 16, 32, 64, 128]),
        "bcq_threshold": tune.choice([0.2, 0.3])
    }

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
            train_buffer_path=train_buffer_path,
            events_buffer_path=events_buffer_path,
            val_buffer_path=val_buffer_path,
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
        "--train_buffer",
        type=str,
        required=True,
        help="Path to training buffer"
    )
    parser.add_argument(
        "--events_buffer",
        type=str,
        required=False,
        help="Path to events buffer, if using"
    )
    parser.add_argument(
        "--val_buffer",
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
        train_buffer_path=args.train_buffer,
        events_buffer_path=args.events_buffer,
        val_buffer_path=args.val_buffer
    )


if __name__ == "__main__":
    main()
