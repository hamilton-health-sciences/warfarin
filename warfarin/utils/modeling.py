"""Utilities for tuning and evaluation."""

from typing import Optional

from warnings import warn

import os

import io

import pickle

import tensorflow as tf

import pandas as pd

from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler

from ray import tune

from plotnine.exceptions import PlotnineError

from warfarin import config
from warfarin.data import WarfarinReplayBuffer
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
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    # Write the image
    with writer.as_default():
        tf.summary.image(plot_name, image, step=step)
        writer.flush()


def get_dataloader(data_path: str,
                   cache_name: str,
                   batch_size: int,
                   option_means: Optional[dict] = None,
                   use_random: bool = True,
                   **replay_buffer_params):
    """
    Load replay buffer, cache it if necessary, and create a dataloader.

    Args:
        data_path: Path to the data, either the cached `.pkl` file or the
                   `.feather` file from the data processing script.
        cache_name: The name of the file in the cache directory to store the
                    buffer.
        batch_size: Batch size to sample.
        option_means: The mean dose change for each option.
        replay_buffer_params: Additional arguments to the replay buffer.

    Returns:
        data: The replay buffer.
        dl: The dataloader.
    """
    if os.path.splitext(data_path)[-1] == ".pkl":
        data = pickle.load(open(train_data_path, "rb"))
    else:
        df = pd.read_feather(data_path)
        data = WarfarinReplayBuffer(
            df=df,
            device="cuda",
            option_means=option_means,
            **replay_buffer_params
        )
        os.makedirs(config.CACHE_PATH, exist_ok=True)
        buffer_path = os.path.join(config.CACHE_PATH, cache_name)
        pickle.dump(data, open(buffer_path, "wb"))

    if use_random:
        sampler = BatchSampler(
            WeightedRandomSampler(weights=data.sample_prob,
                                  num_samples=len(data),
                                  replacement=True),
            batch_size=batch_size,
            drop_last=False
        )
        dl = DataLoader(data, batch_sampler=sampler)
    else:
        dl = DataLoader(data, batch_size=batch_size)

    return data, dl


def get_dataloaders(train_data_path, val_data_path, batch_size, discount_factor,
                    min_train_trajectory_length, use_random_train=True):
    train_data, train_loader = get_dataloader(
        data_path=train_data_path,
        cache_name="train_buffer.pkl",
        batch_size=batch_size,
        discount_factor=discount_factor,
        min_trajectory_length=min_train_trajectory_length,
        use_random=use_random_train
    )
    val_data, val_loader = get_dataloader(
        data_path=val_data_path,
        cache_name="val_buffer.pkl",
        batch_size=batch_size,
        discount_factor=discount_factor,
        state_transforms=train_data.state_transforms,
        use_random=False
    )

    return train_data, train_loader, val_data, val_loader


def evaluate_policy(epoch, policy, train_data, val_data, behavior_policy,
                    running_state):
    plot_epoch = (epoch % config.PLOT_EVERY == 0)
    train_metrics, train_plots, _, running_state = evaluate_and_plot_policy(
        policy,
        train_data,
        behavior_policy,
        running_state,
        plot=plot_epoch
    )
    val_metrics, val_plots, _, running_state = evaluate_and_plot_policy(
        policy,
        val_data,
        behavior_policy,
        running_state,
        plot=plot_epoch
    )
    if "qloss" in running_state:
        train_metrics["qloss"] = running_state["qloss"]
    metrics = {**{f"train/{k}": v for k, v in train_metrics.items()},
               **{f"val/{k}": v for k, v in val_metrics.items()}}
    plots = {**{f"train/{k}": v for k, v in train_plots.items()},
             **{f"val/{k}": v for k, v in val_plots.items()}}

    return metrics, plots


def checkpoint_and_log(epoch, model, writer, metrics, plots):
    with tune.checkpoint_dir(step=epoch) as ckpt_dir_write:
        if ckpt_dir_write:
            # Checkpoint the model
            ckpt_fn = os.path.join(ckpt_dir_write, "model.pt")
            model.save(ckpt_fn)

            # Store plots for Tensorboard
            for plot_name, plot in plots.items():
                try:
                    store_plot_tensorboard(plot_name, plot, epoch, writer)
                except (ValueError, PlotnineError) as exc:
                    warn(str(exc))

    tune.report(**metrics)
