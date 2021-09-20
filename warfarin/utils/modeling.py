"""Utilities for tuning and evaluation."""

import os

import io

import pickle

import tensorflow as tf

import pandas as pd

from torch.utils.data import DataLoader

from warfarin import config
from warfarin.data import WarfarinReplayBuffer


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
                   **replay_buffer_params):
    """
    Load replay buffer, cache it if necessary, and create a dataloader.

    Args:
        data_path: Path to the data, either the cached `.pkl` file or the
                   `.feather` file from the data processing script.
        cache_name: The name of the file in the cache directory to store the
                    buffer.
        batch_size: Batch size to sample.
        replay_buffer_params: Additional arguments to the replay buffer.

    Returns:
        dl: The dataloader.
    """
    if os.path.splitext(data_path)[-1] == ".pkl":
        data = pickle.load(open(train_data_path, "rb"))
    else:
        df = pd.read_feather(data_path)
        data = WarfarinReplayBuffer(
            df=df,
            device="cuda",
            **replay_buffer_params
        )
        os.makedirs(config.CACHE_PATH, exist_ok=True)
        buffer_path = os.path.join(config.CACHE_PATH, cache_name)
        pickle.dump(data, open(buffer_path, "wb"))
    dl = DataLoader(data, batch_size=batch_size)

    return data, dl
