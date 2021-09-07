import pandas as pd

from warfarin.data import WarfarinReplayBuffer


def main(args):
    df = pd.read_feather(args.input_path)

    replay_buffer = WarfarinReplayBuffer.from_data(
        df,
        args.relative_event_sample_probability
    )



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="The path to the processed input dataframe as a Feather file"
    )
    parser.add_argument(
        "--relative_event_sample_probability",
        type=float,
        default=1,
        help=("If upsampling events, set to > 1. Trajectories ending in events "
              "will then be this factor more likely to be sampled during "
              "training.")
    )
    parsed_args = parser.parse_args()

    main(parsed_args)
