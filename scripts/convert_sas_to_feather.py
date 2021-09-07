"""Convert COMBINE-AF SAS input files to Feather for faster loading."""

import os

import pandas as pd

from warfarin.data.utils import decode


def main(args):
    df = pd.read_sas(args.input_filename, format="sas7bdat")
    df = decode(df)
    df.to_feather(args.output_filename)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_filename",
        type=str,
        required=True,
        help="Path to input SAS (.sas7bdat) file"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        required=True,
        help="Path to output Feather (.feather) file"
    )
    parsed_args = parser.parse_args()

    main(parsed_args)
