"""Preprocess the COMBINE-AF files into data buffers for model development."""

from utils.combine_preprocessing import * #load_data # ...
import argparse
from warnings import warn


def main(args):
    
    if not os.path.exists(f"{args.data_folder}{args.raw_data_folder}"):
        raise Exception(f"The specified data folder does not exist: {args.data_folder}{args.raw_data_folder}")

    print(f"\n----------------------------------------------")
    print(f"Converting the SAS files to feather files for faster readability moving forward.")
    print(f"----------------------------------------------")
    
    inr, events, baseline = load_raw_data_sas(args.data_folder + args.raw_data_folder)
    
    print(f"Storing raw data as feather files...")
    base_path = args.data_folder + args.raw_data_folder
    feather.write_dataframe(inr, base_path + f"inr.feather")
    feather.write_dataframe(baseline, base_path + f"baseline.feather")
    feather.write_dataframe(events, base_path + f"events.feather")
    
    print(f"\nDONE! We can now run the data preprocessing script, which uses the feather files.")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default="./data/", type=str, required=True, help="Path to folder containing data. The raw data should be within this folder. Cleaned data will be stored here as well.")
    parser.add_argument("--raw_data_folder", default="raw_data/", type=str, help="Subfolder within the data folder containing the raw data.")
    args = parser.parse_args()

    main(args)

    