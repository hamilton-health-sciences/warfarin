""" Functions that are used across multiple scripts? """

import feather
import numpy as np
import pandas as pd
import time

from warfarin.config import ADV_EVENTS, EVENTS_TO_KEEP, STATE_COLS_TO_FILL



def load_split_data(base_path, suffix):
    """
    Load train, validation, and test data.

    :param base_path: path to where the dataframes are stored 
    :param suffix: suffix of the split data 
    :return: train, validation, and test dataframes 
    """

    train_data = feather.read_dataframe(base_path + f"train_data{suffix}.feather")
    val_data = feather.read_dataframe(base_path + f"val_data{suffix}.feather")
    test_data = feather.read_dataframe(base_path + f"test_data{suffix}.feather")
    return train_data, val_data, test_data


def get_events_data(train_data):
    """
    Detect adverse events in the data.

    This is used to create the events replay buffer.

    :param train_data: dataframe of training data
    :return: dataframe containing the entry leading to the adverse event and the entry corresponding to adverse event
    """

    detect_event = np.append((train_data[ADV_EVENTS].sum(axis=1) > 0).shift(-1)[:-1], False)
    event_occur = (train_data[ADV_EVENTS].sum(axis=1) > 0)
    mask = np.logical_or(detect_event, event_occur)
    events_data = train_data[mask]
    return events_data


def interpolate_daily(input_df, verbose=False):
    
    t0 = time.time()
    print(f"\nDropping  misc columns, if present...")
    for col in ['SUBJID_NEW_2', 'SUBJID_NEW', 'START_TRAJ_CUMU', 'END_TRAJ_CUMU', 'FIRST_DAY', 'LAST_DAY',
                'IS_NULL_CUMU', 'REMOVE', 'CUMU_MEASUR', 'MISSING_ID', 'START_TRAJ', 'END_TRAJ']:
        if col in input_df.columns:
            input_df = input_df.drop(columns=[col])
            print(f"Dropping col: {col}")

    first_days = pd.DataFrame(input_df.groupby('USUBJID_O_NEW')['STUDY_DAY'].first()).reset_index().rename(
        columns={'STUDY_DAY': 'FIRST_DAY'})
    last_days = pd.DataFrame(input_df.groupby('USUBJID_O_NEW')['STUDY_DAY'].last()).reset_index().rename(
        columns={'STUDY_DAY': 'LAST_DAY'})

    dates = first_days.merge(last_days, on='USUBJID_O_NEW')
    df = pd.DataFrame({})

    df['USUBJID_O_NEW'] = dates['USUBJID_O_NEW']
    df['STUDY_DAY'] = [np.arange(s, e + 1, 1) for s, e in
                       zip(dates['FIRST_DAY'], dates['LAST_DAY'])]

    print(f"\nExploding to daily entries...")
    df_exploded = df.explode(column='STUDY_DAY')

    df_exploded_merged = df_exploded.merge(input_df, how='left',
                                           on=['USUBJID_O_NEW', 'STUDY_DAY'])  # ['INR_VALUE'].interpolate()

    df_exploded_merged['INR_VALUE'] = df_exploded_merged['INR_VALUE'].interpolate()

    df_exploded_merged['TIMESTEP'] = df_exploded_merged['STUDY_DAY']

    for ev in EVENTS_TO_KEEP:
        df_exploded_merged[ev] = df_exploded_merged[ev].fillna(0)
        if verbose:
            print(f"\tFinished filling for event {ev}")

    #         dose_cols = [x for x in STATE_COLS_TO_FILL if "WARFARIN_DOSE" in x]
    #         state_cols = [x for x in STATE_COLS_TO_FILL if "WARFARIN_DOSE" not in x]

    state_cols = STATE_COLS_TO_FILL
    state_cols = [x for x in state_cols if not (
            (x in ["INR_1", "INR_2", "INR_3", "INR_4"]) or ("(" in x) or (")" in x) or ("[" in x) or ("]" in x) or (
            "<" in x) or (">" in x))]
    state_cols = [x for x in state_cols if not (("CONTINENT" in x) or ("SMOKE" in x))] + ["CONTINENT", "SMOKE"]
    dose_cols = [x for x in state_cols if "WARFARIN_DOSE" in x]
    state_cols = [x for x in state_cols if x != "WARFARIN_DOSE"]

    for col in dose_cols:
        if verbose:
            print(f"Filling in column: {col}")
        df_exploded_merged[col] = df_exploded_merged.groupby("USUBJID_O_NEW")[col].fillna(method="bfill")

    for col in state_cols:  # [x for x in state_cols if ("CONTINENT" in x) or ("SMOKE" in x)]:
        if verbose:
            print(f"Filling in column: {col}")
        if col in df_exploded_merged.columns:
            df_exploded_merged[col] = df_exploded_merged.groupby("USUBJID_O_NEW")[col].fillna(method="ffill")

    print(df_exploded_merged.shape)

    print(f"Done creating the interpolated dataframe! Took {time.time() - t0}")

    return df_exploded_merged
