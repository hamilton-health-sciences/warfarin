"""Processing steps for specific trials in COMBINE-AF."""

import numpy as np

from warfarin.data.auditing import auditable
from warfarin.data.utils import split_traj


@auditable()
def preprocess_engage_rocket(inr, baseline):
    """
    Preprocessing steps that are specific to the ENGAGE and ROCKET trials.

    In order, this function:

        1. Subsets to ENGAGE and ROCKET data.
        2. Removes daily warfarin doses that are not multiples of 0.5 or have
           missingness code 99.
        3. Computes the weekly dose from the given daily doses.
        4. Subsets to observed INRs and doses.
        5. Sets an interrupt flag when two consecutive zero doses are observed,
           or we observe an INR of 0.
        6. Splits trajectories on the interrupt flag.

    Args:
        inr: Dataframe of all INR data.
        baseline: Dataframe of all baseline data.

    Returns:
        inr: Processed INR data from ENGAGE + ROCKET.
    """
    # Subset to ENGAGE and ROCKET
    subset_ids = baseline[
        (baseline["TRIAL"].isin(["ROCKET_AF", "ENGAGE"]))
    ]["SUBJID"].unique()
    subset_data = inr[inr["SUBJID"].isin(subset_ids)].copy()

    # Remove doses that are not multiples of 0.5 or are 99 (these represent
    # missing values)
    subset_data.loc[(subset_data["WARFARIN_DOSE"] % 0.5) != 0,
                    "WARFARIN_DOSE"] = np.nan
    subset_data.loc[subset_data["WARFARIN_DOSE"] == 99,
                    "WARFARIN_DOSE"] = np.nan

    # Doses in ENGAGE and ROCKET-AF are recorded as the three daily doses prior
    # to an INR measurement. Compute the 3-day rolling mean and shift it forward
    # 1 day to align it with the corresponding INR measurement. Multiply by
    # 7 to convert to weekly dose. We have validated that it is better to treat
    # missing daily doses as missing-by-chance rather than missing-as-zero by
    # comparing INR-dose change heatmaps.
    subset_data["WARFARIN_DOSE"] = subset_data.groupby(
        "SUBJID", as_index=False
    )["WARFARIN_DOSE"].rolling(3, min_periods=1).mean().groupby(
        "SUBJID", as_index=False
    ).shift(1) * 7.

    # Subset to entries where INR is actually observed, which are now aligned
    # with known weekly doses
    subset_data = subset_data[subset_data["INR_TYPE"] == "Y"]

    # Zero INR values represent unknown data errors, so are set to NaN
    subset_data.loc[subset_data["INR_VALUE"] == 0, "INR_VALUE"] = np.nan

    # Split along dose interruptions when two consecutive visits have zero
    # weekly dose, or there is a NaN INR
    near_zero = ((subset_data["WARFARIN_DOSE"].shift(1) == 0) |
                 (subset_data["WARFARIN_DOSE"].shift(-1) == 0))
    subset_data["INTERRUPT"] = (
        (subset_data["WARFARIN_DOSE"] == 0) & near_zero
    ) | subset_data["INR_VALUE"].isnull()

    subset_data = split_traj(subset_data, reason="DOSE_INR_INTERRUPTION")

    subset_data = subset_data.drop(columns=["INTERRUPT"])

    return subset_data


@auditable()
def preprocess_rely(inr, baseline):
    """
    Preprocessing steps that are specific to RE-LY trial data.

    In order, this function:

        1. Converts average daily doses to weekly doses.

    Args:
        inr: Dataframe of all INR data.
        baseline: Dataframe of all baseline data.

    Returns:
        rely_data: Dataframe of processed RE-LY INR data.
    """
    # Subset to RE-LY participants
    subset_ids = baseline[(baseline["TRIAL"] == "RELY")]["SUBJID"].unique()
    rely_data = inr[inr["SUBJID"].isin(subset_ids)].copy()

    # Convert average daily doses to weekly doses
    rely_data["WARFARIN_DOSE"] = rely_data["WARFARIN_DOSE"] * 7

    # Use trial interruptions data to interrupt trajectories
    rely_data["INTERRUPT"] = rely_data["INTERRUPT_FLAG"].fillna(0.).astype(bool)
    rely_data = split_traj(rely_data, reason="TRIAL_INTERRUPTION")

    # Subset to observed INRs
    rely_data = rely_data[rely_data["INR_TYPE"] == "Y"].copy()

    return rely_data


@auditable()
def preprocess_aristotle(inr, baseline):
    """
    Preprocessing steps that are specific to ARISTOTLE trial data.

    In order, this function:

        1. Converts two typoed negative doses to their positive equivalent.
        2. When dose is NaN, we assume there isn't a visit, and backfill the
           previous dose column accordingly.
        3. Define interruptions as consecutive weekly doses of zero and split
           into trajectories along these interruptions.

    Args:
        inr: Dataframe of all INR data.
        baseline: Dataframe of all baseline data.

    Returns:
        inr: Dataframe of INR data from ARISTOTLE.
    """
    # Subset to ARISTOTLE data
    subset_ids = baseline[(baseline["TRIAL"] == "ARISTOTLE")]["SUBJID"].unique()
    aristotle_data = inr[inr["SUBJID"].isin(subset_ids)].copy()

    # For ARISTOTLE patients, there are 2 negative doses that were manually
    # converted to positive doses as they appear to be typos.
    aristotle_data["WARFARIN_DOSE"] = np.abs(aristotle_data["WARFARIN_DOSE"])

    # Backfill warfarin dose until previous visit is reached.
    aristotle_data["WARFARIN_DOSE"] = aristotle_data.groupby(
        "SUBJID"
    )["WARFARIN_DOSE"].fillna(method="bfill")
    aristotle_data = aristotle_data.dropna(
        subset=["INR_VALUE", "WARFARIN_DOSE"]
    )
    aristotle_data = aristotle_data[aristotle_data["INR_TYPE"] == "Y"]

    # Splitting trajectories along dose interruptions
    near_zero = (
        (aristotle_data["WARFARIN_DOSE"].shift(1) == 0) |
        (aristotle_data["WARFARIN_DOSE"].shift(-1) == 0)
    )
    aristotle_data["INTERRUPT"] = ((aristotle_data["WARFARIN_DOSE"] == 0) &
                                   near_zero)

    aristotle_data = split_traj(aristotle_data, reason="DOSE_INR_INTERRUPTION")

    aristotle_data = aristotle_data.drop(columns=["INTERRUPT"])

    return aristotle_data
