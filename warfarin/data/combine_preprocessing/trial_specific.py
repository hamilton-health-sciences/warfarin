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

    # TODO validate these assumptions?
    # Doses in ENGAGE and ROCKET-AF are recorded as the three daily doses prior
    # to an INR measurement. Compute the 3-day rolling mean and shift it forward
    # 1 day to align it with the corresponding INR measurement. Multiply by
    # 7 to convert to weekly dose.
    subset_data["WARFARIN_DOSE"] = subset_data.groupby(
        "SUBJID", as_index=False
    )["WARFARIN_DOSE"].rolling(3, min_periods=1).mean().groupby(
        "SUBJID", as_index=False
    ).shift(1) * 7.

    # TODO move reporting to separate auditing tool
    nan_entries = sum(
        subset_data[
            subset_data["INR_TYPE"] == "Y"
        ]["WARFARIN_DOSE"].isnull()
    )
    total_entries = subset_data[subset_data["INR_TYPE"] == "Y"].shape[0]
    print(
        f"\tENGAGE, ROCKET_AF: {nan_entries} of {total_entries} "
        f"({nan_entries / total_entries:,.2%}) entries are NaN"
    )

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

    subset_data = split_traj(subset_data)

    subset_data = subset_data.drop(columns=["INTERRUPT"])

    # TODO move reporting to separate auditing tool
    # print(
    #     f"\t{subset_data['SUBJID'].nunique():,.0f} patients, "
    #     f"{subset_data['SUBJID_NEW'].nunique():,.0f} trajectories after "
    #     "splitting along dose interruptions"
    # )

    return subset_data


@auditable()
def preprocess_rely(inr, baseline):
    """
    Preprocessing steps that are specific to RE-LY trial data.

    In order, this function:

        1. Convert average daily doses to weekly doses.
        2. Convert observed INR values of 0 to NaN, as it is unclear what
           happened at those steps.
        3. Split trajectories at points where INR is null.

    Args:
        inr: Dataframe of all INR data.
        baseline: Dataframe of all baseline data.
    Returns:
        rely_data: Dataframe of processed RE-LY INR data.
    """
    # Subset to RE-LY participants
    subset_ids = baseline[(baseline["TRIAL"] == "RELY")]["SUBJID"].unique()
    rely_data = inr[inr["SUBJID"].isin(subset_ids)].copy()

    # Convert daily doses to weekly doses
    rely_data["WARFARIN_DOSE"] = rely_data["WARFARIN_DOSE"] * 7

    # Split along dose interruptions. There are 4 entries in the RE-LY dataset
    # where the INR value is 0. Since it is unclear what the dynamics are around
    # this point, we will remove these transitions and separate trajectories at
    # these points.
    rely_data.loc[rely_data["INR_VALUE"] == 0, "INR_VALUE"] = np.nan

    # Split trajectories in entries where INR is NaN
    rely_data["IS_NULL"] = rely_data["INR_VALUE"].isnull()
    rely_data["IS_NULL_CUMU"] = rely_data.groupby("SUBJID")["IS_NULL"].cumsum()
    rely_data["INTERRUPT"] = np.minimum(1, rely_data["IS_NULL_CUMU"].diff() > 0)
    rely_data = rely_data.drop(["IS_NULL", "IS_NULL_CUMU"], axis=1)

    rely_data = split_traj(rely_data)

    rely_data = rely_data.drop(columns=["INTERRUPT"])

    # TODO extract out to auditing script
    # print(
    #     f"\t{rely_data['SUBJID'].nunique():,.0f} patients, "
    #     f"{rely_data['SUBJID_NEW'].nunique():,.0f} trajectories after "
    #     "splitting along dose interruptions"
    # )

    return rely_data


@auditable()
def preprocess_aristotle(inr, baseline):
    """
    Preprocessing steps that are specific to ARISTOTLE trial data.

    In order, this function:

        1. Converts two typoed negative doses to their positive equivalent.
        2. Handles cases where the INR is observed on day `t`, but the visit
           where the previous dose is recorded doesn't occur until day `t + 1`.
        3. When dose is NaN, we assume there isn't a visit, and backfill the
           previous dose column accordingly. (TODO correct interpretation?)
        4. Define interruptions as consecutive weekly doses of zero and split
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

    # TODO move to auditing script
    # print(
    #     "\tThere are "
    #     f"{len(aristotle_data[aristotle_data['WARFARIN_DOSE'] < 0])} entries "
    #     "with negative entries. Taking absolute value..."
    # )

    # For ARISTOTLE patients, there are 2 negative doses that were manually
    # converted to positive doses as they appear to be typos.
    aristotle_data["WARFARIN_DOSE"] = np.abs(aristotle_data["WARFARIN_DOSE"])

    # TODO is this appropriate for evaluation? i.e. did the clinician actually
    # make a decision?
    # Handling lag between INR and Warfarin dose. Specifically, in cases where
    # the INR is observed but there is no corresponding dose, we pull in the
    # previous dose from the next observation.
    aristotle_data["PREV_DOSE"] = aristotle_data.groupby(
        "SUBJID"
    )["WARFARIN_DOSE"].shift(-1)
    mask = (~aristotle_data["PREV_DOSE"].isnull() &
            aristotle_data["WARFARIN_DOSE"].isnull() &
            (aristotle_data["INR_TYPE"] == "Y"))
    aristotle_data.loc[mask,
                       "WARFARIN_DOSE"] = aristotle_data.loc[mask, "PREV_DOSE"]
    aristotle_data = aristotle_data.drop(["PREV_DOSE"], axis=1)
    # TODO move to auditing script
    # num_lags = sum(mask)
    # print(f"\tIdentified {num_lags} potential lags")

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

    aristotle_data = split_traj(aristotle_data)

    aristotle_data = aristotle_data.drop(columns=["INTERRUPT"])

    # aristotle_data[aristotle_data["INTERRUPT"]].groupby(
    #     "SUBJID"
    # ).size().hist()
    # plt.xlabel("Number of INR measurements where dose was interrupted")
    # plt.ylabel("Count")

    # TODO move to auditing script
    # print(
    #     f"\t{aristotle_data['SUBJID'].nunique():,.0f} patients, "
    #     f"{aristotle_data['SUBJID_NEW'].nunique():,.0f} trajectories after "
    #     "splitting along dose interruptions"
    # )

    return aristotle_data
