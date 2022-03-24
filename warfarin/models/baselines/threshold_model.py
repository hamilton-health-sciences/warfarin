"""Baseline model with the rules-based approach."""

import numpy as np


class ThresholdModel:
    """
    Implements the warfarin-dosing threshold model given in [1].

    Specifically, it's a rule-based approach:
        * If the INR is less than 1.5, increase the dose by 15%.
        * If the INR is between 1.5 and 2, increase the dose by 10%.
        * If the INR is between 3 and 5, then decrease the dose by 10%.
        * If the INR is greater than 5, then decrease the dose by 15%.
        * Otherwise, maintain the dose.

    [1] doi:10.1161/CIRCULATIONAHA.112.101808
    """

    def select_action(self, previous_inr, current_inr):
        """
        Give the threshold algorithm's action.

        Note: The action codes given assume a 7-dimensional action space.

        Args:
            previous_inr: The INR at the previous check-in.
            current_inr: The INR at the current check-in.

        Returns:
            model_actions: The decision of the algorithm based on the rules.
        """
        conditions = [
            current_inr <= 1.5,
            current_inr < 2.,
            (current_inr >= 2.) & (current_inr <= 3.),
            (current_inr > 3.) & (current_inr < 5),
            current_inr >= 5.
        ]
        actions = [5, 4, 3, 2, 1]
        model_actions = np.select(conditions, actions)

        return model_actions

    def select_action_quant(self, previous_inr, current_inr):
        """
        Give the threshold algorithm's action in value of dose change.

        Args:
            previous_inr: The INR at the previous check-in.
            current_inr: The INR at the current check-in.

        Returns:
            model_actions: The decision of the algorithm based on the rules.
        """
        conditions = [
            current_inr <= 1.5,
            current_inr < 2.,
            (current_inr >= 2.) & (current_inr <= 3.),
            (current_inr > 3.) & (current_inr < 5),
            current_inr >= 5.
        ]
        actions = [0.15, 0.1, 0., -0.1, -0.15]
        model_actions = np.select(conditions, actions)

        # Convert to relative dose changes
        model_actions += 1.

        return model_actions
