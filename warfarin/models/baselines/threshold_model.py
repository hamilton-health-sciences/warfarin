"""Baseline model with the rules-based approach."""

import numpy as np


class ThresholdModel:
    """
    Implements the warfarin-dosing threshold model given in [1].

    Specifically, it's a rule-based approach:
        * If the INR is less than 1 or greater than 4, do not make a decision.
        * If the INR is between 1 and 1.5, increase the dose by 15%.
        * If the INR is between 1.5 and 2, increase the dose by 10%.
        * If the INR is greater than 3 twice in a row, decrease the dose by 10%.
        * Otherwise, maintain the dose.

    [1] Nieuwlaat, R. et al. Randomised comparison of a simple warfarin dosing
        algorithm  versus a computerised anticoagulation management system for
        control of warfarin maintenance therapy. Thromb Haemost, 2012.
    """
    def __init__(self):
        self.conditions = [
            (current_inr < 1.) | (current_inr > 4.),
            current_inr < 1.5,
            current_inr < 2.,
            (current_inr >= 2.) & (current_inr <= 3.),
            (current_inr > 3.) & (np.isnan(previous_inr) |
                                  (previous_inr <= 3.)),
            (current_inr > 3.) & (previous_inr > 3.)
        ]

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
        actions = [np.nan, 5, 4, 3, 3, 2]
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
        actions = [np.nan, 0.15, 0.1, 0., 0., -0.1]
        model_actions = np.select(self.conditions, actions)

        return model_actions
