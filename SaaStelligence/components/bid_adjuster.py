# components/bid_adjuster.py
"""
Handles the adjustment of advertising bids based on performance metrics.

This component uses configured thresholds and multipliers to adjust a base bid
value according to the predicted conversion rate (CVR).
"""
from SaaStelligence.config.config import CONFIG

class BidAdjuster:
    """
    Adjusts advertising bids based on predicted CVR and configured parameters.
    """
    def __init__(self):
        """
        Initializes the BidAdjuster.
        Bid adjustment parameters are accessed directly from CONFIG within methods.
        """
        pass

    def adjust_bid(self, predicted_cvr: float, cpa_budget: float) -> float:
        """
        Adjusts the bid based on the predicted conversion rate (CVR).

        The logic uses a base bid and adjusts it up or down if the CVR is above
        a high threshold or below a low threshold, respectively. The adjustment
        factors and thresholds are defined in the application configuration.

        Args:
            predicted_cvr: The predicted conversion rate for the ad/keyword.
            cpa_budget: The target Cost Per Acquisition (CPA). Currently unused
                        in the logic but retained in signature for future use.

        Returns:
            The adjusted bid value.
        """
        # Retrieve bid parameters from configuration
        base_bid = CONFIG.BID_BASE_BID
        high_cvr_threshold = CONFIG.BID_CVR_THRESHOLD_HIGH
        low_cvr_threshold = CONFIG.BID_CVR_THRESHOLD_LOW
        high_cvr_multiplier = CONFIG.BID_ADJUSTMENT_FACTOR_HIGH
        low_cvr_multiplier = CONFIG.BID_ADJUSTMENT_FACTOR_LOW

        # Apply bidding logic
        if predicted_cvr > high_cvr_threshold:
            # If CVR is high, increase the bid
            return base_bid * high_cvr_multiplier
        elif predicted_cvr < low_cvr_threshold:
            # If CVR is low, decrease the bid
            return base_bid * low_cvr_multiplier
        else:
            # Otherwise, use the base bid
            return base_bid
