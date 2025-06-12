# components/funnel_router.py
"""
Handles routing users to different funnels based on their intent.

This component uses a configurable map to determine the appropriate funnel
for a given user intent. It supports A/B testing by allowing multiple
funnel options for a single intent.
"""
import random
from SaaStelligence.config.config import CONFIG

class FunnelRouter:
    """
    Routes users to funnels based on intent, supporting A/B testing of funnels.
    """
    def __init__(self):
        """
        Initializes the FunnelRouter.
        Configuration for funnel mapping is accessed directly from CONFIG within methods.
        """
        pass

    def route_to_funnel(self, intent: str, variation_index: int = None) -> dict:
        """
        Determines the funnel for a given user intent, with A/B test variation support.

        The funnel mapping is defined in `CONFIG.FUNNEL_MAP`. If an intent in the map
        points to a list of funnels, this method will select one. If a valid
        `variation_index` is provided, the specific funnel at that index is chosen.
        Otherwise, a funnel is selected randomly from the list. If an intent maps to
        a single funnel string, that funnel is used. If an intent is not found in the
        map, the `CONFIG.DEFAULT_FUNNEL_NAME` is used.

        Args:
            intent: The user intent for which to determine the funnel.
            variation_index: Optional specific index of the funnel if A/B testing
                             multiple funnels for the given intent.

        Returns:
            A dictionary containing:
                - "funnel_name": The name of the chosen funnel.
                - "funnel_variant_index_chosen": The index of the chosen funnel if selected
                  from a list (0 or greater). Set to `None` if the funnel was a direct
                  string mapping (not part of a list). Set to -2 if it was a fallback
                  to default due to an empty list for the intent.
        """
        # Get the destination (single funnel string or list of funnels) from config
        # If intent not found, use the default funnel name
        destination_config = CONFIG.FUNNEL_MAP.get(intent, CONFIG.DEFAULT_FUNNEL_NAME)

        chosen_funnel_name = None
        chosen_funnel_variant_idx = None # Default for single/direct mapping

        if isinstance(destination_config, list):
            if not destination_config:
                # If the list is empty for a configured intent, fallback to default.
                chosen_funnel_name = CONFIG.DEFAULT_FUNNEL_NAME
                chosen_funnel_variant_idx = -2 # Special index indicating fallback from empty list
            elif variation_index is not None and 0 <= variation_index < len(destination_config):
                # Use specified funnel if index is valid
                chosen_funnel_name = destination_config[variation_index]
                chosen_funnel_variant_idx = variation_index
            else:
                # Randomly select a funnel from the list if no valid index provided
                chosen_funnel_variant_idx = random.randrange(len(destination_config))
                chosen_funnel_name = destination_config[chosen_funnel_variant_idx]
        else:
            # Destination is a single string (direct mapping or default)
            chosen_funnel_name = destination_config
            # chosen_funnel_variant_idx remains None for direct string mappings

        return {
            "funnel_name": chosen_funnel_name,
            "funnel_variant_index_chosen": chosen_funnel_variant_idx
        }
