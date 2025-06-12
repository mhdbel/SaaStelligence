# SaaStelligence/config/config.py
"""
Centralized configuration for the SAAStelligence application.

This file defines the Config class, which holds various settings used across
different components of the agent. It loads sensitive information and environment-specific
parameters from a .env file.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file into the environment
load_dotenv()

class Config:
    """
    Configuration class for SAAStelligence.

    Attributes are defined as class variables and typically loaded from environment
    variables or set to default values.
    """

    # --- API Keys and External Service Credentials ---
    # These should be stored in the .env file for security and environment flexibility.
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # For AdGenerator (LangChain/OpenAI)
    INTENT_MODEL_PATH = os.getenv("INTENT_MODEL_PATH", "models/intent_classifier.h5") # For IntentDetector
    META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")      # For RetargetingManager (Facebook/Meta Ads)
    GOOGLE_ADS_CLIENT_ID = os.getenv("GOOGLE_ADS_CLIENT_ID")# For RetargetingManager (Google Ads)
    HUBSPOT_API_KEY = os.getenv("HUBSPOT_API_KEY")          # Example for potential CRM integration

    # --- AdGenerator Component Configurations ---
    # Settings for ad creative generation, including A/B testing of prompt templates.
    AD_PROMPT_TEMPLATES = [  # List of prompt templates for A/B testing
        """
Based on this intent: {intent}, generate a high-conversion ad copy for SaaS lead gen (Variant A).
Make it emotionally engaging, include urgency or scarcity where appropriate.
Output only the ad text.
""",
        """
Based on this intent: {intent}, create a compelling ad for SaaS lead generation (Variant B).
Focus on clarity and a strong call to action.
Output only the ad text.
"""
    ]
    AD_LLM_TEMPERATURE = 0.7  # Temperature for the LLM used in ad generation

    # --- FunnelRouter Component Configurations ---
    # Defines mapping from intents to funnels, supporting A/B testing of different funnels.
    FUNNEL_MAP = {
        'workflow_automation': ['funnel_workflow_v1', 'funnel_workflow_v2'], # Intent maps to a list of funnels for A/B testing
        'sales_team_efficiency': 'funnel_sales_default',                     # Intent maps to a single funnel
        'project_management': ['funnel_pm_alpha', 'funnel_pm_beta', 'funnel_pm_gamma'], # Example with three variants
        'customer_support': 'funnel_support_main',
        'marketing_automation': ['funnel_marketing_leadgen', 'funnel_marketing_nurture']
    }
    DEFAULT_FUNNEL_NAME = 'default_general_funnel'  # Fallback funnel if an intent is not in FUNNEL_MAP or list is empty

    # --- BidAdjuster Component Configurations ---
    # Parameters for the bid adjustment logic.
    BID_BASE_BID = 10.0               # Default base bid value
    BID_CVR_THRESHOLD_HIGH = 0.05     # CVR above which bid might be increased
    BID_CVR_THRESHOLD_LOW = 0.02      # CVR below which bid might be decreased
    BID_ADJUSTMENT_FACTOR_HIGH = 1.2  # Multiplier for bids when CVR is high
    BID_ADJUSTMENT_FACTOR_LOW = 0.8   # Multiplier for bids when CVR is low

    # --- General A/B Testing Configurations ---
    # AB_TEST_ASSIGNMENT_KEY could be used for more advanced consistent user assignment to test variants.
    # For example, hash user_id to pick a variant. Currently, selection is random in components if not specified.
    AB_TEST_ASSIGNMENT_KEY = "user_id"

    # --- RetargetingManager Component Configurations ---
    # Defines retargeting actions for different user behaviors (last_action events).
    # Each action specifies the platform and necessary parameters for a (simulated) API call.
    RETARGETING_ACTIONS = {
        'email_submitted': [
            {'platform': 'meta', 'type': 'custom_audience', 'audience_id': 'meta_audience_123', 'description': 'Add to Meta Custom Audience for email submissions'},
            {'platform': 'google_ads', 'type': 'remarketing_list', 'list_id': 'google_list_abc', 'description': 'Add to Google Ads Remarketing List for email submissions'}
        ],
        'form_abandoned': [
            {'platform': 'meta', 'type': 'custom_audience', 'audience_id': 'meta_audience_456', 'description': 'Add to Meta Custom Audience for form abandons'},
        ],
        'viewed_pricing_page': [ # Example of another user action trigger
            {'platform': 'meta', 'type': 'pixel_event', 'event_name': 'ViewPricing', 'description': 'Fire Meta Pixel for pricing page view'},
            {'platform': 'google_ads', 'type': 'conversion_event', 'conversion_id': 'google_conv_789', 'description': 'Fire Google Ads conversion for pricing page view'}
        ]
    }

    # --- BehavioralScorer Component Configurations ---
    # Defines points assigned for different user actions, used for behavioral scoring.
    BEHAVIORAL_ACTION_SCORES = {
        'processed_query_high_intent': 5,    # e.g., for 'sales_team_efficiency' intent
        'processed_query_medium_intent': 3,  # e.g., for 'workflow_automation' intent
        'processed_query_low_intent': 1,     # e.g., for general queries or less specific intents
        'email_submitted': 10,               # Example of a high-value conversion action
        'form_abandoned': -2,                # Example of a negative behavioral signal
        'viewed_pricing_page': 7             # Example of a significant engagement action
    }

# Create a single, global instance of the Config class for easy access from other modules.
CONFIG = Config()
