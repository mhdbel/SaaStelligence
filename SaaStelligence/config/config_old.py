# config/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Preserved variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    INTENT_MODEL_PATH = os.getenv("INTENT_MODEL_PATH", "models/intent_classifier.h5")
    META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
    GOOGLE_ADS_CLIENT_ID = os.getenv("GOOGLE_ADS_CLIENT_ID")
    HUBSPOT_API_KEY = os.getenv("HUBSPOT_API_KEY")

    # For AdGenerator - A/B Testing
    AD_PROMPT_TEMPLATES = [
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
    AD_LLM_TEMPERATURE = 0.7

    # For FunnelRouter - A/B Testing
    FUNNEL_MAP = {
        'workflow_automation': ['funnel_workflow_v1', 'funnel_workflow_v2'],
        'sales_team_efficiency': 'funnel_sales_default',
        'project_management': ['funnel_pm_alpha', 'funnel_pm_beta', 'funnel_pm_gamma'],
        'customer_support': 'funnel_support_main',
        'marketing_automation': ['funnel_marketing_leadgen', 'funnel_marketing_nurture']
    }
    DEFAULT_FUNNEL_NAME = 'default_general_funnel'

    # For BidAdjuster
    BID_BASE_BID = 10.0
    BID_CVR_THRESHOLD_HIGH = 0.05
    BID_CVR_THRESHOLD_LOW = 0.02
    BID_ADJUSTMENT_FACTOR_HIGH = 1.2
    BID_ADJUSTMENT_FACTOR_LOW = 0.8

    # A/B Testing assignment key
    AB_TEST_ASSIGNMENT_KEY = "user_id"

    # Retargeting Configuration
    RETARGETING_ACTIONS = {
        'email_submitted': [
            {'platform': 'meta', 'type': 'custom_audience', 'audience_id': 'meta_audience_123', 'description': 'Add to Meta Custom Audience for email submissions'},
            {'platform': 'google_ads', 'type': 'remarketing_list', 'list_id': 'google_list_abc', 'description': 'Add to Google Ads Remarketing List for email submissions'}
        ],
        'form_abandoned': [
            {'platform': 'meta', 'type': 'custom_audience', 'audience_id': 'meta_audience_456', 'description': 'Add to Meta Custom Audience for form abandons'},
        ],
        'viewed_pricing_page': [
            {'platform': 'meta', 'type': 'pixel_event', 'event_name': 'ViewPricing', 'description': 'Fire Meta Pixel for pricing page view'},
            {'platform': 'google_ads', 'type': 'conversion_event', 'conversion_id': 'google_conv_789', 'description': 'Fire Google Ads conversion for pricing page view'}
        ]
    }

    # New Behavioral Scoring Configuration
    BEHAVIORAL_ACTION_SCORES = {
        'processed_query_high_intent': 5, # e.g., for 'sales_team_efficiency'
        'processed_query_medium_intent': 3, # e.g., for 'workflow_automation'
        'processed_query_low_intent': 1,    # e.g., for general queries or less specific intents
        'email_submitted': 10, # Example of a high-value action
        'form_abandoned': -2, # Example of a negative action
        'viewed_pricing_page': 7
    }

CONFIG = Config()
