# SaaStelligence/agents/saastelligence_agent.py
"""
Core of the SAAStelligence autonomous agent.

This module defines the SAAStelligenceAgent class, which orchestrates various
components to process user queries, generate ads, route to funnels, adjust bids,
manage retargeting, and score user behavior.
"""
import pandas as pd
import os
from dotenv import load_dotenv

from SaaStelligence.components.intent_detector import IntentDetector
from SaaStelligence.components.ad_generator import AdGenerator
from SaaStelligence.components.funnel_router import FunnelRouter
from SaaStelligence.components.bid_adjuster import BidAdjuster
from SaaStelligence.components.retargeting_manager import RetargetingManager
from SaaStelligence.components.behavioral_scorer import BehavioralScorer

load_dotenv() # Load environment variables from .env file

class SAAStelligenceAgent:
    """
    The main SAAStelligence agent class.

    Coordinates various specialized components to handle tasks related to
    SaaS lead generation, such as intent detection, ad generation,
    funnel routing, bid adjustment, retargeting, and behavioral scoring.
    """
    def __init__(self):
        """
        Initializes the SAAStelligenceAgent and its components.

        Each component is responsible for a specific part of the agent's
        functionality.
        """
        self.intent_detector = IntentDetector()
        self.ad_generator = AdGenerator()
        self.funnel_router = FunnelRouter()
        self.bid_adjuster = BidAdjuster()
        self.retargeting_manager = RetargetingManager()
        self.behavioral_scorer = BehavioralScorer()

    def train_from_feedback(self, conversion_data: list[dict]):
        """
        Placeholder for training or fine-tuning models based on feedback.

        In a real implementation, this method would process conversion data
        (or other forms of feedback) to update machine learning models,
        adjust strategies, or log data for offline training.

        Args:
            conversion_data: A list of dictionaries, where each dictionary
                             represents a feedback event (e.g., a conversion).
                             The structure of these dictionaries would depend on
                             the specific data being collected.
        """
        # Example: Convert feedback data to a pandas DataFrame for processing
        df = pd.DataFrame(conversion_data)
        # In a production system, this data would be used to:
        # - Retrain the intent detection model
        # - Fine-tune ad generation prompts or models
        # - Adjust bidding strategies or CVR predictions
        # - Update A/B testing configurations or analyze results
        print("SAAStelligenceAgent: Received feedback data. Training model with new feedback (simulated)...")
        # For now, this method is a placeholder.
        # Example: if df contains 'intent' and 'predicted_cvr', it might be used for the BidAdjuster model.
        # X = df[['intent', 'predicted_cvr']]
        # y = df['converted']
        # self.bid_adjuster.train_model(X,y) # If BidAdjuster had such a method

    def report_performance(self) -> dict:
        """
        Placeholder for generating a performance report.

        This method would typically gather key metrics from various components
        or a centralized analytics store to provide a summary of the agent's
        performance.

        Returns:
            A dictionary containing key performance indicators (KPIs).
            Currently returns static example data.
        """
        # In a real system, metrics would be dynamically fetched or calculated.
        # Example:
        # ctr = self.ad_generator.get_ctr()
        # cvr = self.funnel_router.get_conversion_rate()
        # cpa = self.bid_adjuster.get_average_cpa()
        # leads = self.analytics_store.get_total_leads()
        metrics = {
            'CTR': 0.028,  # Click-Through Rate
            'CVR': 0.067,  # Conversion Rate
            'CPA': 28.00,  # Cost Per Acquisition
            'Leads': 3500  # Total leads generated
        }
        print("SAAStelligenceAgent: Performance Summary:", metrics)
        return metrics

    def run(self, user_query: str, user_id: str = None, last_action: str = None) -> dict:
        """
        Processes a user query and executes the agent's decision-making logic.

        This is the main operational method of the agent. It involves:
        1. Detecting user intent.
        2. Generating ad copy (potentially A/B testing different templates).
        3. Routing to a funnel (potentially A/B testing different funnels).
        4. Adjusting bids.
        5. Determining retargeting tasks.
        6. Tracking user actions for behavioral scoring.

        Args:
            user_query: The query string from the user.
            user_id: Optional unique identifier for the user.
            last_action: Optional string representing the last significant action
                         taken by the user, used for retargeting and behavioral scoring.

        Returns:
            A dictionary containing the results of the agent's processing,
            including detected intent, generated ad copy, chosen funnel, bid,
            retargeting tasks, and behavioral score.
        """
        # 1. Detect intent
        intent = self.intent_detector.detect_intent(user_query)

        # 2. Generate ad copy (components handle A/B selection internally if not specified)
        ad_generation_result = self.ad_generator.generate_ad(intent=intent)
        ad_copy = ad_generation_result["ad_copy"]
        ad_template_index_chosen = ad_generation_result["template_index_chosen"]

        # 3. Route to funnel (components handle A/B selection internally)
        funnel_routing_result = self.funnel_router.route_to_funnel(intent=intent)
        funnel = funnel_routing_result["funnel_name"]
        funnel_variant_index_chosen = funnel_routing_result["funnel_variant_index_chosen"]

        # 4. Adjust bid (using example CVR and budget for now)
        # In a real scenario, predicted_cvr might come from another component or historical data.
        bid = self.bid_adjuster.adjust_bid(predicted_cvr=0.05, cpa_budget=45)

        # 5. Determine retargeting tasks
        retargeting_tasks_list = []
        if user_id and last_action:
            retargeting_tasks_list = self.retargeting_manager.retarget_user(user_id, last_action)

        # 6. Behavioral scoring
        current_behavioral_score = 0 # Default score if no user_id
        if user_id:
            # Determine action type for scoring based on intent (simplified)
            action_to_track_query = 'processed_query_low_intent' # Default
            if intent == 'sales_team_efficiency':
                action_to_track_query = 'processed_query_high_intent'
            elif intent == 'workflow_automation':
                action_to_track_query = 'processed_query_medium_intent'

            self.behavioral_scorer.track_action(
                user_id,
                action_to_track_query,
                details={'intent': intent, 'query': user_query}
            )

            # Track the `last_action` from parameters as a separate behavioral event if present
            if last_action:
                 self.behavioral_scorer.track_action(
                     user_id,
                     last_action,
                     details={'source': 'run_method_last_action_parameter'}
                 )
            current_behavioral_score = self.behavioral_scorer.get_score(user_id)

        # Consolidate results
        return {
            'intent': intent,
            'ad_copy': ad_copy,
            'ad_template_index_chosen': ad_template_index_chosen,
            'funnel': funnel,
            'funnel_variant_index_chosen': funnel_variant_index_chosen,
            'bid': bid,
            'retargeting_tasks': retargeting_tasks_list,
            'behavioral_score': current_behavioral_score
        }
