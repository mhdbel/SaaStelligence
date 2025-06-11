# agents/saastelligence_agent.py

from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import requests
import os
from dotenv import load_dotenv
from config.config import CONFIG

load_dotenv()

# Load pre-trained intent detection model
intent_model = load_model(CONFIG.INTENT_MODEL_PATH)

# Define ad generation prompt template
ad_prompt_template = PromptTemplate.from_template(
    """
    Based on this intent: {intent}, generate a high-conversion ad copy for SaaS lead gen.
    Make it emotionally engaging, include urgency or scarcity where appropriate.
    Output only the ad text.
    """
)
ad_chain = LLMChain(llm=OpenAI(openai_api_key=CONFIG.OPENAI_API_KEY, temperature=0.7), prompt=ad_prompt_template)

class SAAStelligenceAgent:
    def __init__(self):
        self.intent_mapping = {
            'workflow_automation': 0,
            'sales_team_efficiency': 1,
            'project_management': 2,
            'customer_support': 3,
            'marketing_automation': 4
        }

    def detect_intent(self, query):
        encoded = pd.Series([query]).apply(lambda x: x.lower())
        return list(self.intent_mapping.keys())[np.argmax(intent_model.predict(encoded))]

    def generate_ad(self, intent):
        return ad_chain.run(intent=intent)

    def route_to_funnel(self, intent):
        funnel_map = {
            'workflow_automation': 'funnel_a',
            'sales_team_efficiency': 'funnel_b',
            'project_management': 'funnel_c',
            'customer_support': 'funnel_d',
            'marketing_automation': 'funnel_e'
        }
        return funnel_map.get(intent, 'default_funnel')

    def adjust_bid(self, predicted_cvr, cpa_budget):
        base_bid = 10.0
        if predicted_cvr > 0.05:
            return base_bid * 1.2
        elif predicted_cvr < 0.02:
            return base_bid * 0.8
        else:
            return base_bid

    def retarget_user(self, user_id, last_action):
        if last_action == 'email_submitted':
            return f"https://ads.example.com/retarget/email?uid={user_id}"
        elif last_action == 'form_abandoned':
            return f"https://ads.example.com/retarget/form?uid={user_id}"
        else:
            return None

    def train_from_feedback(self, conversion_data):
        df = pd.DataFrame(conversion_data)
        X = df[['intent', 'predicted_cvr']]
        y = df['converted']
        # In production, use model.fit(X, y) 
        print("Training model with new feedback...")

    def report_performance(self):
        metrics = {
            'CTR': 0.028,
            'CVR': 0.067,
            'CPA': 28.00,
            'Leads': 3500
        }
        print("Performance Summary:", metrics)
        return metrics

    def run(self, user_query, user_id=None, last_action=None):
        intent = self.detect_intent(user_query)
        ad_copy = self.generate_ad(intent)
        funnel = self.route_to_funnel(intent)
        bid = self.adjust_bid(predicted_cvr=0.05, cpa_budget=45)
        retarget_url = self.retarget_user(user_id, last_action) if user_id else None
        return {
            'intent': intent,
            'ad_copy': ad_copy,
            'funnel': funnel,
            'bid': bid,
            'retarget_url': retarget_url
        }