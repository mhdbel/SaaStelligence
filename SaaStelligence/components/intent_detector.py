# components/intent_detector.py
"""
Handles the detection of user intent from a given query.

This component uses a pre-trained machine learning model to classify
queries into predefined intent categories.
"""
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from SaaStelligence.config.config import CONFIG
from dotenv import load_dotenv

load_dotenv() # Ensures .env variables are loaded for CONFIG

class IntentDetector:
    """
    Detects user intent using a loaded Keras model and predefined intent mappings.
    """
    def __init__(self):
        """
        Initializes the IntentDetector.

        Loads the intent detection model and sets up the intent mapping.
        The model path is retrieved from the application configuration.
        """
        self.intent_mapping = {
            'workflow_automation': 0,
            'sales_team_efficiency': 1,
            'project_management': 2,
            'customer_support': 3,
            'marketing_automation': 4
        }
        # Load the pre-trained intent detection model
        # CONFIG.INTENT_MODEL_PATH should point to a trained .h5 Keras model file.
        self.intent_model = load_model(CONFIG.INTENT_MODEL_PATH)

    def detect_intent(self, query: str) -> str:
        """
        Detects the intent of a given user query.

        Args:
            query: The user's query string.

        Returns:
            A string representing the detected intent (e.g., 'workflow_automation').
        """
        # Preprocess the query similar to how the model was trained
        # (e.g., lowercasing, potentially tokenization if model expects it)
        encoded_query = pd.Series([query]).apply(lambda x: x.lower())

        # Get model prediction (typically an array of probabilities or one-hot encoded vector)
        prediction = self.intent_model.predict(encoded_query)

        # Determine the intent with the highest probability
        # np.argmax finds the index of the maximum value in the prediction array.
        # This index corresponds to an intent in our mapping.
        detected_intent_index = np.argmax(prediction)

        # Convert the index back to the intent string label
        # Assumes self.intent_mapping keys are ordered consistently if direct list conversion is used.
        # A more robust mapping might involve mapping integer outputs of model back to specific labels
        # if the model output isn't directly an index into a simple list of keys.
        # For now, assuming direct mapping via list(keys) is how it's intended.
        return list(self.intent_mapping.keys())[detected_intent_index]
