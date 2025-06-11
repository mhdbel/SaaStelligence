# config/config.py

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    INTENT_MODEL_PATH = os.getenv("INTENT_MODEL_PATH", "models/intent_classifier.h5")
    META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
    GOOGLE_ADS_CLIENT_ID = os.getenv("GOOGLE_ADS_CLIENT_ID")
    HUBSPOT_API_KEY = os.getenv("HUBSPOT_API_KEY")

CONFIG = Config()