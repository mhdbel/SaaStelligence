# config/config.py

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    INTENT_MODEL_PATH = os.getenv("INTENT_MODEL_PATH", "models/intent_classifier.h5")
    TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "models/tokenizer.json")
    CONVERSIONS_DATA_PATH = os.getenv("CONVERSIONS_DATA_PATH", "data/conversions.csv")
    META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
    GOOGLE_ADS_CLIENT_ID = os.getenv("GOOGLE_ADS_CLIENT_ID")
    HUBSPOT_API_KEY = os.getenv("HUBSPOT_API_KEY")
    GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH")

CONFIG = Config()