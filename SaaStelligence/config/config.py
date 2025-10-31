# config/config.py

import os
from pathlib import Path

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv() -> bool:
        return False


load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    INTENT_MODEL_PATH = os.getenv(
        "INTENT_MODEL_PATH", str(BASE_DIR / "models" / "intent_classifier.json")
    )
    CONVERSIONS_DATA_PATH = os.getenv(
        "CONVERSIONS_DATA_PATH", str(BASE_DIR / "data" / "conversions.csv")
    )
    META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
    GOOGLE_ADS_CLIENT_ID = os.getenv("GOOGLE_ADS_CLIENT_ID")
    HUBSPOT_API_KEY = os.getenv("HUBSPOT_API_KEY")
    GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH")


CONFIG = Config()
