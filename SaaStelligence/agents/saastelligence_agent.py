"""Core agent orchestration for the SAAStelligence pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from config.config import CONFIG
from models.train_intent_model import (
    INTENT_CATEGORIES,
    MAX_SEQUENCE_LENGTH,
    REQUIRED_DATA_COLUMNS,
    train_intent_classifier,
)
from utils.report_utils import format_metrics
from utils.repoting_dashboard import update_dashboard


class SAAStelligenceAgent:
    """Encapsulates intent detection, ad generation, bidding and reporting."""

    def __init__(self) -> None:
        self.intent_mapping = {intent: idx for idx, intent in enumerate(INTENT_CATEGORIES)}
        self._intent_model = None
        self._tokenizer = None
        self._max_sequence_length = MAX_SEQUENCE_LENGTH
        self._ad_chain = self._build_ad_chain()
        self._data_path = Path(CONFIG.CONVERSIONS_DATA_PATH)
        self._tokenizer_path = Path(CONFIG.TOKENIZER_PATH)

    # ------------------------------------------------------------------
    # Model and tokenizer management
    # ------------------------------------------------------------------
    def _build_ad_chain(self) -> Optional[LLMChain]:
        if not CONFIG.OPENAI_API_KEY:
            return None

        ad_prompt_template = PromptTemplate.from_template(
            """
            Based on this intent: {intent}, generate a high-conversion ad copy for SaaS lead gen.
            Make it emotionally engaging, include urgency or scarcity where appropriate.
            Output only the ad text.
            """
        )
        return LLMChain(
            llm=OpenAI(openai_api_key=CONFIG.OPENAI_API_KEY, temperature=0.7),
            prompt=ad_prompt_template,
        )

    def _ensure_model(self):
        if self._intent_model is None:
            model_path = Path(CONFIG.INTENT_MODEL_PATH)
            if not model_path.exists():
                dataset = self._load_performance_data()
                if dataset.empty:
                    raise FileNotFoundError(
                        f"Intent model not found at {model_path} and no dataset available to train a fallback "
                        "model. Populate data/conversions.csv and run the training script."
                    )
                model, tokenizer = train_intent_classifier(
                    dataframe=dataset,
                    model_path=model_path,
                    tokenizer_path=self._tokenizer_path,
                )
                self._tokenizer = tokenizer
                self._max_sequence_length = MAX_SEQUENCE_LENGTH
                self._intent_model = model
            else:
                self._intent_model = load_model(model_path)
        return self._intent_model

    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            if not self._tokenizer_path.exists():
                raise FileNotFoundError(
                    f"Tokenizer artifact not found at {self._tokenizer_path}. Train the model to generate it."
                )
            with open(self._tokenizer_path, "r", encoding="utf-8") as handle:
                artifact = json.load(handle)
            self._tokenizer = tokenizer_from_json(artifact["tokenizer_config"])
            self._max_sequence_length = artifact.get(
                "max_sequence_length", MAX_SEQUENCE_LENGTH
            )
        return self._tokenizer

    # ------------------------------------------------------------------
    # Core agent capabilities
    # ------------------------------------------------------------------
    def detect_intent(self, query: str) -> Tuple[str, float]:
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        model = self._ensure_model()
        tokenizer = self._ensure_tokenizer()

        sequence = tokenizer.texts_to_sequences([query])
        padded = pad_sequences(
            sequence,
            maxlen=self._max_sequence_length,
            padding="post",
            truncating="post",
        )

        probabilities = model.predict(padded, verbose=0)[0]
        intent_index = int(np.argmax(probabilities))
        intent = INTENT_CATEGORIES[intent_index]
        confidence = float(probabilities[intent_index])
        return intent, confidence

    def generate_ad(self, intent: str) -> str:
        if self._ad_chain is None:
            return (
                "OpenAI API key not configured. Provide OPENAI_API_KEY to enable dynamic ad copy generation."
            )
        return self._ad_chain.run(intent=intent)

    def route_to_funnel(self, intent: str) -> str:
        funnel_map = {
            "workflow_automation": "funnel_a",
            "sales_team_efficiency": "funnel_b",
            "project_management": "funnel_c",
            "customer_support": "funnel_d",
            "marketing_automation": "funnel_e",
        }
        return funnel_map.get(intent, "default_funnel")

    def adjust_bid(self, predicted_cvr: float, cpa_budget: float) -> float:
        base_bid = 10.0
        if predicted_cvr <= 0:
            return round(base_bid * 0.6, 2)

        target_cvr = min(0.15, max(0.02, 1 / (cpa_budget / base_bid)))
        performance_ratio = predicted_cvr / target_cvr

        if performance_ratio >= 1.2:
            multiplier = 1.3
        elif performance_ratio >= 1.0:
            multiplier = 1.15
        elif performance_ratio <= 0.6:
            multiplier = 0.7
        else:
            multiplier = 0.9

        return round(base_bid * multiplier, 2)

    def retarget_user(self, user_id: Optional[str], last_action: Optional[str]):
        if not user_id:
            return None
        if last_action == "email_submitted":
            return f"https://ads.example.com/retarget/email?uid={user_id}"
        if last_action == "form_abandoned":
            return f"https://ads.example.com/retarget/form?uid={user_id}"
        return None

    # ------------------------------------------------------------------
    # Data, training, and reporting helpers
    # ------------------------------------------------------------------
    def _load_performance_data(self) -> pd.DataFrame:
        if not self._data_path.exists():
            return pd.DataFrame(columns=sorted(REQUIRED_DATA_COLUMNS | {"clicks", "impressions", "cost"}))
        df = pd.read_csv(self._data_path)
        if "converted" in df.columns:
            df["converted"] = pd.to_numeric(df["converted"], errors="coerce").fillna(0).astype(int)
        return df

    def _estimate_intent_cvr(self, intent: str) -> float:
        df = self._load_performance_data()
        if df.empty:
            return 0.03

        if "clicks" not in df.columns or "converted" not in df.columns:
            return 0.03

        intent_df = df[df["intent"] == intent]
        if intent_df.empty:
            intent_df = df

        clicks = intent_df.get("clicks", pd.Series(dtype=float)).sum()
        if clicks <= 0:
            clicks = len(intent_df)

        conversions = intent_df.get("converted", pd.Series(dtype=float)).sum()
        if clicks <= 0:
            return 0.03
        return float(conversions / clicks)

    def train_from_feedback(self, conversion_data):
        if not conversion_data:
            raise ValueError("conversion_data must contain at least one record.")

        df = pd.DataFrame(conversion_data)
        missing = REQUIRED_DATA_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Feedback data missing required fields: {sorted(missing)}"
            )

        df["converted"] = df["converted"].apply(lambda value: int(bool(value)))
        for column in ["clicks", "impressions", "cost"]:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        existing = self._load_performance_data()
        combined = pd.concat([existing, df], ignore_index=True)
        combined.to_csv(self._data_path, index=False)

        train_intent_classifier(
            dataframe=combined,
            model_path=Path(CONFIG.INTENT_MODEL_PATH),
            tokenizer_path=self._tokenizer_path,
        )

        # Reset cached artifacts so subsequent calls use the updated model.
        self._intent_model = None
        self._tokenizer = None

        return {"records_trained": len(combined)}

    def _calculate_reporting_metrics(self) -> Dict[str, float]:
        df = self._load_performance_data()
        if df.empty:
            return {"CTR": 0.0, "CVR": 0.0, "CPA": 0.0, "Leads": 0}

        impressions = df.get("impressions", pd.Series(dtype=float)).sum()
        clicks = df.get("clicks", pd.Series(dtype=float)).sum()
        conversions = df.get("converted", pd.Series(dtype=float)).sum()
        cost = df.get("cost", pd.Series(dtype=float)).sum()

        if impressions <= 0:
            impressions = max(clicks, len(df))
        if clicks <= 0:
            clicks = len(df)

        ctr = clicks / impressions if impressions else 0.0
        cvr = conversions / clicks if clicks else 0.0
        cpa = (cost / conversions) if conversions and cost else 0.0

        return {
            "CTR": ctr,
            "CVR": cvr,
            "CPA": cpa,
            "Leads": int(conversions),
        }

    def report_performance(self):
        metrics = self._calculate_reporting_metrics()
        formatted = format_metrics(metrics)
        update_dashboard(
            formatted,
            credentials_path=Path(CONFIG.GOOGLE_CREDENTIALS_PATH)
            if CONFIG.GOOGLE_CREDENTIALS_PATH
            else None,
        )
        return formatted

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    def run(self, user_query, user_id=None, last_action=None):
        intent, confidence = self.detect_intent(user_query)
        ad_copy = self.generate_ad(intent)
        funnel = self.route_to_funnel(intent)
        predicted_cvr = self._estimate_intent_cvr(intent)
        bid = self.adjust_bid(predicted_cvr=predicted_cvr, cpa_budget=45)
        retarget_url = self.retarget_user(user_id, last_action)
        return {
            "intent": intent,
            "intent_confidence": round(confidence, 4),
            "ad_copy": ad_copy,
            "funnel": funnel,
            "predicted_cvr": round(predicted_cvr, 4),
            "bid": bid,
            "retarget_url": retarget_url,
        }