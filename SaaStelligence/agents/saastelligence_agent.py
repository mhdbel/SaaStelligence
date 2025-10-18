"""Core agent orchestration for the SAAStelligence pipeline."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover - optional dependency
    from langchain import LLMChain
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate
except Exception:  # pragma: no cover - optional dependency
    LLMChain = None  # type: ignore
    OpenAI = None  # type: ignore
    PromptTemplate = None  # type: ignore

from SaaStelligence.config.config import CONFIG
from SaaStelligence.models.train_intent_model import (
    INTENT_CATEGORIES,
    REQUIRED_DATA_COLUMNS,
    load_model as load_intent_model,
    train_intent_classifier,
)
from SaaStelligence.utils.report_utils import format_metrics
from SaaStelligence.utils.repoting_dashboard import update_dashboard

NUM_INTENTS = len(INTENT_CATEGORIES)
OPTIONAL_PERFORMANCE_COLUMNS = ["clicks", "impressions", "cost"]
CSV_FIELD_ORDER = ["query_text", "intent", "converted", *OPTIONAL_PERFORMANCE_COLUMNS]


def _normalize_record(record: Dict[str, str]) -> Dict[str, object]:
    normalized: Dict[str, object] = {
        "query_text": record.get("query_text", ""),
        "intent": record.get("intent", ""),
        "converted": int(float(record.get("converted", 0)) > 0),
    }
    for column in OPTIONAL_PERFORMANCE_COLUMNS:
        value = record.get(column)
        if value in (None, ""):
            normalized[column] = 0.0
        else:
            try:
                normalized[column] = float(value)
            except ValueError:
                normalized[column] = 0.0
    return normalized


class SAAStelligenceAgent:
    """Encapsulates intent detection, ad generation, bidding and reporting."""

    def __init__(self) -> None:
        self.intent_mapping = {intent: idx for idx, intent in enumerate(INTENT_CATEGORIES)}
        self._intent_model = None
        self._model_path = Path(CONFIG.INTENT_MODEL_PATH)
        self._ad_chain = self._build_ad_chain()
        self._data_path = Path(CONFIG.CONVERSIONS_DATA_PATH)

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------
    def _build_ad_chain(self) -> Optional[LLMChain]:
        if not CONFIG.OPENAI_API_KEY or LLMChain is None or OpenAI is None or PromptTemplate is None:
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
            if not self._model_path.exists():
                dataset = self._load_performance_data()
                if not dataset:
                    raise FileNotFoundError(
                        f"Intent model not found at {self._model_path} and no dataset available to train a fallback "
                        "model. Populate data/conversions.csv and run the training script."
                    )
                self._intent_model = train_intent_classifier(
                    records=[{key: str(value) for key, value in record.items()} for record in dataset],
                    model_path=self._model_path,
                )
            else:
                self._intent_model = load_intent_model(self._model_path)
        return self._intent_model

    # ------------------------------------------------------------------
    # Core agent capabilities
    # ------------------------------------------------------------------
    def detect_intent(self, query: str) -> Tuple[str, float]:
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        model = self._ensure_model()
        probabilities = model.predict_proba(query)
        intent_index = max(range(NUM_INTENTS), key=lambda idx: probabilities[idx])
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
    def _load_performance_data(self) -> List[Dict[str, object]]:
        if not self._data_path.exists():
            return []
        with self._data_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            records = [_normalize_record(row) for row in reader]
        return records

    def _write_performance_data(self, records: Iterable[Dict[str, object]]) -> None:
        rows = list(records)
        if not rows:
            return
        with self._data_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELD_ORDER)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "query_text": row.get("query_text", ""),
                    "intent": row.get("intent", ""),
                    "converted": int(row.get("converted", 0)),
                    "clicks": row.get("clicks", 0),
                    "impressions": row.get("impressions", 0),
                    "cost": row.get("cost", 0),
                })

    def _estimate_intent_cvr(self, intent: str) -> float:
        records = self._load_performance_data()
        if not records:
            return 0.03

        intent_records = [record for record in records if record.get("intent") == intent] or records
        clicks = sum(float(record.get("clicks", 0)) for record in intent_records)
        if clicks <= 0:
            clicks = float(len(intent_records))
        conversions = sum(int(record.get("converted", 0)) for record in intent_records)
        if clicks <= 0:
            return 0.03
        return float(conversions / clicks)

    def train_from_feedback(self, conversion_data):
        if not conversion_data:
            raise ValueError("conversion_data must contain at least one record.")

        normalized_rows: List[Dict[str, object]] = []
        for row in conversion_data:
            missing = REQUIRED_DATA_COLUMNS - set(row.keys())
            if missing:
                raise ValueError(f"Feedback data missing required fields: {sorted(missing)}")
            normalized_rows.append(_normalize_record(row))

        existing = self._load_performance_data()
        combined = existing + normalized_rows
        self._write_performance_data(combined)

        train_intent_classifier(
            records=[{key: str(value) for key, value in record.items()} for record in combined],
            model_path=self._model_path,
        )

        # Reset cached artifacts so subsequent calls use the updated model.
        self._intent_model = None

        return {"records_trained": len(combined)}

    def _calculate_reporting_metrics(self) -> Dict[str, float]:
        records = self._load_performance_data()
        if not records:
            return {"CTR": 0.0, "CVR": 0.0, "CPA": 0.0, "Leads": 0}

        impressions = sum(float(record.get("impressions", 0)) for record in records)
        clicks = sum(float(record.get("clicks", 0)) for record in records)
        conversions = sum(int(record.get("converted", 0)) for record in records)
        cost = sum(float(record.get("cost", 0)) for record in records)

        if impressions <= 0:
            impressions = max(clicks, float(len(records)))
        if clicks <= 0:
            clicks = float(len(records))

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
